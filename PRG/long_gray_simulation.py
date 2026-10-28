"""Record continuous gray-screen activity for PRG analysis.

Example (from the repository root)::

    python3 PRG/long_gray_simulation.py [--restore_from /path/to/Best_model]

Match the training run's network selection, seed, readout, integration scheme,
and online-voltage-penalty settings when restoring a checkpoint. Only the
recorded outputs are quantized: spikes to uint8 and normalized voltages to
float16. No training configuration is loaded automatically.
"""

from concurrent.futures import ThreadPoolExecutor
import json
import math
import os
import zlib
from contextlib import nullcontext
from pathlib import Path
import sys
from datetime import datetime, timezone
from time import perf_counter

import h5py
import numpy as np
from absl import app, flags as absl_flags

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))


def register_flags(flag_values=None):
    """Register the recorder CLI, optionally into isolated flags for tests."""
    flag_values = absl_flags.FLAGS if flag_values is None else flag_values
    definitions = {
        "string": {
            "data_dir": ("GLIF_network_nll_full", "Network data directory."),
            "restore_from": ("", "Checkpoint directory, or subdirectory of --ckpt_dir."),
            "ckpt_dir": ("", "Optional training run directory."),
            "output_file": ("", "New HDF5 path; default: unique PRG/results subdirectory."),
            "voltage_penalty_mode": ("range", "Must match the training run."),
        },
        "integer": {
            "neurons": (203816, "Simulated neurons; 0 selects all neurons."),
            "n_input": (17400, "Number of LGN inputs."),
            "n_output": (2, "Training model readout width."),
            "neurons_per_output": (16, "Training model readout population size."),
            "cue_duration": (40, "Training model readout cue duration."),
            "seq_len": (500, "Maximum steps per rollout and disk write (1 ms/step)."),
            "warmup_ms": (5000, "Discarded gray-screen settling time in ms."),
            "duration_ms": (60000, "Recorded gray-screen duration in ms."),
            "seed": (3000, "Network seed; must match the training run."),
            "run_session": (0, "Noise stream offset for repeat recordings."),
            "max_delay": (0, "Maximum synaptic delay; 0 derives it from the data."),
        },
        "float": {
            "plot_core_radius": (400.0, "Radius of the recorded core in micrometres."),
            "loss_core_radius": (400.0, "Core radius used to select training readout neurons."),
            "input_weight_scale": (1.0, "Training model input weight scale."),
            "dampening_factor": (0.05, "Training model surrogate damping."),
            "recurrent_dampening_factor": (1.0, "Training recurrent damping."),
            "gauss_std": (0.28, "Training surrogate Gaussian width."),
            "lr_scale": (1.0, "Training model weight parameterization scale."),
        },
        "boolean": {
            "caching": (True, "Reuse cached network data."),
            "compress_model_cache": (False, "Use LZ4 network caches."),
            "core_only": (False, "Simulate only core neurons; must match training."),
            "connected_selection": (True, "Use connected network selection."),
            "random_weights": (False, "Use randomized initial weights."),
            "uniform_weights": (False, "Use uniform initial weights."),
            "hard_reset": (True, "Clamp voltage to reset during refractoriness; must match training."),
            "neuron_output": (False, "Training model readout scheme."),
            "pseudo_gauss": (False, "Use the Gaussian surrogate."),
            "use_online_voltage_loss": (True, "Must match the checkpoint's state inputs."),
            "restore_runtime_dtype_cast": (True, "Allow in-memory checkpoint dtype conversion."),
            "bmtk_compat_lgn": (True, "Use BMTK-compatible LGN probabilities."),
        },
    }
    for kind, items in definitions.items():
        define = getattr(absl_flags, f"DEFINE_{kind}")
        for name, (default, help_text) in items.items():
            define(name, default, help_text, flag_values=flag_values)
    enums = {
        "dtype": ("float32", ["float16", "float32"]),
        "acceleration": ("auto", ["auto", "cuda", "tensorflow"]),
        "integration_scheme": ("exact", ["exact", "euler"]),
        "surrogate_gradient": ("gaussian", ["triangular", "gaussian", "slayer"]),
        "lgn_backend": ("grating", ["grating", "movie", "tensorflow"]),
        "neuron_layout": ("morton", ["canonical", "morton"]),
        "lgn_row_order": ("original", ["original", "retinotopic"]),
    }
    for name, (default, choices) in enums.items():
        absl_flags.DEFINE_enum(name, default, choices, name.replace("_", " "),
                               flag_values=flag_values)
    return flag_values


def validate_flags(flags):
    """Reject invalid recording requests before loading a model."""
    from v1_model_utils.training_utils import validate_n_input

    validate_n_input(flags.n_input)
    for name in ("seq_len", "duration_ms", "n_output", "neurons_per_output"):
        if getattr(flags, name) <= 0:
            raise ValueError(f"--{name} must be positive.")
    if flags.warmup_ms < 0 or flags.neurons < 0:
        raise ValueError("--warmup_ms and --neurons must be nonnegative.")
    if not np.isfinite(flags.plot_core_radius) or flags.plot_core_radius <= 0:
        raise ValueError("--plot_core_radius must be finite and positive.")
    if flags.output_file and Path(flags.output_file).expanduser().exists():
        raise FileExistsError(flags.output_file)


def record_simulation(output_path, flags, neuron_ids, bmtk_ids, initial_state, step,
                      *, checkpoint="", memory_info=None):
    """Warm up and stream one trajectory; ``step(state, steps, record)`` advances it.

    ``step`` returns ``(state, spikes, voltages)`` with two-dimensional arrays
    in the supplied neuron order, or ``None`` arrays during warm-up. This small
    interface also permits testing the disk/continuity path without a GPU.
    ``memory_info`` optionally queries TensorFlow's GPU allocator statistics.
    """
    if flags.seq_len <= 0 or flags.duration_ms <= 0 or flags.warmup_ms < 0:
        raise ValueError("Chunk length and duration must be positive; warm-up must be nonnegative.")
    neuron_ids, bmtk_ids = np.asarray(neuron_ids), np.asarray(bmtk_ids)
    if neuron_ids.ndim != 1 or bmtk_ids.shape != neuron_ids.shape or not neuron_ids.size:
        raise ValueError("neuron_ids and bmtk_ids must be matching, nonempty vectors.")
    if len(np.unique(neuron_ids)) != len(neuron_ids):
        raise ValueError("neuron_ids must be unique.")
    order = np.argsort(neuron_ids)
    # main() records the neurons sorted, so their columns need no copy.
    columns = order if np.any(np.diff(neuron_ids) < 0) else slice(None)
    output_path = Path(output_path).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    settings = flags.flag_values_dict() if hasattr(flags, "flag_values_dict") else vars(flags)
    n_neurons = len(neuron_ids)
    # Every rollout covers whole time chunks, which are then written directly.
    time_chunk = min(math.gcd(flags.seq_len, 100), flags.duration_ms)
    neuron_chunk = min(n_neurons, max(1, 1024**2 // (2 * time_chunk)))
    with h5py.File(output_path, "x") as recording, \
            ThreadPoolExecutor(len(os.sched_getaffinity(0))) as pool:
        recording.create_dataset("neuron_ids", data=neuron_ids[order])
        recording.create_dataset("bmtk_ids", data=bmtk_ids[order])
        shape = (flags.duration_ms, n_neurons)
        options = dict(chunks=(time_chunk, neuron_chunk), compression="gzip",
                       compression_opts=1, shuffle=True)
        spikes = recording.create_dataset("spikes", shape, dtype=np.uint8, **options)
        voltages = recording.create_dataset("voltages", shape, dtype=np.float16, **options)
        voltages.attrs["units"] = "dimensionless normalized membrane voltage"
        voltages.attrs["reset"] = 0.0
        voltages.attrs["threshold"] = 1.0
        recording.attrs.update(
            data_dir=str(Path(flags.data_dir).expanduser().resolve()),
            checkpoint=checkpoint, dt_ms=1, warmup_ms=flags.warmup_ms,
            duration_ms=flags.duration_ms, completed_steps=0,
            flags_json=json.dumps(settings, default=str, sort_keys=True),
            time_reference="First saved sample is 1 ms after the end of warm-up.",
        )
        recording.flush()
        state = initial_state
        for start in range(0, flags.warmup_ms, flags.seq_len):
            state, _, _ = step(state, min(flags.seq_len, flags.warmup_ms - start), False)
        print(f"Warm-up complete ({flags.warmup_ms} ms). Recording to {output_path}", flush=True)
        recording_start = perf_counter()
        # The previous rollout's chunks deflate in the pool while the next one runs.
        pending, end = [], 0
        try:
            for start in range(0, flags.duration_ms, flags.seq_len):
                chunk_start = perf_counter()
                length = min(flags.seq_len, flags.duration_ms - start)
                state, chunk_spikes, chunk_voltages = step(state, length, True)
                if np.shape(chunk_spikes) != (length, n_neurons) or np.shape(chunk_voltages) != (length, n_neurons):
                    raise ValueError("Rollout outputs do not match the requested time/neuron shape.")
                write_chunks(recording, pending, end)
                pending = [
                    *deflate_chunks(pool, spikes, start, np.asarray(chunk_spikes, np.uint8)[:, columns]),
                    *deflate_chunks(pool, voltages, start, np.asarray(chunk_voltages, np.float16)[:, columns]),
                ]
                end = start + length
                now = perf_counter()
                try:
                    memory = memory_info() if memory_info is not None else None
                except ValueError:
                    memory = None  # The device may not support allocator statistics.
                peak = f"{memory['peak'] / 1024**3:.2f} GiB" if memory is not None else "N/A"
                print(
                    f"Recorded {end}/{flags.duration_ms} ms "
                    f"| chunk: {now - chunk_start:.2f} s "
                    f"| elapsed: {now - recording_start:.2f} s "
                    f"| TF peak: {peak}",
                    flush=True,
                )
        finally:
            # Also commits the last finished rollout when a later one fails.
            write_chunks(recording, pending, end)
    return state


def deflate_chunks(pool, dataset, start, rows):
    """Futures of the whole ``dataset`` chunks that ``rows`` fill from time ``start``.

    Each future deflates one chunk exactly as the dataset's shuffle and gzip
    filters would, so threads share the compression that HDF5 runs serially.
    HDF5 is not thread-safe, so the threads never touch ``dataset``.
    """
    chunks, dtype, level = dataset.chunks, dataset.dtype, dataset.compression_opts
    chunk_time, chunk_neurons = chunks

    def deflate(time, neuron):
        block = np.zeros(chunks, dtype)  # Edge chunks are padded.
        part = rows[time:time + chunk_time, neuron:neuron + chunk_neurons]
        block[:part.shape[0], :part.shape[1]] = part
        planes = block.view(np.uint8).reshape(-1, block.itemsize).T  # Byte shuffle.
        return (start + time, neuron), zlib.compress(planes.tobytes(), level)

    return [(dataset, pool.submit(deflate, time, neuron))
            for time in range(0, len(rows), chunk_time)
            for neuron in range(0, rows.shape[1], chunk_neurons)]


def write_chunks(recording, pending, end):
    """Write the ``deflate_chunks`` futures and commit the recording up to ``end``."""
    for dataset, chunk in pending:
        dataset.id.write_direct_chunk(*chunk.result())
    recording.attrs.modify("completed_steps", end)
    recording.flush()


def make_step(model, probability, dtype, *, seed=3000, run_session=0, strategy=None):
    """Build state-only warm-up and sequence recording calls sharing one cell."""
    import tensorflow as tf
    from v1_model_utils import models, tf_utils

    strategy = strategy or tf.distribute.get_strategy()
    rsnn = model.get_layer("rsnn")
    state_model = models.build_state_only_model(model, rsnn)
    sequence_model = models.build_sequence_and_state_model(model, rsnn)
    sequence_count = sequence_model._v1_sequence_output_count
    seeds = tf_utils.DistributedSeedHelper(seed + run_session,
                                          rsnn.cell.noise_stream, rsnn.cell.noise_seed)
    draw = 0

    def roll_out(probabilities, state, draw_id, record):
        seeds.advance_noise_seed()
        with tf.device("/CPU:0"):
            input_seed = seeds.spontaneous_seed(draw_id)
        uniforms = tf.random.stateless_uniform(
            tf.concat([[1], tf.shape(probabilities)], axis=0),
            seed=input_seed, dtype=tf.float32,
        )
        x = tf.cast(uniforms < tf.cast(probabilities, tf.float32), dtype)
        pair_weights = rsnn.cell.lgn_pair_weights()
        pair_scope = (
            nullcontext() if pair_weights is None
            else rsnn.cell.lgn_pair_weights_scope(pair_weights)
        )
        with pair_scope:
            if not record:
                return tuple(tf.nest.flatten(state_model([x, *state])))
            outputs = tf.nest.flatten(sequence_model([x, *state]))
        # Voltages are saved as float16: casting on the GPU halves the copy.
        return tuple(outputs[sequence_count:]), outputs[0][0], tf.cast(outputs[1][0], tf.float16)

    @tf.function
    def distributed_rollout(probabilities, state, draw_id, record):
        return strategy.run(roll_out, args=(probabilities, state, draw_id, record))

    def step(state, n_steps, record):
        nonlocal draw
        draw += 1
        outputs = distributed_rollout(probability[:n_steps], state,
                                      tf.constant(draw, tf.int32), record)
        if not record:
            return outputs, None, None
        new_state, spikes, voltages = outputs
        return new_state, spikes.numpy(), voltages.numpy()

    return step


def main(_):
    flags = absl_flags.FLAGS
    validate_flags(flags)
    os.environ.setdefault("TF_GPU_THREAD_MODE", "global")
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "0")
    from v1_model_utils.cuda_caches import configure_cuda_caches

    configure_cuda_caches()
    import tensorflow as tf
    from v1_model_utils import tf_utils

    # One trajectory always uses one device, even outside a scheduler allocation.
    gpus = tf.config.list_physical_devices("GPU")
    tf.config.set_visible_devices(gpus[:1], "GPU")
    tf_utils.configure_gpu_memory_growth()
    tf_utils.print_tensorflow_runtime_info()
    checkpoint = ""
    if flags.restore_from:
        checkpoint, _ = tf_utils.resolve_checkpoint_directory(flags)
        if checkpoint is None:
            raise FileNotFoundError("No valid checkpoint found in --restore_from.")

    from v1_model_utils import cuda_csr_recurrent, load_sparse, models
    from v1_model_utils import other_v1_utils, spatial_layout, training_utils
    from v1_model_utils.cuda_prebuild import prebuild_for_run
    import stim_dataset

    if flags.acceleration != "tensorflow":
        prebuild_for_run()
    tf_utils.configure_reproducibility(flags.seed)
    tf_utils.enable_tensorflow_optimizations(enabled=True, memory_optimizer=False)
    _, dtype = tf_utils.configure_policy_and_dtype(flags.dtype)
    strategy = tf_utils.create_distribution_strategy(single_gpu_strategy="mirrored")
    load_fn = load_sparse.cached_load_v1 if flags.caching else load_sparse.load_v1
    network, lgn_input, bkg_input = load_fn(flags, flags.neurons)
    layout = spatial_layout.build_layout(network, flags.neuron_layout)
    if not layout.is_identity:
        network, lgn_input, bkg_input = spatial_layout.apply_layout(
            layout, network, lgn_input, bkg_input)
    lgn_order = spatial_layout.build_lgn_row_order(lgn_input, flags.lgn_row_order)
    if not lgn_order.is_identity:
        lgn_input = spatial_layout.apply_lgn_row_order(lgn_order, lgn_input)
    edge_orders = None
    if cuda_csr_recurrent.DIRECT_CSR:
        network, lgn_input, bkg_input, edge_orders = spatial_layout.apply_csr_edge_order(
            network, lgn_input, bkg_input, max_delay=flags.max_delay)
    core_mask = other_v1_utils.isolate_core_neurons(
        network, radius=flags.plot_core_radius, data_dir=flags.data_dir)
    output_ids = np.flatnonzero(core_mask).astype(np.int32)
    # Canonical order: the model's per-step gather leaves the columns ready to save.
    output_ids = output_ids[np.argsort(layout.new_to_old[output_ids])]
    if not len(output_ids):
        raise ValueError("The requested core contains no neurons.")
    training_utils.require_int32_safe_rnn_output(1, flags.seq_len, 2 * len(output_ids))

    with strategy.scope():
        def build_model(model_dtype, compact=True):
            model = models.create_model(
                network, lgn_input, bkg_input, seq_len=flags.seq_len,
                n_input=flags.n_input, n_output=flags.n_output, dtype=model_dtype,
                batch_size=1, cue_duration=flags.cue_duration,
                input_weight_scale=flags.input_weight_scale,
                dampening_factor=flags.dampening_factor,
                recurrent_dampening_factor=flags.recurrent_dampening_factor,
                gauss_std=flags.gauss_std, lr_scale=flags.lr_scale,
                train_input=False, train_noise=False, train_recurrent=False,
                train_recurrent_per_type=False, neuron_output=flags.neuron_output,
                pseudo_gauss=flags.pseudo_gauss,
                surrogate_gradient="gaussian" if flags.pseudo_gauss else flags.surrogate_gradient,
                use_state_input=True, return_state=True, hard_reset=flags.hard_reset,
                integration_scheme=flags.integration_scheme, add_metric=False,
                max_delay=flags.max_delay, current_input=False, seed=flags.seed,
                acceleration=flags.acceleration, output_spike_dtype=tf.uint8 if compact else None,
                output_neuron_ids=output_ids if compact else None,
                return_voltage_sequences=True,
                track_voltage_penalty=flags.use_online_voltage_loss,
                voltage_penalty_mode=flags.voltage_penalty_mode,
                neuron_layout=layout, edge_orders=edge_orders, lgn_row_order=lgn_order,
            )
            model.build((1, flags.seq_len, flags.n_input))
            return model

        model = build_model(dtype)
        if checkpoint:
            # Restore the prefix resolved before construction, even if training
            # has since written a newer checkpoint into the same directory.
            checkpoint_dtype = tf_utils.infer_checkpoint_model_dtype(checkpoint)
            variable_dtype = tf_utils._variable_dtype_name(
                tf.as_dtype(variable.dtype).name for variable in model.variables
            )
            dtype_mismatch = checkpoint_dtype is not None and checkpoint_dtype != variable_dtype
            if dtype_mismatch and not flags.restore_runtime_dtype_cast:
                raise ValueError(
                    "Checkpoint/model dtype mismatch; enable --restore_runtime_dtype_cast."
                )
            ignored_variables = (
                {"projection_v1/kernel", "projection_v1/bias"}
                if len(output_ids) < network["n_nodes"] else ()
            )
            print(f"Restoring checkpoint from {checkpoint}", flush=True)
            if dtype_mismatch or ignored_variables:
                tf_utils.restore_model_with_runtime_dtype_cast(
                    target_model=model,
                    build_model_fn=lambda model_dtype: build_model(model_dtype, compact=False),
                    checkpoint_directory=checkpoint,
                    checkpoint_dtype_name=checkpoint_dtype or flags.dtype,
                    target_dtype_name=flags.dtype,
                    ignored_target_variable_names=ignored_variables,
                )
            else:
                tf_utils.restore_and_rebase(tf.train.Checkpoint(model=model), checkpoint, model)
        probability = stim_dataset.load_or_compute_spontaneous_lgn_probabilities(
            seq_len=flags.seq_len, n_input=flags.n_input, data_dir=flags.data_dir,
            bmtk_compat=flags.bmtk_compat_lgn, seed=flags.seed,
            output_dtype=tf.float32, lgn_backend=flags.lgn_backend,
        )
        step = make_step(model, probability, dtype, seed=flags.seed,
                         run_session=flags.run_session, strategy=strategy)
        initial_state = model.get_layer("rsnn").cell.zero_state(1, dtype=dtype)

    output_path = flags.output_file
    if not output_path:
        stamp = datetime.now(timezone.utc).strftime("prg_%Y%m%d_%H%M%S")
        output_path = REPOSITORY_ROOT / "PRG" / "results" / stamp / "simulation_data.hdf5"
    record_simulation(
        output_path, flags, layout.new_to_old[output_ids],
        np.asarray(network["tf_id_to_bmtk_id"])[output_ids], initial_state, step,
        checkpoint=checkpoint,
        memory_info=(lambda: tf.config.experimental.get_memory_info("GPU:0")) if gpus else None,
    )


if __name__ == "__main__":
    register_flags()
    app.run(main)
