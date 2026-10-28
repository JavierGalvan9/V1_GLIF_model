import matplotlib
matplotlib.use('agg')# to avoid GUI request on clusters
# ruff: noqa: E402  # CUDA cache configuration must precede TensorFlow imports.
import os

# Define the environment variables for optimal GPU performance
os.environ['TF_GPU_THREAD_MODE'] = 'global'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'  # before import tensorflow
# Prefer the default BFC allocator: cuda_malloc_async retained GPU pool memory
# and reached OOM earlier in this repository's workloads despite lower live usage.
# os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
# os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

from v1_model_utils.cuda_caches import configure_cuda_caches
configure_cuda_caches()  # before CUDA and XLA initialize

import absl
import numpy as np
import tensorflow as tf
from time import time
import logging
from v1_model_utils import tf_utils
from v1_model_utils import cuda_csr_recurrent, spatial_layout, training_utils
tf.get_logger().setLevel(logging.INFO)


@tf.function(jit_compile=True)
def count_spikes(spikes):
    """Total spike count; XLA fuses the cast, so no int64 copy of the spikes exists."""
    return tf.reduce_sum(tf.cast(spikes, tf.int64))


def main(_):
    flags = absl.app.flags.FLAGS
    training_utils.validate_n_input(flags.n_input)
    # Allow for memory growth (also to observe memory consumption)
    tf_utils.configure_gpu_memory_growth()
    # Display TensorFlow and CUDA runtime information for debugging and verification purposes.
    tf_utils.print_tensorflow_runtime_info()

    # Import TensorFlow-heavy project modules only after GPU memory is configured.
    from v1_model_utils import load_sparse, models
    from v1_model_utils.callbacks import OsiDsiCallbacks, printgpu
    import stim_dataset
    from v1_model_utils.cuda_prebuild import prebuild_for_run

    # Build the missing CUDA operator libraries together rather than one by one on first use.
    prebuild_for_run()

    tf_utils.configure_reproducibility(flags.seed)
    # Configure TensorFlow optimization options that have proven to be beneficial for our model and training setup. These options can improve the performance of the model by optimizing the computational graph and memory usage.
    # Grappler's memory optimizer is off: on large networks (~1M neurons) it overestimates the peak of the
    # per-chunk graph and swaps tensors to host memory every timestep (13.3 vs 8.9 s per angle). Inference
    # keeps no gradient tape, so those swaps never save memory.
    tf_utils.enable_tensorflow_optimizations(enabled=True, memory_optimizer=False)

    # Create the tensorflow datafile for the network
    os.makedirs(os.path.join(flags.data_dir, 'tf_data'), exist_ok=True)

    # Load the checkpoint model if it exists
    flag_str, logdir, current_epoch = tf_utils.configure_run_paths(
        flags,
        task_name=flags.task_name,
    )

    # Configure the dtype policy
    mixed_precision, dtype = tf_utils.configure_policy_and_dtype(flags.dtype)

    strategy = tf_utils.create_distribution_strategy(
        single_gpu_strategy="mirrored",
    )

    per_replica_batch_size = flags.n_trials_per_angle #flags.batch_size
    global_batch_size = per_replica_batch_size * strategy.num_replicas_in_sync
    print(f'Per replica batch size: {per_replica_batch_size}')
    print(f'Global batch size: {global_batch_size}\n')

    # Define 2 outputs that correspond to having more cues top or bottom
    # Note that two different output conventions can be used:
    # 1) Linear readouts from all neurons in the model (softmax)
    # 2) Selecting a population of neurons that report a binary decision
    # with high firing rate (flag --neuron_output)
    # n_output = 2

    # Load data and select appropriate number of neurons and inputs
    t0 = time()
    if flags.caching:
        load_fn = load_sparse.cached_load_v1
    else:
        load_fn = load_sparse.load_v1

    network, lgn_input, bkg_input = load_fn(flags, flags.neurons, flag_str=flag_str)
    print(f"Model files loading: {time()-t0:.2f} seconds\n")

    # Renumber neurons for CSR cache locality, then reorder the edges into the
    # order the kernels index weights by. Checkpoints and every per-neuron
    # artefact this script writes stay in the canonical order and are translated
    # at those boundaries.
    neuron_layout = spatial_layout.build_layout(network, flags.neuron_layout)
    if not neuron_layout.is_identity:
        network, lgn_input, bkg_input = spatial_layout.apply_layout(
            neuron_layout, network, lgn_input, bkg_input
        )
        print(f"Neuron layout: {neuron_layout.mode}")
    # Renumber LGN rows so consecutive forward blocks drive nearby V1
    # territory. Orthogonal to the neuron layout, and applied before the CSR
    # edge order so that permutation carries it; the kernels are unchanged.
    lgn_row_order = spatial_layout.build_lgn_row_order(
        lgn_input, flags.lgn_row_order
    )
    if not lgn_row_order.is_identity:
        lgn_input = spatial_layout.apply_lgn_row_order(lgn_row_order, lgn_input)
        print(f"LGN row order: {lgn_row_order.mode}")

    edge_orders = None
    if cuda_csr_recurrent.DIRECT_CSR:
        network, lgn_input, bkg_input, edge_orders = (
            spatial_layout.apply_csr_edge_order(
                network, lgn_input, bkg_input, max_delay=flags.max_delay
            )
        )

    if flags.track_core_only:
        from v1_model_utils import other_v1_utils

        core_mask = other_v1_utils.isolate_core_neurons(
            network,
            radius=flags.plot_core_radius,
            data_dir=flags.data_dir,
        )
        output_neuron_ids = np.flatnonzero(core_mask).astype(np.int32)
    else:
        core_mask = None
        output_neuron_ids = None

    # Define the scope in which the model training will be executed
    with strategy.scope():
        t0 = time()

        def build_model_with_dtype(model_dtype, compact_output=flags.track_core_only):
            temp_model = models.create_model(
                network,
                lgn_input,
                bkg_input,
                seq_len=flags.seq_len,
                n_input=flags.n_input,
                n_output=flags.n_output,
                cue_duration=flags.cue_duration,
                dtype=model_dtype,
                batch_size=per_replica_batch_size,
                input_weight_scale=flags.input_weight_scale,
                dampening_factor=flags.dampening_factor,
                recurrent_dampening_factor=flags.recurrent_dampening_factor,
                gauss_std=flags.gauss_std,
                lr_scale=flags.lr_scale,
                train_input=False,
                train_noise=False,
                compute_lgn_activity_gradient=flags.compute_lgn_activity_gradient,
                compute_bkg_activity_gradient=flags.compute_bkg_activity_gradient,
                train_recurrent=False,
                train_recurrent_per_type=False,
                neuron_output=flags.neuron_output,
                pseudo_gauss=flags.pseudo_gauss,
                surrogate_gradient=(
                    "gaussian" if flags.pseudo_gauss else flags.surrogate_gradient
                ),
                use_state_input=True,
                return_state=True,
                hard_reset=flags.hard_reset,
                integration_scheme=flags.integration_scheme,
                add_metric=False,
                max_delay=flags.max_delay,  # 0 = auto-compute from SONATA data
                current_input=flags.current_input,
                use_dummy_state_input=False,
                seed=flags.seed,
                acceleration=flags.acceleration,
                # The recurrent state stays in compute dtype; only the exposed,
                # already-stacked compact spike sequence is quantized for inference.
                output_spike_dtype=(
                    tf.uint8 if compact_output else None
                ),
                output_neuron_ids=output_neuron_ids if compact_output else None,
                return_voltage_sequences=flags.track_voltage,
                # Checkpoints locate the cell by its position among the model's
                # layers, and the penalty state adds an input layer before it.
                track_voltage_penalty=flags.use_online_voltage_loss,
                voltage_penalty_mode=flags.voltage_penalty_mode,
                neuron_layout=neuron_layout,
                edge_orders=edge_orders,
                lgn_row_order=lgn_row_order,
            )
            temp_model.build((per_replica_batch_size, flags.seq_len, flags.n_input))
            return temp_model

        # Build the model
        model = build_model_with_dtype(dtype)
        print(f"Model built in {time()-t0:.2f} s\n")

        # Store the initial model variables that are going to be trained
        # model_variables_dict = {'Initial': {var.name: var.numpy().astype(np.float16) for var in model.trainable_variables}}
        model_variables_dict = {'Initial': {var.name: var.numpy() for var in model.trainable_variables}} # float32 dtype

        # Restore model from a checkpoint if it exists.
        checkpoint, logdir, current_epoch = tf_utils.restore_evaluation_checkpoint(
            flags,
            model,
            lambda model_dtype: build_model_with_dtype(
                model_dtype, compact_output=False
            ),
            logdir,
            current_epoch=current_epoch,
            result_name="OSI/DSI",
            evaluate_untrained=flags.evaluate_untrained,
            # The core-only output narrows the (unused) readout to the core.
            runtime_cast_ignored_variables={
                "projection_v1/kernel",
                "projection_v1/bias",
            } if flags.track_core_only and len(output_neuron_ids) < network["n_nodes"] else (),
        )

        # model_variables_dict['Best'] =  {var.name: var.numpy().astype(np.float16) for var in model.trainable_variables}
        model_variables_dict['Best'] =  {var.name: var.numpy() for var in model.trainable_variables} # float32 dtype
        print("Model variables stored in dictionary\n")

        # Build the sequence-and-state model used by the optimized Keras RNN loop.
        rsnn_layer = model.get_layer('rsnn')
        sequence_and_state_model = models.build_sequence_and_state_model(
            model, rsnn_layer, name="osi_dsi_sequence_and_state_extractor"
        )
        n_sequence_outputs = sequence_and_state_model._v1_sequence_output_count
        # Each --seq_len chunk is simulated in rollout chunks of
        # --gradient_checkpoint_chunk_size steps, one model call each, and every
        # rollout chunk's spikes leave the GPU as soon as they exist. The
        # per-replica batch here is --n_trials_per_angle, which is exactly the
        # knob raised to tighten the OSI/DSI statistics. Above 2**31 elements the
        # stock Keras RNN output assembly (TensorArray.stack) of one model call
        # returns the wrong spikes with no error at all, so refuse it outright.
        # The call stacks the tracked neurons' spikes, and their voltages after
        # them with --track_voltage.
        rollout_chunk_size = (
            min(flags.gradient_checkpoint_chunk_size, flags.seq_len)
            if flags.segmented_rollout else flags.seq_len
        )
        output_width = network["n_nodes"] if output_neuron_ids is None else len(output_neuron_ids)
        training_utils.require_int32_safe_rnn_output(
            per_replica_batch_size, rollout_chunk_size,
            output_width * (2 if flags.track_voltage else 1),
        )
        print(f"RNN rollout in chunks of {rollout_chunk_size} steps.")

        # The gray-screen warm-up only needs the final state, so the state-only
        # model keeps it from building a [batch, seq_len, n_neurons] sequence
        # that is discarded immediately -- and from tripping the same limit.
        try:
            state_model = models.build_state_only_model(model, rsnn_layer)
        except Exception as e:
            state_model = None
            print(
                f"Warning: failed to build state-only model ({e}); "
                "using the full sequence model for the gray state."
            )

        # zero_state = rsnn_layer.cell.zero_state(per_replica_batch_size, dtype=dtype)
        # state_variables = tf.nest.map_structure(lambda a: tf.Variable(
        #     a, trainable=False, synchronization=tf.VariableSynchronization.ON_READ
        # ), zero_state)

        # Load the spontaneous probabilities once
        spontaneous_prob = stim_dataset.load_or_compute_spontaneous_lgn_probabilities(
            seq_len=flags.seq_len,
            n_input=flags.n_input,
            data_dir=flags.data_dir,
            bmtk_compat=flags.bmtk_compat_lgn,
            seed=flags.seed,
            output_dtype=tf.float32,  # sampled in float32 (generate_spontaneous_spikes)
            lgn_backend=flags.lgn_backend,
        )
        # Handle the random seed for spontaneous spike generation and BKG noise.
        # Offset by run_session so different evaluation sessions of the SAME network
        # (which must all use the same --seed to rebuild an identical network for
        # checkpoint restoration) still get distinct noise streams.
        seed_helper = tf_utils.DistributedSeedHelper(
            flags.seed + flags.run_session,
            rsnn_layer.cell.noise_stream,
            rsnn_layer.cell.noise_seed,
        )

        # LGN firing rates to the different angles
        DG_angles = np.arange(0, 360, 45)

        # Load the OSI/DSI LGN firing probabilites dataset once
        lgn_firing_probabilities_dict = stim_dataset.load_or_compute_osi_dsi_lgn_probabilities(
            seq_len=flags.seq_len,
            spont_duration=flags.spont_duration,
            evoked_duration=flags.evoked_duration,
            n_input=flags.n_input,
            data_dir=flags.data_dir,
            rotation=flags.rotation,
            seed=flags.seed,
            output_dtype=tf.float32,  # sampled in float32 (generate_spontaneous_spikes)
            angles=DG_angles,
            strategy=strategy,
            lgn_backend=flags.lgn_backend,
        )

    def rollout_pieces(probability):
        """``probability`` [steps, n_input] split into its rollout chunks."""
        return [probability[start:start + rollout_chunk_size]
                for start in range(0, probability.shape[0], rollout_chunk_size)]

    def roll_out(pieces, state, first_draw, on_chunk=None):
        """Final state of a rollout driven by the LGN spike probabilities
        ``pieces`` (rollout_pieces), in rollout chunks whose inputs are the spontaneous
        draws ``first_draw``, ``first_draw + 1``, ... Each chunk's input is drawn and cast
        only after the previous chunk's state exists: the draws do not depend on
        the state, so in a graph they would otherwise all run, and hold their
        memory, up front. ``on_chunk(start, x, sequences)`` receives every
        chunk's bool input and sequence outputs; without it only the final state
        is computed."""
        seed_helper.advance_noise_seed()
        state = tuple(state)
        # One computation of the LGN pair sums for all the chunks.
        with rsnn_layer.cell.lgn_pair_weights_scope():
            for draw, piece in enumerate(pieces):
                start = draw * rollout_chunk_size
                with tf.control_dependencies(state[:1]):
                    x = generate_spontaneous_spikes(piece, per_replica_batch_size, first_draw + draw)
                inputs = (tf.cast(x, dtype), state)
                if on_chunk is None and state_model is not None:
                    state = tuple(tf.nest.flatten(state_model(inputs)))
                    continue
                outputs = tf.nest.flatten(sequence_and_state_model(inputs))
                state = tuple(outputs[n_sequence_outputs:])
                if on_chunk is not None:
                    on_chunk(start, x, outputs[:n_sequence_outputs])
        return state

    # Generate spontaneous spikes efficiently
    @tf.function
    def generate_spontaneous_spikes(probability, batch_size, draw):
        """Spikes [batch_size, *probability.shape]. The probabilities and uniforms are
        float32 whatever `dtype` is: float16 uniforms take only 1024 values, so
        P(u < p) would be ceil(1024 p) / 1024 (+14% spikes at spontaneous rates).
        """
        # On the GPU, the seed's fold-ins would each end in a device-to-host
        # read (int32 results live in host memory) that waits for the GPU.
        with tf.device('/CPU:0'):
            seed = seed_helper.spontaneous_seed(draw)
        random_uniform = tf.random.stateless_uniform(
            tf.concat([[batch_size], tf.shape(probability)], axis=0),
            seed=seed,
            dtype=tf.float32,
        )
        return tf.less(random_uniform, tf.cast(probability, tf.float32))

    def generate_gray_state(first_draw):
        zero_state = rsnn_layer.cell.zero_state(per_replica_batch_size, dtype=dtype)
        return roll_out(rollout_pieces(spontaneous_prob), zero_state, first_draw)

    @tf.function
    def distributed_generate_gray_state(first_draw):
        return strategy.run(generate_gray_state, args=(first_draw,))

    # The LGN draws are numbered on the host (DistributedSeedHelper.spontaneous_seed).
    draws_per_chunk = len(range(0, flags.seq_len, rollout_chunk_size))

    # Match the original estimator semantics: initialize the gray-screen state
    # once, outside per-angle timing, and reuse it as an immutable starting state.
    gray_state = distributed_generate_gray_state(tf.constant(1))
    next_draw = 1 + draws_per_chunk

    # Define the callbacks for the OSI/DSI analysis
    stim_duration = flags.spont_duration + flags.evoked_duration
    sim_duration = int(np.ceil(stim_duration/flags.seq_len)) * flags.seq_len
    post_delay = sim_duration - stim_duration
    tracked_runtime_ids = (
        np.arange(network['n_nodes'])
        if output_neuron_ids is None
        else output_neuron_ids
    )
    canonical_neuron_ids = neuron_layout.new_to_old[tracked_runtime_ids]
    callbacks = OsiDsiCallbacks(network, lgn_input, bkg_input, flags, logdir, current_epoch=current_epoch,
                                pre_delay=flags.spont_duration, post_delay=post_delay, model_variables_init=model_variables_dict,
                                neuron_ids=canonical_neuron_ids)

    chunk_size = flags.seq_len
    # Determine which neurons to track based on track_core_only flag
    if flags.track_core_only:
        n_tracked_neurons = np.sum(core_mask)
        print(f"Tracking only core neurons: {n_tracked_neurons} out of {network['n_nodes']} total neurons")
    else:
        core_mask = None
        n_tracked_neurons = network['n_nodes']
        print(f"Tracking all neurons: {n_tracked_neurons}")

    print(f"Resolved model acceleration: {rsnn_layer.cell.resolved_acceleration}")

    spikes = np.zeros((flags.n_trials_per_angle, len(DG_angles), sim_duration, n_tracked_neurons), dtype=np.uint8)
    voltages = (
        np.zeros((sim_duration, n_tracked_neurons), dtype=np.float16)
        if flags.track_voltage
        else None
    )
    if voltages is not None:
        print(
            "Voltage tracking enabled. Allocating "
            f"{voltages.nbytes / (1024**3):.2f} GiB for the first trial."
        )

    def store_spikes(chunk_spikes, angle_id, start):
        """Host sink: one rollout chunk's spikes into the spike array."""
        spikes[:, angle_id, start:start + chunk_spikes.shape[1]] = chunk_spikes
        return np.int32(0)

    def store_voltage(chunk_voltage, start):
        """Host sink: the first trial's voltage of one rollout chunk."""
        voltages[start:start + chunk_voltage.shape[0]] = chunk_voltage
        return np.int32(0)

    def to_host(sink, *args):
        # Stateless, so that no later stateful op waits for the host; the
        # caller returns the token so the sink is not pruned.
        return tf.numpy_function(sink, args, tf.int32, stateful=False)

    def simulate_chunk(pieces, state, first_draw, angle_id, start, store_voltages):
        """One --seq_len chunk of every trial. Each rollout chunk's spikes (and,
        with ``store_voltages``, the first trial's voltage) go to
        the host sinks as soon as the model produces them: the copies and host
        writes overlap the next rollout chunks, and only the spike count and the
        first trial's input stay on the GPU."""
        counts, stored, first_trial = [], [], []

        def on_chunk(chunk_start, x, sequences):
            first_trial.append(x[0])
            chunk_spikes = tf.cast(sequences[0], tf.uint8)
            counts.append(count_spikes(chunk_spikes))
            stored.append(to_host(store_spikes, chunk_spikes, angle_id, start + chunk_start))
            if store_voltages:
                stored.append(to_host(store_voltage, sequences[1][0], start + chunk_start))

        state = roll_out(pieces, state, first_draw, on_chunk)
        return state, tf.add_n(counts), tf.concat(first_trial, axis=0), tf.stack(stored)

    @tf.function
    def distributed_simulate_chunk(pieces, state, first_draw, angle_id, start, store_voltages):
        return strategy.run(
            simulate_chunk, args=(pieces, state, first_draw, angle_id, start, store_voltages)
        )

    for angle_id, angle in enumerate(DG_angles):
        print(f'Running angle {angle}...')
        # load LGN firing rates for the given angle and calculate spiking probability
        lgn_prob = lgn_firing_probabilities_dict[angle]

        t0 = time()
        continuing_state = gray_state
        chunk_counts, first_trial_chunks = [], []
        for start in range(0, sim_duration, chunk_size):
            # Host tensors: the call copies each piece to the GPU while the
            # previous one is simulated.
            with tf.device('/CPU:0'):
                pieces = [tf.constant(piece)
                          for piece in rollout_pieces(lgn_prob[start:start + chunk_size])]
            continuing_state, count, first_trial, _ = distributed_simulate_chunk(
                pieces, continuing_state, tf.constant(next_draw),
                tf.constant(angle_id), tf.constant(start),
                voltages is not None and angle_id == 0,
            )
            next_draw += draws_per_chunk
            chunk_counts.append(count)
            if angle_id == 0:
                first_trial_chunks.append(first_trial)

        # Track GPU memory and inference time
        average_rate = int(tf.add_n(chunk_counts)) / spikes[:, angle_id].size
        callbacks.track_performance(t0, average_rate=average_rate, gpu_id=0)

        print(f'    Angle processing time: {time() - t0:.2f}s')
        for gpu_id in range(len(strategy.extended.worker_devices)):
            printgpu(gpu_id=gpu_id)

        if angle_id == 0:
            # Raster plot for 0 degree orientation
            callbacks.single_trial_callbacks(
                lgn_row_order.to_runtime(tf.concat(first_trial_chunks, axis=0).numpy()[np.newaxis]),
                spikes[:, 0, :, :],
                y=angle,
            )
            if voltages is not None:
                voltage_path = os.path.join(logdir, "voltage_trace.npy")
                # Columns follow the runtime numbering; save them canonically so
                # the trace matches a canonical run neuron for neuron.
                np.save(
                    voltage_path,
                    spatial_layout.tracked_to_canonical(
                        neuron_layout, voltages, output_neuron_ids, axis=1
                    ),
                )
                print(f"Voltage traces saved to {voltage_path}")
        if not flags.calculate_osi_dsi:
            break

    # Save the performance metrics
    callbacks.save_inference_metrics()

    # Do the OSI/DSI analysis
    if flags.calculate_osi_dsi:
        callbacks.osi_dsi_analysis(spikes, DG_angles)

    # # Save the spikes in a pickle file
    # from v1_model_utils import other_v1_utils
    # core_mask = other_v1_utils.isolate_core_neurons(network, radius=flags.plot_core_radius, data_dir=flags.data_dir)
    # spikes_plot = spikes[:, :, :, core_mask]
    # with open(f'v1_core_osi_spikes.pkl', 'wb') as f:
    #     pkl.dump(spikes_plot, f)


if __name__ == '__main__':
    _data_dir = 'GLIF_network'
    _results_dir = 'Simulation_results'

    absl.app.flags.DEFINE_string('task_name', 'drifting_gratings_firing_rates_distr' , '')
    absl.app.flags.DEFINE_string('data_dir', _data_dir, '')
    absl.app.flags.DEFINE_string('results_dir', _results_dir, '')
    absl.app.flags.DEFINE_string('restore_from', 'Intermediate_checkpoints', '')
    absl.app.flags.DEFINE_boolean('evaluate_untrained', False, 'Evaluate the seeded model before training, without restoring a checkpoint.')
    absl.app.flags.DEFINE_string('comment', '', '')
    absl.app.flags.DEFINE_string('delays', '100,0', '')
    absl.app.flags.DEFINE_string('scale', '2,2', '')
    absl.app.flags.DEFINE_string('optimizer', 'exp_adam', '')
    absl.app.flags.DEFINE_enum(
        'dtype', 'float32', ['float16', 'float32'], 'Model numeric dtype.'
    )
    absl.app.flags.DEFINE_enum(
        'acceleration',
        'auto',
        ['auto', 'cuda', 'tensorflow'],
        'Model acceleration mode.',
    )
    absl.app.flags.DEFINE_float('learning_rate', .005, '')
    absl.app.flags.DEFINE_string('lr_schedule', 'none',
        "Learning-rate schedule. Options: 'none' or 'warmup_cosine'.",
    )
    absl.app.flags.DEFINE_float('lr_warmup_start_lr', 0.08,
        'Warmup start learning rate (used when lr_schedule=warmup_cosine).',
    )
    absl.app.flags.DEFINE_float('lr_warmup_target_lr', 0.04,
        'Warmup end learning rate (used when lr_schedule=warmup_cosine).',
    )
    absl.app.flags.DEFINE_integer('lr_warmup_steps', 120,
        'Number of linear warmup steps (used when lr_schedule=warmup_cosine).',
    )
    absl.app.flags.DEFINE_float('lr_cosine_min_lr', 0.001,
        'Final cosine learning rate floor (used when lr_schedule=warmup_cosine).',
    )
    absl.app.flags.DEFINE_integer('lr_cosine_steps', 880,
        'Number of cosine decay steps after warmup (used when lr_schedule=warmup_cosine).',
    )
    absl.app.flags.DEFINE_float('rate_cost', 10000., '')
    absl.app.flags.DEFINE_float('voltage_cost', 1., '')
    absl.app.flags.DEFINE_float('sync_cost', 1.5, '')
    absl.app.flags.DEFINE_float('osi_cost', 20., '')
    absl.app.flags.DEFINE_float('annulus_loss_weight', 0.1, '')
    absl.app.flags.DEFINE_string('osi_loss_method', 'crowd_osi', '')
    absl.app.flags.DEFINE_float('osi_loss_subtraction_ratio', 1., '')
    absl.app.flags.DEFINE_float('dampening_factor', 0.05, '')
    absl.app.flags.DEFINE_float('recurrent_dampening_factor', 1.0, '')
    absl.app.flags.DEFINE_float('input_weight_scale', 1., '')
    absl.app.flags.DEFINE_float('gauss_std', 0.28, '')
    absl.app.flags.DEFINE_float('recurrent_weight_regularization', 0., '')
    absl.app.flags.DEFINE_string('recurrent_weight_regularizer_type', "emd", '')
    absl.app.flags.DEFINE_string('voltage_penalty_mode', 'range', '')
    absl.app.flags.DEFINE_float('lr_scale', 1., '')
    absl.app.flags.DEFINE_float('temporal_f', 2., '')
    absl.app.flags.DEFINE_float('max_time', -1, '')

    absl.app.flags.DEFINE_integer('n_runs', 1, '')
    absl.app.flags.DEFINE_integer('run_session', 0, '')
    absl.app.flags.DEFINE_integer('n_epochs', 75, '')
    absl.app.flags.DEFINE_integer('batch_size', 1, '')
    absl.app.flags.DEFINE_integer('neurons', 0, '')  # 0 to take all neurons
    absl.app.flags.DEFINE_integer('steps_per_epoch', 25, '')# EA and garret dose not need this many but pure classification needs 781 = int(50000/64)
    absl.app.flags.DEFINE_integer('val_steps', 1, '')# EA and garret dose not need this many but pure classification needs 156 = int(10000/64)
    # number of LGN filters in visual space (input population)
    absl.app.flags.DEFINE_integer('n_input', 17400, '')
    absl.app.flags.DEFINE_integer("n_output", 2, "")
    absl.app.flags.DEFINE_integer('seq_len', 500, '')
    absl.app.flags.DEFINE_integer('spont_duration', 2000, '')
    absl.app.flags.DEFINE_integer('evoked_duration', 2000, '')
    absl.app.flags.DEFINE_integer('seed', 3000, '')
    absl.app.flags.DEFINE_integer('neurons_per_output', 16, '')
    absl.app.flags.DEFINE_integer('max_delay', 0, 'Maximum synaptic delay in ms (0 = auto from data)')

    absl.app.flags.DEFINE_integer('n_cues', 3, '')
    absl.app.flags.DEFINE_integer('recall_duration', 40, '')
    absl.app.flags.DEFINE_integer('cue_duration', 40, '')
    absl.app.flags.DEFINE_integer('interval_duration', 40, '')
    absl.app.flags.DEFINE_integer('examples_in_epoch', 32, '')
    absl.app.flags.DEFINE_integer('validation_examples', 16, '')
    absl.app.flags.DEFINE_integer('n_trials_per_angle', 10, '')
    absl.app.flags.DEFINE_integer('fano_samples', 512, '')

    # absl.app.flags.DEFINE_boolean('float16', False, '')
    absl.app.flags.DEFINE_boolean('caching', True, '')
    absl.app.flags.DEFINE_boolean(
        'compress_model_cache', False,
        'Store constructed V1 model caches with LZ4 compression (.pkl.lz4).',
    )
    absl.app.flags.DEFINE_boolean('core_only', False, '')
    absl.app.flags.DEFINE_boolean('core_loss', False, '')
    absl.app.flags.DEFINE_boolean('all_neuron_rate_loss', False, '')  # whethre you want to enforce rate loss to all neurons
    absl.app.flags.DEFINE_float('loss_core_radius', 400.0, '') # 0 is not using core loss
    absl.app.flags.DEFINE_float('plot_core_radius', 400.0, '') # 0 is not using core plot

    absl.app.flags.DEFINE_boolean('hard_reset', True, 'Clamp voltage to reset during refractoriness; must match training.')
    absl.app.flags.DEFINE_enum(
        "integration_scheme", "exact", ["exact", "euler"],
        "Membrane step integrator; must match the value used for training.")
    absl.app.flags.DEFINE_boolean('train_input', False, '')
    absl.app.flags.DEFINE_boolean('train_noise', False, '')
    absl.app.flags.DEFINE_boolean('compute_lgn_activity_gradient', False, '')
    absl.app.flags.DEFINE_boolean('compute_bkg_activity_gradient', False, '')
    absl.app.flags.DEFINE_boolean('train_recurrent', True, '')
    absl.app.flags.DEFINE_boolean('train_recurrent_per_type', False, '')
    # absl.app.flags.DEFINE_boolean('train_recurrent_per_type', False, '')
    absl.app.flags.DEFINE_boolean('connected_selection', True, '')
    absl.app.flags.DEFINE_boolean('neuron_output', False, '')
    # absl.app.flags.DEFINE_boolean('hard_only', False, '')
    absl.app.flags.DEFINE_boolean('pseudo_gauss', False, '')
    absl.app.flags.DEFINE_enum(
        'surrogate_gradient',
        'gaussian',
        ['triangular', 'gaussian', 'slayer'],
        'Surrogate derivative used for spike generation.',
    )
    absl.app.flags.DEFINE_boolean("bmtk_compat_lgn", True, "")
    absl.app.flags.DEFINE_enum(
        "lgn_backend", "grating", ["grating", "movie", "tensorflow"],
        "LGN implementation: 'grating' (the exact drifting-grating CUDA op; other stimuli use the general CUDA movie op), 'movie' (the general CUDA movie op for every stimulus) or 'tensorflow' (the TensorFlow filters).",
    )
    absl.app.flags.DEFINE_boolean("average_grad_for_cell_type", False, "")
    absl.app.flags.DEFINE_boolean("reset_every_step", False, "")
    absl.app.flags.DEFINE_boolean("spontaneous_training", False, "")
    absl.app.flags.DEFINE_boolean("sequential_stimuli", False, "")
    absl.app.flags.DEFINE_boolean(
        "use_online_voltage_loss", True,
        "Must match the training run: its voltage-penalty state shifts the "
        "checkpointed layer positions, so a mismatch leaves the model unrestored.",
    )
    absl.app.flags.DEFINE_boolean("debug_gradients", False, "")
    absl.app.flags.DEFINE_boolean("calculate_osi_dsi", True, "")
    absl.app.flags.DEFINE_boolean('random_weights', False, '')
    absl.app.flags.DEFINE_boolean('uniform_weights', False, '')
    absl.app.flags.DEFINE_boolean('restore_runtime_dtype_cast', True, 'Enable in-memory checkpoint dtype conversion when checkpoint and requested model dtype differ.')
    absl.app.flags.DEFINE_boolean("current_input", False, "")
    absl.app.flags.DEFINE_boolean(
        "segmented_rollout",
        True,
        "Simulate each --seq_len chunk in model calls of --gradient_checkpoint_chunk_size "
        "steps, whose spikes go to the host as each call ends; off makes one call per "
        "--seq_len chunk. A call whose stacked output passes 2**31 elements is refused.",
    )
    absl.app.flags.DEFINE_integer(
        "gradient_checkpoint_chunk_size",
        25,
        "Steps per model call of the segmented rollout.",
    )
    absl.app.flags.DEFINE_boolean("track_core_only", False, "Track spikes only from core neurons to reduce memory usage")
    absl.app.flags.DEFINE_boolean("track_voltage", False, "Track and save membrane voltage traces during simulation")

    absl.app.flags.DEFINE_string("rotation", "ccw", "")
    absl.app.flags.DEFINE_string('ckpt_dir', '', '')
    absl.app.flags.DEFINE_string('neuropixels_df', 'Neuropixels_data/v1_OSI_DSI_DF.csv', 'File name of the Neuropixels DataFrame for OSI/DSI analysis.')
    absl.app.flags.DEFINE_enum(
        "lgn_row_order",
        "original",
        list(spatial_layout.LGN_ROW_ORDERS),
        "Runtime LGN row numbering. 'retinotopic' ranks LGN rows by the mean "
        "V1 index they drive, so consecutive forward blocks scatter into "
        "nearby postsynaptic memory. Measured worth about 2% of the LGN "
        "forward kernel, and only once the postsynaptic currents array "
        "outgrows L2 - roughly 0.09% of a batch-64 step, and nothing at batch "
        "32 - so it stays off by default. It exists because a tensor-row WMMA "
        "kernel would make the same ordering pay through fragment reuse "
        "instead of cache capacity, which does not depend on the batch. "
        "Checkpoints stay in the canonical row order either way.",
    )
    absl.app.flags.DEFINE_enum(
        "neuron_layout",
        "morton",
        list(spatial_layout.LAYOUTS),
        "Runtime neuron numbering. 'morton' renumbers neurons along a 2-D "
        "space-filling curve, which takes 11% off the recurrent forward kernel "
        "and 21% off the LGN forward kernel; inference has no backward pass, so "
        "those kernels are a larger share of the step here than in training. It "
        "reorders FP16 atomic accumulation, so results shift by about the "
        "run-to-run noise floor; use 'canonical' for bit-comparability with "
        "historical runs. Saved per-neuron outputs stay canonical either way.",
    )

    absl.app.run(main)
