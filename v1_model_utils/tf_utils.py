import json
import os
import re
import pickle as pkl
import random
import ctypes.util
import gc
from v1_model_utils.toolkit import get_random_identifier
import numpy as np
import tensorflow as tf
from packaging import version


def print_tensorflow_runtime_info():
    """Print and return TensorFlow/CUDA runtime information."""
    build_info = tf.sysconfig.get_build_info()
    info = {
        "cuda_version": build_info.get("cuda_version"),
        "cudnn_version": build_info.get("cudnn_version"),
        "tensorflow_version": tf.__version__,
        "cudart_path": ctypes.util.find_library("cudart"),
    }

    print("--- CUDA version: ", info["cuda_version"])
    print("--- CUDNN version: ", info["cudnn_version"])
    print("--- TensorFlow version: ", info["tensorflow_version"])
    print("--- CUDA Library path: ", info["cudart_path"])
    return info


def configure_reproducibility(seed):
    """Set NumPy/Python/TensorFlow seeds for repeatability."""
    seed = int(seed)
    tf.keras.utils.set_random_seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    random.seed(seed)
    print(f"Reproducibility seed: {seed}")


def configure_gpu_memory_growth():
    """Enable memory growth on all visible GPUs."""
    physical_devices = tf.config.list_physical_devices("GPU")
    for dev in physical_devices:
        try:
            tf.config.experimental.set_memory_growth(dev, True)
            print(f"Memory growth enabled for device {dev}")
        except (RuntimeError, ValueError) as exc:
            print(f"Could not enable memory growth for device {dev}: {exc}")
    print("- Num GPUs Available: ", len(physical_devices), "\n")
    return physical_devices


def enable_tensorflow_optimizations(enabled=True):
    """Enable TensorFlow graph optimizations for training and inference."""
    # tf.config.optimizer.set_experimental_options(
    #     {"cudnn_use_autotune": bool(enabled)}
    # )
    if enabled:
        tf.config.optimizer.set_experimental_options({
            "cudnn_use_autotune": True,   # not useful
            "layout_optimizer": True,     # good
            "constant_folding": True,     # good
            "shape_optimization": True, # barely good
            "remapping": True, # slightly faster and less memory without it
            "arithmetic_optimization": True, # really good for speed
            "dependency_optimization": True, # good
            "loop_optimization": True, #good
            "function_optimization": True, # good
            "scoped_allocator_optimization": True, # good
            "pin_to_host_optimization": False, # trigger errors in multi area model and when using fp32
            "implementation_selector": True, # good
            "auto_parallel": True, # good
            # "disable_model_pruning": False, # needs to be false to allow pruning of training subgraph
            "min_graph_nodes": 0, # good to set to 0 to allow optimization of small subgraphs, which is important for our model with many small ops
        })
        ## tf.config.optimizer.set_jit("autoclustering") # does not work here because of Ragged/Sparse custom-gradient path


def configure_policy_and_dtype(dtype_name):
    """
    Set TensorFlow mixed-precision policy and return the matching tf.DType.

    Supported values: float16, bfloat16, float32
    """
    dtype_name = tf.dtypes.as_dtype(dtype_name).name

    if dtype_name == "float16":
        policy_name = "mixed_float16"
        resolved_dtype = tf.float16
        print("Mixed precision (float16) enabled!")
    elif dtype_name == "bfloat16":
        policy_name = "mixed_bfloat16"
        resolved_dtype = tf.bfloat16
        print("Mixed precision (bfloat16) enabled!")
    elif dtype_name == "float32":
        policy_name = "float32"
        resolved_dtype = tf.float32
    else:
        raise ValueError(
            f"Unsupported dtype '{dtype_name}'. Use one of: float16, bfloat16, float32."
        )

    if version.parse(tf.__version__) < version.parse("2.4.0"):
        from tensorflow.keras.mixed_precision import experimental as mixed_precision
        policy = mixed_precision.Policy(policy_name)
        mixed_precision.set_policy(policy)
    else:
        from tensorflow.keras import mixed_precision
        mixed_precision.set_global_policy(policy_name)

    return mixed_precision, resolved_dtype


def create_distribution_strategy(
    physical_devices=None,
    use_hierarchical_all_reduce=True,
    single_gpu_strategy="mirrored",
):
    """
    Create a distribution strategy using the common project defaults.

    - Multi-GPU: MirroredStrategy, optionally with HierarchicalCopyAllReduce.
    - Single-GPU: MirroredStrategy (default) or OneDeviceStrategy.
    - CPU-only: OneDeviceStrategy('/cpu:0').
    """
    if physical_devices is None:
        physical_devices = tf.config.list_physical_devices("GPU")

    # Use NCCL for multi-GPU communication to avoid CPU fallback
    if len(physical_devices) > 1:
        if use_hierarchical_all_reduce:
            # Use HierarhicalCopyAllReduce to avoid NCCL issues with Blackwell GPUs
            return tf.distribute.MirroredStrategy(
                cross_device_ops=tf.distribute.HierarchicalCopyAllReduce()
            )
        # strategy = tf.distribute.MirroredStrategy(cross_device_ops=tf.distribute.ReductionToOneDevice(reduce_to_device="cpu:0")) # slowest option
        else:
            return tf.distribute.MirroredStrategy(cross_device_ops=tf.distribute.NcclAllReduce())

    if len(physical_devices) == 1:
        if single_gpu_strategy == "one_device":
            # dont use this strategy since single device gpu increases largely the memory required to allocate the pre_ind_table tensor on GPU memory, which is already large and can cause OOM errors. MirroredStrategy with one GPU does not have this issue (it places it in the CPU) and is more memory efficient.
            return tf.distribute.OneDeviceStrategy(device="/gpu:0")
        if single_gpu_strategy == "mirrored":
            return tf.distribute.MirroredStrategy()
        raise ValueError(
            "single_gpu_strategy must be 'mirrored' or 'one_device'."
        )

    return tf.distribute.OneDeviceStrategy(device="/cpu:0")


def stateless_fold_in(seed, data):
    """Fold data into a stateless RNG seed across TensorFlow versions."""
    fold_in_fn = getattr(tf.random, "stateless_fold_in", None)
    if fold_in_fn is None:
        experimental_random = getattr(tf.random, "experimental", None)
        if experimental_random is None:
            raise AttributeError("TensorFlow random stateless fold-in is unavailable.")
        fold_in_fn = getattr(experimental_random, "stateless_fold_in", None)
        if fold_in_fn is None:
            raise AttributeError("TensorFlow random stateless fold-in is unavailable.")
    return fold_in_fn(seed, data)


def configure_run_paths(
    flags,
    task_name=None,
    flag_names=None,
):
    """Build the common results/logdir context used by entrypoint scripts."""
    if flag_names is None:
        flag_names = (
            "n_input",
            "core_only",
            "connected_selection",
            "random_weights",
            "uniform_weights",
        )

    logdir = flags.ckpt_dir
    current_epoch = 0

    if logdir == "":
        flag_str = f"v1_{flags.neurons}"
        for name, value in flags.flag_values_dict().items():
            if value != flags[name].default and name in flag_names:
                flag_str += f"_{name}_{value}"

        results_dir = os.path.join(flags.results_dir, flag_str)
        os.makedirs(results_dir, exist_ok=True)
        print("Simulation results path: ", results_dir)
        # Generate a ticker for the current simulation
        sim_name = get_random_identifier("b_")
        logdir = os.path.join(results_dir, sim_name)
        if task_name is not None:
            print(f"> Results for {task_name} will be stored in:\n {logdir} \n")
        else:
            print(f"> Results will be stored in:\n {logdir} \n")
    else:
        flag_str = logdir.split(os.path.sep)[-2]
        current_epoch = flags.run_session * flags.n_epochs

    return flag_str, logdir, current_epoch


class DistributedSeedHelper(tf.Module):
    """Manage distributed RNG streams for noise and spontaneous LGN sampling."""

    def __init__(
        self,
        seed,
        noise_stream,
        noise_seed,
        stream_name="spontaneous_seed_stream",
        noise_seed_offset=700001,
        spontaneous_seed_offset=900001,
    ):
        super().__init__(name="distributed_seed_helper")
        self.noise_stream = noise_stream
        self.noise_seed = noise_seed
        self.spontaneous_seed_stream = tf.Variable(
            tf.constant(0, dtype=tf.int64), trainable=False, dtype=tf.int64, name=stream_name
        )

        self.noise_seed_base = tf.constant(int(seed), dtype=tf.int64)
        max_int32 = 2**31 - 1
        spontaneous_seed_a = int(seed + noise_seed_offset) % max_int32
        spontaneous_seed_b = int(seed + spontaneous_seed_offset) % max_int32
        self.spontaneous_seed_base = tf.constant(
            [spontaneous_seed_a, spontaneous_seed_b], dtype=tf.int32
        )

    def advance_noise_seed(self):
        stream_id = self.noise_stream.assign_add(tf.constant(1, dtype=tf.int64))
        self.noise_seed.assign(self.noise_seed_base + stream_id)

    def next_spontaneous_seed(self):
        stream_id = self.spontaneous_seed_stream.assign_add(tf.constant(1, dtype=tf.int64))
        seed_value = stateless_fold_in(
            self.spontaneous_seed_base, tf.cast(stream_id, tf.int32)
        )
        replica_context = tf.distribute.get_replica_context()
        if replica_context is None:
            replica_id = tf.constant(0, dtype=tf.int32)
        else:
            replica_id = tf.cast(replica_context.replica_id_in_sync_group, tf.int32)
        return stateless_fold_in(seed_value, replica_id)

    def reset_spontaneous_seed_stream(self):
        self.spontaneous_seed_stream.assign(tf.constant(0, dtype=tf.int64))


def load_training_state(logdir, restore_from=""):
    candidates = []
    if logdir:
        candidates.append(os.path.join(logdir, "train_end_data.pkl"))
    if restore_from:
        candidates.append(os.path.join(os.path.dirname(restore_from), "train_end_data.pkl"))

    for candidate in candidates:
        if candidate and os.path.exists(candidate):
            with open(candidate, "rb") as handle:
                return pkl.load(handle), candidate
    return None, None


def resolve_performance_statistics_path(logdir):
    """Resolve the aggregate performance CSV location from a run logdir."""
    abs_logdir = os.path.abspath(logdir)
    current = abs_logdir

    while True:
        if os.path.basename(current) == "Simulation_results":
            return os.path.join(current, "performance_statistics.csv")
        parent = os.path.dirname(current)
        if parent == current:
            break
        current = parent

    # Fallback for tests or ad hoc directories that are not under Simulation_results.
    parent = os.path.dirname(abs_logdir)
    grandparent = os.path.dirname(parent)
    results_root = grandparent if grandparent and grandparent != parent else parent
    return os.path.join(results_root, "performance_statistics.csv")


def append_performance_statistics(
    logdir,
    n_neurons,
    n_edges,
    batch_size,
    seq_len,
    mean_rate,
    sem_rate,
    mean_step_time,
    sem_step_time,
    mean_gpu_memory,
    sem_gpu_memory,
    mode,
):
    stats_file = resolve_performance_statistics_path(logdir)
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    canonical_repo_stats_file = os.path.join(
        repo_root, "Simulation_results", "performance_statistics.csv"
    )

    if (
        os.path.abspath(stats_file) == canonical_repo_stats_file
        and not os.path.exists(stats_file)
    ):
        os.makedirs(os.path.dirname(stats_file), exist_ok=True)

    file_exists = os.path.isfile(stats_file)
    sim_name = os.path.basename(os.path.normpath(logdir))
    os.makedirs(os.path.dirname(stats_file), exist_ok=True)

    with open(stats_file, "a", encoding="utf-8") as handle:
        if not file_exists:
            handle.write(
                "sim_name,n_neurons,n_edges,batch_size,seq_len,"
                "mean_rate,sem_rate,mean_step_time,sem_step_time,"
                "mean_gpu_memory,sem_gpu_memory,mode\n"
            )
        handle.write(
            f"{sim_name},{n_neurons},{n_edges},{batch_size},{seq_len},"
            f"{mean_rate:.4f},{sem_rate:.4f},{mean_step_time:.4f},"
            f"{sem_step_time:.4f},{mean_gpu_memory:.4f},{sem_gpu_memory:.4f},"
            f"\"{mode}\"\n"
        )
    return stats_file


def resolve_checkpoint_directory(flags):
    if flags.ckpt_dir != '' and flags.restore_from != '':
        candidate_dir = os.path.join(flags.ckpt_dir, flags.restore_from)
        if os.path.exists(candidate_dir):
            latest_ckpt = tf.train.latest_checkpoint(candidate_dir)
            if latest_ckpt:
                return latest_ckpt, 'ckpt_dir'
    if flags.restore_from != '' and os.path.exists(flags.restore_from):
        latest_ckpt = tf.train.latest_checkpoint(flags.restore_from)
        if latest_ckpt:
            return latest_ckpt, 'path'
    return None, None


def restore_training_checkpoint(
    flags,
    model,
    optimizer,
    learning_rate,
    mixed_precision_module=None,
    checkpoint_subdir="Intermediate_checkpoints"
):
    """Restore a training checkpoint and rebuild the optimizer if needed."""
    from v1_model_utils.other_v1_utils import optimizers_match
    from v1_model_utils.optimizers import create_optimizer

    checkpoint = None
    checkpoint_directory = None

    if flags.ckpt_dir != '' and os.path.exists(
        os.path.join(flags.ckpt_dir, checkpoint_subdir)
    ):
        checkpoint_directory = tf.train.latest_checkpoint(
            os.path.join(flags.ckpt_dir, checkpoint_subdir)
        )
        if checkpoint_directory is None:
            print(
                f"No checkpoint found in {os.path.join(flags.ckpt_dir, checkpoint_subdir)}. Starting from scratch...\n"
            )
            return checkpoint, optimizer, checkpoint_directory

        print(f'Restoring checkpoint from {checkpoint_directory}...')
        optimizer_continuing = optimizers_match(optimizer, checkpoint_directory)
        if not optimizer_continuing:
            print("Optimizer does not match the checkpoint. Using a new optimizer.")
            optimizer = create_optimizer(
                flags,
                learning_rate,
                model.trainable_variables,
                mixed_precision_module=mixed_precision_module,
            )
            checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)
            checkpoint.restore(checkpoint_directory).expect_partial()#.assert_consumed()
            print('Checkpoint restored with a new optimizer.')
        else:
            checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)
            checkpoint.restore(checkpoint_directory).expect_partial()#.assert_consumed()
            print('Checkpoint restored!')
        return checkpoint, optimizer, checkpoint_directory

    if flags.restore_from != '' and os.path.exists(flags.restore_from):
        checkpoint_directory = tf.train.latest_checkpoint(flags.restore_from)
        if checkpoint_directory is None:
            print(
                f"No checkpoint found in {flags.restore_from}. Starting from scratch...\n"
            )
            return checkpoint, optimizer, checkpoint_directory

        print(
            f'Restoring checkpoint from {checkpoint_directory} with the restore_from option...'
        )
        optimizer_continuing = optimizers_match(optimizer, checkpoint_directory)
        if not optimizer_continuing:
            print("Optimizer does not match the checkpoint. Using a new optimizer.")
            optimizer = create_optimizer(
                flags,
                learning_rate,
                model.trainable_variables,
                mixed_precision_module=mixed_precision_module,
            )
            checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)
            checkpoint.restore(checkpoint_directory).expect_partial()#.assert_consumed()
            print('Checkpoint restored with a new optimizer.')
        else:
            checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)
            checkpoint.restore(checkpoint_directory).expect_partial()#.assert_consumed()
            print('Checkpoint restored!')
        return checkpoint, optimizer, checkpoint_directory

    print(f"No checkpoint found in {flags.ckpt_dir} or {flags.restore_from}. Starting from scratch...\n")
    return checkpoint, optimizer, checkpoint_directory


def restore_evaluation_checkpoint(
    flags,
    model,
    build_model_fn,
    logdir,
    current_epoch=0,
    result_name="OSI/DSI",
):
    """Restore a model checkpoint for evaluation-style scripts."""
    checkpoint_directory, checkpoint_source = resolve_checkpoint_directory(flags)
    checkpoint_model_dtype = infer_checkpoint_model_dtype(checkpoint_directory)
    checkpoint_dtype_mismatch = (
        checkpoint_directory is not None
        and checkpoint_model_dtype is not None
        and checkpoint_model_dtype != flags.dtype
    )

    if checkpoint_dtype_mismatch:
        print(
            f"Checkpoint model dtype '{checkpoint_model_dtype}' differs from requested "
            f"evaluation dtype '{flags.dtype}'."
        )

    if checkpoint_dtype_mismatch and not flags.restore_runtime_dtype_cast:
        raise ValueError(
            "Checkpoint/model dtype mismatch detected. Re-run with "
            "--restore_runtime_dtype_cast to enable in-memory conversion "
            "(no checkpoint is re-saved)."
        )

    checkpoint = None
    if checkpoint_directory is not None:
        print(f'Restoring checkpoint from {checkpoint_directory}...')

        if checkpoint_dtype_mismatch:
            restore_model_with_runtime_dtype_cast(
                target_model=model,
                build_model_fn=build_model_fn,
                checkpoint_directory=checkpoint_directory,
                checkpoint_dtype_name=checkpoint_model_dtype,
                target_dtype_name=flags.dtype,
            )
            print('Checkpoint restored via in-memory dtype conversion (no checkpoint re-save).')
        else:
            checkpoint = tf.train.Checkpoint(model=model)
            checkpoint.restore(checkpoint_directory).expect_partial()
            print('Checkpoint restored!')

        if checkpoint_source == 'ckpt_dir' and flags.restore_from == "Best_model":
            logdir = checkpoint_directory + "_results"
            current_epoch_file = os.path.basename(checkpoint_directory)
            match = re.search(r"ckpt-(\d+)", current_epoch_file)
            if match:
                current_epoch = int(match.group(1))

        print(f'{result_name} results for epoch {current_epoch} will be saved in: {logdir}\n')
    else:
        print(f"No checkpoint found in {flags.ckpt_dir} or {flags.restore_from}. Starting from scratch...\n")

    return checkpoint, logdir, current_epoch


def _dtype_name_from_ckpt_dtype(raw_dtype):
    try:
        return tf.dtypes.as_dtype(raw_dtype).name
    except (TypeError, ValueError):
        try:
            return tf.dtypes.as_dtype(int(raw_dtype)).name
        except (TypeError, ValueError):
            return None


def infer_checkpoint_model_dtype(checkpoint_directory):
    if not checkpoint_directory:
        return None
    try:
        dtype_map = tf.train.load_checkpoint(checkpoint_directory).get_variable_to_dtype_map()
    except Exception as e:
        print(f'Could not inspect checkpoint dtypes at {checkpoint_directory}: {e}')
        return None

    model_dtype_names = set()
    for tensor_name, raw_dtype in dtype_map.items():
        if not tensor_name.startswith('model/'):
            continue
        dtype_name = _dtype_name_from_ckpt_dtype(raw_dtype)
        if dtype_name in ('float16', 'bfloat16', 'float32'):
            model_dtype_names.add(dtype_name)

    if len(model_dtype_names) == 1:
        return next(iter(model_dtype_names))

    if len(model_dtype_names) > 1:
        sorted_names = sorted(model_dtype_names)
        print(f'Found mixed model dtypes in checkpoint: {sorted_names}')
        # Mixed-precision checkpoints commonly contain float32 plus one low-precision
        # dtype. Prefer the low-precision dtype as the checkpoint model dtype.
        if 'float16' in model_dtype_names and 'bfloat16' not in model_dtype_names:
            return 'float16'
        if 'bfloat16' in model_dtype_names and 'float16' not in model_dtype_names:
            return 'bfloat16'
    return None


def restore_model_with_runtime_dtype_cast(
    target_model,
    build_model_fn,
    checkpoint_directory,
    checkpoint_dtype_name,
    target_dtype_name,
):
    def _canonical_name(var_name):
        return var_name.split(':', 1)[0]

    source_dtype = configure_policy_and_dtype(checkpoint_dtype_name)
    source_model = build_model_fn(source_dtype)
    source_checkpoint = tf.train.Checkpoint(model=source_model)
    source_checkpoint.restore(checkpoint_directory).expect_partial()

    try:
        source_vars_by_name = {_canonical_name(var.name): var for var in source_model.variables}
        target_vars_by_name = {_canonical_name(var.name): var for var in target_model.variables}

        missing_in_source = []
        shape_mismatches = []
        assigned_count = 0

        for target_name, target_var in target_vars_by_name.items():
            source_var = source_vars_by_name.get(target_name)
            if source_var is None and target_name.endswith("sparse_recurrent_weights_compute"):
                base_name = target_name.replace("_compute", "")
                source_var = source_vars_by_name.get(base_name)

            if source_var is None:
                missing_in_source.append(target_name)
                continue

            if source_var.shape != target_var.shape:
                shape_mismatches.append(f'{target_name} ({source_var.shape} != {target_var.shape})')
                continue

            target_var.assign(tf.cast(source_var.read_value(), target_var.dtype))
            assigned_count += 1

        if shape_mismatches:
            preview = ', '.join(shape_mismatches[:5])
            raise ValueError(
                'Cannot runtime-cast checkpoint: variable shape mismatches found. '
                f'Examples: {preview}'
            )

        if missing_in_source:
            preview = ', '.join(missing_in_source[:5])
            raise ValueError(
                'Cannot runtime-cast checkpoint: target variables missing in source model. '
                f'Examples: {preview}'
            )

        extra_in_source = sorted(set(source_vars_by_name) - set(target_vars_by_name))
        if extra_in_source:
            preview = ', '.join(extra_in_source[:5])
            print(
                f'Runtime-cast restore: ignoring {len(extra_in_source)} source-only '
                f'variables (example: {preview}).'
            )

        print(f'Runtime-cast restore: assigned {assigned_count} variables.')
    finally:
        del source_checkpoint
        del source_model
        gc.collect()
        configure_policy_and_dtype(target_dtype_name)
