
import os
# Define the environment variables for optimal GPU performance.
# These must be set before importing TensorFlow.
os.environ['TF_GPU_THREAD_MODE'] = 'global'
# Prefer the optimized recurrent/interarea dW kernel and warn before falling
# back to TensorFlow when the custom op is unavailable. An explicit environment
# value (for example ``off`` or ``required``) still takes precedence.
os.environ.setdefault('LM_V1_CUDA_SYNAPTIC_DW', 'auto')
# Trial XLA auto-clustering for the fixed-shape recurrent training graph does not work since the network activity is sparse.
# os.environ['TF_XLA_FLAGS'] = '--tf_xla_auto_jit=1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0' # before import tensorflow
# Prefer the default BFC allocator for training: cuda_malloc_async retained GPU
# pool memory and reached OOM earlier in this workload despite lower live usage.
# os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
# os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
# os.environ['TF_ENABLE_ONEDNN_OPTS']= '1'

import absl
import numpy as np
import tensorflow as tf
import pickle as pkl
from packaging import version
from time import time
import ctypes.util
import random
import logging
logging.getLogger().setLevel(logging.INFO)

from Model_utils.stimulus_weights import StimulusWeights


GPU_INT32_INDEX_LIMIT = np.iinfo(np.int32).max


def assert_gpu_index_limits(networks, lgn_inputs, bkg_inputs, batch_size, seq_len):
    """Fail before model construction when GPU int32 indexing would overflow.

    TensorFlow's GPU Split kernel indexes the complete sequence tensor with
    int32, while the sparse-matmul kernels use an int32-sized work product of
    ``nnz(A) * output.shape[1]``.  The sparse operations in BillehColumn are
    called once per timestep, so ``output.shape[1]`` is the per-replica batch.
    """
    batch_size = int(batch_size)
    seq_len = int(seq_len)
    for area, network in networks.items():
        split_elements = batch_size * seq_len * int(network["n_nodes"])
        assert split_elements < GPU_INT32_INDEX_LIMIT, (
            "GPU Split input is too large: "
            f"area={area}, batch_size={batch_size}, seq_len={seq_len}, "
            f"neurons={network['n_nodes']}, elements={split_elements:,} "
            f"must be < {GPU_INT32_INDEX_LIMIT:,}. Reduce batch_size/seq_len "
            "or use branch-wise rollout."
        )

    sparse_matrices = []

    for area, network in networks.items():
        sparse_matrices.append(
            (f"{area} recurrent", network["synapses"]["indices"])
        )
        for source_area, connection in network["interarea_synapses"].items():
            sparse_matrices.append(
                (
                    f"{source_area}->{area} inter-area",
                    connection["indices"],
                )
            )

    for area, inputs in lgn_inputs.items():
        if inputs is not None:
            sparse_matrices.append((f"LGN->{area}", inputs["indices"]))
    for area, inputs in bkg_inputs.items():
        if inputs is not None:
            sparse_matrices.append((f"background->{area}", inputs["indices"]))

    for name, indices in sparse_matrices:
        if indices is None:
            continue
        nnz = int(np.asarray(indices).shape[0])
        spmm_work = nnz * batch_size
        assert spmm_work < GPU_INT32_INDEX_LIMIT, (
            "GPU SpMM work size is too large: "
            f"{name} has nnz(A)={nnz:,}, output.shape[1]={batch_size}, "
            f"nnz(A)*output.shape[1]={spmm_work:,} must be < "
            f"{GPU_INT32_INDEX_LIMIT:,}. Reduce batch_size or split the "
            "SpMM workload."
        )


def configure_gpu_memory_growth():
    physical_devices = tf.config.list_physical_devices("GPU")
    for dev in physical_devices:
        try:
            tf.config.experimental.set_memory_growth(dev, True)
            print(f"Memory growth enabled for device {dev}")
        except (RuntimeError, ValueError) as e:
            print(f"Could not enable memory growth for device {dev}: {e}")
    print("- Num GPUs Available: ", len(physical_devices), '\n')
    return physical_devices


def configure_reproducibility(seed):
    seed = int(seed)
    tf.keras.utils.set_random_seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    random.seed(seed)
    print(f"Reproducibility seed: {seed}")


def natural_scene_projection_protocol_compatible(stored, current):
    """Return whether saved calibration references match the active protocol."""
    if current is None:
        return stored is None
    if stored is None:
        # Metadata predates the RPD projection mode. Legacy references used
        # the deterministic contrast-rotation protocol.
        return current.get("mode") == "contrast_rotations"
    if not isinstance(stored, dict) or not isinstance(current, dict):
        return False
    return stored == current


def annulus_mask_from_core(core_mask):
    """Return the complement of one area's core, or None for no annulus."""
    if core_mask is None:
        return None
    annulus_mask = tf.logical_not(tf.convert_to_tensor(core_mask, tf.bool))
    if not bool(tf.reduce_any(annulus_mask).numpy()):
        return None
    return annulus_mask


def update_rate_emas(
    v1_ema,
    lm_ema,
    v1_rates,
    lm_rates,
    decay,
    update_state=True,
):
    """Update training rate normalizers without mutating them in evaluation."""
    if update_state:
        v1_ema.assign(decay * v1_ema + (1 - decay) * v1_rates)
        lm_ema.assign(decay * lm_ema + (1 - decay) * lm_rates)


def run_gray_state_rollout(state_model, sequence_and_state_model, inputs):
    """Return warm-up final state, using the full sequence model as fallback."""
    if state_model is not None:
        return state_model(inputs)
    full_outputs = sequence_and_state_model(inputs)
    if not isinstance(full_outputs, (tuple, list)) or len(full_outputs) < 2:
        raise ValueError(
            "The sequence-and-state fallback must return sequences followed "
            "by at least one final-state tensor."
        )
    return tuple(full_outputs[1:])


def build_learning_rate(flags, linear_warmup_cosine_decay_cls):
    if flags.lr_schedule == "none":
        print(f"Learning-rate schedule: none (constant lr={flags.learning_rate:.6g})")
        return flags.learning_rate

    if flags.lr_schedule == "warmup_cosine":
        schedule = linear_warmup_cosine_decay_cls(
            warmup_start_lr=flags.lr_warmup_start_lr,
            warmup_target_lr=flags.lr_warmup_target_lr,
            warmup_steps=flags.lr_warmup_steps,
            cosine_steps=flags.lr_cosine_steps,
            min_lr=flags.lr_cosine_min_lr,
        )
        print(
            "Learning-rate schedule: warmup_cosine "
            f"(warmup: {flags.lr_warmup_start_lr:.6g}->{flags.lr_warmup_target_lr:.6g} "
            f"in {flags.lr_warmup_steps} steps, cosine: "
            f"{flags.lr_warmup_target_lr:.6g}->{flags.lr_cosine_min_lr:.6g} "
            f"in {flags.lr_cosine_steps} steps)"
        )
        return schedule

    raise ValueError(
        f"Invalid lr_schedule '{flags.lr_schedule}'. "
        "Supported values are: 'none', 'warmup_cosine'."
    )


def main(_):
    flags = absl.app.flags.FLAGS
    stimulus_weights = StimulusWeights(flags)
    stimulus_names = ("dg", "sp", "ns")
    # Allow for memory growth (also to observe memory consumption)
    physical_devices = configure_gpu_memory_growth()

    print("--- CUDA version: ", tf.sysconfig.get_build_info()["cuda_version"])
    print("--- CUDNN version: ", tf.sysconfig.get_build_info()["cudnn_version"])
    print("--- TensorFlow version: ", tf.__version__)
    # For CUDA Runtime API
    lib_path = ctypes.util.find_library("cudart")
    print("--- CUDA Library path: ", lib_path)

    # Import TensorFlow-heavy project modules only after GPU memory is configured.
    from Model_utils import load_sparse, models, other_billeh_utils, stim_dataset, toolkit
    import Model_utils.loss_functions as losses
    from Model_utils.callbacks import Callbacks
    from Model_utils.optimizers import ExponentiatedAdam, LinearWarmupCosineDecay

    if version.parse(tf.__version__) < version.parse("2.4.0"):
        from tensorflow.keras.mixed_precision import experimental as mixed_precision
    else:
        from tensorflow.keras import mixed_precision

    # Set the seeds for reproducibility - use seed for stimulus generation
    configure_reproducibility(flags.seed)
    # tf.config.optimizer.set_experimental_options({"cudnn_use_autotune": True})
    # Configure TensorFlow optimization options that have proven to be beneficial for our model and training setup. These options can improve the performance of the model by optimizing the computational graph and memory usage.
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
        "pin_to_host_optimization": False, # it causes segmentation fault error in the V1-LM model
        "implementation_selector": True, # good
        "auto_parallel": True, # good
        # "disable_model_pruning": False, # needs to be false to allow pruning of training subgraph
        "min_graph_nodes": 0, # good to set to 0 to allow optimization of small subgraphs, which is important for our model with many small ops
    })

    # Create the tensorflow datafile for the network
    os.makedirs(os.path.join(flags.data_dir, 'tf_data'), exist_ok=True)

    # Define 2 outputs that correspond to having more cues top or bottom
    # Note that two different output conventions can be used:
    # 1) Linear readouts from all neurons in the model (softmax)
    # 2) Selecting a population of neurons that report a binary decision
    # with high firing rate (flag --neuron_output)
    # n_output = 2

    # Load data of Billeh et al. (2020) and select appropriate number of neurons and inputs
    # Create the v1-lm model - use model_seed for model creation
    t0 = time()
    if flags.caching:
        load_fn = load_sparse.cached_load_billeh
    else:
        load_fn = load_sparse.load_billeh

    networks, lgn_inputs, bkg_inputs = load_fn(flags)
    v1_neurons = networks['v1']['n_nodes']
    v1_column_radius = networks['v1']['column_radius']
    lm_neurons = networks['lm']['n_nodes']
    lm_column_radius = networks['lm']['column_radius']
    print(f"Model files loading: {time()-t0:.2f} seconds\n")

    if physical_devices:
        assert_gpu_index_limits(
            networks,
            lgn_inputs,
            bkg_inputs,
            batch_size=flags.batch_size,
            seq_len=flags.seq_len,
        )

    delays = [int(a) for a in flags.delays.split(',') if a != '']

    # if flags.realistic_neurons_ratio:
    #     # Select the connectivity rules in the network
    #     v1_to_lm_neurons_ratio = 7.010391285652859
    #     n_neurons = {'v1': flags.v1_neurons,
    #                  'lm': int(flags.v1_neurons/v1_to_lm_neurons_ratio)}
    # elif flags.realistic_radius:
    #     column_radius = {'v1': flags.v1_radius,
    #                      'lm': flags.lm_radius}
    # else:
    #     n_neurons = {'v1': flags.v1_neurons,
    #                  'lm': flags.lm_neurons}

    logdir = flags.ckpt_dir
    if logdir == '':
        if flags.realistic_radius:
            flag_str = f'v1_{v1_column_radius}microns_lm_{lm_column_radius}microns_model_{flags.model_seed}'
        else:
            flag_str = f'v1_{v1_neurons}_lm_{lm_neurons}_model_{flags.model_seed}'

        for name, value in flags.flag_values_dict().items():
            if value != flags[name].default and name in ['n_input', 'core_only', 'connected_selection', 'interarea_weight_distribution', 'E4_weight_factor', 'randomize_recurrent_weights']:
                flag_str += f'_{name}_{value}'
        flag_str += f'_{stimulus_weights.tag}'
        # Define flag string as the second part of results_path
        results_dir = f'{flags.results_dir}/{flag_str}'
        os.makedirs(results_dir, exist_ok=True)
        # Generate a ticker for the current simulation
        sim_name = toolkit.get_random_identifier('b_')
        logdir = os.path.join(results_dir, sim_name)
        print(
            f'> Results for {flags.task_name} will be stored in:\n {logdir} \n')
    # else:
    #     flag_str = logdir.split(os.path.sep)[-2]

    # Can be used to try half precision training
    if flags.dtype == 'float16':
        if version.parse(tf.__version__) < version.parse("2.4.0"):
            policy = mixed_precision.Policy("mixed_float16")
            mixed_precision.set_policy(policy)
        else:
            mixed_precision.set_global_policy('mixed_float16')
        dtype = tf.float16
        print('Mixed precision (float16) enabled!')
    elif flags.dtype == 'bfloat16':
        if version.parse(tf.__version__) < version.parse("2.4.0"):
            policy = mixed_precision.Policy("mixed_bfloat16")
            mixed_precision.set_policy(policy)
        else:
            mixed_precision.set_global_policy('mixed_bfloat16')
        dtype = tf.bfloat16
        print('Mixed precision (bfloat16) enabled!')
    else:
        dtype = tf.float32

    # n_workers, n_gpus_per_worker = 1, 1
    # model is being run on multiple GPUs or CPUs, and the results are being reduced to a single CPU device.
    # In this case, the reduce_to_device argument is set to "cpu:0", which means that the results are being reduced to the first CPU device.
    # device = "/gpu:0" if tf.config.list_physical_devices("GPU") else "/cpu:0"
    # strategy = tf.distribute.OneDeviceStrategy(device=device) # Single device strategy
    # Use NCCL for multi-GPU all-reduce to avoid CPU fallback
    if len(physical_devices) > 1:
        # strategy = tf.distribute.MirroredStrategy(cross_device_ops=tf.distribute.NcclAllReduce())
        # Use HierarchicalCopyAllReduce to avoid NCCL issues with Blackwell GPUs
        strategy = tf.distribute.MirroredStrategy(cross_device_ops=tf.distribute.HierarchicalCopyAllReduce())
    elif len(physical_devices) == 1:
        if flags.single_gpu_strategy == "one_device":
            strategy = tf.distribute.OneDeviceStrategy(device="/gpu:0")
        else:
            strategy = tf.distribute.MirroredStrategy()
    else:
        strategy = tf.distribute.OneDeviceStrategy(device="/cpu:0")
    # Fallback option to reduce all results to GPU:0 (slowest)
    # strategy = tf.distribute.MirroredStrategy(
    #     cross_device_ops=tf.distribute.ReductionToOneDevice(reduce_to_device="/gpu:0")
    # )

    per_replica_batch_size = flags.batch_size
    grating_batch_size = flags.grating_batch_size
    gray_batch_size = flags.gray_batch_size
    natural_batch_size = flags.natural_batch_size
    stimulus_batch_sizes = (
        gray_batch_size, grating_batch_size, natural_batch_size
    )
    if any(size <= 0 for size in stimulus_batch_sizes):
        raise ValueError("All per-replica stimulus batch sizes must be positive.")
    if sum(stimulus_batch_sizes) != per_replica_batch_size:
        raise ValueError(
            "batch_size must equal gray_batch_size + grating_batch_size + "
            "natural_batch_size."
        )
    global_batch_size = per_replica_batch_size * strategy.num_replicas_in_sync
    print(f'Per replica batch size: {per_replica_batch_size}')
    print(f'Global batch size: {global_batch_size}')
    print(
        f'Stimulus batch sizes (per replica): grating={grating_batch_size}, '
        f'gray={gray_batch_size}, natural={natural_batch_size}\n'
    )
    print(f'Training with current input: {flags.current_input}')
    print(f'Pseudo derivative gaussian: {flags.pseudo_gauss}')
    print(
        "Stimulus weights: "
        f"dg={stimulus_weights.dg:g}, "
        f"sp={stimulus_weights.sp:g}, "
        f"ns={stimulus_weights.ns:g}"
    )
    print(f"Active training stimuli: {', '.join(stimulus_weights.active_names)}\n")

    if flags.gradient_checkpointing:
        if not 1 <= flags.gradient_checkpoint_chunk_size <= flags.seq_len:
            raise ValueError(
                "gradient_checkpoint_chunk_size must satisfy "
                "1 <= chunk size <= seq_len when gradient checkpointing is enabled."
            )
        n_checkpoint_chunks = (
            flags.seq_len + flags.gradient_checkpoint_chunk_size - 1
        ) // flags.gradient_checkpoint_chunk_size
        print(
            "Segmented gradient checkpointing enabled: "
            f"chunk_size={flags.gradient_checkpoint_chunk_size}, "
            f"chunks={n_checkpoint_chunks}\n"
        )

    # Define the scope in which the model training will be executed
    with strategy.scope():
        t0 = time()

        noise_scales = [float(a) for a in flags.noise_scales.split(',') if a != '']
        if len(noise_scales) == 2:
            noise_scale = (noise_scales[0], noise_scales[1])
        else:
            noise_scale = (1.0, 1.0)

        # Build the model
        model = models.create_model(
            networks,
            lgn_inputs,
            bkg_inputs,
            seq_len=flags.seq_len,
            n_input=flags.n_input,
            n_output=flags.n_output,
            dtype=dtype,
            input_weight_scale=flags.input_weight_scale,
            interarea_runtime_scale=flags.interarea_runtime_scale,
            recurrent_weight_init_scale=flags.recurrent_weight_init_scale,
            recurrent_runtime_scale=flags.recurrent_runtime_scale,
            interarea_weight_init_scale=flags.interarea_weight_init_scale,
            recurrent_dampening_factor=flags.recurrent_dampening_factor,
            dampening_factor=flags.dampening_factor,
            gauss_std=flags.gauss_std,
            lr_scale=flags.lr_scale,
            train_recurrent_v1=flags.train_recurrent_v1,
            train_recurrent_lm=flags.train_recurrent_lm,
            train_input=flags.train_input,
            train_interarea_lm_v1=flags.train_interarea_lm_v1,
            train_interarea_v1_lm=flags.train_interarea_v1_lm,
            train_noise=flags.train_noise,
            batch_size=per_replica_batch_size,
            pseudo_gauss=flags.pseudo_gauss,
            use_state_input=True,
            return_state=True,
            hard_reset=flags.hard_reset,
            connected_recurrent_connections=flags.connected_recurrent_connections,
            connected_areas=flags.connected_areas,
            connected_noise=flags.connected_noise,
            add_rate_metric=False,
            max_delay=5,
            neuron_output=flags.neuron_output,
            # output_completed_valid_from_time=120,
            # output_abstract_valid_from_time=100,
            current_input=flags.current_input,
            track_voltage_penalty=True,
            voltage_penalty_mode=flags.voltage_penalty_mode,
            return_voltage_sequences=flags.return_voltage_sequences,
            seed=flags.seed,
            use_dummy_state_input=False,
            noise_type=flags.noise_type,
            decoded_noise_path=flags.decoded_noise_path,
            noise_scale=noise_scale
        )

        model.build((per_replica_batch_size, flags.seq_len, flags.n_input))
        print(f"Model built in {time()-t0:.2f} s\n")

        # # Store the initial model variables that are going to be trained
        # model_variables_dict = {'Initial': {var.name: var.numpy().astype(
        #     np.float16) for var in model.trainable_variables}}

        # Define the optimizer
        learning_rate = build_learning_rate(flags, LinearWarmupCosineDecay)

        def create_optimizer():
            if flags.optimizer == 'adam':
                base_optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, epsilon=1e-11)
            elif flags.optimizer == 'exp_adam':
                base_optimizer = ExponentiatedAdam(learning_rate=learning_rate, epsilon=1e-11)
            elif flags.optimizer == 'sgd':
                base_optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.0, nesterov=False)
            else:
                print(f"Invalid optimizer: {flags.optimizer}")
                raise ValueError

            # The base optimizer needs to be built before restoring from checkpoint.
            base_optimizer.build(model.trainable_variables)

            if flags.dtype == 'float16':
                # Prevent gradient underflow in mixed-float16 training.
                base_optimizer = mixed_precision.LossScaleOptimizer(base_optimizer)

            return base_optimizer

        # Define the optimizer
        optimizer = create_optimizer()

        # Restore model and optimizer from a checkpoint if it exists
        checkpoint = None
        if flags.ckpt_dir != '' and os.path.exists(os.path.join(flags.ckpt_dir, "Intermediate_checkpoints")):
            checkpoint_directory = tf.train.latest_checkpoint(os.path.join(flags.ckpt_dir, "Intermediate_checkpoints"))
            print(f'Restoring checkpoint from {checkpoint_directory}...')
            optimizer_continuing = other_billeh_utils.optimizers_match(optimizer, checkpoint_directory)
            if not optimizer_continuing:
                # Define the optimizer
                optimizer = create_optimizer()
                # Restore the model
                checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)
                # .assert_consumed()
                checkpoint.restore(checkpoint_directory).expect_partial()
            else:
                # Restore the model
                checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)
                checkpoint.restore(checkpoint_directory).expect_partial()#.assert_consumed()
        # Option to resume the training from a checkpoint from a previous training session
        elif flags.restore_from != '' and os.path.exists(flags.restore_from):
            checkpoint_directory = tf.train.latest_checkpoint(flags.restore_from)
            print(
                f'Restoring checkpoint from {checkpoint_directory} with the restore_from option...')
            optimizer_continuing = other_billeh_utils.optimizers_match(optimizer, checkpoint_directory)
            if not optimizer_continuing:
                # Define the optimizer
                optimizer = create_optimizer()
                # Restore the model
                checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)
                # .assert_consumed()
                checkpoint.restore(checkpoint_directory).expect_partial()
            else:
                # Restore the model
                checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)
                checkpoint.restore(checkpoint_directory).expect_partial()#.assert_consumed()
        else:
            print(
                f"No checkpoint found in {flags.ckpt_dir} or {flags.restore_from}. Starting from scratch...\n")

        resumed_from_checkpoint = checkpoint is not None

        # model_variables_dict['Best'] = {var.name: var.numpy().astype(
        #     np.float16) for var in model.trainable_variables}
        # print(f"Model variables stored in dictionary\n")

        # if flags.realistic_neurons_ratio:
    #     # Select the connectivity rules in the network
    #     v1_to_lm_neurons_ratio = 7.010391285652859
    #     n_neurons = {'v1': flags.v1_neurons,
    #                  'lm': int(flags.v1_neurons/v1_to_lm_neurons_ratio)}
    # elif flags.realistic_radius:
    #     column_radius = {'v1': flags.v1_radius,
    #                      'lm': flags.lm_radius}
    # else:
    #     n_neurons = {'v1': flags.v1_neurons,
    #                  'lm': flags.lm_neurons}

        ### BUILD THE LOSS AND REGULARIZER FUNCTIONS ###
        # Create rate and voltage regularizers
        if flags.core_loss:
            if flags.realistic_radius and v1_column_radius >= 400:
                v1_core_mask = other_billeh_utils.isolate_core_neurons(
                    networks['v1'], column_name='v1', radius=400, data_dir=flags.data_dir)
                v1_core_mask = tf.constant(v1_core_mask, dtype=tf.bool)
            elif v1_neurons > 51978:
                v1_core_mask = other_billeh_utils.isolate_core_neurons(
                    networks['v1'], column_name='v1', n_selected_neurons=51978, data_dir=flags.data_dir)
                v1_core_mask = tf.constant(v1_core_mask, dtype=tf.bool)
            else:
                v1_core_mask = None

            if flags.realistic_radius and lm_column_radius >= 170:
                lm_core_mask = other_billeh_utils.isolate_core_neurons(
                    networks['lm'], column_name='lm', radius=170, data_dir=flags.data_dir)
                lm_core_mask = tf.constant(lm_core_mask, dtype=tf.bool)
            elif lm_neurons > 7414:
                lm_core_mask = other_billeh_utils.isolate_core_neurons(
                    networks['lm'], column_name='lm', n_selected_neurons=7414, data_dir=flags.data_dir)
                lm_core_mask = tf.constant(lm_core_mask, dtype=tf.bool)
            else:
                lm_core_mask = None
        else:
            v1_core_mask = None
            lm_core_mask = None

        # Extract outputs of intermediate keras layers to get access to spikes and membrane voltages of the model
        rsnn_layer = model.get_layer('rsnn')

        ### RECURRENT REGULARIZERS ###
        v1_recurrent_regularizer = None
        v1_l2_recurrent_regularizer = None
        weight_regularizer_dtype = rsnn_layer.cell.v1.recurrent_weight_values.dtype
        if flags.train_recurrent_v1 and flags.recurrent_weight_regularization > 0:
            v1_recurrent_regularizer = losses.StiffRegularizer(flags.recurrent_weight_regularization,
                                                               networks['v1'],
                                                               penalize_relative_change=True,
                                                               dtype=weight_regularizer_dtype)
            v1_l2_recurrent_regularizer = losses.L2Regularizer(flags.recurrent_weight_regularization,
                                                               networks['v1'],
                                                               penalize_relative_change=False,
                                                               dtype=weight_regularizer_dtype)

        lm_recurrent_regularizer = None
        lm_l2_recurrent_regularizer = None
        if flags.train_recurrent_lm and flags.recurrent_weight_regularization > 0:
            lm_recurrent_regularizer = losses.StiffRegularizer(flags.recurrent_weight_regularization,
                                                               networks['lm'],
                                                               penalize_relative_change=True,
                                                               dtype=weight_regularizer_dtype)
            lm_l2_recurrent_regularizer = losses.L2Regularizer(flags.recurrent_weight_regularization,
                                                               networks['lm'],
                                                               penalize_relative_change=False,
                                                               dtype=weight_regularizer_dtype)

        ### INTERAREA REGULARIZERS ###
        v1_lm_regularizer = None
        if flags.train_interarea_v1_lm:
            v1_lm_regularizer = losses.L2Regularizer(flags.interarea_weight_regularization, networks['v1'], penalize_relative_change=True, recurrent_weights=False, source_area='lm', dtype=weight_regularizer_dtype)

        lm_v1_regularizer = None
        if flags.train_interarea_lm_v1:
            lm_v1_regularizer = losses.L2Regularizer(flags.interarea_weight_regularization, networks['lm'], penalize_relative_change=True, recurrent_weights=False, source_area='v1', dtype=weight_regularizer_dtype)

        ### EVOKED RATES REGULARIZERS ###
        v1_evoked_rate_regularizer = losses.SpikeRateDistributionTarget(networks['v1'], stimulus_type='drifting_gratings', rate_cost=flags.rate_cost, pre_delay=delays[0], post_delay=delays[1],
                                                                        data_dir=flags.data_dir, area='v1', core_mask=v1_core_mask, seed=flags.seed, dtype=tf.float32,
                                                                        neuropixels_df=flags.v1_neuropixels_df)
        lm_evoked_rate_regularizer = losses.SpikeRateDistributionTarget(networks['lm'], stimulus_type='drifting_gratings', rate_cost=flags.rate_cost, pre_delay=delays[0], post_delay=delays[1],
                                                                        data_dir=flags.data_dir, area='lm', core_mask=lm_core_mask, seed=flags.seed, dtype=tf.float32,
                                                                        neuropixels_df=flags.lm_neuropixels_df)

        ### SPONTANEOUS RATES REGULARIZERS ###
        v1_spont_rate_regularizer = losses.SpikeRateDistributionTarget(networks['v1'], stimulus_type='spontaneous', rate_cost=flags.rate_cost, pre_delay=delays[0], post_delay=delays[1],
                                                                       data_dir=flags.data_dir, area='v1', core_mask=v1_core_mask, seed=flags.seed, dtype=tf.float32,
                                                                       neuropixels_df=flags.v1_neuropixels_df)
        lm_spont_rate_regularizer = losses.SpikeRateDistributionTarget(networks['lm'], stimulus_type='spontaneous', rate_cost=flags.rate_cost, pre_delay=delays[0], post_delay=delays[1],
                                                                       data_dir=flags.data_dir, area='lm', core_mask=lm_core_mask, seed=flags.seed, dtype=tf.float32,
                                                                       neuropixels_df=flags.lm_neuropixels_df)

        ### NATURAL SCENES RATES REGULARIZERS ###
        natural_rate_cost = (
            flags.rate_cost if stimulus_weights.is_enabled("ns") else 0.0
        )
        v1_natural_rate_regularizer = losses.SpikeRateDistributionTarget(
            networks['v1'],
            stimulus_type='natural_stimuli',
            rate_cost=natural_rate_cost,
            pre_delay=delays[0],
            post_delay=delays[1],
            data_dir=flags.data_dir,
            area='v1',
            core_mask=v1_core_mask,
            seed=flags.seed,
            dtype=tf.float32,
            neuropixels_df=flags.v1_neuropixels_df,
            joint_natural_rate_loss=flags.natural_joint_rate_loss,
            natural_joint_direction_mode=flags.natural_joint_direction_mode,
            natural_rpd_concentration=flags.natural_rpd_concentration,
            natural_rpd_stream_id=0,
        )
        lm_natural_rate_regularizer = losses.SpikeRateDistributionTarget(
            networks['lm'],
            stimulus_type='natural_stimuli',
            rate_cost=natural_rate_cost,
            pre_delay=delays[0],
            post_delay=delays[1],
            data_dir=flags.data_dir,
            area='lm',
            core_mask=lm_core_mask,
            seed=flags.seed,
            dtype=tf.float32,
            neuropixels_df=flags.lm_neuropixels_df,
            joint_natural_rate_loss=flags.natural_joint_rate_loss,
            natural_joint_direction_mode=flags.natural_joint_direction_mode,
            natural_rpd_concentration=flags.natural_rpd_concentration,
            natural_rpd_stream_id=1,
        )
        # Assigned by the deterministic pre-training calibration pass below.
        # Variables, rather than Python scalars, keep the compiled train step
        # stable while allowing exact reference values to be restored.
        natural_scene_marginal_reference = tf.Variable(
            1.0, trainable=False, dtype=tf.float32,
            name="natural_scene_marginal_reference",
        )
        natural_scene_joint_reference = tf.Variable(
            1.0, trainable=False, dtype=tf.float32,
            name="natural_scene_joint_reference",
        )
        natural_scene_projection_protocol = {
            "version": 1,
            "mode": flags.natural_joint_direction_mode,
            "rpd_concentration": float(flags.natural_rpd_concentration),
            "n_extra_directions": losses.NATURAL_JOINT_RPD_PROJECTIONS,
        }

        ### SYNCHRONIZATION REGULARIZERS ###
        v1_evoked_sync_loss = losses.SynchronizationLoss(networks['v1'], sync_cost=flags.sync_cost, area='v1', core_mask=v1_core_mask, t_start=0.2, t_end=flags.seq_len/1000, n_samples=flags.fano_samples, dtype=tf.float32,
                                                         stimulus_type='drifting_gratings', experimental_data_dir='data/Neuropixels_synchronization_data', data_dir=flags.data_dir)
        lm_evoked_sync_loss = losses.SynchronizationLoss(networks['lm'], sync_cost=flags.sync_cost, area='lm', core_mask=lm_core_mask, t_start=0.2, t_end=flags.seq_len/1000, n_samples=flags.fano_samples, dtype=tf.float32,
                                                         stimulus_type='drifting_gratings', experimental_data_dir='data/Neuropixels_synchronization_data', data_dir=flags.data_dir)

        v1_spont_sync_loss = losses.SynchronizationLoss(networks['v1'], sync_cost=flags.sync_cost, area='v1', core_mask=v1_core_mask, t_start=0.2, t_end=flags.seq_len/1000, n_samples=flags.fano_samples, dtype=tf.float32,
                                                        stimulus_type='spontaneous', experimental_data_dir='data/Neuropixels_synchronization_data', data_dir=flags.data_dir)
        lm_spont_sync_loss = losses.SynchronizationLoss(networks['lm'], sync_cost=flags.sync_cost, area='lm', core_mask=lm_core_mask, t_start=0.2, t_end=flags.seq_len/1000, n_samples=flags.fano_samples, dtype=tf.float32,
                                                        stimulus_type='spontaneous', experimental_data_dir='data/Neuropixels_synchronization_data', data_dir=flags.data_dir)

        ### ORIENTATION SELECTIVITY INDEX (OSI) AND DIRECTION SELECTIVITY INDEX (DSI) LOSSES ###
        # here we need information of the layer mask for the OSI loss
        if flags.osi_loss_method == 'neuropixels_fr':
            v1_layer_info = other_billeh_utils.get_layer_info(networks['v1'], data_dir=flags.data_dir)
            lm_layer_info = other_billeh_utils.get_layer_info(networks['lm'], data_dir=flags.data_dir)
        else:
            v1_layer_info = None
            lm_layer_info = None

        # Create an ExponentialMovingAverage object
        # Define the decay factor for the exponential moving average
        ema_decay = 0.95
        # Initialize exponential moving averages for V1 and LM firing rates
        train_end_data_path = os.path.join(logdir, 'train_end_data.pkl')
        if resumed_from_checkpoint and os.path.exists(train_end_data_path):
            with open(train_end_data_path, 'rb') as f:
                data_loaded = pkl.load(f)
                v1_ema = tf.Variable(data_loaded.get("v1_ema", 0.003), trainable=False, name='V1_EMA')
                lm_ema = tf.Variable(data_loaded.get("lm_ema", 0.003), trainable=False, name='LM_EMA')
            print(f"Loaded EMA state from {train_end_data_path}.")
        else:
            if (not resumed_from_checkpoint) and os.path.exists(train_end_data_path):
                print(
                    f"Found {train_end_data_path}, but no checkpoint resume was requested. "
                    "Ignoring stored EMA state for a clean reproducible start."
                )
            # 3 Hz is near the average FR of cortex
            v1_ema = tf.Variable(tf.constant(0.003, shape=(v1_neurons,), dtype=tf.float32), trainable=False, name='V1_EMA')
            lm_ema = tf.Variable(tf.constant(0.003, shape=(lm_neurons,), dtype=tf.float32), trainable=False, name='LM_EMA')

        # if training for spontaneous firing rates set the osi loss to 0
        v1_OSI_DSI_Loss = losses.OrientationSelectivityLoss(network=networks['v1'], osi_cost=flags.osi_cost, area='v1',
                                                            pre_delay=delays[0], post_delay=delays[1],
                                                            dtype=tf.float32, core_mask=v1_core_mask,
                                                            method=flags.osi_loss_method,
                                                            subtraction_ratio=flags.osi_loss_subtraction_ratio,
                                                            layer_info=v1_layer_info,
                                                            neuropixels_df=flags.v1_neuropixels_df,
                                                            data_dir=flags.data_dir)
        lm_OSI_DSI_Loss = losses.OrientationSelectivityLoss(network=networks['lm'], osi_cost=flags.osi_cost, area='lm',
                                                            pre_delay=delays[0], post_delay=delays[1],
                                                            dtype=tf.float32, core_mask=lm_core_mask,
                                                            method=flags.osi_loss_method,
                                                            subtraction_ratio=flags.osi_loss_subtraction_ratio,
                                                            layer_info=lm_layer_info,
                                                            neuropixels_df=flags.lm_neuropixels_df,
                                                            data_dir=flags.data_dir)

        ### ANNULUS REGULARIZERS ###
        v1_annulus_mask = annulus_mask_from_core(v1_core_mask)
        lm_annulus_mask = annulus_mask_from_core(lm_core_mask)
        if v1_annulus_mask is not None:
            # Add rate regularizer for the annulus
            v1_annulus_spont_rate_regularizer = losses.SpikeRateDistributionTarget(networks['v1'], stimulus_type='spontaneous', rate_cost=flags.annulus_loss_weight*flags.rate_cost, pre_delay=delays[0], post_delay=delays[1],
                                                                                   data_dir=flags.data_dir, area='v1', core_mask=v1_annulus_mask, rates_dampening=1., seed=flags.seed, dtype=tf.float32,
                                                                                   neuropixels_df=flags.v1_neuropixels_df)
            v1_annulus_evoked_rate_regularizer = losses.SpikeRateDistributionTarget(networks['v1'], stimulus_type='drifting_gratings', rate_cost=flags.annulus_loss_weight*flags.rate_cost, pre_delay=delays[0], post_delay=delays[1],
                                                                                    data_dir=flags.data_dir, area='v1', core_mask=v1_annulus_mask, rates_dampening=1., seed=flags.seed, dtype=tf.float32,
                                                                                    neuropixels_df=flags.v1_neuropixels_df)
            v1_annulus_natural_rate_regularizer = losses.SpikeRateDistributionTarget(networks['v1'], stimulus_type='natural_stimuli', rate_cost=flags.annulus_loss_weight*natural_rate_cost, pre_delay=delays[0], post_delay=delays[1],
                                                                                     data_dir=flags.data_dir, area='v1', core_mask=v1_annulus_mask, rates_dampening=1., seed=flags.seed, dtype=tf.float32,
                                                                                     neuropixels_df=flags.v1_neuropixels_df, joint_natural_rate_loss=flags.natural_joint_rate_loss,
                                                                                     natural_joint_direction_mode=flags.natural_joint_direction_mode, natural_rpd_concentration=flags.natural_rpd_concentration,
                                                                                     natural_rpd_stream_id=2)
            # Add OSI/DSI regularizer for the annulus
            v1_annulus_OSI_DSI_Loss = losses.OrientationSelectivityLoss(network=networks['v1'], osi_cost=flags.annulus_loss_weight*flags.osi_cost, area='v1',
                                                                        pre_delay=delays[0], post_delay=delays[1],
                                                                        dtype=tf.float32, core_mask=v1_annulus_mask,
                                                                        method=flags.osi_loss_method,
                                                                        subtraction_ratio=flags.osi_loss_subtraction_ratio,
                                                                        layer_info=v1_layer_info,
                                                                        neuropixels_df=flags.v1_neuropixels_df,
                                                                        data_dir=flags.data_dir)

        if lm_annulus_mask is not None:
            lm_annulus_spont_rate_regularizer = losses.SpikeRateDistributionTarget(networks['lm'], stimulus_type='spontaneous', rate_cost=flags.annulus_loss_weight*flags.rate_cost, pre_delay=delays[0], post_delay=delays[1],
                                                                                   data_dir=flags.data_dir, area='lm', core_mask=lm_annulus_mask, rates_dampening=1., seed=flags.seed, dtype=tf.float32,
                                                                                   neuropixels_df=flags.lm_neuropixels_df)
            lm_annulus_evoked_rate_regularizer = losses.SpikeRateDistributionTarget(networks['lm'], stimulus_type='drifting_gratings', rate_cost=flags.annulus_loss_weight*flags.rate_cost, pre_delay=delays[0], post_delay=delays[1],
                                                                                    data_dir=flags.data_dir, area='lm', core_mask=lm_annulus_mask, rates_dampening=1., seed=flags.seed, dtype=tf.float32,
                                                                                    neuropixels_df=flags.lm_neuropixels_df)
            lm_annulus_natural_rate_regularizer = losses.SpikeRateDistributionTarget(networks['lm'], stimulus_type='natural_stimuli', rate_cost=flags.annulus_loss_weight*natural_rate_cost, pre_delay=delays[0], post_delay=delays[1],
                                                                                     data_dir=flags.data_dir, area='lm', core_mask=lm_annulus_mask, rates_dampening=1., seed=flags.seed, dtype=tf.float32,
                                                                                     neuropixels_df=flags.lm_neuropixels_df, joint_natural_rate_loss=flags.natural_joint_rate_loss,
                                                                                     natural_joint_direction_mode=flags.natural_joint_direction_mode, natural_rpd_concentration=flags.natural_rpd_concentration,
                                                                                     natural_rpd_stream_id=3)
            # Add OSI/DSI regularizer for the annulus
            lm_annulus_OSI_DSI_Loss = losses.OrientationSelectivityLoss(network=networks['lm'], osi_cost=flags.annulus_loss_weight*flags.osi_cost, area='lm',
                                                                        pre_delay=delays[0], post_delay=delays[1],
                                                                        dtype=tf.float32, core_mask=lm_annulus_mask,
                                                                        method=flags.osi_loss_method,
                                                                        subtraction_ratio=flags.osi_loss_subtraction_ratio,
                                                                        layer_info=lm_layer_info,
                                                                        neuropixels_df=flags.lm_neuropixels_df,
                                                                        data_dir=flags.data_dir)

        # prediction_layer = model.get_layer('prediction')
        # abstract_layer = model.get_layer('abstract_output')
        # Training needs spikes plus the compact final voltage-penalty states.
        extractor_model = models.build_voltage_tracking_sequence_model(
            model,
            rsnn_layer,
            name="training_voltage_tracking_extractor",
        )
        sequence_and_state_model = models.build_sequence_and_state_model(
            model,
            rsnn_layer,
            name="training_sequence_and_state_extractor",
        )
        # State-only model to avoid storing full sequences when only the final state is needed.
        try:
            state_model = models.build_state_only_model(model, rsnn_layer)
        except Exception as e:
            state_model = None
            print(f"Warning: failed to build state-only model ({e}); using full model for gray state.")

        # These "dummy" zeros are injected to the models membrane voltage
        # Provides the opportunity to compute gradients wrt. membrane voltages at all time steps
        # Not important for general use
        # n_total_neurons = v1_neurons + lm_neurons
        # dummy_zeros = tf.zeros((per_replica_batch_size, flags.seq_len, n_total_neurons), dtype)
        # Initial state used for gray-screen warmup and training.
        # zero_state = rsnn_layer.cell.zero_state_multi_areas(per_replica_batch_size, dtype=dtype)
        # state_variables = tf.nest.map_structure(lambda a: tf.Variable(
        #     a, trainable=False, synchronization=tf.VariableSynchronization.ON_READ
        # ), zero_state)

        # Add other metrics and losses
        train_loss = tf.keras.metrics.Mean()
        train_firing_rate = tf.keras.metrics.Mean()
        train_rate_loss = tf.keras.metrics.Mean()
        train_rate_dg_loss = tf.keras.metrics.Mean()
        train_rate_sp_loss = tf.keras.metrics.Mean()
        train_rate_ns_loss = tf.keras.metrics.Mean()
        train_rate_ns_marginal_loss = tf.keras.metrics.Mean()
        train_rate_ns_mean_projection_loss = tf.keras.metrics.Mean()
        train_rate_ns_contrast_basis_projection_loss = tf.keras.metrics.Mean()
        train_rate_ns_rpd_projection_loss = tf.keras.metrics.Mean()
        train_rate_ns_contrast_rotation_projection_loss = tf.keras.metrics.Mean()
        train_rate_ns_joint_loss = tf.keras.metrics.Mean()
        train_natural_scene_composite = tf.keras.metrics.Mean()
        train_voltage_loss = tf.keras.metrics.Mean()
        train_regularizer_loss = tf.keras.metrics.Mean()
        train_osi_dsi_loss = tf.keras.metrics.Mean()
        train_sync_loss = tf.keras.metrics.Mean()

        val_loss = tf.keras.metrics.Mean()
        val_firing_rate = tf.keras.metrics.Mean()
        val_rate_loss = tf.keras.metrics.Mean()
        val_rate_dg_loss = tf.keras.metrics.Mean()
        val_rate_sp_loss = tf.keras.metrics.Mean()
        val_rate_ns_loss = tf.keras.metrics.Mean()
        val_rate_ns_marginal_loss = tf.keras.metrics.Mean()
        val_rate_ns_mean_projection_loss = tf.keras.metrics.Mean()
        val_rate_ns_contrast_basis_projection_loss = tf.keras.metrics.Mean()
        val_rate_ns_rpd_projection_loss = tf.keras.metrics.Mean()
        val_rate_ns_contrast_rotation_projection_loss = tf.keras.metrics.Mean()
        val_rate_ns_joint_loss = tf.keras.metrics.Mean()
        val_natural_scene_composite = tf.keras.metrics.Mean()
        val_voltage_loss = tf.keras.metrics.Mean()
        val_regularizer_loss = tf.keras.metrics.Mean()
        val_osi_dsi_loss = tf.keras.metrics.Mean()
        val_sync_loss = tf.keras.metrics.Mean()

        def reset_train_metrics():
            train_loss.reset_states(),
            train_firing_rate.reset_states(),
            train_firing_rate.reset_states()
            train_rate_loss.reset_states()
            train_rate_dg_loss.reset_states()
            train_rate_sp_loss.reset_states()
            train_rate_ns_loss.reset_states()
            train_rate_ns_marginal_loss.reset_states()
            train_rate_ns_mean_projection_loss.reset_states()
            train_rate_ns_contrast_basis_projection_loss.reset_states()
            train_rate_ns_rpd_projection_loss.reset_states()
            train_rate_ns_contrast_rotation_projection_loss.reset_states()
            train_rate_ns_joint_loss.reset_states()
            train_natural_scene_composite.reset_states()
            train_voltage_loss.reset_states()
            train_regularizer_loss.reset_states()
            train_osi_dsi_loss.reset_states()
            train_sync_loss.reset_states()

        def reset_validation_metrics():
            val_loss.reset_states(),
            val_firing_rate.reset_states(),
            val_rate_loss.reset_states(),
            val_rate_dg_loss.reset_states(),
            val_rate_sp_loss.reset_states(),
            val_rate_ns_loss.reset_states(),
            val_rate_ns_marginal_loss.reset_states(),
            val_rate_ns_mean_projection_loss.reset_states(),
            val_rate_ns_contrast_basis_projection_loss.reset_states(),
            val_rate_ns_rpd_projection_loss.reset_states(),
            val_rate_ns_contrast_rotation_projection_loss.reset_states(),
            val_rate_ns_joint_loss.reset_states(),
            val_natural_scene_composite.reset_states(),
            val_voltage_loss.reset_states(),
            val_regularizer_loss.reset_states(),
            val_osi_dsi_loss.reset_states(),
            val_sync_loss.reset_states()

        # Precompute spontaneous LGN firing rates once
        def get_spontaneous_lgn_probs():
            cache_dir = os.path.join(flags.data_dir, "tf_data")
            cache_file = os.path.join(
                cache_dir,
                f"spontaneous_lgn_probabilities_n_input_{flags.n_input}_seqlen_{flags.seq_len}.pkl",
            )
            if os.path.exists(cache_file):
                with open(cache_file, "rb") as f:
                    probs = pkl.load(f)
                print("Loaded cached spontaneous LGN firing rates.")
            else:
                # Compute and cache the spontaneous firing rates using proper gray screen function
                rates = next(iter(stim_dataset.generate_gray_screen_stimulus(
                    seq_len=flags.seq_len,
                    n_input=flags.n_input,
                    return_firing_rates=True,
                    data_dir=flags.data_dir,
                    dtype=tf.float32,
                    seed=flags.seed,
                )))
                probs = 1 - tf.exp(-tf.cast(rates, tf.float32) / 1000.0)
                os.makedirs(cache_dir, exist_ok=True)
                with open(cache_file, "wb") as f:
                    pkl.dump(probs.numpy().astype(np.float32), f)
                print("Computed and cached spontaneous LGN firing rates.")

            probs = tf.convert_to_tensor(probs)
            return tf.cast(probs, dtype)

        # Load the spontaneous probabilities once (seq_len, n_input)
        spontaneous_prob_base = get_spontaneous_lgn_probs()
        spontaneous_seed_stream = tf.Variable(0, trainable=False, dtype=tf.int64, name="spontaneous_seed_stream"
        )
        evaluation_spontaneous_seed_stream = tf.Variable(
            0,
            trainable=False,
            dtype=tf.int64,
            name="evaluation_spontaneous_seed_stream",
        )
        evaluation_noise_seed_stream = tf.Variable(
            0,
            trainable=False,
            dtype=tf.int64,
            name="evaluation_noise_seed_stream",
        )

    noise_seed_base = tf.constant(flags.seed, dtype=tf.int64)
    max_int32 = 2**31 - 1
    spont_seed_a = int(flags.seed + 700001) % max_int32
    spont_seed_b = int(flags.seed + 900001) % max_int32
    spontaneous_seed_base = tf.constant(
        [spont_seed_a, spont_seed_b], dtype=tf.int32
    )

    def advance_noise_seed(update_training_state=True):
        # Evaluation has its own stream so it cannot perturb future training.
        stream = (
            rsnn_layer.cell.noise_stream
            if update_training_state
            else evaluation_noise_seed_stream
        )
        stream_id = stream.assign_add(tf.constant(1, dtype=tf.int64))
        rsnn_layer.cell.noise_seed.assign(noise_seed_base + stream_id)

    def next_spontaneous_seed(update_training_state=True):
        stream = (
            spontaneous_seed_stream
            if update_training_state
            else evaluation_spontaneous_seed_stream
        )
        stream_id = stream.assign_add(tf.constant(1, dtype=tf.int64))
        seed = tf.random.experimental.stateless_fold_in(
            spontaneous_seed_base, tf.cast(stream_id, tf.int32)
        )
        replica_context = tf.distribute.get_replica_context()
        if replica_context is None:
            replica_id = tf.constant(0, dtype=tf.int32)
        else:
            replica_id = tf.cast(replica_context.replica_id_in_sync_group, tf.int32)
        return tf.random.experimental.stateless_fold_in(seed, replica_id)

    if flags.gradient_checkpointing:
        segmented_extractor = models.SegmentedRecomputeRunner(
            sequence_and_state_model,
            sequence_length=flags.seq_len,
            chunk_size=flags.gradient_checkpoint_chunk_size,
            differentiate_inputs=False, # if one wants to train the LGN units (not the input_weights) then this should be True
        )

        def extractor_forward(x, state_vars):
            flat_outputs = segmented_extractor(x, state_vars)
            sequence_outputs = flat_outputs[
                :segmented_extractor.n_sequence_outputs
            ]
            final_state = flat_outputs[
                segmented_extractor.n_sequence_outputs:
            ]
            return sequence_outputs + tuple(final_state[-2:])
    else:
        def extractor_forward(x, state_vars):
            return extractor_model((x, state_vars))

    def run_extractor(x, state_variables, update_training_state=True):
        if x.dtype == tf.bool:
            x = tf.cast(x, dtype)
        advance_noise_seed(update_training_state=update_training_state)
        # Gray-screen warmup initializes neural state only; its voltage loss is
        # not part of any training or validation objective.
        state_variables = tuple(state_variables[:-2]) + (
            tf.zeros_like(state_variables[-2]),
            tf.zeros_like(state_variables[-1]),
        )
        return extractor_forward(x, state_variables)

    def _unpack_voltage_tracking_outputs(outputs):
        if flags.return_voltage_sequences:
            v1_z, v1_v, lm_z, lm_v, v1_penalty, lm_penalty = outputs
            return (v1_z, v1_v, lm_z, lm_v), v1_z, lm_z, v1_penalty, lm_penalty
        v1_z, lm_z, v1_penalty, lm_penalty = outputs
        return (v1_z, lm_z), v1_z, lm_z, v1_penalty, lm_penalty

    def _firing_rate(v1_z, lm_z):
        v1_sum = tf.reduce_sum(tf.cast(v1_z, tf.float32))
        lm_sum = tf.reduce_sum(tf.cast(lm_z, tf.float32))
        total_count = tf.cast(tf.size(v1_z) + tf.size(lm_z), tf.float32)
        return (v1_sum + lm_sum) / total_count

    def _voltage_loss(v1_penalty, lm_penalty, sequence_length):
        normalizer = tf.cast(sequence_length, tf.float32)
        v1_voltage_loss = tf.reduce_mean(tf.cast(v1_penalty, tf.float32)) / normalizer
        lm_voltage_loss = tf.reduce_mean(tf.cast(lm_penalty, tf.float32)) / normalizer
        return (v1_voltage_loss + lm_voltage_loss) * tf.cast(flags.voltage_cost / 2, tf.float32)

    def _zero_branch_aux(firing_rate):
        zero = tf.constant(0.0, dtype=tf.float32)
        return {
            "rate_loss": zero,
            "voltage_loss": zero,
            "osi_dsi_loss": zero,
            "sync_loss": zero,
            "firing_rate": firing_rate,
        }

    def _branch_total_loss(branch_aux):
        return tf.nn.scale_regularization_loss(
            branch_aux["rate_loss"]
            + branch_aux["voltage_loss"]
            + branch_aux["osi_dsi_loss"]
            + branch_aux["sync_loss"]
        )

    def _compute_regularizers_loss():
        regularizers_loss = tf.constant(0.0, dtype=tf.float32)
        if flags.train_recurrent_v1 and flags.recurrent_weight_regularization > 0:
            v1_recurrent_stiff_regularizer = v1_recurrent_regularizer(
                rsnn_layer.cell.v1.recurrent_weight_values
            )
            v1_l2_recurrent_regularizer_values = v1_l2_recurrent_regularizer(
                rsnn_layer.cell.v1.recurrent_weight_values
            )
            regularizers_loss += (
                v1_recurrent_stiff_regularizer
                + 1e-2 * v1_l2_recurrent_regularizer_values
            )

        if flags.train_recurrent_lm and flags.recurrent_weight_regularization > 0:
            lm_recurrent_stiff_regularizer = lm_recurrent_regularizer(
                rsnn_layer.cell.lm.recurrent_weight_values
            )
            lm_l2_recurrent_regularizer_values = lm_l2_recurrent_regularizer(
                rsnn_layer.cell.lm.recurrent_weight_values
            )
            regularizers_loss += (
                lm_recurrent_stiff_regularizer
                + 1e-2 * lm_l2_recurrent_regularizer_values
            )

        return regularizers_loss

    def _aggregate_aux(aux):
        return {
            "rate_loss": stimulus_weights.weighted_sum(
                {name: aux[name]["rate_loss"] for name in stimulus_names}
            ),
            "voltage_loss": stimulus_weights.weighted_sum(
                {name: aux[name]["voltage_loss"] for name in stimulus_names}
            ),
            "osi_dsi_loss": stimulus_weights.weighted_sum(
                {name: aux[name]["osi_dsi_loss"] for name in stimulus_names}
            ),
            "sync_loss": stimulus_weights.weighted_sum(
                {name: aux[name]["sync_loss"] for name in stimulus_names}
            ),
            "firing_rate": stimulus_weights.weighted_mean(
                {name: aux[name]["firing_rate"] for name in stimulus_names}
            ),
        }

    def _compute_dg_aux(
        v1_z_dg,
        lm_z_dg,
        v1_penalty_dg,
        lm_penalty_dg,
        sequence_length,
        y_gratings,
        trim,
        update_training_state,
    ):
        dg_firing_rate = _firing_rate(v1_z_dg, lm_z_dg)
        if not stimulus_weights.is_enabled("dg"):
            return _zero_branch_aux(dg_firing_rate)
        dg_voltage_loss = _voltage_loss(v1_penalty_dg, lm_penalty_dg, sequence_length)

        v1_rates = v1_evoked_rate_regularizer.rates_from_spikes(
            v1_z_dg, trim
        )
        lm_rates = lm_evoked_rate_regularizer.rates_from_spikes(
            lm_z_dg, trim
        )
        update_rate_emas(
            v1_ema,
            lm_ema,
            v1_rates,
            lm_rates,
            ema_decay,
            update_state=update_training_state,
        )

        dg_rate_loss = (
            v1_evoked_rate_regularizer.loss_from_rates(v1_rates)
            + lm_evoked_rate_regularizer.loss_from_rates(lm_rates)
        ) / 2
        dg_osi_loss = (
            v1_OSI_DSI_Loss(
                v1_z_dg, y_gratings, trim, normalizer=v1_ema
            )
            + lm_OSI_DSI_Loss(
                lm_z_dg, y_gratings, trim, normalizer=lm_ema
            )
        ) / 2
        dg_sync_loss = (
            v1_evoked_sync_loss(
                v1_z_dg, trim, advance_seed=update_training_state
            )
            + lm_evoked_sync_loss(
                lm_z_dg, trim, advance_seed=update_training_state
            )
        ) / 2

        if v1_annulus_mask is not None:
            dg_rate_loss += (
                v1_annulus_evoked_rate_regularizer.loss_from_rates(v1_rates) / 2
            )
            dg_osi_loss += v1_annulus_OSI_DSI_Loss(
                v1_z_dg, y_gratings, trim, normalizer=v1_ema
            ) / 2
        if lm_annulus_mask is not None:
            dg_rate_loss += (
                lm_annulus_evoked_rate_regularizer.loss_from_rates(lm_rates) / 2
            )
            dg_osi_loss += lm_annulus_OSI_DSI_Loss(
                lm_z_dg, y_gratings, trim, normalizer=lm_ema
            ) / 2

        return {
            "rate_loss": dg_rate_loss,
            "voltage_loss": dg_voltage_loss,
            "osi_dsi_loss": dg_osi_loss,
            "sync_loss": dg_sync_loss,
            "firing_rate": dg_firing_rate,
        }

    def _compute_sp_aux(
        v1_z_sp,
        lm_z_sp,
        v1_penalty_sp,
        lm_penalty_sp,
        sequence_length,
        trim,
        update_training_state,
    ):
        sp_firing_rate = _firing_rate(v1_z_sp, lm_z_sp)
        if not stimulus_weights.is_enabled("sp"):
            return _zero_branch_aux(sp_firing_rate)
        sp_voltage_loss = _voltage_loss(v1_penalty_sp, lm_penalty_sp, sequence_length)

        v1_rates = v1_spont_rate_regularizer.rates_from_spikes(
            v1_z_sp, trim
        )
        lm_rates = lm_spont_rate_regularizer.rates_from_spikes(
            lm_z_sp, trim
        )
        sp_rate_loss = (
            v1_spont_rate_regularizer.loss_from_rates(v1_rates)
            + lm_spont_rate_regularizer.loss_from_rates(lm_rates)
        ) / 2
        sp_sync_loss = (
            v1_spont_sync_loss(
                v1_z_sp, trim, advance_seed=update_training_state
            )
            + lm_spont_sync_loss(
                lm_z_sp, trim, advance_seed=update_training_state
            )
        ) / 2
        sp_osi_loss = tf.constant(0.0, dtype=tf.float32)

        if v1_annulus_mask is not None:
            sp_rate_loss += (
                v1_annulus_spont_rate_regularizer.loss_from_rates(v1_rates) / 2
            )
        if lm_annulus_mask is not None:
            sp_rate_loss += (
                lm_annulus_spont_rate_regularizer.loss_from_rates(lm_rates) / 2
            )

        return {
            "rate_loss": sp_rate_loss,
            "voltage_loss": sp_voltage_loss,
            "osi_dsi_loss": sp_osi_loss,
            "sync_loss": sp_sync_loss,
            "firing_rate": sp_firing_rate,
        }

    def _compute_ns_aux(
        v1_z_ns,
        lm_z_ns,
        v1_penalty_ns,
        lm_penalty_ns,
        sequence_length,
        natural_scene_ids,
        trim,
        projection_step=0,
    ):
        ns_firing_rate = _firing_rate(v1_z_ns, lm_z_ns)
        v1_rates = v1_natural_rate_regularizer.rates_from_spikes(
            v1_z_ns, trim
        )
        lm_rates = lm_natural_rate_regularizer.rates_from_spikes(
            lm_z_ns, trim
        )
        if not stimulus_weights.is_enabled("ns"):
            aux = _zero_branch_aux(ns_firing_rate)
            zero = tf.constant(0.0, dtype=tf.float32)
            aux.update({
                "v1_rates": v1_rates,
                "lm_rates": lm_rates,
                "natural_scene_composite": zero,
                "rate_loss_components": {
                    key: zero
                    for key in losses.NATURAL_SCENE_LOSS_COMPONENT_KEYS
                },
            })
            return aux
        ns_voltage_loss = _voltage_loss(v1_penalty_ns, lm_penalty_ns, sequence_length)

        v1_rate_loss, v1_components = (
            v1_natural_rate_regularizer.loss_from_rates(
                v1_rates,
                scene_ids=natural_scene_ids,
                projection_step=projection_step,
                return_components=True,
            )
        )
        lm_rate_loss, lm_components = (
            lm_natural_rate_regularizer.loss_from_rates(
                lm_rates,
                scene_ids=natural_scene_ids,
                projection_step=projection_step,
                return_components=True,
            )
        )
        ns_rate_loss_components = {
            key: (v1_components[key] + lm_components[key]) / 2
            for key in losses.NATURAL_SCENE_LOSS_COMPONENT_KEYS
        }
        ns_osi_loss = tf.constant(0.0, dtype=tf.float32)
        ns_sync_loss = tf.constant(0.0, dtype=tf.float32)

        if v1_annulus_mask is not None:
            _, v1_annulus_components = (
                v1_annulus_natural_rate_regularizer.loss_from_rates(
                    v1_rates,
                    scene_ids=natural_scene_ids,
                    projection_step=projection_step,
                    return_components=True,
                )
            )
            for key in losses.NATURAL_SCENE_LOSS_COMPONENT_KEYS:
                ns_rate_loss_components[key] += v1_annulus_components[key] / 2
        if lm_annulus_mask is not None:
            _, lm_annulus_components = (
                lm_annulus_natural_rate_regularizer.loss_from_rates(
                    lm_rates,
                    scene_ids=natural_scene_ids,
                    projection_step=projection_step,
                    return_components=True,
                )
            )
            for key in losses.NATURAL_SCENE_LOSS_COMPONENT_KEYS:
                ns_rate_loss_components[key] += lm_annulus_components[key] / 2

        if flags.natural_joint_rate_loss:
            epsilon = tf.constant(1e-8, dtype=tf.float32)
            marginal_reference = tf.maximum(
                natural_scene_marginal_reference, epsilon
            )
            joint_reference = tf.maximum(natural_scene_joint_reference, epsilon)
            reference_scale = marginal_reference + joint_reference
            ns_rate_loss = reference_scale * tf.constant(0.5, tf.float32) * (
                ns_rate_loss_components["marginal"] / marginal_reference
                + ns_rate_loss_components["joint"] / joint_reference
            )
        else:
            ns_rate_loss = ns_rate_loss_components["marginal"]

        return {
            "rate_loss": ns_rate_loss,
            "rate_loss_components": ns_rate_loss_components,
            "natural_scene_composite": ns_rate_loss,
            "voltage_loss": ns_voltage_loss,
            "osi_dsi_loss": ns_osi_loss,
            "sync_loss": ns_sync_loss,
            "firing_rate": ns_firing_rate,
            "v1_rates": v1_rates,
            "lm_rates": lm_rates,
        }

    # @tf.function
    def roll_out(_x, _y, _state_variables, spontaneous=False, trim=True):
        # _initial_state = tf.nest.map_structure(lambda _a: _a.read_value(), state_variables)
        # Access initial state values directly
        _sequences, _v1_z, _lm_z, _v1_penalty, _lm_penalty = (
            _unpack_voltage_tracking_outputs(run_extractor(_x, _state_variables))
        )

        # update state_variables with the new model state
        # new_state = tuple(_out[1:])
        # tf.nest.map_structure(lambda a, b: a.assign(b), state_variables, new_state)

        voltage_loss = _voltage_loss(
            _v1_penalty, _lm_penalty, tf.shape(_x)[1]
        )

        regularizers_loss = tf.constant(0.0, dtype=tf.float32)
        if flags.train_recurrent_v1 and flags.recurrent_weight_regularization > 0:
            v1_recurrent_stiff_regularizer = v1_recurrent_regularizer(rsnn_layer.cell.v1.recurrent_weight_values)
            v1_l2_recurrent_regularizer_values = v1_l2_recurrent_regularizer(rsnn_layer.cell.v1.recurrent_weight_values)
            regularizers_loss += v1_recurrent_stiff_regularizer + \
                1e-2 * v1_l2_recurrent_regularizer_values

        if flags.train_recurrent_lm and flags.recurrent_weight_regularization > 0:
            lm_recurrent_stiff_regularizer = lm_recurrent_regularizer(rsnn_layer.cell.lm.recurrent_weight_values)
            lm_l2_recurrent_regularizer_values = lm_l2_recurrent_regularizer(rsnn_layer.cell.lm.recurrent_weight_values)
            regularizers_loss += lm_recurrent_stiff_regularizer + \
                1e-2 * lm_l2_recurrent_regularizer_values

        if flags.train_interarea_v1_lm:
            regularizers_loss += v1_lm_regularizer(rsnn_layer.cell.v1.interarea_weight_values['lm'])

        if flags.train_interarea_lm_v1:
            regularizers_loss += lm_v1_regularizer(rsnn_layer.cell.lm.interarea_weight_values['v1'])

        if spontaneous:
            v1_rates = v1_spont_rate_regularizer.rates_from_spikes(
                _v1_z, trim
            )
            lm_rates = lm_spont_rate_regularizer.rates_from_spikes(
                _lm_z, trim
            )
            v1_rate_loss = v1_spont_rate_regularizer.loss_from_rates(
                v1_rates
            )
            lm_rate_loss = lm_spont_rate_regularizer.loss_from_rates(
                lm_rates
            )
            rate_loss = (v1_rate_loss + lm_rate_loss) / 2
            osi_dsi_loss = tf.constant(0.0, dtype=tf.float32)
            v1_sync_loss = v1_spont_sync_loss(_v1_z, trim)
            lm_sync_loss = lm_spont_sync_loss(_lm_z, trim)
            sync_loss = (v1_sync_loss + lm_sync_loss) / 2
        else:
            # update the exponential moving average of the firing rates over drifting gratings presentations
            v1_rates = v1_evoked_rate_regularizer.rates_from_spikes(
                _v1_z, trim
            )
            lm_rates = lm_evoked_rate_regularizer.rates_from_spikes(
                _lm_z, trim
            )
            # Update the EMAs
            v1_ema.assign(ema_decay * v1_ema + (1 - ema_decay) * v1_rates)
            lm_ema.assign(ema_decay * lm_ema + (1 - ema_decay) * lm_rates)
            # Compute the final term only after the first three terms have been computed
            v1_rate_loss = v1_evoked_rate_regularizer.loss_from_rates(
                v1_rates
            )
            lm_rate_loss = lm_evoked_rate_regularizer.loss_from_rates(
                lm_rates
            )
            rate_loss = (v1_rate_loss + lm_rate_loss) / 2
            v1_osi_dsi_loss = v1_OSI_DSI_Loss(_v1_z, _y, trim, normalizer=v1_ema)
            lm_osi_dsi_loss = lm_OSI_DSI_Loss(_lm_z, _y, trim, normalizer=lm_ema)
            osi_dsi_loss = (v1_osi_dsi_loss + lm_osi_dsi_loss) / 2
            v1_sync_loss = v1_evoked_sync_loss(_v1_z, trim)
            lm_sync_loss = lm_evoked_sync_loss(_lm_z, trim)
            sync_loss = (v1_sync_loss + lm_sync_loss) / 2

        # Compute each area's annulus independently. Dividing every area term
        # by two preserves the original two-area weighting when both exist.
        if v1_annulus_mask is not None:
            if spontaneous:
                rate_loss += (
                    v1_annulus_spont_rate_regularizer.loss_from_rates(v1_rates)
                    / 2
                )
            else:
                rate_loss += (
                    v1_annulus_evoked_rate_regularizer.loss_from_rates(v1_rates)
                    / 2
                )
                osi_dsi_loss += v1_annulus_OSI_DSI_Loss(
                    _v1_z, _y, trim, normalizer=v1_ema
                ) / 2

        if lm_annulus_mask is not None:
            if spontaneous:
                rate_loss += (
                    lm_annulus_spont_rate_regularizer.loss_from_rates(lm_rates)
                    / 2
                )
            else:
                rate_loss += (
                    lm_annulus_evoked_rate_regularizer.loss_from_rates(lm_rates)
                    / 2
                )
                osi_dsi_loss += lm_annulus_OSI_DSI_Loss(
                    _lm_z, _y, trim, normalizer=lm_ema
                ) / 2

        # Rescale the losses based on the number of replicas
        _loss = tf.nn.scale_regularization_loss(
            rate_loss + voltage_loss + regularizers_loss + osi_dsi_loss + sync_loss)
        _aux = dict(rate_loss=rate_loss, voltage_loss=voltage_loss,
                    regularizer_loss=regularizers_loss, osi_dsi_loss=osi_dsi_loss, sync_loss=sync_loss)

        return _sequences, _loss, _aux

    def roll_out_combined(
        _x_gratings,
        _y,
        _x_spontaneous,
        _x_natural,
        _natural_scene_ids,
        _state_variables,
        trim=True,
        projection_step=0,
        update_training_state=True,
    ):
        x_concat = tf.concat([_x_gratings, _x_spontaneous, _x_natural], axis=0)
        _sequences, _v1_z_full, _lm_z_full, _v1_penalty_full, _lm_penalty_full = (
            _unpack_voltage_tracking_outputs(
                run_extractor(
                    x_concat,
                    _state_variables,
                    update_training_state=update_training_state,
                )
            )
        )

        grating_batch_size_local = tf.shape(_x_gratings)[0]
        spont_batch_size_local = tf.shape(_x_spontaneous)[0]
        natural_batch_size_local = tf.shape(_x_natural)[0]
        split_sizes = tf.stack(
            [grating_batch_size_local, spont_batch_size_local, natural_batch_size_local]
        )
        sequence_length = tf.shape(x_concat)[1]

        v1_z_dg, v1_z_sp, v1_z_ns = tf.split(_v1_z_full, split_sizes, axis=0)
        lm_z_dg, lm_z_sp, lm_z_ns = tf.split(_lm_z_full, split_sizes, axis=0)
        v1_penalty_dg, v1_penalty_sp, v1_penalty_ns = tf.split(_v1_penalty_full, split_sizes, axis=0)
        lm_penalty_dg, lm_penalty_sp, lm_penalty_ns = tf.split(_lm_penalty_full, split_sizes, axis=0)

        regularizers_loss = _compute_regularizers_loss()

        dg_aux = _compute_dg_aux(
            v1_z_dg,
            lm_z_dg,
            v1_penalty_dg,
            lm_penalty_dg,
            sequence_length,
            _y,
            trim,
            update_training_state,
        )
        sp_aux = _compute_sp_aux(
            v1_z_sp,
            lm_z_sp,
            v1_penalty_sp,
            lm_penalty_sp,
            sequence_length,
            trim,
            update_training_state,
        )
        ns_aux = _compute_ns_aux(
            v1_z_ns,
            lm_z_ns,
            v1_penalty_ns,
            lm_penalty_ns,
            sequence_length,
            _natural_scene_ids,
            trim,
            projection_step=projection_step,
        )

        dg_loss = _branch_total_loss(dg_aux)
        sp_loss = _branch_total_loss(sp_aux)
        ns_loss = _branch_total_loss(ns_aux)
        total_loss = stimulus_weights.weighted_sum(
            {"dg": dg_loss, "sp": sp_loss, "ns": ns_loss}
        ) + tf.nn.scale_regularization_loss(regularizers_loss)

        aux = {
            "dg": dg_aux,
            "sp": sp_aux,
            "ns": ns_aux,
            "regularizer_loss": regularizers_loss,
        }

        return _sequences, total_loss, aux

    def train_step(
        _x_gratings,
        _y,
        _x_spontaneous,
        _x_natural,
        _natural_scene_ids,
        state_variables,
        trim=True,
        return_sequences=False,
    ):
        use_loss_scaling = (flags.dtype == 'float16' and hasattr(optimizer, 'get_scaled_loss'))
        # Forward propagation of the model
        with tf.GradientTape() as tape:
            _out, _loss, _aux = roll_out_combined(
                _x_gratings,
                _y,
                _x_spontaneous,
                _x_natural,
                _natural_scene_ids,
                state_variables,
                trim=trim,
                projection_step=optimizer.iterations,
            )
            aggregate_aux = _aggregate_aux(_aux)
            # Scale the loss for float16
            loss_for_grad = optimizer.get_scaled_loss(_loss) if use_loss_scaling else _loss

        grad = tape.gradient(loss_for_grad, model.trainable_variables)
        if use_loss_scaling:
            grad = optimizer.get_unscaled_gradients(grad)

        # The optimizer will aggregate the gradients across replicas automatically before applying them by default,
        # so the losses have to be properly scaled to account for the number of replicas
        # https://www.tensorflow.org/tutorials/distribute/custom_training
        # https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/keras/optimizer_v2/optimizer_v2.py#L741
        # , experimental_aggregate_gradients=False)
        optimizer.apply_gradients(zip(grad, model.trainable_variables))
        # for g, v in zip(grad, model.trainable_variables):
        #     tf.print(f"Gradient for {v.name}: ", g)

        # Backpropagation of the model
        train_loss.update_state(_loss * strategy.num_replicas_in_sync)
        train_firing_rate.update_state(aggregate_aux["firing_rate"])
        train_rate_loss.update_state(aggregate_aux["rate_loss"])
        train_rate_dg_loss.update_state(
            stimulus_weights.dg * _aux["dg"]["rate_loss"]
        )
        train_rate_sp_loss.update_state(
            stimulus_weights.sp * _aux["sp"]["rate_loss"]
        )
        train_rate_ns_loss.update_state(
            stimulus_weights.ns * _aux["ns"]["rate_loss"]
        )
        train_rate_ns_marginal_loss.update_state(
            stimulus_weights.ns
            * _aux["ns"]["rate_loss_components"]["marginal"]
        )
        train_rate_ns_mean_projection_loss.update_state(
            stimulus_weights.ns
            * _aux["ns"]["rate_loss_components"]["mean_projection"]
        )
        train_rate_ns_contrast_basis_projection_loss.update_state(
            stimulus_weights.ns
            * _aux["ns"]["rate_loss_components"]["contrast_basis_projection"]
        )
        train_rate_ns_rpd_projection_loss.update_state(
            stimulus_weights.ns
            * _aux["ns"]["rate_loss_components"]["rpd_projection"]
        )
        train_rate_ns_contrast_rotation_projection_loss.update_state(
            stimulus_weights.ns
            * _aux["ns"]["rate_loss_components"]["contrast_rotation_projection"]
        )
        train_rate_ns_joint_loss.update_state(
            stimulus_weights.ns * _aux["ns"]["rate_loss_components"]["joint"]
        )
        train_natural_scene_composite.update_state(
            stimulus_weights.ns * _aux["ns"]["natural_scene_composite"]
        )
        train_voltage_loss.update_state(aggregate_aux["voltage_loss"])
        train_regularizer_loss.update_state(_aux["regularizer_loss"])
        train_sync_loss.update_state(aggregate_aux["sync_loss"])
        train_osi_dsi_loss.update_state(aggregate_aux["osi_dsi_loss"])

        if return_sequences:
            return _loss, _aux, _out  # , grad

    @tf.function
    def distributed_train_step(
        x_gratings,
        y,
        x_spontaneous,
        x_natural,
        natural_scene_ids,
        state_variables,
        trim,
        return_sequences=False,
    ):
        return strategy.run(
            train_step,
            args=(
                x_gratings,
                y,
                x_spontaneous,
                x_natural,
                natural_scene_ids,
                state_variables,
                trim,
                return_sequences,
            ),
        )

    # @tf.function
    def split_train_step(
        _x,
        _y,
        state_variables,
        _x_spontaneous,
        _x_natural,
        _natural_scene_ids,
        trim=True,
        capture_spikes=False,
    ):
        if capture_spikes:
            _loss, _, _out = distributed_train_step(
                _x,
                _y,
                _x_spontaneous,
                _x_natural,
                _natural_scene_ids,
                state_variables,
                trim,
                return_sequences=True,
            )

            local_sequences = tf.nest.map_structure(
                lambda value: strategy.experimental_local_results(value)[0],
                _out,
            )
            if flags.return_voltage_sequences:
                v1_z_full, v1_v_full, lm_z_full, lm_v_full = local_sequences
            else:
                v1_z_full, lm_z_full = local_sequences
            local_x = strategy.experimental_local_results(_x)[0]
            grating_batch_size_local = tf.shape(local_x)[0]
            local_x_spont = strategy.experimental_local_results(_x_spontaneous)[0]
            spont_batch_size_local = tf.shape(local_x_spont)[0]
            local_x_natural = strategy.experimental_local_results(_x_natural)[0]
            natural_batch_size_local = tf.shape(local_x_natural)[0]
            split_sizes = tf.stack(
                [grating_batch_size_local, spont_batch_size_local,
                    natural_batch_size_local]
            )

            v1_spikes, v1_spikes_spont, v1_spikes_ns = tf.split(v1_z_full, split_sizes, axis=0)
            lm_spikes, lm_spikes_spont, lm_spikes_ns = tf.split(lm_z_full, split_sizes, axis=0)
            model_spikes = (v1_spikes, lm_spikes, v1_spikes_spont, lm_spikes_spont, v1_spikes_ns, lm_spikes_ns)
            if flags.return_voltage_sequences:
                v1_v_dg, _, _ = tf.split(v1_v_full, split_sizes, axis=0)
                lm_v_dg, _, _ = tf.split(lm_v_full, split_sizes, axis=0)
                tf.print('  V1 voltage gratings: ', tf.reduce_max(v1_v_dg), tf.reduce_min(v1_v_dg))
                tf.print('  LM voltage gratings: ', tf.reduce_max(lm_v_dg), tf.reduce_min(lm_v_dg))

        else:
            distributed_train_step(
                _x,
                _y,
                _x_spontaneous,
                _x_natural,
                _natural_scene_ids,
                state_variables,
                trim,
                return_sequences=False,
            )
            model_spikes = (None, None, None, None, None, None)

        # _loss_gratings = strategy.reduce(tf.distribute.ReduceOp.SUM, _loss_gratings, axis=None)
        # _loss_spontaneous = strategy.reduce(tf.distribute.ReduceOp.SUM, _loss_spontaneous, axis=None)
        # _loss = _loss_gratings + _loss_spontaneous

        # Accumulate gradients
        # average_gradients = [tf.add(g1, g2) / 2.0 for g1, g2 in zip(grad_spontaneous, grad_gratings)]
        # average_gradients = [g / 2.0 for g in accumulated_gradients]
        # Apply average gradients
        # optimizer.apply_gradients(zip(average_gradients, model.trainable_variables))

        # ### Backpropagation of the model
        # train_loss.update_state(_loss)
        # rate = tf.reduce_mean(tf.concat([_out_gratings[0][0][0], _out_gratings[0][2][0], _out_spontaneous[0][0][0], _out_spontaneous[0][2][0]], axis=-1))
        # train_firing_rate.update_state(rate)
        # train_rate_loss.update_state(_aux_gratings['rate_loss'] + _aux_spontaneous['rate_loss'])
        # train_voltage_loss.update_state(_aux_gratings['voltage_loss'] + _aux_spontaneous['voltage_loss'])
        # train_regularizer_loss.update_state(_aux_gratings['regularizer_loss'] + _aux_spontaneous['regularizer_loss'])
        # train_osi_dsi_loss.update_state(_aux_gratings['osi_dsi_loss'] + _aux_spontaneous['osi_dsi_loss'])
        # train_sync_loss.update_state(_aux_gratings['sync_loss'] + _aux_spontaneous['sync_loss'])

        rate_loss = train_rate_loss.result()
        voltage_loss = train_voltage_loss.result()
        regularizers_loss = train_regularizer_loss.result()
        sync_loss = train_sync_loss.result()
        osi_dsi_loss = train_osi_dsi_loss.result()
        # For display, reconstruct the total as the sum of the individual components so it matches the printout
        _loss = rate_loss + voltage_loss + regularizers_loss + osi_dsi_loss + sync_loss
        rate = train_firing_rate.result()

        # rate_loss = _aux_gratings['rate_loss'] + _aux_spontaneous['rate_loss']
        # voltage_loss = _aux_gratings['voltage_loss'] + _aux_spontaneous['voltage_loss']
        # regularizers_loss = _aux_gratings['regularizer_loss'] + _aux_spontaneous['regularizer_loss']
        # osi_dsi_loss = _aux_gratings['osi_dsi_loss'] + _aux_spontaneous['osi_dsi_loss']
        # sync_loss = _aux_gratings['sync_loss'] + _aux_spontaneous['sync_loss']

        step_values = [_loss, rate, rate_loss, voltage_loss,
                       regularizers_loss, osi_dsi_loss, sync_loss]
        # step_values = [
        #     strategy.reduce(tf.distribute.ReduceOp.MEAN if i in [1] else tf.distribute.ReduceOp.SUM, value, axis=None)
        #     for i, value in enumerate(step_values)
        # ]

        return model_spikes, step_values

    def validation_step(
        x,
        y,
        x_spontaneous,
        x_natural,
        natural_scene_ids,
        state_variables,
        output_spikes=True,
    ):
        _out, _loss, _aux = roll_out_combined(
            x,
            y,
            x_spontaneous,
            x_natural,
            natural_scene_ids,
            state_variables,
            projection_step=tf.constant(0, tf.int64),
            update_training_state=False,
        )
        aggregate_aux = _aggregate_aux(_aux)

        natural_sample_weight = tf.cast(tf.shape(x_natural)[0], tf.float32)
        val_loss.update_state(
            _loss * strategy.num_replicas_in_sync,
            sample_weight=natural_sample_weight,
        )
        val_firing_rate.update_state(
            aggregate_aux["firing_rate"], sample_weight=natural_sample_weight
        )
        val_rate_loss.update_state(
            aggregate_aux["rate_loss"], sample_weight=natural_sample_weight
        )
        val_rate_dg_loss.update_state(
            stimulus_weights.dg * _aux["dg"]["rate_loss"],
            sample_weight=natural_sample_weight,
        )
        val_rate_sp_loss.update_state(
            stimulus_weights.sp * _aux["sp"]["rate_loss"],
            sample_weight=natural_sample_weight,
        )
        val_rate_ns_loss.update_state(
            stimulus_weights.ns * _aux["ns"]["rate_loss"],
            sample_weight=natural_sample_weight,
        )
        val_rate_ns_marginal_loss.update_state(
            stimulus_weights.ns
            * _aux["ns"]["rate_loss_components"]["marginal"],
            sample_weight=natural_sample_weight,
        )
        val_rate_ns_mean_projection_loss.update_state(
            stimulus_weights.ns
            * _aux["ns"]["rate_loss_components"]["mean_projection"],
            sample_weight=natural_sample_weight,
        )
        val_rate_ns_contrast_basis_projection_loss.update_state(
            stimulus_weights.ns
            * _aux["ns"]["rate_loss_components"]["contrast_basis_projection"],
            sample_weight=natural_sample_weight,
        )
        val_rate_ns_rpd_projection_loss.update_state(
            stimulus_weights.ns
            * _aux["ns"]["rate_loss_components"]["rpd_projection"],
            sample_weight=natural_sample_weight,
        )
        val_rate_ns_contrast_rotation_projection_loss.update_state(
            stimulus_weights.ns
            * _aux["ns"]["rate_loss_components"]["contrast_rotation_projection"],
            sample_weight=natural_sample_weight,
        )
        val_rate_ns_joint_loss.update_state(
            stimulus_weights.ns * _aux["ns"]["rate_loss_components"]["joint"],
            sample_weight=natural_sample_weight,
        )
        val_natural_scene_composite.update_state(
            stimulus_weights.ns * _aux["ns"]["natural_scene_composite"],
            sample_weight=natural_sample_weight,
        )
        val_voltage_loss.update_state(
            aggregate_aux["voltage_loss"], sample_weight=natural_sample_weight
        )
        val_regularizer_loss.update_state(
            _aux["regularizer_loss"], sample_weight=natural_sample_weight
        )
        val_sync_loss.update_state(
            aggregate_aux["sync_loss"], sample_weight=natural_sample_weight
        )
        val_osi_dsi_loss.update_state(
            aggregate_aux["osi_dsi_loss"], sample_weight=natural_sample_weight
        )

        if flags.return_voltage_sequences:
            v1_z_full, _, lm_z_full, _ = _out
        else:
            v1_z_full, lm_z_full = _out
        grating_batch_size_local = tf.shape(x)[0]
        spont_batch_size_local = tf.shape(x_spontaneous)[0]
        natural_batch_size_local = tf.shape(x_natural)[0]
        split_sizes = tf.stack([grating_batch_size_local, spont_batch_size_local, natural_batch_size_local])

        v1_spikes, v1_spikes_spont, v1_spikes_ns = tf.split(v1_z_full, split_sizes, axis=0)
        lm_spikes, lm_spikes_spont, lm_spikes_ns = tf.split(lm_z_full, split_sizes, axis=0)

        # tf.nest.map_structure(lambda _a, _b: _a.assign(_b), list(state_variables), _out[1:])
        if output_spikes:
            model_spikes = (v1_spikes, lm_spikes, v1_spikes_spont, lm_spikes_spont, v1_spikes_ns, lm_spikes_ns)
            return model_spikes, _aux["ns"]["v1_rates"], _aux["ns"]["lm_rates"]
        return _aux["ns"]["v1_rates"], _aux["ns"]["lm_rates"]

    @tf.function
    def distributed_validation_step(
        x,
        y,
        x_spontaneous,
        x_natural,
        natural_scene_ids,
        state_variables,
        output_spikes=True,
    ):
        return strategy.run(
            validation_step,
            args=(
                x,
                y,
                x_spontaneous,
                x_natural,
                natural_scene_ids,
                state_variables,
                output_spikes,
            ),
        )

    ### LGN INPUT ###
    # Keep the standard five panels when possible, but add panels if a
    # smaller natural batch would otherwise provide fewer than two slots per
    # held-out scene.
    validation_steps = stim_dataset.balanced_scene_panel_count(
        flags.natural_scenes_validation_count, natural_batch_size
    )

    def get_functional_dataset_fn(
        regular=False,
        natural_scene_subset="train",
        finite_natural_scene_pass=False,
        natural_scene_panels=None,
    ):
        def _f(input_context):
            per_replica_batch_size_local = input_context.get_per_replica_batch_size(global_batch_size)
            if per_replica_batch_size_local != per_replica_batch_size:
                raise ValueError(
                    "Distributed input context produced an unexpected per-replica "
                    f"batch size: {per_replica_batch_size_local} != "
                    f"{per_replica_batch_size}."
                )
            grating_batch_size_local = grating_batch_size
            natural_batch_size_local = natural_batch_size

            pipeline_seed = flags.seed + 10000 + int(input_context.input_pipeline_id)
            # Single unified generator avoids concurrent GPU operations from
            # separate generators that cause shape-mismatch errors in _fold_in_seed.
            return stim_dataset.generate_functional_multistim_dataset(
                seq_len=flags.seq_len,
                pre_delay=delays[0],
                post_delay=delays[1],
                grating_batch_size=grating_batch_size_local,
                natural_batch_size=natural_batch_size_local,
                n_input=flags.n_input,
                current_input=flags.current_input,
                regular=regular,
                temporal_f=flags.temporal_f,
                rotation=flags.rotation,
                billeh_phase=True,
                data_dir=flags.data_dir,
                bmtk_compat=flags.bmtk_compat_lgn,
                cache_dir=flags.natural_scenes_cache_dir,
                experiment_id=flags.natural_scenes_experiment_id,
                natural_scene_subset=natural_scene_subset,
                natural_scenes_validation_count=flags.natural_scenes_validation_count,
                natural_scenes_split_seed=flags.natural_scenes_split_seed,
                finite_natural_scene_pass=finite_natural_scene_pass,
                natural_scene_panels=natural_scene_panels,
                dtype=dtype,
                seed=pipeline_seed,
            )

        return _f


    # Generate spontaneous spikes efficiently
    def generate_spontaneous_spikes(batch_size, update_training_state=True):
        batch_size = tf.cast(batch_size, tf.int32)
        # Stop gradient for shape operations to avoid int32 dtype warnings
        prob_shape = tf.stop_gradient(tf.shape(spontaneous_prob_base))
        target_shape = tf.concat([[batch_size], prob_shape], axis=0)
        random_uniform = tf.random.stateless_uniform(
            target_shape,
            seed=next_spontaneous_seed(
                update_training_state=update_training_state
            ),
            dtype=dtype,
        )
        return tf.less(random_uniform, spontaneous_prob_base[tf.newaxis, ...])

    @tf.function
    def distributed_generate_spontaneous_spikes(
        batch_size, update_training_state=True
    ):
        return strategy.run(
            generate_spontaneous_spikes,
            args=(batch_size, update_training_state),
        )

    def generate_gray_state(batch_size, update_training_state=True):
        batch_size = tf.cast(batch_size, tf.int32)
        x = generate_spontaneous_spikes(
            batch_size, update_training_state=update_training_state
        )
        if x.dtype == tf.bool:
            x = tf.cast(x, dtype)
        init_state = rsnn_layer.cell.zero_state_multi_areas(batch_size, dtype=dtype)
        advance_noise_seed(update_training_state=update_training_state)
        inputs = [x]
        inputs.extend(list(init_state))
        if state_model is None:
            print(
                "No state-only model provided; running the full sequence "
                "model for gray-state warm-up."
            )
        return run_gray_state_rollout(
            state_model,
            sequence_and_state_model,
            tuple(inputs),
        )

    @tf.function
    def distributed_generate_gray_state(batch_size):
        # Run generate_gray_state on each replica
        return strategy.run(generate_gray_state, args=(batch_size,))

    @tf.function
    def distributed_generate_validation_state(x_gratings, x_spontaneous, x_natural):
        def _generate(local_gratings, local_spontaneous, local_natural):
            local_batch_size = (
                tf.shape(local_gratings)[0]
                + tf.shape(local_spontaneous)[0]
                + tf.shape(local_natural)[0]
            )
            return generate_gray_state(
                local_batch_size, update_training_state=False
            )

        return strategy.run(
            _generate, args=(x_gratings, x_spontaneous, x_natural)
        )

    train_data_set = strategy.distribute_datasets_from_function(
        get_functional_dataset_fn(natural_scene_subset="train")
    )

    natural_scenes_path = os.path.join(
        flags.natural_scenes_cache_dir, "natural_scenes.npy"
    )
    if not os.path.exists(natural_scenes_path):
        raise FileNotFoundError(
            "Natural-scene dataset construction did not create the expected "
            f"cache file: {natural_scenes_path}"
        )
    cached_natural_scenes = np.load(natural_scenes_path, mmap_mode="r")
    cached_scene_ids = tuple(range(int(cached_natural_scenes.shape[0])))
    del cached_natural_scenes
    v1_target_scene_ids = tuple(v1_natural_rate_regularizer.natural_scene_ids)
    lm_target_scene_ids = tuple(lm_natural_rate_regularizer.natural_scene_ids)
    if not (
        cached_scene_ids
        == v1_target_scene_ids
        == lm_target_scene_ids
        == tuple(range(118))
    ):
        raise ValueError(
            "Natural-scene ID contract mismatch: generator IDs must be 0..117 "
            "and must map directly to matching VISp/VISl frame columns. "
            f"cache={cached_scene_ids}, VISp={v1_target_scene_ids}, "
            f"VISl={lm_target_scene_ids}"
        )
    print(
        "Validated natural-scene ID mapping: scene_id k selects "
        "firing_rate_ns_frame_k_hz for k=0..117."
    )
    training_scene_ids, expected_validation_scene_ids = stim_dataset._split_natural_scene_indices(
        len(cached_scene_ids),
        n_validation_scenes=flags.natural_scenes_validation_count,
        split_seed=flags.natural_scenes_split_seed,
    )
    distributed_validation_panels = stim_dataset.distributed_scene_panel_count(
        validation_steps, strategy.num_replicas_in_sync
    )
    validation_panels = stim_dataset.fixed_balanced_scene_panels(
        expected_validation_scene_ids,
        panel_size=natural_batch_size,
        n_panels=distributed_validation_panels,
        seed=flags.natural_scenes_split_seed,
    )
    calibration_panels = stim_dataset.fixed_scene_panels(
        training_scene_ids,
        panel_size=natural_batch_size,
        n_panels=distributed_validation_panels,
        seed=flags.seed + 7001,
    )
    diagnostic_steps = stim_dataset.natural_scene_validation_steps(
        flags.natural_scenes_validation_count,
        natural_batch_size,
        strategy.num_replicas_in_sync,
    )
    diagnostic_panels = stim_dataset.fixed_complete_scene_panels(
        expected_validation_scene_ids,
        panel_size=natural_batch_size,
        n_panels=stim_dataset.distributed_scene_panel_count(
            diagnostic_steps, strategy.num_replicas_in_sync
        ),
        seed=flags.natural_scenes_split_seed + 1,
    )
    val_data_set = strategy.distribute_datasets_from_function(
        get_functional_dataset_fn(
            natural_scene_subset="validation",
            natural_scene_panels=validation_panels,
        )
    )
    calibration_data_set = strategy.distribute_datasets_from_function(
        get_functional_dataset_fn(
            natural_scene_subset="train",
            natural_scene_panels=calibration_panels,
        )
    )
    diagnostic_val_data_set = strategy.distribute_datasets_from_function(
        get_functional_dataset_fn(
            natural_scene_subset="validation",
            natural_scene_panels=diagnostic_panels,
        )
    )
    print(
        "Natural-scene validation uses "
        f"{validation_steps} fixed balanced panels of "
        f"{natural_batch_size} held-out scenes per replica."
    )

    def calibrate_natural_scene_loss_references():
        """Measure fixed raw scales before the first optimizer update."""
        reset_validation_metrics()
        calibration_iterator = iter(calibration_data_set)
        for _ in range(validation_steps):
            x, y, x_natural, scene_ids = next(calibration_iterator)
            x_spontaneous = distributed_generate_spontaneous_spikes(
                gray_batch_size, update_training_state=False
            )
            state = distributed_generate_validation_state(
                x, x_spontaneous, x_natural
            )
            distributed_validation_step(
                x, y, x_spontaneous, x_natural, scene_ids, state,
                output_spikes=False,
            )
        epsilon = tf.constant(1e-8, dtype=tf.float32)
        natural_scene_marginal_reference.assign(tf.maximum(
            val_rate_ns_marginal_loss.result(), epsilon
        ))
        natural_scene_joint_reference.assign(tf.maximum(
            val_rate_ns_joint_loss.result(), epsilon
        ))
        reset_validation_metrics()
        print(
            "Natural-scene calibration references: "
            f"marginal={natural_scene_marginal_reference.numpy():.6g}, "
            f"joint={natural_scene_joint_reference.numpy():.6g}"
        )

    restored_natural_scene_references = False
    if resumed_from_checkpoint and os.path.exists(train_end_data_path):
        with open(train_end_data_path, "rb") as reference_file:
            reference_data = pkl.load(reference_file)
        marginal_reference = reference_data.get("natural_scene_marginal_reference")
        joint_reference = reference_data.get("natural_scene_joint_reference")
        stored_protocol = reference_data.get(
            "natural_scene_projection_protocol"
        )
        if (
            marginal_reference is not None
            and joint_reference is not None
            and natural_scene_projection_protocol_compatible(
                stored_protocol, natural_scene_projection_protocol
            )
        ):
            natural_scene_marginal_reference.assign(marginal_reference)
            natural_scene_joint_reference.assign(joint_reference)
            restored_natural_scene_references = True
        elif marginal_reference is not None or joint_reference is not None:
            print(
                "Stored natural-scene calibration uses a different projection "
                "protocol; recalibrating references."
            )
    if flags.natural_joint_rate_loss and not restored_natural_scene_references:
        calibrate_natural_scene_loss_references()

    ############################ TRAINING #############################
    stop = False
    # Initialize your callbacks
    metric_keys = ['train_loss', 'train_firing_rate', 'train_rate_loss', 'train_voltage_loss',
                   'train_regularizer_loss', 'train_osi_dsi_loss', 'train_sync_loss',
                   'train_natural_scene_composite', 'val_loss', 'val_firing_rate',
                   'val_rate_loss', 'val_voltage_loss', 'val_regularizer_loss',
                   'val_osi_dsi_loss', 'val_sync_loss', 'val_natural_scene_composite']

    callbacks = Callbacks(networks, lgn_inputs, bkg_inputs, model, optimizer, flags, logdir, strategy,
                          metric_keys, pre_delay=delays[0], post_delay=delays[1],
                          checkpoint=checkpoint, spontaneous_training=flags.spontaneous_training,
                          natural_scene_rate_targets={
                              "v1": v1_natural_rate_regularizer.natural_scene_diagnostic_targets,
                              "lm": lm_natural_rate_regularizer.natural_scene_diagnostic_targets,
                          },
                          expected_validation_scene_ids=expected_validation_scene_ids)

    callbacks.on_train_begin()

    def capture_full_natural_scene_diagnostics():
        """Return one no-repeat held-out response table without scoring it."""
        diagnostic_iterator = iter(diagnostic_val_data_set)
        id_chunks, v1_chunks, lm_chunks = [], [], []
        for _ in range(diagnostic_steps):
            x_diagnostic, y_diagnostic, x_natural_diagnostic, ids_diagnostic = (
                next(diagnostic_iterator)
            )
            x_spontaneous_diagnostic = distributed_generate_spontaneous_spikes(
                gray_batch_size, update_training_state=False
            )
            state_diagnostic = distributed_generate_validation_state(
                x_diagnostic,
                x_spontaneous_diagnostic,
                x_natural_diagnostic,
            )
            v1_rates, lm_rates = distributed_validation_step(
                x_diagnostic, y_diagnostic, x_spontaneous_diagnostic,
                x_natural_diagnostic, ids_diagnostic, state_diagnostic,
                output_spikes=False,
            )
            for ids_replica, v1_replica, lm_replica in zip(
                strategy.experimental_local_results(ids_diagnostic),
                strategy.experimental_local_results(v1_rates),
                strategy.experimental_local_results(lm_rates),
            ):
                id_chunks.append(ids_replica.numpy())
                v1_chunks.append(v1_replica.numpy().astype(np.float32))
                lm_chunks.append(lm_replica.numpy().astype(np.float32))
        all_ids = np.concatenate(id_chunks)
        all_rates = {
            "v1": np.concatenate(v1_chunks),
            "lm": np.concatenate(lm_chunks),
        }
        selected_positions = []
        for scene_id in expected_validation_scene_ids:
            matches = np.flatnonzero(all_ids == scene_id)
            if matches.size == 0:
                raise ValueError(
                    f"Distributed diagnostics omitted held-out scene {scene_id}."
                )
            selected_positions.append(int(matches[0]))
        selected_positions = np.asarray(selected_positions, dtype=np.int64)
        return all_ids[selected_positions], {
            area: rates[selected_positions] for area, rates in all_rates.items()
        }

    # chunknum = 1
    # max_working_fr = {}   # defined for each chunknum
    n_prev_epochs = flags.run_session * flags.n_epochs

    # import datetime
    # profiler_logdir = f"{logdir}/logs/profile/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    # # Set steps to profile
    # profile_start_step = 1
    # profile_end_step = 7

    # Load the dataset iterator - this must be done inside the epoch loop
    it = iter(train_data_set)
    for epoch in range(n_prev_epochs, n_prev_epochs + flags.n_epochs):
        callbacks.on_epoch_start()
        # Reset the model state to the gray state
        gray_state = distributed_generate_gray_state(per_replica_batch_size)
        # distributed_reset_state(gray_state)

        for step in range(flags.steps_per_epoch):
            callbacks.on_step_start()
            profile_this_step = step == flags.profile_train_step
            cuda_profiler = None
            if profile_this_step:
                if flags.profile_backend == "tensorflow":
                    profiler_logdir = flags.profile_logdir or os.path.join(
                        logdir, "logs", "profile"
                    )
                    os.makedirs(profiler_logdir, exist_ok=True)
                    tf.profiler.experimental.start(
                        profiler_logdir,
                        options=tf.profiler.experimental.ProfilerOptions(
                            host_tracer_level=2,
                            python_tracer_level=1,
                            device_tracer_level=1,
                        ),
                    )
                    print(
                        f"TensorFlow profiler started for epoch={epoch}, step={step}: "
                        f"{profiler_logdir}"
                    )
                elif flags.profile_backend == "cuda":
                    cuda_profiler = ctypes.CDLL(ctypes.util.find_library("cudart"))
                    error_code = cuda_profiler.cudaProfilerStart()
                    if error_code != 0:
                        raise RuntimeError(
                            f"cudaProfilerStart failed with error code {error_code}."
                        )
                    print(f"CUDA profiler range started for epoch={epoch}, step={step}.")
                else:
                    print(f"Timing-only profile started for epoch={epoch}, step={step}.")
                profile_step_start = time()

            # try resetting every iteration
            if flags.reset_every_step:
                gray_state = distributed_generate_gray_state(per_replica_batch_size)

            # Generate LGN spikes
            x, y, x_natural, natural_scene_id = next(it)  # x dtype tf.bool
            x_spontaneous = distributed_generate_spontaneous_spikes(
                gray_batch_size
            )

            try:
                with tf.profiler.experimental.Trace(
                    "functional_train_step",
                    step_num=epoch * flags.steps_per_epoch + step,
                    _r=1,
                ):
                    _, step_values = split_train_step(
                        x,
                        y,
                        gray_state,
                        x_spontaneous,
                        x_natural,
                        natural_scene_id,
                        trim=True,
                        capture_spikes=False,
                    )
                # break
            except tf.errors.ResourceExhaustedError as e:
                raise RuntimeError(
                    "ResourceExhaustedError during training. "
                    f"Epoch={epoch}, step={step}. Reduce memory pressure by lowering "
                    "--batch_size/--seq_len, enabling --sequential_stimuli, or reducing "
                    "network size."
                ) from e

            if profile_this_step:
                # Materialize the small result tensors so all asynchronous device
                # work belongs to the captured step before the profiler stops.
                for value in tf.nest.flatten(step_values):
                    if hasattr(value, "numpy"):
                        value.numpy()
                memory = tf.config.experimental.get_memory_info("GPU:0")
                profile_step_elapsed = time() - profile_step_start
                if flags.profile_backend == "tensorflow":
                    tf.profiler.experimental.stop(save=True)
                elif flags.profile_backend == "cuda":
                    error_code = cuda_profiler.cudaProfilerStop()
                    if error_code != 0:
                        raise RuntimeError(
                            f"cudaProfilerStop failed with error code {error_code}."
                        )
                print(
                    f"{flags.profile_backend.capitalize()} profiler stopped; "
                    f"profiled_step_wall_time={profile_step_elapsed:.4f} s, "
                    f"current_gpu_memory={memory['current'] / 2**20:.2f} MiB, "
                    f"peak_gpu_memory={memory['peak'] / 2**20:.2f} MiB"
                )
                if flags.exit_after_profile:
                    print("Exiting after the requested profiled training step.")
                    return
            callbacks.on_step_end(step_values, y, verbose=True)

        # test_it = iter(test_data_set)
        # test_it = it
        # gray_state = distributed_generate_gray_state(global_batch_size)
        # for step in range(flags.val_steps):
        #     x, y, _, w = next(test_it)
        #     # Generate LGN spikes
        #     x_spontaneous = generate_spontaneous_spikes(spontaneous_prob)
        #     # gray_state = generate_gray_state()
        #     # distributed_reset_state(gray_state)
        #     v1_spikes_spont, lm_spikes_spont = distributed_validation_step(x_spontaneous, y, gray_state, output_spikes=True)
        #     v1_spikes, lm_spikes = distributed_validation_step(x, y, gray_state, output_spikes=True)
        #     # _out, _, _, _, bkg_noise = distributed_roll_out(x_spontaneous, y_spontaneous, w_spontaneous)

        ## VALIDATION AFTER EACH EPOCH
        val_it = iter(val_data_set)
        validation_scene_id_chunks = []
        validation_v1_rate_chunks = []
        validation_lm_rate_chunks = []
        for val_step in range(validation_steps):
            x, y, x_natural, natural_scene_id = next(val_it)
            x_spontaneous = distributed_generate_spontaneous_spikes(
                gray_batch_size, update_training_state=False
            )
            val_gray_state = distributed_generate_validation_state(
                x, x_spontaneous, x_natural
            )
            capture_spikes = val_step == validation_steps - 1
            validation_output = distributed_validation_step(
                x,
                y,
                x_spontaneous,
                x_natural,
                natural_scene_id,
                val_gray_state,
                output_spikes=capture_spikes,
            )
            if capture_spikes:
                model_spikes, v1_rates_step, lm_rates_step = validation_output
            else:
                v1_rates_step, lm_rates_step = validation_output

            local_scene_ids = strategy.experimental_local_results(
                natural_scene_id
            )
            local_v1_rates = strategy.experimental_local_results(v1_rates_step)
            local_lm_rates = strategy.experimental_local_results(lm_rates_step)
            for ids_replica, v1_replica, lm_replica in zip(
                local_scene_ids, local_v1_rates, local_lm_rates
            ):
                validation_scene_id_chunks.append(ids_replica.numpy())
                validation_v1_rate_chunks.append(
                    v1_replica.numpy().astype(np.float32)
                )
                validation_lm_rate_chunks.append(
                    lm_replica.numpy().astype(np.float32)
                )

        # validation_scene_ids = np.concatenate(
        #     validation_scene_id_chunks, axis=0
        # )
        # validation_natural_rates = {
        #     "v1": np.concatenate(validation_v1_rate_chunks, axis=0),
        #     "lm": np.concatenate(validation_lm_rate_chunks, axis=0),
        # }

        v1_spikes, lm_spikes, v1_spikes_spont, lm_spikes_spont, v1_spikes_ns, lm_spikes_ns = model_spikes
        # get the first replica of the training spikes
        v1_spikes = strategy.experimental_local_results(v1_spikes)[0]
        lm_spikes = strategy.experimental_local_results(lm_spikes)[0]
        v1_spikes_spont = strategy.experimental_local_results(v1_spikes_spont)[0]
        lm_spikes_spont = strategy.experimental_local_results(lm_spikes_spont)[0]
        v1_spikes_ns = strategy.experimental_local_results(v1_spikes_ns)[0]
        lm_spikes_ns = strategy.experimental_local_results(lm_spikes_ns)[0]

        if strategy.num_replicas_in_sync > 1:
            x = strategy.experimental_local_results(x)[0]
            x_spontaneous = strategy.experimental_local_results(x_spontaneous)[0]
            x_natural = strategy.experimental_local_results(x_natural)[0]
            y = strategy.experimental_local_results(y)[0]
            natural_scene_id = strategy.experimental_local_results(natural_scene_id)[0]

        train_values = [a.result().numpy() for a in [train_loss, train_firing_rate, train_rate_loss, train_voltage_loss,
                                                     train_regularizer_loss, train_osi_dsi_loss, train_sync_loss,
                                                     train_natural_scene_composite]]
        val_values = [a.result().numpy() for a in [val_loss, val_firing_rate, val_rate_loss, val_voltage_loss,
                                                   val_regularizer_loss, val_osi_dsi_loss, val_sync_loss,
                                                   val_natural_scene_composite]]
        rate_loss_values = {
            "train_rate_loss": float(train_rate_loss.result().numpy()),
            "train_rate_dg_loss": float(train_rate_dg_loss.result().numpy()),
            "train_rate_sp_loss": float(train_rate_sp_loss.result().numpy()),
            "train_rate_ns_loss": float(train_rate_ns_loss.result().numpy()),
            "train_rate_ns_marginal_loss": float(
                train_rate_ns_marginal_loss.result().numpy()
            ),
            "train_rate_ns_mean_projection_loss": float(
                train_rate_ns_mean_projection_loss.result().numpy()
            ),
            "train_rate_ns_contrast_basis_projection_loss": float(
                train_rate_ns_contrast_basis_projection_loss.result().numpy()
            ),
            "train_rate_ns_rpd_projection_loss": float(
                train_rate_ns_rpd_projection_loss.result().numpy()
            ),
            "train_rate_ns_contrast_rotation_projection_loss": float(
                train_rate_ns_contrast_rotation_projection_loss.result().numpy()
            ),
            "train_rate_ns_joint_loss": float(
                train_rate_ns_joint_loss.result().numpy()
            ),
            "train_natural_scene_composite": float(
                train_natural_scene_composite.result().numpy()
            ),
            "val_rate_loss": float(val_rate_loss.result().numpy()),
            "val_rate_dg_loss": float(val_rate_dg_loss.result().numpy()),
            "val_rate_sp_loss": float(val_rate_sp_loss.result().numpy()),
            "val_rate_ns_loss": float(val_rate_ns_loss.result().numpy()),
            "val_rate_ns_marginal_loss": float(
                val_rate_ns_marginal_loss.result().numpy()
            ),
            "val_rate_ns_mean_projection_loss": float(
                val_rate_ns_mean_projection_loss.result().numpy()
            ),
            "val_rate_ns_contrast_basis_projection_loss": float(
                val_rate_ns_contrast_basis_projection_loss.result().numpy()
            ),
            "val_rate_ns_rpd_projection_loss": float(
                val_rate_ns_rpd_projection_loss.result().numpy()
            ),
            "val_rate_ns_contrast_rotation_projection_loss": float(
                val_rate_ns_contrast_rotation_projection_loss.result().numpy()
            ),
            "val_rate_ns_joint_loss": float(
                val_rate_ns_joint_loss.result().numpy()
            ),
            "val_natural_scene_composite": float(
                val_natural_scene_composite.result().numpy()
            ),
        }
        # for our case, training set and testing set are undistinguishible
        # val_values = train_values

        metric_values = train_values + val_values
        diagnostic_scene_ids = None
        diagnostic_natural_rates = None
        selection_score = metric_values[
            metric_keys.index(callbacks.selection_metric_key)
        ]
        if stimulus_weights.is_enabled("ns") and callbacks.would_improve(
            selection_score
        ):
            diagnostic_scene_ids, diagnostic_natural_rates = (
                capture_full_natural_scene_diagnostics()
            )

        # Weight diagnostics are intentionally evaluated once per epoch. Running
        # these full-vector reductions in split_train_step adds device
        # synchronization and scans every large synapse tensor at every step.
        tf.print(
            '  V1 rec weights: ',
            tf.reduce_min(rsnn_layer.cell.v1.recurrent_weight_values),
            tf.reduce_mean(rsnn_layer.cell.v1.recurrent_weight_values),
            tf.reduce_max(rsnn_layer.cell.v1.recurrent_weight_values),
        )
        tf.print(
            '  LM rec weights: ',
            tf.reduce_min(rsnn_layer.cell.lm.recurrent_weight_values),
            tf.reduce_mean(rsnn_layer.cell.lm.recurrent_weight_values),
            tf.reduce_max(rsnn_layer.cell.lm.recurrent_weight_values),
        )
        tf.print(
            '  LM to V1 weights: ',
            tf.reduce_min(rsnn_layer.cell.v1.interarea_weight_values['lm']),
            tf.reduce_mean(rsnn_layer.cell.v1.interarea_weight_values['lm']),
            tf.reduce_max(rsnn_layer.cell.v1.interarea_weight_values['lm']),
        )
        tf.print(
            '  V1 to LM weights: ',
            tf.reduce_min(rsnn_layer.cell.lm.interarea_weight_values['v1']),
            tf.reduce_mean(rsnn_layer.cell.lm.interarea_weight_values['v1']),
            tf.reduce_max(rsnn_layer.cell.lm.interarea_weight_values['v1']),
        )

        stop = callbacks.on_epoch_end(x, v1_spikes, lm_spikes, y, metric_values, verbose=True,
                                      x_spont=x_spontaneous, v1_spikes_spont=v1_spikes_spont, lm_spikes_spont=lm_spikes_spont,
                                      x_ns=x_natural, v1_spikes_ns=v1_spikes_ns, lm_spikes_ns=lm_spikes_ns,
                                      natural_scene_ids=diagnostic_scene_ids,
                                      natural_scene_rates=diagnostic_natural_rates,
                                      rate_loss_values=rate_loss_values)
        if stop:
            break

        # Reset the metrics for the next epoch
        reset_train_metrics()
        reset_validation_metrics()

    normalizers = {
        'v1_ema': v1_ema.numpy(),
        'lm_ema': lm_ema.numpy(),
        'natural_scene_marginal_reference': natural_scene_marginal_reference.numpy(),
        'natural_scene_joint_reference': natural_scene_joint_reference.numpy(),
        'natural_scene_projection_protocol': natural_scene_projection_protocol,
    }
    callbacks.on_train_end(metric_values, normalizers=normalizers)


if __name__ == '__main__':
    _data_dir = 'GLIF_network'
    _results_dir = 'Simulation_results'

    absl.app.flags.DEFINE_string('task_name', 'functional_natural_training', '')
    absl.app.flags.DEFINE_string('data_dir', _data_dir, '')
    absl.app.flags.DEFINE_string('results_dir', _results_dir, '')
    absl.app.flags.DEFINE_string('ckpt_dir', '', '')
    absl.app.flags.DEFINE_string('restore_from', '', '')
    absl.app.flags.DEFINE_string('comment', '', '')
    absl.app.flags.DEFINE_string('interarea_weight_distribution', 'billeh_uniform_weights', '')
    absl.app.flags.DEFINE_string('delays', '0,0', '') # 50,0
    absl.app.flags.DEFINE_string('noise_scales', '2,2', '')
    absl.app.flags.DEFINE_string("noise_type", "poisson", "")  # poisson or experimental
    absl.app.flags.DEFINE_string("decoded_noise_path", "data/empirical_noise_data/additive_noise.npy", "")
    absl.app.flags.DEFINE_string('optimizer', 'exp_adam', '')
    absl.app.flags.DEFINE_string('v1_neuropixels_df', 'data/Neuropixels_data/OSI_DSI_neuropixels_VISp_v5.csv',
                                 'File name of the Neuropixels DataFrame for OSI/DSI analysis.')
    absl.app.flags.DEFINE_string('lm_neuropixels_df', 'data/Neuropixels_data/OSI_DSI_neuropixels_VISl_v5.csv',
                                 'File name of the Neuropixels DataFrame for OSI/DSI analysis.')
    absl.app.flags.DEFINE_string('dtype', 'float32', '')
    absl.app.flags.DEFINE_string("rotation", "ccw", "")
    absl.app.flags.DEFINE_string("natural_scenes_cache_dir", ".cache/natural_scenes", "")

    absl.app.flags.DEFINE_float('learning_rate', 0.1, '')
    absl.app.flags.DEFINE_string('lr_schedule', 'warmup_cosine',
        "Learning-rate schedule. Options: 'none' or 'warmup_cosine'.",
    )
    absl.app.flags.DEFINE_float('lr_warmup_start_lr', 0.003,
        'Warmup start learning rate (used when lr_schedule=warmup_cosine).',
    )
    absl.app.flags.DEFINE_float('lr_warmup_target_lr', 0.1,
        'Warmup end learning rate (used when lr_schedule=warmup_cosine).',
    )
    absl.app.flags.DEFINE_integer('lr_warmup_steps', 50,
        'Number of linear warmup steps (used when lr_schedule=warmup_cosine).',
    )
    absl.app.flags.DEFINE_float('lr_cosine_min_lr', 0.001,
        'Final cosine learning rate floor (used when lr_schedule=warmup_cosine).',
    )
    absl.app.flags.DEFINE_integer('lr_cosine_steps', 950,
        'Number of cosine decay steps after warmup (used when lr_schedule=warmup_cosine).',
    )
    absl.app.flags.DEFINE_float('rate_cost', 10000., '')
    absl.app.flags.DEFINE_float('voltage_cost', 100., '')
    absl.app.flags.DEFINE_string('voltage_penalty_mode', 'range',
                                 'Type of penalization for voltage. Options: range, threshold')
    absl.app.flags.DEFINE_boolean('return_voltage_sequences', False,
                                  'Return full voltage sequences for training diagnostics.')
    absl.app.flags.DEFINE_float('sync_cost', 1., '')
    absl.app.flags.DEFINE_float('osi_cost', 20., '')
    absl.app.flags.DEFINE_float('annulus_loss_weight', 0.1, '')
    absl.app.flags.DEFINE_float('dg_weight', 1.0, '')
    absl.app.flags.DEFINE_float('sp_weight', 1.0, '')
    absl.app.flags.DEFINE_float('ns_weight', 1.0, '')
    absl.app.flags.DEFINE_string('osi_loss_method', 'crowd_osi', '')
    absl.app.flags.DEFINE_float('osi_loss_subtraction_ratio', 0., '')
    absl.app.flags.DEFINE_float('dampening_factor', 1.0, '')
    absl.app.flags.DEFINE_float('recurrent_dampening_factor', 0.1, '')
    absl.app.flags.DEFINE_float('input_weight_scale', 1., '')
    absl.app.flags.DEFINE_float("interarea_runtime_scale", 1.0, "")
    absl.app.flags.DEFINE_float('recurrent_weight_init_scale', 1., '')
    absl.app.flags.DEFINE_float('recurrent_runtime_scale', 1., '')
    absl.app.flags.DEFINE_float('interarea_weight_init_scale', 1., '')
    absl.app.flags.DEFINE_float('gauss_std', .3, '')
    absl.app.flags.DEFINE_float('recurrent_weight_regularization', 0., '')
    absl.app.flags.DEFINE_float('interarea_weight_regularization', 0., '')
    absl.app.flags.DEFINE_float('lr_scale', 1., '')
    absl.app.flags.DEFINE_float('input_f0', 0.2, '')
    absl.app.flags.DEFINE_float('E4_weight_factor', 4., '')
    absl.app.flags.DEFINE_float('temporal_f', 2., '')
    absl.app.flags.DEFINE_float('max_time', -1, '')

    absl.app.flags.DEFINE_integer('n_runs', 1, '')
    absl.app.flags.DEFINE_integer('run_session', 0, '')
    absl.app.flags.DEFINE_integer('n_epochs', 20, '')
    # absl.app.flags.DEFINE_integer('osi_dsi_eval_period', 50, '') # number of epochs for osi/dsi evaluation if n_runs = 1
    absl.app.flags.DEFINE_integer(
        'batch_size', 12,
        'Total per-replica batch size; must equal the three stimulus counts.')
    absl.app.flags.DEFINE_integer(
        'gray_batch_size', 1, 'Gray samples per replica.')
    absl.app.flags.DEFINE_integer(
        'grating_batch_size', 3, 'Grating samples per replica.')
    absl.app.flags.DEFINE_integer(
        'natural_batch_size', 8, 'Natural-scene samples per replica.')
    absl.app.flags.DEFINE_boolean(
        'natural_joint_rate_loss', True,
        'Use joint projections in the natural-scene firing-rate loss.')
    absl.app.flags.DEFINE_enum(
        'natural_joint_direction_mode',
        'rpd',
        ['rpd', 'contrast_rotations'],
        'Extra joint projection bank for natural-scene firing rates.',
    )
    absl.app.flags.DEFINE_float(
        'natural_rpd_concentration',
        32.0,
        'Positive Power-Spherical concentration for natural-scene RPDs.',
    )
    absl.app.flags.DEFINE_integer('v1_neurons', 10, '')  # -1 to take all neurons
    absl.app.flags.DEFINE_integer('lm_neurons', 10, '')  # -1 to take all neurons
    absl.app.flags.DEFINE_integer('v1_radius', 845, '')
    absl.app.flags.DEFINE_integer('lm_radius', 400, '')
    # EA and garret dose not need this many but pure classification needs 781 = int(50000/64)
    absl.app.flags.DEFINE_integer('steps_per_epoch', 10, '')
    absl.app.flags.DEFINE_integer(
        'profile_train_step',
        -1,
        'Zero-based training step to capture with TensorFlow Profiler; -1 disables profiling.',
    )
    absl.app.flags.DEFINE_string(
        'profile_logdir',
        '',
        'TensorFlow Profiler output directory; defaults below the run log directory.',
    )
    absl.app.flags.DEFINE_enum(
        'profile_backend',
        'tensorflow',
        ['tensorflow', 'cuda', 'none'],
        'Profiler backend: TensorFlow XSpace, CUDA capture range, or timing only.',
    )
    absl.app.flags.DEFINE_boolean(
        'exit_after_profile',
        False,
        'Exit before validation and checkpoint callbacks after the profiled step.',
    )
    absl.app.flags.DEFINE_enum(
        'single_gpu_strategy',
        'mirrored',
        ['mirrored', 'one_device'],
        'Distribution strategy used when exactly one GPU is visible.',
    )
    # EA and garret dose not need this many but pure classification needs 156 = int(10000/64)
    absl.app.flags.DEFINE_integer(
        'val_steps',
        18,
        'Deprecated for functional training; validation steps are derived from '
        'the held-out natural-scene count and effective natural batch size.',
    )
    # number of LGN filters in visual space (input population)
    absl.app.flags.DEFINE_integer('n_input', 17400, '')
    absl.app.flags.DEFINE_integer('seq_len', 500, '')
    absl.app.flags.DEFINE_integer(
        'gradient_checkpoint_chunk_size',
        50,
        'Timesteps per exact-BPTT recomputation chunk.',
    )
    absl.app.flags.DEFINE_integer('natural_scenes_experiment_id', 501498760, '')
    absl.app.flags.DEFINE_integer('natural_scenes_validation_count', 18,
                                  'Number of held-out natural scenes reserved for validation/testing.')
    absl.app.flags.DEFINE_integer('natural_scenes_split_seed', 0,
                                  'Seed used to create the fixed natural-scene train/validation split.')
    absl.app.flags.DEFINE_integer('n_cues', 3, '')
    absl.app.flags.DEFINE_integer('recall_duration', 40, '')
    absl.app.flags.DEFINE_integer('cue_duration', 40, '')
    absl.app.flags.DEFINE_integer('interval_duration', 40, '')
    absl.app.flags.DEFINE_integer('examples_in_epoch', 32, '')
    absl.app.flags.DEFINE_integer('validation_examples', 16, '')
    absl.app.flags.DEFINE_integer('seed', 3000,
                                  'Seed for stimulus generation and training randomness')
    absl.app.flags.DEFINE_integer('model_seed', 3000,
                                  'Seed for model structure creation (neurons, connections, etc.)')
    absl.app.flags.DEFINE_integer('neurons_per_output', 16, '')
    absl.app.flags.DEFINE_integer('n_trials_per_angle', 10, '')
    absl.app.flags.DEFINE_integer('n_output', 10, '')
    absl.app.flags.DEFINE_integer('fano_samples', 500, '')
    # absl.app.flags.DEFINE_integer('fano_duration', 300, '')

    # absl.app.flags.DEFINE_boolean('float16', False, '')
    absl.app.flags.DEFINE_boolean('caching', True, '')
    absl.app.flags.DEFINE_boolean('core_only', False, '')
    absl.app.flags.DEFINE_boolean('core_loss', False, '')
    absl.app.flags.DEFINE_boolean('hard_reset', False, '')
    absl.app.flags.DEFINE_boolean('disconnect_lm_L6_inhibition', False, '')
    absl.app.flags.DEFINE_boolean('disconnect_v1_lm_L6_excitatory_projections', False, '')
    absl.app.flags.DEFINE_boolean('randomize_recurrent_weights', False, '')
    absl.app.flags.DEFINE_boolean('realistic_neurons_ratio', False, '')
    absl.app.flags.DEFINE_boolean('realistic_radius', True, '')
    absl.app.flags.DEFINE_boolean('train_recurrent_v1', False, '')
    absl.app.flags.DEFINE_boolean('train_recurrent_lm', False, '')
    absl.app.flags.DEFINE_boolean('train_input', False, '')
    absl.app.flags.DEFINE_boolean('train_interarea_lm_v1', False, '')
    absl.app.flags.DEFINE_boolean('train_interarea_v1_lm', False, '')
    absl.app.flags.DEFINE_boolean('train_noise', False, '')
    # absl.app.flags.DEFINE_boolean('train_recurrent_per_type', False, '')
    absl.app.flags.DEFINE_boolean('connected_selection', True, '')
    absl.app.flags.DEFINE_boolean('neuron_output', False, '')

    # absl.app.flags.DEFINE_boolean('hard_only', False, '')
    absl.app.flags.DEFINE_boolean('visualize_test', False, '')
    absl.app.flags.DEFINE_boolean('pseudo_gauss', False, '')
    absl.app.flags.DEFINE_boolean("bmtk_compat_lgn", True, "")
    absl.app.flags.DEFINE_boolean("reset_every_step", False, "")
    absl.app.flags.DEFINE_boolean("spontaneous_training", False, "")
    absl.app.flags.DEFINE_boolean("spontaneous_uniform_distribution_constraint", False, "")
    absl.app.flags.DEFINE_boolean("current_input", False, "")
    absl.app.flags.DEFINE_boolean("gradient_checkpointing", False, "")
    absl.app.flags.DEFINE_boolean("connected_areas", True, "")
    absl.app.flags.DEFINE_boolean("connected_recurrent_connections", True, "")
    absl.app.flags.DEFINE_boolean("connected_noise", True, "")

    absl.app.run(main)
