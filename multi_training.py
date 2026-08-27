import matplotlib
matplotlib.use('agg')# to avoid GUI request on clusters
import os

# Define the environment variables for optimal GPU performance
os.environ['TF_GPU_THREAD_MODE'] = 'global'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'  # before import tensorflow
# Prefer the default BFC allocator: cuda_malloc_async retained GPU pool memory
# and reached OOM earlier in this repository's workloads despite lower live usage.
# os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
# os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

import absl
import socket
# import re
import copy
import numpy as np
import pandas as pd
import tensorflow as tf
import pickle as pkl
from time import strftime, time
import logging
from v1_model_utils import tf_utils
tf.get_logger().setLevel(logging.INFO)
# logging.getLogger().setLevel(logging.INFO)


def main(_):
    flags = absl.app.flags.FLAGS
    if flags.profile_gpu_index >= 0:
        all_gpus = tf.config.list_physical_devices("GPU")
        if not all_gpus:
            raise RuntimeError("No physical GPUs are visible to TensorFlow.")
        if flags.profile_gpu_index >= len(all_gpus):
            raise ValueError(
                f"profile_gpu_index={flags.profile_gpu_index} is out of range "
                f"for {len(all_gpus)} visible GPU(s)."
            )
        tf.config.set_visible_devices(all_gpus[flags.profile_gpu_index], "GPU")
        print(f"Profiling on physical GPU index {flags.profile_gpu_index}: {all_gpus[flags.profile_gpu_index]}")
    # Allow for memory growth (also to observe memory consumption)
    physical_devices = tf_utils.configure_gpu_memory_growth()
    # Display TensorFlow and CUDA runtime information for debugging and verification purposes.
    tf_utils.print_tensorflow_runtime_info()

    # Import TensorFlow-heavy project modules only after GPU memory is configured.
    import stim_dataset
    from v1_model_utils.callbacks import Callbacks
    import v1_model_utils.loss_functions as losses
    from v1_model_utils.model_metrics_analysis import calculate_OSI_DSI
    from v1_model_utils import load_sparse, models, other_v1_utils, training_utils
    from v1_model_utils import optimizers as optimizer_utils

    # Configure seed for reproducibility
    tf_utils.configure_reproducibility(flags.seed)
    # Configure TensorFlow optimization options that have proven to be beneficial for our model and training setup. These options can improve the performance of the model by optimizing the computational graph and memory usage.
    tf_utils.enable_tensorflow_optimizations(enabled=True)

    # Create the tensorflow datafile for the network
    os.makedirs(os.path.join(flags.data_dir, 'tf_data'), exist_ok=True)

    flag_str, logdir, _ = tf_utils.configure_run_paths(
        flags,
        task_name=flags.task_name,
    )

    # Configure the dtype policy
    mixed_precision, dtype = tf_utils.configure_policy_and_dtype(flags.dtype)

    # Use HierarchicalCopyAllReduce to avoid NCCL issues with Blackwell GPUs
    strategy = tf_utils.create_distribution_strategy(
        physical_devices=physical_devices,
        use_hierarchical_all_reduce=True,
        single_gpu_strategy=flags.single_gpu_strategy,
    )

    per_replica_batch_size = flags.batch_size
    batch_multiplier = 1 if flags.sequential_stimuli else 2
    real_batch_size = per_replica_batch_size * batch_multiplier
    global_batch_size = per_replica_batch_size * strategy.num_replicas_in_sync
    print(f'Per replica batch size: {per_replica_batch_size}')
    if flags.sequential_stimuli:
        print('Sequential stimuli updates enabled (memory friendly).')
        print(f'Model batch size: {real_batch_size}')
        if per_replica_batch_size != 1:
            print(f'Warning: sequential_stimuli is intended for batch_size=1; got {per_replica_batch_size}.')
    else:
        print(f'Real batch size (evoked+spont): {real_batch_size}')
    print(f'Global batch size: {global_batch_size}\n')
    print(f'Training with current input: {flags.current_input}')
    print(f'Pseudo derivative gaussian: {flags.pseudo_gauss}')

    # Load or create the network building files configuration
    t0 = time()
    if flags.caching:
        load_fn = load_sparse.cached_load_v1
    else:
        load_fn = load_sparse.load_v1
    network, lgn_input, bkg_input = load_fn(flags, flags.neurons, flag_str=flag_str)
    print(f"Model files loading: {time()-t0:.2f} seconds\n")

    pre_delay, post_delay = training_utils.parse_delays(flags.delays)
    delays = [pre_delay, post_delay]

    # Define the scope in which the model training will be executed
    with strategy.scope():
        t0 = time()
        # # Enable TensorFlow Profiler
        model = models.create_model(
            network,
            lgn_input,
            bkg_input,
            seq_len=flags.seq_len,
            n_input=flags.n_input,
            n_output=flags.n_output,
            cue_duration=flags.cue_duration,
            dtype=dtype,
            batch_size=real_batch_size,
            input_weight_scale=flags.input_weight_scale,
            dampening_factor=flags.dampening_factor,
            recurrent_dampening_factor=flags.recurrent_dampening_factor,
            voltage_gradient_dampening=flags.voltage_gradient_dampening,
            gauss_std=flags.gauss_std,
            lr_scale=flags.lr_scale,
            train_input=flags.train_input,
            train_noise=flags.train_noise,
            train_recurrent=flags.train_recurrent,
            train_recurrent_per_type=flags.train_recurrent_per_type,
            neuron_output=flags.neuron_output,
            pseudo_gauss=flags.pseudo_gauss,
            use_state_input=True,
            return_state=True,
            hard_reset=flags.hard_reset,
            add_metric=False,
            max_delay=flags.max_delay,  # 0 = auto-compute from SONATA data
            current_input=flags.current_input,
            seed=flags.seed,
            use_dummy_state_input=False,
            synaptic_current_backend=flags.synaptic_current_backend,
        )

        # Initialize the weights of the model based on the specified input shape. It operates in eager mode.
        # It does not construct a computational graph of the model operations, but prepares the model layers and weights
        model.build((real_batch_size, flags.seq_len, flags.n_input))
        print(f"Model built in {time()-t0:.2f} s\n")

        # Store the initial model variables that are going to be trained
        model_variables_dict = {'Initial': {var.name: var.numpy().astype(np.float16) for var in model.trainable_variables}}

        # Define the learning rate
        learning_rate = optimizer_utils.build_learning_rate(flags)

        # Define the optimizer
        optimizer = optimizer_utils.create_optimizer(
            flags,
            learning_rate,
            model.trainable_variables,
            mixed_precision_module=mixed_precision,
        )

        # Restore model and optimizer from a checkpoint if it exists.
        checkpoint, optimizer, _checkpoint_directory = tf_utils.restore_training_checkpoint(
            flags,
            model,
            optimizer,
            learning_rate,
            mixed_precision_module=mixed_precision,
        )

        model_variables_dict['Best'] = {var.name: var.numpy().astype(np.float16) for var in model.trainable_variables}
        print("Model variables stored in dictionary\n")

        ### BUILD THE LOSS AND REGULARIZER FUNCTIONS ###
        # Create rate and voltage regularizers
        core_mask = None
        annulus_mask = None
        if flags.loss_core_radius > 0:
            core_mask_np = other_v1_utils.isolate_core_neurons(network, radius=flags.loss_core_radius, data_dir=flags.data_dir)
            # if core_mask is all True, set it to None.
            if core_mask_np.all():
                print("All neurons are in the core region. Core mask is set to None.")
            else:
                # report how many neurons are selected.
                print(f"Core mask is set to {core_mask_np.sum()} neurons.")
                core_mask = tf.constant(core_mask_np, dtype=tf.bool)
                annulus_mask = tf.constant(~core_mask_np, dtype=tf.bool)

        # Extract outputs of intermediate keras layers to get access to spikes and membrane voltages of the model
        rsnn_layer = model.get_layer("rsnn")
        # prediction_layer = model.get_layer('prediction')

        ### RECURRENT REGULARIZERS ###
        if flags.recurrent_weight_regularization > 0 and flags.uniform_weights:
            print("Uniform weights are set to True. Loading the network with original weights for regularizer.")
            dummy_flags = copy.deepcopy(flags)
            dummy_flags.uniform_weights = False  # read network with original weights
            rec_reg_network, _, _ = load_fn(dummy_flags, dummy_flags.neurons, flag_str='')
        else:
            rec_reg_network = network

        # Keep weight regularizers in the master recurrent-weight dtype
        # (typically float32 under mixed precision).
        weight_regularizer_dtype = rsnn_layer.cell.recurrent_weight_values.dtype
        rec_weight_regularizer = None
        if flags.train_recurrent and flags.recurrent_weight_regularization > 0:
            if flags.recurrent_weight_regularizer_type == 'mean':
                print("Using mean regularizer")
                rec_weight_regularizer = losses.MeanStiffRegularizer(
                    flags.recurrent_weight_regularization,
                    rec_reg_network,
                    penalize_relative_change=True,
                    dtype=weight_regularizer_dtype,
                )
            elif flags.recurrent_weight_regularizer_type == 'emd':
                print("Using emd regularizer")
                rec_weight_regularizer = losses.EarthMoversDistanceRegularizer(
                    flags.recurrent_weight_regularization,
                    rec_reg_network,
                    dtype=weight_regularizer_dtype,
                )
            else:
                raise ValueError(
                    f"Invalid recurrent weight regularizer type: {flags.recurrent_weight_regularizer_type}")

        ### EVOKED RATES REGULARIZERS ###
        rate_core_mask = None if flags.all_neuron_rate_loss else core_mask
        evoked_rate_regularizer = losses.SpikeRateDistributionTarget(network, stimulus_type='drifting_gratings', rate_cost=flags.rate_cost, pre_delay=delays[0], post_delay=delays[1],
                                                                     data_dir=flags.data_dir, core_mask=rate_core_mask, seed=flags.seed, dtype=tf.float32, neuropixels_df=flags.neuropixels_df)
        # model.add_loss(lambda: evoked_rate_regularizer(rsnn_layer.output[0][0]))

        ### SPONTANEOUS RATES REGULARIZERS ###
        spont_rate_regularizer = losses.SpikeRateDistributionTarget(network, stimulus_type='spontaneous', rate_cost=flags.rate_cost, pre_delay=delays[0], post_delay=delays[1],
                                                                    data_dir=flags.data_dir, core_mask=rate_core_mask, seed=flags.seed, dtype=tf.float32, neuropixels_df=flags.neuropixels_df)
        # model.add_loss(lambda: spont_rate_regularizer(rsnn_layer.output[0][0]))

        ### VOLTAGE REGULARIZERS ###
        # Keep heavy voltage regularizer math in fp16 when mixed precision is enabled.
        voltage_regularizer_dtype = tf.float16 if flags.dtype == 'float16' else tf.float32
        voltage_regularizer = losses.VoltageRegularization(
            rsnn_layer.cell,
            voltage_cost=flags.voltage_cost,
            dtype=voltage_regularizer_dtype,
            penalty_mode=flags.voltage_penalty_mode,
        )
        # model.add_loss(lambda: voltage_regularizer(rsnn_layer.output[0][1]))

        ### SYNCHRONIZATION REGULARIZERS ###
        if flags.sync_cost == 0.0:
            def zero_sync_loss(_spikes, trim=True):
                return tf.constant(0.0, dtype=tf.float32)

            evoked_sync_loss = zero_sync_loss
            spont_sync_loss = zero_sync_loss
        else:
            evoked_sync_loss = losses.SynchronizationLoss(network, sync_cost=flags.sync_cost, core_mask=core_mask, t_start=0.2, t_end=flags.seq_len/1000, n_samples=flags.fano_samples,
                                                          dtype=tf.float32, stimulus_type='drifting_gratings', neuropixels_data_dir='Synchronization_data',
                                                          data_dir=flags.data_dir)
            # model.add_loss(lambda: evoked_sync_loss(rsnn_layer.output[0][0]))

            spont_sync_loss = losses.SynchronizationLoss(network, sync_cost=flags.sync_cost, core_mask=core_mask, t_start=0.2, t_end=flags.seq_len/1000, n_samples=flags.fano_samples,
                                                         dtype=tf.float32, stimulus_type='spontaneous', neuropixels_data_dir='Synchronization_data',
                                                         data_dir=flags.data_dir)
        # model.add_loss(lambda: spont_sync_loss(rsnn_layer.output[0][0]))

        ### OSI / DSI LOSSES ###
        # Define the decay factor for the exponential moving average
        ema_decay = 0.95
        # Initialize exponential moving averages for V1 and LM firing rates
        train_end_data = {}
        if os.path.exists(os.path.join(logdir, 'train_end_data.pkl')):
            with open(os.path.join(logdir, 'train_end_data.pkl'), 'rb') as f:
                train_end_data = pkl.load(f)
        # 3 Hz is near the average FR of cortex.
        default_v1_ema = tf.constant(
            0.003, shape=(network["n_nodes"],), dtype=tf.float32
        )
        restored_v1_ema = train_end_data.get("v1_ema")
        if restored_v1_ema is None:
            v1_ema_value = default_v1_ema
        else:
            restored_shape = tuple(np.asarray(restored_v1_ema).shape)
            expected_shape = (network["n_nodes"],)
            if restored_shape != expected_shape:
                print(
                    "Warning: ignoring restored v1_ema with shape "
                    f"{restored_shape}; expected {expected_shape}."
                )
                v1_ema_value = default_v1_ema
            else:
                v1_ema_value = restored_v1_ema
        v1_ema = tf.Variable(v1_ema_value, trainable=False, name='V1_EMA')

        OSI_DSI_Loss = losses.OrientationSelectivityLoss(network=network, osi_cost=flags.osi_cost,
                                                         pre_delay=delays[0], post_delay=delays[1],
                                                         dtype=tf.float32, core_mask=core_mask,
                                                         method=flags.osi_loss_method,
                                                         subtraction_ratio=flags.osi_loss_subtraction_ratio,
                                                         neuropixels_df=flags.neuropixels_df,
                                                         data_dir=flags.data_dir,
                                                         rolling_decay=flags.rolling_decay,
                                                         rolling_target_sample_ess=flags.rolling_target_sample_ess,
                                                         rolling_batch_size=per_replica_batch_size,
                                                         rolling_gradient_correction=flags.rolling_gradient_correction,
                                                         rolling_max_gradient_scale=flags.rolling_max_gradient_scale,
                                                         rolling_warmup=flags.rolling_warmup)
        # placeholder_angle = tf.constant(0, dtype=tf.float32, shape=(per_replica_batch_size, 1))
        # model.add_loss(lambda: OSI_DSI_Loss(rsnn_layer.output[0][0], placeholder_angle, trim=True, normalizer=v1_ema))

        ### ANNULUS REGULARIZERS ###
        if annulus_mask is not None:
            annulus_spont_rate_regularizer = losses.SpikeRateDistributionTarget(network, stimulus_type='spontaneous', rate_cost=flags.annulus_loss_weight*flags.rate_cost, pre_delay=delays[0], post_delay=delays[1],
                                                                                data_dir=flags.data_dir, core_mask=annulus_mask, seed=flags.seed, dtype=tf.float32, neuropixels_df=flags.neuropixels_df)
            # model.add_loss(lambda: annulus_spont_rate_regularizer(rsnn_layer.output[0][0]))
            annulus_evoked_rate_regularizer = losses.SpikeRateDistributionTarget(network, stimulus_type='drifting_gratings', rate_cost=flags.annulus_loss_weight*flags.rate_cost, pre_delay=delays[0], post_delay=delays[1],
                                                                                 data_dir=flags.data_dir, core_mask=annulus_mask, seed=flags.seed, dtype=tf.float32, neuropixels_df=flags.neuropixels_df)
            # model.add_loss(lambda: annulus_evoked_rate_regularizer(rsnn_layer.output[0][0]))

            # Add OSI/DSI regularizer for the annulus
            annulus_OSI_DSI_Loss = losses.OrientationSelectivityLoss(network=network, osi_cost=flags.annulus_loss_weight*flags.osi_cost,
                                                                     pre_delay=delays[0], post_delay=delays[1],
                                                                     dtype=tf.float32, core_mask=annulus_mask,
                                                                     method=flags.osi_loss_method,
                                                                     subtraction_ratio=flags.osi_loss_subtraction_ratio,
                                                                     neuropixels_df=flags.neuropixels_df,
                                                                     data_dir=flags.data_dir,
                                                                     rolling_decay=flags.rolling_decay,
                                                                     rolling_target_sample_ess=flags.rolling_target_sample_ess,
                                                                     rolling_batch_size=per_replica_batch_size,
                                                                     rolling_gradient_correction=flags.rolling_gradient_correction,
                                                                     rolling_max_gradient_scale=flags.rolling_max_gradient_scale,
                                                                     rolling_warmup=flags.rolling_warmup)
            # placeholder_angle = tf.constant(0, dtype=tf.float32, shape=(per_replica_batch_size, 1))
            # model.add_loss(lambda: annulus_OSI_DSI_Loss(rsnn_layer.output[0][0], placeholder_angle, trim=True, normalizer=v1_ema))

        def _restore_rolling_state(loss_obj, state, label):
            if flags.osi_loss_method not in (
                "rolling_osi_emd",
                "adaptative_crowd_osi",
            ) or not state:
                return
            try:
                if loss_obj.set_rolling_state(state):
                    print(f"Restored {flags.osi_loss_method} state for {label}.")
            except ValueError as exc:
                print(f"Warning: could not restore {flags.osi_loss_method} state for {label}: {exc}")

        rolling_state_key = None
        if flags.osi_loss_method == "rolling_osi_emd":
            rolling_state_key = "rolling_osi_emd_state"
        elif flags.osi_loss_method == "adaptative_crowd_osi":
            rolling_state_key = "adaptative_crowd_osi_state"

        rolling_state_payload = (
            train_end_data.get(rolling_state_key, {}) if rolling_state_key else {}
        )
        if isinstance(rolling_state_payload, dict):
            core_state = rolling_state_payload.get("core")
            if core_state is None and "ori_real" in rolling_state_payload:
                core_state = rolling_state_payload
            _restore_rolling_state(OSI_DSI_Loss, core_state, "core")
            if annulus_mask is not None:
                _restore_rolling_state(
                    annulus_OSI_DSI_Loss,
                    rolling_state_payload.get("annulus"),
                    "annulus",
                )

        extractor_model = models.build_sequence_only_model(model, rsnn_layer)
        state_fallback_model = None
        # State-only model to avoid storing full sequences when only the final state is needed.
        try:
            state_model = models.build_state_only_model(model, rsnn_layer)
        except Exception as e:
            state_model = None
            state_fallback_model = tf.keras.Model(
                inputs=model.inputs,
                outputs=rsnn_layer.output[1:],
                name="rsnn_state_fallback",
            )
            print(
                f"Warning: failed to build state-only model ({e}); "
                "using the original final-state graph for gray state."
            )

        # Loss from Guozhang classification task (unused in our case)
        # loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
        #     from_logits=False, reduction=tf.keras.losses.Reduction.NONE)
        # def compute_loss(_l, _p, _w):
        #     per_example_loss = loss_object(_l, _p, sample_weight=_w) * strategy.num_replicas_in_sync / tf.reduce_sum(_w)
        #     rec_weight_loss = rec_weight_regularizer(rsnn_layer.cell.recurrent_weight_values)
        #     return tf.nn.compute_average_loss(per_example_loss, global_batch_size=global_batch_size) + rec_weight_loss

        # Initial state used for gray-screen warmup and training.
        # zero_state = rsnn_layer.cell.zero_state(real_batch_size, dtype=dtype)
        # state_variables = tf.nest.map_structure(lambda a: tf.Variable(
        #     a, trainable=False, synchronization=tf.VariableSynchronization.ON_READ
        # ), zero_state)

        # Add other metrics and losses
        train_loss = tf.keras.metrics.Mean()
        train_firing_rate = tf.keras.metrics.Mean()
        train_rate_loss = tf.keras.metrics.Mean()
        train_voltage_loss = tf.keras.metrics.Mean()
        train_regularizer_loss = tf.keras.metrics.Mean()
        train_osi_dsi_loss = tf.keras.metrics.Mean()
        train_sync_loss = tf.keras.metrics.Mean()

        val_loss = tf.keras.metrics.Mean()
        val_firing_rate = tf.keras.metrics.Mean()
        val_rate_loss = tf.keras.metrics.Mean()
        val_voltage_loss = tf.keras.metrics.Mean()
        val_regularizer_loss = tf.keras.metrics.Mean()
        val_osi_dsi_loss = tf.keras.metrics.Mean()
        val_sync_loss = tf.keras.metrics.Mean()

        validation_metrics = (
            val_loss,
            val_firing_rate,
            val_rate_loss,
            val_voltage_loss,
            val_regularizer_loss,
            val_osi_dsi_loss,
            val_sync_loss,
        )

        def reset_train_metrics():
            training_utils.reset_metrics(
                (
                    train_loss,
                    train_firing_rate,
                    train_rate_loss,
                    train_voltage_loss,
                    train_regularizer_loss,
                    train_osi_dsi_loss,
                    train_sync_loss,
                )
            )

        def reset_validation_metrics():
            training_utils.reset_metrics(validation_metrics)

        # Load the spontaneous probabilities once (seq_len, n_input)
        spontaneous_prob_base = stim_dataset.load_or_compute_spontaneous_lgn_probabilities(
            seq_len=flags.seq_len,
            n_input=flags.n_input,
            data_dir=flags.data_dir,
            bmtk_compat=flags.bmtk_compat_lgn,
            seed=flags.seed,
            output_dtype=dtype,
        )
        # Handle the random seed for spontaneous spike generation and BKG noise
        seed_helper = tf_utils.DistributedSeedHelper(
            flags.seed,
            rsnn_layer.cell.noise_stream,
            rsnn_layer.cell.noise_seed,
        )

    if flags.gradient_checkpointing:
        @tf.recompute_grad
        def extractor_forward(x, state_vars):
            # Call extractor model without storing intermediate state variables.
            return extractor_model((x, state_vars))
    else:
        def extractor_forward(x, state_vars):
            return extractor_model((x, state_vars))

    def run_extractor(_x, _state_variables):
        if _x.dtype == tf.bool:
            _x = tf.cast(_x, dtype)
        seed_helper.advance_noise_seed()
        return extractor_forward(_x, _state_variables)

    def _gradient_value_tensor(gradient):
        if isinstance(gradient, tf.IndexedSlices):
            return gradient.values
        return gradient

    def _identity_gradient_with_control(gradient):
        if gradient is None:
            return None
        if isinstance(gradient, tf.IndexedSlices):
            return tf.IndexedSlices(
                tf.identity(gradient.values),
                gradient.indices,
                gradient.dense_shape,
            )
        return tf.identity(gradient)

    def _print_and_check_gradients(gradients, label, spontaneous=None):
        valid_gradients = [g for g in gradients if g is not None]
        if not valid_gradients:
            tf.print(label, "gradients: no non-None gradients")
            return gradients

        grad_values = [_gradient_value_tensor(g) for g in valid_gradients]
        finite_flags = [tf.reduce_all(tf.math.is_finite(g)) for g in grad_values]
        all_finite = tf.reduce_all(tf.stack(finite_flags, axis=0))
        nonfinite_count = tf.add_n([
            tf.reduce_sum(tf.cast(tf.logical_not(tf.math.is_finite(g)), tf.int32))
            for g in grad_values
        ])
        grad_abs = [
            tf.reshape(tf.cast(tf.abs(g), tf.float32), [-1])
            for g in grad_values
        ]
        grad_max = tf.reduce_max(
            tf.stack([tf.reduce_max(g_abs) for g_abs in grad_abs], axis=0)
        )
        grad_sum = tf.add_n([tf.reduce_sum(g_abs) for g_abs in grad_abs])
        grad_count = tf.add_n([
            tf.cast(tf.size(g_abs), tf.float32)
            for g_abs in grad_abs
        ])
        print_args = [
            label,
            "grad_all_finite=", all_finite,
            "nonfinite_count=", nonfinite_count,
            "max_abs=", grad_max,
            "mean_abs=", grad_sum / tf.maximum(grad_count, 1.0),
            "n_tensors=", len(valid_gradients),
        ]
        if spontaneous is not None:
            print_args.insert(1, "spontaneous=")
            print_args.insert(2, spontaneous)
        tf.print(*print_args)

        assert_op = tf.debugging.assert_equal(
            all_finite,
            tf.constant(True),
            message=f"{label} gradients contain NaN/Inf values.",
        )
        with tf.control_dependencies([assert_op]):
            return [_identity_gradient_with_control(g) for g in gradients]

    def _compute_losses_from_activity(
        _z, _v, y, spontaneous, trim, regularizers_loss, update_state=True
    ):

        # keep final scalar aggregation in float32
        voltage_loss = tf.cast(voltage_regularizer(_v), tf.float32)
        spontaneous = tf.cast(spontaneous, tf.bool)

        def _evoked_losses():
            rate_loss = evoked_rate_regularizer(_z, trim)
            if update_state:
                # Update only during training; validation reads the existing normalizer.
                v1_evoked_rates = tf.reduce_mean(
                    tf.cast(
                        _z[:, delays[0]: flags.seq_len - delays[1], :],
                        tf.float32,
                    ),
                    (0, 1),
                )
                v1_ema.assign(
                    ema_decay * v1_ema + (1 - ema_decay) * v1_evoked_rates
                )
            osi_dsi_loss = OSI_DSI_Loss(
                _z,
                y,
                trim,
                normalizer=v1_ema,
                update_state=update_state,
            )
            sync_loss = evoked_sync_loss(_z, trim)

            if annulus_mask is not None:
                rate_loss += annulus_evoked_rate_regularizer(_z, trim)
                osi_dsi_loss += annulus_OSI_DSI_Loss(
                    _z,
                    y,
                    trim,
                    normalizer=v1_ema,
                    update_state=update_state,
                )

            return rate_loss, osi_dsi_loss, sync_loss

        def _spontaneous_losses():
            rate_loss = spont_rate_regularizer(_z, trim)
            osi_dsi_loss = tf.constant(0.0, dtype=tf.float32)
            sync_loss = spont_sync_loss(_z, trim)

            if annulus_mask is not None:
                rate_loss += annulus_spont_rate_regularizer(_z, trim)

            return rate_loss, osi_dsi_loss, sync_loss

        rate_loss, osi_dsi_loss, sync_loss = tf.cond(
            spontaneous, _spontaneous_losses, _evoked_losses
        )

        _aux = dict(
            rate_loss=rate_loss,
            voltage_loss=voltage_loss,
            osi_dsi_loss=osi_dsi_loss,
            regularizer_loss=regularizers_loss,
            sync_loss=sync_loss,
        )
        # Rescale the losses based on the number of replicas
        _loss = tf.nn.scale_regularization_loss(
            rate_loss + voltage_loss + regularizers_loss + osi_dsi_loss + sync_loss
        )

        return _loss, _aux

    def roll_out(x, y, initial_state, spontaneous=False, trim=True, update_loss_state=True):

        # _initial_state = tf.nest.map_structure(lambda _a: _a.read_value(), initial_state)
        _out = run_extractor(x, initial_state)
        _z, _v = _out

        # # update state_variables with the new model state
        # new_state = tuple(_out[1:])
        # tf.nest.map_structure(lambda a, b: a.assign(b), initial_state, new_state)
        regularizers_loss = tf.constant(0.0, dtype=tf.float32)
        if flags.train_recurrent and flags.recurrent_weight_regularization > 0:
            regularizers_loss += tf.cast(
                rec_weight_regularizer(
                    rsnn_layer.cell.recurrent_weight_values), tf.float32
            )

        _loss, _aux = _compute_losses_from_activity(
            _z, _v, y, spontaneous, trim, regularizers_loss,
            update_state=update_loss_state,
        )

        return _out, _loss, _aux

    def roll_out_combined(x, y, x_spontaneous, initial_state, trim=True, update_loss_state=True):

        x_concat = tf.concat([x, x_spontaneous], axis=0)

        _out = run_extractor(x_concat, initial_state)
        _z_full, _v_full = _out

        _z_evoked = _z_full[:per_replica_batch_size]
        _v_evoked = _v_full[:per_replica_batch_size]
        _z_spont = _z_full[per_replica_batch_size:]
        _v_spont = _v_full[per_replica_batch_size:]

        regularizers_loss = tf.constant(0.0, dtype=tf.float32)
        if flags.train_recurrent and flags.recurrent_weight_regularization > 0:
            regularizers_loss += tf.cast(
                rec_weight_regularizer(
                    rsnn_layer.cell.recurrent_weight_values), tf.float32
            )

        evoked_loss, evoked_aux = _compute_losses_from_activity(
            _z_evoked, _v_evoked, y, False, trim, regularizers_loss,
            update_state=update_loss_state,
        )
        spont_loss, spont_aux = _compute_losses_from_activity(
            _z_spont, _v_spont, y, True, trim, regularizers_loss,
            update_state=update_loss_state,
        )

        return _out, evoked_loss, spont_loss, evoked_aux, spont_aux

    def train_step_combined(x, y, x_spontaneous, state_variables, trim, return_sequences=False):
        # Forward propagation of the model (single call for evoked + spontaneous)
        with tf.GradientTape() as tape:
            _out, evoked_loss, spont_loss, evoked_aux, spont_aux = roll_out_combined(
                x, y, x_spontaneous, state_variables, trim=trim
            )
            total_loss = tf.cast(evoked_loss + spont_loss, tf.float32)
            loss_for_grad = optimizer_utils.scale_loss_for_optimizer(optimizer, total_loss)

        # Backpropagation of the model (gradients computation and application)
        grad = tape.gradient(loss_for_grad, model.trainable_variables)
        grad = optimizer_utils.unscale_gradients_for_optimizer(optimizer, grad)

        # # The optimizer will aggregate the gradients across replicas automatically before applying them by default,
        # # so the losses have to be properly scaled to account for the number of replicas
        # # https://www.tensorflow.org/tutorials/distribute/custom_training
        # # https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/keras/optimizer_v2/optimizer_v2.py#L741
        # optimizer.apply_gradients(zip(combined_grads, model.trainable_variables))
        if flags.debug_gradients:
            grad = _print_and_check_gradients(grad, "[Combined]")

        optimizer.apply_gradients(zip(grad, model.trainable_variables))

        mean_aux = {
            "rate_loss": (evoked_aux["rate_loss"] + spont_aux["rate_loss"]) / 2.0,
            "voltage_loss": (evoked_aux["voltage_loss"] + spont_aux["voltage_loss"]) / 2.0,
            "osi_dsi_loss": evoked_aux["osi_dsi_loss"],
            "regularizer_loss": evoked_aux["regularizer_loss"],
            "sync_loss": (evoked_aux["sync_loss"] + spont_aux["sync_loss"]) / 2.0,
        }

        # Backpropagation of the model (metrics)
        mean_loss = (evoked_loss + spont_loss) / 2.0
        train_loss.update_state(mean_loss * strategy.num_replicas_in_sync)
        rate = tf.reduce_mean(tf.cast(_out[0], tf.float32))
        train_firing_rate.update_state(rate)
        train_rate_loss.update_state(mean_aux["rate_loss"])
        train_voltage_loss.update_state(mean_aux["voltage_loss"])
        train_regularizer_loss.update_state(mean_aux["regularizer_loss"])
        train_sync_loss.update_state(mean_aux["sync_loss"])
        train_osi_dsi_loss.update_state(mean_aux["osi_dsi_loss"])

        if return_sequences:
            return total_loss, mean_aux, _out

    def train_step_sequential(x, y, x_spontaneous, state_variables, trim, spontaneous=False, return_sequences=False):
        spontaneous = tf.cast(spontaneous, tf.bool)
        metric_weight = tf.cast(0.5, tf.float32)
        _x = tf.cond(spontaneous, lambda: x_spontaneous, lambda: x)

        with tf.GradientTape() as tape:
            _out, _loss, _aux = roll_out(
                _x, y, state_variables, spontaneous=spontaneous, trim=trim
            )
            _loss = tf.cast(_loss, tf.float32)
            loss_for_grad = optimizer_utils.scale_loss_for_optimizer(optimizer, _loss)

        grad = tape.gradient(loss_for_grad, model.trainable_variables)
        grad = optimizer_utils.unscale_gradients_for_optimizer(optimizer, grad)

        if flags.debug_gradients:
            grad = _print_and_check_gradients(
                grad, "[Sequential]", spontaneous=spontaneous
            )

        optimizer.apply_gradients(zip(grad, model.trainable_variables))

        train_loss.update_state(
            _loss * strategy.num_replicas_in_sync, sample_weight=metric_weight
        )
        rate = tf.reduce_mean(tf.cast(_out[0], tf.float32))
        train_firing_rate.update_state(rate)
        train_rate_loss.update_state(_aux["rate_loss"], sample_weight=metric_weight)
        train_voltage_loss.update_state(_aux["voltage_loss"], sample_weight=metric_weight)
        train_regularizer_loss.update_state(_aux["regularizer_loss"], sample_weight=metric_weight)
        train_sync_loss.update_state(_aux["sync_loss"], sample_weight=metric_weight)
        osi_weight = tf.where(
            spontaneous,
            tf.constant(0.0, dtype=tf.float32),
            metric_weight,
        )
        train_osi_dsi_loss.update_state(_aux["osi_dsi_loss"], sample_weight=osi_weight)

        if return_sequences:
            return _loss, _aux, _out

    # @tf.function
    # def distributed_train_step(x, y, state_variables, spontaneous, trim):
    #     _loss, _aux, _out, grad = train_step(x, y, state_variables, spontaneous, trim)
    #     return _loss, _aux, _out, grad

    # def combine_gradients(_x, _y, state_variables, _x_spontaneous, trim=True):
    #     evoked_loss, _evoked_aux, _evoked_out, evoked_grad = distributed_train_step(_x, _y, state_variables, False, trim)
    #     spont_loss, _spont_aux, _spont_out, spont_grad = distributed_train_step(_x_spontaneous, _y, state_variables, True, trim)
    #     # Combine gradients
    #     combined_gradients = []
    #     for evo_grad, spo_grad in zip(evoked_grad, spont_grad):
    #         combined_gradients.append(evo_grad + spo_grad)

    #     # Apply combined gradients
    #     optimizer.apply_gradients(zip(combined_gradients, model.trainable_variables))

    #     return evoked_loss, _evoked_aux, _evoked_out, spont_loss, _spont_aux, _spont_out

    # @tf.function
    # def split_train_step(_x, _y, state_variables, _x_spontaneous, trim=True):
    #     evoked_loss, _evoked_aux, _out_evoked, spont_loss, _spont_aux, _out_spontaneous = strategy.run(combine_gradients, args=(_x, _y, state_variables, _x_spontaneous, trim))

    #     v1_spikes_evoked = strategy.experimental_local_results(_out_evoked)[0][0][0]
    #     v1_spikes_spont = strategy.experimental_local_results(_out_spontaneous)[0][0][0]
    #     model_spikes = (v1_spikes_evoked, v1_spikes_spont)

    #     rate_loss = train_rate_loss.result()
    #     voltage_loss = train_voltage_loss.result()
    #     regularizers_loss = train_regularizer_loss.result()
    #     sync_loss = train_sync_loss.result()
    #     osi_dsi_loss = train_osi_dsi_loss.result()
    #     _loss = train_loss.result()
    #     rate = train_firing_rate.result()

    #     step_values = [_loss, rate, rate_loss, voltage_loss, regularizers_loss, osi_dsi_loss, sync_loss]

    #     return model_spikes, step_values

    @tf.function
    def distributed_train_step(
        x,
        y,
        x_spontaneous,
        state_variables,
        trim,
        return_sequences=False,
        spontaneous=False,
    ):
        if flags.sequential_stimuli:
            return strategy.run(
                train_step_sequential,
                args=(x, y, x_spontaneous, state_variables, trim,
                      spontaneous, return_sequences),
            )
        return strategy.run(
            train_step_combined, args=(
                x, y, x_spontaneous, state_variables, trim, return_sequences)
        )

    def split_train_step(x, y, state_variables, x_spontaneous, trim=True, capture_spikes=False):
        if capture_spikes:
            if flags.sequential_stimuli:
                _, _, out_evoked = distributed_train_step(
                    x,
                    y,
                    x_spontaneous,
                    state_variables,
                    trim,
                    return_sequences=True,
                    spontaneous=False
                )
                _, _, out_spont = distributed_train_step(
                    x,
                    y,
                    x_spontaneous,
                    state_variables,
                    trim,
                    return_sequences=True,
                    spontaneous=True
                )
                v1_spikes_evoked = strategy.experimental_local_results(out_evoked)[0][0]
                v1_spikes_spont = strategy.experimental_local_results(out_spont)[0][0]
            else:
                _loss, _aux, _out = distributed_train_step(
                    x, y, x_spontaneous, state_variables, trim, return_sequences=True
                )
                v1_spikes_full = strategy.experimental_local_results(_out)[0][0]
                v1_spikes_evoked = v1_spikes_full[:per_replica_batch_size]
                v1_spikes_spont = v1_spikes_full[per_replica_batch_size:]
            model_spikes = (v1_spikes_evoked, v1_spikes_spont)
        else:
            if flags.sequential_stimuli:
                distributed_train_step(
                    x,
                    y,
                    x_spontaneous,
                    state_variables,
                    trim,
                    return_sequences=False,
                    spontaneous=False,
                )
                distributed_train_step(
                    x,
                    y,
                    x_spontaneous,
                    state_variables,
                    trim,
                    return_sequences=False,
                    spontaneous=True,
                )
            else:
                distributed_train_step(x, y, x_spontaneous, state_variables, trim, return_sequences=False)
            model_spikes = (None, None)

        rate_loss = train_rate_loss.result()
        voltage_loss = train_voltage_loss.result()
        regularizers_loss = train_regularizer_loss.result()
        sync_loss = train_sync_loss.result()
        osi_dsi_loss = train_osi_dsi_loss.result()
        _loss = train_loss.result()
        rate = train_firing_rate.result()

        step_values = [_loss, rate, rate_loss, voltage_loss,
                       regularizers_loss, osi_dsi_loss, sync_loss]

        return model_spikes, step_values

    ### LGN INPUT ###
    # Define the function that generates the dataset for our task
    def get_gratings_dataset_fn(regular=False):
        def _f(input_context):
            batch_size = input_context.get_per_replica_batch_size(global_batch_size)
            pipeline_seed = flags.seed + 10000 + int(input_context.input_pipeline_id)
            _data_set = (stim_dataset.generate_drifting_grating_tuning(
                seq_len=flags.seq_len,
                pre_delay=delays[0],
                post_delay=delays[1],
                n_input=flags.n_input,
                data_dir=flags.data_dir,
                regular=regular,
                bmtk_compat=flags.bmtk_compat_lgn,
                rotation=flags.rotation,
                dtype=dtype,
                seed=pipeline_seed,
            )
                .batch(batch_size)
                .prefetch(tf.data.AUTOTUNE)
            )

            return _data_set
        return _f

    def get_gray_dataset_fn():
        def _f(input_context):
            batch_size = input_context.get_per_replica_batch_size(global_batch_size)
            pipeline_seed = flags.seed + 20000 + int(input_context.input_pipeline_id)
            _gray_data_set = (stim_dataset.generate_gray_screen_stimulus(
                seq_len=flags.seq_len,
                n_input=flags.n_input,
                data_dir=flags.data_dir,
                return_firing_rates=False,
                dtype=dtype,
                seed=pipeline_seed,
            )
                .batch(batch_size)
                .prefetch(tf.data.AUTOTUNE)
            )

            return _gray_data_set
        return _f

    # We define the dataset generates function under the strategy scope for a randomly selected orientation or gray screen
    if flags.spontaneous_training:
        train_data_set = strategy.distribute_datasets_from_function(get_gray_dataset_fn())
    else:
        train_data_set = strategy.distribute_datasets_from_function(get_gratings_dataset_fn())

    def sample_probability_batch(probability, batch_size, current_input=False):
        batch_size = tf.cast(batch_size, tf.int32)
        probability = tf.cast(probability, dtype)
        target_shape = tf.concat([[batch_size], tf.shape(probability)], axis=0)
        probability = tf.broadcast_to(probability, target_shape)
        if current_input:
            return tf.cast(probability * tf.cast(1.3, dtype), dtype)
        random_uniform = tf.random.stateless_uniform(
            tf.shape(probability),
            seed=seed_helper.next_spontaneous_seed(),
            dtype=dtype,
        )
        return tf.less(random_uniform, probability)

    def generate_spontaneous_spikes(batch_size):
        return sample_probability_batch(spontaneous_prob_base, batch_size)

    @tf.function
    def distributed_sample_probability_batch(probability, batch_size, current_input=False):
        return strategy.run(
            sample_probability_batch,
            args=(probability, batch_size),
            kwargs={"current_input": current_input},
        )

    def generate_gray_state(batch_size):
        batch_size = tf.cast(batch_size, tf.int32)
        x = generate_spontaneous_spikes(batch_size)
        if x.dtype == tf.bool:
            x = tf.cast(x, dtype)
        init_state = rsnn_layer.cell.zero_state(batch_size, dtype=dtype)
        if state_model is not None:
            seed_helper.advance_noise_seed()
            inputs = [x]
            inputs.extend(list(init_state))
            state_out = state_model(tuple(inputs))
            return state_out  # tuple(tf.nest.flatten(state_out))
        seed_helper.advance_noise_seed()
        inputs = [x]
        inputs.extend(list(init_state))
        return state_fallback_model(tuple(inputs))

    @tf.function
    def distributed_generate_gray_state(batch_size):
        # Run generate_gray_state on each replica
        return strategy.run(generate_gray_state, args=(batch_size,))

    def validation_step(x, state_variables):
        _out = run_extractor(x, state_variables)
        return _out[0], _out[1]

    @tf.function
    def distributed_validation_step(x, state_variables):
        return strategy.run(validation_step, args=(x, state_variables))

    def resampled_emd(model_values, target_values):
        model_values = np.asarray(model_values, dtype=np.float32)
        target_values = np.asarray(target_values, dtype=np.float32)
        model_values = model_values[np.isfinite(model_values)]
        target_values = target_values[np.isfinite(target_values)]
        if model_values.size == 0 or target_values.size == 0:
            return None
        target_resampled = losses.resample_sorted_distribution(
            target_values,
            model_values.size,
        )
        return float(np.mean(np.abs(np.sort(model_values) - target_resampled)))

    def equal_cell_type_emd(
        model_values,
        target_df,
        target_column,
        population_ids,
        target_scale=1.0,
    ):
        emds = []
        for cell_type in losses.CELL_TYPE_ORDER:
            node_ids = np.asarray(population_ids.get(cell_type, []), dtype=np.int32)
            if node_ids.size == 0:
                continue
            target_values = target_df.loc[target_df["cell_type"] == cell_type, target_column]
            if target_values.empty:
                continue
            emd = resampled_emd(
                np.asarray(model_values)[node_ids],
                target_values.dropna().to_numpy(dtype=np.float32) * float(target_scale),
            )
            if emd is not None:
                emds.append(emd)

        return float(np.mean(emds))

    def load_neuropixels_targets(neuropixels_df):
        features = [
            "ecephys_unit_id",
            "cell_type",
            "firing_rate_sp",
            "Spont_Rate(Hz)",
            "Ave_Rate(Hz)",
            "max_mean_rate(Hz)",
            "OSI",
            "DSI",
        ]
        available = pd.read_csv(neuropixels_df, sep=" ", nrows=0).columns
        usecols = [column for column in features if column in available]
        target_df = pd.read_csv(neuropixels_df, sep=" ", usecols=usecols).dropna(how="all")
        if "Spont_Rate(Hz)" in target_df.columns and "firing_rate_sp" not in target_df.columns:
            target_df.rename(columns={"Spont_Rate(Hz)": "firing_rate_sp"}, inplace=True)
        target_df = target_df[target_df["cell_type"].notna()].copy()
        target_df["cell_type"] = target_df["cell_type"].apply(
            losses.neuropixels_cell_type_to_cell_type
        )
        if "max_mean_rate(Hz)" in target_df.columns:
            nonresponding = target_df["max_mean_rate(Hz)"] < 0.5
            target_df.loc[nonresponding, ["OSI", "DSI"]] = np.nan
        return target_df

    def collect_local_validation_outputs(distributed_x, distributed_z, distributed_v, keep):
        local_x = strategy.experimental_local_results(distributed_x)
        local_z = strategy.experimental_local_results(distributed_z)
        local_v = strategy.experimental_local_results(distributed_v)
        xs = []
        zs = []
        vs = []
        for x_replica, z_replica, v_replica in zip(local_x, local_z, local_v):
            xs.append(x_replica)
            zs.append(z_replica)
            vs.append(v_replica)
        return (
            tf.concat(xs, axis=0)[:keep],
            tf.concat(zs, axis=0)[:keep],
            tf.concat(vs, axis=0)[:keep],
        )

    def trimmed_rates_hz(spikes):
        trimmed = spikes[:, delays[0]:flags.seq_len - delays[1], :]
        return np.mean(trimmed.astype(np.float32), axis=1) * 1000.0

    def validation_rate_loss_tensor(spikes, spontaneous):
        if spontaneous:
            rate_loss = spont_rate_regularizer(spikes, trim=True)
            if annulus_mask is not None:
                rate_loss += annulus_spont_rate_regularizer(spikes, trim=True)
        else:
            rate_loss = evoked_rate_regularizer(spikes, trim=True)
            if annulus_mask is not None:
                rate_loss += annulus_evoked_rate_regularizer(spikes, trim=True)
        return tf.cast(rate_loss, tf.float32)

    def distributed_validation_rate_loss(spikes, spontaneous):
        replica_losses = strategy.run(
            validation_rate_loss_tensor,
            args=(spikes, spontaneous),
        )
        return float(
            strategy.reduce(
                tf.distribute.ReduceOp.MEAN,
                replica_losses,
                axis=None,
            ).numpy()
        )

    def run_gray_validation_repeats(repeats):
        spont_rates = []
        spont_rate_losses = []
        spont_voltage_losses = []
        spont_sync_losses = []
        representative = None
        completed = 0

        while completed < repeats:
            keep = min(global_batch_size, repeats - completed)
            x = distributed_sample_probability_batch(
                spontaneous_prob_base,
                per_replica_batch_size,
            )
            state = distributed_generate_gray_state(per_replica_batch_size)
            z, v = distributed_validation_step(x, state)
            rate_loss = distributed_validation_rate_loss(z, spontaneous=True)
            x_local, z_local, v_local = collect_local_validation_outputs(x, z, v, keep)
            z_np = z_local.numpy()
            spont_rates.append(trimmed_rates_hz(z_np))
            spont_rate_losses.append(rate_loss)
            spont_voltage_losses.append(
                float(tf.cast(voltage_regularizer(v_local), tf.float32).numpy())
            )
            spont_sync_losses.append(
                float(tf.cast(spont_sync_loss(z_local, trim=True), tf.float32).numpy())
            )
            if representative is None:
                representative = (
                    x_local[:1],
                    z_local[:1],
                )
            completed += keep

        return {
            "spont_rates": np.concatenate(spont_rates, axis=0),
            "spont_rate_loss": float(np.mean(spont_rate_losses)),
            "spont_voltage_loss": float(np.mean(spont_voltage_losses)),
            "spont_sync_loss": float(np.mean(spont_sync_losses)),
            "representative": representative,
        }

    def run_osi_dsi_validation_repeats(probability, angle, repeats, collect_spikes=False):
        evoked_rates = []
        evoked_rate_losses = []
        evoked_voltage_losses = []
        evoked_sync_losses = []
        protocol_spikes = []
        representative = None
        completed = 0
        protocol_mask = np.asarray(core_mask, dtype=bool)

        while completed < repeats:
            keep = min(global_batch_size, repeats - completed)
            x = distributed_sample_probability_batch(probability, per_replica_batch_size)
            state = distributed_generate_gray_state(per_replica_batch_size)
            z_evoked, v_evoked = distributed_validation_step(x, state)
            rate_loss = distributed_validation_rate_loss(z_evoked, spontaneous=False)
            x_local, z_evoked, v_evoked = collect_local_validation_outputs(x, z_evoked, v_evoked, keep)
            z_evoked_np = z_evoked.numpy()
            evoked_rates.append(trimmed_rates_hz(z_evoked_np))
            evoked_rate_losses.append(rate_loss)
            evoked_voltage_losses.append(
                float(tf.cast(voltage_regularizer(v_evoked), tf.float32).numpy())
            )
            evoked_sync_losses.append(
                float(tf.cast(evoked_sync_loss(z_evoked, trim=True), tf.float32).numpy())
            )
            if collect_spikes:
                if protocol_mask is None:
                    protocol_spikes.append(z_evoked_np.astype(np.float32))
                else:
                    protocol_spikes.append(z_evoked_np[:, :, protocol_mask].astype(np.float32))
            if representative is None:
                representative = (
                    x_local[:1],
                    z_evoked[:1],
                    tf.constant([[float(angle)]], dtype=dtype),
                )
            completed += keep

        collected_protocol_spikes = None
        if collect_spikes:
            collected_protocol_spikes = np.concatenate(protocol_spikes, axis=0)
        return {
            "evoked_rates": np.concatenate(evoked_rates, axis=0),
            "evoked_rate_loss": float(np.mean(evoked_rate_losses)),
            "evoked_voltage_loss": float(np.mean(evoked_voltage_losses)),
            "evoked_sync_loss": float(np.mean(evoked_sync_losses)),
            "protocol_spikes": collected_protocol_spikes,
            "representative": representative,
        }

    def update_validation_metrics_on_replica(*values):
        for metric, value in zip(validation_metrics, values):
            metric.update_state(value)

    @tf.function
    def distributed_reset_validation_metrics():
        strategy.run(reset_validation_metrics)

    @tf.function
    def distributed_update_validation_metrics(*values):
        distributed_reset_validation_metrics()
        strategy.run(
            update_validation_metrics_on_replica,
            args=values,
        )

    def update_validation_metrics(metric_values):
        distributed_update_validation_metrics(
            *tuple(tf.constant(value, dtype=tf.float32) for value in metric_values)
        )

    def run_protocol_validation():
        spont_result = run_gray_validation_repeats(protocol_n_trials)
        evoked_rates_by_angle = []
        evoked_spikes_by_angle = []
        evoked_rate_losses = []
        evoked_voltage_losses = []
        evoked_sync_losses = []
        representative_evoked = None

        for angle in protocol_angles:
            result = run_osi_dsi_validation_repeats(
                osi_dsi_lgn_probabilities[int(angle)],
                float(angle),
                protocol_n_trials,
                collect_spikes=True,
            )
            evoked_rates_by_angle.append(result["evoked_rates"])
            evoked_spikes_by_angle.append(result["protocol_spikes"])
            evoked_rate_losses.append(result["evoked_rate_loss"])
            evoked_voltage_losses.append(result["evoked_voltage_loss"])
            evoked_sync_losses.append(result["evoked_sync_loss"])
            if representative_evoked is None:
                x_rep, z_rep, y_rep = result["representative"]
                representative_evoked = (x_rep, z_rep, y_rep)

        evoked_rates = np.stack(evoked_rates_by_angle, axis=1)
        protocol_spikes = np.stack(evoked_spikes_by_angle, axis=1)
        osi_mask = np.asarray(core_mask, dtype=bool)
        osi_rates = evoked_rates[:, :, osi_mask] if osi_mask is not None else evoked_rates
        osi_dsi_df = calculate_OSI_DSI(
            osi_rates,
            network,
            session="drifting_gratings",
            DG_angles=protocol_angles,
            core_radius=flags.loss_core_radius if flags.loss_core_radius > 0 else None,
            remove_zero_rate_neurons=False,
            directory="",
            save_df=False,
            data_dir=flags.data_dir,
        )

        cell_type_populations_ids = losses.get_population_neuron_ids(
            network,
            data_dir=flags.data_dir,
            core_mask=core_mask,
            reindex_selected=False,
        )
        rate_loss = (
            float(np.mean(evoked_rate_losses)) + float(spont_result["spont_rate_loss"])
        ) / 2.0

        osi_values = np.full(network["n_nodes"], np.nan, dtype=np.float32)
        dsi_values = np.full(network["n_nodes"], np.nan, dtype=np.float32)
        if osi_mask is None:
            osi_values[:] = osi_dsi_df["OSI"].to_numpy(dtype=np.float32)
            dsi_values[:] = osi_dsi_df["DSI"].to_numpy(dtype=np.float32)
        else:
            selected_ids = np.flatnonzero(osi_mask)
            osi_values[selected_ids] = osi_dsi_df["OSI"].to_numpy(dtype=np.float32)
            dsi_values[selected_ids] = osi_dsi_df["DSI"].to_numpy(dtype=np.float32)

        osi_emd = equal_cell_type_emd(
            osi_values,
            protocol_target_df,
            "OSI",
            cell_type_populations_ids,
        )
        dsi_emd = equal_cell_type_emd(
            dsi_values,
            protocol_target_df,
            "DSI",
            cell_type_populations_ids,
        )
        osi_dsi_loss = flags.osi_cost * (osi_emd + dsi_emd)
        voltage_loss = (
            float(np.mean(evoked_voltage_losses)) + float(spont_result["spont_voltage_loss"])
        ) / 2.0
        sync_loss = (
            float(np.mean(evoked_sync_losses)) + float(spont_result["spont_sync_loss"])
        ) / 2.0

        regularizer_loss = 0.
        if flags.train_recurrent and flags.recurrent_weight_regularization > 0:
            regularizer_loss = float(
                rec_weight_regularizer(rsnn_layer.cell.recurrent_weight_values).numpy())

        loss_value = rate_loss + voltage_loss + regularizer_loss + osi_dsi_loss + sync_loss
        firing_rate = (
            float(np.mean(evoked_rates) / 1000.0)
            + float(np.mean(spont_result["spont_rates"]) / 1000.0)
        ) / 2.0
        metric_values = [
            float(loss_value),
            float(firing_rate),
            float(rate_loss),
            float(voltage_loss),
            float(regularizer_loss),
            float(osi_dsi_loss),
            float(sync_loss),
        ]
        update_validation_metrics(metric_values)
        x_rep, z_rep, y_rep = representative_evoked
        x_spont_rep, z_spont_rep = spont_result["representative"]
        return (
            [metric.result().numpy() for metric in validation_metrics],
            x_rep,
            z_rep,
            y_rep,
            x_spont_rep,
            z_spont_rep,
            protocol_spikes,
            protocol_angles,
        )

    # def reset_state(new_state):
    #     tf.nest.map_structure(lambda a, b: a.assign(b), state_variables, new_state)

    # # @tf.function
    # def distributed_reset_state(new_state):
    #     strategy.run(reset_state, args=(new_state,))

    # def get_next_chunknum(chunknum, seq_len, direction='up'):
    #     # get the next chunk number (diviser) for seq_len.
    #     if direction == 'up':
    #         chunknum += 1
    #         # check if it is a valid diviser
    #         while seq_len % chunknum != 0:
    #             chunknum += 1
    #             if chunknum >= seq_len:
    #                 print('Chunk number reached seq_len')
    #                 return seq_len
    #     elif direction == 'down':
    #         chunknum -= 1
    #         while seq_len % chunknum != 0:
    #             chunknum -= 1
    #             if chunknum <= 1:
    #                 print('Chunk number reached 1')
    #                 return 1
    #     else:
    #         raise ValueError(f"Invalid direction: {direction}")
    #     return chunknum

    ############################ TRAINING #############################

    stop = False
    # Initialize your callbacks
    metric_keys = ['train_loss', 'train_firing_rate', 'train_rate_loss', 'train_voltage_loss',
                   'train_regularizer_loss', 'train_osi_dsi_loss', 'train_sync_loss', 'val_loss',
                   'val_firing_rate', 'val_rate_loss', 'val_voltage_loss', 'val_regularizer_loss', 'val_osi_dsi_loss', 'val_sync_loss']

    callbacks = Callbacks(network, lgn_input, bkg_input, model, optimizer, flags, logdir, strategy,
                          metric_keys, pre_delay=delays[0], post_delay=delays[1], model_variables_init=model_variables_dict,
                          checkpoint=checkpoint, spontaneous_training=flags.spontaneous_training)

    protocol_angles = np.asarray(tuple(range(0, 360, 45)), dtype=np.int32)
    protocol_n_trials = 10
    protocol_target_df = load_neuropixels_targets(flags.neuropixels_df)

    osi_dsi_lgn_probabilities = stim_dataset.load_or_compute_osi_dsi_lgn_probabilities(
        seq_len=flags.seq_len,
        pre_delay=delays[0],
        post_delay=delays[1],
        n_input=flags.n_input,
        data_dir=flags.data_dir,
        rotation=flags.rotation,
        seed=flags.seed,
        output_dtype=dtype,
        angles=protocol_angles,
        strategy=strategy,
        bmtk_compat=flags.bmtk_compat_lgn,
        current_input=flags.current_input,
        cache_prefix="protocol_validation_lgn_probabilities",
    )

    callbacks.on_train_begin()
    # chunknum = 1
    # max_working_fr = {}   # defined for each chunknum
    n_prev_epochs = flags.run_session * flags.n_epochs  # used for resuming training and logging correct epoch numbers in that case
    profiler_logdir = None
    profiler_finished = False
    if flags.profile_train_step:
        profiler_logdir = flags.profile_logdir or os.path.join(
            logdir, "logs", "profile", strftime("%Y%m%d-%H%M%S")
        )
        os.makedirs(profiler_logdir, exist_ok=True)
        print(
            "TensorFlow profiler enabled for "
            f"epoch={n_prev_epochs}, step={flags.profile_step}. "
            f"Trace output: {profiler_logdir}"
        )

    # import datetime
    # profiler_logdir = f"{logdir}/logs/profile/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    # Set steps to profile
    # profile_start_step = 1
    # profile_end_step = 7

    # Load the dataset iterator
    it = iter(train_data_set)

    for epoch in range(n_prev_epochs, n_prev_epochs + flags.n_epochs):
        callbacks.on_epoch_start()
        # Reset the model state to the gray state
        gray_state = distributed_generate_gray_state(real_batch_size)

        # tf.profiler.experimental.start(logdir=logdir)
        for step in range(flags.steps_per_epoch):
            callbacks.on_step_start()
            # Start profiler at specified step
            # if step == profile_start_step:
            #     tf.profiler.experimental.start(logdir=logdir)

            # try resetting every iteration
            if flags.reset_every_step:
                gray_state = distributed_generate_gray_state(real_batch_size)

            # Generate LGN spikes
            x, y, _, _ = next(it)  # x dtype tf.bool
            x_spontaneous = distributed_sample_probability_batch(
                spontaneous_prob_base,
                per_replica_batch_size,
            )
            profile_this_step = (
                flags.profile_train_step
                and not profiler_finished
                and epoch == n_prev_epochs
                and step == flags.profile_step
            )

            # with tf.profiler.experimental.Trace('train', step_num=step, _r=1):
            # while True:
            try:
                # x_chunks = tf.split(x, chunknum, axis=1)
                # x_spont_chunks = tf.split(x_spontaneous, chunknum, axis=1)
                # seq_len_local = x.shape[1] // chunknum
                # for j in range(chunknum):
                #     x_chunk = x_chunks[j]
                #     x_spont_chunk = x_spont_chunks[j]
                #     # Profile specific steps
                #     # if profile_start_step <= step <= profile_end_step:
                #     #     with tf.profiler.experimental.Trace('train', step_num=step, _r=1):
                #     #         model_spikes, step_values = distributed_split_train_step(x_chunk, y, w, x_spont_chunk, trim=chunknum==1)
                #     # else:
                #     model_spikes, step_values = distributed_split_train_step(x_chunk, y, gray_state, x_spont_chunk, trim=chunknum==1)
                # # distributed_train_step(x, y, w, trim=chunknum==1)
                # model_spikes, step_values = distributed_split_train_step(x, y, gray_state, x_spontaneous, trim=chunknum==1)
                if profile_this_step:
                    options = tf.profiler.experimental.ProfilerOptions(
                        host_tracer_level=2,
                        python_tracer_level=1,
                        device_tracer_level=1,
                    )
                    tf.profiler.experimental.start(profiler_logdir, options=options)
                    try:
                        with tf.profiler.experimental.Trace("train_step", step_num=step, _r=1):
                            _, step_values = split_train_step(
                                x, y, gray_state, x_spontaneous, trim=True, capture_spikes=False
                            )
                    finally:
                        tf.profiler.experimental.stop()
                    profiler_finished = True
                    print(f"TensorFlow profiler trace written to: {profiler_logdir}")
                else:
                    _, step_values = split_train_step(
                        x, y, gray_state, x_spontaneous, trim=True, capture_spikes=False
                    )
                # break
            except tf.errors.ResourceExhaustedError as e:
                raise RuntimeError(
                    "ResourceExhaustedError during training. "
                    f"Epoch={epoch}, step={step}. Reduce memory pressure by lowering "
                    "--batch_size/--seq_len, enabling --sequential_stimuli, or reducing "
                    "network size."
                ) from e


            callbacks.on_step_end(step_values, y, verbose=True)
            if profiler_finished and flags.profile_stop_after_capture:
                print("Stopping after captured profiler step (--profile_stop_after_capture).")
                return

        # tf.profiler.experimental.stop()

        ### VALIDATION AFTER EACH EPOCH
        (
            val_values,
            x_val,
            v1_spikes_val,
            y_val,
            x_spont_val,
            v1_spikes_spont_val,
            protocol_spikes,
            protocol_angles_epoch,
        ) = run_protocol_validation()

        train_values = [a.result().numpy() for a in [train_loss, train_firing_rate,
                                                     train_rate_loss, train_voltage_loss, train_regularizer_loss,
                                                     train_osi_dsi_loss, train_sync_loss]]
        metric_values = train_values + val_values

        stop = callbacks.on_epoch_end(
            x_val,
            v1_spikes_val,
            y_val,
            metric_values,
            verbose=True,
            x_spont=x_spont_val,
            v1_spikes_spont=v1_spikes_spont_val,
            protocol_spikes=protocol_spikes,
            protocol_angles=protocol_angles_epoch,
        )

        if stop:
            break

        # Reset the metrics for the next epoch
        reset_train_metrics()
        distributed_reset_validation_metrics()

    normalizers = {'v1_ema': v1_ema.numpy()}
    if flags.osi_loss_method in ("rolling_osi_emd", "adaptative_crowd_osi"):
        rolling_state = {"core": OSI_DSI_Loss.get_rolling_state()}
        if annulus_mask is not None:
            rolling_state["annulus"] = annulus_OSI_DSI_Loss.get_rolling_state()
        if flags.osi_loss_method == "rolling_osi_emd":
            normalizers["rolling_osi_emd_state"] = rolling_state
        else:
            normalizers["adaptative_crowd_osi_state"] = rolling_state
    callbacks.on_train_end(metric_values, normalizers=normalizers)


if __name__ == '__main__':
    hostname = socket.gethostname()
    print("*" * 80)
    print(hostname)
    print("*" * 80)
    # make a condition for different machines. The allen institute has
    # cluster host name to be n??? where ??? is 3 digit number.
    # let's make regex for that.
    # if hostname.count('alleninstitute') > 0 or re.search(r'n\d{3}', hostname) is not None:
    #     _data_dir = '/allen/programs/mindscope/workgroups/realistic-model/shinya.ito/tensorflow_new/V1_GLIF_model/GLIF_network'
    #     _results_dir = '/allen/programs/mindscope/workgroups/realistic-model/shinya.ito/tensorflow_new/V1_GLIF_model/Simulation_results'
    # else:
    #     _data_dir = '/home/jgalvan/Desktop/Neurocoding/V1_GLIF_model/GLIF_network'
    #     _results_dir = '/home/jgalvan/Desktop/Neurocoding/V1_GLIF_model/Simulation_results'

    # absl.app.flags.DEFINE_string('data_dir', _data_dir, '')
    absl.app.flags.DEFINE_string('data_dir', 'GLIF_network', '')
    # absl.app.flags.DEFINE_string('results_dir', _results_dir, '')
    absl.app.flags.DEFINE_string('results_dir', 'Simulation_results', '')
    absl.app.flags.DEFINE_string(
        'task_name', 'drifting_gratings_firing_rates_distr', '')

    # absl.app.flags.DEFINE_string('restore_from', '../results/multi_training/b_53dw/results/ckpt-49', '')
    absl.app.flags.DEFINE_string('restore_from', '', '')
    absl.app.flags.DEFINE_string('comment', '', '')
    absl.app.flags.DEFINE_string('delays', '0,0', '')
    # absl.app.flags.DEFINE_string('neuron_model', 'GLIF3', '')
    absl.app.flags.DEFINE_string('scale', '2,2', '')
    absl.app.flags.DEFINE_string('dtype', 'float16', '')
    absl.app.flags.DEFINE_enum(
        'synaptic_current_backend',
        'cuda',
        ['cuda', 'tensorflow'],
        'Recurrent synaptic-current implementation.',
    )
    absl.app.flags.DEFINE_string('rotation', 'ccw', '')
    absl.app.flags.DEFINE_string('ckpt_dir', '', '')
    absl.app.flags.DEFINE_string('osi_loss_method', 'crowd_osi', '')
    absl.app.flags.DEFINE_string('optimizer', 'exp_adam', '')
    absl.app.flags.DEFINE_string('neuropixels_df', 'Neuropixels_data/OSI_DSI_neuropixels_v4.csv',
                                 'File name of the Neuropixels DataFrame for OSI/DSI analysis.')

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
    absl.app.flags.DEFINE_float('sync_cost', 1.5, '')
    absl.app.flags.DEFINE_float('voltage_cost', 1., '')
    absl.app.flags.DEFINE_float('osi_cost', 20., '')
    absl.app.flags.DEFINE_float('annulus_loss_weight', 0.1, '')
    absl.app.flags.DEFINE_float('osi_loss_subtraction_ratio', 1., '')
    absl.app.flags.DEFINE_float(
        'rolling_decay',
        -1.0,
        'EMA decay for rolling_osi_emd. Set < 0 to auto-compute from batch_size and rolling_target_sample_ess.',
    )
    absl.app.flags.DEFINE_float(
        'rolling_target_sample_ess',
        80.0,
        'Target effective sample size in samples used when rolling_decay < 0 (auto mode).',
    )
    absl.app.flags.DEFINE_boolean(
        'rolling_gradient_correction',
        False,
        'Scale current-batch gradients through the rolling OSI/DSI EMA without changing forward values.',
    )
    absl.app.flags.DEFINE_float(
        'rolling_max_gradient_scale',
        20.0,
        'Maximum gradient scale used by rolling_gradient_correction.',
    )
    absl.app.flags.DEFINE_boolean(
        'rolling_warmup',
        True,
        'Ramp rolling OSI/DSI loss by current EMA effective sample size during cold start.',
    )
    absl.app.flags.DEFINE_float('dampening_factor', 0.1, '')
    absl.app.flags.DEFINE_float("recurrent_dampening_factor", 0.1, "")
    absl.app.flags.DEFINE_float('input_weight_scale', 1., '')
    absl.app.flags.DEFINE_float('gauss_std', .3, '')
    absl.app.flags.DEFINE_float('recurrent_weight_regularization', 10., '')
    absl.app.flags.DEFINE_string('recurrent_weight_regularizer_type', 'emd',
                                 'Type of recurrent weight regularizer. Options: mean, stiff, kl_lognormal, emd')
    absl.app.flags.DEFINE_string('voltage_penalty_mode', 'threshold',
                                 'Type of penalization for voltage. Options: range, threshold')
    absl.app.flags.DEFINE_float('lr_scale', 1., '')
    # absl.app.flags.DEFINE_float('p_reappear', .5, '')
    absl.app.flags.DEFINE_float('max_time', -1, '')
    # absl.app.flags.DEFINE_float('max_time', 0.05, '')
    # absl.app.flags.DEFINE_float('scale_w_e', -1, '')
    # absl.app.flags.DEFINE_float('sti_intensity', 2., '')
    absl.app.flags.DEFINE_float('temporal_f', 2., '')
    # 0 is not using core loss
    absl.app.flags.DEFINE_float('loss_core_radius', 200.0, '')
    # 0 is not using core plot
    absl.app.flags.DEFINE_float('plot_core_radius', 200.0, '')

    absl.app.flags.DEFINE_integer('n_runs', 1, '')
    absl.app.flags.DEFINE_integer('run_session', 0, '')
    absl.app.flags.DEFINE_integer('n_epochs', 75, '')
    # number of epochs for osi/dsi evaluation if n_runs = 1
    absl.app.flags.DEFINE_integer('osi_dsi_eval_period', 1, '')
    absl.app.flags.DEFINE_integer('batch_size', 5, '')
    absl.app.flags.DEFINE_integer('neurons', 0, '')  # 0 to take all neurons
    absl.app.flags.DEFINE_integer("n_input", 17400, "")
    absl.app.flags.DEFINE_integer('seq_len', 500, '')
    # absl.app.flags.DEFINE_integer('im_slice', 100, '')
    absl.app.flags.DEFINE_integer('seed', 3000, '')
    # absl.app.flags.DEFINE_integer('port', 12778, '')
    absl.app.flags.DEFINE_integer("n_output", 2, "")
    absl.app.flags.DEFINE_integer('neurons_per_output', 16, '')
    # EA and garret dose not need this many but pure classification needs 781 = int(50000/64)
    absl.app.flags.DEFINE_integer('steps_per_epoch', 25, '')
    # EA and garret dose not need this many but pure classification needs 156 = int(10000/64)
    absl.app.flags.DEFINE_integer('val_steps', 1, '')
    absl.app.flags.DEFINE_integer('max_delay', 0, 'Maximum synaptic delay in ms (0 = auto from data)')
    # absl.app.flags.DEFINE_integer('n_plots', 1, '')
    absl.app.flags.DEFINE_integer('n_trials_per_angle', 10, '')
    absl.app.flags.DEFINE_integer("cue_duration", 40, "")
    absl.app.flags.DEFINE_integer('fano_samples', 500, '')

    # absl.app.flags.DEFINE_integer('pre_chunks', 3, '')
    # absl.app.flags.DEFINE_integer('post_chunks', 8, '') # the pure classification task only need 1 but to make consistent with other tasks one has to make up here
    # absl.app.flags.DEFINE_integer('pre_delay', 50, '')
    # absl.app.flags.DEFINE_integer('post_delay', 450, '')

    # absl.app.flags.DEFINE_boolean('use_rand_connectivity', False, '')
    # absl.app.flags.DEFINE_boolean('use_uniform_neuron_type', False, '')
    # absl.app.flags.DEFINE_boolean('use_only_one_type', False, '')
    # absl.app.flags.DEFINE_boolean('use_dale_law', True, '')
    # if one wants to use caching, remember to update the caching function
    absl.app.flags.DEFINE_boolean('caching', True, '')
    # a little confusing.
    absl.app.flags.DEFINE_boolean('core_only', False, '')
    # not used. should be retired.
    absl.app.flags.DEFINE_boolean('core_loss', False, '')
    # whethre you want to enforce rate loss to all neurons
    absl.app.flags.DEFINE_boolean('all_neuron_rate_loss', False, '')
    # absl.app.flags.DEFINE_boolean('train_input', True, '')
    absl.app.flags.DEFINE_boolean('train_input', False, '')
    absl.app.flags.DEFINE_boolean('train_noise', True, '')
    absl.app.flags.DEFINE_boolean('train_recurrent', True, '')
    absl.app.flags.DEFINE_boolean('train_recurrent_per_type', False, '')
    absl.app.flags.DEFINE_boolean('connected_selection', True, '')
    absl.app.flags.DEFINE_boolean('neuron_output', False, '')
    # absl.app.flags.DEFINE_boolean('localized_readout', True, '')
    # absl.app.flags.DEFINE_boolean('current_input', True, '')
    # absl.app.flags.DEFINE_boolean('use_rand_ini_w', True, '')
    # absl.app.flags.DEFINE_boolean('use_decoded_noise', True, '')
    # absl.app.flags.DEFINE_boolean('from_lgn', True, '')
    # absl.app.flags.DEFINE_boolean("float16", False, "")
    absl.app.flags.DEFINE_boolean("hard_reset", False, "")
    absl.app.flags.DEFINE_boolean("pseudo_gauss", False, "")
    absl.app.flags.DEFINE_boolean("bmtk_compat_lgn", True, "")
    absl.app.flags.DEFINE_boolean("reset_every_step", False, "")
    absl.app.flags.DEFINE_boolean("spontaneous_training", False, "")
    absl.app.flags.DEFINE_boolean('random_weights', False, '')
    absl.app.flags.DEFINE_boolean('uniform_weights', False, '')
    absl.app.flags.DEFINE_boolean("current_input", False, "")
    absl.app.flags.DEFINE_boolean("gradient_checkpointing", True, "")
    absl.app.flags.DEFINE_float("voltage_gradient_dampening", 0.5, "")
    absl.app.flags.DEFINE_boolean(
        "sequential_stimuli", True, "Run evoked and spontaneous stimuli sequentially but convergence would be slower and worse (memory friendly; intended for batch_size=1).")
    absl.app.flags.DEFINE_boolean(
        "profile_train_step",
        False,
        "Capture one real training step with TensorFlow profiler.",
    )
    absl.app.flags.DEFINE_integer(
        "profile_step",
        1,
        "Zero-based training-loop step to profile when profile_train_step is enabled.",
    )
    absl.app.flags.DEFINE_string(
        "profile_logdir",
        "",
        "Profiler output directory. Defaults to <run_logdir>/logs/profile/<timestamp>.",
    )
    absl.app.flags.DEFINE_boolean(
        "profile_stop_after_capture",
        True,
        "Exit immediately after the profiled training step is captured.",
    )
    absl.app.flags.DEFINE_integer(
        "profile_gpu_index",
        -1,
        "Optional physical GPU index to expose before creating the TensorFlow strategy.",
    )
    absl.app.flags.DEFINE_enum(
        "single_gpu_strategy",
        "mirrored",
        ["mirrored", "one_device"],
        "Distribution strategy to use when exactly one GPU is visible.",
    )
    absl.app.flags.DEFINE_boolean(
        "debug_gradients",
        False,
        "Print and assert finite gradients at every training update.",
    )

    absl.app.run(main)
