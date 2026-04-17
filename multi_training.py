import matplotlib
matplotlib.use('agg')# to avoid GUI request on clusters
import os

# Define the environment variables for optimal GPU performance
os.environ['TF_GPU_THREAD_MODE'] = 'global'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'  # before import tensorflow
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
# os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

import absl
import socket
import re
import copy
import numpy as np
import tensorflow as tf
import pickle as pkl
from time import time
import logging
from v1_model_utils import tf_utils
tf.get_logger().setLevel(logging.INFO)
# logging.getLogger().setLevel(logging.INFO)


def main(_):
    flags = absl.app.flags.FLAGS
    # Allow for memory growth (also to observe memory consumption)
    physical_devices = tf_utils.configure_gpu_memory_growth()
    # Display TensorFlow and CUDA runtime information for debugging and verification purposes.
    tf_utils.print_tensorflow_runtime_info()

    # Import TensorFlow-heavy project modules only after GPU memory is configured.
    import stim_dataset
    from v1_model_utils.callbacks import Callbacks
    import v1_model_utils.loss_functions as losses
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
        single_gpu_strategy="mirrored",
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
            use_dummy_state_input=False
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
        print(f"Model variables stored in dictionary\n")

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
        rsnn_layer.cell.refresh_recurrent_weight_shadow()
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
        if os.path.exists(os.path.join(logdir, 'train_end_data.pkl')):
            with open(os.path.join(logdir, 'train_end_data.pkl'), 'rb') as f:
                data_loaded = pkl.load(f)
                v1_ema = tf.Variable(data_loaded.get("v1_ema", 0.003), trainable=False, name='V1_EMA')
        else:
            # 3 Hz is near the average FR of cortex
            v1_ema = tf.Variable(tf.constant(0.003, shape=(
                network["n_nodes"],), dtype=tf.float32), trainable=False, name='V1_EMA')

        OSI_DSI_Loss = losses.OrientationSelectivityLoss(network=network, osi_cost=flags.osi_cost,
                                                         pre_delay=delays[0], post_delay=delays[1],
                                                         dtype=tf.float32, core_mask=core_mask,
                                                         method=flags.osi_loss_method,
                                                         subtraction_ratio=flags.osi_loss_subtraction_ratio,
                                                         neuropixels_df=flags.neuropixels_df,
                                                         data_dir=flags.data_dir)
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
                                                                     data_dir=flags.data_dir)
            # placeholder_angle = tf.constant(0, dtype=tf.float32, shape=(per_replica_batch_size, 1))
            # model.add_loss(lambda: annulus_OSI_DSI_Loss(rsnn_layer.output[0][0], placeholder_angle, trim=True, normalizer=v1_ema))

        extractor_model = tf.keras.Model(inputs=model.inputs, outputs=rsnn_layer.output)
        #  outputs=[rsnn_layer.output, model.output, prediction_layer.output])
        # State-only model to avoid storing full sequences when only the final state is needed.
        try:
            state_model = models.build_state_only_model(model, rsnn_layer)
        except Exception as e:
            state_model = None
            print(f"Warning: failed to build state-only model ({e}); using full model for gray state.")

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
            training_utils.reset_metrics(
                (
                    val_loss,
                    val_firing_rate,
                    val_rate_loss,
                    val_voltage_loss,
                    val_regularizer_loss,
                    val_osi_dsi_loss,
                    val_sync_loss,
                )
            )

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

    def _compute_losses_from_activity(_z, _v, y, spontaneous, trim, regularizers_loss):

        # keep final scalar aggregation in float32
        voltage_loss = tf.cast(voltage_regularizer(_v), tf.float32)
        spontaneous = tf.cast(spontaneous, tf.bool)

        def _evoked_losses():
            rate_loss = evoked_rate_regularizer(_z, trim)
            # update the exponential moving average of the firing rates over drifting gratings presentation
            v1_evoked_rates = tf.reduce_mean(
                tf.cast(_z[:, delays[0]: flags.seq_len - delays[1], :], tf.float32), (0, 1)
            )
            v1_ema.assign(ema_decay * v1_ema + (1 - ema_decay) * v1_evoked_rates)
            osi_dsi_loss = OSI_DSI_Loss(_z, y, trim, normalizer=v1_ema)
            sync_loss = evoked_sync_loss(_z, trim)

            if annulus_mask is not None:
                rate_loss += annulus_evoked_rate_regularizer(_z, trim)
                osi_dsi_loss += annulus_OSI_DSI_Loss(_z, y, trim, normalizer=v1_ema)

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

    def roll_out(x, y, initial_state, spontaneous=False, trim=True):

        # _initial_state = tf.nest.map_structure(lambda _a: _a.read_value(), initial_state)
        _out = run_extractor(x, initial_state)
        _z, _v = _out[0]

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
            _z, _v, y, spontaneous, trim, regularizers_loss
        )

        return _out, _loss, _aux

    def roll_out_combined(x, y, x_spontaneous, initial_state, trim=True):

        x_concat = tf.concat([x, x_spontaneous], axis=0)

        _out = run_extractor(x_concat, initial_state)
        _z_full, _v_full = _out[0]

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
            _z_evoked, _v_evoked, y, False, trim, regularizers_loss
        )
        spont_loss, spont_aux = _compute_losses_from_activity(
            _z_spont, _v_spont, y, True, trim, regularizers_loss
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
        # Optional debug print; very expensive, keep off in multi-GPU
        if flags.debug_gradients:
            valid_grad_abs = [tf.reshape(tf.abs(g), [-1]) for g in grad if g is not None]
            if valid_grad_abs:
                grad_max = tf.reduce_max(
                    tf.stack([tf.reduce_max(g_abs) for g_abs in valid_grad_abs], axis=0)
                )
                grad_sum = tf.add_n([tf.reduce_sum(g_abs) for g_abs in valid_grad_abs])
                grad_count = tf.add_n(
                    [tf.cast(tf.size(g_abs), tf.float32) for g_abs in valid_grad_abs]
                )
                tf.print(
                    "[Combined] Grad max:",
                    grad_max,
                    " mean:",
                    grad_sum / grad_count,
                )

        optimizer.apply_gradients(zip(grad, model.trainable_variables))
        rsnn_layer.cell.refresh_recurrent_weight_shadow()

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
        rate = tf.reduce_mean(tf.cast(_out[0][0], tf.float32))
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

        optimizer.apply_gradients(zip(grad, model.trainable_variables))
        rsnn_layer.cell.refresh_recurrent_weight_shadow()

        train_loss.update_state(
            _loss * strategy.num_replicas_in_sync, sample_weight=metric_weight
        )
        rate = tf.reduce_mean(tf.cast(_out[0][0], tf.float32))
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
                v1_spikes_evoked = strategy.experimental_local_results(out_evoked)[0][0][0]
                v1_spikes_spont = strategy.experimental_local_results(out_spont)[0][0][0]
            else:
                _loss, _aux, _out = distributed_train_step(
                    x, y, x_spontaneous, state_variables, trim, return_sequences=True
                )
                v1_spikes_full = strategy.experimental_local_results(_out)[0][0][0]
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

    def validation_step(x, y, state_variables, x_spontaneous, output_spikes=True):

        if flags.sequential_stimuli:
            _aux = {}
            _out_evoked, _loss_evoked, _aux_evoked = roll_out(x, y, state_variables, spontaneous=False)
            _out_spont, _loss_spont, _aux_spont = roll_out(x_spontaneous, y, state_variables, spontaneous=True)
            _z_evoked = _out_evoked[0][0]
            _z_spont = _out_spont[0][0]
        else:
            _out, _loss_evoked, _loss_spont, _aux_evoked, _aux_spont  = roll_out_combined(x, y, x_spontaneous, state_variables)
            _z_evoked = _out[0][0][:per_replica_batch_size]
            _z_spont = _out[0][0][per_replica_batch_size:]

        _loss = (_loss_evoked + _loss_spont) / 2.0
        _aux = {k: (_aux_evoked[k] + _aux_spont[k]) / 2.0 for k in ["rate_loss", "voltage_loss",  "sync_loss", "regularizer_loss"]}
        _aux["osi_dsi_loss"] = _aux_evoked["osi_dsi_loss"]  # only evoked contributes to OSI/DSI loss

        val_loss.update_state(_loss * strategy.num_replicas_in_sync)
        rate = (tf.reduce_mean(tf.cast(_z_evoked, tf.float32)) + tf.reduce_mean(tf.cast(_z_spont, tf.float32))) / 2.0
        val_firing_rate.update_state(rate)
        val_rate_loss.update_state(_aux['rate_loss'])
        val_voltage_loss.update_state(_aux['voltage_loss'])
        val_regularizer_loss.update_state(_aux['regularizer_loss'])
        val_osi_dsi_loss.update_state(_aux['osi_dsi_loss'])
        val_sync_loss.update_state(_aux['sync_loss'])

        # tf.nest.map_structure(lambda _a, _b: _a.assign(_b), list(state_variables), _out[1:])
        if output_spikes:
            return _z_evoked, _z_spont

    @tf.function
    def distributed_validation_step(x, y, state_variables, x_spontaneous, output_spikes=True):
        return strategy.run(
            validation_step,
            args=(x, y, state_variables, x_spontaneous, output_spikes),
        )

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

    def generate_spontaneous_spikes(batch_size):
        batch_size = tf.cast(batch_size, tf.int32)
        # Stop gradient for shape operations to avoid int32 dtype warnings
        prob_shape = tf.stop_gradient(tf.shape(spontaneous_prob_base))
        target_shape = tf.concat([[batch_size], prob_shape], axis=0)
        spontaneous_prob = tf.broadcast_to(spontaneous_prob_base, target_shape)
        random_uniform = tf.random.stateless_uniform(
            tf.shape(spontaneous_prob),
            seed=seed_helper.next_spontaneous_seed(),
            dtype=dtype,
        )
        return tf.less(random_uniform, spontaneous_prob)

    @tf.function
    def distributed_generate_spontaneous_spikes(batch_size):
        return strategy.run(generate_spontaneous_spikes, args=(batch_size,))

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
        else:
            _out = run_extractor(x, init_state)
            return _out[1:]

    @tf.function
    def distributed_generate_gray_state(batch_size):
        # Run generate_gray_state on each replica
        return strategy.run(generate_gray_state, args=(batch_size,))

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

    callbacks.on_train_begin()
    # chunknum = 1
    # max_working_fr = {}   # defined for each chunknum
    n_prev_epochs = flags.run_session * flags.n_epochs  # used for resuming training and logging correct epoch numbers in that case

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
            x_spontaneous = distributed_generate_spontaneous_spikes(per_replica_batch_size)

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

            # # update max working fr for the chunk num
            # current_fr = step_values[2].numpy()
            # if chunknum not in max_working_fr:
            #     max_working_fr[chunknum] = current_fr
            # else:
            #     max_working_fr[chunknum] = max(max_working_fr[chunknum], current_fr)
            # # determine if the chunknum should be decreased
            # if chunknum > 1:
            #     chunknum_down = get_next_chunknum(chunknum, flags.seq_len, direction='down')
            #     if chunknum_down in max_working_fr:
            #         if current_fr < max_working_fr[chunknum_down]:
            #             chunknum = chunknum_down
            #             # Clear the session to reset the graph state
            #             tf.keras.backend.clear_session()
            #             print("Decreasing chunknum to: ", chunknum)
            #             print(current_fr, max_working_fr)
            #             print(max_working_fr)
            #     else:  # data not available, estimate from the current one.
            #         fr_ratio = current_fr / max_working_fr[chunknum]
            #         chunknum_ratio = chunknum_down / chunknum
            #         print(current_fr, max_working_fr, fr_ratio, chunknum_ratio)
            #         if fr_ratio < chunknum_ratio:  # potentially good to decrease
            #             chunknum = chunknum_down
            #             # Clear the session to reset the graph state
            #             tf.keras.backend.clear_session()
            #             print("Tentatively decreasing chunknum to: ", chunknum)

            # Stop profiler after profiling steps
            # if step == profile_end_step:
            #     tf.profiler.experimental.stop()

            callbacks.on_step_end(step_values, y, verbose=True)

        # tf.profiler.experimental.stop()

        ### VALIDATION AFTER EACH EPOCH
        v1_spikes, v1_spikes_spont = distributed_validation_step(
            x,
            y,
            gray_state,
            x_spontaneous,
            output_spikes=True
        )
        # get the first replica of the training spikes
        v1_spikes = strategy.experimental_local_results(v1_spikes)[0]
        v1_spikes_spont = strategy.experimental_local_results(v1_spikes_spont)[0]
        if strategy.num_replicas_in_sync > 1:
            x = strategy.experimental_local_results(x)[0]
            y = strategy.experimental_local_results(y)[0]

        train_values = [a.result().numpy() for a in [train_loss, train_firing_rate,
                                                     train_rate_loss, train_voltage_loss, train_regularizer_loss,
                                                     train_osi_dsi_loss, train_sync_loss]]
        # val_values = train_values
        val_values = [a.result().numpy() for a in [val_loss, val_firing_rate,
                                                   val_rate_loss, val_voltage_loss,
                                                   val_regularizer_loss,
                                                   val_osi_dsi_loss, val_sync_loss]]
        metric_values = train_values + val_values

        stop = callbacks.on_epoch_end(x, v1_spikes, y, metric_values, verbose=True,
                                      x_spont=x_spontaneous, v1_spikes_spont=v1_spikes_spont)

        if stop:
            break

        # Reset the metrics for the next epoch
        reset_train_metrics()
        reset_validation_metrics()

    normalizers = {'v1_ema': v1_ema.numpy()}
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
    absl.app.flags.DEFINE_float('osi_loss_subtraction_ratio', 0., '')
    absl.app.flags.DEFINE_float('dampening_factor', .1, '')
    absl.app.flags.DEFINE_float("recurrent_dampening_factor", .1, "")
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
    # New: enable/disable very expensive gradient logging
    absl.app.flags.DEFINE_boolean("debug_gradients", False, "")

    absl.app.run(main)
