import matplotlib
matplotlib.use('agg')# to avoid GUI request on clusters
import os

# Define the environment variables for optimal GPU performance
os.environ['TF_GPU_THREAD_MODE'] = 'global'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'  # before import tensorflow
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
# os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

import absl
import re
import copy
import numpy as np
import tensorflow as tf
from time import time
import logging
from v1_model_utils import tf_utils
tf.get_logger().setLevel(logging.INFO)


def main(_):
    flags = absl.app.flags.FLAGS
    # Allow for memory growth (also to observe memory consumption)
    physical_devices = tf_utils.configure_gpu_memory_growth()
    # Display TensorFlow and CUDA runtime information for debugging and verification purposes.
    tf_utils.print_tensorflow_runtime_info()

    # Import TensorFlow-heavy project modules only after GPU memory is configured.
    from v1_model_utils import load_sparse, models
    from v1_model_utils.callbacks import OsiDsiCallbacks, printgpu
    import stim_dataset

    tf_utils.configure_reproducibility(flags.seed)
    # Configure TensorFlow optimization options that have proven to be beneficial for our model and training setup. These options can improve the performance of the model by optimizing the computational graph and memory usage.
    tf_utils.enable_tensorflow_optimizations(enabled=True)

    # Create the tensorflow datafile for the network
    os.makedirs(os.path.join(flags.data_dir, 'tf_data'), exist_ok=True)

    # Load the checkpoint model if it exists
    flag_str, logdir, current_epoch = tf_utils.configure_run_paths(
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

    # Create a deep copy of flags and override uniform_weights to False
    # This ensures initial weight distribution is preserved for visualization
    if flags.uniform_weights:
        modified_flags = copy.deepcopy(flags)
        modified_flags.uniform_weights = False
        load_fn = load_sparse.load_v1  # revert to original load function
    else:
        modified_flags = flags

    # Use modified flags for loading network
    network, lgn_input, bkg_input = load_fn(modified_flags, flags.neurons, flag_str=flag_str)
    print(f"Model files loading: {time()-t0:.2f} seconds\n")

    # Define the scope in which the model training will be executed
    with strategy.scope():
        t0 = time()

        def build_model_with_dtype(model_dtype):
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
                max_delay=5,
                current_input=flags.current_input,
                use_dummy_state_input=False,
                seed=flags.seed
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
            build_model_with_dtype,
            logdir,
            current_epoch=current_epoch,
            result_name="OSI/DSI",
        )

        # model_variables_dict['Best'] =  {var.name: var.numpy().astype(np.float16) for var in model.trainable_variables}
        model_variables_dict['Best'] =  {var.name: var.numpy() for var in model.trainable_variables} # float32 dtype
        print(f"Model variables stored in dictionary\n")

        # Build the model layers
        rsnn_layer = model.get_layer('rsnn')
        rsnn_layer.cell.refresh_recurrent_weight_shadow()
        # prediction_layer = model.get_layer('prediction')
        # abstract_layer = model.get_layer('abstract_output')
        extractor_model = tf.keras.Model(inputs=model.inputs, outputs=rsnn_layer.output)

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
            output_dtype=dtype,
        )
        # Handle the random seed for spontaneous spike generation and BKG noise
        seed_helper = tf_utils.DistributedSeedHelper(
            flags.seed,
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
            output_dtype=dtype,
            angles=DG_angles,
            strategy=strategy,
        )

    def roll_out(_x, _state_variables):
        # _initial_state = tf.nest.map_structure(lambda _a: _a.read_value(), state_variables)
        if _x.dtype == tf.bool:
            _x = tf.cast(_x, dtype)
        seed_helper.advance_noise_seed()
        _out = extractor_model((_x, _state_variables))
        z = _out[0][0]
        v = _out[0][1]
        # update state_variables with the new model state
        new_state = tuple(_out[1:])
        # tf.nest.map_structure(lambda a, b: a.assign(b), state_variables, new_state)

        return z, v, new_state

    @tf.function
    def distributed_roll_out(x, state_variables):
        z, v, new_state = strategy.run(roll_out, args=(x, state_variables))
        return z, v, new_state

    # Generate spontaneous spikes efficiently
    @tf.function
    def generate_spontaneous_spikes(spontaneous_prob):
        # random_uniform = tf.random.uniform(tf.shape(spontaneous_prob), dtype=dtype)
        random_uniform = tf.random.stateless_uniform(
            tf.shape(spontaneous_prob),
            seed=seed_helper.next_spontaneous_seed(),
            dtype=dtype,
        )
        return tf.less(random_uniform, spontaneous_prob)

    def generate_gray_state(batch_size):
        # Generate LGN spikes
        batch_size = tf.cast(batch_size, tf.int32)
        prob = tf.tile(tf.expand_dims(spontaneous_prob, axis=0), [batch_size, 1, 1])
        x = generate_spontaneous_spikes(prob)
        x = tf.cast(x, dtype)
        # Simulate the network with a gray screen
        zero_state = rsnn_layer.cell.zero_state(batch_size, dtype=dtype)
        _, _, new_state = roll_out(x, zero_state)
        return new_state

    @tf.function
    def distributed_generate_gray_state(batch_size):
        # Run generate_gray_state on each replica
        return strategy.run(generate_gray_state, args=(batch_size,))

    # Generate the gray state
    gray_state = distributed_generate_gray_state(per_replica_batch_size)
    continuing_state = gray_state

    # Define the callbacks for the OSI/DSI analysis
    stim_duration = flags.spont_duration + flags.evoked_duration
    sim_duration = int(np.ceil(stim_duration/flags.seq_len)) * flags.seq_len
    post_delay = sim_duration - stim_duration
    callbacks = OsiDsiCallbacks(network, lgn_input, bkg_input, flags, logdir, current_epoch=current_epoch,
                                pre_delay=flags.spont_duration, post_delay=post_delay, model_variables_init=model_variables_dict)

    n_iterations = int(np.ceil(flags.n_trials_per_angle / per_replica_batch_size))
    chunk_size = flags.seq_len
    num_chunks = int(np.ceil(sim_duration/chunk_size))

    # Determine which neurons to track based on track_core_only flag
    if flags.track_core_only:
        from v1_model_utils import other_v1_utils
        core_mask = other_v1_utils.isolate_core_neurons(network, radius=flags.plot_core_radius, data_dir=flags.data_dir)
        n_tracked_neurons = np.sum(core_mask)
        print(f"Tracking only core neurons: {n_tracked_neurons} out of {network['n_nodes']} total neurons")
    else:
        core_mask = None
        n_tracked_neurons = network['n_nodes']
        print(f"Tracking all neurons: {n_tracked_neurons}")

    spikes = np.zeros((flags.n_trials_per_angle, len(DG_angles), sim_duration, n_tracked_neurons), dtype=np.uint8)

    # Initialize voltage tracking arrays if requested
    if flags.track_voltage:
        voltages = np.zeros((sim_duration, n_tracked_neurons), dtype=np.float16)
        print(f"Voltage tracking enabled. Allocating {voltages.nbytes / (1024**3):.2f} GB for voltage storage.")

    for angle_id, angle in enumerate(DG_angles):
        print(f'Running angle {angle}...')
        # load LGN firing rates for the given angle and calculate spiking probability
        lgn_prob = lgn_firing_probabilities_dict[angle]
        lgn_prob = tf.tile(tf.expand_dims(lgn_prob, axis=0), [per_replica_batch_size, 1, 1])

        t0 = time()

        for iter_id in range(n_iterations):
            start_idx = iter_id * per_replica_batch_size
            end_idx = min((iter_id + 1) * per_replica_batch_size, flags.n_trials_per_angle)
            iteration_length = end_idx - start_idx

            lgn_spikes = generate_spontaneous_spikes(lgn_prob)
            # Reset the memory stats
            # tf.config.experimental.reset_memory_stats('GPU:0')
            continuing_state = distributed_generate_gray_state(per_replica_batch_size)

            for i in range(num_chunks):
                chunk = lgn_spikes[:, i * chunk_size : (i + 1) * chunk_size, :]
                v1_z_chunk, v1_v_chunk, continuing_state = distributed_roll_out(chunk, continuing_state)
                # Extract spikes based on tracking mode
                if flags.track_core_only:
                    tracked_spikes = v1_z_chunk.numpy()[:iteration_length, :, core_mask].astype(np.uint8)
                    if flags.track_voltage and iter_id == 0 and angle_id == 0:
                        tracked_voltage = v1_v_chunk.numpy()[:iteration_length, :, core_mask].astype(np.float16)
                else:
                    tracked_spikes = v1_z_chunk.numpy()[:iteration_length, :, :].astype(np.uint8)
                    if flags.track_voltage and iter_id == 0 and angle_id == 0:
                        tracked_voltage = v1_v_chunk.numpy()[:iteration_length, :, :].astype(np.float16)

                spikes[start_idx:end_idx, angle_id, i * chunk_size : (i + 1) * chunk_size, :] += tracked_spikes
                if flags.track_voltage and iter_id == 0 and angle_id == 0:
                    voltages[i * chunk_size : (i + 1) * chunk_size, :] = tracked_voltage[0, :, :]

        # Track GPU memory and inference time
        # average_rate = np.mean(spikes[start_idx:end_idx, angle_id, :, :].astype(np.float32))
        average_rate = np.mean(spikes[:, angle_id, :, :].astype(np.float32))
        callbacks.track_performance(t0, average_rate=average_rate, gpu_id=0)

        print(f'    Angle processing time: {time() - t0:.2f}s')
        for gpu_id in range(len(strategy.extended.worker_devices)):
            printgpu(gpu_id=gpu_id)

        if angle_id == 0:
            # Raster plot for 0 degree orientation
            callbacks.single_trial_callbacks(lgn_spikes.numpy(), spikes[:, 0, :, :], y=angle)
            # Save voltage traces if tracking was enabled
            if flags.track_voltage:
                voltage_path = os.path.join(logdir, 'voltage_trace.npy')
                np.save(voltage_path, voltages)
                print(f"Voltage traces saved to {voltage_path}")

        if not flags.calculate_osi_dsi:
            break

    # Save the performance metrics
    callbacks.save_inference_metrics()

    # Save voltage traces if tracking was enabled
    if flags.track_voltage:
        voltage_path = os.path.join(logdir, 'voltage_trace.npy')
        np.save(voltage_path, voltages)
        print(f"Voltage traces saved to {voltage_path}")

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
    absl.app.flags.DEFINE_string('comment', '', '')
    absl.app.flags.DEFINE_string('delays', '100,0', '')
    absl.app.flags.DEFINE_string('scale', '2,2', '')
    absl.app.flags.DEFINE_string('optimizer', 'adam', '')
    absl.app.flags.DEFINE_string('dtype', 'float32', '')
    absl.app.flags.DEFINE_float('learning_rate', .01, '')
    absl.app.flags.DEFINE_string('lr_schedule', 'warmup_cosine',
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
    absl.app.flags.DEFINE_float('rate_cost', 100., '')
    absl.app.flags.DEFINE_float('voltage_cost', 1., '')
    absl.app.flags.DEFINE_float('sync_cost', 1., '')
    absl.app.flags.DEFINE_float('osi_cost', 1., '')
    absl.app.flags.DEFINE_string('osi_loss_method', 'crowd_osi', '')
    absl.app.flags.DEFINE_float('osi_loss_subtraction_ratio', 1., '')
    absl.app.flags.DEFINE_float('dampening_factor', .5, '')
    absl.app.flags.DEFINE_float('recurrent_dampening_factor', .5, '')
    absl.app.flags.DEFINE_float('input_weight_scale', 1., '')
    absl.app.flags.DEFINE_float('gauss_std', .3, '')
    absl.app.flags.DEFINE_float('recurrent_weight_regularization', 0., '')
    absl.app.flags.DEFINE_string('recurrent_weight_regularizer_type', "mean", '')
    absl.app.flags.DEFINE_string('voltage_penalty_mode', 'range', '')
    absl.app.flags.DEFINE_float('lr_scale', 1., '')
    absl.app.flags.DEFINE_float('temporal_f', 2., '')
    absl.app.flags.DEFINE_float('max_time', -1, '')

    absl.app.flags.DEFINE_integer('n_runs', 1, '')
    absl.app.flags.DEFINE_integer('run_session', 0, '')
    absl.app.flags.DEFINE_integer('n_epochs', 50, '')
    absl.app.flags.DEFINE_integer('batch_size', 1, '')
    absl.app.flags.DEFINE_integer('neurons', 0, '')  # 0 to take all neurons
    absl.app.flags.DEFINE_integer('steps_per_epoch', 20, '')# EA and garret dose not need this many but pure classification needs 781 = int(50000/64)
    absl.app.flags.DEFINE_integer('val_steps', 1, '')# EA and garret dose not need this many but pure classification needs 156 = int(10000/64)
    # number of LGN filters in visual space (input population)
    absl.app.flags.DEFINE_integer('n_input', 17400, '')
    absl.app.flags.DEFINE_integer("n_output", 2, "")
    absl.app.flags.DEFINE_integer('seq_len', 600, '')
    absl.app.flags.DEFINE_integer('spont_duration', 2000, '')
    absl.app.flags.DEFINE_integer('evoked_duration', 2000, '')
    absl.app.flags.DEFINE_integer('seed', 3000, '')
    absl.app.flags.DEFINE_integer('neurons_per_output', 16, '')

    absl.app.flags.DEFINE_integer('n_cues', 3, '')
    absl.app.flags.DEFINE_integer('recall_duration', 40, '')
    absl.app.flags.DEFINE_integer('cue_duration', 40, '')
    absl.app.flags.DEFINE_integer('interval_duration', 40, '')
    absl.app.flags.DEFINE_integer('examples_in_epoch', 32, '')
    absl.app.flags.DEFINE_integer('validation_examples', 16, '')
    absl.app.flags.DEFINE_integer('n_trials_per_angle', 10, '')
    absl.app.flags.DEFINE_integer('fano_samples', 500, '')

    # absl.app.flags.DEFINE_boolean('float16', False, '')
    absl.app.flags.DEFINE_boolean('caching', True, '')
    absl.app.flags.DEFINE_boolean('core_only', False, '')
    absl.app.flags.DEFINE_boolean('core_loss', False, '')
    absl.app.flags.DEFINE_boolean('all_neuron_rate_loss', False, '')  # whethre you want to enforce rate loss to all neurons
    absl.app.flags.DEFINE_float('loss_core_radius', 400.0, '') # 0 is not using core loss
    absl.app.flags.DEFINE_float('plot_core_radius', 400.0, '') # 0 is not using core plot

    absl.app.flags.DEFINE_boolean('hard_reset', False, '')
    absl.app.flags.DEFINE_boolean('train_input', False, '')
    absl.app.flags.DEFINE_boolean('train_noise', False, '')
    absl.app.flags.DEFINE_boolean('train_recurrent', True, '')
    absl.app.flags.DEFINE_boolean('train_recurrent_per_type', False, '')
    # absl.app.flags.DEFINE_boolean('train_recurrent_per_type', False, '')
    absl.app.flags.DEFINE_boolean('connected_selection', True, '')
    absl.app.flags.DEFINE_boolean('neuron_output', False, '')
    # absl.app.flags.DEFINE_boolean('hard_only', False, '')
    absl.app.flags.DEFINE_boolean('pseudo_gauss', False, '')
    absl.app.flags.DEFINE_boolean("bmtk_compat_lgn", True, "")
    absl.app.flags.DEFINE_boolean("average_grad_for_cell_type", False, "")
    absl.app.flags.DEFINE_boolean("reset_every_step", False, "")
    absl.app.flags.DEFINE_boolean("spontaneous_training", False, "")
    absl.app.flags.DEFINE_boolean("calculate_osi_dsi", True, "")
    absl.app.flags.DEFINE_boolean('random_weights', False, '')
    absl.app.flags.DEFINE_boolean('uniform_weights', False, '')
    absl.app.flags.DEFINE_boolean('restore_runtime_dtype_cast', True, 'Enable in-memory checkpoint dtype conversion when checkpoint and requested model dtype differ.')
    absl.app.flags.DEFINE_boolean("current_input", False, "")
    absl.app.flags.DEFINE_boolean("gradient_checkpointing", True, "")
    absl.app.flags.DEFINE_boolean("track_core_only", False, "Track spikes only from core neurons to reduce memory usage")
    absl.app.flags.DEFINE_boolean("track_voltage", False, "Track and save membrane voltage traces during simulation")
    absl.app.flags.DEFINE_float("voltage_gradient_dampening", 0.5, "")

    absl.app.flags.DEFINE_string("rotation", "ccw", "")
    absl.app.flags.DEFINE_string('ckpt_dir', '', '')
    absl.app.flags.DEFINE_string('neuropixels_df', 'Neuropixels_data/v1_OSI_DSI_DF.csv', 'File name of the Neuropixels DataFrame for OSI/DSI analysis.')

    absl.app.run(main)
