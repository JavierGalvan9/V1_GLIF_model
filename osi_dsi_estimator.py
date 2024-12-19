import matplotlib
matplotlib.use('agg')# to avoid GUI request on clusters
import os

# Define the environment variables for optimal GPU performance
os.environ['TF_GPU_THREAD_MODE'] = 'global'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'  # before import tensorflow
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
# os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

import absl
import numpy as np
import tensorflow as tf
import pickle as pkl
from packaging import version

if version.parse(tf.__version__) < version.parse("2.4.0"):
    from tensorflow.keras.mixed_precision import experimental as mixed_precision
else:
    from tensorflow.keras import mixed_precision

from v1_model_utils import load_sparse, models, toolkit
from v1_model_utils.callbacks import OsiDsiCallbacks, printgpu
import stim_dataset
from time import time
import ctypes.util
import random


print("--- CUDA version: ", tf.sysconfig.get_build_info()["cuda_version"])
print("--- CUDNN version: ", tf.sysconfig.get_build_info()["cudnn_version"])
print("--- TensorFlow version: ", tf.__version__)

# For CUDA Runtime API
lib_path = ctypes.util.find_library('cudart')
print("--- CUDA Library path: ", lib_path)


def main(_):
    # Allow for memory growth (also to observe memory consumption)
    physical_devices = tf.config.list_physical_devices("GPU")
    for dev in physical_devices:
        try:
            tf.config.experimental.set_memory_growth(dev, True)
            print(f"Memory growth enabled for device {dev}")
        except:
            print(f"Invalid device {dev} or cannot modify virtual devices once initialized.")
            pass
    print("- Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')), '\n')

    flags = absl.app.flags.FLAGS
    # Set the seeds for reproducibility
    np.random.seed(flags.seed)
    tf.random.set_seed(flags.seed)
    random.seed(flags.seed)

    # Load the checkpoint model if it exists
    logdir = flags.ckpt_dir
    if logdir == '':
        flag_str = f'v1_{flags.neurons}'
        for name, value in flags.flag_values_dict().items():
            if value != flags[name].default and name in ['n_input', 'core_only', 'connected_selection', 'data_dir', 'random_weights']:
                flag_str += f'_{name}_{value}'
        # Define flag string as the second part of results_path
        results_dir = f'{flags.results_dir}/{flag_str}'
        os.makedirs(results_dir, exist_ok=True)
        print('Simulation results path: ', results_dir)
        # Generate a ticker for the current simulation
        sim_name = toolkit.get_random_identifier('b_')
        logdir = os.path.join(results_dir, sim_name)
        print(f'> Results for {flags.task_name} will be stored in:\n {logdir} \n')
        current_epoch = 0
    else:
        flag_str = logdir.split(os.path.sep)[-2]
         # Load current epoch from the file
        if os.path.exists(os.path.join(logdir, 'train_end_data.pkl')):
            with open(os.path.join(logdir, 'train_end_data.pkl'), 'rb') as f:
                data_loaded = pkl.load(f)
            current_epoch = len(data_loaded['epoch_metric_values']['train_loss'])
        else:
            current_epoch = (flags.run_session + 1) * flags.n_epochs
        
    # Can be used to try half precision training
    if flags.dtype=='float16':
        if version.parse(tf.__version__) < version.parse("2.4.0"):
            policy = mixed_precision.Policy("mixed_float16")
            mixed_precision.set_policy(policy)
        else:
            mixed_precision.set_global_policy('mixed_float16')
        dtype = tf.float16
        print('Mixed precision (float16) enabled!')
    elif flags.dtype=='bfloat16':
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
    # strategy = tf.distribute.MirroredStrategy(cross_device_ops=tf.distribute.ReductionToOneDevice(reduce_to_device="cpu:0")) 
    # device = "/gpu:0" if tf.config.list_physical_devices("GPU") else "/cpu:0"
    # strategy = tf.distribute.OneDeviceStrategy(device=device)
    strategy = tf.distribute.MirroredStrategy()

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

    # Define the scope in which the model training will be executed
    with strategy.scope():
        t0 = time()
        # Build the model
        model = models.create_model(
            network, 
            lgn_input, 
            bkg_input, 
            seq_len=flags.seq_len,
            n_input=flags.n_input, 
            n_output=flags.n_output,
            cue_duration=flags.cue_duration,
            dtype=dtype, 
            input_weight_scale=flags.input_weight_scale,
            dampening_factor=flags.dampening_factor, 
            recurrent_dampening_factor=flags.recurrent_dampening_factor,
            gauss_std=flags.gauss_std, 
            lr_scale=flags.lr_scale,
            train_input=flags.train_input,
            train_noise=flags.train_noise,
            train_recurrent=flags.train_recurrent,
            train_recurrent_per_type=flags.train_recurrent_per_type,
            neuron_output=flags.neuron_output,
            batch_size=per_replica_batch_size, 
            pseudo_gauss=flags.pseudo_gauss, 
            use_state_input=True, 
            return_state=True,
            hard_reset=flags.hard_reset,
            add_metric=False,
            max_delay=5, 
            current_input=flags.current_input
            )

        # del lgn_input, bkg_input

        model.build((per_replica_batch_size, flags.seq_len, flags.n_input))
        print(f"Model built in {time()-t0:.2f} s\n")

        # Store the initial model variables that are going to be trained
        # model_variables_dict = {'Initial': {var.name: var.numpy().astype(np.float16) for var in model.trainable_variables}}
        model_variables_dict = {'Initial': {var.name: var.numpy().astype(np.float16) for var in model.trainable_variables}}

        # # Define the optimizer
        # if flags.optimizer == 'adam':
        #     optimizer = tf.keras.optimizers.Adam(flags.learning_rate, epsilon=1e-11)  
        # elif flags.optimizer == 'sgd':
        #     optimizer = tf.keras.optimizers.SGD(flags.learning_rate, momentum=0.0, nesterov=False)
        # else:
        #     print(f"Invalid optimizer: {flags.optimizer}")
        #     raise ValueError

        # optimizer.build(model.trainable_variables)

        # Restore model and optimizer from a checkpoint if it exists
        if flags.ckpt_dir != '' and os.path.exists(os.path.join(flags.ckpt_dir, flags.restore_from)):
            checkpoint_directory = tf.train.latest_checkpoint(os.path.join(flags.ckpt_dir, flags.restore_from))
            print(f'Restoring checkpoint from {checkpoint_directory}...')
            checkpoint = tf.train.Checkpoint(model=model)
            checkpoint.restore(checkpoint_directory).expect_partial()
            if flags.restore_from == "Best_model":
                logdir = checkpoint_directory + "_results"
                current_epoch_file = os.path.basename(checkpoint_directory)
                # Extract the number if a match is found
                match = re.search(r"ckpt-(\d+)", current_epoch_file)
                if match:
                    current_epoch = int(match.group(1))
            print('Checkpoint restored!')
            print(f'OSI/DSI results for epoch {current_epoch} will be saved in: {logdir}\n')
        elif flags.restore_from != '' and os.path.exists(flags.restore_from):
            checkpoint_directory = tf.train.latest_checkpoint(flags.restore_from)
            print(f'Restoring checkpoint from {checkpoint_directory}...')
            checkpoint = tf.train.Checkpoint(model=model)
            checkpoint.restore(checkpoint_directory).expect_partial() #.assert_consumed()
            print('Checkpoint restored!')
            print(f'OSI/DSI results for epoch {current_epoch} will be saved in: {logdir}\n')
        else:
            print(f"No checkpoint found in {flags.ckpt_dir} or {flags.restore_from}. Starting from scratch...\n")

        # model_variables_dict['Best'] =  {var.name: var.numpy().astype(np.float16) for var in model.trainable_variables}
        model_variables_dict['Best'] =  {var.name: var.numpy().astype(np.float16) for var in model.trainable_variables}
        print(f"Model variables stored in dictionary\n")

        # Build the model layers
        rsnn_layer = model.get_layer('rsnn')
        # prediction_layer = model.get_layer('prediction')
        # abstract_layer = model.get_layer('abstract_output')
        extractor_model = tf.keras.Model(inputs=model.inputs,
                                         outputs=rsnn_layer.output)

        zero_state = rsnn_layer.cell.zero_state(per_replica_batch_size, dtype=dtype)
        # state_variables = tf.nest.map_structure(lambda a: tf.Variable(
        #     a, trainable=False, synchronization=tf.VariableSynchronization.ON_READ
        # ), zero_state)

        # Precompute spontaneous LGN firing rates once
        def compute_spontaneous_lgn_firing_rates():
            cache_dir = "lgn_model/.cache_lgn"
            cache_file = os.path.join(cache_dir, f"spontaneous_lgn_probabilities_n_input_{flags.n_input}_seqlen_{flags.seq_len}.pkl")
            if os.path.exists(cache_file):
                with open(cache_file, "rb") as f:
                    spontaneous_prob = pkl.load(f)
                print("Loaded cached spontaneous LGN firing rates.")
            else:
                # Compute and cache the spontaneous firing rates
                spontaneous_lgn_firing_rates = stim_dataset.generate_drifting_grating_tuning(
                    seq_len=flags.seq_len,
                    pre_delay=flags.seq_len,
                    post_delay=0,
                    n_input=flags.n_input,
                    data_dir=flags.data_dir,
                    rotation=flags.rotation,
                    return_firing_rates=True,
                    dtype=dtype
                )
                spontaneous_lgn_firing_rates = next(iter(spontaneous_lgn_firing_rates))
                spontaneous_prob = 1 - tf.exp(-tf.cast(spontaneous_lgn_firing_rates, dtype) / 1000.)
                # Save to cache
                os.makedirs(cache_dir, exist_ok=True)
                with open(cache_file, "wb") as f:
                    pkl.dump(spontaneous_prob, f)
                print("Computed and cached spontaneous LGN firing rates.")
            
            # repeat the spontaneous firing rates with shape (seqlen, n_input) to match the batch size 
            spontaneous_prob = tf.tile(tf.expand_dims(spontaneous_prob, axis=0), [per_replica_batch_size, 1, 1])

            return tf.cast(spontaneous_prob, dtype=dtype)

        # Load the spontaneous probabilities once
        spontaneous_prob = compute_spontaneous_lgn_firing_rates()

        # LGN firing rates to the different angles
        DG_angles = np.arange(0, 360, 45)
        # Precompute the OSI/DSI LGN firing rates dataset
        def compute_osi_dsi_lgn_firing_rates():
            cache_dir = "lgn_model/.cache_lgn"
            post_delay = flags.seq_len - (2500 % flags.seq_len) if flags.seq_len < 2500 else 0
            osi_seq_len = 2500 + post_delay
            osi_dataset_path = os.path.join(cache_dir, f"osi_dsi_lgn_probabilities_n_input_{flags.n_input}_seqlen_{osi_seq_len}.pkl")

            if os.path.exists(osi_dataset_path):
                # Load cached OSI/DSI LGN firing rates dataset
                with open(osi_dataset_path, "rb") as f:
                    lgn_firing_probabilities_dict = pkl.load(f)
                print("Loaded cached OSI/DSI LGN firing rates dataset.")
            else:
                # Create and cache the OSI/DSI LGN firing rates dataset
                print("Creating OSI/DSI dataset...")
                lgn_firing_rates = stim_dataset.generate_drifting_grating_tuning(
                    seq_len=osi_seq_len,
                    pre_delay=500,
                    post_delay=post_delay,
                    n_input=flags.n_input,
                    regular=True,
                    return_firing_rates=True,
                    rotation=flags.rotation,
                    dtype=dtype
                    )

                # Distribute and generate the dataset
                osi_dsi_data_set = iter(lgn_firing_rates)
                lgn_firing_probabilities_dict = {}
                # Compute and store LGN firing rates for each angle
                for angle in DG_angles:
                    t0 = time()
                    angle_lgn_firing_rates = next(osi_dsi_data_set)
                    lgn_prob = 1 - tf.exp(-tf.cast(angle_lgn_firing_rates, dtype) / 1000.)
                    lgn_firing_probabilities_dict[angle] = lgn_prob
                    print(f"Angle {angle} done.")
                    print(f"    LGN running time: {time() - t0:.2f}s")
                    for gpu_id in range(len(strategy.extended.worker_devices)):
                        printgpu(gpu_id=gpu_id)

                # Save the OSI/DSI dataset to cache
                os.makedirs(cache_dir, exist_ok=True)
                with open(osi_dataset_path, "wb") as f:
                    pkl.dump(lgn_firing_probabilities_dict, f)
                print("OSI/DSI dataset created and cached successfully!")

            return lgn_firing_probabilities_dict

        # Load the OSI/DSI LGN firing probabilites dataset once
        lgn_firing_probabilities_dict = compute_osi_dsi_lgn_firing_rates()

    def roll_out(_x, _state_variables):
        # _initial_state = tf.nest.map_structure(lambda _a: _a.read_value(), state_variables)
        _initial_state = _state_variables
        seq_len = tf.shape(_x)[1]
        dummy_zeros = tf.zeros((per_replica_batch_size, seq_len, flags.neurons), dtype)
        _out = extractor_model((_x, dummy_zeros, _initial_state))
        z = _out[0][0]
        # update state_variables with the new model state
        new_state = tuple(_out[1:])
        # tf.nest.map_structure(lambda a, b: a.assign(b), state_variables, new_state)

        return z, new_state

    @tf.function
    def distributed_roll_out(x, state_variables):
        _z, new_state = strategy.run(roll_out, args=(x, state_variables))
        return _z, new_state

    # Generate spontaneous spikes efficiently
    @tf.function
    def generate_spontaneous_spikes(spontaneous_prob):
        random_uniform = tf.random.uniform(tf.shape(spontaneous_prob), dtype=dtype)
        return tf.less(random_uniform, spontaneous_prob)
            
    def generate_gray_state(spontaneous_prob):
        # Generate LGN spikes
        x = generate_spontaneous_spikes(spontaneous_prob)
        # Simulate the network with a gray screen   
        _, new_state = roll_out(x, zero_state)
        return new_state
    
    @tf.function
    def distributed_generate_gray_state(spontaneous_prob):
        # Run generate_gray_state on each replica
        return strategy.run(generate_gray_state, args=(spontaneous_prob,))
    
    # Generate the gray state
    gray_state = distributed_generate_gray_state(spontaneous_prob)
    continuing_state = gray_state
    
    # # LGN firing rates to the different angles
    # DG_angles = np.arange(0, 360, 45)
    # osi_dataset_path = os.path.join('OSI_DSI_dataset', 'lgn_firing_rates.pkl')
    # if not os.path.exists(osi_dataset_path):
    #     print('Creating OSI/DSI dataset...')
    #     # Define OSI/DSI dataset
    #     def get_osi_dsi_dataset_fn(regular=False):
    #         def _f(input_context):
    #             post_delay = flags.seq_len - (2500 % flags.seq_len)
    #             _lgn_firing_rates = stim_dataset.generate_drifting_grating_tuning(
    #                 seq_len=2500+post_delay,
    #                 pre_delay=500,
    #                 post_delay = post_delay,
    #                 n_input=flags.n_input,
    #                 regular=regular,
    #                 return_firing_rates=True,
    #                 rotation=flags.rotation,
    #                 dtype=dtype
    #             ).batch(1)
                            
    #             return _lgn_firing_rates
    #         return _f
    
    #     osi_dsi_data_set = strategy.distribute_datasets_from_function(get_osi_dsi_dataset_fn(regular=True))
    #     test_it = iter(osi_dsi_data_set)
    #     lgn_firing_rates_dict = {}  # Dictionary to store firing rates
    #     for angle_id, angle in enumerate(DG_angles):
    #         t0 = time()
    #         lgn_firing_rates = next(test_it)
    #         lgn_firing_rates_dict[angle] = lgn_firing_rates.numpy()
    #         print(f'Angle {angle} done.')
    #         print(f'    Trial running time: {time() - t0:.2f}s')
    #         for gpu_id in range(len(strategy.extended.worker_devices)):
    #             printgpu(gpu_id=gpu_id)

    #     # Save the dataset      
    #     results_dir = os.path.join("OSI_DSI_dataset")
    #     os.makedirs(results_dir, exist_ok=True)
    #     with open(osi_dataset_path, 'wb') as f:
    #         pkl.dump(lgn_firing_rates_dict, f)
    #     print('OSI/DSI dataset created successfully!')
    # else:
    #     # Load the LGN firing rates dataset
    #     with open(osi_dataset_path, 'rb') as f:
    #         lgn_firing_rates_dict = pkl.load(f)

    # # Define the dataset for the gray simulation
    # def get_gray_dataset_fn():
    #     def _f(input_context):
    #         _lgn_firing_rates = stim_dataset.generate_drifting_grating_tuning(
    #             seq_len=flags.seq_len,
    #             pre_delay=flags.seq_len,
    #             post_delay=0,
    #             n_input=flags.n_input,
    #             rotation=flags.rotation,
    #             billeh_phase=True,
    #             return_firing_rates=True,
    #             dtype=dtype
    #         ).batch(per_replica_batch_size)
                        
    #         return _lgn_firing_rates
    #     return _f

    # # We define the dataset generates function under the strategy scope for a randomly selected orientation       
    # # test_data_set = strategy.distribute_datasets_from_function(get_dataset_fn(regular=True))   
    # gray_data_set = strategy.distribute_datasets_from_function(get_gray_dataset_fn())
    # gray_it = iter(gray_data_set)
    # spontaneous_lgn_firing_rates = next(iter(gray_data_set))   
    # spontaneous_lgn_firing_rates = tf.constant(spontaneous_lgn_firing_rates, dtype=dtype)
    # # load LGN spontaneous firing rates 
    # spontaneous_prob = 1 - tf.exp(-spontaneous_lgn_firing_rates / 1000.)
    # # Generate LGN spikes
    # x = tf.random.uniform(tf.shape(spontaneous_prob), dtype=dtype) < spontaneous_prob
    # # Run a gray simulation to get the model state
    # tf.nest.map_structure(lambda a, b: a.assign(b), state_variables, zero_state)    
    # _,  gray_state = distributed_roll_out(x)
    # del gray_data_set, gray_it, spontaneous_lgn_firing_rates, spontaneous_prob, x

    # def reset_state(reset_type='zero', new_state=None):
    #     if reset_type == 'zero':
    #         tf.nest.map_structure(lambda a, b: a.assign(b), state_variables, zero_state)
    #     elif reset_type == 'gray':
    #         # Run a gray simulation to get the model state
    #         tf.nest.map_structure(lambda a, b: a.assign(b), state_variables, new_state)
    #     elif reset_type == 'continue':
    #         # Continue on the previous model state
    #         # No action needed, as the state_variables will not be modified
    #         pass
    #     else:
    #         raise ValueError(f"Invalid reset_type: {reset_type}")

    # # @tf.function
    # def distributed_reset_state(reset_type, gray_state=None):
    #     if reset_type == 'gray':
    #         strategy.run(reset_state, args=(reset_type, gray_state))
    #     else:
    #         strategy.run(reset_state, args=(reset_type, zero_state))

    # Define the callbacks for the OSI/DSI analysis
    sim_duration = int(np.ceil(2500/flags.seq_len)) * flags.seq_len
    post_delay = sim_duration - 2500
    callbacks = OsiDsiCallbacks(network, lgn_input, bkg_input, flags, logdir, current_epoch=current_epoch,
                                pre_delay=500, post_delay=post_delay, model_variables_init=model_variables_dict)

    n_iterations = int(np.ceil(flags.n_trials_per_angle / per_replica_batch_size))
    chunk_size = flags.seq_len
    num_chunks = int(np.ceil(2500/chunk_size))
    spikes = np.zeros((flags.n_trials_per_angle, len(DG_angles), sim_duration, network['n_nodes']), dtype=np.uint8)

    for angle_id, angle in enumerate(DG_angles):
        print(f'Running angle {angle}...')
        # load LGN firign rates for the given angle and calculate spiking probability
        t0 = time()
        # load LGN firign rates for the given angle and calculate spiking probability
        lgn_prob = lgn_firing_probabilities_dict[angle]
        lgn_prob = tf.tile(tf.expand_dims(lgn_prob, axis=0), [per_replica_batch_size, 1, 1])

        for iter_id in range(n_iterations):
            start_idx = iter_id * per_replica_batch_size
            end_idx = min((iter_id + 1) * per_replica_batch_size, flags.n_trials_per_angle)
            iteration_length = end_idx - start_idx

            lgn_spikes = generate_spontaneous_spikes(lgn_prob)
            # Reset the memory stats
            # tf.config.experimental.reset_memory_stats('GPU:0')
            
            for i in range(num_chunks):
                chunk = lgn_spikes[:, i * chunk_size : (i + 1) * chunk_size, :]
                v1_z_chunk, continuing_state = distributed_roll_out(chunk, continuing_state)
                # v1_spikes[angle_id, i * chunk_size : (i + 1) * chunk_size, :] += v1_z_chunk.numpy()[0, :, :].astype(float)
                # lm_spikes[angle_id, i * chunk_size : (i + 1) * chunk_size, :] += lm_z_chunk.numpy()[0, :, :].astype(float)
                spikes[start_idx:end_idx, angle_id, i * chunk_size : (i + 1) * chunk_size, :] += v1_z_chunk.numpy()[:iteration_length, :, :].astype(np.uint8)
                
            if angle_id == 0:
                # Raster plot for 0 degree orientation
                callbacks.single_trial_callbacks(lgn_spikes.numpy(), spikes[0], y=angle)
    
            print(f'    Angle processing time: {time() - t0:.2f}s')
            for gpu_id in range(len(strategy.extended.worker_devices)):
                printgpu(gpu_id=gpu_id)

            if not flags.calculate_osi_dsi:
                break

    # Do the OSI/DSI analysis       
    if flags.calculate_osi_dsi:
        callbacks.osi_dsi_analysis(spikes, DG_angles)


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
    absl.app.flags.DEFINE_float('lr_scale', 1., '')
    absl.app.flags.DEFINE_float('temporal_f', 2., '')
    absl.app.flags.DEFINE_float('max_time', -1, '')

    absl.app.flags.DEFINE_integer('n_runs', 1, '')
    absl.app.flags.DEFINE_integer('run_session', 0, '')
    absl.app.flags.DEFINE_integer('n_epochs', 50, '')
    absl.app.flags.DEFINE_integer('batch_size', 1, '')
    absl.app.flags.DEFINE_integer('neurons', 65871, '')  # -1 to take all neurons
    absl.app.flags.DEFINE_integer('steps_per_epoch', 20, '')# EA and garret dose not need this many but pure classification needs 781 = int(50000/64)
    absl.app.flags.DEFINE_integer('val_steps', 1, '')# EA and garret dose not need this many but pure classification needs 156 = int(10000/64)
    # number of LGN filters in visual space (input population)
    absl.app.flags.DEFINE_integer('n_input', 17400, '')
    absl.app.flags.DEFINE_integer("n_output", 2, "")
    absl.app.flags.DEFINE_integer('seq_len', 600, '')
    absl.app.flags.DEFINE_integer('seed', 3000, '')
    absl.app.flags.DEFINE_integer('neurons_per_output', 16, '')

    absl.app.flags.DEFINE_integer('n_cues', 3, '')
    absl.app.flags.DEFINE_integer('recall_duration', 40, '')
    absl.app.flags.DEFINE_integer('cue_duration', 40, '')
    absl.app.flags.DEFINE_integer('interval_duration', 40, '')
    absl.app.flags.DEFINE_integer('examples_in_epoch', 32, '')
    absl.app.flags.DEFINE_integer('validation_examples', 16, '')
    absl.app.flags.DEFINE_integer('n_trials_per_angle', 10, '')

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
    absl.app.flags.DEFINE_boolean("current_input", False, "")

    absl.app.flags.DEFINE_string("rotation", "ccw", "")
    absl.app.flags.DEFINE_string('ckpt_dir', '', '')

    absl.app.run(main)
