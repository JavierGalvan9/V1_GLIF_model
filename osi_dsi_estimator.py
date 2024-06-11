import matplotlib
matplotlib.use('agg')# to avoid GUI request on clusters
import os

# Define the environment variables for optimal GPU performance
os.environ['TF_GPU_THREAD_MODE'] = 'gpu_private'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # before import tensorflow
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
from v1_model_utils.model_metrics_analysis import ModelMetricsAnalysis
from v1_model_utils.plotting_utils import InputActivityFigure 
import stim_dataset

from time import time
import ctypes.util


print("--- CUDA version: ", tf.sysconfig.get_build_info()["cuda_version"])
print("--- CUDNN version: ", tf.sysconfig.get_build_info()["cudnn_version"])
print("--- TensorFlow version: ", tf.__version__)

# For CUDA Runtime API
lib_path = ctypes.util.find_library('cudart')
print("--- CUDA Library path: ", lib_path)

debug = False

def printgpu(verbose=0):
    if tf.config.list_physical_devices('GPU'):
        meminfo = tf.config.experimental.get_memory_info('GPU:0')
        current = meminfo['current'] / 1024**3
        peak = meminfo['peak'] / 1024**3
        if verbose == 0:
            print(f"GPU memory use: {current:.2f} GB / Peak: {peak:.2f} GB")
        if verbose == 1:
            return current, peak

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

    # Load the checkpoint model if it exists
    logdir = flags.ckpt_dir
    if logdir == '':
        flag_str = f'v1_{flags.neurons}'
        for name, value in flags.flag_values_dict().items():
            if value != flags[name].default and name in ['n_input', 'core_only', 'connected_selection']:
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
        current_epoch = (flags.run_session + 1) * flags.n_epochs
        
    # Can be used to try half precision training
    if flags.float16:
        if version.parse(tf.__version__) < version.parse("2.4.0"):
            policy = mixed_precision.Policy("mixed_float16")
            mixed_precision.set_policy(policy)
        else:
            mixed_precision.set_global_policy('mixed_float16')
        dtype = tf.float16
    else:
        dtype = tf.float32

    n_workers, n_gpus_per_worker = 1, 1
    # model is being run on multiple GPUs or CPUs, and the results are being reduced to a single CPU device. 
    # In this case, the reduce_to_device argument is set to "cpu:0", which means that the results are being reduced to the first CPU device.
    # strategy = tf.distribute.MirroredStrategy(cross_device_ops=tf.distribute.ReductionToOneDevice(reduce_to_device="cpu:0")) 
    device = "/gpu:0" if tf.config.list_physical_devices("GPU") else "/cpu:0"
    strategy = tf.distribute.OneDeviceStrategy(device=device)

    per_replica_batch_size = flags.batch_size
    global_batch_size = per_replica_batch_size * strategy.num_replicas_in_sync
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
            dtype=tf.float32, 
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
            batch_size=flags.batch_size, 
            pseudo_gauss=flags.pseudo_gauss, 
            use_state_input=True, 
            return_state=True,
            hard_reset=flags.hard_reset,
            add_metric=True,
            max_delay=5, 
            )

        del lgn_input, bkg_input

        model.build((flags.batch_size, flags.seq_len, flags.n_input))
        print(f"Model built in {time()-t0:.2f} s\n")

        # Define the optimizer
        if flags.optimizer == 'adam':
            optimizer = tf.keras.optimizers.Adam(flags.learning_rate, epsilon=1e-11)  
        elif flags.optimizer == 'sgd':
            optimizer = tf.keras.optimizers.SGD(flags.learning_rate, momentum=0.0, nesterov=False)
        else:
            print(f"Invalid optimizer: {flags.optimizer}")
            raise ValueError

        optimizer.build(model.trainable_variables)

        # Restore model and optimizer from a checkpoint if it exists
        if flags.ckpt_dir != '' and os.path.exists(flags.ckpt_dir):
            print(f'Restoring checkpoint from {flags.ckpt_dir}...')
            checkpoint_directory = tf.train.latest_checkpoint(os.path.join(flags.ckpt_dir, "OSI_DSI_checkpoints"))
            checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)
            checkpoint.restore(checkpoint_directory).assert_consumed()
            print('Checkpoint restored!')
            print(f'OSI/DSI results for epoch {current_epoch} will be saved in: {checkpoint_directory}\n')
        else:
            checkpoint = None # no restoration from any checkpoint

        # Build the model layers
        rsnn_layer = model.get_layer('rsnn')
        prediction_layer = model.get_layer('prediction')
        # abstract_layer = model.get_layer('abstract_output')
        extractor_model = tf.keras.Model(inputs=model.inputs,
                                         outputs=[rsnn_layer.output, model.output, prediction_layer.output])

        zero_state = rsnn_layer.cell.zero_state(flags.batch_size, dtype=dtype)
        state_variables = tf.nest.map_structure(lambda a: tf.Variable(
            a, trainable=False, synchronization=tf.VariableSynchronization.ON_READ
        ), zero_state)

        def roll_out(_x):
            _initial_state = tf.nest.map_structure(lambda _a: _a.read_value(), state_variables)
            dummy_zeros = tf.zeros((flags.batch_size, flags.seq_len, flags.neurons), dtype)
            _out, _p, _ = extractor_model((_x, dummy_zeros, _initial_state))

            z = _out[0][0]
            # update state_variables with the new model state
            new_state = tuple(_out[1:])
            tf.nest.map_structure(lambda a, b: a.assign(b), state_variables, new_state)

            return z

        @tf.function
        def distributed_roll_out(x):
            _z = strategy.run(roll_out, args=(x,))
            return _z
        
        # LGN firing rates to the different angles
        DG_angles = np.arange(0, 360, 45)
        osi_dataset_path = os.path.join('OSI_DSI_dataset', 'lgn_firing_rates.pkl')
        if not os.path.exists(osi_dataset_path):
            print('Creating OSI/DSI dataset...')
            # Define OSI/DSI dataset
            def get_osi_dsi_dataset_fn(regular=False):
                def _f(input_context):
                    post_delay = flags.seq_len - (2500 % flags.seq_len)
                    _lgn_firing_rates = stim_dataset.generate_drifting_grating_tuning(
                        seq_len=2500+post_delay,
                        pre_delay=500,
                        post_delay = post_delay,
                        n_input=flags.n_input,
                        regular=regular,
                        return_firing_rates=True,
                        rotation=flags.rotation,
                    ).batch(1)
                                
                    return _lgn_firing_rates
                return _f
        
            osi_dsi_data_set = strategy.distribute_datasets_from_function(get_osi_dsi_dataset_fn(regular=True))
            test_it = iter(osi_dsi_data_set)
            lgn_firing_rates_dict = {}  # Dictionary to store firing rates
            for angle_id, angle in enumerate(DG_angles):
                t0 = time()
                lgn_firing_rates = next(test_it)
                lgn_firing_rates_dict[angle] = lgn_firing_rates.numpy()
                print(f'Angle {angle} done.')
                print(f'    Trial running time: {time() - t0:.2f}s')
                mem_data = printgpu(verbose=1)
                print(f'    Memory consumption (current - peak): {mem_data[0]:.2f} GB - {mem_data[1]:.2f} GB\n')

            # Save the dataset      
            results_dir = os.path.join("OSI_DSI_dataset")
            os.makedirs(results_dir, exist_ok=True)
            with open(osi_dataset_path, 'wb') as f:
                pkl.dump(lgn_firing_rates_dict, f)
            print('OSI/DSI dataset created successfully!')

        else:
            # Load the LGN firing rates dataset
            with open(osi_dataset_path, 'rb') as f:
                lgn_firing_rates_dict = pkl.load(f)

        print('Starting to plot OSI and DSI...')
        sim_duration = (2500//flags.seq_len + 1) * flags.seq_len
        spikes = np.zeros((8, sim_duration, network['n_nodes']), dtype=float)

        for angle_id, angle in enumerate(range(0, 360, 45)):
            # load LGN firign rates for the given angle and calculate spiking probability
            lgn_fr = lgn_firing_rates_dict[angle]
            lgn_fr = tf.constant(lgn_fr, dtype=tf.float32)
            _p = 1 - tf.exp(-lgn_fr / 1000.)

            for trial_id in range(flags.n_trials_per_angle):
                t0 = time()
                # Reset the memory stats
                tf.config.experimental.reset_memory_stats('GPU:0')

                x = tf.random.uniform(tf.shape(_p)) < _p

                chunk_size = flags.seq_len
                num_chunks = (2500//chunk_size + 1)
                for i in range(num_chunks):
                    chunk = x[:, i * chunk_size : (i + 1) * chunk_size, :]
                    z_chunk = distributed_roll_out(chunk)
                    spikes[angle_id, i * chunk_size : (i + 1) * chunk_size, :] += z_chunk.numpy()[0, :, :].astype(float)
                    
                if trial_id == 0 and angle_id == 0:
                    # Raster plot for 0 degree orientation
                    lgn_spikes = x[:, :2500, :].numpy()
                    z = spikes[:, :2500, :]
                    images_dir = os.path.join(logdir, 'Raster_plots_OSI_DSI')
                    os.makedirs(images_dir, exist_ok=True)
                    graph = InputActivityFigure(
                                                network,
                                                flags.data_dir,
                                                images_dir,
                                                filename=f'Epoch_{current_epoch}_orientation_0_degrees',
                                                frequency=flags.temporal_f,
                                                stimuli_init_time=500,
                                                stimuli_end_time=2500,
                                                reverse=False,
                                                plot_core_only=True,
                                                )
                    graph(lgn_spikes, z)

                print(f'Trial {trial_id+1}/{flags.n_trials_per_angle} - Angle {angle} done.')
                print(f'    Trial running time: {time() - t0:.2f}s')
                mem_data = printgpu(verbose=1)
                print(f'    Memory consumption (current - peak): {mem_data[0]:.2f} GB - {mem_data[1]:.2f} GB\n')


        # Average the spikes over the number of trials
        spikes = spikes/flags.n_trials_per_angle
        spikes = spikes[:, :2500, :]

        # Do the OSI/DSI analysis       
        boxplots_dir = os.path.join(logdir, 'Boxplots_OSI_DSI')
        os.makedirs(boxplots_dir, exist_ok=True)
        metrics_analysis = ModelMetricsAnalysis(network, data_dir=flags.data_dir,
                                                drifting_gratings_init=500, drifting_gratings_end=2500,
                                                core_radius=flags.plot_core_radius)
        metrics_analysis(spikes, DG_angles, metrics=["Rate at preferred direction (Hz)", "OSI", "DSI"], 
                        directory=boxplots_dir, filename=f'Epoch_{current_epoch}')


if __name__ == '__main__':
    _data_dir = 'GLIF_network'
    _results_dir = 'Simulation_results'

    absl.app.flags.DEFINE_string('task_name', 'drifting_gratings_firing_rates_distr' , '')
    absl.app.flags.DEFINE_string('data_dir', _data_dir, '')
    absl.app.flags.DEFINE_string('results_dir', _results_dir, '')
    absl.app.flags.DEFINE_string('restore_from', '', '')
    absl.app.flags.DEFINE_string('comment', '', '')
    absl.app.flags.DEFINE_string('delays', '100,0', '')
    absl.app.flags.DEFINE_string('scale', '2,2', '')
    absl.app.flags.DEFINE_string('optimizer', 'adam', '')

    absl.app.flags.DEFINE_float('learning_rate', .01, '')
    absl.app.flags.DEFINE_float('rate_cost', 100., '')
    absl.app.flags.DEFINE_float('voltage_cost', 1., '')
    absl.app.flags.DEFINE_float('osi_cost', 1., '')
    absl.app.flags.DEFINE_string('osi_loss_method', 'crowd_spikes', '')
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

    absl.app.flags.DEFINE_boolean('float16', False, '')
    absl.app.flags.DEFINE_boolean('caching', True, '')
    absl.app.flags.DEFINE_boolean('core_only', False, '')
    absl.app.flags.DEFINE_boolean('core_loss', False, '')
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
    absl.app.flags.DEFINE_boolean("reset_every_step", False, "")

    absl.app.flags.DEFINE_string("rotation", "ccw", "")
    absl.app.flags.DEFINE_string('ckpt_dir', '', '')

    absl.app.run(main)
