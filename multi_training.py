import matplotlib
matplotlib.use('agg')# to avoid GUI request on clusters
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
import socket
import absl
import json
import time
import contextlib
import datetime as dt
import numpy as np
import pickle as pkl
import tensorflow as tf
import re
from packaging import version
# check the version of tensorflow, and do the right thing.
# if tf.__version__ < "2.4.0": # does not work wor 2.10.1.
if version.parse(tf.__version__) < version.parse("2.4.0"):
    from tensorflow.keras.mixed_precision import experimental as mixed_precision
else:
    from tensorflow.keras import mixed_precision

from general_utils import file_management
from general_utils.other_utils import memory_tracer, timer
from v1_model_utils import load_sparse, other_v1_utils, toolkit
from v1_model_utils.plotting_utils import InputActivityFigure, LaminarPlot, LGN_sample_plot, PopulationActivity, RasterPlot
import stim_dataset
from time import time
import ctypes.util

from v1_model_utils.plotting_utils import InputActivityFigure, ModelMetricsAnalysis
from v1_model_utils.callbacks import Callbacks


# Define the environment variables for optimal GPU performance
# os.environ['TF_GPU_THREAD_MODE'] = 'gpu_private'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

print("--- CUDA version: ", tf.sysconfig.get_build_info()["cuda_version"])
print("--- CUDNN version: ", tf.sysconfig.get_build_info()["cudnn_version"])
print("--- TensorFlow version: ", tf.__version__)

# For CUDA Runtime API
lib_path = ctypes.util.find_library('cudart')
print("--- CUDA Library path: ", lib_path)

debug = False

# def printgpu(verbose=0):
#     if tf.config.list_physical_devices('GPU'):
#         meminfo = tf.config.experimental.get_memory_info('GPU:0')
#         current = meminfo['current'] / 1024**3
#         peak = meminfo['peak'] / 1024**3
#         if verbose == 0:
#             print(f"GPU memory use: {current:.2f} GB / Peak: {peak:.2f} GB")
#         if verbose == 1:
#             return current, peak

def print_vram(check_point_num=0):
    if debug:
        tf_ram = tf.config.experimental.get_memory_usage('GPU:0') / 1024**3
        print(f"Checkpoint {check_point_num}: GPU memory usage: {tf_ram:.3f} GB")
    return


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
    np.random.seed(flags.seed)
    tf.random.set_seed(flags.seed) 

    # Choose optimized version for training depending on whether the user wants to 
    # train the input or the recurrent
    # if flags.train_input or flags.train_noise:
    #     import v1_model_utils.models_train_input as models
    # else:
    #     import v1_model_utils.models as models

    import v1_model_utils.models as models

    results_dir = os.path.join(flags.results_dir, 'multi_training')
    os.makedirs(results_dir, exist_ok=True)
     # Save the flags configuration in a JSON file
    with open(os.path.join(results_dir, 'flags_config.json'), 'w') as fp:
        json.dump(flags.flag_values_dict(), fp)

    # Can be used to try half precision training
    if flags.float16:
        # policy = mixed_precision.Policy("mixed_float16")
        # mixed_precision.set_policy(policy)
        if version.parse(tf.__version__) < version.parse("2.4.0"):
            policy = mixed_precision.Policy("mixed_float16")
            mixed_precision.set_policy(policy)
        else:
            mixed_precision.set_global_policy('mixed_float16')

        dtype = tf.float16
    else:
        dtype = tf.float32

    sim_name = toolkit.get_random_identifier('b_')
    logdir = os.path.join(results_dir, sim_name)
    print(f'> Results for {flags.task_name} will be stored in:\n {logdir} \n')

    n_workers, n_gpus_per_worker = 1, 1
    # model is being run on multiple GPUs or CPUs, and the results are being reduced to a single CPU device. 
    # In this case, the reduce_to_device argument is set to "cpu:0", which means that the results are being reduced to the first CPU device.
    # strategy = tf.distribute.MirroredStrategy(cross_device_ops=tf.distribute.ReductionToOneDevice(reduce_to_device="cpu:0")) 
    device = "/gpu:0" if tf.config.list_physical_devices("GPU") else "/cpu:0"
    strategy = tf.distribute.OneDeviceStrategy(device=device)

    per_replica_batch_size = flags.batch_size
    global_batch_size = per_replica_batch_size * strategy.num_replicas_in_sync
    print(f'Global batch size: {global_batch_size}\n')

    ### Load or create the network building files configuration
    t0 = time()
    if flags.caching:
        load_fn = load_sparse.cached_load_v1
    else:
        load_fn = load_sparse.load_v1
    network, lgn_input, bkg_input = load_fn(flags, flags.neurons)
    print(f"Model files loading: {time()-t0:.2f} seconds\n")
  
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
            input_weight_scale=flags.input_weight_scale,
            dampening_factor=flags.dampening_factor,
            gauss_std=flags.gauss_std,
            lr_scale=flags.learning_rate,
            train_input=flags.train_input,
            train_noise=flags.train_noise,
            train_recurrent=flags.train_recurrent,
            neuron_output=flags.neuron_output,
            recurrent_dampening_factor=flags.recurrent_dampening_factor,
            batch_size=flags.batch_size,
            pseudo_gauss=flags.pseudo_gauss,
            use_state_input=True,
            return_state=True,
        )
        
        del lgn_input, bkg_input

        # Initialize the weights of the model based on the specified input shape. It operates in eager mode.
        # It does not construct a computational graph of the model operations, but prepares the model layers and weights
        model.build((flags.batch_size, flags.seq_len, flags.n_input))
        print(f"Model built in {time()-t0:.2f} s\n")
        
        # Extract outputs of intermediate keras layers to get access to
        # spikes and membrane voltages of the model
        rsnn_layer = model.get_layer("rsnn")
        rec_weight_regularizer = models.StiffRegularizer(flags.recurrent_weight_regularization,
                                                         rsnn_layer.cell.recurrent_weight_values)

        # Load the firing rates distribution as a regularizer that we have and generate target firing rates for every neuron type
        with open(os.path.join(flags.data_dir, 'np_gratings_firing_rates.pkl'), 'rb') as f:
            target_firing_rates = pkl.load(f)

        tuning_angles = tf.constant(network['tuning_angle'], dtype=dtype)
        cell_type_ids = np.zeros(flags.neurons, dtype=np.int32)
        for i, (key, value) in enumerate(target_firing_rates.items()):
            # identify tne ids that are included in value["ids"]
            neuron_ids = np.where(np.isin(network["node_type_ids"], value["ids"]))[0]
            neuron_ids = tf.cast(neuron_ids, dtype=tf.int32)
            target_firing_rates[key]['neuron_ids'] = neuron_ids
            type_n_neurons = len(neuron_ids)
            sorted_target_rates = models.sample_firing_rates(value["rates"], type_n_neurons, flags.seed)
            target_firing_rates[key]['sorted_target_rates'] = tf.cast(sorted_target_rates, dtype=tf.float32) 
            cell_type_ids[neuron_ids] = i

        # make a new array for neurons that contains neuron type ids.
        pre_cell_type_ids = cell_type_ids[network["synapses"]["indices"][:, 0]]
        post_cell_type_ids = cell_type_ids[network["synapses"]["indices"][:, 1] % flags.neurons]
        # division by neurons is needed to cancel the delay embedding.
        
        # assign connection type id to each unique pair of the sending and recieving ids.
        connection_ids = np.zeros(network["synapses"]["indices"].shape[0], dtype=np.int32)
        connection_ids = pre_cell_type_ids * 1000 + post_cell_type_ids
        
        # for each connection type, make a list of synapse indices.
        connection_type_ids = np.unique(connection_ids)
        connection_type_ids = connection_type_ids[connection_type_ids != 0]
        connection_type_ids = np.sort(connection_type_ids)
        connection_type_ids = connection_type_ids.tolist()
        same_connection_type_indices = []
        for i in connection_type_ids:
            same_connection_type_indices.append(np.where(connection_ids == i)[0])
        
        # Create rate and voltage regularizers
        rate_distribution_regularizer = models.SpikeRateDistributionTarget(target_firing_rates, flags.rate_cost, dtype=dtype)
        # rate_distribution_regularizer = models.SpikeRateDistributionRegularization(target_firing_rates, flags.rate_cost)
        rate_loss = rate_distribution_regularizer(rsnn_layer.output[0][0])
        voltage_regularizer = models.VoltageRegularization(rsnn_layer.cell, flags.voltage_cost, dtype=dtype)
        voltage_loss = voltage_regularizer(rsnn_layer.output[0][1]) 
        model.add_loss(rate_loss)
        model.add_loss(voltage_loss)
        model.add_metric(rate_loss, name='rate_loss')
        model.add_metric(voltage_loss, name='voltage_loss')

        prediction_layer = model.get_layer('prediction')
        extractor_model = tf.keras.Model(inputs=model.inputs,
                                         outputs=[rsnn_layer.output, model.output, prediction_layer.output])
        # extractor_model = tf.keras.Model(inputs=model.inputs, outputs=[rsnn_layer.output])

        # Loss from Guozhang classification task (unused in our case)
        loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=False, reduction=tf.keras.losses.Reduction.NONE)
        def compute_loss(_l, _p, _w):
            per_example_loss = loss_object(_l, _p, sample_weight=_w) * strategy.num_replicas_in_sync / tf.reduce_sum(_w)
            rec_weight_loss = rec_weight_regularizer(rsnn_layer.cell.recurrent_weight_values)
            return tf.nn.compute_average_loss(per_example_loss, global_batch_size=global_batch_size) + rec_weight_loss

        def calculate_delta_angle(stim_angle, tuning_angle):
            # angle unit is degrees.
            # this function calculates the difference between stim_angle and tuning_angle,
            # but it is fine to have the opposite direction.
            # so, delta angle is always between -90 and 90.
            # they are both vector, so dimension matche is needed.
            # stim_angle is a length of batch size
            # tuning_angle is a length of n_neurons

            # delta_angle = stim_angle - tuning_angle
            delta_angle = tf.expand_dims(stim_angle, axis=1) - tuning_angle
            delta_angle = tf.where(delta_angle > 90, delta_angle - 180, delta_angle)
            delta_angle = tf.where(delta_angle < -90, delta_angle + 180, delta_angle)
            # # do it twice to make sure
            delta_angle = tf.where(delta_angle > 90, delta_angle - 180, delta_angle)
            delta_angle = tf.where(delta_angle < -90, delta_angle + 180, delta_angle)

            return delta_angle

        def compute_loss_gratings(_y, _z):
            # I need to access the tuning angle. of all the neurons.
            _y = tf.cast(_y, dtype)
            delta_angle = calculate_delta_angle(_y, tuning_angles)
            # sum spikes in _z, and multiply with delta_angle.
            sum_angle = tf.reduce_mean(_z, axis=[1]) * delta_angle
            # make a huber loss for this.
            # angle_loss = tf.keras.losses.Huber(delta=1, reduction=tf.keras.losses.Reduction.SUM)(sum_angle, tf.zeros_like(sum_angle))
            angle_loss = tf.reduce_mean(tf.abs(sum_angle))
            # it might be nice to add regularization of weights
            rec_weight_loss = rec_weight_regularizer(rsnn_layer.cell.recurrent_weight_values)

            return angle_loss + rec_weight_loss
        
        optimizer = tf.keras.optimizers.Adam(flags.learning_rate, epsilon=1e-11)  

        zero_state = rsnn_layer.cell.zero_state(flags.batch_size, dtype=dtype)
        state_variables = tf.nest.map_structure(lambda a: tf.Variable(
            a, trainable=False, synchronization=tf.VariableSynchronization.ON_READ
        ), zero_state)
        optimizer.build(model.trainable_variables)  # Add this line to build the optimizer with the trainable variables

        # Add other metrics and losses
        train_loss = tf.keras.metrics.Mean()
        val_loss = tf.keras.metrics.Mean()
        train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()
        val_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()
        train_firing_rate = tf.keras.metrics.Mean()
        val_firing_rate = tf.keras.metrics.Mean()
        train_rate_loss = tf.keras.metrics.Mean()
        val_rate_loss = tf.keras.metrics.Mean()
        train_voltage_loss = tf.keras.metrics.Mean()
        val_voltage_loss = tf.keras.metrics.Mean()

        def reset_train_metrics():
            train_loss.reset_states(), train_accuracy.reset_states(), train_firing_rate.reset_states()
            train_rate_loss.reset_states(), train_voltage_loss.reset_states()

        def reset_validation_metrics():
            val_loss.reset_states(), val_accuracy.reset_states(), val_firing_rate.reset_states()
            val_rate_loss.reset_states(), val_voltage_loss.reset_states()

    # checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)

    def roll_out(_x, _y, _w, output_spikes=False):
        _initial_state = tf.nest.map_structure(lambda _a: _a.read_value(), state_variables)
        dummy_zeros = tf.zeros((flags.batch_size, flags.seq_len, flags.neurons), dtype)
        _out, _p, _ = extractor_model((_x, dummy_zeros, _initial_state))

        _z, _v, _input_current = _out[0]
        # update state_variables with the new model state
        new_state = tuple(_out[1:])
        tf.nest.map_structure(lambda a, b: a.assign(b), state_variables, new_state)

        voltage_loss = voltage_regularizer(_v) 
        rate_loss = rate_distribution_regularizer(_z)
        classification_loss = compute_loss_gratings(_y, _z)

        _aux = dict(rate_loss=rate_loss, voltage_loss=voltage_loss)
        _loss = classification_loss + rate_loss + voltage_loss

        return _out, _p, _loss, _aux

    @tf.function
    def distributed_roll_out(x, y, w, output_spikes=True):
        _out, _p, _loss, _aux = strategy.run(roll_out, args=(x, y, w))
        if output_spikes:
            return _out[0][0]
        else:
            return _out, _p, _loss, _aux

    
    def calculate_grad_average(g, v, grad_average_ind):
        print('Calculating gradient average...')
        newg = tf.zeros_like(g)
        for inds in grad_average_ind:
            v_gathered = tf.gather(v, inds)
            g_gathered = tf.gather(g, inds)
            # Avoid division by zero and replace NaNs
            safe_divisor = tf.where(v_gathered != 0, v_gathered, tf.ones_like(v_gathered))
            mean_frac_change = tf.reduce_mean(g_gathered / safe_divisor)
            # mean_frac_change = tf.reduce_mean(tf.gather(g, inds) / tf.gather(v, inds))
            mean_frac_change = tf.where(tf.math.is_nan(mean_frac_change), tf.zeros_like(mean_frac_change), mean_frac_change)
            mean_rel_change = tf.gather(v, inds) * mean_frac_change
            # newg[inds].assign(mean_rel_change)
            newg = tf.tensor_scatter_nd_update(newg, tf.expand_dims(inds, axis=1), mean_rel_change)
        
        return newg

    def train_step(_x, _y, _w, grad_average_ind=None):
        ### Forward propagation of the model
        with tf.GradientTape() as tape:
            _out, _p, _loss, _aux = roll_out(_x, _y, _w)

        ### Backpropagation of the model
        _op = train_accuracy.update_state(_y, _p, sample_weight=_w)
        with tf.control_dependencies([_op]):
            _op = train_loss.update_state(_loss)
        _rate = tf.reduce_mean(_out[0][0])
        with tf.control_dependencies([_op]):
            _op = train_firing_rate.update_state(_rate)
        with tf.control_dependencies([_op]):
            _op = train_rate_loss.update_state(_aux['rate_loss'])
        with tf.control_dependencies([_op]):
            _op = train_voltage_loss.update_state(_aux['voltage_loss'])

        grad = tape.gradient(_loss, model.trainable_variables)
             
        for g, v in zip(grad, model.trainable_variables):
            tf.print(f'{v.name} optimization')
            tf.print('First Var / grad: ', v[0].dtype, g[0].dtype)
            # tf.print('First Var / grad: ', v[0], g[0])
            tf.print('Loss, total_gradients : ', _loss, tf.reduce_sum(tf.math.abs(g)))
            tf.print(g)
            tf.print(v)
            with tf.control_dependencies([_op]):
                # if the trainable variable is recurrent connection, average the gradient
                # for each cell type pair.
                if grad_average_ind is not None:
                    if v.name == "sparse_recurrent_weights:0":
                        newg = calculate_grad_average(g, v, grad_average_ind)
                        _op = optimizer.apply_gradients([(newg, v)])
                    else:
                        _op = optimizer.apply_gradients([(g, v)])
                else:
                    _op = optimizer.apply_gradients([(g, v)])

            # print the number of nan values in the gradients and the variables.
            tf.print(v)
            # tf.print(f'Number of nan in gradients: {tf.reduce_sum(tf.cast(tf.math.is_nan(g), tf.float32))}')
            # tf.print(f'Number of nan in variables: {tf.reduce_sum(tf.cast(tf.math.is_nan(v), tf.float32))}')


    @tf.function
    def distributed_train_step(x, y, weights, grad_average_ind=None):
        strategy.run(train_step, args=(x, y, weights, grad_average_ind))

    # @tf.function
    # def distributed_train_step(dist_inputs):
    #     per_replica_losses = mirrored_strategy.run(train_step, args=(dist_inputs,))
    #     return mirrored_strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses,
    #                             axis=None)

    def validation_step(_x, _y, _w, output_spikes=True):
        _out, _p, _loss, _aux = roll_out(_x, _y, _w)
        _op = val_accuracy.update_state(_y, _p, sample_weight=_w)
        with tf.control_dependencies([_op]):
            _op = val_loss.update_state(_loss)
        _rate = tf.reduce_mean(_out[0][0])
        with tf.control_dependencies([_op]):
            _op = val_firing_rate.update_state(_rate)
        with tf.control_dependencies([_op]):
            _op = val_rate_loss.update_state(_aux['rate_loss'])
        with tf.control_dependencies([_op]):
            _op = val_voltage_loss.update_state(_aux['voltage_loss'])
        # tf.nest.map_structure(lambda _a, _b: _a.assign(_b), list(state_variables), _out[1:])

        if output_spikes:
            return _out[0][0]


    @tf.function
    def distributed_validation_step(x, y, weights, output_spikes=True):
        if output_spikes:
            return strategy.run(validation_step, args=(x, y, weights, output_spikes))
        else:
            strategy.run(validation_step, args=(x, y, weights))

    # Define the function that generates the dataset for our task
    def get_dataset_fn(regular=False):
        def _f(input_context):
            delays = [int(a) for a in flags.delays.split(',') if a != '']
            _data_set = stim_dataset.generate_drifting_grating_tuning(
                seq_len=flags.seq_len,
                pre_delay=delays[0],
                post_delay=delays[1],
                n_input=flags.n_input,
                regular=regular
            ).batch(per_replica_batch_size)
                        
            return _data_set
        return _f

    def get_gray_dataset_fn():
        def _f(input_context):
            _gray_data_set = stim_dataset.generate_drifting_grating_tuning(
                seq_len=flags.seq_len,
                pre_delay=flags.seq_len,
                post_delay=0,
                n_input=flags.n_input,
            ).batch(per_replica_batch_size)
                        
            return _gray_data_set
        return _f
       
    # We define the dataset generates function under the strategy scope for a randomly selected orientation       
    # test_data_set = strategy.distribute_datasets_from_function(get_dataset_fn(regular=True))
    train_data_set = strategy.distribute_datasets_from_function(get_dataset_fn())      
    gray_data_set = strategy.distribute_datasets_from_function(get_gray_dataset_fn())

    # def reset_state():
    #     tf.nest.map_structure(lambda a, b: a.assign(b), state_variables, zero_state)

    def reset_state(reset_type='zero', new_state=None):
        if reset_type == 'zero':
            tf.nest.map_structure(lambda a, b: a.assign(b), state_variables, zero_state)
        elif reset_type == 'gray':
            # Run a gray simulation to get the model state
            tf.nest.map_structure(lambda a, b: a.assign(b), state_variables, new_state)
        elif reset_type == 'continue':
            # Continue on the previous model state
            # No action needed, as the state_variables will not be modified
            pass
        else:
            raise ValueError(f"Invalid reset_type: {reset_type}")

    # @tf.function
    def distributed_reset_state(reset_type, gray_state=None):
        if reset_type == 'gray':
            if gray_state is None:
                gray_it = next(iter(gray_data_set))
                x, y, _, w = gray_it
                tf.nest.map_structure(lambda a, b: a.assign(b), state_variables, zero_state)
                _out, _p, _loss, _aux = distributed_roll_out(x, y, w, output_spikes=False)
                gray_state = tuple(_out[1:])
                strategy.run(reset_state, args=(reset_type, gray_state))
                return gray_state
            else:
                strategy.run(reset_state, args=(reset_type, gray_state))

        else:
            strategy.run(reset_state, args=(reset_type, zero_state))


    ############################ TRAINING #############################

    stop = False
    # Initialize your callbacks
    metric_keys = ['train_accuracy', 'train_loss', 'train_firing_rate', 'train_rate_loss',
            'train_voltage_loss', 'val_accuracy', 'val_loss',
            'val_firing_rate', 'val_rate_loss', 'val_voltage_loss']
    callbacks = Callbacks(model, optimizer, distributed_roll_out, network, flags, logdir, strategy, metric_keys)
    
    callbacks.on_train_begin()
    for epoch in range(flags.n_epochs):
        callbacks.on_epoch_start()  
        # Reset the model state to the gray state    
        gray_state = distributed_reset_state('gray')  
        
        # Load the dataset iterator - this must be done inside the epoch loop
        it = iter(train_data_set)

        # tf.profiler.experimental.start(logdir=logdir)
        for step in range(flags.steps_per_epoch):
            callbacks.on_step_start()
            distributed_reset_state('gray', gray_state=gray_state)

            x, y, _, w = next(it) # x dtype tf.bool
    
            # with tf.profiler.experimental.Trace('train', step_num=step, _r=1):
            if flags.average_grad_for_cell_type:
                distributed_train_step(x, y, w, grad_average_ind=same_connection_type_indices)
            else:
                distributed_train_step(x, y, w)
                
            train_values = [a.result().numpy() for a in [train_accuracy, train_loss, train_firing_rate, 
                                                         train_rate_loss, train_voltage_loss]]

            stop = callbacks.on_step_end(train_values, y, verbose=True)

        # tf.profiler.experimental.stop() 

        # ## VALIDATION AFTER EACH EPOCH
            
        # test_it = iter(test_data_set)
        test_it = it
        for step in range(flags.val_steps):
            x, y, _, w = next(test_it)
            distributed_reset_state('gray', gray_state=gray_state)
            z = distributed_validation_step(x, y, w, output_spikes=True) 

        val_values = [a.result().numpy() for a in [val_accuracy, val_loss, val_firing_rate, 
                                                   val_rate_loss, val_voltage_loss]]
        metric_values = train_values + val_values

        # if the model train loss is minimal, save the model.
        callbacks.on_epoch_end(z, x, y, metric_values)

        if stop:
            print(f'[ Maximum optimization time of {flags.max_time:.2f}h reached ]')
            break
        
        # Reset the metrics for the next epoch
        reset_train_metrics()
        reset_validation_metrics()

    callbacks.on_train_end(metric_values)

 
if __name__ == '__main__':
    hostname = socket.gethostname()
    print("*" * 80)
    print(hostname)
    print("*" * 80)
    # make a condition for different machines. The allen institute has
    # cluster host name to be n??? where ??? is 3 digit number.
    # let's make regex for that.
    if hostname.count('alleninstitute') > 0 or re.search(r'n\d{3}', hostname) is not None:
        _data_dir = '/allen/programs/mindscope/workgroups/realistic-model/shinya.ito/tensorflow_new/V1_GLIF_model/GLIF_network'
        _results_dir = '/allen/programs/mindscope/workgroups/realistic-model/shinya.ito/tensorflow_new/V1_GLIF_model/results'
    else: 
        _data_dir = '/home/jgalvan/Desktop/Neurocoding/V1_GLIF_model/GLIF_network'
        _results_dir = '/home/jgalvan/Desktop/Neurocoding/V1_GLIF_model/GLIF_network/results'

    absl.app.flags.DEFINE_string('data_dir', _data_dir, '')
    absl.app.flags.DEFINE_string('results_dir', _results_dir, '')
    absl.app.flags.DEFINE_string('task_name', 'drifting_gratings_firing_rates_distr' , '')
    
    # absl.app.flags.DEFINE_string('restore_from', '', '')
    # absl.app.flags.DEFINE_string('restore_from', '../results/multi_training/b_53dw/results/ckpt-49', '')
    absl.app.flags.DEFINE_string('restore_from', '', '')
    absl.app.flags.DEFINE_string('comment', '', '')
    absl.app.flags.DEFINE_string('delays', '50,50', '')
    # absl.app.flags.DEFINE_string('neuron_model', 'GLIF3', '')
    absl.app.flags.DEFINE_string('scale', '2,2', '')

    absl.app.flags.DEFINE_float('learning_rate', .01, '')
    absl.app.flags.DEFINE_float('rate_cost', 10., '')
    absl.app.flags.DEFINE_float('voltage_cost', .00001, '')
    absl.app.flags.DEFINE_float('dampening_factor', .5, '')
    absl.app.flags.DEFINE_float('gauss_std', .28, '')
    absl.app.flags.DEFINE_float('recurrent_weight_regularization', 0., '')
    # absl.app.flags.DEFINE_float('p_reappear', .5, '')
    absl.app.flags.DEFINE_float('max_time', -1, '')
    # absl.app.flags.DEFINE_float('max_time', 0.05, '')
    # absl.app.flags.DEFINE_float('scale_w_e', -1, '')
    # absl.app.flags.DEFINE_float('sti_intensity', 2., '')
    absl.app.flags.DEFINE_float('input_weight_scale', 1., '')
    absl.app.flags.DEFINE_float('temporal_f', 2., '')

    absl.app.flags.DEFINE_integer('n_epochs', 51, '')
    absl.app.flags.DEFINE_integer('batch_size', 1, '')
    absl.app.flags.DEFINE_integer('neurons', 66634, '')
    absl.app.flags.DEFINE_integer("n_input", 17400, "")  
    absl.app.flags.DEFINE_integer('seq_len', 600, '')
    # absl.app.flags.DEFINE_integer('im_slice', 100, '')
    absl.app.flags.DEFINE_integer('seed', 3000, '')
    # absl.app.flags.DEFINE_integer('port', 12778, '')
    absl.app.flags.DEFINE_integer("n_output", 2, "")
    absl.app.flags.DEFINE_integer('neurons_per_output', 30, '')
    absl.app.flags.DEFINE_integer('steps_per_epoch', 2, '')# EA and garret dose not need this many but pure classification needs 781 = int(50000/64)
    absl.app.flags.DEFINE_integer('val_steps', 1, '')# EA and garret dose not need this many but pure classification needs 156 = int(10000/64)
    # absl.app.flags.DEFINE_integer('max_delay', 5, '')
    # absl.app.flags.DEFINE_integer('n_plots', 1, '')

    # absl.app.flags.DEFINE_integer('pre_chunks', 3, '')
    # absl.app.flags.DEFINE_integer('post_chunks', 8, '') # the pure calssification task only need 1 but to make consistent with other tasks one has to make up here
    absl.app.flags.DEFINE_integer('pre_delay', 50, '')
    absl.app.flags.DEFINE_integer('post_delay', 450, '')

    # absl.app.flags.DEFINE_boolean('use_rand_connectivity', False, '')
    # absl.app.flags.DEFINE_boolean('use_uniform_neuron_type', False, '')
    # absl.app.flags.DEFINE_boolean('use_only_one_type', False, '')
    # absl.app.flags.DEFINE_boolean('use_dale_law', True, '')
    absl.app.flags.DEFINE_boolean('caching', True, '') # if one wants to use caching, remember to update the caching function
    absl.app.flags.DEFINE_boolean('core_only', False, '')
    # absl.app.flags.DEFINE_boolean('train_input', True, '')
    absl.app.flags.DEFINE_boolean('train_input', False, '')
    absl.app.flags.DEFINE_boolean('train_noise', False, '')
    absl.app.flags.DEFINE_boolean('train_recurrent', True, '')
    absl.app.flags.DEFINE_boolean('connected_selection', True, '')
    absl.app.flags.DEFINE_boolean('neuron_output', False, '')
    # absl.app.flags.DEFINE_boolean('localized_readout', True, '')
    # absl.app.flags.DEFINE_boolean('current_input', True, '')
    # absl.app.flags.DEFINE_boolean('use_rand_ini_w', True, '')
    # absl.app.flags.DEFINE_boolean('use_decoded_noise', True, '')
    # absl.app.flags.DEFINE_boolean('from_lgn', True, '')
    absl.app.flags.DEFINE_boolean("float16", False, "")

    absl.app.flags.DEFINE_integer("cue_duration", 40, "")
    absl.app.flags.DEFINE_float("recurrent_dampening_factor", 0.5, "")
    absl.app.flags.DEFINE_boolean("hard_reset", False, "")
    absl.app.flags.DEFINE_boolean("pseudo_gauss", False, "")
    absl.app.flags.DEFINE_boolean("average_grad_for_cell_type", False, "")

    absl.app.run(main)

