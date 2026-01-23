import matplotlib
matplotlib.use('agg')# to avoid GUI request on clusters
import os

# Define the environment variables for optimal GPU performance
os.environ['TF_GPU_THREAD_MODE'] = 'gpu_private'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # before import tensorflow
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
# os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

import socket
import re
import absl
import numpy as np
import tensorflow as tf
import pickle as pkl
from packaging import version
# check the version of tensorflow, and do the right thing.
# if tf.__version__ < "2.4.0": # does not work wor 2.10.1.
if version.parse(tf.__version__) < version.parse("2.4.0"):
    from tensorflow.keras.mixed_precision import experimental as mixed_precision
else:
    from tensorflow.keras import mixed_precision

from v1_model_utils import load_sparse, models, other_v1_utils, toolkit
import v1_model_utils.loss_functions as losses
from v1_model_utils.callbacks import Callbacks
# from general_utils import file_management
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

    logdir = flags.ckpt_dir
    if logdir == '':
        flag_str = f'v1_{flags.neurons}'
        for name, value in flags.flag_values_dict().items():
            if value != flags[name].default and name in ['n_input', 'core_only', 'connected_selection', 'data_dir']:
                flag_str += f'_{name}_{value}'
        # Define flag string as the second part of results_path
        results_dir = f'{flags.results_dir}/{flag_str}'
        os.makedirs(results_dir, exist_ok=True)
        print('Simulation results path: ', results_dir)
        # Generate a ticker for the current simulation
        sim_name = toolkit.get_random_identifier('b_')
        logdir = os.path.join(results_dir, sim_name)
        print(f'> Results for {flags.task_name} will be stored in:\n {logdir} \n')
    else:
        flag_str = logdir.split(os.path.sep)[-2]

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
    network, lgn_input, bkg_input = load_fn(flags, flags.neurons, flag_str=flag_str)
    print(f"Model files loading: {time()-t0:.2f} seconds\n")

    delays = [int(a) for a in flags.delays.split(',') if a != '']
  
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
            use_dummy_state_input=False,
        )
        
        # Initialize the weights of the model based on the specified input shape. It operates in eager mode.
        # It does not construct a computational graph of the model operations, but prepares the model layers and weights
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

        # Store the initial model variables that are going to be trained
        model_variables_dict = {'Initial': {var.name: var.numpy().astype(np.float16) for var in model.trainable_variables}}

        # Restore model and optimizer from a checkpoint if it exists
        if flags.ckpt_dir != '' and os.path.exists(os.path.join(flags.ckpt_dir, "OSI_DSI_checkpoints")):
            print(f'Restoring checkpoint from {flags.ckpt_dir}...')
            checkpoint_directory = tf.train.latest_checkpoint(os.path.join(flags.ckpt_dir, "OSI_DSI_checkpoints"))
            checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)
            checkpoint.restore(checkpoint_directory).assert_consumed()
            print('Checkpoint restored!')
        elif flags.restore_from != '' and os.path.exists(flags.restore_from):
            print(f'Restoring checkpoint from {flags.restore_from} with the restore_from option...')
            checkpoint_directory = tf.train.latest_checkpoint(flags.restore_from)
            checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)
            checkpoint.restore(checkpoint_directory).assert_consumed()
            print('Checkpoint restored!')
        else:
            checkpoint = None

        ### BUILD THE LOSS AND REGULARIZER FUNCTIONS ###
        # Create rate and voltage regularizers
        if flags.loss_core_radius > 0:
            core_mask = other_v1_utils.isolate_core_neurons(network, radius=flags.loss_core_radius, data_dir=flags.data_dir)
            # if core_mask is all True, set it to None.
            if core_mask.all():
                core_mask = None
                print("All neurons are in the core region. Core mask is set to None.")
            else:
                # report how many neurons are selected.
                print(f"Core mask is set to {core_mask.sum()} neurons.")
                core_mask = tf.constant(core_mask, dtype=tf.bool)
        else:
            core_mask = None
            
        # if flags.core_loss and flags.neurons > 65871:
        #     core_radius = 400
        #     core_mask = other_v1_utils.isolate_core_neurons(network, radius=core_radius, data_dir=flags.data_dir)
        #     core_mask = tf.constant(core_mask, dtype=tf.bool)
        # else:
        #     core_mask = None

        # Extract outputs of intermediate keras layers to get access to
        # spikes and membrane voltages of the model
        rsnn_layer = model.get_layer("rsnn")

        # Load the firing rates distribution as a regularizer that we have and generate target firing rates for every neuron type
        # with open(os.path.join(flags.data_dir, 'np_gratings_firing_rates.pkl'), 'rb') as f:
        #     target_firing_rates = pkl.load(f) # they are in Hz and divided by 1000 to make it in kHz and match the dt = 1 ms

        # for i, (key, value) in enumerate(target_firing_rates.items()):
        #     # identify tne ids that are included in value["ids"]
        #     neuron_ids = np.where(np.isin(networks['V1']["node_type_ids"], value["ids"]))[0]
        #     neuron_ids = tf.cast(neuron_ids, dtype=tf.int32)
        #     type_n_neurons = len(neuron_ids)
        #     sorted_target_rates = models.sample_firing_rates(value["rates"], type_n_neurons, flags.seed)
        #     target_firing_rates[key]['sorted_target_rates'] = tf.cast(sorted_target_rates, dtype=tf.float32) 

        # rec_weight_regularizer = losses.StiffRegularizer(flags.recurrent_weight_regularization, rsnn_layer.cell.recurrent_weight_values)
        # rec_weight_l2_regularizer = losses.L2Regularizer(flags.recurrent_weight_regularization, rsnn_layer.cell.recurrent_weight_values)

        rate_core_mask = None if flags.all_neuron_rate_loss else core_mask
        rate_distribution_regularizer = losses.SpikeRateDistributionTarget(network, flags.rate_cost, pre_delay=delays[0], post_delay=delays[1], 
                                                                            data_dir=flags.data_dir, core_mask=rate_core_mask, seed=flags.seed, dtype=dtype)
        # rate_distribution_regularizer = models.SpikeRateDistributionRegularization(target_firing_rates, flags.rate_cost)
        rate_loss = rate_distribution_regularizer(rsnn_layer.output[0][0])

        voltage_regularizer = losses.VoltageRegularization(rsnn_layer.cell, flags.voltage_cost, dtype=dtype, core_mask=core_mask)
        voltage_loss = voltage_regularizer(rsnn_layer.output[0][1]) 

        # Create an ExponentialMovingAverage object
        # Define the decay factor for the exponential moving average
        ema_decay = 0.95
        # Initialize exponential moving averages for V1 and LM firing rates
        if os.path.exists(os.path.join(logdir, 'train_end_data.pkl')):
            with open(os.path.join(logdir, 'train_end_data.pkl'), 'rb') as f:
                data_loaded = pkl.load(f)
                v1_ema = tf.Variable(data_loaded['v1_ema'], trainable=False, name='V1_EMA')
        else:
            # 3 Hz is near the average FR of cortex
            v1_ema = tf.Variable(0.003 * tf.ones(shape=(flags.neurons,)), trainable=False, name='V1_EMA')

        # here we need information of the layer mask for the OSI loss
        if flags.osi_loss_method == 'neuropixels_fr':
            layer_info = other_v1_utils.get_layer_info(network)
        else:
            layer_info = None

        OSI_DSI_Loss = losses.OrientationSelectivityLoss(network=network, osi_cost=flags.osi_cost, 
                                                    pre_delay=delays[0], post_delay=delays[1], 
                                                    dtype=dtype, core_mask=core_mask,
                                                    method=flags.osi_loss_method,
                                                    subtraction_ratio=flags.osi_loss_subtraction_ratio,
                                                    layer_info=layer_info)
        osi_dsi_loss = OSI_DSI_Loss(rsnn_layer.output[0][0], tf.constant(0, dtype=tf.float32, shape=(1,1)), trim=True, normalizer=v1_ema)[0] # this is just a placeholder
        # osi_dsi_loss = OSI_DSI_Loss(rsnn_layer.output[0][0], tf.constant(0, dtype=tf.float32, shape=(1,1)), trim=True) 

        model.add_loss(rate_loss)
        model.add_loss(voltage_loss)
        model.add_loss(osi_dsi_loss)
        model.add_metric(rate_loss, name='rate_loss')
        model.add_metric(voltage_loss, name='voltage_loss')
        model.add_metric(osi_dsi_loss, name='osi_dsi_loss')

        prediction_layer = model.get_layer('prediction')
        extractor_model = tf.keras.Model(inputs=model.inputs,
                                         outputs=[rsnn_layer.output, model.output, prediction_layer.output])

        # Loss from Guozhang classification task (unused in our case)
        # loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
        #     from_logits=False, reduction=tf.keras.losses.Reduction.NONE)
        # def compute_loss(_l, _p, _w):
        #     per_example_loss = loss_object(_l, _p, sample_weight=_w) * strategy.num_replicas_in_sync / tf.reduce_sum(_w)
        #     rec_weight_loss = rec_weight_regularizer(rsnn_layer.cell.recurrent_weight_values)
        #     return tf.nn.compute_average_loss(per_example_loss, global_batch_size=global_batch_size) + rec_weight_loss

        zero_state = rsnn_layer.cell.zero_state(flags.batch_size, dtype=dtype)
        state_variables = tf.nest.map_structure(lambda a: tf.Variable(
            a, trainable=False, synchronization=tf.VariableSynchronization.ON_READ
        ), zero_state)

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
        train_osi_dsi_loss = tf.keras.metrics.Mean()
        val_osi_dsi_loss = tf.keras.metrics.Mean()

        def reset_train_metrics():
            train_loss.reset_states(), train_accuracy.reset_states(), train_firing_rate.reset_states()
            train_rate_loss.reset_states(), train_voltage_loss.reset_states(), train_osi_dsi_loss.reset_states()

        def reset_validation_metrics():
            val_loss.reset_states(), val_accuracy.reset_states(), val_firing_rate.reset_states()
            val_rate_loss.reset_states(), val_voltage_loss.reset_states(), val_osi_dsi_loss.reset_states()


    def roll_out(_x, _y, _w, trim=True):
        _initial_state = tf.nest.map_structure(lambda _a: _a.read_value(), state_variables)
        seq_len = tf.shape(_x)[1]
        if _x.dtype == tf.bool:
            _x = tf.cast(_x, dtype)
        _out, _p, _ = extractor_model((_x, _initial_state))

        _z, _v, _ = _out[0]
        # update state_variables with the new model state
        new_state = tuple(_out[1:])
        tf.nest.map_structure(lambda a, b: a.assign(b), state_variables, new_state)
        # update the exponential moving average of the firing rates
        v1_evoked_rates = _z[:, delays[0]:seq_len-delays[1], :]
        v1_evoked_rates = tf.reduce_mean(v1_evoked_rates, (0, 1))
        v1_evoked_rates = tf.stop_gradient(v1_evoked_rates)
        # Update the EMAs
        v1_ema.assign(ema_decay * v1_ema + (1 - ema_decay) * v1_evoked_rates)
        # tf.print('V1_ema: ', tf.reduce_mean(v1_ema), tf.reduce_mean(v1_evoked_rates), v1_ema)

        voltage_loss = voltage_regularizer(_v)  # trim is irrelevant for this
        rate_loss = rate_distribution_regularizer(_z, trim)
        osi_dsi_loss = OSI_DSI_Loss(_z, _y, trim, normalizer=v1_ema)
        # tf.print(flags.osi_cost, osi_dsi_loss[0])
        # tf.print('V1 OSI losses: ')
        # tf.print(osi_dsi_loss)
        # weights_l2_regularizer = rec_weight_regularizer(rsnn_layer.cell.recurrent_weight_values)

        _aux = dict(rate_loss=rate_loss, voltage_loss=voltage_loss, osi_dsi_loss=osi_dsi_loss[0])
        _loss = osi_dsi_loss[0] + rate_loss + voltage_loss #+ weights_l2_regularizer
        # tf.print(osi_dsi_loss[0], rate_loss, voltage_loss) #, weights_l2_regularizer)

        return _out, _p, _loss, _aux

    @tf.function
    def distributed_roll_out(x, y, w, output_spikes=True):
        _out, _p, _loss, _aux = strategy.run(roll_out, args=(x, y, w))
        if output_spikes:
            return _out[0][0]
        else:
            return _out, _p, _loss, _aux


    def train_step(_x, _y, _w, trim=True, output_metrics=False):
        ### Forward propagation of the model
        with tf.GradientTape() as tape:
            _out, _p, _loss, _aux = roll_out(_x, _y, _w, trim=trim)

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

        with tf.control_dependencies([_op]):
            _op = train_osi_dsi_loss.update_state(_aux['osi_dsi_loss'])
        
        grad = tape.gradient(_loss, model.trainable_variables)
        for g, v in zip(grad, model.trainable_variables):
            # tf.print(f'{v.name} optimization')
            # tf.print('Loss, total_gradients : ', _loss, tf.reduce_sum(tf.math.abs(g)))
            with tf.control_dependencies([_op]):
                _op = optimizer.apply_gradients([(g, v)])

        if output_metrics:
            return [0., _loss, _rate, _aux['rate_loss'], _aux['voltage_loss'], _aux['osi_dsi_loss']]


    @tf.function
    def distributed_train_step(x, y, weights, trim, output_metrics=False):
        if output_metrics:
            _out = strategy.run(train_step, args=(x, y, weights, trim, output_metrics))
            return _out
        else:
            strategy.run(train_step, args=(x, y, weights, trim))  

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
        with tf.control_dependencies([_op]):
            _op = val_osi_dsi_loss.update_state(_aux['osi_dsi_loss'])
            
        # tf.nest.map_structure(lambda _a, _b: _a.assign(_b), list(state_variables), _out[1:])
        if output_spikes:
            return _out[0][0]


    @tf.function
    def distributed_validation_step(x, y, weights, output_spikes=True):
        if output_spikes:
            return strategy.run(validation_step, args=(x, y, weights, output_spikes))
        else:
            strategy.run(validation_step, args=(x, y, weights))

    ### LGN INPUT ###
    # Define the function that generates the dataset for our task
    def get_dataset_fn(regular=False):
        def _f(input_context):
            _data_set = stim_dataset.generate_drifting_grating_tuning(
                seq_len=flags.seq_len,
                pre_delay=delays[0],
                post_delay=delays[1],
                n_input=flags.n_input,
                data_dir=flags.data_dir,
                regular=regular,
                bmtk_compat=flags.bmtk_compat_lgn,
                rotation=flags.rotation,
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
                data_dir=flags.data_dir,
                rotation=flags.rotation,
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

    @tf.function
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

    def get_next_chunknum(chunknum, seq_len, direction='up'):
        # get the next chunk number (diviser) for seq_len.
        if direction == 'up':
            chunknum += 1
            # check if it is a valid diviser
            while seq_len % chunknum != 0:
                chunknum += 1
                if chunknum >= seq_len:
                    print('Chunk number reached seq_len')
                    return seq_len
        elif direction == 'down':
            chunknum -= 1
            while seq_len % chunknum != 0:
                chunknum -= 1
                if chunknum <= 1:
                    print('Chunk number reached 1')
                    return 1
        else:
            raise ValueError(f"Invalid direction: {direction}")
        return chunknum

    ############################ TRAINING #############################

    stop = False
    # Initialize your callbacks
    metric_keys = ['train_accuracy', 'train_loss', 'train_firing_rate', 'train_rate_loss',
            'train_voltage_loss', 'train_osi_dsi_loss', 'val_accuracy', 'val_loss',
            'val_firing_rate', 'val_rate_loss', 'val_voltage_loss', 'val_osi_dsi_loss']
    
    callbacks = Callbacks(network, lgn_input, bkg_input, model, optimizer, distributed_roll_out, flags, logdir, strategy, 
                        metric_keys, pre_delay=delays[0], post_delay=delays[1], model_variables_init=model_variables_dict,
                        checkpoint=checkpoint)
    
    callbacks.on_train_begin()
    chunknum = 1
    max_working_fr = {}   # defined for each chunknum
    n_prev_epochs = flags.run_session * flags.n_epochs
    for epoch in range(n_prev_epochs, n_prev_epochs + flags.n_epochs):
        callbacks.on_epoch_start()  
        # Reset the model state to the gray state    
        gray_state = distributed_reset_state('gray')  
        
        # Load the dataset iterator - this must be done inside the epoch loop
        it = iter(train_data_set)

        # tf.profiler.experimental.start(logdir=logdir)
        for step in range(flags.steps_per_epoch):
            callbacks.on_step_start()
            # try resetting every iteration
            if flags.reset_every_step:
                distributed_reset_state('gray')
            else:
                distributed_reset_state('gray', gray_state=gray_state)

            x, y, _, w = next(it) # x dtype tf.bool
    
            # with tf.profiler.experimental.Trace('train', step_num=step, _r=1):
            while True:
                try:
                    x_chunks = tf.split(x, chunknum, axis=1)
                    seq_len_local = x.shape[1] // chunknum
                    for j in range(chunknum):
                        x_chunk = x_chunks[j]
                        step_values = distributed_train_step(x_chunk, y, w, trim=chunknum==1, output_metrics=True)
                    # distributed_train_step(x, y, w, trim=chunknum==1)
                    break
                except tf.errors.ResourceExhaustedError as e:
                    print("OOM error occured")
                    import gc
                    gc.collect()
                    # increase the chunknum
                    chunknum = get_next_chunknum(chunknum, flags.seq_len, direction='up')
                    print("Increasing chunknum to: ", chunknum)
                    print("BPTT truncation: ", flags.seq_len / chunknum)
                    
            # update max working fr for the chunk num
            current_fr = step_values[2].numpy()
            if chunknum not in max_working_fr:
                max_working_fr[chunknum] = current_fr
            else:
                max_working_fr[chunknum] = max(max_working_fr[chunknum], current_fr)
            # determine if the chunknum should be decreased
            if chunknum > 1:
                chunknum_down = get_next_chunknum(chunknum, flags.seq_len, direction='down')
                if chunknum_down in max_working_fr:
                    if current_fr < max_working_fr[chunknum_down]:
                        chunknum = chunknum_down
                        print("Decreasing chunknum to: ", chunknum)
                        print(current_fr, max_working_fr)
                        print(max_working_fr)
                else:  # data not available, estimate from the current one.
                    fr_ratio = current_fr / max_working_fr[chunknum]
                    chunknum_ratio = chunknum_down / chunknum
                    print(current_fr, max_working_fr, fr_ratio, chunknum_ratio)
                    if fr_ratio < chunknum_ratio:  # potentially good to decrease
                        chunknum = chunknum_down
                        print("Tentatively decreasing chunknum to: ", chunknum)
                
            callbacks.on_step_end(step_values, y, verbose=True)

        # tf.profiler.experimental.stop() 

        # ## VALIDATION AFTER EACH EPOCH
        # test_it = iter(test_data_set)
        test_it = it
        for step in range(flags.val_steps):
            x, y, _, w = next(test_it)
            distributed_reset_state('gray', gray_state=gray_state)
            z = distributed_validation_step(x, y, w, output_spikes=True) 

        train_values = [a.result().numpy() for a in [train_accuracy, train_loss, train_firing_rate, 
                                                    train_rate_loss, train_voltage_loss, train_osi_dsi_loss]]
        val_values = train_values
        # val_values = [a.result().numpy() for a in [val_accuracy, val_loss, val_firing_rate, 
        #                                            val_rate_loss, val_voltage_loss, val_osi_dsi_loss]]
        metric_values = train_values + val_values

        # if the model train loss is minimal, save the model.
        stop = callbacks.on_epoch_end(x, z, y, metric_values, verbose=True)

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
    if hostname.count('alleninstitute') > 0 or re.search(r'n\d{3}', hostname) is not None:
        _data_dir = '/allen/programs/mindscope/workgroups/realistic-model/shinya.ito/tensorflow_new/V1_GLIF_model/GLIF_network'
        _results_dir = '/allen/programs/mindscope/workgroups/realistic-model/shinya.ito/tensorflow_new/V1_GLIF_model/Simulation_results'
    else: 
        _data_dir = '/home/jgalvan/Desktop/Neurocoding/V1_GLIF_model/GLIF_network'
        _results_dir = '/home/jgalvan/Desktop/Neurocoding/V1_GLIF_model/Simulation_results'

    absl.app.flags.DEFINE_string('data_dir', _data_dir, '')
    absl.app.flags.DEFINE_string('results_dir', _results_dir, '')
    absl.app.flags.DEFINE_string('task_name', 'drifting_gratings_firing_rates_distr' , '')
    
    # absl.app.flags.DEFINE_string('restore_from', '', '')
    # absl.app.flags.DEFINE_string('restore_from', '../results/multi_training/b_53dw/results/ckpt-49', '')
    absl.app.flags.DEFINE_string('restore_from', '', '')
    absl.app.flags.DEFINE_string('comment', '', '')
    absl.app.flags.DEFINE_string('delays', '100,0', '')
    # absl.app.flags.DEFINE_string('neuron_model', 'GLIF3', '')
    absl.app.flags.DEFINE_string('scale', '2,2', '')

    absl.app.flags.DEFINE_string('optimizer', 'adam', '')
    absl.app.flags.DEFINE_float('learning_rate', .01, '')
    absl.app.flags.DEFINE_float('rate_cost', 100., '')
    # absl.app.flags.DEFINE_float('voltage_cost', .00001, '')
    absl.app.flags.DEFINE_float('voltage_cost', 1., '')
    absl.app.flags.DEFINE_float('osi_cost', 1., '')
    absl.app.flags.DEFINE_string('osi_loss_method', 'crowd_osi', '')
    absl.app.flags.DEFINE_float('osi_loss_subtraction_ratio', 1., '')
    absl.app.flags.DEFINE_string('rotation', 'ccw', '')
    absl.app.flags.DEFINE_float('dampening_factor', .5, '')
    absl.app.flags.DEFINE_float("recurrent_dampening_factor", 0.5, "")
    absl.app.flags.DEFINE_float('input_weight_scale', 1., '')
    absl.app.flags.DEFINE_float('gauss_std', .3, '')
    absl.app.flags.DEFINE_float('recurrent_weight_regularization', 0., '')
    absl.app.flags.DEFINE_float('lr_scale', 1., '')
    # absl.app.flags.DEFINE_float('p_reappear', .5, '')
    absl.app.flags.DEFINE_float('max_time', -1, '')
    # absl.app.flags.DEFINE_float('max_time', 0.05, '')
    # absl.app.flags.DEFINE_float('scale_w_e', -1, '')
    # absl.app.flags.DEFINE_float('sti_intensity', 2., '')
    absl.app.flags.DEFINE_float('temporal_f', 2., '')

    absl.app.flags.DEFINE_integer('n_runs', 1, '')
    absl.app.flags.DEFINE_integer('run_session', 0, '')
    absl.app.flags.DEFINE_integer('n_epochs', 50, '')
    absl.app.flags.DEFINE_integer('osi_dsi_eval_period', 50, '') # number of epochs for osi/dsi evaluation if n_runs = 1
    absl.app.flags.DEFINE_integer('batch_size', 1, '')
    absl.app.flags.DEFINE_integer('neurons', 65871, '')
    absl.app.flags.DEFINE_integer("n_input", 17400, "")  
    absl.app.flags.DEFINE_integer('seq_len', 600, '')
    # absl.app.flags.DEFINE_integer('im_slice', 100, '')
    absl.app.flags.DEFINE_integer('seed', 3000, '')
    # absl.app.flags.DEFINE_integer('port', 12778, '')
    absl.app.flags.DEFINE_integer("n_output", 2, "")
    absl.app.flags.DEFINE_integer('neurons_per_output', 16, '')
    absl.app.flags.DEFINE_integer('steps_per_epoch', 20, '')# EA and garret dose not need this many but pure classification needs 781 = int(50000/64)
    absl.app.flags.DEFINE_integer('val_steps', 1, '')# EA and garret dose not need this many but pure classification needs 156 = int(10000/64)
    # absl.app.flags.DEFINE_integer('max_delay', 5, '')
    # absl.app.flags.DEFINE_integer('n_plots', 1, '')

    # absl.app.flags.DEFINE_integer('pre_chunks', 3, '')
    # absl.app.flags.DEFINE_integer('post_chunks', 8, '') # the pure classification task only need 1 but to make consistent with other tasks one has to make up here
    # absl.app.flags.DEFINE_integer('pre_delay', 50, '')
    # absl.app.flags.DEFINE_integer('post_delay', 450, '')

    # absl.app.flags.DEFINE_boolean('use_rand_connectivity', False, '')
    # absl.app.flags.DEFINE_boolean('use_uniform_neuron_type', False, '')
    # absl.app.flags.DEFINE_boolean('use_only_one_type', False, '')
    # absl.app.flags.DEFINE_boolean('use_dale_law', True, '')
    absl.app.flags.DEFINE_boolean('caching', True, '') # if one wants to use caching, remember to update the caching function
    absl.app.flags.DEFINE_boolean('core_only', False, '')  # a little confusing.
    absl.app.flags.DEFINE_boolean('core_loss', False, '')  # not used. should be retired.
    absl.app.flags.DEFINE_boolean('all_neuron_rate_loss', False, '')  # whethre you want to enforce rate loss to all neurons
    absl.app.flags.DEFINE_float('loss_core_radius', 400.0, '') # 0 is not using core loss
    absl.app.flags.DEFINE_float('plot_core_radius', 400.0, '') # 0 is not using core plot

    # absl.app.flags.DEFINE_boolean('train_input', True, '')
    absl.app.flags.DEFINE_boolean('train_input', False, '')
    absl.app.flags.DEFINE_boolean('train_noise', False, '')
    absl.app.flags.DEFINE_boolean('train_recurrent', True, '')
    absl.app.flags.DEFINE_boolean('train_recurrent_per_type', False, '')
    absl.app.flags.DEFINE_boolean('connected_selection', True, '')
    absl.app.flags.DEFINE_boolean('neuron_output', False, '')
    # absl.app.flags.DEFINE_boolean('localized_readout', True, '')
    # absl.app.flags.DEFINE_boolean('current_input', True, '')
    # absl.app.flags.DEFINE_boolean('use_rand_ini_w', True, '')
    # absl.app.flags.DEFINE_boolean('use_decoded_noise', True, '')
    # absl.app.flags.DEFINE_boolean('from_lgn', True, '')
    absl.app.flags.DEFINE_boolean("float16", False, "")

    absl.app.flags.DEFINE_integer("cue_duration", 40, "")
    absl.app.flags.DEFINE_boolean("hard_reset", False, "")
    absl.app.flags.DEFINE_boolean("pseudo_gauss", False, "")
    absl.app.flags.DEFINE_boolean("bmtk_compat_lgn", True, "")
    absl.app.flags.DEFINE_boolean("reset_every_step", False, "")

    absl.app.flags.DEFINE_string('ckpt_dir', '', '')

    absl.app.run(main)
