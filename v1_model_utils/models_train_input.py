import numpy as np
import tensorflow as tf
import psutil
import os 
import pickle as pkl
from time import time
# from memory_profiler import profile
from pympler.asizeof import asizeof, asized
from numba import njit


# print("TensorFlow version:", tf.__version__) # 2.4.1
import subprocess

class GPUMemoryTracker:
    def __init__(self):
        result = subprocess.run(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,nounits,noheader'],
                                stdout=subprocess.PIPE, encoding='utf-8') # MiB
        self.previous_used = float(result.stdout.strip())
    
    def get_gpu_memory(self):
        # Function to get the allocated, free and total memory of a GPU
        result = subprocess.run(['nvidia-smi', '--query-gpu=memory.used,memory.free,memory.total', '--format=csv,nounits,noheader'],
                                stdout=subprocess.PIPE, encoding='utf-8') # MiB
        used, free, total = [float(x) for x in result.stdout.strip().split(',')]
        
        increase = used - self.previous_used
        self.previous_used = used
        
        tf.print("---- GPU Memory ----")
        tf.print(f"  Total: {round(total / 1024, 2)} GiB")
        tf.print(f"  Available: {round(free / 1024, 2)} GiB")
        tf.print(f"  Used: {round(used / 1024, 2)} GiB")
        tf.print(f"  Increase: {round(increase / 1024, 2)} GiB")
        tf.print('')


# Define a custom gradient for the spike function.
# Diverse functions can be used to define the gradient.
# Here we provide variations of this functions depending on
# the gradient type and the precision of the input tensor.

def gauss_pseudo(v_scaled, sigma, amplitude):
    return tf.math.exp(-tf.square(v_scaled) / tf.square(sigma)) * amplitude


def pseudo_derivative(v_scaled, dampening_factor):
    return dampening_factor * tf.maximum(1 - tf.abs(v_scaled), 0)


# def slayer_pseudo(v_scaled, sigma, amplitude):
#     return tf.math.exp(-sigma * tf.abs(v_scaled)) * amplitude


@tf.custom_gradient
def spike_gauss(v_scaled, sigma, amplitude):
    z_ = tf.greater(v_scaled, 0.0)
    z_ = tf.cast(z_, tf.float32)

    def grad(dy):
        de_dz = dy
        dz_dv_scaled = gauss_pseudo(v_scaled, sigma, amplitude)
        de_dv_scaled = de_dz * dz_dv_scaled

        return [de_dv_scaled, tf.zeros_like(sigma), tf.zeros_like(amplitude)]

    return tf.identity(z_, name="spike_gauss"), grad


@tf.custom_gradient
def spike_gauss_16(v_scaled, sigma, amplitude):
    z_ = tf.greater(v_scaled, 0.0)
    z_ = tf.cast(z_, tf.float16)

    def grad(dy):
        de_dz = dy
        dz_dv_scaled = gauss_pseudo(v_scaled, sigma, amplitude)

        de_dv_scaled = de_dz * dz_dv_scaled

        return [de_dv_scaled, tf.zeros_like(sigma), tf.zeros_like(amplitude)]

    return tf.identity(z_, name="spike_gauss"), grad


# @tf.custom_gradient
# def spike_slayer(v_scaled, sigma, amplitude):
#     z_ = tf.greater(v_scaled, 0.0)
#     z_ = tf.cast(z_, tf.float32)

#     def grad(dy):
#         de_dz = dy
#         dz_dv_scaled = slayer_pseudo(v_scaled, sigma, amplitude)

#         de_dv_scaled = de_dz * dz_dv_scaled

#         return [de_dv_scaled, tf.zeros_like(sigma), tf.zeros_like(amplitude)]

#     return tf.identity(z_, name="spike_slayer"), grad


@tf.custom_gradient
def spike_function(v_scaled, dampening_factor):
    z_ = tf.greater(v_scaled, 0.0)
    z_ = tf.cast(z_, tf.float32)

    def grad(dy):
        de_dz = dy
        dz_dv_scaled = pseudo_derivative(v_scaled, dampening_factor)

        de_dv_scaled = de_dz * dz_dv_scaled

        return [de_dv_scaled, tf.zeros_like(dampening_factor)]

    return tf.identity(z_, name="spike_function"), grad


@tf.custom_gradient
def spike_function_16(v_scaled, dampening_factor):
    z_ = tf.greater(v_scaled, 0.0)
    z_ = tf.cast(z_, tf.float16)

    def grad(dy):
        de_dz = dy
        dz_dv_scaled = pseudo_derivative(v_scaled, dampening_factor)

        de_dv_scaled = de_dz * dz_dv_scaled

        return [de_dv_scaled, tf.zeros_like(dampening_factor)]

    return tf.identity(z_, name="spike_function"), grad


@tf.custom_gradient
def spike_function_b16(v_scaled, dampening_factor):
    z_ = tf.greater(v_scaled, 0.0)
    z_ = tf.cast(z_, tf.bfloat16)

    def grad(dy):
        de_dz = dy
        dz_dv_scaled = pseudo_derivative(v_scaled, dampening_factor)

        de_dv_scaled = de_dz * dz_dv_scaled

        return [de_dv_scaled, tf.zeros_like(dampening_factor)]

    return tf.identity(z_, name="spike_function"), grad


def exp_convolve(tensor, decay=0.8, reverse=False, initializer=None, axis=0):
    rank = len(tensor.get_shape())
    perm = np.arange(rank)
    perm[0], perm[axis] = perm[axis], perm[0]
    tensor = tf.transpose(tensor, perm)

    if initializer is None:
        initializer = tf.zeros_like(tensor[0])

    def scan_fun(_acc, _t):
        return _acc * decay + _t

    filtered = tf.scan(scan_fun, tensor, reverse=reverse,
                       initializer=initializer)

    filtered = tf.transpose(filtered, perm)
    return filtered


class BackgroundNoiseLayer(tf.keras.layers.Layer):
    def __init__(self, indices, weights, dense_shape, # tau_syn_weights, 
                 synaptic_weights, syn_ids, batch_size, seq_len,
                 lr_scale=1., bkg_firing_rate=250, n_bkg_units=100, 
                 dtype=tf.float32, **kwargs):
        super().__init__(**kwargs)
        self._dtype = dtype
        self._bkg_weights = weights
        self._bkg_indices = indices
        self._dense_shape = dense_shape
        # self._tau_syn_weights = tau_syn_weights
        self._batch_size = batch_size
        self._seq_len = seq_len
        self._n_syn_basis = synaptic_weights.shape[1]
        self._lr_scale = lr_scale
        self._bkg_firing_rate = bkg_firing_rate
        self._n_bkg_units = n_bkg_units
        self.bkg_input_weights_factors = synaptic_weights[syn_ids]

    # @tf.function
    def calculate_bkg_i_in(self, inputs):
        # This function performs the tensor multiplication to calculate the recurrent currents at each timestep
        i_in = tf.TensorArray(dtype=self._compute_dtype, size=self._n_syn_basis)
        for r_id in range(self._n_syn_basis):
            weights_syn_receptors = self._bkg_weights * self.bkg_input_weights_factors[:, r_id]
            sparse_w_in = tf.sparse.SparseTensor(
                self._bkg_indices,
                tf.cast(weights_syn_receptors, self._compute_dtype), 
                self._dense_shape,
            )
            i_receptor = tf.sparse.sparse_dense_matmul(
                                                        sparse_w_in,
                                                        inputs,
                                                        adjoint_b=True
                                                    )
            # Append i_receptor to the TensorArray
            i_in = i_in.write(r_id, i_receptor)
        # Stack the TensorArray into a single tensor
        i_in = i_in.stack()
        return i_in

    def call(self, inp): # inp only provides the shape
        t0 = time()   
        tf.print(' ------------- BKG - call --------------')
        tf.print(inp.shape)
        # Generate the background spikes
        rest_of_brain = tf.cast(tf.random.uniform(
                (self._batch_size, self._seq_len, self._n_bkg_units)) < self._bkg_firing_rate * .001, 
                self._compute_dtype)
        rest_of_brain = tf.reshape(rest_of_brain, (self._batch_size * self._seq_len, self._n_bkg_units)) # (batch_size*sequence_length, input_dim)
        # Create a TensorArray to save the results for every receptor type
        noise_input = self.calculate_bkg_i_in(tf.cast(rest_of_brain, self._compute_dtype))
        noise_input = tf.transpose(noise_input) # New shape (3000, 66634, 5)
        # Reshape properly the input current
        noise_input = tf.reshape(noise_input, (self._batch_size, self._seq_len, -1)) # (1, 3000, 333170)
        tf.print('>>> BKG sparse layer processing time: ', time()-t0)

        return noise_input
    

# class BackgroundNoiseLayer(tf.keras.layers.Layer):
#     def __init__(self, indices, weights, dense_shape, # tau_syn_weights, 
#                  synaptic_weights, syn_ids, batch_size, seq_len,
#                  lr_scale=1., bkg_firing_rate=250, n_bkg_units=100, 
#                  dtype=tf.float32, **kwargs):
#         super().__init__(**kwargs)
#         self._dtype = dtype
#         self._bkg_weights = weights
#         self._bkg_indices = indices
#         self._dense_shape = dense_shape
#         # self._tau_syn_weights = tau_syn_weights
#         self._batch_size = batch_size
#         self._seq_len = seq_len
#         self._n_syn_basis = synaptic_weights.shape[1]
#         self._lr_scale = lr_scale
#         self._bkg_firing_rate = bkg_firing_rate
#         self._n_bkg_units = n_bkg_units
#         # self.bkg_input_weights_factors = synaptic_weights[syn_ids]
#         self.bkg_input_weights_factors = tf.constant(synaptic_weights[syn_ids], dtype=self._compute_dtype)
#         # self.prepare_sparse_weight()

#     def prepare_sparse_weight(self, bkg_weights, verbose=False):
#         sparse_w_bkg_list = []
#         print("bkg_weights: ", bkg_weights)
#         print("bkg_input_weights_factors: ", self.bkg_input_weights_factors)
#         if verbose:
#             tf.print("preparing the input sparse weight")
#         for r_id in tf.range(self._n_syn_basis):
#             weights_syn_receptors = bkg_weights * self.bkg_input_weights_factors[:, r_id]
#             sparse_w_bkg_list.append(weights_syn_receptors)

#         return sparse_w_bkg_list     

#     def calculate_bkg_i_in(self, inputs, weights):
#         # This function performs the tensor multiplication to calculate the recurrent currents at each timestep
#         i_in = tf.TensorArray(dtype=self._compute_dtype, size=self._n_syn_basis)
#         for r_id in range(self._n_syn_basis): # ? can this be a tf.range?
#             sparse_w_bkg = tf.sparse.SparseTensor(
#                 self._bkg_indices,
#                 weights[r_id], 
#                 self._dense_shape,
#             )
#             i_receptor = tf.sparse.sparse_dense_matmul(
#                                                         sparse_w_bkg,
#                                                         inputs,
#                                                         adjoint_b=True
#                                                     )
#             i_in = i_in.write(r_id, i_receptor)
#         i_in = i_in.stack()
#         return i_in

#     def generate_bkg_noise(self):
#         # Generate the background spikes
#         rest_of_brain = tf.cast(tf.random.uniform((self._batch_size * self._seq_len, self._n_bkg_units)) < self._bkg_firing_rate * 0.001, self._compute_dtype)
#         return rest_of_brain

#     # @tf.function
#     def call(self, inp): # inp only provides the shape
#         t0 = time()
#         # Generate the background spikes
#         rest_of_brain = self.generate_bkg_noise()
#         # Update the weights
#         print('Okey')
#         bkg_weights = self.prepare_sparse_weight(self._bkg_weights)
#         tf.print('Okey')
#         # Calculate the input current
#         noise_input = self.calculate_bkg_i_in(rest_of_brain, bkg_weights) 
#         noise_input = tf.transpose(noise_input) # New shape (3000, 66634, 5)
#         # Reshape properly the input current
#         noise_input = tf.reshape(noise_input, (self._batch_size, self._seq_len, -1)) # (1, 3000, 333170)
#         tf.print('>>> BKG sparse layer processing time: ', time()-t0)

#         return noise_input


class SparseLayer(tf.keras.layers.Layer):
    def __init__(self, indices, weights, dense_shape, synaptic_weights, syn_ids, # tau_syn_weights,
                 n_neurons, lr_scale=1.0, dtype=tf.float32, **kwargs,):
        super().__init__(**kwargs)
        self._dtype = dtype
        self._lr_scale = lr_scale
        self._indices = indices
        self._input_weights = weights
        # self._tau_syn_weights = tau_syn_weights
        self._n_syn_basis = synaptic_weights.shape[1] # 5
        self._dense_shape = dense_shape
        self._n_neurons = n_neurons

        self.input_weights_factors = synaptic_weights[syn_ids]

        # Define a threshold that determines whether to compute the sparse
        # matrix multiplication directly or split it into smaller batches in a GPU.
        # The value is calculated to ensure that output.shape[1] * nnz(a) > 2^31, 
        # where output.shape[1] is the time_length, and nnz(a) is the number of non-zero elements in the sparse matrix.
        nnz_sparse_matrix = self._indices.shape[0]
        self._max_batch = int(2**31 / nnz_sparse_matrix)


    # @tf.function(input_signature=[tf.TensorSpec(shape=[None, None], dtype=tf.int16)])
    # @tf.function
    def calculate_i_in(self, inputs):
        # This function performs the tensor multiplication to calculate the recurrent currents at each timestep
        i_in = tf.TensorArray(dtype=self._compute_dtype, size=self._n_syn_basis)
        for r_id in range(self._n_syn_basis):
            weights_syn_receptors = self._input_weights * self.input_weights_factors[:, r_id]
            sparse_w_in = tf.sparse.SparseTensor(
                self._indices,
                tf.cast(weights_syn_receptors, self._compute_dtype), 
                self._dense_shape,
            )
            i_receptor = tf.sparse.sparse_dense_matmul(
                                                        sparse_w_in,
                                                        tf.cast(inputs, self._compute_dtype),
                                                        adjoint_b=True
                                                    )
            # Append i_receptor to the TensorArray
            i_in = i_in.write(r_id, i_receptor)
        # Stack the TensorArray into a single tensor
        i_in = i_in.stack()
        return i_in

    def call(self, inp):
        t0 = time()
        # replace any None values in the shape of inp with the actual values obtained
        # from the input tensor at runtime (tf.shape(inp)).
        # This is necessary because the SparseTensor multiplication operation requires
        # a fully defined shape.
        inp_shape = inp.get_shape().as_list()
        shp = [dim if dim is not None else tf.shape(inp)[i] for i, dim in enumerate(inp_shape)]
        batch_size = shp[0] * shp[1]
        print(f"The ratio of the current input batch size to the maximum batch size is {batch_size}/{self._max_batch}")
        inp = tf.cast(inp, self._compute_dtype)
        inp = tf.reshape(inp, (shp[0] * shp[1], shp[2])) # (batch_size*sequence_length, input_dim)

        if shp[0] * shp[1] < self._max_batch:
            # the sparse tensor multiplication can be directly performed
            print('Processing input tensor directly.')
            input_current = self.calculate_i_in(tf.cast(inp, self._compute_dtype)) # (5, 1000, 600)
            input_current = tf.transpose(input_current) # (600, 1000, 5)

        else: 
            # Define the current batch size and calculate the number of chunks
            batch_size = tf.shape(inp)[0]
            if batch_size % self._max_batch == 0:
                num_chunks = int(batch_size / self._max_batch)
                num_pad_elements = 0
            else: # add 1 chunk if the quotient is not an integer
                num_chunks = int(batch_size / self._max_batch) + 1
                # Padd the input with 0's to ensure all chunks have the same size for the matrix multiplication
                num_pad_elements = num_chunks * self._max_batch - tf.shape(inp)[0]
                inp = tf.pad(inp, [(0, num_pad_elements), (0, 0)])

            print(f'Chunking input tensor into {num_chunks} batches.')
            inp = tf.cast(inp, self._compute_dtype)

            # Initialize a tensor array to hold the partial results of every chunk
            result_array = tf.TensorArray(dtype=self._compute_dtype, size=num_chunks)
            # Iterate over the chunks
            for i in tf.range(num_chunks):
                start_idx = int(i * self._max_batch)
                end_idx = int((i + 1) * self._max_batch)
                chunk = inp[start_idx:end_idx, :]
                chunk = tf.reshape(chunk, (self._max_batch, -1))
                partial_input_current = self.calculate_i_in(tf.cast(chunk, self._compute_dtype))  # ( 5, 66634, 68)
                # Store the partial result in the tensor array     
                result_array = result_array.write(i, partial_input_current)
            
            # Concatenate the partial results to get the final result
            result_array = result_array.stack() # ( 9, 5, 66634, 68)
            result_array = tf.transpose(result_array, perm=[1, 2, 3, 0]) # New shape (5, 66634, 68, 9)
            result_array = tf.reshape(result_array, (self._n_syn_basis, -1, num_chunks * self._max_batch)) # New shape (5, 66634, 612)
            result_array = tf.transpose(result_array, perm=[2, 1, 0]) # New shape (612, 66634, 5)
            
            if num_pad_elements > 0: # Remove the padded 0's
                result_array = result_array[:-num_pad_elements, :] # New shape (600, 66634, 5)
            input_current = tf.cast(result_array, self._compute_dtype)

        # Reshape properly the input current
        input_current = tf.reshape(input_current, (shp[0], shp[1], -1)) # New shape (1, 3000, 333170)
        print('>>> Input sparse layer processing time: ', time()-t0)

        return input_current


class SignedConstraint(tf.keras.constraints.Constraint):
    def __init__(self, positive):
        self._positive = positive

    def __call__(self, w):
        sign_corrected_w = tf.where(
            self._positive, tf.nn.relu(w), -tf.nn.relu(-w))
        return sign_corrected_w


class SparseSignedConstraint(tf.keras.constraints.Constraint):
    def __init__(self, mask, positive):
        self._mask = mask
        self._positive = positive

    def __call__(self, w):
        sign_corrected_w = tf.where(
            self._positive, tf.nn.relu(w), -tf.nn.relu(-w))
        return tf.where(self._mask, sign_corrected_w, tf.zeros_like(sign_corrected_w))


class StiffRegularizer(tf.keras.regularizers.Regularizer):
    def __init__(self, strength, initial_value):
        super().__init__()
        self._strength = strength
        self._initial_value = tf.Variable(initial_value, trainable=False)

    def __call__(self, x):
        return self._strength * tf.reduce_sum(tf.square(x - self._initial_value))

# @profile
class V1Column(tf.keras.layers.Layer):
    # @profile
    def __init__(
        self,
        network,
        lgn_input,
        bkg_input,
        dt=1.0,
        gauss_std=0.5,
        dampening_factor=0.3,
        recurrent_dampening_factor=0.4,
        input_weight_scale=1.0,
        recurrent_weight_scale=1.0,
        lr_scale=1.0,
        spike_gradient=False,
        max_delay=5,
        pseudo_gauss=False,
        train_recurrent=True,
        train_input=True,
        hard_reset=True,
    ):
        super().__init__()
        self._params = network["node_params"]
        # Rescale the voltages to have them near 0, as we wanted the effective step size
        # for the weights to be normalized when learning (weights are scaled similarly)
        voltage_scale = self._params["V_th"] - self._params["E_L"]
        voltage_offset = self._params["E_L"]
        self._params["V_th"] = (self._params["V_th"] - voltage_offset) / voltage_scale
        self._params["E_L"] = (self._params["E_L"] - voltage_offset) / voltage_scale
        self._params["V_reset"] = (self._params["V_reset"] - voltage_offset) / voltage_scale
        self._params["asc_amps"] = (self._params["asc_amps"] / voltage_scale[..., None])  # _params['asc_amps'] has shape (111, 2)
        # Define the other model variables
        self._node_type_ids = network["node_type_ids"]
        self._n_syn_basis = 5
        self._dt = dt
        self._recurrent_dampening = recurrent_dampening_factor
        self._pseudo_gauss = pseudo_gauss
        self._lr_scale = lr_scale
        self._spike_gradient = spike_gradient
        self._hard_reset = hard_reset
        self._n_neurons = network["n_nodes"]
        self._dampening_factor = tf.cast(dampening_factor, self._compute_dtype)
        self._gauss_std = tf.cast(gauss_std, self._compute_dtype)
        # Determine the membrane time decay constant
        tau = (self._params["C_m"] / self._params["g"])
        self._decay = np.exp(-dt / tau)
        self._current_factor = 1 / self._params["C_m"] * (1 - self._decay) * tau
        # Determine the synaptic dynamic parameters for each of the 5 basis receptors.
        path='GLIF_network/synaptic_data/tau_basis.npy'
        self._tau_syn = np.load(path)
        self.syn_decay = np.exp(-dt / self._tau_syn)
        self.psc_initial = np.e / self._tau_syn

        # Find the maximum delay in the network
        self.max_delay = int(np.round(np.min([np.max(network["synapses"]["delays"]), max_delay])))
        # Define the state size of the network
        self.state_size = (
            self._n_neurons * self.max_delay,  # z buffer
            self._n_neurons,  # v
            self._n_neurons,  # r
            self._n_neurons,  # asc 1
            self._n_neurons,  # asc 2
            self._n_neurons * self._n_syn_basis,  # psc rise
            self._n_neurons * self._n_syn_basis,  # psc
        )

        def _f(_v, trainable=False):
            return tf.Variable(
                tf.cast(self._gather(_v), self._compute_dtype), trainable=trainable
            )

        def inv_sigmoid(_x):
            return tf.math.log(_x / (1 - _x))

        def custom_val(_v, trainable=False):
            _v = tf.Variable(
                tf.cast(inv_sigmoid(self._gather(_v)), self._compute_dtype),
                trainable=trainable,
            )

            def _g():
                return tf.nn.sigmoid(_v.read_value())

            return _v, _g

        self.t_ref = _f(self._params["t_ref"])  # refractory time
        self.v_reset = _f(self._params["V_reset"])
        self.asc_amps = _f(self._params["asc_amps"], trainable=False)
        _k = self._params["k"]
        # inverse sigmoid of the adaptation rate constant (1/ms)
        self.param_k, self.param_k_read = custom_val(_k, trainable=False)
        self.k = self.param_k_read()
        self.v_th = _f(self._params["V_th"])
        self.e_l = _f(self._params["E_L"])
        self.normalizer = self.v_th - self.e_l
        self.param_g = _f(self._params["g"])
        self.gathered_g = self.param_g * self.e_l
        self.decay = _f(self._decay)
        self.current_factor = _f(self._current_factor)
        self.voltage_scale = _f(voltage_scale)
        self.voltage_offset = _f(voltage_offset)
        # self.recurrent_weights = None

        # Find the synaptic basis representation for each neuron
        path = os.path.join('GLIF_network', 'syn_id_to_syn_weights_dict.pkl')
        with open(path, "rb") as f:
            syn_id_to_syn_weights_dict = pkl.load(f)
        synaptic_weights = np.array([syn_id_to_syn_weights_dict[i] for i in range(len(syn_id_to_syn_weights_dict))])
        # This should be a tensor
        self.synaptic_weights = synaptic_weights

        ### Network recurrent connectivity ###
        indices, weights, dense_shape, syn_ids = (
            network["synapses"]["indices"],
            network["synapses"]["weights"],
            network["synapses"]["dense_shape"],
            network["synapses"]["syn_ids"]
        )
        # Scale down the recurrent weights
        weights = (weights/voltage_scale[self._node_type_ids[indices[:, 0]]]).astype(np.float32)
        
        # recurrent_weights_factors = self.synaptic_weights[syn_ids] # syn_ids.shape = n_synapses, synaptic_weights.shape = (79, n_syn_basis=5), 
        # recurrent_weights_factors.shape = (n_synapses, n_syn_basis=5)

        # Use the maximum delay to clip the synaptic delays
        delays = np.round(np.clip(network["synapses"]["delays"], dt, self.max_delay)/dt).astype(np.int32)
        dense_shape = dense_shape[0], self.max_delay * dense_shape[1]
        # Notice that in dense_shape, the first column (presynaptic neuron) has size n_neurons
        # and the second column (postsynaptic neuron) has size max_delay*n_neurons

        # Introduce the delays in the postsynaptic neuron indices
        indices[:, 1] = indices[:, 1] + self._n_neurons * (delays - 1)

        # Define the Tensorflow variables
        self.recurrent_indices = tf.Variable(indices, dtype=tf.int64, trainable=False)
        self.pre_ind_table = self.make_pre_ind_table(indices)

        # convert it to tensor
        # self.recurrent_indices = tf.convert_to_tensor(self.recurrent_indices, dtype=tf.int64)
        self.recurrent_dense_shape = dense_shape
        # Set the sign of the connections (exc or inh)
        self.recurrent_weight_positive = tf.Variable(
            weights >= 0.0, name="recurrent_weights_sign", trainable=False)
        # Scale the weights
        self.recurrent_weight_values = tf.Variable(
            weights * recurrent_weight_scale / lr_scale,
            name="sparse_recurrent_weights",
            # to ensure that the weights always keep their sign
            constraint=SignedConstraint(self.recurrent_weight_positive),
            trainable=train_recurrent,
        ) # shape = (n_synapses,)

        # self.recurrent_weights_factors = synaptic_weights[syn_ids]
        # print('Recurrent weights factors shape', self.recurrent_weights_factors.shape)
        self.recurrent_weights_factors = tf.Variable(
            synaptic_weights[syn_ids],
            name="recurrent_weights_factors",
            trainable=False,
            dtype=self._compute_dtype
        )
       
        # self.sparse_w_rec = self.prepare_sparse_weight()
        # self.prepare_sparse_weight()

        print(f"> Recurrent synapses {len(indices)}")

        del indices, weights, dense_shape, syn_ids


        ### LGN input connectivity ###
        self.lgn_input_dense_shape = (self._n_neurons, lgn_input["n_inputs"],)
        input_indices, input_weights, input_syn_ids = (
            lgn_input["indices"],
            lgn_input["weights"],
            lgn_input["syn_ids"]
        )

        # self.input_tau_syn_weights = lgn_input["tau_syn_weights_array"]
        # Scale down the input weights
        input_weights = (input_weights/ voltage_scale[self._node_type_ids[input_indices[:, 0]]]).astype(np.float32)
        input_delays = np.round(np.clip(lgn_input["delays"], dt, self.max_delay)/dt).astype(np.int32)
        # Introduce the delays in the postsynaptic neuron indices
        # input_indices[:, 1] = input_indices[:, 1] + self._n_neurons * (input_delays - 1)
        self.input_indices = tf.Variable(input_indices, trainable=False, dtype=tf.int64)

        # Define the Tensorflow variables
        self.input_weight_positive = tf.Variable(
            input_weights >= 0.0, name="input_weights_sign", trainable=False)
        self.input_weight_values = tf.Variable(
            input_weights * input_weight_scale / lr_scale,
            name="sparse_input_weights",
            constraint=SignedConstraint(self.input_weight_positive),
            trainable=train_input,
        )
        self.input_syn_ids = input_syn_ids
        self.input_weights_factors = synaptic_weights[self.input_syn_ids]

        # input_weights_factors = synaptic_weights[input_syn_ids]
        # input_weight_values_mod = self.input_weight_values[:, np.newaxis] * input_weights_factors
        # input_weight_values_mod_list = tf.split(input_weight_values_mod, self._n_syn_basis, axis=1)
        # self.input_sparse_w = []
        # for input_weight in input_weight_values_mod_list:
        #     sparse_w_in = tf.sparse.SparseTensor(
        #         self.input_indices,
        #         input_weight[:, 0],
        #         self.lgn_input_dense_shape,
        #     )
        #     self.input_sparse_w.append(tf.cast(sparse_w_in, self._compute_dtype))
        print(f"> LGN input synapses {len(input_indices)}")
        del input_indices, input_weights, input_syn_ids 

        ### BKG input connectivity ###
        self.bkg_input_dense_shape = (self._n_neurons, bkg_input["n_inputs"],)
        bkg_input_indices, bkg_input_weights, bkg_input_syn_ids = (
            bkg_input["indices"],
            bkg_input["weights"],
            bkg_input["syn_ids"]
        )

        # bkg_input_indices = bkg_input["indices"]
        # self.bkg_input_tau_syn_weights = bkg_input["tau_syn_weights_array"]
        bkg_input_weights = (bkg_input_weights/voltage_scale[self._node_type_ids[bkg_input_indices[:, 0]]]).astype(np.float32)
        bkg_input_delays = np.round(np.clip(bkg_input["delays"], dt, self.max_delay)/dt).astype(np.int32)
        # Introduce the delays in the postsynaptic neuron indices
        # input_indices[:, 1] = input_indices[:, 1] + self._n_neurons * (input_delays - 1)
        self.bkg_input_indices = tf.Variable(bkg_input_indices, trainable=False, dtype=tf.int64)

        # Define Tensorflow variables
        self.bkg_input_weight_positive = tf.Variable(
            bkg_input_weights >= 0.0, name="bkg_input_weights_sign", trainable=False)
        self.bkg_input_weights = tf.Variable(
            bkg_input_weights * input_weight_scale / lr_scale, 
            name="rest_of_brain_weights", 
            constraint=SignedConstraint(self.bkg_input_weight_positive),
            trainable=train_input
        )
        self.bkg_input_syn_ids = bkg_input_syn_ids

        # bkg_input_weights_factors = synaptic_weights[bkg_input_syn_ids]
        # bkg_input_weight_values_mod = self.bkg_input_weights[:, np.newaxis]*bkg_input_weights_factors
        # bkg_input_weight_values_mod_list = tf.split(bkg_input_weight_values_mod, self._n_syn_basis, axis=1)
        # self.bkg_input_sparse_w = []
        # for bkg_input_weight in bkg_input_weight_values_mod_list:
        #     sparse_w_in = tf.sparse.SparseTensor(
        #         self.bkg_input_indices,
        #         bkg_input_weight[:, 0],
        #         self.bkg_input_dense_shape,
        #     )
        #     self.bkg_input_sparse_w.append(tf.cast(sparse_w_in, self._compute_dtype))    

        print(f"> BKG input synapses {len(bkg_input_indices)}")
        del bkg_input_indices, bkg_input_weights, bkg_input_syn_ids, synaptic_weights
    
    def make_pre_ind_table(self, indices):
        """ This function creates a table that maps the presynaptyc index to 
        the indices of the recurrent_indices tensor. it takes a dimension of
        (number_of_neurons * max_delay) x (largest out-degree)
        
        If this causes address overflow, Try using TensorArray instead.
        
        """
        pre_inds = indices[:, 1]
        uni, counts = np.unique(pre_inds, return_counts=True)
        max_elem = np.max(counts)
        n_elem = self._n_neurons * self.max_delay
        
        # checking the possibility of address overflow
        if n_elem * max_elem > 2**31:
            # with my observation, this never happens with the current model.
            # with all 296991 neurons, the largest out-degree is 1548.
            # this results in 1,838,968,272, which is barely below 2**31 (~2.1 billion)
            print("n_elem: ", n_elem)
            print("max_elem: ", max_elem)
            print("n_elem * max_elem: ", n_elem * max_elem)
            print("n_elem * max_elem > 2**31")
            raise ValueError("It will cause address overflow. Time to think about a different approach.")
        
        @njit
        def make_table(pre_inds, n_elem, max_elem):
            # first, make a big array to allocate memory
            arr = np.ones((n_elem, max_elem), dtype=np.int32) * -1
            arr_inds = np.zeros(n_elem, dtype=np.int32)
            for i in range(pre_inds.shape[0]):
                arr[pre_inds[i], arr_inds[pre_inds[i]]] = i
                arr_inds[pre_inds[i]] += 1
            return arr
        
        table = make_table(pre_inds, n_elem, max_elem)
        # exit with int64 for faster processing on a GPU (don't know why...)
        table = tf.convert_to_tensor(table, dtype=tf.int64)
        # table = tf.convert_to_tensor(table, dtype=tf.int32)
        # table = tf.cast(tf.stack(table, axis=0), tf.int32)
        return table


    def get_new_inds_table(self, non_zero_cols):
        """ a new function that prepares new sparse indices tensor.
        This effectively does 'gather' operation for the sparse tensor.
        It utilizes the pre_ind_table to find the indices of the recurrent_indices

        """
        pre_inds = self.recurrent_indices[:, 1]
        post_inds = self.recurrent_indices[:, 0]
        all_inds = tf.gather(self.pre_ind_table, non_zero_cols)
        all_inds = tf.reshape(all_inds, [-1])  # flatten the tensor
        
        # remove unecessary -1's
        all_inds = tf.boolean_mask(all_inds, all_inds >= 0)
        
        # sort to make it compatible with sparse tensor creation
        # all_inds = tf.sort(all_inds)
        # inds = tf.cast(all_inds, tf.int32)
        inds = tf.sort(all_inds)
        
        remaining_pre = tf.gather(pre_inds, inds)
        uniq_pre_inds, idx = tf.unique(remaining_pre, out_idx=tf.int64)

        new_pre = tf.gather(idx, tf.range(tf.size(inds)))
        new_post = tf.gather(post_inds, inds)
        new_indices = tf.stack((new_post, new_pre), axis=1)
        return new_indices, inds

    # @tf.function
    def calculate_i_rec(self, rec_z_buf):
        # This function performs the tensor multiplication to calculate the recurrent currents at each timestep
        # This is a new faster implementation that uses the pre_ind_table
        # Memory consumption and processing time depends on the number of
        # spiking neurons
        # this faster method uses sparseness of the rec_z_buf.
        # it identifies the non_zero rows of rec_z_buf and only computes the
        # sparse matrix multiplication for those rows.
        i_rec = tf.TensorArray(dtype=self._compute_dtype, size=self._n_syn_basis)
        # find the non-zero rows of rec_z_buf
        non_zero_cols = tf.where(rec_z_buf)[:, 1]
        nnz = tf.cast(tf.shape(non_zero_cols)[0], dtype=tf.int64)  # number of non zero
        if nnz == 0: # nothing is firing
            i_rec = tf.zeros((self._n_syn_basis * self._n_neurons, 1), dtype=self._compute_dtype)
        else:
            sliced_rec_z_buf = tf.gather(rec_z_buf, non_zero_cols, axis=1)
            
            # let's make sparse arrays for multiplication
            # new_indices will be a version of indices that only contains the non-zero columns
            # in the non_zero_cols, and changes the indices accordingly.
            new_indices, inds = self.get_new_inds_table(non_zero_cols)
            if tf.shape(inds)[0] == 0:  # if firing cells do not have any outputs
                i_rec = tf.zeros((self._n_syn_basis * self._n_neurons, 1), dtype=self._compute_dtype)
            else:
                picked_weights = tf.gather(self.recurrent_weight_values, inds)
                
                for r_id in range(self._n_syn_basis):
                        
                    weights_syn_receptors = picked_weights * tf.gather(self.recurrent_weights_factors[:, r_id], inds)
                    sliced_sparse = tf.sparse.SparseTensor(
                        # tf.cast(new_indices, tf.int64),
                        new_indices,
                        weights_syn_receptors,
                        [self.recurrent_dense_shape[0], nnz]
                    )
                    i_receptor = tf.sparse.sparse_dense_matmul(
                                                                sliced_sparse,
                                                                sliced_rec_z_buf,
                                                                adjoint_b=True
                                                            )
                    # Append i_receptor to the TensorArray
                    i_rec = i_rec.write(r_id, i_receptor)
                # Stack the TensorArray into a single tensor
                i_rec = i_rec.stack()
       
        return i_rec
       
        
    # @tf.function
    def update_psc(self, psc, psc_rise, rec_inputs, syn_decay, psc_initial, dt):
        new_psc_rise = psc_rise * self.syn_decay + rec_inputs * self.psc_initial
        new_psc = psc * self.syn_decay + self._dt * self.syn_decay * psc_rise
        return new_psc, new_psc_rise

    # @tf.function
    def update_asc(self, asc_1, asc_2, k, prev_z, asc_amps, dt):
        exp_dt_k_1 = tf.exp(-dt * k[:, 0])
        exp_dt_k_2 = tf.exp(-dt * k[:, 1])
        new_asc_1 = exp_dt_k_1 * asc_1 + prev_z * asc_amps[:, 0]
        new_asc_2 = exp_dt_k_2 * asc_2 + prev_z * asc_amps[:, 1]
        return new_asc_1, new_asc_2

    # @tf.function
    def zero_state(self, batch_size, dtype=tf.float32):
        # The neurons membrane voltage start the simulation at their reset value
        v0 = tf.ones((batch_size, self._n_neurons), dtype) * tf.cast(
            self.v_th * 0.0 + 1.0 * self.v_reset, dtype)
        z0_buf = tf.zeros(
            (batch_size, self._n_neurons * self.max_delay), dtype)
        r0 = tf.zeros((batch_size, self._n_neurons), dtype)
        asc_10 = tf.zeros((batch_size, self._n_neurons), dtype)
        asc_20 = tf.zeros((batch_size, self._n_neurons), dtype)
        psc_rise0 = tf.zeros(
            (batch_size, self._n_neurons * self._n_syn_basis), dtype
        )
        psc0 = tf.zeros((batch_size, self._n_neurons *
                        self._n_syn_basis), dtype)
        return z0_buf, v0, r0, asc_10, asc_20, psc_rise0, psc0

    # @tf.function
    def _gather(self, prop):
        return tf.gather(prop, self._node_type_ids)

    def calculate_i_in(self, inputs):
        # This function performs the tensor multiplication to calculate the recurrent currents at each timestep
        i_in = tf.TensorArray(dtype=self._compute_dtype, size=self._n_syn_basis)
        for r_id in range(self._n_syn_basis):
            weights_syn_receptors = self.input_weight_values * self.input_weights_factors[:, r_id]
            sparse_w_in = tf.sparse.SparseTensor(
                self.input_indices,
                tf.cast(weights_syn_receptors, self._compute_dtype), 
                self.lgn_input_dense_shape,
            )
            i_receptor = tf.sparse.sparse_dense_matmul(
                                                        sparse_w_in,
                                                        tf.cast(inputs, self._compute_dtype),
                                                        adjoint_b=True
                                                    )
            # Append i_receptor to the TensorArray
            i_in = i_in.write(r_id, i_receptor)
        # Stack the TensorArray into a single tensor
        i_in = i_in.stack()
        return i_in

    # @tf.function
    def call(self, inputs, state, constants=None):
        # tf.print('------------- MODEL CALLING OUT --------------')
        # tracker = GPUMemoryTracker()

        batch_size = inputs.shape[0]
        if batch_size is None:
            batch_size = tf.shape(inputs)[0]

        if self._spike_gradient:
            state_input = tf.zeros((1,))
        else:
            state_input = tf.zeros((4,))
        if constants is not None:
            if self._spike_gradient:
                # external_current = inputs[:, : self._n_neurons * self._n_syn_basis]
                # state_input = inputs[:, self._n_neurons * self._n_syn_basis:]
                lgn_spikes = inputs[:, :-self._n_neurons]
                state_input = inputs[:, -self._n_neurons:]
            else:
                # external_current = inputs[:, : self._n_neurons * self._n_syn_basis]
                # state_input = inputs[:, self._n_neurons * self._n_syn_basis:]
                lgn_spikes = inputs[:, :-self._n_neurons]
                state_input = inputs[:, -self._n_neurons:]
                state_input = tf.reshape(state_input, 
                                        (batch_size, self._n_neurons, self._n_syn_basis))
                
        inp_shape = lgn_spikes.get_shape().as_list()
        print('Input shape: ', inp_shape)
        shp = [dim if dim is not None else tf.shape(inputs)[i] for i, dim in enumerate(inp_shape)]
        # print(f"The ratio of the current input batch size to the maximum batch size is {batch_size}/{self._max_batch}")
        inp = tf.cast(lgn_spikes, self._compute_dtype)
        inp = tf.reshape(inp, (shp[0], shp[1])) # (batch_size*sequence_length, input_dim)
        input_current = self.calculate_i_in(tf.cast(inp, self._compute_dtype)) # (5, 1000, 600)
        print(input_current.shape)
        external_current = tf.transpose(input_current) # (600, 1000, 5)
        print(input_current.shape)

        # external_current = inputs   # inputs shape (1, 399804)
        
        # Extract the network variables from the state
        z_buf, v, r, asc_1, asc_2, psc_rise, psc = state
        # Define the previous max_delay spike matrix
        shaped_z_buf = tf.reshape(z_buf, (-1, self.max_delay, self._n_neurons))  
        prev_z = shaped_z_buf[:, 0]  # previous spikes with shape (neurons,)
        dampened_z_buf = z_buf * self._recurrent_dampening  # dampened version of z_buf # no entiendo muy bien la utilidad de esto
        # Now we use tf.stop_gradient to prevent the term (z_buf - dampened_z_buf) to be trained
        rec_z_buf = (tf.stop_gradient(z_buf - dampened_z_buf) + dampened_z_buf)  
        # Reshape the psc variables
        psc_rise = tf.reshape(psc_rise, (batch_size, self._n_neurons, self._n_syn_basis))
        psc = tf.reshape(psc, (batch_size, self._n_neurons, self._n_syn_basis))

        ### Calculate the recurrent input current ###
        i_rec = self.calculate_i_rec(rec_z_buf)
        i_rec = tf.transpose(i_rec)
        rec_inputs = tf.cast(i_rec, self._compute_dtype)
        rec_inputs = tf.reshape(rec_inputs,
            (batch_size, self._n_neurons, self._n_syn_basis),)

        ### Add the external current to the recurrent current ###
        # external_current = tf.reshape(external_current,
        #     (batch_size, self._n_neurons, self._n_syn_basis),)
        rec_inputs = rec_inputs + external_current
        # Scale with the learning rate
        rec_inputs = rec_inputs * self._lr_scale

        if constants is not None and not self._spike_gradient:
            rec_inputs = rec_inputs + state_input * self._lr_scale

        # Calculate the new psc variables
        # new_psc_rise = psc_rise * self.syn_decay + rec_inputs * self.psc_initial
        # new_psc = psc * self.syn_decay + self._dt * self.syn_decay * psc_rise
        new_psc, new_psc_rise = self.update_psc(psc, psc_rise, rec_inputs, self.syn_decay, self.psc_initial, self._dt)

        # New r is a variable that accounts for the refractory period in which a neuron cannot spike
        new_r = tf.nn.relu(r + prev_z * self.t_ref - self._dt)  # =max(r + prev_z * self.t_ref - self._dt, 0)
        
        # Calculate the ASC
        new_asc_1, new_asc_2 = self.update_asc(asc_1, asc_2, self.k, prev_z, self.asc_amps, self._dt)

        if constants is not None and self._spike_gradient:
            input_current = tf.reduce_sum(psc, -1) + state_input
        else:
            input_current = tf.reduce_sum(psc, -1)
        # input_current = tf.reduce_sum(psc, -1)

        # Add all the postsynaptic current sources
        c1 = input_current + asc_1 + asc_2 + self.gathered_g

        # Calculate the new voltage values
        decayed_v = self.decay * v
        # Update the voltage according to the LIF equation and the refractory period
        if self._hard_reset:
            # Here we keep the voltage at the reset value during the refractory period
            new_v = tf.where(new_r > 0.0, self.v_reset, decayed_v + self.current_factor * c1)
            # Here we make a hard reset and let the voltage freely evolve but we do not let the
            # neuron spike during the refractory period
            # new_v = tf.where(prev_z > 0., self.v_reset, decayed_v + self.current_factor * c1)
        else:
            reset_current = prev_z * (self.v_reset - self.v_th)
            new_v = decayed_v + self.current_factor * c1 + reset_current

        # Generate the network spikes
        v_sc = (new_v - self.v_th) / self.normalizer
        if self._pseudo_gauss:
            if self._compute_dtype == tf.bfloat16:
                new_z = spike_function_b16(v_sc, self._dampening_factor)
            elif self._compute_dtype == tf.float16:
                new_z = spike_gauss_16(v_sc, self._gauss_std, self._dampening_factor)
            else:
                new_z = spike_gauss(v_sc, self._gauss_std, self._dampening_factor)
        else:
            if self._compute_dtype == tf.float16:
                new_z = spike_function_16(v_sc, self._dampening_factor)
            else:
                new_z = spike_function(v_sc, self._dampening_factor)

        # Generate the new spikes if the refractory period is concluded
        new_z = tf.where(new_r > 0.0, tf.zeros_like(new_z), new_z)
        
        # Reshape the network variables
        new_psc = tf.reshape(new_psc, (batch_size, self._n_neurons * self._n_syn_basis))
        new_psc_rise = tf.reshape(new_psc_rise, (batch_size, self._n_neurons * self._n_syn_basis))
        # Add current spikes to the buffer
        new_shaped_z_buf = tf.concat((new_z[:, None], shaped_z_buf[:, :-1]), 1)
        new_z_buf = tf.reshape(new_shaped_z_buf, (-1, self._n_neurons * self.max_delay))

        # Define the model outputs and the new state of the network
        outputs = (
            new_z,
            new_v * self.voltage_scale + self.voltage_offset,
            (input_current + new_asc_1 + new_asc_2) * self.voltage_scale,
        )

        new_state = (
            new_z_buf,
            new_v,
            new_r,
            new_asc_1,
            new_asc_2,
            new_psc_rise,
            new_psc,
        )

        return outputs, new_state


# @tf.function
def huber_quantile_loss(u, tau, kappa):
    branch_1 = tf.abs(tau - tf.cast(u <= 0, tf.float32)) / \
        (2 * kappa) * tf.square(u)
    branch_2 = tf.abs(tau - tf.cast(u <= 0, tf.float32)) * \
        (tf.abs(u) - 0.5 * kappa)
    return tf.where(tf.abs(u) <= kappa, branch_1, branch_2)

### To calculate the loss of firing rates between neuron types
# @tf.function
def compute_spike_rate_target_loss(_spikes, target_rates):
    # TODO: define this function
    # target_rates is a dictionary that contains all the cell types.
    # I should iterate on them, and add the cost for each one at the end.
    # spikes will have a shape of (batch_size, n_steps, n_neurons)
    losses = []
    for key, value in target_rates.items():
        spikes_type = tf.gather(_spikes, value["neuron_ids"], axis=-1)
        loss_type = compute_spike_rate_distribution_loss(spikes_type, value["sorted_target_rates"])
        losses.append(tf.reduce_mean(loss_type))

    return tf.reduce_sum(losses, axis=0)


class SpikeRateDistributionTarget:
    """ Instead of regularization, treat it as a target.
        The main difference is that this class will calculate the loss
        for each subtypes of the neurons."""
    def __init__(self, target_rates, rate_cost=.5):
        self._rate_cost = rate_cost
        self._target_rates = target_rates

    def __call__(self, spikes):
        reg_loss = compute_spike_rate_target_loss(spikes, self._target_rates) * self._rate_cost
        return reg_loss


def compute_spike_rate_distribution_loss(_spikes, target_rate):
    _rate = tf.reduce_mean(_spikes, (0, 1))
    ind = tf.range(target_rate.shape[0])
    rand_ind = tf.random.shuffle(ind)
    _rate = tf.gather(_rate, rand_ind)
    sorted_rate = tf.sort(_rate)
    u = target_rate - sorted_rate
    tau = (tf.cast(tf.range(target_rate.shape[0]), tf.float32) + 1) / target_rate.shape[
        0
    ]
    loss = huber_quantile_loss(u, tau, 0.002)

    return loss


class SpikeRateDistributionRegularization:
    def __init__(self, target_rates, rate_cost=0.5):
        self._rate_cost = rate_cost
        self._target_rates = target_rates

    def __call__(self, spikes):
        reg_loss = (
            compute_spike_rate_distribution_loss(spikes, self._target_rates)
            * self._rate_cost
        )
        reg_loss = tf.reduce_sum(reg_loss)

        return reg_loss


class VoltageRegularization:
    def __init__(self, cell, voltage_cost=1e-5):
        self._voltage_cost = voltage_cost
        self._cell = cell

    def __call__(self, voltages):
        voltage_32 = (
            tf.cast(voltages, tf.float32) - self._cell.voltage_offset
        ) / self._cell.voltage_scale
        v_pos = tf.square(tf.nn.relu(voltage_32 - 1.0))
        v_neg = tf.square(tf.nn.relu(-voltage_32 + 1.0))
        voltage_loss = (
            tf.reduce_mean(tf.reduce_sum(v_pos + v_neg, -1)) *
            self._voltage_cost
        )
        return voltage_loss


def sample_firing_rates(firing_rates, n_neurons, rnd_seed):
    sorted_firing_rates = np.sort(firing_rates)
    percentiles = (np.arange(firing_rates.shape[-1]) + 1).astype(np.float32) / firing_rates.shape[-1]
    rate_rd = np.random.RandomState(seed=rnd_seed)
    x_rand = rate_rd.uniform(size=n_neurons)
    target_firing_rates = np.sort(np.interp(x_rand, percentiles, sorted_firing_rates))
    
    return target_firing_rates


# @profile
def create_model(
    network,
    lgn_input,
    bkg_input,
    seq_len=100,
    n_input=10,
    n_output=2,
    cue_duration=20,
    dtype=tf.float32,
    input_weight_scale=1.0,
    gauss_std=0.5,
    dampening_factor=0.2,
    lr_scale=800.0,
    train_recurrent=True,
    train_input=True,
    neuron_output=False,
    recurrent_dampening_factor=0.5,
    use_state_input=False,
    return_state=False,
    return_sequences=False,
    down_sample=50,
    add_metric=True,
    max_delay=5,
    batch_size=None,
    pseudo_gauss=False,
    hard_reset=False,
):

    # Create the input layer of the model
    x = tf.keras.layers.Input(shape=(seq_len, n_input,))
    neurons = network["n_nodes"]

    # Create an input layer for the initial state of the RNN
    state_input_holder = tf.keras.layers.Input(shape=(seq_len, neurons))
    state_input = tf.cast(tf.identity(state_input_holder), dtype)  

    # If batch_size is not provided as an argument, it is automatically inferred from the
    # first dimension of x using tf.shape().
    if batch_size is None:
        batch_size = tf.shape(x)[0]
    else:
        batch_size = batch_size

    # Create the V1Column cell
    print('Creating the V1 column')
    time0 = time()
    cell = V1Column(
        network,
        lgn_input,
        bkg_input,
        gauss_std=gauss_std,
        dampening_factor=dampening_factor,
        input_weight_scale=input_weight_scale,
        lr_scale=lr_scale,
        spike_gradient=True,
        recurrent_dampening_factor=recurrent_dampening_factor,
        max_delay=max_delay,
        pseudo_gauss=pseudo_gauss,
        train_recurrent=train_recurrent,
        train_input=train_input,
        hard_reset=hard_reset,
    )
    print(f"V1Column created in {time()-time0:.2f} seconds")

    # initialize the RNN state to zero using the zero_state() method of the V1Column class.
    zero_state = cell.zero_state(batch_size, dtype)

    if use_state_input:
        # The shape of each input tensor matches the shape of the corresponding
        # tensor in the zero_state tuple, except for the batch dimension. The batch
        # dimension is left unspecified, allowing the tensor to be fed variable-sized
        # batches of data.
        initial_state_holder = tf.nest.map_structure(lambda _x: tf.keras.layers.Input(shape=_x.shape[1:]), zero_state)
        # The code then copies the input tensors into the rnn_initial_state variable
        # using tf.nest.map_structure(). This creates a nested structure of tensors with
        # the same shape as the original zero_state structure.
        rnn_initial_state = tf.nest.map_structure(tf.identity, initial_state_holder)
        # In both cases, the code creates a constants tensor using tf.zeros_like() or
        # tf.zeros(). This tensor is used to provide constant input to the RNN during
        # computation. The shape of the constants tensor matches the batch_size.
        constants = tf.zeros_like(rnn_initial_state[0][:, 0], dtype)
    else:
        rnn_initial_state = zero_state
        constants = tf.zeros((batch_size,))

    # Create the LGN input layer of the model
    # x = tf.cast(x, tf.int16)
    # rnn_inputs = SparseLayer(
    #     cell.input_indices,
    #     cell.input_weight_values,
    #     cell.lgn_input_dense_shape,
    #     cell.synaptic_weights,
    #     cell.input_syn_ids,
    #     # cell.input_tau_syn_weights,
    #     cell._n_neurons,
    #     lr_scale=lr_scale,
    #     dtype=dtype,
    #     name="input_layer",
    # )(x)

    # # Create the BKG input layer of the model
    # noise_inputs = BackgroundNoiseLayer(
    #     cell.bkg_input_indices,
    #     cell.bkg_input_weights,
    #     cell.bkg_input_dense_shape,
    #     cell.synaptic_weights, 
    #     cell.bkg_input_syn_ids,
    #     batch_size, 
    #     seq_len,
    #     lr_scale=lr_scale,
    #     dtype=dtype,
    #     name="noise_layer",
    # )(x) # the input is provided just because in a Keras custom layer, the call method should accept input

    # rnn_inputs = rnn_inputs + noise_inputs

    print("Sparse layer created")
    print(x.shape)
    print(state_input)

    # Concatenate the input layer with the initial state of the RNN
    # rnn_inputs = tf.cast(rnn_inputs, dtype)
    # full_inputs = tf.concat((rnn_inputs, state_input), -1)

    full_inputs = tf.concat((x, state_input), -1)

    print('Full inputs shape: ', full_inputs.shape)
    # Create the RNN layer of the model using the V1Column cell
    # The RNN layer returns the output of the RNN layer and the final state of the RNN
    # layer. The output of the RNN layer is a tensor of shape (batch_size, seq_len,
    # neurons). The final state of the RNN layer is a tuple of tensors, each of shape
    # (batch_size, neurons).
    rnn = tf.keras.layers.RNN(
        cell, return_sequences=True, return_state=return_state, name="rsnn"
    )

    print("RNN layer created")

    # Apply the rnn layer to the full_inputs tensor
    t0 = time()
    out = rnn(full_inputs, initial_state=rnn_initial_state, constants=constants)
    # process = psutil.Process()
    # mem = process.memory_info().rss / (1024 * 1024)  # in MB
    # tf.print("Memory consumption:", mem)
    # print('>>> Recurrent layer processing time: ', time()-t0)
    # Check if the return_state argument is True or False and assign the output of the
    # RNN layer to the hidden variable accordingly.
    if return_state:
        hidden = out[0]
        new_state = out[1:]
    else:
        hidden = out
    spikes = hidden[0]
    voltage = hidden[1]
    # computes the mean of the spikes tensor along the second and third dimensions
    # (which represent time and neurons),
    rate = tf.cast(tf.reduce_mean(spikes, (1, 2)), tf.float32)

    # The neuron output option selects only the output neurons from the spikes tensor
    if neuron_output:
        # The output_spikes tensor is computed by taking a linear combination
        # of the current spikes and the previous spikes, with the coefficients
        # determined by the dampening_factor. This serves to reduce the volatility
        # of the output spikes, making them more stable.
        output_spikes = 1 / dampening_factor * spikes + (
            1 - 1 / dampening_factor
        ) * tf.stop_gradient(spikes)
        # The output tensor is then computed by selecting the spikes from the
        # output neurons and scaling them by a learned factor. The scale factor
        # is computed using a softplus activation function applied to the output
        # of a dense layer, and the threshold is computed by passing the output
        # spikes through another dense layer.
        output = tf.gather(
            output_spikes, network["readout_neuron_ids"], axis=2)
        output = tf.reduce_mean(output, -1)
        scale = 1 + tf.nn.softplus(
            tf.keras.layers.Dense(1)(tf.zeros_like(output[:1, :1]))
        )
        thresh = tf.keras.layers.Dense(1)(tf.zeros_like(output))
        output = tf.stack([thresh[..., 0], output[..., -1]], -1) * scale
    # If neuron_output is False, then the output tensor is simply the result of
    # passing the spikes tensor through a dense layer with n_output units.
    else:
        output = tf.keras.layers.Dense(n_output, name="projection", trainable=False)(spikes)
        # output = tf.keras.layers.Dense(n_output, name="projection", trainable=True)(spikes)
        
    # Finally, a prediction layer is created
    output = tf.keras.layers.Lambda(lambda _a: _a, name="prediction")(output)

    # If return_sequences is True, then the mean_output tensor is computed by
    # averaging over sequences of length down_sample in the output tensor.
    # Otherwise, mean_output is simply the mean of the last cue_duration time steps
    # of the output tensor.
    if return_sequences:
        mean_output = tf.reshape(
            output, (-1, int(seq_len / down_sample), down_sample, n_output)
        )
        mean_output = tf.reduce_mean(mean_output, 2)
        mean_output = tf.nn.softmax(mean_output, axis=-1)
    else:
        mean_output = tf.reduce_mean(output[:, -cue_duration:], 1)
        mean_output = tf.nn.softmax(mean_output)

    if use_state_input:
        many_input_model = tf.keras.Model(
            inputs=[x, state_input_holder,
                    initial_state_holder], outputs=mean_output
        )
        # many_input_model = tf.keras.Model(
        #     inputs=[x, state_input_holder,initial_state_holder],
        #     outputs=[])
    else:
        many_input_model = tf.keras.Model(
            inputs=[x, state_input_holder], outputs=mean_output
        )
        # many_input_model = tf.keras.Model(
        #     inputs=[x, state_input_holder])

    if add_metric:
        # add the firing rate of the neurons as a metric to the model
        many_input_model.add_metric(rate, name="rate")

    return many_input_model


# if name is main run the create model function
if __name__ == "__main__":
    # load the network
    import load_sparse

    n_input = 17400
    n_neurons = 1574

    network, lgn_input, bkg_input = load_sparse.cached_load_v1(
        n_input,
        n_neurons,
        True,
        "GLIF_network",
        seed=3000,
        connected_selection=False,
        n_output=2,
        neurons_per_output=16,
    )
    # create the model
    model = create_model(
        network,
        lgn_input,
        bkg_input,
        seq_len=100,
        n_input=n_input,
        n_output=2,
        cue_duration=20,
        dtype=tf.float32,
        input_weight_scale=1.0,
        gauss_std=0.5,
        dampening_factor=0.2,
        lr_scale=800.0,
        train_recurrent=True,
        train_input=True,
        neuron_output=False,
        recurrent_dampening_factor=0.5,
        use_state_input=False,
        return_state=False,
        return_sequences=False,
        down_sample=50,
        add_metric=True,
        max_delay=5,
        batch_size=1,
        pseudo_gauss=False,
        hard_reset=True,
    )
