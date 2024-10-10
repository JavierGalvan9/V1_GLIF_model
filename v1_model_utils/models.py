import numpy as np
import tensorflow as tf
import os 
import pickle as pkl
from numba import njit
# import subprocess
from . import other_v1_utils


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

def make_pre_ind_table(indices, n_source_neurons=197613):
    """
    This function creates a table that maps presynaptic indices to 
    the indices of the recurrent_indices tensor using a RaggedTensor.
    This approach ensures that every presynaptic neuron, even those with no
    postsynaptic connections, has an entry in the RaggedTensor.
    """
    pre_inds = indices[:, 1]
    n_syn = pre_inds.shape[0]
    # Since pre_inds may not be sorted, we sort them along with synapse_indices
    sorted_pre_inds, sorted_synapse_indices = tf.math.top_k(-pre_inds, k=n_syn)
    sorted_pre_inds = -sorted_pre_inds  # Undo the negation to get the sorted pre_inds
    # Count occurrences (out-degrees) for each presynaptic neuron using bincount
    counts = tf.math.bincount(tf.cast(sorted_pre_inds, dtype=tf.int32), minlength=n_source_neurons)
    # Create row_splits that covers all presynaptic neurons (0 to n_source_neurons)
    row_splits = tf.concat([[0], tf.cumsum(counts)], axis=0)
    # Create the RaggedTensor with empty rows for missing neurons
    rt = tf.RaggedTensor.from_row_splits(sorted_synapse_indices, row_splits)

    return rt

def get_new_inds_table(indices, non_zero_cols, pre_ind_table):
    """Optimized function that prepares new sparse indices tensor."""
    # Gather the rows corresponding to the non_zero_cols
    selected_rows = tf.gather(pre_ind_table, non_zero_cols)
    # Flatten the selected rows to get all_inds
    all_inds = selected_rows.flat_values
    # Gather from indices using all_inds
    new_indices = tf.gather(indices, all_inds)

    return new_indices, all_inds

class BackgroundNoiseLayer(tf.keras.layers.Layer):
    """
    This class calculates the input currents from the BKG noise by processing all timesteps at once."
    For that reason is unfeasible if the user wants to train the LGN -> V1 weights.
    Each call takes 0.03 seconds for 600 ms of simulation.

    Returns:
        _type_: input_currents (self._compute_dtype)
    """
    def __init__(self, indices, weights, dense_shape,  
                 weights_factors, batch_size, seq_len,
                 bkg_firing_rate=250, n_bkg_units=100, 
                 dtype=tf.float32, **kwargs):
        super().__init__(**kwargs)
        self._dtype = dtype
        self._bkg_weights = weights
        self._bkg_indices = indices
        self._dense_shape = dense_shape
        self._bkg_input_weights_factors = weights_factors
        self._batch_size = batch_size
        self._seq_len = seq_len
        self._n_syn_basis = weights_factors.shape[1]
        # self._lr_scale = lr_scale
        self._bkg_firing_rate = bkg_firing_rate
        self._n_bkg_units = n_bkg_units

    def calculate_bkg_i_in(self, inputs):
        # This function performs the tensor multiplication to calculate the recurrent currents at each timestep
        i_in = tf.TensorArray(dtype=self._dtype, size=self._n_syn_basis)
        for r_id in range(self._n_syn_basis):
            weights_syn_receptors = self._bkg_weights * self._bkg_input_weights_factors[:, r_id]
            sparse_w_in = tf.sparse.SparseTensor(
                self._bkg_indices,
                weights_syn_receptors, 
                self._dense_shape,
            )
            i_receptor = tf.sparse.sparse_dense_matmul(
                                                        sparse_w_in,
                                                        inputs,
                                                        adjoint_b=True
                                                    )
            # Optionally cast the output back to float16
            if i_receptor.dtype != self._dtype:
                i_receptor = tf.cast(i_receptor, dtype=self._dtype)

            # Append i_receptor to the TensorArray
            i_in = i_in.write(r_id, i_receptor)

        # Stack the TensorArray into a single tensor
        i_in = i_in.stack()

        return i_in

    def call(self, inp): # inp only provides the shape
        # Generate the background spikes
        seq_len = tf.shape(inp)[1]
        # rest_of_brain = tf.random.poisson(shape=(self._batch_size, seq_len, self._n_bkg_units), 
        #                                 lam=self._bkg_firing_rate * .001, 
        #                                 seed=42,
        #                                 dtype=self._compute_dtype) # this implementation is slower
        rest_of_brain = tf.cast(tf.random.uniform(
                (self._batch_size, seq_len, self._n_bkg_units)) < self._bkg_firing_rate * .001, 
                tf.float32) # (1, 600, 100)

        rest_of_brain = tf.reshape(rest_of_brain, (self._batch_size * seq_len, self._n_bkg_units)) # (batch_size*sequence_length, input_dim)
        # Create a TensorArray to save the results for every receptor type
        noise_input = self.calculate_bkg_i_in(rest_of_brain) # (5, 65871, 600)
        noise_input = tf.transpose(noise_input) # (600, 50000, 5) # New shape (3000, 65871, 5)
        # Reshape properly the input current
        noise_input = tf.reshape(noise_input, (self._batch_size, seq_len, -1)) # (1, 600, 250000) # (1, 3000, 333170)

        return noise_input
    

class LGNInputLayerCell(tf.keras.layers.Layer):
    def __init__(self, indices, weights, dense_shape, weights_factors,
                 dtype=tf.float32, **kwargs):
        super().__init__(**kwargs)
        self._indices = indices
        self._input_weights = weights
        self._input_weights_factors = weights_factors
        # self._n_syn_basis = weights_factors.shape[1]  # Number of receptors
        self._dense_shape = dense_shape
        self._dtype = dtype
        # Precompute the synapses table
        self.pre_ind_table = make_pre_ind_table(indices, n_source_neurons=dense_shape[1])

    @property
    def state_size(self):
        # No states are maintained in this cell
        return []

    # def build(self, input_shape):
    #     # If you have any trainable variables, initialize them here
    #     pass

    @tf.function
    def call(self, inputs_t, states):
        # inputs_t: Shape [batch_size, input_dim]
        batch_size = tf.shape(inputs_t)[0]
        # Compute the input current for the timestep
        non_zero_cols = tf.where(inputs_t > 0)[:, 1]
        new_indices, inds = get_new_inds_table(self._indices, non_zero_cols, self.pre_ind_table)
        # Sort the segment IDs and corresponding data
        sorted_indices = tf.argsort(new_indices[:, 0])
        sorted_segment_ids = tf.gather(new_indices[:, 0], sorted_indices)
        sorted_inds = tf.gather(inds, sorted_indices)
        # Get the weights for each active synapse
        gathered_weights = tf.gather(self._input_weights, sorted_inds, axis=0)
        gathered_factors = tf.gather(self._input_weights_factors, sorted_inds, axis=0)
        sorted_data = gathered_weights * gathered_factors
        # Calculate the total LGN input current received by each neuron
        i_in = tf.math.unsorted_segment_sum(
            sorted_data,
            sorted_segment_ids,
            num_segments=self._dense_shape[0]
        )
        # Optionally cast the output back to float16
        if i_in.dtype != self._dtype:
            i_in = tf.cast(i_in, dtype=self._dtype)

        # Add batch dimension
        i_in = tf.expand_dims(i_in, axis=0)  # Shape: [1, n_post_neurons, n_syn_basis]
        i_in = tf.reshape(i_in, [batch_size, -1])
        # Since no states are maintained, return empty state
        return i_in, []

class LGNInputLayer(tf.keras.layers.Layer):
    """
    Calculates input currents from the LGN by processing one timestep at a time using a custom RNN cell.
    """
    def __init__(self, indices, weights, dense_shape, weights_factors,
                 dtype=tf.float32, **kwargs):
        super().__init__(**kwargs)
        self.input_cell = LGNInputLayerCell(
            indices, weights, dense_shape, weights_factors,
            dtype=dtype, **kwargs
        )
        # Create the input RNN layer with the custom cell to recursively process all the inputs by timesteps
        self.input_rnn = tf.keras.layers.RNN(self.input_cell, return_sequences=True, return_state=False, name='lgn_rsnn')

    def call(self, inputs, **kwargs):
        # inputs: Shape [batch_size, seq_len, input_dim]
        input_current = self.input_rnn(inputs, **kwargs)  # Outputs: [batch_size, seq_len, n_postsynaptic_neurons]
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


class ClipConstraint(tf.keras.constraints.Constraint):
    def __init__(self, lower_limit, upper_limit):
        self._lower_limit = lower_limit
        self._upper_limit = upper_limit

    def __call__(self, w):
        return tf.clip_by_value(w, self._lower_limit, self._upper_limit)

class V1Column(tf.keras.layers.Layer):
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
        train_recurrent_per_type=True,
        train_input=True,
        train_noise=True,
        hard_reset=True,
    ):
        super().__init__()
        _params = dict(network["node_params"])
        # Rescale the voltages to have them near 0, as we wanted the effective step size
        # for the weights to be normalized when learning (weights are scaled similarly)
        voltage_scale = _params["V_th"] - _params["E_L"]
        voltage_offset = _params["E_L"]
        _params["V_th"] = (_params["V_th"] - voltage_offset) / voltage_scale
        _params["E_L"] = (_params["E_L"] - voltage_offset) / voltage_scale
        _params["V_reset"] = (_params["V_reset"] - voltage_offset) / voltage_scale
        _params["asc_amps"] = (_params["asc_amps"] / voltage_scale[..., None])  # _params['asc_amps'] has shape (111, 2)
        # Define the other model variables
        self._node_type_ids = np.array(network["node_type_ids"])
        self._n_syn_basis = 5
        self._dt = dt
        self._recurrent_dampening = tf.cast(recurrent_dampening_factor, self._compute_dtype)
        self._dampening_factor = tf.cast(dampening_factor, self._compute_dtype)
        self._pseudo_gauss = pseudo_gauss
        self._lr_scale = lr_scale
        self._spike_gradient = spike_gradient
        self._hard_reset = hard_reset
        self._n_neurons = int(network["n_nodes"])
        self._gauss_std = tf.cast(gauss_std, self._compute_dtype)
        # Determine the membrane time decay constant
        tau = _params["C_m"] / _params["g"]
        membrane_decay = np.exp(-dt / tau)
        current_factor = 1 / _params["C_m"] * (1 - membrane_decay) * tau

        # Determine the synaptic dynamic parameters for each of the 5 basis receptors.
        path='GLIF_network/synaptic_data/tau_basis.npy' # [0.7579732  1.33243834 2.34228851 4.11750046 7.23813909]
        tau_syns = np.load(path)
        syn_decay = np.exp(-dt / tau_syns)
        syn_decay = tf.constant(syn_decay, dtype=self._compute_dtype)
        syn_decay = tf.tile(syn_decay, [self._n_neurons])
        self.syn_decay = tf.expand_dims(syn_decay, axis=0) # expand the dimension for processing different receptor types
        psc_initial = np.e / tau_syns
        psc_initial = tf.constant(psc_initial, dtype=self._compute_dtype)
        psc_initial = tf.tile(psc_initial, [self._n_neurons])
        self.psc_initial = tf.expand_dims(psc_initial, axis=0) # expand the dimension for processing different receptor types

        # Find the maximum delay in the network
        self.max_delay = int(np.round(np.min([np.max(network["synapses"]["delays"]), max_delay])))
        
        def _gather(prop):
            return tf.gather(prop, self._node_type_ids)
    
        def _f(_v, trainable=False):
            return tf.Variable(tf.cast(_gather(_v), self._compute_dtype), trainable=trainable)

        def inv_sigmoid(_x):
            return tf.math.log(_x / (1 - _x))

        def custom_val(_v, trainable=False):
            _v = tf.Variable(
                tf.cast(inv_sigmoid(_gather(_v)), self._compute_dtype),
                trainable=trainable,
            )
            def _g():
                return tf.nn.sigmoid(_v.read_value())

            return _v, _g

        # Gather the neuron parameters for every neuron
        self.t_ref = _f(_params["t_ref"])  # refractory time
        self.v_reset = _f(_params["V_reset"])
        self.asc_amps = _f(_params["asc_amps"], trainable=False)
        _k = _params["k"]
        # inverse sigmoid of the adaptation rate constant (1/ms)
        param_k, param_k_read = custom_val(_k, trainable=False)
        k = param_k_read()
        self.exp_dt_k = tf.cast(tf.exp(-self._dt * k), self._compute_dtype)

        self.v_th = _f(_params["V_th"])
        self.v_gap = self.v_reset - self.v_th
        e_l = _f(_params["E_L"])
        self.normalizer = self.v_th - e_l
        param_g = _f(_params["g"])
        self.gathered_g = param_g * e_l

        self.decay = _f(membrane_decay)
        self.current_factor = _f(current_factor)
        self.voltage_scale = _f(voltage_scale)
        self.voltage_offset = _f(voltage_offset)

        # Find the synaptic basis representation for each synaptic type
        path = os.path.join('GLIF_network', 'syn_id_to_syn_weights_dict.pkl')
        with open(path, "rb") as f:
            syn_id_to_syn_weights_dict = pkl.load(f)
        synaptic_basis_weights = np.array(list(syn_id_to_syn_weights_dict.values()))
        self.synaptic_basis_weights = tf.constant(synaptic_basis_weights, dtype=tf.float32)

        ### Network recurrent connectivity ###
        indices = np.array(network["synapses"]["indices"])
        weights = np.array(network["synapses"]["weights"])
        dense_shape = np.array(network["synapses"]["dense_shape"])
        syn_ids = np.array(network["synapses"]["syn_ids"])
        delays = np.array(network["synapses"]["delays"])

        # Scale down the recurrent weights
        weights = (weights/voltage_scale[self._node_type_ids[indices[:, 0]]])      

        # Use the maximum delay to clip the synaptic delays
        delays = np.round(np.clip(delays, dt, self.max_delay)/dt).astype(np.int32)
        # Introduce the delays in the presynaptic neuron indices
        indices[:, 1] = indices[:, 1] + self._n_neurons * (delays - 1)

        self.recurrent_dense_shape = dense_shape[0], self.max_delay * dense_shape[1] 
        #the first column (presynaptic neuron) has size n_neurons and the second column (postsynaptic neuron) has size max_delay*n_neurons

        # Define the Tensorflow variables
        self.recurrent_indices = tf.Variable(indices, dtype=tf.int64, trainable=False)
        self.pre_ind_table = make_pre_ind_table(indices, n_source_neurons=self.recurrent_dense_shape[1])

        # add dimension for the weights factors - TensorShape([23525415, 1])
        weights = tf.expand_dims(weights, axis=1) 
        # Set the sign of the connections (exc or inh)
        recurrent_weight_positive = tf.Variable(
            weights >= 0.0, name="recurrent_weights_sign", trainable=False)

        # if training the recurrent connection per type, turn off recurrent training
        # of individual connections
        if train_recurrent:
            if train_recurrent_per_type:
                individual_training = False
                per_type_training = True
            else:
                individual_training = True
                per_type_training = False
        else:
            individual_training = False
            per_type_training = False

        # Scale the weights
        self.recurrent_weight_values = tf.Variable(
            weights * recurrent_weight_scale / lr_scale, 
            name="sparse_recurrent_weights",
            constraint=SignedConstraint(recurrent_weight_positive),
            trainable=individual_training,
            dtype=tf.float32
        ) # shape = (n_synapses,)

        # prepare per_type variable, if required
        if per_type_training:
            self.per_type_training = True
            self.connection_type_ids = other_v1_utils.connection_type_ids(network)
            max_id = np.max(self.connection_type_ids) + 1
            # prepare a variable and gather with type ids.
            self.recurrent_per_type_weight_values = tf.Variable(
                tf.ones(max_id),
                name="recurrent_per_type_weights",
                constraint=ClipConstraint(0.2, 5.0),
                trainable=True,
                dtype=tf.float32
            ) # shape = (n_connection_types (21 * 21))
            # multiply this to the weights (this needs to be done in the loop)
        else:
            self.per_type_training = False
            
        syn_ids = tf.constant(syn_ids, dtype=tf.int32)
        self.recurrent_weights_factors = tf.gather(self.synaptic_basis_weights, syn_ids, axis=0) # TensorShape([23525415, 5])
        print(f"    > # Recurrent synapses: {len(indices)}")

        del indices, weights, dense_shape, syn_ids, delays

        ### LGN input connectivity ###
        self.lgn_input_dense_shape = (self._n_neurons, lgn_input["n_inputs"],)
        input_indices = np.array(lgn_input["indices"])
        input_weights = np.array(lgn_input["weights"])
        input_syn_ids = np.array(lgn_input["syn_ids"])
        input_delays = np.array(lgn_input["delays"])

        # Scale down the input weights
        input_weights = (input_weights/ voltage_scale[self._node_type_ids[input_indices[:, 0]]])
        input_delays = np.round(np.clip(input_delays, dt, self.max_delay)/dt).astype(np.int32)
        # Introduce the delays in the postsynaptic neuron indices
        # input_indices[:, 1] = input_indices[:, 1] + self._n_neurons * (input_delays - 1)
        self.input_indices = tf.Variable(input_indices, trainable=False, dtype=tf.int64)

        # Define the Tensorflow variables
        input_weights = tf.expand_dims(input_weights, axis=1) # add dimension for the weights factors - TensorShape([23525415, 1])

        input_weight_positive = tf.Variable(
            input_weights >= 0.0, name="input_weights_sign", trainable=False)
        self.input_weight_values = tf.Variable(
            input_weights * input_weight_scale / lr_scale,
            name="sparse_input_weights",
            constraint=SignedConstraint(input_weight_positive),
            trainable=train_input,
            dtype=tf.float32
        )

        input_syn_ids = tf.constant(input_syn_ids, dtype=tf.int32)
        self.input_weights_factors = tf.gather(self.synaptic_basis_weights, input_syn_ids, axis=0)

        print(f"    > # LGN input synapses {len(input_indices)}")
        del input_indices, input_weights, input_syn_ids, input_delays

        ### BKG input connectivity ###
        self.bkg_input_dense_shape = (self._n_neurons, bkg_input["n_inputs"],)
        bkg_input_indices = np.array(bkg_input['indices'])
        bkg_input_weights = np.array(bkg_input['weights'])
        bkg_input_syn_ids = np.array(bkg_input['syn_ids'])
        bkg_input_delays = np.array(bkg_input['delays'])

        bkg_input_weights = (bkg_input_weights/voltage_scale[self._node_type_ids[bkg_input_indices[:, 0]]])
        bkg_input_delays = np.round(np.clip(bkg_input_delays, dt, self.max_delay)/dt).astype(np.int32)
        # Introduce the delays in the postsynaptic neuron indices
        # bkg_input_indices[:, 1] = bkg_input_indices[:, 1] + self._n_neurons * (bkg_input_delays - 1)
        self.bkg_input_indices = tf.Variable(bkg_input_indices, trainable=False, dtype=tf.int64)

        # Define Tensorflow variables
        bkg_input_weight_positive = tf.Variable(
            bkg_input_weights >= 0.0, name="bkg_input_weights_sign", trainable=False)
        self.bkg_input_weights = tf.Variable(
            bkg_input_weights * input_weight_scale / lr_scale, 
            name="rest_of_brain_weights", 
            constraint=SignedConstraint(bkg_input_weight_positive),
            trainable=train_noise,
            dtype=tf.float32
        )

        bkg_input_syn_ids = tf.constant(bkg_input_syn_ids, dtype=tf.int32)
        self.bkg_input_weights_factors = tf.gather(self.synaptic_basis_weights, bkg_input_syn_ids, axis=0)

        print(f"    > # BKG input synapses {len(bkg_input_indices)}")
        del bkg_input_indices, bkg_input_weights, bkg_input_syn_ids, bkg_input_delays

    def calculate_i_rec(self, rec_z_buf):
        # This is a new faster implementation that uses the pre_ind_table as a raggedTensor and exploits
        # the sparseness of the rec_z_buf.
        # it identifies the non_zero rows of rec_z_buf and only computes the
        # contributions for those rows.        
        non_zero_cols = tf.where(rec_z_buf > 0)[:, 1]
        new_indices, inds = get_new_inds_table(self.recurrent_indices, non_zero_cols, self.pre_ind_table)
        # Sort the segment IDs and corresponding data
        sorted_indices = tf.argsort(new_indices[:, 0])
        sorted_segment_ids = tf.gather(new_indices[:, 0], sorted_indices)
        sorted_inds = tf.gather(inds, sorted_indices)
        # Get the weights for each active synapse
        gathered_weights = tf.gather(self.recurrent_weight_values, sorted_inds, axis=0)
        gathered_factors = tf.gather(self.recurrent_weights_factors, sorted_inds, axis=0)
        sorted_data = gathered_weights * gathered_factors
        if self.per_type_training:
            per_type_weights = tf.expand_dims(tf.gather(self.recurrent_per_type_weight_values, 
                                                        tf.gather(self.connection_type_ids, sorted_inds)), axis=1)
            sorted_data = sorted_data * per_type_weights
        # Calculate the total recurrent current received by each neuron
        i_rec = tf.math.unsorted_segment_sum(
            sorted_data,
            sorted_segment_ids,
            num_segments=self._n_neurons
        )

        if i_rec.dtype != self._compute_dtype:
            i_rec = tf.cast(i_rec, dtype=self._compute_dtype)
        # Add batch dimension
        i_rec = tf.expand_dims(i_rec, axis=0)
        i_rec = tf.reshape(i_rec, [1, -1])
            
        return i_rec
              
    def update_psc(self, psc, psc_rise, rec_inputs):
        new_psc_rise = psc_rise * self.syn_decay + rec_inputs * self.psc_initial
        new_psc = psc * self.syn_decay + self._dt * self.syn_decay * psc_rise
        return new_psc, new_psc_rise

    @property
    def state_size(self):
        # Define the state size of the network
        state_size = (
            self._n_neurons * self.max_delay,  # z buffer
            self._n_neurons,  # v
            self._n_neurons,  # r
            self._n_neurons * 2,  # asc
            self._n_neurons * self._n_syn_basis,  # psc rise
            self._n_neurons * self._n_syn_basis,  # psc
        )
        return state_size
    
    def zero_state(self, batch_size, dtype=tf.float32):
        # The neurons membrane voltage start the simulation at their reset value
        v0 = tf.ones((batch_size, self._n_neurons), dtype) * tf.cast(
            self.v_th * 0.0 + 1.0 * self.v_reset, dtype)
        z0_buf = tf.zeros((batch_size, self._n_neurons * self.max_delay), dtype)
        r0 = tf.zeros((batch_size, self._n_neurons), dtype)
        asc = tf.zeros((batch_size, self._n_neurons * 2), dtype)
        psc_rise0 = tf.zeros((batch_size, self._n_neurons * self._n_syn_basis), dtype)
        psc0 = tf.zeros((batch_size, self._n_neurons * self._n_syn_basis), dtype)

        return z0_buf, v0, r0, asc, psc_rise0, psc0

    @tf.function
    def call(self, inputs, state, constants=None):

        # Get all the model inputs
        external_current = inputs[:, :self._n_neurons*self._n_syn_basis] # external inputs shape (1, 399804)
        bkg_noise = inputs[:, self._n_neurons*self._n_syn_basis:-self._n_neurons]
        state_input = inputs[:, -self._n_neurons:] # dummy zeros
        batch_size = tf.shape(bkg_noise)[0]

        if self._spike_gradient:
            state_input = tf.zeros((1,), dtype=self._compute_dtype)
        else:
            state_input = tf.zeros((4,), dtype=self._compute_dtype)
                
        # Extract the network variables from the state
        z_buf, v, r, asc, psc_rise, psc = state
        # Get previous spikes
        prev_z = z_buf[:, :self._n_neurons]  # Shape: [batch_size, n_neurons]
        # Define the spikes buffer
        dampened_z_buf = z_buf * self._recurrent_dampening  # dampened version of z_buf # no entiendo muy bien la utilidad de esto
        # Now we use tf.stop_gradient to prevent the term (z_buf - dampened_z_buf) to be trained
        rec_z_buf = (tf.stop_gradient(z_buf - dampened_z_buf) + dampened_z_buf)  
        # Calculate the recurrent postsynaptic currents
        i_rec = self.calculate_i_rec(rec_z_buf)
        # Add all the current sources
        rec_inputs = i_rec + external_current + bkg_noise
        # Scale with the learning rate
        rec_inputs = rec_inputs * self._lr_scale
        
        if constants is not None and not self._spike_gradient:
            rec_inputs = rec_inputs + state_input * self._lr_scale

        # Calculate the new psc variables
        new_psc, new_psc_rise = self.update_psc(psc, psc_rise, rec_inputs)        
        # Calculate the ASC
        asc = tf.reshape(asc, (batch_size, self._n_neurons, 2))
        new_asc = self.exp_dt_k * asc + tf.expand_dims(prev_z, axis=-1) * self.asc_amps
        new_asc = tf.reshape(new_asc, (batch_size, self._n_neurons * 2))
        # Calculate the postsynaptic current 
        input_current = tf.reshape(psc, (batch_size, self._n_neurons, self._n_syn_basis))
        input_current = tf.reduce_sum(input_current, -1) # sum over receptors
        if constants is not None and self._spike_gradient:
            input_current += state_input

        # Add all the postsynaptic current sources
        c1 = input_current + tf.reduce_sum(asc, axis=-1) + self.gathered_g

        # Calculate the new voltage values
        decayed_v = self.decay * v
        reset_current = prev_z * self.v_gap
        new_v = decayed_v + self.current_factor * c1 + reset_current

        # Update the voltage according to the LIF equation and the refractory period
        # New r is a variable that accounts for the refractory period in which a neuron cannot spike
        new_r = tf.nn.relu(r + prev_z * self.t_ref - self._dt)  # =max(r + prev_z * self.t_ref - self._dt, 0)
        if self._hard_reset:
            # Here we keep the voltage at the reset value during the refractory period
            new_v = tf.where(new_r > 0.0, self.v_reset, new_v)
            # Here we make a hard reset and let the voltage freely evolve but we do not let the
            # neuron spike during the refractory period

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
        # Add current spikes to the buffer
        new_z_buf = tf.concat([new_z, z_buf[:, :-self._n_neurons]], axis=1)  # Shift buffer

        # Define the model outputs and the new state of the network
        outputs = (
            new_z,
            new_v * self.voltage_scale + self.voltage_offset,
            (input_current + tf.reduce_sum(asc, axis=-1)) * self.voltage_scale,
        )

        new_state = (
            new_z_buf,
            new_v,
            new_r,
            new_asc,
            new_psc_rise,
            new_psc,
        )

        return outputs, new_state


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
    recurrent_dampening_factor=0.5,
    lr_scale=800.0,
    train_recurrent=True,
    train_recurrent_per_type=False,
    train_input=True,
    train_noise=True,
    neuron_output=False,
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
    x = tf.keras.layers.Input(shape=(None, n_input,))
    neurons = network["n_nodes"]

    # Create an input layer for the initial state of the RNN
    state_input_holder = tf.keras.layers.Input(shape=(None, neurons))
    state_input = tf.cast(tf.identity(state_input_holder), dtype)  

    # If batch_size is not provided as an argument, it is automatically inferred from the
    # first dimension of x using tf.shape().
    if batch_size is None:
        batch_size = tf.shape(x)[0]
    else:
        batch_size = batch_size

    # Create the V1Column cell
    print('Creating the V1 column...')
    # time0 = time()
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
        train_recurrent_per_type=train_recurrent_per_type,
        train_input=train_input,
        train_noise=train_noise,
        hard_reset=hard_reset,
    )

    # initialize the RNN state to zero using the zero_state() method of the V1Column class.
    zero_state = cell.zero_state(batch_size, dtype)

    if use_state_input:
        # The shape of each input tensor matches the shape of the corresponding
        # tensor in the zero_state tuple, except for the batch dimension. The batch
        # dimension is left unspecified, allowing the tensor to be fed variable-sized
        # batches of data.
        initial_state_holder = tf.nest.map_structure(lambda _x: tf.keras.layers.Input(shape=_x.shape[1:], dtype=_x.dtype), zero_state)
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
    rnn_inputs = LGNInputLayer(
        cell.input_indices,
        cell.input_weight_values,
        cell.lgn_input_dense_shape,
        cell.input_weights_factors,
        # lr_scale=lr_scale,
        dtype=dtype,
        name="input_layer",
    )(x)

    # Create the BKG input layer of the model
    bkg_inputs = BackgroundNoiseLayer(
        cell.bkg_input_indices,
        cell.bkg_input_weights,
        cell.bkg_input_dense_shape,
        cell.bkg_input_weights_factors, 
        batch_size, 
        seq_len,
        # lr_scale=lr_scale,
        dtype=dtype,
        name="noise_layer",
    )(x) # the input is provided just because in a Keras custom layer, the call method should accept input

    # Concatenate the input layer with the initial state of the RNN
    full_inputs = tf.concat((rnn_inputs, bkg_inputs, state_input), -1) # (None, 600, 5*n_neurons+n_neurons)
    
    # Create the RNN layer of the model using the V1Column cell
    # The RNN layer returns the output of the RNN layer and the final state of the RNN
    # layer. The output of the RNN layer is a tensor of shape (batch_size, seq_len,
    # neurons). The final state of the RNN layer is a tuple of tensors, each of shape
    # (batch_size, neurons).
    rnn = tf.keras.layers.RNN(cell, return_sequences=True, return_state=return_state, name="rsnn")

    # Apply the rnn layer to the full_inputs tensor
    out = rnn(full_inputs, initial_state=rnn_initial_state, constants=constants)

    # Check if the return_state argument is True or False and assign the output of the
    # RNN layer to the hidden variable accordingly.
    if return_state:
        hidden = out[0]
        # new_state = out[1:]
    else:
        hidden = out

    spikes = hidden[0]
    voltage = hidden[1]

    # computes the mean of the spikes tensor along the second and third dimensions
    # (which represent time and neurons),
    rate = tf.reduce_mean(spikes, (1, 2))

    # The neuron output option selects only the output neurons from the spikes tensor
    if neuron_output:
        # The output_spikes tensor is computed by taking a linear combination
        # of the current spikes and the previous spikes, with the coefficients
        # determined by the dampening_factor. This serves to reduce the volatility
        # of the output spikes, making them more stable.
        output_spikes = 1 / dampening_factor * spikes + (1 - 1 / dampening_factor) * tf.stop_gradient(spikes)
        # The output tensor is then computed by selecting the spikes from the
        # output neurons and scaling them by a learned factor. The scale factor
        # is computed using a softplus activation function applied to the output
        # of a dense layer, and the threshold is computed by passing the output
        # spikes through another dense layer.
        output = tf.gather(output_spikes, network["readout_neuron_ids"], axis=2)
        output = tf.reduce_mean(output, -1)
        scale = 1 + tf.nn.softplus(tf.keras.layers.Dense(1)(tf.zeros_like(output[:1, :1])))
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
    # if return_sequences:
    #     mean_output = tf.reshape(output, (-1, int(seq_len / down_sample), down_sample, n_output))
    #     mean_output = tf.reduce_mean(mean_output, 2)
    #     mean_output = tf.nn.softmax(mean_output, axis=-1)
    # else:
    #     mean_output = tf.reduce_mean(output[:, -cue_duration:], 1)
    #     mean_output = tf.nn.softmax(mean_output)

    if use_state_input:
        # many_input_model = tf.keras.Model(
        #     inputs=[x, state_input_holder, initial_state_holder], 
        #     outputs=mean_output
        # )
        many_input_model = tf.keras.Model(
            inputs=[x, state_input_holder,initial_state_holder],
            outputs=[output])
    else:
        # many_input_model = tf.keras.Model(
        #     inputs=[x, state_input_holder], 
        #     outputs=mean_output
        # )
        many_input_model = tf.keras.Model(
            inputs=[x, state_input_holder],
            outputs=[output])

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
        train_recurrent_per_type=False,
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
