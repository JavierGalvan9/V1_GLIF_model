import numpy as np
import tensorflow as tf


def gauss_pseudo(v_scaled, sigma, amplitude):
    return tf.math.exp(-tf.square(v_scaled) / tf.square(sigma)) * amplitude


def pseudo_derivative(v_scaled, dampening_factor):
    return dampening_factor * tf.maximum(1 - tf.abs(v_scaled), 0)

def slayer_pseudo(v_scaled, sigma, amplitude):
    return tf.math.exp(-sigma * tf.abs(v_scaled)) * amplitude


@tf.custom_gradient
def spike_gauss_16(v_scaled, sigma, amplitude):
    z_ = tf.greater(v_scaled, 0.)
    z_ = tf.cast(z_, tf.float16)

    def grad(dy):
        de_dz = dy
        dz_dv_scaled = gauss_pseudo(v_scaled, sigma, amplitude)

        de_dv_scaled = de_dz * dz_dv_scaled

        return [de_dv_scaled,
                tf.zeros_like(sigma), tf.zeros_like(amplitude)]

    return tf.identity(z_, name='spike_gauss'), grad


@tf.custom_gradient
def spike_gauss(v_scaled, sigma, amplitude):
    z_ = tf.greater(v_scaled, 0.)
    z_ = tf.cast(z_, tf.float32)

    def grad(dy):
        de_dz = dy
        dz_dv_scaled = gauss_pseudo(v_scaled, sigma, amplitude)
        de_dv_scaled = de_dz * dz_dv_scaled

        return [de_dv_scaled,
                tf.zeros_like(sigma), tf.zeros_like(amplitude)]

    return tf.identity(z_, name='spike_gauss'), grad

@tf.custom_gradient
def spike_slayer(v_scaled, sigma, amplitude):
    z_ = tf.greater(v_scaled, 0.)
    z_ = tf.cast(z_, tf.float32)

    def grad(dy):
        de_dz = dy
        dz_dv_scaled = slayer_pseudo(v_scaled, sigma, amplitude)

        de_dv_scaled = de_dz * dz_dv_scaled

        return [de_dv_scaled,
                tf.zeros_like(sigma), tf.zeros_like(amplitude)]

    return tf.identity(z_, name='spike_slayer'), grad


@tf.custom_gradient
def spike_function(v_scaled, dampening_factor):
    z_ = tf.greater(v_scaled, 0.)
    z_ = tf.cast(z_, tf.float32)

    def grad(dy):
        de_dz = dy
        dz_dv_scaled = pseudo_derivative(v_scaled, dampening_factor)

        de_dv_scaled = de_dz * dz_dv_scaled

        return [de_dv_scaled,
                tf.zeros_like(dampening_factor)]

    return tf.identity(z_, name="spike_function"), grad


@tf.custom_gradient
def spike_function_16(v_scaled, dampening_factor):
    z_ = tf.greater(v_scaled, 0.)
    z_ = tf.cast(z_, tf.float16)

    def grad(dy):
        de_dz = dy
        dz_dv_scaled = pseudo_derivative(v_scaled, dampening_factor)

        de_dv_scaled = de_dz * dz_dv_scaled

        return [de_dv_scaled,
                tf.zeros_like(dampening_factor)]

    return tf.identity(z_, name="spike_function"), grad


@tf.custom_gradient
def spike_function_b16(v_scaled, dampening_factor):
    z_ = tf.greater(v_scaled, 0.)
    z_ = tf.cast(z_, tf.bfloat16)

    def grad(dy):
        de_dz = dy
        dz_dv_scaled = pseudo_derivative(v_scaled, dampening_factor)

        de_dv_scaled = de_dz * dz_dv_scaled

        return [de_dv_scaled,
                tf.zeros_like(dampening_factor)]

    return tf.identity(z_, name="spike_function"), grad


def exp_convolve(tensor, decay=.8, reverse=False, initializer=None, axis=0):
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


class SparseLayer(tf.keras.layers.Layer):
    def __init__(self, indices, weights, dense_shape, bkg_weights, lr_scale=1., dtype=tf.float32, **kwargs):
        super().__init__(**kwargs)
        self._indices = indices
        self._weights = weights
        self._dense_shape = dense_shape
        # self._max_batch = int(2**31 / weights.shape[0]) #int(2**40 / weights.shape[0]) #int(2**31 / weights.shape[0])
        self._max_batch = int(2**40 / weights.shape[0]) #int(2**40 / weights.shape[0])
        self._dtype = dtype        
        self._bkg_weights = bkg_weights
        self._lr_scale = lr_scale

    def call(self, inp):
        tf_shp = tf.unstack(tf.shape(inp))
        shp = inp.shape.as_list()
        for i, a in enumerate(shp):
            if a is None:
                shp[i] = tf_shp[i]

        sparse_w_in = tf.sparse.SparseTensor(
            self._indices, self._weights, self._dense_shape)
        # print(sparse_w_in.shape) # (923696, 17400)
        inp = tf.reshape(inp, (shp[0] * shp[1], shp[2]))
        # print(inp.shape) #(2500, 17400)
        # print(shp)#[1, 2500, 17400]
        # print(self._weights.shape) #(3506881,)
        # print(self._max_batch) #612
        if shp[0] * shp[1] < self._max_batch:
            input_current = tf.sparse.sparse_dense_matmul(
                sparse_w_in, tf.cast(inp, tf.float32), adjoint_b=True)
            input_current = tf.transpose(input_current)
            input_current = tf.cast(input_current, self._dtype)
        else:
            _batch_size = shp[0] * shp[1] #2500
            _n_iter = tf.ones((), tf.int32)
            # while _batch_size > self._max_batch:
            for _i in tf.range(int(np.log(self._max_batch) * 2)): #12
                _sel = tf.greater(_batch_size, self._max_batch) #True
                _batch_size = tf.where(_sel, tf.cast(
                    _batch_size / 2, tf.int32), _batch_size)
                # _n_iter = tf.where(_sel, _n_iter + 1, _n_iter)
                _n_iter = tf.where(_sel, _n_iter * 2, _n_iter) # creo que esto es lo correcto

            # _batch_size, _n_iter = 312, 4
            results = tf.TensorArray(self._dtype, size=_n_iter)
            for _i in tf.range(_n_iter):
                partial_input_current = tf.sparse.sparse_dense_matmul(
                    sparse_w_in, tf.cast(inp[_i * _batch_size:(_i + 1) * _batch_size], tf.float32), adjoint_b=True)
                partial_input_current = tf.cast(
                    tf.transpose(partial_input_current), self._dtype)
                print(_i, partial_input_current.shape)
                results = results.write(_i, partial_input_current)
                _i += 1
            input_current = results.stack()

        rest_of_brain = tf.reduce_sum(tf.cast(
            tf.random.uniform((shp[0], shp[1], 10)) < .1, self._compute_dtype), -1)
        noise_input = tf.cast(
            self._bkg_weights[None, None], self._compute_dtype) * rest_of_brain[..., None] / 10.
        input_current = tf.reshape(
            input_current, (shp[0], shp[1], -1)) + noise_input
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


class BillehColumn(tf.keras.layers.Layer):
    def __init__(self, network, input_population, bkg_weights,
                 dt=1., gauss_std=.5, dampening_factor=.3, recurrent_dampening_factor=.4,
                 input_weight_scale=1., recurrent_weight_scale=1.,
                 lr_scale=1., spike_gradient=False, max_delay=5, pseudo_gauss=False,
                 train_recurrent=True, train_input=True, hard_reset=True):
        super().__init__()
        self._params = network['node_params']
        
        # Rescale the voltages to have them near 0, as we wanted the effective step size 
        # for the weights to be normalized when learning (weights are scaled similarly)
        voltage_scale = self._params['V_th'] - self._params['E_L']
        voltage_offset = self._params['E_L']
        self._params['V_th'] = (self._params['V_th'] -
                                voltage_offset) / voltage_scale
        self._params['E_L'] = (self._params['E_L'] -
                               voltage_offset) / voltage_scale
        self._params['V_reset'] = (
            self._params['V_reset'] - voltage_offset) / voltage_scale
        self._params['asc_amps'] = self._params['asc_amps'] / \
            voltage_scale[..., None]   # _params['asc_amps'] has shape (111, 2)

        self._node_type_ids = network['node_type_ids']
        self._dt = dt
        self._recurrent_dampening = recurrent_dampening_factor
        self._pseudo_gauss = pseudo_gauss

        self._lr_scale = lr_scale

        self._spike_gradient = spike_gradient
        
        self._hard_reset = hard_reset

        n_receptors = network['node_params']['tau_syn'].shape[1] # we have 4 receptors (soma, dendrites, etc) for each neuron
        self._n_receptors = n_receptors
        self._n_neurons = network['n_nodes']
        self._dampening_factor = tf.cast(dampening_factor, self._compute_dtype)
        self._gauss_std = tf.cast(gauss_std, self._compute_dtype)

        tau = self._params['C_m'] / self._params['g'] # determine the membrane time decay constant
        self._decay = np.exp(-dt / tau)
        self._current_factor = 1 / \
            self._params['C_m'] * (1 - self._decay) * tau
        self._syn_decay = np.exp(-dt / np.array(self._params['tau_syn']))
        self._psc_initial = np.e / np.array(self._params['tau_syn'])

        # synapses: target_ids, source_ids, weights, delays
        # this are the axonal delays
        self.max_delay = int(
            np.round(np.min([np.max(network['synapses']['delays']), max_delay])))

        self.state_size = (
            self._n_neurons * self.max_delay,                # z buffer
            self._n_neurons,                                 # v
            self._n_neurons,                                 # r
            self._n_neurons,                                 # asc 1
            self._n_neurons,                                 # asc 2
            n_receptors * self._n_neurons,                   # psc rise
            n_receptors * self._n_neurons,                   # psc
        )

        def _f(_v, trainable=False):
            return tf.Variable(tf.cast(self._gather(_v), self._compute_dtype), trainable=trainable)

        def inv_sigmoid(_x):
            return tf.math.log(_x / (1 - _x))

        def custom_val(_v, trainable=False):
            _v = tf.Variable(tf.cast(inv_sigmoid(self._gather(
                _v)), self._compute_dtype), trainable=trainable)

            def _g():
                return tf.nn.sigmoid(_v.read_value())

            return _v, _g

        self.v_reset = _f(self._params['V_reset'])
        self.syn_decay = _f(self._syn_decay)
        self.psc_initial = _f(self._psc_initial)
        self.t_ref = _f(self._params['t_ref'])  # refractory time
        # print(set(self._params['t_ref']))
        self.asc_amps = _f(self._params['asc_amps'], trainable=False)
        _k = self._params['k']
        self.param_k, self.param_k_read = custom_val(_k, trainable=False)
        self.v_th = _f(self._params['V_th'])
        self.e_l = _f(self._params['E_L'])
        self.param_g = _f(self._params['g'])
        self.decay = _f(self._decay)
        self.current_factor = _f(self._current_factor)
        self.voltage_scale = _f(voltage_scale)
        self.voltage_offset = _f(voltage_offset)
        self.recurrent_weights = None

        indices, weights, dense_shape = \
            network['synapses']['indices'], network['synapses']['weights'], network['synapses']['dense_shape']
        weights = weights / \
            voltage_scale[self._node_type_ids[indices[:, 0] //
                                             self._n_receptors]]  # scale down the weights

        delays = np.round(np.clip(
            network['synapses']['delays'], dt, self.max_delay) / dt).astype(np.int32)
        dense_shape = dense_shape[0], self.max_delay * dense_shape[1] 
        # Notice that in dense_shape, the first column (presynaptic neuron) has size receptors*n_neurons 
        # and the second column (postsynaptic neuron) has size max_delay*n_neurons
        
        indices[:, 1] = indices[:, 1] + self._n_neurons * (delays - 1)
        
        
        weights = weights.astype(np.float32)
        print(f'> Recurrent synapses {len(indices)}')
        input_weights = input_population['weights'].astype(np.float32)
        input_indices = input_population['indices']
        input_weights = input_weights / \
            voltage_scale[self._node_type_ids[input_indices[:,
                                                            0] // self._n_receptors]]
        print(f'> Input synapses {len(input_indices)}')

        input_dense_shape = (self._n_receptors *
                             self._n_neurons, input_population['n_inputs'])

        self.recurrent_weight_positive = tf.Variable(
            weights >= 0., name='recurrent_weights_sign', trainable=False)
        self.input_weight_positive = tf.Variable(
            input_weights >= 0., name='input_weights_sign', trainable=False)
        self.recurrent_weight_values = tf.Variable(
            weights * recurrent_weight_scale / lr_scale, name='sparse_recurrent_weights',
            constraint=SignedConstraint(self.recurrent_weight_positive),
            trainable=train_recurrent)
        self.recurrent_indices = tf.Variable(indices, trainable=False)
        self.recurrent_dense_shape = dense_shape

        self.input_weight_values = tf.Variable(
            input_weights * input_weight_scale / lr_scale, name='sparse_input_weights',
            constraint=SignedConstraint(self.input_weight_positive),
            trainable=train_input)
        self.input_indices = tf.Variable(input_indices, trainable=False)
        self.input_dense_shape = input_dense_shape
        bkg_weights = bkg_weights / \
            np.repeat(voltage_scale[self._node_type_ids], self._n_receptors)
        self.bkg_weights = tf.Variable(
            bkg_weights * 10., name='rest_of_brain_weights', trainable=train_input)

    def compute_input_current(self, inp):
        tf_shp = tf.unstack(tf.shape(inp))
        shp = inp.shape.as_list()
        for i, a in enumerate(shp):
            if a is None:
                shp[i] = tf_shp[i]

        sparse_w_in = tf.sparse.SparseTensor(
            self.input_indices, self.input_weight_values, self.input_dense_shape)
        inp = tf.reshape(inp, (shp[0] * shp[1], shp[2]))
        input_current = tf.sparse.sparse_dense_matmul(
            sparse_w_in, tf.cast(inp, tf.float32), adjoint_b=True)
        input_current = tf.transpose(input_current)

        input_current = tf.reshape(
            input_current, (shp[0], shp[1], 4 * self._n_neurons))
        return input_current

    def zero_state(self, batch_size, dtype=tf.float32):
        # The neurons membrane voltage start the simulation at their reset value
        v0 = tf.ones((batch_size, self._n_neurons), dtype) * \
                tf.cast(self.v_th * .0 + 1. * self.v_reset, dtype)
        z0_buf = tf.zeros(
            (batch_size, self._n_neurons * self.max_delay), dtype)
        r0 = tf.zeros((batch_size, self._n_neurons), dtype)
        asc_10 = tf.zeros((batch_size, self._n_neurons), dtype)
        asc_20 = tf.zeros((batch_size, self._n_neurons), dtype)
        psc_rise0 = tf.zeros(
            (batch_size, self._n_neurons * self._n_receptors), dtype)
        psc0 = tf.zeros(
            (batch_size, self._n_neurons * self._n_receptors), dtype)
        return z0_buf, v0, r0, asc_10, asc_20, psc_rise0, psc0

    def _gather(self, prop):
        return tf.gather(prop, self._node_type_ids)

    def call(self, inputs, state, constants=None):
        batch_size = inputs.shape[0]
        if batch_size is None:
            batch_size = tf.shape(inputs)[0]
        external_current = inputs
        if self._spike_gradient:
            state_input = tf.zeros((1,))
        else:
            state_input = tf.zeros((4,))
        if constants is not None:
            if self._spike_gradient:
                external_current = inputs[:,:self._n_neurons * self._n_receptors]
                state_input = inputs[:, self._n_neurons * self._n_receptors:]
            else:
                external_current = inputs[:,:self._n_neurons * self._n_receptors]
                state_input = inputs[:, self._n_neurons * self._n_receptors:]
                state_input = tf.reshape(state_input, (batch_size, self._n_neurons, 4))
        # external_current = inputs
        z_buf, v, r, asc_1, asc_2, psc_rise, psc = state

        shaped_z_buf = tf.reshape(z_buf, (-1, self.max_delay, self._n_neurons)) #shape (4, 50000)
        prev_z = shaped_z_buf[:, 0] # previous spikes with shape (50000)

        psc_rise = tf.reshape(psc_rise, (batch_size, self._n_neurons, self._n_receptors))
        psc = tf.reshape(psc, (batch_size, self._n_neurons, self._n_receptors))

        dampened_z_buf = z_buf * self._recurrent_dampening
        rec_z_buf = tf.stop_gradient(z_buf - dampened_z_buf) + dampened_z_buf # here we use tf.stop_gradient to prevent the term (z_buf - dampened_z_buf) to be trained

        sparse_w_rec = tf.sparse.SparseTensor(
            self.recurrent_indices, self.recurrent_weight_values, self.recurrent_dense_shape)

        i_rec = tf.sparse.sparse_dense_matmul(
            sparse_w_rec, tf.cast(rec_z_buf, tf.float32), adjoint_b=True)
        i_rec = tf.transpose(i_rec)
        
        rec_inputs = tf.cast(i_rec, self._compute_dtype)
        rec_inputs = tf.reshape(
            rec_inputs + external_current, (batch_size, self._n_neurons, self._n_receptors))
        rec_inputs = rec_inputs * self._lr_scale

        if constants is not None and not self._spike_gradient:
            rec_inputs = rec_inputs + state_input * self._lr_scale
        
        new_psc_rise = psc_rise * self.syn_decay + rec_inputs * self.psc_initial
        new_psc = psc * self.syn_decay + self._dt * self.syn_decay * psc_rise

        # New r is a variable that accounts for the refractory period in which
        # a neuron cannot spike
        new_r = tf.nn.relu(r + prev_z * self.t_ref - self._dt) # =max(r + prev_z * self.t_ref - self._dt, 0)

        k = self.param_k_read()
        asc_amps = self.asc_amps
        new_asc_1 = tf.exp(-self._dt * k[:, 0]) * asc_1 + prev_z * asc_amps[:, 0]
        new_asc_2 = tf.exp(-self._dt * k[:, 1]) * asc_2 + prev_z * asc_amps[:, 1]

        if constants is not None and self._spike_gradient:
            input_current = tf.reduce_sum(psc, -1) + state_input
        else:
            input_current = tf.reduce_sum(psc, -1)
        
        decayed_v = self.decay * v
        gathered_g = self.param_g * self.e_l
        c1 = input_current + asc_1 + asc_2 + gathered_g
        
        if self._hard_reset:
            # Here we keep the voltage at the reset value during the refractory period
            new_v = tf.where(new_r > 0., self.v_reset, decayed_v + self.current_factor * c1)
            # Here we make a hard reset and let the voltage freely evolve but we do not let the 
            # neuron spike during the refractory period
            # new_v = tf.where(prev_z > 0., self.v_reset, decayed_v + self.current_factor * c1)
        else:
            reset_current = prev_z * (self.v_reset - self.v_th)
            new_v = decayed_v + self.current_factor * c1 + reset_current
        
        normalizer = self.v_th - self.e_l
        v_sc = (new_v - self.v_th) / normalizer
        new_z = spike_slayer(v_sc, 5., .6) # If v_sc is greater than 0 then there is a spike
        
        if False:
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
        
        new_z = tf.where(new_r > 0., tf.zeros_like(new_z), new_z)
        new_psc = tf.reshape(new_psc, (batch_size, self._n_neurons * self._n_receptors))
        new_psc_rise = tf.reshape(new_psc_rise, (batch_size, self._n_neurons * self._n_receptors))
        
        # new_z = tf.cast(new_z, tf.float16)
        
        new_shaped_z_buf = tf.concat((new_z[:, None], shaped_z_buf[:, :-1]), 1)
        new_z_buf = tf.reshape(new_shaped_z_buf, (-1, self._n_neurons * self.max_delay))
        
        outputs = (new_z, new_v * self.voltage_scale + self.voltage_offset, input_current + new_asc_1 + new_asc_2)
        new_state = (new_z_buf, new_v, new_r, new_asc_1, new_asc_2, new_psc_rise, new_psc)

        return outputs, new_state


def huber_quantile_loss(u, tau, kappa):
    branch_1 = tf.abs(tau - tf.cast(u <= 0, tf.float32)) / \
        (2 * kappa) * tf.square(u)
    branch_2 = tf.abs(tau - tf.cast(u <= 0, tf.float32)) * \
        (tf.abs(u) - .5 * kappa)
    return tf.where(tf.abs(u) <= kappa, branch_1, branch_2)


def compute_spike_rate_distribution_loss(_spikes, target_rate):
    _rate = tf.reduce_mean(_spikes, (0, 1))
    ind = tf.range(target_rate.shape[0])
    rand_ind = tf.random.shuffle(ind)
    _rate = tf.gather(_rate, rand_ind)
    sorted_rate = tf.sort(_rate)
    u = target_rate - sorted_rate
    tau = (
        tf.cast(tf.range(target_rate.shape[0]), tf.float32) + 1) / target_rate.shape[0]
    loss = huber_quantile_loss(u, tau, .002)

    return loss


class SpikeRateDistributionRegularization:
    def __init__(self, target_rates, rate_cost=.5):
        self._rate_cost = rate_cost
        self._target_rates = target_rates

    def __call__(self, spikes):
        reg_loss = compute_spike_rate_distribution_loss(
            spikes, self._target_rates) * self._rate_cost
        reg_loss = tf.reduce_sum(reg_loss)

        return reg_loss


class VoltageRegularization:
    def __init__(self, cell, voltage_cost=1e-5):
        self._voltage_cost = voltage_cost
        self._cell = cell

    def __call__(self, voltages):
        voltage_32 = (tf.cast(voltages, tf.float32) -
                      self._cell.voltage_offset) / self._cell.voltage_scale
        v_pos = tf.square(tf.nn.relu(voltage_32 - 1.))
        v_neg = tf.square(tf.nn.relu(-voltage_32 + 1.))
        voltage_loss = tf.reduce_mean(tf.reduce_sum(
            v_pos + v_neg, -1)) * self._voltage_cost
        return voltage_loss


def create_model(network, input_population, bkg_weights, seq_len=100, n_input=10, n_output=2,
                 cue_duration=20, dtype=tf.float32, input_weight_scale=1., gauss_std=.5,
                 dampening_factor=.2, lr_scale=800., train_recurrent=True,
                 train_input=True, neuron_output=False, recurrent_dampening_factor=.5,
                 use_state_input=False, return_state=False, return_sequences=False, down_sample=50,
                 add_metric=True, max_delay=5, batch_size=None, pseudo_gauss=False,
                 hard_reset=True):

    # Create the input of the model
    x = tf.keras.layers.Input(shape=(seq_len, n_input,))
    neurons = network['n_nodes']
    state_input_holder = tf.keras.layers.Input(shape=(seq_len, neurons))
    state_input = tf.cast(tf.identity(state_input_holder), dtype) # tf.cast() changes the dtype

    if batch_size is None:
        batch_size = tf.shape(x)[0] # if batch_size is None just update after each seq_len
    else:
        batch_size = batch_size

    cell = BillehColumn(network, input_population, bkg_weights,
                        gauss_std=gauss_std, dampening_factor=dampening_factor,
                        input_weight_scale=input_weight_scale, lr_scale=lr_scale, spike_gradient=True,
                        recurrent_dampening_factor=recurrent_dampening_factor, max_delay=max_delay,
                        pseudo_gauss=pseudo_gauss, train_recurrent=train_recurrent, train_input=train_input,
                        hard_reset=hard_reset)

    zero_state = cell.zero_state(batch_size, dtype)
    if use_state_input:
        initial_state_holder = tf.nest.map_structure(
            lambda _x: tf.keras.layers.Input(shape=_x.shape[1:]), zero_state)
        rnn_initial_state = tf.nest.map_structure(
            tf.identity, initial_state_holder)
        constants = tf.zeros_like(rnn_initial_state[0][:, 0], dtype)
    else:
        rnn_initial_state = zero_state
        constants = tf.zeros((batch_size,))
  
    rnn_inputs = SparseLayer(
        cell.input_indices, cell.input_weight_values, cell.input_dense_shape,
        cell.bkg_weights, lr_scale=lr_scale, dtype=dtype, name='input_layer')(x)

    rnn_inputs = tf.cast(rnn_inputs, dtype)
    full_inputs = tf.concat((rnn_inputs, state_input), -1)

    rnn = tf.keras.layers.RNN(
        cell, return_sequences=True, return_state=return_state, name='rsnn')
    out = rnn(full_inputs, initial_state=rnn_initial_state,
              constants=constants)
    if return_state:
        hidden = out[0]
        new_state = out[1:]
    else:
        hidden = out
    spikes = hidden[0]
    voltage = hidden[1]
    rate = tf.cast(tf.reduce_mean(spikes, (1, 2)), tf.float32)

    if neuron_output:
        output_spikes = 1 / dampening_factor * spikes + \
            (1 - 1 / dampening_factor) * tf.stop_gradient(spikes)
        output = tf.gather(
            output_spikes, network['readout_neuron_ids'], axis=2)
        output = tf.reduce_mean(output, -1)
        scale = (1 + tf.nn.softplus(tf.keras.layers.Dense(1)
                                    (tf.zeros_like(output[:1, :1]))))
        thresh = tf.keras.layers.Dense(1)(tf.zeros_like(output))
        output = tf.stack([thresh[..., 0], output[..., -1]], -1) * scale
    else:
        output = tf.keras.layers.Dense(
            n_output, name='projection', trainable=True)(spikes)

    output = tf.keras.layers.Lambda(lambda _a: _a, name='prediction')(output)

    if return_sequences:
        mean_output = tf.reshape(
            output, (-1, int(seq_len / down_sample), down_sample, n_output))
        mean_output = tf.reduce_mean(mean_output, 2)
        mean_output = tf.nn.softmax(mean_output, axis=-1)
    else:
        mean_output = tf.reduce_mean(output[:, -cue_duration:], 1)
        mean_output = tf.nn.softmax(mean_output)

    if use_state_input:
        many_input_model = tf.keras.Model(
            inputs=[x, state_input_holder, initial_state_holder], outputs=mean_output)
    else:
        many_input_model = tf.keras.Model(
            inputs=[x, state_input_holder], outputs=mean_output)

    if add_metric:
        many_input_model.add_metric(rate, name='rate')

    return many_input_model
