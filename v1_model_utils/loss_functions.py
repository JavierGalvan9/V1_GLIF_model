import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Layer


class StiffRegularizer(tf.keras.regularizers.Regularizer):
    def __init__(self, strength, initial_value):
        super().__init__()
        self._strength = strength
        self._initial_value = tf.Variable(initial_value, trainable=False)

    def __call__(self, x):
        return self._strength * tf.reduce_sum(tf.square(x - self._initial_value))

        
def sample_firing_rates(firing_rates, n_neurons, rnd_seed):
    sorted_firing_rates = np.sort(firing_rates)
    percentiles = (np.arange(firing_rates.shape[-1])).astype(np.float32) / (firing_rates.shape[-1] - 1)
    rate_rd = np.random.RandomState(seed=rnd_seed)
    x_rand = rate_rd.uniform(size=n_neurons)
    target_firing_rates = np.sort(np.interp(x_rand, percentiles, sorted_firing_rates))
    
    return target_firing_rates


def huber_quantile_loss(u, tau, kappa, dtype=tf.float32):
    tau = tf.cast(tau, dtype)
    num = tf.abs(tau - tf.cast(u <= 0, dtype))

    branch_1 = num / (2 * kappa) * tf.square(u)
    branch_2 = num * (tf.abs(u) - 0.5 * kappa)
    return tf.where(tf.abs(u) <= kappa, branch_1, branch_2)

### To calculate the loss of firing rates between neuron types
def compute_spike_rate_target_loss(_spikes, target_rates, dtype=tf.float32):
    # TODO: define this function
    # target_rates is a dictionary that contains all the cell types.
    # I should iterate on them, and add the cost for each one at the end.
    # spikes will have a shape of (batch_size, n_steps, n_neurons)
    # losses = []
    total_loss = tf.constant(0.0, dtype=dtype)
    rates = tf.reduce_mean(_spikes, (0, 1))
    # if core_mask is not None:
    #     core_neurons_ids = np.where(core_mask)[0]

    for i, (key, value) in enumerate(target_rates.items()):
        if tf.size(value["neuron_ids"]) != 0:
            _rate_type = tf.gather(rates, value["neuron_ids"])
            target_rate = value["sorted_target_rates"]
            # if core_mask is not None:
            #     key_core_mask = np.isin(value["neuron_ids"], core_neurons_ids)
            #     neuron_ids =  np.where(key_core_mask)[0]
            #     _rate_type = tf.gather(rates, neuron_ids)
            #     target_rate = value["sorted_target_rates"][key_core_mask]
            # else:
            #     _rate_type = tf.gather(rates, value["neuron_ids"])
            #     target_rate = value["sorted_target_rates"]

            loss_type = compute_spike_rate_distribution_loss(_rate_type, target_rate, dtype=dtype)
            mean_loss_type = tf.reduce_mean(loss_type)
        else:
            mean_loss_type = tf.constant(0, dtype=dtype)

        # losses.append(mean_loss_type)
        total_loss += mean_loss_type
        
    # total_loss = tf.reduce_sum(losses, axis=0)
    return total_loss

def compute_spike_rate_distribution_loss(_rates, target_rate, dtype=tf.float32):
    # tf.print(f"target_rate: {target_rate}")
    # ind = tf.range(target_rate.shape[0])
    # rand_ind = tf.random.shuffle(ind)
    # _rate = tf.gather(_rates, rand_ind)
    # sorted_rate = tf.sort(_rate)
    sorted_rate = tf.sort(_rates)
    # u = target_rate - sorted_rate
    u = sorted_rate - target_rate
    # tau = (tf.range(target_rate.shape[0]), dtype) + 1) / target_rate.shape[0]
    tau = (tf.range(target_rate.shape[0]) + 1) / target_rate.shape[0]
    # loss = huber_quantile_loss(u, tau, 0.002, dtype=dtype)
    loss = huber_quantile_loss(u, tau, 0.1, dtype=dtype)

    return loss


class SpikeRateDistributionTarget:
    """ Instead of regularization, treat it as a target.
        The main difference is that this class will calculate the loss
        for each subtypes of the neurons."""
    def __init__(self, target_rates, rate_cost=.5, pre_delay=None, post_delay=None, core_mask=None, dtype=tf.float32):
        self._rate_cost = rate_cost
        self._target_rates = target_rates
        self._pre_delay = pre_delay
        self._post_delay = post_delay
        self._core_mask = core_mask
        self._dtype = dtype

    def __call__(self, spikes, trim=True):
        if trim:
            if self._pre_delay is not None:
                spikes = spikes[:, self._pre_delay:, :]
            if self._post_delay is not None and self._post_delay != 0:
                spikes = spikes[:, :-self._post_delay, :]
        if self._core_mask is not None:
            spikes = tf.boolean_mask(spikes, self._core_mask, axis=2)
        reg_loss = compute_spike_rate_target_loss(spikes, self._target_rates, dtype=self._dtype) * self._rate_cost
        return reg_loss


# class SpikeRateDistributionRegularization:
#     def __init__(self, target_rates, rate_cost=0.5, dtype=tf.float32):
#         self._rate_cost = rate_cost
#         self._target_rates = target_rates
#         self._dtype = dtype

#     def __call__(self, spikes):
#         reg_loss = (
#             compute_spike_rate_distribution_loss(spikes, self._target_rates, dtype=self._dtype)
#             * self._rate_cost
#         )
#         reg_loss = tf.reduce_sum(reg_loss)

#         return reg_loss


class VoltageRegularization:
    def __init__(self, cell, voltage_cost=1e-5, dtype=tf.float32, core_mask=None):
        self._voltage_cost = voltage_cost
        self._cell = cell
        self._dtype = dtype
        self._core_mask = core_mask

        if core_mask is None:
            self.voltage_offset = self._cell.voltage_offset
            self.voltage_scale = self._cell.voltage_scale
        else:
            self.voltage_offset = tf.boolean_mask(self._cell.voltage_offset, core_mask)
            self.voltage_scale = tf.boolean_mask(self._cell.voltage_scale, core_mask)

    def __call__(self, voltages):
        if self._core_mask is not None:
            voltages = tf.boolean_mask(voltages, self._core_mask, axis=2)
            
        voltage_32 = (voltages - self.voltage_offset) / self.voltage_scale
        v_pos = tf.square(tf.nn.relu(voltage_32 - 1.0))
        v_neg = tf.square(tf.nn.relu(-voltage_32 + 1.0))
        voltage_loss = tf.reduce_mean(tf.reduce_mean(v_pos + v_neg, -1)) * self._voltage_cost
        return voltage_loss



class CustomMeanLayer(Layer):
    def call(self, inputs):
        spike_rates, mask = inputs
        masked_data = tf.boolean_mask(spike_rates, mask)
        return tf.reduce_mean(masked_data)


class OrientationSelectivityLoss:
    def __init__(self, tuning_angles, osi_cost=1e-5, pre_delay=None, post_delay=None, dtype=tf.float32, core_mask=None, method="crowd_osi", subtraction_ratio=1.0, layer_info=None):
        self._tuning_angles = tuning_angles
        self._osi_cost = osi_cost
        self._pre_delay = pre_delay
        self._post_delay = post_delay
        self._dtype = dtype
        self._core_mask = core_mask
        self._method = method
        self._subtraction_ratio = subtraction_ratio  # only for crowd_spikes method
        if (self._core_mask is not None) and (self._method == "crowd_spikes"):
            self._tuning_angles = tf.boolean_mask(self._tuning_angles, self._core_mask)
        self._layer_info = layer_info  # needed for neuropixels_fr method
        # the layer_info should be a dictionary that contains
        # the cell id of the corresponding layer.
        # the keys should be something like "EXC_L23" or "PV_L5"
        

    def vonmises_model_fr(self, structure, population):
        from scipy.stats import vonmises
        paramdic = self._von_mises_params
        _params = paramdic[structure][population]
        if len(_params) == 4:
            mu, kappa, a, b = _params
        vonmises_pdf = vonmises(kappa, loc=mu).pdf
        
        angles = np.deg2rad(np.arange(-85, 86, 10)) * 2  # *2 needed to make it proper model
        model_fr = a + b * vonmises_pdf(angles)

        return model_fr

    def calculate_delta_angle(self, stim_angle, tuning_angle):
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
        # # do it twice to make sure everything is between -90 and 90.
        delta_angle = tf.where(delta_angle > 90, delta_angle - 180, delta_angle)
        delta_angle = tf.where(delta_angle < -90, delta_angle + 180, delta_angle)

        return delta_angle
    
    def __call__(self, spikes, angle, trim):
        if self._method == "crowd_osi":
            return self.crowd_osi_loss(spikes, angle, trim)
        elif self._method == "crowd_spikes":
            return self.crowd_spikes_loss(spikes, angle, trim)
        elif self._method == "neuropixels_fr":
            return self.neuropixels_fr_loss(spikes, angle, trim)
        
        
    def neuropixels_fr_loss(self, spikes, angle, trim=True):
        # if the trget fr is not set, construct them
        if not hasattr(self, "_target_frs"):

            # self._von_mises_params = np.load("GLIF_network/param_dict_orientation.npy")
            # pickle instead
            with open("GLIF_network/param_dict_orientation.pkl", 'rb') as f:
                self._von_mises_params = pickle.load(f)
            # get the model values with 10 degree increments 
            structure = "VISp"
            self._target_frs = {}
            for key in self._layer_info.keys():
                self._target_frs[key] = self.vonmises_model_fr(structure, key)
                # TODO: convert it to tensor if needed.
        
        spikes = self.spike_trimming(spikes, trim)
        # assuming 1 ms bins
        spike_rates = tf.reduce_mean(spikes, axis=[0, 1]) / spikes.shape[1] * 1000
        angle_bins = tf.constant(np.arange(-90, 91, 10), dtype=tf.float32)
        nbins = angle_bins.shape[0] - 1
        # now, process each layer
        # losses = tf.TensorArray(tf.float32, size=len(self._layer_info))
        losses = []
        delta_angle = self.calculate_delta_angle(angle, self._tuning_angles)
        
        custom_mean_layer = CustomMeanLayer()

        
        for key, value in self._layer_info.items():
            # first, calculate delta_angle
            
            # rates = tf.TensorArray(tf.float32, size=nbins)
            rates_list = []
            for i in range(nbins):
                mask = (delta_angle >= angle_bins[i]) & (delta_angle < angle_bins[i+1])
                # take the intersection with core mask
                mask = tf.logical_and(mask, self._core_mask)
                mask = tf.logical_and(mask, value)
                # mask = mask.flatten()
                # doesn't work.
                mask = tf.reshape(mask, [-1])
                mean_val = custom_mean_layer([spike_rates, mask])
                # rates_ = rates.write(i, mean_val)
                rates_list.append(mean_val)
                # rates = rates.write(i, tf.reduce_mean(tf.boolean_mask(spike_rates, mask)))

            # calculate the loss
            # rates = rates.stack()
            rates = tf.stack(rates_list)
            loss = tf.reduce_mean(tf.square(rates - self._target_frs[key]))
            # if key == "EXC_L6":
                # print the results!
                # tf.print("Layer6: ", rates)
                # tf.print("target: ", self._target_frs[key])
            # losses = losses.write(i, loss)
            losses.append(loss)
        
        # final_loss = tf.reduce_sum(losses.stack()) * self._osi_cost
        final_loss = tf.reduce_mean(tf.stack(losses)) * self._osi_cost
        return final_loss
        
    
    def crowd_osi_loss(self, spikes, angle, trim=True):
        angle = tf.cast(angle, self._dtype) 
        delta_angle = tf.expand_dims(angle, axis=1) -  self._tuning_angles
        delta_angle = delta_angle * (np.pi / 180)

        if self._core_mask is not None:
            spikes = tf.boolean_mask(spikes, self._core_mask, axis=2)
            
        # if self._pre_delay is not None:
        #     spikes = spikes[:, self._pre_delay:, :]
        # if self._post_delay is not None and self._post_delay != 0:
        #     spikes = spikes[:, :-self._post_delay, :]
        spikes = self.spike_trimming(spikes, trim)


    #     # sum spikes in _z, and multiply with delta_angle.
    #     mean_spikes = tf.reduce_mean(spikes, axis=[1]) 

    #     # Convert mean_spikes to a complex tensor with zero imaginary part
    #     mean_spikes = tf.cast(mean_spikes, tf.complex64)

    #     # Calculate weighted responses for OSI numerator
    #     # Adjust for preferred orientation by incorporating e^(2i(theta - theta_pref))
    #     weighted_responses_numerator = mean_spikes * tf.exp(tf.complex(0.0, 2.0 * delta_angle))
    #     approximated_numerator = tf.reduce_sum(weighted_responses_numerator)
        
    #     # Calculate denominator as the sum of mean_spikes
    #     approximated_denominator = tf.reduce_sum(mean_spikes)
        
    #     # Calculate OSI approximation
    #     osi_approx = tf.abs(approximated_numerator / tf.cast(approximated_denominator, tf.complex64))

    #     return tf.square(osi_approx - 1) * self._osi_cost

    def spike_trimming(self, spikes, trim=True):
        # remove pre and post delays
        if not trim:
            return spikes
        
        if self._pre_delay is not None:
            spikes = spikes[:, self._pre_delay:, :]
        if self._post_delay is not None and self._post_delay != 0:
            spikes = spikes[:, :-self._post_delay, :]
        return spikes
    
    def crowd_spikes_loss(self, spikes, angle, trim=True):
        # I need to access the tuning angle. of all the neurons.
        angle = tf.cast(angle, self._dtype)

        if self._core_mask is not None:
            spikes = tf.boolean_mask(spikes, self._core_mask, axis=2)
            
        # if self._pre_delay is not None:
        #     spikes = spikes[:, self._pre_delay:, :]
        # if self._post_delay is not None and self._post_delay != 0:
        #     spikes = spikes[:, :-self._post_delay, :]
        spikes = self.spike_trimming(spikes, trim)

        delta_angle = self.calculate_delta_angle(angle, self._tuning_angles)
        # sum spikes in _z, and multiply with delta_angle.
        mean_spikes = tf.reduce_mean(spikes, axis=[1]) 
        mean_angle = mean_spikes * delta_angle
        # Here, the expected value with random firing to subtract
        # (this prevents the osi loss to drive the firing rates to go to zero.)
        expected_sum_angle = tf.reduce_mean(mean_spikes) * 45
        
        angle_loss = tf.reduce_mean(tf.abs(mean_angle)) - expected_sum_angle * self._subtraction_ratio
        return tf.abs(angle_loss) * self._osi_cost
    
