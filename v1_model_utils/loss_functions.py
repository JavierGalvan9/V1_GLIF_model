import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Layer
import pandas as pd
import pickle as pkl
import os
import sys
sys.path.append(os.path.join(os.getcwd(), "v1_model_utils"))
import other_v1_utils


class StiffRegularizer(tf.keras.regularizers.Regularizer):
    def __init__(self, strength, initial_value):
        super().__init__()
        self._strength = strength
        self._initial_value = tf.Variable(initial_value, trainable=False)

    def __call__(self, x):
        return self._strength * tf.reduce_mean(tf.square(x - self._initial_value))
    
class L2Regularizer(tf.keras.regularizers.Regularizer):
    def __init__(self, strength, initial_value):
        super().__init__()
        self._strength = strength
        self._initial_value = tf.Variable(initial_value, trainable=False)

    def __call__(self, x):
        return self._strength * tf.reduce_mean(tf.square(x))

        
def sample_firing_rates(firing_rates, n_neurons, rnd_seed):
    sorted_firing_rates = np.sort(firing_rates)
    # percentiles = (np.arange(firing_rates.shape[-1])).astype(np.float32) / (firing_rates.shape[-1] - 1)
    percentiles = np.linspace(0, 1, sorted_firing_rates.size)
    rate_rd = np.random.RandomState(seed=rnd_seed)
    x_rand = rate_rd.uniform(low=0, high=1, size=n_neurons)
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
    total_loss = tf.constant(0.0, dtype=dtype)
    rates = tf.reduce_mean(_spikes, (0, 1))
    # if core_mask is not None:
    #     core_neurons_ids = np.where(core_mask)[0]
    cell_count = 0

    for key, value in target_rates.items():
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
            mean_loss_type = tf.reduce_sum(loss_type)
            cell_count += tf.size(value["neuron_ids"])
        else:
            mean_loss_type = tf.constant(0, dtype=dtype)

        # losses.append(mean_loss_type)
        total_loss += mean_loss_type
        
    total_loss = total_loss / float(cell_count)
    # total_loss = tf.reduce_sum(losses, axis=0)
    return total_loss

def compute_spike_rate_distribution_loss(_rates, target_rate, dtype=tf.float32):
    # Firstly we shuffle the current model rates to avoid bias towards a particular tuning angles (inherited from neurons ordering in the network)
    ind = tf.range(target_rate.shape[0])
    rand_ind = tf.random.shuffle(ind)
    _rates = tf.gather(_rates, rand_ind)
    sorted_rate = tf.sort(_rates)
    # u = target_rate - sorted_rate
    u = sorted_rate - target_rate
    # tau = (tf.range(target_rate.shape[0]), dtype) + 1) / target_rate.shape[0]
    tau = (tf.range(target_rate.shape[0]) + 1) / target_rate.shape[0]
    loss = huber_quantile_loss(u, tau, 0.002, dtype=dtype)
    # loss = huber_quantile_loss(u, tau, 0.1, dtype=dtype)

    return loss

def process_neuropixels_data(path=''):
    # Load data
    neuropixels_data_path = f'Neuropixels_data/cortical_metrics_1.4.csv'
    df_all = pd.read_csv(neuropixels_data_path, sep=",")
    # Exc and PV have sufficient number of cells, so we'll filter out non-V1 Exc and PV.
    # SST and VIP are small populations, so let's keep also non-V1 neurons
    exclude = (df_all["cell_type"].isnull() | df_all["cell_type"].str.contains("EXC") | df_all["cell_type"].str.contains("PV")) \
            & (df_all["ecephys_structure_acronym"] != 'VISp')
    df = df_all[~exclude]
    print(f"Original: {df_all.shape[0]} cells,   filtered: {df.shape[0]} cells")

    # Some cells have very large values of RF. They are likely not-good fits, so ignore.
    df.loc[(df["width_rf"] > 100), "width_rf"] = np.nan
    df.loc[(df["height_rf"] > 100), "height_rf"] = np.nan

    # Save the processed table
    df.to_csv(f'Neuropixels_data/v1_OSI_DSI_DF.csv', sep=" ", index=False)
    return df


def neuropixels_cell_type_to_cell_type(pop_name):
    # Convert pop_name in the neuropixels cell type to cell types. E.g, 'EXC_L23' -> 'L2/3 Exc', 'PV_L5' -> 'L5 PV'
    layer = pop_name.split('_')[1]
    class_name = pop_name.split('_')[0]
    if "2" in layer:
        layer = "L2/3"
    elif layer == "L1":
        return "L1 Htr3a"  # special case
    if class_name == "EXC":
        class_name = "Exc"
    if class_name == 'Htr3a':
        class_name = 'VIP'

    return f"{layer} {class_name}"


class SpikeRateDistributionTarget:
    """ Instead of regularization, treat it as a target.
        The main difference is that this class will calculate the loss
        for each subtypes of the neurons."""
    def __init__(self, network, rate_cost=.5, pre_delay=None, post_delay=None, 
                 data_dir='GLIF_network', core_mask=None, seed=42, dtype=tf.float32):
        self._network = network
        self._rate_cost = rate_cost
        self._pre_delay = pre_delay
        self._post_delay = post_delay
        self._core_mask = core_mask
        if self._core_mask is not None:
            self._np_core_mask = self._core_mask.numpy()
        self._data_dir = data_dir
        self._dtype = dtype
        self._seed = seed
        self.neuropixels_feature = 'Ave_Rate(Hz)'
        self._target_rates = self.get_neuropixels_firing_rates()

    def get_neuropixels_firing_rates(self):
        """
        Processes neuropixels data to obtain neurons average firing rates.

        Returns:
            dict: Dictionary containing rates and node_type_ids for each population query.
        """
        # Load data
        neuropixels_data_path = f'Neuropixels_data/v1_OSI_DSI_DF.csv'
        if not os.path.exists(neuropixels_data_path):
            np_df = process_neuropixels_data(path=neuropixels_data_path)
        else:
            np_df = pd.read_csv(neuropixels_data_path, index_col=0, sep=" ")

        area_node_types = pd.read_csv(os.path.join(self._data_dir, f'network/v1_node_types.csv'), sep=" ")

        # Define population queries
        query_mapping = {
            "i1H": 'ALL_L1',
            "e23": 'EXC_L23',
            "i23P": 'PV_L23',
            "i23S": 'SST_L23',
            "i23V": 'VIP_L23',
            "e4": 'EXC_L4',
            "i4P": 'PV_L4',
            "i4S": 'SST_L4',
            "i4V": 'VIP_L4',
            "e5": 'EXC_L5',
            "i5P": 'PV_L5',
            "i5S": 'SST_L5',
            "i5V": 'VIP_L5',
            "e6": 'EXC_L6',
            "i6P": 'PV_L6',
            "i6S": 'SST_L6',
            "i6V": 'VIP_L6'
        }

        # define the reverse mapping
        reversed_query_mapping = {v:k for k, v in query_mapping.items()}

        # Process rates
        type_rates_dict = {
                            reversed_query_mapping[cell_type]: np.append(subdf[self.neuropixels_feature].dropna().values / 1000, 0)
                            # reversed_query_mapping[cell_type]: np.sort(np.append(subdf[self.neuropixels_feature].dropna().values / 1000, 0)) # the rates are sorted again later so is redundant
                            for cell_type, subdf in np_df.groupby("cell_type")
                        }

        # Identify node_type_ids for each population query
        pop_ids = {query: area_node_types[area_node_types.pop_name.str.contains(query)].index.values for query in query_mapping.keys()}

        # Create a dictionary with rates and IDs
        target_firing_rates = {pop_query: {'rates': type_rates_dict[pop_query], 'ids': pop_ids[pop_query]} for pop_query in pop_ids.keys()}
        
        for key, value in target_firing_rates.items():
            # identify tne ids that are included in value["ids"]
            neuron_ids = np.where(np.isin(self._network["node_type_ids"], value["ids"]))[0]
            if self._core_mask is not None:
                # if core_mask is not None, use only neurons in the core
                neuron_ids = neuron_ids[self._np_core_mask[neuron_ids]]

            neuron_ids = tf.cast(neuron_ids, dtype=tf.int32)
            target_firing_rates[key]['neuron_ids'] = neuron_ids
            type_n_neurons = len(neuron_ids)
            sorted_target_rates = sample_firing_rates(value["rates"], type_n_neurons, self._seed)
            target_firing_rates[key]['sorted_target_rates'] = tf.cast(sorted_target_rates, dtype=tf.float32) 

        return target_firing_rates    

    def __call__(self, spikes, trim=True):
        if trim:
            if self._pre_delay is not None:
                spikes = spikes[:, self._pre_delay:, :]
            if self._post_delay is not None and self._post_delay != 0:
                spikes = spikes[:, :-self._post_delay, :]
        
        # if self._core_mask is not None:
        #     spikes = tf.boolean_mask(spikes, self._core_mask, axis=2)
        
        reg_loss = compute_spike_rate_target_loss(spikes, self._target_rates, dtype=self._dtype)
        
        return reg_loss * self._rate_cost

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
        voltage_loss = tf.reduce_mean(tf.reduce_mean(v_pos + v_neg, -1)) 

        return voltage_loss * self._voltage_cost



class CustomMeanLayer(Layer):
    def call(self, inputs):
        spike_rates, mask = inputs
        masked_data = tf.boolean_mask(spike_rates, mask)
        return tf.reduce_mean(masked_data)


class OrientationSelectivityLoss:
    def __init__(self, network=None, osi_cost=1e-5, pre_delay=None, post_delay=None, dtype=tf.float32, 
                 core_mask=None, method="crowd_osi", subtraction_ratio=1.0, layer_info=None):
        self._tuning_angles = tf.constant(network['tuning_angle'], dtype=dtype) 
        self._network = network
        self._osi_cost = osi_cost
        self._pre_delay = pre_delay
        self._post_delay = post_delay
        self._dtype = dtype
        self._core_mask = core_mask
        self._method = method
        self._subtraction_ratio = subtraction_ratio  # only for crowd_spikes method
        if (self._core_mask is not None) and (self._method == "crowd_spikes" or self._method == "crowd_osi"):
            self._tuning_angles = tf.boolean_mask(self._tuning_angles, self._core_mask)
        
        if self._method == "neuropixels_fr":
            self._layer_info = layer_info  # needed for neuropixels_fr method
            # the layer_info should be a dictionary that contains
            # the cell id of the corresponding layer.
            # the keys should be something like "EXC_L23" or "PV_L5"   

        elif self._method == "crowd_osi":
            # Get the target OSI
            self._target_osi = self.get_neuropixels_osi_dsi()

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
    
    def spike_trimming(self, spikes, trim=True):
        # remove pre and post delays
        if not trim:
            return spikes
        
        if self._pre_delay is not None:
            spikes = spikes[:, self._pre_delay:, :]
        if self._post_delay is not None and self._post_delay != 0:
            spikes = spikes[:, :-self._post_delay, :]
        return spikes
    
    def get_neuropixels_osi_dsi(self):
        """
        Processes neuropixels data to obtain neurons average firing rates.

        Returns:
            dict: Dictionary containing rates and node_type_ids for each population query.
        """
        # Load data
        neuropixels_data_path = f'Neuropixels_data/v1_OSI_DSI_DF.csv'
        if not os.path.exists(neuropixels_data_path):
            np_df = process_neuropixels_data(path=neuropixels_data_path)
        else:
            np_df = pd.read_csv(neuropixels_data_path, index_col=0, sep=" ")

        osi_dsi_df = np_df[['cell_type', 'OSI', 'DSI', "Ave_Rate(Hz)", "max_mean_rate(Hz)"]]
        nonresponding = osi_dsi_df["max_mean_rate(Hz)"] < 0.5
        osi_dsi_df.loc[nonresponding, "OSI"] = np.nan
        osi_dsi_df.loc[nonresponding, "DSI"] = np.nan
        osi_dsi_df = osi_dsi_df[osi_dsi_df["Ave_Rate(Hz)"] != 0]
        osi_dsi_df.dropna(inplace=True)
        osi_dsi_df["cell_type"] = osi_dsi_df["cell_type"].apply(neuropixels_cell_type_to_cell_type)
        # osi_df.groupby("cell_type")['OSI'].mean()
        # osi_df.groupby("cell_type")['OSI'].median()
        osi_target = osi_dsi_df.groupby("cell_type")['OSI'].mean()
        dsi_target = osi_dsi_df.groupby("cell_type")['DSI'].mean()

        original_pop_names = other_v1_utils.pop_names(self._network)
        if self._core_mask is not None:
            original_pop_names = original_pop_names[self._core_mask] 

        cell_types = np.array([other_v1_utils.pop_name_to_cell_type(pop_name, ignore_l5e_subtypes=True) for pop_name in original_pop_names])
        node_ids = np.arange(len(cell_types))
        cell_ids = {key: node_ids[cell_types == key] for key in set(osi_dsi_df['cell_type'])}

        # osi_target = osi_df.groupby("cell_type")['OSI'].mean()
        # osi_target = osi_df.groupby("cell_type")['OSI'].median()
        # osi_df.groupby("cell_type")['OSI'].median()
        # convert to dict
        osi_dsi_exp_dict = {key: {'OSI': val, 'DSI': dsi_target[key], 'ids': cell_ids[key]} for key, val in osi_target.to_dict().items()}

        return osi_dsi_exp_dict
        
        
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
    
    def neuropixels_fr_loss(self, spikes, angle, trim=True):
        # if the trget fr is not set, construct them
        if not hasattr(self, "_target_frs"):

            # self._von_mises_params = np.load("GLIF_network/param_dict_orientation.npy")
            # pickle instead
            with open("GLIF_network/param_dict_orientation.pkl", 'rb') as f:
                self._von_mises_params = pkl.load(f)
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
        
    
    def crowd_spikes_loss(self, spikes, angle, trim=True):
        # I need to access the tuning angle. of all the neurons.
        angle = tf.cast(angle, self._dtype)

        if self._core_mask is not None:
            spikes = tf.boolean_mask(spikes, self._core_mask, axis=2)
            
        spikes = self.spike_trimming(spikes, trim)

        delta_angle = self.calculate_delta_angle(angle, self._tuning_angles)
        # sum spikes in _z, and multiply with delta_angle.
        mean_spikes = tf.reduce_mean(spikes, axis=[1]) 
        mean_angle = mean_spikes * delta_angle
        # Here, the expected value with random firing to subtract
        # (this prevents the osi loss to drive the firing rates to go to zero.)
        expected_sum_angle = tf.reduce_mean(mean_spikes) * 45
        
        angle_loss = tf.reduce_mean(tf.abs(mean_angle)) - expected_sum_angle * self._subtraction_ratio
        
        return angle_loss * self._osi_cost

    def crowd_osi_loss(self, spikes, angle, trim=True, normalizer=None):
        # Calculate the angle deltas between current angle and tuning angle
        # angle = tf.cast(angle, self._dtype) 
        angle = tf.cast(angle[0][0], self._dtype) 
        delta_angle = tf.expand_dims(angle, axis=0) -  self._tuning_angles
        # i want the delta_angle to be within 0-360
        # clipped_delta_angle = tf.math.floormod(delta_angle, 360)
        radians_delta_angle = delta_angle * (np.pi / 180)
        # Instead of complex numbers, use cosine and sine separately
        osi_cos_component = tf.math.cos(2.0 * radians_delta_angle)
        dsi_cos_component = tf.math.cos(radians_delta_angle)
        # osi_sin_component = tf.math.sin(2.0 * radians_delta_angle)
        # dsi_sin_component = tf.math.sin(radians_delta_angle)

        if self._core_mask is not None:
            spikes = tf.boolean_mask(spikes, self._core_mask, axis=2)
            
        spikes = self.spike_trimming(spikes, trim)
        # sum spikes in _z, and multiply with delta_angle.
        rates = tf.reduce_mean(tf.cast(spikes, dtype=self._dtype), axis=[0, 1]) 
        if normalizer is not None:
            if self._core_mask is not None:
                normalizer = tf.boolean_mask(normalizer, self._core_mask, axis=0)
            # Minimum threshold for each element of the normalizer
            min_normalizer_value = 0.0005
            # Use tf.maximum to ensure each element of normalizer does not fall below min_normalizer_value
            normalizer = tf.maximum(normalizer, min_normalizer_value)
            rates = rates / normalizer

        # For weighted responses, we separately consider the contributions from cosine and sine
        weighted_osi_cos_responses = rates * osi_cos_component
        weighted_dsi_cos_responses = rates * dsi_cos_component
        # weighted_osi_sin_responses = rates * osi_sin_component
        # weighted_dsi_sin_responses = rates * dsi_sin_component

        total_osi_loss = tf.constant(0.0, dtype=self._dtype)
        total_dsi_loss = tf.constant(0.0, dtype=self._dtype)
        # penalization_terms = tf.constant(0.0, dtype=self._dtype)
        individual_osi_loss = {}
        # individual_dsi_loss = {}
        # individual_penalization_loss = {}

        for key, value in self._target_osi.items():
            if tf.size(value["ids"]) != 0:
                _rates_type = tf.gather(rates, value['ids'])
                _weighted_osi_cos_responses_type = tf.gather(weighted_osi_cos_responses, value['ids'])
                _weighted_dsi_cos_responses_type = tf.gather(weighted_dsi_cos_responses, value['ids'])
                # _weighted_osi_sin_responses_type = tf.gather(weighted_osi_sin_responses, value['ids'])
                # _weighted_dsi_sin_responses_type = tf.gather(weighted_dsi_sin_responses, value['ids'])

                # Define small epsilon values to avoid differentiability issues when 0 spikes are recorded within the population
                epsilon1 = 0.0005
                # Calculate the approximated OSI for the population
                approximated_osi_numerator = tf.reduce_mean(_weighted_osi_cos_responses_type)
                approximated_dsi_numerator = tf.reduce_mean(_weighted_dsi_cos_responses_type)
                # approximated_denominator = tf.maximum(tf.reduce_sum(tf.abs(_weighted_cos_responses_type)), epsilon2)
                approximated_denominator = tf.maximum(tf.reduce_mean(_rates_type), epsilon1)

                osi_approx_type = approximated_osi_numerator / approximated_denominator
                dsi_approx_type = approximated_dsi_numerator / approximated_denominator

                # osi_penalization = tf.math.square(tf.reduce_mean(_weighted_osi_sin_responses_type) / approximated_denominator)
                # dsi_penalization = tf.math.square(tf.reduce_mean(_weighted_dsi_sin_responses_type) / approximated_denominator)

                # Calculate the OSI loss
                osi_loss_type = tf.math.square(osi_approx_type - value['OSI'])
                dsi_loss_type = tf.math.square(dsi_approx_type - value['DSI'])

                individual_osi_loss[key] = osi_loss_type
                # individual_dsi_loss[key] = dsi_loss_type
                # individual_penalization_loss[key] = osi_penalization + dsi_penalization

                cell_count_type = tf.size(value["ids"])
                total_osi_loss += osi_loss_type * float(cell_count_type)
                total_dsi_loss += dsi_loss_type * float(cell_count_type)
                # penalization_terms += osi_penalization + dsi_penalization
                cell_count += cell_count_type
            else:
                individual_osi_loss[key] = 0.0
                # individual_dsi_loss[key] = 0.0
                # individual_penalization_loss[key] = 0.0
                pass
            
        # return (total_osi_loss + total_dsi_loss + penalization_terms) * self._osi_cost
        # return (total_osi_loss + total_dsi_loss) * self._osi_cost
        total_osi_loss = total_osi_loss / float(cell_count)
        total_dsi_loss = total_dsi_loss / float(cell_count)
        return (total_osi_loss + total_dsi_loss) * self._osi_cost, individual_osi_loss
    

    def __call__(self, spikes, angle, trim, normalizer=None):
        if self._method == "crowd_osi":
            return self.crowd_osi_loss(spikes, angle, trim, normalizer=normalizer)
        elif self._method == "crowd_spikes":
            return self.crowd_spikes_loss(spikes, angle, trim)
        elif self._method == "neuropixels_fr":
            return self.neuropixels_fr_loss(spikes, angle, trim)
        
