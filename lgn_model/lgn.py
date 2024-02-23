import os
# import tqdm
import socket
import pickle as pkl
import numpy as np
import pandas as pd
import h5py
import tensorflow as tf
import matplotlib.pyplot as plt

# import pdb

from bmtk.simulator.filternet.lgnmodel.fitfuns import makeBasis_StimKernel
from bmtk.simulator.filternet.lgnmodel.spatialfilter import GaussianSpatialFilter
from bmtk.simulator.filternet.lgnmodel.temporalfilter import TemporalFilterCosineBump
from bmtk.simulator.filternet.lgnmodel.util_fns import get_tcross_from_temporal_kernel
try:  # this is old version of bmtk
    from bmtk.simulator.filternet.lgnmodel.util_fns import get_data_metrics_for_each_subclass
except ImportError:  # new bmtk imports
    from bmtk.simulator.filternet.lgnmodel.cellmetrics import get_data_metrics_for_each_subclass


def create_temporal_filter(inp_dict):
    opt_wts = inp_dict['opt_wts']
    opt_kpeaks = inp_dict['opt_kpeaks']
    opt_delays = inp_dict['opt_delays']
    temporal_filter = TemporalFilterCosineBump(opt_wts, opt_kpeaks, opt_delays)

    return temporal_filter

def create_one_unit_of_two_subunit_filter(prs, ttp_exp):
    filt = create_temporal_filter(prs)
    tcross_ind = get_tcross_from_temporal_kernel(filt.get_kernel(threshold=-1.0).kernel)
    filt_sum = filt.get_kernel(threshold=-1.0).kernel[:tcross_ind].sum()

    # Calculate delay offset needed to match response latency with data and rebuild temporal filter
    del_offset = ttp_exp - tcross_ind
    if del_offset >= 0:
        delays = prs['opt_delays']
        delays[0] = delays[0] + del_offset
        delays[1] = delays[1] + del_offset
        prs['opt_delays'] = delays
        filt_new = create_temporal_filter(prs)
    else:
        print('del_offset < 0')

    return filt_new, filt_sum

def temporal_filter(all_spatial_responses, temporal_kernels):
    tr_spatial_responses = tf.pad(
        all_spatial_responses[None, :, None, :],
        ((0, 0), (temporal_kernels.shape[-1] - 1, 0), (0, 0), (0, 0)))

    tr_temporal_kernels = tf.transpose(temporal_kernels)[:, None, :, None]
    filtered_output = tf.nn.depthwise_conv2d(
        tr_spatial_responses, tr_temporal_kernels, strides=[1, 1, 1, 1], padding='VALID')[0, :, 0]
    return filtered_output


def transfer_function(arg__a):
    _h = tf.cast(arg__a >= 0, tf.float32)
    return _h * arg__a


def select_spatial(x, y, convolved_movie):
    i1 = tf.cast(tf.stack([tf.floor(y), tf.floor(x)], axis=-1), dtype=tf.int32)
    i2 = tf.cast(tf.stack([tf.math.ceil(y), tf.floor(x)], axis=-1), dtype=tf.int32)
    i3 = tf.cast(tf.stack([tf.floor(y), tf.math.ceil(x)], axis=-1), dtype=tf.int32)
    i4 = tf.cast(tf.stack([tf.math.ceil(y), tf.math.ceil(x)], axis=-1), dtype=tf.int32)

    transposed_convolved_movie = tf.transpose(convolved_movie, perm=[1, 2, 0])

    sr1 = tf.gather_nd(transposed_convolved_movie, i1)
    sr2 = tf.gather_nd(transposed_convolved_movie, i2)
    sr3 = tf.gather_nd(transposed_convolved_movie, i3)
    sr4 = tf.gather_nd(transposed_convolved_movie, i4)

    ss = tf.stack([sr1, sr2, sr3, sr4], axis=0)

    y_factor = y - tf.floor(y)
    x_factor = x - tf.floor(x)

    weights = tf.stack([
    (1 - x_factor) * (1 - y_factor),
    (1 - x_factor) * y_factor,
    x_factor * (1 - y_factor),
    x_factor * y_factor
    ], axis=0)

    spatial_responses = tf.reduce_sum(ss * tf.expand_dims(weights, axis=-1), axis=0)
    spatial_responses = tf.transpose(spatial_responses)

    return spatial_responses


def create_lgn_units_info(csv_path='/home/jgalvan/Desktop/Neurocoding/V1_GLIF_model/GLIF_network/network/lgn_node_types.csv', 
                          h5_path='/home/jgalvan/Desktop/Neurocoding/V1_GLIF_model/GLIF_network/network//lgn_nodes.h5',
                          filename='/home/jgalvan/Desktop/Neurocoding/V1_GLIF_model/lgn_model/data/lgn_full_col_cells.csv'
                          ):
    filename = os.path.join('data', filename)
    # Load both the h5 file and the csv file
    csv_file = pd.read_csv(csv_path, sep=' ')
    features = ['id', 'model_id', 'x', 'y', 'ei', 'location', 'spatial_size', 'kpeaks_dom_0', 'kpeaks_dom_1', 'weight_dom_0', 'weight_dom_1', 'delay_dom_0', 'delay_dom_1', 'kpeaks_non_dom_0', 'kpeaks_non_dom_1', 'weight_non_dom_0', 'weight_non_dom_1', 'delay_non_dom_0', 'delay_non_dom_1', 'tuning_angle', 'sf_sep']
    df = pd.DataFrame(columns=features)

    with h5py.File(h5_path, 'r') as h5_file:
        node_id = h5_file['nodes']['lgn']['node_id'][:]
        node_type_id = h5_file['nodes']['lgn']['node_type_id'][:]
        for feature in h5_file['nodes']['lgn']['0'].keys():
            df[feature] = np.array(h5_file['nodes']['lgn']['0'][feature][:], dtype=np.float32)

    node_info = {}
    for index, row in csv_file.iterrows():
        node_info[row['node_type_id']] = {'model_id': row['pop_name'], 'location': row['location'], 'ei': row['ei']}

    df['id'] = node_id
    df['model_id'] = [node_info[node_type_id[i]]['model_id'] for i in range(len(node_type_id))]
    df['location'] = [node_info[node_type_id[i]]['location'] for i in range(len(node_type_id))]
    df['ei'] = [node_info[node_type_id[i]]['ei'] for i in range(len(node_type_id))]

    df.to_csv(filename, index=False, sep=' ', na_rep='NaN')
    return df


class LGN(object):
    # @profile
    def __init__(self, row_size=80, col_size=120, lgn_data_path=None, n_input=None):
        filename = f'lgn_full_col_cells_{col_size}x{row_size}.csv'
        root_path = os.path.split(__file__)[0]
        root_path = os.path.join(root_path, 'data')
        lgn_data_path = os.path.join(root_path, filename)
        if os.path.exists(lgn_data_path):
            d = pd.read_csv(lgn_data_path, delimiter=' ')
        else:
            print('Creating LGN units info')
            # making the LGN file generation work in more generic environments
            model_path = os.path.split(__file__)[0]
            # go up one folder and add "GLIF_network" to the path
            model_path = os.path.split(model_path)[0]
            model_path = os.path.join(model_path, 'GLIF_network')
            model_path = os.path.join(model_path, 'network')
            lgn_node_path = os.path.join(model_path, 'lgn_nodes.h5')
            lge_node_type_path = os.path.join(model_path, 'lgn_node_types.csv')
            d = create_lgn_units_info(filename=lgn_data_path, csv_path=lge_node_type_path, h5_path=lgn_node_path)
                
        spatial_sizes = d['spatial_size'].to_numpy(dtype=np.float32)
        self.spatial_sizes = spatial_sizes
        model_id = d['model_id'].to_numpy()
        self.model_id = model_id
        amplitude = np.array([1. if a.count('ON') > 0 else -1. for a in model_id])
        non_dom_amplitude = np.zeros_like(amplitude)
        is_composite = np.array([a.count('ON') > 0 and a.count(
            'OFF') > 0 for a in model_id]).astype(np.float32)
        self.is_composite = is_composite
        x = d['x'].to_numpy(dtype=np.float32)
        y = d['y'].to_numpy(dtype=np.float32)

        non_dominant_x = np.zeros_like(x)
        non_dominant_y = np.zeros_like(y)
        tuning_angle = d['tuning_angle'].to_numpy(dtype=np.float32)
        subfield_separation = d['sf_sep'].to_numpy(dtype=np.float32)  # for composite cells

        s_path = os.path.join(root_path, f'spontaneous_firing_rates_{col_size}x{row_size}.pkl')
        if not os.path.exists(s_path):
            cell_type = [a[:a.find('_')] for a in model_id]
            tf_str = [a[a.find('_') + 1:] for a in model_id]
            spontaneous_firing_rates = []
            print('Computing spontaneous firing rates')
            # for a, b in tqdm.tqdm(zip(cell_type, tf_str), total=len(model_id)):
            for a, b in zip(cell_type, tf_str):
                if a.count('ON') > 0 and a.count('OFF') > 0:
                    spontaneous_firing_rates.append(-1)
                else:
                    spontaneous_firing_rate = get_data_metrics_for_each_subclass(a)[b]['spont_exp']
                    spontaneous_firing_rates.append(spontaneous_firing_rate[0])
            spontaneous_firing_rates = np.array(spontaneous_firing_rates)
            with open(s_path, 'wb') as f:
                pkl.dump(spontaneous_firing_rates, f)
                print('Caching spontaneous firing rates')
        else:
            with open(s_path, 'rb') as f:
                spontaneous_firing_rates = pkl.load(f)

        temporal_peaks_dom = np.stack((d['kpeaks_dom_0'].to_numpy(dtype=np.float32), d['kpeaks_dom_1'].to_numpy(dtype=np.float32)), -1)
        temporal_weights = np.stack((d['weight_dom_0'].to_numpy(dtype=np.float32), d['weight_dom_1'].to_numpy(dtype=np.float32)), -1)
        temporal_delays = np.stack((d['delay_dom_0'].to_numpy(dtype=np.float32), d['delay_dom_1'].to_numpy(dtype=np.float32)), -1)

        temporal_peaks_non_dom = np.stack((d['kpeaks_non_dom_0'].to_numpy(dtype=np.float32), d['kpeaks_non_dom_1'].to_numpy(dtype=np.float32)), -1)
        temporal_weights_non_dom = np.stack((d['weight_non_dom_0'].to_numpy(dtype=np.float32), d['weight_non_dom_1'].to_numpy(dtype=np.float32)), -1)
        temporal_delays_non_dom = np.stack((d['delay_non_dom_0'].to_numpy(dtype=np.float32), d['delay_non_dom_1'].to_numpy(dtype=np.float32)), -1)

        # values from bmtk
        t_path = os.path.join(root_path, f'temporal_kernels_{col_size}x{row_size}.pkl')
        kernel_length = 700
        if not os.path.exists(t_path):
            nkt = 600
            dom_temporal_kernels = []
            non_dom_temporal_kernels = []
            print('Computing temporal kernels')
            # for i in tqdm.tqdm(range(x.shape[0])):
            for i in range(x.shape[0]):
                dom_temporal_kernel = np.zeros((kernel_length,), np.float32)
                non_dom_temporal_kernel = np.zeros((kernel_length,), np.float32)
                if model_id[i].count('ON') > 0 and model_id[i].count('OFF') > 0:
                    non_dom_params = dict(
                        opt_wts=temporal_weights_non_dom[i],
                        opt_kpeaks=temporal_peaks_non_dom[i],
                        opt_delays=temporal_delays_non_dom[i]
                    )
                    dom_params = dict(
                        opt_wts=temporal_weights[i],
                        opt_kpeaks=temporal_peaks_dom[i],
                        opt_delays=temporal_delays[i]
                    )
                    amp_on = 1.0  # set the non-dominant subunit amplitude to unity

                    if model_id[i].count('sONsOFF_001') > 0:
                        non_dom_filter, non_dom_sum = create_one_unit_of_two_subunit_filter(non_dom_params, 121.)
                        dom_filter, dom_sum = create_one_unit_of_two_subunit_filter(dom_params, 115.)

                        spont = 4.0
                        max_roff = 35.0
                        max_ron = 21.0
                        amp_off = -(max_roff / max_ron) * (non_dom_sum / dom_sum) * amp_on - (
                            spont * (max_roff - max_ron)) / (max_ron * dom_sum)
                    elif model_id[i].count('sONtOFF_001') > 0:
                        non_dom_filter, non_dom_sum = create_one_unit_of_two_subunit_filter(non_dom_params, 93.5)
                        dom_filter, dom_sum = create_one_unit_of_two_subunit_filter(dom_params, 64.8)

                        spont = 5.5
                        max_roff = 46.0
                        max_ron = 31.0
                        amp_off = -0.7 * (max_roff / max_ron) * (non_dom_sum / dom_sum) * amp_on - (
                            spont * (max_roff - max_ron)) / (max_ron * dom_sum)
                    else:
                        raise ValueError('Unknown cell type')
                    non_dom_amplitude[i] = amp_on
                    amplitude[i] = amp_off
                    spontaneous_firing_rates[i] = spont / 2

                    hor_offset = np.cos(tuning_angle[i] * np.pi / 180.) * subfield_separation[i] + x[i]
                    vert_offset = np.sin(tuning_angle[i] * np.pi / 180.) * subfield_separation[i] + y[i]
                    non_dominant_x[i] = hor_offset
                    non_dominant_y[i] = vert_offset
                    dom_temporal_kernel[-len(dom_filter.kernel_data):] = dom_filter.kernel_data[::-1]
                    non_dom_temporal_kernel[-len(non_dom_filter.kernel_data):] = non_dom_filter.kernel_data[::-1]
                else:
                    dd = dict(neye=0, ncos=2, kpeaks=temporal_peaks_dom[i], b=.3, delays=[temporal_delays[i].astype(int)])
                    kernel_data = np.dot(makeBasis_StimKernel(dd, nkt), temporal_weights[i])
                    dom_temporal_kernel[-len(kernel_data):] = kernel_data

                dom_temporal_kernels.append(dom_temporal_kernel)
                non_dom_temporal_kernels.append(non_dom_temporal_kernel)

            dom_temporal_kernels = np.array(dom_temporal_kernels).astype(np.float32)
            non_dom_temporal_kernels = np.array(non_dom_temporal_kernels).astype(np.float32)
            to_save = dict(
                dom_temporal_kernels=dom_temporal_kernels,
                non_dom_temporal_kernels=non_dom_temporal_kernels,
                non_dominant_x=non_dominant_x,
                non_dominant_y=non_dominant_y,
                amplitude=amplitude.astype(np.float32),
                non_dom_amplitude=non_dom_amplitude.astype(np.float32),
                spontaneous_firing_rates=spontaneous_firing_rates
            )
            with open(t_path, 'wb') as f:
                pkl.dump(to_save, f)
                print('Caching temporal kernels...')
        else:
            with open(t_path, 'rb') as f:
                loaded = pkl.load(f)
            dom_temporal_kernels = loaded['dom_temporal_kernels']
            non_dom_temporal_kernels = loaded['non_dom_temporal_kernels']
            non_dominant_x = loaded['non_dominant_x']
            non_dominant_y = loaded['non_dominant_y']
            amplitude = loaded['amplitude']
            non_dom_amplitude = loaded['non_dom_amplitude']
            spontaneous_firing_rates = loaded['spontaneous_firing_rates']
        truncation = np.min(np.sum(np.cumsum(np.abs(dom_temporal_kernels), axis=1) <= 1e-6, 1))
        non_dom_truncation = np.min(np.sum(np.cumsum(np.abs(non_dom_temporal_kernels), axis=1) <= 1e-6, 1))
        truncation = np.min([truncation, non_dom_truncation])
        # print(f'Could truncate {truncation} steps from filter')

        x = x * (col_size-1) / col_size  # 239 / 240
        y = y * (row_size-1) / row_size  # 119 / 120
        x[np.floor(x) < 0] = 0.
        y[np.floor(y) < 0] = 0.
        x[np.ceil(x) >= float(col_size-1)] = float(col_size-1)
        y[np.ceil(y) >= float(row_size-1)] = float(row_size-1)

        non_dominant_x = non_dominant_x * (col_size-1) / col_size  # 239 / 240
        non_dominant_y = non_dominant_y * (row_size-1) / row_size
        non_dominant_x[np.floor(non_dominant_x) < 0] = 0.
        non_dominant_y[np.floor(non_dominant_y) < 0] = 0.
        non_dominant_x[np.ceil(non_dominant_x) >= float(col_size-1)] = float(col_size-1)
        non_dominant_y[np.ceil(non_dominant_y) >= float(row_size-1)] = float(row_size-1)

        # prepare the spatial kernels in advance and store in TF format
        d_spatial = 1.
        spatial_range = np.arange(0, 15, d_spatial)
        x_range = np.arange(-50, 51)  # define the spatial kernel max size
        y_range = np.arange(-50, 51)

        # kernels = []
        gaussian_filters = []
        for i in range(len(spatial_range) - 1):
            sigma = np.round(np.mean(spatial_range[i:i+2])) / 3
            original_filter = GaussianSpatialFilter(translate=(0., 0.), sigma=(sigma, sigma), origin=(0., 0.))
            kernel = original_filter.get_kernel(x_range, y_range, amplitude=1.).full()
            # kernels.append(kernel)
            nonzero_inds = np.where(np.abs(kernel) > 1e-9)
            rm, rM = nonzero_inds[0].min(), nonzero_inds[0].max()
            cm, cM = nonzero_inds[1].min(), nonzero_inds[1].max()
            kernel = kernel[rm:rM + 1, cm:cM + 1]
            gaussian_filter = kernel[..., None, None]
            # gaussian_filter0 = tf.convert_to_tensor(gaussian_filter, dtype=tf.float32)
            gaussian_filter = tf.constant(gaussian_filter, dtype=tf.float32) # this is faster by assuming that gaussian_filter is unmutable
            gaussian_filters.append(gaussian_filter)

        if n_input is None:
            self.x = x
            self.y = y
            self.non_dominant_x = non_dominant_x
            self.non_dominant_y = non_dominant_y
            self.amplitude = amplitude
            self.non_dom_amplitude = non_dom_amplitude
            self.spontaneous_firing_rates = spontaneous_firing_rates
            self.dom_temporal_kernels = dom_temporal_kernels
            self.non_dom_temporal_kernels = non_dom_temporal_kernels
            # self.kernels = kernels
            self.gaussian_filters = gaussian_filters
        else:
            self.x = x[:n_input]
            self.y = y[:n_input]
            self.non_dominant_x = non_dominant_x[:n_input]
            self.non_dominant_y = non_dominant_y[:n_input]
            self.amplitude = amplitude[:n_input]
            self.non_dom_amplitude = non_dom_amplitude[:n_input]
            self.spontaneous_firing_rates = spontaneous_firing_rates[:n_input]
            self.dom_temporal_kernels = dom_temporal_kernels[:n_input, :]
            self.non_dom_temporal_kernels = non_dom_temporal_kernels[:n_input, :]
            # self.kernels = kernels
            self.gaussian_filters = gaussian_filters
            
            # other properties that are defined above needs to be also truncated
            self.spatial_sizes = self.spatial_sizes[:n_input]
            self.model_id = self.model_id[:n_input]
            self.is_composite = self.is_composite[:n_input]

    @tf.function
    def spatial_response(self, movie, bmtk_compat=True):
        d_spatial = 1
        spatial_range = tf.range(0, 15, d_spatial, dtype=tf.float32)
        # spatial_range = tf.range(0, 15, d_spatial, dtype=tf.float32)

        # Preprocess data outside the loop if they don't change
        x = tf.constant(self.x, dtype=tf.float32)
        y = tf.constant(self.y, dtype=tf.float32)
        non_dominant_x = tf.constant(self.non_dominant_x, dtype=tf.float32)
        non_dominant_y = tf.constant(self.non_dominant_y, dtype=tf.float32)
        spatial_sizes = tf.constant(self.spatial_sizes, dtype=tf.float32)
        # movie = tf.constant(movie, dtype=tf.float32)
        if not isinstance(movie, tf.Tensor):
            movie = tf.constant(movie, dtype=tf.float32)
            print(f'Movie type: {type(movie)}')
       
        all_spatial_responses = []
        all_non_dom_spatial_responses = []
        neuron_ids = []

        for i in range(len(spatial_range)-1):
            sel = tf.math.logical_and(spatial_sizes < spatial_range[i + 1], spatial_sizes >= spatial_range[i])
            num_selected = tf.reduce_sum(tf.cast(sel, dtype=tf.int32))
            # if num_selected == 0:
            #     tf.print('No neurons selected')
            #     continue
            
            # Construct spatial filter
            gaussian_filter = self.gaussian_filters[i]  # Assuming self.gaussian_filters is a list of precomputed filters

            # Apply it
            convolved_movie = tf.nn.conv2d(movie, gaussian_filter, strides=[1, 1, 1, 1], padding='SAME')
            
            # Making BMTK compatible by normalizing the edge values
            if bmtk_compat:
                ones = tf.ones_like(movie)
                gaussian_fraction = tf.nn.conv2d(ones, gaussian_filter, strides=[1, 1, 1, 1], padding='SAME')
                convolved_movie = convolved_movie / gaussian_fraction
            
            convolved_movie = convolved_movie[..., 0]  # Assuming you only need one channel

            spatial_responses = select_spatial(tf.boolean_mask(x, sel), tf.boolean_mask(y, sel), convolved_movie)
            non_dom_spatial_responses = select_spatial(tf.boolean_mask(non_dominant_x, sel), tf.boolean_mask(non_dominant_y, sel), convolved_movie)

            all_spatial_responses.append(spatial_responses)
            all_non_dom_spatial_responses.append(non_dom_spatial_responses)
            selected_indices = tf.where(sel)[:, 0]
            neuron_ids.append(selected_indices)

        neuron_ids = tf.concat(neuron_ids, axis=0)
        neuron_ids = tf.cast(neuron_ids, dtype=tf.int32)
        all_spatial_responses = tf.concat(all_spatial_responses, axis=1)
        all_non_dom_spatial_responses = tf.concat(all_non_dom_spatial_responses, axis=1)

        sorted_neuron_ids_indices = tf.argsort(neuron_ids)
        all_spatial_responses = tf.gather(all_spatial_responses, sorted_neuron_ids_indices, axis=1)
        all_non_dom_spatial_responses = tf.gather(all_non_dom_spatial_responses, sorted_neuron_ids_indices, axis=1)

        return all_spatial_responses, all_non_dom_spatial_responses

    @tf.function
    def firing_rates_from_spatial(self, all_spatial_responses, all_non_dom_spatial_responses):
        dom_filtered_output = temporal_filter(all_spatial_responses, self.dom_temporal_kernels)
        non_dom_filtered_output = temporal_filter(all_non_dom_spatial_responses, self.non_dom_temporal_kernels)
        # combined_filtered_output = dom_filtered_output * amplitude + non_dom_filtered_output * non_dom_amplitude
        firing_rates = transfer_function(dom_filtered_output * self.amplitude + self.spontaneous_firing_rates)
        multi_firing_rates = firing_rates + transfer_function(
            non_dom_filtered_output * self.non_dom_amplitude + self.spontaneous_firing_rates)
        firing_rates = firing_rates * \
            (1 - self.is_composite) + multi_firing_rates * self.is_composite

        return firing_rates


def main():
    from check_filter import load_example_movie
    movie = load_example_movie(duration=2000, onset=1000, offset=1100)

    lgn = LGN()
    spatial = lgn.spatial_response(movie)
    firing_rates = lgn.firing_rates_from_spatial(*spatial)

    # fig, ax = plt.subplots(figsize=(12, 12))
    fig = plt.figure(figsize=(12, 12))
    gs = fig.add_gridspec(5, 1)
    ax = fig.add_subplot(gs[:4])
    if False:
        import h5py
        f = h5py.File(
            '/data/allen/v1_model/go_nogo_image_outputs/stim_0.h5_f_tot.h5', mode='r')
        d = np.array(f['firing_rates_Hz'])
        data = firing_rates[:, :4000].numpy().T - d[:4000]
        abs_max = np.abs(data).max()
        p = ax.pcolormesh(data, cmap='seismic', vmin=-abs_max, vmax=abs_max)
    else:
        data = firing_rates.numpy().T
        p = ax.pcolormesh(data, cmap='cividis')
    plt.colorbar(p, ax=ax)
    ax = fig.add_subplot(gs[4])
    ax.plot(data.mean(0))
    fig.savefig('temp.png', dpi=300)
    plt.show()


if __name__ == '__main__':
    main()
