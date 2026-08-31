import os
# import tqdm
# import socket
import pickle as pkl
import numpy as np
import pandas as pd
import h5py
import tensorflow as tf
import matplotlib.pyplot as plt
# import pdb

try:
    from numba import njit
    HAS_NUMBA = True
except Exception:
    HAS_NUMBA = False

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
        ((0, 0), (temporal_kernels.shape[0] - 1, 0), (0, 0), (0, 0)))

    filtered_output = tf.nn.depthwise_conv2d(
        tr_spatial_responses, temporal_kernels[:, None, :, None], strides=[1, 1, 1, 1], padding='VALID')[0, :, 0]
    return filtered_output

def transfer_function(arg__a, dtype=tf.float32):
    _h = tf.cast(arg__a >= 0, dtype)
    return _h * arg__a

def _bilinear_metadata(x, y, width):
    """Precompute flattened indices and weights for bilinear sampling."""
    x0 = np.floor(x).astype(np.int32)
    x1 = np.ceil(x).astype(np.int32)
    y0 = np.floor(y).astype(np.int32)
    y1 = np.ceil(y).astype(np.int32)
    x_fraction = x - x0
    y_fraction = y - y0
    indices = np.stack(
        (y0 * width + x0, y1 * width + x0, y0 * width + x1, y1 * width + x1),
        axis=1,
    )
    weights = np.stack(
        (
            (1 - x_fraction) * (1 - y_fraction),
            (1 - x_fraction) * y_fraction,
            x_fraction * (1 - y_fraction),
            x_fraction * y_fraction,
        ),
        axis=1,
    ).astype(np.float32)
    return indices, weights


def _sample_spatial(flattened_movie, indices, weights):
    values = tf.gather(flattened_movie, indices, axis=1)
    return tf.reduce_sum(values * weights[None, ...], axis=-1)


def create_lgn_units_info(csv_path='/home/jgalvan/Desktop/Neurocoding/V1_GLIF_model/GLIF_network/network/lgn_node_types.csv', 
                          h5_path='/home/jgalvan/Desktop/Neurocoding/V1_GLIF_model/GLIF_network/network//lgn_nodes.h5',
                          filename='/home/jgalvan/Desktop/Neurocoding/V1_GLIF_model/lgn_model/data/lgn_full_col_cells.csv'
                          ):
    # filename = os.path.join('data', filename)
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
   
if HAS_NUMBA:
    @njit(cache=True)
    def _assign_spatial_bin_ids_numba(spatial_sizes, spatial_range):
        """Assign each spatial size to a [low, high) range bin index, or -1 if none."""
        n = spatial_sizes.shape[0]
        n_bins = spatial_range.shape[0] - 1
        out = np.full(n, -1, dtype=np.int32)
        for idx in range(n):
            v = spatial_sizes[idx]
            for b in range(n_bins):
                if (v >= spatial_range[b]) and (v < spatial_range[b + 1]):
                    out[idx] = b
                    break
        return out
else:
    def _assign_spatial_bin_ids_numba(spatial_sizes, spatial_range):
        n_bins = len(spatial_range) - 1
        out = np.full(spatial_sizes.shape[0], -1, dtype=np.int32)
        for b in range(n_bins):
            sel = np.logical_and(spatial_sizes >= spatial_range[b], spatial_sizes < spatial_range[b + 1])
            out[sel] = b
        return out

class LGN(object):
    """
    Drop-in LGN class with __init__ preprocessing moved to Numba/Numpy.
    """

    def __init__(self, row_size=80, col_size=120, data_dir='GLIF_network', n_input=None, dtype=tf.float32):
        self.row_size = row_size
        self.col_size = col_size
        filename = f'lgn_full_col_cells_{col_size}x{row_size}.csv'
        lgn_code_dir = os.path.split(__file__)[0]
        # go up one folder and add "GLIF_network" to the path
        root_dir = os.path.split(lgn_code_dir)[0]
        if os.path.isabs(data_dir):
            data_dir_abs = data_dir
        else:
            data_dir_abs = os.path.join(root_dir, data_dir)

        lgn_data_dir = os.path.join(data_dir_abs, 'tf_data')
        lgn_data_path = os.path.join(lgn_data_dir, filename)
        # root_path = os.path.split(__file__)[0]
        # root_path = os.path.join(root_path, 'data')
        # lgn_data_path = os.path.join(root_path, filename)

        if os.path.exists(lgn_data_path):
            d = pd.read_csv(lgn_data_path, delimiter=' ')
        else:
            print('Creating LGN units info')
            # making the LGN file generation work in more generic environments
            # model_path = os.path.split(__file__)[0]
            # go up one folder and add "GLIF_network" to the path
            # model_path = os.path.split(model_path)[0]
            # model_path = os.path.join(model_path, 'GLIF_network')
            os.makedirs(lgn_data_dir, exist_ok=True)
            network_dir = os.path.join(data_dir_abs, 'network')
            lgn_node_path = os.path.join(network_dir, 'lgn_nodes.h5')
            lgn_node_type_path = os.path.join(network_dir, 'lgn_node_types.csv')
            d = create_lgn_units_info(
                filename=lgn_data_path,
                csv_path=lgn_node_type_path,
                h5_path=lgn_node_path,
            )

        # CHANGE 1: Apply n_input selection immediately after loading data
        if n_input is not None and n_input < len(d):
            # Select first n_input neurons
            d = d.iloc[:n_input].copy()
            print(f"Selected first {n_input} LGN units out of {len(d)} available")

        # CHANGE 2: Update cache file names to include n_input
        n_units = len(d)
        s_path = os.path.join(lgn_data_dir, f'spontaneous_firing_rates_{col_size}x{row_size}_n{n_units}.pkl')
        t_path = os.path.join(lgn_data_dir, f'temporal_kernels_{col_size}x{row_size}_n{n_units}.pkl')
        spatial_path = os.path.join(lgn_data_dir, f'spatial_kernels_{col_size}x{row_size}_n{n_units}.pkl')

        # Load basic information about the LGN units
        self.dtype = dtype
        model_id = d['model_id'].to_numpy()
        amplitude = np.array([1.0 if m.count('ON') > 0 else -1.0 for m in model_id], dtype=np.float32)
        non_dom_amplitude = np.zeros_like(amplitude)
        is_composite = np.array([('ON' in m and 'OFF' in m) for m in model_id], dtype=np.float32)

        # Load the spontaneous firing rates
        if not os.path.exists(s_path):
            cell_type = [a[:a.find('_')] for a in model_id]
            tf_str = [a[a.find('_') + 1:] for a in model_id]
            spontaneous_firing_rates = []
            print('Computing spontaneous firing rates')
            for a, b in zip(cell_type, tf_str):
                if a.count('ON') > 0 and a.count('OFF') > 0:
                    spontaneous_firing_rates.append(-1.0)
                else:
                    spontaneous_firing_rate = get_data_metrics_for_each_subclass(a)[b]['spont_exp']
                    spontaneous_firing_rates.append(spontaneous_firing_rate[0])
            spontaneous_firing_rates = np.array(spontaneous_firing_rates, dtype=np.float32)
            with open(s_path, 'wb') as f:
                pkl.dump(spontaneous_firing_rates, f)
                print('Caching spontaneous firing rates')
        else:
            with open(s_path, 'rb') as f:
                spontaneous_firing_rates = np.asarray(pkl.load(f), dtype=np.float32)

        # Load the temporal kernels
        if not os.path.exists(t_path):
            nkt = 600
            kernel_length = 700
            dom_temporal_kernels = []
            non_dom_temporal_kernels = []
            print('Computing temporal kernels')
            # Load spatial features of the elliptical subfields
            tuning_angle = d['tuning_angle'].to_numpy(dtype=np.float32)
            subfield_separation = d['sf_sep'].to_numpy(dtype=np.float32)
            x = d['x'].to_numpy(dtype=np.float32)
            y = d['y'].to_numpy(dtype=np.float32)
            non_dominant_x = np.zeros_like(x)
            non_dominant_y = np.zeros_like(y)

            # Load the temporal kernels features
            temporal_peaks_dom = np.stack(
                (d['kpeaks_dom_0'].to_numpy(dtype=np.float32), d['kpeaks_dom_1'].to_numpy(dtype=np.float32)),
                axis=-1,
            )
            temporal_weights = np.stack(
                (d['weight_dom_0'].to_numpy(dtype=np.float32), d['weight_dom_1'].to_numpy(dtype=np.float32)),
                axis=-1,
            )
            temporal_delays = np.stack(
                (d['delay_dom_0'].to_numpy(dtype=np.float32), d['delay_dom_1'].to_numpy(dtype=np.float32)),
                axis=-1,
            )

            temporal_peaks_non_dom = np.stack(
                (d['kpeaks_non_dom_0'].to_numpy(dtype=np.float32), d['kpeaks_non_dom_1'].to_numpy(dtype=np.float32)),
                axis=-1,
            )
            temporal_weights_non_dom = np.stack(
                (d['weight_non_dom_0'].to_numpy(dtype=np.float32), d['weight_non_dom_1'].to_numpy(dtype=np.float32)),
                axis=-1,
            )
            temporal_delays_non_dom = np.stack(
                (d['delay_non_dom_0'].to_numpy(dtype=np.float32), d['delay_non_dom_1'].to_numpy(dtype=np.float32)),
                axis=-1,
            )

            for i in range(x.shape[0]):
                dom_temporal_kernel = np.zeros((kernel_length,), np.float32)
                non_dom_temporal_kernel = np.zeros((kernel_length,), np.float32)
                if model_id[i].count('ON') > 0 and model_id[i].count('OFF') > 0:
                    non_dom_params = dict(
                        opt_wts=temporal_weights_non_dom[i],
                        opt_kpeaks=temporal_peaks_non_dom[i],
                        opt_delays=temporal_delays_non_dom[i],
                    )
                    dom_params = dict(
                        opt_wts=temporal_weights[i],
                        opt_kpeaks=temporal_peaks_dom[i],
                        opt_delays=temporal_delays[i],
                    )
                    amp_on = 1.0 # set the non-dominant subunit amplitude to unity

                    if model_id[i].count('sONsOFF_001') > 0:
                        non_dom_filter, non_dom_sum = create_one_unit_of_two_subunit_filter(non_dom_params, 121.0)
                        dom_filter, dom_sum = create_one_unit_of_two_subunit_filter(dom_params, 115.0)
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
                    spontaneous_firing_rates[i] = spont / 2.0

                    hor_offset = np.cos(tuning_angle[i] * np.pi / 180.0) * subfield_separation[i] + x[i]
                    vert_offset = np.sin(tuning_angle[i] * np.pi / 180.0) * subfield_separation[i] + y[i]
                    non_dominant_x[i] = hor_offset
                    non_dominant_y[i] = vert_offset
                    dom_temporal_kernel[-len(dom_filter.kernel_data):] = dom_filter.kernel_data[::-1]
                    non_dom_temporal_kernel[-len(non_dom_filter.kernel_data):] = non_dom_filter.kernel_data[::-1]
                else:
                    dd = dict(
                        neye=0,
                        ncos=2,
                        kpeaks=temporal_peaks_dom[i],
                        b=0.3,
                        delays=[temporal_delays[i].astype(int)],
                    )
                    kernel_data = np.dot(makeBasis_StimKernel(dd, nkt), temporal_weights[i])
                    dom_temporal_kernel[-len(kernel_data):] = kernel_data

                dom_temporal_kernels.append(dom_temporal_kernel)
                non_dom_temporal_kernels.append(non_dom_temporal_kernel)

            dom_temporal_kernels = np.asarray(dom_temporal_kernels, dtype=np.float32)
            non_dom_temporal_kernels = np.asarray(non_dom_temporal_kernels, dtype=np.float32)

            # Apply truncation
            dom_cumsum = np.cumsum(np.abs(dom_temporal_kernels), axis=1)
            non_dom_cumsum = np.cumsum(np.abs(non_dom_temporal_kernels), axis=1)
            # Find the minimum number of steps where cumulative sum is below threshold
            threshold = 1e-6
            # For dominant kernels: compute truncation points for every filters
            dom_truncation_points = np.sum(dom_cumsum <= threshold, axis=1)
            # For non-dominant kernels: only include filters that are non-zero in the truncation calculation
            non_dom_truncation_points = np.where(
                np.sum(np.abs(non_dom_temporal_kernels), axis=1) > 0,
                np.sum(non_dom_cumsum <= threshold, axis=1),
                np.inf,
            )
            # Find the minimum truncation point while ignoring zero filters (set to np.inf to avoid affecting the min calculation)
            dom_truncation = int(np.min(dom_truncation_points))
            # non_dom_truncation = int(np.min(non_dom_truncation_points))
            # Handle the case where all non-dominant kernels are zero (no composite cells)
            if np.all(np.isinf(non_dom_truncation_points)):
                # No composite cells, use only dominant truncation
                non_dom_truncation = dom_truncation
            else:
                non_dom_truncation = int(np.min(non_dom_truncation_points))

            # Apply the truncation to both dominant and non-dominant temporal kernels
            truncation = int(np.min([dom_truncation, non_dom_truncation]))
            # Truncate and transpose the kernels from the truncation point onwards
            dom_temporal_kernels = dom_temporal_kernels[:, dom_truncation:].T
            non_dom_temporal_kernels = non_dom_temporal_kernels[:, non_dom_truncation:].T
            print(f'Kernels truncated from time step {truncation} onwards.')

            to_save = dict(
                dom_temporal_kernels=dom_temporal_kernels,
                non_dom_temporal_kernels=non_dom_temporal_kernels,
                non_dominant_x=non_dominant_x,
                non_dominant_y=non_dominant_y,
                amplitude=amplitude.astype(np.float32),
                non_dom_amplitude=non_dom_amplitude.astype(np.float32),
                spontaneous_firing_rates=np.asarray(spontaneous_firing_rates, dtype=np.float32),
            )
            with open(t_path, 'wb') as f:
                pkl.dump(to_save, f)
                print('Caching temporal kernels...')
        else:
            with open(t_path, 'rb') as f:
                loaded = pkl.load(f)
            dom_temporal_kernels = np.asarray(loaded['dom_temporal_kernels'], dtype=np.float32)
            non_dom_temporal_kernels = np.asarray(loaded['non_dom_temporal_kernels'], dtype=np.float32)
            non_dominant_x = np.asarray(loaded['non_dominant_x'], dtype=np.float32)
            non_dominant_y = np.asarray(loaded['non_dominant_y'], dtype=np.float32)
            amplitude = np.asarray(loaded['amplitude'], dtype=np.float32)
            non_dom_amplitude = np.asarray(loaded['non_dom_amplitude'], dtype=np.float32)
            spontaneous_firing_rates = np.asarray(loaded['spontaneous_firing_rates'], dtype=np.float32)
            x = d['x'].to_numpy(dtype=np.float32)
            y = d['y'].to_numpy(dtype=np.float32)

        # Load the spatial kernels
        if not os.path.exists(spatial_path):
            print('Computing spatial kernels...')
            # Scale x and y within the range
            col_max = float(col_size - 1)
            row_max = float(row_size - 1)
            # Clamp x and y to stay within [0, col_max] and [0, row_max] respectively
            x = np.clip(x * col_max / col_size, 0, col_max)
            y = np.clip(y * row_max / row_size, 0, row_max)
            # Clamp non_dominant_x and non_dominant_y to stay within [0, col_max] and [0, row_max] respectively
            non_dominant_x = np.clip(non_dominant_x * col_max / col_size, 0, col_max)
            non_dominant_y = np.clip(non_dominant_y * row_max / row_size, 0, row_max)

            # prepare the spatial kernels in advance and store in TF format
            d_spatial = 1.0
            spatial_range = np.arange(0, 15, d_spatial, dtype=np.float32)
            x_range = np.arange(-50, 51)
            y_range = np.arange(-50, 51)
            # Load the spatial sizes of the LGN units
            spatial_sizes = d['spatial_size'].to_numpy(dtype=np.float32)

            bin_ids = _assign_spatial_bin_ids_numba(spatial_sizes, spatial_range)
            gaussian_filters = []
            spatial_range_indices = []

            for i in range(len(spatial_range) - 1):
                # check if there is any neuron in the spatial range
                indices = np.where(bin_ids == i)[0].astype(np.int32)
                if indices.size == 0:
                    continue
                # Precompute indices for each spatial range during initialization
                spatial_range_indices.append(indices)
                #considering the spatial range as 3 x sigma of the gaussian filter, we can compute the sigma of the Gaussian filters as:
                sigma = (spatial_range[i] + d_spatial / 2.0) / 3.0
                original_filter = GaussianSpatialFilter(
                    translate=(0.0, 0.0), sigma=(sigma, sigma), origin=(0.0, 0.0)
                )
                kernel = original_filter.get_kernel(x_range, y_range, amplitude=1.0).full()
                nonzero_inds = np.where(np.abs(kernel) > 1e-9)
                rm, rM = nonzero_inds[0].min(), nonzero_inds[0].max()
                cm, cM = nonzero_inds[1].min(), nonzero_inds[1].max()
                kernel = kernel[rm:rM + 1, cm:cM + 1]
                gaussian_filter = kernel[..., None, None].astype(np.float32, copy=False)
                gaussian_filters.append(gaussian_filter)

            # Concatenate all the ids and sort them
            if len(spatial_range_indices) > 0:
                neuron_ids = np.concatenate(spatial_range_indices, axis=0).astype(np.int32, copy=False)
                sorted_neuron_ids_indices = np.argsort(neuron_ids).astype(np.int32, copy=False)
            else:
                sorted_neuron_ids_indices = np.empty((0,), dtype=np.int32)

            # Save the spatial kernels
            to_save = dict(
                x=x,
                y=y,
                non_dominant_x=non_dominant_x,
                non_dominant_y=non_dominant_y,
                gaussian_filters=gaussian_filters,
                spatial_range_indices=spatial_range_indices,
                sorted_neuron_ids_indices=sorted_neuron_ids_indices,
            )
            with open(spatial_path, 'wb') as f:
                pkl.dump(to_save, f)
                print('Caching spatial kernels...')
        else:
            with open(spatial_path, 'rb') as f:
                loaded = pkl.load(f)
            x = np.asarray(loaded['x'], dtype=np.float32)
            y = np.asarray(loaded['y'], dtype=np.float32)
            non_dominant_x = np.asarray(loaded['non_dominant_x'], dtype=np.float32)
            non_dominant_y = np.asarray(loaded['non_dominant_y'], dtype=np.float32)
            gaussian_filters = [np.asarray(gf, dtype=np.float32) for gf in loaded['gaussian_filters']]
            spatial_range_indices = [np.asarray(a, dtype=np.int32) for a in loaded['spatial_range_indices']]
            sorted_neuron_ids_indices = np.asarray(loaded['sorted_neuron_ids_indices'], dtype=np.int32)

        # Preprocess data tensors outside the loop if they don't change
        self.x = tf.constant(x, dtype=dtype)
        self.y = tf.constant(y, dtype=dtype)
        self.non_dominant_x = tf.constant(non_dominant_x, dtype=dtype)
        self.non_dominant_y = tf.constant(non_dominant_y, dtype=dtype)
        self.amplitude = tf.constant(amplitude, dtype=dtype)
        self.non_dom_amplitude = tf.constant(non_dom_amplitude, dtype=dtype)
        self.is_composite = tf.constant(is_composite, dtype=dtype)
        self.spontaneous_firing_rates = tf.constant(spontaneous_firing_rates, dtype=dtype)

        self.dom_temporal_kernels = tf.convert_to_tensor(dom_temporal_kernels, dtype=dtype)
        self.non_dom_temporal_kernels = tf.convert_to_tensor(non_dom_temporal_kernels, dtype=dtype)
        self.gaussian_filters = [tf.convert_to_tensor(gf, dtype=dtype) for gf in gaussian_filters]
        self.spatial_range_indices = spatial_range_indices
        self.sorted_neuron_ids_indices = tf.convert_to_tensor(sorted_neuron_ids_indices, dtype=tf.int32)

        self.vertical_filters = []
        self.horizontal_filters = []
        self.edge_reciprocals = []
        for gaussian_filter in gaussian_filters:
            matrix = gaussian_filter[:, :, 0, 0]
            u, singular_values, vh = np.linalg.svd(matrix, full_matrices=False)
            if singular_values[1:].sum() > singular_values[0] * 1e-5:
                raise ValueError('Expected a rank-one Gaussian spatial kernel')
            scale = np.sqrt(singular_values[0])
            vertical_filter = (u[:, 0] * scale)[:, None, None, None]
            horizontal_filter = (vh[0] * scale)[None, :, None, None]
            self.vertical_filters.append(tf.constant(vertical_filter, dtype=dtype))
            self.horizontal_filters.append(tf.constant(horizontal_filter, dtype=dtype))
            edge_fraction = tf.nn.conv2d(
                tf.ones((1, row_size, col_size, 1), dtype=dtype),
                tf.constant(gaussian_filter, dtype=dtype),
                strides=1,
                padding='SAME',
            )
            self.edge_reciprocals.append(tf.math.reciprocal(edge_fraction))

        max_vertical = max(value.shape[0] for value in self.vertical_filters)
        max_horizontal = max(value.shape[1] for value in self.horizontal_filters)

        def center_pad(value, target, axis):
            padding = target - value.shape[axis]
            before = padding // 2
            after = padding - before
            widths = [(0, 0)] * value.ndim
            widths[axis] = (before, after)
            return np.pad(value, widths)

        packed_vertical = np.concatenate(
            [
                center_pad(np.asarray(value), max_vertical, axis=0)
                for value in self.vertical_filters
            ],
            axis=3,
        )
        packed_horizontal = np.concatenate(
            [
                center_pad(np.asarray(value), max_horizontal, axis=1)
                for value in self.horizontal_filters
            ],
            axis=2,
        )
        self.packed_vertical_filters = tf.constant(packed_vertical, dtype=dtype)
        self.packed_horizontal_filters = tf.constant(packed_horizontal, dtype=dtype)

        composite_mask = is_composite.astype(bool)
        composite_ids = np.flatnonzero(composite_mask).astype(np.int32)
        self.n_composite = composite_ids.size
        self.composite_ids = tf.constant(composite_ids)
        self.composite_non_dom_kernels = tf.gather(
            self.non_dom_temporal_kernels, self.composite_ids, axis=1
        )
        self.composite_non_dom_amplitude = tf.gather(self.non_dom_amplitude, self.composite_ids)
        self.composite_spontaneous_rates = tf.gather(
            self.spontaneous_firing_rates, self.composite_ids
        )

        self.dominant_sample_indices = []
        self.dominant_sample_weights = []
        self.non_dominant_sample_indices = []
        self.non_dominant_sample_weights = []
        grouped_composite_ids = []
        for indices in spatial_range_indices:
            dominant_indices, dominant_weights = _bilinear_metadata(
                x[indices], y[indices], col_size
            )
            selected_ids = indices[composite_mask[indices]]
            non_dominant_indices, non_dominant_weights = _bilinear_metadata(
                non_dominant_x[selected_ids], non_dominant_y[selected_ids], col_size
            )
            self.dominant_sample_indices.append(tf.constant(dominant_indices))
            self.dominant_sample_weights.append(tf.constant(dominant_weights, dtype=dtype))
            self.non_dominant_sample_indices.append(tf.constant(non_dominant_indices))
            self.non_dominant_sample_weights.append(
                tf.constant(non_dominant_weights, dtype=dtype)
            )
            grouped_composite_ids.append(selected_ids)
        grouped_composite_ids = np.concatenate(grouped_composite_ids)
        self.composite_sort_indices = tf.constant(
            np.argsort(grouped_composite_ids).astype(np.int32)
        )

    @tf.function
    def spatial_response(self, movie, bmtk_compat=True):
        """Return dominant responses and compact non-dominant composite responses."""
        movie = tf.cast(movie, dtype=self.dtype)
        convolved_movies = tf.nn.conv2d(
            movie, self.packed_vertical_filters, strides=1, padding='SAME'
        )
        convolved_movies = tf.nn.depthwise_conv2d(
            convolved_movies,
            self.packed_horizontal_filters,
            strides=(1, 1, 1, 1),
            padding='SAME',
        )
        all_spatial_responses = []
        all_non_dom_spatial_responses = []
        for i, _ in enumerate(self.spatial_range_indices):
            convolved_movie = convolved_movies[..., i:i + 1]
            if bmtk_compat:
                convolved_movie *= self.edge_reciprocals[i]
            flattened_movie = tf.reshape(
                convolved_movie[..., 0], (tf.shape(movie)[0], -1)
            )
            all_spatial_responses.append(
                _sample_spatial(
                    flattened_movie,
                    self.dominant_sample_indices[i],
                    self.dominant_sample_weights[i],
                )
            )
            all_non_dom_spatial_responses.append(
                _sample_spatial(
                    flattened_movie,
                    self.non_dominant_sample_indices[i],
                    self.non_dominant_sample_weights[i],
                )
            )

        all_spatial_responses = tf.concat(all_spatial_responses, axis=1)
        all_non_dom_spatial_responses = tf.concat(all_non_dom_spatial_responses, axis=1)
        all_spatial_responses = tf.gather(all_spatial_responses, self.sorted_neuron_ids_indices, axis=1)
        all_non_dom_spatial_responses = tf.gather(
            all_non_dom_spatial_responses, self.composite_sort_indices, axis=1
        )

        return all_spatial_responses, all_non_dom_spatial_responses

    @tf.function(jit_compile=True)
    def firing_rates_from_spatial(self, all_spatial_responses, all_non_dom_spatial_responses):
        dom_filtered_output = temporal_filter(all_spatial_responses, self.dom_temporal_kernels)
        dom_firing_rates = transfer_function(
            dom_filtered_output * self.amplitude + self.spontaneous_firing_rates,
            dtype=self.dtype,
        )
        if self.n_composite == 0:
            return dom_firing_rates

        non_dom_filtered_output = temporal_filter(
            all_non_dom_spatial_responses, self.composite_non_dom_kernels
        )
        composite_firing_rates = transfer_function(
            non_dom_filtered_output * self.composite_non_dom_amplitude
            + self.composite_spontaneous_rates,
            dtype=self.dtype,
        )
        non_dom_firing_rates = tf.transpose(
            tf.scatter_nd(
                self.composite_ids[:, None],
                tf.transpose(composite_firing_rates),
                (tf.shape(all_spatial_responses)[1], tf.shape(all_spatial_responses)[0]),
            )
        )
        return dom_firing_rates + non_dom_firing_rates


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
