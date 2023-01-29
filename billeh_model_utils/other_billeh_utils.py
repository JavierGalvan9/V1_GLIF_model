# -*- coding: utf-8 -*-
"""
Created on Thu Jan  6 19:43:44 2022

@author: javig
"""


import pandas as pd
import os
import sys
import glob
import numpy as np
import h5py
import time
from scipy.ndimage import gaussian_filter1d
parentDir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(parentDir, "general_utils"))
import file_management

def pop_names(network, data_dir='GLIF_network'):
    path_to_csv = os.path.join(data_dir, 'network/v1_node_types.csv')
    path_to_h5 = os.path.join(data_dir, 'network/v1_nodes.h5')
    node_types = pd.read_csv(path_to_csv, sep=' ')
    node_h5 = h5py.File(path_to_h5, mode='r')
    node_type_id_to_pop_name = dict()
    for nid in np.unique(node_h5['nodes']['v1']['node_type_id']): 
        # if not np.unique all of the 230924 model neurons ids are considered, 
        # but nearly all of them are repeated since there are only 111 different indices
        ind_list = np.where(node_types.node_type_id == nid)[0]
        assert len(ind_list) == 1
        node_type_id_to_pop_name[nid] = node_types.pop_name[ind_list[0]]
    true_pop_names = []  # it contains the pop_name of all the 230,924 neurons
    for nid in node_h5['nodes']['v1']['node_type_id']:
        true_pop_names.append(node_type_id_to_pop_name[nid])
     # Select population names of neurons in the present network (core)
    true_pop_names = np.array(true_pop_names)[network['tf_id_to_bmtk_id']]
    
    return true_pop_names

def angle_tunning(network, data_dir='GLIF_network'):
    path_to_h5 = os.path.join(data_dir, 'network/v1_nodes.h5')
    node_h5 = h5py.File(path_to_h5, mode='r')
    angle_tunning = np.array(node_h5['nodes']['v1']['0']['tuning_angle'][:])[network['tf_id_to_bmtk_id']]    
    
    return angle_tunning

def isolate_core_neurons(network, data_dir='GLIF_network'):
    path_to_h5 = os.path.join(data_dir, 'network/v1_nodes.h5')
    node_h5 = h5py.File(path_to_h5, mode='r')
    x = np.array(node_h5['nodes']['v1']['0']['x'])
    z = np.array(node_h5['nodes']['v1']['0']['z'])
    r = np.sqrt(x ** 2 + z ** 2)
    r_network_nodes = r[network['tf_id_to_bmtk_id']]
    selected_mask = r_network_nodes < 400
       
    return selected_mask
    
def isolate_neurons(network, neuron_population='e23', data_dir='GLIF_network'):
    n_neurons = network['n_nodes']
    node_types = pd.read_csv(os.path.join(data_dir, 'network/v1_node_types.csv'), sep=' ')
    path_to_h5 = os.path.join(data_dir, 'network/v1_nodes.h5')
    node_h5 = h5py.File(path_to_h5, mode='r')
    node_type_id_to_pop_name = dict()
    for nid in np.unique(node_h5['nodes']['v1']['node_type_id']): 
        # if not np.unique all of the 230924 model neurons ids are considered, 
        # but nearly all of them are repeated since there are only 111 different indices
        ind_list = np.where(node_types.node_type_id == nid)[0]
        assert len(ind_list) == 1
        node_type_id_to_pop_name[nid] = node_types.pop_name[ind_list[0]]   
    
    node_type_ids = np.array(node_h5['nodes']['v1']['node_type_id'])
    true_node_type_ids = node_type_ids[network['tf_id_to_bmtk_id']] 
    selected_mask = np.zeros(n_neurons, bool)
    for pop_id, pop_name in node_type_id_to_pop_name.items():
        # if pop_name[0] == neuron_population[0] and pop_name[1] == neuron_population[1]:
        if neuron_population in pop_name:
            # choose all the neurons of the given pop_id
            sel = true_node_type_ids == pop_id
            selected_mask = np.logical_or(selected_mask, sel)
    
    return selected_mask
    
def firing_rates_smoothing(z, sampling_rate=60, window_size=100): #window_size=300
    n_simulations, simulation_length, n_neurons = z.shape
    sampling_interval = int(1000/sampling_rate) #ms
    window_size = int(np.round(window_size/sampling_interval))
    #z = z.reshape(n_simulations, simulation_length, z.shape[1])
    z_chunks = [z[:, x:x+sampling_interval, :] for x in range(0, simulation_length, sampling_interval)]
    sampled_firing_rates = np.array([np.sum(group, axis = 1) * sampling_rate for group in z_chunks])  # (simulation_length, n_simulations, n_neurons)
    smoothed_fr = gaussian_filter1d(sampled_firing_rates, window_size, axis=0)
    smoothed_fr = np.swapaxes(smoothed_fr, 0, 1)
    return smoothed_fr, sampling_interval

def voltage_spike_effect_correction(v, z, pre_spike_gap=2, post_spike_gap=3):
    n_simulations, simulation_length, n_neurons = v.shape
    v = v.reshape((n_simulations*simulation_length, n_neurons))
    z = z.reshape((n_simulations*simulation_length, n_neurons))
    # Find the spike times excluding the last miliseconds
    spikes_idx, neurons_idx = np.where(z[:-post_spike_gap,:]==1)
    # Filter early spikes
    mask = spikes_idx>= pre_spike_gap
    spikes_idx = spikes_idx[mask]
    neurons_idx = neurons_idx[mask]
    for t_idx, n_idx in zip(spikes_idx, neurons_idx):
        pre_t_idx = t_idx-pre_spike_gap
        post_t_idx = t_idx+post_spike_gap
        prev_value = v[pre_t_idx, n_idx]
        post_value = v[post_t_idx, n_idx]
        # Make a linear interpolation of the voltage in the surroundings of the spike
        step = (post_value-prev_value)/(pre_spike_gap+post_spike_gap+1)
        if step==0:
            new_values = np.ones(pre_spike_gap+post_spike_gap+1)*post_value
        else:
            new_values = np.arange(prev_value, post_value, step)
        v[pre_t_idx:post_t_idx+1, n_idx] = new_values
    v = v.reshape((n_simulations, simulation_length, n_neurons))
    return v

############################ DATA SAVING AND LOADING METHODS #########################
class SaveSimDataHDF5:
    def __init__(self, flags, keys, data_path, network, save_core_only=True, dtype=np.float16):
        self.keys = keys
        self.data_path = data_path
        os.makedirs(self.data_path, exist_ok=True)
        self.dtype = dtype
        if save_core_only:
            self.core_mask = isolate_core_neurons(network, data_dir=flags.data_dir)
        else:
            self.core_mask = np.full(flags.neurons, True)
        self.V1_data_shape = (flags.n_simulations, flags.seq_len, flags.neurons)
        self.V1_core_data_shape = (flags.n_simulations, flags.seq_len, self.core_mask.sum())
        self.LGN_data_shape = (flags.n_simulations, flags.seq_len, flags.n_input)
        with h5py.File(os.path.join(self.data_path, 'simulation_data.hdf5'), 'w') as f:
            g = f.create_group('Data')
            for key in self.keys:
                if key=='z':
                    g.create_dataset(key, self.V1_data_shape, dtype=np.uint8, 
                                     chunks=True, compression='gzip', shuffle=True)
                elif key=='z_lgn':
                    g.create_dataset(key, self.LGN_data_shape, dtype=np.uint8, 
                                     chunks=True, compression='gzip', shuffle=True)
                else:
                    g.create_dataset(key, self.V1_core_data_shape, dtype=self.dtype, 
                                     chunks=True, compression='gzip', shuffle=True)
            for flag, val in flags.flag_values_dict().items():
                if isinstance(val, (float, int, str, bool)):
                    g.attrs[flag] = val
            g.attrs['Date'] = time.time()
                
    def __call__(self, simulation_data, trial):
        with h5py.File(os.path.join(self.data_path, 'simulation_data.hdf5'), 'a') as f:
            for key, val in simulation_data.items():
                if key in ['z', 'z_lgn']:
                    val = np.array(val).astype(np.uint8)
                    # val = np.packbits(val)
                else:
                    val = np.array(val)[:, :, self.core_mask].astype(self.dtype)
                f['Data'][key][trial, :, :] = val
    
    
# def save_simulation_results_h5df(flags, simulation_data, network, data_path, trial, save_core_only=True,
#                             dtype=np.float16):
#     if save_core_only:
#         core_mask = isolate_core_neurons(network, data_dir=flags.data_dir)
#     else:
#         core_mask = np.full(flags.neurons, True)
        
#     with h5py.File(os.path.join(data_path, 'Simulation_data.h5'), "wb") as f:
#         for key, val in simulation_data.items():
#             if key in ['z', 'z_lgn']:
#                 val = np.array(val).astype(np.uint8)
#                 val = np.packbits(val)
#             else:
#                 val = np.array(val)[:, :, core_mask].astype(dtype)
#             key_grp = f.create_group(key)
#             key_grp.create_dataset(trial, data=val, compression='gzip')
        
    # for key, val in simulation_data.items():
    #     if key in ['z', 'z_lgn']:
    #         val = np.array(val).astype(np.uint8)
    #         val = np.packbits(val)
    #         file_management.save_lzma(val, f'{key}_{trial}.lzma', data_path)
    #     else:
    #         val = np.array(val)[:, :, core_mask].astype(dtype)
    #         file_management.save_lzma(val, f'{key}_{trial}.lzma', data_path)
            

class SaveSimData:
    def __init__(self, flags, keys, data_path, network, save_core_only=True, 
                 compress_data=True, dtype=np.float16):
        self.keys = keys
        self.data_path = data_path
        os.makedirs(self.data_path, exist_ok=True)
        self.dtype = dtype
        if save_core_only:
            self.core_mask = isolate_core_neurons(network, data_dir=flags.data_dir)
        else:
            self.core_mask = np.full(flags.neurons, True)
        self.V1_data_shape = (flags.n_simulations, flags.seq_len, flags.neurons)
        self.V1_core_data_shape = (flags.n_simulations, flags.seq_len, self.core_mask.sum())
        self.LGN_data_shape = (flags.n_simulations, flags.seq_len, flags.n_input)
        if compress_data:
            self.save_method = file_management.save_lzma
        else:
            self.save_method = file_management.save_pickle
                
    def __call__(self, simulation_data, trial):
        for key, val in simulation_data.items():
            if key in ['z', 'z_lgn']:
                val = np.array(val).astype(np.uint8)
                # val = np.packbits(val)
            else:
                val = np.array(val)[:, :, self.core_mask].astype(self.dtype)
            self.save_method(val, f'{key}_{trial}', self.data_path)
            

# def save_simulation_results(flags, simulation_data, network, data_path, trial, save_core_only=True,
#                             compress_data=True, dtype=np.float16):
#     if save_core_only:
#         core_mask = isolate_core_neurons(network, data_dir=flags.data_dir)
#     else:
#         core_mask = np.full(flags.neurons, True)

#     if compress_data:
#         save_method = file_management.save_lzma
#     else:
#         save_method = file_management.save_pickle
        
#     for key, val in simulation_data.items():
#         if key in ['z', 'z_lgn']:
#             val = np.array(val).astype(np.uint8)
#             val = np.packbits(val)
#         else:
#             val = np.array(val)[:, :, core_mask].astype(dtype)
#         save_method(val, f'{key}_{trial}', data_path)


def load_simulation_results(full_data_path, n_simulations=None, skip_first_simulation=False, 
                            variables=None, simulation_length=2500, n_neurons=230924, 
                            n_core_neurons=51978, n_input=17400,
                            compress_data=True, dtype=np.float16):
    if compress_data:
        load_method = file_management.load_lzma
    else:
        load_method = file_management.load_pickle
    
    if n_simulations is None:
        n_simulations = len(glob.glob(os.path.join(full_data_path, 'v*')))
    first_simulation = 0
    last_simulation = n_simulations
    if skip_first_simulation:
        n_simulations -= 1
        first_simulation += 1
    if variables == None:
        variables = ['v', 'z', 'input_current', 'recurrent_current', 'bottom_up_current', 'z_lgn']
    if type(variables) == str:
        variables = [variables]
    data = {key: (np.zeros((n_simulations, simulation_length, n_input), np.uint8) if key=='z_lgn' 
                  else np.zeros((n_simulations, simulation_length, n_neurons), np.uint8) if key=='z' 
                  else np.zeros((n_simulations, simulation_length, n_core_neurons), dtype))
            for key in variables}

    for i in range(first_simulation, last_simulation):
        for key, value in data.items():
            key_trial_file = glob.glob(os.path.join(full_data_path, f'{key}_{i}.*'))[0]
            data_array = load_method(key_trial_file)
            # if key == 'z':
                # unpacked_array = np.unpackbits(data_array)
                # data_array = unpacked_array.reshape((1,simulation_length,n_neurons))
            # elif key == 'z_lgn':
                # unpacked_array = np.unpackbits(data_array)
                # data_array = unpacked_array.reshape((1,simulation_length,n_input))
            if key in ['z', 'z_lgn']:
                data[key][(i-first_simulation):(i+1-first_simulation), :,:] = data_array.astype(np.uint8)
            else:
                data[key][(i-first_simulation):(i+1-first_simulation), :,:] = data_array.astype(np.float32)
            
    # if len(variables) == 1:
    #     data = data[key]
        
    return data, n_simulations


def load_simulation_results_hdf5(full_data_path, n_simulations=None, skip_first_simulation=False, 
                                variables=None):
    # Prepare dictionary to store the simulation metadata
    flags_dict = {}
    with h5py.File(full_data_path, 'r') as f:
        dataset = f['Data']
        flags_dict.update(dataset.attrs)
        # Get the simulation features
        if n_simulations is None:
            n_simulations = dataset['z'].shape[0]
        first_simulation = 0
        last_simulation = n_simulations
        if skip_first_simulation:
            n_simulations -= 1
            first_simulation += 1
        # Select the variables for the extraction
        if variables == None:
            variables = ['v', 'z', 'input_current', 'recurrent_current', 'bottom_up_current', 'z_lgn']
        if type(variables) == str:
            variables = [variables]
        # Extract the simulation data
        data = {}
        for key in variables:
            if key in ['z', 'z_lgn']:
               data[key] = np.array(dataset[key][first_simulation:last_simulation, :,:]).astype(np.uint8) 
            else:
                data[key] = np.array(dataset[key][first_simulation:last_simulation, :,:]).astype(np.float32)
            
    # if len(variables) == 1:
    #     data = data[key]
        
    return data, flags_dict, n_simulations