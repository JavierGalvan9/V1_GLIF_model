import os
import pickle as pkl

import h5py
import numpy as np
import pandas as pd
from numba import njit


@njit
def sort_indices(indices, weights, delays):
    max_ind = np.max(indices) + 1
    # sort indices by considering first all the targets of node 0, then all of node 1, ...
    q = indices[:, 0] * max_ind + indices[:, 1] 
    sorted_ind = np.argsort(q)
    return indices[sorted_ind], weights[sorted_ind], delays[sorted_ind]

def get_model_params(d, d_index):
    params_dict = d['nodes'][d_index]['params']

    new_params_dict = {
    'EL_reference':params_dict['E_L']*10**-3,  # rest potential (also reset value)
    'EL':0,
    'C':params_dict['C_m']*10**-12,
    'R_input':1/(params_dict['g']*10**-9),
    'tau':(1/(params_dict['g']*10**-9))*params_dict['C_m']*10**-12,
    'asc_amp_array':np.array(params_dict['asc_amps'])*10**-12,
    'asc_tau_array':1/(np.array(params_dict['k'])*10**3),
    'th_inf':(params_dict['V_th'] - params_dict['E_L'])*10**-3,
    'dt':10**-3,
    #dt:1.0 / sweep_data['sampling_rate']
    }
    
    return new_params_dict

@njit
def Ij1(Ij, asc_tau_array, dt):
    Ij1 = Ij*np.exp(-(1/asc_tau_array)*dt)
    return Ij1

@njit
def dV(I_ext, Ij, V, C, R_input, EL):
    dV = (1/C)*(I_ext + np.sum(Ij) - (1/R_input)*(V-EL))
    return dV

def calculate_rheobase(new_params_dict):
    EL = new_params_dict['EL']
    C = new_params_dict['C']
    R_input = new_params_dict['R_input']
    asc_tau_array = new_params_dict['asc_tau_array']
    th_inf = new_params_dict['th_inf']
    dt = new_params_dict['dt']
    rheobase = None
    dt = 10**-3
    current = 10**-11
    while rheobase is None:
        stim = np.zeros(2000)
        stim[0:1000] = current
        stim[1000:2000] = 0
        stimulus = stim
        stimulus_size = len(stimulus)
        V_values = np.zeros(stimulus_size)
        asc_values = np.zeros((stimulus_size, 2)) # this will remain as 0 all the time since no spike take place
        V0 = V_values[0]
        asc0 = asc_values[0,:]
        time_step = 0
        while time_step < stimulus_size:
            I_ext = stimulus[time_step]
            V1 = V0 + dt*dV(I_ext, asc0, V0, C, R_input, EL)
            asc1 = Ij1(asc0, asc_tau_array, dt)
            if V1 > th_inf:
                rheobase = current
                break
            else:
                # there was no spike, store the next voltages
                V_values[time_step] = V1 
                asc_values[time_step,:] = asc1
                V0 = V1
                asc0 = asc1
                time_step += 1 
        current += 10**-14
        
    return rheobase


def load_network(path='GLIF_network/network_dat.pkl',
                 h5_path='GLIF_network/network/v1_nodes.h5',
                 core_only=True, n_neurons=None, v_old=None, seed=3000, connected_selection=False):
    rd = np.random.RandomState(seed=seed)

    with open(path, 'rb') as f:
        d = pkl.load(f) #d is a dictionary with 'nodes' and 'edges' keys

  ### This file contains the data related to each neuron class.
  # The nodes key is a list of 111 entries (one per neuron class) with the following information:
  #  'ids' (bmtk indices of the class neurons): array([173915, 174530, 175234, ..., 230780], dtype=uint32) 
  #  'params': {'asc_init': [0.0, 0.0],
  #             'V_th': -34.78002413066345,
  #             'g': 4.332666343216805,
  #             'E_L': -71.3196309407552,
  #             'k': [0.003, 0.029999999999999992],
  #             'C_m': 61.776013140488196,
  #             'V_reset': -71.3196309407552,
  #             'V_dynamics_method': 'linear_exact',
  #             'tau_syn': [5.5, 8.5, 2.8, 5.8],
  #             't_ref': 2.2,
  #             'asc_amps': [-6.621493991981387, -68.56339310938284]
  #             'rheobase': [values....]}
  
  ### The 'edges' key is a list of 1783 entries (one per edge class) with the following information:
  #  'source': array([   86,   195,   874, ..., 26266, 26563, 26755], dtype=uint64), # bmtk indices
  #  'target': array([13289, 13289, 13289, ..., 26843, 26843, 26843], dtype=uint64), # bmtk indices
  #  'params': {'model': 'static_synapse',
  #             'receptor_type': 1,
  #             'delay': 1.5,
  #             'weight': array([2.05360475e-07, 1.18761259e-20, 1.04067864e-12, ...,
  #                              3.33087865e-34, 1.26318969e-03, 1.20919572e-01])}

    n_nodes = sum([len(a['ids']) for a in d['nodes']]) # 230924 total neurons
    n_edges = sum([len(a['source']) for a in d['edges']]) # 70139111 total edges
    # max_delay = max([a['params']['delay'] for a in d['edges']])

    bmtk_id_to_tf_id = np.arange(n_nodes)
    tf_id_to_bmtk_id = np.arange(n_nodes)

    edges = d['edges']
    h5_file = h5py.File(h5_path, 'r')  
    # This file gives us the:
        # '0': coordinates of each point (and other information we are not using)
        # 'node_group_id': all nodes have the index 0
        # 'node_group_index': same as node_id
        # 'node_id': bmtk index of each node (node_id[0]=0, node_id[1]=1, ...)
        # 'node_type_id': 518290966, 539742766,... for each node
    assert np.diff(h5_file['nodes']['v1']['node_id']).var() < 1e-12
    x = np.array(h5_file['nodes']['v1']['0']['x'])
    y = np.array(h5_file['nodes']['v1']['0']['y'])
    z = np.array(h5_file['nodes']['v1']['0']['z'])
    # its a cylinder where the y variable is just the depth
    r = np.sqrt(x ** 2 + z ** 2)

    # sel is a boolean array with True value in the indices of selected neurons
    if connected_selection: # this condition takes the n_neurons closest neurons 
        sorted_ind = np.argsort(r) #order according to radius distance
        sel = np.zeros(n_nodes, np.bool)
        sel[sorted_ind[:n_neurons]] = True  # keep only the nearest n_neurons
        print(f'> Maximum sample radius: {r[sorted_ind[n_neurons - 1]]:.2f}')
    elif core_only: # this condition makes all the neurons to be within distance 400 micrometers from the origin (core)
                    # 51,978 maximum value for n_neurons in this case
        sel = r < 400
        if n_neurons is not None and n_neurons > 0:
            inds, = np.where(sel)  # indices where the condition is satisfied
            take_inds = rd.choice(inds, size=n_neurons, replace=False)
            sel[:] = False
            sel[take_inds] = True
    elif n_neurons is not None and n_neurons > 0: # this condition takes random neurons from all the V1
        legit_neurons = np.arange(n_nodes)
        take_inds = rd.choice(legit_neurons, size=n_neurons, replace=False)
        sel = np.zeros(n_nodes, np.bool)
        sel[take_inds] = True 
        
    #elif n_neurons == -1:
    #    sel = np.full(n_nodes, True)  


    n_nodes = np.sum(sel) # number of nodes selected
    tf_id_to_bmtk_id = tf_id_to_bmtk_id[sel]  # tf idx '0' corresponds to 'tf_id_to_bmtk_id[0]' bmtk idx
    bmtk_id_to_tf_id = np.zeros_like(bmtk_id_to_tf_id) - 1
    for tf_id, bmtk_id in enumerate(tf_id_to_bmtk_id):
        bmtk_id_to_tf_id[bmtk_id] = tf_id 
        
    # bmtk idx '0' corresponds to 'bmtk_id_to_tf_id[0]' tf idx which can be '-1' in case 
    #the bmtk node is not in the tensorflow selection or another value in case it belongs the selection
    x = x[sel]
    y = y[sel]
    z = z[sel]

    # from all the model edges, lets see how many correspond to the selected nodes
    n_edges = 0
    for edge in edges:
        target_tf_ids = bmtk_id_to_tf_id[np.array(edge['target'])]
        source_tf_ids = bmtk_id_to_tf_id[np.array(edge['source'])]
        edge_exists = np.logical_and(target_tf_ids >= 0, source_tf_ids >= 0)
        n_edges += np.sum(edge_exists)

    print(f'> Number of Neurons: {n_nodes}')
    print(f'> Number of Synapses: {n_edges}')

    # Save in a dictionary the properties of each of the 111 node types
    n_node_types = len(d['nodes'])
    node_params = dict(
        V_th=np.zeros(n_node_types, np.float32),
        g=np.zeros(n_node_types, np.float32),
        E_L=np.zeros(n_node_types, np.float32),
        k=np.zeros((n_node_types, 2), np.float32),
        C_m=np.zeros(n_node_types, np.float32),
        V_reset=np.zeros(n_node_types, np.float32),
        tau_syn=np.zeros((n_node_types, 4), np.float32),
        t_ref=np.zeros(n_node_types, np.float32),
        asc_amps=np.zeros((n_node_types, 2), np.float32),
        rheobase=np.zeros(n_node_types, np.float32)
    )

    # give every selected node of a given node type an index according to tf ids
    node_type_ids = np.zeros(n_nodes, np.int64)
    save_new_d_dict = False
    for i, node_type in enumerate(d['nodes']):
        tf_ids = bmtk_id_to_tf_id[np.array(node_type['ids'])] # get ALL the nodes of the given node type
        tf_ids = tf_ids[tf_ids >= 0] # choose only those that belong to our model
        node_type_ids[tf_ids] = i # assign them all the same id (which does not relate with the neuron type)
        
        if 'rheobase' not in node_type['params'].keys():
            new_params_dict = get_model_params(d, i)
            rheobase = calculate_rheobase(new_params_dict)
            d['nodes'][i]['params']['rheobase'] = rheobase
            node_type['params']['rheobase'] = rheobase
            save_new_d_dict = True
        
        for k, v in node_params.items():
            v[i] = node_type['params'][k]  # save in a dict the information of the nodes
            
    if save_new_d_dict:
        with open(path, 'wb') as handle:
            pkl.dump(d, handle)
            
    # each node has 4 different inputs (soma, dendrites, etc) with different properties each
    dense_shape = (4 * n_nodes, n_nodes)
    indices = np.zeros((n_edges, 2), dtype=np.int64)
    weights = np.zeros(n_edges, np.float32)
    delays = np.zeros(n_edges, np.float32)

    current_edge = 0
    for edge in edges:
        r = edge['params']['receptor_type'] - 1  # Indentify the which of the 4 types of inputs we have
        # r takes values whithin 0 - 3
        target_tf_ids = bmtk_id_to_tf_id[np.array(edge['target'])]
        source_tf_ids = bmtk_id_to_tf_id[np.array(edge['source'])]
        edge_exists = np.logical_and(target_tf_ids >= 0, source_tf_ids >= 0)
        # select the edges whithin our model
        target_tf_ids = target_tf_ids[edge_exists]
        source_tf_ids = source_tf_ids[edge_exists]
        weights_tf = edge['params']['weight'][edge_exists]
        delays_tf = edge['params']['delay'] # all the edges of a given type have the same delay
        n_new_edge = np.sum(edge_exists)
        indices[current_edge:current_edge +
                n_new_edge] = np.array([target_tf_ids * 4 + r, source_tf_ids]).T  
        # we multiply by 4 and add r to identify the receptor_type easily: 
        # if target id is divisible by 4 the receptor_type is 0, 
        # if it is rest is 1 by dividing by 4 then its receptor type is 1, and so on...
        weights[current_edge:current_edge + n_new_edge] = weights_tf
        delays[current_edge:current_edge + n_new_edge] = delays_tf
        current_edge += n_new_edge
    # sort indices by considering first all the targets of node 0, then all of node 1, ...
    indices, weights, delays = sort_indices(indices, weights, delays)

    network = dict(
        x=x, y=y, z=z,
        n_nodes=n_nodes,
        n_edges=n_edges,
        node_params=node_params,
        node_type_ids=node_type_ids,
        synapses=dict(indices=indices, weights=weights, delays=delays,
                      dense_shape=dense_shape),
        tf_id_to_bmtk_id=tf_id_to_bmtk_id,
        bmtk_id_to_tf_id=bmtk_id_to_tf_id
    )
    return network


# Here we load the 17400 neurons that act as input in the model
def load_input(path='GLIF_network/input_dat.pkl',
               start=0,
               duration=3000,
               dt=1,
               bmtk_id_to_tf_id=None):
    with open(path, 'rb') as f:
        d = pkl.load(f) #d contains two populations (LGN and background inputs), each of them with two elements:
            # [0] a dict with 'ids' and 'spikes'
            # [1] a list of edges types, each of them with their 'source', 'target' and 'params'
            # The first population (LGN) has 86 different edges and the source ids go from 0 to 17399
            # The second population (background) is only formed by the source index 0 (single background node) 
            # and projects to all V1 neurons with 21 different edges types and weights

    input_populations = []
    for input_population in d:
        post_indices = []
        pre_indices = []
        weights = []
        delays = []

        for edge in input_population[1]:
            r = edge['params']['receptor_type'] - 1 # Indentify the which of the 4 types of inputs we have
            # r takes values whithin 0 - 3
            target_tf_id = np.array(edge['target'])
            source_tf_id = np.array(edge['source'])
            weights_tf = np.array(edge['params']['weight'])
            delays_tf = np.zeros_like(weights_tf) + edge['params']['delay']
            if bmtk_id_to_tf_id is not None:
                # check if the given edges exist in our model 
                # (notice that only the target must exist since the source is whithin the LGN module)
                # This means that source index is whithin 0-17400
                target_tf_id = bmtk_id_to_tf_id[target_tf_id]
                edge_exists = target_tf_id >= 0
                target_tf_id = target_tf_id[edge_exists]
                source_tf_id = source_tf_id[edge_exists]
                weights_tf = weights_tf[edge_exists]
                delays_tf = delays_tf[edge_exists]
            # we multiply by 4 the indices and add r to identify the receptor_type easily: 
            # if target id is divisible by 4 the receptor_type is 0, 
            # if it is rest is 1 by dividing by 4 then its receptor type is 1, and so on...
            post_indices.extend(4 * target_tf_id + r) # extend acts by extending the list with the given object
            pre_indices.extend(source_tf_id)
            weights.extend(weights_tf)
            delays.append(delays_tf)
        indices = np.stack([post_indices, pre_indices], -1) # first column are the post indices and second column the pre indices
        weights = np.array(weights)
        delays = np.concatenate(delays)
        # sort indices by considering first all the sources of target node 0, then all of node 1, ...
        indices, weights, delays = sort_indices(indices, weights, delays)

        n_neurons = len(input_population[0]['ids']) # 17400
        spikes = np.zeros((int(duration / dt), n_neurons))
        # now we save the spikes of the input population
        for i, sp in zip(input_population[0]['ids'], input_population[0]['spikes']):
            sp = sp[np.logical_and(start < sp, sp < start + duration)] - start # consider only the spikes whithin the period we are taking
            sp = (sp / dt).astype(np.int)
            for s in sp:
                spikes[s, i] += 1

        input_populations.append(dict(n_inputs=n_neurons, indices=indices.astype(
            np.int64), weights=weights, delays=delays, spikes=spikes))
    return input_populations


def reduce_input_population(input_population, new_n_input, seed=3000):
    rd = np.random.RandomState(seed=seed)

    in_ind = input_population['indices']
    in_weights = input_population['weights']
    in_delays = input_population['delays']

    # we take input_population['n_inputs'] neurons from a list of new_n_input with replace, 
    # which means that in the end there can be less than new_n_input neurons of the LGN,
    # but they are randonmly selected
    assignment = rd.choice(np.arange(new_n_input),
                           size=input_population['n_inputs'], replace=True)
    
    weight_dict = dict()
    delays_dict = dict()
    for input_neuron in range(input_population['n_inputs']): # go through all the asignment selection made 
        assigned_neuron = assignment[input_neuron]
        sel = in_ind[:, 1] == input_neuron  # consider that neurons connected to the input_neuron
        sel_post_inds = in_ind[sel, 0] # keep that neurons connected to the input_neuron
        sel_weights = in_weights[sel]
        sel_delays = in_delays[sel]
        for post_ind, weight, delay in zip(sel_post_inds, sel_weights, sel_delays):
            t_inds = post_ind, assigned_neuron  # tuple with the indices of the post model neuron and the pre LGN neuron
            if t_inds not in weight_dict.keys(): # in case the key hasnt been already created
                weight_dict[t_inds] = 0.   
            weight_dict[t_inds] += weight # in case a LGN unit connection is repeated we consider that the weights are add up
            delays_dict[t_inds] = delay
    n_synapses = len(weight_dict)
    # we now save the synapses in arrays of indices and weights
    new_in_ind = np.zeros((n_synapses, 2), np.int64)
    new_in_weights = np.zeros(n_synapses)
    new_in_delays = np.zeros(n_synapses)
    for i, (t_ind, w) in enumerate(weight_dict.items()):
        new_in_ind[i] = t_ind
        new_in_weights[i] = w
        new_in_delays[i] = delays_dict[t_ind]
    new_in_ind, new_in_weights, new_in_delays = sort_indices(new_in_ind, new_in_weights, new_in_delays)
    new_input_population = dict(
        n_inputs=new_n_input, indices=new_in_ind, weights=new_in_weights, delays=new_in_delays, spikes=None)

    return new_input_population


def load_billeh(n_input, n_neurons, core_only, data_dir, seed=3000, connected_selection=False, n_output=2,
                neurons_per_output=16, v_old=None):
    network = load_network(
        path=os.path.join(data_dir, 'network_dat.pkl'),
        h5_path=os.path.join(data_dir, 'network/v1_nodes.h5'), core_only=core_only, v_old=v_old, n_neurons=n_neurons,
        seed=seed, connected_selection=connected_selection)
    inputs = load_input(
        start=1000, duration=1000, dt=1, path=os.path.join(data_dir, 'input_dat.pkl'),
        bmtk_id_to_tf_id=network['bmtk_id_to_tf_id'])
    df = pd.read_csv(os.path.join(
        data_dir, 'network/v1_node_types.csv'), delimiter=' ')

    ###### Select random l5e neurons #########
    l5e_types_indices = []
    for a in df.iterrows():
        if a[1]['pop_name'].startswith('e5'):
            l5e_types_indices.append(a[0])
    l5e_types_indices = np.array(l5e_types_indices)
    l5e_neuron_sel = np.zeros(network['n_nodes'], np.bool)
    for l5e_type_index in l5e_types_indices:
        is_l5_type = network['node_type_ids'] == l5e_type_index
        l5e_neuron_sel = np.logical_or(l5e_neuron_sel, is_l5_type)
    network['l5e_types'] = l5e_types_indices
    network['l5e_neuron_sel'] = l5e_neuron_sel
    print(f'> Number of L5e Neurons: {np.sum(l5e_neuron_sel)}')

    # assert that you have enough l5 neurons for all the outputs and then choose n_output * neurons_per_output random neurons
    # assert np.sum(l5e_neuron_sel) > n_output * neurons_per_output 
    rd = np.random.RandomState(seed=seed)
    l5e_neuron_indices = np.where(l5e_neuron_sel)[0]
    readout_neurons = rd.choice(
        l5e_neuron_indices, size=n_output * neurons_per_output, replace=False)
    readout_neurons = readout_neurons.reshape((n_output, neurons_per_output))
    network['readout_neuron_ids'] = readout_neurons
    ##########################################

    input_population = inputs[0]
    bkg = inputs[1] # contains the single background node that projects to all V1 neurons
    bkg_weights = np.zeros((network['n_nodes'] * 4,), np.float32)
    bkg_weights[bkg['indices'][:, 0]] = bkg['weights']
    if n_input != 17400:
        input_population = reduce_input_population(
            input_population, n_input, seed=seed)
    # return input_population, network, bkg_weights
    return input_population, network, bkg, bkg_weights


# If the model already exist we can load it, or if it does not just save it for future occasions
def cached_load_billeh(n_input, n_neurons, core_only, data_dir, seed=3000, connected_selection=False, n_output=2,
                       neurons_per_output=16):
    store = False
    input_population, network, bkg, bkg_weights = None, None, None, None
    flag_str = f'in{n_input}_rec{n_neurons}_s{seed}_c{core_only}_con{connected_selection}'
    flag_str += f'_out{n_output}_nper{neurons_per_output}'
    file_dir = os.path.split(__file__)[0]
    cache_path = os.path.join(
        file_dir, f'.cache/billeh_network_{flag_str}.pkl')
    # os.remove(cache_path)
    if os.path.exists(cache_path):
        try:
            with open(cache_path, 'rb') as f:
                input_population, network, bkg, bkg_weights = pkl.load(f)
                print(f'> Sucessfully restored Billeh model from {cache_path}')
        except Exception as e:
            print(e)
            store = True
    else:
        store = True
    if input_population is None or network is None or bkg is None or bkg_weights is None:
        input_population, network, bkg, bkg_weights = load_billeh(
            n_input, n_neurons, core_only, data_dir, seed,
            connected_selection=connected_selection, n_output=n_output,
            neurons_per_output=neurons_per_output)
    if store:
        os.makedirs(os.path.join(file_dir, '.cache'), exist_ok=True)
        with open(cache_path, 'wb') as f:
            pkl.dump((input_population, network, bkg, bkg_weights), f)
        print(f'> Cached Billeh model in {cache_path}')
    return input_population, network, bkg, bkg_weights
