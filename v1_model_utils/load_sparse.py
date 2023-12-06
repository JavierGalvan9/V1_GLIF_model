import os
import pickle as pkl
import json

import h5py
import numpy as np
import pandas as pd
from numba import njit
from time import time

from memory_profiler import profile

def sort_indices(indices, *arrays):
    max_ind = np.max(indices) + 1
    q = indices[:, 0] * max_ind + indices[:, 1]
    sorted_ind = np.argsort(q)
    sorted_arrays = list(map(lambda arr: arr[sorted_ind], [indices, *arrays]))
    return tuple(sorted_arrays)


# @njit
# def sort_input_indices(indices, weights, delays, tau_syn_weights_array):
#     max_ind = np.max(indices) + 1
#     # sort indices by considering first all the targets of node 0, then all of node 1, ...
#     q = indices[:, 0] * max_ind + indices[:, 1]
#     sorted_ind = np.argsort(q)
    
#     return (indices[sorted_ind], weights[sorted_ind], delays[sorted_ind], tau_syn_weights_array[sorted_ind, :])


def create_network_dat(data_dir='GLIF_network/network', source='v1', target='v1', 
                        output_file='GLIF_network/network_dat.pkl', save_pkl=True):
    # Extract the network data into a pickle file from the SONATA files
    new_network_dat = {"nodes": [], "edges": []}

    # If the source and target are v1, then extract nodes dynamic parameters
    if source == "v1" and target == "v1":
        # Load the nodes file and create a dictionary with the node ids and the node parameters
        node_file = f'{source}_nodes.h5'
        node_types_file = f'{source}_node_types.csv'
        nodes_h5_path = os.path.join(data_dir, node_file)
        node_types_df = pd.read_csv(os.path.join(data_dir, node_types_file), delimiter=" ")
        cell_models_path = os.path.join('GLIF_network', "components", "cell_models")

        with h5py.File(nodes_h5_path, "r") as nodes_h5_file:
            source_node_ids = np.array(nodes_h5_file["nodes"][source]["node_id"], dtype=np.int32)
            source_node_type_ids = np.array(nodes_h5_file["nodes"][source]["node_type_id"], dtype=np.int32)

        for node_type_id in node_types_df["node_type_id"]:
            mask = source_node_type_ids == node_type_id
            new_pop_dict = {"ids": source_node_ids[mask].astype(np.int32)}

            node_params_file = os.path.join(cell_models_path, f"{node_type_id}_glif_lif_asc_config.json")
            with open(node_params_file) as f:
                cell_model_dict = json.load(f)

            # rename V_m key in dictionary to V_reset
            cell_model_dict["V_reset"] = cell_model_dict.pop("V_m")
            # rename asc_decay key in dictionary to k
            cell_model_dict["k"] = cell_model_dict.pop("asc_decay")
            new_pop_dict["params"] = cell_model_dict

            new_network_dat["nodes"].append(new_pop_dict)

    # Process edges
    edge_file = f'{source}_{target}_edges.h5'
    edge_types_file = f'{source}_{target}_edge_types.csv'
    edges_h5_path = os.path.join(data_dir, edge_file)
    edges_type_df = pd.read_csv(os.path.join(data_dir, edge_types_file), delimiter=" ")
    synaptic_models_path = os.path.join('GLIF_network', 'components', "synaptic_models")
    basis_function_weights_df = pd.read_csv('GLIF_network/synaptic_data/basis_function_weights.csv', index_col=0)
    print(f'Saving basis function weights for {source}-{target}')
    # Map the synaptic model to a given id using a dictionary
    path = os.path.join('GLIF_network', 'synaptic_models_to_syn_id_dict.pkl')
    if not os.path.exists(path):
        synaptic_models_to_syn_id_dict = dict()
        syn_id_to_syn_weights_dict = dict()
        for syn_id, file in enumerate(os.listdir(synaptic_models_path)):
            if file.endswith(".json"):
                synaptic_model = file.split('.')[0]
                synaptic_models_to_syn_id_dict[synaptic_model] = syn_id
                tau_syn_weights = basis_function_weights_df.loc[synaptic_model].values
                syn_id_to_syn_weights_dict[syn_id] = tau_syn_weights
        with open(path, "wb") as f:
            pkl.dump(synaptic_models_to_syn_id_dict, f)
        with open('GLIF_network/syn_id_to_syn_weights_dict.pkl', "wb") as f:
            pkl.dump(syn_id_to_syn_weights_dict, f)
    else:
        with open(path, "rb") as f:
            synaptic_models_to_syn_id_dict = pkl.load(f)

    with h5py.File(edges_h5_path, "r") as edges_h5_file:
        source_to_target = f"{source}_to_{target}"
        edge_type_ids = np.array(edges_h5_file["edges"][source_to_target]["edge_type_id"], dtype=np.int32)
        source_node_ids = np.array(edges_h5_file["edges"][source_to_target]["source_node_id"], dtype=np.int32)
        target_node_ids = np.array(edges_h5_file["edges"][source_to_target]["target_node_id"], dtype=np.int32)
        if source != 'bkg':
            syn_weights = np.array(edges_h5_file["edges"][source_to_target]["0"]["syn_weight"], dtype=np.float32)

    if source == 'bkg':
        syn_weights = np.zeros(len(source_node_ids), dtype=np.float32)

    for idx, (edge_type_id, edge_model, edge_delay, edge_dyn_params) in edges_type_df[
        ["edge_type_id", "model_template", "delay", "dynamics_params"]].iterrows():

        mask = edge_type_ids == edge_type_id

        if source == 'bkg':
            syn_weights[mask] = edges_type_df.loc[idx, 'syn_weight']

        synaptic_model = edge_dyn_params.split('.')[0]
        tau_syn_weights = basis_function_weights_df.loc[synaptic_model].values
        edge_params_file = os.path.join(synaptic_models_path, edge_dyn_params)
        with open(edge_params_file) as f:
            synaptic_model_dict = json.load(f)
            receptor_type = synaptic_model_dict["receptor_type"]

        syn_id = synaptic_models_to_syn_id_dict[synaptic_model]
        new_pop_dict = {
            "source": source_node_ids[mask],
            "target": target_node_ids[mask],
            "params": {
                "model": edge_model,
                "delay": edge_delay,
                "weight": syn_weights[mask],
                "synaptic_model": synaptic_model,
                "receptor_type": receptor_type,
                "tau_syn_weights": tau_syn_weights,
                "syn_id": syn_id
            },
        }

        new_network_dat["edges"].append(new_pop_dict)

    if save_pkl:    
        with open(output_file, "wb") as f:
            pkl.dump(new_network_dat, f)

    return new_network_dat

# @profile
def load_network(
    path="GLIF_network/network_dat.pkl",
    h5_path="GLIF_network/network/v1_nodes.h5",
    core_only=True,
    n_neurons=296991,
    seed=3000,
    connected_selection=False,
    n_syn_basis=5):

    rd = np.random.RandomState(seed=seed)

    # Create / Load the network_dat pickle file from the SONATA files
    if not os.path.exists(path):
        print("Creating network_dat.pkl file...")
        d = create_network_dat(data_dir='GLIF_network/network', source='v1', target='v1',
                                output_file='GLIF_network/network_dat.pkl', save_pkl=True)
    else:
        print("Loading network_dat.pkl file...")
        with open(path, "rb") as f:
            d = pkl.load(f)  # d is a dictionary with 'nodes' and 'edges' keys

    # This file contains the data related to each neuron class.
    # The nodes key is a list of 201 entries (one per neuron class) with the following information:
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

    # The 'edges' key is a list of 1783 entries (one per edge class) with the following information:
    #  'source': array([   86,   195,   874, ..., 26266, 26563, 26755], dtype=uint64), # bmtk indices
    #  'target': array([13289, 13289, 13289, ..., 26843, 26843, 26843], dtype=uint64), # bmtk indices
    #  'params': {'model': 'static_synapse',
    #             'receptor_type': 1,
    #             'delay': 1.5,
    #             'weight': array([2.05360475e-07, 1.18761259e-20, 1.04067864e-12, ...,
    #                              3.33087865e-34, 1.26318969e-03, 1.20919572e-01])}

    # Get the number of nodes in the full V1 network
    n_nodes = sum([len(a["ids"]) for a in d["nodes"]])  # 296991 total neurons
    # Create arrays to convert between bmtk and tf ides
    bmtk_id_to_tf_id = np.arange(n_nodes, dtype=np.int32)
    tf_id_to_bmtk_id = np.arange(n_nodes, dtype=np.int32)

    # Extract from the SONATA file the nodes information
    h5_file = h5py.File(h5_path, "r")
    assert np.diff(h5_file["nodes"]["v1"]["node_id"]).var() < 1e-12
    x = np.array(h5_file["nodes"]["v1"]["0"]["x"], dtype=np.float32) # horizontal axis
    y = np.array(h5_file["nodes"]["v1"]["0"]["y"], dtype=np.float32) # depth
    z = np.array(h5_file["nodes"]["v1"]["0"]["z"], dtype=np.float32) # horizontal axis
    tuning_angle = np.array(h5_file['nodes']['v1']['0']['tuning_angle'], dtype=np.float32)
    node_type_id = np.array(h5_file['nodes']['v1']['node_type_id'], dtype=np.int32)
    r = np.sqrt(x**2 + z**2)  # the maximum radius is 845

    ### CHOOSE THE NETWORK NODES ###
    if n_neurons > 296991:
        raise ValueError("There are only 296991 neurons in the network")
    
    elif connected_selection: # this condition takes the n_neurons closest neurons to the origin
        sorted_ind = np.argsort(r)
        sel = np.zeros(n_nodes, np.bool_)
        sel[sorted_ind[:n_neurons]] = True
        print(f"> Maximum sample radius: {r[sorted_ind[n_neurons - 1]]:.2f}")
    
    elif core_only: # this condition takes the n_neurons closest neurons to the origin (core)
        sel = r < 400
        if n_neurons > 66634:
            raise ValueError("There are only 66634 neurons in the network core")
        elif n_neurons > 0 and n_neurons <= 66634:
            (inds,) = np.where(sel)
            take_inds = rd.choice(inds, size=n_neurons, replace=False)
            sel[:] = False
            sel[take_inds] = True
    
    elif n_neurons > 0 and n_neurons <= 296991: # this condition randomly selects neurons from the whole V1
        legit_neurons = np.arange(n_nodes)
        take_inds = rd.choice(legit_neurons, size=n_neurons, replace=False)
        sel = np.zeros(n_nodes, np.bool_)
        sel[take_inds] = True
    
    else: # if no condition is met, all neurons are selected
        sel = np.ones(n_nodes, np.bool_)

    # Get the number of neurons in the chosen network and update the traslation arrays
    n_nodes = np.sum(sel)
    tf_id_to_bmtk_id = tf_id_to_bmtk_id[sel] # tf idx '0' corresponds to 'tf_id_to_bmtk_id[0]' bmtk idx
    bmtk_id_to_tf_id = np.zeros_like(bmtk_id_to_tf_id) - 1
    for tf_id, bmtk_id in enumerate(tf_id_to_bmtk_id):
        bmtk_id_to_tf_id[bmtk_id] = tf_id
    # bmtk idx '0' corresponds to 'bmtk_id_to_tf_id[0]' tf idx which can be '-1' in case
    # the bmtk node is not in the tensorflow selection or another value in case it belongs the selection
    
    # Get the properties from the network neurons
    x = x[sel]
    y = y[sel]
    z = z[sel]
    tuning_angle = tuning_angle[sel]
    node_type_id = node_type_id[sel]

    # Count the number of edges in the current network
    edges = d["edges"]
    n_edges = 0
    for edge in edges:
        target_tf_ids = bmtk_id_to_tf_id[np.array(edge["target"], dtype=np.int32)]
        source_tf_ids = bmtk_id_to_tf_id[np.array(edge["source"], dtype=np.int32)]
        edge_exists = np.logical_and(target_tf_ids >= 0, source_tf_ids >= 0)
        n_edges += np.sum(edge_exists)

    print(f"> Number of Neurons: {n_nodes}")
    print(f"> Number of Synapses: {n_edges}")

    # GET THE NODES PARAMETERS
    n_node_types = len(d["nodes"])
    node_params = dict(
        V_th=np.zeros(n_node_types, np.float32),
        g=np.zeros(n_node_types, np.float32),
        E_L=np.zeros(n_node_types, np.float32),
        k=np.zeros((n_node_types, 2), np.float32),
        C_m=np.zeros(n_node_types, np.float32),
        V_reset=np.zeros(n_node_types, np.float32),
        t_ref=np.zeros(n_node_types, np.float32),
        asc_amps=np.zeros((n_node_types, 2), np.float32),
    )

    node_type_ids = np.zeros(n_nodes, np.int32)
    for i, node_type in enumerate(d["nodes"]):
        # get ALL the nodes of the given node type
        tf_ids = bmtk_id_to_tf_id[np.array(node_type["ids"], dtype=np.int32)]
        # choose only those that belong to our model
        tf_ids = tf_ids[tf_ids >= 0]
        # assign them all the same id (which does not relate with the neuron type)
        node_type_ids[tf_ids] = i
        for k, v in node_params.items():
            # save in a dict the information of the nodes
            v[i] = node_type["params"][k]

    # GET THE EDGES INFORMATION
    dense_shape = (n_nodes, n_nodes)
    indices = np.zeros((n_edges, 2), dtype=np.int32) # Tf requires int64 for indices in sparse matrix 
    weights = np.zeros(n_edges, np.float32)
    delays = np.zeros(n_edges, np.float16)
    syn_ids = np.zeros(n_edges, np.uint8)

    current_edge = 0
    for edge in edges:
        # Identify which of the 10 types of inputs we have
        # r = edge["params"]["receptor_type"] - 1
        # r takes values within 0 - 9
        target_tf_ids = bmtk_id_to_tf_id[np.array(edge["target"], dtype=np.int32)]
        source_tf_ids = bmtk_id_to_tf_id[np.array(edge["source"], dtype=np.int32)]
        edge_exists = np.logical_and(target_tf_ids >= 0, source_tf_ids >= 0)
        # select the edges within our model
        target_tf_ids = target_tf_ids[edge_exists]
        source_tf_ids = source_tf_ids[edge_exists]
        weights_tf = edge["params"]["weight"][edge_exists]
        # tau_syn_weights = edge["params"]["tau_syn_weights"]
        
        # all the edges of a given type have the same delay and synaptic id
        delays_tf = edge["params"]["delay"]
        syn_id = edge["params"]["syn_id"]
        n_new_edge = np.sum(edge_exists)
        # we multiply by max_n_receptors and add r to identify the receptor_type easily:
        # if target id is divisible by max_n_receptors the receptor_type is 0,
        # if it is rest is 1 by dividing by max_n_receptors then its receptor type is 1, and so on...
        # indices[current_edge: current_edge + n_new_edge] = np.array(
        #     [target_tf_ids * max_n_receptors + r, source_tf_ids]
        # ).T

        indices[current_edge: current_edge + n_new_edge] = np.array(
            [target_tf_ids, source_tf_ids]
        ).T

        weights[current_edge: current_edge + n_new_edge] = weights_tf
        delays[current_edge: current_edge + n_new_edge] = delays_tf
        # tau_syn_weights_array[current_edge: current_edge + n_new_edge, :] = tau_syn_weights
        syn_ids[current_edge: current_edge + n_new_edge] = syn_id
        current_edge += n_new_edge
    # sort indices by considering first all the targets of node 0, then all of node 1, ...
    # indices, weights, delays, tau_syn_weights_array, syn_ids = sort_indices(indices, weights, delays, tau_syn_weights_array, syn_ids)
    indices, weights, delays, syn_ids = sort_indices(indices, weights, delays, syn_ids)

    # # i want the weights array to be of the same shape as tau_syn_weights_array. make copies of the weights along a new axis to achieve this
    # weights = np.repeat(weights[:, np.newaxis], n_syn_basis, axis=1)    
    # # same for the delays
    # delays = np.repeat(delays[:, np.newaxis], n_syn_basis, axis=1)
    # delays = delays.flatten()

    # # i want the indices array to be of the same length as tau_syn_weights_array shape[1]. 
    # # make copies of the indices along its first axis adding 1 for every copy 
    # # (this is because the indices are multiplied by the number of receptors)

    # # element wise multiply the weights and tau_syn_weights_array
    # weights = weights * tau_syn_weights_array
    # # flatten the weights
    # weights = weights.flatten()

    # # indices = indices * n_syn_basis
    # indices[:, 0] = indices[:, 0] * n_syn_basis
    # indices = np.repeat(indices, n_syn_basis, axis=0)
    # indices[:, 0] += np.tile(np.arange(n_syn_basis), n_edges)

    network = dict(
        x=x, y=y, z=z,
        tuning_angle=tuning_angle,
        node_type_id=node_type_id,
        n_nodes=n_nodes,
        n_edges=n_edges,
        node_params=node_params,
        node_type_ids=node_type_ids,
        synapses=dict(indices=indices, weights=weights, delays=delays, 
                    #   tau_syn_weights_array=tau_syn_weights_array,
                      syn_ids=syn_ids,
                      dense_shape=dense_shape),
        tf_id_to_bmtk_id=tf_id_to_bmtk_id,
        bmtk_id_to_tf_id=bmtk_id_to_tf_id,
    )

    return network


def load_input_tf(
    lgn_input,
    start=0,
    duration=3000,
    dt=1,
    bmtk_id_to_tf_id=None,
    # max_n_receptors=10
):        
    # Unite the edges information of the LGN and the background noise
    input_edges = [lgn_input['edges']]
       
    input_populations = []
    for idx, input_population in enumerate(input_edges):
        post_indices = []
        pre_indices = []
        weights = []
        delays = []
        syn_ids = []

        for idx, edge in enumerate(input_population):
            target_bmtk_id = tf.constant(edge["target"], dtype=tf.int32)
            source_tf_id = tf.constant(edge["source"], dtype=tf.int32)
            weights_tf = tf.constant(edge["params"]["weight"], dtype=tf.float32)
            delays_tf = tf.zeros_like(weights_tf, dtype=tf.float16) + edge["params"]["delay"]
            syn_id = tf.zeros_like(weights_tf, dtype=tf.uint8) + edge["params"]["syn_id"]
            
            if bmtk_id_to_tf_id is not None:
                target_tf_id = bmtk_id_to_tf_id[target_bmtk_id]
                edge_exists = tf.greater_equal(target_tf_id, 0)
                target_tf_id = tf.boolean_mask(target_tf_id, edge_exists)
                source_tf_id = tf.boolean_mask(source_tf_id, edge_exists)
                weights_tf = tf.boolean_mask(weights_tf, edge_exists)
                delays_tf = tf.boolean_mask(delays_tf, edge_exists)
                syn_id = tf.boolean_mask(syn_id, edge_exists)

                new_target_tf_id = target_tf_id
                post_indices.append(new_target_tf_id)
                pre_indices.append(source_tf_id)
                weights.append(weights_tf)
                delays.append(delays_tf)
                syn_ids.append(syn_id)

        post_indices = tf.concat(post_indices, axis=0)
        pre_indices = tf.concat(pre_indices, axis=0)
        weights = tf.concat(weights, axis=0)
        delays = tf.concat(delays, axis=0)
        syn_ids = tf.concat(syn_ids, axis=0)

        indices = tf.stack([post_indices, pre_indices], axis=-1)
        indices = tf.cast(indices, dtype=tf.int32)
        weights = tf.cast(weights, dtype=tf.float32)
        delays = tf.cast(delays, dtype=tf.float16)
        syn_ids = tf.cast(syn_ids, dtype=tf.uint8)

        # Sort indices
        indices, weights, delays, syn_ids = sort_indices_tf(indices, weights, delays, syn_ids)

        n_inputs = 17400

        input_populations.append(
            dict(
                n_inputs=n_inputs,
                indices=indices.numpy(),
                weights=weights.numpy(),
                delays=delays.numpy(),
                syn_ids=syn_ids.numpy(),
            )
        )
    
    return input_populations

def load_input_np(
    lgn_input,
    start=0,
    duration=3000,
    dt=1,
    bmtk_id_to_tf_id=None,
    # max_n_receptors=10
):
        
    # Unite the edges information of the LGN and the background noise
    input_edges = [lgn_input['edges']]
       
    input_populations = []    
    for idx, input_population in enumerate(input_edges):
        post_indices = []
        pre_indices = []
        weights = []
        delays = []
        syn_ids = []

        for edge in input_population:  # input_population[1]:
            # Identify the which of the 10 types of inputs we have
            # r = edge["params"]["receptor_type"] - 1
            # r takes values within 0 - 9
            target_bmtk_id = np.array(edge["target"], dtype=np.int32)
            source_tf_id = np.array(edge["source"], dtype=np.int32)
            weights_tf = np.array(edge["params"]["weight"], dtype=np.float32)
            
            # delays_tf = np.array(edge["params"]["delay"], dtype=np.float16)
            # syn_id = np.array(edge["params"]["syn_id"], dtype=np.uint8)
            delays_tf = (np.zeros_like(weights_tf) + edge["params"]["delay"]).astype(np.float16)
            syn_id = (np.zeros_like(weights_tf) + edge["params"]["syn_id"]).astype(np.uint8)

            if bmtk_id_to_tf_id is not None:
                # check if the given edges exist in our model
                # (notice that only the target must exist since the source is within the LGN module)
                # This means that source index is within 0-17400
                target_tf_id = bmtk_id_to_tf_id[target_bmtk_id]
                edge_exists = target_tf_id >= 0
                target_tf_id = target_tf_id[edge_exists]
                source_tf_id = source_tf_id[edge_exists]
                weights_tf = weights_tf[edge_exists]
                delays_tf = delays_tf[edge_exists]
                syn_id = syn_id[edge_exists]

                # we multiply by 10 the indices and add r to identify the receptor_type easily:
                # if target id is divisible by 10 the receptor_type is 0,
                # if it is rest is 1 by dividing by 10 then its receptor type is 1, and so on...
                # extend acts by extending the list with the given object
                # new_target_tf_id = target_tf_id * max_n_receptors + r
                new_target_tf_id = target_tf_id
                post_indices.extend(new_target_tf_id)
                pre_indices.extend(source_tf_id)
                weights.extend(weights_tf)
                delays.extend(delays_tf)
                syn_ids.extend(syn_id)

        # first column are the post indices and second column the pre indices
        indices = np.stack([post_indices, pre_indices], axis=-1).astype(np.int32)
        weights = np.array(weights, dtype=np.float32)
        delays = np.array(delays, dtype=np.float16) # np.concatenate(delays)
        syn_ids = np.array(syn_ids, dtype=np.uint8)
        # sort indices by considering first all the sources of target node 0, then all of node 1, ...
        # indices, weights, delays = sort_input_indices(indices, weights, delays)

        # indices, weights = sort_indices(indices, weights)
        indices, weights, delays, syn_ids = sort_indices(indices, weights, delays, syn_ids)

        n_inputs = 17400

        input_populations.append(
            dict(
                n_inputs=n_inputs,
                indices=indices,
                weights=weights,
                delays=delays,
                syn_ids=syn_ids,
            )
        )


    return input_populations


# Here we load the input from the LGN units and the background noise
# @profile
def load_input(
    lgn_path="GLIF_network/lgn_input_dat.pkl",
    bkg_path="GLIF_network/bkg_input_dat.pkl",
    start=0,
    duration=3000,
    dt=1,
    bmtk_id_to_tf_id=None,
    # max_n_receptors=10
):
    # LOAD THE LGN INPUT
    if not os.path.exists(lgn_path):
        print("Creating lgn_input_dat.pkl file...")
        # Process LGN input network
        lgn_input = create_network_dat(data_dir='GLIF_network/network', source='lgn', target='v1', 
                                        output_file=lgn_path, save_pkl=True)
        print("Done.")
    else:
        with open(lgn_path, "rb") as f:
            lgn_input = pkl.load(f)


    t0 = time()
    input_populations = load_input_np(lgn_input, bmtk_id_to_tf_id=bmtk_id_to_tf_id)
    print('Time to load input numpy: ', time() - t0)

    t0 = time()
    input_populations = load_input_tf(lgn_input, bmtk_id_to_tf_id=bmtk_id_to_tf_id)
    print('Time to load input tensorflow: ', time() - t0)


    # LOAD THE BACKGROUND INPUT
    if not os.path.exists(bkg_path):
        print("Creating bkg_input_dat.pkl file...")
        # Process LGN input network
        bkg_input = create_network_dat(data_dir='GLIF_network/network', source='bkg', target='v1', 
                                        output_file=bkg_path, save_pkl=True)
        print("Done.")
    else:
        with open(bkg_path, "rb") as f:
            bkg_input = pkl.load(f)
        
    # Unite the edges information of the LGN and the background noise
    input_edges = [lgn_input['edges'], bkg_input['edges']]
       
    input_populations = []
    # for idx, input_population in enumerate(input_edges):
    #     post_indices = []
    #     pre_indices = []
    #     weights = []
    #     delays = []
    #     syn_ids = []

    #     for idx, edge in enumerate(input_population):
    #         print(idx)
    #         target_bmtk_id = tf.constant(edge["target"], dtype=tf.int32)
    #         source_tf_id = tf.constant(edge["source"], dtype=tf.int32)
    #         weights_tf = tf.constant(edge["params"]["weight"], dtype=tf.float32)
    #         # delays_tf = tf.constant(edge["params"]["delay"], dtype=tf.float16)
    #         # syn_id = tf.constant(edge["params"]["syn_id"], dtype=tf.uint8)

    #         # delays_tf = (np.zeros_like(weights_tf) + edge["params"]["delay"]).astype(np.float16)
    # #         syn_id = (np.zeros_like(weights_tf) + edge["params"]["syn_id"]).astype(np.uint8)

            
    #         print(delays_tf)

    #         if bmtk_id_to_tf_id is not None:
    #             target_tf_id = bmtk_id_to_tf_id[target_bmtk_id]
    #             edge_exists = tf.greater_equal(target_tf_id, 0)
    #             target_tf_id = tf.boolean_mask(target_tf_id, edge_exists)
    #             source_tf_id = tf.boolean_mask(source_tf_id, edge_exists)
    #             weights_tf = tf.boolean_mask(weights_tf, edge_exists)
    #             delays_tf = tf.boolean_mask(delays_tf, edge_exists)
    #             syn_id = tf.boolean_mask(syn_id, edge_exists)

    #             new_target_tf_id = target_tf_id
    #             post_indices.append(new_target_tf_id)
    #             pre_indices.append(source_tf_id)
    #             weights.append(weights_tf)
    #             delays.append(delays_tf)
    #             syn_ids.append(syn_id)

    #     post_indices = tf.concat(post_indices, axis=0)
    #     pre_indices = tf.concat(pre_indices, axis=0)
    #     weights = tf.concat(weights, axis=0)
    #     delays = tf.concat(delays, axis=0)
    #     syn_ids = tf.concat(syn_ids, axis=0)

    #     indices = tf.stack([post_indices, pre_indices], axis=-1)
    #     indices = tf.cast(indices, dtype=tf.int32)
    #     weights = tf.cast(weights, dtype=tf.float32)
    #     delays = tf.cast(delays, dtype=tf.float16)
    #     syn_ids = tf.cast(syn_ids, dtype=tf.uint8)

    #     # Sort indices
    #     sorted_indices = tf.argsort(indices[:, 0])
    #     indices = tf.gather(indices, sorted_indices)
    #     weights = tf.gather(weights, sorted_indices)
    #     delays = tf.gather(delays, sorted_indices)
    #     syn_ids = tf.gather(syn_ids, sorted_indices)

    #     input_populations.append((indices, weights, delays, syn_ids))
    
    for idx, input_population in enumerate(input_edges):
        post_indices = []
        pre_indices = []
        weights = []
        delays = []
        syn_ids = []

        for edge in input_population:  # input_population[1]:
            # Identify the which of the 10 types of inputs we have
            # r = edge["params"]["receptor_type"] - 1
            # r takes values within 0 - 9
            target_bmtk_id = np.array(edge["target"], dtype=np.int32)
            source_tf_id = np.array(edge["source"], dtype=np.int32)
            weights_tf = np.array(edge["params"]["weight"], dtype=np.float32)
            
            # delays_tf = np.array(edge["params"]["delay"], dtype=np.float16)
            # syn_id = np.array(edge["params"]["syn_id"], dtype=np.uint8)
            delays_tf = (np.zeros_like(weights_tf) + edge["params"]["delay"]).astype(np.float16)
            syn_id = (np.zeros_like(weights_tf) + edge["params"]["syn_id"]).astype(np.uint8)

            if bmtk_id_to_tf_id is not None:
                # check if the given edges exist in our model
                # (notice that only the target must exist since the source is within the LGN module)
                # This means that source index is within 0-17400
                target_tf_id = bmtk_id_to_tf_id[target_bmtk_id]
                edge_exists = target_tf_id >= 0
                target_tf_id = target_tf_id[edge_exists]
                source_tf_id = source_tf_id[edge_exists]
                weights_tf = weights_tf[edge_exists]
                delays_tf = delays_tf[edge_exists]
                syn_id = syn_id[edge_exists]

                # we multiply by 10 the indices and add r to identify the receptor_type easily:
                # if target id is divisible by 10 the receptor_type is 0,
                # if it is rest is 1 by dividing by 10 then its receptor type is 1, and so on...
                # extend acts by extending the list with the given object
                # new_target_tf_id = target_tf_id * max_n_receptors + r
                new_target_tf_id = target_tf_id
                post_indices.extend(new_target_tf_id)
                pre_indices.extend(source_tf_id)
                weights.extend(weights_tf)
                delays.extend(delays_tf)
                syn_ids.extend(syn_id)

        # first column are the post indices and second column the pre indices
        indices = np.stack([post_indices, pre_indices], axis=-1).astype(np.int32)
        weights = np.array(weights, dtype=np.float32)
        delays = np.array(delays, dtype=np.float16) # np.concatenate(delays)
        syn_ids = np.array(syn_ids, dtype=np.uint8)
        # sort indices by considering first all the sources of target node 0, then all of node 1, ...
        # indices, weights, delays = sort_input_indices(indices, weights, delays)

        # indices, weights = sort_indices(indices, weights)
        indices, weights, delays, syn_ids = sort_indices(indices, weights, delays, syn_ids)

        if idx == 0:
            # we load the LGN nodes and their positions
            lgn_nodes_h5_file = h5py.File("GLIF_network/network/lgn_nodes.h5", "r")
            n_inputs = len(lgn_nodes_h5_file["nodes"]["lgn"]["node_id"])
        elif idx == 1:
            # we load the background nodes and their positions
            bkg_nodes_h5_file = h5py.File("GLIF_network/network/bkg_nodes.h5", "r")
            n_inputs = len(bkg_nodes_h5_file["nodes"]["bkg"]["node_id"])

        input_populations.append(
            dict(
                n_inputs=n_inputs,
                indices=indices,
                weights=weights,
                delays=delays,
                syn_ids=syn_ids,
            )
        )

        # n_neurons = len(input_population[0]['ids'])  # 17400
        # spikes = np.zeros((int(duration / dt), n_neurons))
        # # now we save the spikes of the input population
        # for i, sp in zip(input_population[0]['ids'], input_population[0]['spikes']):
        #     # consider only the spikes within the period we are taking
        #     sp = sp[np.logical_and(start < sp, sp < start + duration)] - start
        #     sp = (sp / dt).astype(np.int)
        #     for s in sp:
        #         spikes[s, i] += 1

        # input_populations.append(dict(n_inputs=n_neurons, indices=indices.astype(
        #     np.int64), weights=weights, delays=delays, spikes=spikes))
    return input_populations


def reduce_input_population(input_population, new_n_input, seed=3000):
    rd = np.random.RandomState(seed=seed)

    in_ind = input_population["indices"]
    in_weights = input_population["weights"]
    in_delays = input_population["delays"]
    in_syn_ids = input_population["syn_ids"]

    # we take input_population['n_inputs'] neurons from a list of new_n_input with replace,
    # which means that in the end there can be less than new_n_input neurons of the LGN,
    # but they are randonmly selected
    assignment = rd.choice(np.arange(new_n_input, dtype=np.int32), size=input_population["n_inputs"], replace=True)

    weight_dict = dict()
    delays_dict = dict()
    syn_ids_dict = dict()
    # go through all the asignment selection made
    for input_neuron in range(input_population["n_inputs"]):
        assigned_neuron = assignment[input_neuron]
        # consider that neurons connected to the input_neuron
        sel = in_ind[:, 1] == input_neuron
        # keep that neurons connected to the input_neuron
        sel_post_inds = in_ind[sel, 0]
        sel_weights = in_weights[sel]
        sel_delays = in_delays[sel]
        sel_syn_ids = in_syn_ids[sel]
        for post_ind, weight, delay, syn_id in zip(sel_post_inds, sel_weights, sel_delays, sel_syn_ids):
            # tuple with the indices of the post model neuron and the pre LGN neuron
            t_inds = post_ind, assigned_neuron
            if t_inds not in weight_dict.keys():  # in case the key hasnt been already created
                weight_dict[t_inds] = 0.0
            # in case a LGN unit connection is repeated we consider that the weights are add up
            weight_dict[t_inds] += weight
            delays_dict[t_inds] = delay
            syn_ids_dict[t_inds] = syn_id

    n_synapses = len(weight_dict)
    # we now save the synapses in arrays of indices and weights
    new_in_ind = np.zeros((n_synapses, 2), dtype=np.int32)
    new_in_weights = np.zeros(n_synapses, dtype=np.float32)
    new_in_delays = np.zeros(n_synapses, dtype=np.float16)
    new_in_syn_ids = np.zeros(n_synapses, dtype=np.uint8)
    for i, (t_ind, w) in enumerate(weight_dict.items()):
        new_in_ind[i] = t_ind
        new_in_weights[i] = w
        new_in_delays[i] = delays_dict[t_ind]
        new_in_syn_ids[i] = syn_ids_dict[t_ind]

    # new_in_ind, new_in_weights, new_in_delays = sort_input_indices(
    #     new_in_ind, new_in_weights, new_in_delays
    # )

    new_in_ind, new_in_weights, new_in_delays, new_in_syn_ids = sort_indices(
        new_in_ind, new_in_weights, new_in_delays, new_in_syn_ids
    )
    
    new_input_population = dict(
        n_inputs=new_n_input,
        indices=new_in_ind,
        weights=new_in_weights,
        delays=new_in_delays,
        syn_ids=new_in_syn_ids,
        # spikes=None,
    )

    return new_input_population

  
# @profile
def load_v1(flags, n_neurons):

    # Initialize the network 
    network = load_network(
        path=os.path.join(flags.data_dir, "network_dat.pkl"),
        h5_path=os.path.join(flags.data_dir, "network/v1_nodes.h5"),
        core_only=flags.core_only,
        n_neurons=n_neurons,
        seed=flags.seed,
        connected_selection=flags.connected_selection,
    )

    ###### Select random l5e neurons for tracking output #########
    df = pd.read_csv(os.path.join(
        flags.data_dir, "network/v1_node_types.csv"), delimiter=" ")
    
    l5e_types_indices = []
    for a in df.iterrows():
        if a[1]["pop_name"].startswith("e5"):
            l5e_types_indices.append(a[0])
    l5e_types_indices = np.array(l5e_types_indices)
    l5e_neuron_sel = np.zeros(network["n_nodes"], np.bool_)
    for l5e_type_index in l5e_types_indices:
        is_l5_type = network["node_type_ids"] == l5e_type_index
        l5e_neuron_sel = np.logical_or(l5e_neuron_sel, is_l5_type)
    network["l5e_types"] = l5e_types_indices
    network["l5e_neuron_sel"] = l5e_neuron_sel
    print(f"> Number of L5e Neurons: {np.sum(l5e_neuron_sel)}")

    # assert that you have enough l5 neurons for all the outputs and then choose n_output * neurons_per_output random neurons
    # assert np.sum(l5e_neuron_sel) > n_output * neurons_per_output
    rd = np.random.RandomState(seed=flags.seed)
    l5e_neuron_indices = np.where(l5e_neuron_sel)[0]
    readout_neurons = rd.choice(
        l5e_neuron_indices, size=flags.n_output * flags.neurons_per_output, replace=False
    )
    readout_neurons = readout_neurons.reshape((flags.n_output, flags.neurons_per_output))
    network["readout_neuron_ids"] = readout_neurons
    ##########################################

    ### Load the LGN and BKG input of the model
    print('Loading input...')
    t0 = time()
    inputs = load_input(
        start=1000, 
        duration=1000,
        dt=1,
        lgn_path=os.path.join(flags.data_dir, "lgn_input_dat.pkl"),
        bkg_path=os.path.join(flags.data_dir, "bkg_input_dat.pkl"),
        bmtk_id_to_tf_id=network["bmtk_id_to_tf_id"],
    )
    print('Done. Took %.2f seconds' % (time() - t0))

    lgn_input = inputs[0]
    bkg_input = inputs[1]

    # If required reduce the number of LGN inputs
    if flags.n_input != 17400:
        lgn_input = reduce_input_population(lgn_input, flags.n_input, seed=flags.seed)

    return network, lgn_input, bkg_input


# If the model already exist we can load it, or if it does not just save it for future occasions
def cached_load_v1(flags, n_neurons):
    store = False
    network, lgn_input, bkg_input = None, None, None
    flag_str = (f"neurons_{n_neurons}_n_input_{flags.n_input}_s{flags.seed}_c{flags.core_only}_con{flags.connected_selection}")
    file_dir = os.path.split(__file__)[0]
    cache_path = os.path.join(file_dir, f".cache/V1_network_{flag_str}.pkl")
    print(f"> Looking for cached V1 model in {cache_path}")
    
    if os.path.exists(cache_path):
        try:
            with open(cache_path, "rb") as f:
                network, lgn_input, bkg_input = pkl.load(f)
                print(f"> Sucessfully restored V1 model from {cache_path}")
        except Exception as e:
            print(e)
            store = True
    else:
        store = True

    if lgn_input is None or network is None or bkg_input is None:
        network, lgn_input, bkg_input = load_v1(flags=flags, n_neurons=n_neurons)

    if store:
        os.makedirs(os.path.join(file_dir, ".cache"), exist_ok=True)
        with open(cache_path, "wb") as f:
            pkl.dump((network, lgn_input, bkg_input), f)
        print(f"> Cached V1 model in {cache_path}")

    return network, lgn_input, bkg_input

