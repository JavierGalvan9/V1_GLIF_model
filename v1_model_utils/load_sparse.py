import os
import pickle as pkl
import json

import h5py
import numpy as np
import pandas as pd
from numba import njit
from time import time

@njit
def sort_indices(indices, weights, delays, tau_syn_weights_array):
    max_ind = np.max(indices) + 1
    # sort indices by considering first all the targets of node 0, then all of node 1, ...
    q = indices[:, 0] * max_ind + indices[:, 1]
    sorted_ind = np.argsort(q)
    
    return indices[sorted_ind], weights[sorted_ind], delays[sorted_ind], tau_syn_weights_array[sorted_ind, :]

# @njit
# def sort_input_indices(indices, weights, delays):
#     max_ind = np.max(indices) + 1
#     # sort indices by considering first all the targets of node 0, then all of node 1, ...
#     q = indices[:, 0] * max_ind + indices[:, 1]
#     sorted_ind = np.argsort(q)
    
#     return indices[sorted_ind], weights[sorted_ind], delays[sorted_ind]


# def create_network_dat(data_dir='GLIF_network/network'):
#     # Extract the network data into a pickle file from the SONATA files
#     new_network_dat = {"nodes": [], "edges": []}

#     # Load the nodes file and create a dictionary with the node ids and the node parameters
#     nodes_h5_path = os.path.join(data_dir, "v1_nodes.h5")
#     node_types_df = pd.read_csv(os.path.join(data_dir, "v1_node_types.csv"), delimiter=" ")
#     cell_models_path = os.path.join(data_dir, "components", "cell_models")

#     for node_type_id in node_types_df["node_type_id"]:
#         with h5py.File(nodes_h5_path, "r") as nodes_h5_file:
#             v1_node_ids = np.array(nodes_h5_file["nodes"]["v1"]["node_id"])
#             v1_node_type_ids = np.array(nodes_h5_file["nodes"]["v1"]["node_type_id"])

#         mask = v1_node_type_ids == node_type_id
#         new_pop_dict = {"ids": v1_node_ids[mask].astype(np.uint32)}

#         # open json file with node parameters
#         node_params_file = os.path.join(cell_models_path, f"{node_type_id}_glif_lif_asc_config.json")
#         with open(node_params_file) as f:
#             cell_model_dict = json.load(f)

#         # rename V_m key in dictionary to V_reset
#         cell_model_dict["V_reset"] = cell_model_dict.pop("V_m")
#         # rename asc_decay key in dictionary to k
#         cell_model_dict["k"] = cell_model_dict.pop("asc_decay")
#         new_pop_dict["params"] = cell_model_dict

#         new_network_dat["nodes"].append(new_pop_dict)

#     # Load the edges file and create a dictionary with the edge ids and the edge parameters
#     edges_h5_path = os.path.join(data_dir, "v1_v1_edges.h5")
#     edges_type_df = pd.read_csv(os.path.join(data_dir, "v1_v1_edge_types.csv"), delimiter=" ")
#     synaptic_models_path = os.path.join(data_dir, "components", "synaptic_models")
#     basis_function_weights_df = pd.read_csv('GLIF_network/synaptic_data/basis_function_weights.csv', index_col=0)

#     for _, (edge_type_id, edge_model, edge_delay, edge_dyn_params) in edges_type_df[
#         ["edge_type_id", "model_template", "delay", "dynamics_params"]].iterrows():

#         with h5py.File(edges_h5_path, "r") as edges_h5_file:
#             edge_type_ids = np.array(edges_h5_file["edges"]["v1_to_v1"]["edge_type_id"])
#             source_node_ids = np.array(edges_h5_file["edges"]["v1_to_v1"]["source_node_id"])
#             target_node_ids = np.array(edges_h5_file["edges"]["v1_to_v1"]["target_node_id"])
#             syn_weights = np.array(edges_h5_file["edges"]["v1_to_v1"]["0"]["syn_weight"])

#         mask = edge_type_ids == edge_type_id
#         synaptic_model = edge_dyn_params.split('.')[0]
#         tau_syn_weights = basis_function_weights_df.loc[synaptic_model].values

#         edge_params_file = os.path.join(synaptic_models_path, edge_dyn_params)
#         with open(edge_params_file) as f:
#             synaptic_model_dict = json.load(f)
#             receptor_type = synaptic_model_dict["receptor_type"]

#         new_pop_dict = {
#             "source": source_node_ids[mask],
#             "target": target_node_ids[mask],
#             "params": {
#                 "model": edge_model,
#                 "delay": edge_delay,
#                 "weight": syn_weights[mask],
#                 "synaptic_model": synaptic_model,
#                 "receptor_type": receptor_type,
#                 "tau_syn_weights": tau_syn_weights,
#             },
#         }

#         new_network_dat["edges"].append(new_pop_dict)

#     with open("GLIF_network/network_dat.pkl", "wb") as f:
#         pkl.dump(new_network_dat, f)


# def create_input_dat(data_dir='GLIF_network/network'):
#     # Extract the input (LGN and BKG) data into a pickle file from the SONATA files
#     new_input_network = []

#     ### LGN input network ###
#     lgn_input_network = []
#     edges_h5_path = os.path.join(data_dir, "lgn_v1_edges.h5")
#     edges_type_df = pd.read_csv(os.path.join(data_dir, "lgn_v1_edge_types.csv"), delimiter=" ")
#     synaptic_models_path = os.path.join(data_dir, 'components', "synaptic_models")
#     basis_function_weights_df = pd.read_csv('GLIF_network/synaptic_data/basis_function_weights.csv', index_col=0)

#     for _, (edge_type_id, edge_model, edge_delay, edge_dyn_params) in edges_type_df[
#         ["edge_type_id", "model_template", "delay", "dynamics_params"]].iterrows():

#         with h5py.File(edges_h5_path, "r") as edges_h5_file:
#             edge_type_ids = np.array(edges_h5_file["edges"]["lgn_to_v1"]["edge_type_id"])
#             source_node_ids = np.array(edges_h5_file["edges"]["lgn_to_v1"]["source_node_id"])
#             target_node_ids = np.array(edges_h5_file["edges"]["lgn_to_v1"]["target_node_id"])
#             syn_weights = np.array(edges_h5_file["edges"]["lgn_to_v1"]["0"]["syn_weight"])

#         mask = edge_type_ids == edge_type_id
#         synaptic_model = edge_dyn_params.split('.')[0]
#         tau_syn_weights = basis_function_weights.loc[synaptic_model].values

#         edge_params_file = os.path.join(synaptic_models_path, edge_dyn_params)
#         with open(edge_params_file) as f:
#             synaptic_model_dict = json.load(f)
#             receptor_type = synaptic_model_dict["receptor_type"]
   
#         new_pop_dict = {
#             "source": source_node_ids[mask],
#             "target": target_node_ids[mask],
#             "params": {
#                 "model": edge_model,
#                 "delay": edge_delay,
#                 "weight": syn_weights[mask],
#                 "synaptic_model": synaptic_model,
#                 "receptor_type": receptor_type,
#                 "tau_syn_weights": tau_syn_weights,
#             },
#         }

#         lgn_input_network.append(new_pop_dict)

#     new_input_network.append(lgn_input_network)

#     ### BKG input network ###
#     bkg_input_network = []
#     edges_h5_path = os.path.join(data_dir, "bkg_v1_edges.h5")
#     edges_type_df = pd.read_csv(os.path.join(data_dir, "bkg_v1_edge_types.csv"), delimiter=" ")
#     synaptic_models_path = os.path.join(data_dir, 'components', "synaptic_models")
#     basis_function_weights_df = pd.read_csv('GLIF_network/synaptic_data/basis_function_weights.csv', index_col=0)

#     for _, (edge_type_id, edge_model, edge_delay, edge_dyn_params) in edges_type_df[
#         ["edge_type_id", "model_template", "delay", "dynamics_params"]].iterrows():

#         with h5py.File(edges_h5_path, "r") as edges_h5_file:
#             edge_type_ids = np.array(edges_h5_file["edges"]["bkg_to_v1"]["edge_type_id"])
#             source_node_ids = np.array(edges_h5_file["edges"]["bkg_to_v1"]["source_node_id"])
#             target_node_ids = np.array(edges_h5_file["edges"]["bkg_to_v1"]["target_node_id"])
#             syn_weights = np.array(edges_h5_file["edges"]["bkg_to_v1"]["0"]["syn_weight"])

#         mask = edge_type_ids == edge_type_id
#         synaptic_model = edge_dyn_params.split('.')[0]
#         tau_syn_weights = basis_function_weights.loc[synaptic_model].values

#         edge_params_file = os.path.join(synaptic_models_path, edge_dyn_params)
#         with open(edge_params_file) as f:
#             synaptic_model_dict = json.load(f)
#             receptor_type = synaptic_model_dict["receptor_type"]
   
#         new_pop_dict = {
#             "source": source_node_ids[mask],
#             "target": target_node_ids[mask],
#             "params": {
#                 "model": edge_model,
#                 "delay": edge_delay,
#                 "weight": syn_weights[mask],
#                 "synaptic_model": synaptic_model,
#                 "receptor_type": receptor_type,
#                 "tau_syn_weights": tau_syn_weights,
#             },
#         }

#         bkg_input_network.append(new_pop_dict)

#     new_input_network.append(bkg_input_network)

#     with open("GLIF_network/input_dat.pkl", "wb") as file:
#         pkl.dump(new_input_network, file)



def create_network_dat(data_dir='GLIF_network/network', source='v1', target='v1', 
                        output_file='GLIF_network/network_dat.pkl', save_dat=True):
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
            source_node_ids = np.array(nodes_h5_file["nodes"][source]["node_id"])
            source_node_type_ids = np.array(nodes_h5_file["nodes"][source]["node_type_id"])

        for node_type_id in node_types_df["node_type_id"]:
            mask = source_node_type_ids == node_type_id
            new_pop_dict = {"ids": source_node_ids[mask].astype(np.uint32)}

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

    with h5py.File(edges_h5_path, "r") as edges_h5_file:
        source_to_target = f"{source}_to_{target}"
        edge_type_ids = np.array(edges_h5_file["edges"][source_to_target]["edge_type_id"])
        source_node_ids = np.array(edges_h5_file["edges"][source_to_target]["source_node_id"])
        target_node_ids = np.array(edges_h5_file["edges"][source_to_target]["target_node_id"])
        if source != 'bkg':
            syn_weights = np.array(edges_h5_file["edges"][source_to_target]["0"]["syn_weight"])

    if source == 'bkg':
        syn_weights = np.zeros(len(source_node_ids))

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
            },
        }

        new_network_dat["edges"].append(new_pop_dict)

    if save_dat:    
        with open(output_file, "wb") as f:
            pkl.dump(new_network_dat, f)

    return new_network_dat


def load_network(
    path="GLIF_network/network_dat.pkl",
    h5_path="GLIF_network/network/v1_nodes.h5",
    core_only=True,
    n_neurons=296991,
    seed=3000,
    connected_selection=False):

    rd = np.random.RandomState(seed=seed)

    # if network path does not exist, create it
    if not os.path.exists(path):
        print("Creating network_dat.pkl file...")
        # Process V1 network
        create_network_dat(data_dir='GLIF_network/network', source='v1', target='v1',
                            output_file='GLIF_network/network_dat.pkl')
        print("Done.")

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

    n_nodes = sum([len(a["ids"]) for a in d["nodes"]])  # 296991 total neurons
    n_edges = sum([len(a["source"]) for a in d["edges"]])  # 150655767 total edges
    # max_delay = max([a['params']['delay'] for a in d['edges']])

    bmtk_id_to_tf_id = np.arange(n_nodes)
    tf_id_to_bmtk_id = np.arange(n_nodes)

    edges = d["edges"]
    h5_file = h5py.File(h5_path, "r")
    # This file gives us the:
    # '0': coordinates of each point (and other information we are not using)
    # 'node_group_id': all nodes have the index 0
    # 'node_group_index': same as node_id
    # 'node_id': bmtk index of each node (node_id[0]=0, node_id[1]=1, ...)
    # 'node_type_id': 518290966, 539742766,... for each node
    assert np.diff(h5_file["nodes"]["v1"]["node_id"]).var() < 1e-12
    x = np.array(h5_file["nodes"]["v1"]["0"]["x"]) # horizontal axis
    y = np.array(h5_file["nodes"]["v1"]["0"]["y"]) # depth
    z = np.array(h5_file["nodes"]["v1"]["0"]["z"]) # horizontal axis
    tuning_angle = np.array(h5_file['nodes']['v1']['0']['tuning_angle'])
    node_type_id = np.array(h5_file['nodes']['v1']['node_type_id'])
    r = np.sqrt(x**2 + z**2)  # the maximum radius is 845

    # Check if the number of neurons is not too large
    if n_neurons > 296991:
        raise ValueError("There are only 296991 neurons in the network")
    
    ### Select neurons to keep in the network
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
    
    elif n_neurons > 0 and n_neurons <= 296991: # this condition samples neurons from the whole V1
        legit_neurons = np.arange(n_nodes)
        take_inds = rd.choice(legit_neurons, size=n_neurons, replace=False)
        sel = np.zeros(n_nodes, np.bool_)
        sel[take_inds] = True
    
    else: # if no condition is met, all neurons are selected
        sel = np.ones(n_nodes, np.bool_)

    # Update the number of neurons
    n_nodes = np.sum(sel)
    # tf idx '0' corresponds to 'tf_id_to_bmtk_id[0]' bmtk idx
    tf_id_to_bmtk_id = tf_id_to_bmtk_id[sel]
    bmtk_id_to_tf_id = np.zeros_like(bmtk_id_to_tf_id) - 1
    for tf_id, bmtk_id in enumerate(tf_id_to_bmtk_id):
        bmtk_id_to_tf_id[bmtk_id] = tf_id

    # bmtk idx '0' corresponds to 'bmtk_id_to_tf_id[0]' tf idx which can be '-1' in case
    # the bmtk node is not in the tensorflow selection or another value in case it belongs the selection
    x = x[sel]
    y = y[sel]
    z = z[sel]
    tuning_angle = tuning_angle[sel]
    node_type_id = node_type_id[sel]

    # from all the model edges, lets see how many correspond to the selected nodes
    n_edges = 0
    for edge in edges:
        target_tf_ids = bmtk_id_to_tf_id[np.array(edge["target"])]
        source_tf_ids = bmtk_id_to_tf_id[np.array(edge["source"])]
        edge_exists = np.logical_and(target_tf_ids >= 0, source_tf_ids >= 0)
        n_edges += np.sum(edge_exists)

    print(f"> Number of Neurons: {n_nodes}")
    print(f"> Number of Synapses: {n_edges}")

    # Calculate the maximum number of receptors (synaptic time constants)
    # max_n_receptors = 10
    # max_n_receptors = 0
    # for node_type in d["nodes"]:
    #     n_receptors = len(node_type["params"]["tau_syn"])
    #     max_n_receptors = max(max_n_receptors, n_receptors)
    # print("Maximum number of receptors:", max_n_receptors)

    n_syn_basis = 5
    # get the n_syn_basis from the tau_basis.npy file shape

    n_syn_basis = len(np.load("GLIF_network/synaptic_data/tau_basis.npy"))
    print("Number of synaptic basis:", n_syn_basis)

    # Save in a dictionary the properties of each of the 111 node types
    n_node_types = len(d["nodes"])
    node_params = dict(
        V_th=np.zeros(n_node_types, np.float32),
        g=np.zeros(n_node_types, np.float32),
        E_L=np.zeros(n_node_types, np.float32),
        k=np.zeros((n_node_types, 2), np.float32),
        C_m=np.zeros(n_node_types, np.float32),
        V_reset=np.zeros(n_node_types, np.float32),
        # tau_syn=np.zeros((n_node_types, max_n_receptors), np.float32),  # 10 is the maximum number of different synaptic types a neuron has
        t_ref=np.zeros(n_node_types, np.float32),
        asc_amps=np.zeros((n_node_types, 2), np.float32),
    )

    # give every selected node of a given node type an index according to tf ids
    node_type_ids = np.zeros(n_nodes, np.int64)
    for i, node_type in enumerate(d["nodes"]):
        # get ALL the nodes of the given node type
        tf_ids = bmtk_id_to_tf_id[np.array(node_type["ids"])]
        # choose only those that belong to our model
        tf_ids = tf_ids[tf_ids >= 0]
        # assign them all the same id (which does not relate with the neuron type)
        node_type_ids[tf_ids] = i
        for k, v in node_params.items():
            # save in a dict the information of the nodes
            v[i] = node_type["params"][k]
            # if k == "tau_syn":
            #     n_receptors = len(node_type["params"][k])
            #     v[i, : n_receptors] = node_type["params"][k]
            # else:
            #     v[i] = node_type["params"][k]

    # each node has at most 10 different inputs with different tau_syn each
    # dense_shape = (max_n_receptors * n_nodes, n_nodes)
    # dense_shape = (n_syn_basis * n_nodes, n_nodes)
    dense_shape = (n_nodes, n_nodes)
    
    indices = np.zeros((n_edges, 2), dtype=np.int64)
    weights = np.zeros(n_edges, np.float32)
    delays = np.zeros(n_edges, np.float32)
    tau_syn_weights_array = np.zeros((n_edges, n_syn_basis), dtype=np.float32)

    current_edge = 0
    for edge in edges:
        # Identify which of the 10 types of inputs we have
        # r = edge["params"]["receptor_type"] - 1
        # r takes values within 0 - 9
        target_tf_ids = bmtk_id_to_tf_id[np.array(edge["target"])]
        source_tf_ids = bmtk_id_to_tf_id[np.array(edge["source"])]
        edge_exists = np.logical_and(target_tf_ids >= 0, source_tf_ids >= 0)
        # select the edges within our model
        target_tf_ids = target_tf_ids[edge_exists]
        source_tf_ids = source_tf_ids[edge_exists]
        weights_tf = edge["params"]["weight"][edge_exists]
        tau_syn_weights = edge["params"]["tau_syn_weights"]
        # all the edges of a given type have the same delay
        delays_tf = edge["params"]["delay"]
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
        tau_syn_weights_array[current_edge: current_edge + n_new_edge, :] = tau_syn_weights
        current_edge += n_new_edge
    # sort indices by considering first all the targets of node 0, then all of node 1, ...
    indices, weights, delays, tau_syn_weights_array = sort_indices(indices, weights, delays, tau_syn_weights_array)

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
                      tau_syn_weights_array=tau_syn_weights_array,
                      dense_shape=dense_shape),
        tf_id_to_bmtk_id=tf_id_to_bmtk_id,
        bmtk_id_to_tf_id=bmtk_id_to_tf_id,
    )

    return network


# Here we load the input from the LGN units and the background noise
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
                                        output_file=lgn_path, save_dat=True)
        print("Done.")
    else:
        with open(lgn_path, "rb") as f:
            lgn_input = pkl.load(f)

    # LOAD THE BACKGROUND INPUT
    if not os.path.exists(bkg_path):
        print("Creating bkg_input_dat.pkl file...")
        # Process LGN input network
        bkg_input = create_network_dat(data_dir='GLIF_network/network', source='bkg', target='v1', 
                                        output_file=bkg_path, save_dat=True)
        print("Done.")
    else:
        with open(bkg_path, "rb") as f:
            bkg_input = pkl.load(f)
        
    # Unite the edges information of the LGN and the background noise
    input_edges = [lgn_input['edges'], bkg_input['edges']]
       
    input_populations = []
    for idx, input_population in enumerate(input_edges):
        post_indices = []
        pre_indices = []
        weights = []
        delays = []
        tau_syn_weights = []

        for edge in input_population:  # input_population[1]:
            # Identify the which of the 10 types of inputs we have
            # r = edge["params"]["receptor_type"] - 1
            # r takes values within 0 - 9
            target_bmtk_id = np.array(edge["target"])
            source_tf_id = np.array(edge["source"])
            weights_tf = np.array(edge["params"]["weight"])
            delays_tf = np.zeros_like(weights_tf) + edge["params"]["delay"]
            tau_syn_weights_tf = np.zeros((len(delays_tf), len(edge["params"]["tau_syn_weights"])), dtype=np.float32) + edge["params"]["tau_syn_weights"]

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
                tau_syn_weights_tf = tau_syn_weights_tf[edge_exists, :]

                # we multiply by 10 the indices and add r to identify the receptor_type easily:
                # if target id is divisible by 10 the receptor_type is 0,
                # if it is rest is 1 by dividing by 10 then its receptor type is 1, and so on...
                # extend acts by extending the list with the given object
                # new_target_tf_id = target_tf_id * max_n_receptors + r
                new_target_tf_id = target_tf_id
                post_indices.extend(new_target_tf_id)
                pre_indices.extend(source_tf_id)
                weights.extend(weights_tf)
                delays.append(delays_tf)
                tau_syn_weights.append(tau_syn_weights_tf)

        # first column are the post indices and second column the pre indices
        indices = np.stack([post_indices, pre_indices], -1)
        weights = np.array(weights)
        delays = np.concatenate(delays)
        tau_syn_weights_array = np.concatenate(tau_syn_weights)
        # sort indices by considering first all the sources of target node 0, then all of node 1, ...
        # indices, weights, delays = sort_input_indices(indices, weights, delays)
        indices, weights, delays, tau_syn_weights_array = sort_indices(indices, weights, delays, tau_syn_weights_array)

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
                indices=indices.astype(np.int64),
                weights=weights,
                delays=delays,
                tau_syn_weights_array=tau_syn_weights_array,
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

    # we take input_population['n_inputs'] neurons from a list of new_n_input with replace,
    # which means that in the end there can be less than new_n_input neurons of the LGN,
    # but they are randonmly selected
    assignment = rd.choice(np.arange(new_n_input), size=input_population["n_inputs"], replace=True)

    weight_dict = dict()
    delays_dict = dict()
    # go through all the asignment selection made
    for input_neuron in range(input_population["n_inputs"]):
        assigned_neuron = assignment[input_neuron]
        # consider that neurons connected to the input_neuron
        sel = in_ind[:, 1] == input_neuron
        # keep that neurons connected to the input_neuron
        sel_post_inds = in_ind[sel, 0]
        sel_weights = in_weights[sel]
        sel_delays = in_delays[sel]
        for post_ind, weight, delay in zip(sel_post_inds, sel_weights, sel_delays):
            # tuple with the indices of the post model neuron and the pre LGN neuron
            t_inds = post_ind, assigned_neuron
            if t_inds not in weight_dict.keys():  # in case the key hasnt been already created
                weight_dict[t_inds] = 0.0
            # in case a LGN unit connection is repeated we consider that the weights are add up
            weight_dict[t_inds] += weight
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

    new_in_ind, new_in_weights, new_in_delays = sort_input_indices(
        new_in_ind, new_in_weights, new_in_delays
    )
    new_input_population = dict(
        n_inputs=new_n_input,
        indices=new_in_ind,
        weights=new_in_weights,
        delays=new_in_delays,
        # spikes=None,
    )

    return new_input_population


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

    # Define the maximum number of receptors received by every neuron
    # max_n_receptors = int(
    #     network["synapses"]["dense_shape"][0] / network["synapses"]["dense_shape"][1]
    # )  

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

    # Load the LGN and BKG input of the model
    print('Loading input...')
    inputs = load_input(
        start=1000, 
        duration=1000,
        dt=1,
        lgn_path=os.path.join(flags.data_dir, "lgn_input_dat.pkl"),
        bkg_path=os.path.join(flags.data_dir, "bkg_input_dat.pkl"),
        bmtk_id_to_tf_id=network["bmtk_id_to_tf_id"],
        # max_n_receptors=max_n_receptors
    )

    # Extract the LGN and BKG inputs
    lgn_input = inputs[0]
    bkg_input = inputs[1]
    # bkg_weights = np.zeros((network["n_nodes"] * max_n_receptors,), np.float32)
    # bkg_weights = np.zeros((network["n_nodes"],), np.float32)
    # n_syn_basis = len(np.load("GLIF_network/synaptic_data/tau_basis.npy"))
    # print("Number of synaptic basis:", n_syn_basis)
    # # bkg_weights[bkg["indices"][:, 0]] = bkg["weights"]

    # bkg_weights = np.zeros((network["n_nodes"], n_syn_basis,), np.float32)

    # print(bkg_input)
    # print(bkg_weights.shape)
    # print(bkg_input["tau_syn_weights_array"].shape)

    # # count the number of edges for every neuron
    # n_edges_per_neuron = np.zeros(network["n_nodes"], np.int64)
    # for i in range(network["n_nodes"]):
    #     n_edges_per_neuron[i] = np.sum(bkg_input["indices"][:, 0] == i)

    # print('n_edges_per_neuron', n_edges_per_neuron)
    
    # for i in range(n_syn_basis):
    #     print(bkg_input["weights"].shape)
    #     print(bkg_input["tau_syn_weights_array"].shape)
    #     print((bkg_input["weights"] * bkg_input["tau_syn_weights_array"][:, i]).shape)
    #     bkg_input_weights[:, i] = bkg_input["weights"] * bkg_input["tau_syn_weights_array"][:, i]

    # If required reduce the number of LGN inputs
    if flags.n_input != 17400:
        lgn_input = reduce_input_population(lgn_input, flags.n_input, seed=flags.seed)

    return network, lgn_input, bkg_input #, bkg_weights


# If the model already exist we can load it, or if it does not just save it for future occasions
def cached_load_v1(flags, n_neurons):
    store = False
    # lgn_input, network, bkg, bkg_weights = None, None, None, None
    network, lgn_input, bkg_input = None, None, None
    flag_str = (f"neurons_{n_neurons}_n_input_{flags.n_input}_s{flags.seed}_c{flags.core_only}_con{flags.connected_selection}")
    # flag_str += f"_out{flags.n_output}_nper{flags.neurons_per_output}"
    file_dir = os.path.split(__file__)[0]
    cache_path = os.path.join(file_dir, f".cache/V1_network_{flag_str}.pkl")
    print(f"> Looking for cached V1 model in {cache_path}")
    
    if os.path.exists(cache_path):
        try:
            with open(cache_path, "rb") as f:
                # lgn_input, network, bkg, bkg_weights = pkl.load(f)
                network, lgn_input, bkg_input = pkl.load(f)
                print(f"> Sucessfully restored V1 model from {cache_path}")
        except Exception as e:
            print(e)
            store = True
    else:
        store = True
    # if lgn_input is None or network is None or bkg is None or bkg_weights is None:
    #     lgn_input, network, bkg, bkg_weights = load_v1(flags=flags, n_neurons=n_neurons)

    if lgn_input is None or network is None or bkg_input is None:
        network, lgn_input, bkg_input = load_v1(flags=flags, n_neurons=n_neurons)

    if store:
        os.makedirs(os.path.join(file_dir, ".cache"), exist_ok=True)
        with open(cache_path, "wb") as f:
            # pkl.dump((lgn_input, network, bkg, bkg_weights), f)
            pkl.dump((network, lgn_input, bkg_input), f)
        print(f"> Cached V1 model in {cache_path}")

    # return lgn_input, network, bkg, bkg_weights
    return network, lgn_input, bkg_input


# if __name__ == "__main__":
#     load_v1(flags, n_neurons=10000)
