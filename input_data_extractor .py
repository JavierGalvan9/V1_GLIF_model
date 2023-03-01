import json
import os
import pickle as pkl

import h5py
import numpy as np
import pandas as pd

# Create a new_input_network.dat
data_dir = "GLIF_network/network"
new_input_network = []
lgn_input_network = []

lgn_edges_h5_file = h5py.File('GLIF_network/network/lgn_v1_edges.h5', "r")
edge_type_ids = np.array(
    lgn_edges_h5_file["edges"]["lgn_to_v1"]["edge_type_id"])
source_node_ids = np.array(
    lgn_edges_h5_file['edges']['lgn_to_v1']['source_node_id'])
target_node_ids = np.array(
    lgn_edges_h5_file['edges']['lgn_to_v1']['target_node_id'])
syn_weights = np.array(
    lgn_edges_h5_file["edges"]["lgn_to_v1"]["0"]["syn_weight"])

edges_df = pd.read_csv(
    'GLIF_network/network/lgn_v1_edge_types.csv', delimiter=' ')
synaptic_models_path = os.path.join(
    'biorealistic-v1-model', 'tiny', 'components', 'synaptic_models')

for idx, (edge_type_id, edge_model, edge_delay) in edges_df[
    ["edge_type_id", "model_template", "delay"]
].iterrows():
    mask = edge_type_ids == edge_type_id
    new_pop_dict = {
        "source": source_node_ids[mask],
        "target": target_node_ids[mask],
        "params": {
            "model": edge_model,
            "delay": edge_delay,
            "weight": syn_weights[mask],
        },
    }
    # open json file with edge parameters
    dynamic_params_json = edges_df["dynamics_params"].loc[idx]
    with open(os.path.join(synaptic_models_path, dynamic_params_json)) as f:
        synaptic_model_dict = json.load(f)
    new_pop_dict["params"]["receptor_type"] = synaptic_model_dict["receptor_type"]
    lgn_input_network.append(new_pop_dict)

new_input_network.append(lgn_input_network)

bkg_input_network = []

bkg_edges_h5_file = h5py.File('GLIF_network/network/bkg_v1_edges.h5', "r")
edge_type_ids = np.array(
    bkg_edges_h5_file["edges"]["bkg_to_v1"]["edge_type_id"])
source_node_ids = np.array(
    bkg_edges_h5_file['edges']['bkg_to_v1']['source_node_id'])
target_node_ids = np.array(
    bkg_edges_h5_file['edges']['bkg_to_v1']['target_node_id'])
syn_weights = np.array(
    bkg_edges_h5_file["edges"]["bkg_to_v1"]["0"]["syn_weight"])

edges_df = pd.read_csv(
    'GLIF_network/network/bkg_v1_edge_types.csv', delimiter=' ')
synaptic_models_path = os.path.join(
    'biorealistic-v1-model', 'tiny', 'components', 'synaptic_models')

for idx, (edge_type_id, edge_model, edge_delay) in edges_df[
    ["edge_type_id", "model_template", "delay"]
].iterrows():
    mask = edge_type_ids == edge_type_id
    new_pop_dict = {
        "source": source_node_ids[mask],
        "target": target_node_ids[mask],
        "params": {
            "model": edge_model,
            "delay": edge_delay,
            "weight": syn_weights[mask],
        },
    }
    # open json file with edge parameters
    dynamic_params_json = edges_df["dynamics_params"].loc[idx]
    with open(os.path.join(synaptic_models_path, dynamic_params_json)) as f:
        synaptic_model_dict = json.load(f)
    new_pop_dict["params"]["receptor_type"] = synaptic_model_dict["receptor_type"]
    lgn_input_network.append(new_pop_dict)

new_input_network.append(lgn_input_network)

with open(os.path.join(data_dir, "new_input_dat.pkl"), "wb") as file:
    pkl.dump(new_input_network, file)
