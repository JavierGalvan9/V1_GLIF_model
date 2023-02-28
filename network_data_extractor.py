import json
import os
import pickle as pkl

import h5py
import numpy as np
import pandas as pd

# Create a new_network.dat
data_dir = "GLIF_network/network"
new_network = {"nodes": [], "edges": []}

nodes_h5_path = os.path.join(data_dir, "v1_nodes.h5")
nodes_h5_file = h5py.File(nodes_h5_path, "r")
v1_node_ids = np.array(nodes_h5_file["nodes"]["v1"]["node_id"])
v1_node_type_ids = np.array(nodes_h5_file["nodes"]["v1"]["node_type_id"])
nodes_df = pd.read_csv(os.path.join(
    data_dir, "v1_node_types.csv"), delimiter=" ")
cell_models_path = os.path.join(
    'biorealistic-v1-model', 'tiny_shinya', 'components', 'cell_models')
for node_type_id in nodes_df["node_type_id"]:
    mask = v1_node_type_ids == node_type_id
    new_pop_dict = {"ids": v1_node_ids[mask].astype(np.uint32)}
    # open json file with node parameters
    with open(os.path.join(cell_models_path, f"{node_type_id}_glif_lif_asc_config.json")) as f:
        cell_model_dict = json.load(f)
    new_pop_dict["params"] = cell_model_dict
    # rename V_m key in dictionary to V_reset
    new_pop_dict["params"]["V_reset"] = new_pop_dict["params"].pop("V_m")
    # rename asc_decay key in dictionary to k
    new_pop_dict["params"]["k"] = new_pop_dict["params"].pop("asc_decay")
    new_network["nodes"].append(new_pop_dict)


edges_h5_path = os.path.join(data_dir, "v1_v1_edges.h5")
edges_h5_file = h5py.File(edges_h5_path, "r")
edge_type_ids = np.array(edges_h5_file["edges"]["v1_to_v1"]["edge_type_id"])
source_node_ids = np.array(
    edges_h5_file["edges"]["v1_to_v1"]["source_node_id"])
target_node_ids = np.array(
    edges_h5_file["edges"]["v1_to_v1"]["target_node_id"])
syn_weights = np.array(edges_h5_file["edges"]["v1_to_v1"]["0"]["syn_weight"])
edges_df = pd.read_csv(
    "GLIF_network/network/v1_v1_edge_types.csv", delimiter=" ")
synaptic_models_path = os.path.join(
    'biorealistic-v1-model', 'tiny_shinya', 'components', 'synaptic_models')

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
    new_network["edges"].append(new_pop_dict)

with open(os.path.join(data_dir, "new_network_dat.pkl"), "wb") as file:
    pkl.dump(new_network, file)
