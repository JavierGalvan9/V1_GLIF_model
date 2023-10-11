# import json
import os
import pickle as pkl
import json
import h5py
import numpy as np
import pandas as pd

# Create a new_input_network.dat
data_dir = "GLIF_network/network"
new_input_network = []

### LGN input network ###
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

# Correct the naming of the synaptic_models file in the new V1 version
edges_df['dynamics_params'] = edges_df['dynamics_params'].str.replace('lgn', 'exc')
synaptic_models_path = os.path.join(data_dir, 'components', "synaptic_models")
    # 'biorealistic-v1-model', 'tiny', 'components', 'synaptic_models')

path=os.path.join(data_dir, 'basis_function_weights.csv')
basis_function_weights = pd.read_csv(path, index_col=0)
for dyn_params in edges_df['dynamics_params'].unique():
    try:
        with open(os.path.join(synaptic_models_path, dyn_params)) as f:
            synaptic_model_dict = json.load(f)
        synaptic_type = dyn_params.split('.')[0]
        synaptic_model_dict['tau_syn_weights'] = basis_function_weights.loc[synaptic_type].values.tolist()
        # save the results in a new file
        with open(os.path.join(synaptic_models_path, dyn_params), 'w') as f:
            json.dump(synaptic_model_dict, f)
    
    except:
        pass


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
    new_pop_dict["params"]["synaptic_model"] = dynamic_params_json.split(".")[0]
    new_pop_dict["params"]["receptor_type"] = synaptic_model_dict["receptor_type"]
    new_pop_dict["params"]["tau_syn_weights"] = synaptic_model_dict["tau_syn_weights"]
    lgn_input_network.append(new_pop_dict)
new_input_network.append(lgn_input_network)


### BKG input network ###
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
# Correct the naming of the params file in the new V1 version
edges_df['dynamics_params'] = edges_df['dynamics_params'].str.replace('lgn', 'exc')

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
    new_pop_dict["params"]["synaptic_model"] = dynamic_params_json.split(".")[0]
    new_pop_dict["params"]["receptor_type"] = synaptic_model_dict["receptor_type"]
    new_pop_dict["params"]["tau_syn_weights"] = synaptic_model_dict["tau_syn_weights"]

    bkg_input_network.append(new_pop_dict)

new_input_network.append(bkg_input_network)

with open(os.path.join(data_dir, "new_input_dat.pkl"), "wb") as file:
    pkl.dump(new_input_network, file)
