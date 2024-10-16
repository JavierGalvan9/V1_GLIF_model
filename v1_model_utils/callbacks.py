import os
import json
import numpy as np
import pandas as pd
import tensorflow as tf
import datetime as dt
import subprocess
from time import time
import pickle as pkl
from numba import njit
from matplotlib import pyplot as plt
import seaborn as sns
# from scipy.signal import correlate
from scipy.signal import welch
from scipy.stats import ks_2samp
import stim_dataset
from v1_model_utils import other_v1_utils
from v1_model_utils.plotting_utils import InputActivityFigure, PopulationActivity
from v1_model_utils.model_metrics_analysis import ModelMetricsAnalysis
from v1_model_utils.model_metrics_analysis import calculate_Firing_Rate, get_borders, draw_borders


def printgpu(gpu_id=0):
    if tf.config.list_physical_devices('GPU'):
        # Check TensorFlow memory info
        meminfo = tf.config.experimental.get_memory_info(f'GPU:{gpu_id}')
        current = meminfo['current'] / 1024**3
        peak = meminfo['peak'] / 1024**3
        print(f'    TensorFlow GPU {gpu_id} Memory Usage: {current:.2f} GiB, Peak Usage: {peak:.2f} GiB')
        # Check GPU memory using nvidia-smi
        result = subprocess.run(['nvidia-smi', '--query-gpu=memory.used,memory.free,memory.total', '--format=csv,nounits,noheader'],
                            stdout=subprocess.PIPE, encoding='utf-8') # MiB
        used, free, total = [float(x)/1024 for x in result.stdout.strip().split(',')]
        print(f"    Total GPU Memory Usage: Used: {used:.2f} GiB, Free: {free:.2f} GiB, Total: {total:.2f} GiB")

        # return current, peak

def compose_str(metrics_values):
        _acc, _loss, _rate, _rate_loss, _voltage_loss, _regularizers_loss, _osi_dsi_loss, _sync_loss = metrics_values
        _s = f'Loss {_loss:.4f}, '
        _s += f'RLoss {_rate_loss:.4f}, '
        _s += f'VLoss {_voltage_loss:.4f}, '
        _s += f'RegLoss {_regularizers_loss:.4f}, '
        _s += f'OLoss {_osi_dsi_loss:.4f}, '
        _s += f'SLoss {_sync_loss:.4f}, '
        _s += f'Accuracy {_acc:.4f}, '
        _s += f'Rate {_rate:.4f}'
        return _s

def compute_ks_statistics(df, metric='Weight', min_n_sample=15):
    """
    Compute the Kolmogorov-Smirnov statistic and similarity scores for each cell type in the dataframe.
    Parameters:
    - df: pd.DataFrame, contains data with columns 'data_type' and 'Ave_Rate(Hz)', and indexed by cell type.
    Returns:
    - mean_similarity_score: float, the mean of the similarity scores computed across all cell types.
    """
    # Get unique cell types
    # cell_types = df.index.unique()
    cell_types = df['Post_names'].unique()
    # Initialize a dictionary to store the results
    ks_results = {}
    similarity_scores = {}
    # Iterate over cell types
    for cell_type in cell_types:
        # Filter data for current cell type from two different data types
        # df1 = df.loc[(df.index == cell_type) & (df['data_type'] == 'V1/LM GLIF model'), metric]
        # df2 = df.loc[(df.index == cell_type) & (df['data_type'] == 'Neuropixels'), metric]
        df1 = df.loc[(df['Post_names'] == cell_type) & (df['Weight Type'] == 'Initial weight'), metric]
        df2 = df.loc[(df['Post_names'] == cell_type) & (df['Weight Type'] == 'Final weight'), metric]
        # Drop NA values
        df1.dropna(inplace=True)
        df2.dropna(inplace=True)
        # Calculate the Kolmogorov-Smirnov statistic
        if len(df1) >= min_n_sample and len(df2) >= min_n_sample:
            ks_stat, p_value = ks_2samp(df1, df2)
            ks_results[cell_type] = (ks_stat, p_value)
            similarity_scores[cell_type] = 1 - ks_stat

    # Calculate the mean of the similarity scores and return it
    mean_similarity_score = np.mean(list(similarity_scores.values()))
    return mean_similarity_score

# Define a function to compute the exponential decay of a spike train
def exponential_decay_filter(spike_train, tau=20):
    decay_factor = np.exp(-1/tau)
    continuous_signal = np.zeros_like(spike_train, dtype=float)
    continuous_signal[0] = spike_train[0]
    for i in range(1, len(spike_train)):
        continuous_signal[i] = decay_factor * continuous_signal[i-1] + spike_train[i]
    return continuous_signal

# Define a function to calculate the power spectrum
def calculate_power_spectrum(signal, fs=1000):
    f, Pxx = welch(signal, fs, nperseg=100)
    return f, Pxx

@njit
def pop_fano(spikes, bin_sizes):
    fanos = np.zeros(len(bin_sizes))
    for i, bin_width in enumerate(bin_sizes):
        bin_size = int(np.round(bin_width * 1000))
        max_index = spikes.shape[0] // bin_size * bin_size
        # drop the last bin if it is not complete
        # sum over neurons to get the spike counts
        trimmed_spikes = np.sum(spikes[:max_index, :], axis=1) 
        trimmed_spikes = np.reshape(trimmed_spikes, (max_index // bin_size, bin_size, -1))
        # sum over the bins
        sp_counts = np.sum(trimmed_spikes, axis=1)
        # Calculate the mean of the spike counts
        mean_count = np.mean(sp_counts)
        if mean_count > 0:
            # Calculate the Fano Factor
            fanos[i] = np.var(sp_counts) / mean_count
                 
    return fanos

# # create a class for callbacks in other training sessions (e.g. validation, testing)
class OsiDsiCallbacks:
    def __init__(self, network, lgn_input, bkg_input, flags, logdir, current_epoch=0,
                pre_delay=50, post_delay=50, model_variables_init=None):
        self.n_neurons = flags.neurons
        self.network = network
        self.lgn_input = lgn_input
        self.bkg_input = bkg_input
        self.flags = flags
        self.logdir = logdir
        self.pre_delay = pre_delay
        self.post_delay = post_delay
        self.current_epoch = current_epoch
        self.model_variables_dict = model_variables_init
        # Analize changes in trainable variables.
        if self.model_variables_dict is not None:
            for var in self.model_variables_dict['Best'].keys():
                t0 = time()
                self.trainable_variable_change_heatmaps_and_distributions(var)
                print(f'Time spent in {var}: {time()-t0}')

    def trainable_variable_change_heatmaps_and_distributions(self, variable):
        node_types_voltage_scale = (self.network['node_params']['V_th'] - self.network['node_params']['E_L']).astype(np.float16)
        node_type_ids = self.network['node_type_ids']
        if 'rest_of_brain_weights' in variable:
            voltage_scale = node_types_voltage_scale[node_type_ids[self.bkg_input['indices'][:, 0]]]
            self.node_to_pop_weights_analysis(self.bkg_input['indices'], variable=variable, voltage_scale=voltage_scale)
        elif'sparse_input_weights' in variable:
            voltage_scale = node_types_voltage_scale[node_type_ids[self.lgn_input['indices'][:, 0]]]
            self.node_to_pop_weights_analysis(self.lgn_input['indices'], variable=variable, voltage_scale=voltage_scale)
        elif 'sparse_recurrent_weights' in variable:
            indices = self.network['synapses']['indices']
            voltage_scale = node_types_voltage_scale[node_type_ids[indices[:, 0]]]
            self.pop_to_pop_weights_analysis(indices, variable=variable, voltage_scale=voltage_scale)
            self.pop_to_pop_weights_distribution(indices, variable=variable, voltage_scale=voltage_scale)

    def node_to_pop_weights_analysis(self, indices, variable='', voltage_scale=None):
        pop_names = other_v1_utils.pop_names(self.network)
        target_cell_types = [other_v1_utils.pop_name_to_cell_type(pop_name) for pop_name in pop_names]
        # if 'rest_of_brain_weights' in variable:
        #     post_indices =  np.repeat(indices[:, 0], 4)
        #     voltage_scale = np.repeat(voltage_scale, 4)
        # else:
        #     post_indices = indices[:, 0]

        post_indices = indices[:, 0]
        post_cell_types = [target_cell_types[i] for i in post_indices]

        if voltage_scale is not None:
            # Normalize the weights by the voltage scale
            initial_weights = self.model_variables_dict['Initial'][variable] * voltage_scale
            final_weights = self.model_variables_dict['Best'][variable] * voltage_scale
        else:
            initial_weights = self.model_variables_dict['Initial'][variable]
            final_weights = self.model_variables_dict['Best'][variable]

        # Create DataFrame with all the necessary data
        df = pd.DataFrame({
            'Post_names': post_cell_types * 2,  # Duplicate node names for initial and final weights
            'Weight': initial_weights.tolist() + final_weights.tolist(),  # Combine initial and final weights
            'Weight Type': ['Initial weight'] * len(initial_weights) + ['Final weight'] * len(final_weights)  # Distinguish between initial and final weights
        })

        # Count the number of cell_types fro each type
        # cell_type_counts = df['Post_names'].value_counts()

        # Sort the dataframe by Node Name and then by Type to ensure consistent order
        df = df.sort_values(['Post_names', 'Weight Type'])

        # Plotting
        boxplots_dir = os.path.join(self.logdir, f'Boxplots/{variable}')
        os.makedirs(boxplots_dir, exist_ok=True)
        # fig, axs = plt.subplots(2, 1, figsize=(12, 14))
        fig = plt.figure(figsize=(12, 6))
        #get the axis of the figure
        ax = fig.gca()
        similarity_score = compute_ks_statistics(df, metric='Weight', min_n_sample=15)
        hue_order = ['Initial weight', 'Final weight']
        # sns.boxplot(x='Node Name', y='Weight Change', data=df)
        # sns.barplot(x='Post_names', y='Weight', hue='State', hue_order=hue_order, data=df)
        sns.boxplot(x='Post_names', y='Weight', hue='Weight Type', hue_order=hue_order, data=df, ax=ax, width=0.7, fliersize=1.)
        # plt.axhline(0, color='black', linewidth=1)  # include a horizontal black line at 0
        ax.set_yscale('log')
        # ax.set_ylim(bottom=0)  # Set bottom limit of y-axis to 0
        ax.tick_params(axis="x", labelrotation=90)
        ax.set_title(f'{variable}')
        ax.legend(loc='upper right')
        if similarity_score is not None:
            ax.text(0.9, 0.1, f'S: {similarity_score:.2f}', transform=ax.transAxes, fontsize=14,
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
        plt.tight_layout()
        plt.savefig(os.path.join(boxplots_dir, f'{variable}.png'), dpi=300, transparent=False)
        plt.close()

    def pop_to_pop_weights_distribution(self, indices, variable='', voltage_scale=None):
        source_pop_names = other_v1_utils.pop_names(self.network)
        source_cell_types = [other_v1_utils.pop_name_to_cell_type(pop_name) for pop_name in source_pop_names]
        target_pop_names = other_v1_utils.pop_names(self.network)
        target_cell_types = [other_v1_utils.pop_name_to_cell_type(pop_name) for pop_name in target_pop_names]
        post_cell_types = [target_cell_types[i] for i in indices[:, 0]]
        pre_cell_types = [source_cell_types[i] for i in indices[:, 1]]

        if voltage_scale is not None:
            # Normalize the weights by the voltage scale
            initial_weights = self.model_variables_dict['Initial'][variable] * voltage_scale
            final_weights = self.model_variables_dict['Best'][variable] * voltage_scale
        else:
            initial_weights = self.model_variables_dict['Initial'][variable]
            final_weights = self.model_variables_dict['Best'][variable]

        df = pd.DataFrame({
            'Post_names': post_cell_types,
            'Pre_names': pre_cell_types,
            'Initial weight': initial_weights,
            'Final weight': final_weights,
        })

        # Melt DataFrame to long format
        df_melted = df.melt(id_vars=['Post_names', 'Pre_names'], value_vars=['Initial weight', 'Final weight'], 
                            var_name='Weight Type', value_name='Weight')
        df_melted['Weight'] = np.abs(df_melted['Weight'])
        # Create directory for saving plots
        boxplots_dir = os.path.join(self.logdir, f'Boxplots/{variable}_distribution')
        os.makedirs(boxplots_dir, exist_ok=True)
        # Establish the order of the neuron types in the boxplots
        cell_type_order = np.sort(df['Pre_names'].unique())
        target_type_order = np.sort(df['Post_names'].unique())

        # Define the palette
        palette = {"Initial weight": "#87CEEB", "Final weight": "#FFA500"}
        # Create subplots
        num_pre_names = len(cell_type_order)

        num_columns = 4
        num_rows = (num_pre_names + num_columns - 1) // num_columns
        fig, axes = plt.subplots(num_rows, num_columns, figsize=(24, 6 * num_rows))
        # Flatten the axes array and handle the first row separately
        axes = axes.flatten()
        for i, pre_name in enumerate(cell_type_order):
            if num_pre_names == 17 and i == 0:
                i = 1
            elif num_pre_names == 17 and i != 0:
                i += 3
            
            ax = axes[i] 
            subset_df = df_melted[df_melted['Pre_names'] == pre_name]
            similarity_score = compute_ks_statistics(subset_df, metric='Weight', min_n_sample=15)
            # subset_cell_type_order = np.sort(subset_df['Post_names'].unique())
            # Create boxplot for Initial and Final weights
            sns.boxplot(data=subset_df, x='Post_names', y='Weight', hue='Weight Type', order=target_type_order, ax=ax, palette=palette, 
                        width=0.7, fliersize=1.)
            # sns.violinplot(data=subset_df, x='Post_names', y='Weight', hue='Weight Type', order=subset_cell_type_order, ax=ax, palette=palette, width=0.7,
            #                split=True, inner="quart", gap=0.2)
            ax.set_title(f'Source Cell Type: {pre_name}')
            ax.set_ylabel(r'$\vert$ Synaptic Weight (pA)$\vert$', fontsize=12)
            # ax.set_yscale('symlog', linthresh=0.001)
            ax.set_yscale('log')
            if i % num_columns == 0 or i == 1:  # First column
                ax.set_ylabel(r'$\vert$ Synaptic Weight (pA)$\vert$', fontsize=12)
            else:
                ax.set_ylabel('')
            if i >= (num_rows - 1) * num_columns:
                ax.set_xlabel('Target Cell Type')
            else:
                ax.set_xlabel('')
            ax.tick_params(axis="x", labelrotation=90)
            # Apply shadings to each layer
            xticklabel = ax.get_xticklabels()
            borders = get_borders(xticklabel)
            # change y limit
            # if 'E' in pre_name:
            #     bottom_limit = 0
            #     upper_limit = 1000
            # else:
            #     bottom_limit = -500
            #     upper_limit = 0
            bottom_limit = 0.01
            upper_limit = 100
            ax.set_ylim(bottom=bottom_limit, top=upper_limit)
            # get the current ylim
            ylim = ax.get_ylim()
            draw_borders(ax, borders, ylim)
            # ax.legend(loc='best')
            if i == 1 and num_pre_names == 17:
                ax.legend(loc='upper right')
            elif i==0 and num_pre_names != 17:
                ax.legend(loc='upper left')
            else:
                ax.get_legend().remove()

            if similarity_score is not None:
                ax.text(0.82, 0.95, f'S: {similarity_score:.2f}', transform=ax.transAxes, fontsize=14,
                        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
        
        # Remove any empty subplots
        if num_pre_names == 17:
            for j in [0, 2, 3]:
                fig.delaxes(axes[j])

        # Adjust layout
        plt.tight_layout()
        plt.savefig(os.path.join(boxplots_dir, f'{variable}.png'), dpi=300, transparent=False)
        plt.close()

    def pop_to_pop_weights_analysis(self, indices, variable='', voltage_scale=None):
        source_pop_names = other_v1_utils.pop_names(self.network)
        source_cell_types = [other_v1_utils.pop_name_to_cell_type(pop_name) for pop_name in source_pop_names]
        target_pop_names = other_v1_utils.pop_names(self.network)
        target_cell_types = [other_v1_utils.pop_name_to_cell_type(pop_name) for pop_name in target_pop_names]
        post_cell_types = [target_cell_types[i] for i in indices[:, 0]]
        pre_cell_types = [source_cell_types[i] for i in indices[:, 1]]

        ### Initial Weight ###
        if voltage_scale is not None:
            # Normalize the weights by the voltage scale
            initial_weights = self.model_variables_dict['Initial'][variable] * voltage_scale
            final_weights = self.model_variables_dict['Best'][variable] * voltage_scale
        else:
            initial_weights = self.model_variables_dict['Initial'][variable]
            final_weights = self.model_variables_dict['Best'][variable]

        weight_changes = final_weights - initial_weights
        df = pd.DataFrame({'Post_names': post_cell_types, 
                            'Pre_names':pre_cell_types, 
                            'Initial weight': initial_weights, 
                            'Final weight': final_weights, 
                            'Weight Change': weight_changes})
        
        # Calculate global min and max for color normalization
        # global_grouped_df = df.groupby(['Pre_names', 'Post_names'])[['Initial weight', 'Final weight']].mean().reset_index()
        # global_min = global_grouped_df[['Initial weight', 'Final weight']].min().min()
        # global_max = global_grouped_df[['Initial weight', 'Final weight']].max().max()
        # global_min = df[['Initial weight', 'Final weight']].min().min()
        # global_max = df[['Initial weight', 'Final weight']].max().max()

        boxplots_dir = os.path.join(self.logdir, f'Boxplots/{variable}')
        os.makedirs(boxplots_dir, exist_ok=True)

        # Plot for Initial Weight
        if not os.path.exists(os.path.join(self.logdir, 'Boxplots', variable, 'Initial_weight.png')):
            grouped_df = df.groupby(['Pre_names', 'Post_names'])['Initial weight'].mean().reset_index()
            # Create a pivot table to reshape the data for the heatmap
            pivot_df = grouped_df.pivot(index='Pre_names', columns='Post_names', values='Initial weight')
            # Plot heatmap
            fig = plt.figure(figsize=(12, 6))
            # heatmap = sns.heatmap(pivot_df, cmap='RdBu_r', annot=False, cbar=False, center=0, vmin=global_min, vmax=global_max)
            # heatmap = sns.heatmap(pivot_df, cmap='RdBu_r', annot=False, cbar=False, center=0, vmin=-0.5, vmax=0.5)
            heatmap = sns.heatmap(pivot_df, cmap='RdBu_r', annot=False, cbar=False, center=0, vmin=-20, vmax=20)
            # plt.xlabel(f'V1')
            # plt.ylabel(f'V1')
            plt.xticks(rotation=90)
            plt.gca().set_aspect('equal')
            plt.title(f'{variable}')
            # Create a separate color bar axis
            cbar_ax = plt.gcf().add_axes([0.92, 0.15, 0.02, 0.55])  # [left, bottom, width, height]
            # Plot color bar
            cbar = plt.colorbar(heatmap.collections[0], cax=cbar_ax)
            cbar.set_label('Initial Weight (pA)')
            plt.savefig(os.path.join(boxplots_dir, f'Initial_weight.png'), dpi=300, transparent=False)
            plt.close()

        ### Final Weight ###
        grouped_df = df.groupby(['Pre_names', 'Post_names'])['Final weight'].mean().reset_index()
        # Create a pivot table to reshape the data for the heatmap
        pivot_df = grouped_df.pivot(index='Pre_names', columns='Post_names', values='Final weight')
        # Plot heatmap
        plt.figure(figsize=(12, 6))
        # heatmap = sns.heatmap(pivot_df, cmap='RdBu_r', annot=False, cbar=False, center=0, vmin=global_min, vmax=global_max)
        # heatmap = sns.heatmap(pivot_df, cmap='RdBu_r', annot=False, cbar=False, center=0, vmin=-0.5, vmax=0.5)
        heatmap = sns.heatmap(pivot_df, cmap='RdBu_r', annot=False, cbar=False, center=0, vmin=-20, vmax=20)
        # plt.xlabel(f'V1')
        # plt.ylabel(f'V1')
        plt.xticks(rotation=90)
        plt.gca().set_aspect('equal')
        plt.title(f'{variable}')
        # Create a separate color bar axis
        cbar_ax = plt.gcf().add_axes([0.92, 0.15, 0.02, 0.55])  # [left, bottom, width, height]
        # Plot color bar
        cbar = plt.colorbar(heatmap.collections[0], cax=cbar_ax)
        cbar.set_label('Final Weight (pA)')
        plt.savefig(os.path.join(boxplots_dir, f'Final_weight.png'), dpi=300, transparent=False)
        plt.close()

        ### Weight change ###
        grouped_df = df.groupby(['Pre_names', 'Post_names'])['Weight Change'].mean().reset_index()
        # Create a pivot table to reshape the data for the heatmap
        try:
            pivot_df = grouped_df.pivot(index='Pre_names', columns='Post_names', values='Weight Change')
            # Plot heatmap
            plt.figure(figsize=(12, 6))
            # heatmap = sns.heatmap(pivot_df, cmap='RdBu_r', annot=False, cbar=False, center=0)
            # heatmap = sns.heatmap(pivot_df, cmap='RdBu_r', annot=False, cbar=False, center=0, vmin=-0.5, vmax=0.5)
            heatmap = sns.heatmap(pivot_df, cmap='RdBu_r', annot=False, cbar=False, center=0, vmin=-20, vmax=20)
            # plt.xlabel(f'V1')
            # plt.ylabel(f'V1')
            plt.xticks(rotation=90)
            plt.gca().set_aspect('equal')
            plt.title(f'{variable}')
            # Create a separate color bar axis
            cbar_ax = plt.gcf().add_axes([0.92, 0.15, 0.02, 0.55])  # [left, bottom, width, height]
            # Plot color bar
            cbar = plt.colorbar(heatmap.collections[0], cax=cbar_ax)
            cbar.set_label('Weight Change (pA)')
            plt.savefig(os.path.join(boxplots_dir, f'Weight_change.png'), dpi=300, transparent=False)
            plt.close()
        except:
            print('Skipping the plot for the weight change heatmap...')
            # raise the actual error
            print(grouped_df)
        
    def fano_factor(self, spikes, t_start=0.7, t_end=2.5, n_samples=100, analyze_core_only=True):
        
        if analyze_core_only:
            # Isolate the core neurons
            pop_names = other_v1_utils.pop_names(self.network, core_radius=self.flags.loss_core_radius)
            core_mask = other_v1_utils.isolate_core_neurons(self.network, radius=self.flags.loss_core_radius, data_dir=self.flags.data_dir)
            n_core_neurons = np.sum(core_mask)
            spikes = spikes[:, :, :, core_mask]
        else:
            n_core_neurons = spikes.shape[-1]
            pop_names = other_v1_utils.pop_names(self.network)

        # Calculate the Fano Factor for the spikes
        node_ei = np.array([pop_name[0] for pop_name in pop_names])
        node_id = np.arange(n_core_neurons)
        # Get the IDs for excitatory neurons
        node_id_e = node_id[node_ei == 'e']
        # Reshape spikes data 
        new_spikes = spikes[:, :, int(1000*t_start):int(1000*t_end), :]
        new_spikes = new_spikes.reshape(-1, new_spikes.shape[2], new_spikes.shape[3])
        n_trials, seq_len, n_neurons = new_spikes.shape
        # # Prepare timestamps (avoiding unnecessary memory allocation)
        # time = np.arange(t_start, t_end, 0.001).astype(np.float32)
        # most_single_neuron_spikes_in_trial = np.max(np.sum(new_spikes, axis=1))
        # # Preallocate using the known shape
        # spikes_timestamps = np.full((n_trials, n_neurons, most_single_neuron_spikes_in_trial), np.nan)
        # for i in range(n_trials):
        #     for j in range(n_neurons):
        #         timestamps = time[new_spikes[i, :, j]]
        #         spikes_timestamps[i, j, :len(timestamps)] = timestamps

        # Generate Fano factors across random samples
        fanos = []
        # Pre-define bin sizes
        bin_sizes = np.logspace(-3, 0, 20)
        # using the simulation length, limit bin_sizes to define at least 2 bins
        bin_sizes_mask = bin_sizes < (t_end - t_start)/2
        bin_sizes = bin_sizes[bin_sizes_mask]
        # Vectorize the sampling process
        sample_counts = np.random.normal(70, 30, n_samples).astype(int)
        # ensure that the sample counts are at least 15
        sample_counts = np.maximum(sample_counts, 15)
        # ensure that the sample counts are less than the number of neurons
        sample_counts = np.minimum(sample_counts, len(node_id_e))
        # trial_ids =np.random.choice(np.arange(n_trials), n_samples, replace=False)
        trial_ids = np.random.randint(n_trials, size=n_samples)

        for i in range(n_samples):
            random_trial_id = trial_ids[i]
            sample_num = sample_counts[i]
            sample_ids = np.random.choice(node_id_e, sample_num, replace=False)
            # selected_spikes = np.concatenate([spikes_timestamps[random_trial_id][np.isin(node_id, sample_ids), :]])
            # selected_spikes = selected_spikes[~np.isnan(selected_spikes)]
            selected_spikes = new_spikes[random_trial_id][:, np.isin(node_id, sample_ids)]
            # if there are spikes use pop_fano
            if np.sum(selected_spikes) > 0:
                fano = pop_fano(selected_spikes, bin_sizes)
                fanos.append(fano)

        fanos = np.array(fanos)
        # mean_fano = np.mean(fanos, axis=0)
        return fanos, bin_sizes
        
    def fanos_figure(self, spikes, n_samples=100, analyze_core_only=True, data_dir='Synchronization_data'):
        # Calculate fano factors for both sessions
        evoked_fanos, evoked_bin_sizes = self.fano_factor(spikes, t_start=0.7, t_end=2.5, n_samples=n_samples, analyze_core_only=analyze_core_only)
        spontaneous_fanos, spont_bin_sizes = self.fano_factor(spikes, t_start=0.2, t_end=0.5, n_samples=n_samples, analyze_core_only=analyze_core_only)

        # Calculate mean, standard deviation, and SEM of the Fano factors
        evoked_fanos_mean = np.nanmean(evoked_fanos, axis=0)
        evoked_fanos_std = np.nanstd(evoked_fanos, axis=0)
        evoked_fanos_sem = evoked_fanos_std / np.sqrt(n_samples)

        spontaneous_fanos_mean = np.nanmean(spontaneous_fanos, axis=0)
        spontaneous_fanos_std = np.nanstd(spontaneous_fanos, axis=0)
        spontaneous_fanos_sem = spontaneous_fanos_std / np.sqrt(n_samples)

        # find the frequency of the maximum
        evoked_max_fano = np.nanmax(evoked_fanos_mean)
        evoked_max_fano_freq = 1/(2*evoked_bin_sizes[np.nanargmax(evoked_fanos_mean)])
        spontaneous_max_fano = np.nanmax(spontaneous_fanos_mean)
        spontaneous_max_fano_freq = 1/(2*spont_bin_sizes[np.nanargmax(spontaneous_fanos_mean)])

        # Calculate the evoked experimental error committed
        # evoked_exp_data_path = 'Synchronization_data/all_fano_300ms_evoked.npy'
        evoked_exp_data_path = os.path.join(data_dir, 'Fano_factor_v1', 'v1_fano_running_1800ms_evoked.npy')

        # load the experimental data
        evoked_exp_fanos = np.load(evoked_exp_data_path, allow_pickle=True)
        n_experimental_samples = evoked_exp_fanos.shape[0]
        # Calculate mean, standard deviation, and SEM of the Fano factors
        evoked_exp_fanos_mean = np.nanmean(evoked_exp_fanos, axis=0)
        # filter bin_sizes where the experimental data is not nan or zero
        # evoked_exp_fanos_mean = evoked_exp_fanos_mean[bin_sizes_mask]
        evoked_exp_fanos_std = np.nanstd(evoked_exp_fanos, axis=0)
        evoked_exp_fanos_sem = evoked_exp_fanos_std / np.sqrt(n_experimental_samples)

        # Calculate the spontaneous experimental error committed
        # spont_exp_data_path = 'Synchronization_data/all_fano_300ms_spont.npy'
        spont_exp_data_path = os.path.join(data_dir, 'Fano_factor_v1', 'v1_fano_running_300ms_spont.npy')
        # load the experimental data
        spont_exp_fanos = np.load(spont_exp_data_path, allow_pickle=True)
        n_experimental_samples = spont_exp_fanos.shape[0]
        # Calculate mean, standard deviation, and SEM of the Fano factors
        spont_exp_fanos_mean = np.nanmean(spont_exp_fanos, axis=0)
        # filter bin_sizes where the experimental data is not nan or zero
        # spont_exp_fanos_mean = spont_exp_fanos_mean[bin_sizes_mask]
        spont_exp_fanos_std = np.nanstd(spont_exp_fanos, axis=0)
        spont_exp_fanos_sem = spont_exp_fanos_std / np.sqrt(n_experimental_samples)
        
        # Plot the Fano Factor
        fig, axs = plt.subplots(1, 2, figsize=(12, 6), sharex=True, sharey=True)
        # plot fanos with error bars
        axs[0].errorbar(evoked_bin_sizes, evoked_fanos_mean, yerr=evoked_fanos_sem, fmt='o-', label='Evoked Model', color='blue')
        axs[0].errorbar(evoked_bin_sizes, evoked_exp_fanos_mean[:len(evoked_bin_sizes)], yerr=evoked_exp_fanos_sem[:len(evoked_bin_sizes)], fmt='o-', label='Evoked Experimental', color='k')
        axs[0].set_xscale("log")
        axs[0].set_title(f'V1 - Max: {evoked_max_fano:.2f}, Freq: {evoked_max_fano_freq:.1f} Hz', fontsize=16)
        axs[0].set_xlabel('Bin Size (s)', fontsize=14)
        axs[0].set_ylabel('Fano Factor', fontsize=14)
        axs[0].legend(fontsize=14)

        axs[1].errorbar(spont_bin_sizes, spontaneous_fanos_mean, yerr=spontaneous_fanos_sem, fmt='o-', label='Spontaneous Model', color='orange')
        axs[1].errorbar(spont_bin_sizes, spont_exp_fanos_mean[:len(spont_bin_sizes)], yerr=spont_exp_fanos_sem[:len(spont_bin_sizes)], fmt='o-', label='Spontaneous Experimental', color='k')
        axs[1].set_xscale("log")
        axs[1].set_title(f'V1 - Max: {spontaneous_max_fano:.2f}, Freq: {spontaneous_max_fano_freq:.1f} Hz', fontsize=16)
        axs[1].set_xlabel('Bin Size (s)', fontsize=14)
        axs[1].legend(fontsize=14)

        plt.tight_layout()
        os.makedirs(os.path.join(self.logdir, 'Fano_Factor'), exist_ok=True)
        plt.savefig(os.path.join(self.logdir, 'Fano_Factor', f'V1_epoch_{self.current_epoch}.png'), dpi=300, transparent=False)
        plt.close()
    
    def power_spectrum(self, v1_spikes, v1_spikes_spont=None, fs=1000, directory=''):
        # Sum the spikes over the batch size and all neurons to get a single spiking activity signal for each area
        combined_spiking_activity_v1 = v1_spikes.mean(axis=1)
        v1_signal = exponential_decay_filter(combined_spiking_activity_v1)
        # # Compute the power spectrum for the combined signal for each area
        # seq_len = combined_spiking_activity_v1.shape[0]
        # fs = 1000.0
        # frequencies = np.fft.fftfreq(seq_len, d=1/fs)
        # fft_values_v1 = np.fft.fft(combined_spiking_activity_v1)
        # fft_values_lm = np.fft.fft(combined_spiking_activity_lm)
        # power_spectrum_v1 = np.abs(fft_values_v1) ** 2 / seq_len
        # power_spectrum_lm = np.abs(fft_values_lm) ** 2 / seq_len

        # Calculate the power spectrum
        # Sampling frequency (1 kHz)
        f_v1, power_spectrum_v1 = calculate_power_spectrum(v1_signal, fs)

        # Plot the power spectrum
        plt.figure(figsize=(12, 8))
        sns.lineplot(x=f_v1, y=power_spectrum_v1, label='V1', color='blue')

        if v1_spikes_spont is not None:
            combined_spiking_activity_v1_spont = v1_spikes_spont.mean(axis=1)
            v1_signal_spont = exponential_decay_filter(combined_spiking_activity_v1_spont)
            f_v1_spont, power_spectrum_v1_spont = calculate_power_spectrum(v1_signal_spont, fs)
            sns.lineplot(x=f_v1_spont, y=power_spectrum_v1_spont, label='V1 Spontaneous', linestyle='--', color='blue')

        # Remove the 0 Hz component for plotting
        # positive_frequencies = frequencies[:seq_len // 2]
        # positive_power_spectrum_v1 = power_spectrum_v1[:seq_len // 2]
        # positive_power_spectrum_lm = power_spectrum_lm[:seq_len // 2]
        # # # Set up the Seaborn style
        # sns.set(style="ticks")
        # plt.semilogy()
        plt.xlim([0, 50])
        plt.title('Power Spectral Density of Neuronal Spiking Activity in V1', fontsize=16)
        plt.xlabel('Frequency (Hz)', fontsize=14)
        plt.ylabel('Power Spectral Density [1/Hz]', fontsize=14)
        plt.legend()
        # plt.grid(True)
        plt.tight_layout()
        os.makedirs(directory, exist_ok=True)
        plt.savefig(os.path.join(directory, f'epoch_{self.current_epoch}.png'), dpi=300, transparent=False)
        plt.close()

    def plot_populations_activity(self, v1_spikes):
        # Plot the mean firing rate of the population of neurons
        filename = f'Epoch_{self.current_epoch}'
        Population_activity = PopulationActivity(n_neurons=self.n_neurons, network=self.network, 
                                                stimuli_init_time=500, stimuli_end_time=2500, 
                                                image_path=self.logdir, filename=filename, data_dir=self.flags.data_dir,
                                                core_radius=self.flags.plot_core_radius)
        Population_activity(v1_spikes, plot_core_only=True, bin_size=10)

    def plot_raster(self, x, v1_spikes, angle=0):
        seq_len = v1_spikes.shape[1]
        images_dir = os.path.join(self.logdir, 'Raster_plots_OSI_DSI')
        os.makedirs(images_dir, exist_ok=True)
        graph = InputActivityFigure(
                                    self.network,
                                    self.flags.data_dir,
                                    images_dir,
                                    filename=f'Epoch_{self.current_epoch}_orientation_{angle}_degrees',
                                    frequency=self.flags.temporal_f,
                                    stimuli_init_time=self.pre_delay,
                                    stimuli_end_time=seq_len-self.post_delay,
                                    reverse=False,
                                    plot_core_only=True,
                                    core_radius=self.flags.plot_core_radius,
                                    )
        graph(x, v1_spikes)

    def plot_population_firing_rates_vs_tuning_angle(self, spikes, DG_angles, core_radius=400):
        # Save the spikes
        spikes_dir = os.path.join(self.logdir, 'Spikes_OSI_DSI')
        os.makedirs(spikes_dir, exist_ok=True)

        # Isolate the core neurons
        core_mask = other_v1_utils.isolate_core_neurons(self.network, radius=self.flags.plot_core_radius, data_dir=self.flags.data_dir)
        spikes = spikes[:, :, :, core_mask]
        spikes = np.sum(spikes, axis=0).astype(np.float32)/self.flags.n_trials_per_angle
        seq_len = spikes.shape[1]

        tuning_angles = self.network['tuning_angle'][core_mask]
        for angle_id, angle in enumerate(DG_angles):
            firingRates = calculate_Firing_Rate(spikes[angle_id, :, :], stimulus_init=self.pre_delay, stimulus_end=seq_len-self.post_delay, temporal_axis=0)
            x = tuning_angles
            y = firingRates
            # Define bins for delta_angle
            bins = np.linspace(np.min(x), np.max(x), 50)
            # Compute average rates for each bin
            average_rates = []
            for i in range(len(bins)-1):
                mask = (x >= bins[i]) & (x < bins[i+1])
                average_rates.append(np.mean(y[mask]))
            # Create bar plot
            plt.figure(figsize=(10, 6))
            plt.bar(bins[:-1], average_rates, width=np.diff(bins))
            plt.xlabel('Tuning Angle')
            plt.ylabel('Average Rates')
            plt.title(f'Gratings Angle: {angle}')
            plt.savefig(os.path.join(spikes_dir, f'v1_spikes_angle_{angle}.png'))
            plt.close()

    def single_trial_callbacks(self, x, v1_spikes, y, bkg_noise=None):
        # Separate the spontaneous and evoked spikes for the power spectrum
        # x_spont_spikes = x[0, :self.pre_delay, :]
        # x_evoked_spikes = x[0, self.pre_delay:-self.post_delay, :]
        v1_spont_spikes = v1_spikes[0, :self.pre_delay, :].astype(np.float32)
        v1_evoked_spikes = v1_spikes[0, self.pre_delay:-self.post_delay, :].astype(np.float32)
        # Plot the power spectrum of the neuronal activity
        self.power_spectrum(v1_evoked_spikes, v1_spikes_spont=v1_spont_spikes,
                            directory=os.path.join(self.logdir, 'Power_spectrum_OSI_DSI'))
        # Plot the population activity
        self.plot_populations_activity(v1_spikes)
        # Plot the raster plot
        self.plot_raster(x, v1_spikes, angle=y)

    def osi_dsi_analysis(self, v1_spikes, DG_angles):
        # # Average the spikes over the number of trials
        # v1_spikes = np.sum(v1_spikes, axis=0).astype(np.float32)/self.flags.n_trials_per_angle
        # lm_spikes = np.sum(lm_spikes, axis=0).astype(np.float32)/self.flags.n_trials_per_angle

        # Do the OSI/DSI analysis       
        boxplots_dir = os.path.join(self.logdir, 'Boxplots_OSI_DSI')
        os.makedirs(boxplots_dir, exist_ok=True)
        fr_boxplots_dir = os.path.join(self.logdir, f'Boxplots_OSI_DSI/Ave_Rate(Hz)')
        os.makedirs(fr_boxplots_dir, exist_ok=True)
        spontaneous_boxplots_dir = os.path.join(self.logdir, 'Boxplots_OSI_DSI/Spontaneous rate (Hz)')
        os.makedirs(spontaneous_boxplots_dir, exist_ok=True)
         
        # Fano factor analysis
        self.fanos_figure(v1_spikes, n_samples=100, analyze_core_only=True)
        # Plot the tuning angle analysis
        self.plot_population_firing_rates_vs_tuning_angle(v1_spikes, DG_angles)
        # Estimate tuning parameters from the model neurons
        metrics_analysis = ModelMetricsAnalysis(v1_spikes, DG_angles, self.network, data_dir=self.flags.data_dir,
                                                drifting_gratings_init=500, drifting_gratings_end=2500,
                                                spontaneous_init=0, spontaneous_end=500,
                                                core_radius=self.flags.plot_core_radius, df_directory=self.logdir, save_df=True)
        # Figure for OSI/DSI boxplots
        metrics_analysis(metrics=["Rate at preferred direction (Hz)", "OSI", "DSI"], directory=boxplots_dir, filename=f'Epoch_{self.current_epoch}')
        # Figure for Average firing rate boxplots
        metrics_analysis(metrics=["Ave_Rate(Hz)"], directory=fr_boxplots_dir, filename=f'Epoch_{self.current_epoch}')   
        # Spontaneous rates figure
        metrics_analysis(metrics=['Spontaneous rate (Hz)'], directory=spontaneous_boxplots_dir, filename=f'Epoch_{self.current_epoch}') 


class Callbacks:
    def __init__(self, network, lgn_input, bkg_input, model, optimizer, distributed_roll_out, flags, logdir, strategy, 
                metrics_keys, pre_delay=50, post_delay=50, checkpoint=None, model_variables_init=None, 
                save_optimizer=True, spontaneous_training=False):
        
        self.n_neurons = flags.neurons
        self.network = network
        self.lgn_input = lgn_input
        self.bkg_input = bkg_input
        if spontaneous_training:
            self.neuropixels_feature = 'Spontaneous rate (Hz)'
        else:
            self.neuropixels_feature = 'Ave_Rate(Hz)'  
        self.model = model
        self.optimizer = optimizer
        self.flags = flags
        self.logdir = logdir
        self.strategy = strategy
        self.distributed_roll_out = distributed_roll_out
        self.metrics_keys = metrics_keys
        self.pre_delay = pre_delay
        self.post_delay = post_delay
        self.step = 0
        self.step_running_time = []
        self.model_variables_dict = model_variables_init
        self.initial_metric_values = None
        self.summary_writer = tf.summary.create_file_writer(self.logdir)
        with open(os.path.join(self.logdir, 'config.json'), 'w') as f:
            json.dump(flags.flag_values_dict(), f, indent=4)

        if checkpoint is None:
            if save_optimizer:
                checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)
            else:
                checkpoint = tf.train.Checkpoint(model=model)
            self.min_val_loss = float('inf')
            self.no_improve_epochs = 0
            self.checkpoint_epochs = 0
            # create a dictionary to save the values of the metric keys after each epoch
            self.epoch_metric_values = {key: [] for key in self.metrics_keys}
            self.epoch_metric_values['sync'] = []
        else:
            # Load epoch_metric_values and min_val_loss from the file
            if os.path.exists(os.path.join(self.logdir, 'train_end_data.pkl')):
                with open(os.path.join(self.logdir, 'train_end_data.pkl'), 'rb') as f:
                    data_loaded = pkl.load(f)
                self.epoch_metric_values = data_loaded['epoch_metric_values']
                self.min_val_loss = data_loaded['min_val_loss']
                self.no_improve_epochs = data_loaded['no_improve_epochs']
                self.checkpoint_epochs = len(data_loaded['epoch_metric_values']['train_loss'])
            elif os.path.exists(os.path.join(os.path.dirname(flags.restore_from), 'train_end_data.pkl')):
                with open(os.path.join(os.path.dirname(flags.restore_from), 'train_end_data.pkl'), 'rb') as f:
                    data_loaded = pkl.load(f)
                self.epoch_metric_values = data_loaded['epoch_metric_values']
                self.min_val_loss = data_loaded['min_val_loss']
                self.no_improve_epochs = data_loaded['no_improve_epochs']
                self.checkpoint_epochs = len(data_loaded['epoch_metric_values']['train_loss'])
            else:
                print('No train_end_data.pkl file found. Initializing...')
                self.epoch_metric_values = {key: [] for key in self.metrics_keys}
                self.epoch_metric_values['sync'] = []
                self.min_val_loss = float('inf')
                self.no_improve_epochs = 0
                self.checkpoint_epochs = 0

        self.total_epochs = flags.n_runs * flags.n_epochs + self.checkpoint_epochs
        # Manager for the best model
        self.best_manager = tf.train.CheckpointManager(
            checkpoint, directory=self.logdir + '/Best_model', max_to_keep=1
        )
        self.latest_manager = tf.train.CheckpointManager(
            checkpoint, directory=self.logdir + '/latest', max_to_keep=1
        )
        # Manager for osi/dsi checkpoints 
        self.epoch_manager = tf.train.CheckpointManager(
            checkpoint, directory=self.logdir + '/OSI_DSI_checkpoints', max_to_keep=5
        )

    def on_train_begin(self):
        print("---------- Training started at ", dt.datetime.now().strftime('%d-%m-%Y %H:%M'), ' ----------\n')
        self.train_start_time = time()
        # self.epoch = self.flags.run_session * self.flags.n_epochs
        self.epoch = self.checkpoint_epochs

    def on_train_end(self, metric_values, normalizers=None):
        self.train_end_time = time()
        self.final_metric_values = metric_values
        print("\n ---------- Training ended at ", dt.datetime.now().strftime('%d-%m-%Y %H:%M'), ' ----------\n')
        print(f"Total time spent: {self.train_end_time - self.train_start_time:.2f} seconds")
        print(f"Average step time: {np.mean(self.step_running_time):.2f} seconds\n")
        # Determine the maximum key length for formatting the table
        max_key_length = max(len(key) for key in self.metrics_keys)

        # Start of the Markdown table
        print(f"| {'Metric':<{max_key_length}} | {'Initial Value':<{max_key_length}} | {'Final Value':<{max_key_length}} |")
        print(f"|{'-' * (max_key_length + 2)}|{'-' * (max_key_length + 2)}|{'-' * (max_key_length + 2)}|")

        n_metrics = len(self.initial_metric_values)//2
        for initial, final, key in zip(self.initial_metric_values[n_metrics:], self.final_metric_values[n_metrics:], self.metrics_keys[n_metrics:]):
            print(f"| {key:<{max_key_length}} | {initial:<{max_key_length}.3f} | {final:<{max_key_length}.3f} |")

        # Save epoch_metric_values and min_val_loss to a file
        data_to_save = {
            'epoch_metric_values': self.epoch_metric_values,
            'min_val_loss': self.min_val_loss,
            'no_improve_epochs': self.no_improve_epochs
        }
        if normalizers is not None:
            data_to_save['v1_ema'] = normalizers['v1_ema']

        with open(os.path.join(self.logdir, 'train_end_data.pkl'), 'wb') as f:
            pkl.dump(data_to_save, f)

        # # Analize changes in trainable variables.
        # for var in self.model_variables_dict['Best'].keys():
        #     t0 = time()
        #     self.variable_change_analysis(var)
        #     print(f'Time spent in {var}: {time()-t0}')

        if self.flags.n_runs > 1:
            self.plot_osi_dsi(parallel=True)

    def on_epoch_start(self):
        self.epoch += 1
        # self.step_counter.assign_add(1)
        self.epoch_init_time = time()
        date_str = dt.datetime.now().strftime('%d-%m-%Y %H:%M')
        print(f'Epoch {self.epoch:2d}/{self.total_epochs} @ {date_str}')
        tf.print(f'\nEpoch {self.epoch:2d}/{self.total_epochs} @ {date_str}')

    def on_epoch_end(self, x, v1_spikes, y, metric_values, bkg_noise=None, verbose=True,
                    x_spont=None, v1_spikes_spont=None):
        
        if v1_spikes.dtype == tf.float16:
            v1_spikes = v1_spikes.numpy().astype(np.float32)
            x = x.numpy().astype(np.float32)
            y = y.numpy().astype(np.float32)
            if x_spont is not None:
                x_spont = x_spont.numpy().astype(np.float32)
                v1_spikes_spont = v1_spikes_spont.numpy().astype(np.float32)
        else:
            v1_spikes = v1_spikes.numpy()
            x = x.numpy()
            y = y.numpy()
            if x_spont is not None:
                x_spont = x_spont.numpy()
                v1_spikes_spont = v1_spikes_spont.numpy()

        self.step = 0
        if self.initial_metric_values is None:
            self.initial_metric_values = metric_values
        
        if verbose:
            print_str = f'  Validation: - Angle: {y[0][0]:.2f}\n' 
            val_values = metric_values[len(metric_values)//2:]
            print_str += '    ' + compose_str(val_values) 
            print(print_str)
            for gpu_id in range(len(self.strategy.extended.worker_devices)):
                printgpu(gpu_id=gpu_id)

        # val_classification_loss = metric_values[6] - metric_values[8] - metric_values[9] 
        # metric_values.append(val_classification_loss)
        # self.epoch_metric_values = {key: value + [metric_values[i]] for i, (key, value) in enumerate(self.epoch_metric_values.items())}
        for i, (key, value) in enumerate(self.epoch_metric_values.items()):
            if key not in ['sync']:
                self.epoch_metric_values[key] = value + [metric_values[i]]

        if 'val_loss' in self.metrics_keys:
            val_loss_index = self.metrics_keys.index('val_loss')
            val_loss_value = metric_values[val_loss_index]
        else:
            val_loss_index = self.metrics_keys.index('train_loss')
            val_loss_value = metric_values[val_loss_index]

        self.plot_losses_curves()
        
        # # save latest model every 10 epochs
        # if self.epoch % 10 == 0:
        #     self.save_latest_model()    

        if val_loss_value < self.min_val_loss:
        # if True:
            self.min_val_loss = val_loss_value
            self.no_improve_epochs = 0
            self.save_best_model()
            self.plot_mean_firing_rate_boxplot(v1_spikes, y)

            if v1_spikes_spont is not None:
                self.plot_spontaneous_boxplot(v1_spikes_spont, y)
                self.composed_raster(x, v1_spikes, x_spont, v1_spikes_spont, y)
                self.composed_raster(x, v1_spikes, x_spont, v1_spikes_spont, y, plot_core_only=False)
                # self.plot_lgn_activity(x, x_spont)
                # self.plot_populations_activity(v1_spikes, v1_spikes_spont)
            else:
                self.plot_raster(x, v1_spikes, y)
            
            # self.model_variables_dict['Best'] = {var.name: var.numpy() for var in self.model.trainable_variables}
            self.model_variables_dict['Best'] = {
                var.name: var.numpy().astype(np.float16) if len(var.shape) == 1 else var[:, 0].numpy().astype(np.float16)
                for var in self.model.trainable_variables
            }
        else:
            self.no_improve_epochs += 1

        # Plot osi_dsi if only 1 run and the osi/dsi period is reached
        if self.flags.n_runs == 1 and (self.epoch % self.flags.osi_dsi_eval_period == 0 or self.epoch==1):
            self.plot_osi_dsi(parallel=False)
           
        with self.summary_writer.as_default():
            for k, v in zip(self.metrics_keys, metric_values):
                tf.summary.scalar(k, v, step=self.epoch)

        # EARLY STOPPING CONDITIONS
        if (0 < self.flags.max_time < (time() - self.epoch_init_time) / 3600):
            print(f'[ Maximum optimization time of {self.flags.max_time:.2f}h reached ]')
            stop = True
        elif self.no_improve_epochs >= 500:
            print("Early stopping: Validation loss has not improved for 50 epochs.")
            stop = True  
        else:
            stop = False

        return stop

    def on_step_start(self):
        self.step += 1
        self.step_init_time = time()
        # reset the gpu memory stat
        tf.config.experimental.reset_memory_stats('GPU:0')

    def on_step_end(self, train_values, y, verbose=True):
        self.step_running_time.append(time() - self.step_init_time)
        if verbose:
            print_str = f'  Step {self.step:2d}/{self.flags.steps_per_epoch} - Angle: {y[0][0]:.2f}\n'
            print_str += '    ' + compose_str(train_values)
            print(print_str)
            print(f'    Step running time: {time() - self.step_init_time:.2f}s')
            for gpu_id in range(len(self.strategy.extended.worker_devices)):
                printgpu(gpu_id=gpu_id)
         
    def save_latest_model(self):
        try:
            p = self.latest_manager.save(checkpoint_number=self.epoch)
            print(f'Latest model saved in {p}\n')    
        except:
            print("Saving failed. Maybe next time?")    

    def save_best_model(self):
        # self.step_counter.assign_add(1)
        print(f'[ Saving the model at epoch {self.epoch} ]')
        try:
            p = self.best_manager.save(checkpoint_number=self.epoch)
            print(f'Model saved in {p}\n')
        except:
            print("Saving failed. Maybe next time?")

    def plot_losses_curves(self):
        plotting_metrics = ['val_loss', 'val_osi_dsi_loss', 'val_rate_loss', 'val_voltage_loss', 'val_regularizer_loss', 'val_sync_loss']
        # plotting_metrics = ['val_loss', 'val_osi_dsi_loss', 'val_rate_loss', 'val_voltage_loss', 'val_regularizer_loss']
        images_dir = os.path.join(self.logdir, 'Loss_curves')
        os.makedirs(images_dir, exist_ok=True)

        # start_epoch = 6 if self.epoch > 5 else 1
        start_epoch = 1
        epochs = range(start_epoch, self.epoch + 1)

        fig, axs = plt.subplots(3, 2, figsize=(12, 18))
        axs = axs.ravel()  # Flatten the array for easy indexing
        for i, metric_key in enumerate(plotting_metrics):
            ax = axs[i]
            if metric_key == 'val_loss':
                color = 'red'
            else:
                color = 'blue'
            ax.plot(epochs, self.epoch_metric_values[metric_key][start_epoch-1:], color=color)
            ax.set_xlabel('Epoch')
            ax.set_ylabel(metric_key)
        plt.tight_layout()
        plt.savefig(os.path.join(images_dir, f'losses_curves_epoch.png'), dpi=300, transparent=False)
        plt.close()
    
    def plot_raster(self, x, v1_spikes, y):
        seq_len = v1_spikes.shape[1]
        images_dir = os.path.join(self.logdir, 'Raster_plots')
        os.makedirs(images_dir, exist_ok=True)
        graph = InputActivityFigure(
                                    self.network,
                                    self.flags.data_dir,
                                    images_dir,
                                    filename=f'Epoch_{self.epoch}',
                                    frequency=self.flags.temporal_f,
                                    stimuli_init_time=self.pre_delay,
                                    stimuli_end_time=seq_len-self.post_delay,
                                    reverse=False,
                                    plot_core_only=True,
                                    core_radius=self.flags.plot_core_radius,
                                    )
        graph(x, v1_spikes)

    def composed_raster(self, x, v1_spikes, x_spont, v1_spikes_spont, y, plot_core_only=True):
        # concatenate the normal and spontaneous arrays
        x = np.concatenate((x_spont, x), axis=1)
        v1_spikes = np.concatenate((v1_spikes_spont, v1_spikes), axis=1)
        seq_len = v1_spikes.shape[1]
        images_dir = os.path.join(self.logdir, 'Raster_plots')
        if plot_core_only:
            images_dir = os.path.join(images_dir, 'Core_only')
        else:
            images_dir = os.path.join(images_dir, 'Full')
        os.makedirs(images_dir, exist_ok=True)
        graph = InputActivityFigure(
                                    self.network,
                                    self.flags.data_dir,
                                    images_dir,
                                    filename=f'Epoch_{self.epoch}_complete',
                                    frequency=self.flags.temporal_f,
                                    stimuli_init_time=seq_len+self.pre_delay,
                                    stimuli_end_time=2*seq_len-self.post_delay,
                                    reverse=False,
                                    plot_core_only=plot_core_only,
                                    core_radius=self.flags.plot_core_radius,
                                    )
        graph(x, v1_spikes)

    def plot_lgn_activity(self, x, x_spont):
        x = x[0, :, :]
        x_spont = x_spont[0, :, :]
        x = np.concatenate((x_spont, x), axis=0)
        x_mean = np.mean(x, axis=1)
        plt.figure(figsize=(10, 5))
        plt.plot(x_mean)
        plt.title('Mean input activity')
        plt.xlabel('Time (ms)')
        plt.ylabel('Mean input activity')
        os.makedirs(os.path.join(self.logdir, 'Populations activity'), exist_ok=True)
        plt.savefig(os.path.join(self.logdir, 'Populations activity', f'LGN_population_activity_epoch_{self.epoch}.png'))
        plt.close()

    def plot_populations_activity(self, v1_spikes, v1_spikes_spont):
        v1_spikes = np.concatenate((v1_spikes_spont, v1_spikes), axis=1)
        seq_len = v1_spikes.shape[1]
        # Plot the mean firing rate of the population of neurons
        filename = f'Epoch_{self.epoch}'
        Population_activity = PopulationActivity(n_neurons=self.n_neurons, network=self.network, 
                                                stimuli_init_time=self.pre_delay, stimuli_end_time=seq_len-self.post_delay, 
                                                image_path=self.logdir, filename=filename, data_dir=self.flags.data_dir,
                                                core_radius=self.flags.plot_core_radius)
        Population_activity(v1_spikes, plot_core_only=True, bin_size=10)

    def plot_mean_firing_rate_boxplot(self, v1_spikes, y):
        seq_len = v1_spikes.shape[1]
        DG_angles = y
        boxplots_dir = os.path.join(self.logdir, f'Boxplots/{self.neuropixels_feature}')
        os.makedirs(boxplots_dir, exist_ok=True)        
        if self.neuropixels_feature == "Ave_Rate(Hz)":
            metrics_analysis = ModelMetricsAnalysis(v1_spikes, DG_angles, self.network, data_dir=self.flags.data_dir, 
                                                    drifting_gratings_init=self.pre_delay, drifting_gratings_end=seq_len-self.post_delay,
                                                    core_radius=self.flags.plot_core_radius, df_directory=self.logdir, save_df=False) 
        elif self.neuropixels_feature == 'Spontaneous rate (Hz)':
            metrics_analysis = ModelMetricsAnalysis(v1_spikes, DG_angles, self.network, data_dir=self.flags.data_dir, 
                                                    spontaneous_init=self.pre_delay, spontaneous_end=seq_len-self.post_delay,
                                                    core_radius=self.flags.plot_core_radius, df_directory=self.logdir, save_df=False) 
        # Figure for Average firing rate boxplots      
        metrics_analysis(metrics=[self.neuropixels_feature], directory=boxplots_dir, filename=f'Epoch_{self.epoch}')    
                
    def plot_spontaneous_boxplot(self, v1_spikes, y):
        DG_angles = y
        seq_len = v1_spikes.shape[1]
        boxplots_dir = os.path.join(self.logdir, f'Boxplots/Spontaneous')
        os.makedirs(boxplots_dir, exist_ok=True)
        metrics_analysis = ModelMetricsAnalysis(v1_spikes, DG_angles, self.network, data_dir=self.flags.data_dir, 
                                                spontaneous_init=self.pre_delay, spontaneous_end=seq_len-self.post_delay,
                                                core_radius=self.flags.plot_core_radius, df_directory=self.logdir, save_df=False) 
        # Figure for Average firing rate boxplots
        metrics_analysis(metrics=['Spontaneous rate (Hz)'], directory=boxplots_dir, filename=f'Epoch_{self.epoch}')   
                       
    def variable_change_analysis(self, variable):
        if 'rest_of_brain_weights' in variable:
            self.node_to_pop_weights_analysis(self.bkg_input['indices'], variable=variable)
        elif 'sparse_input_weights' in variable:
            self.node_to_pop_weights_analysis(self.lgn_input['indices'], variable=variable)
        elif 'sparse_recurrent_weights' in variable:
            self.pop_to_pop_weights_analysis(self.network['synapses']['indices'], variable=variable)
        
    def node_to_pop_weights_analysis(self, indices, variable=''):
        pop_names = other_v1_utils.pop_names(self.network)
        target_cell_types = [other_v1_utils.pop_name_to_cell_type(pop_name) for pop_name in pop_names]
        post_indices = indices[:, 0]
        post_cell_types = [target_cell_types[i] for i in post_indices]

        # Create DataFrame with all the necessary data
        df = pd.DataFrame({
            'Cell type': post_cell_types * 2,  # Duplicate node names for initial and final weights
            'Weight': self.model_variables_dict['Initial'][variable].tolist() + self.model_variables_dict['Best'][variable].tolist(),  # Combine initial and final weights
            'State': ['Initial'] * len(self.model_variables_dict['Initial'][variable]) + ['Final'] * len(self.model_variables_dict['Best'][variable])  # Distinguish between initial and final weights
        })

        # Sort the dataframe by Node Name and then by Type to ensure consistent order
        df = df.sort_values(['Cell type', 'State'])

        # Plotting
        boxplots_dir = os.path.join(self.logdir, f'Boxplots/{variable}')
        os.makedirs(boxplots_dir, exist_ok=True)
        fig, axs = plt.subplots(2, 1, figsize=(12, 14))

        fig = plt.figure(figsize=(12, 6))
        hue_order = ['Initial', 'Final']
        # sns.boxplot(x='Node Name', y='Weight Change', data=df)
        sns.barplot(x='Cell type', y='Weight', hue='State', hue_order=hue_order, data=df)
        # sns.boxplot(x='Node Name', y='Weight', hue='Type', hue_order=hue_order, data=df)
        # plt.axhline(0, color='black', linewidth=1)  # include a horizontal black line at 0
        plt.xticks(rotation=90)  # Rotate x-axis labels for better readability
        plt.title(f'{variable}')
        plt.tight_layout()
        plt.savefig(os.path.join(boxplots_dir, f'{variable}.png'), dpi=300, transparent=False)
        plt.close()

    def pop_to_pop_weights_analysis(self, indices, variable=''):
        pop_names = other_v1_utils.pop_names(self.network)
        cell_types = [other_v1_utils.pop_name_to_cell_type(pop_name) for pop_name in pop_names]
        post_cell_types = [cell_types[i] for i in indices[:, 0]]
        pre_cell_types = [cell_types[i] for i in indices[:, 1]]

        ### Initial Weight ###
        weight_changes = self.model_variables_dict['Best'][variable] - self.model_variables_dict['Initial'][variable]
        df = pd.DataFrame({'Post Name': post_cell_types, 'Pre_names':pre_cell_types, 
                            'Initial weight': self.model_variables_dict['Initial'][variable], 'Final weight': self.model_variables_dict['Best'][variable], 
                            'Weight Change': weight_changes})
        
        # Calculate global min and max for color normalization
        global_grouped_df = df.groupby(['Pre_names', 'Post Name'])[['Initial weight', 'Final weight']].mean().reset_index()
        global_min = global_grouped_df[['Initial weight', 'Final weight']].min().min()
        global_max = global_grouped_df[['Initial weight', 'Final weight']].max().max()

        # Plot for Initial Weight
        grouped_df = df.groupby(['Pre_names', 'Post Name'])['Initial weight'].mean().reset_index()
        # Create a pivot table to reshape the data for the heatmap
        pivot_df = grouped_df.pivot(index='Pre_names', columns='Post Name', values='Initial weight')
        # Plot heatmap
        boxplots_dir = os.path.join(self.logdir, f'Boxplots/{variable}')
        os.makedirs(boxplots_dir, exist_ok=True)
        fig = plt.figure(figsize=(12, 6))
        heatmap = sns.heatmap(pivot_df, cmap='RdBu_r', annot=False, cbar=False, center=0, vmin=global_min, vmax=global_max)
        # heatmap = sns.heatmap(pivot_df, cmap='RdBu_r', annot=False, cbar=False, center=0, vmin=-0.5, vmax=0.5)
        plt.xticks(rotation=90)
        plt.gca().set_aspect('equal')
        plt.title(f'{variable}')
        # Create a separate color bar axis
        cbar_ax = plt.gcf().add_axes([0.92, 0.15, 0.02, 0.55])  # [left, bottom, width, height]
        # Plot color bar
        cbar = plt.colorbar(heatmap.collections[0], cax=cbar_ax)
        cbar.set_label('Initial Weight')
        plt.savefig(os.path.join(boxplots_dir, f'Initial_weight.png'), dpi=300, transparent=False)
        plt.close()

        ### Final Weight ###
        grouped_df = df.groupby(['Pre_names', 'Post Name'])['Final weight'].mean().reset_index()
        # Create a pivot table to reshape the data for the heatmap
        pivot_df = grouped_df.pivot(index='Pre_names', columns='Post Name', values='Final weight')
        # Plot heatmap
        plt.figure(figsize=(12, 6))
        heatmap = sns.heatmap(pivot_df, cmap='RdBu_r', annot=False, cbar=False, center=0, vmin=global_min, vmax=global_max)
        # heatmap = sns.heatmap(pivot_df, cmap='RdBu_r', annot=False, cbar=False, center=0, vmin=-0.5, vmax=0.5)
        plt.xticks(rotation=90)
        plt.gca().set_aspect('equal')
        plt.title(f'{variable}')
        # Create a separate color bar axis
        cbar_ax = plt.gcf().add_axes([0.92, 0.15, 0.02, 0.55])  # [left, bottom, width, height]
        # Plot color bar
        cbar = plt.colorbar(heatmap.collections[0], cax=cbar_ax)
        cbar.set_label('Final Weight')
        plt.savefig(os.path.join(boxplots_dir, f'Final_weight.png'), dpi=300, transparent=False)
        plt.close()

        ### Weight change ###
        grouped_df = df.groupby(['Pre_names', 'Post Name'])['Weight Change'].mean().reset_index()
        # Create a pivot table to reshape the data for the heatmap
        pivot_df = grouped_df.pivot(index='Pre_names', columns='Post Name', values='Weight Change')
        # Plot heatmap
        plt.figure(figsize=(12, 6))
        heatmap = sns.heatmap(pivot_df, cmap='RdBu_r', annot=False, cbar=False, center=0, vmin=global_min, vmax=global_max)
        # heatmap = sns.heatmap(pivot_df, cmap='RdBu_r', annot=False, cbar=False)
        plt.xticks(rotation=90)
        plt.gca().set_aspect('equal')
        plt.title(f'{variable}')
        # Create a separate color bar axis
        cbar_ax = plt.gcf().add_axes([0.92, 0.15, 0.02, 0.55])  # [left, bottom, width, height]
        # Plot color bar
        cbar = plt.colorbar(heatmap.collections[0], cax=cbar_ax)
        cbar.set_label('Weight Change')
        plt.savefig(os.path.join(boxplots_dir, f'Weight_change.png'), dpi=300, transparent=False)
        plt.close()

    def plot_osi_dsi(self, parallel=False):
        print('Starting to plot OSI and DSI...')
        # Save the checkpoint to reload weights in the osi_dsi_estimator
        if parallel:
            p = self.epoch_manager.save(checkpoint_number=self.epoch)
            print(f'Checkpoint model saved in {p}\n')
        else:              
             # osi_dsi_data_set = self.strategy.distribute_datasets_from_function(self.get_osi_dsi_dataset_fn(regular=True))
            DG_angles = np.arange(0, 360, 45)
            osi_dataset_path = os.path.join('OSI_DSI_dataset', 'lgn_firing_rates.pkl')
            if not os.path.exists(osi_dataset_path):
                print('Creating OSI/DSI dataset...')
                # Define OSI/DSI dataset
                def get_osi_dsi_dataset_fn(regular=False):
                    def _f(input_context):
                        post_delay = self.flags.seq_len - (2500 % self.flags.seq_len)
                        _lgn_firing_rates = stim_dataset.generate_drifting_grating_tuning(
                            seq_len=2500+post_delay,
                            pre_delay=500,
                            post_delay = post_delay,
                            n_input=self.flags.n_input,
                            regular=regular,
                            return_firing_rates=True,
                            rotation=self.flags.rotation,
                        ).batch(1)
                                    
                        return _lgn_firing_rates
                    return _f
            
                osi_dsi_data_set = self.strategy.distribute_datasets_from_function(get_osi_dsi_dataset_fn(regular=True))
                test_it = iter(osi_dsi_data_set)
                lgn_firing_rates_dict = {}  # Dictionary to store firing rates
                for angle_id, angle in enumerate(DG_angles):
                    t0 = time()
                    lgn_firing_rates = next(test_it)
                    lgn_firing_rates_dict[angle] = lgn_firing_rates.numpy()
                    print(f'Angle {angle} done.')
                    print(f'    Trial running time: {time() - t0:.2f}s')
                    for gpu_id in range(len(self.strategy.extended.worker_devices)):
                        printgpu(gpu_id=gpu_id)

                # Save the dataset      
                results_dir = os.path.join("OSI_DSI_dataset")
                os.makedirs(results_dir, exist_ok=True)
                with open(osi_dataset_path, 'wb') as f:
                    pkl.dump(lgn_firing_rates_dict, f)
                print('OSI/DSI dataset created successfully!')

            else:
                # Load the LGN firing rates dataset
                with open(osi_dataset_path, 'rb') as f:
                    lgn_firing_rates_dict = pkl.load(f)

            callbacks = OsiDsiCallbacks(self.network, self.lgn_input, self.bkg_input, self.flags, self.logdir, current_epoch=self.epoch,
                                        pre_delay=500, post_delay=500, model_variables_init=None)

            sim_duration = (2500//self.flags.seq_len + 1) * self.flags.seq_len
            n_trials_per_angle = 3
            spikes = np.zeros((n_trials_per_angle, len(DG_angles), sim_duration, self.flags.neurons), dtype=float)
            
            for angle_id, angle in enumerate(DG_angles):
                # load LGN firign rates for the given angle and calculate spiking probability
                lgn_fr = lgn_firing_rates_dict[angle]
                lgn_fr = tf.constant(lgn_fr, dtype=tf.float32)
                _p = 1 - tf.exp(-lgn_fr / 1000.)

                # test_it = iter(osi_dsi_data_set)
                for trial_id in range(n_trials_per_angle):
                    t0 = time()
                    # Reset the memory stats
                    tf.config.experimental.reset_memory_stats('GPU:0')
                    # Generate LGN spikes
                    x = tf.random.uniform(tf.shape(_p)) < _p
                    y = tf.constant(angle, dtype=tf.float32, shape=(1,1))
                    w = tf.constant(sim_duration, dtype=tf.float32, shape=(1,))

                    # x, y, _, w = next(test_it)
                    chunk_size = self.flags.seq_len
                    num_chunks = (2500//chunk_size + 1)
                    for i in range(num_chunks):
                        chunk = x[:, i * chunk_size : (i + 1) * chunk_size, :]
                        _out, _, _, _, _  = self.distributed_roll_out(chunk, y, w)
                        spikes_chunk = _out[0][0]
                        spikes[trial_id, angle_id, i * chunk_size : (i + 1) * chunk_size, :] += spikes_chunk.numpy()[0, :, :].astype(np.uint8)

                    if trial_id == 0 and angle_id == 0:
                        # Raster plot for 0 degree orientation
                        callbacks.single_trial_callbacks(x.numpy(), spikes[0], y=angle)

                    print(f'Trial {trial_id}/{n_trials_per_angle} - Angle {angle} done.')
                    print(f'    Trial running time: {time() - t0:.2f}s')
                    for gpu_id in range(len(self.strategy.extended.worker_devices)):
                        printgpu(gpu_id=gpu_id)
        
            # Do the OSI/DSI analysis    
            callbacks.osi_dsi_analysis(spikes, DG_angles)   