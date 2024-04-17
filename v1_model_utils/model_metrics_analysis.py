# -*- coding: utf-8 -*-
"""
Created on Thu Dec  8 17:28:39 2022

@author: UX325
"""

import os
import sys
import absl
import pandas as pd
import numpy as np
import math
import tensorflow as tf
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle
sys.path.append(os.path.join(os.getcwd(), "v1_model_utils"))
import other_v1_utils
import load_sparse

mpl.style.use('default')
# rd = np.random.RandomState(seed=42)


def calculate_Firing_Rate(z, drifting_gratings_init=500, drifting_gratings_end=2500):
    dg_spikes = z[:, drifting_gratings_init:drifting_gratings_end, :]
    # if the number of dimensions of dg_spikes is 2, reshape it to 3 adding an additional first dimension
    # if dg_spikes.ndim == 2:
    #     dg_spikes = dg_spikes.reshape(1, dg_spikes.shape[0], dg_spikes.shape[1])
    mean_dg_spikes = np.mean(dg_spikes, axis=0)
    mean_firing_rates = np.sum(mean_dg_spikes, axis=0)/((drifting_gratings_end-drifting_gratings_init)/1000)
    
    return mean_firing_rates

def calculate_OSI_DSI(rates_df, network, DG_angles=range(0,360, 45), n_selected_neurons=None, core_radius=None, remove_zero_rate_neurons=False):
    
    # Get the pop names of the neurons
    if n_selected_neurons is not None:
        pop_names = other_v1_utils.pop_names(network, n_selected_neurons=n_selected_neurons) 
    elif core_radius is not None:
        pop_names = other_v1_utils.pop_names(network, core_radius=core_radius)
    else:
        pop_names = other_v1_utils.pop_names(network)

    # Get the number of neurons and DG angles
    n_neurons = len(pop_names)
    node_ids = np.arange(n_neurons)
    
    # Get the firing rates for every neuron and DG angle
    all_rates = np.array([g["Ave_Rate(Hz)"] for _, g in rates_df.groupby("DG_angle")]).T
    average_rates = np.mean(all_rates, axis=1)

    # Find the preferred DG angle for each neuron
    preferred_angle_ind = np.argmax(all_rates, axis=1)
    preferred_rates = np.max(all_rates, axis=1)
    preferred_DG_angle = np.array(DG_angles)[preferred_angle_ind]

    # Calculate the DSI and OSI
    phase_rad = np.array(DG_angles) * np.pi / 180.0
    denominator = all_rates.sum(axis=1)
    dsi = np.where(denominator != 0, 
               np.abs((all_rates * np.exp(1j * phase_rad)).sum(axis=1)) / denominator, 
               np.nan)
    osi = np.where(denominator != 0,
                np.abs((all_rates * np.exp(2j * phase_rad)).sum(axis=1)) / denominator,
                np.nan)

    # Save the results in a dataframe
    osi_df = pd.DataFrame()
    osi_df["node_id"] = node_ids
    osi_df["pop_name"] = pop_names
    osi_df["DSI"] = dsi
    osi_df["OSI"] = osi
    osi_df["preferred_angle"] = preferred_DG_angle
    osi_df["max_mean_rate(Hz)"] = preferred_rates
    osi_df["Ave_Rate(Hz)"] = average_rates
    # osi_df['firing_rate_sp'] = average_rates

    if remove_zero_rate_neurons:
        osi_df = osi_df[osi_df["Ave_Rate(Hz)"] != 0]

    return osi_df


class ModelMetricsAnalysis:    

    def __init__(self, network, neuropixels_feature="Ave_Rate(Hz)", data_dir='GLIF_network', directory='', filename='', n_trials=1, drifting_gratings_init=50, 
                 drifting_gratings_end=550, analyze_core_only=True):
        self.n_neurons = network['n_nodes']
        self.network = network
        self.data_dir = data_dir 
        self.n_trials = n_trials
        self.drifting_gratings_init = drifting_gratings_init
        self.drifting_gratings_end = drifting_gratings_end
        self.analyze_core_only = analyze_core_only
        self.neuropixels_feature = neuropixels_feature
        self.directory = os.path.join(directory)
        self.filename = filename
    
    def __call__(self, spikes, DG_angles, axis=None):

        # Isolate the core neurons if necessary
        if self.analyze_core_only:
            core_neurons = 65871
            core_radius = 400
            # n_neurons_plot = 65871

            # Calculate the core_neurons mask
            if self.n_neurons > core_neurons:
                self.core_mask = other_v1_utils.isolate_core_neurons(self.network, radius=core_radius, data_dir=self.data_dir)
                # self.core_mask = other_v1_utils.isolate_core_neurons(self.network, n_selected_neurons=core_neurons, data_dir=self.data_dir) 
                # self.n_neurons = core_neurons
                # if n_neurons is overridden, it won't run for the second time...
                n_neurons_plot = core_neurons
            else:
                self.core_mask = np.full(self.n_neurons, True)
                n_neurons_plot = self.n_neurons
        else:
            self.core_mask = np.full(self.n_neurons, True)
            # core_radius = None
            n_neurons_plot = self.n_neurons

        spikes = spikes[:, :, self.core_mask]
       
        # Calculate the firing rates along every orientation            
        # if spikes shape is (n_angles, n_time_steps, n_neurons) reshape it to (n_angles, n_trials, n_time_steps, n_neurons)
        if spikes.shape[0] == len(DG_angles):
            spikes = spikes.reshape(len(DG_angles), self.n_trials, spikes.shape[-2], n_neurons_plot)
        
        firing_rates_df = self.create_firing_rates_df(n_neurons_plot, spikes, n_trials=self.n_trials, 
                                                      drifting_gratings_init=self.drifting_gratings_init, drifting_gratings_end=self.drifting_gratings_end, 
                                                      DG_angles=DG_angles)
        
        # Save the firing rates
        # firing_rates_df.to_csv(os.path.join(self.save_dir, f"V1_DG_firing_rates_df.csv"), sep=" ", index=False)

        # Calculate the orientation and direction selectivity indices
        metrics_df = calculate_OSI_DSI(firing_rates_df, self.network, DG_angles=DG_angles, n_selected_neurons=n_neurons_plot,
                                        remove_zero_rate_neurons=True)
        # metrics_df.to_csv(os.path.join(self.directory, f"V1_OSI_DSI_DF.csv"), sep=" ", index=False)

        # Make the boxplots to compare with the neuropixels data
        if len(DG_angles) == 1:
            metrics = [self.neuropixels_feature]
        else:
            metrics = ["Rate at preferred direction (Hz)", "OSI", "DSI"]

        boxplot = MetricsBoxplot(save_dir=self.directory, filename=self.filename)
        boxplot.plot(metrics=metrics, metrics_df=metrics_df, axis=axis)

    def create_firing_rates_df(self, n_neurons, spikes, n_trials=10, drifting_gratings_init=50, 
                               drifting_gratings_end=550, DG_angles=np.arange(0, 360, 45)):
        
        # Calculate the firing rates for each neuron in each orientation
        firing_rates_df = pd.DataFrame(columns=["DG_angle", "node_id", "Ave_Rate(Hz)"])
        node_ids = np.arange(n_neurons)

        # Iterate through each orientation
        for angle_id, angle in enumerate(DG_angles): 
            # Load the simulation results
            firingRates = calculate_Firing_Rate(spikes[angle_id, :, :, :], drifting_gratings_init=drifting_gratings_init, drifting_gratings_end=drifting_gratings_end)
            data = {
                    "DG_angle": float(angle),
                    "node_id": node_ids,
                    "Ave_Rate(Hz)": firingRates
                }
            df = pd.DataFrame(data)
            # Drop empty or all-NA columns before concatenation
            df = df.dropna(axis=1, how='all')
            # how many nan rows are there?
            firing_rates_df = pd.concat([firing_rates_df, df], ignore_index=True)

        return firing_rates_df
    

class MetricsBoxplot:
    def __init__(self, save_dir='Metrics_analysis', filename=''):
        self.save_dir = save_dir
        self.filename = filename
        self.osi_dfs = []

    @staticmethod
    def pop_name_to_cell_type(pop_name):
        # Convert pop_name in the old format to cell types. E.g., 'e4Rorb' -> 'L4 Exc', 'i4Pvalb' -> 'L4 PV', 'i23Sst' -> 'L2/3 SST'
        shift = 0  # letter shift for L23
        layer = pop_name[1]
        if layer == "2":
            layer = "2/3"
            shift = 1
        elif layer == "1":
            return "L1 Htr3a"  # special case

        class_name = pop_name[2 + shift :]
        if class_name == "Pvalb":
            subclass = "PV"
        elif class_name == "Sst":
            subclass = "SST"
        elif (class_name == "Vip") or (class_name == "Htr3a"):
            subclass = "VIP"
        else:  # excitatory
            subclass = "Exc"

        return f"L{layer} {subclass}"

    @staticmethod
    def neuropixels_cell_type_to_cell_type(pop_name):
        # Convert pop_name in the neuropixels cell type to cell types. E.g, 'EXC_L23' -> 'L2/3 Exc', 'PV_L5' -> 'L5 PV'
        layer = pop_name.split('_')[1]
        class_name = pop_name.split('_')[0]
        if "2" in layer:
            layer = "L2/3"
        elif layer == "L1":
            return "L1 Htr3a"  # special case
        if class_name == "EXC":
            class_name = "Exc"

        return f"{layer} {class_name}"

    @staticmethod
    def get_borders(ticklabel):
        prev_layer = "1"
        borders = [-0.5]
        for i in ticklabel:
            x = i.get_position()[0]
            text = i.get_text()
            if text[1] != prev_layer:
                borders.append(x - 0.5)
                prev_layer = text[1]
        borders.append(x + 0.5)
        return borders

    @staticmethod
    def draw_borders(ax, borders, ylim):
        for i in range(0, len(borders), 2):
            w = borders[i + 1] - borders[i]
            h = ylim[1] - ylim[0]
            ax.add_patch(
                Rectangle((borders[i], ylim[0]), w, h, alpha=0.08, color="k", zorder=-10)
            )
        return ax   

    def get_osi_dsi_df(self, metric_file="v1_OSI_DSI_DF.csv", data_source_name="", feature='', data_dir=""):
        # Load the data csv file and remove rows with empty cell type
        # if metric_file is a dataframe, then do not load it
        if isinstance(metric_file, pd.DataFrame):
            df = metric_file
        else:
            df = pd.read_csv(f"{data_dir}/{metric_file}", sep=" ")

        # Rename the cell types
        if data_dir == "Neuropixels_data":
            df = df[df['cell_type'].notna()]
            df["cell_type"] = df["cell_type"].apply(self.neuropixels_cell_type_to_cell_type)
        elif data_dir == 'Billeh_column_metrics':
            df["cell_type"] = df["pop_name"].apply(self.pop_name_to_cell_type)
        elif data_dir == "NEST_metrics":
            df["cell_type"] = df["pop_name"].apply(self.pop_name_to_cell_type)
            # plot only neurons within 200 um.
            df = df[(df["x"] ** 2 + df["z"] ** 2) < (200 ** 2)]
        else:
            df["cell_type"] = df["pop_name"].apply(self.pop_name_to_cell_type)

        # Rename the maximum rate column
        df.rename(columns={"max_mean_rate(Hz)": "Rate at preferred direction (Hz)"}, inplace=True)
        # df.rename(columns={"firing_rate_sp": "Spontaneous rate (Hz)"}, inplace=True)
        # df.rename(columns={"Ave_Rate(Hz)": "Average rate (Hz)"}, inplace=True)

        # Cut off neurons with low firing rate at the preferred direction
        nonresponding = df["Rate at preferred direction (Hz)"] < 0.5
        df.loc[nonresponding, "OSI"] = np.nan
        df.loc[nonresponding, "DSI"] = np.nan

        # Sort the neurons by neuron types
        df = df.sort_values(by="cell_type")

        # Add a column for the data source name
        if len(data_source_name) > 0:
            df["data_type"] = data_source_name
        else:
            df["data_type"] = data_dir

        # columns = ["cell_type", "data_type", "Rate at preferred direction (Hz)", "OSI", "DSI", 'Ave_Rate(Hz)', 'Spontaneous rate (Hz)']
        columns = ["cell_type", "data_type", "Rate at preferred direction (Hz)", "OSI", "DSI", 'Ave_Rate(Hz)']
        df = df[columns]

        return df

    def plot(self, metrics=["Rate at preferred direction (Hz)", "OSI", "DSI"], metrics_df=None, axis=None):
        # Get the dataframes for the model and Neuropixels OSI and DSI 
        if metrics_df is None:
            metrics_df = f"v1_OSI_DSI_DF.csv"

        self.osi_dfs.append(self.get_osi_dsi_df(metric_file=metrics_df, data_source_name="V1 GLIF model", data_dir=self.save_dir))
        self.osi_dfs.append(self.get_osi_dsi_df(metric_file=f"v1_OSI_DSI_DF.csv", data_source_name="Neuropixels", data_dir='Neuropixels_data'))
        # self.osi_dfs.append(self.get_osi_dsi_df(metric_file=f"V1_OSI_DSI_DF.csv", data_source_name="Billeh et al (2020)", data_dir='Billeh_column_metrics'))
        # self.osi_dfs.append(self.get_osi_dsi_df(metric_file=f"v1_OSI_DSI_DF_pop_name.csv", data_source_name="NEST simulation", data_dir='NEST_metrics'))
        self.osi_dfs.append(self.get_osi_dsi_df(metric_file=f"V1_OSI_DSI_DF_pop_name.csv", data_source_name="NEST simulation", data_dir='NEST_metrics'))
        df = pd.concat(self.osi_dfs, ignore_index=True)
        # df.to_csv(os.path.join('Borrar', f"help_DG_firing_rates_df.csv"), sep=" ", index=False)

        # Create a figure to compare several model metrics against Neuropixels data
        n_metrics = len(metrics)
        height = int(7*n_metrics)

        if axis is None:
            fig, axs = plt.subplots(n_metrics, 1, figsize=(12, height))
            # fig, axs = plt.subplots(n_metrics, 1, figsize=(12, 20))
            if n_metrics == 1:
                axs = [axs]
        else:
            axs = [axis]

        color_pal = {
            "V1 GLIF model": "tab:orange",
            "Neuropixels": "tab:gray",
            "Billeh et al (2020)": "tab:blue",
            "NEST simulation": "tab:pink"
        }

        # Establish the order of the neuron types in the boxplots
        cell_type_order = np.sort(df['cell_type'].unique())

        for idx, metric in enumerate(metrics):
            if metric in ["Rate at preferred direction (Hz)", 'Ave_Rate(Hz)']: #, 'Spontaneous rate (Hz)']:
                ylims = [0, 100]
            else:
                ylims = [0, 1]

            plot_one_metric(axs[idx], df, metric, ylims, color_pal, hue_order=cell_type_order)

        axs[0].legend(loc="upper right", fontsize=16)
        # axs[0].set_title(f"V1", fontsize=20)
        if len(axs) > 1:
            for i in range(len(axs)-1):
                axs[i].set_xticklabels([])

        xticklabel = axs[n_metrics-1].get_xticklabels()
        for label in xticklabel:
            label.set_fontsize(14)
            label.set_weight("bold")

        if axis is None:
            plt.tight_layout()
            os.makedirs(self.save_dir, exist_ok=True)
            plt.savefig(os.path.join(self.save_dir, self.filename+'.png'), dpi=300, transparent=False)
            plt.close()


def plot_one_metric(ax, df, metric_name, ylim, cpal=None, hue_order=None):

    sns.boxplot(
        x="cell_type",
        y=metric_name,
        hue="data_type",
        order=hue_order,
        data=df,
        ax=ax,
        width=0.7,
        palette=cpal,
    )
    ax.tick_params(axis="x", labelrotation=90)
    ax.set_ylim(ylim)
    ax.set_xlabel("")
    # Modify the label sizes
    ax.set_ylabel(metric_name, fontsize=16)
    yticklabel = ax.get_yticklabels()
    for label in yticklabel:
        label.set_fontsize(14)

    # Apply shadings to each layer
    xticklabel = ax.get_xticklabels()
    borders = MetricsBoxplot.get_borders(xticklabel)
    MetricsBoxplot.draw_borders(ax, borders, ylim)

    # Hide the legend
    ax.get_legend().remove()

    return ax



class OneShotTuningAnalysis:
    def __init__(self, network, data_dir='GLIF_network', directory='', drifting_gratings_init=50, 
                 drifting_gratings_end=550, analyze_core_only=True):
        self.n_neurons = network['n_nodes']
        self.network = network
        self.data_dir = data_dir 
        self.drifting_gratings_init = drifting_gratings_init
        self.drifting_gratings_end = drifting_gratings_end
        self.analyze_core_only = analyze_core_only
        self.directory = os.path.join(directory)
        os.makedirs(self.directory, exist_ok=True)

    def __call__(self, spikes, current_orientation):
        self.current_orientation = current_orientation[0][0]
        
        # Isolate the core neurons if necessary
        if self.analyze_core_only:
            core_neurons = 65871
            core_radius = 400
            # n_neurons_plot = 65871

            # Calculate the core_neurons mask
            if self.n_neurons > core_neurons:
                self.core_mask = other_v1_utils.isolate_core_neurons(self.network, radius=core_radius, data_dir=self.data_dir)
                # self.core_mask = other_v1_utils.isolate_core_neurons(self.network, n_selected_neurons=core_neurons, data_dir=self.data_dir) 
                # self.n_neurons = core_neurons
                # if n_neurons is overridden, it won't run for the second time...
                n_neurons_plot = core_neurons
            else:
                self.core_mask = np.full(self.n_neurons, True)
                n_neurons_plot = self.n_neurons
        else:
            self.core_mask = np.full(self.n_neurons, True)
            # core_radius = None
            n_neurons_plot = self.n_neurons

        spikes = spikes[:, :, self.core_mask]
        # Calculate the firing rates along every orientation
        # Calculate the firing rates for each neuron in the given configuration
        self.firing_rate = calculate_Firing_Rate(spikes, drifting_gratings_init=self.drifting_gratings_init, drifting_gratings_end=self.drifting_gratings_end)
        self.tuning_angles = self.network['tuning_angle'][self.core_mask]
        self.pop_names = other_v1_utils.pop_names(self.network, n_selected_neurons=n_neurons_plot)
        self.cell_types = np.array([MetricsBoxplot.pop_name_to_cell_type(x) for x in self.pop_names])

        # Get the orientation angles and orientation assignments using the 'assign_orientation' method
        self.orientation_angles, self.orientation_assignments = self.assign_orientation(self.tuning_angles, n_neurons_plot)

    def assign_orientation(self, tuning_angles, n_neurons):
        # Assign each neuron to an orientation based on its preferred angle 
        orientation_angles = np.arange(0, 360, 45)
        n_angles = len(orientation_angles)
        orientation_assignments = np.zeros((n_neurons, n_angles)).astype(np.bool_)
        for i, angle in enumerate(orientation_angles):
            circular_diff = (tuning_angles - angle) % 360
            circular_diff = np.abs(np.minimum(circular_diff, 360 - circular_diff))
            orientation_assignments[:, i] = (circular_diff < 10).astype(np.bool_)

        return orientation_angles, orientation_assignments

    def plot_tuning_curves(self, epoch, remove_zero_rate_neurons=True):
        # Define the grid size
        nrows, ncols = 5, 4
        fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(20, 25))  # Share x-axis among all plots
        plt.subplots_adjust(hspace=0.4)  # Adjust space between plots

        # Flatten the array of axes for easy iteration
        axs = axs.flatten()
        unique_cell_types = sorted(set(self.cell_types))  # Ensure consistent order of cell types

        # Loop over each unique cell type to plot tuning curves separately for each type
        for cell_id, cell_type in enumerate(unique_cell_types):
            # Create a boolean mask to filter neurons of the current cell type
            cell_type_mask = self.cell_types == cell_type
            # Initialize dictionaries to store mean and standard deviation of firing rates for each orientation angle
            orientation_firing_rates_dict = {}
            orientation_firing_rates_std_dict = {}
            # Loop over the orientation angles to calculate mean and standard deviation of firing rates
            for i, angle in enumerate(self.orientation_angles):
                # Create a boolean mask to filter neurons of the current cell type and orientation
                angle_mask = np.logical_and(self.orientation_assignments[:, i], cell_type_mask)
                firing_rates_at_angle = self.firing_rate[angle_mask]
                if remove_zero_rate_neurons:
                    zero_rate_mask = firing_rates_at_angle != 0
                    firing_rates_at_angle = firing_rates_at_angle[zero_rate_mask]
                    # fr_std = fr_std[zero_rate_mask]
                    # filtered_orientation_angles = np.array(self.orientation_angles)[zero_rate_mask]   
                # else:
                #     filtered_orientation_angles = self.orientation_angles   
                # Calculate the mean and standard deviation of firing rates for the current orientation angle
                orientation_firing_rates_dict[angle] = np.mean(firing_rates_at_angle)
                orientation_firing_rates_std_dict[angle] = np.std(firing_rates_at_angle)

            # Extract the firing rates and standard deviations from dictionaries
            fr = np.array(list(orientation_firing_rates_dict.values()))
            fr_std = np.array(list(orientation_firing_rates_std_dict.values()))

            # if remove_zero_rate_neurons:
            #     zero_rate_mask = fr != 0
            #     fr = fr[zero_rate_mask]
            #     fr_std = fr_std[zero_rate_mask]
            #     filtered_orientation_angles = np.array(self.orientation_angles)[zero_rate_mask]   
            # else:
            #     filtered_orientation_angles = self.orientation_angles   

            # Plot the tuning curve as a line plot with error bars on the respective subplot
            ax = axs[cell_id]
            ax.errorbar(self.orientation_angles, fr, yerr=fr_std, color='black', fmt='-o', label=f'{cell_type} tuning')

            # Add vertical line for the current orientation
            ax.axvline(x=self.current_orientation, color='red', linestyle='--', label=f'Current orientation: {self.current_orientation:.2f}\u00b0')
            ax.legend()  # Show legend
            
            # Set the x-axis tick positions and labels to be the orientation angles
            ax.set_xticks(self.orientation_angles)
            ax.set_xlabel('Tuning angle')
            ax.set_ylim(bottom=0)
            
            # Set the title of the plot using the area and current cell type
            ax.set_title(f'{cell_type}')
            
            # Set the y-axis label
            ax.set_ylabel('Firing rate (Hz)')
            
        # Hide the unused subplots
        for i in range(len(unique_cell_types), nrows * ncols):
            fig.delaxes(axs[i])

        plt.tight_layout()
        # plt.suptitle(f'{self.area} Tuning Curves', fontsize=20, y=1.02)  # Add main title and adjust its position
        path = os.path.join(self.directory, f'Tuning_curves')
        os.makedirs(path, exist_ok=True)
        plt.savefig(os.path.join(path, f'epoch_{epoch}.png'), dpi=300, transparent=False)
        plt.close()

    def plot_max_rate_boxplots(self, epoch, remove_zero_rate_neurons=True, axis=None):
        # Create a DataFrame to store firing rates, preferred angles, and cell types.
        firing_rates_df = pd.DataFrame({'pop_name': self.pop_names, 'Rate at preferred direction (Hz)': np.full(len(self.firing_rate), np.nan)})
        circular_diff = (self.tuning_angles - self.current_orientation) % 360
        circular_diff = np.abs(np.minimum(circular_diff, 360 - circular_diff))
        # Isolate neurons that prefer the current orientation
        preferred_mask = (np.abs(circular_diff) < 10).astype(np.bool_)
        
        # Iterate over unique cell types to calculate preferred firing rates and store them in the DataFrame.
        unique_cell_types = sorted(set(self.cell_types))
        for i, cell_type in enumerate(unique_cell_types):
            # Create a boolean mask to filter neurons of the current cell type
            cell_type_mask = self.cell_types == cell_type
            mask = np.logical_and(preferred_mask, cell_type_mask)
            preferred_firing_rate = self.firing_rate[mask]
            firing_rates_df.loc[mask, "Rate at preferred direction (Hz)"] = preferred_firing_rate

        if remove_zero_rate_neurons:
            firing_rates_df = firing_rates_df[firing_rates_df["Rate at preferred direction (Hz)"] != 0]

        # Add a column to the DataFrame indicating the data type.
        firing_rates_df['data_type'] = 'V1 GLIF model'
        firing_rates_df['OSI'] = np.nan
        firing_rates_df['DSI'] = np.nan
        firing_rates_df['Ave_Rate(Hz)'] = np.nan
        # firing_rates_df['firing_rate_sp'] = np.nan

        # Create an instance of MetricsBoxplot and get OSI and DSI data from a file.
        filename = f'Epoch_{epoch}'
        boxplot = MetricsBoxplot(save_dir=self.directory, filename=filename)
        metrics = ["Rate at preferred direction (Hz)"]
        boxplot.plot(metrics=metrics, metrics_df=firing_rates_df, axis=axis)


# def main(_):
#     flags = absl.app.flags.FLAGS    
#     model_analysis = ModelMetricsAnalysis(flags)
#     model_analysis('v1')
#     model_analysis('lm')
    
# if __name__ == '__main__':
    
#     absl.app.flags.DEFINE_integer('v1_neurons', 230924, '')
#     absl.app.flags.DEFINE_integer('lm_neurons', 32940, '')
#     absl.app.flags.DEFINE_integer('gratings_frequency', 2, '')
#     absl.app.flags.DEFINE_integer('n_simulations', None, '')
#     absl.app.flags.DEFINE_integer('seed', 3000, '')
#     absl.app.flags.DEFINE_boolean('skip_first_simulation', False, '')
#     absl.app.flags.DEFINE_boolean('connected_selection', True, '')
#     absl.app.flags.DEFINE_boolean('caching', True, '')
#     absl.app.flags.DEFINE_boolean('core_only', False, '')
#     absl.app.flags.DEFINE_boolean('hard_reset', False, '')
#     absl.app.flags.DEFINE_string('interarea_weight_distribution', 'billeh_like', '')
#     absl.app.flags.DEFINE_boolean('analyze_core_only', True, '')
#     absl.app.flags.DEFINE_float('E4_weight_factor', 1., '')
#     absl.app.flags.DEFINE_boolean('disconnect_lm_L6_inhibition', False, '')
#     absl.app.flags.DEFINE_integer('n_input', 17400, '')
#     absl.app.flags.DEFINE_integer('seq_len', 3000, '')
#     absl.app.flags.DEFINE_string('data_dir', 'GLIF_network', '')
    
    
#     absl.app.run(main)  