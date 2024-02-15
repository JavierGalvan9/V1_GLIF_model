import os

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import patches
import seaborn as sns
from . import other_v1_utils, toolkit


class InputActivityFigure:
    def __init__(
        self,
        network,
        data_dir,
        images_dir="Images",
        filename="Raster_plot",
        batch_ind=0,
        scale=3.0,
        frequency=2,
        stimuli_init_time=500,
        stimuli_end_time=1500,
        reverse=False,
        plot_core_only=True,
    ):
        self.figure = plt.figure(
            figsize=toolkit.cm2inch((15 * scale, 11 * scale)))
        gs = self.figure.add_gridspec(10, 1)
        self.input_ax = self.figure.add_subplot(gs[:3])
        self.activity_ax = self.figure.add_subplot(gs[3:-1])
        self.drifting_grating_ax = self.figure.add_subplot(gs[-1])

        self.inputs_plot = RasterPlot(
            batch_ind=batch_ind,
            stimuli_init_time=stimuli_init_time,
            stimuli_end_time=stimuli_end_time,
            scale=scale,
            y_label="LGN Neuron ID",
            alpha=0.05,
        )
        self.laminar_plot = LaminarPlot(
            network,
            data_dir,
            batch_ind=batch_ind,
            stimuli_init_time=stimuli_init_time,
            stimuli_end_time=stimuli_end_time,
            scale=scale,
            alpha=0.2,
            plot_core_only=plot_core_only,
        )
        self.drifting_grating_plot = DriftingGrating(
            frequency=frequency,
            stimuli_init_time=stimuli_init_time,
            stimuli_end_time=stimuli_end_time,
            reverse=reverse,
            scale=scale,
        )

        self.tightened = True  # False
        self.scale = scale
        self.network = network
        self.n_neurons = self.network["n_nodes"]
        self.batch_ind = batch_ind
        self.plot_core_only = plot_core_only
        self.images_dir = images_dir
        self.filename = filename

    def __call__(self, inputs, spikes):
        self.input_ax.clear()
        self.activity_ax.clear()
        self.drifting_grating_ax.clear()

        self.inputs_plot(self.input_ax, inputs)
        self.input_ax.set_xticklabels([])
        toolkit.apply_style(self.input_ax, scale=self.scale)

        self.laminar_plot(self.activity_ax, spikes)
        self.activity_ax.set_xticklabels([])
        toolkit.apply_style(self.activity_ax, scale=self.scale)

        simulation_length = spikes.shape[1]
        self.drifting_grating_plot(self.drifting_grating_ax, simulation_length)
        toolkit.apply_style(self.drifting_grating_ax, scale=self.scale)

        if not self.tightened:
            self.figure.tight_layout()
            self.tightened = True

        self.figure.savefig(
            os.path.join(self.images_dir, self.filename), dpi=300, transparent=False
        )

        plt.close(self.figure)

        # return self.figure


class InputActivityFigureWithoutStimulus:
    def __init__(
        self,
        network,
        data_dir,
        images_dir="Images",
        filename="Raster_plot",
        batch_ind=0,
        scale=3.0,
        stimuli_init_time=500,
        stimuli_end_time=1500,
        plot_core_only=True,
    ):
        self.figure = plt.figure(
            figsize=toolkit.cm2inch((15 * scale, 11 * scale)))
        gs = self.figure.add_gridspec(10, 1)
        self.input_ax = self.figure.add_subplot(gs[:3])
        self.activity_ax = self.figure.add_subplot(gs[3:])

        self.inputs_plot = RasterPlot(
            batch_ind=batch_ind,
            stimuli_init_time=500,
            stimuli_end_time=1500,
            scale=scale,
            y_label="LGN Neuron ID",
            alpha=0.05,
        )
        self.laminar_plot = LaminarPlot(
            network,
            data_dir,
            batch_ind=batch_ind,
            stimuli_init_time=500,
            stimuli_end_time=1500,
            scale=scale,
            alpha=0.2,
            plot_core_only=plot_core_only,
        )

        self.tightened = True  # False
        self.scale = scale
        self.network = network
        self.n_neurons = self.network["n_nodes"]
        self.batch_ind = batch_ind
        self.plot_core_only = plot_core_only
        self.images_dir = images_dir
        self.filename = filename

    def __call__(self, inputs, spikes):
        self.input_ax.clear()
        self.activity_ax.clear()

        self.inputs_plot(self.input_ax, inputs)
        self.input_ax.set_xticklabels([])
        toolkit.apply_style(self.input_ax, scale=self.scale)

        self.laminar_plot(self.activity_ax, spikes)
        self.activity_ax.set_xticklabels([])
        toolkit.apply_style(self.activity_ax, scale=self.scale)

        # self.drifting_grating_plot(self.drifting_grating_ax, spikes)
        # toolkit.apply_style(self.drifting_grating_ax, scale=self.scale)

        if not self.tightened:
            self.figure.tight_layout()
            self.tightened = True

        self.figure.savefig(
            os.path.join(self.images_dir, self.filename), dpi=300, transparent=False
        )

        # return self.figure
        plt.close(self.figure)


def pop_ordering(x):
    if (
        x[1:3].count("23") > 0
    ):  # count('str') finds if the string belongs to the given string
        # Those neurons belonging to layers 2/3 assign then to layer 2 by default (representation purposes)
        p_c = 2  # p_c represents the layer number
    else:
        p_c = int(x[1:2])
    if x[0] == "e":
        inter_order = (
            4  # inter_order represents the neurons type order inside the layer
        )
    elif x.count("Vip") or x.count("Htr3a") > 0:
        inter_order = 1
    elif x.count("Sst") > 0:
        inter_order = 2
    elif x.count("Pvalb") > 0:
        inter_order = 3
    else:
        print(x)
        raise ValueError()
    ordering = p_c * 10 + inter_order
    return ordering


class RasterPlot:
    def __init__(
        self,
        batch_ind=0,
        stimuli_init_time=500,
        stimuli_end_time=1500,
        scale=2.0,
        marker_size=1.0,
        alpha=0.03,
        color="r",
        y_label="Neuron ID",
    ):
        self.batch_ind = batch_ind
        self.stimuli_init_time = stimuli_init_time
        self.stimuli_end_time = stimuli_end_time
        self.scale = scale
        self.marker_size = marker_size
        self.alpha = alpha
        self.color = color
        self.y_label = y_label

    def __call__(self, ax, spikes):
        # This method plots the spike train (spikes) that enters the network
        n_elements = np.prod(spikes.shape)
        non_binary_frac = (
            np.sum(np.logical_and(spikes > 1e-3, spikes < 1 - 1e-3)) / n_elements
        )
        if non_binary_frac > 0.01:
            rate = -np.log(1 - spikes[self.batch_ind] / 1.3) * 1000
            # rate = rate.reshape((rate.shape[0], int(rate.shape[1] / 100), 100)).mean(-1)
            p = ax.pcolormesh(rate.T, cmap="cividis")
            toolkit.do_inset_colorbar(ax, p, "")
            ax.set_ylim([0, rate.shape[-1]])
            ax.set_yticks([0, rate.shape[-1]])
            # ax.set_yticklabels([0, rate.shape[-1] * 100])
            ax.set_yticklabels([0, rate.shape[-1]])
            ax.set_ylabel(self.y_label, fontsize=20)
        else:
            # Take the times where the spikes occur
            times, ids = np.where(
                spikes[self.batch_ind].astype(float) > 0.5)
            ax.plot(
                times, ids, ".", color=self.color, ms=self.marker_size, alpha=self.alpha
            )
            ax.set_ylim([0, spikes.shape[-1]])
            ax.set_yticks([0, spikes.shape[-1]])
            ax.set_ylabel(self.y_label, fontsize=20)

        ax.axvline(
            self.stimuli_init_time,
            linestyle="dashed",
            color="k",
            linewidth=1.5,
            alpha=1,
        )
        ax.axvline(
            self.stimuli_end_time, linestyle="dashed", color="k", linewidth=1.5, alpha=1
        )
        ax.set_xlim([0, spikes.shape[1]])
        ax.set_xticks([0, spikes.shape[1]])
        ax.tick_params(axis="both", which="major", labelsize=18)


class LaminarPlot:
    def __init__(
        self,
        network,
        data_dir,
        batch_ind=0,
        stimuli_init_time=500,
        stimuli_end_time=1500,
        scale=2.0,
        marker_size=1.0,
        alpha=0.2,
        plot_core_only=True,
    ):
        self.batch_ind = batch_ind
        self.stimuli_init_time = stimuli_init_time
        self.stimuli_end_time = stimuli_end_time
        self.scale = scale
        self.marker_size = marker_size
        self.alpha = alpha
        self.data_dir = data_dir
        self.network = network
        self.n_neurons = network["n_nodes"]

        if plot_core_only:
            core_neurons = 16679 #65871 
            core_radius = 200 #400
            if self.n_neurons > core_neurons:
                self.n_neurons = core_neurons
            self.core_mask = other_v1_utils.isolate_core_neurons(
                self.network, radius=core_radius, data_dir=self.data_dir
            )
        else:
            self.core_mask = np.full(self.n_neurons, True)

         # use the true_pop_names, true_node_type_ids to create a dictionary with the node_type_id as key and the pop_name as value
        # since many of them are repeated we can use the unique function to get the unique pop_names

        node_types = pd.read_csv(
            os.path.join(self.data_dir, "network/v1_node_types.csv"), sep=" "
        )
        path_to_h5 = os.path.join(self.data_dir, "network/v1_nodes.h5")
        with h5py.File(path_to_h5, mode='r') as node_h5:
            # Create mapping from node_type_id to pop_name
            node_types.set_index('node_type_id', inplace=True)
            node_type_id_to_pop_name = node_types['pop_name'].to_dict()

            # Map node_type_id to pop_name for all neurons and select population names of neurons in the present network 
            node_type_ids = node_h5['nodes']['v1']['node_type_id'][()][network['tf_id_to_bmtk_id']]
            true_pop_names = np.array([node_type_id_to_pop_name[nid] for nid in node_type_ids])

            # Select population names of neurons in the present network (core)
            true_pop_names = true_pop_names[self.core_mask]
            true_node_type_ids = node_type_ids[self.core_mask]
    
        # Now order the pop_names
        #  according to their layer and type
        pop_orders = dict(
            sorted(
                node_type_id_to_pop_name.items(), key=lambda item: pop_ordering(item[1])
            )
        )

        # Now we convert the neuron id (related to its pop_name) to an index related to its position in the y axis
        neuron_id_to_y = (
            np.zeros(self.n_neurons, np.int32) - 1
        )  # rest 1 to check at the end if every neuron has an index
        current_ind = 0

        self.e_mask = np.zeros(self.n_neurons, np.bool_)
        self.htr3a_mask = np.zeros(self.n_neurons, np.bool_)
        self.vip_mask = np.zeros(self.n_neurons, np.bool_)
        self.sst_mask = np.zeros(self.n_neurons, np.bool_)
        self.pvalb_mask = np.zeros(self.n_neurons, np.bool_)

        layer_bounds = []
        ie_bounds = []
        current_pop_name = "e0"

        for pop_id, pop_name in pop_orders.items():
            # choose all the neurons of the given pop_id
            sel = true_node_type_ids == pop_id
            _n = np.sum(sel)
            pop_y_positions = np.arange(current_ind, current_ind + _n)
            tuning_angles = network['tuning_angle'][self.core_mask][sel]
            sorted_indices = np.argsort(tuning_angles)
            pop_y_positions = pop_y_positions[sorted_indices]
            # order the neurons by type and tuning angle in the y axis
            neuron_id_to_y[sel] = pop_y_positions

            if int(pop_name[1]) > int(current_pop_name[1]):
                # register the change of layer
                layer_bounds.append(current_ind)
            if current_pop_name[0] == "i" and pop_name[0] == "e":
                # register the change of neuron type: exc -> inh
                ie_bounds.append(current_ind)

            # #Now introduce the masks for the different neuron types
            if pop_name[0] == "e":
                self.e_mask = np.logical_or(self.e_mask, sel)
            elif pop_name.count("Htr3a") > 0:
                self.htr3a_mask = np.logical_or(self.htr3a_mask, sel)
            elif pop_name.count("Vip") > 0:
                self.vip_mask = np.logical_or(self.vip_mask, sel)
            elif pop_name.count("Sst") > 0:
                self.sst_mask = np.logical_or(self.sst_mask, sel)
            elif pop_name.count("Pvalb") > 0:
                self.pvalb_mask = np.logical_or(self.pvalb_mask, sel)
            else:
                raise ValueError(f"Unknown population {pop_name}")
            current_ind += _n
            current_pop_name = pop_name
        # check that an y id has been given to every neuron
        assert np.sum(neuron_id_to_y < 0) == 0
        self.layer_bounds = layer_bounds

        ######### For l5e neurons  ###########
        # l5e_min, l5e_max = ie_bounds[-2], layer_bounds[-1]
        # n_l5e = l5e_max - l5e_min

        # n_readout_pops = network['readout_neuron_ids'].shape[0]
        # dist = int(n_l5e / n_readout_pops)
        # #####################################

        y_to_neuron_id = np.zeros(self.n_neurons, np.int32)
        y_to_neuron_id[neuron_id_to_y] = np.arange(self.n_neurons)
        assert np.all(y_to_neuron_id[neuron_id_to_y]
                      == np.arange(self.n_neurons))
        # y_to_neuron_id: E.g., la neurona séptima por orden de capas tiene id 0, y_to_neuron_id[7]=0
        # neuron_id_to_y: E.g., la neurona con id 0 es la séptima por orden de capas, neuron_id_to_y[0] = 7

        # ##### For l5e neurons #####
        # neurons_per_readout = network['readout_neuron_ids'].shape[1]

        # for i in range(n_readout_pops):
        #     desired_y = np.arange(neurons_per_readout) + \
        #         int(dist / 2) + dist * i + l5e_min
        #     for j in range(neurons_per_readout):
        #         other_id = y_to_neuron_id[desired_y[j]]
        #         readout_id = network['readout_neuron_ids'][i, j]
        #         old_readout_y = neuron_id_to_y[readout_id]
        #         neuron_id_to_y[readout_id], neuron_id_to_y[other_id] = desired_y[j], neuron_id_to_y[readout_id]
        #         y_to_neuron_id[old_readout_y], y_to_neuron_id[desired_y[j]
        #                                                       ] = other_id, readout_id
        ###########################

        # plot the L1 top and L6 bottom
        self.neuron_id_to_y = self.n_neurons - neuron_id_to_y  

    def __call__(self, ax, spikes):
        scale = self.scale
        ms = self.marker_size
        alpha = self.alpha
        seq_len = spikes.shape[1]
        layer_label = ["1", "2/3", "4", "5", "6"]
        for i, (y, h) in enumerate(
            zip(self.layer_bounds, np.diff(
                self.layer_bounds, append=[self.n_neurons]))
        ):
            ax.annotate(
                f"L{layer_label[i]}",
                (5, (self.n_neurons - y - h / 2)),
                fontsize=5 * scale,
                va="center",
            )

            if i % 2 != 0:
                continue
            rect = patches.Rectangle(
                (0, self.n_neurons - y - h), seq_len, h, color="gray", alpha=0.1
            )
            ax.add_patch(rect)

        spikes = np.array(spikes)
        spikes = np.transpose(spikes[self.batch_ind, :, self.core_mask])

        # e
        times, ids = np.where(spikes * self.e_mask[None, :].astype(float))
        _y = self.neuron_id_to_y[ids]
        ax.plot(times, _y, ".", color="r", ms=ms, alpha=alpha)

        # Htr3a
        times, ids = np.where(
            spikes * self.htr3a_mask[None, :].astype(float))
        _y = self.neuron_id_to_y[ids]
        ax.plot(times, _y, ".", color="pink", ms=ms, alpha=alpha)

        # vip
        times, ids = np.where(spikes * self.vip_mask[None, :].astype(float))
        _y = self.neuron_id_to_y[ids]
        ax.plot(times, _y, ".", color="darkviolet", ms=ms, alpha=alpha)

        # sst
        times, ids = np.where(spikes * self.sst_mask[None, :].astype(float))
        _y = self.neuron_id_to_y[ids]
        ax.plot(times, _y, ".", color="g", ms=ms, alpha=alpha)

        # pvalb
        times, ids = np.where(
            spikes * self.pvalb_mask[None, :].astype(float))
        _y = self.neuron_id_to_y[ids]
        ax.plot(times, _y, ".", color="b", ms=ms, alpha=alpha)

        ##### For l5e neurons #####

        # for i, readout_neuron_ids in enumerate(self.network['readout_neuron_ids']):
        #     if len(self.network['readout_neuron_ids']) == 2 and i == 0:
        #         continue
        #     sel = np.zeros(self.n_neurons)
        #     sel[readout_neuron_ids] = 1.
        #     times, ids = np.where(
        #         spikes[self.batch_ind] * sel[None, :].astype(float))
        #     _y = self.neuron_id_to_y[ids]
        #     ax.plot(times, _y, '.', color='k', ms=ms, alpha=alpha)

        ###########################
        ax.plot([-1, -1], [-1, -1], ".", color="pink",
                ms=6, alpha=0.9, label="Htr3a")
        ax.plot([-1, -1], [-1, -1], ".", color="darkviolet", 
                ms=6, alpha=0.9, label="Vip")
        ax.plot([-1, -1], [-1, -1], ".", color="g",
                ms=6, alpha=0.9, label="Sst")
        ax.plot([-1, -1], [-1, -1], ".", color="b",
                ms=6, alpha=0.9, label="Pvalb")
        ax.plot([-1, -1], [-1, -1], ".", color="r",
                ms=6, alpha=0.9, label="Excitatory")
        # ax.plot([-1, -1], [-1, -1], '.', color='k',
        #         ms=4, alpha=.9, label='Readout (L5e)')

        # bg = patches.Rectangle((480 / 2050 * seq_len, 0), 300 / 2050 * seq_len,
        #                        220 / 1000 * self.n_neurons, color='white', alpha=.9, zorder=101)
        # ax.add_patch(bg)
        # ax.legend(frameon=True, facecolor='white', framealpha=.9, edgecolor='white',
        #           fontsize=5 * scale, loc='center', bbox_to_anchor=(.3, .12)).set_zorder(102)
        ax.axvline(
            self.stimuli_init_time,
            linestyle="dashed",
            color="k",
            linewidth=1.5,
            alpha=1,
        )
        ax.axvline(
            self.stimuli_end_time, linestyle="dashed", color="k", linewidth=1.5, alpha=1
        )
        ax.set_ylim([0, self.n_neurons])
        ax.set_yticks([0, self.n_neurons])
        ax.set_ylabel("Network Neuron ID", fontsize=20)
        ax.set_xlim([0, seq_len])
        ax.set_xticks([0, seq_len])
        ax.tick_params(axis="both", which="major", labelsize=18)


class DriftingGrating:
    def __init__(
        self,
        scale=2.0,
        frequency=2.0,
        stimuli_init_time=500,
        stimuli_end_time=1500,
        reverse=False,
        marker_size=1.0,
        alpha=1,
        color="g",
    ):
        self.marker_size = marker_size
        self.alpha = alpha
        self.color = color
        self.scale = scale
        self.stimuli_init_time = stimuli_init_time
        self.stimuli_end_time = stimuli_end_time
        self.reverse = reverse
        self.frequency = frequency

    def __call__(self, ax, simulation_length, stimulus_length=None):
        if stimulus_length is None:
            stimulus_length = simulation_length

        times = np.arange(stimulus_length)
        stimuli_speed = np.zeros((stimulus_length))
        if self.reverse:
            stimuli_speed[: self.stimuli_init_time] = self.frequency
            stimuli_speed[self.stimuli_end_time:] = self.frequency
        else:
            stimuli_speed[
                self.stimuli_init_time: self.stimuli_end_time
            ] = self.frequency

        ax.plot(
            times,
            stimuli_speed,
            color=self.color,
            ms=self.marker_size,
            alpha=self.alpha,
            linewidth=2 * self.scale,
        )
        ax.set_ylabel("TF \n [Hz]")
        ax.set_yticks([0, self.frequency])
        ax.set_yticklabels(["0", f"{self.frequency}"])
        ax.set_xlim([0, stimulus_length])
        ax.set_xticks(np.linspace(0, stimulus_length, 6))
        ax.set_xticklabels([str(int(x))
                           for x in np.linspace(0, simulation_length, 6)])
        # ax.set_xlabel('Time [ms]', fontsize=20)
        ax.set_xlabel("Time [ms]")
        # ax.tick_params(axis='both', which='major', labelsize=18)


class LGN_sample_plot:
    # Plot one realization of the LGN units response
    def __init__(
        self,
        firing_rates,
        spikes,
        stimuli_init_time=500,
        stimuli_end_time=1500,
        images_dir="Images",
        n_samples=2,
        directory="LGN units",
    ):
        self.firing_rates = firing_rates[0, :, :]
        self.spikes = spikes
        self.stimuli_init_time = stimuli_init_time
        self.stimuli_end_time = stimuli_end_time
        self.firing_rates_shape = self.firing_rates.shape
        self.n_samples = n_samples
        self.images_dir = images_dir
        self.directory = directory

    def __call__(self):
        for neuron_idx in np.random.choice(
            range(self.firing_rates_shape[1]), size=self.n_samples
        ):
            times = np.linspace(
                0, self.firing_rates_shape[0], self.firing_rates_shape[0]
            )

            fig, axs = plt.subplots(2, sharex=True)
            axs[0].plot(
                times, self.firing_rates[:, neuron_idx], color="r", ms=1, alpha=0.7
            )
            axs[0].set_ylabel("Firing rate [Hz]")
            axs[1].plot(
                times, self.spikes[0, :, neuron_idx], color="b", ms=1, alpha=0.7
            )
            axs[1].set_yticks([0, 1])
            axs[1].set_ylim(0, 1)
            axs[1].set_xlabel("Time [ms]")
            axs[1].set_ylabel("Spikes")

            for subplot in range(2):
                axs[subplot].axvline(
                    self.stimuli_init_time,
                    linestyle="dashed",
                    color="gray",
                    linewidth=3,
                )
                axs[subplot].axvline(
                    self.stimuli_end_time, linestyle="dashed", color="gray", linewidth=3
                )

            fig.suptitle(f"LGN unit idx:{neuron_idx}")
            path = os.path.join(self.images_dir, self.directory)
            os.makedirs(path, exist_ok=True)
            fig.savefig(os.path.join(
                path, f"LGN unit idx_{neuron_idx}.png"), dpi=300)
            # close figure
            plt.close(fig)


class PopulationActivity:
    def __init__(
        self,
        n_neurons,
        network,
        stimuli_init_time=500,
        stimuli_end_time=1500,
        image_path="",
        data_dir="",
    ):
        self.data_dir = data_dir
        self.n_neurons = n_neurons
        self.network = network
        self.stimuli_init_time = stimuli_init_time
        self.stimuli_end_time = stimuli_end_time
        self.images_path = image_path
        os.makedirs(self.images_path, exist_ok=True)

    def __call__(self, spikes, plot_core_only=True, bin_size=10):
        if plot_core_only:
            if self.n_neurons > 66634:
                self.n_neurons = 66634
            self.core_mask = other_v1_utils.isolate_core_neurons(
                self.network, data_dir=self.data_dir
            )
        else:
            self.core_mask = np.full(self.n_neurons, True)

        self.spikes = np.array(spikes)[0, :, self.core_mask]
        self.spikes = np.transpose(self.spikes)
        self.neurons_ordering()
        self.plot_populations_activity(bin_size)
        self.subplot_populations_activity(bin_size)

    def neurons_ordering(self):
        node_types = pd.read_csv(
            os.path.join(self.data_dir, "network/v1_node_types.csv"), sep=" "
        )
        path_to_h5 = os.path.join(self.data_dir, "network/v1_nodes.h5")
        node_h5 = h5py.File(path_to_h5, mode="r")
        node_type_id_to_pop_name = dict()
        for nid in np.unique(node_h5["nodes"]["v1"]["node_type_id"]):
            # if not np.unique all of the 230924 model neurons ids are considered,
            # but nearly all of them are repeated since there are only 111 different indices
            ind_list = np.where(node_types.node_type_id == nid)[0]
            assert len(ind_list) == 1
            node_type_id_to_pop_name[nid] = node_types.pop_name[ind_list[0]]

        node_type_ids = np.array(node_h5["nodes"]["v1"]["node_type_id"])
        # Select population names of neurons in the present network (core)
        true_node_type_ids = node_type_ids[
            self.network["tf_id_to_bmtk_id"][self.core_mask]
        ]
        # Now order the pop_names according to their layer and type
        pop_orders = dict(
            sorted(
                node_type_id_to_pop_name.items(), key=lambda item: pop_ordering(item[1])
            )
        )

        # Now we convert the neuron id (related to its pop_name) to an index related to its position in the y axis
        neuron_id_to_y = (
            np.zeros(self.n_neurons, np.int32) - 1
        )  # rest 1 to check at the end if every neuron has an index
        current_ind = 0
        self.layer_bounds = []
        self.ie_bounds = []
        current_pop_name = "e0"

        for pop_id, pop_name in pop_orders.items():
            # choose all the neurons of the given pop_id
            sel = true_node_type_ids == pop_id
            _n = np.sum(sel)
            # order the neurons by type in the y axis
            neuron_id_to_y[sel] = np.arange(current_ind, current_ind + _n)
            if int(pop_name[1]) > int(current_pop_name[1]):
                # register the change of layer
                self.layer_bounds.append(current_ind)
            if current_pop_name[0] == "i" and pop_name[0] == "e":
                # register the change of neuron type: exc -> inh
                self.ie_bounds.append(current_ind)
            current_ind += _n
            current_pop_name = pop_name

        # check that an y id has been given to every neuron
        assert np.sum(neuron_id_to_y < 0) == 0
        self.y_to_neuron_id = np.zeros(self.n_neurons, np.int32)
        self.y_to_neuron_id[neuron_id_to_y] = np.arange(self.n_neurons)
        assert np.all(
            self.y_to_neuron_id[neuron_id_to_y] == np.arange(self.n_neurons))

    def plot_populations_activity(self, bin_size=10):
        layers_label = ["i1", "i23", "e23", "i4", "e4", "i5", "e5", "i6", "e6"]
        neuron_class_bounds = np.concatenate(
            (self.ie_bounds, self.layer_bounds))
        neuron_class_bounds = np.append(neuron_class_bounds, self.n_neurons)
        neuron_class_bounds.sort()

        for idx, label in enumerate(layers_label):
            init_idx = neuron_class_bounds[idx]
            end_idx = neuron_class_bounds[idx + 1]
            neuron_ids = self.y_to_neuron_id[init_idx:end_idx]
            n_neurons_class = len(neuron_ids)
            class_spikes = self.spikes[:, neuron_ids]
            m, n = class_spikes.shape
            H, W = int(m / bin_size), 1  # block-size
            n_spikes_bin = class_spikes.reshape(
                H, m // H, W, n // W).sum(axis=(1, 3))
            population_activity = n_spikes_bin / \
                (n_neurons_class * bin_size * 0.001)

            fig = plt.figure()
            plt.plot(
                np.arange(0, self.spikes.shape[0], bin_size), population_activity)
            plt.axvline(
                self.stimuli_init_time,
                linestyle="dashed",
                color="gray",
                linewidth=1,
                zorder=10,
            )
            plt.axvline(
                self.stimuli_end_time,
                linestyle="dashed",
                color="gray",
                linewidth=1,
                zorder=10,
            )
            plt.xlabel("Time (ms)")
            plt.ylabel("Population activity (Hz)")
            plt.suptitle(f"Population activity of {label} neurons")
            path = os.path.join(self.images_path, "Populations activity")
            os.makedirs(path, exist_ok=True)
            fig.tight_layout()
            fig.savefig(os.path.join(
                path, f"{label}_population_activity.png"), dpi=300)
            plt.close(fig)

    def subplot_populations_activity(self, bin_size=10):
        layers_label = [
            "Inhibitory L1 neurons",
            "Inhibitory L23 neurons",
            "Excitatory L23 neurons",
            "Inhibitory L4 neurons",
            "Excitatory L4 neurons",
            "Inhibitory L5 neurons",
            "Excitatory L5 neurons",
            "Inhibitory L6 neurons",
            "Excitatory L6 neurons",
        ]
        neuron_class_bounds = np.concatenate(
            (self.ie_bounds, self.layer_bounds))
        neuron_class_bounds = np.append(neuron_class_bounds, self.n_neurons)
        neuron_class_bounds.sort()

        population_activity_dict = {}

        for idx, label in enumerate(layers_label):
            init_idx = neuron_class_bounds[idx]
            end_idx = neuron_class_bounds[idx + 1]
            neuron_ids = self.y_to_neuron_id[init_idx:end_idx]
            n_neurons_class = len(neuron_ids)
            class_spikes = self.spikes[:, neuron_ids]
            m, n = class_spikes.shape
            H, W = int(m / bin_size), 1  # block-size
            n_spikes_bin = class_spikes.reshape(
                H, m // H, W, n // W).sum(axis=(1, 3))
            population_activity = n_spikes_bin / \
                (n_neurons_class * bin_size * 0.001)
            population_activity_dict[label] = population_activity

        time = np.arange(0, self.spikes.shape[0], bin_size)
        fig = plt.figure(constrained_layout=False)
        # fig.set_constrained_layout_pads(w_pad=4 / 72, h_pad=4 / 72, hspace=0.15, wspace=0.15)
        ax1 = plt.subplot(5, 1, 1)
        plt.plot(
            time,
            population_activity_dict["Inhibitory L1 neurons"],
            label="Inhibitory L1 neurons",
            color="b",
        )
        plt.legend(fontsize=6)
        plt.tick_params(axis="both", labelsize=7)
        # plt.xlabel('Time (ms)', fontsize=7)
        plt.setp(ax1.get_xticklabels(), visible=False)
        plt.ylabel("Population \n activity (Hz)", fontsize=7)
        plt.axvline(
            self.stimuli_init_time,
            linestyle="dashed",
            color="gray",
            linewidth=1,
            zorder=10,
        )
        plt.axvline(
            self.stimuli_end_time,
            linestyle="dashed",
            color="gray",
            linewidth=1,
            zorder=10,
        )

        ax2 = None
        for i in range(3, 9):
            if i % 2 == 1:
                ax1 = plt.subplot(5, 2, i, sharex=ax1, sharey=ax1)
                plt.plot(
                    time,
                    population_activity_dict[layers_label[i - 2]],
                    label=layers_label[i - 2],
                    color="b",
                )
                plt.ylabel("Population \n activity (Hz)", fontsize=7)
                plt.setp(ax1.get_xticklabels(), visible=False)
                plt.legend(fontsize=6, loc="upper right")
                plt.tick_params(axis="both", labelsize=7)
                plt.axvline(
                    self.stimuli_init_time,
                    linestyle="dashed",
                    color="gray",
                    linewidth=1,
                    zorder=10,
                )
                plt.axvline(
                    1500, linestyle="dashed", color="gray", linewidth=1, zorder=10
                )
            else:
                if ax2 == None:
                    ax2 = plt.subplot(5, 2, i, sharex=ax1)
                else:
                    ax2 = plt.subplot(5, 2, i, sharex=ax2, sharey=ax2)
                plt.plot(
                    time,
                    population_activity_dict[layers_label[i - 2]],
                    label=layers_label[i - 2],
                    color="r",
                )
                plt.setp(ax2.get_xticklabels(), visible=False)
                plt.legend(fontsize=6, loc="upper right")
                plt.tick_params(axis="both", labelsize=7)
                plt.axvline(
                    self.stimuli_init_time,
                    linestyle="dashed",
                    color="gray",
                    linewidth=1,
                    zorder=10,
                )
                plt.axvline(
                    self.stimuli_end_time,
                    linestyle="dashed",
                    color="gray",
                    linewidth=1,
                    zorder=10,
                )

        ax1 = plt.subplot(5, 2, 9, sharex=ax1, sharey=ax1)
        plt.plot(
            time,
            population_activity_dict[layers_label[7]],
            label=layers_label[7],
            color="b",
        )
        plt.ylabel("Population \n activity (Hz)", fontsize=7)
        plt.xlabel("Time [ms]", fontsize=7)
        plt.tick_params(axis="both", labelsize=7)
        plt.legend(fontsize=6, loc="upper right")
        plt.axvline(
            self.stimuli_init_time,
            linestyle="dashed",
            color="gray",
            linewidth=1,
            zorder=10,
        )
        plt.axvline(
            self.stimuli_end_time,
            linestyle="dashed",
            color="gray",
            linewidth=1,
            zorder=10,
        )

        ax2 = plt.subplot(5, 2, 10, sharex=ax2, sharey=ax2)
        plt.plot(
            time,
            population_activity_dict[layers_label[8]],
            label=layers_label[8],
            color="r",
        )
        plt.xlabel("Time [ms]", fontsize=7)
        plt.tick_params(axis="both", labelsize=7)
        plt.legend(fontsize=6, loc="upper right")
        plt.axvline(
            self.stimuli_init_time,
            linestyle="dashed",
            color="gray",
            linewidth=1,
            zorder=10,
        )
        plt.axvline(
            self.stimuli_end_time,
            linestyle="dashed",
            color="gray",
            linewidth=1,
            zorder=10,
        )

        plt.subplots_adjust(
            left=0.1, bottom=0.07, right=0.99, top=0.99, wspace=0.17, hspace=0.17
        )

        path = os.path.join(self.images_path, "Populations activity")
        os.makedirs(path, exist_ok=True)
        # fig.tight_layout()
        fig.savefig(os.path.join(
            path, "subplot_population_activity.png"), dpi=300)
        plt.close(fig)


def calculate_Firing_Rate(z, drifting_gratings_init=500, drifting_gratings_end=2500):
    dg_spikes = z[:, drifting_gratings_init:drifting_gratings_end, :]
    # if the number of dimensions of dg_spikes is 2, reshape it to 3 adding an additional first dimension
    # if dg_spikes.ndim == 2:
    #     dg_spikes = dg_spikes.reshape(1, dg_spikes.shape[0], dg_spikes.shape[1])
    mean_dg_spikes = np.mean(dg_spikes, axis=0)
    mean_firing_rates = np.sum(mean_dg_spikes, axis=0)/((drifting_gratings_end-drifting_gratings_init)/1000)
    
    return mean_firing_rates

def calculate_OSI_DSI(rates_df, network, DG_angles=range(0,360, 45), core_radius=None, core_mask=None, remove_zero_rate_neurons=False):
    
    # Get the pop names of the neurons
    if core_radius is not None:
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
    # dsi = np.abs((all_rates * np.exp(1j * phase_rad)).sum(axis=1) / denominator)
    # osi = np.abs((all_rates * np.exp(2j * phase_rad)).sum(axis=1) / denominator)

    dsi = np.where(denominator != 0, 
               np.abs((all_rates * np.exp(1j * phase_rad)).sum(axis=1)) / denominator, 
               np.nan)
    osi = np.where(denominator != 0,
                np.abs((all_rates * np.exp(2j * phase_rad)).sum(axis=1)) / denominator,
                np.nan)


    print(f"DSI: {dsi}")
    print(f"dsi shape: {dsi.shape}")
    print(f"node_ids shape: {node_ids.shape}")


    
    # Save the results in a dataframe
    osi_df = pd.DataFrame()
    osi_df["node_id"] = node_ids
    osi_df["pop_name"] = pop_names
    # osi_df["DSI"] = dsi[core_mask]
    # osi_df["OSI"] = osi[core_mask]
    # osi_df["preferred_angle"] = preferred_DG_angle[core_mask]
    # osi_df["max_mean_rate(Hz)"] = preferred_rates[core_mask]
    # osi_df["Avg_Rate(Hz)"] = average_rates[core_mask]
    osi_df["DSI"] = dsi
    osi_df["OSI"] = osi
    osi_df["preferred_angle"] = preferred_DG_angle
    osi_df["max_mean_rate(Hz)"] = preferred_rates
    osi_df["Ave_Rate(Hz)"] = average_rates

    if remove_zero_rate_neurons:
        osi_df = osi_df[osi_df["Ave_Rate(Hz)"] != 0]

    return osi_df


class ModelMetricsAnalysis:    

    def __init__(self, network, neurons, data_dir='', directory='', filename='', n_trials=1, drifting_gratings_init=50, 
                 drifting_gratings_end=550, analyze_core_only=True):
        self.n_neurons = neurons
        self.network = network
        self.data_dir = data_dir 
        self.n_trials = n_trials
        self.drifting_gratings_init = drifting_gratings_init
        self.drifting_gratings_end = drifting_gratings_end
        self.analyze_core_only = analyze_core_only
        self.directory=directory
        self.filename = filename
    
    def __call__(self, spikes, DG_angles):

        # Isolate the core neurons if necessary
        if self.analyze_core_only:
            core_neurons = 16679 #65871 
            core_radius = 200 #400
            
            # Calculate the core_neurons mask
            if self.n_neurons > core_neurons:
                self.core_mask = other_v1_utils.isolate_core_neurons(self.network, radius=core_radius, data_dir=self.data_dir) 
                # self.n_neurons = core_neurons
                # if n_neurons is overridden, it won't run for the second time...
                n_neurons_plot = core_neurons
            else:
                self.core_mask = np.full(self.n_neurons, True)
                n_neurons_plot = self.n_neurons

        else:
            self.core_mask = np.full(self.n_neurons, True)
            core_radius = None

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
        metrics_df = calculate_OSI_DSI(firing_rates_df, self.network, DG_angles=DG_angles, core_radius=core_radius, core_mask=self.core_mask)
        # metrics_df.to_csv(os.path.join(self.directory, f"V1_OSI_DSI_DF.csv"), sep=" ", index=False)

        # Make the boxplots to compare with the neuropixels data
        if len(DG_angles) == 1:
            metrics = ["Ave_Rate(Hz)"]
        else:
            metrics = ["Rate at preferred direction (Hz)", "OSI", "DSI"]

        boxplot = MetricsBoxplot(save_dir=self.directory, filename=self.filename)
        boxplot.plot(metrics=metrics, metrics_df=metrics_df)

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
                patches.Rectangle((borders[i], ylim[0]), w, h, alpha=0.08, color="k", zorder=-10)
            )
        return ax   

    def get_osi_dsi_df(self, metric_file="V1_OSI_DSI_DF.csv", data_source_name="", data_dir=""):
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

        columns = ["cell_type", "data_type", "Rate at preferred direction (Hz)", "OSI", "DSI", 'Ave_Rate(Hz)']
        df = df[columns]

        return df

    def plot(self, metrics=["Rate at preferred direction (Hz)", "OSI", "DSI"], metrics_df=None):
        # Get the dataframes for the model and Neuropixels OSI and DSI 
        if metrics_df is None:
            metrics_df = f"V1_OSI_DSI_DF.csv"

        # print(metrics_df)

        self.osi_dfs.append(self.get_osi_dsi_df(metric_file=metrics_df, data_source_name="V1 GLIF model", data_dir=self.save_dir))
        self.osi_dfs.append(self.get_osi_dsi_df(metric_file=f"V1_OSI_DSI_DF.csv", data_source_name="Neuropixels", data_dir='Neuropixels_data'))
        self.osi_dfs.append(self.get_osi_dsi_df(metric_file=f"V1_OSI_DSI_DF.csv", data_source_name="Billeh et al (2020)", data_dir='Billeh_column_metrics'))
        self.osi_dfs.append(self.get_osi_dsi_df(metric_file=f"V1_OSI_DSI_DF_pop_name.csv", data_source_name="NEST simulation", data_dir='NEST_metrics'))
        
        df = pd.concat(self.osi_dfs, ignore_index=True)
        # df.to_csv(os.path.join('Borrar', f"help_DG_firing_rates_df.csv"), sep=" ", index=False)

        # Create a figure to compare several model metrics against Neuropixels data
        n_metrics = len(metrics)
        height = int(7*n_metrics)
        fig, axs = plt.subplots(n_metrics, 1, figsize=(12, height))
        # fig, axs = plt.subplots(n_metrics, 1, figsize=(12, 20))
        if n_metrics == 1:
            axs = [axs]

        color_pal = {
            "V1 GLIF model": "tab:orange",
            "Neuropixels": "tab:gray",
            "Billeh et al (2020)": "tab:blue",
            "NEST simulation": "tab:pink"
        }

        # Establish the order of the neuron types in the boxplots
        cell_type_order = np.sort(df['cell_type'].unique())

        for idx, metric in enumerate(metrics):
            if metric in ["Rate at preferred direction (Hz)", 'Ave_Rate(Hz)']:
                ylims = [0, 100]
            else:
                ylims = [0, 1]

            plot_one_metric(axs[idx], df, metric, ylims, color_pal, hue_order=cell_type_order)

        axs[0].legend(loc="upper right", fontsize=16)
        if len(axs) > 1:
            for i in range(len(axs)-1):
                axs[i].set_xticklabels([])

        xticklabel = axs[n_metrics-1].get_xticklabels()
        for label in xticklabel:
            label.set_fontsize(14)
            label.set_weight("bold")

        plt.tight_layout()
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