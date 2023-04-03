

import json
import os
from time import time

import absl
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.mixed_precision import experimental as mixed_precision

from general_utils import file_management
from general_utils.other_utils import memory_tracer, timer
from billeh_model_utils import load_sparse, models, other_billeh_utils, toolkit
from billeh_model_utils.plotting_utils import InputActivityFigure, LaminarPlot, LGN_sample_plot, PopulationActivity, RasterPlot
# import data_sets


class PlotCallback(tf.keras.callbacks.Callback):
    """Periodically plot the activity of the model based on the same example"""

    def __init__(
        self,
        test_data_set,
        extractor_model,
        network,
        data_dir,
        batch_ind=0,
        scale=2,
        path=None,
        prefix="",
    ):
        super().__init__()
        test_iter = iter(test_data_set)
        self._test_example = next(test_iter)
        self._extractor_model = extractor_model

        self.figure = plt.figure(
            figsize=toolkit.cm2inch((15 * scale, 11 * scale)))
        gs = self.figure.add_gridspec(5, 1)
        self.input_ax = self.figure.add_subplot(gs[0])
        self.activity_ax = self.figure.add_subplot(gs[1:-1])
        self.output_ax = self.figure.add_subplot(gs[-1])

        self.inputs_plot = RasterPlot(
            batch_ind=batch_ind, scale=scale, y_label="Input Neuron ID", alpha=1.0
        )
        self.laminar_plot = LaminarPlot(
            network, data_dir, batch_ind=batch_ind, scale=scale, alpha=0.4
        )
        self.tightened = False
        self.scale = scale
        self.network = network
        self.batch_ind = batch_ind
        self._counter = 0
        self._path = path
        self._prefix = prefix

    def on_epoch_begin(self, epoch, logs=None):
        inputs = self._test_example[0]
        targets = self._test_example[1]
        (z, v), prediction, all_prediction = self._extractor_model(inputs)

        self.input_ax.clear()
        self.activity_ax.clear()
        self.output_ax.clear()
        self.inputs_plot(self.input_ax, inputs[0].numpy())
        self.input_ax.set_xticklabels([])
        toolkit.apply_style(self.input_ax, scale=self.scale)
        self.laminar_plot(self.activity_ax, z.numpy())
        self.activity_ax.set_xticklabels([])
        toolkit.apply_style(self.activity_ax, scale=self.scale)

        all_pred_np = tf.nn.softmax(
            models.exp_convolve(
                all_prediction[self.batch_ind], axis=0, decay=0.8)
        ).numpy()
        self.output_ax.plot(
            all_pred_np[:, 1], "r", alpha=0.7, lw=self.scale, label="Up"
        )
        self.output_ax.plot(
            all_pred_np[:, 0], "b", alpha=0.7, lw=self.scale, label="Down"
        )
        self.output_ax.set_ylim([0, 1])
        self.output_ax.set_yticks([0, 1])
        self.output_ax.set_xlim([0, all_pred_np.shape[0]])
        self.output_ax.set_xticks([0, all_pred_np.shape[0]])
        self.output_ax.legend(frameon=False, fontsize=5 * self.scale)
        self.output_ax.set_xlabel("Time in ms")
        self.output_ax.set_ylabel("Probability")
        toolkit.apply_style(self.output_ax, scale=self.scale)

        if not self.tightened:
            self.figure.tight_layout()
            self.tightened = True

        self._counter += 1
        if self._path is not None:
            self.figure.savefig(
                os.path.join(
                    self._path, f"{self._prefix}raster_epoch_{self._counter}.png"
                ),
                dpi=300,
            )


# @timer
# @memory_tracer
def main(_):
    flags = absl.app.flags.FLAGS
    np.random.seed(flags.seed)
    tf.random.set_seed(flags.seed)

    # flags.data_dir = os.path.join(os.path.dirname(os.getcwd()), flags.data_dir)
    # Create directory to save the simulation results
    path = os.path.join(
        flags.save_dir,
        f"orien_{str(flags.gratings_orientation)}_freq_{str(flags.gratings_frequency)}_reverse_{str(flags.reverse)}_rec_{flags.neurons}",
    )
    os.makedirs(path, exist_ok=True)
    # with open(os.path.join(path,'flags_config.json'), 'w') as fp:
    #     json.dump(flags.flag_values_dict(), fp)

    # Allow for memory growth (also to observe memory consumption)
    physical_devices = tf.config.list_physical_devices("GPU")
    try:
        for dev in physical_devices:
            tf.config.experimental.set_memory_growth(dev, True)
    except:
        # Invalid device or cannot modify virtual devices once initialized.
        print("Invalid device or cannot modify virtual devices once initialized.")
        pass

    for gpu in physical_devices:
        print("GPU:", gpu.name)
        print("Memory capacity:", gpu.memory_limit)

    # Can be used to try half precision training
    if flags.float16:
        policy = mixed_precision.Policy("mixed_float16")
        mixed_precision.set_policy(policy)
        dtype = tf.float16
    else:
        dtype = tf.float32

    # Load data of Billeh et al. (2020) and select appropriate number of neurons and inputs
    if flags.caching:
        load_fn = load_sparse.cached_load_billeh
    else:
        load_fn = load_sparse.load_billeh

    input_weight_scale = 1.0

    ### MODEL INPUT ###
    lgn_firing_rates_filename = f"orientation_{str(flags.gratings_orientation)}&TF_{str(float(flags.gratings_frequency))}&SF_0.04&reverse_{str(flags.reverse)}&init_screen_dur_1.0&visual_flow_dur_1.0&end_screen_dur_1.0&min_value_-1&max_value_1&contrast_0.8&dt_0.001&height_120&width_240&init_gray_screen_False&end_gray_screen_False.lzma"

    with open(
        os.path.join(
            flags.data_dir, "input", "Drifting_gratings", lgn_firing_rates_filename
        ),
        "rb",
    ) as f:
        firing_rates = file_management.load_lzma(f)

    # lgn_firing_rates_filename = 'drifting_grating_firing_rates_2_0.pkl'
    # with open(os.path.join(flags.data_dir,'input',lgn_firing_rates_filename), 'rb') as f:
    #     firing_rates = pkl.load(f)

    # firing_rates are the probability of spikes/seconds so we convert that to spike/ms
    # then, if random number in 0-1 is lower that firing_rate in spike/ms, there is a
    # neuron spike at that ms
    # We remove the first 500 ms of lgn firing rates, which have shape (3000, 17400), to allow then reach the stationary
    firing_rates = firing_rates[None,
                                500: flags.seq_len + 500]  # (1,2500,17400)
    # firing_rates = firing_rates[None, :]

    # Build the network
    input_population, network, bkg, bkg_weights = load_fn(
        flags.n_input,
        flags.neurons,
        flags.core_only,
        flags.data_dir,
        flags.seed,
        flags.connected_selection,
        flags.n_output,
        flags.neurons_per_output,
    )

    # Create the model
    model = models.create_model(
        network,
        input_population,
        bkg_weights,
        seq_len=flags.seq_len,
        n_input=flags.n_input,
        n_output=flags.n_output,
        cue_duration=flags.recall_duration,
        dtype=dtype,
        input_weight_scale=input_weight_scale,
        dampening_factor=flags.dampening_factor,
        gauss_std=flags.gauss_std,
        lr_scale=flags.lr_scale,
        train_recurrent=flags.train_recurrent,
        neuron_output=flags.neuron_output,
        recurrent_dampening_factor=flags.recurrent_dampening_factor,
        batch_size=flags.batch_size,
        pseudo_gauss=flags.pseudo_gauss,
        use_state_input=True,
        return_state=True,
        hard_reset=flags.hard_reset,
    )

    model.build((flags.batch_size, flags.seq_len, flags.n_input))

    # Extract outputs of intermediate keras layers to get access to
    # spikes and membrane voltages of the model
    rsnn_layer = model.get_layer("rsnn")
    prediction_layer = model.get_layer("prediction")

    extractor_model = tf.keras.Model(
        inputs=model.inputs,
        outputs=[rsnn_layer.output, model.output[0], prediction_layer.output],
    )

    # Loss used for training (evidence accumulation is a classification task)
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=False, reduction=tf.keras.losses.Reduction.SUM
    )

    # These "dummy" zeros are injected to the models membrane voltage
    # Provides the opportunity to compute gradients wrt. membrane voltages at all time steps
    # Not important for general use
    zero_state = rsnn_layer.cell.zero_state(flags.batch_size)
    dummy_zeros = tf.zeros(
        (flags.batch_size, flags.seq_len, flags.neurons), dtype)
    inputs = tf.zeros((flags.batch_size, flags.seq_len, flags.n_input), dtype)
    state = zero_state

    time_per_sim = 0
    time_per_save = 0
    keys = [
        "z",
        "v",
        "input_current",
        # "recurrent_current",
        # "bottom_up_current",
        "z_lgn",
    ]
    # Select the dtype of the data saved
    if flags.save_float16:
        save_dtype = np.float16
    else:
        save_dtype = np.float32
    data_path = os.path.join(path, "Data")
    SimulationDataHDF5 = other_billeh_utils.SaveSimDataHDF5(
        flags, keys, data_path, network, save_core_only=True, dtype=save_dtype
    )
    # SimulationData = other_billeh_utils.SaveSimData(flags, keys, data_path, network,
    #                                                 save_core_only=True, compress_data=True,
    #                                                 dtype=np.float16)
    for trial in range(0, flags.n_simulations):
        print("{trial}:new trial".format(trial=trial))
        inputs = (
            np.random.uniform(size=inputs.shape, low=0.0, high=1.0)
            < firing_rates * 0.001
        ).astype(np.uint8)
        t_init_sim = time()
        out = extractor_model((inputs, dummy_zeros, state))
        time_per_sim += time() - t_init_sim

        t_init_save = time()
        if flags.output_currents:
            z, v, input_current, recurrent_current, bottom_up_current = out[0][0]
            simulation_data = {
                "z": z,
                "v": v,
                "input_current": input_current,
                "recurrent_current": recurrent_current,
                "bottom_up_current": bottom_up_current,
                "z_lgn": inputs,
            }
        else:
            z, v, input_current = out[0][0]
            simulation_data = {
                "z": z,
                "v": v,
                "input_current": input_current,
                "z_lgn": inputs,
            }

        # SimulationData(simulation_data, trial)
        SimulationDataHDF5(simulation_data, trial)
        time_per_save += time() - t_init_save

        state = out[0][1:]

    # Save the simulation metadata
    time_per_sim /= flags.n_simulations
    time_per_save /= flags.n_simulations
    metadata_path = os.path.join(path, "Simulation stats")
    with open(metadata_path, "w") as out_file:
        out_file.write(f"Consumed time per simulation: {time_per_sim}\n")
        out_file.write(f"Consumed time saving: {time_per_save}\n")
    # Make the plots
    raster_filename = "Raster_plot.png"
    image_path = os.path.join(path, "Images general")
    os.makedirs(image_path, exist_ok=True)
    # Scatter plot the responses of both the core neurons of V1 and the LGN units
    graph = InputActivityFigure(
        network,
        flags.data_dir,
        image_path,
        filename=raster_filename,
        frequency=flags.gratings_frequency,
        stimuli_init_time=500,
        stimuli_end_time=1500,
        reverse=flags.reverse,
        plot_core_only=False,
    )
    graph(inputs, z)
    # Plot LGN units firing rates
    LGN_units = LGN_sample_plot(
        firing_rates,
        inputs,
        stimuli_init_time=500,
        stimuli_end_time=1500,
        images_dir=image_path,
        n_samples=2,
    )
    LGN_units()
    # Plot the mean firing rate of the population of neurons
    Population_activity = PopulationActivity(
        n_neurons=flags.neurons,
        network=network,
        image_path=image_path,
        data_dir=flags.data_dir,
    )
    Population_activity(z, plot_core_only=True, bin_size=10)

    ### TRAINING ###

    # Load a typical distribution of firing rates to which the model is regularized to
    # during training
    # with open(os.path.joiNo don(flags.data_dir, 'garrett_firing_rates.pkl'), 'rb') as f:
    #     firing_rates = pkl.load(f)
    # sorted_firing_rates = np.sort(firing_rates)
    # percentiles = (np.arange(
    #     firing_rates.shape[-1]) + 1).astype(np.float32) / firing_rates.shape[-1]
    # rate_rd = np.random.RandomState(seed=3000)
    # x_rand = rate_rd.uniform(size=flags.neurons)
    # target_firing_rates = np.sort(
    #     np.interp(x_rand, percentiles, sorted_firing_rates))

    # # ---
    # # Training disrupts the firing properties of the model
    # # To counteract, two types of regularizations are used
    # # 1) Firing rate regularization keeps the distribution of the firing rates close
    # # to the previously loaded distribution of firing rates
    # rate_distribution_regularizer = models.SpikeRateDistributionRegularization(
    #     target_firing_rates, flags.rate_cost)
    # # 2) Voltage regularization penalizes membrane voltages that are below resting potential or above threshold
    # voltage_regularizer = models.VoltageRegularization(
    #     rsnn_layer.cell, flags.voltage_cost)

    # rate_loss = rate_distribution_regularizer(rsnn_layer.output[0])
    # voltage_loss = voltage_regularizer(rsnn_layer.output[1])
    # model.add_loss(rate_loss)
    # model.add_loss(voltage_loss)
    # model.add_metric(rate_loss, name='rate_loss')
    # model.add_metric(voltage_loss, name='voltage_loss')

    # def compute_loss(_target, _pred):
    #     return loss_object(_target, _pred) / global_batch_size

    # # Adaptive learning rates
    # optimizer = tf.keras.optimizers.Adam(flags.learning_rate)
    # model.compile(optimizer, compute_loss, metrics=['accuracy'])

    # # Restore weights from a checkpoint if desired
    # if flags.restore_from != '':
    #     model.load_weights(flags.restore_from)
    #     print(f'> Model successfully restored from {flags.restore_from}')

    # def get_dataset_fn(_n, _n_cues):
    #     def _f(_):
    #         _data_set = data_sets.create_evidence_accumulation(
    #             batch_size=global_batch_size, n_input=flags.n_input, seq_len=flags.seq_len,
    #             recall_duration=flags.recall_duration, examples_in_epoch=_n, n_cues=_n_cues,
    #             hard_only=flags.hard_only, t_cue=flags.cue_duration, t_interval=flags.interval_duration,
    #             input_f0=flags.input_f0
    #         ).repeat().map(_expand)
    #         return _data_set
    #     return _f

    # test_data_set = get_dataset_fn(
    #     flags.validation_examples, flags.n_cues)(None)

    # # Bookkeeping of simulations
    # sim_name = toolkit.get_random_identifier('b_')
    # results_dir = os.path.join(
    #     flags.results_dir, 'drifting_gratings_simulation')
    # print(f'> Results will be stored in {os.path.join(results_dir, sim_name)}')
    # os.makedirs(results_dir, exist_ok=True)

    # cm = simmanager.SimManager(sim_name, results_dir, write_protect_dirs=False, tee_stdx_to='output.log')

    # with cm:
    #     # Save the settings with which the script was invoked
    #     with open(os.path.join(cm.paths.data_path, 'flags.json'), 'w') as f:
    #         json.dump(flags.flag_values_dict(), f, indent=4)

    #     # Apply a learning curriculum using iteratively more cues up to the desired number
    #     for n_cues in range(1, flags.n_cues + 1, 2):
    #         train_data_set = get_dataset_fn(flags.examples_in_epoch, n_cues)(None)

    #         vis_data = test_data_set if flags.visualize_test else train_data_set
    #         # Define callbacks that are used for visualizing network activity (see above),
    #         # for stopping the training if the task is solved, and for saving the model
    #         plot_callback = PlotCallback(vis_data, extractor_model, network, flags.data_dir,
    #                                      path=cm.paths.results_path, prefix=f'cue_{n_cues}_')
    #         fit_callbacks = [
    #             plot_callback,
    #             callbacks.StopAt('accuracy', .99),
    #             tf.keras.callbacks.ModelCheckpoint(
    #                 filepath=os.path.join(cm.paths.results_path, 'model'),
    #                 monitor='val_accuracy', save_weights_only=True),
    #             tf.keras.callbacks.TensorBoard(log_dir=cm.paths.results_path)
    #         ]

    #         # Perform training
    #         model.fit(train_data_set, steps_per_epoch=flags.examples_in_epoch, epochs=flags.n_epochs,
    #                   validation_data=test_data_set, validation_steps=flags.validation_examples,
    #                   callbacks=fit_callbacks)


if __name__ == "__main__":
    _data_dir = "GLIF_network"
    _results_dir = "GLIF_network/results"
    _save_dir = "Simulation_results"
    _images_dir = "Images_general"

    absl.app.flags.DEFINE_string("data_dir", _data_dir, "")
    absl.app.flags.DEFINE_string("results_dir", _results_dir, "")
    absl.app.flags.DEFINE_string("save_dir", _save_dir, "")
    absl.app.flags.DEFINE_string("images_dir", _images_dir, "")
    absl.app.flags.DEFINE_string("restore_from", "", "")
    absl.app.flags.DEFINE_string("comment", "", "")
    absl.app.flags.DEFINE_integer("gratings_orientation", 0, "")
    absl.app.flags.DEFINE_integer("gratings_frequency", 2, "")
    absl.app.flags.DEFINE_integer("n_simulations", 20, "")

    absl.app.flags.DEFINE_float("learning_rate", 0.001, "")
    absl.app.flags.DEFINE_float("rate_cost", 0.0, "")
    absl.app.flags.DEFINE_float("voltage_cost", 0.001, "")
    absl.app.flags.DEFINE_float("dampening_factor", 0.1, "")
    absl.app.flags.DEFINE_float("recurrent_dampening_factor", 0.5, "")
    absl.app.flags.DEFINE_float("gauss_std", 0.3, "")
    absl.app.flags.DEFINE_float("lr_scale", 1.0, "")
    absl.app.flags.DEFINE_float("input_f0", 0.2, "")

    absl.app.flags.DEFINE_integer("n_epochs", 20, "")
    absl.app.flags.DEFINE_integer("batch_size", 16, "")
    absl.app.flags.DEFINE_integer(
        "neurons", 1574, ""
    )  # 230,924 to take all neurons, 51,978 to take only core neurons
    absl.app.flags.DEFINE_integer(
        "n_input", 17400, ""
    )  # number of LGN filters in visual space (input population)
    absl.app.flags.DEFINE_integer("seq_len", 2500, "")
    absl.app.flags.DEFINE_integer("n_cues", 3, "")
    absl.app.flags.DEFINE_integer("recall_duration", 40, "")
    absl.app.flags.DEFINE_integer("cue_duration", 40, "")
    absl.app.flags.DEFINE_integer("interval_duration", 40, "")
    absl.app.flags.DEFINE_integer("examples_in_epoch", 32, "")
    absl.app.flags.DEFINE_integer("validation_examples", 16, "")
    absl.app.flags.DEFINE_integer("seed", 3000, "")
    absl.app.flags.DEFINE_integer("neurons_per_output", 16, "")
    # Define 2 outputs that correspond to having more cues top or bottom
    # Note that two different output conventions can be used:
    # 1) Linear readouts from all neurons in the model (softmax)
    # 2) Selecting a population of neurons that report a binary decision
    # with high firing rate (flag --neuron_output)
    absl.app.flags.DEFINE_integer("n_output", 2, "")

    absl.app.flags.DEFINE_boolean("reverse", False, "")
    absl.app.flags.DEFINE_boolean("output_currents", False, "")
    absl.app.flags.DEFINE_boolean("float16", False, "")
    absl.app.flags.DEFINE_boolean("save_float16", True, "")
    absl.app.flags.DEFINE_boolean("caching", True, "")
    absl.app.flags.DEFINE_boolean("core_only", False, "")
    absl.app.flags.DEFINE_boolean("train_recurrent", True, "")
    absl.app.flags.DEFINE_boolean("connected_selection", True, "")
    absl.app.flags.DEFINE_boolean("neuron_output", True, "")
    absl.app.flags.DEFINE_boolean("hard_only", False, "")
    absl.app.flags.DEFINE_boolean("visualize_test", False, "")
    absl.app.flags.DEFINE_boolean("pseudo_gauss", False, "")
    absl.app.flags.DEFINE_boolean("hard_reset", True, "")

    absl.app.run(main)
