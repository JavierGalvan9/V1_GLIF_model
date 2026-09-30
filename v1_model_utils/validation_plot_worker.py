"""Render validation plots from a CPU-only snapshot in a separate process."""

import os
import sys
import time

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["MPLBACKEND"] = "Agg"

import numpy as np
import pandas as pd

from v1_model_utils.model_metrics_analysis import MetricsBoxplot
from v1_model_utils import other_v1_utils
from v1_model_utils.plotting_utils import InputActivityFigure


def _rate_boxplot(rates, network, data_dir, core_radius, metric, directory,
                  filename, neuropixels_df):
    if core_radius > 0:
        mask = other_v1_utils.isolate_core_neurons(
            network, radius=core_radius, data_dir=data_dir,
        )
        rates = rates[mask]
    names = other_v1_utils.pop_names(
        network, core_radius=core_radius if core_radius > 0 else None,
        data_dir=data_dir,
    )
    metrics = pd.DataFrame({"pop_name": names, metric: rates})
    os.makedirs(directory, exist_ok=True)
    MetricsBoxplot(save_dir=directory, filename=filename).plot(
        metrics=[metric], metrics_df=metrics, neuropixels_df=neuropixels_df,
    )


def _raster(spikes, inputs, network, data_dir, directory, filename,
            frequency, pre_delay, post_delay, core_radius):
    os.makedirs(directory, exist_ok=True)
    InputActivityFigure(
        network, data_dir, directory, filename=filename,
        frequency=frequency, stimuli_init_time=pre_delay,
        stimuli_end_time=spikes.shape[1] - post_delay,
        reverse=False, plot_core_only=True, core_radius=core_radius,
    )(inputs, spikes)


def epoch_main(payload_path):
    started = time.perf_counter()
    stage = started
    timings = {}
    succeeded = False

    def mark(name):
        nonlocal stage
        now = time.perf_counter()
        timings[name] = now - stage
        stage = now

    try:
        with np.load(payload_path, allow_pickle=False) as payload:
            logdir = str(payload["logdir"])
            epoch = int(payload["epoch"])
            data_dir = str(payload["data_dir"])
            neuropixels_df = str(payload["neuropixels_df"])
            pre_delay = int(payload["pre_delay"])
            post_delay = int(payload["post_delay"])
            plot_core_radius = float(payload["plot_core_radius"])
            network = {
                "n_nodes": int(payload["n_nodes"]),
                "tf_id_to_bmtk_id": payload["tf_id_to_bmtk_id"],
                "tuning_angle": payload["tuning_angle"],
            }
            from v1_model_utils.callbacks import Callbacks

            plotter = object.__new__(Callbacks)
            plotter.logdir = logdir
            plotter.epoch_metric_values = {
                key.removeprefix("history_"): payload[key].copy()
                for key in payload.files if key.startswith("history_")
            }
            plotter.plot_losses_curves()
            mark("loss_curves")

            if "osi" in payload:
                metrics = pd.DataFrame({
                    "pop_name": payload["pop_names"],
                    "OSI": payload["osi"], "DSI": payload["dsi"],
                })
                directory = os.path.join(logdir, "Boxplots_OSI_DSI")
                os.makedirs(directory, exist_ok=True)
                MetricsBoxplot(save_dir=directory,
                               filename=f"Epoch_{epoch}_osi_dsi").plot(
                    metrics=["OSI", "DSI"], metrics_df=metrics,
                    neuropixels_df=neuropixels_df,
                )
            mark("osi_dsi")

            if bool(payload["best"]):
                spikes = np.unpackbits(
                    payload["spikes_packed"], axis=-1,
                    count=network["n_nodes"],
                )
                inputs = payload["inputs"]
                _rate_boxplot(
                    payload["evoked_rates"], network, data_dir, plot_core_radius,
                    "Evoked rate (Hz)",
                    os.path.join(logdir, "Boxplots", "Evoked rate (Hz)"),
                    f"Epoch_{epoch}", neuropixels_df,
                )
                mark("evoked_rate")
                _raster(
                    spikes, inputs, network, data_dir,
                    os.path.join(logdir, "Raster_plots"),
                    f"Epoch_{epoch}_drifting_gratings", float(payload["frequency"]),
                    pre_delay, post_delay, plot_core_radius,
                )
                mark("evoked_raster")
                if "spikes_spont_packed" in payload:
                    spont = np.unpackbits(
                        payload["spikes_spont_packed"], axis=-1,
                        count=network["n_nodes"],
                    )
                    _rate_boxplot(
                        payload["spont_rates"], network, data_dir, plot_core_radius,
                        "Spontaneous rate (Hz)",
                        os.path.join(logdir, "Boxplots", "Spontaneous"),
                        f"Epoch_{epoch}", neuropixels_df,
                    )
                    mark("spont_rate")
                    _raster(
                        spont, payload["inputs_spont"], network, data_dir,
                        os.path.join(logdir, "Raster_plots"),
                        f"Epoch_{epoch}_spontaneous", float(payload["frequency"]),
                        pre_delay, post_delay, plot_core_radius,
                    )
                    mark("spont_raster")
        timings["total"] = time.perf_counter() - started
        print("PLOT_WORKER_TIMING " + " ".join(
            f"{name}={duration:.3f}s" for name, duration in timings.items()
        ), flush=True)
        succeeded = True
    finally:
        if succeeded:
            os.unlink(payload_path)


if __name__ == "__main__":
    epoch_main(sys.argv[1])
