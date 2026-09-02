"""Canonical identities for persisted metrics and plotted LGN activity."""

import numpy as np
import pandas as pd

from v1_model_utils import model_metrics_analysis, spatial_layout


def test_saved_metrics_use_canonical_neuron_ids(monkeypatch, tmp_path):
    runtime_populations = np.array(["p2", "p0", "p1"])
    monkeypatch.setattr(
        model_metrics_analysis.other_v1_utils,
        "pop_names",
        lambda *args, **kwargs: runtime_populations,
    )
    canonical_ids = np.array([2, 0, 1])
    evoked_rates = np.array([[[1.0, 2.0, 3.0], [3.0, 4.0, 5.0]]])

    model_metrics_analysis.calculate_OSI_DSI(
        evoked_rates,
        {"n_nodes": 3},
        DG_angles=np.array([0, 180]),
        core_radius=None,
        directory=tmp_path,
        save_df=True,
        neuron_ids=canonical_ids,
    )

    spontaneous_rates = np.array([[[10.0, 20.0, 30.0]]])
    model_metrics_analysis.calculate_OSI_DSI(
        spontaneous_rates,
        {"n_nodes": 3},
        session="spontaneous",
        core_radius=None,
        directory=tmp_path,
        save_df=True,
        neuron_ids=canonical_ids,
    )

    saved = pd.read_csv(tmp_path / "v1_features_df.csv", sep=" ")
    np.testing.assert_array_equal(saved["node_id"], [0, 1, 2])
    np.testing.assert_array_equal(saved["pop_name"], ["p0", "p1", "p2"])
    np.testing.assert_allclose(saved["Ave_Rate(Hz)"], [3.0, 4.0, 2.0])
    np.testing.assert_allclose(saved["firing_rate_sp"], [20.0, 30.0, 10.0])


def test_core_metrics_keep_the_selected_neurons_canonical_ids(
    monkeypatch, tmp_path
):
    core_mask = np.array([True, False, True, False])
    runtime_populations = np.array(["p2", "p0", "p3", "p1"])
    monkeypatch.setattr(
        model_metrics_analysis.other_v1_utils,
        "isolate_core_neurons",
        lambda *args, **kwargs: core_mask,
    )
    monkeypatch.setattr(
        model_metrics_analysis.other_v1_utils,
        "pop_names",
        lambda *args, **kwargs: runtime_populations[core_mask],
    )
    spikes = np.zeros((1, 2, 2, 4), dtype=np.uint8)
    spikes[..., 0] = 1

    model_metrics_analysis.ModelMetricsAnalysis(
        spikes,
        np.array([0, 180]),
        {"n_nodes": 4},
        drifting_gratings_init=0,
        drifting_gratings_end=2,
        core_radius=400,
        save_df=True,
        df_directory=tmp_path,
        neuron_ids=np.array([2, 0, 3, 1]),
    )

    saved = pd.read_csv(tmp_path / "v1_features_df.csv", sep=" ")
    np.testing.assert_array_equal(saved["node_id"], [2, 3])
    np.testing.assert_array_equal(saved["pop_name"], ["p2", "p3"])


def test_lgn_values_follow_the_runtime_row_order():
    lgn = {
        "n_inputs": 4,
        "indices": np.array([[0, 0], [9, 0], [8, 1], [1, 2], [2, 3]]),
    }
    order = spatial_layout.build_lgn_row_order(
        lgn, spatial_layout.LGN_RETINOTOPIC
    )
    canonical = np.arange(4)
    np.testing.assert_array_equal(
        order.to_runtime(canonical), canonical[order.new_to_old]
    )
