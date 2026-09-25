"""Correctness of the spatial neuron layout used by the CUDA CSR kernels."""

import numpy as np
import pytest

from general_utils.analysis_utils import (
    freedman_diaconis,
    freedman_diaconis_bin_sizes,
)
from v1_model_utils import spatial_layout


N_NODES = 512
N_EDGES = 4096
N_LGN = 64


def test_histogram_helpers_accept_integer_arrays_with_numpy_2():
    values = np.arange(10, dtype=np.int32)
    width = freedman_diaconis(values)
    bins = freedman_diaconis_bin_sizes(
        np.column_stack((values, values * 2))
    )

    assert np.issubdtype(type(width), np.floating)
    assert width > 0
    assert len(bins) == 2
    assert all(isinstance(value, int) for value in bins)


def _network(seed=7):
    rng = np.random.default_rng(seed)
    # Two cortical sheets so the depth key has something to separate.
    x = rng.uniform(-400, 400, N_NODES)
    z = rng.uniform(-400, 400, N_NODES)
    y = rng.choice([-100.0, -300.0], N_NODES)
    tf_id_to_bmtk_id = rng.permutation(N_NODES).astype(np.int32)
    bmtk_id_to_tf_id = np.full(N_NODES, -1, np.int32)
    bmtk_id_to_tf_id[tf_id_to_bmtk_id] = np.arange(N_NODES, dtype=np.int32)
    # Leave a sentinel behind to mimic an unselected BMTK node.
    bmtk_id_to_tf_id[tf_id_to_bmtk_id[0]] = -1
    indices = np.stack(
        (rng.integers(0, N_NODES, N_EDGES), rng.integers(0, N_NODES, N_EDGES)),
        axis=1,
    ).astype(np.int64)
    network = {
        "x": x,
        "y": y,
        "z": z,
        "tuning_angle": rng.uniform(0, 360, N_NODES),
        "node_type_ids": rng.integers(0, 17, N_NODES).astype(np.int32),
        "l5e_neuron_sel": rng.random(N_NODES) < 0.1,
        "n_nodes": N_NODES,
        "n_edges": N_EDGES,
        "node_params": {"t_ref": rng.uniform(1, 3, 17)},
        "tf_id_to_bmtk_id": tf_id_to_bmtk_id,
        "bmtk_id_to_tf_id": bmtk_id_to_tf_id,
        "readout_neuron_ids": rng.integers(0, N_NODES, (2, 8)).astype(np.int64),
        "data_dir": "GLIF_network",
        "synapses": {
            "indices": indices,
            "weights": rng.normal(0, 1, N_EDGES),
            "syn_ids": rng.integers(0, 90, N_EDGES).astype(np.int64),
            "delays": rng.uniform(1, 3, N_EDGES),
        },
    }
    lgn = {
        "n_inputs": N_LGN,
        "indices": np.stack(
            (rng.integers(0, N_NODES, 1024), rng.integers(0, N_LGN, 1024)), axis=1
        ).astype(np.int64),
        "weights": rng.normal(0, 1, 1024),
        "syn_ids": rng.integers(0, 90, 1024).astype(np.int64),
    }
    bkg = {
        "n_inputs": 1,
        "indices": np.stack(
            (rng.integers(0, N_NODES, 256), np.zeros(256, np.int64)), axis=1
        ).astype(np.int64),
        "weights": rng.normal(0, 1, 256),
        "syn_ids": rng.integers(0, 90, 256).astype(np.int64),
    }
    return network, lgn, bkg


def test_identity_layout_is_a_noop():
    network, lgn, bkg = _network()
    layout = spatial_layout.build_layout(network, spatial_layout.CANONICAL)
    assert layout.is_identity
    assert spatial_layout.apply_layout(layout, network, lgn, bkg)[0] is network


def test_unknown_layout_rejected():
    network, _, _ = _network()
    with pytest.raises(ValueError, match="unknown neuron layout"):
        spatial_layout.build_layout(network, "hilbert")


def test_permutation_is_a_bijection():
    network, _, _ = _network()
    layout = spatial_layout.build_layout(network, spatial_layout.MORTON)
    assert np.array_equal(
        layout.old_to_new[layout.new_to_old], np.arange(N_NODES)
    )
    assert np.array_equal(
        layout.new_to_old[layout.old_to_new], np.arange(N_NODES)
    )
    assert np.array_equal(np.sort(layout.new_to_old), np.arange(N_NODES))


def test_neuron_aligned_round_trip():
    network, _, _ = _network()
    layout = spatial_layout.build_layout(network, spatial_layout.MORTON)
    values = np.arange(N_NODES * 2, dtype=np.float64).reshape(N_NODES, 2)
    assert np.array_equal(layout.to_canonical(layout.to_runtime(values)), values)


def test_wrong_length_rejected():
    network, _, _ = _network()
    layout = spatial_layout.build_layout(network, spatial_layout.MORTON)
    with pytest.raises(ValueError, match="expected .* neurons"):
        layout.to_runtime(np.zeros(N_NODES + 1))


def test_applied_layout_describes_the_same_graph():
    network, lgn, bkg = _network()
    layout = spatial_layout.build_layout(network, spatial_layout.MORTON)
    permuted, permuted_lgn, permuted_bkg = spatial_layout.apply_layout(
        layout, network, lgn, bkg
    )

    def edge_set(indices, source):
        weights = np.asarray(source["weights"])
        syn_ids = np.asarray(source["syn_ids"])
        return sorted(zip(indices[:, 0].tolist(), indices[:, 1].tolist(),
                          weights.tolist(), syn_ids.tolist()))

    # Recurrent edges keep their identity once endpoints are mapped back.
    restored = np.array(permuted["synapses"]["indices"])
    restored[:, 0] = layout.new_to_old[restored[:, 0]]
    restored[:, 1] = layout.new_to_old[restored[:, 1]]
    assert edge_set(restored, permuted["synapses"]) == edge_set(
        np.asarray(network["synapses"]["indices"]), network["synapses"]
    )

    # External populations only relabel their postsynaptic column.
    for original, moved in ((lgn, permuted_lgn), (bkg, permuted_bkg)):
        restored = np.array(moved["indices"])
        restored[:, 0] = layout.new_to_old[restored[:, 0]]
        assert edge_set(restored, moved) == edge_set(
            np.asarray(original["indices"]), original
        )


def test_edge_order_is_preserved():
    """Trainable weight order must not move: checkpoints depend on it."""
    network, lgn, bkg = _network()
    layout = spatial_layout.build_layout(network, spatial_layout.MORTON)
    permuted, permuted_lgn, _ = spatial_layout.apply_layout(
        layout, network, lgn, bkg
    )
    for original, moved in (
        (network["synapses"], permuted["synapses"]),
        (lgn, permuted_lgn),
    ):
        for field in ("weights", "syn_ids"):
            assert np.array_equal(
                np.asarray(original[field]), np.asarray(moved[field])
            )


def test_neuron_aligned_fields_follow_the_layout():
    network, lgn, bkg = _network()
    layout = spatial_layout.build_layout(network, spatial_layout.MORTON)
    permuted, _, _ = spatial_layout.apply_layout(layout, network, lgn, bkg)
    for field in ("x", "y", "z", "tuning_angle", "node_type_ids",
                  "l5e_neuron_sel", "tf_id_to_bmtk_id"):
        assert np.array_equal(
            np.asarray(permuted[field]),
            np.asarray(network[field])[layout.new_to_old],
        ), field
    # Per-type parameters and scalars are untouched.
    assert permuted["node_params"] is network["node_params"]
    assert permuted["n_nodes"] == N_NODES


def test_sentinel_preserved_and_inverse_consistent():
    network, lgn, bkg = _network()
    layout = spatial_layout.build_layout(network, spatial_layout.MORTON)
    permuted, _, _ = spatial_layout.apply_layout(layout, network, lgn, bkg)
    forward = np.asarray(permuted["tf_id_to_bmtk_id"])
    inverse = np.asarray(permuted["bmtk_id_to_tf_id"])
    original_inverse = np.asarray(network["bmtk_id_to_tf_id"])
    assert (inverse == -1).sum() == (original_inverse == -1).sum()
    mapped = inverse >= 0
    # Every retained BMTK node still points at the neuron holding its data.
    assert np.array_equal(forward[inverse[mapped]], np.flatnonzero(mapped))


def test_readout_ids_point_at_the_same_neurons():
    network, lgn, bkg = _network()
    layout = spatial_layout.build_layout(network, spatial_layout.MORTON)
    permuted, _, _ = spatial_layout.apply_layout(layout, network, lgn, bkg)
    assert np.array_equal(
        layout.new_to_old[np.asarray(permuted["readout_neuron_ids"])],
        np.asarray(network["readout_neuron_ids"]),
    )


def test_neuron_state_translation_round_trips_nested_payloads():
    network, _, _ = _network()
    layout = spatial_layout.build_layout(network, spatial_layout.MORTON)
    payload = {
        "v1_ema": np.linspace(0, 1, N_NODES),
        "min_val_loss": 0.5,
        "epoch_metric_values": {"rate": [1.0, 2.0, 3.0]},
        "adaptative_crowd_osi_state": {
            "core": {
                "ori_real": np.arange(N_NODES, dtype=np.float32),
                "weight_sum": 12.0,
            },
        },
    }
    runtime = spatial_layout.translate_neuron_state(layout, payload, to_runtime=True)
    assert np.array_equal(
        runtime["v1_ema"], payload["v1_ema"][layout.new_to_old]
    )
    assert np.array_equal(
        runtime["adaptative_crowd_osi_state"]["core"]["ori_real"],
        payload["adaptative_crowd_osi_state"]["core"]["ori_real"][layout.new_to_old],
    )
    # Non neuron-aligned entries pass through untouched.
    assert runtime["min_val_loss"] == 0.5
    assert runtime["epoch_metric_values"]["rate"] == [1.0, 2.0, 3.0]
    assert runtime["adaptative_crowd_osi_state"]["core"]["weight_sum"] == 12.0

    canonical = spatial_layout.translate_neuron_state(
        layout, runtime, to_runtime=False
    )
    assert np.allclose(canonical["v1_ema"], payload["v1_ema"])
    assert np.allclose(
        canonical["adaptative_crowd_osi_state"]["core"]["ori_real"],
        payload["adaptative_crowd_osi_state"]["core"]["ori_real"],
    )


def test_delayed_presynaptic_relabelling_keeps_delay_blocks():
    network, _, _ = _network()
    layout = spatial_layout.build_layout(network, spatial_layout.MORTON)
    delays = np.repeat([0, 1, 2], N_NODES)
    neurons = np.tile(np.arange(N_NODES), 3)
    expanded = neurons + N_NODES * delays
    runtime = layout.relabel_delayed(expanded)
    # The delay block is preserved and the neuron inside it is relabelled.
    assert np.array_equal(runtime // N_NODES, delays)
    assert np.array_equal(runtime % N_NODES, layout.old_to_new[neurons])
    assert np.array_equal(
        layout.relabel_delayed(runtime, to_canonical=True), expanded
    )


def test_morton_layout_improves_postsynaptic_locality():
    """The whole point: postsynaptic ids inside a row get closer together."""
    network, lgn, bkg = _network()
    layout = spatial_layout.build_layout(network, spatial_layout.MORTON)
    permuted, _, _ = spatial_layout.apply_layout(layout, network, lgn, bkg)

    def mean_row_spread(indices):
        order = np.lexsort((indices[:, 0], indices[:, 1]))
        pre = indices[order, 1]
        post = indices[order, 0].astype(np.int64)
        boundaries = np.flatnonzero(np.diff(pre)) + 1
        return np.mean([
            np.abs(np.diff(chunk)).mean()
            for chunk in np.split(post, boundaries)
            if chunk.size > 1
        ])

    canonical = mean_row_spread(np.asarray(network["synapses"]["indices"]))
    morton = mean_row_spread(np.asarray(permuted["synapses"]["indices"]))
    # Random connectivity on a spatial substrate: Morton must not be worse.
    assert morton <= canonical


# --- CSR edge ordering -------------------------------------------------------

def test_resolve_max_delay_matches_the_model_rule():
    network, lgn, bkg = _network()
    lgn = {**lgn, "delays": np.full(1024, 2.4)}
    bkg = {**bkg, "delays": np.full(256, 1.2)}
    network["synapses"]["delays"] = np.full(N_EDGES, 2.9)
    assert spatial_layout.resolve_max_delay(network, lgn, bkg) == 3
    # A positive cap clips the value taken from the data.
    assert spatial_layout.resolve_max_delay(network, lgn, bkg, max_delay=2) == 2
    # A cap above the data does not extend it.
    assert spatial_layout.resolve_max_delay(network, lgn, bkg, max_delay=9) == 3


def test_csr_edge_order_makes_the_operator_permutation_the_identity():
    """The crux: our replicated key must equal the operator's own key."""
    from v1_model_utils.cuda_csr_recurrent import (
        build_csr_connectivity as build_recurrent,
    )
    from v1_model_utils.cuda_csr_external import (
        build_csr_connectivity as build_external,
    )

    network, lgn, bkg = _network()
    network["synapses"]["delays"] = np.random.default_rng(3).uniform(1, 3, N_EDGES)
    ordered_net, ordered_lgn, ordered_bkg, orders = (
        spatial_layout.apply_csr_edge_order(network, lgn, bkg)
    )
    steps = spatial_layout.resolve_max_delay(network, lgn, bkg)

    # Recurrent: rebuild the delay-expanded indices the model passes in.
    synapses = ordered_net["synapses"]
    indices = np.asarray(synapses["indices"])
    delay_steps = np.round(np.clip(np.asarray(synapses["delays"]), 1.0, steps)).astype(np.int64)
    expanded = np.stack(
        (indices[:, 0], indices[:, 1] + N_NODES * (delay_steps - 1)), axis=1
    )
    recurrent = build_recurrent(
        expanded, np.asarray(synapses["syn_ids"]),
        n_pre=N_NODES * steps, n_post=N_NODES, weights_csr_ordered=True,
    )
    np.testing.assert_array_equal(
        recurrent.edge_order, np.arange(N_EDGES, dtype=np.uint32)
    )

    for source, n_pre in ((ordered_lgn, N_LGN), (ordered_bkg, 1)):
        external = build_external(
            np.asarray(source["indices"]), np.asarray(source["syn_ids"]),
            n_pre=n_pre, n_post=N_NODES, weights_csr_ordered=True,
        )
        np.testing.assert_array_equal(
            external.edge_order, np.arange(np.asarray(source["indices"]).shape[0],
                                           dtype=np.uint32)
        )
    assert set(orders) == {"recurrent", "lgn", "bkg"}


def test_csr_edge_order_preserves_each_edge():
    network, lgn, bkg = _network()
    ordered_net, ordered_lgn, _, orders = spatial_layout.apply_csr_edge_order(
        network, lgn, bkg
    )
    for original, moved, key in (
        (network["synapses"], ordered_net["synapses"], "recurrent"),
        (lgn, ordered_lgn, "lgn"),
    ):
        order = orders[key]
        for field in ("indices", "weights", "syn_ids"):
            np.testing.assert_array_equal(
                np.asarray(moved[field]), np.asarray(original[field])[order]
            )


def test_edge_value_translation_round_trips():
    network, lgn, bkg = _network()
    _, _, _, orders = spatial_layout.apply_csr_edge_order(network, lgn, bkg)
    order = orders["recurrent"]
    canonical = np.arange(N_EDGES, dtype=np.float32)
    runtime = spatial_layout.to_csr_edges(canonical, order)
    assert np.array_equal(
        spatial_layout.to_original_edges(runtime, order), canonical
    )
    # Reordering is a permutation, not a change of contents.
    assert np.array_equal(np.sort(runtime), canonical)


def test_declared_csr_order_is_verified_not_trusted():
    """A caller lying about the ordering must fail, not silently mispair."""
    from v1_model_utils.cuda_csr_recurrent import build_csr_connectivity

    indices = np.array([[2, 0], [0, 1], [1, 0]], dtype=np.int64)
    types = np.array([0, 1, 0], dtype=np.int64)
    with pytest.raises(ValueError, match="not the identity"):
        build_csr_connectivity(
            indices, types, n_pre=2, n_post=3, weights_csr_ordered=True
        )


def test_missing_csr_declaration_is_refused_under_direct_csr():
    from v1_model_utils.cuda_csr_recurrent import wrapper

    indices = np.array([[0, 0], [1, 0], [2, 1]], dtype=np.int64)
    types = np.array([0, 0, 0], dtype=np.int64)
    connectivity = wrapper.build_csr_connectivity(indices, types, n_pre=2, n_post=3)
    assert not connectivity.weights_csr_ordered
    if wrapper.DIRECT_CSR:
        with pytest.raises(ValueError, match="CSR edge order"):
            wrapper.require_csr_ordered_weights(connectivity, "recurrent connectivity")
    else:
        wrapper.require_csr_ordered_weights(connectivity, "recurrent connectivity")


def test_tracked_traces_are_saved_in_canonical_order():
    """Per-neuron traces recorded in runtime order must round-trip to canonical."""
    network, _, _ = _network()
    layout = spatial_layout.build_layout(network, spatial_layout.MORTON)
    timesteps = 5
    # Value of a trace column identifies the canonical neuron it belongs to.
    canonical_trace = np.tile(np.arange(N_NODES, dtype=np.float32), (timesteps, 1))
    runtime_trace = canonical_trace[:, layout.new_to_old]

    restored = spatial_layout.tracked_to_canonical(layout, runtime_trace, axis=1)
    np.testing.assert_array_equal(restored, canonical_trace)


def test_tracked_subset_traces_order_by_canonical_id():
    """A core-only trace records a subset of neurons in runtime id order."""
    network, _, _ = _network()
    layout = spatial_layout.build_layout(network, spatial_layout.MORTON)
    # A radius-style mask evaluated in runtime space, as isolate_core_neurons gives.
    x = np.asarray(network["x"])[layout.new_to_old]
    z = np.asarray(network["z"])[layout.new_to_old]
    core = (x ** 2 + z ** 2) < 200.0 ** 2
    tracked_ids = np.flatnonzero(core)
    assert 0 < tracked_ids.size < N_NODES

    # Column j holds the canonical id of the runtime neuron tracked_ids[j].
    runtime_trace = layout.new_to_old[tracked_ids].astype(np.float32)[None, :]
    restored = spatial_layout.tracked_to_canonical(
        layout, runtime_trace, tracked_ids, axis=1
    )
    # The same neurons a canonical run would track, in ascending canonical id.
    expected = np.sort(layout.new_to_old[tracked_ids]).astype(np.float32)
    np.testing.assert_array_equal(restored[0], expected)
    # The set of tracked neurons is layout-independent.
    assert set(restored[0].tolist()) == set(layout.new_to_old[tracked_ids].tolist())


def test_tracked_translation_is_a_noop_for_canonical():
    network, _, _ = _network()
    layout = spatial_layout.build_layout(network, spatial_layout.CANONICAL)
    trace = np.arange(N_NODES, dtype=np.float32)[None, :]
    assert spatial_layout.tracked_to_canonical(layout, trace, axis=1) is trace
