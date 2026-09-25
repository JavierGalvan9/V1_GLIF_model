"""Correctness of the retinotopic LGN row order.

The order is purely a relabelling of LGN presynaptic ids, so the properties
worth pinning are that it is a bijection, that the connectivity it produces
describes the same graph, that it composes with the neuron layout and the CSR
edge order, and that the spike-stream gather in ``V1Column`` is the inverse of
the relabel rather than the relabel itself - the direction is easy to get
backwards and silently trains on permuted input.
"""

import numpy as np
import pytest

from v1_model_utils import spatial_layout


N_NODES = 512
N_EDGES = 4096
N_LGN = 64
N_LGN_EDGES = 1024


def _network(seed=7):
    """A small network with the fields the layout helpers touch."""
    rng = np.random.default_rng(seed)
    x = rng.uniform(-400, 400, N_NODES)
    z = rng.uniform(-400, 400, N_NODES)
    y = rng.choice([-100.0, -300.0], N_NODES)
    tf_id_to_bmtk_id = rng.permutation(N_NODES).astype(np.int32)
    bmtk_id_to_tf_id = np.full(N_NODES, -1, np.int32)
    bmtk_id_to_tf_id[tf_id_to_bmtk_id] = np.arange(N_NODES, dtype=np.int32)
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
        "readout_neuron_ids": rng.integers(0, N_NODES, (2, 8)).astype(np.int64),
        "n_nodes": N_NODES,
        "n_edges": N_EDGES,
        "node_params": {"t_ref": rng.uniform(1, 3, 17)},
        "tf_id_to_bmtk_id": tf_id_to_bmtk_id,
        "bmtk_id_to_tf_id": bmtk_id_to_tf_id,
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
            (
                rng.integers(0, N_NODES, N_LGN_EDGES),
                rng.integers(0, N_LGN, N_LGN_EDGES),
            ),
            axis=1,
        ).astype(np.int64),
        "weights": rng.normal(0, 1, N_LGN_EDGES),
        "syn_ids": rng.integers(0, 90, N_LGN_EDGES).astype(np.int64),
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


def _order(lgn, mode=spatial_layout.LGN_RETINOTOPIC):
    return spatial_layout.build_lgn_row_order(lgn, mode)


def test_identity_order_is_a_noop():
    _, lgn, _ = _network()
    order = _order(lgn, spatial_layout.LGN_ORIGINAL)
    assert order.is_identity
    assert order.n_rows == N_LGN
    assert spatial_layout.apply_lgn_row_order(order, lgn) is lgn


def test_unknown_order_rejected():
    _, lgn, _ = _network()
    with pytest.raises(ValueError, match="unknown LGN row order"):
        spatial_layout.build_lgn_row_order(lgn, "spiral")


def test_permutation_is_a_bijection():
    _, lgn, _ = _network()
    order = _order(lgn)
    assert np.array_equal(order.old_to_new[order.new_to_old], np.arange(N_LGN))
    assert np.array_equal(order.new_to_old[order.old_to_new], np.arange(N_LGN))


def test_relabel_round_trips():
    _, lgn, _ = _network()
    order = _order(lgn)
    ids = np.asarray(lgn["indices"])[:, 1]
    assert np.array_equal(order.relabel_to_canonical(order.relabel(ids)), ids)


def test_relabel_preserves_sentinels():
    _, lgn, _ = _network()
    order = _order(lgn)
    ids = np.array([-1, 0, N_LGN - 1, -1], np.int64)
    relabelled = order.relabel(ids)
    assert np.array_equal(relabelled[[0, 3]], [-1, -1])
    assert np.array_equal(order.relabel_to_canonical(relabelled), ids)


def test_only_the_presynaptic_column_moves():
    _, lgn, _ = _network()
    order = _order(lgn)
    reordered = spatial_layout.apply_lgn_row_order(order, lgn)
    original = np.asarray(lgn["indices"])
    moved = np.asarray(reordered["indices"])
    assert np.array_equal(moved[:, 0], original[:, 0])
    assert np.array_equal(moved[:, 1], order.old_to_new[original[:, 1]])
    # Every other field is per-edge and untouched: the edge set is the same,
    # only the row labels changed.
    for name in ("weights", "syn_ids"):
        assert np.array_equal(reordered[name], lgn[name])
    assert lgn["indices"] is not reordered["indices"]


def test_edge_set_is_preserved():
    """The relabelled graph must contain exactly the same connections."""
    _, lgn, _ = _network()
    order = _order(lgn)
    reordered = spatial_layout.apply_lgn_row_order(order, lgn)
    before = np.asarray(lgn["indices"])
    after = np.asarray(reordered["indices"]).copy()
    after[:, 1] = order.new_to_old[after[:, 1]]
    assert np.array_equal(after, before)


def test_rows_are_sorted_by_target_centroid():
    """The whole point: runtime row order follows postsynaptic position."""
    _, lgn, _ = _network()
    order = _order(lgn)
    reordered = spatial_layout.apply_lgn_row_order(order, lgn)
    indices = np.asarray(reordered["indices"])
    total = np.bincount(
        indices[:, 1], weights=indices[:, 0].astype(np.float64), minlength=N_LGN
    )
    counts = np.maximum(np.bincount(indices[:, 1], minlength=N_LGN), 1)
    centroid = total / counts
    assert np.all(np.diff(centroid) >= 0)


def test_cross_row_locality_improves():
    """The proxy the order exists to move, on the real ordering direction."""
    _, lgn, _ = _network()
    order = _order(lgn)
    reordered = spatial_layout.apply_lgn_row_order(order, lgn)

    def gap(source):
        indices = np.asarray(source["indices"])
        total = np.bincount(
            indices[:, 1], weights=indices[:, 0].astype(np.float64),
            minlength=N_LGN,
        )
        counts = np.maximum(np.bincount(indices[:, 1], minlength=N_LGN), 1)
        return np.abs(np.diff(total / counts)).mean()

    assert gap(reordered) < gap(lgn)


def test_rows_without_edges_are_placed_not_dropped():
    _, lgn, _ = _network()
    lgn = dict(lgn)
    indices = np.asarray(lgn["indices"])
    # Silence one row entirely and check it still gets a unique label.
    lgn["indices"] = indices[indices[:, 1] != 3]
    order = _order(lgn)
    assert order.n_rows == N_LGN
    assert np.array_equal(order.old_to_new[order.new_to_old], np.arange(N_LGN))


def test_presynaptic_id_beyond_row_count_rejected():
    _, lgn, _ = _network()
    order = _order(lgn)
    broken = dict(lgn)
    indices = np.array(lgn["indices"])
    indices[0, 1] = N_LGN
    broken["indices"] = indices
    with pytest.raises(ValueError, match="exceeds the"):
        spatial_layout.apply_lgn_row_order(order, broken)


def test_composes_with_neuron_layout_and_csr_edge_order():
    """The three permutations must stack without corrupting the graph."""
    network, lgn, bkg = _network()
    layout = spatial_layout.build_layout(network, spatial_layout.MORTON)
    network, lgn, bkg = spatial_layout.apply_layout(layout, network, lgn, bkg)
    order = _order(lgn)
    lgn = spatial_layout.apply_lgn_row_order(order, lgn)
    _, ordered_lgn, _, orders = spatial_layout.apply_csr_edge_order(
        network, lgn, bkg
    )
    indices = np.asarray(ordered_lgn["indices"])
    # CSR order sorts by presynaptic row, so the runtime rows come out sorted.
    assert np.all(np.diff(indices[:, 1]) >= 0)
    # And the edge permutation is what carries the weights, so undoing it must
    # recover the pre-CSR arrays exactly.
    assert np.array_equal(
        spatial_layout.to_original_edges(ordered_lgn["weights"], orders["lgn"]),
        np.asarray(lgn["weights"]),
    )


def test_gather_direction_matches_the_relabel():
    """``new_to_old`` gathers the canonical stream into runtime order.

    ``V1Column._permute_lgn_input`` gathers with ``new_to_old``; using
    ``old_to_new`` instead would still be a permutation and would not fail any
    shape check, so pin the direction against the relabel.
    """
    _, lgn, _ = _network()
    order = _order(lgn)
    stream = np.arange(N_LGN, dtype=np.float32) * 10.0
    runtime = stream[order.new_to_old]
    # Row r of the runtime stream must carry the canonical row that relabels
    # to r, which is what the kernels index when they read spikes[:, r].
    for canonical in range(N_LGN):
        assert runtime[order.relabel(np.array([canonical]))[0]] == stream[canonical]


def test_target_neuron_ids_survive_the_full_stack():
    """A postsynaptic id must still name the same neuron after all three."""
    network, lgn, bkg = _network()
    layout = spatial_layout.build_layout(network, spatial_layout.MORTON)
    moved_network, moved_lgn, moved_bkg = spatial_layout.apply_layout(
        layout, network, lgn, bkg
    )
    order = _order(moved_lgn)
    moved_lgn = spatial_layout.apply_lgn_row_order(order, moved_lgn)
    canonical_post = np.asarray(lgn["indices"])[:, 0]
    runtime_post = np.asarray(moved_lgn["indices"])[:, 0]
    assert np.array_equal(
        layout.relabel_to_canonical(runtime_post), canonical_post
    )
    assert runtime_post.max() < N_NODES
