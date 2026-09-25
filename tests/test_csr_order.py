"""The packed-key CSR ordering must match the lexsort it replaces.

Every operator asserts that the network it receives is already in its own CSR
order, so a permutation that differs from the reference by even one edge would
pair weights with the wrong synapses. These tests pin the equivalence on the
shapes the model actually builds, including the cases where the packing has to
give up and fall back to ``np.lexsort``.
"""

import numpy as np
import pytest

from v1_model_utils import csr_order


def _reference_order(keys):
    """The ``np.lexsort`` formulation the operators used before packing."""
    original = np.arange(np.asarray(keys[0]).size, dtype=np.uint32)
    return np.lexsort((original, *reversed(keys))).astype(np.uint32)


def _reference_pairs(post_ids, synapse_types):
    """The ``np.unique`` formulation ``_compact_pairs`` used before counting."""
    codes = post_ids.astype(np.uint64) * 256 + synapse_types.astype(np.uint64)
    return np.unique(codes, return_inverse=True)


def _random_edges(rng, n_edges, n_pre, n_post, n_types=8):
    return (
        rng.integers(0, n_pre, n_edges),
        rng.integers(0, n_post, n_edges),
        rng.integers(0, n_types, n_edges),
    )


@pytest.mark.parametrize("n_edges", [0, 1, 2, 17, 5000])
def test_edge_order_matches_lexsort(n_edges):
    rng = np.random.default_rng(n_edges)
    pre, post, types = _random_edges(rng, n_edges, 37, 23)
    keys = (pre, post, types)
    np.testing.assert_array_equal(csr_order.edge_order(keys), _reference_order(keys))


def test_edge_order_matches_lexsort_for_two_keys():
    """The external operator sorts on ``(pre, post)`` with no synapse type."""
    rng = np.random.default_rng(7)
    pre, post, _ = _random_edges(rng, 4000, 11, 9)
    keys = (pre, post)
    np.testing.assert_array_equal(csr_order.edge_order(keys), _reference_order(keys))


def test_ties_keep_the_callers_edge_order():
    """Equal keys must not be permuted: the CSR layout relies on stability."""
    keys = (np.zeros(64, np.int64), np.zeros(64, np.int64), np.zeros(64, np.int64))
    np.testing.assert_array_equal(
        csr_order.edge_order(keys), np.arange(64, dtype=np.uint32)
    )


def test_identity_check_agrees_with_the_sort():
    """``is_identity_order`` must answer exactly what sorting would reveal."""
    rng = np.random.default_rng(3)
    pre, post, types = _random_edges(rng, 3000, 13, 29)
    keys = (pre, post, types)
    assert csr_order.is_identity_order(keys) is False

    order = csr_order.edge_order(keys)
    sorted_keys = tuple(key[order] for key in keys)
    assert csr_order.is_identity_order(sorted_keys) is True
    np.testing.assert_array_equal(
        csr_order.edge_order(sorted_keys), np.arange(order.size, dtype=np.uint32)
    )


def test_identity_check_sees_a_single_swapped_edge():
    """One transposed pair is the failure the operators' assertion exists for."""
    rng = np.random.default_rng(5)
    pre, post, types = _random_edges(rng, 500, 7, 7)
    keys = tuple(key[csr_order.edge_order((pre, post, types))] for key in (pre, post, types))
    assert csr_order.is_identity_order(keys) is True

    swapped = [key.copy() for key in keys]
    for key in swapped:
        key[[0, -1]] = key[[-1, 0]]
    assert csr_order.is_identity_order(tuple(swapped)) is False


def test_keys_too_wide_to_pack_fall_back_to_lexsort():
    """Keys that overflow 64 bits must still sort exactly, not wrap around."""
    rng = np.random.default_rng(11)
    wide = rng.integers(0, 2**62, 400)
    keys = (wide, rng.integers(0, 2**10, 400))
    assert csr_order._packed_key(keys) is None
    np.testing.assert_array_equal(csr_order.edge_order(keys), _reference_order(keys))
    assert csr_order.is_identity_order(keys) is False
    assert csr_order.is_identity_order(tuple(np.sort(k) for k in (wide,))) is True


def test_negative_keys_fall_back_to_lexsort():
    """Unsigned packing cannot represent the sentinel ids some loaders carry."""
    keys = (np.array([-3, 5, -3, 0]), np.array([1, 0, 0, 2]))
    assert csr_order._packed_key(keys) is None
    np.testing.assert_array_equal(csr_order.edge_order(keys), _reference_order(keys))


@pytest.mark.parametrize("n_edges", [0, 1, 1000])
def test_compact_pairs_matches_numpy_unique(n_edges):
    rng = np.random.default_rng(n_edges + 1)
    post = rng.integers(0, 50, n_edges).astype(np.uint32)
    types = rng.integers(0, 6, n_edges).astype(np.uint8)
    codes, pair_ids = csr_order.compact_pairs(post, types)
    ref_codes, ref_ids = _reference_pairs(post, types)

    np.testing.assert_array_equal(codes.astype(np.uint64), ref_codes)
    np.testing.assert_array_equal(pair_ids.astype(np.int64), ref_ids)
    np.testing.assert_array_equal((codes >> 8).astype(np.uint32)[pair_ids], post)
    np.testing.assert_array_equal((codes & 0xFF).astype(np.uint8)[pair_ids], types)


def test_compact_pairs_covers_the_full_synapse_type_range():
    """Type 255 must not collide with the neighbouring postsynaptic neuron."""
    post = np.array([0, 0, 1, 1], np.uint32)
    types = np.array([0, 255, 0, 255], np.uint8)
    codes, pair_ids = csr_order.compact_pairs(post, types)
    ref_codes, ref_ids = _reference_pairs(post, types)
    np.testing.assert_array_equal(codes.astype(np.uint64), ref_codes)
    np.testing.assert_array_equal(pair_ids.astype(np.int64), ref_ids)
    assert codes.size == 4
