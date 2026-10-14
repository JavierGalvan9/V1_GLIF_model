"""Exact host-side projection sharing must preserve edge/type semantics."""

import numpy as np
import pytest

from v1_model_utils.cuda_csr_recurrent.canonical_projection import canonical_pairs


def test_duplicate_basis_rows_share_projection_without_changing_edge_order():
    basis = np.array([[1, 2], [3, 4], [1, 2]], np.float32)
    posts = np.array([7, 7, 9, 7], np.uint32)
    types = np.array([0, 2, 2, 1], np.uint8)
    edges = np.array([3, 1, 0, 2, 1], np.uint32)
    result = canonical_pairs(posts, types, edges, basis)
    assert result['n_pairs'] == 3
    np.testing.assert_array_equal(result['pair_posts'][result['pair_ids']], posts[edges])
    np.testing.assert_array_equal(
        basis[result['pair_types'][result['pair_ids']]].view(np.uint32),
        basis[types[edges]].view(np.uint32),
    )


def test_signed_zero_and_nan_payloads_remain_distinct():
    bits = np.array([[0], [0x80000000], [0x7FC00001], [0x7FC00002], [0x7FC00001]], np.uint32)
    result = canonical_pairs(
        np.zeros(5, np.uint32), np.arange(5, dtype=np.uint8),
        np.arange(5, dtype=np.uint32), bits.view(np.float32),
    )
    assert result['n_pairs'] == 4
    assert result['pair_ids'][2] == result['pair_ids'][4]
    np.testing.assert_array_equal(bits[result['pair_types'][result['pair_ids']]], bits)


def test_empty_edges():
    result = canonical_pairs(
        np.array([], np.uint32), np.array([], np.uint8), np.array([], np.uint32),
        np.ones((2, 4), np.float32),
    )
    assert result['n_pairs'] == 0
    assert result['pair_ids'].dtype == np.uint32


@pytest.mark.parametrize('basis,types', [(np.ones(4), [0]), (np.ones((1, 4)), [1])])
def test_invalid_basis_rejected(basis, types):
    with pytest.raises(ValueError):
        canonical_pairs(np.array([0], np.uint32), np.array(types, np.uint8),
                        np.array([0], np.uint32), basis)
