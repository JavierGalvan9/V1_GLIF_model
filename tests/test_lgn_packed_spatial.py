"""Parity tests for packed LGN separable spatial filtering."""

import pytest
import tensorflow as tf

from lgn_model.lgn import LGN, _sample_spatial


def _reference(lgn, movie, bmtk_compat):
    dominant = []
    non_dominant = []
    for index in range(len(lgn.spatial_range_indices)):
        convolved = tf.nn.conv2d(
            movie, lgn.vertical_filters[index], strides=1, padding="SAME"
        )
        convolved = tf.nn.conv2d(
            convolved, lgn.horizontal_filters[index], strides=1, padding="SAME"
        )
        if bmtk_compat:
            convolved *= lgn.edge_reciprocals[index]
        flattened = tf.reshape(convolved[..., 0], (tf.shape(movie)[0], -1))
        dominant.append(
            _sample_spatial(
                flattened,
                lgn.dominant_sample_indices[index],
                lgn.dominant_sample_weights[index],
            )
        )
        non_dominant.append(
            _sample_spatial(
                flattened,
                lgn.non_dominant_sample_indices[index],
                lgn.non_dominant_sample_weights[index],
            )
        )
    dominant = tf.gather(
        tf.concat(dominant, axis=1), lgn.sorted_neuron_ids_indices, axis=1
    )
    non_dominant = tf.gather(
        tf.concat(non_dominant, axis=1), lgn.composite_sort_indices, axis=1
    )
    return dominant, non_dominant


@pytest.mark.parametrize("dtype", [tf.float32, tf.float16])
@pytest.mark.parametrize("bmtk_compat", [False, True])
def test_packed_spatial_matches_sequential(dtype, bmtk_compat):
    lgn = LGN(n_input=17400, dtype=dtype, data_dir="GLIF_network_nll_full")
    movie = tf.random.stateless_uniform(
        (7, lgn.row_size, lgn.col_size, 1), seed=(137, 311), dtype=dtype
    )
    expected = _reference(lgn, movie, bmtk_compat)
    actual = lgn.spatial_response(movie, bmtk_compat)
    tolerance = 2e-3 if dtype == tf.float16 else 2e-6
    for want, got in zip(expected, actual):
        tf.debugging.assert_near(got, want, atol=tolerance, rtol=tolerance)
