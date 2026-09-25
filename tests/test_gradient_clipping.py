"""Behavioral tests for training-gradient stabilization."""

import tensorflow as tf

from v1_model_utils.optimizers import clip_gradients_by_global_norm


def test_global_norm_clipping_preserves_missing_gradients_and_reports_raw_norm():
    gradients = [tf.constant([3.0, 4.0]), None, tf.constant([0.0])]

    clipped, raw_norm = clip_gradients_by_global_norm(gradients, 2.5)

    tf.debugging.assert_near(raw_norm, 5.0)
    tf.debugging.assert_near(clipped[0], [1.5, 2.0])
    assert clipped[1] is None
    tf.debugging.assert_equal(clipped[2], [0.0])


def test_nonpositive_global_norm_limit_leaves_gradients_unchanged():
    gradients = [tf.constant([3.0, 4.0])]

    unclipped, raw_norm = clip_gradients_by_global_norm(gradients, 0.0)

    tf.debugging.assert_near(raw_norm, 5.0)
    tf.debugging.assert_equal(unclipped[0], gradients[0])
