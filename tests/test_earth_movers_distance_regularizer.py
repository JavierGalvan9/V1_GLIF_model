import unittest
from unittest import mock

import numpy as np
import tensorflow as tf

from v1_model_utils.loss_functions import EarthMoversDistanceRegularizer


def _network(initial_weights):
    initial_weights = np.asarray(initial_weights, dtype=np.float32)
    n_edges = initial_weights.size
    return {
        "node_params": {
            "V_th": np.array([2.0, 3.0], dtype=np.float32),
            "E_L": np.array([0.0, 1.0], dtype=np.float32),
        },
        "node_type_ids": np.array([0, 1], dtype=np.int32),
        "synapses": {
            "indices": np.column_stack(
                [np.arange(n_edges) % 2, np.arange(n_edges) % 2]
            ).astype(np.int32),
            "weights": initial_weights,
        },
    }


def _reference_emd(current, initial, groups, strength):
    voltage_scale = np.array([2.0, 2.0], dtype=np.float32)
    scaled_initial = initial / voltage_scale[np.arange(initial.size) % 2]
    losses = []
    for group in np.unique(groups):
        mask = groups == group
        losses.append(
            np.mean(np.abs(np.sort(current[mask]) - np.sort(scaled_initial[mask])))
        )
    return np.float32(strength * np.mean(losses)) if losses else np.float32(0.0)


def _legacy_tf_emd(current, initial, groups, strength):
    voltage_scale = np.array([2.0, 2.0], dtype=np.float32)
    scaled_initial = initial / voltage_scale[np.arange(initial.size) % 2]
    losses = []
    for group in np.unique(groups):
        indices = np.flatnonzero(groups == group)
        current_group = tf.gather(current, indices)
        initial_group = tf.constant(np.sort(scaled_initial[indices]), tf.float32)
        losses.append(
            tf.reduce_mean(tf.abs(tf.sort(current_group) - initial_group))
        )
    return tf.cast(strength, tf.float32) * tf.reduce_mean(tf.stack(losses))


class EarthMoversDistanceRegularizerTests(unittest.TestCase):
    def _regularizer(self, initial, groups, strength=1.75):
        patcher = mock.patch(
            "v1_model_utils.loss_functions.other_v1_utils.connection_type_ids",
            return_value=np.asarray(groups, dtype=np.int64),
        )
        with patcher:
            return EarthMoversDistanceRegularizer(strength, _network(initial))

    def test_matches_reference_value_and_dense_gradient_for_vector(self):
        initial = np.array([8.0, 2.0, 6.0, 10.0, 4.0], dtype=np.float32)
        groups = np.array([7, 2, 7, 2, 9], dtype=np.int64)
        current = tf.Variable([1.5, 3.0, 2.0, 6.5, 1.0], dtype=tf.float32)
        regularizer = self._regularizer(initial, groups)

        with tf.GradientTape() as tape:
            loss = regularizer(current)
        gradient = tf.convert_to_tensor(tape.gradient(loss, current))

        with tf.GradientTape() as legacy_tape:
            legacy_loss = _legacy_tf_emd(current, initial, groups, 1.75)
        legacy_gradient = tf.convert_to_tensor(
            legacy_tape.gradient(legacy_loss, current)
        )

        expected = _reference_emd(current.numpy(), initial, groups, 1.75)
        self.assertEqual(loss.dtype, tf.float32)
        np.testing.assert_allclose(loss.numpy(), expected, rtol=1e-6, atol=1e-6)
        self.assertEqual(gradient.shape, current.shape)
        np.testing.assert_array_equal(loss.numpy(), legacy_loss.numpy())
        np.testing.assert_allclose(
            gradient.numpy(), legacy_gradient.numpy(), rtol=2e-7, atol=1e-7
        )

        epsilon = np.float32(1e-3)
        numerical = np.empty(current.shape, dtype=np.float32)
        values = current.numpy()
        for index in range(values.size):
            plus = values.copy()
            minus = values.copy()
            plus[index] += epsilon
            minus[index] -= epsilon
            numerical[index] = (
                _reference_emd(plus, initial, groups, 1.75)
                - _reference_emd(minus, initial, groups, 1.75)
            ) / (2 * epsilon)
        np.testing.assert_allclose(gradient.numpy(), numerical, rtol=2e-3, atol=2e-3)

    def test_column_weights_and_singleton_groups_preserve_shape_and_value(self):
        initial = np.array([4.0, 8.0, 2.0], dtype=np.float32)
        groups = np.array([3, 8, 3], dtype=np.int64)
        current = tf.Variable([[2.5], [5.0], [0.5]], dtype=tf.float64)
        regularizer = self._regularizer(initial, groups, strength=0.5)

        with tf.GradientTape() as tape:
            loss = regularizer(current)
        gradient = tape.gradient(loss, current)

        expected = _reference_emd(current.numpy()[:, 0], initial, groups, 0.5)
        self.assertEqual(loss.dtype, tf.float32)
        np.testing.assert_allclose(loss.numpy(), expected, rtol=1e-6, atol=1e-6)
        self.assertEqual(gradient.shape, current.shape)

    def test_gradient_is_dense_and_matches_legacy_for_many_unequal_groups(self):
        # Production has 359 groups whose sizes span several orders of
        # magnitude, so exercise unequal sizes rather than a uniform split.
        rng = np.random.default_rng(11)
        sizes = [1, 2, 3, 5, 8, 13, 21, 34]
        groups = np.repeat(np.arange(len(sizes)) * 3 + 2, sizes).astype(np.int64)
        n_edges = groups.size
        # Distinct values keep the argsort permutation unique, so the analytic
        # gradient and autodiff-through-sort must agree elementwise.
        initial = rng.permutation(n_edges).astype(np.float32) + 1.0
        current = tf.Variable(
            rng.permutation(n_edges).astype(np.float32) * 0.25, dtype=tf.float32
        )
        regularizer = self._regularizer(initial, groups)

        with tf.GradientTape() as tape:
            loss = regularizer(current)
        gradient = tape.gradient(loss, current)

        with tf.GradientTape() as legacy_tape:
            legacy_loss = _legacy_tf_emd(current, initial, groups, 1.75)
        legacy_gradient = tf.convert_to_tensor(
            legacy_tape.gradient(legacy_loss, current)
        )

        # The analytic gradient scatters straight into edge order, so it must
        # arrive dense rather than as IndexedSlices over a full permutation.
        self.assertIsInstance(gradient, tf.Tensor)
        self.assertEqual(gradient.shape, current.shape)
        np.testing.assert_allclose(
            loss.numpy(), legacy_loss.numpy(), rtol=1e-6, atol=1e-6
        )
        np.testing.assert_allclose(
            gradient.numpy(), legacy_gradient.numpy(), rtol=2e-6, atol=1e-8
        )

    def test_tied_weights_preserve_loss_and_per_group_gradient_mass(self):
        # With ties the argsort permutation is not unique, so only the loss and
        # the per-group gradient mass are permutation-invariant; assert those
        # rather than an elementwise match that would depend on sort details.
        initial = np.array([5.0, 5.0, 1.0, 9.0, 5.0, 3.0], dtype=np.float32)
        groups = np.array([1, 1, 1, 4, 4, 4], dtype=np.int64)
        current = tf.Variable([2.0, 2.0, 7.0, 3.0, 3.0, 3.0], dtype=tf.float32)
        regularizer = self._regularizer(initial, groups)

        with tf.GradientTape() as tape:
            loss = regularizer(current)
        gradient = tape.gradient(loss, current).numpy()

        with tf.GradientTape() as legacy_tape:
            legacy_loss = _legacy_tf_emd(current, initial, groups, 1.75)
        legacy_gradient = tf.convert_to_tensor(
            legacy_tape.gradient(legacy_loss, current)
        ).numpy()

        expected = _reference_emd(current.numpy(), initial, groups, 1.75)
        np.testing.assert_allclose(loss.numpy(), expected, rtol=1e-6, atol=1e-6)
        for group in np.unique(groups):
            mask = groups == group
            np.testing.assert_allclose(
                gradient[mask].sum(), legacy_gradient[mask].sum(),
                rtol=2e-6, atol=1e-8,
            )

    def test_empty_connectivity_returns_zero_with_empty_dense_gradient(self):
        current = tf.Variable([], dtype=tf.float32)
        regularizer = self._regularizer([], [], strength=2.0)

        with tf.GradientTape() as tape:
            loss = regularizer(current)
        gradient = tape.gradient(loss, current)

        self.assertEqual(loss.numpy(), 0.0)
        self.assertIsInstance(gradient, tf.Tensor)
        self.assertEqual(gradient.shape, (0,))


if __name__ == "__main__":
    unittest.main()
