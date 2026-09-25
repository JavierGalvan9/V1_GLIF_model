import unittest

import numpy as np
import tensorflow as tf

from v1_model_utils.loss_functions import OrientationSelectivityLoss


def build_synthetic_rolling_loss(decay=0.5, gradient_correction=True):
    loss = OrientationSelectivityLoss.__new__(OrientationSelectivityLoss)
    loss._method = "rolling_osi_emd"
    loss._osi_cost = 1.0
    loss._pre_delay = None
    loss._post_delay = None
    loss._dtype = tf.float32
    loss._core_mask = None
    loss._tf_pi = tf.constant(np.pi, dtype=tf.float32)
    loss._tuning_angles = tf.constant([0.0, 45.0, 90.0, 135.0], dtype=tf.float32)
    loss._min_rates_threshold = tf.constant(0.0005, dtype=tf.float32)
    loss._rolling_decay = tf.constant(decay, dtype=tf.float32)
    loss._rolling_one_minus_decay = tf.constant(1.0 - decay, dtype=tf.float32)
    loss._rolling_epsilon = tf.constant(1e-6, dtype=tf.float32)
    loss._rolling_gradient_correction = gradient_correction
    loss._rolling_max_gradient_scale = tf.constant(20.0, dtype=tf.float32)
    loss._rolling_warmup = False
    loss._rolling_target_sample_ess = tf.constant(80.0, dtype=tf.float32)
    loss._rolling_config_batch_size = tf.constant(3.0, dtype=tf.float32)

    zeros = tf.zeros((4,), dtype=tf.float32)
    loss._rolling_ori_real = tf.Variable(zeros, trainable=False)
    loss._rolling_ori_imag = tf.Variable(zeros, trainable=False)
    loss._rolling_dir_real = tf.Variable(zeros, trainable=False)
    loss._rolling_dir_imag = tf.Variable(zeros, trainable=False)
    loss._rolling_denominator = tf.Variable(zeros, trainable=False)
    loss._rolling_weight_sum = tf.Variable(tf.constant(0.0, dtype=tf.float32), trainable=False)
    loss._rolling_weight_sq_sum = tf.Variable(tf.constant(0.0, dtype=tf.float32), trainable=False)

    row_splits = tf.constant([0, 4], dtype=tf.int32)
    loss._emd_group_indices = tf.RaggedTensor.from_row_splits(
        tf.constant([0, 1, 2, 3], dtype=tf.int32),
        row_splits,
        validate=False,
    )
    loss._osi_target_distributions = tf.RaggedTensor.from_row_splits(
        tf.constant([0.05, 0.20, 0.45, 0.70], dtype=tf.float32),
        row_splits,
        validate=False,
    )
    loss._dsi_target_distributions = tf.RaggedTensor.from_row_splits(
        tf.constant([0.02, 0.10, 0.30, 0.55], dtype=tf.float32),
        row_splits,
        validate=False,
    )
    loss.cell_type_count = tf.constant([4.0], dtype=tf.float32)
    loss._n_node_types = tf.constant(1, dtype=tf.int32)
    return loss


def build_synthetic_adaptative_loss(experimental_counts=(1000.0, 1000.0)):
    loss = OrientationSelectivityLoss.__new__(OrientationSelectivityLoss)
    loss._method = "adaptative_crowd_osi"
    loss._osi_cost = 1.0
    loss._pre_delay = None
    loss._post_delay = None
    loss._dtype = tf.float32
    loss._core_mask = None
    loss._tf_pi = tf.constant(np.pi, dtype=tf.float32)
    loss._tuning_angles = tf.constant([0.0, 45.0, 90.0, 135.0], dtype=tf.float32)
    loss._min_rates_threshold = tf.constant(0.0005, dtype=tf.float32)
    loss._rolling_decay = tf.constant(0.5, dtype=tf.float32)
    loss._rolling_one_minus_decay = tf.constant(0.5, dtype=tf.float32)
    loss._rolling_epsilon = tf.constant(1e-6, dtype=tf.float32)
    loss._rolling_gradient_correction = False
    loss._rolling_max_gradient_scale = tf.constant(20.0, dtype=tf.float32)
    loss._rolling_warmup = False
    loss._rolling_target_sample_ess = tf.constant(80.0, dtype=tf.float32)
    loss._rolling_config_batch_size = tf.constant(3.0, dtype=tf.float32)
    loss._adaptative_scale_min = tf.constant(0.4, dtype=tf.float32)
    loss._adaptative_scale_max = tf.constant(1.0, dtype=tf.float32)
    loss._adaptative_shrink_k = tf.constant(20.0, dtype=tf.float32)

    zeros = tf.zeros((4,), dtype=tf.float32)
    loss._rolling_ori_real = tf.Variable(zeros, trainable=False)
    loss._rolling_ori_imag = tf.Variable(zeros, trainable=False)
    loss._rolling_dir_real = tf.Variable(zeros, trainable=False)
    loss._rolling_dir_imag = tf.Variable(zeros, trainable=False)
    loss._rolling_denominator = tf.Variable(zeros, trainable=False)
    loss._rolling_weight_sum = tf.Variable(tf.constant(0.0, dtype=tf.float32), trainable=False)
    loss._rolling_weight_sq_sum = tf.Variable(tf.constant(0.0, dtype=tf.float32), trainable=False)

    loss.node_type_ids = tf.constant([0, 0, 1, 1], dtype=tf.int32)
    loss._n_node_types = 2
    loss.cell_type_count = tf.constant([2.0, 2.0], dtype=tf.float32)
    loss.experimental_cell_type_count = tf.constant(
        experimental_counts, dtype=tf.float32
    )
    loss.osi_target_values = tf.constant([0.5, 0.5], dtype=tf.float32)
    loss.dsi_target_values = tf.constant([0.25, 0.25], dtype=tf.float32)
    return loss


def build_synthetic_crowd_loss():
    loss = OrientationSelectivityLoss.__new__(OrientationSelectivityLoss)
    loss._method = "crowd_osi"
    loss._osi_cost = 1.0
    loss._pre_delay = None
    loss._post_delay = None
    loss._dtype = tf.float32
    loss._core_mask = tf.constant(
        [True, False, True, True, False, True], dtype=tf.bool
    )
    loss._tf_pi = tf.constant(np.pi, dtype=tf.float32)
    loss._tuning_angles = tf.constant(
        [0.0, 45.0, 90.0, 135.0], dtype=tf.float32
    )
    loss._min_rates_threshold = tf.constant(0.0005, dtype=tf.float32)
    loss.node_type_ids = tf.constant([0, 0, 1, 1], dtype=tf.int32)
    loss._n_node_types = 2
    loss.cell_type_count = tf.constant([2.0, 2.0], dtype=tf.float32)
    loss.osi_target_values = tf.constant([0.5, 0.6], dtype=tf.float32)
    loss.dsi_target_values = tf.constant([0.25, 0.35], dtype=tf.float32)
    return loss


def synthetic_spikes_variable():
    values = np.array(
        [
            [[0.10, 0.30, 0.80, 0.20], [0.20, 0.40, 0.70, 0.30]],
            [[0.50, 0.20, 0.10, 0.90], [0.60, 0.25, 0.15, 0.80]],
            [[0.30, 0.90, 0.20, 0.40], [0.35, 0.85, 0.25, 0.45]],
        ],
        dtype=np.float32,
    )
    return tf.Variable(values)


def synthetic_masked_spikes_variable():
    values = np.array(
        [
            [
                [0.10, 0.20, 0.30, 0.80, 0.40, 0.20],
                [0.20, 0.25, 0.40, 0.70, 0.50, 0.30],
            ],
            [
                [0.50, 0.10, 0.20, 0.10, 0.30, 0.90],
                [0.60, 0.15, 0.25, 0.15, 0.35, 0.80],
            ],
            [
                [0.30, 0.60, 0.90, 0.20, 0.10, 0.40],
                [0.35, 0.55, 0.85, 0.25, 0.15, 0.45],
            ],
        ],
        dtype=np.float32,
    )
    return tf.Variable(values)


class RateBasedOsiApiTests(unittest.TestCase):
    def assert_paths_match(
        self, build_loss, build_spikes, normalizer, update_state=False
    ):
        angle = tf.constant([0.0, 45.0, 90.0], dtype=tf.float32)

        spike_loss_object = build_loss()
        spike_input = build_spikes()
        with tf.GradientTape() as tape:
            spike_value = spike_loss_object(
                spike_input,
                angle,
                trim=False,
                normalizer=normalizer,
                update_state=update_state,
            )
        spike_gradient = tape.gradient(spike_value, spike_input)

        rate_loss_object = build_loss()
        rate_input = build_spikes()
        with tf.GradientTape() as tape:
            rates = rate_loss_object.rates_per_sample_from_spikes(
                rate_input, trim=False
            )
            rate_value = rate_loss_object.loss_from_rates(
                rates,
                angle,
                normalizer=normalizer,
                update_state=update_state,
            )
        rate_gradient = tape.gradient(rate_value, rate_input)

        np.testing.assert_allclose(
            rate_value.numpy(), spike_value.numpy(), rtol=1e-6, atol=1e-6
        )
        np.testing.assert_allclose(
            rate_gradient.numpy(),
            spike_gradient.numpy(),
            rtol=1e-5,
            atol=1e-6,
        )
        if rate_loss_object._uses_rolling_state():
            spike_state = spike_loss_object.get_rolling_state()
            rate_state = rate_loss_object.get_rolling_state()
            for key in spike_state:
                np.testing.assert_allclose(
                    rate_state[key], spike_state[key], rtol=1e-6, atol=1e-6
                )

    def test_crowd_osi_paths_match_with_core_mask_and_normalizer(self):
        self.assert_paths_match(
            build_synthetic_crowd_loss,
            synthetic_masked_spikes_variable,
            tf.constant(
                [0.8, 1.1, 0.9, 1.2, 0.7, 1.0], dtype=tf.float32
            ),
        )

    def test_adaptative_crowd_osi_paths_match_without_state_update(self):
        self.assert_paths_match(
            build_synthetic_adaptative_loss,
            synthetic_spikes_variable,
            tf.constant([0.8, 0.9, 1.0, 1.1], dtype=tf.float32),
        )

    def test_rolling_osi_emd_paths_match_without_state_update(self):
        self.assert_paths_match(
            lambda: build_synthetic_rolling_loss(decay=0.9277),
            synthetic_spikes_variable,
            tf.constant([0.8, 0.9, 1.0, 1.1], dtype=tf.float32),
        )

    def test_adaptative_state_updates_match_exactly_once(self):
        self.assert_paths_match(
            build_synthetic_adaptative_loss,
            synthetic_spikes_variable,
            tf.ones((4,), dtype=tf.float32),
            update_state=True,
        )

    def test_rolling_state_updates_match_exactly_once(self):
        self.assert_paths_match(
            lambda: build_synthetic_rolling_loss(decay=0.9277),
            synthetic_spikes_variable,
            tf.ones((4,), dtype=tf.float32),
            update_state=True,
        )


class RollingOsiLossTests(unittest.TestCase):
    def test_update_state_false_does_not_mutate_rolling_variables(self):
        loss = build_synthetic_rolling_loss(decay=0.9277)
        spikes = synthetic_spikes_variable()
        angle = tf.constant([0.0, 45.0, 90.0], dtype=tf.float32)

        before = loss.get_rolling_state()
        value = loss(spikes, angle, trim=False, update_state=False)
        after = loss.get_rolling_state()

        self.assertTrue(np.isfinite(float(value.numpy())))
        for key in before:
            np.testing.assert_allclose(after[key], before[key])

    def test_update_state_true_mutates_rolling_variables(self):
        loss = build_synthetic_rolling_loss(decay=0.9277)
        spikes = synthetic_spikes_variable()
        angle = tf.constant([0.0, 45.0, 90.0], dtype=tf.float32)

        before = loss.get_rolling_state()
        value = loss(spikes, angle, trim=False, update_state=True)
        after = loss.get_rolling_state()

        self.assertTrue(np.isfinite(float(value.numpy())))
        self.assertGreater(
            sum(np.sum(np.abs(after[key] - before[key])) for key in before),
            0.0,
        )

    def test_rolling_state_round_trip_and_shape_check(self):
        loss = build_synthetic_rolling_loss(decay=0.5)
        state = {
            "ori_real": np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32),
            "ori_imag": np.array([0.5, 0.4, 0.3, 0.2], dtype=np.float32),
            "dir_real": np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32),
            "dir_imag": np.array([0.9, 0.8, 0.7, 0.6], dtype=np.float32),
            "denominator": np.array([2.0, 2.1, 2.2, 2.3], dtype=np.float32),
        }

        self.assertTrue(loss.set_rolling_state(state))
        restored = loss.get_rolling_state()
        for key, value in state.items():
            np.testing.assert_allclose(restored[key], value)

        bad_state = dict(state)
        bad_state["ori_real"] = np.array([1.0, 2.0], dtype=np.float32)
        with self.assertRaises(ValueError):
            loss.set_rolling_state(bad_state)

    def test_gradient_correction_preserves_forward_value_and_scales_gradient(self):
        angle = tf.constant([0.0, 45.0, 90.0], dtype=tf.float32)
        plain_loss = build_synthetic_rolling_loss(
            decay=0.9277, gradient_correction=False
        )
        corrected_loss = build_synthetic_rolling_loss(
            decay=0.9277, gradient_correction=True
        )

        plain_spikes = synthetic_spikes_variable()
        with tf.GradientTape() as tape:
            plain_value = plain_loss(
                plain_spikes, angle, trim=False, update_state=False
            )
        plain_grad = tape.gradient(plain_value, plain_spikes)

        corrected_spikes = synthetic_spikes_variable()
        with tf.GradientTape() as tape:
            corrected_value = corrected_loss(
                corrected_spikes, angle, trim=False, update_state=False
            )
        corrected_grad = tape.gradient(corrected_value, corrected_spikes)

        np.testing.assert_allclose(
            corrected_value.numpy(), plain_value.numpy(), rtol=1e-6, atol=1e-6
        )
        plain_norm = float(tf.reduce_sum(tf.abs(plain_grad)).numpy())
        corrected_norm = float(tf.reduce_sum(tf.abs(corrected_grad)).numpy())
        self.assertGreater(corrected_norm, plain_norm * 5.0)

    def test_rolling_loss_accepts_legacy_positional_trim_with_normalizer(self):
        loss = build_synthetic_rolling_loss(decay=0.9277)
        spikes = synthetic_spikes_variable()
        angle = tf.constant([0.0, 45.0, 90.0], dtype=tf.float32)
        normalizer = tf.ones((4,), dtype=tf.float32)

        value = loss.rolling_osi_emd_loss(
            spikes,
            angle,
            True,
            normalizer=normalizer,
            update_state=False,
        )

        self.assertTrue(np.isfinite(float(value.numpy())))

    def test_rolling_components_expose_quadrature_phase(self):
        loss = build_synthetic_rolling_loss(decay=0.0, gradient_correction=False)
        rates = tf.ones((1, 4), dtype=tf.float32)
        radians_delta_angle = tf.fill((1, 4), np.pi / 4.0)

        (
            _osi_magnitude,
            _dsi_magnitude,
            osi_real,
            osi_imag,
            _dsi_real,
            _dsi_imag,
            _warmup_scale,
        ) = loss._update_rolling_selectivity_estimates(
            rates, radians_delta_angle, update_state=False
        )

        self.assertLess(abs(float(tf.reduce_mean(osi_real).numpy())), 1e-5)
        self.assertGreater(float(tf.reduce_mean(osi_imag).numpy()), 0.99)

    def test_rolling_loss_penalizes_quadrature_when_magnitude_matches_target(self):
        loss = build_synthetic_rolling_loss(decay=0.0, gradient_correction=False)
        loss._tuning_angles = tf.zeros((4,), dtype=tf.float32)
        row_splits = tf.constant([0, 4], dtype=tf.int32)
        loss._osi_target_distributions = tf.RaggedTensor.from_row_splits(
            tf.ones((4,), dtype=tf.float32),
            row_splits,
            validate=False,
        )
        loss._dsi_target_distributions = tf.RaggedTensor.from_row_splits(
            tf.fill((4,), np.float32(np.sqrt(0.5))),
            row_splits,
            validate=False,
        )
        spikes = tf.ones((1, 2, 4), dtype=tf.float32)
        angle = tf.constant([45.0], dtype=tf.float32)

        value = loss(spikes, angle, trim=False, update_state=False)

        self.assertGreater(float(value.numpy()), 2.0)


class AdaptativeCrowdOsiLossTests(unittest.TestCase):
    def test_crowd_moment_core_exposes_quadrature_component(self):
        loss = build_synthetic_adaptative_loss()
        rates = tf.ones((1, 4), dtype=tf.float32)
        radians_delta_angle = tf.fill((1, 4), np.pi / 4.0)

        osi_real, osi_imag, _, _ = loss._compute_crowd_moment_core(
            rates,
            radians_delta_angle,
            tf.shape(rates)[0],
            loss.node_type_ids,
            loss._n_node_types,
        )

        self.assertLess(abs(float(tf.reduce_mean(osi_real).numpy())), 1e-5)
        self.assertGreater(float(tf.reduce_mean(osi_imag).numpy()), 0.99)

    def test_adaptative_scale_is_clipped_to_configured_bounds(self):
        loss = build_synthetic_adaptative_loss(
            experimental_counts=(1e9, 1e9)
        )
        real = tf.constant([1.0, 1.0, 1.0, -1.0], dtype=tf.float32)
        imag = tf.zeros((4,), dtype=tf.float32)
        denominator = tf.ones((4,), dtype=tf.float32)

        scale = loss._adaptative_scale_from_moments(real, imag, denominator)

        np.testing.assert_allclose(scale.numpy(), [1.0, 0.4], rtol=1e-5)

    def test_sparse_type_scale_moves_toward_global_scale(self):
        loss = build_synthetic_adaptative_loss(experimental_counts=(1000.0, 1.0))
        real = tf.constant([1.0, 1.0, 1.0, -1.0], dtype=tf.float32)
        imag = tf.zeros((4,), dtype=tf.float32)
        denominator = tf.ones((4,), dtype=tf.float32)

        scale = loss._adaptative_scale_from_moments(real, imag, denominator).numpy()

        self.assertGreater(scale[1], 0.4)
        self.assertLess(scale[1], 0.7)
        np.testing.assert_allclose(scale[1], 0.6857143, rtol=1e-5)

    def test_update_state_false_does_not_mutate_adaptative_state(self):
        loss = build_synthetic_adaptative_loss()
        spikes = synthetic_spikes_variable()
        angle = tf.constant([0.0, 45.0, 90.0], dtype=tf.float32)

        before = loss.get_rolling_state()
        value = loss(spikes, angle, trim=False, update_state=False)
        after = loss.get_rolling_state()

        self.assertTrue(np.isfinite(float(value.numpy())))
        for key in before:
            np.testing.assert_allclose(after[key], before[key])

    def test_update_state_true_mutates_adaptative_state(self):
        loss = build_synthetic_adaptative_loss()
        spikes = synthetic_spikes_variable()
        angle = tf.constant([0.0, 45.0, 90.0], dtype=tf.float32)

        before = loss.get_rolling_state()
        value = loss(spikes, angle, trim=False, update_state=True)
        after = loss.get_rolling_state()

        self.assertTrue(np.isfinite(float(value.numpy())))
        self.assertGreater(
            sum(np.sum(np.abs(after[key] - before[key])) for key in before),
            0.0,
        )

    def test_adaptative_state_round_trip_and_shape_check(self):
        loss = build_synthetic_adaptative_loss()
        state = {
            "ori_real": np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32),
            "ori_imag": np.array([0.5, 0.4, 0.3, 0.2], dtype=np.float32),
            "dir_real": np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32),
            "dir_imag": np.array([0.9, 0.8, 0.7, 0.6], dtype=np.float32),
            "denominator": np.array([2.0, 2.1, 2.2, 2.3], dtype=np.float32),
        }

        self.assertTrue(loss.set_rolling_state(state))
        restored = loss.get_rolling_state()
        for key, value in state.items():
            np.testing.assert_allclose(restored[key], value)

        bad_state = dict(state)
        bad_state["denominator"] = np.array([1.0, 2.0], dtype=np.float32)
        with self.assertRaises(ValueError):
            loss.set_rolling_state(bad_state)

    def test_adaptative_loss_is_finite_for_zero_rates(self):
        loss = build_synthetic_adaptative_loss()
        spikes = tf.Variable(tf.zeros((3, 2, 4), dtype=tf.float32))
        angle = tf.constant([0.0, 45.0, 90.0], dtype=tf.float32)

        value = loss(spikes, angle, trim=False, update_state=True)

        self.assertTrue(np.isfinite(float(value.numpy())))


if __name__ == "__main__":
    unittest.main()
