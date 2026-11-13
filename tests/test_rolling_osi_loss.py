import unittest

import numpy as np
import tensorflow as tf

from v1_model_utils import spatial_layout
from v1_model_utils.loss_functions import ROLLING_STATE_FORMAT, OrientationSelectivityLoss


def build_synthetic_rolling_loss(decay=0.5, max_gradient_scale=20.0):
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
    loss._rolling_max_gradient_scale = tf.constant(max_gradient_scale, dtype=tf.float32)
    loss._rolling_warmup = False
    loss._rolling_target_sample_ess = tf.constant(80.0, dtype=tf.float32)
    loss._batch_emd_alignment_weight = 1.0
    loss._initialize_rolling_state_variables(4)

    loss._emd_group_indices = tf.ragged.constant([[0, 1, 2, 3]], dtype=tf.int32)
    loss._osi_empirical_distributions = tf.ragged.constant(
        [[0.05, 0.20, 0.45, 0.70]], dtype=tf.float32
    )
    loss._dsi_empirical_distributions = tf.ragged.constant(
        [[0.02, 0.10, 0.30, 0.55]], dtype=tf.float32
    )
    loss.cell_type_count = tf.constant([4.0], dtype=tf.float32)
    loss._n_node_types = tf.constant(1, dtype=tf.int32)
    return loss


def tuned_rates(angles, amplitudes, mean_rate_hz=1.0):
    """Rates [angles, neurons] (spikes/ms) tuned to 0 degrees, one amplitude each."""
    theta = np.deg2rad(angles)[:, None]
    amplitudes = np.asarray(amplitudes, dtype=np.float32)[None, :]
    curve = 1.0 + amplitudes * (np.cos(2.0 * theta) + 0.5 * np.cos(theta))
    return (mean_rate_hz / 1000.0 * curve).astype(np.float32)


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

    def test_rolling_osi_emd_paths_match_without_state_update(self):
        self.assert_paths_match(
            lambda: build_synthetic_rolling_loss(decay=0.9277),
            synthetic_spikes_variable,
            tf.constant([0.8, 0.9, 1.0, 1.1], dtype=tf.float32),
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
        # One row per replica copy (here one).
        state = {
            "ori_real": np.array([[1.0, 2.0, 3.0, 4.0]], dtype=np.float32),
            "ori_imag": np.array([[0.5, 0.4, 0.3, 0.2]], dtype=np.float32),
            "dir_real": np.array([[0.1, 0.2, 0.3, 0.4]], dtype=np.float32),
            "dir_imag": np.array([[0.9, 0.8, 0.7, 0.6]], dtype=np.float32),
            "denominator": np.array([[2.0, 2.1, 2.2, 2.3]], dtype=np.float32),
            "weight_sum": np.array([0.75], dtype=np.float32),
            "weight_sq_sum": np.array([0.1], dtype=np.float32),
            "format": ROLLING_STATE_FORMAT,
        }

        self.assertTrue(loss.set_rolling_state(state))
        restored = loss.get_rolling_state()
        self.assertEqual(restored.keys(), state.keys())
        for key, value in state.items():
            np.testing.assert_allclose(restored[key], value)

        for key, value in (("ori_real", np.array([[1.0, 2.0]])), ("ori_real", state["ori_real"][0])):
            with self.assertRaises(ValueError):
                loss.set_rolling_state({**state, key: value})
        without_weights = {k: v for k, v in state.items() if k != "weight_sum"}
        with self.assertRaisesRegex(ValueError, "weight_sum"):
            loss.set_rolling_state(without_weights)

    def test_core_rolling_state_follows_its_neurons_across_layouts(self):
        """A core-only state saved in one neuron layout restores in another.

        The loss covers the core (4 of 6 neurons, runtime order); the saved
        rows are network-sized, so train_end_data's layout translation moves
        every value with its neuron.
        """
        canonical_core = np.array([True, False, True, True, False, True])
        new_to_old = np.array([3, 0, 5, 1, 4, 2], dtype=np.uint32)
        morton = spatial_layout.NeuronLayout(
            mode=spatial_layout.MORTON, new_to_old=new_to_old,
            old_to_new=np.argsort(new_to_old).astype(np.uint32),
        )

        def core_loss(core_mask):
            loss = build_synthetic_rolling_loss()
            loss._core_mask = tf.constant(core_mask)
            loss._initialize_rolling_state_variables(int(core_mask.sum()))
            return loss

        runtime_core = canonical_core[new_to_old]
        saving = core_loss(runtime_core)
        # Each core neuron's value is ten times its canonical id.
        saving._rolling_ori_real.assign(10.0 * new_to_old[runtime_core].astype(np.float32))
        saved = spatial_layout.translate_neuron_state(
            morton, {"rolling_osi_emd_state": {"core": saving.get_rolling_state()}},
            to_runtime=False,
        )["rolling_osi_emd_state"]["core"]
        np.testing.assert_array_equal(
            saved["ori_real"][0][canonical_core], 10.0 * np.flatnonzero(canonical_core)
        )

        restoring = core_loss(canonical_core)  # canonical layout: identity
        self.assertTrue(restoring.set_rolling_state(saved))
        np.testing.assert_array_equal(
            restoring._rolling_ori_real.numpy(), 10.0 * np.flatnonzero(canonical_core)
        )

    def test_rolling_state_without_format_tag_is_rejected_untouched(self):
        # The earlier rolling_osi_emd saved an EMA of v1_ema-normalized rates.
        loss = build_synthetic_rolling_loss(decay=0.5)
        rates = tf.constant(tuned_rates([0.0, 90.0, 180.0], [0.2, 0.4, 0.6, 0.8]))
        loss.loss_from_rates(rates, tf.constant([0.0, 90.0, 180.0]))
        before = loss.get_rolling_state()
        legacy = {key: value[0] * 300.0 for key, value in before.items() if key != "format"}

        with self.assertRaisesRegex(ValueError, "format"):
            loss.set_rolling_state(legacy)
        after = loss.get_rolling_state()
        for key in before:
            np.testing.assert_array_equal(after[key], before[key])

    def test_restored_rolling_state_continues_bit_exactly(self):
        angles = np.array([0.0, 90.0, 180.0, 270.0], dtype=np.float32)
        batches = [
            tf.constant(tuned_rates(angles, amplitudes, mean_rate_hz=rate))
            for amplitudes, rate in (
                ([0.1, 0.5, 0.3, 0.9], 1.0),
                ([0.4, 0.2, 0.6, 0.1], 2.0),
                ([0.7, 0.3, 0.2, 0.5], 1.5),
            )
        ]
        angle = tf.constant(angles)
        original = build_synthetic_rolling_loss(decay=0.8)
        original._rolling_warmup = True
        for rates in batches[:2]:
            original.loss_from_rates(rates, angle)
        saved = original.get_rolling_state()

        resumed = build_synthetic_rolling_loss(decay=0.8)
        resumed._rolling_warmup = True
        self.assertTrue(resumed.set_rolling_state(saved))
        for key, value in resumed.get_rolling_state().items():
            np.testing.assert_array_equal(value, saved[key])
        self.assertEqual(
            float(original.loss_from_rates(batches[2], angle)),
            float(resumed.loss_from_rates(batches[2], angle)),
        )

    def test_gradient_correction_preserves_forward_value_and_scales_gradient(self):
        # A cap of 1 leaves the raw EMA gradient, whose current-batch share is (1-d).
        angle = tf.constant([0.0, 45.0, 90.0], dtype=tf.float32)
        plain_loss = build_synthetic_rolling_loss(decay=0.9277, max_gradient_scale=1.0)
        corrected_loss = build_synthetic_rolling_loss(decay=0.9277)

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
        np.testing.assert_allclose(
            corrected_grad.numpy(), plain_grad.numpy() / (1.0 - 0.9277), rtol=1e-4, atol=1e-9
        )

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
        loss = build_synthetic_rolling_loss(decay=0.0)
        rates = tf.ones((1, 4), dtype=tf.float32)
        radians_delta_angle = tf.fill((1, 4), np.pi / 4.0)

        (
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
        loss = build_synthetic_rolling_loss(decay=0.0)
        loss._tuning_angles = tf.zeros((4,), dtype=tf.float32)
        loss._osi_empirical_distributions = tf.ragged.constant(
            [[1.0]], dtype=tf.float32
        )
        loss._dsi_empirical_distributions = tf.ragged.constant(
            [[np.sqrt(0.5)]], dtype=tf.float32
        )
        spikes = tf.ones((1, 2, 4), dtype=tf.float32)
        angle = tf.constant([45.0], dtype=tf.float32)

        value = loss(spikes, angle, trim=False, update_state=False)

        self.assertGreater(float(value.numpy()), 2.0)

    def test_rolling_loss_is_finite_for_zero_rates(self):
        loss = build_synthetic_rolling_loss(decay=0.5)
        spikes = tf.Variable(tf.zeros((3, 2, 4), dtype=tf.float32))
        angle = tf.constant([0.0, 45.0, 90.0], dtype=tf.float32)

        value = loss(spikes, angle, trim=False, update_state=True)

        self.assertTrue(np.isfinite(float(value.numpy())))

    def test_without_memory_matches_batch_osi_emd(self):
        """With decay 0 the rolling loss is exactly batch_osi_emd."""
        angles = np.arange(0, 360, 15, dtype=np.float32)
        rates = tuned_rates(angles, [0.0, 0.3, 0.6, 0.9])
        rates[:, 0] = 0.0002  # below the 0.5 Hz preferred-rate cutoff
        loss = build_synthetic_rolling_loss(decay=0.0)
        loss._tuning_angles = tf.constant([0.0, 0.0, 0.0, 30.0], dtype=tf.float32)

        rolling = loss.rolling_osi_emd_loss_from_rates(
            tf.constant(rates), tf.constant(angles), update_state=False
        )
        batch = loss.batch_osi_emd_loss_from_rates(
            tf.constant(rates), tf.constant(angles)
        )

        self.assertGreater(float(batch.numpy()), 0.0)
        np.testing.assert_allclose(rolling.numpy(), batch.numpy(), rtol=1e-5)

    def test_mask_uses_current_batch_preferred_rate(self):
        """A neuron silent in this batch leaves the EMD even if its EMA is active."""
        angles = np.arange(0, 360, 15, dtype=np.float32)
        loss = build_synthetic_rolling_loss(decay=0.5)
        loss._tuning_angles = tf.zeros((4,), dtype=tf.float32)
        loss._emd_group_indices = tf.ragged.constant([[0, 1]], dtype=tf.int32)
        loss._osi_empirical_distributions = tf.ragged.constant([[0.0]], dtype=tf.float32)
        loss._dsi_empirical_distributions = tf.ragged.constant([[0.0]], dtype=tf.float32)
        loss._batch_emd_alignment_weight = 0.0
        rates = tuned_rates(angles, [0.0, 0.0, 0.0, 0.0])
        rates[:, 1] = tuned_rates(angles, [0.8])[:, 0]
        loss.rolling_osi_emd_loss_from_rates(tf.constant(rates), tf.constant(angles))

        rates[:, 1] = 0.0002
        value = loss.rolling_osi_emd_loss_from_rates(
            tf.constant(rates), tf.constant(angles), update_state=False
        )

        # Only the untuned neuron 0 remains; it matches the zero target.
        self.assertLess(float(value.numpy()), 1e-5)

    def test_emd_term_alone_drives_rolling_selectivity_toward_target(self):
        """Without the |Im| term the EMD must still carry a gradient.

        A gradient-free tf.random.shuffle of the values before the sort would
        return no gradient here.
        """
        angles = np.arange(0, 360, 15, dtype=np.float32)
        loss = build_synthetic_rolling_loss(decay=0.5)
        loss._tuning_angles = tf.zeros((4,), dtype=tf.float32)
        loss._batch_emd_alignment_weight = 0.0
        loss._osi_empirical_distributions = tf.ragged.constant(
            [[0.5, 0.6, 0.7, 0.8]], dtype=tf.float32
        )
        loss._dsi_empirical_distributions = tf.ragged.constant(
            [[0.4, 0.5, 0.6, 0.7]], dtype=tf.float32
        )
        rates = tf.Variable(tuned_rates(angles, [0.05, 0.1, 0.15, 0.2]))
        loss.rolling_osi_emd_loss_from_rates(rates, tf.constant(angles))

        with tf.GradientTape() as tape:
            value = loss.rolling_osi_emd_loss_from_rates(
                rates, tf.constant(angles), update_state=False
            )
        gradient = tape.gradient(value, rates)
        self.assertIsNotNone(gradient)
        self.assertGreater(float(tf.reduce_max(tf.abs(gradient))), 0.0)

        def estimates(sample_rates):
            return loss._update_rolling_selectivity_estimates(
                sample_rates,
                loss._radians_delta_angle(tf.constant(angles)),
                update_state=False,
            )

        osi_before, _, dsi_before, _, _ = estimates(rates)
        stepped = rates - 1e-5 * gradient / tf.reduce_max(tf.abs(gradient))
        osi_after, _, dsi_after, _, _ = estimates(stepped)
        self.assertTrue(np.all(osi_after.numpy() > osi_before.numpy()))
        self.assertTrue(np.all(dsi_after.numpy() > dsi_before.numpy()))


class CrowdOsiLossTests(unittest.TestCase):
    def test_crowd_moment_core_exposes_quadrature_component(self):
        loss = build_synthetic_crowd_loss()
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


if __name__ == "__main__":
    unittest.main()
