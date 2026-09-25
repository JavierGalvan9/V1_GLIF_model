import unittest

import numpy as np
import tensorflow as tf

from multi_training import (
    annulus_mask_from_core,
    run_gray_state_rollout,
    update_rate_emas,
)
from v1_model_utils.loss_functions import (
    SpikeRateDistributionTarget,
    compute_spike_rate_target_loss,
    interpolate_empirical_quantile_midpoints,
)


def build_synthetic_rate_target():
    target = SpikeRateDistributionTarget.__new__(SpikeRateDistributionTarget)
    target._pre_delay = 1
    target._post_delay = 1
    target._dtype = tf.float32
    target._rate_cost = 2.5
    target._target_rates = {
        "exc": {
            "neuron_ids": tf.constant([0, 2], dtype=tf.int32),
            "n_model_neurons": 2,
            "sorted_target_rates": tf.constant([0.15, 0.55], dtype=tf.float32),
        },
        "inh": {
            "neuron_ids": tf.constant([1, 3], dtype=tf.int32),
            "n_model_neurons": 2,
            "sorted_target_rates": tf.constant([0.25, 0.45], dtype=tf.float32),
        },
    }
    return target


def synthetic_spikes(dtype=tf.float16):
    values = np.array(
        [
            [
                [0.0, 0.1, 0.2, 0.3],
                [0.2, 0.3, 0.4, 0.5],
                [0.4, 0.5, 0.6, 0.7],
                [0.6, 0.7, 0.8, 0.9],
            ],
            [
                [0.9, 0.8, 0.7, 0.6],
                [0.7, 0.6, 0.5, 0.4],
                [0.5, 0.4, 0.3, 0.2],
                [0.3, 0.2, 0.1, 0.0],
            ],
        ],
        dtype=np.float32,
    )
    return tf.Variable(tf.cast(values, dtype))


class SharedRateTargetTests(unittest.TestCase):
    def test_multi_training_helpers_preserve_gray_state_and_ema_contracts(self):
        self.assertIsNone(annulus_mask_from_core(None))
        self.assertIsNone(
            annulus_mask_from_core(tf.constant([True, True], dtype=tf.bool))
        )
        np.testing.assert_array_equal(
            annulus_mask_from_core(
                tf.constant([True, False, True], dtype=tf.bool)
            ).numpy(),
            [False, True, False],
        )

        ema = tf.Variable([0.1, 0.2], dtype=tf.float32)
        rates = tf.constant([0.5, 0.6], dtype=tf.float32)
        update_rate_emas(ema, rates, decay=0.8, update_state=False)
        np.testing.assert_allclose(ema.numpy(), [0.1, 0.2])
        update_rate_emas(ema, rates, decay=0.8)
        np.testing.assert_allclose(ema.numpy(), [0.18, 0.28], rtol=1e-6)

        inputs = (tf.constant([1.0]),)
        state_model = lambda _: (tf.constant([2.0]), tf.constant([3.0]))
        fallback_model = lambda _: (
            (tf.constant([10.0]), tf.constant([11.0])),
            tf.constant([4.0]),
            tf.constant([5.0]),
        )
        state = run_gray_state_rollout(state_model, fallback_model, inputs)
        fallback_state = run_gray_state_rollout(None, fallback_model, inputs)
        np.testing.assert_array_equal(state[0].numpy(), [2.0])
        np.testing.assert_array_equal(fallback_state[0].numpy(), [4.0])
        np.testing.assert_array_equal(fallback_state[1].numpy(), [5.0])
        with self.assertRaisesRegex(ValueError, "final-state tensor"):
            run_gray_state_rollout(None, lambda _: (tf.constant([1.0]),), ())

    def test_quantile_midpoints_preserve_equal_sized_distribution(self):
        firing_rates = np.array([0.1, 0.2, 0.6, 0.9], dtype=np.float32)

        target = interpolate_empirical_quantile_midpoints(
            firing_rates, len(firing_rates)
        )

        np.testing.assert_allclose(target, firing_rates)

    def test_rate_target_requires_modeled_neurons(self):
        target_rates = {
            "empty": {
                "neuron_ids": tf.constant([], dtype=tf.int32),
                "n_model_neurons": 0,
                "sorted_target_rates": tf.constant([], dtype=tf.float32),
            }
        }

        with self.assertRaisesRegex(ValueError, "No modeled neurons"):
            compute_spike_rate_target_loss(
                tf.constant([0.1], dtype=tf.float32), target_rates
            )

    def test_rate_summaries_trim_cast_and_preserve_full_population(self):
        target = build_synthetic_rate_target()
        spikes = synthetic_spikes()

        rates_per_sample = target.rates_per_sample_from_spikes(
            spikes, trim=True
        )
        rates = target.rates_from_spikes(spikes, trim=True)
        expected_per_sample = tf.reduce_mean(
            tf.cast(spikes[:, 1:-1, :], tf.float32), axis=1
        )

        self.assertEqual(rates_per_sample.dtype, tf.float32)
        self.assertEqual(rates_per_sample.shape, (2, 4))
        np.testing.assert_allclose(
            rates_per_sample.numpy(), expected_per_sample.numpy(), rtol=5e-4, atol=2e-4
        )
        np.testing.assert_allclose(
            rates.numpy(),
            tf.reduce_mean(expected_per_sample, axis=0).numpy(),
            rtol=5e-4,
            atol=2e-4,
        )

    def test_spike_and_precomputed_rate_paths_match_values_and_gradients(self):
        target = build_synthetic_rate_target()

        spike_input = synthetic_spikes(dtype=tf.float32)
        with tf.GradientTape() as tape:
            spike_loss = target(spike_input, trim=True)
        spike_gradient = tape.gradient(spike_loss, spike_input)

        rate_input = synthetic_spikes(dtype=tf.float32)
        with tf.GradientTape() as tape:
            rates = target.rates_from_spikes(rate_input, trim=True)
            rate_loss = target.loss_from_rates(rates)
        rate_gradient = tape.gradient(rate_loss, rate_input)

        np.testing.assert_allclose(
            rate_loss.numpy(), spike_loss.numpy(), rtol=1e-6, atol=1e-6
        )
        np.testing.assert_allclose(
            rate_gradient.numpy(),
            spike_gradient.numpy(),
            rtol=1e-6,
            atol=1e-6,
        )

    def test_loss_from_rates_accepts_mixed_input_dtype(self):
        target = build_synthetic_rate_target()
        rates = tf.constant([0.2, 0.3, 0.4, 0.5], dtype=tf.float16)

        value = target.loss_from_rates(rates)

        self.assertEqual(value.dtype, tf.float32)
        self.assertTrue(np.isfinite(float(value.numpy())))


if __name__ == "__main__":
    unittest.main()
