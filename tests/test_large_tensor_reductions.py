"""Regression tests for INT32-safe spike-sequence reductions."""

import unittest

import numpy as np
import tensorflow as tf

from v1_model_utils import loss_functions


class TemporalReductionTest(unittest.TestCase):
    def test_weighted_branch_rates_match_combined_metric(self):
        evoked = tf.random.stateless_uniform((3, 7, 5), seed=(11, 12))
        spontaneous = tf.random.stateless_uniform((5, 7, 5), seed=(13, 14))
        expected = tf.reduce_mean(tf.concat([evoked, spontaneous], axis=0))
        actual = (
            tf.reduce_mean(evoked) * tf.cast(tf.shape(evoked)[0], tf.float32)
            + tf.reduce_mean(spontaneous)
            * tf.cast(tf.shape(spontaneous)[0], tf.float32)
        ) / tf.cast(tf.shape(evoked)[0] + tf.shape(spontaneous)[0], tf.float32)

        np.testing.assert_allclose(actual, expected, rtol=1e-6)

    def test_float16_binary_spikes_are_summed_before_casting(self):
        spikes = tf.cast(
            tf.random.stateless_uniform((3, 53, 17), seed=(303, 404)) < 0.2,
            tf.float16,
        )
        expected_counts = tf.reduce_sum(tf.cast(spikes, tf.float32), axis=1)
        actual_counts = loss_functions.temporal_sum(
            spikes,
            dtype=tf.float32,
            chunk_size=25,
            full_tensor_element_limit=0,
        )

        np.testing.assert_array_equal(actual_counts, expected_counts)

    def test_small_tensor_uses_equivalent_full_reduction(self):
        spikes = tf.random.stateless_normal((2, 7, 5), seed=(505, 606))
        expected = tf.reduce_sum(spikes, axis=1)
        actual = loss_functions.temporal_sum(spikes)

        np.testing.assert_array_equal(actual, expected)

    def test_temporal_outputs_and_gradients_match_full_reductions(self):
        spikes = tf.random.stateless_normal((3, 11, 17), seed=(101, 202))

        with tf.GradientTape() as reference_tape:
            reference_tape.watch(spikes)
            reference_rates = tf.reduce_mean(spikes, axis=1)
            reference_mean = tf.reduce_mean(spikes)
            reference_loss = tf.reduce_sum(tf.square(reference_rates)) + reference_mean
        reference_gradient = reference_tape.gradient(reference_loss, spikes)

        for chunk_size in (1, 3, 5, 11):
            with self.subTest(chunk_size=chunk_size):
                with tf.GradientTape() as chunked_tape:
                    chunked_tape.watch(spikes)
                    counts = loss_functions.temporal_sum(
                        spikes,
                        chunk_size=chunk_size,
                        full_tensor_element_limit=0,
                    )
                    rates = counts / tf.cast(tf.shape(spikes)[1], tf.float32)
                    mean = loss_functions.temporal_mean(
                        spikes,
                        chunk_size=chunk_size,
                        full_tensor_element_limit=0,
                    )
                    loss = tf.reduce_sum(tf.square(rates)) + mean
                gradient = chunked_tape.gradient(loss, spikes)

                np.testing.assert_allclose(rates, reference_rates, rtol=1e-6)
                np.testing.assert_allclose(
                    mean, reference_mean, rtol=2e-6, atol=2e-8
                )
                np.testing.assert_allclose(
                    loss, reference_loss, rtol=2e-6, atol=2e-8
                )
                np.testing.assert_allclose(
                    gradient, reference_gradient, rtol=2e-6, atol=2e-6
                )

    def test_rejects_dynamic_time_dimension(self):
        @tf.function(input_signature=[tf.TensorSpec((None, None, 3), tf.float32)])
        def reduce_dynamic_time(values):
            return loss_functions.temporal_sum(values)

        with self.assertRaisesRegex(ValueError, "static sequence length"):
            reduce_dynamic_time.get_concrete_function()


if __name__ == "__main__":
    unittest.main()
