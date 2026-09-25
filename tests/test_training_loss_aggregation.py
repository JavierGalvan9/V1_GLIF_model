import unittest

import numpy as np
import tensorflow as tf

from multi_training import (
    paired_stimulus_loss,
    paired_stimulus_loss_contribution,
    sample_weighted_mean,
    validation_rate_loss_from_rates,
)


class _SquaredMeanRateTarget:
    def loss_from_rates(self, rates):
        return tf.square(tf.reduce_mean(rates))


class TrainingLossAggregationTests(unittest.TestCase):
    def test_partial_validation_batches_are_weighted_by_retained_trials(self):
        self.assertAlmostEqual(
            sample_weighted_mean([2.0, 10.0], [8, 2]),
            3.6,
        )

    def test_protocol_rate_loss_uses_only_retained_trials_as_one_population(self):
        # The last two rows model padding from a full validation batch.  They
        # must not influence the requested three-trial protocol objective.
        retained_rates_hz = np.array([[1000.0], [2000.0], [3000.0]])
        padded_rates_hz = np.array([[1000.0], [2000.0], [3000.0], [9000.0], [9000.0]])

        retained_loss = validation_rate_loss_from_rates(
            retained_rates_hz, _SquaredMeanRateTarget()
        )
        padded_loss = validation_rate_loss_from_rates(
            padded_rates_hz, _SquaredMeanRateTarget()
        )

        self.assertAlmostEqual(float(retained_loss.numpy()), 4.0)
        self.assertNotAlmostEqual(
            float(retained_loss.numpy()), float(padded_loss.numpy())
        )

    def test_protocol_rate_loss_adds_annulus_target_on_same_retained_rates(self):
        rates_hz = np.array([[1000.0], [3000.0]])

        loss = validation_rate_loss_from_rates(
            rates_hz,
            _SquaredMeanRateTarget(),
            annulus_regularizer=_SquaredMeanRateTarget(),
        )

        self.assertAlmostEqual(float(loss.numpy()), 8.0)

    def test_paired_objective_matches_validation_component_weighting(self):
        # Common evoked/spontaneous terms are averaged, while OSI/DSI occurs
        # only for evoked activity and keeps its configured full weight.
        evoked_loss = tf.constant(13.0)  # common=3, OSI/DSI=10
        spontaneous_loss = tf.constant(5.0)
        osi_dsi_loss = tf.constant(10.0)

        objective = paired_stimulus_loss(
            evoked_loss, spontaneous_loss, osi_dsi_loss
        )

        self.assertAlmostEqual(float(objective.numpy()), 14.0)

    def test_paired_objective_gradient_keeps_full_osi_weight(self):
        parameter = tf.Variable(2.0)
        with tf.GradientTape() as tape:
            evoked_common = parameter
            spontaneous_common = 3.0 * parameter
            osi_dsi = 5.0 * parameter
            objective = paired_stimulus_loss(
                evoked_common + osi_dsi,
                spontaneous_common,
                osi_dsi,
            )

        gradient = tape.gradient(objective, parameter)

        # 0.5 * (1 + 3) + 5 = 7
        self.assertAlmostEqual(float(gradient.numpy()), 7.0)

    def test_sequential_contributions_equal_combined_objective(self):
        evoked_loss = tf.constant(13.0)
        spontaneous_loss = tf.constant(5.0)
        osi_dsi_loss = tf.constant(10.0)

        sequential = paired_stimulus_loss_contribution(
            evoked_loss, osi_dsi_loss, spontaneous=False
        ) + paired_stimulus_loss_contribution(
            spontaneous_loss, tf.constant(0.0), spontaneous=True
        )
        combined = paired_stimulus_loss(
            evoked_loss, spontaneous_loss, osi_dsi_loss
        )

        self.assertAlmostEqual(float(sequential.numpy()), float(combined.numpy()))


if __name__ == "__main__":
    unittest.main()
