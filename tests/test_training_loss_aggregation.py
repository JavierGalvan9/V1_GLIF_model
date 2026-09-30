import unittest

import numpy as np
import tensorflow as tf

from multi_training import (
    sample_weighted_mean,
    validation_rate_loss_from_rates,
)


class _SquaredMeanRateTarget:
    def loss_from_rates(self, rates):
        return tf.square(tf.reduce_mean(rates))


class ValidationLossAggregationTests(unittest.TestCase):
    def test_batch_summaries_are_weighted_by_sample_count(self):
        self.assertAlmostEqual(
            sample_weighted_mean([2.0, 10.0], [8, 2]),
            3.6,
        )

    def test_protocol_rate_loss_uses_population_mean_rates(self):
        mean_rates_hz = np.array([1000.0, 3000.0])
        loss = validation_rate_loss_from_rates(
            mean_rates_hz, _SquaredMeanRateTarget()
        )
        self.assertAlmostEqual(float(loss.numpy()), 4.0)

    def test_protocol_rate_loss_adds_annulus_target_on_same_mean_rates(self):
        rates_hz = np.array([1000.0, 3000.0])

        loss = validation_rate_loss_from_rates(
            rates_hz,
            _SquaredMeanRateTarget(),
            annulus_regularizer=_SquaredMeanRateTarget(),
        )

        self.assertAlmostEqual(float(loss.numpy()), 8.0)

if __name__ == "__main__":
    unittest.main()
