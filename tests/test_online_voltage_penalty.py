import unittest

import numpy as np
import tensorflow as tf

from v1_model_utils.models import V1Column


def _legacy_voltage_penalty(voltages, penalty_mode):
    """Match VoltageRegularization before its scalar cost is applied."""
    if penalty_mode == "range":
        penalty = tf.square(tf.nn.relu(tf.abs(voltages - 0.5) - 0.5))
    else:
        penalty = tf.square(voltages - 1.0)
    # Preserve the legacy reduction order, including its fp32 neuron reduction.
    penalty = tf.reduce_mean(penalty, axis=0)
    penalty = tf.reduce_mean(penalty, axis=0)
    return tf.reduce_mean(tf.cast(penalty, tf.float32), axis=0)


def _online_voltage_penalty(voltages, penalty_mode):
    """Apply V1Column's per-step accumulator and normalize over time/batch."""
    cell = type("VoltagePenaltyCell", (), {
        "_n_neurons": int(voltages.shape[-1]),
        "_voltage_penalty_mode": penalty_mode,
    })()
    # The accumulator is fp32 regardless of the voltage compute dtype, matching
    # V1Column.zero_state.
    accumulator = tf.zeros(tf.shape(voltages)[0], dtype=tf.float32)
    for timestep in tf.unstack(voltages, axis=1):
        accumulator += V1Column._voltage_penalty_mean_step(cell, timestep)
    return tf.reduce_mean(accumulator) / tf.cast(
        tf.shape(voltages)[1], tf.float32
    )


class OnlineVoltagePenaltyTest(unittest.TestCase):
    def _assert_matches_legacy(self, dtype, penalty_mode):
        # Values deliberately include both sides of the valid [0, 1] range.
        values = tf.constant(
            [
                [[-0.4, 0.0, 0.2, 1.0, 1.6], [0.1, 0.4, 0.8, 1.2, 2.1],
                 [-1.0, 0.5, 0.7, 1.0, 1.4]],
                [[-0.2, 0.3, 0.6, 0.9, 1.3], [0.0, 0.5, 1.0, 1.5, 2.0],
                 [-0.7, 0.2, 0.9, 1.1, 1.8]],
            ],
            dtype=dtype,
        )
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(values)
            legacy = _legacy_voltage_penalty(values, penalty_mode)
            online = _online_voltage_penalty(values, penalty_mode)
        legacy_gradient = tape.gradient(legacy, values)
        online_gradient = tape.gradient(online, values)
        del tape

        rtol, atol = (3e-3, 3e-4) if dtype == tf.float16 else (2e-5, 2e-6)
        np.testing.assert_allclose(
            online.numpy(), legacy.numpy(), rtol=rtol, atol=atol
        )
        np.testing.assert_allclose(
            online_gradient.numpy(), legacy_gradient.numpy(), rtol=rtol, atol=atol
        )

    def test_online_range_penalty_matches_legacy_in_float32_and_float16(self):
        for dtype in (tf.float32, tf.float16):
            with self.subTest(dtype=dtype.name):
                self._assert_matches_legacy(dtype, "range")

    def test_large_fp16_voltage_excursion_stays_finite(self):
        """A single runaway neuron must not poison the batch mean.

        Nothing in the LIF update bounds the membrane voltage of a neuron that
        never spikes, so |v| can reach several hundred. Squaring that in fp16
        overflows to inf, which is what turned the voltage loss NaN while every
        other loss kept evolving.
        """
        for penalty_mode in ("range", "threshold"):
            with self.subTest(penalty_mode=penalty_mode):
                values = np.full((2, 3, 5), 0.5, dtype=np.float16)
                values[0, 1, 2] = -900.0  # well past the fp16 square limit
                voltages = tf.constant(values)
                with tf.GradientTape() as tape:
                    tape.watch(voltages)
                    online = _online_voltage_penalty(voltages, penalty_mode)
                gradient = tape.gradient(online, voltages)
                self.assertTrue(np.isfinite(online.numpy()))
                # The gradient is returned in the fp16 voltage dtype, so the
                # runaway entry itself saturates; every other entry must stay
                # finite rather than inherit a NaN.
                gradient = gradient.numpy()
                gradient[0, 1, 2] = 0.0
                self.assertTrue(np.all(np.isfinite(gradient)))

    def test_online_threshold_penalty_matches_legacy_in_float32_and_float16(self):
        for dtype in (tf.float32, tf.float16):
            with self.subTest(dtype=dtype.name):
                self._assert_matches_legacy(dtype, "threshold")


if __name__ == "__main__":
    unittest.main()
