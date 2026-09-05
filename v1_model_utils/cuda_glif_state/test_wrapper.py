import unittest

import tensorflow as tf

from v1_model_utils import models
from v1_model_utils.cuda_glif_state import wrapper


class _Cell:
    _gauss_std = tf.constant(0.4, tf.float32)
    _dampening_factor = tf.constant(0.3, tf.float32)


def _reference_transition(voltage, refractory, history, surrogate):
    if surrogate == "gaussian":
        spikes = models.spike_gauss(voltage, _Cell._gauss_std, _Cell._dampening_factor)
    elif surrogate == "slayer":
        spikes = models.spike_slayer(voltage, _Cell._gauss_std, _Cell._dampening_factor)
    else:
        spikes = models.spike_function(voltage, _Cell._dampening_factor)
    spikes = tf.where(refractory, tf.zeros_like(spikes), spikes)
    return spikes, tf.concat([spikes, history[:, :-tf.shape(spikes)[1]]], axis=1)


class SurrogateResolutionTest(unittest.TestCase):
    def test_legacy_pseudo_gauss_alias(self):
        self.assertEqual(models.resolve_surrogate_gradient(None, True), "gaussian")

    def test_all_surrogates_are_accepted(self):
        for surrogate in models.SURROGATE_GRADIENTS:
            self.assertEqual(models.resolve_surrogate_gradient(surrogate), surrogate)

    def test_conflicting_legacy_flag_is_rejected(self):
        with self.assertRaises(ValueError):
            models.resolve_surrogate_gradient("slayer", True)


@unittest.skipUnless(tf.config.list_physical_devices("GPU"), "CUDA GPU required")
class CudaSurrogateParityTest(tf.test.TestCase):
    def test_forward_and_backward_match_tensorflow(self):
        voltage_values = [[-1.2, -0.4, 0.0], [0.2, 0.8, 1.4]]
        refractory = tf.constant(
            [[False, True, False], [False, False, True]], tf.bool
        )
        history_values = [
            [0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
            [1.0, 0.0, 1.0, 0.0, 1.0, 0.0],
        ]
        spike_weights = tf.constant([[0.7, -0.2, 0.4], [0.1, 0.8, -0.5]])
        history_weights = tf.reshape(tf.range(12, dtype=tf.float32) / 10.0, [2, 6])

        for surrogate in models.SURROGATE_GRADIENTS:
            with self.subTest(surrogate=surrogate):
                cell = _Cell()
                cell._surrogate_gradient = surrogate
                cuda_voltage = tf.Variable(voltage_values, tf.float32)
                cuda_history = tf.Variable(history_values, tf.float32)
                with tf.GradientTape() as tape:
                    cuda_spikes, cuda_new_history = wrapper._spike_and_shift(
                        cuda_voltage, refractory, cuda_history, cell=cell
                    )
                    cuda_loss = tf.reduce_sum(cuda_spikes * spike_weights)
                    cuda_loss += tf.reduce_sum(cuda_new_history * history_weights)
                cuda_gradients = tape.gradient(cuda_loss, [cuda_voltage, cuda_history])

                tf_voltage = tf.Variable(voltage_values, tf.float32)
                tf_history = tf.Variable(history_values, tf.float32)
                with tf.GradientTape() as tape:
                    tf_spikes, tf_new_history = _reference_transition(
                        tf_voltage, refractory, tf_history, surrogate
                    )
                    tf_loss = tf.reduce_sum(tf_spikes * spike_weights)
                    tf_loss += tf.reduce_sum(tf_new_history * history_weights)
                tf_gradients = tape.gradient(tf_loss, [tf_voltage, tf_history])

                self.assertAllClose(cuda_spikes, tf_spikes)
                self.assertAllClose(cuda_new_history, tf_new_history)
                self.assertAllClose(cuda_gradients[0], tf_gradients[0], atol=1e-6)
                self.assertAllClose(cuda_gradients[1], tf_gradients[1])


if __name__ == "__main__":
    unittest.main()
