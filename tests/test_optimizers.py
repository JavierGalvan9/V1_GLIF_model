import unittest

import tensorflow as tf

from v1_model_utils.optimizers import ExponentiatedAdam


class ExponentiatedAdamTest(unittest.TestCase):
    def test_large_dense_update_matches_known_two_step_result(self):
        variable = tf.Variable([1.0, -2.0, 0.5], dtype=tf.float32)
        optimizer = ExponentiatedAdam(
            learning_rate=0.01,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-7,
        )

        optimizer.apply_gradients(
            [(tf.constant([0.1, -0.2, 0.3]), variable)]
        )
        optimizer.apply_gradients(
            [(tf.constant([-0.4, 0.5, -0.6]), variable)]
        )

        self.assertAllClose(
            variable,
            [0.99560499, -1.98887562, 0.49684059],
            rtol=3e-5,
            atol=3e-7,
        )
        self.assertNotIn("dense_jit_min_elements", optimizer.get_config())

    def assertAllClose(self, actual, expected, rtol, atol):
        tf.debugging.assert_near(
            actual, tf.constant(expected, dtype=actual.dtype), rtol=rtol, atol=atol
        )


if __name__ == "__main__":
    unittest.main()
