import os
import pickle
import unittest

import numpy as np
import tensorflow as tf

from v1_model_utils import models


NETWORK = os.path.join("GLIF_network_nll_full", "tf_data", "V1_network_v1_1000.pkl")


def _cached_triple(path):
    with open(path, "rb") as handle:
        cached = pickle.load(handle)
    return cached["payload"] if isinstance(cached, dict) else cached


def _loop_condition_ops(layer, inputs):
    """Op types in the condition of every while loop the layer traces."""
    graph = tf.function(layer).get_concrete_function(
        tf.TensorSpec(inputs.shape, inputs.dtype)
    ).graph
    library = {fn.signature.name: fn for fn in graph.as_graph_def().library.function}
    names = [op.get_attr("cond").name for op in graph.get_operations()
             if op.type in ("While", "StatelessWhile")]
    return {node.op for name in names for node in library[name].node_def}


class RnnLoopConditionTest(unittest.TestCase):
    """The RNN loop must not synchronise the host on every timestep."""

    def test_condition_has_no_iteration_cap(self):
        inputs = tf.zeros((2, 5, 3))
        keras_ops = _loop_condition_ops(
            tf.keras.layers.RNN(tf.keras.layers.SimpleRNNCell(4)), inputs
        )
        # The control: stock Keras ANDs its iteration cap into the condition.
        self.assertIn("LogicalAnd", keras_ops)
        ops = _loop_condition_ops(
            models.HeterogeneousStateRNN(tf.keras.layers.SimpleRNNCell(4)), inputs
        )
        self.assertNotIn("LogicalAnd", ops)
        self.assertIn("Less", ops)


@unittest.skipUnless(os.path.exists(NETWORK), f"needs the cached network {NETWORK}")
class RnnLoopEquivalenceTest(unittest.TestCase):
    """The repository loop reproduces Keras' loop bit for bit on the CPU."""

    @classmethod
    def setUpClass(cls):
        cls.network, cls.lgn_input, cls.bkg_input = _cached_triple(NETWORK)

    def _rnn(self, return_sequences):
        cell = models.V1Column(
            self.network, self.lgn_input, self.bkg_input,
            batch_size=2, max_delay=1, train_recurrent=True,
            train_recurrent_per_type=False, train_input=True,
            train_noise=True, noise_seed=17, acceleration="tensorflow",
        )
        return models.HeterogeneousStateRNN(
            cell, return_sequences=return_sequences, return_state=True
        )

    def test_outputs_states_and_gradients_match_keras(self):
        inputs = tf.cast(
            tf.random.stateless_uniform((2, 6, self.lgn_input["n_inputs"]), (3, 4)) < 0.05,
            tf.float32,
        )
        for return_sequences in (True, False):
            with self.subTest(return_sequences=return_sequences):
                rnn = self._rnn(return_sequences)
                state = rnn.cell.zero_state(2)
                rnn(inputs, initial_state=state)

                @tf.function
                def run(loop):
                    with tf.GradientTape() as tape:
                        last, outputs, states = loop(
                            sequences=inputs, initial_state=list(state), mask=None
                        )
                        result = outputs if return_sequences else last
                        loss = tf.add_n([
                            tf.reduce_sum(tf.cast(value, tf.float32))
                            for value in tf.nest.flatten((result, states))
                            if value.dtype.is_floating
                        ])
                    gradients = tape.gradient(loss, rnn.trainable_variables)
                    return result, states, gradients

                ours = run(rnn.inner_loop)
                keras = run(lambda **kwargs: tf.keras.layers.RNN.inner_loop(rnn, **kwargs))
                for mine, reference in zip(tf.nest.flatten(ours), tf.nest.flatten(keras)):
                    np.testing.assert_array_equal(mine.numpy(), reference.numpy())


if __name__ == "__main__":
    unittest.main()
