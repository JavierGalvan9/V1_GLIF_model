import pickle
import unittest

import numpy as np
import tensorflow as tf

from v1_model_utils import loss_functions, models


def _cached_triple(path):
    """Return the cached ``(network, lgn, bkg)`` triple from either cache format.

    ``load_sparse`` wraps the payload in a versioned dict; older caches on disk
    are the bare triple.
    """
    with open(path, "rb") as handle:
        cached = pickle.load(handle)
    return cached["payload"] if isinstance(cached, dict) else cached



class OnlineVoltageLossTest(unittest.TestCase):
    """Contract tests for the public V1Column legacy/online output modes."""

    @classmethod
    def setUpClass(cls):
        cls.network, cls.lgn_input, cls.bkg_input = _cached_triple(
            "GLIF_network_nll_full/tf_data/V1_network_v1_10.pkl"
        )

    def _cell(self, *, online, penalty_mode):
        return models.V1Column(
            self.network, self.lgn_input, self.bkg_input,
            batch_size=1, max_delay=1, train_recurrent=True,
            train_recurrent_per_type=False, train_input=False,
            train_noise=False, noise_seed=17, acceleration="tensorflow",
            track_voltage_penalty=online, voltage_penalty_mode=penalty_mode,
            return_voltage_sequences=not online,
        )

    def _run(self, cell, inputs, state=None):
        if state is None:
            state = cell.zero_state(1)
        rnn = models.HeterogeneousStateRNN(
            cell, return_sequences=True, return_state=True
        )
        outputs = rnn(inputs, initial_state=state)
        return models._normalize_v1_rnn_outputs(rnn, outputs)

    def test_online_loss_matches_legacy_loss_and_gradients(self):
        inputs = tf.cast(
            tf.random.stateless_uniform((1, 4, self.lgn_input["n_inputs"]), (1, 2)) < 0.01,
            tf.float32,
        )
        for penalty_mode in ("range", "threshold"):
            with self.subTest(penalty_mode=penalty_mode):
                legacy = self._cell(online=False, penalty_mode=penalty_mode)
                online = self._cell(online=True, penalty_mode=penalty_mode)
                self._run(legacy, inputs)
                self._run(online, inputs)
                online.set_weights(legacy.get_weights())

                with tf.GradientTape() as legacy_tape:
                    legacy_out = self._run(legacy, inputs)
                    legacy_z, legacy_v = legacy_out[0]
                    legacy_loss = loss_functions.VoltageRegularization(
                        legacy, voltage_cost=1.0, penalty_mode=penalty_mode
                    )(legacy_v)
                legacy_gradients = legacy_tape.gradient(
                    legacy_loss, legacy.trainable_variables
                )

                with tf.GradientTape() as online_tape:
                    online_out = self._run(online, inputs)
                    online_z = online_out[0]
                    online_loss = tf.reduce_mean(online_out[-1]) / 4.0
                online_gradients = online_tape.gradient(
                    online_loss, online.trainable_variables
                )

                np.testing.assert_array_equal(online_z.numpy(), legacy_z.numpy())
                np.testing.assert_allclose(
                    online_out[2].numpy(), legacy_out[2].numpy(), rtol=2e-5, atol=2e-6
                )
                np.testing.assert_allclose(
                    online_loss.numpy(), legacy_loss.numpy(), rtol=2e-5, atol=2e-6
                )
                for expected, actual in zip(legacy_gradients, online_gradients):
                    self.assertEqual(expected is None, actual is None)
                    if expected is not None:
                        np.testing.assert_allclose(
                            actual.numpy(), expected.numpy(), rtol=2e-5, atol=2e-6
                        )

    def test_create_model_exposes_spikes_and_final_penalty_only(self):
        model = models.create_model(
            self.network, self.lgn_input, self.bkg_input, seq_len=4,
            n_input=self.lgn_input["n_inputs"], batch_size=1,
            use_state_input=True, return_state=True, max_delay=1,
            train_recurrent=False, train_noise=False, acceleration="tensorflow",
            track_voltage_penalty=True, return_voltage_sequences=False,
        )
        rsnn = model.get_layer("rsnn")
        extractor = models.build_sequence_and_state_model(model, rsnn)
        outputs = extractor((
            tf.zeros((1, 4, self.lgn_input["n_inputs"])),
            rsnn.cell.zero_state(1),
        ))
        sequences = tf.nest.flatten(outputs[0])
        self.assertEqual(len(sequences), 1)
        self.assertEqual(tuple(sequences[0].shape), (1, 4, self.network["n_nodes"]))
        self.assertEqual(tuple(outputs[-1].shape), (1, 1))

    def test_online_penalty_preserves_column_shape_for_multi_example_batch(self):
        batch_size = 3
        cell = self._cell(online=True, penalty_mode="range")
        inputs = tf.zeros(
            (batch_size, 2, self.lgn_input["n_inputs"]), tf.float32
        )
        state = cell.zero_state(batch_size)

        rnn = models.HeterogeneousStateRNN(
            cell, return_sequences=True, return_state=True
        )

        @tf.function
        def rollout(sequence, initial_state):
            return rnn(sequence, initial_state=initial_state)

        outputs = rollout(inputs, state)

        self.assertEqual(tuple(outputs[-1].shape), (batch_size, 1))

    def test_reset_voltage_penalty_preserves_warm_neural_state(self):
        cell = self._cell(online=True, penalty_mode="range")
        warm_inputs = tf.ones((1, 3, self.lgn_input["n_inputs"]), tf.float32)
        warm_outputs = self._run(cell, warm_inputs)
        warm_state = tuple(warm_outputs[1:])

        reset_state = models.reset_voltage_penalty_state(cell, warm_state)

        for expected, actual in zip(warm_state[:-1], reset_state[:-1]):
            np.testing.assert_array_equal(actual.numpy(), expected.numpy())
        np.testing.assert_array_equal(
            reset_state[-1].numpy(), np.zeros((1, 1), dtype=np.float32)
        )
        self.assertGreater(float(warm_state[-1][0, 0]), 0.0)

    def test_warm_started_online_loss_and_gradients_match_legacy(self):
        warm_inputs = tf.cast(
            tf.random.stateless_uniform((1, 3, self.lgn_input["n_inputs"]), (5, 6)) < 0.01,
            tf.float32,
        )
        stimulus_inputs = tf.cast(
            tf.random.stateless_uniform((1, 4, self.lgn_input["n_inputs"]), (7, 8)) < 0.01,
            tf.float32,
        )
        legacy = self._cell(online=False, penalty_mode="range")
        online = self._cell(online=True, penalty_mode="range")
        self._run(legacy, warm_inputs)
        self._run(online, warm_inputs)
        online.set_weights(legacy.get_weights())
        legacy_warm_state = tuple(self._run(legacy, warm_inputs)[1:])
        online_warm_state = models.reset_voltage_penalty_state(
            online, tuple(self._run(online, warm_inputs)[1:])
        )

        with tf.GradientTape() as legacy_tape:
            legacy_out = self._run(legacy, stimulus_inputs, legacy_warm_state)
            legacy_z, legacy_v = legacy_out[0]
            legacy_loss = loss_functions.VoltageRegularization(
                legacy, voltage_cost=1.0, penalty_mode="range"
            )(legacy_v)
        legacy_gradients = legacy_tape.gradient(legacy_loss, legacy.trainable_variables)

        with tf.GradientTape() as online_tape:
            online_out = self._run(online, stimulus_inputs, online_warm_state)
            online_z = online_out[0]
            online_loss = tf.reduce_mean(online_out[-1]) / 4.0
        online_gradients = online_tape.gradient(online_loss, online.trainable_variables)

        np.testing.assert_array_equal(online_z.numpy(), legacy_z.numpy())
        np.testing.assert_allclose(online_loss.numpy(), legacy_loss.numpy(), rtol=2e-5, atol=2e-6)
        for expected, actual in zip(legacy_gradients, online_gradients):
            self.assertEqual(expected is None, actual is None)
            if expected is not None:
                np.testing.assert_allclose(actual.numpy(), expected.numpy(), rtol=2e-5, atol=2e-6)

    def test_fp16_actual_width_loss_and_voltage_gradients_match(self):
        n_neurons = 203816
        sequence_length = 7
        initial_voltage = tf.cast(
            tf.random.stateless_uniform(
                (1, sequence_length, n_neurons),
                (13, 17),
                minval=-0.25,
                maxval=1.25,
            ),
            tf.float16,
        )
        for penalty_mode in ("range", "threshold"):
            with self.subTest(penalty_mode=penalty_mode):
                voltage = tf.Variable(initial_voltage)
                with tf.GradientTape() as legacy_tape:
                    legacy_loss = loss_functions.VoltageRegularization(
                        None, voltage_cost=1.0, penalty_mode=penalty_mode
                    )(voltage)
                legacy_gradient = legacy_tape.gradient(legacy_loss, voltage)

                with tf.GradientTape() as online_tape:
                    step_penalties = [
                        models.compute_voltage_penalty_mean_step(
                            voltage[:, step], n_neurons, penalty_mode
                        )
                        for step in range(sequence_length)
                    ]
                    online_loss = (
                        tf.reduce_mean(tf.add_n(step_penalties))
                        / float(sequence_length)
                    )
                online_gradient = online_tape.gradient(online_loss, voltage)

                np.testing.assert_allclose(
                    online_loss.numpy(), legacy_loss.numpy(), rtol=5e-4, atol=2e-6
                )
                np.testing.assert_allclose(
                    online_gradient.numpy(),
                    legacy_gradient.numpy(),
                    rtol=5e-3,
                    atol=2 * np.nextafter(np.float16(0), np.float16(1)),
                )


if __name__ == "__main__":
    unittest.main()
