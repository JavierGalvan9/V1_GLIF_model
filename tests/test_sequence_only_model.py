import unittest

import numpy as np
import tensorflow as tf

from v1_model_utils.models import (
    ClassificationReadoutLayer,
    FiringRateMetricLayer,
    build_sequence_only_model,
)


class ToyRecurrentLayer(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__(name="toy_rsnn")
        self.projection = tf.keras.layers.Dense(
            3,
            kernel_initializer=tf.keras.initializers.GlorotUniform(seed=7),
        )

    def call(self, inputs):
        spikes = self.projection(inputs)
        voltages = 2.0 * spikes
        final_state = tf.reduce_mean(spikes, axis=1)
        return (spikes, voltages), final_state


class SequenceOnlyModelTests(unittest.TestCase):
    def test_classification_readout_serialization_round_trip(self):
        layer = ClassificationReadoutLayer(
            readout_neuron_ids=[np.array([0, 1]), np.array([2, 3])],
            n_output=2,
            dampening_factor=0.3,
            seq_len=4,
            down_sample=2,
        )
        inputs = tf.ones((1, 4, 4), dtype=tf.float32)
        expected = layer(inputs)

        restored = ClassificationReadoutLayer.from_config(layer.get_config())
        restored(inputs)
        restored.set_weights(layer.get_weights())

        np.testing.assert_allclose(restored(inputs).numpy(), expected.numpy())

    def test_firing_rate_metric_layer_tracks_and_passes_spikes(self):
        inputs = tf.keras.Input(shape=(2, 3))
        layer = FiringRateMetricLayer()
        model = tf.keras.Model(inputs, layer(inputs))
        spikes = np.array([[[0.0, 1.0, 0.0], [1.0, 1.0, 0.0]]])

        output = model(spikes)

        np.testing.assert_array_equal(output.numpy(), spikes)
        self.assertAlmostEqual(float(layer.rate.result()), 0.5)
        self.assertEqual([metric.name for metric in model.metrics], ["rate"])

    def build_models(self):
        inputs = tf.keras.Input(shape=(4, 2), name="stimulus")
        rsnn_layer = ToyRecurrentLayer()
        sequences, final_state = rsnn_layer(inputs)
        readout_layer = tf.keras.layers.Dense(1, name="unused_readout")
        readout = readout_layer(final_state)
        full_model = tf.keras.Model(inputs=inputs, outputs=readout)
        sequence_model = build_sequence_only_model(full_model, rsnn_layer)
        return full_model, sequence_model, rsnn_layer, readout_layer, sequences

    def test_returns_only_sequence_pair_and_matches_rsnn_output(self):
        (
            full_model,
            sequence_model,
            _rsnn_layer,
            readout_layer,
            sequences,
        ) = self.build_models()
        reference_model = tf.keras.Model(
            inputs=full_model.inputs, outputs=sequences
        )
        inputs = tf.reshape(tf.range(16, dtype=tf.float32), (2, 4, 2))

        actual = sequence_model(inputs)
        expected = reference_model(inputs)

        self.assertEqual(len(actual), 2)
        for actual_tensor, expected_tensor in zip(actual, expected):
            np.testing.assert_allclose(
                actual_tensor.numpy(), expected_tensor.numpy()
            )
        self.assertNotIn(readout_layer, sequence_model.layers)
        self.assertEqual(
            {variable.path for variable in sequence_model.trainable_variables},
            {
                variable.path
                for variable in reference_model.trainable_variables
            },
        )

    def test_preserves_gradients_to_recurrent_variables(self):
        _, sequence_model, rsnn_layer, _, _ = self.build_models()
        inputs = tf.ones((2, 4, 2), dtype=tf.float32)

        with tf.GradientTape() as tape:
            spikes, voltages = sequence_model(inputs)
            loss = tf.reduce_sum(spikes + voltages)
        gradients = tape.gradient(loss, rsnn_layer.trainable_variables)

        self.assertTrue(gradients)
        self.assertTrue(all(gradient is not None for gradient in gradients))
        self.assertTrue(
            all(
                np.all(np.isfinite(gradient.numpy()))
                for gradient in gradients
            )
        )

if __name__ == "__main__":
    unittest.main()
