"""Grating Bernoulli sampling stays FP32 under mixed precision."""

import pytest
import tensorflow as tf

import stim_dataset


@pytest.mark.parametrize("dtype", [tf.float16, tf.float32])
def test_tensorflow_grating_sampling_uses_float32(monkeypatch, dtype):
    monkeypatch.setattr(stim_dataset.lgn_module, "LGN", lambda **kwargs: None)
    grating = stim_dataset.DriftingGratingLGN(
        64, 0, 0, n_input=8, dtype=dtype, lgn_backend="tensorflow"
    )
    rates = tf.reshape(tf.linspace(0.01, 100.0, 512), (64, 8))
    rates = tf.cast(rates, dtype)
    monkeypatch.setattr(grating, "firing_rates", lambda theta, phase: rates)
    expected_probability = -tf.math.expm1(-tf.cast(rates, tf.float32) / 1000.0)
    theta = tf.zeros((2, 1), dtype)
    phase = tf.zeros((2,), dtype)
    seeds = tf.constant([[7, 3], [11, 5]])

    probabilities = grating.batch_probabilities(theta, phase)
    assert probabilities.dtype == tf.float32
    tf.debugging.assert_equal(probabilities[0], expected_probability)
    expected_spikes = tf.stack([
        tf.random.stateless_uniform((64, 8), seed, dtype=tf.float32)
        < expected_probability
        for seed in seeds
    ])
    tf.debugging.assert_equal(grating.batch_spikes(theta, phase, seeds), expected_spikes)
    tf.debugging.assert_equal(grating.spikes(0, 0, seeds[0]), expected_spikes[0])
    assert grating.spikes(0, 0, current_input=True).dtype == dtype

    uniforms = tf.random.stateless_uniform((64, 8), (19, 2), dtype=tf.float32)

    def uniform(shape, dtype):
        assert dtype == tf.float32
        tf.debugging.assert_equal(shape, [64, 8])
        return uniforms

    monkeypatch.setattr(tf.random, "uniform", uniform)
    tf.debugging.assert_equal(grating.spikes(0, 0), uniforms < expected_probability)
