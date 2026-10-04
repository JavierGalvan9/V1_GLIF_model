"""Regression coverage for shapes outside the optimized CUDA history path."""

import numpy as np
import pytest
import tensorflow as tf

from v1_model_utils.cuda_csr_recurrent.wrapper import (
    build_csr_connectivity,
    calculate_recurrent_csr_currents,
    spike_queue,
)


@pytest.mark.parametrize("slots", [8, 9, 12, 32])
@pytest.mark.parametrize("basis_dim", [3, 4, 5])
@pytest.mark.parametrize("dtype", [tf.float16, tf.float32])
@pytest.mark.parametrize("resource_mode", ["0", "1"])
@pytest.mark.parametrize("accumulate", [False, True])
def test_long_history_with_carried_queues_matches_flattened(
    slots, basis_dim, dtype, resource_mode, accumulate, monkeypatch
):
    """A long queued history must retain forward and gradient semantics."""
    monkeypatch.setenv("V1_CSR_RESOURCE_MODE", resource_mode)
    width, batch, n_post = 3, 3, 5
    pre = np.arange(slots * width)
    post = pre % n_post
    indices = np.stack((post, pre), axis=1)
    connectivity = build_csr_connectivity(
        indices, np.zeros(pre.size, np.uint8), n_pre=pre.size,
        n_post=n_post, weights_csr_ordered=True,
    )
    rng = np.random.default_rng(410)
    history = tuple(
        tf.Variable((rng.random((batch, width)) < 0.4).astype(dtype.as_numpy_dtype))
        for _ in range(slots)
    )
    weights = tf.Variable(rng.uniform(0.01, 0.1, pre.size).astype(np.float32))
    basis = tf.constant(rng.uniform(0.1, 0.4, (1, basis_dim)), tf.float32)
    upstream = tf.constant(rng.normal(size=(batch * n_post, basis_dim)), dtype)
    initial = tf.Variable(tf.fill((batch * n_post, basis_dim), tf.cast(0.1, dtype)))
    records = tuple(spike_queue(slot) for slot in history[1:])

    def evaluate(carried):
        with tf.GradientTape() as tape:
            operand = history if carried else tf.concat(history, axis=1)
            result = calculate_recurrent_csr_currents(
                operand, weights, basis, 0.3, connectivity,
                initial=initial if accumulate else None,
                queues=records if carried else None,
            )
            currents = result[0] if carried else result
            loss = tf.reduce_sum(tf.cast(currents, tf.float32) * tf.cast(upstream, tf.float32))
        operands = (*history, weights, initial) if accumulate else (*history, weights)
        gradients = tape.gradient(loss, operands)
        return currents.numpy(), [gradient.numpy() for gradient in gradients], result

    expected, expected_gradients, _ = evaluate(False)
    actual, gradients, result = evaluate(True)
    tolerance = 3e-3 if dtype == tf.float16 else 2e-6
    np.testing.assert_allclose(actual, expected, rtol=tolerance, atol=tolerance)
    for actual_gradient, expected_gradient in zip(gradients, expected_gradients):
        np.testing.assert_allclose(
            actual_gradient, expected_gradient, rtol=tolerance, atol=tolerance
        )
    record = result[1].numpy()
    reference = spike_queue(history[0]).numpy()
    capacity = batch * width
    count = int(reference[capacity])
    np.testing.assert_array_equal(record[:count], reference[:count])
    np.testing.assert_array_equal(record[capacity:], reference[capacity:])
