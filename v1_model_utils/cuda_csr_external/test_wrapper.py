"""Public-interface regression tests for external CUDA currents."""

import numpy as np
import pytest
import tensorflow as tf

from v1_model_utils.cuda_csr_external.wrapper import (
    build_csr_connectivity,
    calculate_external_csr_currents,
)


@pytest.mark.parametrize("batch,n_basis", ((32, 4), (32, 3), (33, 5)))
def test_external_gradients_match_independent_reference(batch, n_basis):
    rng = np.random.default_rng(20260915 + batch + n_basis)
    n_pre, n_post, n_types = 7, 11, 3
    pre = np.repeat(np.arange(n_pre), (19, 2, 33, 1, 17, 5, 24))
    post = rng.integers(0, n_post, pre.size, dtype=np.int64)
    synapse_types = rng.integers(0, n_types, pre.size, dtype=np.int64)
    order = np.lexsort((np.arange(pre.size), synapse_types, post, pre))
    indices = np.stack((post[order], pre[order]), axis=1)
    synapse_types = synapse_types[order]
    connectivity = build_csr_connectivity(
        indices,
        synapse_types,
        n_pre=n_pre,
        n_post=n_post,
        weights_csr_ordered=True,
    )

    activity_np = (rng.random((batch, n_pre)) < 0.2).astype(np.float16)
    weights_np = rng.normal(size=pre.size).astype(np.float32)
    basis_np = rng.normal(size=(n_types, n_basis)).astype(np.float16)
    upstream_np = rng.normal(size=(batch, n_post, n_basis)).astype(np.float16)
    activity = tf.Variable(activity_np)
    weights = tf.Variable(weights_np)
    basis = tf.constant(basis_np)
    with tf.GradientTape() as tape:
        currents = calculate_external_csr_currents(
            activity, weights, basis, connectivity
        )
        loss = tf.reduce_sum(currents * tf.reshape(upstream_np, currents.shape))
    activity_grad, weight_grad = tape.gradient(loss, (activity, weights))

    expected_activity_grad = np.zeros((batch, n_pre), dtype=np.float32)
    expected_weight_grad = np.zeros(pre.size, dtype=np.float32)
    for edge, (target, source) in enumerate(indices):
        projection = np.sum(
            upstream_np[:, target].astype(np.float32)
            * basis_np[synapse_types[edge]].astype(np.float32),
            axis=1,
        )
        expected_activity_grad[:, source] += projection * weights_np[edge]
        expected_weight_grad[edge] = np.sum(
            projection * activity_np[:, source].astype(np.float32)
        )

    np.testing.assert_allclose(
        activity_grad.numpy(),
        expected_activity_grad.astype(np.float16),
        rtol=3e-3,
        atol=1e-2,
    )
    np.testing.assert_allclose(
        weight_grad.numpy(), expected_weight_grad, rtol=3e-3, atol=1e-2
    )


@pytest.mark.skipif(not tf.config.list_physical_devices("GPU"), reason="CUDA GPU required")
@pytest.mark.parametrize("batch", [3, 5, 7, 9, 31, 33])
@pytest.mark.parametrize("resource_mode", ["0", "1"])
def test_short_row_activity_gradient_matches_reference(batch, resource_mode, monkeypatch):
    """Odd batches with many short rows must retain dense and empty-row gradients."""
    monkeypatch.setenv("V1_CSR_RESOURCE_MODE", resource_mode)
    rng = np.random.default_rng(20261003)
    n_pre, n_post = 4609, 127
    pre = np.repeat(np.arange(n_pre - 1), 2)
    post = rng.integers(n_post, size=pre.size)
    types = rng.integers(3, size=pre.size)
    order = np.lexsort((np.arange(pre.size), types, post, pre))
    pre, post, types = pre[order], post[order], types[order]
    connectivity = build_csr_connectivity(
        np.stack((post, pre), axis=1), types, n_pre=n_pre, n_post=n_post,
        weights_csr_ordered=True, needs_activity_backward=True,
    )
    weights = rng.normal(0, 0.1, pre.size).astype(np.float32)
    basis = rng.normal(0, 0.1, (3, 4)).astype(np.float32)
    upstream = rng.normal(0, 0.1, (batch, n_post, 4)).astype(np.float16)
    activity = tf.Variable((rng.random((batch, n_pre)) < 0.03).astype(np.float16))
    with tf.GradientTape() as tape:
        currents = calculate_external_csr_currents(
            activity, tf.constant(weights), tf.constant(basis), connectivity,
            compute_activity_gradient=True, compute_weight_gradient=False,
        )
        loss = tf.reduce_sum(currents * tf.reshape(upstream, currents.shape))
    actual = tape.gradient(loss, activity).numpy()
    expected = np.zeros((batch, n_pre), np.float32)
    for sample in range(batch):
        projections = np.sum(upstream[sample, post].astype(np.float32) * basis[types], axis=1)
        np.add.at(expected[sample], pre, projections * weights)
    np.testing.assert_allclose(actual, expected.astype(np.float16), rtol=3e-3, atol=2e-5)
    np.testing.assert_array_equal(actual[:, -1], 0)
