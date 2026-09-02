"""Public-interface regression tests for recurrent CUDA currents."""

import numpy as np
import tensorflow as tf

from v1_model_utils.cuda_csr_recurrent.wrapper import (
    build_csr_connectivity,
    calculate_recurrent_csr_currents,
)


def test_batch32_pair_projection_matches_independent_reference():
    rng = np.random.default_rng(20260902)
    n_pre, n_post, n_types, batch = 7, 11, 3, 32
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

    spikes_np = (rng.random((batch, n_pre)) < 0.25).astype(np.float16)
    weights_np = rng.normal(size=pre.size).astype(np.float32)
    basis_np = rng.normal(size=(n_types, 4)).astype(np.float16)
    upstream_np = rng.normal(size=(batch, n_post, 4)).astype(np.float16)
    dampening = np.float16(0.1)

    spikes = tf.Variable(spikes_np)
    weights = tf.Variable(weights_np)
    basis = tf.constant(basis_np)
    with tf.GradientTape() as tape:
        currents = calculate_recurrent_csr_currents(
            spikes, weights, basis, dampening, connectivity
        )
        loss = tf.reduce_sum(currents * tf.reshape(upstream_np, currents.shape))
    spike_grad, weight_grad = tape.gradient(loss, (spikes, weights))

    expected_currents = np.zeros((batch, n_post, 4), dtype=np.float32)
    expected_spike_grad = np.zeros((batch, n_pre), dtype=np.float32)
    expected_weight_grad = np.zeros(pre.size, dtype=np.float32)
    for edge, (target, source) in enumerate(indices):
        projection = np.sum(
            upstream_np[:, target].astype(np.float32)
            * basis_np[synapse_types[edge]].astype(np.float32),
            axis=1,
        )
        expected_currents[:, target] += (
            spikes_np[:, source, None].astype(np.float32)
            * weights_np[edge]
            * basis_np[synapse_types[edge]].astype(np.float32)
        )
        expected_spike_grad[:, source] += projection * weights_np[edge] * dampening
        expected_weight_grad[edge] = np.sum(
            projection * spikes_np[:, source].astype(np.float32)
        )

    np.testing.assert_allclose(
        currents.numpy().reshape(batch, n_post, 4),
        expected_currents.astype(np.float16),
        rtol=3e-3,
        atol=1e-2,
    )
    np.testing.assert_allclose(
        spike_grad.numpy(), expected_spike_grad.astype(np.float16), rtol=3e-3, atol=1e-2
    )
    np.testing.assert_allclose(
        weight_grad.numpy(), expected_weight_grad, rtol=3e-3, atol=1e-2
    )
