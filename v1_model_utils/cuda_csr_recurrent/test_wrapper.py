"""Public-interface regression tests for recurrent CUDA currents."""

import numpy as np
import pytest
import tensorflow as tf

from v1_model_utils.cuda_csr_recurrent.wrapper import (
    build_csr_connectivity,
    calculate_recurrent_csr_currents,
    spike_queue,
)


@pytest.mark.parametrize(
    "batch,n_basis", ((1, 4), (32, 4), (32, 3), (33, 5), (64, 4), (65, 4), (130, 3))
)
def test_pair_projection_matches_independent_reference(batch, n_basis):
    rng = np.random.default_rng(20260902)
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

    spikes_np = (rng.random((batch, n_pre)) < 0.25).astype(np.float16)
    weights_np = rng.normal(size=pre.size).astype(np.float32)
    basis_np = rng.normal(size=(n_types, n_basis)).astype(np.float16)
    upstream_np = rng.normal(size=(batch, n_post, n_basis)).astype(np.float16)
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

    expected_currents = np.zeros((batch, n_post, n_basis), dtype=np.float32)
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
        currents.numpy().reshape(batch, n_post, n_basis),
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


@pytest.mark.parametrize("slots", (2, 3))
@pytest.mark.parametrize("dtype", (np.float16, np.float32))
@pytest.mark.parametrize("batch", (1, 64, 65))
def test_carried_queue_records_give_the_same_currents(batch, dtype, slots):
    """Carried slot records replace sweeps only: same currents, and the newest
    slot's record is the one spike_queue builds.

    Every target receives at most two edges, so each current is a sum of at
    most two terms, whose value does not depend on the order the forward's
    atomics add them in: the currents are exactly reproducible.
    """
    rng = np.random.default_rng(20261001)
    width, n_post, n_types = 40, 200, 3
    n_pre = slots * width
    post = rng.permutation(np.repeat(np.arange(n_post), 2))[: 3 * n_pre]
    pre = np.sort(rng.integers(0, n_pre, post.size))
    synapse_types = rng.integers(0, n_types, pre.size)
    order = np.lexsort((np.arange(pre.size), synapse_types, post, pre))
    connectivity = build_csr_connectivity(
        np.stack((post[order], pre[order]), axis=1), synapse_types[order],
        n_pre=n_pre, n_post=n_post, weights_csr_ordered=True,
    )
    history = tuple(
        tf.constant((rng.random((batch, width)) < 0.3).astype(dtype)) for _ in range(slots)
    )
    weights = tf.constant(rng.normal(size=pre.size).astype(np.float32))
    basis = tf.constant(rng.normal(size=(n_types, 4)).astype(np.float32))
    plain = calculate_recurrent_csr_currents(history, weights, basis, 0.3, connectivity)
    carried, record = calculate_recurrent_csr_currents(
        history, weights, basis, 0.3, connectivity,
        queues=tuple(spike_queue(slot) for slot in history[1:]),
    )
    np.testing.assert_array_equal(carried.numpy().view(np.uint8), plain.numpy().view(np.uint8))
    # A record: the active entries (room for batch * width), their count, the
    # batch + 1 sample starts. The room past the count is unused.
    capacity = batch * width

    def valid(words):
        count = int(words[capacity])
        return words[:count], words[capacity:]

    for got, want in zip(valid(record.numpy()), valid(spike_queue(history[0]).numpy())):
        np.testing.assert_array_equal(got, want)
