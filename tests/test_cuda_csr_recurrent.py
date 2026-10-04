import dataclasses

import numpy as np
import pytest
import tensorflow as tf

from v1_model_utils.cuda_csr_recurrent import (
    DIRECT_CSR,
    accumulate_recurrent_weight_gradient,
    build_csr_connectivity,
    calculate_recurrent_csr_currents,
)


POWER_OF_TWO_BATCHES = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512)


def _csr_sorted(indices, synapse_types, weights=None):
    """Put edges in the operator's CSR order, as the model's network is.

    With DIRECT_CSR the kernels index weights by CSR position, so a fixture has
    to be ordered the same way the reordered network is.
    """
    order = np.lexsort((
        np.arange(indices.shape[0]), synapse_types, indices[:, 0], indices[:, 1]
    ))
    if weights is None:
        return indices[order], synapse_types[order]
    return indices[order], synapse_types[order], weights[order]


def _connectivity(indices, synapse_types, n_pre, n_post):
    """Build connectivity, declaring the CSR ordering the kernels require."""
    return build_csr_connectivity(
        indices, synapse_types, n_pre=n_pre, n_post=n_post,
        weights_csr_ordered=DIRECT_CSR,
    )


def _fixture(batch_size, n_basis):
    rng = np.random.default_rng(1234 + batch_size + n_basis)
    indices = np.array(
        [[2, 0], [0, 1], [1, 2], [1, 0], [2, 2], [0, 3], [2, 1]],
        dtype=np.int64,
    )
    synapse_types = np.array([1, 0, 2, 1, 0, 2, 1], dtype=np.int64)
    spikes = rng.uniform(0.1, 1.0, size=(batch_size, 4)).astype(np.float32)
    spikes[rng.random(spikes.shape) < 0.5] = 0.0
    weights = rng.normal(size=indices.shape[0]).astype(np.float32)
    basis = rng.normal(size=(3, n_basis)).astype(np.float32)
    upstream = rng.normal(size=(batch_size * 3, n_basis)).astype(np.float32)
    indices, synapse_types, weights = _csr_sorted(indices, synapse_types, weights)
    return indices, synapse_types, spikes, weights, basis, upstream


def _reference(indices, synapse_types, spikes, weights, basis, upstream, dampening):
    batch_size, n_pre = spikes.shape
    n_post, n_basis = 3, basis.shape[1]
    currents = np.zeros((batch_size * n_post, n_basis), np.float32)
    spike_grad = np.zeros_like(spikes)
    weight_grad = np.zeros_like(weights)
    for edge, (post, pre) in enumerate(indices):
        vector = weights[edge] * basis[synapse_types[edge]]
        for batch in range(batch_size):
            currents[batch * n_post + post] += spikes[batch, pre] * vector
            projected = np.dot(upstream[batch * n_post + post], basis[synapse_types[edge]])
            spike_grad[batch, pre] += dampening * weights[edge] * projected
            weight_grad[edge] += spikes[batch, pre] * projected
    return currents, spike_grad, weight_grad


@pytest.mark.skipif(not tf.config.list_physical_devices("GPU"), reason="CUDA GPU required")
@pytest.mark.parametrize("n_basis", [4, 5])
@pytest.mark.parametrize("batch_size", POWER_OF_TWO_BATCHES + (3, 6, 10, 24, 48))
def test_forward_and_backward_match_reference(batch_size, n_basis):
    indices, synapse_types, spikes, weights, basis, upstream = _fixture(batch_size, n_basis)
    connectivity = _connectivity(indices, synapse_types, n_pre=4, n_post=3)
    dampening = 0.37
    spikes_tensor = tf.constant(spikes)
    master_weights = tf.Variable(weights)
    with tf.GradientTape() as tape:
        tape.watch(spikes_tensor)
        currents = calculate_recurrent_csr_currents(
            spikes_tensor,
            master_weights,
            tf.constant(basis),
            dampening,
            connectivity,
        )
        loss = tf.reduce_sum(currents * upstream)
    spike_grad, weight_grad = tape.gradient(loss, (spikes_tensor, master_weights))
    expected = _reference(indices, synapse_types, spikes, weights, basis, upstream, dampening)
    np.testing.assert_allclose(currents.numpy(), expected[0], rtol=2e-5, atol=2e-5)
    np.testing.assert_allclose(spike_grad.numpy(), expected[1], rtol=2e-5, atol=2e-5)
    np.testing.assert_allclose(weight_grad.numpy(), expected[2], rtol=2e-5, atol=2e-5)


@pytest.mark.skipif(not tf.config.list_physical_devices("GPU"), reason="CUDA GPU required")
def test_tf_function_empty_activity_and_weight_updates():
    indices, synapse_types, spikes, weights, basis, upstream = _fixture(3, 4)
    connectivity = _connectivity(indices, synapse_types, n_pre=4, n_post=3)
    master_weights = tf.Variable(weights)

    @tf.function
    def run(spike_values):
        with tf.GradientTape() as tape:
            tape.watch(spike_values)
            currents = calculate_recurrent_csr_currents(
                spike_values, master_weights, basis, 0.5, connectivity
            )
            loss = tf.reduce_sum(currents * upstream)
        return currents, tape.gradient(loss, (spike_values, master_weights))

    currents, (spike_grad, weight_grad) = run(tf.zeros_like(spikes))
    np.testing.assert_array_equal(currents.numpy(), 0.0)
    np.testing.assert_array_equal(weight_grad.numpy(), 0.0)
    assert np.all(np.isfinite(spike_grad.numpy()))
    master_weights.assign_add(tf.ones_like(master_weights))
    updated, _ = run(tf.constant(spikes))
    assert np.any(updated.numpy() != 0.0)


@pytest.mark.skipif(not tf.config.list_physical_devices("GPU"), reason="CUDA GPU required")
def test_mixed_precision_uses_fp32_weights_and_fp32_weight_gradient():
    indices, synapse_types, spikes, weights, basis, upstream = _fixture(2, 4)
    connectivity = _connectivity(indices, synapse_types, n_pre=4, n_post=3)
    spikes = tf.constant(spikes, tf.float16)
    master_weights = tf.Variable(weights, dtype=tf.float32)
    with tf.GradientTape() as tape:
        tape.watch(spikes)
        currents = calculate_recurrent_csr_currents(
            spikes,
            master_weights,
            tf.constant(basis, tf.float16),
            0.1,
            connectivity,
        )
        loss = tf.reduce_sum(currents * tf.cast(upstream, tf.float16))
    spike_grad, weight_grad = tape.gradient(loss, (spikes, master_weights))
    assert currents.dtype == tf.float16
    assert spike_grad.dtype == tf.float16
    assert weight_grad.dtype == tf.float32
    assert np.all(np.isfinite(weight_grad.numpy()))


def test_compact_pairs_describe_every_csr_edge():
    """pair_ids must reproduce each edge's (postsynaptic, synapse type) pair."""
    indices, synapse_types, _, _, _, _ = _fixture(32, 4)
    connectivity = _connectivity(indices, synapse_types, n_pre=4, n_post=3)
    order = connectivity.edge_order
    csr_posts = indices[order, 0]
    csr_types = synapse_types[order]
    pair_ids = connectivity.pair_ids.numpy()
    pair_posts = connectivity.pair_posts.numpy()
    pair_types = connectivity.pair_types.numpy()

    assert pair_ids.shape == (indices.shape[0],)
    assert connectivity.n_pairs == pair_posts.size == pair_types.size
    assert connectivity.n_pairs == len(set(zip(csr_posts.tolist(), csr_types.tolist())))
    np.testing.assert_array_equal(pair_posts[pair_ids], csr_posts)
    np.testing.assert_array_equal(pair_types[pair_ids], csr_types)


def _dense_fixture(batch, n_basis, seed):
    """Several hundred edges per row, so a slicing or reduction error cannot hide."""
    rng = np.random.default_rng(seed)
    n_pre, n_post, n_types, n_edges = 48, 96, 7, 6000
    indices = np.stack(
        (rng.integers(0, n_post, n_edges), rng.integers(0, n_pre, n_edges)), axis=1
    ).astype(np.int64)
    synapse_types = rng.integers(0, n_types, n_edges).astype(np.int64)
    spikes = rng.uniform(0.1, 1.0, (batch, n_pre)).astype(np.float32)
    spikes[rng.random(spikes.shape) < 0.4] = 0.0
    weights = rng.normal(size=n_edges).astype(np.float32)
    basis = rng.normal(size=(n_types, n_basis)).astype(np.float32)
    upstream = rng.normal(size=(batch * n_post, n_basis)).astype(np.float32)
    indices, synapse_types, weights = _csr_sorted(indices, synapse_types, weights)
    return indices, synapse_types, spikes, weights, basis, upstream, n_pre, n_post


def _relative_error(got, want):
    got = np.asarray(got, np.float64)
    return np.linalg.norm(got - want) / np.linalg.norm(want)


@pytest.mark.skipif(not tf.config.list_physical_devices("GPU"), reason="CUDA GPU required")
@pytest.mark.parametrize("dtype", [tf.float32, tf.float16])
@pytest.mark.parametrize("n_basis", [3, 4, 5])
@pytest.mark.parametrize("batch", [1, 3, 8, 24, 32, 48, 64, 128])
def test_backward_matches_float64_at_every_shape(batch, n_basis, dtype):
    """Every batch size, basis dimension and dtype runs the sliced backward.

    Against an FP64 reference built from the same (dtype-rounded) inputs: the
    weight gradient is rebuilt in FP32 and must be near FP32 round-off; the FP16
    spike gradient goes through a scaled FP16 projection, so it is held to the
    2e-3 relative-Frobenius gate the kernel was accepted under.
    """
    (indices, synapse_types, spikes, weights, basis, upstream,
     n_pre, n_post) = _dense_fixture(batch, n_basis, 90210 + batch + n_basis)
    np_dtype = np.float16 if dtype == tf.float16 else np.float32
    spikes = spikes.astype(np_dtype).astype(np.float64)
    upstream = upstream.astype(np_dtype).astype(np.float64)
    connectivity = _connectivity(indices, synapse_types, n_pre, n_post)
    spikes_tensor = tf.constant(spikes, dtype)
    master = tf.Variable(weights)
    with tf.GradientTape() as tape:
        tape.watch(spikes_tensor)
        currents = calculate_recurrent_csr_currents(
            spikes_tensor, master, tf.constant(basis), 0.5, connectivity
        )
        loss = tf.reduce_sum(tf.cast(currents, tf.float32) * tf.constant(upstream, tf.float32))
    spike_grad, weight_grad = tape.gradient(loss, (spikes_tensor, master))

    projected = np.sum(   # [batch, edge]
        upstream.reshape(batch, n_post, n_basis)[:, indices[:, 0]]
        * basis[synapse_types], axis=-1,
    )
    expected_weight = np.einsum("be,be->e", spikes[:, indices[:, 1]], projected)
    expected_spike = np.zeros((batch, n_pre))
    np.add.at(expected_spike.T, indices[:, 1], (0.5 * weights * projected).T)
    expected_currents = _float64_forward(indices, synapse_types, spikes, weights, basis, n_post)

    fp16 = dtype == tf.float16
    assert _relative_error(currents, expected_currents) < (2e-3 if fp16 else 1e-6)
    assert _relative_error(spike_grad, expected_spike) < (2e-3 if fp16 else 1e-6)
    assert _relative_error(weight_grad, expected_weight) < 1e-6


@pytest.mark.skipif(not tf.config.list_physical_devices("GPU"), reason="CUDA GPU required")
@pytest.mark.parametrize("resource_mode", ["0", "1"])
@pytest.mark.parametrize("dtype", [tf.float32, tf.float16])
@pytest.mark.parametrize("n_basis", [4, 5])
@pytest.mark.parametrize("batch", [1, 3, 32, 48])
def test_accumulating_backward_adds_the_weight_gradient_in_place(
    batch, n_basis, dtype, resource_mode, monkeypatch
):
    """Inside the accumulation block the weight gradient lands in the variable.

    It is added to what the variable held, one addend per edge, so the result
    is bitwise the old value plus the dense weight gradient; the spike gradient
    is unchanged and the weights get no gradient of their own. A live read of
    the old value must survive the update, so the op copies on write.
    """
    monkeypatch.setenv("V1_CSR_RESOURCE_MODE", resource_mode)
    (indices, synapse_types, spikes, weights, basis, upstream,
     n_pre, n_post) = _dense_fixture(batch, n_basis, 4242 + batch + n_basis)
    connectivity = _connectivity(indices, synapse_types, n_pre, n_post)
    assert (connectivity.resource_name is not None) == (resource_mode == "1")
    spikes_tensor = tf.constant(spikes, dtype)
    master = tf.Variable(weights)

    def gradients():
        with tf.GradientTape() as tape:
            tape.watch(spikes_tensor)
            currents = calculate_recurrent_csr_currents(
                spikes_tensor, master, tf.constant(basis), 0.5, connectivity
            )
            loss = tf.reduce_sum(tf.cast(currents, tf.float32) * tf.constant(upstream))
        return tape.gradient(loss, (spikes_tensor, master))

    spike_grad, weight_grad = gradients()
    offset = np.random.default_rng(batch).normal(size=weights.shape).astype(np.float32)
    accumulator = tf.Variable(offset)
    live_read = accumulator.read_value()
    with accumulate_recurrent_weight_gradient(accumulator.handle):
        accumulated_spike_grad, no_weight_grad = gradients()

    assert no_weight_grad is None
    np.testing.assert_array_equal(accumulated_spike_grad.numpy(), spike_grad.numpy())
    np.testing.assert_array_equal(live_read.numpy(), offset)
    np.testing.assert_array_equal(accumulator.numpy(), offset + weight_grad.numpy())


@pytest.mark.skipif(not tf.config.list_physical_devices("GPU"), reason="CUDA GPU required")
def test_accumulating_backward_rejects_a_mismatched_accumulator():
    indices, synapse_types, spikes, weights, basis, upstream = _fixture(2, 4)
    connectivity = _connectivity(indices, synapse_types, 4, 3)
    spikes_tensor = tf.constant(spikes)
    accumulator = tf.Variable(tf.zeros(weights.size + 1))
    with tf.GradientTape() as tape:
        tape.watch(spikes_tensor)
        currents = calculate_recurrent_csr_currents(
            spikes_tensor, tf.constant(weights), tf.constant(basis), 0.5, connectivity
        )
    with accumulate_recurrent_weight_gradient(accumulator.handle):
        with pytest.raises(tf.errors.InvalidArgumentError, match="accumulator"):
            tape.gradient(currents, spikes_tensor, output_gradients=tf.constant(upstream))


@pytest.mark.skipif(not tf.config.list_physical_devices("GPU"), reason="CUDA GPU required")
@pytest.mark.parametrize("n_basis", [4, 5])
@pytest.mark.parametrize("batch_size", [1, 8, 32, 3])
def test_initial_accumulates_into_the_currents(batch_size, n_basis):
    """Seeding `initial` must equal adding it afterwards, gradients included."""
    indices, synapse_types, spikes, weights, basis, upstream = _fixture(batch_size, n_basis)
    connectivity = _connectivity(indices, synapse_types, n_pre=4, n_post=3)
    rng = np.random.default_rng(4242)
    seed = rng.normal(size=(batch_size * 3, n_basis)).astype(np.float32)

    def run(use_initial):
        spikes_tensor = tf.constant(spikes)
        master = tf.Variable(weights)
        seed_tensor = tf.Variable(seed)
        with tf.GradientTape() as tape:
            tape.watch(spikes_tensor)
            currents = calculate_recurrent_csr_currents(
                spikes_tensor, master, tf.constant(basis), 0.37, connectivity,
                initial=seed_tensor if use_initial else None,
            )
            if not use_initial:
                currents = currents + seed_tensor
            loss = tf.reduce_sum(currents * upstream)
        grads = tape.gradient(loss, (spikes_tensor, master, seed_tensor))
        return [np.asarray(currents)] + [np.asarray(g) for g in grads]

    for got, want, name in zip(run(True), run(False),
                               ("currents", "spike_grad", "weight_grad", "initial_grad")):
        np.testing.assert_allclose(got, want, rtol=2e-5, atol=2e-5, err_msg=name)


@pytest.mark.skipif(not tf.config.list_physical_devices("GPU"), reason="CUDA GPU required")
def test_initial_must_match_the_output_shape():
    indices, synapse_types, spikes, weights, basis, _ = _fixture(8, 4)
    connectivity = _connectivity(indices, synapse_types, n_pre=4, n_post=3)
    with pytest.raises(tf.errors.InvalidArgumentError, match="initial"):
        calculate_recurrent_csr_currents(
            tf.constant(spikes), tf.Variable(weights), tf.constant(basis),
            0.37, connectivity, initial=tf.zeros((5, 4)),
        ).numpy()


@pytest.mark.skipif(not tf.config.list_physical_devices("GPU"), reason="CUDA GPU required")
@pytest.mark.parametrize("use_initial", [False, True])
def test_graph_mode_gradients_with_and_without_an_accumulator(use_initial):
    """Graph mode validates gradient shapes against inputs.

    Without an accumulator the `initial` input is an empty sentinel, so its
    gradient must stay unset; returning an output-shaped gradient there fails
    only once the op is traced inside a tf.function.
    """
    indices, synapse_types, spikes, weights, basis, upstream = _fixture(32, 4)
    connectivity = _connectivity(indices, synapse_types, n_pre=4, n_post=3)
    master = tf.Variable(weights)
    seed = tf.Variable(np.zeros((32 * 3, 4), np.float32))

    @tf.function
    def step(spike_values):
        with tf.GradientTape() as tape:
            tape.watch(spike_values)
            currents = calculate_recurrent_csr_currents(
                spike_values, master, tf.constant(basis), 0.37, connectivity,
                initial=seed if use_initial else None,
            )
            loss = tf.reduce_sum(currents * upstream)
        return tape.gradient(loss, (spike_values, master))

    spike_grad, weight_grad = step(tf.constant(spikes))
    assert np.isfinite(np.asarray(spike_grad)).all()
    assert np.isfinite(np.asarray(weight_grad)).all()


def _float64_forward(indices, synapse_types, spikes, weights, basis, n_post):
    currents = np.zeros((spikes.shape[0], n_post, basis.shape[1]))
    for (post, pre), synapse_type, weight in zip(indices, synapse_types, weights):
        currents[:, post] += np.outer(spikes[:, pre], weight * basis[synapse_type])
    return currents.reshape(-1, basis.shape[1])


@pytest.mark.skipif(not tf.config.list_physical_devices("GPU"), reason="CUDA GPU required")
@pytest.mark.parametrize("aggregate", [True, False])
@pytest.mark.parametrize("dtype", [tf.float16, tf.float32])
@pytest.mark.parametrize("n_basis", [3, 4, 5])
@pytest.mark.parametrize("use_initial", [False, True])
def test_forward_aggregates_same_post_runs(use_initial, n_basis, dtype, aggregate):
    """Runs of edges onto one postsynaptic neuron are summed before the atomic.

    Each row sends several synapse types to the same targets, as LGN rows do, and
    is long enough to span several warps, so runs cross warp boundaries too.
    """
    rng = np.random.default_rng(7)
    n_pre, n_post, n_types, batch = 6, 40, 5, 32
    posts = np.repeat(np.arange(n_post), 3)
    indices = np.concatenate(
        [np.stack((posts, np.full_like(posts, pre)), axis=1) for pre in range(n_pre)]
    ).astype(np.int64)
    synapse_types = rng.integers(0, n_types, indices.shape[0]).astype(np.int64)
    weights = rng.normal(size=indices.shape[0]).astype(np.float32)
    indices, synapse_types, weights = _csr_sorted(indices, synapse_types, weights)
    np_dtype = np.float16 if dtype == tf.float16 else np.float32
    spikes = (rng.random((batch, n_pre)) < 0.5).astype(np_dtype)
    basis = rng.normal(size=(n_types, n_basis)).astype(np.float32)
    seed = rng.normal(size=(batch * n_post, n_basis)).astype(np_dtype)
    connectivity = _connectivity(indices, synapse_types, n_pre, n_post)
    assert connectivity.repeats_targets
    # Both scatter paths are exact; the flag only chooses the faster one.
    connectivity = dataclasses.replace(connectivity, repeats_targets=aggregate)
    currents = calculate_recurrent_csr_currents(
        tf.constant(spikes), tf.Variable(weights), tf.constant(basis), 0.37,
        connectivity, initial=tf.constant(seed) if use_initial else None,
    )
    expected = _float64_forward(
        indices, synapse_types, spikes.astype(np.float64), weights, basis, n_post
    ) + (seed if use_initial else 0.0)
    if dtype == tf.float16:
        # FP16 atomics round every partial sum, and partial sums here reach ~8
        # before cancelling, so single elements are off by a few FP16 steps at
        # that magnitude, in an order-dependent place. Judge the whole buffer.
        assert _relative_error(currents, expected) < 2e-3
    else:
        np.testing.assert_allclose(np.asarray(currents, np.float64), expected,
                                   rtol=1e-5, atol=1e-5)


@pytest.mark.skipif(not tf.config.list_physical_devices("GPU"), reason="CUDA GPU required")
def test_forward_basis_reaches_the_kernel_in_fp32():
    """A basis value FP16 cannot represent must not be rounded on the way in."""
    indices, synapse_types, spikes, weights, _, _ = _fixture(32, 4)
    basis = np.full((3, 4), 1.0 + 2.0 ** -12, np.float32)   # FP16 rounds it to 1
    connectivity = _connectivity(indices, synapse_types, n_pre=4, n_post=3)
    currents = calculate_recurrent_csr_currents(
        tf.constant(spikes), tf.Variable(weights), tf.constant(basis), 0.37,
        connectivity,
    )
    expected = _float64_forward(indices, synapse_types, spikes, weights, basis, 3)
    np.testing.assert_allclose(currents.numpy(), expected, rtol=1e-6, atol=1e-6)


@pytest.mark.skipif(not tf.config.list_physical_devices("GPU"), reason="CUDA GPU required")
@pytest.mark.parametrize("batch", [8, 32, 48, 64])
def test_event_weight_gradient_skips_silent_rows(batch):
    """The FP16 small-batch backward against an FP64 reference.

    Its weight gradient is event driven: rows that never fire get exactly zero,
    and firing rows are rebuilt in FP32 from the upstream gradient. Rows without
    edges must still receive a zero spike gradient.
    """
    rng = np.random.default_rng(314)
    n_pre, n_post, n_types, n_edges = 64, 80, 6, 4000
    wired = np.arange(0, n_pre, 2)   # odd rows have no edges
    indices = np.stack(
        (rng.integers(0, n_post, n_edges), rng.choice(wired, n_edges)), axis=1
    ).astype(np.int64)
    synapse_types = rng.integers(0, n_types, n_edges).astype(np.int64)
    weights = rng.normal(size=n_edges).astype(np.float32)
    indices, synapse_types, weights = _csr_sorted(indices, synapse_types, weights)
    spikes = (rng.random((batch, n_pre)) < 0.3).astype(np.float16)
    spikes[:, wired[::3]] = 0.0   # a third of the wired rows never fire
    basis = rng.normal(size=(n_types, 4)).astype(np.float32)
    upstream = rng.normal(size=(batch * n_post, 4)).astype(np.float16)
    connectivity = _connectivity(indices, synapse_types, n_pre, n_post)
    spikes_tensor = tf.constant(spikes)
    master = tf.Variable(weights)
    with tf.GradientTape() as tape:
        tape.watch(spikes_tensor)
        currents = calculate_recurrent_csr_currents(
            spikes_tensor, master, tf.constant(basis), 0.37, connectivity
        )
        loss = tf.reduce_sum(tf.cast(currents, tf.float32) * upstream)
    spike_grad, weight_grad = tape.gradient(loss, (spikes_tensor, master))
    weight_grad = weight_grad.numpy()

    projected = np.sum(   # [batch, edge]
        upstream.astype(np.float64).reshape(batch, n_post, 4)[:, indices[:, 0]]
        * basis[synapse_types], axis=-1,
    )
    expected_weight = np.einsum("be,be->e", spikes[:, indices[:, 1]], projected)
    expected_spike = np.zeros((batch, n_pre))
    np.add.at(expected_spike.T, indices[:, 1], (0.37 * weights * projected).T)

    silent = np.isin(indices[:, 1], wired[::3])
    assert np.all(weight_grad[silent] == 0.0)
    np.testing.assert_allclose(weight_grad, expected_weight, rtol=1e-5, atol=1e-4)
    np.testing.assert_allclose(np.asarray(spike_grad, np.float64), expected_spike,
                               rtol=5e-3, atol=5e-3)
    assert np.all(np.asarray(spike_grad)[:, 1::2] == 0.0)


@pytest.mark.skipif(not tf.config.list_physical_devices("GPU"), reason="CUDA GPU required")
def test_forward_handles_rows_beyond_the_old_queue_encoding():
    n_pre = 2 ** 21 + 1
    connectivity = _connectivity(
        np.array([[0, n_pre - 1]], np.int64), np.array([0], np.int64), n_pre, 1
    )
    spikes = tf.scatter_nd([[1, n_pre - 1]], [1.0], (2, n_pre))
    currents = calculate_recurrent_csr_currents(
        spikes, tf.Variable([2.0]), tf.ones((1, 4)), 0.37, connectivity,
    ).numpy()
    np.testing.assert_array_equal(currents[0], 0.0)
    np.testing.assert_array_equal(currents[1], 2.0)


@pytest.mark.skipif(not tf.config.list_physical_devices("GPU"), reason="CUDA GPU required")
@pytest.mark.parametrize("resource_mode", ["0", "1"])
def test_validation_graphs_share_forward_tile_memory(monkeypatch, resource_mode):
    """Two unrolled validation graphs must not retain forty CSR tile tables."""
    monkeypatch.setenv("V1_CSR_RESOURCE_MODE", resource_mode)
    n_pre, n_post = 32768, 8192
    indices = np.column_stack((
        np.tile([0, 2048, 4096, 6144], n_pre),
        np.repeat(np.arange(n_pre), 4),
    ))
    with tf.device("/GPU:0"):
        connectivity = _connectivity(
            indices, np.zeros(len(indices), np.int64), n_pre, n_post
        )
        weights = tf.Variable(tf.fill((len(indices),), 1.0 / n_pre))
        basis = tf.ones((1, 4))
        spikes = tf.ones((1, n_pre))

        def make_rollout(chunks):
            @tf.function
            def rollout(activity):
                for _ in range(chunks):
                    currents = calculate_recurrent_csr_currents(
                        activity, weights, basis, 0.1, connectivity
                    )
                    activity = activity + tf.reduce_sum(currents) * 1e-5
                return tf.reduce_mean(activity)

            return rollout

        warmup = make_rollout(1)
        warmup(spikes).numpy()
        before = tf.config.experimental.get_memory_info("GPU:0")["current"]
        gray, evoked = make_rollout(20), make_rollout(20)
        for rollout in (gray, evoked):
            np.testing.assert_allclose(
                rollout(spikes).numpy(), (1.0 + 16e-5) ** 20, rtol=2e-5
            )
        after = tf.config.experimental.get_memory_info("GPU:0")["current"]
        assert after - before < 8 * 2**20, (
            f"Validation retained {(after - before) / 2**20:.2f} MiB of extra "
            "GPU memory for unchanged connectivity"
        )


@pytest.mark.skipif(not tf.config.list_physical_devices("GPU"), reason="CUDA GPU required")
def test_shared_forward_tables_survive_connectivity_changes(monkeypatch):
    """Changing one operator's CSR must not mutate another operator's table."""
    monkeypatch.setenv("V1_CSR_RESOURCE_MODE", "0")
    n_post = 8192
    with tf.device("/GPU:0"):
        connectivity = _connectivity(
            np.array([[0, 0], [6144, 1]], np.int64),
            np.zeros(2, np.int64), 2, n_post,
        )
        weights, basis = tf.Variable([1.0, 1.0]), tf.ones((1, 4))
        changed_posts = tf.constant([6144, 0], tf.uint32)

        @tf.function
        def original(activity):
            return calculate_recurrent_csr_currents(
                activity, weights, basis, 0.1, connectivity
            )

        @tf.function
        def changing(activity, posts):
            return calculate_recurrent_csr_currents(
                activity, weights, basis, 0.1,
                dataclasses.replace(connectivity, post_ids=posts),
            )

        # FP16 and FP32 share metadata but configure distinct CUDA kernels.
        for dtype in (tf.float32, tf.float16):
            activity = tf.constant([[1.0, 2.0]], dtype)
            expected = np.zeros((n_post, 4), np.float32)
            expected[0], expected[6144] = 1.0, 2.0
            changed_expected = np.zeros_like(expected)
            changed_expected[0], changed_expected[6144] = 2.0, 1.0
            np.testing.assert_array_equal(original(activity).numpy(), expected)
            for posts, target in (
                (connectivity.post_ids, expected),
                (changed_posts, changed_expected),
                (connectivity.post_ids, expected),
            ):
                np.testing.assert_array_equal(changing(activity, posts).numpy(), target)
                np.testing.assert_array_equal(original(activity).numpy(), expected)


@pytest.mark.skipif(not tf.config.list_physical_devices("GPU"), reason="CUDA GPU required")
@pytest.mark.parametrize("resource_mode", ["0", "1"])
def test_shared_forward_tables_respect_output_size(monkeypatch, resource_mode):
    """The same CSR can need masks or binary search for different output sizes."""
    monkeypatch.setenv("V1_CSR_RESOURCE_MODE", resource_mode)
    with tf.device("/GPU:0"):
        connectivity = _connectivity(
            np.array([[0, 0]], np.int64), np.zeros(1, np.int64), 1, 8192
        )
        weights, basis = tf.Variable([2.0]), tf.ones((1, 4))
        activity = tf.ones((1, 1))
        runs = []
        def make_run(csr):
            @tf.function
            def run(spikes):
                return calculate_recurrent_csr_currents(
                    spikes, weights, basis, 0.1, csr
                )

            return run

        # Above 64 tiles of the default 5,120 targets, the table has no masks.
        for n_post in (8192, 327681):
            run = make_run(dataclasses.replace(connectivity, n_post=n_post))
            runs.append(run)
            expected = np.zeros((n_post, 4), np.float32)
            expected[0] = 2.0
            np.testing.assert_array_equal(run(activity).numpy(), expected)
        np.testing.assert_array_equal(runs[0](activity).numpy()[0], 2.0)
