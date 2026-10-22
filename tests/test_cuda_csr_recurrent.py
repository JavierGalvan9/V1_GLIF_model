import dataclasses
import functools

import numpy as np
import pytest
import tensorflow as tf

from v1_model_utils.cuda_csr_recurrent import (
    DIRECT_CSR,
    accumulate_recurrent_weight_gradient,
    build_csr_connectivity,
    calculate_recurrent_csr_currents,
    recurrent_backward,
    spike_activity_masks,
    spike_queue,
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


def _dense_fixture(batch, n_basis, seed, n_pre=48):
    """Several hundred edges per row, so a slicing or reduction error cannot hide."""
    rng = np.random.default_rng(seed)
    n_post, n_types, n_edges = 96, 7, 6000
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
    got, want = np.asarray(got, np.float64), np.asarray(want, np.float64)
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


def test_edge_arrays_of_the_wrong_length_are_rejected():
    """The kernels index edge arrays unchecked: a short weight vector would
    read (and its gradient write) out of bounds instead of failing."""
    indices, synapse_types, spikes, weights, basis, _ = _fixture(2, 4)
    connectivity = _connectivity(indices, synapse_types, n_pre=4, n_post=3)
    with pytest.raises(tf.errors.InvalidArgumentError, match="one entry per edge"):
        calculate_recurrent_csr_currents(
            tf.constant(spikes), tf.Variable(weights[:-1]), tf.constant(basis),
            0.37, connectivity,
        ).numpy()


def test_basis_must_cover_every_synapse_type():
    indices, synapse_types, spikes, weights, basis, _ = _fixture(2, 4)
    connectivity = _connectivity(indices, synapse_types, n_pre=4, n_post=3)
    with pytest.raises(ValueError, match="synapse types"):
        calculate_recurrent_csr_currents(
            tf.constant(spikes), tf.Variable(weights), tf.constant(basis[:2]),
            0.37, connectivity,
        )


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
    edge_vectors = np.asarray(weights, np.float64)[:, None] * basis[synapse_types]
    currents = np.zeros((spikes.shape[0], n_post, basis.shape[1]))
    for row, drive in zip(currents, np.asarray(spikes, np.float64)[:, indices[:, 1]]):
        for column, values in enumerate((drive[:, None] * edge_vectors).T):
            row[:, column] = np.bincount(indices[:, 0], values, minlength=n_post)
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


@functools.lru_cache(maxsize=None)
def _unique_edges(n_pre, n_post, n_edges, n_types=5):
    """Random distinct (post, pre) edges in CSR order, with their weights."""
    rng = np.random.default_rng(n_post)
    flat = np.unique(rng.integers(0, n_pre * n_post, n_edges))
    indices = np.stack((flat % n_post, flat // n_post), axis=1)
    synapse_types = rng.integers(0, n_types, flat.size)
    weights = rng.normal(size=flat.size).astype(np.float32)
    return _csr_sorted(indices, synapse_types, weights)


@pytest.mark.skipif(not tf.config.list_physical_devices("GPU"), reason="CUDA GPU required")
@pytest.mark.parametrize("use_initial", [False, True])
@pytest.mark.parametrize("dtype", [tf.float32, tf.float16])
@pytest.mark.parametrize("batch", [1, 5, 33])
@pytest.mark.parametrize("n_post, n_edges", [(12_000, 400_000), (330_000, 1_500_000)])
def test_tiled_forward_matches_float64(n_post, n_edges, batch, dtype, use_initial):
    """Targets spanning 3 tiles (masked table) and over 64 (binary search).

    The last sample of a batch never fires, so its rows hold exactly `initial`.
    """
    n_pre = 4000
    indices, synapse_types, weights = _unique_edges(n_pre, n_post, n_edges)
    connectivity = _connectivity(indices, synapse_types, n_pre, n_post)
    rng = np.random.default_rng(batch)
    spikes = (rng.random((batch, n_pre)) < 0.2).astype(dtype.as_numpy_dtype)
    if batch > 1:
        spikes[-1] = 0.0
    basis = rng.normal(size=(5, 4)).astype(np.float32)
    seed = rng.normal(size=(batch * n_post, 4)).astype(dtype.as_numpy_dtype)
    currents = calculate_recurrent_csr_currents(
        tf.constant(spikes), tf.Variable(weights), tf.constant(basis), 0.37,
        connectivity, initial=tf.constant(seed) if use_initial else None,
    ).numpy()
    seed = seed if use_initial else np.zeros_like(seed)
    expected = _float64_forward(indices, synapse_types, spikes, weights, basis, n_post)
    assert _relative_error(currents, expected + seed) < (
        2e-3 if dtype == tf.float16 else 1e-6
    )
    if batch > 1:
        np.testing.assert_array_equal(currents[-n_post:], seed[-n_post:])


@pytest.mark.skipif(not tf.config.list_physical_devices("GPU"), reason="CUDA GPU required")
@pytest.mark.parametrize("dtype", [tf.float32, tf.float16])
@pytest.mark.parametrize("n_post, n_edges", [(12_000, 400_000), (330_000, 1_500_000)])
def test_silent_batch_writes_every_output(n_post, n_edges, dtype):
    """With no active entry no edge is swept; the output must still be zeroed,
    not left as whatever the allocator hands back (pre-filled with junk here)."""
    n_pre, batch = 4000, 5
    indices, synapse_types, weights = _unique_edges(n_pre, n_post, n_edges)
    connectivity = _connectivity(indices, synapse_types, n_pre, n_post)
    junk = tf.fill((batch * n_post, 4), tf.constant(7.0, dtype))
    del junk
    currents = calculate_recurrent_csr_currents(
        tf.zeros((batch, n_pre), dtype), tf.Variable(weights), tf.ones((5, 4)), 0.37,
        connectivity,
    )
    np.testing.assert_array_equal(currents.numpy(), 0.0)


@pytest.mark.skipif(not tf.config.list_physical_devices("GPU"), reason="CUDA GPU required")
@pytest.mark.parametrize("dtype", [tf.float32, tf.float16])
@pytest.mark.parametrize("batch", [1, 5, 32])
@pytest.mark.parametrize("n_slots", range(2, 8))
def test_spike_slots_match_the_concatenated_history(n_slots, batch, dtype):
    """A history passed as its delay slots is the concatenated op, slot by slot.

    Each slot gets its own share of the spike gradient, and the in-place weight
    gradient accumulator sees the same gradient as the single-slot call.
    """
    (indices, synapse_types, spikes, weights, basis, upstream,
     n_pre, n_post) = _dense_fixture(batch, 4, 77 * n_slots + batch, n_pre=48 * n_slots)
    connectivity = _connectivity(indices, synapse_types, n_pre, n_post)
    master = tf.Variable(weights)

    def gradients(history):
        with tf.GradientTape() as tape:
            tape.watch(history)
            currents = calculate_recurrent_csr_currents(
                history, master, tf.constant(basis), 0.5, connectivity
            )
            loss = tf.reduce_sum(tf.cast(currents, tf.float32) * upstream)
        return (currents, *tape.gradient(loss, (history, master)))

    history = tf.constant(spikes, dtype)
    slots = tuple(tf.split(history, n_slots, axis=1))
    currents, spike_grad, weight_grad = gradients(history)
    slot_currents, slot_grads, slot_weight_grad = gradients(slots)
    assert len(slot_grads) == n_slots
    # Atomics make neither path bitwise reproducible; FP16 rounds each partial sum.
    tolerance = 2e-3 if dtype == tf.float16 else 1e-6
    assert _relative_error(slot_currents, currents.numpy()) < tolerance
    assert _relative_error(tf.concat(slot_grads, axis=1), spike_grad.numpy()) < tolerance
    assert _relative_error(slot_weight_grad, weight_grad.numpy()) < 1e-6

    offset = np.random.default_rng(n_slots).normal(size=weights.shape).astype(np.float32)
    accumulator = tf.Variable(offset)
    with accumulate_recurrent_weight_gradient(accumulator.handle):
        _, accumulated_grads, no_weight_grad = gradients(slots)
    assert no_weight_grad is None
    assert _relative_error(tf.concat(accumulated_grads, axis=1), spike_grad.numpy()) < tolerance
    assert _relative_error(accumulator.numpy() - offset, weight_grad.numpy()) < 1e-6


@pytest.mark.skipif(not tf.config.list_physical_devices("GPU"), reason="CUDA GPU required")
@pytest.mark.parametrize("dtype", [tf.float32, tf.float16])
@pytest.mark.parametrize("batch", [1, 33, 64, 65, 130])
@pytest.mark.parametrize("n_slots", [1, 3])
def test_activity_masks_give_bitwise_the_swept_backward(n_slots, batch, dtype):
    """Carried activity masks replace the weight gradient's sweep over the slots.

    The masks hold bit j of word w of column c set exactly when sample 32 w + j
    of that column is non-zero, and both backward ops return bitwise the same
    gradients with them as without.
    """
    (indices, synapse_types, spikes, weights, basis, upstream,
     n_pre, n_post) = _dense_fixture(batch, 4, 13 * batch + n_slots, n_pre=48 * n_slots)
    spikes[:, ::7] = -0.0  # signed zeros are inactive too
    c = _connectivity(indices, synapse_types, n_pre, n_post)
    slots = tuple(tf.split(tf.constant(spikes, dtype), n_slots, axis=1))
    masks = [spike_activity_masks(slot) for slot in slots]
    words = (batch + 31) // 32
    for slot, mask in zip(slots, masks):
        assert mask.dtype == tf.uint32 and mask.shape == (slot.shape[1], words)
        bits = (mask.numpy()[:, :, None] >> np.arange(32, dtype=np.uint32)) & 1
        active = np.zeros((slot.shape[1], 32 * words), bool)
        active[:, :batch] = slot.numpy().T != 0
        np.testing.assert_array_equal(bits.reshape(slot.shape[1], -1).astype(bool), active)

    def backward(activity_masks):
        return recurrent_backward(
            c, slots, tf.constant(upstream, dtype), tf.constant(weights), tf.constant(basis),
            0.5, c.post_ids, c.synapse_types, c.row_splits, c.edge_ids, c.nonempty_rows,
            c.pair_ids, c.pair_posts, c.pair_types, activity_masks=activity_masks,
        )

    swept_grads, swept_weight_grad = backward(())
    masked_grads, masked_weight_grad = backward(masks)
    for swept, masked in zip(swept_grads, masked_grads):
        np.testing.assert_array_equal(masked.numpy().view(np.uint8), swept.numpy().view(np.uint8))
    np.testing.assert_array_equal(masked_weight_grad.numpy().view(np.uint8),
                                  swept_weight_grad.numpy().view(np.uint8))
    offset = np.random.default_rng(batch).normal(size=weights.shape).astype(np.float32)
    accumulator = tf.Variable(offset)
    with accumulate_recurrent_weight_gradient(accumulator.handle):
        accumulated_grads, no_weight_grad = backward(masks)
    assert no_weight_grad is None
    for swept, accumulated in zip(swept_grads, accumulated_grads):
        np.testing.assert_array_equal(accumulated.numpy().view(np.uint8),
                                      swept.numpy().view(np.uint8))
    np.testing.assert_array_equal(accumulator.numpy(), offset + swept_weight_grad.numpy())
    with pytest.raises(tf.errors.InvalidArgumentError, match="activity_masks"):
        backward(masks[:-1] + [masks[-1][:-1]])


@pytest.mark.skipif(not tf.config.list_physical_devices("GPU"), reason="CUDA GPU required")
@pytest.mark.parametrize("dtype", [tf.float32, tf.float16])
@pytest.mark.parametrize("batch", [1, 64, 65])
def test_abs_max_bound_gives_bitwise_the_measured_backward(batch, dtype):
    """A precomputed bound (float bits of the largest finite |current_grad|, as
    the GLIF backward's emit_abs_max returns it) replaces the operator's own
    pass and leaves both backward ops' outputs bitwise unchanged."""
    (indices, synapse_types, spikes, weights, basis, upstream,
     n_pre, n_post) = _dense_fixture(batch, 4, 29 * batch + 3, n_pre=96)
    upstream[0, 0] = np.inf  # non-finite entries are left out of the bound
    c = _connectivity(indices, synapse_types, n_pre, n_post)
    slots = tuple(tf.split(tf.constant(spikes, dtype), 3, axis=1))
    current_grad = tf.constant(upstream, dtype)
    magnitudes = np.abs(current_grad.numpy().astype(np.float32))
    bound = tf.constant(
        [magnitudes[np.isfinite(magnitudes)].max().view(np.uint32)], tf.uint32
    )

    def backward(abs_max_bound):
        return recurrent_backward(
            c, slots, current_grad, tf.constant(weights), tf.constant(basis),
            0.5, c.post_ids, c.synapse_types, c.row_splits, c.edge_ids, c.nonempty_rows,
            c.pair_ids, c.pair_posts, c.pair_types, abs_max_bound=abs_max_bound,
        )

    measured_grads, measured_weight_grad = backward(None)
    bounded_grads, bounded_weight_grad = backward(bound)
    for measured, bounded in zip(measured_grads, bounded_grads):
        np.testing.assert_array_equal(bounded.numpy().view(np.uint8),
                                      measured.numpy().view(np.uint8))
    np.testing.assert_array_equal(bounded_weight_grad.numpy().view(np.uint8),
                                  measured_weight_grad.numpy().view(np.uint8))
    accumulator = tf.Variable(tf.zeros_like(measured_weight_grad))
    with accumulate_recurrent_weight_gradient(accumulator.handle):
        accumulated_grads, _ = backward(bound)
    for measured, accumulated in zip(measured_grads, accumulated_grads):
        np.testing.assert_array_equal(accumulated.numpy().view(np.uint8),
                                      measured.numpy().view(np.uint8))
    np.testing.assert_array_equal(accumulator.numpy(), measured_weight_grad.numpy())
    with pytest.raises(tf.errors.InvalidArgumentError, match="abs_max_bound"):
        backward(tf.concat([bound, bound], 0))


@pytest.mark.skipif(not tf.config.list_physical_devices("GPU"), reason="CUDA GPU required")
@pytest.mark.parametrize("dtype", [tf.float32, tf.float16])
@pytest.mark.parametrize("batch", [1, 64, 65])
def test_spike_grad_addends_equal_a_following_add(batch, dtype):
    """Per-slot addends (none for the oldest slot) give bitwise what adding them
    to the returned gradients gives, in both backward ops, signed zeros included."""
    (indices, synapse_types, spikes, weights, basis, upstream,
     n_pre, n_post) = _dense_fixture(batch, 4, 31 * batch + 5, n_pre=96)
    c = _connectivity(indices, synapse_types, n_pre, n_post)
    slots = tuple(tf.split(tf.constant(spikes, dtype), 3, axis=1))
    rng = np.random.default_rng(batch)
    addends = []
    for slot in slots[:2]:
        values = rng.normal(size=slot.shape).astype(np.float32)
        values[rng.random(slot.shape) < 0.2] = -0.0
        addends.append(tf.constant(values, dtype))
    addends.append(tf.zeros([0], dtype))

    def backward(spike_grad_addends):
        return recurrent_backward(
            c, slots, tf.constant(upstream, dtype), tf.constant(weights), tf.constant(basis),
            0.5, c.post_ids, c.synapse_types, c.row_splits, c.edge_ids, c.nonempty_rows,
            c.pair_ids, c.pair_posts, c.pair_types, spike_grad_addends=spike_grad_addends,
        )

    plain_grads, plain_weight_grad = backward(())
    expected = [plain + addend for plain, addend in zip(plain_grads[:2], addends[:2])]
    expected.append(plain_grads[2])
    added_grads, added_weight_grad = backward(addends)
    for want, got in zip(expected, added_grads):
        np.testing.assert_array_equal(got.numpy().view(np.uint8), want.numpy().view(np.uint8))
    np.testing.assert_array_equal(added_weight_grad.numpy(), plain_weight_grad.numpy())
    accumulator = tf.Variable(tf.zeros_like(plain_weight_grad))
    with accumulate_recurrent_weight_gradient(accumulator.handle):
        accumulated_grads, _ = backward(addends)
    for want, got in zip(expected, accumulated_grads):
        np.testing.assert_array_equal(got.numpy().view(np.uint8), want.numpy().view(np.uint8))
    with pytest.raises(tf.errors.InvalidArgumentError, match="spike_grad_addends"):
        backward(addends[:2] + [addends[0][:, :-1]])


@pytest.mark.skipif(not tf.config.list_physical_devices("GPU"), reason="CUDA GPU required")
@pytest.mark.parametrize("resource_mode", ["0", "1"])
@pytest.mark.parametrize("dtype", [tf.float32, tf.float16])
def test_carried_queue_records_match_sweeping_every_slot(dtype, resource_mode, monkeypatch):
    """Carry records step to step as the model loop does: each step's returned
    record becomes the next step's queue of slot 1, and the oldest is dropped."""
    monkeypatch.setenv("V1_CSR_RESOURCE_MODE", resource_mode)
    n_slots, width, n_post, batch, steps = 3, 2000, 12_000, 3, 10
    indices, synapse_types, weights = _unique_edges(n_slots * width, n_post, 80_000)
    connectivity = _connectivity(indices, synapse_types, n_slots * width, n_post)
    master = tf.Variable(weights)
    basis = tf.constant(np.random.default_rng(0).normal(size=(5, 4)), tf.float32)
    rng = np.random.default_rng(1)
    sequence = tf.constant(
        (rng.random((steps + n_slots, batch, width)) < 0.2).astype(dtype.as_numpy_dtype)
    )

    @tf.function
    def rollout(carry):
        slots = tuple(sequence[n_slots - 1 - slot] for slot in range(n_slots))
        queues = tuple(spike_queue(slot) for slot in slots[1:])
        outputs = tf.TensorArray(dtype, size=steps)
        for step in tf.range(steps):
            if carry:
                currents, record = calculate_recurrent_csr_currents(
                    slots, master, basis, 0.5, connectivity, queues=queues
                )
                queues = (record,) + queues[:-1]
            else:
                currents = calculate_recurrent_csr_currents(
                    slots, master, basis, 0.5, connectivity
                )
            outputs = outputs.write(step, currents)
            slots = (sequence[step + n_slots],) + slots[:-1]
        return outputs.stack()

    carried, swept = rollout(True).numpy(), rollout(False).numpy()
    assert np.any(swept != 0.0)
    # Atomics reorder the sums between runs, so match to round-off, not bitwise.
    assert _relative_error(carried, swept) < (1e-3 if dtype == tf.float16 else 1e-6)


@pytest.mark.skipif(not tf.config.list_physical_devices("GPU"), reason="CUDA GPU required")
@pytest.mark.parametrize("dtype", [tf.float32, tf.float16])
def test_empty_batch_with_slots_and_queues(dtype):
    indices, synapse_types, _, weights, basis, _ = _fixture(1, 4)
    connectivity = _connectivity(indices, synapse_types, n_pre=4, n_post=3)
    slots = (tf.zeros((0, 2), dtype), tf.zeros((0, 2), dtype))
    master = tf.Variable(weights)
    with tf.GradientTape() as tape:
        tape.watch(slots)
        currents, record = calculate_recurrent_csr_currents(
            slots, master, tf.constant(basis), 0.5, connectivity,
            queues=(spike_queue(slots[1]),),
        )
        loss = tf.reduce_sum(tf.cast(currents, tf.float32))
    slot_grads, weight_grad = tape.gradient(loss, (slots, master))
    assert currents.shape == (0, 4)
    assert record.dtype == tf.uint32
    assert [grad.shape for grad in slot_grads] == [(0, 2), (0, 2)]
    np.testing.assert_array_equal(weight_grad.numpy(), 0.0)


@pytest.mark.skipif(not tf.config.list_physical_devices("GPU"), reason="CUDA GPU required")
@pytest.mark.parametrize("use_initial", [False, True])
@pytest.mark.parametrize("dtype", [tf.float32, tf.float16])
def test_connectivity_without_edges(dtype, use_initial):
    connectivity = _connectivity(np.zeros((0, 2), np.int64), np.zeros(0, np.int64), 10, 7)
    spikes = tf.ones((3, 10), dtype)
    master = tf.Variable(np.zeros(0, np.float32))
    seed = tf.constant(np.random.default_rng(0).normal(size=(21, 4)), dtype)
    with tf.GradientTape() as tape:
        tape.watch(spikes)
        currents = calculate_recurrent_csr_currents(
            spikes, master, tf.ones((1, 4)), 0.5, connectivity,
            initial=seed if use_initial else None,
        )
        loss = tf.reduce_sum(tf.cast(currents, tf.float32))
    spike_grad, weight_grad = tape.gradient(loss, (spikes, master))
    np.testing.assert_array_equal(currents.numpy(), seed.numpy() if use_initial else 0.0)
    np.testing.assert_array_equal(spike_grad.numpy(), np.zeros((3, 10)))
    assert weight_grad.shape == (0,)


# The tile sweep (CsrTileSweepKernel) serves FP16 and FP32 forwards with four
# basis columns of rows that repeat targets (the LGN) in one tensor, at every
# batch: narrow tiles while the wide grid is at most half the SMs, wide ones
# otherwise (SweepTargets). The three-slot recurrent history uses it only fused
# with the LGN (tests/test_cuda_csr_external.py).
SWEEP_BATCH = 64


def _lgn_like_edges(n_pre, n_post, targets_per_row, seed):
    """Rows sending one to four synapse types to each target, as LGN rows do, and
    one long run per row that crosses warps; tile-edge targets are included."""
    rng = np.random.default_rng(seed)
    edges = []
    for pre in range(n_pre):
        targets = np.unique(np.concatenate((
            rng.integers(0, n_post, targets_per_row), [5119, 5120, n_post - 1])))
        repeats = rng.integers(1, 5, targets.size)
        repeats[rng.integers(0, targets.size)] = 70
        posts = np.repeat(targets, repeats)
        edges.append(np.stack((posts, np.full_like(posts, pre)), axis=1))
    indices = np.concatenate(edges).astype(np.int64)
    synapse_types = rng.integers(0, 5, indices.shape[0]).astype(np.int64)
    weights = rng.normal(size=indices.shape[0]).astype(np.float32)
    return _csr_sorted(indices, synapse_types, weights)


@pytest.mark.skipif(not tf.config.list_physical_devices("GPU"), reason="CUDA GPU required")
@pytest.mark.parametrize("batch", [1, SWEEP_BATCH])
@pytest.mark.parametrize("use_initial", [False, True])
@pytest.mark.parametrize("n_post, targets_per_row", [(12_000, 1500), (330_000, 2500)])
def test_tile_sweep_lgn_matches_float64(n_post, targets_per_row, use_initial, batch):
    """The FP16 sweep of target-repeating rows (narrow tiles at batch 1, wide at
    64; masked table at 3 wide tiles, binary search above 64) is at least as
    accurate as the global-atomic scatter, which the same rows split into two
    slots select on the same samples; a silent sample's rows hold exactly
    `initial`."""
    n_pre = 300
    indices, synapse_types, weights = _lgn_like_edges(n_pre, n_post, targets_per_row, n_post)
    connectivity = _connectivity(indices, synapse_types, n_pre, n_post)
    assert connectivity.repeats_targets
    rng = np.random.default_rng(3)
    spikes = (rng.random((batch, n_pre)) < 0.3).astype(np.float16)
    silent = batch // 2
    spikes[silent] = 0.0
    basis = rng.normal(size=(5, 4)).astype(np.float32)
    seed = rng.normal(size=(batch * n_post, 4)).astype(np.float16)
    if not use_initial:
        seed[:] = 0.0
    expected = _float64_forward(indices, synapse_types, spikes, weights, basis, n_post) + seed
    errors = []
    for values in (tf.constant(spikes), tuple(tf.split(tf.constant(spikes), 2, axis=1))):
        currents = calculate_recurrent_csr_currents(
            values, tf.Variable(weights), tf.constant(basis), 0.37, connectivity,
            initial=tf.constant(seed) if use_initial else None,
        ).numpy()
        np.testing.assert_array_equal(currents[silent * n_post:(silent + 1) * n_post],
                                      seed[silent * n_post:(silent + 1) * n_post])
        error = np.abs(currents.astype(np.float64) - expected)
        errors.append((error.max(), error.mean()))
    (sweep_max, sweep_mean), (atomic_max, atomic_mean) = errors
    assert sweep_max <= atomic_max and sweep_mean <= atomic_mean, errors


def _two_contribution_history(width, n_post, n_slots, seed):
    """A history CSR and spikes in which every target receives at most two
    contributions per sample: row r feeds targets r % G + k G, and a sample fires
    at most two rows of a class G, so every FP32 summation order agrees."""
    rng = np.random.default_rng(seed)
    classes = 97
    n_pre = n_slots * width
    posts = np.arange(0, n_post - classes, classes)
    indices = np.concatenate([
        np.stack((posts + pre % classes, np.full_like(posts, pre)), axis=1)
        for pre in range(n_pre)]).astype(np.int64)
    synapse_types = rng.integers(0, 5, indices.shape[0]).astype(np.int64)
    weights = rng.normal(size=indices.shape[0]).astype(np.float32)
    spikes = np.zeros((SWEEP_BATCH, n_pre), np.float16)
    for sample in range(SWEEP_BATCH):
        for group in range(classes):
            rows = np.arange(group, n_pre, classes)
            spikes[sample, rng.choice(rows, size=rng.integers(0, 3), replace=False)] = 1.0
    return _csr_sorted(indices, synapse_types, weights) + (spikes,)


@pytest.mark.skipif(not tf.config.list_physical_devices("GPU"), reason="CUDA GPU required")
@pytest.mark.parametrize("resource_mode", ["0", "1"])
@pytest.mark.parametrize("use_initial", [False, True])
@pytest.mark.parametrize("n_post", [12_000, 330_000])
def test_tile_sweep_records_bitwise_twoedge(n_post, use_initial, resource_mode, monkeypatch):
    """Three slots of a batch-64 FP16 history with carried queue records (the
    newest built by BuildMissingRecords) and without (the merged queue) give the
    same currents bit for bit when no target receives more than two
    contributions, and the returned record is the newest slot's."""
    monkeypatch.setenv("V1_CSR_RESOURCE_MODE", resource_mode)
    n_slots, width = 3, 1500
    indices, synapse_types, weights, spikes = _two_contribution_history(
        width, n_post, n_slots, n_post)
    connectivity = _connectivity(indices, synapse_types, n_slots * width, n_post)
    assert not connectivity.repeats_targets
    slots = tuple(tf.split(tf.constant(spikes), n_slots, axis=1))
    basis = tf.constant(np.random.default_rng(1).normal(size=(5, 4)), tf.float32)
    initial = tf.constant(np.random.default_rng(2).normal(
        size=(SWEEP_BATCH * n_post, 4)).astype(np.float16)) if use_initial else None
    master = tf.Variable(weights)
    swept, record = calculate_recurrent_csr_currents(
        slots, master, basis, 0.5, connectivity, initial=initial,
        queues=tuple(spike_queue(slot) for slot in slots[1:]))
    merged = calculate_recurrent_csr_currents(
        slots, master, basis, 0.5, connectivity, initial=initial)
    assert np.any(merged.numpy() != (0.0 if initial is None else initial.numpy()))
    np.testing.assert_array_equal(swept.numpy(), merged.numpy())
    # A record is its active entries, then their count and the sample starts;
    # the slack after the entries is never read.
    capacity = SWEEP_BATCH * width
    for got, want in ((record.numpy(), spike_queue(slots[0]).numpy()),):
        count = int(want[capacity])
        np.testing.assert_array_equal(got[:count], want[:count])
        np.testing.assert_array_equal(got[capacity:], want[capacity:])


@pytest.mark.skipif(not tf.config.list_physical_devices("GPU"), reason="CUDA GPU required")
@pytest.mark.parametrize("resource_mode", ["0", "1"])
def test_tile_sweep_graphs_share_tile_memory(monkeypatch, resource_mode):
    """Batch-64 validation graphs reach the sweep's tables through the shared
    registry too, rather than each operator keeping its own."""
    monkeypatch.setenv("V1_CSR_RESOURCE_MODE", resource_mode)
    n_pre, n_post = 32768, 8192
    indices = np.column_stack((
        np.tile([0, 0, 2048, 4096, 6144], n_pre),
        np.repeat(np.arange(n_pre), 5),
    ))
    with tf.device("/GPU:0"):
        connectivity = _connectivity(
            indices, np.zeros(len(indices), np.int64), n_pre, n_post
        )
        assert connectivity.repeats_targets
        weights = tf.Variable(tf.fill((len(indices),), 1.0 / n_pre))
        spikes = tf.ones((SWEEP_BATCH, n_pre), tf.float16)

        def make_rollout(chunks):
            @tf.function
            def rollout(activity):
                total = tf.zeros((), tf.float32)
                for _ in range(chunks):
                    currents = calculate_recurrent_csr_currents(
                        activity, weights, tf.ones((1, 4)), 0.1, connectivity
                    )
                    total += tf.reduce_sum(tf.cast(currents, tf.float32))
                return total

            return rollout

        make_rollout(1)(spikes).numpy()
        before = tf.config.experimental.get_memory_info("GPU:0")["current"]
        for rollout in (make_rollout(20), make_rollout(20)):
            # 5 edges x 4 basis columns per row, each 1 / n_pre, per sample.
            np.testing.assert_allclose(rollout(spikes).numpy(), 20 * SWEEP_BATCH * 20,
                                       rtol=1e-3)
        after = tf.config.experimental.get_memory_info("GPU:0")["current"]
        assert after - before < 8 * 2**20, (
            f"Validation retained {(after - before) / 2**20:.2f} MiB of extra "
            "GPU memory for unchanged connectivity"
        )
