import numpy as np
import pytest
import tensorflow as tf

from v1_model_utils.cuda_csr_recurrent import (
    DIRECT_CSR,
    SPECIALIZED_BATCH_SIZES,
    build_csr_connectivity,
    calculate_recurrent_csr_currents,
    kernel_variant,
)


SPECIALIZED_BATCHES = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512)


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


def test_dispatch_contract():
    assert SPECIALIZED_BATCH_SIZES == SPECIALIZED_BATCHES
    for batch_size in SPECIALIZED_BATCHES:
        assert kernel_variant(4, batch_size) == f"basis4_batch{batch_size}"
        assert kernel_variant(5, batch_size) == f"generic_basis_batch{batch_size}"
    assert kernel_variant(4, 3) == "basis4_generic_batch"
    assert kernel_variant(5, 3) == "generic_basis_generic_batch"


@pytest.mark.skipif(not tf.config.list_physical_devices("GPU"), reason="CUDA GPU required")
@pytest.mark.parametrize("n_basis", [4, 5])
@pytest.mark.parametrize("batch_size", SPECIALIZED_BATCHES + (3, 6, 10))
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


def test_pair_projection_only_claims_qualified_power_of_two_shapes():
    from v1_model_utils.cuda_csr_recurrent.wrapper import pair_projection_applies

    indices, synapse_types, _, _, _, _ = _fixture(32, 4)
    connectivity = _connectivity(indices, synapse_types, n_pre=4, n_post=3)
    for batch, n_basis, expected in (
        (1, 4, True), (16, 4, True), (32, 4, True), (64, 4, True),
        (512, 4, True), (32, 5, False), (3, 4, False), (1024, 4, False),
    ):
        spikes = tf.zeros((batch, 4), tf.float16)
        basis = tf.zeros((3, n_basis))
        assert pair_projection_applies(spikes, basis, connectivity) is expected


@pytest.mark.skipif(not tf.config.list_physical_devices("GPU"), reason="CUDA GPU required")
@pytest.mark.parametrize("dtype", [tf.float32, tf.float16])
def test_pair_projected_backward_matches_general_kernel(dtype):
    """The batch-32 specialization must agree with the general kernel.

    A denser fixture than ``_fixture``: several hundred edges per presynaptic
    row, so a mistake in the per-edge warp reduction cannot hide.
    """
    from v1_model_utils.cuda_csr_recurrent import wrapper

    rng = np.random.default_rng(90210)
    n_pre, n_post, n_types, batch = 48, 96, 7, 32
    n_edges = 6000
    indices = np.stack(
        (rng.integers(0, n_post, n_edges), rng.integers(0, n_pre, n_edges)), axis=1
    ).astype(np.int64)
    synapse_types = rng.integers(0, n_types, n_edges).astype(np.int64)
    spikes = rng.uniform(0.1, 1.0, (batch, n_pre)).astype(np.float32)
    spikes[rng.random(spikes.shape) < 0.4] = 0.0
    weights = rng.normal(size=n_edges).astype(np.float32)
    basis = rng.normal(size=(n_types, 4)).astype(np.float32)
    upstream = rng.normal(size=(batch * n_post, 4)).astype(np.float32)
    indices, synapse_types, weights = _csr_sorted(indices, synapse_types, weights)
    connectivity = _connectivity(indices, synapse_types, n_pre, n_post)
    assert wrapper.pair_projection_applies(
        tf.zeros((batch, n_pre), dtype), tf.zeros((n_types, 4), dtype), connectivity
    ) is (dtype == tf.float16)

    def run(force_general):
        original = wrapper.pair_projection_applies
        if force_general:
            wrapper.pair_projection_applies = lambda *args, **kwargs: False
        try:
            spikes_tensor = tf.constant(spikes, dtype)
            master = tf.Variable(weights)
            with tf.GradientTape() as tape:
                tape.watch(spikes_tensor)
                currents = calculate_recurrent_csr_currents(
                    spikes_tensor, master, tf.constant(basis, dtype),
                    0.37, connectivity,
                )
                loss = tf.reduce_sum(
                    tf.cast(currents, tf.float32) * upstream
                )
            grads = tape.gradient(loss, (spikes_tensor, master))
            return [np.asarray(g, np.float32) for g in grads]
        finally:
            wrapper.pair_projection_applies = original

    specialized = run(force_general=False)
    general = run(force_general=True)
    tolerance = 2e-5 if dtype == tf.float32 else 3e-3
    for got, want, name in zip(specialized, general, ("spike_grad", "weight_grad")):
        np.testing.assert_allclose(
            got, want, rtol=tolerance, atol=tolerance, err_msg=name
        )


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
