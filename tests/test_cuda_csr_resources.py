import numpy as np
import pytest
import tensorflow as tf

from v1_model_utils.cuda_csr_resources import resource_mode_enabled
from v1_model_utils.cuda_csr_external import (
    build_csr_connectivity as build_external_connectivity,
    calculate_external_csr_currents,
)
from v1_model_utils.cuda_csr_recurrent import (
    DIRECT_CSR,
    build_csr_connectivity as _build_recurrent_connectivity,
    calculate_recurrent_csr_currents,
)


def build_csr_connectivity(*args, **kwargs):
    """Declare the CSR edge ordering the DIRECT_CSR kernels require.

    Every fixture below is already written in CSR order; the builder asserts it.
    """
    kwargs.setdefault("weights_csr_ordered", DIRECT_CSR)
    return _build_recurrent_connectivity(*args, **kwargs)


def build_external_csr(*args, **kwargs):
    kwargs.setdefault("weights_csr_ordered", DIRECT_CSR)
    return build_external_connectivity(*args, **kwargs)


def test_resource_mode_follows_the_worker_marker(monkeypatch):
    monkeypatch.delenv("V1_CSR_RESOURCE_MODE", raising=False)
    monkeypatch.setattr(tf.config, "list_logical_devices", lambda _: ["GPU:0"])
    monkeypatch.delenv("V1_DISTRIBUTED_WORKER", raising=False)
    assert not resource_mode_enabled()
    monkeypatch.setenv("V1_DISTRIBUTED_WORKER", "1")
    assert resource_mode_enabled()
    monkeypatch.setenv("V1_DISTRIBUTED_WORKER", "0")
    assert not resource_mode_enabled()


def test_resource_mode_override_wins(monkeypatch):
    monkeypatch.setenv("V1_DISTRIBUTED_WORKER", "1")
    monkeypatch.setenv("V1_CSR_RESOURCE_MODE", "0")
    assert not resource_mode_enabled()
    monkeypatch.delenv("V1_DISTRIBUTED_WORKER", raising=False)
    monkeypatch.setenv("V1_CSR_RESOURCE_MODE", "1")
    assert resource_mode_enabled()


def test_resource_mode_is_not_inferred_from_visible_gpus(monkeypatch):
    """Visibility is not the contract; the entry point declares the mode.

    Otherwise any script that merely sees two GPUs while using one would
    silently switch connectivity backend.
    """
    monkeypatch.delenv("V1_CSR_RESOURCE_MODE", raising=False)
    monkeypatch.delenv("V1_DISTRIBUTED_WORKER", raising=False)
    monkeypatch.setattr(
        tf.config, "list_logical_devices", lambda _: ["GPU:0", "GPU:1"]
    )
    assert not resource_mode_enabled()


@pytest.mark.skipif(not tf.config.list_physical_devices("GPU"), reason="CUDA GPU required")
def test_resource_recurrent_forward_and_gradients_match_tensor_backend(monkeypatch):
    indices = np.array([[0, 0], [1, 0], [0, 1], [1, 2]], np.int64)
    types = np.array([0, 1, 1, 0], np.int64)
    spikes = tf.constant([[1.0, 0.5, 0.25], [0.0, 1.0, 0.5]])
    basis = tf.constant([[1.0, 0.5, -0.5, 0.25], [0.5, -1.0, 0.25, 1.0]])
    upstream = tf.reshape(tf.range(16, dtype=tf.float32) / 7, (4, 4))

    monkeypatch.setenv("V1_CSR_RESOURCE_MODE", "0")
    tensor_connectivity = build_csr_connectivity(indices, types, 3, 2)
    tensor_weights = tf.Variable([0.2, -0.4, 0.7, 0.1])
    with tf.GradientTape() as tape:
        tape.watch(spikes)
        tensor_output = calculate_recurrent_csr_currents(
            spikes, tensor_weights, basis, 0.37, tensor_connectivity
        )
        tensor_loss = tf.reduce_sum(tensor_output * upstream)
    tensor_grads = tape.gradient(tensor_loss, (spikes, tensor_weights))

    monkeypatch.setenv("V1_CSR_RESOURCE_MODE", "1")
    resource_connectivity = build_csr_connectivity(indices, types, 3, 2)
    resource_weights = tf.Variable(tensor_weights.numpy())
    with tf.GradientTape() as tape:
        tape.watch(spikes)
        resource_output = calculate_recurrent_csr_currents(
            spikes, resource_weights, basis, 0.37, resource_connectivity
        )
        resource_loss = tf.reduce_sum(resource_output * upstream)
    resource_grads = tape.gradient(resource_loss, (spikes, resource_weights))

    np.testing.assert_allclose(resource_output, tensor_output, rtol=2e-5, atol=2e-5)
    np.testing.assert_allclose(resource_grads[0], tensor_grads[0], rtol=2e-5, atol=2e-5)
    np.testing.assert_allclose(resource_grads[1], tensor_grads[1], rtol=2e-5, atol=2e-5)


@pytest.mark.skipif(
    len(tf.config.list_physical_devices("GPU")) != 1,
    reason="resource workers require exactly one visible CUDA GPU",
)
@pytest.mark.parametrize(
    ("compute_activity_gradient", "compute_weight_gradient"),
    [(False, False), (False, True), (True, False), (True, True)],
)
@pytest.mark.parametrize("n_basis", [3, 4])
@pytest.mark.parametrize("dtype", [tf.float16, tf.float32])
def test_resource_external_backward_modes_match_tensor_backend(
    monkeypatch, compute_activity_gradient, compute_weight_gradient, n_basis, dtype
):
    indices = np.array([[0, 0], [1, 0], [0, 1], [1, 2]], np.int64)
    types = np.array([0, 1, 1, 0], np.int64)
    activity = tf.constant([[1.0, 0.5, 0.25], [0.0, 1.0, 0.5]], dtype=dtype)
    basis = tf.cast(
        tf.constant([[1.0, 0.5, -0.5, 0.25], [0.5, -1.0, 0.25, 1.0]])[
            :, :n_basis
        ],
        dtype,
    )
    upstream = tf.cast(
        tf.reshape(tf.range(4 * n_basis, dtype=tf.float32) / 7, (4, n_basis)),
        dtype,
    )

    def evaluate(connectivity):
        weights = tf.Variable([0.2, -0.4, 0.7, 0.1])
        with tf.GradientTape() as tape:
            tape.watch(activity)
            output = calculate_external_csr_currents(
                activity,
                weights,
                basis,
                connectivity,
                compute_activity_gradient=compute_activity_gradient,
                compute_weight_gradient=compute_weight_gradient,
            )
            loss = tf.reduce_sum(output * upstream)
        return output, tape.gradient(loss, (activity, weights))

    monkeypatch.setenv("V1_CSR_RESOURCE_MODE", "0")
    expected = evaluate(build_external_csr(indices, types, 3, 2))
    monkeypatch.setenv("V1_CSR_RESOURCE_MODE", "1")
    actual = evaluate(build_external_csr(indices, types, 3, 2))

    tolerance = 2e-3 if dtype == tf.float16 else 2e-5
    np.testing.assert_allclose(actual[0], expected[0], rtol=tolerance, atol=tolerance)
    if compute_activity_gradient:
        np.testing.assert_allclose(
            actual[1][0], expected[1][0], rtol=tolerance, atol=tolerance
        )
    else:
        assert actual[1][0] is expected[1][0] is None
    if compute_weight_gradient:
        np.testing.assert_allclose(
            actual[1][1], expected[1][1], rtol=tolerance, atol=tolerance
        )
    else:
        assert actual[1][1] is expected[1][1] is None


@pytest.mark.skipif(
    len(tf.config.list_physical_devices("GPU")) != 1,
    reason="resource workers require exactly one visible CUDA GPU",
)
def test_multiple_resources_are_isolated(monkeypatch):
    monkeypatch.setenv("V1_CSR_RESOURCE_MODE", "1")
    first = build_csr_connectivity(
        np.array([[0, 0]], np.int64), np.array([0], np.int64), 1, 1
    )
    second = build_csr_connectivity(
        np.array([[0, 0], [0, 1]], np.int64), np.array([0, 0], np.int64), 2, 1
    )
    assert first.resource_name != second.resource_name
    first_output = calculate_recurrent_csr_currents(
        tf.constant([[2.0]]), tf.constant([3.0]), tf.constant([[1.0]]), 1.0, first
    )
    second_output = calculate_recurrent_csr_currents(
        tf.constant([[2.0, 5.0]]),
        tf.constant([3.0, 7.0]),
        tf.constant([[1.0]]),
        1.0,
        second,
    )
    np.testing.assert_allclose(first_output, [[6.0]])
    np.testing.assert_allclose(second_output, [[41.0]])


@pytest.mark.skipif(
    not tf.config.list_physical_devices("GPU"), reason="CUDA GPU required"
)
@pytest.mark.parametrize("dtype", [tf.float32, tf.float16])
def test_resource_backward_matches_tensor_backend_at_the_hot_batch(
    monkeypatch, dtype
):
    """The resource backend must reach the batch-32 pair-projected backward.

    The tensor backend specializes the recurrent backward for a static batch of
    32 with the four-column basis, which is the production per-replica shape.
    Replicated training runs on the resource backend, so a resource replica has
    to select the same kernel and produce the same gradients.
    """
    rng = np.random.default_rng(4242)
    n_pre, n_post, n_types, batch = 48, 96, 7, 32
    n_edges = 6000
    indices = np.stack(
        (rng.integers(0, n_post, n_edges), rng.integers(0, n_pre, n_edges)),
        axis=1,
    ).astype(np.int64)
    synapse_types = rng.integers(0, n_types, n_edges).astype(np.int64)
    # The builder asserts the caller already holds CSR order, and its sort key
    # includes the synapse type.
    order = np.lexsort(
        (np.arange(n_edges), synapse_types, indices[:, 0], indices[:, 1])
    )
    indices = indices[order]
    synapse_types = synapse_types[order]

    spikes_host = rng.uniform(0.1, 1.0, (batch, n_pre)).astype(np.float32)
    spikes_host[rng.random(spikes_host.shape) < 0.4] = 0.0
    weights_host = rng.normal(size=n_edges).astype(np.float32)
    basis_host = rng.normal(size=(n_types, 4)).astype(np.float32)
    upstream = rng.normal(size=(batch * n_post, 4)).astype(np.float32)

    def run():
        connectivity = build_csr_connectivity(
            indices, synapse_types, n_pre, n_post
        )
        spikes = tf.constant(spikes_host, dtype)
        weights = tf.Variable(weights_host)
        with tf.GradientTape() as tape:
            tape.watch(spikes)
            currents = calculate_recurrent_csr_currents(
                spikes, weights, tf.constant(basis_host, dtype), 0.37, connectivity
            )
            loss = tf.reduce_sum(tf.cast(currents, tf.float32) * upstream)
        gradients = tape.gradient(loss, (spikes, weights))
        return [np.asarray(gradient, np.float32) for gradient in gradients]

    monkeypatch.setenv("V1_CSR_RESOURCE_MODE", "0")
    tensor_gradients = run()
    monkeypatch.setenv("V1_CSR_RESOURCE_MODE", "1")
    resource_gradients = run()

    tolerance = 2e-5 if dtype == tf.float32 else 3e-3
    for got, want, name in zip(
        resource_gradients, tensor_gradients, ("spike_grad", "weight_grad")
    ):
        np.testing.assert_allclose(
            got, want, rtol=tolerance, atol=tolerance, err_msg=name
        )


@pytest.mark.skipif(
    not tf.config.list_physical_devices("GPU"), reason="CUDA GPU required"
)
def test_pair_projected_backward_is_refused_without_the_projection(monkeypatch):
    """A connectivity built without a backward cannot serve the specialization.

    The LGN and BKG inputs upload an empty pair projection because their
    backward never runs. Asking the resource operator for the pair-projected
    kernel anyway has to fail loudly rather than read absent metadata.
    """
    from v1_model_utils.cuda_csr_resources import load_ops

    monkeypatch.setenv("V1_CSR_RESOURCE_MODE", "1")
    indices = np.array([[0, 0], [1, 0], [0, 1], [1, 2]], np.int64)
    types = np.array([0, 0, 0, 0], np.int64)
    connectivity = build_external_csr(
        indices, types, 3, 2, needs_backward=False
    )
    assert connectivity.resource_name is not None
    assert int(tf.size(connectivity.pair_ids)) == 0

    with pytest.raises(tf.errors.InvalidArgumentError, match="pair projection"):
        load_ops().v1_csr_backward_resource(
            tf.zeros((32, 3)),
            tf.zeros((32 * 2, 4)),
            tf.zeros((4,)),
            tf.zeros((1, 4)),
            tf.constant(0.3),
            n_post=2,
            n_edges=4,
            resource_name=connectivity.resource_name,
            pair_projected=True,
        )


@pytest.mark.skipif(
    not tf.config.list_physical_devices("GPU"), reason="CUDA GPU required"
)
def test_resource_forward_accumulates_onto_initial_currents(monkeypatch):
    """The resource forward must chain current sources like the tensor one.

    ``V1Column`` seeds each current operator with the previous source's output
    so no pass over the [batch * n_post, n_basis] current buffer is needed per
    source. Replicated training runs on the resource backend, so it has to
    accept the same accumulator instead of paying for separate adds.
    """
    indices = np.array([[0, 0], [1, 0], [0, 1], [1, 2]], np.int64)
    types = np.array([0, 1, 1, 0], np.int64)
    spikes = tf.constant([[1.0, 0.5, 0.25], [0.0, 1.0, 0.5]])
    basis = tf.constant([[1.0, 0.5, -0.5, 0.25], [0.5, -1.0, 0.25, 1.0]])
    seed = tf.reshape(tf.range(16, dtype=tf.float32) / 3, (4, 4))
    upstream = tf.reshape(tf.range(16, dtype=tf.float32) / 7, (4, 4))

    def run(with_initial):
        connectivity = build_csr_connectivity(indices, types, 3, 2)
        weights = tf.Variable([0.2, -0.4, 0.7, 0.1])
        with tf.GradientTape() as tape:
            tape.watch(spikes)
            currents = calculate_recurrent_csr_currents(
                spikes,
                weights,
                basis,
                0.37,
                connectivity,
                initial=seed if with_initial else None,
            )
            loss = tf.reduce_sum(currents * upstream)
        return currents, tape.gradient(loss, (spikes, weights))

    monkeypatch.setenv("V1_CSR_RESOURCE_MODE", "1")
    plain_currents, plain_grads = run(with_initial=False)
    seeded_currents, seeded_grads = run(with_initial=True)

    np.testing.assert_allclose(
        seeded_currents.numpy(),
        plain_currents.numpy() + seed.numpy(),
        rtol=1e-6,
        atol=1e-6,
    )
    # An additive accumulator must not change the gradient of anything else.
    for seeded, plain, name in zip(
        seeded_grads, plain_grads, ("spike_grad", "weight_grad")
    ):
        np.testing.assert_allclose(
            seeded.numpy(), plain.numpy(), rtol=1e-6, atol=1e-6, err_msg=name
        )


@pytest.mark.skipif(
    not tf.config.list_physical_devices("GPU"), reason="CUDA GPU required"
)
def test_resource_initial_receives_the_upstream_gradient(monkeypatch):
    monkeypatch.setenv("V1_CSR_RESOURCE_MODE", "1")
    indices = np.array([[0, 0], [1, 0], [0, 1], [1, 2]], np.int64)
    types = np.array([0, 1, 1, 0], np.int64)
    connectivity = build_csr_connectivity(indices, types, 3, 2)
    spikes = tf.constant([[1.0, 0.5, 0.25], [0.0, 1.0, 0.5]])
    basis = tf.constant([[1.0, 0.5, -0.5, 0.25], [0.5, -1.0, 0.25, 1.0]])
    upstream = tf.reshape(tf.range(16, dtype=tf.float32) / 7, (4, 4))
    seed = tf.Variable(tf.zeros((4, 4)))

    with tf.GradientTape() as tape:
        currents = calculate_recurrent_csr_currents(
            spikes, tf.constant([0.2, -0.4, 0.7, 0.1]), basis, 0.37,
            connectivity, initial=seed,
        )
        loss = tf.reduce_sum(currents * upstream)
    np.testing.assert_allclose(
        tape.gradient(loss, seed).numpy(), upstream.numpy(), rtol=1e-6, atol=1e-6
    )
