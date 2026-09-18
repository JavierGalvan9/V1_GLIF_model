import numpy as np
import pytest
import tensorflow as tf
from types import SimpleNamespace

from v1_model_utils.cuda_csr_external import (
    SPECIALIZED_BATCH_SIZES,
    build_csr_connectivity,
    calculate_external_csr_currents,
    kernel_variant,
)
from v1_model_utils.cuda_csr_recurrent import DIRECT_CSR
from v1_model_utils.models import V1Column


RUNTIME_BATCH_SIZES = (3, 5, 9)


def _csr_sorted(indices, synapse_types, weights):
    """Put edges in the external operator's CSR order.

    Its sort key omits the synapse type, unlike the recurrent operator's.
    """
    order = np.lexsort((np.arange(indices.shape[0]), indices[:, 0], indices[:, 1]))
    return indices[order], synapse_types[order], weights[order]


def _connectivity(indices, synapse_types, n_pre, n_post):
    """Build connectivity, declaring the CSR ordering the kernels require."""
    return build_csr_connectivity(
        indices, synapse_types, n_pre=n_pre, n_post=n_post,
        weights_csr_ordered=DIRECT_CSR,
    )


def _fixture(batch_size, n_basis, dtype=np.float32):
    rng = np.random.default_rng(731 + batch_size + n_basis)
    indices = np.array(
        [[2, 0], [0, 1], [1, 2], [1, 0], [2, 2], [0, 3], [2, 1]],
        dtype=np.int64,
    )
    synapse_types = np.array([1, 0, 2, 1, 0, 2, 1], dtype=np.int64)
    activity = rng.integers(0, 4, size=(batch_size, 4)).astype(dtype)
    weights = rng.normal(size=len(indices)).astype(np.float32)
    basis = rng.normal(size=(3, n_basis)).astype(dtype)
    upstream = rng.normal(size=(batch_size * 3, n_basis)).astype(dtype)
    indices, synapse_types, weights = _csr_sorted(indices, synapse_types, weights)
    return indices, synapse_types, activity, weights, basis, upstream


def _reference(indices, synapse_types, activity, weights, basis, upstream):
    batch_size = activity.shape[0]
    n_post = 3
    currents = np.zeros((batch_size * n_post, basis.shape[1]), np.float32)
    activity_grad = np.zeros_like(activity, dtype=np.float32)
    weight_grad = np.zeros_like(weights)
    for edge, (post, pre) in enumerate(indices):
        basis_vector = basis[synapse_types[edge]].astype(np.float32)
        for batch in range(batch_size):
            flat_post = batch * n_post + post
            currents[flat_post] += activity[batch, pre] * weights[edge] * basis_vector
            projected = np.dot(upstream[flat_post].astype(np.float32), basis_vector)
            activity_grad[batch, pre] += weights[edge] * projected
            weight_grad[edge] += activity[batch, pre] * projected
    return currents, activity_grad, weight_grad


def test_dispatch_contract():
    assert SPECIALIZED_BATCH_SIZES == (1, 2, 4, 8, 16, 32, 64, 128, 256, 512)
    for batch_size in SPECIALIZED_BATCH_SIZES:
        assert kernel_variant(4, batch_size) == f"basis4_batch{batch_size}"
        assert kernel_variant(5, batch_size) == f"generic_basis_batch{batch_size}"
    assert kernel_variant(4, 3) == "basis4_runtime_batch"
    assert kernel_variant(5, 3) == "generic_basis_runtime_batch"


def test_invalid_connectivity_metadata():
    with pytest.raises(ValueError, match="shape"):
        build_csr_connectivity(np.zeros((2, 3)), np.zeros(2), 2, 2)
    with pytest.raises(ValueError, match="outside"):
        build_csr_connectivity(np.array([[2, 0]]), np.zeros(1), 1, 2)
    with pytest.raises(ValueError, match="uint8"):
        build_csr_connectivity(np.array([[0, 0]]), np.array([256]), 1, 1)


def _fixed_four_bkg_fixture(dtype=np.float32):
    rng = np.random.default_rng(915)
    indices = np.array(
        [
            [0, 1], [1, 4], [2, 8], [0, 13], [1, 19], [2, 27],
            [0, 35], [1, 44], [2, 58], [0, 63], [1, 77], [2, 91],
        ],
        dtype=np.int64,
    )
    types = np.arange(len(indices), dtype=np.int64) % 3
    activity = rng.integers(0, 2, size=(2, 100)).astype(dtype)
    weights = rng.normal(size=len(indices)).astype(np.float32)
    basis = rng.normal(size=(3, 4)).astype(dtype)
    upstream = rng.normal(size=(6, 4)).astype(dtype)
    indices, types, weights = _csr_sorted(indices, types, weights)
    return indices, types, activity, weights, basis, upstream


def test_fixed_four_bkg_metadata_is_guarded():
    indices, types, *_ = _fixed_four_bkg_fixture()
    connectivity = _connectivity(indices, types, n_pre=100, n_post=3)
    assert connectivity.incoming_row_splits is not None
    np.testing.assert_array_equal(connectivity.incoming_row_splits, [0, 4, 8, 12])

    unordered_weights = build_csr_connectivity(
        indices, types, n_pre=100, n_post=3, weights_csr_ordered=False
    )
    assert unordered_weights.incoming_row_splits is None

    uneven = indices.copy()
    uneven[0, 0] = 1
    uneven, uneven_types, _ = _csr_sorted(uneven, types, np.arange(len(types)))
    fallback = _connectivity(uneven, uneven_types, n_pre=100, n_post=3)
    assert fallback.incoming_row_splits is None


@pytest.mark.skipif(not tf.config.list_physical_devices("GPU"), reason="CUDA GPU required")
@pytest.mark.parametrize("dtype", [tf.float16, tf.float32])
def test_fixed_four_bkg_forward_and_gradients(dtype):
    np_dtype = np.float16 if dtype == tf.float16 else np.float32
    fixture = _fixed_four_bkg_fixture(np_dtype)
    indices, types, activity, weights, basis, upstream = fixture
    connectivity = _connectivity(indices, types, n_pre=100, n_post=3)
    activity_tensor = tf.constant(activity, dtype)
    master_weights = tf.Variable(weights)
    with tf.GradientTape() as tape:
        tape.watch(activity_tensor)
        currents = calculate_external_csr_currents(
            activity_tensor, master_weights, tf.constant(basis, dtype), connectivity
        )
        loss = tf.reduce_sum(currents * tf.constant(upstream, dtype))
    activity_grad, weight_grad = tape.gradient(loss, (activity_tensor, master_weights))
    expected = _reference(*fixture)
    tolerance = 2e-3 if dtype == tf.float16 else 2e-5
    np.testing.assert_allclose(currents, expected[0], rtol=tolerance, atol=tolerance)
    np.testing.assert_allclose(
        activity_grad, expected[1], rtol=tolerance, atol=tolerance
    )
    np.testing.assert_allclose(weight_grad, expected[2], rtol=tolerance, atol=tolerance)


@pytest.mark.skipif(not tf.config.list_physical_devices("GPU"), reason="CUDA GPU required")
@pytest.mark.parametrize("n_basis", [4, 5])
@pytest.mark.parametrize("batch_size", SPECIALIZED_BATCH_SIZES + RUNTIME_BATCH_SIZES)
def test_full_forward_and_backward(batch_size, n_basis):
    fixture = _fixture(batch_size, n_basis)
    indices, synapse_types, activity, weights, basis, upstream = fixture
    connectivity = _connectivity(indices, synapse_types, n_pre=4, n_post=3)
    activity_tensor = tf.constant(activity)
    master_weights = tf.Variable(weights)
    with tf.GradientTape() as tape:
        tape.watch(activity_tensor)
        currents = calculate_external_csr_currents(
            activity_tensor,
            master_weights,
            basis,
            connectivity,
            compute_activity_gradient=True,
        )
        loss = tf.reduce_sum(currents * upstream)
    activity_grad, weight_grad = tape.gradient(loss, (activity_tensor, master_weights))
    expected = _reference(*fixture)
    np.testing.assert_allclose(currents.numpy(), expected[0], rtol=2e-5, atol=2e-5)
    np.testing.assert_allclose(activity_grad.numpy(), expected[1], rtol=2e-5, atol=2e-5)
    np.testing.assert_allclose(weight_grad.numpy(), expected[2], rtol=2e-5, atol=2e-5)


@pytest.mark.skipif(not tf.config.list_physical_devices("GPU"), reason="CUDA GPU required")
@pytest.mark.parametrize("n_basis", [3, 4, 7])
@pytest.mark.parametrize("batch_size", SPECIALIZED_BATCH_SIZES + RUNTIME_BATCH_SIZES)
def test_weight_only_backward(batch_size, n_basis):
    fixture = _fixture(batch_size, n_basis)
    indices, synapse_types, activity, weights, basis, upstream = fixture
    connectivity = _connectivity(indices, synapse_types, n_pre=4, n_post=3)
    activity_tensor = tf.constant(activity)
    master_weights = tf.Variable(weights)
    with tf.GradientTape() as tape:
        tape.watch(activity_tensor)
        currents = calculate_external_csr_currents(
            activity_tensor,
            master_weights,
            basis,
            connectivity,
            compute_activity_gradient=False,
        )
        loss = tf.reduce_sum(currents * upstream)
    activity_grad, weight_grad = tape.gradient(loss, (activity_tensor, master_weights))
    expected = _reference(*fixture)
    assert activity_grad is None
    np.testing.assert_allclose(currents.numpy(), expected[0], rtol=2e-5, atol=2e-5)
    np.testing.assert_allclose(weight_grad.numpy(), expected[2], rtol=2e-5, atol=2e-5)


@pytest.mark.skipif(not tf.config.list_physical_devices("GPU"), reason="CUDA GPU required")
@pytest.mark.parametrize("dtype", [tf.float16, tf.float32])
def test_graph_mode_zero_activity_and_changed_weights(dtype):
    fixture = _fixture(3, 4, np.float16 if dtype == tf.float16 else np.float32)
    indices, synapse_types, activity, weights, basis, upstream = fixture
    connectivity = _connectivity(indices, synapse_types, n_pre=4, n_post=3)
    master_weights = tf.Variable(weights, dtype=tf.float32)

    @tf.function
    def run(values):
        with tf.GradientTape() as tape:
            currents = calculate_external_csr_currents(
                values,
                master_weights,
                tf.constant(basis, dtype),
                connectivity,
                compute_activity_gradient=False,
            )
            loss = tf.reduce_sum(currents * tf.cast(upstream, dtype))
        return currents, tape.gradient(loss, master_weights)

    currents, weight_grad = run(tf.zeros_like(tf.constant(activity, dtype)))
    np.testing.assert_array_equal(currents.numpy(), 0)
    np.testing.assert_array_equal(weight_grad.numpy(), 0)
    master_weights.assign_add(tf.ones_like(master_weights))
    changed, changed_grad = run(tf.constant(activity, dtype))
    assert changed.dtype == dtype
    assert changed_grad.dtype == tf.float32
    assert np.all(np.isfinite(changed.numpy()))
    assert np.all(np.isfinite(changed_grad.numpy()))


@pytest.mark.skipif(not tf.config.list_physical_devices("GPU"), reason="CUDA GPU required")
def test_fp16_forward_and_both_backward_modes_match_fp32_oracle():
    fixture = _fixture(2, 4, np.float16)
    indices, synapse_types, activity, weights, basis, upstream = fixture
    connectivity = _connectivity(indices, synapse_types, n_pre=4, n_post=3)
    expected = _reference(*fixture)
    for compute_activity_gradient in (True, False):
        activity_tensor = tf.constant(activity, tf.float16)
        master_weights = tf.Variable(weights, dtype=tf.float32)
        with tf.GradientTape() as tape:
            tape.watch(activity_tensor)
            currents = calculate_external_csr_currents(
                activity_tensor,
                master_weights,
                tf.constant(basis, tf.float16),
                connectivity,
                compute_activity_gradient=compute_activity_gradient,
            )
            loss = tf.reduce_sum(currents * tf.constant(upstream, tf.float16))
        activity_grad, weight_grad = tape.gradient(
            loss, (activity_tensor, master_weights)
        )
        np.testing.assert_allclose(currents.numpy(), expected[0], rtol=2e-3, atol=2e-3)
        np.testing.assert_allclose(weight_grad.numpy(), expected[2], rtol=2e-3, atol=2e-3)
        if compute_activity_gradient:
            np.testing.assert_allclose(
                activity_grad.numpy(), expected[1], rtol=2e-3, atol=2e-3
            )
        else:
            assert activity_grad is None


@pytest.mark.skipif(not tf.config.list_physical_devices("GPU"), reason="CUDA GPU required")
@pytest.mark.parametrize(
    ("activity_enabled", "weight_enabled"),
    [(False, False), (False, True), (True, False), (True, True)],
)
def test_gradients_are_independently_selectable(activity_enabled, weight_enabled):
    fixture = _fixture(32, 4, np.float16)
    indices, synapse_types, activity, weights, basis, upstream = fixture
    connectivity = _connectivity(indices, synapse_types, n_pre=4, n_post=3)
    activity_tensor = tf.constant(activity)
    master_weights = tf.Variable(weights)
    with tf.GradientTape() as tape:
        tape.watch(activity_tensor)
        currents = calculate_external_csr_currents(
            activity_tensor,
            master_weights,
            basis,
            connectivity,
            compute_activity_gradient=activity_enabled,
            compute_weight_gradient=weight_enabled,
        )
        loss = tf.reduce_sum(currents * upstream)
    activity_grad, weight_grad = tape.gradient(
        loss, (activity_tensor, master_weights)
    )
    expected = _reference(*fixture)
    assert (activity_grad is not None) is activity_enabled
    assert (weight_grad is not None) is weight_enabled
    if activity_enabled:
        np.testing.assert_allclose(
            activity_grad.numpy(), expected[1], rtol=2e-3, atol=2e-3
        )
    if weight_enabled:
        np.testing.assert_allclose(
            weight_grad.numpy(), expected[2], rtol=2e-3, atol=2e-3
        )


@pytest.mark.skipif(not tf.config.list_physical_devices("GPU"), reason="CUDA GPU required")
def test_v1_lgn_and_background_adapters_produce_weight_gradients_only():
    indices, synapse_types, activity, weights, basis, _ = _fixture(2, 4)
    connectivity = _connectivity(indices, synapse_types, n_pre=4, n_post=3)
    lgn_weights = tf.Variable(weights, dtype=tf.float32, trainable=True)
    lgn_cell = SimpleNamespace(
        _synaptic_current_backend="cuda",
        compute_dtype=tf.float32,
        variable_dtype=tf.float32,
        input_weight_values=lgn_weights,
        synaptic_basis_weights=tf.constant(basis),
        input_csr=connectivity,
    )
    lgn_activity = tf.constant(activity > 0)
    with tf.GradientTape() as tape:
        lgn_current = V1Column.calculate_input_current_from_spikes(
            lgn_cell, lgn_activity
        )
        lgn_loss = tf.reduce_sum(lgn_current)
    assert tape.gradient(lgn_loss, lgn_weights) is not None

    bkg_weights = tf.Variable(weights, dtype=tf.float32, trainable=True)
    bkg_cell = SimpleNamespace(
        _synaptic_current_backend="cuda",
        compute_dtype=tf.float32,
        variable_dtype=tf.float32,
        bkg_input_dense_shape=(3, 4),
        bkg_spike_prob=tf.constant(3.0),
        noise_seed=tf.Variable(17, trainable=False, dtype=tf.int64),
        bkg_input_weights=bkg_weights,
        synaptic_basis_weights=tf.constant(basis),
        bkg_input_csr=connectivity,
    )
    with tf.GradientTape() as tape:
        bkg_current = V1Column.calculate_noise_current(
            bkg_cell, tf.constant(2), tf.constant([11])
        )
        bkg_loss = tf.reduce_sum(bkg_current)
    assert tape.gradient(bkg_loss, bkg_weights) is not None


@pytest.mark.skipif(not tf.config.list_physical_devices("GPU"), reason="CUDA GPU required")
@pytest.mark.parametrize("activity_enabled", [False, True])
def test_v1_lgn_activity_gradient_flag_preserves_binary_forward(activity_enabled):
    indices, synapse_types, activity, weights, basis, _ = _fixture(2, 4)
    connectivity = _connectivity(indices, synapse_types, n_pre=4, n_post=3)
    cell = SimpleNamespace(
        _synaptic_current_backend="cuda",
        _compute_lgn_activity_gradient=activity_enabled,
        compute_dtype=tf.float32,
        variable_dtype=tf.float32,
        input_weight_values=tf.Variable(weights, trainable=False),
        synaptic_basis_weights=tf.constant(basis),
        input_csr=connectivity,
    )
    values = tf.constant(activity)
    binary_values = tf.cast(values > 0, tf.float32)
    with tf.GradientTape() as tape:
        tape.watch(values)
        output = V1Column.calculate_input_current_from_spikes(cell, values)
        loss = tf.reduce_sum(output)
    gradient = tape.gradient(loss, values)
    expected = calculate_external_csr_currents(
        binary_values,
        cell.input_weight_values,
        basis,
        connectivity,
        compute_activity_gradient=False,
        compute_weight_gradient=False,
    )
    np.testing.assert_allclose(output, expected, rtol=2e-5, atol=2e-5)
    assert (gradient is not None) is activity_enabled
