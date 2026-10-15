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
from v1_model_utils.models import V1Column, poisson_cdf_levels, sample_poisson_counts


RUNTIME_BATCH_SIZES = (3, 5, 9)


def _csr_sorted(indices, synapse_types, weights):
    """Put edges in the external operator's CSR order.

    Its sort key omits the synapse type, unlike the recurrent operator's.
    """
    order = np.lexsort((np.arange(indices.shape[0]), indices[:, 0], indices[:, 1]))
    return indices[order], synapse_types[order], weights[order]


def _connectivity(indices, synapse_types, n_pre, n_post, **kwargs):
    """Build connectivity, declaring the CSR ordering the kernels require."""
    return build_csr_connectivity(
        indices, synapse_types, n_pre=n_pre, n_post=n_post,
        weights_csr_ordered=DIRECT_CSR, **kwargs,
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
@pytest.mark.parametrize("sparse_activity", [True, False])
def test_full_forward_and_backward(batch_size, n_basis, sparse_activity):
    fixture = _fixture(batch_size, n_basis)
    indices, synapse_types, activity, weights, basis, upstream = fixture
    # Event-driven (sparse) and dense pair-projected weight gradients.
    connectivity = _connectivity(
        indices, synapse_types, n_pre=4, n_post=3, sparse_activity=sparse_activity
    )
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
    # FP32 accumulation order (atomic batch tiles on the dense path) leaves
    # round-off proportional to the magnitude of the summed terms.
    np.testing.assert_allclose(weight_grad.numpy(), expected[2], rtol=2e-5,
                               atol=2e-6 * np.abs(expected[2]).max())


@pytest.mark.skipif(not tf.config.list_physical_devices("GPU"), reason="CUDA GPU required")
@pytest.mark.parametrize("n_basis", [3, 4, 7])
@pytest.mark.parametrize("batch_size", SPECIALIZED_BATCH_SIZES + RUNTIME_BATCH_SIZES)
@pytest.mark.parametrize("sparse_activity", [True, False])
def test_weight_only_backward(batch_size, n_basis, sparse_activity):
    fixture = _fixture(batch_size, n_basis)
    indices, synapse_types, activity, weights, basis, upstream = fixture
    connectivity = _connectivity(
        indices, synapse_types, n_pre=4, n_post=3, sparse_activity=sparse_activity,
        needs_activity_backward=False,
    )
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
    # FP32 accumulation order (atomic batch tiles on the dense path) leaves
    # round-off proportional to the magnitude of the summed terms.
    np.testing.assert_allclose(weight_grad.numpy(), expected[2], rtol=2e-5,
                               atol=2e-6 * np.abs(expected[2]).max())


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


def _lgn_like_fixture(batch_size, n_basis, n_pre, n_types=4):
    """FP16-exact LGN-like input: rows of 65-280 edges (mean ~170, beyond the
    64-edge chunk), each target hit twice, and ~30 edges per target."""
    rng = np.random.default_rng(batch_size + 7 * n_basis + n_pre)
    n_post = n_pre * 170 // 30
    pre = np.repeat(np.arange(n_pre), 2 * rng.integers(33, 140, n_pre))
    post = np.repeat(rng.integers(0, n_post, pre.size // 2), 2)
    types = rng.integers(0, n_types, pre.size)
    weights = rng.normal(size=pre.size).astype(np.float32)
    indices, types, weights = _csr_sorted(np.stack((post, pre), axis=1), types, weights)
    activity = (rng.random((batch_size, n_pre)) < 0.3) * rng.integers(1, 3, (batch_size, n_pre))
    basis = rng.normal(size=(n_types, n_basis)).astype(np.float32)
    upstream = rng.normal(size=(batch_size * n_post, n_basis)).astype(np.float16)
    return indices, types, activity.astype(np.float16), weights, basis, upstream


def _float64_reference(indices, types, activity, weights, basis, upstream):
    """Currents and both gradients through sparse per-type weight matrices."""
    from scipy import sparse

    post, pre = indices.T
    batch_size, n_pre = activity.shape
    n_post = upstream.shape[0] // batch_size
    activity = activity.astype(np.float64)
    basis = basis.astype(np.float64)
    projected = upstream.reshape(batch_size, n_post, -1).astype(np.float64) @ basis.T
    currents = np.zeros((batch_size, n_post, basis.shape[1]))
    activity_grad = np.zeros_like(activity)
    for synapse_type, vector in enumerate(basis):
        edges = types == synapse_type
        matrix = sparse.csr_array(
            (weights[edges].astype(np.float64), (post[edges], pre[edges])),
            shape=(n_post, n_pre),
        )
        currents += (matrix @ activity.T).T[..., None] * vector
        activity_grad += (matrix.T @ projected[..., synapse_type].T).T
    weight_grad = sum(a[pre] * p[post, types] for a, p in zip(activity, projected))
    return currents.reshape(-1, basis.shape[1]), activity_grad, weight_grad


@pytest.mark.skipif(not tf.config.list_physical_devices("GPU"), reason="CUDA GPU required")
@pytest.mark.parametrize("n_pre", [40, 4600], ids=["row_split", "chunk_loop"])
@pytest.mark.parametrize("sparse_activity", [True, False])
@pytest.mark.parametrize("n_basis", [4, 5])
@pytest.mark.parametrize("batch_size", [1, 3, 5, 32, 33, 64])
def test_fp16_long_rows_match_float64_oracle(batch_size, n_basis, sparse_activity, n_pre):
    # Under 4,512 rows each row is split across blocks; from there on one block
    # loops over the row's 64-edge chunks. Together with the batches and bases
    # this reaches the fp16 compact, packed, generic and direct kernels.
    fixture = _lgn_like_fixture(batch_size, n_basis, n_pre)
    indices, types, activity, weights, basis, upstream = fixture
    connectivity = _connectivity(
        indices, types, n_pre=n_pre, n_post=upstream.shape[0] // batch_size,
        sparse_activity=sparse_activity,
    )
    assert connectivity.repeats_targets
    activity_tensor = tf.constant(activity)
    master_weights = tf.Variable(weights)
    with tf.GradientTape() as tape:
        tape.watch(activity_tensor)
        currents = calculate_external_csr_currents(
            activity_tensor, master_weights, basis, connectivity
        )
        loss = tf.reduce_sum(tf.cast(currents, tf.float32) * upstream.astype(np.float32))
    activity_grad, weight_grad = tape.gradient(loss, (activity_tensor, master_weights))
    expected = _float64_reference(*fixture)
    currents, activity_grad = currents.numpy(), activity_grad.numpy()
    # The inputs are fp16-exact. The forward sums each run of a repeated target
    # in fp32 but adds it with an fp16 atomic, one rounding per run (~15 here,
    # as in the LGN): measured <= 1.5e-3 of the largest current, the LGN figure
    # in cuda_csr_recurrent/README.md. The backward accumulates in fp32, so the
    # fp16 activity gradient only carries its final rounding (2**-11; 2**-10
    # allows one more ulp) and the fp32 weight gradient fp32 round-off.
    np.testing.assert_allclose(currents, expected[0], rtol=0,
                               atol=4e-3 * np.abs(expected[0]).max())
    np.testing.assert_allclose(activity_grad, expected[1], rtol=2**-10,
                               atol=1e-5 * np.abs(expected[1]).max())
    np.testing.assert_allclose(weight_grad.numpy(), expected[2], rtol=1e-5,
                               atol=1e-6 * np.abs(expected[2]).max())


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
        bkg_count_cdf=tf.constant(poisson_cdf_levels(3.0)),
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
def test_v1_lgn_activity_gradient_flag_uses_spikes_as_is(activity_enabled):
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
    values = tf.cast(tf.constant(activity) > 0, tf.float32)
    with tf.GradientTape() as tape:
        tape.watch(values)
        output = V1Column.calculate_input_current_from_spikes(cell, values)
        loss = tf.reduce_sum(output)
    gradient = tape.gradient(loss, values)
    expected = calculate_external_csr_currents(
        values,
        cell.input_weight_values,
        basis,
        connectivity,
        compute_activity_gradient=False,
        compute_weight_gradient=False,
    )
    np.testing.assert_allclose(output, expected, rtol=2e-5, atol=2e-5)
    assert (gradient is not None) is activity_enabled


@pytest.mark.parametrize("rate", [0.0, 0.25, 3.0, 40.0])
def test_poisson_cdf_levels_reach_one_and_match_scipy(rate):
    from scipy import stats

    levels = poisson_cdf_levels(rate)
    assert np.all(np.diff(levels) >= 0)
    assert levels[-1] >= 1.0 - 2.0**-52
    np.testing.assert_allclose(
        levels, stats.poisson.cdf(np.arange(len(levels)), rate), rtol=0, atol=1e-12
    )


@pytest.mark.parametrize(
    "device", ["/CPU:0"] + (["/GPU:0"] if tf.config.list_physical_devices("GPU") else [])
)
def test_bkg_counts_are_poisson_and_fixed_by_the_seed(device):
    from scipy import stats

    rate = 0.25  # the 250 Hz background at dt = 1 ms
    levels = tf.constant(poisson_cdf_levels(rate))
    with tf.device(device):
        draws = [
            sample_poisson_counts(levels, (32, 100), tf.constant([3000, step]), tf.float16)
            for step in range(300)
        ]
        repeat = sample_poisson_counts(levels, (32, 100), tf.constant([3000, 0]), tf.float16)
    # Placement falls back to the host silently when a GPU kernel is missing.
    assert all(draw.device.endswith(device[1:]) for draw in draws)
    counts = np.stack([draw.numpy() for draw in draws]).astype(np.int64)
    np.testing.assert_array_equal(repeat.numpy(), counts[0])
    assert not np.array_equal(counts[0], counts[1])

    pmf = stats.poisson.pmf(np.arange(3), rate)
    expected = np.append(pmf, 1 - pmf.sum()) * counts.size
    observed = np.bincount(counts.ravel(), minlength=4)
    observed = np.append(observed[:3], observed[3:].sum())
    assert stats.chisquare(observed, expected).pvalue > 1e-4


def _history_and_lgn(batch, dtype, n_post=6000, width=2000, n_slots=3, n_lgn=300, n_basis=4):
    """An `n_slots` recurrent history and an LGN-like input (rows repeating
    targets) onto the same neurons, in their operators' CSR orders."""
    from v1_model_utils.cuda_csr_recurrent import build_csr_connectivity as build_recurrent

    rng = np.random.default_rng(batch + n_post)
    n_pre = n_slots * width
    flat = np.unique(rng.integers(0, n_pre * n_post, 300_000))
    rec_indices = np.stack((flat % n_post, flat // n_post), axis=1)
    rec_types = rng.integers(0, 5, flat.size)
    order = np.lexsort((np.arange(flat.size), rec_types, rec_indices[:, 0], rec_indices[:, 1]))
    rec_indices, rec_types = rec_indices[order], rec_types[order]
    recurrent = build_recurrent(rec_indices, rec_types, n_pre=n_pre, n_post=n_post,
                                weights_csr_ordered=DIRECT_CSR)
    edges = []
    for pre in range(n_lgn):
        targets = np.unique(rng.integers(0, n_post, 800))
        posts = np.repeat(targets, rng.integers(1, 4, targets.size))
        edges.append(np.stack((posts, np.full_like(posts, pre)), axis=1))
    lgn_indices = np.concatenate(edges).astype(np.int64)
    lgn_types = rng.integers(0, 5, len(lgn_indices))
    lgn_weights = rng.normal(size=len(lgn_indices)).astype(np.float32)
    lgn_indices, lgn_types, lgn_weights = _csr_sorted(lgn_indices, lgn_types, lgn_weights)
    lgn = _connectivity(lgn_indices, lgn_types, n_lgn, n_post)
    spikes = (rng.random((batch, n_pre)) < 0.05).astype(dtype.as_numpy_dtype)
    activity = (rng.random((batch, n_lgn)) < 0.2).astype(dtype.as_numpy_dtype)
    return SimpleNamespace(
        recurrent=recurrent, rec_indices=rec_indices, rec_types=rec_types,
        rec_weights=rng.normal(size=len(rec_indices)).astype(np.float32),
        lgn=lgn, lgn_indices=lgn_indices, lgn_types=lgn_types, lgn_weights=lgn_weights,
        slots=tuple(tf.split(tf.constant(spikes), n_slots, axis=1)), spikes=spikes,
        activity=tf.constant(activity),
        basis=rng.normal(size=(5, n_basis)).astype(np.float32),
        upstream=tf.constant(rng.normal(size=(batch * n_post, n_basis)).astype(np.float32)),
        n_post=n_post, width=width,
    )


def _two_sources(case, fused):
    """Currents, record and every gradient of the recurrent + LGN step, through
    the fused wrapper or through the two operators chained as the model did."""
    from v1_model_utils.cuda_csr_external import calculate_recurrent_and_external_currents
    from v1_model_utils.cuda_csr_recurrent import (
        calculate_recurrent_csr_currents,
        spike_queue,
    )

    rec_weights, lgn_weights = tf.Variable(case.rec_weights), tf.Variable(case.lgn_weights)
    queues = tuple(spike_queue(slot) for slot in case.slots[1:])
    basis = tf.constant(case.basis)
    with tf.GradientTape() as tape:
        tape.watch((case.slots, case.activity))
        if fused:
            currents, record = calculate_recurrent_and_external_currents(
                case.slots, rec_weights, basis, 0.37, case.recurrent, queues,
                case.activity, lgn_weights, case.lgn,
            )
        else:
            currents, record = calculate_recurrent_csr_currents(
                case.slots, rec_weights, basis, 0.37, case.recurrent, queues=queues
            )
            currents = calculate_external_csr_currents(
                case.activity, lgn_weights, basis, case.lgn, initial=currents
            )
        loss = tf.reduce_sum(tf.cast(currents, tf.float32) * case.upstream)
    grads = tape.gradient(loss, (case.slots, rec_weights, case.activity, lgn_weights))
    return currents.numpy(), record.numpy(), grads


def _two_source_reference(case):
    n_basis = case.basis.shape[1]

    def source(indices, types, weights, values):
        out = np.zeros((values.shape[0], case.n_post, n_basis))
        for sample in range(values.shape[0]):
            drive = values[sample, indices[:, 1]].astype(np.float64) * weights
            for column in range(n_basis):
                out[sample, :, column] = np.bincount(
                    indices[:, 0], drive * case.basis[types, column], minlength=case.n_post)
        return out.reshape(-1, n_basis)

    return (source(case.rec_indices, case.rec_types, case.rec_weights, case.spikes)
            + source(case.lgn_indices, case.lgn_types, case.lgn_weights,
                     case.activity.numpy()))


@pytest.mark.skipif(not tf.config.list_physical_devices("GPU"), reason="CUDA GPU required")
@pytest.mark.parametrize("resource_mode", ["0", "1"])
@pytest.mark.parametrize("batch", [1, 8, 64])
def test_fused_recurrent_and_lgn_match_the_two_operators(monkeypatch, resource_mode, batch):
    """FP16, narrow tiles (batch 1 and 8 on these 6,000 neurons) and wide ones
    (64), with tensor connectivities (V1CsrForwardFused) and GPU-local resources
    (V1CsrForwardFusedResource): the fused sweep rounds once, so it is at least
    as accurate as the chained operators; its gradients come from the same
    backward operators and are bitwise equal (the activity gradient up to that
    operator's own run-to-run spread), and so is the queue record."""
    from v1_model_utils.cuda_csr_external import fusable

    monkeypatch.setenv("V1_CSR_RESOURCE_MODE", resource_mode)
    case = _history_and_lgn(batch, tf.float16)
    assert (case.recurrent.resource_name is not None) == (resource_mode == "1")
    assert fusable(case.slots, case.basis, case.recurrent, (), case.activity, case.lgn)
    fused, fused_record, fused_grads = _two_sources(case, fused=True)
    chained, chained_record, chained_grads = _two_sources(case, fused=False)
    reference = _two_source_reference(case)
    fused_error = np.abs(fused.astype(np.float64) - reference)
    chained_error = np.abs(chained.astype(np.float64) - reference)
    assert fused_error.max() <= chained_error.max()
    assert fused_error.mean() <= chained_error.mean()
    capacity = batch * case.width
    count = int(chained_record[capacity])
    np.testing.assert_array_equal(fused_record[:count], chained_record[:count])
    np.testing.assert_array_equal(fused_record[capacity:], chained_record[capacity:])
    (fused_slots, fused_rec, fused_activity, fused_lgn) = fused_grads
    (chained_slots, chained_rec, chained_activity, chained_lgn) = chained_grads
    for got, want in zip((*fused_slots, fused_rec, fused_lgn),
                         (*chained_slots, chained_rec, chained_lgn)):
        np.testing.assert_array_equal(got.numpy(), want.numpy())
    # The activity backward itself orders its FP16 atomics nondeterministically.
    np.testing.assert_allclose(fused_activity.numpy().astype(np.float32),
                               chained_activity.numpy().astype(np.float32), rtol=2e-3, atol=0.1)


@pytest.mark.skipif(not tf.config.list_physical_devices("GPU"), reason="CUDA GPU required")
@pytest.mark.parametrize("resource_mode", ["0", "1"])
@pytest.mark.parametrize("batch, dtype, n_slots, n_basis", [
    (5, tf.float16, 3, 4), (64, tf.float32, 3, 4), (130, tf.float32, 3, 4),
    (130, tf.float16, 3, 4), (64, tf.float16, 2, 4), (64, tf.float16, 3, 3),
])
def test_fused_operator_matches_float64_on_every_path(
    monkeypatch, resource_mode, batch, dtype, n_slots, n_basis
):
    """The fused wrapper (tensor or resource connectivities) and the fused
    operator called directly agree with fp64 on the sweep's configurations
    (every batch, FP16 and FP32, with three slots and four basis columns) and
    off them, where the operator runs the two forwards in sequence (two slots,
    three basis columns), as do the chained operators."""
    from v1_model_utils.cuda_csr_external import fusable
    from v1_model_utils.cuda_csr_external.wrapper import _load_ops
    from v1_model_utils.cuda_csr_recurrent import spike_queue

    monkeypatch.setenv("V1_CSR_RESOURCE_MODE", resource_mode)
    case = _history_and_lgn(batch, dtype, n_slots=n_slots, n_basis=n_basis)
    assert fusable(case.slots, case.basis, case.recurrent, (), case.activity, case.lgn)
    fused, _, _ = _two_sources(case, fused=True)
    chained, _, _ = _two_sources(case, fused=False)
    reference = _two_source_reference(case)
    recurrent_ops, _ = _load_ops()
    direct, _ = recurrent_ops.v1_csr_forward_fused(
        case.slots, tf.constant(case.rec_weights), case.recurrent.post_ids,
        case.recurrent.synapse_types, case.recurrent.row_splits, case.recurrent.edge_ids,
        tf.constant(case.basis), case.activity, tf.constant(case.lgn_weights),
        case.lgn.post_ids, case.lgn.synapse_types, case.lgn.row_splits, case.lgn.edge_ids,
        tuple(spike_queue(slot) for slot in case.slots[1:]), n_post=case.n_post,
        aggregate_runs=case.recurrent.repeats_targets,
        external_aggregate_runs=case.lgn.repeats_targets,
    )
    tolerance = 2e-3 if dtype == tf.float16 else 1e-6
    for currents in (fused, chained, direct.numpy()):
        error = np.linalg.norm(currents.astype(np.float64) - reference)
        assert error / np.linalg.norm(reference) < tolerance


@pytest.mark.skipif(not tf.config.list_physical_devices("GPU"), reason="CUDA GPU required")
@pytest.mark.parametrize("batch, dtype", [(1, tf.float16), (64, tf.float16), (8, tf.float32)])
def test_fused_resource_operator_matches_the_tensor_operator(monkeypatch, batch, dtype):
    """Replicated runs read both connectivities from GPU-local resources
    (V1CsrForwardFusedResource): the same sweep as V1CsrForwardFused, so the
    same queue record, every gradient bitwise (the activity gradient up to its
    operator's own run-to-run spread) and currents within one rounding."""
    outputs = {}
    for mode in ("0", "1"):
        monkeypatch.setenv("V1_CSR_RESOURCE_MODE", mode)
        case = _history_and_lgn(batch, dtype)
        assert (case.recurrent.resource_name is not None) == (mode == "1")
        assert (case.lgn.resource_name is not None) == (mode == "1")
        outputs[mode] = _two_sources(case, fused=True)
    (tensor, tensor_record, tensor_grads), (resource, resource_record, resource_grads) = (
        outputs["0"], outputs["1"])
    capacity = batch * case.width
    count = int(tensor_record[capacity])
    np.testing.assert_array_equal(resource_record[:count], tensor_record[:count])
    np.testing.assert_array_equal(resource_record[capacity:], tensor_record[capacity:])
    (tensor_slots, tensor_rec, tensor_activity, tensor_lgn) = tensor_grads
    (resource_slots, resource_rec, resource_activity, resource_lgn) = resource_grads
    for got, want in zip((*resource_slots, resource_rec, resource_lgn),
                         (*tensor_slots, tensor_rec, tensor_lgn)):
        np.testing.assert_array_equal(got.numpy(), want.numpy())
    np.testing.assert_allclose(resource_activity.numpy().astype(np.float32),
                               tensor_activity.numpy().astype(np.float32), rtol=2e-3, atol=0.1)
    # The tile's shared-memory sums may add in another order: one rounding of
    # the output apart, plus FP32 slack for results near zero.
    ulp = np.spacing(np.abs(tensor).astype(dtype.as_numpy_dtype)).astype(np.float64)
    assert np.all(np.abs(resource.astype(np.float64) - tensor) <= ulp + 1e-4)


@pytest.mark.skipif(not tf.config.list_physical_devices("GPU"), reason="CUDA GPU required")
def test_fused_step_with_a_dynamic_batch():
    """A graph traced with an unknown batch takes the fused operator, which picks
    its path from the run-time batch: the currents equal the eager ones."""
    from v1_model_utils.cuda_csr_external import calculate_recurrent_and_external_currents
    from v1_model_utils.cuda_csr_recurrent import spike_queue

    case = _history_and_lgn(8, tf.float16)
    rec_weights, lgn_weights = tf.constant(case.rec_weights), tf.constant(case.lgn_weights)
    slot = tf.TensorSpec([None, case.width], tf.float16)

    def step(slots, activity):
        queues = tuple(spike_queue(value) for value in slots[1:])
        return calculate_recurrent_and_external_currents(
            slots, rec_weights, tf.constant(case.basis), 0.37, case.recurrent, queues,
            activity, lgn_weights, case.lgn,
        )[0]

    traced = tf.function(step, input_signature=(
        (slot,) * 3, tf.TensorSpec([None, case.activity.shape[1]], tf.float16)))
    reference = _two_source_reference(case)
    for currents in (traced(case.slots, case.activity), step(case.slots, case.activity)):
        error = np.abs(currents.numpy().astype(np.float64) - reference)
        # Rounded once from FP32 sums: within half an FP16 ulp, plus FP32 slack.
        bound = np.spacing(np.abs(reference).astype(np.float16)).astype(np.float64) / 2
        assert np.all(error <= bound + 1e-4)
