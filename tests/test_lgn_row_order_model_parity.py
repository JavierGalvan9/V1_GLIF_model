"""A retinotopically reordered LGN must drive identical V1 input currents.

The row order is a relabelling of the presynaptic side, and ``V1Column``
compensates by gathering the incoming spike stream. Feeding both models the
same canonical stream must therefore produce the same postsynaptic current -
the only test that pins the gather direction inside the cell, since a reversed
gather is still a valid permutation and breaks nothing structurally.

The observable is the LGN input current rather than the cell's voltage: one
timestep of input leaves the membrane at rest, so a voltage comparison passes
whatever the permutation and proves nothing.
"""

import os
import pickle as pkl

import numpy as np
import pytest
import tensorflow as tf

from v1_model_utils import cuda_csr_recurrent, models, spatial_layout

NETWORK_PICKLE = os.path.join(
    "GLIF_network_nll_full", "tf_data", "V1_network_v1_1000.pkl"
)
BATCH = 2
MAX_DELAY = 3
pytestmark = pytest.mark.skipif(
    not os.path.exists(NETWORK_PICKLE),
    reason="cached 1000-neuron network required",
)


def _load():
    """Return the cached triple from either cache format.

    ``load_sparse`` wraps the payload in a versioned dict; older caches on disk
    are the bare triple.
    """
    with open(NETWORK_PICKLE, "rb") as stream:
        cached = pkl.load(stream)
    return cached["payload"] if isinstance(cached, dict) else cached


def _prepare(network, lgn_input, bkg_input, layout):
    """Apply the neuron layout and the CSR edge order the kernels require."""
    if not layout.is_identity:
        network, lgn_input, bkg_input = spatial_layout.apply_layout(
            layout, network, lgn_input, bkg_input
        )
    edge_orders = None
    if cuda_csr_recurrent.DIRECT_CSR:
        network, lgn_input, bkg_input, edge_orders = (
            spatial_layout.apply_csr_edge_order(
                network, lgn_input, bkg_input, max_delay=MAX_DELAY
            )
        )
    return network, lgn_input, bkg_input, edge_orders


def _cell(network, lgn_input, bkg_input, layout, acceleration, edge_orders,
          row_order=None, lgn_activity_gradient=False):
    return models.V1Column(
        network,
        lgn_input,
        bkg_input,
        batch_size=BATCH,
        max_delay=MAX_DELAY,
        train_recurrent=True,
        train_input=False,
        train_noise=True,
        compute_lgn_activity_gradient=lgn_activity_gradient,
        noise_seed=3,
        acceleration=acceleration,
        neuron_layout=layout,
        edge_orders=edge_orders,
        lgn_row_order=row_order,
    )


def _build(network, lgn_input, bkg_input, row_mode, acceleration,
           lgn_activity_gradient=False):
    """Build a cell whose LGN rows are numbered in ``row_mode``."""
    tf.keras.utils.set_random_seed(11)
    layout = spatial_layout.build_layout(network, spatial_layout.CANONICAL)
    row_order = spatial_layout.build_lgn_row_order(lgn_input, row_mode)
    lgn_input = spatial_layout.apply_lgn_row_order(row_order, lgn_input)
    network, lgn_input, bkg_input, edge_orders = _prepare(
        network, lgn_input, bkg_input, layout
    )
    cell = _cell(
        network, lgn_input, bkg_input, layout, acceleration, edge_orders,
        row_order=row_order, lgn_activity_gradient=lgn_activity_gradient,
    )
    return cell, row_order, edge_orders


def _input_current(cell, stream):
    """The postsynaptic current the cell derives from a canonical stream."""
    permuted = cell._permute_lgn_input(stream)
    return np.asarray(
        tf.cast(cell.calculate_input_current_from_spikes(permuted), tf.float32)
    )


def _stream(n_input, density=0.5, seed=5):
    """A dense stream, so a wrong permutation cannot pass by hitting zeros."""
    rng = np.random.default_rng(seed)
    return tf.constant((rng.random((BATCH, n_input)) < density).astype(np.float32))


def test_retinotopic_rows_reproduce_original_currents():
    network, lgn_input, bkg_input = _load()
    stream = _stream(int(lgn_input["n_inputs"]))

    reference, _, _ = _build(
        network, lgn_input, bkg_input, spatial_layout.LGN_ORIGINAL, "tensorflow"
    )
    moved, row_order, _ = _build(
        network, lgn_input, bkg_input, spatial_layout.LGN_RETINOTOPIC,
        "tensorflow",
    )
    assert not row_order.is_identity, "fixture must exercise a real permutation"

    expected = _input_current(reference, stream)
    actual = _input_current(moved, stream)
    assert np.abs(expected).max() > 0, "observable must not be identically zero"
    np.testing.assert_allclose(actual, expected, rtol=2e-3, atol=2e-3)


def test_a_reversed_gather_changes_the_result():
    """Guard the guard: the parity test above must be capable of failing."""
    network, lgn_input, bkg_input = _load()
    stream = _stream(int(lgn_input["n_inputs"]))

    reference, _, _ = _build(
        network, lgn_input, bkg_input, spatial_layout.LGN_ORIGINAL, "tensorflow"
    )
    moved, row_order, _ = _build(
        network, lgn_input, bkg_input, spatial_layout.LGN_RETINOTOPIC,
        "tensorflow",
    )
    expected = _input_current(reference, stream)
    # Deliberately the wrong direction.
    moved._lgn_row_gather = tf.constant(row_order.old_to_new, dtype=tf.int32)
    wrong = _input_current(moved, stream)
    assert not np.allclose(wrong, expected, rtol=2e-3, atol=2e-3)


def test_omitting_the_gather_changes_the_result():
    """And an unpermuted stream must not silently look correct either."""
    network, lgn_input, bkg_input = _load()
    stream = _stream(int(lgn_input["n_inputs"]))

    reference, _, _ = _build(
        network, lgn_input, bkg_input, spatial_layout.LGN_ORIGINAL, "tensorflow"
    )
    moved, _, _ = _build(
        network, lgn_input, bkg_input, spatial_layout.LGN_RETINOTOPIC,
        "tensorflow",
    )
    expected = _input_current(reference, stream)
    moved._lgn_row_gather = None
    assert not np.allclose(
        _input_current(moved, stream), expected, rtol=2e-3, atol=2e-3
    )


def test_checkpoint_keeps_lgn_row_ids_canonical():
    """``input_indices`` column one is an LGN row id and must be translated.

    The variable is also edge-aligned, so ``_translate_edge_layout`` permutes
    its rows at the same time; undo that permutation before comparing labels.
    """
    if not cuda_csr_recurrent.DIRECT_CSR:
        pytest.skip("edge translation only applies with DIRECT_CSR")
    network, lgn_input, bkg_input = _load()
    cell, row_order, edge_orders = _build(
        network, lgn_input, bkg_input, spatial_layout.LGN_RETINOTOPIC,
        "tensorflow",
    )
    if getattr(cell, "input_indices", None) is None:
        pytest.skip("this backend does not retain input_indices")

    runtime = cell.input_indices.numpy().copy()
    cell.translate_checkpointed_layout(to_runtime=False)
    on_disk = spatial_layout.to_csr_edges(
        cell.input_indices.numpy(), edge_orders["lgn"]
    )
    # Postsynaptic ids stay put: the neuron layout here is the identity.
    np.testing.assert_array_equal(on_disk[:, 0], runtime[:, 0])
    np.testing.assert_array_equal(
        on_disk[:, 1], row_order.new_to_old[runtime[:, 1]]
    )
    assert not np.array_equal(on_disk[:, 1], runtime[:, 1])

    cell.translate_checkpointed_layout(to_runtime=True)
    np.testing.assert_array_equal(cell.input_indices.numpy(), runtime)


def test_identity_order_leaves_the_stream_alone():
    network, lgn_input, bkg_input = _load()
    cell, row_order, _ = _build(
        network, lgn_input, bkg_input, spatial_layout.LGN_ORIGINAL, "tensorflow"
    )
    assert row_order.is_identity
    assert cell._lgn_row_gather is None
    stream = _stream(int(lgn_input["n_inputs"]))
    assert cell._permute_lgn_input(stream) is stream


@pytest.mark.skipif(
    not tf.config.list_physical_devices("GPU"),
    reason="only the CUDA backend differentiates the LGN activity",
)
def test_gradient_flows_back_through_the_permutation():
    """The LGN activity gradient must come back in canonical row order.

    ``_permute_lgn_input`` uses ``tf.gather``, whose gradient is the matching
    scatter, so the gradient arriving at the canonical stream must line up with
    the gradient an unpermuted model produces. Permuting the stream outside the
    graph instead would silently hand back a gradient in runtime order.

    The TensorFlow backend selects active inputs with ``tf.where``, which is
    not differentiable in the stream, so this property only exists on the CUDA
    backend with ``compute_lgn_activity_gradient``.
    """
    network, lgn_input, bkg_input = _load()
    stream = _stream(int(lgn_input["n_inputs"]))

    def activity_gradient(row_mode):
        cell, row_order, _ = _build(
            network, lgn_input, bkg_input, row_mode, "cuda",
            lgn_activity_gradient=True,
        )
        watched = tf.identity(stream)
        with tf.GradientTape() as tape:
            tape.watch(watched)
            current = cell.calculate_input_current_from_spikes(
                cell._permute_lgn_input(watched)
            )
            total = tf.reduce_sum(tf.cast(current, tf.float32) ** 2)
        gradient = tape.gradient(total, watched)
        return None if gradient is None else np.asarray(gradient), row_order

    reference, _ = activity_gradient(spatial_layout.LGN_ORIGINAL)
    moved, row_order = activity_gradient(spatial_layout.LGN_RETINOTOPIC)
    if reference is None or moved is None:
        pytest.skip("this configuration does not differentiate the LGN stream")
    assert np.abs(reference).max() > 0, "gradient must not be identically zero"
    assert not row_order.is_identity
    np.testing.assert_allclose(moved, reference, rtol=2e-3, atol=2e-3)
