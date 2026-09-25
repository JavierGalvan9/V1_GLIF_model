"""A relabelled V1Column must compute the same dynamics as a canonical one.

The Morton layout is only a renumbering, so running both layouts from the same
inputs and mapping the relabelled outputs back must reproduce the canonical
result up to floating-point summation order.
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
pytestmark = pytest.mark.skipif(
    not os.path.exists(NETWORK_PICKLE), reason="cached 1000-neuron network required"
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
                network, lgn_input, bkg_input, max_delay=3
            )
        )
    return network, lgn_input, bkg_input, edge_orders


def _cell(network, lgn_input, bkg_input, layout, acceleration, edge_orders=None):
    return models.V1Column(
        network,
        lgn_input,
        bkg_input,
        batch_size=BATCH,
        max_delay=3,
        train_recurrent=True,
        train_input=False,
        train_noise=True,
        noise_seed=3,
        acceleration=acceleration,
        neuron_layout=layout,
        edge_orders=edge_orders,
    )


# Synaptic delay and the refractory state mean the first two timesteps leave
# the membrane exactly at rest, so a single step compares zeros to zeros and
# passes under any permutation. Six steps put the network well into spiking.
STEPS = 6


def _run(network, lgn_input, bkg_input, layout, acceleration, lgn_spikes):
    """Advance the cell ``STEPS`` timesteps and return its last outputs."""
    tf.keras.utils.set_random_seed(11)
    network, lgn_input, bkg_input, edge_orders = _prepare(
        network, lgn_input, bkg_input, layout
    )
    cell = _cell(
        network, lgn_input, bkg_input, layout, acceleration, edge_orders
    )
    state = cell.zero_state(BATCH, tf.float32)
    for _ in range(STEPS):
        outputs, state = cell(lgn_spikes, state)
    spikes, voltage = tf.split(outputs, 2, axis=-1)
    return np.asarray(spikes), np.asarray(voltage)


@pytest.mark.parametrize(
    "acceleration",
    ["tensorflow"] + (
        ["cuda"] if tf.config.list_physical_devices("GPU") else []
    ),
)
def test_morton_layout_reproduces_canonical_dynamics(acceleration):
    network, lgn_input, bkg_input = _load()
    n_input = int(lgn_input["n_inputs"])

    rng = np.random.default_rng(5)
    # Dense enough to drive the column to threshold within STEPS.
    lgn_spikes = tf.constant(
        (rng.random((BATCH, n_input)) < 0.5).astype(np.float32)
    )

    canonical = spatial_layout.build_layout(network, spatial_layout.CANONICAL)
    reference_spikes, reference_voltage = _run(
        network, lgn_input, bkg_input, canonical, acceleration, lgn_spikes
    )

    layout = spatial_layout.build_layout(network, spatial_layout.MORTON)
    moved_spikes, moved_voltage = _run(
        network, lgn_input, bkg_input, layout, acceleration, lgn_spikes
    )

    # Map the relabelled neurons back onto canonical positions.
    restored_spikes = moved_spikes[..., layout.old_to_new]
    restored_voltage = moved_voltage[..., layout.old_to_new]

    assert restored_spikes.shape == reference_spikes.shape
    # Without this the comparison below can pass on all-zero dynamics, which
    # any permutation satisfies.
    assert reference_spikes.any(), "reference produced no spikes to compare"
    assert np.abs(reference_voltage).max() > 0, "reference voltage never moved"
    np.testing.assert_allclose(
        restored_voltage, reference_voltage, rtol=2e-3, atol=2e-3
    )
    np.testing.assert_array_equal(restored_spikes, reference_spikes)


def test_checkpoint_translation_round_trips_weights_and_optimizer_slots():
    """Checkpoints are written in the network's original edge order.

    The runtime keeps weights in CSR order, so a save has to un-permute them and
    a restore has to permute them back - including the optimizer slots that
    mirror each weight vector.
    """
    if not cuda_csr_recurrent.DIRECT_CSR:
        pytest.skip("edge translation only applies with DIRECT_CSR")
    network, lgn_input, bkg_input = _load()
    layout = spatial_layout.build_layout(network, spatial_layout.CANONICAL)
    prepared_net, prepared_lgn, prepared_bkg, edge_orders = _prepare(
        network, lgn_input, bkg_input, layout
    )
    cell = _cell(
        prepared_net, prepared_lgn, prepared_bkg, layout, "tensorflow", edge_orders
    )

    names = {
        "recurrent_weight_values": "recurrent",
        "input_weight_values": "lgn",
        "bkg_input_weights": "bkg",
    }
    runtime = {n: getattr(cell, n).numpy().copy() for n in names}
    # Stand-in optimizer slots, one per weight vector, matched on length.
    slots = {n: tf.Variable(v.copy()) for n, v in runtime.items()}
    optimizer = type("Opt", (), {"variables": lambda self: tuple(slots.values())})()

    cell.translate_checkpointed_layout(to_runtime=False, optimizer=optimizer)
    for name, population in names.items():
        order = edge_orders[population]
        np.testing.assert_array_equal(
            getattr(cell, name).numpy(),
            spatial_layout.to_original_edges(runtime[name], order),
            err_msg=f"{name} was not written in the network's edge order",
        )
        np.testing.assert_array_equal(
            slots[name].numpy(),
            spatial_layout.to_original_edges(runtime[name], order),
            err_msg=f"optimizer slot for {name} did not follow its weights",
        )
    # The recurrent order is a real permutation here, so values must have moved.
    assert not np.array_equal(
        cell.recurrent_weight_values.numpy(), runtime["recurrent_weight_values"]
    )

    cell.translate_checkpointed_layout(to_runtime=True, optimizer=optimizer)
    for name in names:
        np.testing.assert_array_equal(
            getattr(cell, name).numpy(), runtime[name], err_msg=name
        )
        np.testing.assert_array_equal(
            slots[name].numpy(), runtime[name], err_msg=f"slot {name}"
        )


def test_ambiguous_optimizer_slot_lengths_are_refused():
    """Length matching must not silently guess when two populations collide."""
    if not cuda_csr_recurrent.DIRECT_CSR:
        pytest.skip("edge translation only applies with DIRECT_CSR")
    order = np.arange(8, dtype=np.uint32)
    collided = tf.Variable(np.zeros(8, np.float32))

    class Stub(models.V1Column):
        def __init__(self):  # bypass the full network construction
            self._edge_orders = {"recurrent": order, "lgn": order}

        def edge_weight_variables(self):
            yield "recurrent", collided, order
            yield "lgn", collided, order

    optimizer = type("Opt", (), {"variables": lambda self: ()})()
    with pytest.raises(ValueError, match="same edge count"):
        Stub()._optimizer_slots_by_length(optimizer)
