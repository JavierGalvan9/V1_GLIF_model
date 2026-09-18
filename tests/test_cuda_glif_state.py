"""Parity tests for the shared differentiable CUDA state-transition adapter."""

import pickle

import numpy as np
import pytest
import tensorflow as tf

from v1_model_utils import cuda_csr_recurrent, spatial_layout
from v1_model_utils.models import V1Column, _has_homogeneous_cuda_devices


def _cell(
    tmp_path, *, backend, dtype, hard_reset=False, pseudo_gauss=False,
    detach_reset=True, detach_asc_reset=False,
):
    tf.keras.mixed_precision.set_global_policy(
        "mixed_float16" if dtype == tf.float16 else "float32"
    )
    tf_data = tmp_path / "tf_data"
    tf_data.mkdir(parents=True, exist_ok=True)
    with (tf_data / "syn_id_to_syn_weights_dict.pkl").open("wb") as handle:
        pickle.dump({0: np.ones(4, np.float32)}, handle)
    params = {
        "V_th": np.ones(1, np.float32),
        "E_L": np.zeros(1, np.float32),
        "V_reset": np.zeros(1, np.float32),
        "g": np.ones(1, np.float32),
        "C_m": np.ones(1, np.float32),
        "t_ref": np.full(1, 3, np.float32),
        "k": np.array([[0.1, 0.2]], np.float32),
        "asc_amps": np.array([[0.03, -0.02]], np.float32),
    }
    network = {
        "data_dir": str(tmp_path),
        "n_nodes": 8,
        "node_type_ids": np.zeros(8, np.int32),
        "node_params": params,
        "synapses": {
            "indices": np.array([[0, 0], [1, 1]], np.int32),
            "weights": np.array([0.1, 0.2], np.float32),
            "delays": np.array([3, 1], np.float32),
            "syn_ids": np.zeros(2, np.uint8),
            "dense_shape": (8, 8),
        },
    }
    external = {
        "n_inputs": 2,
        "indices": np.array([[0, 0], [1, 1]], np.int32),
        "weights": np.array([0.1, 0.2], np.float32),
        "delays": np.ones(2, np.float32),
        "syn_ids": np.zeros(2, np.uint8),
    }
    # The CUDA kernels index weights by CSR position, so reorder the fixture's
    # edges the same way the training path reorders the loaded network.
    edge_orders = None
    if cuda_csr_recurrent.DIRECT_CSR:
        network, external, bkg, edge_orders = spatial_layout.apply_csr_edge_order(
            network, external, external
        )
    else:
        bkg = external
    return V1Column(
        network,
        external,
        bkg,
        edge_orders=edge_orders,
        batch_size=3,
        hard_reset=hard_reset,
        pseudo_gauss=pseudo_gauss,
        acceleration=backend,
        train_recurrent=False,
        train_input=False,
        train_noise=False,
        voltage_gradient_dampening=0.5,
        detach_reset=detach_reset,
        detach_asc_reset=detach_asc_reset,
    )


def _run(cell, inputs, state):
    watched = [tf.Variable(inputs)] + [tf.Variable(value) for value in state[:6]]
    arguments = tuple(watched[1:]) + state[6:]
    with tf.GradientTape() as tape:
        outputs, new_state = cell(watched[0], arguments)
        values = list(outputs) + [x for x in new_state if x.dtype.is_floating]
        loss = sum((i + 1) * tf.reduce_sum(tf.cast(x, tf.float32)) for i, x in enumerate(values))
    return outputs, new_state, tape.gradient(loss, watched)


def _previous_spike_gradient(cell, inputs, state, state_index):
    previous_spikes = tf.Variable(state[0])
    watched_state = (previous_spikes,) + tuple(state[1:])
    with tf.GradientTape() as tape:
        _, new_state = cell(inputs, watched_state)
        loss = tf.reduce_sum(tf.cast(new_state[state_index], tf.float32))
    return tape.gradient(loss, previous_spikes)


@pytest.mark.parametrize("dtype", [tf.float32, tf.float16])
@pytest.mark.parametrize("hard_reset", [False, True])
def test_cuda_matches_tensorflow_outputs_and_gradients(tmp_path, dtype, hard_reset):
    if not tf.config.list_physical_devices("GPU"):
        pytest.skip("requires CUDA")
    reference = _cell(tmp_path / "reference", backend="tensorflow", dtype=dtype, hard_reset=hard_reset)
    candidate = _cell(tmp_path / "candidate", backend="cuda", dtype=dtype, hard_reset=hard_reset)
    generator = tf.random.Generator.from_seed(713)
    inputs = generator.uniform((3, reference.input_dim), dtype=reference.compute_dtype)
    state = list(reference.zero_state(3))
    state[0] = tf.cast(generator.uniform(state[0].shape) > 0.7, reference.compute_dtype)
    state[1] = generator.uniform(state[1].shape, -0.2, 1.2, dtype=reference.compute_dtype)
    state[2] = tf.cast(generator.uniform(state[2].shape, 0, 4, dtype=tf.int32), reference._refractory_state_dtype)
    for index in (3, 4, 5):
        state[index] = generator.uniform(state[index].shape, -0.2, 0.2, dtype=reference.compute_dtype)
    expected = _run(reference, inputs, tuple(state))
    actual = _run(candidate, inputs, tuple(state))
    tolerance = 3e-3 if dtype == tf.float16 else 3e-6
    for expected_group, actual_group in zip(expected, actual):
        for want, got in zip(expected_group, actual_group):
            if want is None or got is None:
                assert want is got
            else:
                tf.debugging.assert_near(tf.cast(got, tf.float32), tf.cast(want, tf.float32), atol=tolerance, rtol=tolerance)


@pytest.mark.parametrize("hard_reset", [False, True])
def test_cuda_matches_tensorflow_reset_and_asc_gradients(tmp_path, hard_reset):
    if not tf.config.list_physical_devices("GPU"):
        pytest.skip("requires CUDA")
    reference = _cell(
        tmp_path / "reference", backend="tensorflow", dtype=tf.float32,
        hard_reset=hard_reset,
    )
    candidate = _cell(
        tmp_path / "candidate", backend="cuda", dtype=tf.float32,
        hard_reset=hard_reset,
    )
    state = list(reference.zero_state(3))
    previous_spikes = tf.constant([[1.0, 0.0] * 4] * 3)
    state[0] = tf.concat(
        [previous_spikes, tf.zeros((3, state[0].shape[1] - 8))], axis=1
    )
    state[1] = tf.fill(state[1].shape, tf.constant(0.4, tf.float32))
    state[3] = tf.fill(state[3].shape, tf.constant(0.1, tf.float32))
    inputs = tf.ones((3, reference.input_dim), tf.float32)

    expected = _run(reference, inputs, tuple(state))
    actual = _run(candidate, inputs, tuple(state))

    expected_z_gradient = expected[2][1]
    actual_z_gradient = actual[2][1]
    tf.debugging.assert_near(actual_z_gradient, expected_z_gradient, atol=3e-6, rtol=3e-6)
    assert tf.reduce_any(tf.not_equal(expected_z_gradient[:, :8], 0.0))


@pytest.mark.parametrize("backend", ["tensorflow", "cuda"])
def test_voltage_reset_is_detached_but_asc_spike_path_is_differentiable(
    tmp_path, backend
):
    if backend == "cuda" and not tf.config.list_physical_devices("GPU"):
        pytest.skip("requires CUDA")
    cell = _cell(tmp_path, backend=backend, dtype=tf.float32)
    state = list(cell.zero_state(3))
    state[0] = tf.concat(
        [tf.ones((3, 8)), tf.zeros((3, state[0].shape[1] - 8))], axis=1
    )
    inputs = tf.zeros((3, cell.input_dim), tf.float32)

    voltage_gradient = _previous_spike_gradient(cell, inputs, state, state_index=1)
    asc_gradient = _previous_spike_gradient(cell, inputs, state, state_index=3)

    if voltage_gradient is not None:
        tf.debugging.assert_equal(voltage_gradient, tf.zeros_like(voltage_gradient))
    tf.debugging.assert_near(
        asc_gradient[:, :8],
        tf.broadcast_to(tf.reduce_sum(cell.asc_amps, axis=-1), (3, 8)),
    )


@pytest.mark.parametrize("backend", ["tensorflow", "cuda"])
def test_attached_reset_restores_voltage_gradient_to_previous_spike(tmp_path, backend):
    if backend == "cuda" and not tf.config.list_physical_devices("GPU"):
        pytest.skip("requires CUDA")
    cell = _cell(
        tmp_path, backend=backend, dtype=tf.float32, detach_reset=False
    )
    state = list(cell.zero_state(3))
    state[0] = tf.concat(
        [tf.ones((3, 8)), tf.zeros((3, state[0].shape[1] - 8))], axis=1
    )
    inputs = tf.zeros((3, cell.input_dim), tf.float32)

    voltage_gradient = _previous_spike_gradient(cell, inputs, state, state_index=1)

    tf.debugging.assert_near(voltage_gradient[:, :8], -tf.ones((3, 8)))


@pytest.mark.parametrize("backend", ["tensorflow", "cuda"])
def test_detached_asc_reset_removes_only_asc_gradient_to_previous_spike(
    tmp_path, backend
):
    if backend == "cuda" and not tf.config.list_physical_devices("GPU"):
        pytest.skip("requires CUDA")
    cell = _cell(
        tmp_path,
        backend=backend,
        dtype=tf.float32,
        detach_reset=False,
        detach_asc_reset=True,
    )
    state = list(cell.zero_state(3))
    state[0] = tf.concat(
        [tf.ones((3, 8)), tf.zeros((3, state[0].shape[1] - 8))], axis=1
    )
    inputs = tf.zeros((3, cell.input_dim), tf.float32)

    voltage_gradient = _previous_spike_gradient(cell, inputs, state, state_index=1)
    asc_gradient = _previous_spike_gradient(cell, inputs, state, state_index=3)

    tf.debugging.assert_near(voltage_gradient[:, :8], -tf.ones((3, 8)))
    if asc_gradient is not None:
        tf.debugging.assert_equal(asc_gradient, tf.zeros_like(asc_gradient))


def test_auto_uses_cuda_for_pseudo_gaussian(tmp_path):
    # The CUDA backward kernel implements the gaussian surrogate gradient, so
    # `pseudo_gauss=True` no longer forces the state update back to TensorFlow.
    cell = _cell(tmp_path, backend="auto", dtype=tf.float32, pseudo_gauss=True)
    expected = "cuda" if _has_homogeneous_cuda_devices() else "tensorflow"
    assert cell.resolved_acceleration == expected


def test_auto_rejects_mixed_compute_capabilities(monkeypatch):
    devices = (object(), object())
    monkeypatch.setattr(tf.config, "list_physical_devices", lambda kind: devices)
    monkeypatch.setattr(
        tf.config.experimental,
        "get_device_details",
        lambda device: {"compute_capability": (8, 9) if device is devices[0] else (12, 0)},
    )
    assert not _has_homogeneous_cuda_devices()


@pytest.mark.skipif(not tf.config.list_physical_devices("GPU"), reason="CUDA GPU required")
def test_explicit_cuda_supports_pseudo_gaussian(tmp_path):
    cell = _cell(tmp_path, backend="cuda", dtype=tf.float32, pseudo_gauss=True)
    assert cell.resolved_acceleration == "cuda"
    assert cell._surrogate_gradient == "gaussian"


@pytest.mark.parametrize("hard_reset", [False, True])
def test_backward_kernel_ignores_the_refractory_state_under_soft_reset(hard_reset):
    """The fused backward reads the refractory state only under hard reset.

    The kernel uses it in one expression, to mask the voltage gradient when the
    reset is hard. `_dense_state` relies on that: under soft reset it passes
    zeros so the forward tensor never enters the backward graph, which stops
    TensorFlow accumulating the per-timestep history. That history has no GPU
    TensorList kernel for its dtype and would be staged through host memory,
    costing about 21% of a batch-32 training step.

    This pins the kernel contract that makes the substitution safe.
    """
    if not tf.config.list_physical_devices("GPU"):
        pytest.skip("requires CUDA")
    from v1_model_utils.cuda_glif_state import wrapper as glif_wrapper

    glif_ops, _ = glif_wrapper._load_ops()
    rng = np.random.default_rng(97)
    neurons, batch, basis = 48, 4, 4

    def normal(*shape):
        return tf.constant(rng.normal(0, 1, shape).astype(np.float32))

    z = tf.constant((rng.random((batch, neurons)) < 0.2).astype(np.float32))
    arguments = dict(
        asc=normal(batch, neurons * 2),
        rise=normal(batch, neurons * basis),
        syn_decay=tf.constant(rng.uniform(0.5, 0.99, neurons * basis).astype(np.float32)),
        psc_initial=tf.constant(rng.uniform(0.5, 2.0, neurons * basis).astype(np.float32)),
        asc_decay=tf.constant(rng.uniform(0.5, 0.99, neurons * 2).astype(np.float32)),
        decay=tf.constant(rng.uniform(0.8, 0.99, neurons).astype(np.float32)),
        current_factor=tf.constant(rng.uniform(0.5, 2.0, neurons).astype(np.float32)),
        t_ref_steps=tf.constant(rng.integers(2, 5, neurons).astype(np.int8)),
        asc_amps=tf.constant(rng.normal(0, 0.05, neurons * 2).astype(np.float32)),
        dt=tf.constant(1.0),
        gv=normal(batch, neurons),
        ga=normal(batch, neurons * 2),
        grise=normal(batch, neurons * basis),
        gpsc=normal(batch, neurons * basis),
    )

    def backward(refractory):
        return glif_ops.fused_glif_single_backward(
            z, tf.cast(refractory, tf.int8), arguments["asc"], arguments["rise"],
            arguments["syn_decay"], arguments["psc_initial"], arguments["asc_decay"],
            arguments["decay"], arguments["current_factor"], arguments["t_ref_steps"],
            arguments["asc_amps"], arguments["dt"], arguments["gv"], arguments["ga"],
            arguments["grise"], arguments["gpsc"],
            hard_reset=hard_reset, detach_reset=True, detach_asc_reset=False,
        )

    quiescent = backward(tf.zeros((batch, neurons), tf.int32))
    refractory = backward(tf.fill((batch, neurons), tf.constant(4, tf.int32)))
    deltas = [
        float(tf.reduce_max(tf.abs(a - b))) for a, b in zip(quiescent, refractory)
    ]
    if hard_reset:
        assert max(deltas) > 0.0, (
            "hard reset masks the voltage gradient, so the refractory state must "
            "still change the backward result"
        )
    else:
        assert max(deltas) == 0.0, (
            "the soft-reset backward result changed with the refractory state, so "
            "_dense_state may no longer substitute zeros for it"
        )
