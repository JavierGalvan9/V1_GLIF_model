"""Parity tests for the shared differentiable CUDA state-transition adapter."""

import pickle
from types import SimpleNamespace

import numpy as np
import pytest
import tensorflow as tf

from v1_model_utils import cuda_csr_recurrent, spatial_layout
from v1_model_utils.models import V1Column, _has_homogeneous_cuda_devices
from v1_model_utils.cuda_glif_state.wrapper import update_glif_state


@pytest.fixture(autouse=True)
def _restore_global_policy():
    """`_cell` sets the global Keras policy; keep it from leaking to other files."""
    policy = tf.keras.mixed_precision.global_policy()
    yield
    tf.keras.mixed_precision.set_global_policy(policy)


def _cell(
    tmp_path, *, backend, dtype, hard_reset=False, pseudo_gauss=False,
    detach_reset=True, detach_asc_reset=False, integration_scheme="exact",
    t_ref=3, heterogeneous=False, recurrent_delays=(3, 1), **column_options,
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
        "t_ref": np.full(1, t_ref, np.float32),
        "k": np.array([[0.1, 0.2]], np.float32),
        "asc_amps": np.array([[0.03, -0.02]], np.float32),
    }
    node_type_ids = np.zeros(8, np.int32)
    if heterogeneous:
        # Two cell types that differ in every parameter, so a neuron-aligned
        # constant left behind by a permutation changes value.
        params = {
            "V_th": np.ones(2, np.float32),
            "E_L": np.zeros(2, np.float32),
            "V_reset": np.zeros(2, np.float32),
            "g": np.array([1.0, 2.5], np.float32),
            "C_m": np.array([1.0, 40.0], np.float32),
            "t_ref": np.array([t_ref, t_ref + 4], np.float32),
            "k": np.array([[0.1, 0.2], [0.003, 0.3]], np.float32),
            "asc_amps": np.array([[0.03, -0.02], [-0.1, 0.4]], np.float32),
        }
        node_type_ids = np.array([0, 1, 1, 0, 1, 0, 0, 1], np.int32)
    network = {
        "data_dir": str(tmp_path),
        "n_nodes": 8,
        "node_type_ids": node_type_ids,
        "node_params": params,
        "synapses": {
            "indices": np.array([[0, 0], [1, 1]], np.int32),
            "weights": np.array([0.1, 0.2], np.float32),
            "delays": np.array(recurrent_delays, np.float32),
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
        integration_scheme=integration_scheme,
        **column_options,
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


@pytest.mark.parametrize("basis_dim", [1, 2, 5])
@pytest.mark.parametrize("dtype", [tf.float32, tf.float16])
def test_cuda_glif_generic_basis_forward_and_gradients(basis_dim, dtype):
    """The model adapter must use the runtime-basis path outside four bases."""
    if not tf.config.list_physical_devices("GPU"):
        pytest.skip("requires CUDA")
    batch, neurons, slots = 2, 3, 3
    coefficients = np.zeros((neurons, basis_dim, 4), np.float32)
    coefficients[..., 0] = 0.8
    coefficients[..., 1] = 0.3
    coefficients[..., 2] = 0.2
    coefficients[..., 3] = 0.4
    cell = SimpleNamespace(
        syn_coeffs=tf.constant(coefficients),
        asc_decay=tf.zeros((2 * neurons,), tf.float32),
        asc_amps=tf.zeros((2 * neurons,), tf.float32),
        decay=tf.ones((neurons,), tf.float32),
        asc_factor=tf.zeros((2 * neurons,), tf.float32),
        reset_coeff=tf.zeros((neurons,), tf.float32),
        asc_spike_factor=tf.zeros((neurons,), tf.float32),
        t_ref_steps=tf.zeros((neurons,), tf.int8),
        _dt=tf.constant(0.5),
        v_reset=tf.constant(0.0),
        v_th=tf.constant(10.0),
        _gauss_std=tf.constant(1.0),
        _dampening_factor=tf.constant(1.0),
        _return_voltage_sequences=False,
        _hard_reset=False,
        _surrogate_gradient="triangular",
        _detach_reset=True,
        _detach_asc_reset=True,
    )
    state_shape = (batch, neurons * basis_dim)
    rise = tf.Variable(tf.fill(state_shape, tf.cast(0.25, dtype)))
    psc = tf.Variable(tf.fill(state_shape, tf.cast(0.5, dtype)))
    inputs = tf.Variable(tf.fill(state_shape, tf.cast(0.75, dtype)))
    with tf.GradientTape() as tape:
        spikes, _, state = update_glif_state(
            tf.zeros((batch, slots * neurons), dtype),
            tf.zeros((batch, neurons), tf.float32),
            tf.zeros((batch, neurons), tf.int8),
            tf.zeros((batch, 2 * neurons), tf.float32),
            rise, psc, inputs, cell=cell,
        )
        loss = sum(tf.reduce_sum(tf.cast(state[index], tf.float32)) for index in (1, 4, 5))
    rise_grad, psc_grad, input_grad = tape.gradient(loss, (rise, psc, inputs))
    np.testing.assert_array_equal(spikes.numpy(), np.zeros_like(spikes.numpy()))
    np.testing.assert_allclose(state[1].numpy(), 0.2 * basis_dim,
                               rtol=0, atol=2e-6)
    for actual, expected in ((state[4], 0.425), (state[5], 0.5),
                             (rise_grad, 1.6), (psc_grad, 1.0),
                             (input_grad, 0.3)):
        np.testing.assert_allclose(actual.numpy(), expected,
                                   rtol=0, atol=2e-3 if dtype == tf.float16 else 2e-6)


@pytest.mark.parametrize("dtype", [tf.float32, tf.float16])
@pytest.mark.parametrize("hard_reset", [False, True])
@pytest.mark.parametrize("t_ref", [3, 1])
def test_cuda_matches_tensorflow_outputs_and_gradients(tmp_path, dtype, hard_reset, t_ref):
    _assert_backends_match(tmp_path, dtype, hard_reset=hard_reset, t_ref=t_ref)


@pytest.mark.parametrize("dtype", [tf.float32, tf.float16])
@pytest.mark.parametrize("surrogate_gradient", ["triangular", "gaussian", "slayer"])
@pytest.mark.parametrize(
    "options",
    [
        {},
        {"return_voltage_sequences": False, "track_voltage_penalty": True},
        {"output_neuron_ids": np.array([1, 4, 6])},
        {"recurrent_delays": (1, 1)},
    ],
    ids=["voltage-sequences", "online-voltage-penalty", "output-subset", "one-delay-slot"],
)
def test_fused_step_matches_tensorflow_for_every_surrogate_and_output(
    tmp_path, dtype, surrogate_gradient, options
):
    """The fused op thresholds, shifts the history and exposes the voltage.

    Each surrogate is evaluated inside the fused backward; the exposed voltage
    is written by the forward only when voltage sequences are returned; the
    online penalty sends a voltage gradient that bypasses the exposed sequence;
    and a single delay slot leaves no older history to shift.
    """
    _assert_backends_match(
        tmp_path, dtype, surrogate_gradient=surrogate_gradient, **options
    )


def _assert_backends_match(tmp_path, dtype, **cell_options):
    if not tf.config.list_physical_devices("GPU"):
        pytest.skip("requires CUDA")
    reference = _cell(tmp_path / "reference", backend="tensorflow", dtype=dtype,
                      **cell_options)
    candidate = _cell(tmp_path / "candidate", backend="cuda", dtype=dtype,
                      **cell_options)
    generator = tf.random.Generator.from_seed(713)
    # The spike-input contract is binary; the TensorFlow reference treats
    # every positive entry as one spike, while CUDA consumes its value.
    inputs = tf.cast(
        generator.uniform((3, reference.input_dim), dtype=reference.compute_dtype) > 0.5,
        reference.compute_dtype,
    )
    state = list(reference.zero_state(3))
    state[0] = tf.cast(generator.uniform(state[0].shape) > 0.7, reference.compute_dtype)
    state[1] = generator.uniform(state[1].shape, -0.2, 1.2, dtype=state[1].dtype)
    state[2] = tf.cast(generator.uniform(state[2].shape, 0, 4, dtype=tf.int32), reference._refractory_state_dtype)
    for index in (3, 4, 5):
        state[index] = generator.uniform(state[index].shape, -0.2, 0.2, dtype=state[index].dtype)
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
@pytest.mark.parametrize("hard_reset", [False, True])
@pytest.mark.parametrize("t_ref", [3, 1])
def test_a_spike_resets_the_membrane_at_the_start_of_its_step(
    tmp_path, backend, hard_reset, t_ref
):
    """The reset acts at the spike time and the step then propagates it.

    A soft reset subtracts the threshold gap; a hard reset puts the membrane at
    `v_reset`. Either way the value decays across the step, and a hard reset
    then holds it while refractoriness lasts. With a one-step refractory period
    nothing is held, so this is the only place the hard reset shows: it used to
    fall back to the soft one there and keep the overshoot.
    """
    if backend == "cuda" and not tf.config.list_physical_devices("GPU"):
        pytest.skip("requires CUDA")
    cell = _cell(tmp_path, backend=backend, dtype=tf.float32,
                 hard_reset=hard_reset, t_ref=t_ref)
    state = list(cell.zero_state(3))
    state[0] = tf.concat([tf.ones((3, 8)), tf.zeros((3, state[0].shape[1] - 8))], 1)
    state[1] = tf.fill(state[1].shape, 1.3)
    _, new_state = cell(tf.zeros((3, cell.input_dim)), tuple(state))

    start = cell.v_reset if hard_reset else 1.3 - 1.0
    expected = cell.decay * start + cell.asc_spike_factor
    if hard_reset and t_ref > 1:
        expected = tf.broadcast_to(cell.v_reset, expected.shape)
    tf.debugging.assert_near(new_state[1], tf.broadcast_to(expected, (3, 8)),
                             atol=1e-6, rtol=1e-6)


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

    # The reset is detached, so what is left on the membrane is precisely the
    # drive from the ASC this spike injects - which is the differentiable path.
    tf.debugging.assert_near(
        voltage_gradient[:, :8],
        tf.broadcast_to(cell.asc_spike_factor[:8], (3, 8)),
    )
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

    # Both paths are attached: the reset and the ASC this spike injects.
    tf.debugging.assert_near(
        voltage_gradient[:, :8],
        tf.broadcast_to(cell.reset_coeff[:8] + cell.asc_spike_factor[:8], (3, 8)),
    )


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

    # detach_asc_reset is set, so only the reset reaches the membrane.
    tf.debugging.assert_near(
        voltage_gradient[:, :8], tf.broadcast_to(cell.reset_coeff[:8], (3, 8))
    )
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


DERIVED_COEFFICIENTS = (
    "decay", "asc_decay", "asc_amps", "syn_coeffs", "asc_factor",
    "reset_coeff", "asc_spike_factor", "t_ref_steps", "current_factor",
)


def test_derived_constants_are_untracked_variables(tmp_path):
    """Every tensor the cell holds is a variable, and none of them is saved.

    An eager constant lives on one GPU, so a replica on another re-reads it
    across devices on every timestep; a variable created under the strategy
    scope is mirrored instead. They are derived from the node parameters, so
    they are not state, and keeping them out of the checkpoint is also what
    lets a checkpoint written before they existed still restore: the restore
    path asserts that every tracked variable found a value. Keras 3 leaves raw
    tf.Variables out of ``cell.variables``, so the checkpoint keys are the
    check.
    """
    cell = _cell(tmp_path, backend="tensorflow", dtype=tf.float32)
    eager = sorted(
        name for name, value in vars(cell).items() if isinstance(value, tf.Tensor)
    )
    assert not eager, f"eager tensors bound on the cell: {eager}"

    path = tf.train.Checkpoint(model=cell).save(str(tmp_path / "checkpoint"))
    keys = [key for key, _ in tf.train.list_variables(path)]
    derived = set(DERIVED_COEFFICIENTS) | set(cell._neuron_constants) | {
        "v_th", "v_reset", "synaptic_basis_weights", "_dt", "_gauss_std",
        "_dampening_factor", "bkg_count_cdf",
    }
    leaked = sorted(
        name for name in derived
        if any(f"/{name}/.ATTRIBUTES" in key for key in keys)
    )
    assert not leaked, f"derived constants in the checkpoint: {leaked}"


@pytest.mark.skipif(
    len(tf.config.list_physical_devices("GPU")) < 2, reason="two GPUs required"
)
def test_every_replica_reads_the_cell_constants_from_its_own_gpu(tmp_path):
    """Under a two-replica strategy each derived constant has a copy per GPU.

    An eager constant would live on GPU:0 only, and the other replica would
    fetch it across devices on every timestep of the time loop.
    """
    devices = [device.name for device in tf.config.list_logical_devices("GPU")[:2]]
    strategy = tf.distribute.MirroredStrategy(devices)
    with strategy.scope():
        cell = _cell(tmp_path, backend="tensorflow", dtype=tf.float32)
    names = list(cell._neuron_constants) + [
        "v_th", "v_reset", "synaptic_basis_weights", "_dt", "_gauss_std",
        "_dampening_factor", "bkg_count_cdf",
    ]
    masks = {
        f"{name}.constraint": getattr(cell, name).constraint.condition
        for name in ("recurrent_weight_values", "input_weight_values", "bkg_input_weights")
    }
    for name, value in [(name, getattr(cell, name)) for name in names] + list(masks.items()):
        placed = {component.device for component in strategy.experimental_local_results(value)}
        assert len(placed) == 2, f"{name} is not replica-local: {placed}"


@pytest.mark.parametrize("written,restored", [("euler", "exact"), ("exact", "euler")])
def test_a_checkpoint_restores_across_integration_schemes(tmp_path, written, restored):
    """The scheme changes constants only, so checkpoints stay interchangeable."""
    source = _cell(tmp_path / "source", backend="tensorflow", dtype=tf.float32,
                   integration_scheme=written)
    target = _cell(tmp_path / "target", backend="tensorflow", dtype=tf.float32,
                   integration_scheme=restored)
    # save() rather than write() so the checkpoint carries the save_counter the
    # restoring Checkpoint object also creates.
    path = tf.train.Checkpoint(model=source).save(str(tmp_path / "checkpoint"))
    status = tf.train.Checkpoint(model=target).restore(path)
    status.assert_existing_objects_matched()
    status.expect_partial()


@pytest.mark.parametrize("backend", ["tensorflow", "cuda"])
def test_the_two_detach_flags_stay_independent(tmp_path, backend):
    """detach_reset must not touch the ASC path, and vice versa.

    Under the exact propagator a spike does two things to its own step: it
    resets the membrane, and it injects an after-spike current that drives the
    membrane across that step. Both are constants multiplying the same spike, so
    it is tempting to fold them together - but they are gated by different
    flags, and folding them makes detach_reset silently cut up to a third of the
    ASC gradient.
    """
    if backend == "cuda" and not tf.config.list_physical_devices("GPU"):
        pytest.skip("requires CUDA")

    def voltage_gradient(detach_reset, detach_asc_reset):
        cell = _cell(
            tmp_path / f"{backend}-{detach_reset}-{detach_asc_reset}",
            backend=backend, dtype=tf.float32,
            detach_reset=detach_reset, detach_asc_reset=detach_asc_reset,
        )
        state = list(cell.zero_state(3))
        state[0] = tf.concat(
            [tf.ones((3, 8)), tf.zeros((3, state[0].shape[1] - 8))], axis=1
        )
        inputs = tf.zeros((3, cell.input_dim), tf.float32)
        gradient = _previous_spike_gradient(cell, inputs, state, state_index=1)
        gradient = (tf.zeros((3, cell._n_neurons)) if gradient is None
                    else gradient[:, :8])
        return cell, gradient

    cell, both = voltage_gradient(False, False)
    _, reset_only = voltage_gradient(False, True)
    _, asc_only = voltage_gradient(True, False)
    _, neither = voltage_gradient(True, True)

    reset = tf.broadcast_to(cell.reset_coeff[:8], (3, 8))
    asc = tf.broadcast_to(cell.asc_spike_factor[:8], (3, 8))
    tf.debugging.assert_near(both, reset + asc, atol=3e-6)
    tf.debugging.assert_near(reset_only, reset, atol=3e-6)
    tf.debugging.assert_near(asc_only, asc, atol=3e-6)
    tf.debugging.assert_equal(neither, tf.zeros_like(neither))
    # The ASC term is not negligible, so conflating the flags would matter.
    assert float(tf.reduce_max(tf.abs(asc))) > 0.0


def _permuted_layout(n_nodes, seed=3):
    """A non-identity neuron layout, built the way morton() builds one."""
    rng = np.random.default_rng(seed)
    new_to_old = rng.permutation(n_nodes).astype(np.uint32)
    old_to_new = np.empty_like(new_to_old)
    old_to_new[new_to_old] = np.arange(n_nodes, dtype=np.uint32)
    return spatial_layout.NeuronLayout(
        mode=spatial_layout.MORTON, new_to_old=new_to_old, old_to_new=old_to_new
    )


def test_layout_translation_leaves_the_per_neuron_constants_in_place(tmp_path):
    """The constants are rebuilt in runtime order, so no translation moves them.

    The network is relabelled before the column is built, and the constants
    are never checkpointed, so a restore has nothing canonical to bring back.
    Translating them on restore - as the registry once did - gave every
    neuron another neuron's coefficients.
    """
    cell = _cell(tmp_path, backend="tensorflow", dtype=tf.float32,
                 heterogeneous=True)
    cell._neuron_layout = _permuted_layout(cell._n_neurons)

    # Everything per-neuron the step reads must be registered.
    reads = {"decay", "syn_coeffs", "asc_factor", "reset_coeff",
             "asc_spike_factor", "asc_decay", "asc_amps", "t_ref_steps"}
    missing = reads - set(cell._neuron_constants)
    assert not missing, f"not registered: {sorted(missing)}"

    before = {name: getattr(cell, name).numpy().copy()
              for name in cell._neuron_constants}
    type_ids_before = np.asarray(cell._node_type_ids).copy()
    order = cell._neuron_layout.new_to_old
    # With two cell types a permuted constant differs from the original, so
    # the equality checks below would notice one being moved.
    for name in reads:
        assert not np.array_equal(before[name], before[name][order]), name

    for to_runtime in (True, False):
        cell._translate_neuron_layout(to_runtime=to_runtime)
        np.testing.assert_array_equal(cell._node_type_ids, type_ids_before)
        for name, original in before.items():
            np.testing.assert_array_equal(
                getattr(cell, name).numpy(), original,
                err_msg=f"{name} moved when translating to_runtime={to_runtime}")


@pytest.mark.parametrize("backend", ["tensorflow", "cuda"])
@pytest.mark.parametrize("dtype", [tf.float32, tf.float16])
def test_membrane_asc_and_constants_are_float32_under_any_policy(
    tmp_path, backend, dtype
):
    """Only the synaptic state and the spikes follow the compute dtype.

    float16 rounding of the decay factors shifts the simulated time constants,
    and the slow ASCs accumulate float16 error for hundreds of milliseconds,
    so the membrane, the ASCs and every propagator constant stay float32. The
    exposed voltage sequence keeps the compute dtype of the spikes it is packed
    with.
    """
    if backend == "cuda" and not tf.config.list_physical_devices("GPU"):
        pytest.skip("requires CUDA")
    cell = _cell(tmp_path, backend=backend, dtype=dtype)
    assert cell.compute_dtype == dtype.name
    for name in cell._neuron_constants:
        expected = cell._refractory_state_dtype if name == "t_ref_steps" else tf.float32
        assert getattr(cell, name).dtype == expected, name
    state = cell.zero_state(3)
    expected_state = (dtype, tf.float32, cell._refractory_state_dtype, tf.float32,
                      dtype, dtype)
    assert tuple(x.dtype for x in state[:6]) == expected_state
    outputs, new_state = cell(tf.ones((3, cell.input_dim), dtype), state)
    assert tuple(x.dtype for x in new_state[:6]) == expected_state
    assert outputs.dtype == dtype


def test_asc_decay_is_evaluated_in_float64_and_rounded_once(tmp_path):
    """The ASC decay used to go through a float16 inverse-sigmoid round trip.

    That returned k with a relative error of 2e-3 under a mixed-precision
    policy, which compounds to a sustained bias on the slowest adaptation.
    """
    cell = _cell(tmp_path, backend="tensorflow", dtype=tf.float16,
                 heterogeneous=True)
    k = np.array([[0.1, 0.2], [0.003, 0.3]], np.float32)[cell._node_type_ids]
    k = k.astype(np.float64)
    np.testing.assert_array_equal(
        cell.asc_decay.numpy(), np.exp(-1.0 * k).astype(np.float32)
    )
