"""CUDA GLIF state op parity at production widths, with the fused BKG gather
and over multi-step rollouts, plus the op's input validation."""

import pickle
from types import SimpleNamespace

import numpy as np
import pytest
import tensorflow as tf

from v1_model_utils import cuda_csr_recurrent, models, spatial_layout
from v1_model_utils.models import HeterogeneousStateRNN, V1Column
from v1_model_utils.cuda_glif_state.wrapper import update_glif_state

pytestmark = pytest.mark.skipif(
    not tf.config.list_physical_devices("GPU"), reason="requires CUDA"
)


@pytest.fixture(autouse=True)
def _restore_global_policy():
    policy = tf.keras.mixed_precision.global_policy()
    yield
    tf.keras.mixed_precision.set_global_policy(policy)


def _op_cell(neurons, basis, dtype, penalty="range", seed=0):
    """The attributes update_glif_state reads, with per-neuron random constants."""
    rng = np.random.default_rng(seed)

    def uniform(low, high, *shape):
        return tf.constant(rng.uniform(low, high, shape).astype(np.float32))

    coefficients = np.stack([rng.uniform(*bounds, (neurons, basis)) for bounds in
                             ((0.2, 0.99), (0.1, 3.0), (0.0, 0.2), (0.0, 0.1))], -1)
    return SimpleNamespace(
        syn_coeffs=tf.constant(coefficients.astype(np.float32)),
        asc_decay=uniform(0.5, 1, neurons, 2), asc_amps=uniform(-0.3, 0.3, neurons, 2),
        decay=uniform(0.8, 1, neurons), asc_factor=uniform(0, 0.2, neurons, 2),
        reset_coeff=uniform(-1, -0.8, neurons), asc_spike_factor=uniform(-0.05, 0.05, neurons),
        t_ref_steps=tf.constant(rng.integers(1, 5, neurons).astype(np.int8)),
        _dt=tf.constant(1.0), v_reset=tf.constant(0.0), v_th=tf.constant(1.0),
        _gauss_std=tf.constant(0.5), _dampening_factor=tf.constant(0.3),
        _return_voltage_sequences=True, _hard_reset=False,
        _surrogate_gradient="triangular", _detach_reset=False, _detach_asc_reset=False,
        _voltage_penalty_mode=penalty, _n_neurons=neurons, _n_syn_basis=basis,
        compute_dtype=dtype,
    )


def _op_state(cell, batch, n_slots=3, seed=1):
    """(slots, v, r, asc, rise, psc, inputs, penalty_acc) for one step."""
    rng = np.random.default_rng(seed)
    neurons, width, dtype = cell._n_neurons, cell._n_syn_basis * cell._n_neurons, cell.compute_dtype
    slots = tuple(tf.constant(rng.uniform(size=(batch, neurons)) > 0.7, dtype)
                  for _ in range(n_slots))
    return (
        slots,
        tf.constant(rng.uniform(-0.3, 1.4, (batch, neurons)).astype(np.float32)),
        tf.constant(rng.integers(0, 3, (batch, neurons)).astype(np.int8)),
        tf.constant(rng.uniform(-0.2, 0.2, (batch, 2 * neurons)).astype(np.float32)),
        *(tf.constant(rng.uniform(-0.2, 0.2, (batch, width)), dtype) for _ in range(3)),
        tf.constant(rng.uniform(0, 1, (batch, 1)).astype(np.float32)),
    )


def _reference_step(cell, slots, v, r, asc, rise, psc, inputs, penalty_acc):
    """The TensorFlow backend's step, in update_glif_state's output layout."""
    new_v, new_r, new_asc, new_rise, new_psc = V1Column._dense_update_impl(
        cell, v.shape[0], slots[0], v, r, asc, rise, psc, inputs)
    z = models.spike_function(new_v - cell.v_th, cell._dampening_factor)
    z = tf.cast(tf.where(new_r > 0, tf.zeros_like(z), z), cell.compute_dtype)
    penalty = models.compute_voltage_penalty_mean_step(
        new_v, cell._n_neurons, cell._voltage_penalty_mode)
    return (z, tf.cast(new_v, cell.compute_dtype),
            ((z,) + slots[:-1], new_v, new_r, new_asc, new_rise, new_psc),
            penalty_acc + penalty[:, None])


def _op_outputs_and_gradients(step, cell, state):
    slots, v, r, *rest = state
    watched = [tf.Variable(x) for x in slots + (v, *rest)]
    slots, (v, asc, rise, psc, inputs, penalty_acc) = watched[:len(slots)], watched[len(slots):]
    with tf.GradientTape() as tape:
        if step is update_glif_state:
            z, exposed, new_state, new_acc = step(
                tuple(slots), v, r, asc, rise, psc, inputs, cell=cell, penalty_acc=penalty_acc)
        else:
            z, exposed, new_state, new_acc = step(
                cell, tuple(slots), v, r, asc, rise, psc, inputs, penalty_acc)
        history, *new_state = new_state
        outputs = [z, exposed, *history, *new_state, new_acc]
        # Distinct weights per output, so a gradient routed to the wrong one shows.
        loss = sum((i + 1) * tf.reduce_sum(tf.cast(x, tf.float32) * tf.cast(
            tf.range(tf.size(x)) % 7 - 3, tf.float32)[: x.shape[-1]])
            for i, x in enumerate(outputs) if x.dtype.is_floating)
    return outputs, tape.gradient(loss, watched)


@pytest.mark.parametrize("dtype", [tf.float32, tf.float16])
@pytest.mark.parametrize("basis", [3, 4, 5])
@pytest.mark.parametrize("penalty", ["range", "threshold"])
def test_production_width_op_matches_tensorflow(dtype, basis, penalty):
    """At ~70k neurons and an odd batch the forward runs its grid-stride loop
    and tiled basis-3/5 path, and the penalty reduces over many blocks."""
    cell = _op_cell(70001, basis, dtype, penalty)
    state = _op_state(cell, batch=3)
    expected = _op_outputs_and_gradients(_reference_step, cell, state)
    actual = _op_outputs_and_gradients(update_glif_state, cell, state)
    # The reference sums the float16 spike gradients in float16: one ulp of
    # terms near 8 shows on a cancelled sum near 0.2.
    tolerance = 1e-2 if dtype == tf.float16 else 1e-5
    for want, got in zip(sum(expected, []), sum(actual, [])):
        if want is None or got is None:  # the dropped oldest slot
            assert want is got
            continue
        tf.debugging.assert_near(tf.cast(got, tf.float32), tf.cast(want, tf.float32),
                                 atol=tolerance, rtol=tolerance)


def _bkg_inputs(cell, batch, n_bkg=100, seed=2):
    """A fused-BKG argument with four incoming edges per neuron."""
    rng = np.random.default_rng(seed)
    edges = 4 * cell._n_neurons
    empty = tf.zeros([0], tf.int32)  # only the backward reads the rest
    connectivity = SimpleNamespace(
        incoming_pre_ids=tf.constant(rng.integers(0, n_bkg, edges).astype(np.uint32)),
        incoming_edge_ids=tf.constant(np.arange(edges, dtype=np.uint32)),
        incoming_types=tf.zeros([edges], tf.uint8),
        post_ids=empty, synapse_types=empty, row_splits=empty, edge_ids=empty,
        nonempty_rows=empty, pair_ids=empty, pair_posts=empty, pair_types=empty,
    )
    return dict(activity=tf.ones((batch, n_bkg), cell.compute_dtype),
                weights=tf.ones([edges]), basis=tf.ones((1, 4)),
                connectivity=connectivity, trainable=False)


def _break_bkg(**fields):
    def corrupt(cell, state, bkg):
        return cell, state, {**bkg, **fields}
    return corrupt


def _replace_state(index, value):
    def corrupt(cell, state, bkg):
        return cell, state[:index] + (value(state[index]),) + state[index + 1:], bkg
    return corrupt


def _replace_cell(**fields):
    def corrupt(cell, state, bkg):
        return SimpleNamespace(**{**vars(cell), **fields}), state, bkg
    return corrupt


@pytest.mark.parametrize("corrupt", [
    _replace_state(4, lambda rise: rise[:, :-8]),
    _replace_state(6, lambda inputs: tf.concat([inputs, inputs[:, :8]], 1)),
    _replace_cell(syn_coeffs=tf.zeros((8, 3, 4))),
    _replace_cell(t_ref_steps=tf.ones([9], tf.int8)),
    _replace_state(7, lambda acc: tf.zeros((4, 1))),
    _break_bkg(weights=tf.ones([31])),
    _break_bkg(activity=tf.ones((2, 100))),
    _break_bkg(basis=tf.ones((1, 3))),
], ids=["basis-width", "input-width", "syn-coeffs-basis", "t-ref-size",
        "penalty-accumulator", "bkg-weights", "bkg-batch", "bkg-basis"])
def test_op_rejects_inconsistent_shapes(corrupt):
    cell = _op_cell(8, 4, tf.float32)
    state = _op_state(cell, batch=3)
    bkg = _bkg_inputs(cell, batch=3)

    def step(cell, state, bkg):
        slots, *tensors, acc = state
        return update_glif_state(slots, *tensors, cell=cell, penalty_acc=acc, bkg=bkg)

    step(cell, state, bkg)  # the unbroken inputs are accepted
    with pytest.raises(tf.errors.InvalidArgumentError):
        step(*corrupt(cell, state, bkg))


N_NEURONS, N_BKG, N_LGN = 600, 100, 30


def _network_cell(tmp_path, backend, dtype, fuse_bkg=True, **options):
    """Three cell types, delays of 1-3 steps and the production BKG fan-in of
    four edges per neuron from 100 sources, with every weight trainable."""
    tf.keras.mixed_precision.set_global_policy(
        "mixed_float16" if dtype == tf.float16 else "float32")
    rng = np.random.default_rng(3)
    tf_data = tmp_path / "tf_data"
    tf_data.mkdir(parents=True, exist_ok=True)
    with (tf_data / "syn_id_to_syn_weights_dict.pkl").open("wb") as handle:
        pickle.dump({i: rng.uniform(0.2, 1.0, 4).astype(np.float32) for i in range(5)}, handle)
    params = {
        "V_th": rng.uniform(1, 3, 3).astype(np.float32), "E_L": np.zeros(3, np.float32),
        "V_reset": np.zeros(3, np.float32), "g": rng.uniform(2, 20, 3).astype(np.float32),
        "C_m": rng.uniform(50, 300, 3).astype(np.float32),
        "t_ref": np.array([2.0, 3.0, 1.0], np.float32),
        "k": rng.uniform(0.003, 0.3, (3, 2)).astype(np.float32),
        "asc_amps": rng.uniform(-2, 2, (3, 2)).astype(np.float32),
    }

    def edges(n_pre, posts, weight_scale, **extra):
        return {"indices": np.stack([posts, rng.integers(0, n_pre, len(posts))], 1).astype(np.int32),
                "weights": rng.uniform(0, weight_scale, len(posts)).astype(np.float32),
                "delays": np.ones(len(posts), np.float32),
                "syn_ids": rng.integers(0, 5, len(posts)).astype(np.uint8), **extra}

    synapses = edges(N_NEURONS, rng.integers(0, N_NEURONS, 8 * N_NEURONS), 2.0,
                     dense_shape=(N_NEURONS, N_NEURONS))
    synapses["weights"] *= rng.choice([-1, 1], synapses["weights"].shape)
    synapses["delays"] = rng.uniform(1, 3, synapses["delays"].shape).astype(np.float32)
    network = {"data_dir": str(tmp_path), "n_nodes": N_NEURONS,
               "node_type_ids": rng.integers(0, 3, N_NEURONS).astype(np.int32),
               "node_params": params, "synapses": synapses}
    lgn = edges(N_LGN, rng.integers(0, N_NEURONS, 4 * N_NEURONS), 3.0, n_inputs=N_LGN)
    bkg = edges(N_BKG, np.repeat(np.arange(N_NEURONS), 4), 2.0, n_inputs=N_BKG)
    edge_orders = None
    if cuda_csr_recurrent.DIRECT_CSR:
        network, lgn, bkg, edge_orders = spatial_layout.apply_csr_edge_order(network, lgn, bkg)
    cell = V1Column(network, lgn, bkg, edge_orders=edge_orders, batch_size=3,
                    acceleration=backend, train_recurrent=True,
                    train_recurrent_per_type=False, train_input=True, train_noise=True,
                    noise_seed=11, **options)
    fused = cell._fused_bkg_inputs(tf.constant(3, tf.int64), tf.zeros((3, 1), tf.int32))
    assert (fused is not None) == (backend == "cuda")
    if not fuse_bkg:
        cell._fused_bkg_inputs = lambda *_: None
    return cell


@pytest.mark.parametrize("dtype", [tf.float32, tf.float16])
def test_fused_bkg_gather_is_bitwise_the_separate_bkg_operator(tmp_path, dtype):
    """The op adds the BKG current in BkgCsrForward's order, so the rise state
    and the BKG weight gradient do not change by a bit."""
    cell = _network_cell(tmp_path, "cuda", dtype)
    batch = 3
    state = cell.zero_state(batch)
    slots = tuple(tf.split(state[0], cell.max_delay, axis=1))
    inputs = tf.constant(np.random.default_rng(1).uniform(-1, 1, (batch, 4 * N_NEURONS)), dtype)
    step = tf.fill((batch, 1), 7)
    weights = tf.cast(tf.range(4 * N_NEURONS) % 5, tf.float32)

    def rise_and_gradient(fused):
        with tf.GradientTape() as tape:
            if fused:
                bkg = cell._fused_bkg_inputs(tf.constant(batch, tf.int64), step)
                current = inputs
            else:
                bkg = None
                current = tf.reshape(cell.calculate_noise_current(
                    tf.constant(batch, tf.int64), step,
                    initial=tf.reshape(inputs, (-1, 4))), (batch, -1))
            rise = update_glif_state(slots, *state[1:6], current, cell=cell, bkg=bkg)[2][4]
            loss = tf.reduce_sum(tf.cast(rise, tf.float32) * weights)
        return rise, tape.gradient(loss, cell.bkg_input_weights)

    (fused_rise, fused_gradient), (rise, gradient) = map(rise_and_gradient, (True, False))
    tf.debugging.assert_equal(fused_rise, rise)
    tf.debugging.assert_equal(fused_gradient, gradient)
    assert float(tf.reduce_max(tf.abs(gradient))) > 0.0


def _rollout(cell, inputs):
    """Spikes, final state and weight gradients of a full HeterogeneousStateRNN pass."""
    rnn = HeterogeneousStateRNN(cell, return_sequences=True, return_state=True)
    state = list(cell.zero_state(inputs.shape[0]))
    rnn(inputs, initial_state=state)
    variables = sorted(rnn.trainable_variables, key=lambda variable: variable.path)

    @tf.function
    def run():
        with tf.GradientTape() as tape:
            sequence, *final = rnn(inputs, initial_state=state)
            loss = tf.reduce_sum(tf.cast(sequence, tf.float32) * tf.cast(
                tf.range(sequence.shape[-1]) % 7, tf.float32))
            loss += sum(tf.reduce_sum(tf.cast(x, tf.float32)) for x in final if x.dtype.is_floating)
        return sequence, final, tape.gradient(loss, variables)

    return run()


@pytest.mark.parametrize("fuse_bkg", [True, False], ids=["fused-bkg", "separate-bkg"])
@pytest.mark.parametrize("options", [{}, {"hard_reset": True},
                                     {"track_voltage_penalty": True, "detach_reset": False}],
                         ids=["soft-reset", "hard-reset", "voltage-penalty"])
def test_cuda_rollout_matches_tensorflow(tmp_path, options, fuse_bkg):
    """30 steps through delays of up to three steps keep every spike, and the
    trainable recurrent, LGN and BKG weights get the same gradients."""
    inputs = tf.constant(np.random.default_rng(5).uniform(size=(3, 30, N_LGN)) < 0.3, tf.float32)
    expected = _rollout(_network_cell(tmp_path / "tf", "tensorflow", tf.float32, **options), inputs)
    actual = _rollout(_network_cell(tmp_path / "cuda", "cuda", tf.float32,
                                    fuse_bkg=fuse_bkg, **options), inputs)
    spikes = slice(0, N_NEURONS)
    tf.debugging.assert_equal(actual[0][..., spikes], expected[0][..., spikes])
    assert float(tf.reduce_sum(expected[0][..., spikes])) > 0.0
    for want, got in zip(expected[1] + expected[2], actual[1] + actual[2]):
        tf.debugging.assert_near(tf.cast(got, tf.float32), tf.cast(want, tf.float32),
                                 atol=1e-4, rtol=1e-4)
