"""Exercise recorded history transport on CPU, without loading CUDA operators.

The real forward/reverse loops run against portable cell/backward operators.
The fake destructive frame store rejects duplicate reads and missing writes.
These checks cover transport and delay-window routing, not CUDA numerics.
"""

from contextlib import nullcontext
from types import SimpleNamespace
from typing import NamedTuple

import numpy as np
import pytest
import tensorflow as tf

from v1_model_utils import lgn_input_source, models, recorded_adjoint as adjoint


BATCH, WIDTH, DELAY = 2, 35, 3


class _History(NamedTuple):
    slots: tuple


class _Cell:
    compute_dtype = "float32"
    variable_dtype = "float32"
    max_delay = DELAY
    _n_neurons = WIDTH
    _n_syn_basis = 4
    _synaptic_current_backend = "portable"
    _state_backend = "cuda"
    _chain_current_sources = True
    _return_voltage_sequences = False
    _output_neuron_ids = None
    _output_spike_dtype = "float32"
    _compute_lgn_activity_gradient = False
    _compute_bkg_activity_gradient = False
    _track_voltage_penalty = False
    _lgn_delay_buffer_width = 0
    input_dim = WIDTH
    _scales_recurrent_inputs = False
    _recurrent_dampening = 1.0
    _dt = 1.0
    v_th = 1.0
    _gauss_std = 1.0
    _dampening_factor = 1.0
    trainable_variables = ()

    def __init__(self):
        self.recurrent_weight_values = tf.keras.Variable([1.0], trainable=False)
        self.bkg_input_weights = tf.keras.Variable([1.0], trainable=False)
        self.input_weight_values = tf.keras.Variable([1.0], trainable=False)
        self.synaptic_basis_weights = tf.ones((1, 4))
        self.recurrent_csr = SimpleNamespace(resource_name="portable")
        self.bkg_input_csr = self.input_csr = None

    def lgn_pair_weights_scope(self):
        return nullcontext()

    def to_loop_state(self, state):
        return (_History(tuple(tf.split(state[0], DELAY, axis=1))), *state[1:])

    def from_loop_state(self, state):
        return (tf.concat(state[0].slots, axis=1), *state[1:])

    def call(self, inputs, state):
        history, voltage, refractory, asc, rise, psc, clock = state
        spikes = tf.cast(inputs > 0, tf.float32)
        return spikes, (
            _History((spikes, *history.slots[:-1])),
            voltage * 0.5 + inputs + history.slots[-1] * 0.25,
            tf.cast(tf.math.floormod(clock + tf.range(WIDTH)[None], 3) == 0, tf.int32),
            asc * 0.5 + tf.tile(spikes, [1, 2]),
            rise * 0.5 + tf.tile(inputs, [1, 4]),
            psc * 0.5 + rise * 0.25,
            clock + 1,
        )


class _FrameStore:
    def __init__(self):
        self.stores = {}
        self.resources = {}

    def prepare(self, capacity, rollouts):
        # Preallocate resources so tracing does not trigger a variable-creation
        # retrace, which would itself allocate another rollout's store keys.
        for kind in ("spikes", "blocked"):
            self.resources[kind] = []
            for _ in range(rollouts):
                self.resources[kind].append((
                    tf.Variable(tf.zeros((capacity, BATCH, 2), tf.int32)),
                    tf.Variable(tf.zeros((capacity,), tf.int32)),
                    tf.Variable(tf.zeros((capacity,), tf.int32)),
                ))

    def offload_frame(self, value, index, key, capacity, *args, codec, drop_outside=False, order=None):
        assert codec == "raw"
        assert value.dtype == tf.int32
        if key not in self.stores:
            self.stores[key] = self.resources[key.split("_")[0]].pop()
        data, written, taken = self.stores[key]
        assert data.shape == (capacity, *value.shape)
        if drop_outside:  # the op records nothing outside [0, capacity)
            index = tf.convert_to_tensor(index)
            return tf.cond((index >= 0) & (index < capacity),
                           lambda: self._write(value, index, data, written, taken),
                           lambda: tf.zeros([0]))
        return self._write(value, index, data, written, taken)

    @staticmethod
    def _write(value, index, data, written, taken):
        reset = tf.cond(
            tf.equal(index, 0),
            lambda: (written.assign(tf.zeros_like(written)), taken.assign(tf.zeros_like(taken))),
            lambda: (written.read_value(), taken.read_value()),
        )
        with tf.control_dependencies(tf.nest.flatten(reset)):
            check = tf.debugging.assert_equal(written[index], 0)
        with tf.control_dependencies([check]):
            updates = (data.scatter_nd_update([[index]], value[None]),
                       written.scatter_nd_update([[index]], [1]))
        with tf.control_dependencies(updates):
            return tf.zeros([0])

    def take_frame(self, index, key, like, zero_before_start=False, order=None, **kwargs):
        if zero_before_start:  # the op returns zeros before frame 0
            return tf.cond(tf.convert_to_tensor(index) >= 0,
                           lambda: self.take_frame(index, key, like),
                           lambda: tf.zeros(like.shape, like.dtype))
        data, written, taken = self.stores[key]
        checks = (tf.debugging.assert_equal(written[index], 1),
                  tf.debugging.assert_equal(taken[index], 0))
        with tf.control_dependencies(checks):
            value = tf.identity(data[index])
        with tf.control_dependencies([value]):
            update = taken.scatter_nd_update([[index]], [1])
        with tf.control_dependencies([update]):
            return tf.ensure_shape(tf.identity(value), like.shape)

    def assert_consumed(self, rollouts):
        assert len(self.stores) == 2 * rollouts
        assert {key.split("_")[0] for key in self.stores} == {"spikes", "blocked"}
        for _, written, taken in self.stores.values():
            np.testing.assert_array_equal(written.numpy(), 1)
            np.testing.assert_array_equal(taken.numpy(), 1)


def _glif_backward(cell, previous, voltage, blocked, constants, dt, threshold,
                   sigma, dampening, inverse, spike, history0, history1,
                   voltage_g, asc_g, rise_g, psc_g, *args, **kwargs):
    previous = models._unpack_spike_words(previous, WIDTH, tf.float32, 31)
    blocked = models._unpack_spike_words(blocked, WIDTH, tf.float32, 31)
    signal = (previous * 0.125 + blocked * 0.25 + voltage * 0.03125
              + spike + history0 * 0.25 + history1 * 0.125 + voltage_g * 0.5
              + asc_g[:, :WIDTH] * 0.125 + rise_g[:, :WIDTH] * 0.03125
              + psc_g[:, :WIDTH] * 0.03125)
    return (signal, signal * 0.5,
            asc_g * 0.25 + tf.tile(signal * 0.125, [1, 2]),
            rise_g * 0.25 + tf.tile(signal * 0.125, [1, 4]),
            psc_g * 0.25 + tf.tile(signal * 0.125, [1, 4]),
            tf.tile(signal * 0.125, [1, 4]), tf.zeros([1], tf.uint32))


def _recurrent_backward(connectivity, slots, upstream, weights, basis, dampening,
                        *, activity_masks, abs_max_bound, spike_grad_addends):
    signal = tf.reduce_sum(tf.reshape(upstream, (BATCH, WIDTH, 4)), axis=-1)
    gradients = []
    for index, (slot, mask, addend) in enumerate(zip(slots, activity_masks, spike_grad_addends)):
        gradient = signal * ((index + 1) * 0.125) + slot * 0.03125 + mask * 0.0625
        gradients.append(gradient + addend if addend.shape.num_elements() else gradient)
    return tuple(gradients), None


@pytest.fixture
def portable_backend(monkeypatch):
    store = _FrameStore()
    monkeypatch.setattr(adjoint.frame_store, "available", lambda: True)
    monkeypatch.setattr(adjoint.frame_store, "offload_frame", store.offload_frame)
    monkeypatch.setattr(adjoint.frame_store, "take_frame", store.take_frame)
    monkeypatch.setattr(adjoint, "glif_constants", lambda cell: ())
    monkeypatch.setattr(adjoint, "glif_step_backward", _glif_backward)
    monkeypatch.setattr(adjoint, "resource_recurrent_backward", _recurrent_backward)
    monkeypatch.setattr(adjoint, "spike_activity_masks", lambda slot: slot * 0.5)
    # Execute the same packing/statistics TensorFlow code without XLA compilation.
    for module, names in ((models, ("_pack_spike_words", "_unpack_spike_words")),
                          (adjoint, ("_pack_step", "_unpack_frame", "_record_step"))):
        for name in names:
            monkeypatch.setattr(module, name, getattr(module, name).python_function)
    return store


def _runner(length, mode=None):
    return adjoint.RecordedAdjointRunner(
        SimpleNamespace(_v1_sequence_output_count=1), _Cell(), length, min(length, 4),
        differentiate_inputs=False, **({"history_offload": mode} if mode is not None else {}),
    )


@pytest.mark.parametrize("length", [1, 2, 3, 7, 11])
@pytest.mark.parametrize("rollouts", [1, 2])
@pytest.mark.parametrize("mode", ["raw", "adaptive"])
def test_history_offload_matches_resident_adjoint(portable_backend, length, rollouts, mode):
    portable_backend.prepare(length, rollouts)
    resident, offloaded = _runner(length, "none"), _runner(length, mode)
    # Short sequences ordinarily stay resident; exercise transport even at L < delay.
    offloaded._PREFETCH = 0 if length <= offloaded._PREFETCH else offloaded._PREFETCH
    rng = np.random.default_rng(13)
    state = (tf.constant(rng.integers(0, 2, (BATCH, WIDTH * DELAY)), tf.float32),
             tf.constant(rng.normal(size=(BATCH, WIDTH)), tf.float32),
             tf.zeros((BATCH, WIDTH), tf.int32),
             tf.constant(rng.normal(size=(BATCH, WIDTH * 2)), tf.float32),
             tf.constant(rng.normal(size=(BATCH, WIDTH * 4)), tf.float32),
             tf.constant(rng.normal(size=(BATCH, WIDTH * 4)), tf.float32),
             tf.zeros((BATCH, 1), tf.int32))
    inputs = tf.constant(rng.normal(size=(BATCH, length, WIDTH)), tf.float32)
    statistics = adjoint.SpikeStatistics((0, length))

    @tf.function(autograph=False)
    def run(inputs, state):
        results = []
        for runner in (resident, offloaded):
            for rollout in range(rollouts):
                source = lgn_input_source.DenseFrames(inputs + rollout)
                outputs, final, records = runner._forward_records(
                    lambda time, buffer: source.advance(source.tensors, time, buffer),
                    source.initial_buffer(), state, statistics,
                )
                state_gradients = tuple(tf.ones_like(value) for value in final)
                _, gradients, _ = runner._adjoint(
                    source, source.tensors, state, final, records, statistics,
                    (tf.ones_like(outputs[0]),), state_gradients, (),
                )
                results.append((outputs, final, tuple(value for value in gradients if value is not None)))
        return tuple(results)

    with tf.device("/CPU:0"):
        for _ in range(2):
            results = run(inputs, state)
            for index in range(rollouts):
                for expected, actual in zip(tf.nest.flatten(results[index]),
                                            tf.nest.flatten(results[index + rollouts])):
                    np.testing.assert_array_equal(actual.numpy(), expected.numpy())
            portable_backend.assert_consumed(rollouts)


def test_history_offload_configuration_and_fallbacks(portable_backend, monkeypatch):
    assert _runner(11).history_offload == "none"
    assert _runner(11).voltage_offload == "none"
    with pytest.raises(ValueError, match="history_offload"):
        _runner(11, "compressed")
    runner = _runner(11, "raw")
    assert runner._frame_store("spikes", True, BATCH, 2) is None  # eager fallback
    monkeypatch.setattr(adjoint.frame_store, "available", lambda: False)

    @tf.function(autograph=False)
    def fallback():
        assert runner._frame_store("spikes", True, BATCH, 2) is None
        return tf.constant(1)

    assert fallback().numpy() == 1


@pytest.mark.parametrize("mode,codec", [("raw", "raw"), ("adaptive", "tq"),
                                       ("compressed", "tq")])
def test_voltage_offload_selects_codec(portable_backend, monkeypatch, mode, codec):
    length = 11
    runner = adjoint.RecordedAdjointRunner(
        SimpleNamespace(_v1_sequence_output_count=1), _Cell(), length, 4,
        differentiate_inputs=False, voltage_offload=mode,
    )
    monkeypatch.setattr(runner, "_frame_store",
                        lambda name, enabled, *args: name if enabled else None)
    codecs = []

    def offload(value, index, key, *args, codec, drop_outside=False, order=None):
        assert key == "voltages" and value.dtype == tf.float32
        codecs.append(codec)
        return tf.zeros([0])

    monkeypatch.setattr(adjoint.frame_store, "offload_frame", offload)
    state = tuple(tf.zeros((BATCH, size), dtype) for size, dtype in (
        (WIDTH * DELAY, tf.float32), (WIDTH, tf.float32), (WIDTH, tf.int32),
        (WIDTH * 2, tf.float32), (WIDTH * 4, tf.float32), (WIDTH * 4, tf.float32),
        (1, tf.int32),
    ))

    @tf.function(autograph=False)
    def run():
        source = lgn_input_source.DenseFrames(tf.zeros((BATCH, length, WIDTH)))
        outputs, _, _ = runner._forward_records(
            lambda time, buffer: source.advance(source.tensors, time, buffer),
            source.initial_buffer(), state, adjoint.SpikeStatistics((0, length)),
        )
        return outputs

    run()
    assert codecs and set(codecs) == {codec}
