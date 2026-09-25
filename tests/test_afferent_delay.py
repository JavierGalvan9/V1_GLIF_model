"""LGN afferent delay is applied to the input stream."""

import pickle

import numpy as np
import pytest
import tensorflow as tf

from v1_model_utils import cuda_csr_recurrent, spatial_layout
from v1_model_utils.models import V1Column


def _cell(tmp_path, lgn_delay, track_voltage_penalty=False):
    tf.keras.mixed_precision.set_global_policy("float32")
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
        "delays": np.full(2, lgn_delay, np.float32),
        "syn_ids": np.zeros(2, np.uint8),
    }
    bkg = dict(external, delays=np.ones(2, np.float32))
    edge_orders = None
    if cuda_csr_recurrent.DIRECT_CSR:
        network, external, bkg, edge_orders = spatial_layout.apply_csr_edge_order(
            network, external, bkg
        )
    return V1Column(
        network, external, bkg,
        edge_orders=edge_orders,
        batch_size=1,
        acceleration="tensorflow",
        train_recurrent=False, train_input=False, train_noise=False,
        bkg_firing_rate=0.0,
        track_voltage_penalty=track_voltage_penalty,
    )


def _psc_trace(cell, spike_times, n_steps=6):
    state = cell.zero_state(1)
    trace = []
    for step in range(n_steps):
        spikes = tf.constant(
            [[1.0, 1.0]] if step in spike_times else [[0.0, 0.0]],
            dtype=cell.compute_dtype,
        )
        _, state = cell(spikes, state)
        trace.append(float(tf.reduce_sum(tf.abs(state[4]))))
    return np.array(trace)


def test_max_delay_ignores_afferent_delays(tmp_path):
    # The z ring buffer only holds recurrent spikes, so a long LGN delay must
    # not enlarge it.
    assert _cell(tmp_path / "a", lgn_delay=10.0).max_delay == 3


def test_lgn_delay_shifts_the_input_stream(tmp_path):
    fast = _cell(tmp_path / "fast", lgn_delay=1.0)
    slow = _cell(tmp_path / "slow", lgn_delay=1.7)
    assert fast._lgn_delay_steps == 1
    assert slow._lgn_delay_steps == 2

    fast_trace = _psc_trace(fast, spike_times={0})
    slow_trace = _psc_trace(slow, spike_times={0})
    # One extra timestep of latency: the slow cell sees nothing on step 0 and
    # then reproduces the fast cell's trace shifted by one step.
    assert slow_trace[0] == 0.0
    assert fast_trace[0] > 0.0
    np.testing.assert_allclose(slow_trace[1:], fast_trace[:-1], rtol=1e-5)


def test_voltage_penalty_reset_leaves_the_delay_buffer_alone(tmp_path):
    # reset_voltage_penalty_state() zeroes the LAST state element, so the delay
    # buffer must not be appended after the penalty accumulator.
    from v1_model_utils.models import reset_voltage_penalty_state

    cell = _cell(tmp_path / "pen", lgn_delay=1.7, track_voltage_penalty=True)
    state = list(cell.zero_state(1))
    lgn_index = 7
    state[lgn_index] = tf.ones_like(state[lgn_index])
    state[-1] = tf.ones_like(state[-1])

    reset = reset_voltage_penalty_state(cell, tuple(state))
    assert float(tf.reduce_sum(reset[-1])) == 0.0, "penalty was not reset"
    assert float(tf.reduce_sum(reset[lgn_index])) == float(cell.input_dim), (
        "the LGN delay buffer was cleared by the voltage-penalty reset"
    )
