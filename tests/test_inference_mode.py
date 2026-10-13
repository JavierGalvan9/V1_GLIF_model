"""Public model-factory tests for compact inference spike output."""

from __future__ import annotations

import pickle

import numpy as np
import tensorflow as tf

from v1_model_utils import models


def _tiny_network(tmp_path):
    tf_data = tmp_path / "tf_data"
    tf_data.mkdir()
    with (tf_data / "syn_id_to_syn_weights_dict.pkl").open("wb") as handle:
        pickle.dump({0: np.ones(4, dtype=np.float32)}, handle)

    node_params = {
        "V_th": np.array([1.0], np.float32),
        "E_L": np.array([0.0], np.float32),
        "V_reset": np.array([0.0], np.float32),
        "g": np.array([1.0], np.float32),
        "C_m": np.array([1.0], np.float32),
        "t_ref": np.array([1.0], np.float32),
        "k": np.array([[0.1, 0.2]], np.float32),
        "asc_amps": np.zeros((1, 2), np.float32),
    }
    network = {
        "data_dir": str(tmp_path),
        "n_nodes": 2,
        "node_type_ids": np.zeros(2, np.int32),
        "node_params": node_params,
        "synapses": {
            "indices": np.array([[0, 0], [1, 1]], np.int32),
            "weights": np.zeros(2, np.float32),
            "delays": np.ones(2, np.float32),
            "syn_ids": np.zeros(2, np.uint8),
            "dense_shape": (2, 2),
        },
    }
    input_data = {
        "n_inputs": 1,
        "indices": np.array([[0, 0], [1, 0]], np.int32),
        "weights": np.array([10.0, 10.0], np.float32),
        "delays": np.ones(2, np.float32),
        "syn_ids": np.zeros(2, np.uint8),
    }
    background = {
        **input_data,
        "weights": np.zeros(2, np.float32),
    }
    return network, input_data, background


def _extract_sequences(model, inputs):
    rsnn = model.get_layer("rsnn")
    outputs = models._normalize_v1_rnn_outputs(rsnn, rsnn.output)
    extractor = tf.keras.Model(model.inputs, outputs)
    outputs = extractor(inputs)
    return outputs[0][0], tuple(outputs[1:])


def test_output_spike_dtype_preserves_float_recurrent_state(tmp_path):
    network, lgn_input, background = _tiny_network(tmp_path)
    common = dict(
        seq_len=4,
        n_input=1,
        dtype=tf.float32,
        batch_size=1,
        train_recurrent=False,
        train_input=False,
        train_noise=False,
        use_state_input=False,
        return_state=True,
        add_metric=False,
        current_input=True,
        acceleration="tensorflow",
        seed=7,
    )
    baseline = models.create_model(network, lgn_input, background, **common)
    candidate = models.create_model(
        network, lgn_input, background, output_spike_dtype=tf.uint8, **common
    )
    inputs = tf.ones((1, 4, 1), tf.float32)

    baseline_spikes, baseline_state = _extract_sequences(baseline, inputs)
    candidate_spikes, candidate_state = _extract_sequences(candidate, inputs)

    assert baseline_spikes.dtype == tf.float32
    assert candidate_spikes.dtype == tf.uint8
    assert candidate_state[0].dtype == baseline_state[0].dtype
    np.testing.assert_array_equal(candidate_spikes.numpy(), baseline_spikes.numpy())
    np.testing.assert_array_equal(candidate_state[0].numpy(), baseline_state[0].numpy())
    assert np.count_nonzero(candidate_spikes.numpy()) > 0


def test_state_only_rollout_preserves_compact_integer_states(tmp_path, request):
    tf.keras.mixed_precision.set_global_policy("mixed_float16")
    # The policy is global; leaving it set changes every later test's dtypes.
    request.addfinalizer(
        lambda: tf.keras.mixed_precision.set_global_policy("float32")
    )
    network, lgn_input, background = _tiny_network(tmp_path)
    model = models.create_model(
        network,
        lgn_input,
        background,
        seq_len=4,
        n_input=1,
        dtype=tf.float16,
        batch_size=1,
        train_recurrent=False,
        train_input=False,
        train_noise=False,
        use_state_input=True,
        return_state=True,
        add_metric=False,
        current_input=True,
        acceleration="tensorflow",
        track_voltage_penalty=True,
        return_voltage_sequences=False,
        seed=7,
    )
    rsnn = model.get_layer("rsnn")
    initial_state = rsnn.cell.zero_state(1, tf.float16)
    state_model = models.build_state_only_model(model, rsnn)

    state = state_model([tf.ones((1, 4, 1), tf.float16), *initial_state])

    assert initial_state[2].dtype in (tf.int8, tf.int16)
    assert state[2].dtype == initial_state[2].dtype
    assert state[6].dtype == tf.int32
