"""Validation protocol: K grating phases per direction, and direction-averaged losses."""
import os

import numpy as np
import tensorflow as tf

import multi_training
import stim_dataset
from v1_model_utils.loss_functions import SynchronizationLoss

INT32_MAX = np.iinfo(np.int32).max


def test_trials_take_phases_in_global_trial_order():
    """Trial i of replica r is global trial r * B + i and takes phase g mod K."""
    np.testing.assert_array_equal(multi_training.validation_phase_index(0, 3, 8), [0, 1, 2])
    np.testing.assert_array_equal(multi_training.validation_phase_index(1, 3, 8), [3, 4, 5])
    np.testing.assert_array_equal(
        multi_training.validation_phase_index(2, 5, 8), [2, 3, 4, 5, 6]
    )
    # 248 + 8 trials per replica on 2 replicas: every phase 64 times.
    phases = np.concatenate([
        multi_training.validation_phase_index(replica, 256, 8).numpy() for replica in (0, 1)
    ])
    np.testing.assert_array_equal(np.bincount(phases), np.full(8, 64))


def _phase_patterns(n_phases=3, seq_len=10, n_input=12):
    """0/1 probabilities: phase k fires input k at every time step, nothing else."""
    probability = np.zeros((n_phases, seq_len, n_input), np.float32)
    for phase in range(n_phases):
        probability[phase, :, phase] = 1.0
    return tf.constant(probability)


def test_each_trial_samples_its_own_phase():
    probability = _phase_patterns()
    trial_index = tf.constant([2, 0, 1, 1, 2], tf.int32)
    spikes = multi_training.sample_stateless_bernoulli_batch(
        probability, 5, (3, 7), trial_index=trial_index
    )
    np.testing.assert_array_equal(
        spikes.numpy(), tf.gather(probability, trial_index).numpy() > 0.5
    )


def test_per_time_chunk_sampling_equals_the_full_batch_reference():
    """The compact validation step samples [B, chunk, n_input] slices only."""
    probability = _phase_patterns(seq_len=10)
    trial_index = multi_training.validation_phase_index(1, 4, 3)
    chunks = [
        multi_training.sample_stateless_bernoulli_batch(
            probability[..., start:start + 3, :], 4, (start, 11), trial_index=trial_index
        )
        for start in range(0, 10, 3)
    ]
    np.testing.assert_array_equal(
        tf.concat(chunks, axis=1).numpy(),
        tf.gather(probability, trial_index).numpy() > 0.5,
    )


def test_trial_sampling_is_the_same_with_batch_chunking():
    probability = tf.random.stateless_uniform((3, 4, 4), seed=(1, 2))
    trial_index = tf.constant(np.arange(19) % 3, tf.int32)
    expected = multi_training.sample_stateless_bernoulli_batch(
        probability, 19, (17, 29), trial_index=trial_index
    )
    reference = tf.random.stateless_uniform((19, 4, 4), seed=(17, 29)) < tf.gather(
        probability, trial_index
    )
    np.testing.assert_array_equal(expected.numpy(), reference.numpy())
    for chunk_size in (1, 3, 64):
        actual = multi_training.sample_stateless_bernoulli_batch(
            probability, 19, (17, 29), batch_chunk_size=chunk_size,
            full_tensor_element_limit=32, trial_index=trial_index,
        )
        np.testing.assert_array_equal(actual.numpy(), expected.numpy())


def test_full_sequence_trial_sampling_stays_below_int32_kernels():
    """The path without the online voltage loss samples a whole [256, 500, 17400] batch."""
    def sample(probability):
        return multi_training.sample_stateless_bernoulli_batch(
            probability, 256, (17, 29),
            trial_index=multi_training.validation_phase_index(1, 256, 8),
        )

    graph = tf.function(sample, autograph=False).get_concrete_function(
        tf.TensorSpec((8, 500, 17400), tf.float32)
    )
    assert graph.output_shapes == (256, 500, 17400)
    for op in graph.graph.get_operations():
        if op.type in ("StatelessRandomUniformV2", "Less", "GatherV2"):
            assert op.outputs[0].shape.num_elements() <= INT32_MAX, op.type


def _fano_loss():
    """A SynchronizationLoss with only what loss_from_fano_curve reads."""
    loss = SynchronizationLoss.__new__(SynchronizationLoss)
    loss._dtype = tf.float32
    loss._sync_cost = 1.5
    loss._bin_sizes_ms_tf = tf.constant([1, 10, 100, 200], tf.int32)
    loss.experimental_fanos_mean = tf.constant([1.0, 1.2, 1.6, 2.0], tf.float32)
    return loss


def test_evoked_losses_are_taken_on_direction_averaged_rates_and_curves():
    rng = np.random.default_rng(0)
    rates = rng.uniform(0.0, 20.0, (8, 50))
    curves = rng.uniform(0.5, 3.0, (8, 4)).astype(np.float32)
    targets = rng.uniform(0.0, 10.0, 50)
    sync = _fano_loss()

    def rate_loss(values):
        return np.mean((values - targets) ** 2)

    def sync_loss(curve):
        return sync.loss_from_fano_curve(curve, 300).numpy()

    rate_value, sync_value = multi_training.direction_averaged_validation_losses(
        rates, curves, rate_loss, sync_loss
    )
    np.testing.assert_allclose(rate_value, rate_loss(rates.mean(axis=0)))
    np.testing.assert_allclose(sync_value, sync_loss(curves.mean(axis=0)), rtol=1e-6)
    # Not the mean of the per-direction losses, which adds their spread.
    assert rate_value < np.mean([rate_loss(row) for row in rates]) - 1.0
    assert sync_value < np.mean([sync_loss(row) for row in curves]) - 1e-3
    # Bins of 200 ms and more do not fit twice in 300 ms and are left out.
    np.testing.assert_allclose(
        sync_loss(curves.mean(axis=0)),
        1.5 * np.mean((curves.mean(axis=0)[:3] - [1.0, 1.2, 1.6]) ** 2),
        rtol=1e-6,
    )
    assert multi_training.direction_averaged_validation_losses(
        rates, None, rate_loss, sync_loss
    )[1] == 0.0


def test_fano_curve_loss_matches_the_trace_loss():
    traces = tf.cast(tf.random.stateless_uniform((6, 300), seed=(4, 5)) < 0.05, tf.float32)
    sync = _fano_loss()
    sync._bin_sizes_ms = (1, 10, 100, 200)
    sync._epsilon_tf = tf.constant(1e-7, tf.float32)
    np.testing.assert_allclose(
        sync.loss_from_fano_curve(sync.pop_fano_tf(traces), 300).numpy(),
        sync.loss_from_sampled_traces(traces).numpy(),
    )


class _FakeGratingLGN:
    """Rates that encode the (angle, phase) they were asked for."""

    calls = []

    def __init__(self, seq_len, pre_delay, post_delay, n_input, **_):
        self.shape = (seq_len, n_input)

    def firing_rates(self, theta, phase):
        _FakeGratingLGN.calls.append((theta, phase))
        return tf.fill(self.shape, float(theta + phase / 1000.0))


def test_phase_probabilities_are_cached_per_direction(tmp_path, monkeypatch):
    monkeypatch.setattr(stim_dataset, "DriftingGratingLGN", _FakeGratingLGN)
    arguments = dict(
        seq_len=6, pre_delay=1, post_delay=1, n_input=5, data_dir=str(tmp_path),
        angles=np.array([0, 90]), n_phases=4,
    )
    probabilities = stim_dataset.load_or_compute_protocol_phase_lgn_probabilities(**arguments)
    assert sorted(probabilities) == [0, 90]
    assert _FakeGratingLGN.calls == [
        (angle, phase) for angle in (0.0, 90.0) for phase in (0.0, 90.0, 180.0, 270.0)
    ]
    for angle, value in probabilities.items():
        assert value.shape == (4, 6, 5) and value.dtype == np.float32
        rates = angle + np.array([0.0, 90.0, 180.0, 270.0]) / 1000.0
        np.testing.assert_allclose(
            value[:, 0, 0], -np.expm1(-rates / 1000.0), rtol=1e-6
        )
    (cache_file,) = os.listdir(tmp_path / "tf_data")
    assert "phases_4" in cache_file

    _FakeGratingLGN.calls.clear()
    cached = stim_dataset.load_or_compute_protocol_phase_lgn_probabilities(**arguments)
    assert _FakeGratingLGN.calls == []
    np.testing.assert_array_equal(cached[90], probabilities[90])
