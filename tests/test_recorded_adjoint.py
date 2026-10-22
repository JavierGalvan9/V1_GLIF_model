"""The recorded adjoint is the exact gradient of the forward it records.

Each case runs the recorded forward loop once under a tape: TensorFlow's VJP of
that loop (the CUDA operators' own custom gradients) is the reference, and
RecordedAdjointRunner's adjoint, fed the same forward's records, must match it
bit for bit - weights, every floating initial state, all statistics.
"""

import unittest

import numpy as np
import tensorflow as tf

from tests.network_fixtures import load_network
from v1_model_utils import cuda_csr_recurrent, models, spatial_layout
from v1_model_utils import loss_functions as losses
from v1_model_utils.recorded_adjoint import (
    RecordedAdjointRunner,
    SampledTraces,
    SpikeStatistics,
)


class RecordedAdjointTest(unittest.TestCase):
    BATCH, SEQ_LEN = 4, 11

    def setUp(self):
        if not tf.config.list_physical_devices("GPU"):
            self.skipTest("CUDA GPU required")
        tf.keras.mixed_precision.set_global_policy("mixed_float16")
        self.addCleanup(tf.keras.mixed_precision.set_global_policy, "float32")

    def _case(self, chunk_size, count_window, trace_window, paired_samples=False,
              **model_options):
        network, lgn, bkg = load_network(1000)
        edge_orders = None
        if cuda_csr_recurrent.DIRECT_CSR:
            network, lgn, bkg, edge_orders = spatial_layout.apply_csr_edge_order(
                network, lgn, bkg, max_delay=3
            )
        batch, length = self.BATCH, self.SEQ_LEN
        n_input = int(lgn["n_inputs"])
        model = models.create_model(
            network, lgn, bkg, seq_len=length, n_input=n_input, n_output=2,
            dtype=tf.float16, batch_size=batch, max_delay=3, train_recurrent=True,
            train_noise=True, use_state_input=True, return_state=True, add_metric=False,
            seed=3, acceleration="cuda", edge_orders=edge_orders,
            track_voltage_penalty=True, return_voltage_sequences=False,
            recurrent_dampening_factor=0.1, **model_options,
        )
        model.build((batch, length, n_input))
        layer = model.get_layer("rsnn")
        cell = layer.cell
        n = cell._n_neurons
        rng = np.random.default_rng(0)
        inputs = tf.constant((rng.random((batch, length, n_input)) < 0.02).astype(np.float16))
        state = list(cell.zero_state(batch, tf.float32))
        state[0] = tf.constant((rng.random((batch, 3 * n)) < 0.02).astype(np.float16))
        state[1] = tf.constant(rng.normal(0.3, 0.3, (batch, n)).astype(np.float32))
        state = tuple(state)
        sample_ids = rng.integers(0, n, (2, 2, 15))
        if paired_samples:
            # Every row samples eight neurons, seven of them twice.
            sample_ids[..., 8:] = sample_ids[..., :7]
        groups = tuple(
            SampledTraces(
                rows, trace_window, tf.constant(sample_ids[group].astype(np.int32)),
                tf.constant((rng.random((2, 3, 5)) < 0.8).astype(np.float16)),
            )
            for group, rows in enumerate(((0, 2), (2, 4)))
        )
        statistics = SpikeStatistics(count_window, groups)
        count_weights = tf.constant(rng.normal(size=(batch, n)).astype(np.float32))
        trace_weights = [
            tf.constant(rng.normal(size=(6, trace_window[1] - trace_window[0])), tf.float32)
            for _ in groups
        ]
        runner = RecordedAdjointRunner(
            models.build_sequence_and_state_model(model, layer), cell,
            sequence_length=length, chunk_size=chunk_size, differentiate_inputs=False,
            accumulated_variable=cell.accumulated_weight_gradient_variable,
        )
        self.assertTrue(runner.records)

        def loss_of(outputs, final_state):
            counts = tf.cast(outputs[0], tf.float32)
            loss = tf.reduce_sum(counts * count_weights) + 0.01 * tf.reduce_sum(counts ** 2)
            for trace, weights in zip(outputs[1:], trace_weights):
                trace = tf.cast(trace, tf.float32)
                loss += tf.reduce_sum(trace * weights) + 0.02 * tf.reduce_sum(trace ** 2)
            loss += 0.1 * tf.reduce_sum(final_state[1]) + 3.0 * tf.reduce_mean(final_state[-1])
            return loss + 0.2 * tf.reduce_sum(tf.cast(final_state[4], tf.float32))

        variables = model.trainable_variables
        floating = [i for i, value in enumerate(state) if value.dtype.is_floating]
        recurrent = [v.name for v in variables].index("sparse_recurrent_weights")

        @tf.function
        def compare():
            with tf.GradientTape() as tape:
                tape.watch([state[i] for i in floating])
                outputs, final_state, records = runner._forward_records(
                    inputs, state, statistics
                )
                loss = loss_of(outputs, final_state)
            detached = [tf.stop_gradient(v) for v in list(outputs) + list(final_state)]
            with tf.GradientTape() as output_tape:
                output_tape.watch(detached)
                detached_loss = loss_of(detached[:len(outputs)], detached[len(outputs):])
            cotangents = output_tape.gradient(detached_loss, detached)
            with runner._accumulation(True):
                reference = tape.gradient(
                    loss, list(variables) + [state[i] for i in floating],
                    unconnected_gradients=tf.UnconnectedGradients.ZERO,
                )
            reference[recurrent] = runner._accumulated_gradient(reference, tf.float32)
            state_gradients, candidate = runner._adjoint(
                inputs, state, final_state, records, statistics,
                cotangents[:len(outputs)], cotangents[len(outputs):],
                tuple(v.value for v in variables),
            )
            return reference, list(candidate) + [state_gradients[i] for i in floating]

        # Autodiff sums the BKG/LGN weight gradients over the whole loop; the
        # adjoint sums them per chunk, then over chunks (the segmented runner's
        # order), so with several chunks only those two differ, by fp32 rounding.
        summed = {"rest_of_brain_weights", "sparse_input_weights"}
        names = [v.name for v in variables] + [f"state_{i}" for i in floating]
        for name, expected, actual in zip(names, *compare()):
            expected = tf.convert_to_tensor(expected).numpy()
            actual = tf.convert_to_tensor(actual).numpy()
            if chunk_size < length and name in summed:
                np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)
            else:
                np.testing.assert_array_equal(actual, expected)

    def test_matches_autodiff_bitwise(self):
        self._case(self.SEQ_LEN, (0, self.SEQ_LEN), (3, self.SEQ_LEN))

    def test_matches_with_neurons_sampled_twice(self):
        # The dense gradient sums a neuron's two trace cotangents before adding
        # them to its count cotangent.
        self._case(self.SEQ_LEN, (0, self.SEQ_LEN), (3, self.SEQ_LEN), paired_samples=True)

    def test_matches_with_chunks_windows_and_cell_options(self):
        self._case(4, (2, 9), (4, 8), hard_reset=True, detach_reset=False, lr_scale=3.0,
                   train_input=True)


class InPlaceTraceUpdateTest(unittest.TestCase):
    def test_overwriting_the_previous_step_is_add_traces(self):
        """Overwriting the sampled entries of the previous step's cotangent with
        _traced_values gives _add_traces' cotangent bit for bit, with neurons
        sampled up to three times, masked entries and signed zeros. The
        first step copies the counts' cotangent, which stays unchanged."""
        from v1_model_utils.cuda_csr_recurrent.wrapper import overwrite_entries
        from v1_model_utils.recorded_adjoint import (
            _add_traces, _duplicate_runs, _Sampling, _traced_values,
        )

        rng = np.random.default_rng(5)
        batch, width, n_traces, entries = 3, 40, 6, 7
        ids = rng.integers(0, batch * width, n_traces * entries)
        ids[5:8] = ids[0]
        ids[20] = ids[3]
        sampling = _Sampling(
            tf.constant(ids, tf.int64),
            tf.constant((rng.random(ids.size) < 0.7).astype(np.float16)),
            tf.constant(np.repeat(np.arange(n_traces), entries), tf.int64),
            n_traces, entries,
        )
        runs, first = _duplicate_runs(sampling.ids)
        count = rng.normal(size=(batch, width)).astype(np.float16)
        count[:, ::4] = 0  # canonicalized: +0 only
        count_before = count.copy()
        count = tf.constant(count)
        previous = count
        for step in range(4):
            trace = rng.normal(size=n_traces).astype(np.float16)
            trace[::3] = -0.0 if step % 2 else 0.0
            trace = tf.constant(trace)
            expected = _add_traces(count, trace, sampling, runs, first)
            previous = overwrite_entries(
                previous, sampling.ids, _traced_values(count, trace, sampling, runs)
            )
            np.testing.assert_array_equal(
                previous.numpy().view(np.uint16), expected.numpy().view(np.uint16)
            )
        np.testing.assert_array_equal(count.numpy(), count_before)


class SpikeSummaryLossTest(unittest.TestCase):
    """The rate and synchronization losses read a summary as the spikes it sums."""

    def test_rates_and_traces_match_the_dense_sequence(self):
        rng = np.random.default_rng(1)
        spikes = tf.constant((rng.random((3, 20, 12)) < 0.3).astype(np.float16))
        window = losses.spike_trimming_window(20, 2, 3, trim=True)
        self.assertEqual(window, (2, 17))
        trimmed = losses.spike_trimming(spikes, 2, 3, trim=True)
        counts = tf.cast(losses.temporal_sum(trimmed, dtype=tf.float32), tf.float16)
        summary = losses.SpikeSummary(counts, window[1] - window[0])
        target = object.__new__(losses.SpikeRateDistributionTarget)
        target._dtype, target._pre_delay, target._post_delay = tf.float32, 2, 3
        np.testing.assert_array_equal(
            target.rates_per_sample_from_spikes(summary).numpy(),
            target.rates_per_sample_from_spikes(spikes).numpy(),
        )


if __name__ == "__main__":
    unittest.main()
