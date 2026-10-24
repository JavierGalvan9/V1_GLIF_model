"""The recorded adjoint is the exact gradient of the forward it records.

Each case runs the recorded forward loop once under a tape: TensorFlow's VJP of
that loop (the CUDA operators' own custom gradients) is the reference, and
RecordedAdjointRunner's adjoint, fed the same forward's records, must match it
bit for bit - weights, every floating initial state, all statistics.
"""

import os
import unittest
from unittest import mock

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

    def _model(self, **model_options):
        """The 1000-neuron test model, built; its RNN layer and input width."""
        network, lgn, bkg = load_network(1000)
        edge_orders = None
        if cuda_csr_recurrent.DIRECT_CSR:
            network, lgn, bkg, edge_orders = spatial_layout.apply_csr_edge_order(
                network, lgn, bkg, max_delay=3
            )
        n_input = int(lgn["n_inputs"])
        options = dict(
            seq_len=self.SEQ_LEN, n_input=n_input, n_output=2, dtype=tf.float16,
            batch_size=self.BATCH, max_delay=3, train_recurrent=True, train_noise=True,
            use_state_input=True, return_state=True, add_metric=False, seed=3,
            acceleration="cuda", edge_orders=edge_orders, track_voltage_penalty=True,
            return_voltage_sequences=False, recurrent_dampening_factor=0.1,
        )
        model = models.create_model(network, lgn, bkg, **{**options, **model_options})
        model.build((self.BATCH, self.SEQ_LEN, n_input))
        return model, model.get_layer("rsnn"), n_input

    def _runner(self, model, layer, chunk_size, voltage_offload="none", history_offload="none"):
        return RecordedAdjointRunner(
            models.build_sequence_and_state_model(model, layer), layer.cell,
            sequence_length=self.SEQ_LEN, chunk_size=chunk_size, differentiate_inputs=False,
            accumulated_variable=layer.cell.accumulated_weight_gradient_variable,
            voltage_offload=voltage_offload, history_offload=history_offload,
        )

    def _case(self, chunk_size, count_window, trace_window, paired_samples=False,
              offload=None, **model_options):
        model, layer, n_input = self._model(**model_options)
        batch, length = self.BATCH, self.SEQ_LEN
        cell = layer.cell
        n = cell._n_neurons
        spike_dtype = cell.compute_dtype
        rng = np.random.default_rng(0)
        inputs = tf.constant((rng.random((batch, length, n_input)) < 0.02).astype(spike_dtype))
        state = list(cell.zero_state(batch, tf.float32))
        state[0] = tf.constant((rng.random((batch, 3 * n)) < 0.02).astype(spike_dtype))
        state[1] = tf.constant(rng.normal(0.3, 0.3, (batch, n)).astype(np.float32))
        state = tuple(state)
        sample_ids = rng.integers(0, n, (2, 2, 15))
        if paired_samples:
            # Every row samples eight neurons, seven of them twice.
            sample_ids[..., 8:] = sample_ids[..., :7]
        groups = tuple(
            SampledTraces(
                rows, trace_window, tf.constant(sample_ids[group].astype(np.int32)),
                tf.constant((rng.random((2, 3, 5)) < 0.8).astype(spike_dtype)),
            )
            for group, rows in enumerate(((0, 2), (2, 4)))
        )
        statistics = SpikeStatistics(count_window, groups)
        count_weights = tf.constant(rng.normal(size=(batch, n)).astype(np.float32))
        trace_weights = [
            tf.constant(rng.normal(size=(6, trace_window[1] - trace_window[0])), tf.float32)
            for _ in groups
        ]
        offload = dict(offload or {})
        runner = self._runner(
            model, layer, chunk_size, offload.pop("voltage_offload", "none"),
            offload.pop("history_offload", "none"),
        )
        for name, value in offload.items():  # store settings, small enough for this sequence
            setattr(runner, name, value)
        self.assertEqual(runner.unsupported, ())
        self.assertTrue(runner.records)
        stores, history_stores = [], []

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
                stores.append(records[-1])
                history_stores.append(records[-2])
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
        gradients = compare()
        if runner.history_offload != "none":
            # Reuse the traced graph and its stores on the next training step.
            gradients = compare()
        self.assertEqual(bool(stores[0]), runner.voltage_offload != "none")
        self.assertEqual(tuple(map(bool, history_stores[0])),
                         (runner.history_offload != "none",) * 2)
        lossy = runner.voltage_offload in ("adaptive", "compressed")
        for name, expected, actual in zip(names, *gradients):
            expected = tf.convert_to_tensor(expected).numpy()
            actual = tf.convert_to_tensor(actual).numpy()
            if lossy:  # the reverse pass reads (part of) the record within 2**-17
                self.assertTrue(np.isfinite(actual).all())
                actual, expected = np.float64(actual), np.float64(expected)
                error = np.linalg.norm(actual - expected) / max(np.linalg.norm(expected), 1e-30)
                self.assertLess(error, 1e-3, name)
            elif chunk_size < length and name in summed:
                np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)
            else:
                np.testing.assert_array_equal(actual, expected)

    def test_matches_autodiff_bitwise(self):
        self._case(self.SEQ_LEN, (0, self.SEQ_LEN), (3, self.SEQ_LEN))

    def test_matches_autodiff_bitwise_in_float32(self):
        # Also the check of the float32 step gradient link: autodiff's
        # reference here hands the recurrent backward its bound and addends.
        tf.keras.mixed_precision.set_global_policy("float32")
        self._case(self.SEQ_LEN, (0, self.SEQ_LEN), (3, self.SEQ_LEN), dtype=tf.float32)

    def test_matches_with_neurons_sampled_twice(self):
        # The dense gradient sums a neuron's two trace cotangents before adding
        # them to its count cotangent.
        self._case(self.SEQ_LEN, (0, self.SEQ_LEN), (3, self.SEQ_LEN), paired_samples=True)

    def test_matches_with_chunks_windows_and_cell_options(self):
        self._case(4, (2, 9), (4, 8), hard_reset=True, detach_reset=False, lr_scale=3.0,
                   train_input=True)

    def test_matches_with_the_membrane_record_in_host_memory(self):
        from v1_model_utils import cuda_frame_store

        if not cuda_frame_store.available():
            self.skipTest("CUDA frame store unavailable")
        self._case(4, (2, 9), (4, 8), train_input=True, offload=dict(
            voltage_offload="raw", _PREFETCH=3, _COPY_GROUP=2,
        ))
        # A longer resident tail, groups of three.
        self._case(4, (2, 9), (4, 8), offload=dict(
            voltage_offload="raw", _PREFETCH=4, _COPY_GROUP=3,
        ))

    def test_matches_with_packed_histories_in_host_memory(self):
        from v1_model_utils import cuda_frame_store

        if not cuda_frame_store.available():
            self.skipTest("CUDA frame store unavailable")
        for voltage_offload in ("none", "raw"):
            with self.subTest(voltage_offload=voltage_offload):
                self._case(4, (2, 9), (4, 8), train_input=True, offload=dict(
                    voltage_offload=voltage_offload, history_offload="raw",
                    _PREFETCH=3, _COPY_GROUP=2,
                ))

    def test_rejects_an_unknown_history_offload(self):
        model, layer, _ = self._model()
        with self.assertRaisesRegex(ValueError, "history_offload"):
            self._runner(model, layer, self.SEQ_LEN, history_offload="compressed")

    def test_matches_closely_with_the_membrane_record_compressed(self):
        """TQ-pack on a 0 GB/s simulated link: every offloaded frame encoded,
        predicted from the two later frames (the resident tail, raw, as the
        anchors)."""
        from v1_model_utils import cuda_frame_store

        if not cuda_frame_store.available():
            self.skipTest("CUDA frame store unavailable")
        with mock.patch.dict(os.environ, {"V1_FRAME_STORE_LINK_GBPS": "0"}):
            self._case(4, (2, 9), (4, 8), train_input=True, offload=dict(
                voltage_offload="compressed", _PREFETCH=3, _COPY_GROUP=2,
            ))

    def test_names_the_conditions_a_cell_fails(self):
        """A cell returning its voltage sequences has a derivative the adjoint
        lacks; the runner names that condition and refuses statistics rollouts."""
        model, layer, _ = self._model(return_voltage_sequences=True)
        runner = self._runner(model, layer, self.SEQ_LEN)
        self.assertEqual(runner.unsupported, ("no voltage sequences",))
        self.assertFalse(runner.records)
        with self.assertRaisesRegex(ValueError, "no voltage sequences"):
            runner(tf.zeros((self.BATCH, self.SEQ_LEN, 1)), (), SpikeStatistics((0, 1), ()))

    def test_rejects_an_unknown_voltage_offload(self):
        model, layer, _ = self._model()
        with self.assertRaisesRegex(ValueError, "voltage_offload"):
            self._runner(model, layer, self.SEQ_LEN, voltage_offload="lossy")


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


class DroppedPaddingTraceTest(unittest.TestCase):
    def test_dropping_masked_entries_leaves_the_traced_values(self):
        """Padding slots (mask 0, all pointed at one neuron) only add zeros:
        without them the sampled entries get the same values, float64-summed
        (so the order of repeated neurons does not matter)."""
        from v1_model_utils.recorded_adjoint import (
            SampledTraces, SpikeStatistics, _duplicate_runs, _Sampling, _traced_values,
        )

        rng = np.random.default_rng(7)
        rows, per_trial, max_count, width = 3, 4, 9, 30
        counts = rng.integers(3, max_count + 1, (rows, per_trial, 1))
        mask = (np.arange(max_count) < counts).astype(np.float16)
        ids = rng.integers(0, width, (rows, per_trial * max_count))
        ids[:, 5:9] = ids[:, :1]  # repeated neurons within a row
        ids = np.where(mask.reshape(rows, -1) > 0, ids, 0)  # padding points at neuron 0
        group = SampledTraces(
            (0, rows), (0, 2), tf.constant(ids, tf.int32), tf.constant(mask)
        )
        statistics = SpikeStatistics((0, 2), (group,))
        full = _Sampling.of(statistics, width)
        kept = _Sampling.of(statistics, width, drop_masked=True)
        self.assertEqual(kept.ids.shape[0], int(mask.sum()))
        count = tf.constant(rng.normal(size=(rows, width)).astype(np.float16))
        trace = tf.constant(rng.normal(size=rows * per_trial).astype(np.float16))
        values = []
        for sampling in (full, kept):
            out = _traced_values(count, trace, sampling, _duplicate_runs(sampling.ids)[0])
            dense = count.numpy().reshape(-1).copy()
            dense[sampling.ids.numpy()] = out.numpy()
            values.append(dense)
        np.testing.assert_array_equal(values[0].view(np.uint16), values[1].view(np.uint16))


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
