"""CheckpointedStatisticsRunner: the checkpointing rollout returning spike statistics.

Its counts and sampled traces must be those the losses take from the dense
sequence of SegmentedRecomputeRunner (temporal_sum, _gather_population_traces),
and the gradients of a loss of them must be those the dense rollout gets, bit
for bit. The two rollouts run separately, and atomic accumulation orders can
change a run's trajectory: each statistics run is compared against every dense
run seen, a mismatch only counting when the dense runs agree with each other.
"""

import unittest

import numpy as np
import tensorflow as tf

from tests.network_fixtures import load_network
from v1_model_utils import cuda_csr_recurrent, models, spatial_layout
from v1_model_utils import loss_functions as losses
from v1_model_utils.recorded_adjoint import (
    CheckpointedStatisticsRunner,
    SampledTraces,
    SpikeStatistics,
)


class CheckpointedStatisticsTest(unittest.TestCase):
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
        n_input = int(lgn["n_inputs"])
        batch, length = self.BATCH, self.SEQ_LEN
        options = dict(
            seq_len=length, n_input=n_input, n_output=2, dtype=tf.float16,
            batch_size=batch, max_delay=3, train_recurrent=True, train_noise=True,
            use_state_input=True, return_state=True, add_metric=False, seed=3,
            acceleration="cuda", edge_orders=edge_orders, track_voltage_penalty=True,
            return_voltage_sequences=False, recurrent_dampening_factor=0.1,
        )
        model = models.create_model(network, lgn, bkg, **{**options, **model_options})
        model.build((batch, length, n_input))
        layer = model.get_layer("rsnn")
        cell = layer.cell
        core = models.build_sequence_and_state_model(model, layer)
        runner_options = dict(
            sequence_length=length, chunk_size=chunk_size, differentiate_inputs=False,
            accumulated_variable=cell.accumulated_weight_gradient_variable,
        )
        dense = models.SegmentedRecomputeRunner(core, cell=cell, **runner_options)
        summarizing = CheckpointedStatisticsRunner(core, cell, **runner_options)
        self.assertTrue(summarizing.summarizes)
        n = cell._n_neurons
        rng = np.random.default_rng(0)
        inputs = tf.constant((rng.random((batch, length, n_input)) < 0.02).astype(np.float16))
        state = list(cell.zero_state(batch, tf.float32))
        state[0] = tf.constant((rng.random((batch, 3 * n)) < 0.02).astype(np.float16))
        state[1] = tf.constant(rng.normal(0.3, 0.3, (batch, n)).astype(np.float32))
        state = tuple(state)
        sample_ids = rng.integers(0, n, (2, 2, 15))
        if paired_samples:
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

        def loss_of(counts, traces, final_state):
            counts = tf.cast(counts, tf.float32)
            loss = tf.reduce_sum(counts * count_weights) + 0.01 * tf.reduce_sum(counts ** 2)
            for trace, weights in zip(traces, trace_weights):
                trace = tf.cast(trace, tf.float32)
                loss += tf.reduce_sum(trace * weights) + 0.02 * tf.reduce_sum(trace ** 2)
            loss += 0.1 * tf.reduce_sum(final_state[1]) + 3.0 * tf.reduce_mean(final_state[-1])
            return loss + 0.2 * tf.reduce_sum(tf.cast(final_state[4], tf.float32))

        def dense_statistics(spikes):
            """What the losses take from the sequence (SpikeSummaryLossTest)."""
            start, stop = count_window
            counts = tf.cast(losses.temporal_sum(spikes[:, start:stop], dtype=tf.float32),
                             tf.float16)
            start, stop = trace_window
            traces = tuple(
                losses._gather_population_traces(
                    spikes[group.rows[0]:group.rows[1], start:stop], group.sample_ids,
                    group.neuron_mask, group.rows[1] - group.rows[0], stop - start,
                    group.neuron_mask.shape[1], group.neuron_mask.shape[2],
                )
                for group in groups
            )
            return counts, traces

        variables = model.trainable_variables
        floating = [state[i] for i, value in enumerate(state) if value.dtype.is_floating]

        @tf.function
        def run_dense():
            with tf.GradientTape() as tape:
                tape.watch(floating)
                outputs = dense(inputs, state)
                counts, traces = dense_statistics(outputs[0])
                loss = loss_of(counts, traces, outputs[1:])
            return (counts,) + traces, tape.gradient(loss, list(variables) + floating)

        @tf.function
        def run_summarized():
            with tf.GradientTape() as tape:
                tape.watch(floating)
                outputs = summarizing(inputs, state, statistics)
                n_statistics = 1 + len(groups)
                loss = loss_of(outputs[0], outputs[1:n_statistics], outputs[n_statistics:])
            return outputs[:n_statistics], tape.gradient(loss, list(variables) + floating)

        def values(run):
            return [np.asarray(tf.convert_to_tensor(x)) for part in run() for x in part]

        def same(a, b):
            return all(np.array_equal(x, y) for x, y in zip(a, b))

        references = [values(run_dense) for _ in range(3)]
        attempts = [values(run_summarized) for _ in range(2)]
        self.assertTrue(np.any(references[0][0] != 0), "no spikes counted")
        if any(same(a, r) for a in attempts for r in references):
            return
        if not all(same(references[0], other) for other in references[1:]):
            self.skipTest("the forward is not run-to-run deterministic here")
        for index, (a, b) in enumerate(zip(attempts[0], references[0])):
            np.testing.assert_array_equal(a, b, err_msg=f"output {index}")

    def test_matches_the_dense_rollout_bitwise(self):
        self._case(4, (0, self.SEQ_LEN), (3, self.SEQ_LEN))

    def test_matches_with_windows_and_neurons_sampled_twice(self):
        self._case(3, (2, 9), (4, 8), paired_samples=True)

    def test_matches_with_one_chunk_and_cell_options(self):
        # (train_input's LGN weight gradient sums with atomics: not run-to-run
        # deterministic, so it is left to the recorded adjoint's same-trajectory test.)
        self._case(self.SEQ_LEN, (2, 9), (0, 11), hard_reset=True, lr_scale=3.0)


if __name__ == "__main__":
    unittest.main()
