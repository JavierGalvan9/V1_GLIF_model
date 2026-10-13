import os
import unittest
from unittest import mock

import numpy as np
import tensorflow as tf

from tests.network_fixtures import load_network
from v1_model_utils import models
from v1_model_utils.models import SegmentedRecomputeRunner


def _build_core(sequence_length=7, input_size=3, units=4):
    inputs = tf.keras.Input((None, input_size), dtype=tf.float32)
    initial_state = tf.keras.Input((units,), dtype=tf.float32)
    rnn = tf.keras.layers.SimpleRNN(
        units,
        activation="tanh",
        return_sequences=True,
        return_state=True,
        kernel_initializer=tf.keras.initializers.GlorotUniform(seed=11),
        recurrent_initializer=tf.keras.initializers.Orthogonal(seed=12),
        bias_initializer=tf.keras.initializers.RandomNormal(seed=13),
    )
    sequence, final_state = rnn(inputs, initial_state=initial_state)
    return tf.keras.Model(
        inputs=(inputs, (initial_state,)),
        outputs=(sequence, final_state),
    )


def _loss_and_gradients(core, inputs, initial_state, runner=None):
    with tf.GradientTape() as tape:
        tape.watch((inputs, initial_state))
        if runner is None:
            sequence, final_state = core((inputs, (initial_state,)))
        else:
            sequence, final_state = runner(inputs, (initial_state,))
        sequence_weights = tf.reshape(
            tf.linspace(0.25, 1.25, tf.shape(sequence)[1]), (1, -1, 1)
        )
        loss = (
            tf.reduce_sum(sequence * sequence_weights)
            + 0.7 * tf.reduce_sum(tf.square(final_state))
        )
    targets = (inputs, initial_state) + tuple(core.trainable_variables)
    gradients = tape.gradient(loss, targets)
    return (sequence, final_state, loss), gradients


class SegmentedRecomputeRunnerTest(unittest.TestCase):
    def setUp(self):
        tf.keras.backend.clear_session()
        tf.random.set_seed(21)
        self.sequence_length = 7
        self.core = _build_core(sequence_length=self.sequence_length)
        self.inputs = tf.random.stateless_normal((2, self.sequence_length, 3), (1, 2))
        self.initial_state = tf.random.stateless_normal((2, 4), (3, 4))

    def test_outputs_states_and_gradients_match_full_bptt(self):
        expected_outputs, expected_gradients = _loss_and_gradients(
            self.core, self.inputs, self.initial_state
        )

        for chunk_size in (1, 2, self.sequence_length):
            with self.subTest(chunk_size=chunk_size):
                runner = SegmentedRecomputeRunner(
                    self.core,
                    sequence_length=self.sequence_length,
                    chunk_size=chunk_size,
                )
                actual_outputs, actual_gradients = _loss_and_gradients(
                    self.core, self.inputs, self.initial_state, runner
                )
                for expected, actual in zip(expected_outputs, actual_outputs):
                    np.testing.assert_allclose(
                        actual.numpy(), expected.numpy(), rtol=2e-6, atol=2e-6
                    )
                for expected, actual in zip(expected_gradients, actual_gradients):
                    self.assertIsNotNone(actual)
                    np.testing.assert_allclose(
                        actual.numpy(), expected.numpy(), rtol=2e-5, atol=2e-5
                    )

    def test_can_skip_input_gradient_without_losing_variable_gradients(self):
        _, expected_gradients = _loss_and_gradients(
            self.core, self.inputs, self.initial_state
        )
        runner = SegmentedRecomputeRunner(
            self.core,
            sequence_length=self.sequence_length,
            chunk_size=3,
            differentiate_inputs=False,
        )
        _, actual_gradients = _loss_and_gradients(
            self.core, self.inputs, self.initial_state, runner
        )
        self.assertIsNone(actual_gradients[0])
        for expected, actual in zip(expected_gradients[1:], actual_gradients[1:]):
            np.testing.assert_allclose(
                actual.numpy(), expected.numpy(), rtol=2e-5, atol=2e-5
            )

    def test_matches_full_bptt_inside_tf_function(self):
        runner = SegmentedRecomputeRunner(
            self.core,
            sequence_length=self.sequence_length,
            chunk_size=3,
        )

        @tf.function
        def graph_gradients(inputs, initial_state, segmented):
            with tf.GradientTape() as tape:
                tape.watch((inputs, initial_state))
                if segmented:
                    sequence, final_state = runner(inputs, (initial_state,))
                else:
                    sequence, final_state = self.core((inputs, (initial_state,)))
                loss = tf.reduce_sum(sequence) + tf.reduce_sum(final_state)
            return tape.gradient(
                loss,
                (inputs, initial_state) + tuple(self.core.trainable_variables),
            )

        expected = graph_gradients(self.inputs, self.initial_state, False)
        actual = graph_gradients(self.inputs, self.initial_state, True)
        for expected_gradient, actual_gradient in zip(expected, actual):
            np.testing.assert_allclose(
                actual_gradient.numpy(),
                expected_gradient.numpy(),
                rtol=2e-5,
                atol=2e-5,
            )

    def test_rejects_invalid_chunk_sizes(self):
        for chunk_size in (0, self.sequence_length + 1):
            with self.subTest(chunk_size=chunk_size):
                with self.assertRaisesRegex(ValueError, "chunk_size"):
                    SegmentedRecomputeRunner(
                        self.core,
                        sequence_length=self.sequence_length,
                        chunk_size=chunk_size,
                    )

    def test_rejects_runtime_sequence_length_mismatch(self):
        runner = SegmentedRecomputeRunner(
            self.core,
            sequence_length=self.sequence_length,
            chunk_size=3,
        )
        short_inputs = self.inputs[:, :-1]
        with self.assertRaisesRegex(tf.errors.InvalidArgumentError, "input length"):
            runner(short_inputs, (self.initial_state,))


class V1SegmentedRecomputeEquivalenceTest(unittest.TestCase):
    def tearDown(self):
        tf.keras.mixed_precision.set_global_policy("float32")

    def test_v1_sequences_states_and_gradients_match(self):
        sequence_length = 7
        dtype_name = os.environ.get("V1_TEST_DTYPE", "float32")
        if dtype_name == "float16":
            tf.keras.mixed_precision.set_global_policy("mixed_float16")
            dtype = tf.float16
            rtol, atol = 3e-3, 3e-4
        else:
            tf.keras.mixed_precision.set_global_policy("float32")
            dtype = tf.float32
            rtol, atol = 2e-5, 2e-6
        network, lgn_input, bkg_input = load_network(10)
        acceleration = os.environ.get("V1_TEST_ACCELERATION", "tensorflow")
        edge_orders = None
        if acceleration == "cuda":
            network, lgn_input, bkg_input, edge_orders = (
                models.spatial_layout.apply_csr_edge_order(
                    network, lgn_input, bkg_input, max_delay=3
                )
            )
        model = models.create_model(
            network,
            lgn_input,
            bkg_input,
            seq_len=sequence_length,
            n_input=lgn_input["n_inputs"],
            dtype=dtype,
            train_recurrent=True,
            train_recurrent_per_type=False,
            train_input=False,
            train_noise=True,
            use_state_input=True,
            return_state=True,
            add_metric=False,
            batch_size=1,
            seed=31,
            acceleration=acceleration,
            edge_orders=edge_orders,
        )
        rsnn_layer = model.get_layer("rsnn")
        core = models.build_sequence_and_state_model(model, rsnn_layer)
        runner = SegmentedRecomputeRunner(
            core,
            sequence_length=sequence_length,
            chunk_size=3,
            differentiate_inputs=False,
        )
        inputs = tf.cast(
            tf.random.stateless_uniform(
                (1, sequence_length, lgn_input["n_inputs"]), (41, 42)
            ) < 0.01,
            dtype,
        )
        initial_state = rsnn_layer.cell.zero_state(1, dtype)
        differentiable_state = tuple(
            state for state in initial_state
            if state.dtype.is_floating or state.dtype.is_complex
        )

        def evaluate(segmented):
            with tf.GradientTape() as tape:
                tape.watch(differentiable_state)
                if segmented:
                    outputs = runner(inputs, initial_state)
                else:
                    sequence, *state = core((inputs, initial_state))
                    outputs = tuple(tf.nest.flatten(sequence)) + tuple(state)
                z, voltage = outputs[:2]
                final_float_states = tuple(
                    state for state in outputs[2:]
                    if state.dtype.is_floating or state.dtype.is_complex
                )
                # States keep their own precision (fp32 membrane, fp16
                # synaptic state under mixed_float16), so sum in float32.
                loss = (
                    tf.reduce_mean(tf.cast(z, tf.float32))
                    + 0.3 * tf.reduce_mean(tf.cast(voltage, tf.float32))
                    + 0.01 * tf.add_n([
                        tf.reduce_mean(tf.cast(state, tf.float32))
                        for state in final_float_states
                    ])
                )
            targets = differentiable_state + tuple(core.trainable_variables)
            return outputs, loss, tape.gradient(loss, targets)

        expected_outputs, expected_loss, expected_gradients = evaluate(False)
        actual_outputs, actual_loss, actual_gradients = evaluate(True)
        np.testing.assert_allclose(
            actual_loss.numpy(), expected_loss.numpy(), rtol=rtol, atol=atol
        )
        for expected, actual in zip(expected_outputs, actual_outputs):
            if expected.dtype.is_integer:
                np.testing.assert_array_equal(actual.numpy(), expected.numpy())
            else:
                np.testing.assert_allclose(
                    actual.numpy(), expected.numpy(), rtol=rtol, atol=atol
                )
        for expected, actual in zip(expected_gradients, actual_gradients):
            if expected is None:
                if actual is not None:
                    np.testing.assert_array_equal(actual.numpy(), np.zeros_like(actual.numpy()))
            else:
                self.assertIsNotNone(actual)
                np.testing.assert_allclose(
                    tf.convert_to_tensor(actual).numpy(),
                    tf.convert_to_tensor(expected).numpy(),
                    rtol=rtol,
                    atol=atol,
                )

        @tf.function
        def spike_only_gradients():
            with tf.GradientTape() as tape:
                outputs = runner(inputs, initial_state)
                loss = tf.reduce_mean(tf.cast(outputs[0], tf.float32))
            return tape.gradient(loss, core.trainable_variables)

        gradients = spike_only_gradients()
        self.assertTrue(any(gradient is not None for gradient in gradients))

    def test_spike_only_online_penalty_uses_sequence_shape_not_history_shape(self):
        sequence_length = 5
        dtype_name = os.environ.get("V1_TEST_DTYPE", "float32")
        if dtype_name == "float16":
            tf.keras.mixed_precision.set_global_policy("mixed_float16")
            dtype = tf.float16
        else:
            dtype = tf.float32
        network, lgn_input, bkg_input = load_network(10)
        acceleration = os.environ.get("V1_TEST_ACCELERATION", "tensorflow")
        edge_orders = None
        if acceleration == "cuda":
            network, lgn_input, bkg_input, edge_orders = (
                models.spatial_layout.apply_csr_edge_order(
                    network, lgn_input, bkg_input, max_delay=3
                )
            )
        model = models.create_model(
            network,
            lgn_input,
            bkg_input,
            seq_len=sequence_length,
            n_input=lgn_input["n_inputs"],
            dtype=dtype,
            train_recurrent=False,
            train_input=False,
            train_noise=False,
            use_state_input=True,
            return_state=True,
            add_metric=False,
            batch_size=2,
            max_delay=3,
            acceleration=acceleration,
            edge_orders=edge_orders,
            track_voltage_penalty=True,
            return_voltage_sequences=False,
        )
        rsnn_layer = model.get_layer("rsnn")
        core = models.build_sequence_and_state_model(model, rsnn_layer)
        runner = SegmentedRecomputeRunner(
            core, sequence_length=sequence_length, chunk_size=2
        )
        inputs = tf.zeros(
            (2, sequence_length, lgn_input["n_inputs"]), dtype
        )
        initial_state = rsnn_layer.cell.zero_state(2, dtype)

        @tf.function
        def rollout():
            return runner(inputs, initial_state)

        outputs = rollout()
        self.assertEqual(
            tuple(outputs[0].shape),
            (2, sequence_length, network["n_nodes"]),
        )
        self.assertIn(outputs[3].dtype, (tf.int8, tf.int16))
        self.assertEqual(outputs[7].dtype, tf.int32)
        # The voltage-penalty accumulator, the last state, stays float32
        # whatever the policy.
        self.assertEqual(outputs[-1].dtype, tf.float32)

    def test_bitpacked_spike_checkpoints_match_unpacked_runner(self):
        sequence_length = 7
        network, lgn_input, bkg_input = load_network(10)
        model = models.create_model(
            network,
            lgn_input,
            bkg_input,
            seq_len=sequence_length,
            n_input=lgn_input["n_inputs"],
            dtype=tf.float32,
            train_recurrent=True,
            train_recurrent_per_type=False,
            train_input=False,
            train_noise=True,
            use_state_input=True,
            return_state=True,
            add_metric=False,
            batch_size=1,
            seed=31,
            acceleration="tensorflow",
        )
        rsnn_layer = model.get_layer("rsnn")
        core = models.build_sequence_and_state_model(model, rsnn_layer)
        inputs = tf.cast(
            tf.random.stateless_uniform(
                (1, sequence_length, lgn_input["n_inputs"]), (51, 52)
            ) < 0.01,
            tf.float32,
        )
        initial_state = tuple(rsnn_layer.cell.zero_state(1, tf.float32))
        baseline = SegmentedRecomputeRunner(
            core,
            sequence_length=sequence_length,
            chunk_size=3,
            differentiate_inputs=False,
        )
        packed = SegmentedRecomputeRunner(
            core,
            sequence_length=sequence_length,
            chunk_size=3,
            differentiate_inputs=False,
            pack_spike_checkpoints=True,
        )

        @tf.function
        def evaluate(runner):
            with tf.GradientTape() as tape:
                outputs = runner(inputs, initial_state)
                loss = tf.reduce_mean(outputs[0]) + 0.2 * tf.reduce_mean(outputs[1])
            return outputs, tape.gradient(loss, core.trainable_variables)

        expected_outputs, expected_gradients = evaluate(baseline)
        actual_outputs, actual_gradients = evaluate(packed)
        for expected, actual in zip(expected_outputs, actual_outputs):
            np.testing.assert_allclose(actual.numpy(), expected.numpy(), atol=1e-6)
        for expected, actual in zip(expected_gradients, actual_gradients):
            self.assertEqual(expected is None, actual is None)
            if expected is not None:
                np.testing.assert_allclose(
                    tf.convert_to_tensor(actual).numpy(),
                    tf.convert_to_tensor(expected).numpy(),
                    rtol=2e-5,
                    atol=2e-6,
                )


if __name__ == "__main__":
    unittest.main()


class CsrGradientCheckpointCaptureTests(unittest.TestCase):
    """Guard the graph-capture contract of the CSR custom gradients.

    Metadata a custom gradient needs must be passed as an argument of the
    wrapped function, not captured from an enclosing scope. A captured constant
    is threaded incorrectly through `tf.while_loop`'s nested gradient function
    and raises "is in list of internal_captures but not in
    internal_capture_to_output" once the RNN is recomputed in chunks.
    """

    BATCH, SEQ_LEN = 2, 8
    # The spike comparison below needs a noise realization with no neuron at
    # threshold, where the FP16-atomic forward flips it from run to run. This
    # seed reproduces both rollouts for both replicas' seed offsets (0 flips in
    # 20 repeats each; seed 3, the model seed, flips in ~10%).
    NOISE_SEED = 24

    def setUp(self):
        if not tf.config.list_physical_devices("GPU"):
            self.skipTest("CUDA GPU required")
        tf.keras.mixed_precision.set_global_policy("mixed_float16")
        self.addCleanup(tf.keras.mixed_precision.set_global_policy, "float32")
        from v1_model_utils import cuda_csr_recurrent, spatial_layout

        network, lgn, bkg = load_network(1000)
        edge_orders = None
        if cuda_csr_recurrent.DIRECT_CSR:
            network, lgn, bkg, edge_orders = spatial_layout.apply_csr_edge_order(
                network, lgn, bkg, max_delay=3
            )
        n_input = int(lgn["n_inputs"])
        self.strategy = self._strategy()
        with self.strategy.scope():
            self.model = models.create_model(
                network, lgn, bkg, seq_len=self.SEQ_LEN, n_input=n_input, n_output=2,
                dtype=tf.float16, batch_size=self.BATCH, max_delay=3,
                train_recurrent=True, train_input=False, train_noise=True,
                use_state_input=True, return_state=True, add_metric=False,
                seed=3, acceleration="cuda", edge_orders=edge_orders,
            )
            self.model.build((self.BATCH, self.SEQ_LEN, n_input))
            self.cell = self.model.get_layer("rsnn").cell
            self.cell.noise_seed.assign(self.NOISE_SEED)
            self.sequences = models.build_sequence_and_state_model(
                self.model, self.model.get_layer("rsnn")
            )
            self.state = tuple(self.cell.zero_state(self.BATCH, tf.float32))
        rng = np.random.default_rng(0)
        self.inputs = tf.constant(
            (rng.random((self.BATCH, self.SEQ_LEN, n_input)) < 0.02).astype(np.float16)
        )

    def _strategy(self):
        return tf.distribute.get_strategy()

    def _runner(self, chunk_size, accumulated_variable=None):
        with self.strategy.scope():
            return SegmentedRecomputeRunner(
                self.sequences, sequence_length=self.SEQ_LEN, chunk_size=chunk_size,
                differentiate_inputs=False, accumulated_variable=accumulated_variable,
            )

    def _per_replica_gradients(self, rollout):
        """Each replica's own gradients and spikes from ``rollout(inputs)``.

        ``rollout`` returns ``(loss, spikes)``. The gradients are taken before
        any all-reduce, and every replica rolls a differently ordered batch
        out, so a replica that read or wrote another replica's state could not
        reproduce its result.
        """
        def replica_gradients():
            replica = tf.distribute.get_replica_context().replica_id_in_sync_group
            with tf.GradientTape() as tape:
                loss, spikes = rollout(tf.roll(self.inputs, shift=replica, axis=0))
            return tape.gradient(loss, self.model.trainable_variables), spikes

        gradients, spikes = tf.function(
            lambda: self.strategy.run(replica_gradients)
        )()

        def local(values, replica):
            return [
                np.asarray(self.strategy.experimental_local_results(value)[replica])
                for value in values
            ]

        return [
            (local(gradients, replica), local(spikes, replica))
            for replica in range(self.strategy.num_replicas_in_sync)
        ]

    @staticmethod
    def _loss(outputs):
        return tf.add_n([
            tf.reduce_sum(tf.cast(value, tf.float32))
            for value in tf.nest.flatten(outputs)
            if value.dtype.is_floating
        ])

    def test_recomputed_chunks_differentiate_through_the_csr_operators(self):
        runner = self._runner(chunk_size=4)

        def rollout(inputs):
            outputs = runner(inputs, self.state)
            return self._loss(outputs), (outputs[0],)

        for gradients, _ in self._per_replica_gradients(rollout):
            self.assertTrue(gradients)
            for gradient in gradients:
                self.assertTrue(np.isfinite(gradient).all())

    def test_accumulated_weight_gradient_matches_the_dense_path(self):
        """The in-place accumulator yields the gradient the dense path sums.

        Two rollouts in one tape exercise the zero, accumulate, read ordering:
        each backward must hand back only its own rollout's contribution. A
        chunk size that leaves a remainder covers the chunk outside the loop.

        The FP16-atomic forward is not bitwise reproducible: now and then a
        near-threshold neuron fires in one run and not in another, which moves
        the gradient by a few percent whatever the backward. Gradients are
        therefore compared only between runs whose spikes agree, and a replica
        that flipped is retried. With matching spikes the two paths agree to
        ~1e-6; double counting would be of order one.
        """
        weights = self.cell.accumulated_weight_gradient_variable
        self.assertIsNotNone(weights)
        dense = self._runner(chunk_size=3)
        seam = self._runner(chunk_size=3, accumulated_variable=weights)

        def rollouts(runner):
            def rollout(inputs):
                first = runner(inputs, self.state)
                second = runner(inputs[::-1], self.state)
                return (
                    self._loss(first) + 0.5 * self._loss(second),
                    (first[0], second[0]),
                )

            return self._per_replica_gradients(rollout)

        pending = set(range(self.strategy.num_replicas_in_sync))
        for _ in range(4):
            expected, actual = rollouts(dense), rollouts(seam)
            for replica in sorted(pending):
                (want_gradients, want_spikes) = expected[replica]
                (got_gradients, got_spikes) = actual[replica]
                if not all(map(np.array_equal, want_spikes, got_spikes)):
                    continue
                pending.discard(replica)
                for want, got in zip(want_gradients, got_gradients):
                    want, got = want.astype(np.float64), got.astype(np.float64)
                    self.assertLess(
                        np.linalg.norm(got - want) / np.linalg.norm(want), 1e-4
                    )
            if not pending:
                break
        self.assertFalse(pending, "replicas whose forward never reproduced its spikes")
        # Each replica's accumulator still holds its last backward's gradient,
        # so the seam cannot have silently fallen back to the dense path.
        for accumulator in self.strategy.experimental_local_results(
            seam._weight_gradient_accumulator
        ):
            self.assertGreater(np.abs(accumulator.numpy()).max(), 0.0)


class ReplicatedCsrGradientCheckpointTests(CsrGradientCheckpointCaptureTests):
    """The same contracts with two replicas in one process, as multi-GPU training runs.

    Connectivity then lives in per-GPU operator resources and each replica has
    its own accumulator, so every replica must reproduce its own dense-path
    gradient.
    """

    def setUp(self):
        if len(tf.config.list_physical_devices("GPU")) < 2:
            self.skipTest("two CUDA GPUs required")
        environment = mock.patch.dict(os.environ, {"V1_CSR_RESOURCE_MODE": "1"})
        environment.start()
        self.addCleanup(environment.stop)
        super().setUp()
        self.assertGreaterEqual(self.strategy.num_replicas_in_sync, 2)

    def _strategy(self):
        return tf.distribute.MirroredStrategy(
            cross_device_ops=tf.distribute.NcclAllReduce()
        )
