"""TensorFlow's gradient of a CUDA rollout is the same with the step gradient link.

With the link, each timestep's GLIF gradient hands the recurrent backward of the
same timestep the abs-max bound of the currents' gradient and the slots' other
gradients, which the operator adds itself (StepGradientLink); in the segmented
runner's replay the link also carries the slots' activity masks. Each case
differentiates the same small forward with the link and without it (each
operator's gradients summed by TensorFlow), in separate runs: the gradients of the
weights and of every floating initial state must agree bit for bit with those of
one run without the link (the case is skipped when no run matches and the runs
without the link already differ among themselves). (A float32 forward
is not run-to-run deterministic; float32 is checked on one trajectory by
tests/test_recorded_adjoint.py, whose autodiff reference takes the link.)
"""

import unittest
from unittest import mock

import numpy as np
import tensorflow as tf

from tests.network_fixtures import load_network
from v1_model_utils import cuda_csr_recurrent, models, spatial_layout
from v1_model_utils.cuda_csr_recurrent import wrapper as recurrent_wrapper


class StepGradientLinkTest(unittest.TestCase):
    BATCH, SEQ_LEN = 4, 9

    def setUp(self):
        if not tf.config.list_physical_devices("GPU"):
            self.skipTest("CUDA GPU required")

    def _case(self, dtype=tf.float16, chunk_size=None, **model_options):
        if dtype == tf.float16:
            tf.keras.mixed_precision.set_global_policy("mixed_float16")
            self.addCleanup(tf.keras.mixed_precision.set_global_policy, "float32")
        network, lgn, bkg = load_network(1000)
        edge_orders = None
        if cuda_csr_recurrent.DIRECT_CSR:
            network, lgn, bkg, edge_orders = spatial_layout.apply_csr_edge_order(
                network, lgn, bkg, max_delay=3
            )
        n_input = int(lgn["n_inputs"])
        batch, length = self.BATCH, self.SEQ_LEN
        options = dict(
            seq_len=length, n_input=n_input, n_output=2, dtype=dtype, batch_size=batch,
            max_delay=3, train_recurrent=True, train_noise=True, use_state_input=True,
            return_state=True, add_metric=False, seed=3, acceleration="cuda",
            edge_orders=edge_orders, track_voltage_penalty=True,
            return_voltage_sequences=False, recurrent_dampening_factor=0.1,
        )
        model = models.create_model(network, lgn, bkg, **{**options, **model_options})
        model.build((batch, length, n_input))
        layer = model.get_layer("rsnn")
        cell = layer.cell
        core = models.build_sequence_and_state_model(model, layer)
        runner = (
            None if chunk_size is None
            else models.SegmentedRecomputeRunner(
                core, sequence_length=length, chunk_size=chunk_size,
                differentiate_inputs=False, cell=cell,
            )
        )
        n = cell._n_neurons
        rng = np.random.default_rng(0)
        spike_dtype = dtype.as_numpy_dtype
        inputs = tf.constant((rng.random((batch, length, n_input)) < 0.02).astype(spike_dtype))
        state = list(cell.zero_state(batch, tf.float32))
        state[0] = tf.constant((rng.random((batch, 3 * n)) < 0.02).astype(spike_dtype))
        state[1] = tf.constant(rng.normal(0.3, 0.3, (batch, n)).astype(np.float32))
        state = tuple(state)
        weights = tf.constant(rng.normal(size=(batch, length, n)).astype(np.float32))
        variables = model.trainable_variables
        floating = [state[i] for i, value in enumerate(state) if value.dtype.is_floating]
        hand_offs, masked = [], []
        hand_off = recurrent_wrapper.StepGradientLink.hand_off
        take = recurrent_wrapper.StepGradientLink.take

        def counting_take(link):
            options = take(link)
            masked.append("activity_masks" in options)
            return options

        def counting_hand_off(link, *args):
            hand_offs.append(link)
            return hand_off(link, *args)

        def gradients(linked):
            @tf.function
            def run():
                with tf.GradientTape() as tape:
                    tape.watch(floating)
                    outputs = (
                        core((inputs, state)) if runner is None else runner(inputs, state)
                    )
                    outputs = tf.nest.flatten(outputs)
                    spikes, final_state = outputs[0], outputs[1:]
                    loss = tf.reduce_sum(tf.cast(spikes, tf.float32) * weights)
                    loss += 0.1 * tf.reduce_sum(final_state[1])
                    loss += 3.0 * tf.reduce_mean(final_state[-1])
                    loss += 0.2 * tf.reduce_sum(tf.cast(final_state[0], tf.float32))
                return tape.gradient(loss, list(variables) + floating)

            with mock.patch.object(recurrent_wrapper, "LINK_STEP_GRADIENTS", linked), \
                    mock.patch.object(recurrent_wrapper.StepGradientLink, "hand_off",
                                      counting_hand_off), \
                    mock.patch.object(recurrent_wrapper.StepGradientLink, "take", counting_take):
                return run()

        def values(gradients):
            return [np.asarray(tf.convert_to_tensor(g)) for g in gradients]

        def same(a, b):
            return all(np.array_equal(x, y) for x, y in zip(a, b))

        # Atomic accumulation orders can change a run's trajectory: the link is
        # judged against every reference trajectory seen, and a mismatch only
        # counts when the references agree with each other.
        references = [values(gradients(False)) for _ in range(2)]
        self.assertFalse(hand_offs or any(masked))
        attempts = [values(gradients(True)) for _ in range(2)]
        self.assertTrue(hand_offs, "the link never handed off")
        # Only the segmented replay carries the activity masks.
        self.assertEqual(any(masked), runner is not None)
        names = [v.name for v in variables] + [f"state_{i}" for i in range(len(floating))]
        for name, reference in zip(names, references[0]):
            if name in ("sparse_recurrent_weights", "rest_of_brain_weights"):
                self.assertTrue(np.any(reference != 0), f"{name}: zero gradient")
        if any(same(a, r) for a in attempts for r in references):
            return
        if not same(*references):
            self.skipTest("the forward is not run-to-run deterministic here")
        for name, a, b in zip(names, attempts[0], references[0]):
            np.testing.assert_array_equal(a, b, err_msg=name)

    def test_keras_rollout_float16(self):
        self._case()

    def test_keras_rollout_unscaled_currents_and_hard_reset(self):
        self._case(lr_scale=1.0, hard_reset=True)

    def test_segmented_replay(self):
        self._case(chunk_size=4)


if __name__ == "__main__":
    unittest.main()
