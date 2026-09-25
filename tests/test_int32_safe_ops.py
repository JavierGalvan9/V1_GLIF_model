"""Regression tests for the remaining int32 element-limit hazards.

TensorFlow's stock GPU kernels index their operands with int32. Above 2**31
elements the unary elementwise ops (cast, square, ...), the whole-tensor
reductions, gather, axis-0 concat with 16+ inputs, and the fills (ones_like)
all return wrong answers with no error raised, and some of them write outside
their output buffer and corrupt neighbouring tensors. See
int32_overflow_audit_20260902/.

These tests pin the shapes of the fixes, not the GPU behaviour itself: they run
anywhere, including on CPU, and assert that each rewritten code path still
computes what the original computed.
"""

import unittest

import numpy as np
import tensorflow as tf

import multi_training
from v1_model_utils import loss_functions

INT32_MAX = 2**31 - 1


class Int32SafeCastTest(unittest.TestCase):
    """multi_training.int32_safe_cast -- hazard 5."""

    def test_matches_tf_cast_below_the_limit(self):
        values = tf.cast(
            tf.random.stateless_uniform((3, 50, 7), seed=(1, 2)) < 0.2, tf.bool
        )
        np.testing.assert_array_equal(
            multi_training.int32_safe_cast(values, tf.float16).numpy(),
            tf.cast(values, tf.float16).numpy(),
        )

    def test_chunked_path_matches_the_unchunked_one(self):
        """Force chunking with a tiny limit; the answer must not change."""
        values = tf.cast(
            tf.random.stateless_uniform((3, 53, 7), seed=(3, 4)) < 0.2, tf.bool
        )
        chunked = multi_training.int32_safe_cast(
            values, tf.float16, chunk_size=5, full_tensor_element_limit=1
        )
        self.assertEqual(chunked.shape, values.shape)
        self.assertEqual(chunked.dtype, tf.float16)
        np.testing.assert_array_equal(
            chunked.numpy(), tf.cast(values, tf.float16).numpy()
        )

    def test_ragged_final_chunk_is_kept(self):
        """53 is not a multiple of 5: the last chunk is short, not dropped."""
        values = tf.ones((2, 53, 3), tf.float32)
        out = multi_training.int32_safe_cast(
            values, tf.float16, chunk_size=5, full_tensor_element_limit=1
        )
        self.assertEqual(out.shape, (2, 53, 3))
        self.assertEqual(float(tf.reduce_sum(tf.cast(out, tf.float32))), 2 * 53 * 3)

    def test_no_op_when_dtype_already_matches(self):
        values = tf.ones((2, 4, 3), tf.float16)
        self.assertIs(multi_training.int32_safe_cast(values, tf.float16), values)

    def test_rejects_shapes_it_cannot_chunk(self):
        with self.assertRaisesRegex(ValueError, "rank>=2"):
            multi_training.int32_safe_cast(
                tf.ones((8,), tf.float32), tf.float16,
                full_tensor_element_limit=1,
            )

    def test_concatenate_stimulus_batches_casts_and_joins(self):
        grating = tf.cast(tf.ones((2, 6, 3)), tf.bool)
        spontaneous = tf.cast(tf.zeros((3, 6, 3)), tf.bool)
        out = multi_training.concatenate_stimulus_batches(
            grating, spontaneous, tf.float16
        )
        self.assertEqual(out.shape, (5, 6, 3))
        self.assertEqual(out.dtype, tf.float16)
        self.assertEqual(float(tf.reduce_sum(tf.cast(out, tf.float32))), 2 * 6 * 3)


class VoltageCoreSelectionTest(unittest.TestCase):
    """VoltageRegularization -- hazard 2."""

    class _Cell:
        pass

    def _make(self, core_mask, penalty_mode="range"):
        return loss_functions.VoltageRegularization(
            self._Cell(), voltage_cost=1.0, dtype=tf.float32,
            core_mask=core_mask, penalty_mode=penalty_mode,
        )

    def test_selection_matches_a_boolean_mask(self):
        """Gathering static indices must equal the boolean mask it replaced."""
        mask_np = np.array([True, False, True, True, False, False, True])
        voltages = tf.random.stateless_uniform((3, 5, 7), seed=(5, 6)) * 2.5 - 0.7
        for mode in ("range", "threshold"):
            with self.subTest(mode=mode):
                masked = tf.boolean_mask(voltages, mask_np, axis=2)
                expected = self._make(None, mode)(masked)
                actual = self._make(tf.constant(mask_np), mode)(voltages)
                np.testing.assert_allclose(
                    float(actual), float(expected), rtol=1e-6
                )

    def test_no_mask_uses_every_neuron(self):
        voltages = tf.random.stateless_uniform((2, 4, 6), seed=(7, 8))
        reg = self._make(None)
        self.assertGreaterEqual(float(reg(voltages)), 0.0)
        self.assertIsNone(reg._core_indices)

    def test_gradient_reaches_only_core_neurons(self):
        mask_np = np.array([True, False, False, True])
        reg = self._make(tf.constant(mask_np))
        voltages = tf.Variable(
            tf.random.stateless_uniform((2, 3, 4), seed=(9, 10)) * 3.0 - 1.0
        )
        with tf.GradientTape() as tape:
            loss = reg(voltages)
        grad = tape.gradient(loss, voltages).numpy()
        self.assertTrue(np.any(grad[:, :, [0, 3]] != 0.0))
        np.testing.assert_array_equal(grad[:, :, [1, 2]], 0.0)

    def test_non_static_mask_is_rejected(self):
        """A mask that is not known at build time cannot be turned into
        indices, and silently falling back to boolean_mask is the bug."""
        with self.assertRaisesRegex(ValueError, "statically known"):
            self._make(tf.Variable([True, False, True]))


class UnchunkedRnnOutputGuardTest(unittest.TestCase):
    """multi_training.require_int32_safe_rnn_output -- hazard 3."""

    def test_allows_a_shape_under_the_limit(self):
        # 203,816 neurons at batch 32, seq_len 300: 1.96e9, the largest shape
        # the un-checkpointed path handles correctly.
        self.assertEqual(
            multi_training.require_int32_safe_rnn_output(32, 300, 203816),
            32 * 300 * 203816,
        )

    def test_rejects_the_active_203k_batch_64_shape(self):
        with self.assertRaisesRegex(ValueError, "gradient_checkpointing"):
            multi_training.require_int32_safe_rnn_output(64, 500, 203816)

    def test_boundary_is_the_int32_element_limit_exactly(self):
        multi_training.require_int32_safe_rnn_output(1, 1, INT32_MAX)
        with self.assertRaises(ValueError):
            multi_training.require_int32_safe_rnn_output(1, 1, INT32_MAX + 1)

    def test_message_names_the_offending_shape(self):
        with self.assertRaises(ValueError) as ctx:
            multi_training.require_int32_safe_rnn_output(256, 500, 66652)
        self.assertIn("[256, 500, 66652]", str(ctx.exception))


class SpikeBasedOsiLossTest(unittest.TestCase):
    """OrientationSelectivityLoss crowd_spikes / neuropixels_fr -- hazard 4."""

    def _loss(self, network, core_mask=None):
        return loss_functions.OrientationSelectivityLoss(
            network=network, osi_cost=1.0, method="crowd_spikes",
            core_mask=core_mask, dtype=tf.float32, pre_delay=0, post_delay=0,
        )

    def test_crowd_spikes_matches_the_pre_change_formulation(self):
        """Cast+mask+average-over-time, in the old order, on the raw sequence."""
        n_neurons, batch, duration = 6, 3, 40
        network = {"tuning_angle": np.linspace(0, 180, n_neurons).astype(np.float32)}
        angle = tf.constant([[0.0], [45.0], [90.0]])
        spikes = tf.cast(
            tf.random.stateless_uniform((batch, duration, n_neurons), seed=(11, 12))
            < 0.15,
            tf.float16,
        )
        for mask in (None, np.array([True, False, True, True, False, True])):
            with self.subTest(mask="core" if mask is not None else "all"):
                loss = self._loss(network, None if mask is None else tf.constant(mask))
                actual = loss(spikes, angle=angle, trim=False)

                # The formulation this replaced, spelled out: cast the whole
                # sequence, mask it, then average over time.
                reference = tf.cast(spikes, tf.float32)
                if mask is not None:
                    reference = tf.boolean_mask(reference, mask, axis=2)
                mean_spikes = tf.reduce_mean(reference, axis=1)
                mean_angle = mean_spikes * loss.calculate_delta_angle(
                    angle, loss._tuning_angles
                )
                expected = (
                    tf.reduce_mean(tf.abs(mean_angle))
                    - tf.reduce_mean(mean_spikes) * 45 * loss._subtraction_ratio
                ) * loss._osi_cost

                np.testing.assert_allclose(
                    float(actual), float(expected), rtol=1e-5, atol=1e-7
                )

    def test_time_average_is_exact_for_binary_spikes(self):
        """The rewrite sums in the spike dtype before casting; for 0/1 spikes
        over a 500 ms window every partial sum is exact in fp16."""
        spikes = tf.cast(
            tf.random.stateless_uniform((2, 500, 5), seed=(13, 14)) < 0.3, tf.float16
        )
        counts = loss_functions.temporal_sum(spikes, dtype=tf.float32)
        expected = tf.reduce_sum(tf.cast(spikes, tf.float32), axis=1)
        np.testing.assert_array_equal(counts.numpy(), expected.numpy())


if __name__ == "__main__":
    unittest.main()

