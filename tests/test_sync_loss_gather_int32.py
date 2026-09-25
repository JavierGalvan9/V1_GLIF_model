"""Regression tests for the INT32-safe synchronisation-loss gather.

TensorFlow's stock GPU gather indexes its operand with int32, so above 2**31
elements it silently reads the wrong addresses and its gradient aborts the
process. `_gather_population_traces` is XLA-compiled to avoid that. These tests
pin both properties that matters: it still computes what the old inline
formulation computed, and it is correct at a shape that breaks the stock
kernel.

The oversized case needs ~37 GiB of device memory, so it only runs when
`RUN_INT32_OVERFLOW_TESTS=1` and a GPU that large is present.
"""

import os
import unittest

import numpy as np
import tensorflow as tf

from v1_model_utils.loss_functions import _gather_population_traces

INT32_MAX = 2**31 - 1


def _reference(spikes, sample_ids, neuron_mask, n_trials, duration,
               per_trial, max_count):
    """The pre-XLA formulation, kept here as the definition of correct."""
    gathered = tf.gather(spikes, sample_ids, axis=2, batch_dims=1)
    gathered = tf.reshape(gathered, [n_trials, duration, per_trial, max_count])
    selected = tf.reduce_sum(gathered * neuron_mask[:, None, :, :], axis=3)
    return tf.reshape(
        tf.transpose(selected, [0, 2, 1]), [n_trials * per_trial, duration]
    )


def _inputs(n_trials, duration, n_neurons, per_trial, max_count, dtype, seed=0):
    rng = np.random.default_rng(seed)
    spikes = tf.constant(
        (rng.random((n_trials, duration, n_neurons)) < 0.05).astype("float32"),
        dtype=dtype,
    )
    counts = np.clip(
        rng.normal(70, 30, n_trials * per_trial).astype(np.int64), 15, max_count
    )
    column = np.arange(max_count)[None, :]
    mask = column < counts[:, None]
    ids = np.where(mask, rng.integers(0, n_neurons, (n_trials * per_trial, max_count)), 0)
    return (
        spikes,
        tf.constant(ids.reshape(n_trials, per_trial * max_count).astype(np.int32)),
        tf.constant(mask.reshape(n_trials, per_trial, max_count).astype("float32"),
                    dtype=dtype),
    )


class GatherPopulationTracesTest(unittest.TestCase):
    def test_matches_the_pre_xla_formulation(self):
        """Compiling the gather must not change the loss it feeds."""
        n_trials, duration, n_neurons, per_trial, max_count = 4, 60, 500, 3, 90
        for dtype in (tf.float32, tf.float16):
            with self.subTest(dtype=dtype.name):
                spikes, ids, mask = _inputs(
                    n_trials, duration, n_neurons, per_trial, max_count, dtype
                )
                expected = _reference(
                    spikes, ids, mask, n_trials, duration, per_trial, max_count
                )
                actual = _gather_population_traces(
                    spikes, ids, mask, n_trials, duration, per_trial, max_count
                )
                self.assertEqual(actual.shape, (n_trials * per_trial, duration))
                np.testing.assert_array_equal(
                    actual.numpy().astype(np.float32),
                    expected.numpy().astype(np.float32),
                )

    def test_padding_slots_contribute_nothing(self):
        """Padded slots point at a legal address and must be masked away."""
        n_trials, duration, n_neurons, per_trial, max_count = 2, 30, 200, 2, 40
        spikes, ids, mask = _inputs(
            n_trials, duration, n_neurons, per_trial, max_count, tf.float32
        )
        traces = _gather_population_traces(
            spikes, ids, mask, n_trials, duration, per_trial, max_count
        )
        # Zero the mask entirely: every trace must collapse to zero.
        zeroed = _gather_population_traces(
            spikes, ids, tf.zeros_like(mask), n_trials, duration, per_trial, max_count
        )
        self.assertGreater(float(tf.reduce_sum(traces)), 0.0)
        np.testing.assert_array_equal(zeroed.numpy(), np.zeros_like(zeroed.numpy()))

    def test_gradient_reaches_only_selected_neurons(self):
        n_trials, duration, n_neurons, per_trial, max_count = 3, 40, 300, 2, 50
        spikes, ids, mask = _inputs(
            n_trials, duration, n_neurons, per_trial, max_count, tf.float32
        )
        with tf.GradientTape() as tape:
            tape.watch(spikes)
            out = tf.reduce_sum(
                _gather_population_traces(
                    spikes, ids, mask, n_trials, duration, per_trial, max_count
                )
            )
        grad = tape.gradient(out, spikes).numpy()
        self.assertEqual(grad.shape, (n_trials, duration, n_neurons))
        for trial in range(n_trials):
            selected = set(np.asarray(ids[trial]).tolist())
            untouched = [n for n in range(n_neurons) if n not in selected]
            self.assertTrue(untouched, "test needs at least one unselected neuron")
            np.testing.assert_array_equal(
                grad[trial][:, untouched], 0.0
            )

    @unittest.skipUnless(
        os.environ.get("RUN_INT32_OVERFLOW_TESTS") == "1",
        "needs ~37 GiB of device memory; set RUN_INT32_OVERFLOW_TESTS=1",
    )
    def test_correct_above_the_int32_element_limit(self):
        """[64, 300, 203816] is 1.8x the limit and breaks the stock kernel."""
        if not tf.config.list_physical_devices("GPU"):
            self.skipTest("needs a GPU")
        n_trials, duration, n_neurons = 64, 300, 203816
        per_trial, max_count = 8, 180
        self.assertGreater(n_trials * duration * n_neurons, INT32_MAX)

        # Every neuron in trial b holds b + 1, so each trace is exactly
        # (b + 1) * (number of unmasked slots) and a misread is unmistakable.
        parts = [
            tf.fill([1, duration, n_neurons], tf.constant(b + 1, tf.float16))
            for b in range(n_trials)
        ]
        while len(parts) > 1:  # pairwise: concat is only int32-safe below 16 inputs
            parts = [
                tf.concat(parts[i:i + 2], 0) if len(parts[i:i + 2]) == 2 else parts[i]
                for i in range(0, len(parts), 2)
            ]
        spikes = parts[0]

        rng = np.random.default_rng(0)
        counts = np.clip(
            rng.normal(70, 30, n_trials * per_trial).astype(np.int64), 15, max_count
        )
        column = np.arange(max_count)[None, :]
        mask = column < counts[:, None]
        ids = np.where(
            mask, rng.integers(0, n_neurons, (n_trials * per_trial, max_count)), 0
        )
        traces = _gather_population_traces(
            spikes,
            tf.constant(ids.reshape(n_trials, per_trial * max_count).astype(np.int32)),
            tf.constant(mask.reshape(n_trials, per_trial, max_count).astype("float16")),
            n_trials, duration, per_trial, max_count,
        )
        counts_2d = counts.reshape(n_trials, per_trial)
        for trial in (0, n_trials // 2, n_trials - 1):
            for slot in (0, per_trial - 1):
                expected = float((trial + 1) * counts_2d[trial, slot])
                got = float(traces[trial * per_trial + slot, 0])
                # fp16 carries ~3 decimal digits and these sums are far larger
                # than real spike counts, so compare within one ulp.
                self.assertLess(
                    abs(got - expected), 2e-3 * expected,
                    f"trial {trial} slot {slot}: got {got}, want {expected}",
                )


if __name__ == "__main__":
    unittest.main()
