"""Correctness of the single-batch per-neuron OSI/DSI estimator.

`osi_loss_method='batch_osi_emd'` estimates every neuron's OSI and DSI from the
gratings samples in one training batch and matches the resulting distribution,
per cell type, against the Neuropixels one. That only works if the per-neuron
estimate is actually unbiased at realistic batch sizes, so these tests drive it
with synthetic Poisson spikes whose tuning curves -- and therefore whose true
OSI/DSI -- are known exactly by construction.
"""

import unittest

import numpy as np
import tensorflow as tf

from v1_model_utils.loss_functions import OrientationSelectivityLoss

WINDOW_MS = 500.0


def _estimator(tuning_angles, **overrides):
    """An OrientationSelectivityLoss carrying only what the estimator touches.

    Building the real object would load the Neuropixels targets and the BMTK
    node tables; these tests are about the estimator maths, which depends on
    none of that.
    """
    obj = object.__new__(OrientationSelectivityLoss)
    obj._dtype = tf.float32
    obj._tf_pi = tf.constant(np.pi, dtype=tf.float32)
    obj._core_mask = None
    obj._tuning_angles = tf.constant(tuning_angles, dtype=tf.float32)
    obj._min_rates_threshold = tf.constant(0.0005, dtype=tf.float32)
    obj._rolling_epsilon = tf.constant(1e-6, dtype=tf.float32)
    obj._batch_emd_alignment_weight = 1.0
    obj.__dict__.update(overrides)
    return obj


def _single_type_loss(tuning_angle, osi_target=0.4, dsi_target=0.0):
    """Minimal public loss fixture for one neuron and one cell type."""
    loss = _estimator([tuning_angle], _osi_cost=1.0)
    loss._emd_group_indices = tf.ragged.constant([[0]], dtype=tf.int32)
    loss._n_node_types = tf.constant(1, dtype=tf.int32)
    loss._osi_target_distributions = tf.ragged.constant(
        [[osi_target]], dtype=tf.float32
    )
    loss._dsi_target_distributions = tf.ragged.constant(
        [[dsi_target]], dtype=tf.float32
    )
    loss._osi_empirical_distributions = loss._osi_target_distributions
    loss._dsi_empirical_distributions = loss._dsi_target_distributions
    loss.cell_type_count = tf.constant([1.0], dtype=tf.float32)
    return loss


def _tuning_curve(theta_deg, pref_deg, baseline, amp, kappa, null_ratio):
    """Rates in Hz for a pair of opposed von Mises lobes.

    ``null_ratio=1`` gives a purely orientation-tuned cell (DSI 0) and
    ``null_ratio=0`` a fully direction-selective one.
    """
    theta = np.deg2rad(np.atleast_1d(theta_deg))[:, None]
    pref = np.deg2rad(pref_deg)[None, :]
    preferred_lobe = np.exp(kappa * (np.cos(theta - pref) - 1.0))
    null_lobe = np.exp(kappa * (np.cos(theta - pref - np.pi) - 1.0))
    return baseline + amp * (preferred_lobe + null_ratio * null_lobe)


def _true_selectivity(params, n_grid=2880):
    """OSI/DSI of the noiseless tuning curve, by dense numerical integration."""
    grid = np.linspace(0.0, 360.0, n_grid, endpoint=False)
    curve = _tuning_curve(grid, *params)
    theta = np.deg2rad(grid)[:, None]
    denominator = curve.sum(axis=0)
    osi = np.abs((curve * np.exp(2j * theta)).sum(axis=0)) / denominator
    dsi = np.abs((curve * np.exp(1j * theta)).sum(axis=0)) / denominator
    return osi, dsi


def _population(n_neurons=3000, seed=0):
    rng = np.random.default_rng(seed)
    params = (
        rng.uniform(0.0, 360.0, n_neurons),      # preferred angle
        rng.uniform(1.0, 6.0, n_neurons),        # baseline, Hz
        rng.uniform(0.0, 30.0, n_neurons),       # amplitude, Hz (0 -> untuned)
        rng.uniform(0.5, 6.0, n_neurons),        # kappa
        rng.uniform(0.0, 1.0, n_neurons),        # null/preferred ratio
    )
    return params, *_true_selectivity(params)


def _sample(params, batch, seed, rate_scale=1.0):
    """Poisson spike counts over a `WINDOW_MS` window, as spikes per ms."""
    rng = np.random.default_rng(seed)
    angles = rng.uniform(0.0, 360.0, batch)
    lam = _tuning_curve(angles, *params) * rate_scale * WINDOW_MS / 1000.0
    rates = rng.poisson(lam) / WINDOW_MS
    return angles.astype(np.float32), rates.astype(np.float32)


def _naive(rates, angles, harmonic):
    """The uncorrected estimator, i.e. what the debiasing is measured against."""
    theta = np.deg2rad(angles)[:, None]
    moment = (rates * np.exp(1j * harmonic * theta)).sum(axis=0)
    return np.abs(moment) / np.maximum(rates.sum(axis=0), 1e-12)


class BatchOsiEstimatorTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.params, cls.osi_true, cls.dsi_true = _population()
        cls.estimator = _estimator(cls.params[0])

    def _estimate(self, batch, seed, **kwargs):
        angles, rates = _sample(self.params, batch, seed, **kwargs)
        return self.estimator.batch_osi_dsi_estimates(
            tf.constant(rates), tf.constant(angles)
        ), angles, rates

    def test_squared_estimate_is_unbiased(self):
        """OSI^2 and DSI^2 are what the pairwise correction makes unbiased."""
        errors = []
        for seed in range(24):
            angles, rates = _sample(self.params, 56, 100 + seed)
            osi_sq, dsi_sq, _, _ = self.estimator.batch_squared_osi_dsi_estimates(
                tf.constant(rates), tf.constant(angles)
            )
            errors.append(
                (
                    np.mean(osi_sq.numpy() - self.osi_true ** 2),
                    np.mean(dsi_sq.numpy() - self.dsi_true ** 2),
                )
            )
        osi_bias, dsi_bias = np.mean(errors, axis=0)
        self.assertLess(abs(osi_bias), 0.015, f"OSI^2 bias {osi_bias:+.4f}")
        self.assertLess(abs(dsi_bias), 0.015, f"DSI^2 bias {dsi_bias:+.4f}")

    def test_debiasing_removes_the_untuned_noise_floor(self):
        """An untuned neuron must not measure as selective at batch 56.

        This is the whole reason the crowd_osi workaround existed: the naive
        per-neuron estimate reads ~0.14 for cells whose true OSI is ~0.03.
        """
        untuned = self.osi_true < 0.05
        self.assertGreater(untuned.sum(), 100)
        debiased, naive = [], []
        for seed in range(24):
            (osi, _, _, _), angles, rates = self._estimate(56, 200 + seed)
            debiased.append(np.mean(osi.numpy()[untuned]))
            naive.append(np.mean(_naive(rates, angles, 2)[untuned]))
        debiased, naive = np.mean(debiased), np.mean(naive)
        truth = self.osi_true[untuned].mean()
        self.assertLess(debiased - truth, 0.5 * (naive - truth))
        self.assertLess(debiased, 0.08)

    def test_bias_shrinks_with_batch_size(self):
        bias = {}
        for batch in (32, 128, 512):
            errors = []
            for seed in range(12):
                angles, rates = _sample(self.params, batch, 300 + seed)
                osi_sq, _, _, _ = self.estimator.batch_squared_osi_dsi_estimates(
                    tf.constant(rates), tf.constant(angles)
                )
                errors.append(np.mean(osi_sq.numpy() - self.osi_true ** 2))
            bias[batch] = abs(np.mean(errors))
        self.assertLess(bias[128], bias[32])
        self.assertLess(bias[512], bias[128])

    def test_matches_the_eight_angle_protocol_estimator(self):
        """Same definition as `calculate_OSI_DSI`, which validation reports."""
        protocol = np.arange(0, 360, 45).astype(float)
        curve = _tuning_curve(protocol, *self.params)
        theta = np.deg2rad(protocol)[:, None]
        denominator = curve.sum(axis=0)
        osi_8 = np.abs((curve * np.exp(2j * theta)).sum(axis=0)) / denominator
        dsi_8 = np.abs((curve * np.exp(1j * theta)).sum(axis=0)) / denominator

        (osi, dsi, _, _), _, _ = self._estimate(4096, 7)
        for estimated, reference, name in (
            (osi.numpy(), osi_8, "OSI"),
            (dsi.numpy(), dsi_8, "DSI"),
        ):
            self.assertGreater(np.corrcoef(estimated, reference)[0, 1], 0.99, name)
            self.assertLess(np.mean(np.abs(estimated - reference)), 0.03, name)

    def test_normalizer_does_not_change_the_estimate(self):
        """`normalizer` is ignored: the estimate must be bit-identical with it.

        Selectivity is a per-neuron ratio, so the divide would cancel out of
        both the numerator and denominator.
        """
        (osi_plain, dsi_plain, _, _), angles, rates = self._estimate(56, 5)
        normalizer = tf.constant(
            np.maximum(rates.mean(axis=0), 1e-4).astype(np.float32)
        )
        osi_scaled, dsi_scaled, _, _ = self.estimator.batch_osi_dsi_estimates(
            tf.constant(rates), tf.constant(angles), normalizer=normalizer
        )
        np.testing.assert_array_equal(osi_plain.numpy(), osi_scaled.numpy())
        np.testing.assert_array_equal(dsi_plain.numpy(), dsi_scaled.numpy())

    def test_estimate_is_invariant_to_overall_rate_scaling(self):
        """OSI is a ratio: multiplying every rate by a constant changes nothing.

        Both the pairwise numerator and its denominator scale quadratically.
        """
        (osi_plain, dsi_plain, _, _), angles, rates = self._estimate(56, 5)
        osi_scaled, dsi_scaled, _, _ = self.estimator.batch_osi_dsi_estimates(
            tf.constant(rates * 7.0), tf.constant(angles)
        )
        np.testing.assert_allclose(
            osi_plain.numpy(), osi_scaled.numpy(), atol=1e-4
        )
        np.testing.assert_allclose(
            dsi_plain.numpy(), dsi_scaled.numpy(), atol=1e-4
        )

    def test_estimate_is_invariant_to_per_neuron_rate_scaling(self):
        """Each neuron's independent rate scale cancels from its ratio."""
        (osi_plain, _, _, _), angles, rates = self._estimate(56, 5)
        rng = np.random.default_rng(4)
        scale = rng.uniform(0.25, 4.0, rates.shape[1]).astype(np.float32)
        osi_scaled, _, _, _ = self.estimator.batch_osi_dsi_estimates(
            tf.constant(rates * scale[None, :]), tf.constant(angles)
        )
        np.testing.assert_allclose(
            osi_plain.numpy(), osi_scaled.numpy(), atol=1e-5
        )

    def test_orthogonal_power_tracks_preferred_angle_drift(self):
        """The alignment term must read the offset between actual and wired angle."""
        strong = self.osi_true > 0.3
        for offset in (0.0, 10.0, 30.0, 45.0):
            estimator = _estimator(self.params[0] + offset)
            angles, rates = _sample(self.params, 4096, 13)
            _, _, orthogonal, _ = estimator.batch_osi_dsi_estimates(
                tf.constant(rates), tf.constant(angles)
            )
            expected = np.mean(
                self.osi_true[strong] * abs(np.sin(2 * np.deg2rad(offset)))
            )
            self.assertAlmostEqual(
                float(np.mean(orthogonal.numpy()[strong])), expected, delta=0.01,
                msg=f"offset {offset} deg",
            )

    def test_orthogonal_power_has_no_firing_rate_floor(self):
        """Scaling all rates must not change the alignment term.

        Its uncorrected form `mean((B/T)^2)` has a noise floor proportional to
        1/rate, which would let the optimiser lower the OSI loss just by firing
        more. The pairwise correction is what removes that.
        """
        aligned = _estimator(self.params[0])
        values = []
        for rate_scale in (0.5, 4.0):
            per_scale = []
            for seed in range(8):
                angles, rates = _sample(
                    self.params, 56, 600 + seed, rate_scale=rate_scale
                )
                _, _, orthogonal_sq, _ = aligned.batch_squared_osi_dsi_estimates(
                    tf.constant(rates), tf.constant(angles)
                )
                per_scale.append(np.mean(orthogonal_sq.numpy()))
            values.append(np.mean(per_scale))
        self.assertLess(abs(values[0] - values[1]), 0.01, f"{values}")

    def test_silent_and_single_spike_neurons_stay_finite(self):
        angles, rates = _sample(self.params, 56, 17)
        rates[:, :50] = 0.0                       # never fires
        rates[:, 50:100] = 0.0
        rates[0, 50:100] = 1.0 / WINDOW_MS        # one spike, in one trial
        osi, dsi, orthogonal, _ = self.estimator.batch_osi_dsi_estimates(
            tf.constant(rates), tf.constant(angles)
        )
        for tensor in (osi, dsi, orthogonal):
            values = tensor.numpy()
            self.assertTrue(np.all(np.isfinite(values)))
            self.assertLess(values[:100].max(), 1e-3)

    def test_gradients_are_finite_and_reach_weakly_tuned_neurons(self):
        angles, rates = _sample(self.params, 56, 23)
        rates[:, :50] = 0.0
        variable = tf.Variable(rates)
        with tf.GradientTape() as tape:
            osi_sq, dsi_sq, orthogonal_sq, _ = (
                self.estimator.batch_squared_osi_dsi_estimates(
                    variable, tf.constant(angles)
                )
            )
            loss = (
                tf.reduce_mean(osi_sq)
                + tf.reduce_mean(dsi_sq)
                + tf.reduce_mean(orthogonal_sq)
            )
        gradient = tape.gradient(loss, variable).numpy()
        self.assertTrue(np.all(np.isfinite(gradient)))
        self.assertEqual(np.abs(gradient[:, :50]).max(), 0.0)
        # Working in squared space is what keeps every live neuron differentiable;
        # clipping a negative OSI^2 before the sqrt would zero a quarter of them.
        live = np.abs(gradient[:, 100:]).max(axis=0)
        self.assertEqual(np.count_nonzero(live == 0.0), 0)

    def test_recovers_selectivity_from_binary_spike_trains(self):
        """The real entry point takes [batch, time, neurons] spikes."""
        n_neurons, batch, duration = 300, 56, 600
        params = tuple(component[:n_neurons] for component in self.params)
        osi_true, dsi_true = _true_selectivity(params)
        estimator = _estimator(params[0])
        estimator._pre_delay, estimator._post_delay = 50, 50

        rng = np.random.default_rng(3)
        angles = rng.uniform(0.0, 360.0, batch).astype(np.float32)
        probability = _tuning_curve(angles, *params)[:, None, :] / 1000.0
        spikes = (rng.random((batch, duration, n_neurons)) < probability).astype(
            np.float32
        )
        rates = estimator.rates_per_sample_from_spikes(
            tf.constant(spikes), trim=False
        )
        osi, dsi, _, _ = estimator.batch_osi_dsi_estimates(
            rates, tf.constant(angles)
        )
        self.assertGreater(np.corrcoef(osi.numpy(), osi_true)[0, 1], 0.7)
        self.assertGreater(np.corrcoef(dsi.numpy(), dsi_true)[0, 1], 0.7)
        self.assertLess(abs(np.mean(osi.numpy() - osi_true)), 0.06)

    def test_loss_rejects_ninety_degree_orientation_reversal(self):
        """Linear EMD must retain the sign of the aligned component."""
        angles = np.arange(360, dtype=np.float32)
        rates = (1.0 + 0.8 * np.cos(2.0 * np.deg2rad(angles)))[:, None]

        aligned = _single_type_loss(0.0)
        reversed_orientation = _single_type_loss(90.0)
        aligned_loss = aligned.batch_osi_emd_loss_from_rates(
            tf.constant(rates), tf.constant(angles)
        )
        reversed_loss = reversed_orientation.batch_osi_emd_loss_from_rates(
            tf.constant(rates), tf.constant(angles)
        )

        self.assertLess(float(aligned_loss.numpy()), 1e-5)
        self.assertGreater(float(reversed_loss.numpy()), 0.2)

    def test_loss_rejects_opposite_direction(self):
        """A direction opposite to the wired preference is not equivalent."""
        angles = np.arange(360, dtype=np.float32)
        rates = (1.0 + 0.8 * np.cos(np.deg2rad(angles)))[:, None]

        aligned = _single_type_loss(0.0, osi_target=0.0, dsi_target=0.4)
        reversed_direction = _single_type_loss(
            180.0, osi_target=0.0, dsi_target=0.4
        )
        aligned_loss = aligned.batch_osi_emd_loss_from_rates(
            tf.constant(rates), tf.constant(angles)
        )
        reversed_loss = reversed_direction.batch_osi_emd_loss_from_rates(
            tf.constant(rates), tf.constant(angles)
        )

        self.assertLess(float(aligned_loss.numpy()), 1e-5)
        self.assertGreater(float(reversed_loss.numpy()), 0.2)

    def test_loss_excludes_silent_neurons(self):
        """Undefined selectivity must not be inserted into the EMD as zero."""
        angles = np.arange(360, dtype=np.float32)
        active_rates = 1.0 + 0.8 * np.cos(2.0 * np.deg2rad(angles))
        rates = np.stack([active_rates, np.zeros_like(active_rates)], axis=1)
        loss = _single_type_loss(0.0)
        loss._tuning_angles = tf.constant([0.0, 0.0], dtype=tf.float32)
        loss._emd_group_indices = tf.ragged.constant([[0, 1]], dtype=tf.int32)
        loss.cell_type_count = tf.constant([2.0], dtype=tf.float32)

        value = loss.batch_osi_emd_loss_from_rates(
            tf.constant(rates), tf.constant(angles)
        )

        self.assertLess(float(value.numpy()), 1e-5)

    def test_loss_weights_valid_cell_types_by_model_population_size(self):
        """Cell-type EMDs are averaged using their model population sizes."""
        angles = np.arange(360, dtype=np.float32)
        untuned = np.ones_like(angles)
        tuned = 1.0 + 0.8 * np.cos(2.0 * np.deg2rad(angles))
        rates = np.stack([untuned, tuned, tuned], axis=1)
        loss = _estimator([0.0, 0.0, 0.0], _osi_cost=1.0)
        loss._emd_group_indices = tf.ragged.constant(
            [[0], [1, 2]], dtype=tf.int32
        )
        loss._n_node_types = tf.constant(2, dtype=tf.int32)
        loss._osi_empirical_distributions = tf.ragged.constant(
            [[0.0], [0.0]], dtype=tf.float32
        )
        loss._dsi_empirical_distributions = tf.ragged.constant(
            [[0.0], [0.0]], dtype=tf.float32
        )
        loss.cell_type_count = tf.constant([1.0, 2.0], dtype=tf.float32)

        value = loss.batch_osi_emd_loss_from_rates(
            tf.constant(rates), tf.constant(angles)
        )

        self.assertAlmostEqual(float(value.numpy()), 0.8 / 3.0, places=5)

    def test_loss_traces_with_dynamic_active_neuron_count(self):
        """The training loss must support graph execution with a dynamic mask."""
        loss = _single_type_loss(0.0)
        loss._tuning_angles = tf.constant([0.0, 0.0], dtype=tf.float32)
        loss._emd_group_indices = tf.ragged.constant([[0, 1]], dtype=tf.int32)
        loss.cell_type_count = tf.constant([2.0], dtype=tf.float32)
        angles = tf.constant(np.arange(360, dtype=np.float32))

        @tf.function
        def compiled_loss(rates):
            return loss.batch_osi_emd_loss_from_rates(rates, angles)

        active = 1.0 + 0.8 * np.cos(2.0 * np.deg2rad(np.arange(360)))
        one_active = tf.constant(
            np.stack([active, np.zeros_like(active)], axis=1), dtype=tf.float32
        )
        two_active = tf.constant(
            np.stack([active, active], axis=1), dtype=tf.float32
        )

        self.assertLess(float(compiled_loss(one_active).numpy()), 1e-5)
        self.assertLess(float(compiled_loss(two_active).numpy()), 1e-5)


if __name__ == "__main__":
    unittest.main()
