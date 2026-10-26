"""Correctness of the single-batch per-neuron OSI/DSI estimator.

`osi_loss_method='batch_osi_emd'` estimates every neuron's OSI and DSI from the
gratings samples in one training batch and matches the resulting distribution,
per cell type, against the Neuropixels one. That only works if the per-neuron
estimate is actually unbiased at realistic batch sizes, so these tests drive it
with synthetic Poisson spikes whose tuning curves -- and therefore whose true
OSI/DSI -- are known exactly by construction. The estimate is the real
component of the selectivity vector anchored at each neuron's wired preferred
angle, so noise averages out instead of adding a magnitude floor.
"""

import unittest

import numpy as np
import tensorflow as tf

from stim_dataset import stratify_angles
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
    """The per-neuron magnitude, i.e. the noise floor the signed estimate avoids."""
    theta = np.deg2rad(angles)[:, None]
    moment = (rates * np.exp(1j * harmonic * theta)).sum(axis=0)
    return np.abs(moment) / np.maximum(rates.sum(axis=0), 1e-12)


class BatchOsiEstimatorTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.params, cls.osi_true, cls.dsi_true = _population()
        cls.estimator = _estimator(cls.params[0])

    def _components(self, rates, angles, estimator=None):
        """Signed OSI/DSI components as NumPy arrays: osi, osi_imag, dsi, dsi_imag."""
        estimator = self.estimator if estimator is None else estimator
        outputs = estimator._batch_signed_osi_dsi_components(
            tf.constant(rates), tf.constant(angles)
        )
        return [tensor.numpy() for tensor in outputs[:4]]

    def test_signed_estimate_is_nearly_unbiased(self):
        """Anchored at the wired angle, the real component estimates OSI/DSI."""
        errors = []
        for seed in range(24):
            angles, rates = _sample(self.params, 56, 100 + seed)
            osi, _, dsi, _ = self._components(rates, angles)
            errors.append((np.mean(osi - self.osi_true), np.mean(dsi - self.dsi_true)))
        osi_bias, dsi_bias = np.mean(errors, axis=0)
        self.assertLess(abs(osi_bias), 0.015, f"OSI bias {osi_bias:+.4f}")
        self.assertLess(abs(dsi_bias), 0.015, f"DSI bias {dsi_bias:+.4f}")

    def test_signed_estimate_has_no_untuned_noise_floor(self):
        """An untuned neuron must not measure as selective at batch 56.

        The naive per-neuron magnitude reads ~0.14 for cells whose true OSI is
        ~0.03; the signed component averages its noise out instead.
        """
        untuned = self.osi_true < 0.05
        self.assertGreater(untuned.sum(), 100)
        signed, naive = [], []
        for seed in range(24):
            angles, rates = _sample(self.params, 56, 200 + seed)
            signed.append(np.mean(self._components(rates, angles)[0][untuned]))
            naive.append(np.mean(_naive(rates, angles, 2)[untuned]))
        truth = self.osi_true[untuned].mean()
        self.assertLess(abs(np.mean(signed) - truth), 0.01)
        self.assertGreater(np.mean(naive) - truth, 0.05)

    def test_matches_the_eight_angle_protocol_estimator(self):
        """Same definition as `calculate_OSI_DSI`, which validation reports."""
        protocol = np.arange(0, 360, 45).astype(float)
        curve = _tuning_curve(protocol, *self.params)
        theta = np.deg2rad(protocol)[:, None]
        denominator = curve.sum(axis=0)
        osi_8 = np.abs((curve * np.exp(2j * theta)).sum(axis=0)) / denominator
        dsi_8 = np.abs((curve * np.exp(1j * theta)).sum(axis=0)) / denominator

        angles, rates = _sample(self.params, 4096, 7)
        osi, _, dsi, _ = self._components(rates, angles)
        for estimated, reference, name in (
            (osi, osi_8, "OSI"),
            (dsi, dsi_8, "DSI"),
        ):
            self.assertGreater(np.corrcoef(estimated, reference)[0, 1], 0.99, name)
            self.assertLess(np.mean(np.abs(estimated - reference)), 0.03, name)

    def test_normalizer_does_not_change_the_estimate(self):
        """`normalizer` is ignored: selectivity is a per-neuron ratio."""
        angles, rates = _sample(self.params, 56, 5)
        normalizer = tf.constant(np.maximum(rates.mean(axis=0), 1e-4))
        plain = self.estimator._batch_signed_osi_dsi_components(
            tf.constant(rates), tf.constant(angles)
        )
        normalized = self.estimator._batch_signed_osi_dsi_components(
            tf.constant(rates), tf.constant(angles), normalizer=normalizer
        )
        for expected, actual in zip(plain, normalized):
            np.testing.assert_array_equal(expected.numpy(), actual.numpy())

    def test_estimate_is_invariant_to_rate_scaling(self):
        """Overall and per-neuron rate scales cancel from each neuron's ratio."""
        angles, rates = _sample(self.params, 56, 5)
        plain = self._components(rates, angles)
        scale = np.random.default_rng(4).uniform(0.25, 4.0, rates.shape[1])
        for scaled_rates in (rates * 7.0, rates * scale[None, :].astype(np.float32)):
            for expected, actual in zip(plain, self._components(scaled_rates, angles)):
                np.testing.assert_allclose(actual, expected, atol=1e-5)

    def test_imaginary_component_tracks_preferred_angle_drift(self):
        """The alignment term reads the offset between actual and wired angle."""
        strong = self.osi_true > 0.3
        angles, rates = _sample(self.params, 4096, 13)
        for offset in (0.0, 10.0, 30.0, 45.0):
            osi, osi_imag, _, _ = self._components(
                rates, angles, _estimator(self.params[0] + offset)
            )
            phase = 2 * np.deg2rad(offset)
            truth = self.osi_true[strong]
            self.assertAlmostEqual(
                float(np.mean(osi_imag[strong])), -np.mean(truth) * np.sin(phase),
                delta=0.01, msg=f"offset {offset} deg",
            )
            self.assertAlmostEqual(
                float(np.mean(osi[strong])), np.mean(truth) * np.cos(phase),
                delta=0.01, msg=f"offset {offset} deg",
            )

    def test_alignment_term_has_no_firing_rate_floor(self):
        """Firing more must not lower the alignment penalty of aligned cells."""
        values = []
        for rate_scale in (0.5, 4.0):
            per_scale = []
            for seed in range(8):
                angles, rates = _sample(self.params, 56, 600 + seed, rate_scale=rate_scale)
                per_scale.append(np.mean(self._components(rates, angles)[1]))
            values.append(np.mean(per_scale))
        self.assertLess(abs(values[0] - values[1]), 0.01, f"{values}")

    def test_silent_and_single_spike_neurons_stay_finite(self):
        angles, rates = _sample(self.params, 56, 17)
        rates[:, :50] = 0.0                       # never fires
        rates[:, 50:100] = 0.0
        rates[0, 50:100] = 1.0 / WINDOW_MS        # one spike, in one trial
        for values in self._components(rates, angles):
            self.assertTrue(np.all(np.isfinite(values)))
            self.assertEqual(np.abs(values[:50]).max(), 0.0)

    def test_loss_gradients_are_finite_and_skip_silent_neurons(self):
        """A silent neuron's components divide by epsilon; the loss masks it."""
        n_neurons = 300
        angles, rates = _sample(self.params, 56, 23)
        rates = rates[:, :n_neurons]
        rates[:, :50] = 0.0
        loss = _single_type_loss(0.0)
        loss._tuning_angles = tf.constant(self.params[0][:n_neurons], dtype=tf.float32)
        loss._emd_group_indices = tf.ragged.constant([np.arange(n_neurons)], dtype=tf.int32)
        loss._osi_empirical_distributions = tf.ragged.constant(
            [np.linspace(0.0, 0.8, 20)], dtype=tf.float32
        )
        loss._dsi_empirical_distributions = tf.ragged.constant(
            [np.linspace(0.0, 0.5, 20)], dtype=tf.float32
        )
        variable = tf.Variable(rates)
        with tf.GradientTape() as tape:
            value = loss.batch_osi_emd_loss_from_rates(variable, tf.constant(angles))
        gradient = tape.gradient(value, variable).numpy()
        self.assertTrue(np.all(np.isfinite(gradient)))
        self.assertEqual(np.abs(gradient[:, :50]).max(), 0.0)
        self.assertEqual(np.count_nonzero(np.abs(gradient[:, 50:]).max(axis=0) == 0.0), 0)

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
        osi, _, dsi, _ = self._components(rates.numpy(), angles, estimator)
        self.assertGreater(np.corrcoef(osi, osi_true)[0, 1], 0.7)
        self.assertGreater(np.corrcoef(dsi, dsi_true)[0, 1], 0.7)
        self.assertLess(abs(np.mean(osi - osi_true)), 0.06)

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

    def test_emd_term_alone_drives_selectivity_toward_target(self):
        """Without the alignment term the EMD must still carry a gradient.

        A gradient-free shuffle before the sort once zeroed it, leaving only
        the |Im| penalty, which lowers selectivity.
        """
        angles = np.arange(0, 360, 5, dtype=np.float32)
        loss = _single_type_loss(0.0, osi_target=0.4, dsi_target=0.3)
        loss._batch_emd_alignment_weight = 0.0
        rates = tf.Variable(
            (1.0 + 0.1 * np.cos(2.0 * np.deg2rad(angles)))[:, None] / 1000.0
        )
        with tf.GradientTape() as tape:
            value = loss.batch_osi_emd_loss_from_rates(rates, tf.constant(angles))
        gradient = tape.gradient(value, rates)
        self.assertIsNotNone(gradient)
        stepped = rates - 1e-4 * gradient / tf.reduce_max(tf.abs(gradient))
        osi_before, _, dsi_before, _ = _estimator([0.0])._batch_signed_osi_dsi_components(
            rates, tf.constant(angles)
        )
        osi_after, _, dsi_after, _ = _estimator([0.0])._batch_signed_osi_dsi_components(
            stepped, tf.constant(angles)
        )
        self.assertGreater(float(osi_after[0]), float(osi_before[0]))
        self.assertGreater(float(dsi_after[0]), float(dsi_before[0]))

    def test_loss_excludes_neurons_below_half_hertz(self):
        """Neurons under the 0.5 Hz responsiveness cutoff are left out, too."""
        angles = np.arange(360, dtype=np.float32)
        tuned = 1.0 + 0.8 * np.cos(2.0 * np.deg2rad(angles))
        loss = _single_type_loss(0.0)
        loss._tuning_angles = tf.constant([0.0, 0.0], dtype=tf.float32)
        loss._emd_group_indices = tf.ragged.constant([[0, 1]], dtype=tf.int32)
        loss.cell_type_count = tf.constant([2.0], dtype=tf.float32)

        for mean_rate_hz, expected in ((0.4, 0.0), (0.6, 0.4)):
            untuned = np.full_like(tuned, mean_rate_hz / 1000.0)
            value = loss.batch_osi_emd_loss_from_rates(
                tf.constant(np.stack([tuned, untuned], axis=1)), tf.constant(angles)
            )
            # An included untuned neuron misses the 0.4 OSI target by 0.4,
            # which the two-neuron mean halves, and the DSI target is 0.
            self.assertAlmostEqual(
                float(value.numpy()), expected / 2.0, places=5,
                msg=f"{mean_rate_hz} Hz",
            )

    def test_mask_uses_preferred_direction_rate(self):
        """Mean rate below 0.5 Hz but preferred-direction rate above it: included.

        The 8 protocol-direction bins of the batch decide, as for the
        Neuropixels targets and validation, not the batch-mean rate.
        """
        angles = np.arange(0, 360, 5, dtype=np.float32)
        loss = _estimator([0.0, 0.0])
        # 0.9 Hz in the 0-degree bin only: a 0.11 Hz batch mean.
        bin_zero = np.abs(((angles + 22.5) % 360.0) - 22.5) < 22.5
        selective = np.where(bin_zero, 0.9, 0.0) / 1000.0
        weak = np.where(bin_zero, 0.4, 0.0) / 1000.0
        rates = tf.constant(np.stack([selective, weak], axis=1).astype(np.float32))
        self.assertLess(float(tf.reduce_mean(rates[:, 0])), 0.0005)
        active = loss._preferred_direction_active(rates, tf.constant(angles))
        self.assertEqual(active.numpy().tolist(), [True, False])

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


class StratifiedAnglesTest(unittest.TestCase):
    def test_one_angle_per_equal_width_bin(self):
        """Every bin of width 360 / batch holds exactly one grating angle."""
        for dtype, batch in ((tf.float32, 248), (tf.float16, 248), (tf.float32, 7)):
            uniform = tf.random.stateless_uniform(
                (batch, 1), seed=(3, 4), maxval=360.0, dtype=tf.float32
            )
            angles = stratify_angles(tf.cast(uniform, dtype))
            self.assertEqual(angles.dtype, dtype)
            values = angles.numpy().astype(np.float64)[:, 0]
            self.assertTrue(np.all(np.isfinite(values)))
            bins = np.floor(values * batch / 360.0).astype(int)
            # float16 rounds angles near a bin edge (spacing 0.25 deg near 360).
            tolerance = 1 if dtype == tf.float16 else 0
            self.assertLessEqual(np.abs(bins - np.arange(batch)).max(), tolerance)
            self.assertTrue(np.all((values >= 0.0) & (values <= 360.0)))


if __name__ == "__main__":
    unittest.main()
