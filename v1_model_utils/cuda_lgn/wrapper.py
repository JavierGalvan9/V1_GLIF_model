"""CUDA drifting-grating LGN: exact factorization of the movie -> filters -> rate pipeline.

See lgn_grating_ops.cc for the math. Everything that does not depend on the
sample (the temporal responses R, the corner weights, the separable filter
taps) is computed here once, in float64, from the float32 LGN constants.
"""

from pathlib import Path
import warnings

import numpy as np
from scipy.signal import correlate2d
import tensorflow as tf

from v1_model_utils.cuda_operator_cache import ensure_artifact
from v1_model_utils.tf_utils import replica_local_constant


HERE = Path(__file__).resolve().parent
BUILD_FLAGS = ("--expt-relaxed-constexpr",)
GRATING, MOVIE = "lgn_grating_ops", "lgn_movie_ops"
_OPS = {}


def _load_ops(stem=GRATING):
    if stem not in _OPS:
        library = ensure_artifact(
            HERE,
            stem,
            sources=(HERE / "build.py", HERE / "lgn_common.cuh", HERE / f"{stem}.cc",
                     HERE / f"{stem}.cu.cc"),
            build_module="v1_model_utils.cuda_lgn.build",
            build_flags=BUILD_FLAGS,
            build_args=("--stem", stem),
        )
        _OPS[stem] = tf.load_op_library(str(library))
    return _OPS[stem]


def available(stem=GRATING):
    """Whether one CUDA LGN library (GRATING or MOVIE) runs here: a GPU is
    visible and its ops build and load. Each library falls back on its own."""
    if not tf.config.list_physical_devices("GPU"):
        return False
    try:
        _load_ops(stem)
    except Exception as error:  # the TensorFlow path remains correct, only slower
        warnings.warn(f"CUDA LGN {stem} unavailable, using the TensorFlow LGN path: {error}")
        return False
    return True


def _separable_taps(filters):
    """Rank-one float64 factors of each 2-D Gaussian, centered in a common odd width."""
    taps = max(max(f.shape) for f in filters)
    vertical, horizontal = np.zeros((len(filters), taps)), np.zeros((len(filters), taps))
    for k, kernel in enumerate(filters):
        u, singular, vh = np.linalg.svd(kernel, full_matrices=False)
        if singular[1:].sum() > singular[0] * 1e-5 or kernel.shape[0] % 2 == 0 or kernel.shape[1] % 2 == 0:
            raise ValueError("Expected odd, rank-one Gaussian spatial kernels")
        column, row = u[:, 0] * np.sqrt(singular[0]), vh[0] * np.sqrt(singular[0])
        if column.sum() < 0:
            column, row = -column, -row
        before_v, before_h = (taps - column.size) // 2, (taps - row.size) // 2
        vertical[k, before_v:before_v + column.size] = column
        horizontal[k, before_h:before_h + row.size] = row
    return vertical, horizontal


def _bilinear(x, y, reciprocal, scale):
    """Corner coordinates [x0, x1, y0, y1] and weights of (y0,x0), (y1,x0), (y0,x1), (y1,x1)."""
    x0, x1, y0, y1 = np.floor(x), np.ceil(x), np.floor(y), np.ceil(y)
    fx, fy = x - x0, y - y0
    corners = np.stack((x0, x1, y0, y1), axis=1).astype(np.int64)
    weights = np.stack(((1 - fx) * (1 - fy), (1 - fx) * fy, fx * (1 - fy), fx * fy), axis=1)
    rows = corners[:, [2, 3, 2, 3]]
    cols = corners[:, [0, 0, 1, 1]]
    return corners, weights * reciprocal[rows, cols] * scale[:, None]


def _temporal_response(kernels, omega, seq_len, pre_delay, duration, chunk=4096):
    """R[t, n] = sum over lags m with the frame t - m - pre_delay in the stimulus of
    K[L - 1 - m, n] e^{i omega (t - m - pre_delay)}: the baseline's causal depthwise
    convolution of e^{i omega f}, zero-padded before the kernel and outside the window.
    """
    length, units = kernels.shape
    frame = np.arange(seq_len) - pre_delay
    high = np.clip(frame + 1, 0, length)
    low = np.clip(frame - duration + 1, 0, length)
    rotation = np.exp(1j * omega * frame)[:, None]
    shift = np.exp(-1j * omega * np.arange(length))[:, None]
    response = np.empty((seq_len, units, 2), np.float32)
    for start in range(0, units, chunk):
        lagged = kernels[::-1, start:start + chunk].astype(np.float64) * shift
        prefix = np.concatenate((np.zeros((1, lagged.shape[1])), np.cumsum(lagged, axis=0)))
        value = rotation * (prefix[high] - prefix[low])
        response[:, start:start + chunk, 0] = value.real
        response[:, start:start + chunk, 1] = value.imag
    return response


class GratingLGNKernel:
    """The CUDA ops and their constants for one DriftingGratingLGN configuration.

    Constants are replica-local (tf_utils.replica_local_constant), so under a
    distribution strategy every replica reads its own GPU's copy. `chunk`
    samples are sampled per op call, up to the batch's remaining samples.
    The default 16 reuses temporal responses across a sample tile. Set
    ``chunk=1`` to keep only one sample's uniforms alive. At 17,400 units and
    500 steps, 16 samples' float32 uniforms occupy about 531 MiB versus 33 MiB
    for one sample. On SM120, batches 32 and 256 were 2.3–2.6 times faster
    with 16 than with 1 (Experiments/kernel_generalization_20261001).
    """

    def __init__(self, host, *, seq_len, pre_delay, post_delay, temporal_f, cpd, contrast,
                 rows, cols, theta_sign, theta_offset, bmtk_compat=True, chunk=16):
        if not 1 <= chunk <= 16:
            raise ValueError(f"chunk must be in [1, 16], got {chunk}")
        self._ops = _load_ops()
        duration = seq_len - pre_delay - post_delay
        composite = np.asarray(host["is_composite"], bool)
        units = composite.size
        f64 = lambda value: np.asarray(value, np.float32).astype(np.float64)  # noqa: E731
        filters = [f64(kernel) for kernel in host["gaussian_filters"]]
        vertical, horizontal = _separable_taps(filters)

        bins = np.full(units, -1, np.int64)
        for k, indices in enumerate(host["spatial_range_indices"]):
            bins[indices] = k
        if (bins < 0).any():
            raise ValueError("Every LGN unit needs a spatial filter bin")
        corners = np.zeros((units, 8), np.int64)
        weights = np.zeros((units, 8))
        amplitude = f64(host["amplitude"]) * contrast
        non_dom_amplitude = f64(host["non_dom_amplitude"]) * contrast
        x, y = f64(host["x"]), f64(host["y"])
        non_dom_x, non_dom_y = f64(host["non_dominant_x"]), f64(host["non_dominant_y"])
        for k, kernel in enumerate(filters):
            # bmtk_compat: divide by the filtered all-ones frame ('SAME', zero padding).
            reciprocal = (1.0 / correlate2d(np.ones((rows, cols)), kernel, mode="same")
                          if bmtk_compat else np.ones((rows, cols)))
            units_k = np.flatnonzero(bins == k)
            corners[units_k, :4], weights[units_k, :4] = _bilinear(
                x[units_k], y[units_k], reciprocal, amplitude[units_k])
            units_k = units_k[composite[units_k]]
            corners[units_k, 4:], weights[units_k, 4:] = _bilinear(
                non_dom_x[units_k], non_dom_y[units_k], reciprocal, non_dom_amplitude[units_k])

        omega = 2 * np.pi * temporal_f / 1000.0
        composite_ids = np.flatnonzero(composite)
        slot = np.full(units, -1, np.int64)
        slot[composite_ids] = np.arange(composite_ids.size)
        response = _temporal_response(host["dom_temporal_kernels"], omega, seq_len, pre_delay, duration)
        composite_response = _temporal_response(
            np.asarray(host["non_dom_temporal_kernels"])[:, composite_ids], omega, seq_len,
            pre_delay, duration)

        self.seq_len, self.units, self.chunk = seq_len, units, chunk
        self._attrs = dict(theta_sign=float(theta_sign), theta_offset=float(theta_offset),
                           rows=rows, cols=cols)
        self._spatial = [replica_local_constant(value, dtype) for value, dtype in (
            (2 * np.pi * cpd, tf.float64), (vertical, tf.float64), (horizontal, tf.float64),
            (bins, tf.int64), (corners, tf.int64), (weights, tf.float64))]
        self._temporal = [replica_local_constant(value, dtype) for value, dtype in (
            (response, tf.float32), (composite_response, tf.float32), (slot, tf.int64),
            (np.asarray(host["spontaneous_firing_rates"], np.float32), tf.float32))]

    def coefficients(self, theta, phase):
        """Complex spatial coefficients [batch, units, 4] of the batch's gratings."""
        return self._ops.lgn_grating_coefficients(theta, phase, *self._spatial, **self._attrs)

    def probabilities(self, theta, phase, rates=False):
        """Float32 spike probabilities (or rates in Hz), [batch, seq_len, units]."""
        return self._ops.lgn_grating_probabilities(
            self.coefficients(theta, phase), *self._temporal, rates=rates)

    def spikes(self, theta, phase, spike_seeds, dtype=tf.float32):
        """Spikes [batch, seq_len, units]: TensorFlow's stateless uniforms, in `dtype`,
        below the probabilities, sampled `chunk` samples at a time into one buffer.

        float16 uniforms take only 1024 values, so P(u < p) = ceil(1024 p) / 1024
        rather than p (about +5% spikes at LGN rates); use float32.
        """
        coefficients = self.coefficients(theta, phase)
        shape = (self.seq_len, self.units)

        def uniform(seed):
            return tf.random.stateless_uniform(shape, seed=seed, dtype=dtype)

        def sample(spikes, uniforms, offset):
            return self._ops.lgn_grating_spikes(
                spikes, coefficients, *self._temporal, uniforms, offset=offset)

        batch = theta.shape[0]
        if batch is None:  # one chunk when the batch is only known at run time
            uniforms = tf.map_fn(uniform, spike_seeds, fn_output_signature=tf.TensorSpec(shape, dtype))
            return sample(tf.zeros((0,), tf.bool), [uniforms], 0)
        if batch == 0:
            return tf.zeros((0,) + shape, tf.bool)
        # Each chunk's op writes its samples into the previous chunk's output in
        # place, and a chunk's uniforms are drawn only once the previous chunk is
        # sampled, so only one chunk of uniforms is ever alive.
        spikes = tf.zeros((0,), tf.bool)
        for begin in range(0, batch, self.chunk):
            with tf.control_dependencies([spikes] if begin else []):
                uniforms = [uniform(spike_seeds[b]) for b in range(begin, min(begin + self.chunk, batch))]
            spikes = sample(spikes, uniforms, begin)
        return spikes
