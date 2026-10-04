"""Parity tests for the CUDA LGN movie op (v1_model_utils.cuda_lgn.MovieLGNKernel)."""

import os

import numpy as np
import pytest
from scipy.signal import fftconvolve
import tensorflow as tf

from lgn_model.lgn import LGN
from v1_model_utils import cuda_lgn

DATA_DIR = os.environ.get("V1_LGN_TEST_DATA_DIR", "GLIF_network_nll_full")
requires_op = pytest.mark.skipif(not cuda_lgn.available(cuda_lgn.MOVIE), reason="CUDA LGN ops unavailable")


@pytest.fixture(scope="module", autouse=True)
def strict_float32():
    """TensorFlow may run float32 convolutions in TF32 (about 1e-3 Hz off); the
    reference TF path here is strict float32."""
    tf.config.experimental.enable_tensor_float_32_execution(False)
    yield
    tf.config.experimental.enable_tensor_float_32_execution(True)


@pytest.fixture(scope="module")
def lgn(strict_float32):
    return LGN(n_input=17400, dtype=tf.float32, data_dir=DATA_DIR)


def _movie(batch, steps, seed=7):
    """Smoothed noise in [-1, 1], float16-exact, [batch, steps, 80, 120]."""
    noise = tf.random.stateless_normal((batch, steps, 80, 120, 1), seed=(seed, 3))
    noise = tf.nn.avg_pool3d(noise, (8, 5, 5), 1, "SAME")[..., 0] * 4
    return tf.cast(tf.cast(tf.clip_by_value(noise, -1, 1), tf.float16), tf.float32)


def _tf_rates(lgn, movie, bmtk_compat):
    return tf.stack([
        tf.cast(lgn.firing_rates_from_spatial(*lgn.spatial_response(sample[..., None], bmtk_compat)),
                tf.float32)
        for sample in movie
    ])


def _reference_rates(lgn, movie, bmtk_compat):
    """Float64 rates: full 2-D Gaussian correlation, bilinear samples, FFT causal convolution."""
    host = {k: np.asarray(v, np.float64) if not isinstance(v, list) else v
            for k, v in lgn.host_constants.items()}
    movie = np.asarray(movie, np.float64)
    batch, steps = movie.shape[:2]
    composite = np.asarray(lgn.host_constants["is_composite"], bool)
    subunits = np.zeros((2, batch, steps, composite.size))
    for kernel, units in zip(host["gaussian_filters"], host["spatial_range_indices"]):
        kernel = np.asarray(kernel, np.float64)[None, None, ::-1, ::-1]
        filtered = fftconvolve(movie, kernel, mode="same", axes=(2, 3))
        if bmtk_compat:
            filtered /= fftconvolve(np.ones((1, 1, 80, 120)), kernel, mode="same", axes=(2, 3))
        for plane, (x, y, selected) in zip(subunits, (
                ("x", "y", units), ("non_dominant_x", "non_dominant_y", units[composite[units]]))):
            px, py = host[x][selected], host[y][selected]
            x0, y0 = np.floor(px).astype(int), np.floor(py).astype(int)
            x1, y1 = np.ceil(px).astype(int), np.ceil(py).astype(int)
            fx, fy = px - x0, py - y0
            plane[..., selected] = (
                filtered[..., y0, x0] * (1 - fx) * (1 - fy) + filtered[..., y1, x0] * (1 - fx) * fy
                + filtered[..., y0, x1] * fx * (1 - fy) + filtered[..., y1, x1] * fx * fy)

    def causal(signal, kernels):
        return fftconvolve(signal, kernels[None, ::-1], axes=1)[:, :steps]

    rates = np.maximum(causal(subunits[0], host["dom_temporal_kernels"]) * host["amplitude"]
                       + host["spontaneous_firing_rates"], 0)
    ids = np.flatnonzero(composite)
    rates[..., ids] += np.maximum(
        causal(subunits[1][..., ids], host["non_dom_temporal_kernels"][:, ids])
        * host["non_dom_amplitude"][ids] + host["spontaneous_firing_rates"][ids], 0)
    return rates


@requires_op
@pytest.mark.parametrize("bmtk_compat", [True, False])
@pytest.mark.parametrize("steps", [37, 700])  # shorter and longer than the 574-lag kernels
def test_movie_op_matches_tf_float32(lgn, bmtk_compat, steps):
    movie = _movie(3, steps)
    expected = _tf_rates(lgn, movie, bmtk_compat)
    actual = lgn.movie_kernel(bmtk_compat).response(movie, "rates")
    tf.debugging.assert_near(actual, expected, atol=2e-5, rtol=1e-5)


@requires_op
@pytest.mark.parametrize("bmtk_compat", [True, False])
def test_movie_op_matches_float64_reference(lgn, bmtk_compat):
    movie = _movie(2, 60, seed=11)
    reference = _reference_rates(lgn, movie, bmtk_compat)
    actual = lgn.movie_kernel(bmtk_compat).response(movie, "rates").numpy()
    tf32 = _tf_rates(lgn, movie, bmtk_compat).numpy()
    op_error = np.abs(actual - reference).max()
    assert op_error < 1e-5
    assert op_error <= np.abs(tf32 - reference).max()


@requires_op
def test_gray_screen_is_spontaneous(lgn):
    rates = lgn.movie_kernel(True).response(tf.zeros((2, 50, 80, 120), tf.float16), "rates")
    tf.debugging.assert_equal(rates, _tf_rates(lgn, tf.zeros((2, 50, 80, 120)), True))


@requires_op
def test_probabilities_and_spikes(lgn):
    kernel = lgn.movie_kernel(True)
    movie = _movie(5, 90, seed=5)
    seeds = tf.constant([[i, 3 * i + 1] for i in range(5)], tf.int32)
    rates = kernel.response(movie, "rates").numpy().astype(np.float64)
    probabilities = kernel.response(movie, "probabilities")
    np.testing.assert_allclose(probabilities, -np.expm1(-rates / 1000), rtol=2e-7, atol=1e-12)
    uniforms = tf.stack([tf.random.stateless_uniform((90, 17400), seed=s) for s in seeds])
    expected = uniforms < probabilities
    tf.debugging.assert_equal(kernel.spikes(movie, seeds), expected)
    kernel.chunk = 2  # chunked uniforms write into the previous chunk's output in place
    tf.debugging.assert_equal(kernel.spikes(movie, seeds), expected)
    kernel.chunk = 8
    # A batch known only at run time takes one chunk.
    spikes = tf.function(kernel.spikes, input_signature=(
        tf.TensorSpec((None, 90, 80, 120), tf.float32), tf.TensorSpec((None, 2), tf.int32)))
    tf.debugging.assert_equal(spikes(movie, seeds), expected)


@requires_op
def test_firing_rates_api(lgn):
    movie = _movie(2, 40, seed=9)
    rates = lgn.firing_rates(movie)
    assert rates.shape == (2, 40, 17400) and rates.dtype == tf.float32
    single = lgn.firing_rates(movie[0][..., None], output="probabilities")
    tf.debugging.assert_equal(single, lgn.firing_rates(movie, output="probabilities")[0])
    spikes = lgn.firing_rates(movie[1][..., None], output="spikes", spike_seeds=[4, 2])
    probabilities = lgn.firing_rates(movie, output="probabilities")[1]
    tf.debugging.assert_equal(spikes, tf.random.stateless_uniform((40, 17400), seed=(4, 2)) < probabilities)


def test_firing_rates_fallback_matches(lgn, monkeypatch):
    movie = _movie(2, 45, seed=13)
    seeds = tf.constant([[1, 2], [3, 4]], tf.int32)
    if cuda_lgn.available(cuda_lgn.MOVIE):
        expected = {output: lgn.firing_rates(movie, output=output, spike_seeds=seeds)
                    for output in ("rates", "probabilities", "spikes")}
    monkeypatch.setattr(lgn, "movie_kernel", lambda bmtk_compat=True: None)
    rates = lgn.firing_rates(movie)
    tf.debugging.assert_near(rates, _tf_rates(lgn, movie, True), atol=1e-6, rtol=1e-6)
    probabilities = lgn.firing_rates(movie, output="probabilities")
    tf.debugging.assert_near(probabilities, -tf.math.expm1(-rates / 1000), atol=1e-9, rtol=1e-6)
    uniforms = tf.stack([tf.random.stateless_uniform((45, 17400), seed=s) for s in seeds])
    spikes = lgn.firing_rates(movie, output="spikes", spike_seeds=seeds)
    tf.debugging.assert_equal(spikes, uniforms < probabilities)
    if cuda_lgn.available(cuda_lgn.MOVIE):
        tf.debugging.assert_near(expected["rates"], rates, atol=2e-5, rtol=1e-5)
        tf.debugging.assert_near(expected["probabilities"], probabilities, atol=1e-7, rtol=1e-5)


@requires_op
def test_n_input_subset_without_composites():
    lgn = LGN(n_input=2000, dtype=tf.float16, data_dir=DATA_DIR)
    assert lgn.n_composite == 0
    movie = _movie(2, 30, seed=17)
    expected = _tf_rates(LGN(n_input=2000, dtype=tf.float32, data_dir=DATA_DIR), movie, True)
    tf.debugging.assert_near(lgn.firing_rates(movie), expected, atol=2e-5, rtol=1e-5)


@requires_op
def test_empty_batches_and_frame_size(lgn):
    kernel = lgn.movie_kernel(True)
    for shape in ((0, 50, 80, 120), (2, 0, 80, 120)):
        movie = tf.zeros(shape, tf.float16)
        assert kernel.response(movie).shape == shape[:2] + (17400,)
        assert kernel.spikes(movie, tf.zeros((shape[0], 2), tf.int32)).shape == shape[:2] + (17400,)
    with pytest.raises(tf.errors.InvalidArgumentError):
        kernel.response(tf.zeros((1, 5, 80, 121)))


@requires_op
@pytest.mark.skipif(not cuda_lgn.available(cuda_lgn.GRATING), reason="CUDA grating op unavailable")
def test_drifting_grating_backends_agree(strict_float32):
    """The grating op, the movie op and the TensorFlow filters give the same grating
    probabilities, and the CUDA paths the same spikes with or without return_probabilities."""
    from stim_dataset import DriftingGratingLGN

    theta = tf.constant([[30.0], [200.0]])
    phase = tf.constant([10.0, 300.0])
    seeds = tf.constant([[1, 2], [3, 4]], tf.int32)

    def grating(backend, **kwargs):
        return DriftingGratingLGN(120, 20, 10, n_input=17400, data_dir=DATA_DIR, rotation="ccw",
                                  lgn_backend=backend, **kwargs)

    paths = {backend: grating(backend) for backend in ("grating", "movie", "tensorflow")}
    assert {backend: path.backend for backend, path in paths.items()} == {
        "grating": "grating", "movie": "movie", "tensorflow": "tensorflow"}
    probabilities = {backend: path.batch_probabilities(theta, phase) for backend, path in paths.items()}
    tf.debugging.assert_near(probabilities["movie"], probabilities["grating"], atol=1e-7, rtol=0)
    # Under mixed precision the movie is float16, about 5e-4 (measured: p within 1.9e-6 at
    # 32 x 500); the angles are float16-exact, so both ops see one grating.
    half = [grating(backend, dtype=tf.float16).batch_probabilities(
        tf.cast(theta, tf.float16), tf.cast(phase, tf.float16)) for backend in ("movie", "grating")]
    tf.debugging.assert_near(half[0], half[1], atol=5e-6, rtol=0)
    tf.debugging.assert_near(probabilities["tensorflow"], probabilities["grating"], atol=1e-5, rtol=0)
    for backend in ("grating", "movie"):
        spikes = paths[backend].batch_spikes(theta, phase, seeds)
        sampled = grating(backend, return_probabilities=True).batch_spikes(theta, phase, seeds)
        tf.debugging.assert_equal(spikes, sampled)
        uniforms = tf.stack([tf.random.stateless_uniform((120, 17400), seed=s) for s in seeds])
        tf.debugging.assert_equal(spikes, uniforms < probabilities[backend])
    with pytest.raises(ValueError, match="lgn_backend"):
        grating("cudnn")


@pytest.mark.skipif(not cuda_lgn.available(cuda_lgn.GRATING), reason="CUDA grating op unavailable")
@pytest.mark.parametrize("dtype", [tf.float16, tf.float32])
@pytest.mark.parametrize("batch", [3, 17])
def test_grating_default_chunks_match_sequential_and_probabilities(lgn, dtype, batch):
    """A partial sample tile and a second tile preserve per-sample seeded spikes."""
    options = dict(seq_len=17, pre_delay=2, post_delay=3, temporal_f=2, cpd=0.04,
                   contrast=0.8, rows=80, cols=120, theta_sign=-1, theta_offset=0)
    default = cuda_lgn.GratingLGNKernel(lgn.host_constants, **options)
    sequential = cuda_lgn.GratingLGNKernel(lgn.host_constants, chunk=1, **options)
    theta = tf.cast(tf.range(batch) * 11, dtype)
    phase = tf.cast(tf.range(batch) * 17, dtype)
    seeds = tf.stack([tf.range(batch) + 71, tf.fill([batch], 3)], axis=1)
    spikes = tf.function(default.spikes)(theta, phase, seeds)
    tf.debugging.assert_equal(spikes, tf.function(sequential.spikes)(theta, phase, seeds))
    uniforms = tf.stack([
        tf.random.stateless_uniform((17, 17400), seed=seed, dtype=tf.float32)
        for seed in tf.unstack(seeds)
    ])
    tf.debugging.assert_equal(spikes, uniforms < default.probabilities(theta, phase))


@requires_op
def test_chunked_movie_spikes_match_whole_batch(lgn):
    """Building the movie one chunk at a time gives the whole-batch spikes bitwise."""
    kernel = cuda_lgn.MovieLGNKernel(lgn.host_constants, rows=80, cols=120, chunk=4)
    movie = _movie(11, 40, seed=21)  # 11 samples: chunks of 4, 4 and 3
    seeds = tf.reshape(tf.range(22, dtype=tf.int32), (11, 2))
    whole = kernel.spikes(movie, seeds)
    chunked = kernel.spikes(lambda begin, end: movie[begin:end], seeds, batch=11)
    tf.debugging.assert_equal(chunked, whole)
    with pytest.raises(ValueError, match="batch size"):
        kernel.spikes(lambda begin, end: movie[begin:end], seeds)
