"""The CUDA background-noise op must reproduce the TensorFlow sampler bit for bit.

`bkg_poisson_counts` replaces `sample_poisson_counts` on the GPU, so the two
backends (and runs restored across them) draw the same Poisson counts. That
includes the int32 wrap of int64 seeds and the per-replica seed offset.
"""

import numpy as np
import pytest
import tensorflow as tf

from v1_model_utils import cuda_bkg_noise
from v1_model_utils.models import poisson_cdf_levels, sample_poisson_counts

pytestmark = pytest.mark.skipif(
    not cuda_bkg_noise.available(), reason="CUDA BKG noise op unavailable"
)

LEVELS = tf.constant(poisson_cdf_levels(0.25), tf.float64)


def _both_backends(seed, replica, step, shape, dtype):
    with tf.device("/GPU:0"):
        cuda = cuda_bkg_noise.bkg_poisson_counts(
            tf.Variable(np.int64(seed)), replica, tf.constant([step, 5], tf.int32),
            tf.constant(shape), LEVELS, dtype,
        )
        # The documented seed: int32(noise_seed) + replica * 1000003.
        base = tf.cast(tf.constant(seed, tf.int64), tf.int32)
        reference = sample_poisson_counts(
            LEVELS, shape, tf.stack([base + replica * 1000003, tf.constant(step)]), dtype
        )
    return cuda.numpy(), reference.numpy()


@pytest.mark.parametrize("seed", [0, 3000, -5, 2**31 + 7, -(2**33) - 1])
@pytest.mark.parametrize("replica", [0, 3])
@pytest.mark.parametrize("step", [0, 999, -3, 2**31 - 1])
@pytest.mark.parametrize("shape", [(5, 100), (3, 33), (7, 1)])
def test_counts_match_the_tensorflow_sampler(seed, replica, step, shape):
    cuda, reference = _both_backends(seed, replica, step, shape, tf.int32)
    np.testing.assert_array_equal(cuda, reference)


def test_float16_counts_match_the_tensorflow_sampler():
    cuda, reference = _both_backends(11, 1, 4, (5, 100), tf.float16)
    assert cuda.dtype == np.float16
    np.testing.assert_array_equal(cuda, reference)


def test_replicas_and_steps_draw_different_counts():
    """Guard the guard: the comparison above must be able to fail."""
    base, _ = _both_backends(11, 0, 4, (5, 100), tf.int32)
    other_replica, _ = _both_backends(11, 1, 4, (5, 100), tf.int32)
    other_step, _ = _both_backends(11, 0, 5, (5, 100), tf.int32)
    assert not np.array_equal(base, other_replica)
    assert not np.array_equal(base, other_step)


@pytest.mark.parametrize("step", [0, 999, 2**31 - 3])
@pytest.mark.parametrize("dtype", [tf.int32, tf.float16])
def test_num_steps_counts_are_the_single_step_calls(step, dtype):
    """One launch of num_steps steps holds, slice by slice, the separate calls'
    counts at steps step + t (wrapping int32 past 2**31 - 1)."""
    seed, shape, num_steps = tf.Variable(np.int64(-17)), tf.constant((5, 33)), 6
    with tf.device("/GPU:0"):
        steps = cuda_bkg_noise.bkg_poisson_counts(
            seed, 2, tf.constant([step, 5], tf.int32), shape, LEVELS, dtype, num_steps=num_steps
        ).numpy()
        single = [
            cuda_bkg_noise.bkg_poisson_counts(
                seed, 2, tf.constant([step, 5], tf.int32) + t, shape, LEVELS, dtype
            ).numpy()
            for t in range(num_steps)
        ]
    assert steps.shape == (num_steps, 5, 33) and steps.dtype == single[0].dtype
    np.testing.assert_array_equal(steps, np.stack(single))


@pytest.mark.parametrize("backend", ["cuda", "tf"])
def test_sample_bkg_activity_num_steps_is_the_per_step_samples(backend):
    """models.sample_bkg_activity(num_steps=k) stacks the per-step samples on
    both backends (the TensorFlow sampler through map_fn)."""
    from types import SimpleNamespace

    from v1_model_utils.models import sample_bkg_activity

    cell = SimpleNamespace(
        noise_seed=tf.Variable(np.int64(123)), bkg_input_dense_shape=(10, 21),
        bkg_count_cdf=LEVELS, compute_dtype=tf.float16, _synaptic_current_backend=backend,
    )
    noise_step = tf.constant([40, 0], tf.int32)
    with tf.device("/GPU:0"):
        steps = sample_bkg_activity(cell, 3, noise_step, num_steps=4).numpy()
        single = np.stack([sample_bkg_activity(cell, 3, noise_step + t).numpy() for t in range(4)])
    assert steps.shape == (4, 3, 21)
    np.testing.assert_array_equal(steps, single)
