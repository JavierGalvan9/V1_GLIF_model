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
