"""Device-side background Poisson counts (see bkg_noise_ops.cc)."""

import warnings

import tensorflow as tf

from .build import LIBRARY


_OPS = None
_UNAVAILABLE = False


def _load_ops():
    global _OPS
    if _OPS is None:
        _OPS = tf.load_op_library(str(LIBRARY.ensure()))
    return _OPS


def available():
    """Whether the op runs here: a GPU is visible and the op builds and loads."""
    global _UNAVAILABLE
    if _UNAVAILABLE or not tf.config.list_physical_devices("GPU"):
        return False
    try:
        _load_ops()
    except Exception as error:  # the TensorFlow sampler stays correct
        warnings.warn(f"CUDA BKG noise op unavailable, using the TensorFlow sampler: {error}")
        _UNAVAILABLE = True
        return False
    return True


def bkg_poisson_counts(noise_seed, replica_id, step, shape, cdf, dtype, num_steps=0):
    """Counts [batch, n_bkg] in `dtype`, equal to sample_poisson_counts with the
    seed [int32(noise_seed) + replica_id * 1000003, int32(step[0])].

    With num_steps > 0, counts [num_steps, batch, n_bkg] whose slice t is the
    call at step step[0] + t (wrapping int32), in one launch.

    noise_seed is the int64 variable (or tensor) on the GPU; replica_id, step
    and shape are int32, which TensorFlow keeps in host memory.
    """
    return _load_ops().bkg_poisson_counts(
        noise_seed, tf.cast(replica_id, tf.int32), tf.cast(step, tf.int32),
        tf.cast(shape, tf.int32), tf.cast(cdf, tf.float64), T=dtype, num_steps=num_steps)
