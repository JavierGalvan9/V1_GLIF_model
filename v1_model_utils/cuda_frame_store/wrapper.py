"""Park a sequence of GPU frames in pinned host memory (see frame_store_ops.cc)."""

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
    """Whether the ops run here: a GPU is visible and the ops build and load."""
    global _UNAVAILABLE
    if _UNAVAILABLE or not tf.config.list_physical_devices("GPU"):
        return False
    try:
        _load_ops()
    except Exception as error:  # the frames then stay on the GPU
        warnings.warn(f"CUDA frame store unavailable, keeping frames on the GPU: {error}")
        _UNAVAILABLE = True
        return False
    return True


for _op in ("V1FrameStoreOffload", "V1FrameStoreTake"):
    tf.no_gradient(_op)


def order_variable():
    """A variable whose (per-replica) handle the store ops take as ``order``.

    Its value is never read. TensorFlow's automatic control dependencies
    chain every stateful op without a resource input, across replicas, in
    program order; with the handle as input, a replica's store ops are
    ordered among themselves only, and replicas roll out concurrently.
    Create it in the distribution strategy's scope, and pass ``.handle``
    from replica context.
    """
    return tf.Variable(0.0, trainable=False, name="frame_store_order")


def offload_frame(frame, index, key, capacity, resident_from=None, group=1, codec="raw",
                  drop_outside=False, *, order):
    """Hand ``frame`` to slot ``index`` (int32, < ``capacity``) of the host
    store ``key`` of this GPU. Slots at or above ``resident_from`` stay on the
    GPU; the others are copied out in groups of ``group`` as fast as the copy
    engine finishes them (earliest first) and released on the GPU once copied.
    Groups not admitted to the copy stream by the time the GPU passes slot
    ``capacity - 1`` stay on the GPU. With ``codec="tq"`` float32 frames go
    out raw while the link keeps up and TQ-pack encoded (lossy, on the grid
    2**-16, error <= 2**-17) as far as the link needs, also while they stay
    on the GPU. With ``drop_outside`` an index outside [0, ``capacity``)
    records nothing (in place of a ``tf.cond``, whose host predicate
    TensorFlow copies to the GPU and waits for). ``order`` is the replica's
    resource handle that orders its store ops (see ``order_variable``).
    Returns an empty token."""
    return _load_ops().v1_frame_store_offload(
        frame, index, order, key=key, capacity=capacity,
        resident_from=capacity if resident_from is None else resident_from, group=group,
        codec=codec, drop_outside=drop_outside,
    )


def take_frame(index, key, like, prefetch=0, group=1, zero_before_start=False, *, order):
    """Frame ``index`` of the store ``key`` on this GPU, shaped and typed as
    ``like``; with ``prefetch``, then starts fetching frames index - 1 ..
    index - prefetch in groups of ``group``. With ``zero_before_start`` a
    negative index returns zeros (in place of a ``tf.cond``, as above).
    ``order`` as for ``offload_frame``."""
    return _load_ops().v1_frame_store_take(
        index, order, key=key, T=like.dtype, shape=like.shape, prefetch=prefetch, group=group,
        zero_before_start=zero_before_start,
    )
