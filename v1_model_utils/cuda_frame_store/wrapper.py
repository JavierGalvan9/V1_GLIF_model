"""Park a sequence of GPU frames in pinned host memory (see frame_store_ops.cc)."""

from pathlib import Path
import warnings

import tensorflow as tf

from v1_model_utils.cuda_operator_cache import ensure_artifact

from .build import BUILD_FLAGS


HERE = Path(__file__).resolve().parent
_OPS = None
_UNAVAILABLE = False


def _load_ops():
    global _OPS
    if _OPS is None:
        library = ensure_artifact(
            HERE,
            "frame_store_ops",
            sources=(HERE / "build.py", HERE / "frame_store_ops.cc", HERE / "frame_store_ops.cu.cc",
                     HERE / "tq_codec.cu.h"),
            build_module="v1_model_utils.cuda_frame_store.build",
            build_flags=BUILD_FLAGS,
        )
        _OPS = tf.load_op_library(str(library))
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


def offload_frame(frame, index, key, capacity, resident_from=None, group=1, codec="raw"):
    """Hand ``frame`` to slot ``index`` (int32, < ``capacity``) of the host
    store ``key`` of this GPU. Slots at or above ``resident_from`` stay on the
    GPU; the others are copied out in groups of ``group`` as fast as the copy
    engine finishes them (earliest first) and released on the GPU once copied.
    Groups not admitted to the copy stream by the time the GPU passes slot
    ``capacity - 1`` stay on the GPU. With ``codec="tq"`` float32 frames go
    out raw while the link keeps up and TQ-pack encoded (lossy, on the grid
    2**-16, error <= 2**-17) as far as the link needs, also while they stay
    on the GPU. Returns an empty token."""
    return _load_ops().v1_frame_store_offload(
        frame, index, key=key, capacity=capacity,
        resident_from=capacity if resident_from is None else resident_from, group=group,
        codec=codec,
    )


def take_frame(index, key, like, prefetch=0, group=1):
    """Frame ``index`` of the store ``key`` on this GPU, shaped and typed as
    ``like``; with ``prefetch``, then starts fetching frames index - 1 ..
    index - prefetch in groups of ``group``."""
    return _load_ops().v1_frame_store_take(
        index, key=key, T=like.dtype, shape=like.shape, prefetch=prefetch, group=group
    )
