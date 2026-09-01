"""Runtime support for GPU-local opaque CSR connectivity metadata."""

from dataclasses import dataclass
import os
from pathlib import Path
import uuid

import tensorflow as tf

from v1_model_utils.cuda_operator_cache import ensure_artifact
from v1_model_utils.cuda_csr_resources.build import BUILD_FLAGS


_OPS = None


@dataclass(frozen=True)
class ResourceConnectivity:
    """Reference to CSR metadata owned by the local GPU resource manager."""

    name: str


def resource_mode_enabled():
    """Whether this process is a distributed worker using local resources."""
    return os.environ.get("V1_DISTRIBUTED_WORKER") == "1"


def load_ops():
    """Load the architecture-matched resource-current operator library."""
    global _OPS
    if _OPS is None:
        directory = Path(__file__).parent
        library = ensure_artifact(
            directory,
            "csr_resource_ops",
            sources=(
                directory / "csr_resource_ops.cc",
                directory / "csr_resource_ops.cu.cc",
                directory / "build.py",
                directory.parent
                / "cuda_csr_recurrent/csr_recurrent_ops.cc",
                directory.parent
                / "cuda_csr_recurrent/csr_recurrent_ops.cu.cc",
                directory.parent
                / "cuda_csr_external/csr_external_grad_ops.cu.cc",
            ),
            build_module="v1_model_utils.cuda_csr_resources.build",
            build_flags=BUILD_FLAGS,
        )
        _OPS = tf.load_op_library(str(library))
    return _OPS


def initialize_resource(metadata):
    """Copy CSR metadata once to the sole GPU visible to this worker."""
    devices = tf.config.list_physical_devices("GPU")
    if len(devices) != 1:
        raise RuntimeError(
            "V1 distributed workers require exactly one visible GPU; "
            f"found {len(devices)}."
        )
    name = f"v1_csr_{uuid.uuid4().hex}"
    with tf.device("/device:GPU:0"):
        values = [
            tf.cast(getattr(metadata, field), tf.int32)
            for field in (
                "post_ids",
                "synapse_types",
                "row_splits",
                "edge_ids",
                "nonempty_rows",
            )
        ]
        initialized = load_ops().initialize_v1_csr_resource(
            *values, resource_name=f"{name}_gpu0"
        )
    if not bool(initialized.numpy()):
        raise RuntimeError(f"Failed to initialize GPU-local CSR resource {name!r}.")
    return ResourceConnectivity(name)
