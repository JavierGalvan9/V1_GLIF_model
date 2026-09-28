"""Runtime support for GPU-local opaque CSR connectivity metadata."""

from dataclasses import dataclass
import os
from pathlib import Path
import uuid

import tensorflow as tf

from v1_model_utils.cuda_operator_cache import ensure_artifact
from v1_model_utils.cuda_csr_resources.build import build_flags_for


_OPS = None


@dataclass(frozen=True)
class ResourceConnectivity:
    """Reference to CSR metadata owned by the local GPU resource manager."""

    name: str


def resource_mode_enabled():
    """Whether connectivity metadata belongs in GPU-local operator resources.

    Every replicated run needs it. As ordinary graph tensors the CSR metadata
    is placed on one device and copied to every other replica on each use,
    which both unbalances GPU memory and serializes the step behind those
    copies. A resource is created once per device instead, so each replica
    reads metadata that already lives on its own GPU.

    The decision is declared, never inferred from how many GPUs happen to be
    visible: ``create_distribution_strategy`` sets ``V1_CSR_RESOURCE_MODE``
    when it builds a multi-replica strategy, and the one-process-per-GPU
    launcher sets ``V1_DISTRIBUTED_WORKER``. A single replica keeps the plain
    tensor path; both backends run the same kernels, including the
    background-input forward gather.
    """
    override = os.environ.get("V1_CSR_RESOURCE_MODE")
    if override is not None:
        return override == "1"
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
                / "cuda_csr_recurrent/event_weight_grad.cuh",
                directory.parent
                / "cuda_csr_external/csr_external_grad_ops.cu.cc",
                directory.parent
                / "cuda_csr_external/generic_backward_kernels.cuh",
            ),
            build_module="v1_model_utils.cuda_csr_resources.build",
            build_flags=build_flags_for,
        )
        _OPS = tf.load_op_library(str(library))
    return _OPS


# Resource fields and their dtypes, in InitializeV1CsrResource input order: the
# tensor backend's own widths, so both backends run identical kernels.
_METADATA_FIELDS = {
    "post_ids": tf.uint32,
    "synapse_types": tf.uint8,
    "row_splits": tf.uint32,
    "edge_ids": tf.uint32,
    "nonempty_rows": tf.uint32,
    "pair_ids": tf.uint32,
    "pair_posts": tf.uint32,
    "pair_types": tf.uint8,
    # The BKG forward gather's incoming CSR; absent from every other source.
    "incoming_row_splits": tf.uint32,
    "incoming_pre_ids": tf.uint32,
    "incoming_edge_ids": tf.uint32,
    "incoming_types": tf.uint8,
}


def _metadata_value(metadata, field):
    """A resource field in its dtype; one the connectivity lacks is uploaded empty."""
    dtype = _METADATA_FIELDS[field]
    value = getattr(metadata, field, None)
    return tf.zeros((0,), dtype) if value is None else tf.cast(value, dtype)


def _device_suffix(device_name):
    """Return the GPU ordinal the operators append to a resource name.

    ``DeviceResourceName`` in the kernel derives the suffix from the executing
    device, so the Python side has to key the resource it creates by the same
    ordinal rather than by enumeration order.
    """
    return device_name.rsplit(":", 1)[1]


def replica_devices():
    """The GPUs this process places replicas on.

    A one-process-per-GPU worker exposes exactly one GPU, but under
    ``MultiWorkerMirroredStrategy`` its eager context also lists the other
    workers' devices. Touching those from here would place the metadata on
    another worker's GPU - and fails outright before its remote eager service
    is up - so a worker only ever addresses its own device.
    """
    if os.environ.get("V1_DISTRIBUTED_WORKER") == "1":
        return ("/device:GPU:0",)
    return tuple(device.name for device in tf.config.list_logical_devices("GPU"))


def initialize_resource(metadata):
    """Copy CSR metadata once to each GPU this process will run replicas on.

    Every such GPU receives its own resource under one shared base name. The
    forward and backward operators resolve the copy that belongs to the device
    they execute on, so no replica ever reads another GPU's metadata.
    """
    devices = replica_devices()
    if not devices:
        raise RuntimeError("GPU-local CSR resources require a visible GPU.")
    name = f"v1_csr_{uuid.uuid4().hex}"
    for device in devices:
        with tf.device(device):
            values = [_metadata_value(metadata, field) for field in _METADATA_FIELDS]
            initialized = load_ops().initialize_v1_csr_resource(
                *values,
                resource_name=f"{name}_gpu{_device_suffix(device)}",
            )
        if not bool(initialized.numpy()):
            raise RuntimeError(
                f"Failed to initialize CSR resource {name!r} on {device}."
            )
    return ResourceConnectivity(name)
