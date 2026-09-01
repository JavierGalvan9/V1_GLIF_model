"""TensorFlow interface for fused LGN/background synaptic currents."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tensorflow as tf

from v1_model_utils.cuda_operator_cache import ensure_artifact
from v1_model_utils.cuda_csr_external.build import BUILD_FLAGS
from v1_model_utils.cuda_csr_recurrent.build import (
    BUILD_FLAGS as RECURRENT_BUILD_FLAGS,
)
from v1_model_utils.cuda_csr_resources import (
    initialize_resource,
    load_ops as load_resource_ops,
    resource_mode_enabled,
)


SPECIALIZED_BATCH_SIZES = (1, 2, 4, 8, 16, 32, 64, 128, 256)
_OPS = None
_RECURRENT_OPS = None


def _active_rows_or_pairs(values, basis_values):
    """Use grouped CSR rows only for the measured static fast paths."""
    if values.shape[0] in (1, 2, 4, 8, 16, 32, 64, 128) and basis_values.shape[-1] == 4:
        row_ids = tf.cast(
            tf.where(tf.reduce_any(values != tf.cast(0, values.dtype), axis=0))[:, 0],
            tf.int64,
        )
        return tf.stack((tf.zeros_like(row_ids), row_ids), axis=1)
    return tf.where(values != tf.cast(0, values.dtype))


@dataclass(frozen=True)
class CsrConnectivity:
    """Presynaptic CSR metadata with original-edge weight ordering."""

    post_ids: tf.Tensor
    synapse_types: tf.Tensor
    row_splits: tf.Tensor
    edge_ids: tf.Tensor
    nonempty_rows: tf.Tensor
    n_pre: int
    n_post: int
    n_edges: int
    resource_name: str | None = None


def kernel_variant(n_basis, batch_size):
    """Return the selected basis and backward batch specializations."""
    basis = "basis4" if int(n_basis) == 4 else "generic_basis"
    batch = int(batch_size)
    suffix = f"batch{batch}" if batch in SPECIALIZED_BATCH_SIZES else "runtime_batch"
    return f"{basis}_{suffix}"


def build_csr_connectivity(indices, synapse_types, n_pre, n_post):
    """Create compact pre-CSR metadata while preserving edge weight order."""
    indices = np.asarray(indices)
    types = np.asarray(synapse_types)
    if indices.ndim != 2 or indices.shape[1] != 2:
        raise ValueError("indices must have shape [n_edges, 2] as [post, pre]")
    if types.shape != (indices.shape[0],):
        raise ValueError("synapse_types must contain one value per edge")
    limit = np.iinfo(np.uint32).max
    if not 0 < int(n_pre) <= limit or not 0 < int(n_post) <= limit:
        raise ValueError("n_pre and n_post must fit uint32")
    if indices.shape[0] > limit:
        raise ValueError("edge count must fit uint32")
    if np.any(indices < 0) or np.any(indices[:, 0] >= n_post) or np.any(
        indices[:, 1] >= n_pre
    ):
        raise ValueError("connectivity index is outside the declared shape")
    if np.any(types < 0) or np.any(types > np.iinfo(np.uint8).max):
        raise ValueError("synapse types must fit uint8")

    original = np.arange(indices.shape[0], dtype=np.uint32)
    order = np.lexsort((original, indices[:, 0], indices[:, 1])).astype(
        np.uint32, copy=False
    )
    counts = np.bincount(indices[:, 1], minlength=n_pre).astype(np.uint64)
    offsets = np.empty(int(n_pre) + 1, dtype=np.uint32)
    offsets[0] = 0
    offsets[1:] = np.cumsum(counts, dtype=np.uint64).astype(np.uint32)
    connectivity = CsrConnectivity(
        post_ids=tf.constant(indices[order, 0], tf.uint32),
        synapse_types=tf.constant(types[order], tf.uint8),
        row_splits=tf.constant(offsets, tf.uint32),
        edge_ids=tf.constant(order, tf.uint32),
        nonempty_rows=tf.constant(np.flatnonzero(counts).astype(np.uint32)),
        n_pre=int(n_pre),
        n_post=int(n_post),
        n_edges=int(indices.shape[0]),
    )
    if resource_mode_enabled():
        resource = initialize_resource(connectivity)
        connectivity = CsrConnectivity(
            **{
                **connectivity.__dict__,
                "resource_name": resource.name,
            }
        )
    return connectivity


def _load_ops():
    global _OPS, _RECURRENT_OPS
    if _OPS is None:
        recurrent_directory = Path(__file__).parents[1] / "cuda_csr_recurrent"
        recurrent_library = ensure_artifact(
            recurrent_directory,
            "csr_recurrent_ops",
            sources=(
                recurrent_directory / "build.py",
                recurrent_directory / "csr_recurrent_ops.cc",
                recurrent_directory / "csr_recurrent_ops.cu.cc",
            ),
            build_module="v1_model_utils.cuda_csr_recurrent.build",
            build_flags=RECURRENT_BUILD_FLAGS,
        )
        directory = Path(__file__).parent
        library = ensure_artifact(
            directory,
            "csr_external_grad_ops",
            sources=(
                directory / "build.py",
                directory / "csr_external_grad_ops.cc",
                directory / "csr_external_grad_ops.cu.cc",
            ),
            build_module="v1_model_utils.cuda_csr_external.build",
            build_flags=BUILD_FLAGS,
        )
        _RECURRENT_OPS = tf.load_op_library(str(recurrent_library))
        _OPS = tf.load_op_library(str(library))
    return _RECURRENT_OPS, _OPS


def calculate_external_csr_currents(
    activity,
    weights,
    basis,
    connectivity,
    *,
    compute_activity_gradient=True,
):
    """Return currents and original-order FP32 weight gradients.

    When ``compute_activity_gradient`` is false, backward invokes a distinct
    weight-only CUDA op. It neither allocates nor computes an activity-gradient
    tensor. The full derivative remains available for diagnostics and reuse.
    """
    activity = tf.convert_to_tensor(activity)
    weights = tf.convert_to_tensor(weights, tf.float32)
    basis = tf.cast(basis, activity.dtype)
    if activity.shape.rank != 2:
        raise ValueError("activity must be rank two")
    if activity.shape[-1] is not None and int(activity.shape[-1]) != connectivity.n_pre:
        raise ValueError("activity width does not match connectivity.n_pre")
    if basis.shape.rank != 2:
        raise ValueError("basis must be rank two")
    if connectivity.resource_name is not None:
        return _calculate_resource_currents(
            activity,
            weights,
            basis,
            connectivity,
            compute_activity_gradient=compute_activity_gradient,
        )
    recurrent_ops, external_ops = _load_ops()

    @tf.custom_gradient
    def fused(
        values,
        master_weights,
        basis_values,
        post_ids,
        synapse_types,
        row_splits,
        edge_ids,
        nonempty_rows,
    ):
        active = _active_rows_or_pairs(values, basis_values)
        currents = recurrent_ops.v1_csr_forward(
            values,
            active,
            master_weights,
            post_ids,
            synapse_types,
            row_splits,
            edge_ids,
            basis_values,
            n_post=connectivity.n_post,
        )

        def grad(upstream):
            if compute_activity_gradient:
                activity_grad, weight_grad = recurrent_ops.v1_csr_backward(
                    values,
                    upstream,
                    master_weights,
                    post_ids,
                    synapse_types,
                    row_splits,
                    edge_ids,
                    nonempty_rows,
                    basis_values,
                    tf.cast(1, values.dtype),
                    n_post=connectivity.n_post,
                    n_edges=connectivity.n_edges,
                )
            else:
                activity_grad = None
                weight_grad = external_ops.external_csr_weight_backward(
                    values,
                    upstream,
                    post_ids,
                    synapse_types,
                    row_splits,
                    edge_ids,
                    nonempty_rows,
                    basis_values,
                    n_post=connectivity.n_post,
                    n_edges=connectivity.n_edges,
                )
            return activity_grad, weight_grad, None, None, None, None, None, None

        return currents, grad

    return fused(
        activity,
        weights,
        basis,
        connectivity.post_ids,
        connectivity.synapse_types,
        connectivity.row_splits,
        connectivity.edge_ids,
        connectivity.nonempty_rows,
    )


def _calculate_resource_currents(
    activity,
    weights,
    basis,
    connectivity,
    *,
    compute_activity_gradient,
):
    ops = load_resource_ops()

    @tf.custom_gradient
    def fused(values, master_weights, basis_values):
        active = _active_rows_or_pairs(values, basis_values)
        currents = ops.v1_csr_forward_resource(
            values,
            active,
            master_weights,
            basis_values,
            n_post=connectivity.n_post,
            resource_name=connectivity.resource_name,
        )

        def grad(upstream):
            if compute_activity_gradient:
                activity_grad, weight_grad = ops.v1_csr_backward_resource(
                    values,
                    upstream,
                    master_weights,
                    basis_values,
                    tf.cast(1, values.dtype),
                    n_post=connectivity.n_post,
                    n_edges=connectivity.n_edges,
                    resource_name=connectivity.resource_name,
                )
            else:
                activity_grad = None
                weight_grad = ops.external_csr_weight_backward_resource(
                    values,
                    upstream,
                    basis_values,
                    n_post=connectivity.n_post,
                    n_edges=connectivity.n_edges,
                    resource_name=connectivity.resource_name,
                )
            return activity_grad, tf.cast(weight_grad, master_weights.dtype), None

        return currents, grad

    return fused(
        tf.convert_to_tensor(activity),
        tf.convert_to_tensor(weights, tf.float32),
        tf.cast(basis, activity.dtype),
    )
