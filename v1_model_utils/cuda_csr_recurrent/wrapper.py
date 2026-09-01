"""Python interface for the fused recurrent-current CUDA operator."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tensorflow as tf

from v1_model_utils.cuda_operator_cache import ensure_artifact
from v1_model_utils.cuda_csr_recurrent.build import BUILD_FLAGS
from v1_model_utils.cuda_csr_resources import (
    initialize_resource,
    load_ops as load_resource_ops,
    resource_mode_enabled,
)


SPECIALIZED_BATCH_SIZES = (1, 2, 4, 8, 16, 32, 64, 128, 256)
_OPS = None


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
    """Presynaptic CSR metadata with an original-edge permutation."""

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
    """Return the CUDA specialization selected for diagnostic reporting."""
    basis = "basis4" if int(n_basis) == 4 else "generic_basis"
    batch = int(batch_size)
    suffix = f"batch{batch}" if batch in SPECIALIZED_BATCH_SIZES else "generic_batch"
    return f"{basis}_{suffix}"


def _load_ops():
    global _OPS
    if _OPS is None:
        directory = Path(__file__).parent
        library = ensure_artifact(
            directory,
            "csr_recurrent_ops",
            sources=(
                directory / "build.py",
                directory / "csr_recurrent_ops.cc",
                directory / "csr_recurrent_ops.cu.cc",
            ),
            build_module="v1_model_utils.cuda_csr_recurrent.build",
            build_flags=BUILD_FLAGS,
        )
        _OPS = tf.load_op_library(str(library))
    return _OPS


def build_csr_connectivity(indices, synapse_types, n_pre, n_post):
    """Build compact pre-CSR metadata while preserving external edge order."""
    indices = np.asarray(indices)
    synapse_types = np.asarray(synapse_types)
    if indices.ndim != 2 or indices.shape[1] != 2:
        raise ValueError("indices must have shape [n_edges, 2] as [post, pre]")
    if synapse_types.shape != (indices.shape[0],):
        raise ValueError("synapse_types must contain one entry per edge")
    if not (0 < int(n_pre) <= np.iinfo(np.uint32).max):
        raise ValueError("n_pre must fit uint32")
    if not (0 < int(n_post) <= np.iinfo(np.uint32).max):
        raise ValueError("n_post must fit uint32")
    if indices.shape[0] > np.iinfo(np.uint32).max:
        raise ValueError("edge count must fit uint32")
    if np.any(indices < 0) or np.any(indices[:, 0] >= n_post) or np.any(indices[:, 1] >= n_pre):
        raise ValueError("connectivity index is outside the declared shape")
    if np.any(synapse_types < 0) or np.any(synapse_types > np.iinfo(np.uint8).max):
        raise ValueError("synapse types must fit uint8")

    original = np.arange(indices.shape[0], dtype=np.uint32)
    order = np.lexsort((original, synapse_types, indices[:, 0], indices[:, 1])).astype(
        np.uint32, copy=False
    )
    counts = np.bincount(indices[:, 1], minlength=n_pre).astype(np.uint64)
    row_splits = np.empty(n_pre + 1, dtype=np.uint32)
    row_splits[0] = 0
    row_splits[1:] = np.cumsum(counts, dtype=np.uint64).astype(np.uint32)
    nonempty_rows = np.flatnonzero(counts).astype(np.uint32)
    connectivity = CsrConnectivity(
        post_ids=tf.constant(indices[order, 0], tf.uint32),
        synapse_types=tf.constant(synapse_types[order], tf.uint8),
        row_splits=tf.constant(row_splits, tf.uint32),
        edge_ids=tf.constant(order, tf.uint32),
        nonempty_rows=tf.constant(nonempty_rows, tf.uint32),
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


def calculate_recurrent_csr_currents(
    spikes, weights, basis, dampening, connectivity
):
    """Calculate currents with gradients for spikes and original-order weights."""
    if connectivity.resource_name is not None:
        return _calculate_resource_currents(
            spikes, weights, basis, dampening, connectivity
        )
    ops = _load_ops()

    @tf.custom_gradient
    def fused(
        spike_values,
        weight_values,
        basis_values,
        post_ids,
        synapse_types,
        row_splits,
        edge_ids,
        nonempty_rows,
        dampening_value,
    ):
        active = _active_rows_or_pairs(spike_values, basis_values)
        currents = ops.v1_csr_forward(
            spike_values,
            active,
            weight_values,
            post_ids,
            synapse_types,
            row_splits,
            edge_ids,
            basis_values,
            n_post=connectivity.n_post,
        )

        def grad(current_grad):
            spike_grad, weight_grad = ops.v1_csr_backward(
                spike_values,
                current_grad,
                weight_values,
                post_ids,
                synapse_types,
                row_splits,
                edge_ids,
                nonempty_rows,
                basis_values,
                tf.cast(dampening_value, spike_values.dtype),
                n_post=connectivity.n_post,
                n_edges=connectivity.n_edges,
            )
            return (
                spike_grad,
                tf.cast(weight_grad, weight_values.dtype),
                None,
                None,
                None,
                None,
                None,
                None,
                None,
            )

        return currents, grad

    return fused(
        spikes,
        weights,
        basis,
        connectivity.post_ids,
        connectivity.synapse_types,
        connectivity.row_splits,
        connectivity.edge_ids,
        connectivity.nonempty_rows,
        tf.cast(dampening, spikes.dtype),
    )


def _calculate_resource_currents(spikes, weights, basis, dampening, connectivity):
    ops = load_resource_ops()

    @tf.custom_gradient
    def fused(spike_values, weight_values, basis_values, dampening_value):
        active = _active_rows_or_pairs(spike_values, basis_values)
        currents = ops.v1_csr_forward_resource(
            spike_values,
            active,
            weight_values,
            basis_values,
            n_post=connectivity.n_post,
            resource_name=connectivity.resource_name,
        )

        def grad(current_grad):
            spike_grad, weight_grad = ops.v1_csr_backward_resource(
                spike_values,
                current_grad,
                weight_values,
                basis_values,
                tf.cast(dampening_value, spike_values.dtype),
                n_post=connectivity.n_post,
                n_edges=connectivity.n_edges,
                resource_name=connectivity.resource_name,
            )
            return spike_grad, tf.cast(weight_grad, weight_values.dtype), None, None

        return currents, grad

    return fused(spikes, weights, basis, tf.cast(dampening, spikes.dtype))
