"""Python interface for the fused recurrent-current CUDA operator."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tensorflow as tf


SPECIALIZED_BATCH_SIZES = (1, 2, 4, 8, 16, 32, 64, 128, 256)
LIBRARY = Path(__file__).with_name("_synaptic_current_ops.so")
_OPS = None


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


def kernel_variant(n_basis, batch_size):
    """Return the CUDA specialization selected for diagnostic reporting."""
    basis = "basis4" if int(n_basis) == 4 else "generic_basis"
    batch = int(batch_size)
    suffix = f"batch{batch}" if batch in SPECIALIZED_BATCH_SIZES else "generic_batch"
    return f"{basis}_{suffix}"


def _load_ops():
    global _OPS
    if _OPS is None:
        if not LIBRARY.exists():
            raise FileNotFoundError(
                f"CUDA synaptic-current operator is not built; run "
                f"`python -m v1_model_utils.cuda_synaptic_currents.build`: {LIBRARY}"
            )
        _OPS = tf.load_op_library(str(LIBRARY))
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
    return CsrConnectivity(
        post_ids=tf.constant(indices[order, 0], tf.uint32),
        synapse_types=tf.constant(synapse_types[order], tf.uint8),
        row_splits=tf.constant(row_splits, tf.uint32),
        edge_ids=tf.constant(order, tf.uint32),
        nonempty_rows=tf.constant(nonempty_rows, tf.uint32),
        n_pre=int(n_pre),
        n_post=int(n_post),
        n_edges=int(indices.shape[0]),
    )


def calculate_synaptic_currents(spikes, weights, basis, dampening, connectivity):
    """Calculate currents with gradients for spikes and original-order weights."""
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
        active = tf.where(spike_values != tf.cast(0, spike_values.dtype))
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
