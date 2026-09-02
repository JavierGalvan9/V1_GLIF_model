"""Python interface for the fused recurrent-current CUDA operator."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tensorflow as tf

from v1_model_utils.cuda_operator_cache import ensure_artifact
from v1_model_utils.cuda_csr_recurrent.build import BUILD_FLAGS, DIRECT_CSR
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
    """Presynaptic CSR metadata with an original-edge permutation.

    ``edge_ids`` maps a CSR position to the edge's index in the caller's
    original order. ``edge_order`` is the same permutation kept on the host, so
    weights can be moved between the two orders without a device round trip.
    """

    post_ids: tf.Tensor
    synapse_types: tf.Tensor
    row_splits: tf.Tensor
    edge_ids: tf.Tensor
    nonempty_rows: tf.Tensor
    n_pre: int
    n_post: int
    n_edges: int
    resource_name: str | None = None
    edge_order: np.ndarray | None = None
    weights_csr_ordered: bool = False
    # Compact (postsynaptic neuron, synapse type) pairs. `pair_ids` gives the
    # pair of each CSR edge; `pair_posts`/`pair_types` describe each pair once.
    pair_ids: tf.Tensor | None = None
    pair_posts: tf.Tensor | None = None
    pair_types: tf.Tensor | None = None
    n_pairs: int = 0


def _compact_pairs(post_ids, synapse_types):
    """Describe each distinct (postsynaptic neuron, synapse type) pair once.

    The backward pass projects the upstream gradient onto the synaptic basis per
    (post, type) combination. Edges reuse those combinations heavily, so
    projecting the distinct pairs once and indexing them per edge removes almost
    all of the redundant projection work.
    """
    codes = (
        post_ids.astype(np.uint64) * (np.iinfo(np.uint8).max + 1)
        + synapse_types.astype(np.uint64)
    )
    unique_codes, pair_ids = np.unique(codes, return_inverse=True)
    return {
        "pair_ids": tf.constant(pair_ids.astype(np.uint32), tf.uint32),
        "pair_posts": tf.constant(
            (unique_codes >> 8).astype(np.uint32), tf.uint32
        ),
        "pair_types": tf.constant(
            (unique_codes & np.uint64(0xFF)).astype(np.uint8), tf.uint8
        ),
        "n_pairs": int(unique_codes.size),
    }


def to_csr_order(values, connectivity):
    """Reorder original-order edge values into CSR order.

    With :data:`DIRECT_CSR` the kernels index weights by CSR position, so
    anything edge-aligned that they touch has to be moved through here first.
    """
    if not DIRECT_CSR or connectivity.edge_order is None:
        return values
    return np.asarray(values)[connectivity.edge_order]


def to_original_order(values, connectivity):
    """Scatter CSR-order edge values back into the caller's original order."""
    if not DIRECT_CSR or connectivity.edge_order is None:
        return values
    values = np.asarray(values)
    restored = np.empty_like(values)
    restored[connectivity.edge_order] = values
    return restored


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


def require_csr_ordered_weights(connectivity, name):
    """Fail loudly when a caller has not moved its weights into CSR order.

    With :data:`DIRECT_CSR` compiled in, the kernels treat the CSR position as
    the weight index. Silently accepting original-order weights would pair every
    edge with the wrong weight, so callers must declare the ordering.
    """
    if DIRECT_CSR and not connectivity.weights_csr_ordered:
        raise ValueError(
            f"{name} was built with DIRECT_CSR kernels, which require weights in "
            "CSR edge order. Build the model from a network reordered by "
            "spatial_layout.apply_csr_edge_order, or rebuild the operator with "
            "DIRECT_CSR disabled."
        )


def build_csr_connectivity(
    indices, synapse_types, n_pre, n_post, weights_csr_ordered=False
):
    """Build compact pre-CSR metadata while preserving external edge order.

    Set ``weights_csr_ordered`` when the caller's edges are already in this
    operator's CSR order; the derived permutation is then asserted to be the
    identity, so a mismatch surfaces here rather than as silently mispaired
    weights.
    """
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
    if weights_csr_ordered and not np.array_equal(
        order, np.arange(order.size, dtype=order.dtype)
    ):
        raise ValueError(
            "edges were declared to be in CSR order but the derived permutation "
            "is not the identity; the CSR sort key and the caller's edge order "
            "have diverged"
        )
    pairs = _compact_pairs(indices[order, 0], synapse_types[order])
    connectivity = CsrConnectivity(
        post_ids=tf.constant(indices[order, 0], tf.uint32),
        synapse_types=tf.constant(synapse_types[order], tf.uint8),
        row_splits=tf.constant(row_splits, tf.uint32),
        edge_ids=tf.constant(order, tf.uint32),
        nonempty_rows=tf.constant(nonempty_rows, tf.uint32),
        n_pre=int(n_pre),
        n_post=int(n_post),
        n_edges=int(indices.shape[0]),
        edge_order=order,
        weights_csr_ordered=bool(weights_csr_ordered),
        **pairs,
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


def pair_projection_applies(spike_values, basis_values, connectivity):
    """Whether the compact pair-projected backward specialization can run.

    It is written for the measured hot shape only: a static batch of 32 with the
    four-column synaptic basis. Everything else keeps the general kernel.
    """
    if connectivity.pair_ids is None or connectivity.n_pairs == 0:
        return False
    batch = spike_values.shape[0]
    n_basis = basis_values.shape[-1]
    return batch == 32 and n_basis == 4


def empty_like_currents(basis):
    """The sentinel `initial` value meaning "start from zero"."""
    return tf.zeros((0, 0), basis.dtype)


def calculate_recurrent_csr_currents(
    spikes, weights, basis, dampening, connectivity, initial=None
):
    """Calculate currents plus spike and weight gradients.

    ``initial`` accumulates this source's currents on top of another source's
    output, which avoids materializing a separate tensor and adding it later.
    Its gradient is the upstream gradient unchanged.
    """
    require_csr_ordered_weights(connectivity, "recurrent connectivity")
    if connectivity.resource_name is not None:
        # The resource operator has no `initial` input, so fall back to an
        # explicit add rather than dropping it.
        currents = _calculate_resource_currents(
            spikes, weights, basis, dampening, connectivity
        )
        return currents if initial is None else currents + initial
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
        pair_ids,
        pair_posts,
        pair_types,
        initial_values,
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
            initial_values,
            n_post=connectivity.n_post,
        )
        use_pairs = pair_projection_applies(
            spike_values, basis_values, connectivity
        )

        def grad(current_grad):
            common = (
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
            )
            if use_pairs:
                spike_grad, weight_grad = ops.v1_csr_backward_pair_projected(
                    *common,
                    pair_ids,
                    pair_posts,
                    pair_types,
                    n_post=connectivity.n_post,
                    n_edges=connectivity.n_edges,
                )
            else:
                spike_grad, weight_grad = ops.v1_csr_backward(
                    *common,
                    n_post=connectivity.n_post,
                    n_edges=connectivity.n_edges,
                )
            # The trailing gradient belongs to `initial`, which enters the
            # output additively, so the upstream gradient passes straight
            # through. With no accumulator the input is an empty sentinel, and
            # its gradient has to stay unset rather than take the output shape.
            initial_grad = None if initial is None else current_grad
            return (
                spike_grad,
                tf.cast(weight_grad, weight_values.dtype),
            ) + (None,) * 10 + (initial_grad,)

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
        connectivity.pair_ids,
        connectivity.pair_posts,
        connectivity.pair_types,
        empty_like_currents(basis) if initial is None else initial,
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
