"""TensorFlow interface for fused LGN/background synaptic currents."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tensorflow as tf

from v1_model_utils import csr_order
from v1_model_utils.cuda_operator_cache import ensure_artifact
from v1_model_utils.cuda_csr_external.build import build_flags_for, DIRECT_CSR
from v1_model_utils.cuda_csr_recurrent.build import (
    build_flags_for as recurrent_build_flags_for,
)
from v1_model_utils.cuda_csr_recurrent.wrapper import (
    empty_like_currents,
    empty_metadata,
    require_csr_ordered_weights,
)
from v1_model_utils.cuda_csr_resources import (
    initialize_resource,
    load_ops as load_resource_ops,
    resource_mode_enabled,
)


SPECIALIZED_BATCH_SIZES = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512)
_OPS = None
_RECURRENT_OPS = None


def _edge_index_tensor(order):
    """Upload the CSR-to-original permutation only when a kernel reads it.

    Mirrors the recurrent operator: with :data:`DIRECT_CSR` compiled in,
    ``V1_EDGE_INDEX`` expands to the CSR position and no kernel dereferences
    this input. The resource operator checks its length, so it keeps the real
    permutation.
    """
    if DIRECT_CSR and not resource_mode_enabled():
        return empty_metadata(tf.uint32)
    return tf.constant(order, tf.uint32)


@dataclass(frozen=True)
class CsrConnectivity:
    """Presynaptic CSR metadata with original-edge weight ordering.

    ``edge_order`` mirrors ``edge_ids`` on the host so weights can move between
    the caller's order and CSR order without a device round trip. ``edge_ids``
    itself is empty under :data:`DIRECT_CSR`, and the ``pair_*`` projection
    metadata is empty for a connectivity whose backward never runs.
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
    pair_ids: tf.Tensor | None = None
    pair_posts: tf.Tensor | None = None
    pair_types: tf.Tensor | None = None
    n_pairs: int = 0
    incoming_row_splits: tf.Tensor | None = None
    incoming_pre_ids: tf.Tensor | None = None
    incoming_edge_ids: tf.Tensor | None = None
    incoming_types: tf.Tensor | None = None
    # Sparse activity (LGN) takes the event-driven weight gradient; dense
    # activity (the Poisson BKG) the compact pair-projected one.
    sparse_activity: bool = True
    # Whether some row sends two edges to one target; see csr_order.repeats_targets.
    repeats_targets: bool = True


def _compact_pairs(post_ids, synapse_types, needed=True):
    """Return the unique postsynaptic/type projection shared by each edge.

    The activity gradient and the dense weight gradient project onto the basis
    per pair; the event-driven weight gradient reads the edges directly. So a
    sparse connectivity whose activity is never differentiated - the default
    LGN input - carries empty metadata instead of one uint32 per edge that
    nothing reads: 365 MiB on the LGN input of the 203,816-neuron network.
    """
    if not needed:
        return {
            "pair_ids": empty_metadata(tf.uint32),
            "pair_posts": empty_metadata(tf.uint32),
            "pair_types": empty_metadata(tf.uint8),
            "n_pairs": 0,
        }
    unique_codes, pair_ids = csr_order.compact_pairs(post_ids, synapse_types)
    return {
        "pair_ids": tf.constant(pair_ids.astype(np.uint32), tf.uint32),
        "pair_posts": tf.constant((unique_codes >> 8).astype(np.uint32), tf.uint32),
        "pair_types": tf.constant((unique_codes & 0xFF).astype(np.uint8), tf.uint8),
        "n_pairs": int(unique_codes.size),
    }


def _bkg_incoming_metadata(
    post_ids, synapse_types, row_splits, n_pre, n_post, weights_csr_ordered
):
    """Build the fixed-four incoming CSR used by the BKG forward fast path.

    Both backends use it: the tensor operators read it from these tensors, and
    resource mode stores it in each GPU's resource.
    """
    if (
        not weights_csr_ordered
        or n_pre != 100
        or len(post_ids) != 4 * n_post
    ):
        return {}
    post_counts = np.bincount(post_ids, minlength=n_post)
    if post_counts.size != n_post or not np.all(post_counts == 4):
        return {}
    incoming_order = np.argsort(post_ids, kind="stable").astype(np.uint32)
    pre_ids = np.repeat(
        np.arange(n_pre, dtype=np.uint32), np.diff(row_splits).astype(np.int64)
    )
    return {
        "incoming_row_splits": tf.constant(
            np.arange(0, 4 * n_post + 1, 4, dtype=np.uint32)
        ),
        "incoming_pre_ids": tf.constant(pre_ids[incoming_order], tf.uint32),
        "incoming_edge_ids": tf.constant(incoming_order, tf.uint32),
        "incoming_types": tf.constant(synapse_types[incoming_order], tf.uint8),
    }


def uses_bkg_gather(connectivity, basis):
    """Whether the forward takes the fixed-four incoming gather.

    The one gate for both backends, so the tensor and resource paths cannot
    select different kernels for the same connectivity.
    """
    return connectivity.incoming_row_splits is not None and basis.shape[-1] == 4


def kernel_variant(n_basis, batch_size):
    """Return the selected basis and backward batch specializations."""
    basis = "basis4" if int(n_basis) == 4 else "generic_basis"
    batch = int(batch_size)
    suffix = f"batch{batch}" if batch in SPECIALIZED_BATCH_SIZES else "runtime_batch"
    return f"{basis}_{suffix}"


def build_csr_connectivity(
    indices, synapse_types, n_pre, n_post, weights_csr_ordered=False,
    needs_activity_backward=True, sparse_activity=True,
):
    """Create compact pre-CSR metadata while preserving edge weight order.

    Set ``weights_csr_ordered`` when the caller's edges already follow this
    operator's CSR order; the derived permutation is then asserted to be the
    identity. Clear ``needs_activity_backward`` when no activity gradient will
    ever be requested, so a sparse connectivity does not build or upload the
    per-edge pair projection. Clear ``sparse_activity`` for an input that is
    active in most rows every step, such as the Poisson background: its weight
    gradient then uses the dense pair-projected kernel, which needs the
    projection, instead of the event-driven one.
    """
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

    keys = (indices[:, 1], indices[:, 0])
    if weights_csr_ordered:
        if not csr_order.is_identity_order(keys):
            raise ValueError(
                "edges were declared to be in CSR order but the derived "
                "permutation is not the identity"
            )
        # The permutation is the identity, so every gather below it is too.
        order = np.arange(indices.shape[0], dtype=np.uint32)
        ordered_posts = indices[:, 0].astype(np.uint32, copy=False)
        ordered_types = types.astype(np.uint8, copy=False)
    else:
        order = csr_order.edge_order(keys)
        ordered_posts = indices[order, 0].astype(np.uint32, copy=False)
        ordered_types = types[order].astype(np.uint8, copy=False)
    counts = np.bincount(indices[:, 1], minlength=n_pre).astype(np.uint64)
    offsets = np.empty(int(n_pre) + 1, dtype=np.uint32)
    offsets[0] = 0
    offsets[1:] = np.cumsum(counts, dtype=np.uint64).astype(np.uint32)
    connectivity = CsrConnectivity(
        post_ids=tf.constant(ordered_posts, tf.uint32),
        synapse_types=tf.constant(ordered_types, tf.uint8),
        row_splits=tf.constant(offsets, tf.uint32),
        edge_ids=_edge_index_tensor(order),
        nonempty_rows=tf.constant(np.flatnonzero(counts).astype(np.uint32)),
        n_pre=int(n_pre),
        n_post=int(n_post),
        n_edges=int(indices.shape[0]),
        edge_order=order,
        weights_csr_ordered=bool(weights_csr_ordered),
        sparse_activity=bool(sparse_activity),
        repeats_targets=csr_order.repeats_targets(ordered_posts, offsets),
        **_compact_pairs(
            ordered_posts, ordered_types, needs_activity_backward or not sparse_activity
        ),
        **_bkg_incoming_metadata(
            ordered_posts,
            ordered_types,
            offsets,
            int(n_pre),
            int(n_post),
            bool(weights_csr_ordered),
        ),
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
                recurrent_directory / "event_weight_grad.cuh",
            ),
            build_module="v1_model_utils.cuda_csr_recurrent.build",
            build_flags=recurrent_build_flags_for,
        )
        directory = Path(__file__).parent
        library = ensure_artifact(
            directory,
            "csr_external_grad_ops",
            sources=(
                directory / "build.py",
                directory / "csr_external_grad_ops.cc",
                directory / "csr_external_grad_ops.cu.cc",
                directory / "generic_backward_kernels.cuh",
                recurrent_directory / "event_weight_grad.cuh",
            ),
            build_module="v1_model_utils.cuda_csr_external.build",
            build_flags=build_flags_for,
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
    compute_weight_gradient=True,
    initial=None,
):
    """Return currents and original-order FP32 weight gradients.

    Activity and weight gradients are independently selectable. Disabled
    derivatives are neither allocated nor computed.

    ``initial`` accumulates these currents on top of another source's output
    rather than returning a separate tensor for a later add to combine. The
    kernels take the synaptic basis in FP32.
    """
    activity = tf.convert_to_tensor(activity)
    weights = tf.convert_to_tensor(weights, tf.float32)
    basis = tf.cast(basis, tf.float32)
    if activity.shape.rank != 2:
        raise ValueError("activity must be rank two")
    if activity.shape[-1] is not None and int(activity.shape[-1]) != connectivity.n_pre:
        raise ValueError("activity width does not match connectivity.n_pre")
    if basis.shape.rank != 2:
        raise ValueError("basis must be rank two")
    require_csr_ordered_weights(connectivity, "external connectivity")
    if connectivity.resource_name is not None:
        return _calculate_resource_currents(
            activity,
            weights,
            basis,
            connectivity,
            initial=initial,
            compute_activity_gradient=compute_activity_gradient,
            compute_weight_gradient=compute_weight_gradient,
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
        initial_values,
        pair_ids,
        pair_posts,
        pair_types,
    ):
        if uses_bkg_gather(connectivity, basis_values):
            currents = external_ops.bkg_csr_forward(
                values,
                master_weights,
                connectivity.incoming_row_splits,
                connectivity.incoming_pre_ids,
                connectivity.incoming_edge_ids,
                connectivity.incoming_types,
                basis_values,
                initial_values,
                n_post=connectivity.n_post,
            )
        else:
            currents = recurrent_ops.v1_csr_forward(
                [values],
                master_weights,
                post_ids,
                synapse_types,
                row_splits,
                edge_ids,
                basis_values,
                initial_values,
                [],  # no carried spike-history queue records
                n_post=connectivity.n_post,
                aggregate_runs=connectivity.repeats_targets,
            ).currents

        def grad(upstream):
            if compute_activity_gradient:
                activity_grad = external_ops.external_csr_activity_backward(
                    upstream,
                    master_weights,
                    post_ids,
                    synapse_types,
                    row_splits,
                    edge_ids,
                    nonempty_rows,
                    basis_values,
                    pair_ids,
                    pair_posts,
                    pair_types,
                    n_post=connectivity.n_post,
                )
            else:
                activity_grad = None
            if compute_weight_gradient:
                weight_grad = external_ops.external_csr_weight_backward(
                    values,
                    upstream,
                    post_ids,
                    synapse_types,
                    row_splits,
                    edge_ids,
                    nonempty_rows,
                    basis_values,
                    pair_ids,
                    pair_posts,
                    pair_types,
                    sparse_activity=connectivity.sparse_activity,
                    n_post=connectivity.n_post,
                    n_edges=connectivity.n_edges,
                )
            else:
                weight_grad = None
            # `initial` enters additively, so its gradient is the upstream
            # one - unless there is no accumulator, in which case the input is
            # an empty sentinel whose gradient must stay unset.
            initial_grad = None if initial is None else upstream
            return (activity_grad, weight_grad, None, None, None, None, None,
                    None, initial_grad, None, None, None)

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
        empty_like_currents(activity) if initial is None else initial,
        connectivity.pair_ids,
        connectivity.pair_posts,
        connectivity.pair_types,
    )


def _calculate_resource_currents(
    activity,
    weights,
    basis,
    connectivity,
    *,
    initial=None,
    compute_activity_gradient,
    compute_weight_gradient,
):
    ops = load_resource_ops()

    @tf.custom_gradient
    def fused(values, master_weights, basis_values, initial_values):
        if uses_bkg_gather(connectivity, basis_values):
            currents = ops.bkg_csr_forward_resource(
                values,
                master_weights,
                basis_values,
                initial_values,
                n_post=connectivity.n_post,
                resource_name=connectivity.resource_name,
            )
        else:
            currents = ops.v1_csr_forward_resource(
                [values],
                master_weights,
                basis_values,
                initial_values,
                [],  # no carried spike-history queue records
                n_post=connectivity.n_post,
                resource_name=connectivity.resource_name,
                aggregate_runs=connectivity.repeats_targets,
            ).currents

        def grad(upstream):
            if compute_activity_gradient:
                activity_grad = ops.external_csr_activity_backward_resource(
                    upstream,
                    master_weights,
                    basis_values,
                    n_post=connectivity.n_post,
                    resource_name=connectivity.resource_name,
                )
            else:
                activity_grad = None
            if compute_weight_gradient:
                weight_grad = ops.external_csr_weight_backward_resource(
                    values,
                    upstream,
                    basis_values,
                    sparse_activity=connectivity.sparse_activity,
                    n_post=connectivity.n_post,
                    n_edges=connectivity.n_edges,
                    resource_name=connectivity.resource_name,
                )
                weight_grad = tf.cast(weight_grad, master_weights.dtype)
            else:
                weight_grad = None
            # `initial` is accumulated into the output, so its gradient is the
            # upstream gradient unchanged; an absent accumulator keeps None.
            initial_grad = None if initial is None else upstream
            return activity_grad, weight_grad, None, initial_grad

        return currents, grad

    return fused(
        activity, weights, basis,
        empty_like_currents(activity) if initial is None else initial,
    )
