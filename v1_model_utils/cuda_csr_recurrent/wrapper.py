"""Python interface for the fused recurrent-current CUDA operator."""

import contextlib
from dataclasses import dataclass
from pathlib import Path
import threading

import numpy as np
import tensorflow as tf

from v1_model_utils import csr_order
from .canonical_projection import canonical_pairs
from v1_model_utils.cuda_operator_cache import ensure_artifact
from v1_model_utils.cuda_csr_recurrent.build import build_flags_for, DIRECT_CSR
from v1_model_utils.cuda_csr_resources import (
    initialize_resource,
    load_ops as load_resource_ops,
    resource_mode_enabled,
)


_OPS = None
# Per thread, because MirroredStrategy traces every replica in its own thread:
# a replica's backward must only ever see its own accumulator.
_WEIGHT_GRADIENT_ACCUMULATOR = threading.local()


def empty_metadata(dtype):
    """An empty metadata field, placed on the GPU like the non-empty ones.

    TensorFlow has no GPU kernel that fills a uint32 tensor, so
    ``tf.zeros((0,), tf.uint32)`` is created on the host. A GPU op that reads
    such a captured host tensor inside the timestep loop receives it through a
    host-to-device copy every step, and even an empty copy completes only after
    the GPU has run everything queued before it. The op, and all that depends
    on it, is then launched only once the GPU has caught up, which held the
    host in lockstep with the GPU through the whole backward loop. A constant
    built from NumPy is placed on the default device, like every non-empty
    field.
    """
    return tf.constant(np.zeros(0, tf.as_dtype(dtype).as_numpy_dtype))


def _edge_index_tensor(order):
    """Upload the CSR-to-original permutation only when a kernel reads it.

    With :data:`DIRECT_CSR` the kernels index weights by CSR position, so
    ``V1_EDGE_INDEX`` expands to the position itself and no kernel ever
    dereferences this input - it is four dead bytes per edge on the device.
    The resource operator validates its length against ``post_ids``, so that
    path keeps the real permutation.
    """
    if DIRECT_CSR and not resource_mode_enabled():
        return empty_metadata(tf.uint32)
    return tf.constant(order, tf.uint32)


@dataclass(frozen=True)
class CsrConnectivity:
    """Presynaptic CSR metadata with an original-edge permutation.

    ``edge_ids`` maps a CSR position to the edge's index in the caller's
    original order; it is empty when :data:`DIRECT_CSR` makes the kernels index
    weights by CSR position instead. ``edge_order`` is the same permutation
    kept on the host, so weights can be moved between the two orders without a
    device round trip.
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
    # Whether some row sends two edges to one target; see csr_order.repeats_targets.
    repeats_targets: bool = True
    # One more than the largest synapse type: the basis rows the kernels read.
    n_synapse_types: int = 0


def _compact_pairs(post_ids, synapse_types, basis=None):
    """Describe each distinct (postsynaptic neuron, synapse type) pair once.

    The backward pass projects the upstream gradient onto the synaptic basis per
    (post, type) combination. Edges reuse those combinations heavily, so
    projecting the distinct pairs once and indexing them per edge removes almost
    all of the redundant projection work.
    """
    unique_codes, pair_ids = csr_order.compact_pairs(post_ids, synapse_types)
    metadata = {
        "pair_ids": pair_ids.astype(np.uint32),
        "pair_posts": (unique_codes >> 8).astype(np.uint32),
        "pair_types": (unique_codes & 0xFF).astype(np.uint8),
        "n_pairs": int(unique_codes.size),
    }
    if basis is not None:
        metadata = canonical_pairs(
            metadata["pair_posts"], metadata["pair_types"], metadata["pair_ids"], basis
        )
    return {key: value if key == "n_pairs" else tf.constant(value)
            for key, value in metadata.items()}


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
                directory / "event_weight_grad.cuh",
            ),
            build_module="v1_model_utils.cuda_csr_recurrent.build",
            build_flags=build_flags_for,
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
    indices, synapse_types, n_pre, n_post, weights_csr_ordered=False, basis=None
):
    """Build compact pre-CSR metadata while preserving external edge order.

    Set ``weights_csr_ordered`` when the caller's edges are already in this
    operator's CSR order; the derived permutation is then asserted to be the
    identity, so a mismatch surfaces here rather than as silently mispaired
    weights. Optional ``basis`` must be the immutable FP32 basis used by this
    connectivity; it shares only bitwise-identical projection rows. Omitting it
    keeps the baseline metadata and supports callers with mutable bases.
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

    keys = (indices[:, 1], indices[:, 0], synapse_types)
    if weights_csr_ordered:
        if not csr_order.is_identity_order(keys):
            raise ValueError(
                "edges were declared to be in CSR order but the derived "
                "permutation is not the identity; the CSR sort key and the "
                "caller's edge order have diverged"
            )
        # The permutation is the identity, so every gather below it is too.
        order = np.arange(indices.shape[0], dtype=np.uint32)
        ordered_posts, ordered_types = indices[:, 0], synapse_types
    else:
        order = csr_order.edge_order(keys)
        ordered_posts, ordered_types = indices[order, 0], synapse_types[order]
    counts = np.bincount(indices[:, 1], minlength=n_pre).astype(np.uint64)
    row_splits = np.empty(n_pre + 1, dtype=np.uint32)
    row_splits[0] = 0
    row_splits[1:] = np.cumsum(counts, dtype=np.uint64).astype(np.uint32)
    nonempty_rows = np.flatnonzero(counts).astype(np.uint32)
    pairs = _compact_pairs(ordered_posts, ordered_types, basis=basis)
    connectivity = CsrConnectivity(
        post_ids=tf.constant(ordered_posts, tf.uint32),
        synapse_types=tf.constant(ordered_types, tf.uint8),
        row_splits=tf.constant(row_splits, tf.uint32),
        edge_ids=_edge_index_tensor(order),
        nonempty_rows=tf.constant(nonempty_rows, tf.uint32),
        n_pre=int(n_pre),
        n_post=int(n_post),
        n_edges=int(indices.shape[0]),
        edge_order=order,
        weights_csr_ordered=bool(weights_csr_ordered),
        repeats_targets=csr_order.repeats_targets(ordered_posts, row_splits),
        n_synapse_types=int(synapse_types.max()) + 1 if synapse_types.size else 0,
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


@contextlib.contextmanager
def accumulate_recurrent_weight_gradient(handle):
    """Add recurrent weight gradients built in this block into ``handle``.

    ``handle`` is the resource handle of an FP32 ``[n_edges]`` variable, as a
    tensor of the graph that builds the backward. Gradient functions traced in
    the block add their weight gradient to it in place and return ``None`` for
    the weights, so the loop gradient that encloses them never sums a dense
    per-step weight gradient. The caller zeroes the variable first and reads
    the total afterwards.
    """
    previous = getattr(_WEIGHT_GRADIENT_ACCUMULATOR, "handle", None)
    _WEIGHT_GRADIENT_ACCUMULATOR.handle = handle
    try:
        yield
    finally:
        _WEIGHT_GRADIENT_ACCUMULATOR.handle = previous


def _weight_gradient_accumulator():
    return getattr(_WEIGHT_GRADIENT_ACCUMULATOR, "handle", None)


# The most spike tensors one operator call takes (kMaxSpikeSlots in
# event_weight_grad.cuh). A longer spike history is concatenated first.
MAX_SPIKE_SLOTS = 8


def spike_slots(spikes):
    """The presynaptic matrix as the operators take it: a tuple of equal
    [batch, width] tensors side by side (the recurrent spike history, one per
    delay slot, newest first), or one tensor."""
    if not isinstance(spikes, (tuple, list)):
        return (spikes,)
    if len(spikes) > MAX_SPIKE_SLOTS:
        return (tf.concat(spikes, axis=1),)
    return tuple(spikes)


def empty_like_currents(values):
    """The sentinel `initial` value meaning "start from zero".

    Currents take the dtype of the spikes (or activity) that drive them.
    """
    return tf.zeros((0, 0), values.dtype)


def spike_queue(slot):
    """One [batch, width] spike slot's queue record (uint32), the form in which
    calculate_recurrent_csr_currents carries a history's active entries."""
    return _load_ops().v1_spike_queue(slot)


def spike_activity_masks(slot):
    """One [batch, width] spike slot's activity masks (uint32 [width, ceil(batch / 32)]).

    A backward that walks the spike history in reverse can carry them: at the
    step whose slots are (z[t-1], z[t-2], z[t-3]), the first two are the
    previous (later) step's slots 1 and 2, so only z[t-3] needs new masks. Pass
    the step's masks, in slot order, as ``activity_masks`` of
    :func:`recurrent_backward`; the gradients are bitwise the same, and the
    weight gradient then reads the masks instead of sweeping every slot.
    """
    return _load_ops().v1_spike_activity_masks(slot)


def overwrite_entries(base, indices, values):
    """``base`` with its flat entries ``indices`` (int64) replaced by ``values``
    (equal values for repeated indices), like ``tf.tensor_scatter_nd_update``
    (which has no float16 GPU kernel). The op writes into ``base``'s buffer when
    it holds the only reference, so a loop that rewrites a few entries of its
    previous result every step copies nothing else."""
    return _load_ops().v1_overwrite_entries(base, indices, values)


def scaled_abs_max_bound(bound, scale):
    """The abs-max bound (float bits of the largest finite magnitude, as the GLIF
    backward emits it) of ``values * scale``, from that of ``values``.

    The product is formed by the same compute-dtype multiply as the scaled
    values, and its rounding is monotone and symmetric, so the largest finite
    ``|values * scale|`` is ``|bound * scale|`` whenever that is finite. If it
    overflows, the recurrent backward measures the bound itself.
    """
    magnitude = tf.cast(tf.bitcast(bound, tf.float32), scale.dtype) * scale
    return tf.bitcast(tf.cast(tf.abs(magnitude), tf.float32), tf.uint32)


# Whether a StepGradientLink is taken up (read when the gradient is built). Off,
# the recurrent backward measures its bound and sweeps its slots itself, and
# TensorFlow adds each operator's gradients: the reference of the bitwise check
# tests/test_step_gradient_link.py.
LINK_STEP_GRADIENTS = True


class StepGradientLink:
    """What a timestep's GLIF backward hands to the recurrent backward of the
    same timestep when TensorFlow differentiates the step (the segmented
    runner's replay, plain autodiff), as the recorded adjoint passes them
    directly: the abs-max bound of the currents' gradient and, per spike slot,
    the slot's gradient through the GLIF op, which the recurrent operator adds
    as it stores its own (``spike_grad_addends``) instead of TensorFlow adding
    the two afterwards. The gradients are bitwise the same; the operator skips
    its abs-max pass and the adds their passes. A link may also carry the
    slots' activity masks (``spike_activity_masks`` bit-cast to float32,
    carried by the loop from step to step), which spare the operator its
    sweep of every slot.

    V1Column.call makes one per step whose GLIF input currents are the
    recurrent operator's currents (reshaped, other sources accumulated onto
    them with the gradient passed through, times ``scale`` when given) and no
    other gradient reaches them. The recurrent wrapper ``accept``s it at
    forward time when its spike inputs are the link's slots themselves; only
    then does the GLIF gradient ``hand_off`` (and return no slot gradients).
    TensorFlow builds the GLIF gradient before the recurrent one (it needs the
    currents' gradient), and builds the recurrent one whenever a slot gradient
    is needed (the operator consumes the slots), so ``take`` there receives
    what this step handed off.
    """

    def __init__(self, slots, scale=None, masks=()):
        self.slots = tuple(slots)
        self.scale = scale
        self.masks = tuple(masks)
        self.accepted = False
        self._handed = None

    def accept(self, spike_values):
        """Whether the recurrent operator reading ``spike_values`` takes the
        hand-off (they must be this link's slots, in order)."""
        self.accepted = self._same_slots(spike_values)
        return self.accepted

    def hands_off(self, history):
        """Whether the GLIF gradient of the op whose spike history is
        ``history`` hands off (and returns no slot gradients)."""
        return LINK_STEP_GRADIENTS and self.accepted and self._same_slots(history)

    def _same_slots(self, values):
        return len(values) == len(self.slots) and all(
            value is slot for value, slot in zip(values, self.slots)
        )

    def hand_off(self, bound, addends):
        """The GLIF gradient's abs-max bound of its currents' gradient and one
        addend per slot (None for a slot it has no gradient for)."""
        self._handed = (bound, addends)

    def take(self):
        """The recurrent_backward options: the masks, and what was handed off
        (once)."""
        if not LINK_STEP_GRADIENTS:
            return {}
        options = (
            dict(activity_masks=tuple(tf.bitcast(mask, tf.uint32) for mask in self.masks))
            if self.masks else {}
        )
        if self._handed is None:
            return options
        bound, addends = self._handed
        self._handed = None
        if self.scale is not None:
            bound = scaled_abs_max_bound(bound, self.scale)
        return dict(
            options,
            abs_max_bound=bound,
            spike_grad_addends=tuple(
                tf.zeros([0], slot.dtype) if addend is None else addend
                for addend, slot in zip(addends, self.slots)
            ),
        )


def recurrent_backward(
    connectivity, spike_values, current_grad, weight_values, basis_values, dampening_value,
    post_ids, synapse_types, row_splits, edge_ids, nonempty_rows, pair_ids, pair_posts,
    pair_types, activity_masks=(), abs_max_bound=None, spike_grad_addends=(),
):
    """The recurrent spike gradients (one per slot) and weight gradient (None
    when an accumulator takes it, see accumulate_recurrent_weight_gradient).
    ``activity_masks`` is empty or one :func:`spike_activity_masks` per slot.
    ``abs_max_bound`` is None or the [1] uint32 float bits of the largest finite
    ``|current_grad|`` (the GLIF backward's ``emit_abs_max`` output for the
    same tensor), which saves the operator that pass; the results are the same.
    Non-finite bits make the operator measure the bound after all (a rescaled
    bound that overflowed).
    ``spike_grad_addends`` is empty or one tensor per slot, of the slot's shape
    or empty: each slot's gradient is returned plus its addend, as a following
    add of the two would give it, without that add's pass."""
    ops = _load_ops()
    inputs = (
        spike_values,
        current_grad,
        weight_values,
        post_ids,
        synapse_types,
        row_splits,
        edge_ids,
        nonempty_rows,
        basis_values,
        tf.cast(dampening_value, spike_values[0].dtype),
        pair_ids,
        pair_posts,
        pair_types,
    )
    sizes = {"n_post": connectivity.n_post, "n_edges": connectivity.n_edges}
    bound = [] if abs_max_bound is None else [abs_max_bound]
    accumulator = _weight_gradient_accumulator()
    if accumulator is None:
        spike_grads, weight_grad = ops.v1_csr_backward_pair_projected(
            *inputs, list(activity_masks), bound, list(spike_grad_addends), **sizes
        )
        return spike_grads, tf.cast(weight_grad, weight_values.dtype)
    spike_grads = ops.v1_csr_backward_pair_projected_accumulate(
        *inputs, accumulator, list(activity_masks), bound, list(spike_grad_addends),
        **sizes
    )
    return spike_grads, None


def calculate_recurrent_csr_currents(
    spikes, weights, basis, dampening, connectivity, initial=None, queues=None, link=None
):
    """Calculate currents plus spike and weight gradients.

    ``spikes`` is the [batch, n_pre] presynaptic matrix, or the recurrent spike
    history as a tuple of its delay slots (newest first), whose concatenation
    is that matrix; the slots then each get their own gradient, and the history
    uses separate tensors for up to eight slots. Longer histories use the
    generic concatenated path. ``initial`` accumulates this source's currents on
    top of another source's output, which avoids materializing a separate
    tensor and adding it later. Its gradient is the upstream gradient
    unchanged. The kernels take the
    synaptic basis in FP32; pass the unrounded values, since casting an
    already-rounded FP16 basis back up recovers nothing.

    ``queues`` carries a spike history's per-slot queue records (see
    spike_queue) from step to step, so that only the newest slot is swept for
    its active entries: the records of slots 1 to D - 1, which are the
    previous step's records of slots 0 to D - 2. The result is then
    ``(currents, record)``, with ``record`` the newest slot's, for the next
    step's history. The currents are the same as without ``queues``.

    ``link`` is the timestep's StepGradientLink (V1Column.call), taken up by
    the backward when the spike slots are its own.
    """
    require_csr_ordered_weights(connectivity, "recurrent connectivity")
    basis = tf.cast(basis, tf.float32)
    # The kernels read basis[type] unchecked for every edge.
    if basis.shape[0] is not None and basis.shape[0] < connectivity.n_synapse_types:
        raise ValueError(
            f"basis has {basis.shape[0]} rows but the connectivity uses "
            f"{connectivity.n_synapse_types} synapse types"
        )
    history = tuple(spikes) if isinstance(spikes, (tuple, list)) else (spikes,)
    if queues is not None and (len(history) < 2 or len(queues) != len(history) - 1):
        raise ValueError("queues must hold one record per spike slot but the newest")
    if len(history) > MAX_SPIKE_SLOTS and queues is not None:
        # The raw ops carry at most eight pointers. Preserve the public queue
        # contract while the generic path rebuilds the flattened activity queue.
        currents = calculate_recurrent_csr_currents(
            tf.concat(history, axis=1), weights, basis, dampening, connectivity,
            initial=initial,
        )
        return currents, spike_queue(history[0])
    slots = spike_slots(spikes)
    queue_args = () if queues is None else tuple(queues)
    if connectivity.resource_name is not None:
        return _calculate_resource_currents(
            slots, weights, basis, dampening, connectivity, initial=initial,
            queues=queues, link=link,
        )
    ops = _load_ops()

    @tf.custom_gradient
    def fused(
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
        *rest,
    ):
        spike_values, queue_values = rest[:len(slots)], rest[len(slots):]
        linked = link is not None and link.accept(spike_values)
        currents, record = ops.v1_csr_forward(
            spike_values,
            weight_values,
            post_ids,
            synapse_types,
            row_splits,
            edge_ids,
            basis_values,
            initial_values,
            queue_values,
            n_post=connectivity.n_post,
            aggregate_runs=connectivity.repeats_targets,
        )

        def grad(current_grad, *_record_grad):
            spike_grads, weight_grad = recurrent_backward(
                connectivity, spike_values, current_grad, weight_values, basis_values,
                dampening_value, post_ids, synapse_types, row_splits, edge_ids,
                nonempty_rows, pair_ids, pair_posts, pair_types,
                **(link.take() if linked else {}),
            )
            # `initial` enters the output additively, so the upstream gradient
            # passes straight through. With no accumulator the input is an
            # empty sentinel, and its gradient has to stay unset rather than
            # take the output shape.
            initial_grad = None if initial is None else current_grad
            return ((weight_grad,) + (None,) * 10 + (initial_grad, *spike_grads)
                    + (None,) * len(queue_values))

        return (currents if queues is None else (currents, record)), grad

    return fused(
        weights,
        basis,
        connectivity.post_ids,
        connectivity.synapse_types,
        connectivity.row_splits,
        connectivity.edge_ids,
        connectivity.nonempty_rows,
        tf.cast(dampening, slots[0].dtype),
        connectivity.pair_ids,
        connectivity.pair_posts,
        connectivity.pair_types,
        empty_like_currents(slots[0]) if initial is None else initial,
        *slots,
        *queue_args,
    )


def resource_recurrent_backward(
    connectivity, spike_values, current_grad, weight_values, basis_values, dampening_value,
    activity_masks=(), abs_max_bound=None, spike_grad_addends=(),
):
    """recurrent_backward with the connectivity read from its GPU-local resource
    (same optional inputs, same results)."""
    ops = load_resource_ops()
    inputs = (
        spike_values,
        current_grad,
        weight_values,
        basis_values,
        tf.cast(dampening_value, spike_values[0].dtype),
    )
    options = (
        list(activity_masks),
        [] if abs_max_bound is None else [abs_max_bound],
        list(spike_grad_addends),
    )
    attributes = {
        "n_post": connectivity.n_post,
        "n_edges": connectivity.n_edges,
        "resource_name": connectivity.resource_name,
    }
    accumulator = _weight_gradient_accumulator()
    if accumulator is None:
        spike_grads, weight_grad = ops.v1_csr_backward_resource(
            *inputs, *options, **attributes
        )
        return spike_grads, tf.cast(weight_grad, weight_values.dtype)
    spike_grads = ops.v1_csr_backward_accumulate_resource(
        *inputs, accumulator, *options, **attributes
    )
    return spike_grads, None


def _calculate_resource_currents(
    slots, weights, basis, dampening, connectivity, initial=None, queues=None, link=None
):
    ops = load_resource_ops()
    queue_args = () if queues is None else tuple(queues)

    @tf.custom_gradient
    def fused(
        weight_values, basis_values, dampening_value, initial_values, *rest
    ):
        spike_values, queue_values = rest[:len(slots)], rest[len(slots):]
        linked = link is not None and link.accept(spike_values)
        currents, record = ops.v1_csr_forward_resource(
            spike_values,
            weight_values,
            basis_values,
            initial_values,
            queue_values,
            n_post=connectivity.n_post,
            resource_name=connectivity.resource_name,
            aggregate_runs=connectivity.repeats_targets,
        )

        def grad(current_grad, *_record_grad):
            spike_grads, weight_grad = resource_recurrent_backward(
                connectivity, spike_values, current_grad, weight_values, basis_values,
                dampening_value, **(link.take() if linked else {}),
            )
            # `initial` enters the output additively, so the upstream gradient
            # passes straight through. With no accumulator the input is an
            # empty sentinel and its gradient has to stay unset.
            initial_grad = None if initial is None else current_grad
            return ((weight_grad, None, None, initial_grad, *spike_grads)
                    + (None,) * len(queue_values))

        return (currents if queues is None else (currents, record)), grad

    return fused(
        weights,
        basis,
        tf.cast(dampening, slots[0].dtype),
        empty_like_currents(slots[0]) if initial is None else initial,
        *slots,
        *queue_args,
    )
