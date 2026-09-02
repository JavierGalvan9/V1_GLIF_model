"""Spatial neuron layout for the fused CUDA CSR synaptic-current kernels.

Renumbering neurons along a space-filling curve makes the postsynaptic indices
inside every CSR row numerically close, so the fused current kernels touch far
fewer cache lines per warp. The kernels themselves are unchanged: the layout is
purely a relabelling of the loaded network.

The permutation lives in memory only. ``load_sparse`` output, checkpoints and
every other on-disk artefact stay in the canonical order, and
:func:`to_canonical` / :func:`to_runtime` translate at those boundaries.
"""

from dataclasses import dataclass

import numpy as np


CANONICAL = "canonical"
MORTON = "morton"
LAYOUTS = (CANONICAL, MORTON)

# 14 planar bits per axis interleaved as a 2-D Morton code, with a coarse
# 4-bit cortical-depth key above it. Measured as the best of the sweep: a
# 3-bit depth key was a better cache proxy but lost a paired full-kernel
# comparison, and full 3-D Morton was slower than 2-D plus depth.
_PLANAR_BITS = 14
_DEPTH_BITS = 4


def _quantize(values, levels):
    """Map coordinates onto ``0..levels`` preserving their spatial order.

    Quantization is done in float64 so the layout does not depend on the dtype
    the coordinates happen to be stored in. Evaluating it in the stored float32
    instead moves 114 of 203,816 neurons across a bin boundary, which changes
    the permutation but not the locality it buys.
    """
    values = np.asarray(values, np.float64)
    span = np.ptp(values) + 1e-12
    return np.floor((values - values.min()) / span * levels).astype(np.uint32)


def _interleave(values):
    """Spread 14 bits so ``value`` occupies every other bit position."""
    values = values & 0x3FFF
    values = (values | values << 8) & 0x00FF00FF
    values = (values | values << 4) & 0x0F0F0F0F
    values = (values | values << 2) & 0x33333333
    return (values | values << 1) & 0x55555555


def morton_key(x, y, z):
    """Return the anisotropic spatial sort key for every neuron."""
    planar_levels = (1 << _PLANAR_BITS) - 1
    depth_levels = (1 << _DEPTH_BITS) - 1
    return (
        _interleave(_quantize(x, planar_levels))
        | _interleave(_quantize(z, planar_levels)) << 1
        | _quantize(y, depth_levels) << (2 * _PLANAR_BITS)
    )


@dataclass(frozen=True)
class NeuronLayout:
    """Bidirectional map between canonical and runtime neuron numbering."""

    mode: str
    new_to_old: np.ndarray
    old_to_new: np.ndarray

    @classmethod
    def identity(cls, n_nodes):
        order = np.arange(int(n_nodes), dtype=np.uint32)
        return cls(mode=CANONICAL, new_to_old=order, old_to_new=order)

    @classmethod
    def morton(cls, network):
        key = morton_key(network["x"], network["y"], network["z"])
        new_to_old = np.argsort(key, kind="stable").astype(np.uint32)
        old_to_new = np.empty_like(new_to_old)
        old_to_new[new_to_old] = np.arange(new_to_old.size, dtype=np.uint32)
        return cls(mode=MORTON, new_to_old=new_to_old, old_to_new=old_to_new)

    @property
    def n_nodes(self):
        return int(self.new_to_old.size)

    @property
    def is_identity(self):
        return self.mode == CANONICAL

    def to_runtime(self, values, axis=0):
        """Reorder a canonical neuron-aligned array into runtime order."""
        return self._take(values, self.new_to_old, axis)

    def to_canonical(self, values, axis=0):
        """Reorder a runtime neuron-aligned array back into canonical order."""
        return self._take(values, self.old_to_new, axis)

    def relabel(self, ids):
        """Translate canonical neuron identifiers into runtime identifiers.

        Negative entries are sentinels (``load_sparse`` marks unmapped BMTK
        nodes with ``-1``) and are preserved.
        """
        return self._map_ids(ids, self.old_to_new)

    def relabel_to_canonical(self, ids):
        """Translate runtime neuron identifiers back into canonical ones."""
        return self._map_ids(ids, self.new_to_old)

    def relabel_delayed(self, ids, to_canonical=False):
        """Relabel delay-expanded presynaptic ids of the form ``pre + n*(d-1)``."""
        if self.is_identity:
            return ids
        ids = np.asarray(ids)
        mapping = self.new_to_old if to_canonical else self.old_to_new
        neuron = np.mod(ids, self.n_nodes)
        delay_block = ids - neuron
        return delay_block + mapping[neuron].astype(ids.dtype, copy=False)

    def _map_ids(self, ids, mapping):
        if self.is_identity:
            return ids
        ids = np.asarray(ids)
        relabelled = ids.copy()
        mapped = ids >= 0
        relabelled[mapped] = mapping[ids[mapped]].astype(ids.dtype, copy=False)
        return relabelled

    def _take(self, values, order, axis):
        if self.is_identity:
            return values
        values = np.asarray(values)
        if values.shape[axis] != self.n_nodes:
            raise ValueError(
                f"axis {axis} has length {values.shape[axis]}, "
                f"expected {self.n_nodes} neurons"
            )
        return np.take(values, order, axis=axis)


def build_layout(network, mode=CANONICAL):
    """Return the neuron layout selected by ``mode`` for ``network``."""
    if mode not in LAYOUTS:
        raise ValueError(f"unknown neuron layout {mode!r}; expected one of {LAYOUTS}")
    if mode == CANONICAL:
        return NeuronLayout.identity(network["n_nodes"])
    return NeuronLayout.morton(network)


# Length-``n_nodes`` arrays that are *not* indexed by runtime neuron id, so
# they must never be reordered by the generic sweep in :func:`apply_layout`.
# ``bmtk_id_to_tf_id`` is indexed by BMTK node id and only its values move.
_NOT_NEURON_INDEXED = frozenset(("bmtk_id_to_tf_id", "synapses", "node_params"))


LGN_ORIGINAL = "original"
LGN_RETINOTOPIC = "retinotopic"
LGN_ROW_ORDERS = (LGN_ORIGINAL, LGN_RETINOTOPIC)


@dataclass(frozen=True)
class LgnRowOrder:
    """Bidirectional map between canonical and runtime LGN row numbering.

    The external forward kernel launches one block per active presynaptic row
    in ascending row id and scatters into ``currents[post]`` with atomics, so
    the row numbering decides which postsynaptic territory consecutive blocks
    touch. LGN->V1 is retinotopic, so ranking rows by the V1 territory they
    drive makes neighbouring blocks share that territory.

    This is orthogonal to :class:`NeuronLayout`: one renumbers the
    postsynaptic side, the other the presynaptic side. Measured value is
    small - about 2% of the LGN forward kernel, and only once the postsynaptic
    currents array outgrows L2 (see
    ``Benchmarks_metrics/batch64_layout_20260902/REPORT.md``) - so
    :data:`LGN_ORIGINAL` remains the default. It is implemented because a
    tensor-row WMMA kernel would make the same ordering pay through fragment
    reuse rather than cache capacity, which does not depend on the batch.
    """

    mode: str
    new_to_old: np.ndarray
    old_to_new: np.ndarray

    @classmethod
    def identity(cls, n_rows):
        order = np.arange(int(n_rows), dtype=np.uint32)
        return cls(mode=LGN_ORIGINAL, new_to_old=order, old_to_new=order)

    @classmethod
    def retinotopic(cls, lgn_input):
        """Rank rows by the mean postsynaptic index they drive.

        LGN neurons carry no coordinates in ``lgn_input``, but none are needed:
        the retinotopic position of a row is implied by its targets. Every row
        in the 17,400-input model has at least seven edges, so no row falls
        back on a default position.
        """
        indices = np.asarray(lgn_input["indices"])
        n_rows = int(lgn_input["n_inputs"])
        pre = indices[:, 1].astype(np.int64)
        post = indices[:, 0].astype(np.float64)
        total = np.bincount(pre, weights=post, minlength=n_rows)
        counts = np.bincount(pre, minlength=n_rows)
        # A row with no edges keeps position zero; it drives nothing, so where
        # it lands cannot affect the kernel.
        centroid = np.divide(
            total, counts, out=np.zeros_like(total), where=counts > 0
        )
        new_to_old = np.argsort(centroid, kind="stable").astype(np.uint32)
        old_to_new = np.empty_like(new_to_old)
        old_to_new[new_to_old] = np.arange(new_to_old.size, dtype=np.uint32)
        return cls(
            mode=LGN_RETINOTOPIC, new_to_old=new_to_old, old_to_new=old_to_new
        )

    @property
    def n_rows(self):
        return int(self.new_to_old.size)

    @property
    def is_identity(self):
        return self.mode == LGN_ORIGINAL

    def relabel(self, ids):
        """Translate canonical LGN row ids into runtime ids."""
        return self._map_ids(ids, self.old_to_new)

    def relabel_to_canonical(self, ids):
        """Translate runtime LGN row ids back into canonical ids."""
        return self._map_ids(ids, self.new_to_old)

    def _map_ids(self, ids, mapping):
        if self.is_identity:
            return ids
        ids = np.asarray(ids)
        relabelled = ids.copy()
        mapped = ids >= 0
        relabelled[mapped] = mapping[ids[mapped]].astype(ids.dtype, copy=False)
        return relabelled


def build_lgn_row_order(lgn_input, mode=LGN_ORIGINAL):
    """Return the LGN row order selected by ``mode``."""
    if mode not in LGN_ROW_ORDERS:
        raise ValueError(
            f"unknown LGN row order {mode!r}; expected one of {LGN_ROW_ORDERS}"
        )
    if mode == LGN_ORIGINAL:
        return LgnRowOrder.identity(lgn_input["n_inputs"])
    return LgnRowOrder.retinotopic(lgn_input)


def apply_lgn_row_order(order, lgn_input):
    """Relabel LGN presynaptic ids into ``order``'s runtime numbering.

    Only column one of ``indices`` moves. Apply this *before*
    :func:`apply_csr_edge_order` so the resulting CSR permutation carries the
    new row order, which is what lets the forward and weight-backward kernels
    and the checkpoint translation all stay consistent without touching any
    kernel code.

    The spike stream still arrives in canonical row order, so whoever feeds the
    model must gather it into runtime order; :class:`models.V1Column` does this
    with ``new_to_old``.
    """
    if order.is_identity:
        return lgn_input
    lgn_input = dict(lgn_input)
    indices = np.array(lgn_input["indices"])
    if indices[:, 1].max() >= order.n_rows:
        raise ValueError(
            f"LGN presynaptic id {int(indices[:, 1].max())} exceeds the "
            f"{order.n_rows} rows the order was built for"
        )
    indices[:, 1] = order.relabel(indices[:, 1])
    lgn_input["indices"] = indices
    return lgn_input

def resolve_max_delay(network, lgn_input, bkg_input, max_delay=0, dt=1.0):
    """Mirror ``V1Column``'s maximum-delay choice in simulation steps."""
    delays = [np.asarray(network["synapses"]["delays"])]
    for source in (lgn_input, bkg_input):
        if "delays" in source:
            delays.append(np.asarray(source["delays"]))
    data_max_delay = int(np.ceil(np.max(np.concatenate(delays)) / dt))
    if max_delay > 0:
        return min(data_max_delay, int(max_delay))
    return data_max_delay


def recurrent_csr_order(network, max_delay, dt=1.0):
    """Return the CSR edge order the recurrent operator will derive.

    This reproduces ``cuda_csr_recurrent.build_csr_connectivity``, including the
    delay expansion ``V1Column`` applies to presynaptic indices. The operator
    asserts that its own order is the identity once the network is reordered, so
    any drift between the two fails loudly instead of mispairing weights.
    """
    synapses = network["synapses"]
    indices = np.asarray(synapses["indices"])
    delay_steps = np.round(
        np.clip(np.asarray(synapses["delays"]), dt, max_delay) / dt
    ).astype(np.int64)
    pre = indices[:, 1].astype(np.int64) + int(network["n_nodes"]) * (delay_steps - 1)
    types = np.asarray(synapses["syn_ids"])
    original = np.arange(indices.shape[0], dtype=np.uint32)
    return np.lexsort((original, types, indices[:, 0], pre)).astype(np.uint32)


def external_csr_order(source):
    """Return the CSR edge order the external operator will derive.

    Note the key omits the synapse type, matching
    ``cuda_csr_external.build_csr_connectivity``.
    """
    indices = np.asarray(source["indices"])
    original = np.arange(indices.shape[0], dtype=np.uint32)
    return np.lexsort((original, indices[:, 0], indices[:, 1])).astype(np.uint32)


# Fields that are not per-edge even though their length can coincide with the
# edge count on small networks: ``dense_shape`` is a two-element shape tuple.
_NOT_EDGE_INDEXED = frozenset(("dense_shape",))


def _reorder_edges(source, order, n_edges):
    """Reorder every edge-aligned field of one population."""
    reordered = dict(source)
    for name, value in tuple(source.items()):
        if name in _NOT_EDGE_INDEXED or np.isscalar(value):
            continue
        array = np.asarray(value)
        if array.ndim >= 1 and array.shape[0] == n_edges:
            reordered[name] = array[order]
    return reordered


def apply_csr_edge_order(network, lgn_input, bkg_input, max_delay=0, dt=1.0):
    """Reorder every population's edges into the CSR order of its operator.

    With CSR-ordered edges the kernels index weights by CSR position, so the
    random ``edge_ids`` gather leaves every inner loop. Because the whole
    network is reordered, weights, sign masks, per-edge regularizer references
    and connection-type ids are all derived consistently.

    Returns the reordered triple plus the permutations, which the checkpoint
    layer needs to write weights back in the original order.
    """
    steps = resolve_max_delay(network, lgn_input, bkg_input, max_delay, dt)
    recurrent_order = recurrent_csr_order(network, steps, dt)
    lgn_order = external_csr_order(lgn_input)
    bkg_order = external_csr_order(bkg_input)

    network = dict(network)
    synapses = np.asarray(network["synapses"]["indices"]).shape[0]
    network["synapses"] = _reorder_edges(
        network["synapses"], recurrent_order, synapses
    )
    orders = {
        "recurrent": recurrent_order,
        "lgn": lgn_order,
        "bkg": bkg_order,
    }
    return (
        network,
        _reorder_edges(
            lgn_input, lgn_order, np.asarray(lgn_input["indices"]).shape[0]
        ),
        _reorder_edges(
            bkg_input, bkg_order, np.asarray(bkg_input["indices"]).shape[0]
        ),
        orders,
    )


def to_original_edges(values, order):
    """Scatter CSR-order edge values back into the network's original order."""
    values = np.asarray(values)
    restored = np.empty_like(values)
    restored[order] = values
    return restored


def to_csr_edges(values, order):
    """Gather original-order edge values into CSR order."""
    return np.asarray(values)[order]


def tracked_to_canonical(layout, values, tracked_ids=None, axis=-1):
    """Return a runtime-order per-neuron array in canonical neuron order.

    ``tracked_ids`` gives the runtime neuron ids of the entries along ``axis``
    when only a subset is recorded. The entries are then ordered by canonical id,
    which is the order the same subset has in a canonical run, so saved traces
    stay comparable across layouts.
    """
    if layout.is_identity:
        return values
    values = np.asarray(values)
    if tracked_ids is None:
        return np.take(values, layout.old_to_new, axis=axis)
    canonical_ids = layout.new_to_old[np.asarray(tracked_ids)]
    return np.take(values, np.argsort(canonical_ids), axis=axis)


def translate_neuron_state(layout, payload, to_runtime):
    """Translate the per-neuron arrays of a persisted state payload.

    ``train_end_data.pkl`` stores neuron-aligned running statistics (firing-rate
    EMAs, rolling orientation-selectivity accumulators) that are written and
    read across runs. They follow the same canonical-on-disk rule as
    checkpoints, so they are translated on the way in and out.
    """
    if layout.is_identity:
        return payload
    reorder = layout.to_runtime if to_runtime else layout.to_canonical
    if isinstance(payload, dict):
        return {
            key: translate_neuron_state(layout, value, to_runtime)
            for key, value in payload.items()
        }
    array = np.asarray(payload) if not np.isscalar(payload) else None
    if array is not None and array.shape == (layout.n_nodes,) and (
        np.issubdtype(array.dtype, np.floating)
    ):
        return reorder(array)
    return payload


def _relabel_post(source, layout):
    """Relabel the postsynaptic column of an external input population."""
    indices = np.array(source["indices"])
    indices[:, 0] = layout.relabel(indices[:, 0])
    return {**source, "indices": indices}


def apply_layout(layout, network, lgn_input, bkg_input):
    """Return the loaded network triple expressed in runtime neuron order.

    Every neuron-aligned array is reordered and every stored neuron identifier
    is relabelled, so the rest of the pipeline never sees canonical numbering.
    Edge ordering is untouched: the CSR builders derive their own edge order and
    retain a permutation to the trainable weights.
    """
    if layout.is_identity:
        return network, lgn_input, bkg_input

    n_nodes = layout.n_nodes
    if int(network["n_nodes"]) != n_nodes:
        raise ValueError("layout was built for a different network size")

    network = dict(network)
    for name, value in tuple(network.items()):
        if name in _NOT_NEURON_INDEXED or np.isscalar(value):
            continue
        array = np.asarray(value)
        if array.ndim >= 1 and array.shape[0] == n_nodes:
            network[name] = layout.to_runtime(array)
    # ``tf_id_to_bmtk_id`` is neuron-aligned and reordered above; every helper
    # that reads coordinates or node types through it therefore follows the
    # layout automatically. Its inverse stores runtime ids and is relabelled.
    network["bmtk_id_to_tf_id"] = layout.relabel(network["bmtk_id_to_tf_id"])
    network["readout_neuron_ids"] = layout.relabel(network["readout_neuron_ids"])

    synapses = dict(network["synapses"])
    indices = np.array(synapses["indices"])
    indices[:, 0] = layout.relabel(indices[:, 0])
    indices[:, 1] = layout.relabel(indices[:, 1])
    synapses["indices"] = indices
    network["synapses"] = synapses

    return network, _relabel_post(lgn_input, layout), _relabel_post(bkg_input, layout)
