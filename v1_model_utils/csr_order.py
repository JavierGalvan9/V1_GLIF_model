"""Host-side CSR edge ordering shared by the layout pass and the operators.

``spatial_layout`` reorders the loaded network into each operator's CSR order,
and every operator then re-derives that order to check the caller kept its
promise. Both sides sort the same keys, so they build the permutation here and
agree by construction.

The keys are small non-negative integers, so packing them into one uint64 lets
a single radix pass replace the multi-key ``np.lexsort``, and lets the identity
check that the operators perform become a linear scan instead of a second sort.
Packing is skipped when the keys do not fit 64 bits, which keeps the fallback
exact rather than approximate.
"""

import numpy as np


def _packed_key(keys):
    """Pack sort keys, most significant first, into one uint64 per edge.

    Returns ``None`` when the keys need more than 64 bits together, so callers
    fall back to ``np.lexsort`` on the unpacked keys.
    """
    keys = [np.asarray(key) for key in keys]
    if any(key.size and int(key.min()) < 0 for key in keys):
        return None
    widths = [max(1, int(key.max()).bit_length()) if key.size else 1 for key in keys]
    if sum(widths) > 64:
        return None
    packed = np.zeros(keys[0].shape, np.uint64)
    shift = 0
    for key, width in zip(reversed(keys), reversed(widths)):
        packed |= key.astype(np.uint64) << np.uint64(shift)
        shift += width
    return packed


def edge_order(keys):
    """Return the stable permutation sorting edges by ``keys``.

    ``keys`` are ordered most significant first, and ties keep the caller's
    edge order, which is what the operators' CSR layout assumes.
    """
    packed = _packed_key(keys)
    if packed is None:
        original = np.arange(np.asarray(keys[0]).size, dtype=np.uint32)
        return np.lexsort((original, *reversed(keys))).astype(np.uint32)
    return np.argsort(packed, kind="stable").astype(np.uint32)


def is_identity_order(keys):
    """Whether :func:`edge_order` on ``keys`` would be the identity.

    A stable sort leaves an array untouched exactly when its keys are already
    non-decreasing, so the check costs one linear pass instead of the sort it
    replaces.
    """
    packed = _packed_key(keys)
    if packed is None:
        size = np.asarray(keys[0]).size
        return np.array_equal(edge_order(keys), np.arange(size, dtype=np.uint32))
    return bool(np.all(packed[1:] >= packed[:-1]))


def compact_pairs(post_ids, synapse_types):
    """Return the distinct ``(post, type)`` codes and each edge's index into them.

    The codes span ``n_post * 256`` values at most, so counting them in a dense
    array replaces the sort ``np.unique`` would run over one code per edge. The
    distinct codes come out ascending, matching ``np.unique``.
    """
    codes = (np.asarray(post_ids).astype(np.int64) << 8) | np.asarray(
        synapse_types
    ).astype(np.int64)
    if codes.size == 0:
        return np.empty((0,), np.int64), np.empty((0,), np.uint32)
    size = int(codes.max()) + 1
    seen = np.zeros(size, np.bool_)
    seen[codes] = True
    unique_codes = np.flatnonzero(seen)
    lookup = np.zeros(size, np.uint32)
    lookup[unique_codes] = np.arange(unique_codes.size, dtype=np.uint32)
    return unique_codes, lookup[codes]
