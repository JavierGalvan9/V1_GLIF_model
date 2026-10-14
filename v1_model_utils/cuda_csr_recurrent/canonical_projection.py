"""Host-only, bitwise-exact sharing of identical basis projections."""

import numpy as np


def canonical_pairs(pair_posts, pair_types, pair_ids, basis):
    """Return compact metadata, preserving all edge IDs and their order.

    Compare FP32 bit patterns, including signed zero and NaN payloads. Keep the
    first original type as representative; never alter biological type arrays.
    """
    basis = np.ascontiguousarray(basis, dtype=np.float32)
    if basis.ndim != 2:
        raise ValueError("basis must be a matrix")
    if np.asarray(pair_types).size and int(np.max(pair_types)) >= len(basis):
        raise ValueError("pair type exceeds basis rows")
    raw = basis.view(np.uint32)
    _, first, inverse = np.unique(raw, axis=0, return_index=True, return_inverse=True)
    representatives = first[inverse].astype(np.uint8)
    codes = (np.asarray(pair_posts, np.uint64) << np.uint64(8)) | representatives[pair_types]
    unique, remap = np.unique(codes, return_inverse=True)
    remap = remap.astype(np.uint32)
    return {
        "pair_ids": remap[np.asarray(pair_ids)],
        "pair_posts": (unique >> np.uint64(8)).astype(np.uint32),
        "pair_types": (unique & np.uint64(255)).astype(np.uint8),
        "n_pairs": int(len(unique)),
    }
