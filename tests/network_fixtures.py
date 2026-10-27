"""Small cached V1 networks shared by the model-level tests.

``load_sparse.cached_load_v1`` caches each network under ``<data_dir>/tf_data``.
Tests read that cache and skip when it is absent; build the missing ones from
the full network data with

    python -m tests.network_fixtures 10 1000
"""

import os
import pickle
import sys
import unittest
from types import SimpleNamespace

DATA_DIR = "GLIF_network_nll_full"


def _flags():
    """Training defaults, with a single readout neuron so that the L5e
    requirement holds for the smallest fixtures."""
    return SimpleNamespace(
        data_dir=DATA_DIR, seed=3000, n_input=17400, core_only=False,
        connected_selection=True, random_weights=False, uniform_weights=False,
        loss_core_radius=200.0, n_output=1, neurons_per_output=1,
    )


def network_path(n_neurons):
    from v1_model_utils import load_sparse

    path = load_sparse.v1_cache_path(_flags(), n_neurons, f"v1_{n_neurons}")
    key = os.path.basename(path).removesuffix(".pkl").rsplit("_", 1)[1]
    directory = os.path.join(DATA_DIR, "tf_data")
    candidates = (
        path,
        os.path.join(directory, f"V1_network_v1_{n_neurons}_{key}.pkl"),
        os.path.join(directory, f"v1_network_{n_neurons}neurons_legacy.pkl"),
        os.path.join(directory, f"V1_network_v1_{n_neurons}.pkl"),
    )
    return next((candidate for candidate in candidates if os.path.exists(candidate)), path)


def load_network(n_neurons):
    """Return the cached ``(network, lgn_input, bkg_input)`` triple, or skip.

    ``load_sparse`` wraps the payload in a versioned dict; older caches on disk
    are the bare triple.
    """
    path = network_path(n_neurons)
    if not os.path.exists(path):
        raise unittest.SkipTest(
            f"needs the cached network {path}; build it with "
            f"`python -m tests.network_fixtures {n_neurons}`"
        )
    with open(path, "rb") as handle:
        cached = pickle.load(handle)
    network, lgn_input, bkg_input = cached["payload"] if isinstance(cached, dict) else cached
    # Like load_sparse.cached_load_v1: a cache built under another path or host
    # must read its companion files from this checkout's data directory.
    network["data_dir"] = DATA_DIR
    return network, lgn_input, bkg_input


def build_network(n_neurons):
    """Cache an ``n_neurons`` network with the training defaults."""
    from v1_model_utils import load_sparse

    load_sparse.cached_load_v1(_flags(), n_neurons, flag_str=f"v1_{n_neurons}")


if __name__ == "__main__":
    for argument in sys.argv[1:]:
        build_network(int(argument))
