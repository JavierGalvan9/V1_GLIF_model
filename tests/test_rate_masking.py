import os
import unittest

import h5py
import numpy as np

from v1_model_utils import other_v1_utils
from v1_model_utils.loss_functions import compare_rate_mask_population_ids


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def resolve_data_dir():
    candidates = [
        os.path.join(REPO_ROOT, "GLIF_network_nll_full"),
        os.path.join(REPO_ROOT, "GLIF_network_nll_core"),
        os.path.join(REPO_ROOT, "GLIF_network_nll"),
        os.path.join(REPO_ROOT, "GLIF_network"),
        os.path.join(REPO_ROOT, "biorealistic-v1-model", "tiny"),
    ]
    for candidate in candidates:
        if os.path.exists(os.path.join(candidate, "network", "v1_nodes.h5")):
            return candidate
    raise FileNotFoundError("Could not locate a V1 network fixture for rate-masking tests.")


DATA_DIR = resolve_data_dir()


def build_small_network(n_nodes=512):
    with h5py.File(os.path.join(DATA_DIR, "network/v1_nodes.h5"), mode="r") as node_h5:
        node_type_ids = np.array(node_h5["nodes"]["v1"]["node_type_id"][()])[:n_nodes]
    return {
        "n_nodes": n_nodes,
        "tf_id_to_bmtk_id": np.arange(n_nodes, dtype=np.int64),
        "node_type_ids": node_type_ids,
    }


class RateMaskingTests(unittest.TestCase):
    def test_canonical_and_legacy_population_ids_match_without_core_mask(self):
        network = build_small_network()
        comparison = compare_rate_mask_population_ids(network, data_dir=DATA_DIR)
        mismatches = [cell_type for cell_type, values in comparison.items() if not values["match"]]
        self.assertEqual(mismatches, [])

    def test_canonical_and_legacy_population_ids_match_with_core_mask(self):
        network = build_small_network()
        core_mask = other_v1_utils.isolate_core_neurons(
            network, radius=400.0, data_dir=DATA_DIR
        )
        comparison = compare_rate_mask_population_ids(
            network, data_dir=DATA_DIR, core_mask=core_mask
        )
        mismatches = [cell_type for cell_type, values in comparison.items() if not values["match"]]
        self.assertEqual(mismatches, [])


if __name__ == "__main__":
    unittest.main()
