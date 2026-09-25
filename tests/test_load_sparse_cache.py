from types import SimpleNamespace

import h5py
import numpy as np

from v1_model_utils import load_sparse


NODE_PARAMS = {
    "V_th": -50.0,
    "g": 1.0,
    "E_L": -70.0,
    "k": [0.1, 0.2],
    "C_m": 2.0,
    "V_reset": -65.0,
    "t_ref": 2.0,
    "asc_amps": [0.0, 0.0],
}


def _write_minimal_network(tmp_path):
    network_dir = tmp_path / "network"
    cache_dir = tmp_path / "tf_data"
    network_dir.mkdir()
    cache_dir.mkdir()
    with h5py.File(network_dir / "v1_nodes.h5", "w") as handle:
        population = handle.create_group("nodes/v1")
        population.create_dataset("node_id", data=np.arange(6))
        population.create_dataset("node_type_id", data=np.ones(6, dtype=np.int32))
        spatial = population.create_group("0")
        spatial.create_dataset("x", data=np.arange(6, dtype=np.float32))
        spatial.create_dataset("y", data=np.zeros(6, dtype=np.float32))
        spatial.create_dataset("z", data=np.zeros(6, dtype=np.float32))
        spatial.create_dataset("tuning_angle", data=np.zeros(6, dtype=np.float32))

    network_dat = {
        "nodes": [{"ids": np.arange(6), "params": NODE_PARAMS}],
        "edges": [{
            "edge_type_id": 1,
            "source": np.repeat(np.arange(6), 6),
            "target": np.tile(np.arange(6), 6),
            "params": {
                "weight": np.arange(1, 37, dtype=np.float32),
                "delay": 1.0,
                "syn_id": 0,
            },
        }],
    }
    with open(cache_dir / "network_dat.pkl", "wb") as handle:
        load_sparse.pkl.dump(network_dat, handle)


def test_random_selection_returns_exact_requested_population(tmp_path):
    _write_minimal_network(tmp_path)

    network = load_sparse.load_network(
        data_dir=tmp_path,
        core_only=False,
        connected_selection=False,
        n_neurons=2,
        seed=7,
    )

    assert network["n_nodes"] == 2
    assert len(network["tf_id_to_bmtk_id"]) == 2


def test_random_weights_are_reproducible_from_model_seed(tmp_path):
    _write_minimal_network(tmp_path)

    def load(seed):
        return load_sparse.load_network(
            data_dir=tmp_path,
            core_only=False,
            connected_selection=True,
            n_neurons=6,
            seed=seed,
            random_weights=True,
        )["synapses"]["weights"]

    np.testing.assert_array_equal(load(11), load(11))
    assert not np.array_equal(load(11), load(12))


class _Flags(SimpleNamespace):
    def flag_values_dict(self):
        return vars(self)

    def __getitem__(self, name):
        return SimpleNamespace(default=getattr(self, name))


def _flags(data_dir, seed):
    return _Flags(
        data_dir=str(data_dir), neurons=2, n_input=10, core_only=False,
        connected_selection=False, random_weights=False, uniform_weights=False,
        seed=seed, loss_core_radius=400.0, n_output=1, neurons_per_output=1,
    )


def test_cached_load_v1_does_not_reuse_seed_dependent_payload(tmp_path, monkeypatch):
    (tmp_path / "tf_data").mkdir()
    (tmp_path / "network").mkdir()
    (tmp_path / "network" / "v1_nodes.h5").write_bytes(b"source")
    calls = []

    def fake_load(flags, n_neurons):
        calls.append(flags.seed)
        return {"seed": flags.seed}, {"seed": flags.seed}, {"seed": flags.seed}

    monkeypatch.setattr(load_sparse, "load_v1", fake_load)
    first = load_sparse.cached_load_v1(_flags(tmp_path, 1), 2, flag_str="shared")
    second = load_sparse.cached_load_v1(_flags(tmp_path, 2), 2, flag_str="shared")
    repeated = load_sparse.cached_load_v1(_flags(tmp_path, 2), 2, flag_str="shared")

    assert first[0]["seed"] == 1
    assert second[0]["seed"] == 2
    assert repeated[0]["seed"] == 2
    assert calls == [1, 2]


def test_cached_load_v1_invalidates_payload_when_source_changes(tmp_path, monkeypatch):
    (tmp_path / "tf_data").mkdir()
    (tmp_path / "network").mkdir()
    source = tmp_path / "network" / "v1_nodes.h5"
    source.write_bytes(b"first")
    calls = []

    def fake_load(flags, n_neurons):
        calls.append(source.read_bytes())
        return {"source": calls[-1]}, {}, {}

    monkeypatch.setattr(load_sparse, "load_v1", fake_load)
    flags = _flags(tmp_path, 1)
    load_sparse.cached_load_v1(flags, 2, flag_str="shared")
    source.write_bytes(b"second and changed")
    result = load_sparse.cached_load_v1(flags, 2, flag_str="shared")

    assert result[0]["source"] == b"second and changed"
    assert calls == [b"first", b"second and changed"]
