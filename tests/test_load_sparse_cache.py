import os
import shutil
from types import SimpleNamespace

import h5py
import lz4.frame
import pytest
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


def _flags(data_dir, seed, compress_model_cache=False):
    return _Flags(
        data_dir=str(data_dir), neurons=2, n_input=10, core_only=False,
        connected_selection=False, random_weights=False, uniform_weights=False,
        seed=seed, loss_core_radius=400.0, n_output=1, neurons_per_output=1,
        compress_model_cache=compress_model_cache,
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


@pytest.mark.parametrize("compressed", [False, True])
def test_cached_load_v1_invalidates_payload_when_source_changes(tmp_path, monkeypatch, compressed):
    (tmp_path / "tf_data").mkdir()
    (tmp_path / "network").mkdir()
    source = tmp_path / "network" / "v1_nodes.h5"
    source.write_bytes(b"first")
    calls = []

    def fake_load(flags, n_neurons):
        calls.append(source.read_bytes())
        return {"source": calls[-1]}, {}, {}

    monkeypatch.setattr(load_sparse, "load_v1", fake_load)
    flags = _flags(tmp_path, 1, compressed)
    load_sparse.cached_load_v1(flags, 2, flag_str="shared")
    source.write_bytes(b"second and changed")
    result = load_sparse.cached_load_v1(flags, 2, flag_str="shared")

    assert result[0]["source"] == b"second and changed"
    assert calls == [b"first", b"second and changed"]


@pytest.mark.parametrize("compressed", [False, True])
def test_failed_cache_dump_leaves_no_temporary_file(tmp_path, monkeypatch, compressed):
    (tmp_path / "tf_data").mkdir()
    (tmp_path / "network").mkdir()
    monkeypatch.setattr(load_sparse, "load_v1", lambda flags, n_neurons: ({}, {}, {}))

    def full_disk(payload, handle):
        handle.write(b"partial")
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(load_sparse.pkl, "dump", full_disk)
    with pytest.raises(OSError):
        load_sparse.cached_load_v1(_flags(tmp_path, 1, compressed), 2, flag_str="shared")
    assert list((tmp_path / "tf_data").iterdir()) == []


def test_alternating_seeds_reuse_their_own_caches(tmp_path, monkeypatch):
    (tmp_path / "tf_data").mkdir()
    (tmp_path / "network").mkdir()
    calls = []

    def fake_load(flags, n_neurons):
        calls.append(flags.seed)
        return {"seed": flags.seed}, {}, {}

    monkeypatch.setattr(load_sparse, "load_v1", fake_load)
    for seed in (1, 2, 1, 2):
        result = load_sparse.cached_load_v1(_flags(tmp_path, seed), 2, flag_str="shared")
        assert result[0]["seed"] == seed
    assert calls == [1, 2]


@pytest.mark.parametrize("compressed", [False, True])
def test_matching_legacy_cache_is_reused_without_rebuilding(tmp_path, monkeypatch, compressed):
    (tmp_path / "tf_data").mkdir()
    (tmp_path / "network").mkdir()
    flags = _flags(tmp_path, 1, compressed)
    legacy = tmp_path / "tf_data" / "V1_network_shared.pkl"
    with open(legacy, "wb") as handle:
        load_sparse.pkl.dump({
            "version": load_sparse._V1_CACHE_VERSION,
            "provenance": load_sparse._v1_cache_provenance(flags, 2),
            "payload": ({"legacy": True}, {}, {}),
        }, handle)
    monkeypatch.setattr(load_sparse, "load_v1", lambda *args, **kwargs: pytest.fail("rebuilt"))

    result = load_sparse.cached_load_v1(flags, 2, flag_str="shared")

    assert result[0]["legacy"] is True
    assert legacy.exists() == compressed
    assert os.path.exists(load_sparse.v1_cache_path(flags, 2, "shared"))
    if compressed:
        with lz4.frame.open(load_sparse.v1_cache_path(flags, 2, "shared"), "rb") as handle:
            assert load_sparse.pkl.load(handle)["payload"][0]["legacy"] is True


@pytest.mark.parametrize("compressed", [False, True])
def test_cache_restored_from_a_moved_data_dir_points_at_the_new_location(tmp_path, monkeypatch, compressed):
    original = tmp_path / "original"
    (original / "tf_data").mkdir(parents=True)
    (original / "network").mkdir()
    (original / "network" / "v1_nodes.h5").write_bytes(b"source")
    monkeypatch.setattr(
        load_sparse, "load_v1",
        lambda flags, n_neurons: ({"data_dir": flags.data_dir}, {}, {}),
    )
    load_sparse.cached_load_v1(_flags(original, 1, compressed), 2, flag_str="shared")

    # Copying with metadata (as rsync -a does) keeps the provenance valid.
    moved = tmp_path / "moved"
    shutil.copytree(original, moved)
    shutil.rmtree(original)
    monkeypatch.setattr(load_sparse, "load_v1", lambda *args, **kwargs: pytest.fail("rebuilt"))

    result = load_sparse.cached_load_v1(_flags(moved, 1, compressed), 2, flag_str="shared")

    assert result[0]["data_dir"] == str(moved)


@pytest.mark.parametrize("compressed", [False, True])
def test_model_cache_roundtrips_numpy_arrays_without_rebuilding(tmp_path, monkeypatch, compressed):
    flags = _flags(tmp_path, 1, compressed)
    weights = np.arange(12, dtype=np.float32).reshape(3, 4)
    indices = np.array([[0, 1], [2, 3]], dtype=np.int32)
    monkeypatch.setattr(load_sparse, "load_v1", lambda **kwargs: (
        {"synapses": {"weights": weights, "indices": indices}}, {}, {},
    ))
    load_sparse.cached_load_v1(flags, 2, flag_str="shared")
    path = load_sparse.v1_cache_path(flags, 2, "shared")
    with open(path, "rb") as handle:
        assert (handle.read(4) == b"\x04\x22\x4d\x18") == compressed
    monkeypatch.setattr(load_sparse, "load_v1", lambda **kwargs: pytest.fail("rebuilt"))

    network, _, _ = load_sparse.cached_load_v1(flags, 2, flag_str="shared")

    for key, expected in (("weights", weights), ("indices", indices)):
        actual = network["synapses"][key]
        np.testing.assert_array_equal(actual, expected)
        assert actual.dtype == expected.dtype
    assert network["data_dir"] == str(tmp_path)


@pytest.mark.parametrize("initial_compressed", [False, True])
@pytest.mark.parametrize("invalid_preferred", [False, True])
def test_switching_cache_encoding_reuses_payload_and_preserves_original(
    tmp_path, monkeypatch, initial_compressed, invalid_preferred
):
    flags = _flags(tmp_path, 1, initial_compressed)
    monkeypatch.setattr(load_sparse, "load_v1", lambda **kwargs: ({"seed": 1}, {}, {}))
    load_sparse.cached_load_v1(flags, 2, flag_str="shared")
    original_path = load_sparse.v1_cache_path(flags, 2, "shared")
    with open(original_path, "rb") as handle:
        original_bytes = handle.read()
    original_provenance = load_sparse._v1_cache_provenance(flags, 2)
    flags.compress_model_cache = not initial_compressed
    requested_path = load_sparse.v1_cache_path(flags, 2, "shared")
    if invalid_preferred:
        with open(requested_path, "wb") as handle:
            handle.write(b"corrupt cache")
    monkeypatch.setattr(load_sparse, "load_v1", lambda **kwargs: pytest.fail("rebuilt"))

    assert load_sparse.cached_load_v1(flags, 2, flag_str="shared")[0]["seed"] == 1
    assert load_sparse.cached_load_v1(flags, 2, flag_str="shared")[0]["seed"] == 1

    assert load_sparse._v1_cache_provenance(flags, 2) == original_provenance
    with open(original_path, "rb") as handle:
        assert handle.read() == original_bytes
    assert os.path.exists(requested_path)
    assert not list((tmp_path / "tf_data").glob("*.tmp"))


def test_missing_compression_flag_preserves_plain_cache_path(tmp_path):
    flags = _flags(tmp_path, 1)
    plain_path = load_sparse.v1_cache_path(flags, 2, "shared")
    del flags.compress_model_cache
    assert load_sparse.v1_cache_path(flags, 2, "shared") == plain_path


def test_run_labels_share_a_configuration_cache(tmp_path):
    flags = _flags(tmp_path, 1)
    path = load_sparse.v1_cache_path(flags, 2, "single")
    assert path == load_sparse.v1_cache_path(flags, 2, "mirrored")
    assert os.path.basename(path).startswith("v1_network_2neurons_")
    assert path != load_sparse.v1_cache_path(_flags(tmp_path, 2), 2, "single")


@pytest.mark.parametrize("compressed", [False, True])
@pytest.mark.parametrize("legacy_key", [False, True])
def test_previous_cache_names_are_reused_without_rebuilding(
    tmp_path, monkeypatch, compressed, legacy_key
):
    flags = _flags(tmp_path, 1, compressed)
    monkeypatch.setattr(load_sparse, "load_v1", lambda **kwargs: ({"seed": 1}, {}, {}))
    load_sparse.cached_load_v1(flags, 2, flag_str="shared")
    path = load_sparse.v1_cache_path(flags, 2, "shared")
    suffix = ".pkl.lz4" if compressed else ".pkl"
    key = os.path.basename(path).removesuffix(suffix).rsplit("_", 1)[1]
    name = f"V1_network_shared_{key}{suffix}" if legacy_key else f"v1_network_2neurons_legacy{suffix}"
    previous = tmp_path / "tf_data" / name
    os.rename(path, previous)
    before = previous.read_bytes()
    monkeypatch.setattr(load_sparse, "load_v1", lambda **kwargs: pytest.fail("rebuilt"))

    result = load_sparse.cached_load_v1(flags, 2, flag_str="shared")

    assert result[0]["seed"] == 1
    assert not previous.exists()
    with open(path, "rb") as handle:
        assert handle.read() == before


@pytest.mark.parametrize("compressed", [False, True])
def test_failed_replacement_preserves_existing_cache(tmp_path, monkeypatch, compressed):
    (tmp_path / "network").mkdir()
    source = tmp_path / "network" / "v1_nodes.h5"
    source.write_bytes(b"original")
    flags = _flags(tmp_path, 1, compressed)
    monkeypatch.setattr(load_sparse, "load_v1", lambda **kwargs: ({}, {}, {}))
    load_sparse.cached_load_v1(flags, 2, flag_str="shared")
    cache_path = load_sparse.v1_cache_path(flags, 2, "shared")
    with open(cache_path, "rb") as handle:
        original_bytes = handle.read()
    source.write_bytes(b"changed source")

    def full_disk(payload, handle):
        handle.write(b"partial")
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(load_sparse.pkl, "dump", full_disk)
    with pytest.raises(OSError):
        load_sparse.cached_load_v1(flags, 2, flag_str="shared")

    with open(cache_path, "rb") as handle:
        assert handle.read() == original_bytes
    assert not list((tmp_path / "tf_data").glob("*.tmp"))
