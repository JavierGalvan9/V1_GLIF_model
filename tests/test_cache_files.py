"""Cache layout migration must preserve payloads and distinguish settings."""

import json
import pickle

import pytest

from v1_model_utils.cache_files import (
    lgn_cache_details,
    lgn_cache_path,
    migrate_directory,
)
from v1_model_utils import cache_files


PHASE_CACHE = (
    "protocol_phase_lgn_probabilities_lgnv2_n_input_17400_seqlen_500_"
    "pre_0_post_0_rotation_ccw_phases_8_angles_0-45-90-135-180-225-270-315.pkl"
)
OSI_CACHE = (
    "osi_dsi_lgn_probabilities_lgnv2_n_input_17400_seqlen_4000_"
    "pre_2000_post_0_rotation_ccw_seed_3000_spikes_angles_0-45-90-135-180-225-270-315.pkl"
)


@pytest.mark.parametrize("legacy,label", [
    (PHASE_CACHE, "validation_8phases_500ms_"),
    (OSI_CACHE, "eval_osi_dsi_4s_seed3000_"),
    (OSI_CACHE.replace("osi_dsi", "protocol_validation"), "validation_single_4s_seed3000_"),
    ("spontaneous_lgn_probabilities_n_input_17400_seqlen_500.pkl", "spontaneous_500ms_"),
])
def test_lgn_names_and_lossless_migration(tmp_path, legacy, label):
    directory = tmp_path / "tf_data"
    directory.mkdir()
    source = directory / legacy
    contents = pickle.dumps({0: [1.0, 2.0]})
    source.write_bytes(contents)
    original = source.stat()

    migrate_directory(tmp_path)
    path = directory / "lgn_probabilities" / lgn_cache_details(legacy)[0]

    assert path.name.startswith(label)
    assert path.read_bytes() == contents
    assert path.stat().st_ino == original.st_ino
    assert path.stat().st_mtime_ns == original.st_mtime_ns
    assert not source.exists()
    metadata = json.loads(path.with_suffix(".json").read_text())
    assert metadata["legacy_filename"] == legacy
    assert metadata["config"]["n_input"] == 17400
    assert lgn_cache_path(tmp_path, legacy) == str(path)
    assert migrate_directory(tmp_path) == []


@pytest.mark.parametrize("old,new", [
    ("17400", "2000"), ("seqlen_500", "seqlen_1000"),
    ("pre_0", "pre_100"), ("post_0", "post_100"),
    ("ccw", "cw"), ("phases_8", "phases_4"),
    ("0-45-90-135-180-225-270-315", "0-90"), ("lgnv2", "lgnv3"),
])
def test_lgn_hash_distinguishes_configuration(old, new):
    assert lgn_cache_details(PHASE_CACHE)[0] != lgn_cache_details(PHASE_CACHE.replace(old, new))[0]


def test_lgn_hash_distinguishes_seed_and_input_mode():
    name = lgn_cache_details(OSI_CACHE)[0]
    assert name != lgn_cache_details(OSI_CACHE.replace("seed_3000", "seed_3001"))[0]
    assert name != lgn_cache_details(OSI_CACHE.replace("spikes", "current"))[0]


def test_custom_osi_cache_prefix_is_supported():
    legacy = OSI_CACHE.replace("osi_dsi_lgn_probabilities", "custom")
    name, config = lgn_cache_details(legacy)
    assert name.startswith("custom_4s_seed3000_")
    assert config["purpose"] == "custom"


def test_runtime_reuses_a_legacy_lgn_cache(tmp_path):
    directory = tmp_path / "tf_data"
    directory.mkdir()
    (directory / PHASE_CACHE).write_bytes(b"unchanged payload")
    path = lgn_cache_path(tmp_path, PHASE_CACHE)
    assert open(path, "rb").read() == b"unchanged payload"
    assert not (directory / PHASE_CACHE).exists()


@pytest.mark.parametrize("error", [FileExistsError, FileNotFoundError])
def test_concurrent_worker_migration_reuses_completed_cache(tmp_path, monkeypatch, error):
    directory = tmp_path / "tf_data"
    directory.mkdir()
    (directory / PHASE_CACHE).write_bytes(b"shared payload")
    move = cache_files._move_cache

    def another_worker_moved_first(source, destination):
        move(source, destination)
        raise error("another worker completed migration")

    monkeypatch.setattr(cache_files, "_move_cache", another_worker_moved_first)
    path = lgn_cache_path(tmp_path, PHASE_CACHE)
    with open(path, "rb") as handle:
        assert handle.read() == b"shared payload"


def test_migration_refuses_collisions_before_moving_any_files(tmp_path):
    directory = tmp_path / "tf_data"
    directory.mkdir()
    source = directory / PHASE_CACHE
    source.write_bytes(b"old cache")
    target_dir = directory / "lgn_probabilities"
    target_dir.mkdir()
    target = target_dir / lgn_cache_details(PHASE_CACHE)[0]
    target.write_bytes(b"different cache")
    network = directory / "V1_network_v1_10_123456abcdef.pkl"
    network.write_bytes(b"network")

    with pytest.raises(FileExistsError):
        migrate_directory(tmp_path)
    assert source.read_bytes() == b"old cache"
    assert target.read_bytes() == b"different cache"
    assert network.exists()


@pytest.mark.parametrize("suffix", [".pkl", ".pkl.lz4"])
@pytest.mark.parametrize("key", ["", "_123456abcdef"])
def test_network_migration_preserves_bytes(tmp_path, suffix, key):
    directory = tmp_path / "tf_data"
    directory.mkdir()
    source = directory / f"V1_network_v1_1000{key}{suffix}"
    source.write_bytes(b"unchanged network")
    [(old, new)] = migrate_directory(tmp_path)
    expected_key = key.lstrip("_") or "legacy"
    assert new.name == f"v1_network_1000neurons_{expected_key}{suffix}"
    assert new.read_bytes() == b"unchanged network"
    assert not old.exists()
