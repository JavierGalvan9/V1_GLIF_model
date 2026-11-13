"""Readable cache names and lossless migration of the previous disk layout.

Run ``python -m v1_model_utils.cache_files GLIF_network_nll_*`` to migrate
existing caches. Payloads are renamed, never loaded or rewritten.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile


def _duration(milliseconds):
    return f"{milliseconds // 1000}s" if milliseconds % 1000 == 0 else f"{milliseconds}ms"


def lgn_cache_details(legacy_name):
    """Return the short filename and all settings recorded by the old name.

    Backend and BMTK compatibility were not recorded in legacy filenames, so
    these metadata deliberately make no claims about their values.
    """
    spontaneous = re.fullmatch(
        r"spontaneous_lgn_probabilities_n_input_(\d+)_seqlen_(\d+)\.pkl", legacy_name
    )
    if spontaneous:
        n_input, seq_len = map(int, spontaneous.groups())
        config = dict(purpose="spontaneous", n_input=n_input, seq_len=seq_len)
        label = f"spontaneous_{_duration(seq_len)}"
    else:
        match = re.fullmatch(
            r"(.+)_(lgnv\d+)_n_input_(\d+)_seqlen_(\d+)_"
            r"pre_(\d+)_post_(\d+)_rotation_(cw|ccw)_"
            r"(?:phases_(\d+)|seed_(None|-?\d+)_(spikes|current))_angles_([\d-]+)\.pkl",
            legacy_name,
        )
        if match is None:
            raise ValueError(f"Unrecognized LGN cache filename: {legacy_name}")
        prefix, version, n_input, seq_len, pre, post, rotation, phases, seed, mode, angles = match.groups()
        purpose = {
            "osi_dsi_lgn_probabilities": "eval_osi_dsi",
            "protocol_phase_lgn_probabilities": "validation",
            "protocol_validation_lgn_probabilities": "validation_single",
        }.get(prefix, prefix)
        config = dict(
            purpose=purpose, version=version, n_input=int(n_input), seq_len=int(seq_len),
            pre_delay=int(pre), post_delay=int(post), rotation=rotation,
            angles=[int(angle) for angle in angles.split("-")],
        )
        if phases is not None:
            config["n_phases"] = int(phases)
            label = f"{purpose}_{phases}phases_{_duration(int(seq_len))}"
        else:
            config.update(seed=None if seed == "None" else int(seed), current_input=mode == "current")
            label = f"{purpose}_{_duration(int(seq_len))}_seed{seed}"
    label = re.sub(r"[^a-zA-Z0-9_-]", "_", label)
    serialized = json.dumps(config, sort_keys=True, separators=(",", ":"))
    key = hashlib.sha256(serialized.encode()).hexdigest()[:12]
    return f"{label}_{key}.pkl", config


def _write_metadata(path, legacy_name, config):
    metadata = path.with_suffix(".json")
    if metadata.exists():
        return
    contents = dict(config=config, legacy_filename=legacy_name)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as handle:
            temporary = handle.name
            json.dump(contents, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary, metadata)
    finally:
        if temporary is not None and os.path.exists(temporary):
            os.unlink(temporary)


def _move_cache(source, destination):
    """Move without overwriting an existing cache, preserving inode and mtime."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    # link() fails atomically if the destination exists, unlike rename().
    os.link(source, destination)
    source.unlink()


def lgn_cache_path(data_dir, legacy_name):
    """Resolve a cache, moving an old file into the new layout when needed."""
    name, config = lgn_cache_details(legacy_name)
    directory = Path(data_dir) / "tf_data"
    path = directory / "lgn_probabilities" / name
    legacy = directory / legacy_name
    if not path.exists() and legacy.exists():
        try:
            _move_cache(legacy, path)
        except (FileExistsError, FileNotFoundError):
            # Another worker may have migrated this shared cache first.
            if not path.exists():
                raise
    path.parent.mkdir(parents=True, exist_ok=True)
    _write_metadata(path, legacy_name, config)
    return os.fspath(path)


def network_cache_name(n_neurons, key, compressed=False):
    suffix = ".pkl.lz4" if compressed else ".pkl"
    return f"v1_network_{n_neurons}neurons_{key}{suffix}"


def migrate_directory(data_dir):
    """Migrate known legacy names; refuse collisions instead of discarding data."""
    directory = Path(data_dir) / "tf_data"
    moves = []
    for source in sorted(directory.glob("*.pkl*")):
        if "lgn_probabilities_" in source.name:
            name, config = lgn_cache_details(source.name)
            destination = directory / "lgn_probabilities" / name
        elif source.name.startswith("V1_network_"):
            match = re.fullmatch(r"V1_network_v1_(\d+)(?:_([a-f0-9]{12}))?\.pkl(\.lz4)?", source.name)
            if match is None:
                raise ValueError(f"Cannot recover configuration from {source}")
            neurons, key, compressed = match.groups()
            destination = directory / network_cache_name(neurons, key or "legacy", bool(compressed))
            config = None
        else:
            continue
        if destination.exists():
            raise FileExistsError(f"Preserving both caches: {source} and {destination}")
        moves.append((source, destination, config))
    for source, destination, config in moves:
        _move_cache(source, destination)
        if config is not None:
            _write_metadata(destination, source.name, config)
        print(f"{source} -> {destination}")
    return [(source, destination) for source, destination, _ in moves]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directories", nargs="+", type=Path)
    args = parser.parse_args()
    for directory in args.directories:
        migrate_directory(directory)
