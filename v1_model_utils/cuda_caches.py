"""Deterministic GPU selection and persistent CUDA/XLA compilation caches.

This module must stay free of TensorFlow imports: every variable it sets is
read while CUDA and XLA initialize, so entry points call it from their
pre-import environment block.
"""

import os
import subprocess
from pathlib import Path


# The stock TensorFlow wheel ships no Blackwell SASS (it is built for sm_60,
# sm_70, sm_80, sm_89 and compute_90 PTX), so its kernels are JIT-compiled from
# PTX on first use. The driver caches the result, and this budget keeps that
# cache from evicting it and paying the multi-minute compile again.
_JIT_CACHE_BYTES = 2 * 1024**3

_DUMP_FLAG = "--xla_gpu_dump_autotune_results_to"
_LOAD_FLAG = "--xla_gpu_load_autotune_results_from"


def cache_root():
    """Return the persistent cache root, honoring standard cache overrides."""
    configured = os.environ.get("V1_CUDA_CACHE_DIR")
    if configured:
        return Path(configured).expanduser()
    xdg_cache = os.environ.get("XDG_CACHE_HOME")
    base = Path(xdg_cache).expanduser() if xdg_cache else Path.home() / ".cache"
    return base / "v1_glif"


def _visible_gpu_key():
    """Return a filename-safe key naming the GPUs this process will use.

    Autotuning results are only valid for the GPU they were measured on, so the
    cache file is keyed by device. Requires `CUDA_DEVICE_ORDER=PCI_BUS_ID` to
    already be set, which makes `CUDA_VISIBLE_DEVICES` indices line up with the
    ones `nvidia-smi` reports.
    """
    try:
        listing = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10, check=True,
        ).stdout.split("\n")
    except (OSError, subprocess.SubprocessError):
        return "unknown-gpu"

    names = [line.strip() for line in listing if line.strip()]
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible:
        selected = []
        for index in visible.split(","):
            index = index.strip()
            if index.isdigit() and int(index) < len(names):
                selected.append(names[int(index)])
        names = selected
    if not names:
        return "unknown-gpu"
    unique = sorted(set(names))
    return "_".join("".join(
        character if character.isalnum() else "-" for character in name
    ).strip("-") for name in unique)[:120]


def configure_cuda_caches():
    """Pin GPU enumeration order and persist CUDA/XLA compilation artifacts.

    CUDA's default device order is `FASTEST_FIRST`, whose ranking is not stable
    across processes, so the same `CUDA_VISIBLE_DEVICES=0` can select different
    physical GPUs on the same host. Pinning it to `PCI_BUS_ID` makes the index
    mean what `nvidia-smi` shows, which both reproduces device selection and
    lets the autotuning cache be keyed by the GPU it was measured on.

    The autotuning results cached here are the cuDNN and cuBLAS algorithm
    choices XLA measures on first compilation. Reusing them removes those
    measurements from later runs, which also avoids selecting an algorithm from
    the noisy timings behind the runtime's delay-kernel warning. This shortens
    the first step; it does not change steady-state step time.

    Anything already set in the environment wins, so a caller can opt out.
    """
    os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    os.environ.setdefault("CUDA_CACHE_MAXSIZE", str(_JIT_CACHE_BYTES))

    xla_flags = os.environ.get("XLA_FLAGS", "")
    if _DUMP_FLAG in xla_flags or _LOAD_FLAG in xla_flags:
        return
    autotune_dir = cache_root() / "xla_autotune"
    autotune_dir.mkdir(parents=True, exist_ok=True)
    results = autotune_dir / f"{_visible_gpu_key()}.txtpb"

    # Loading a file that does not exist yet is an error, so the first run on a
    # GPU only writes; every later run both loads and refreshes it.
    flags = [xla_flags] if xla_flags else []
    if results.is_file():
        flags.append(f"{_LOAD_FLAG}={results}")
    flags.append(f"{_DUMP_FLAG}={results}")
    os.environ["XLA_FLAGS"] = " ".join(flags)
