"""Build every stale CUDA operator library for the active GPUs in parallel.

Each library otherwise builds lazily, one after another, the first time a run
uses it. Building them all at once up front costs about as long as the slowest
library. The per-library cache locks of :func:`ensure_artifact` keep this safe
alongside other processes and the lazy path, which then finds fresh caches.

Usage: ``python -m v1_model_utils.cuda_prebuild [--architecture 120] [--jobs 4]``
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
import time

import tensorflow as tf

from v1_model_utils.cuda_operator_cache import (
    active_gpu_architecture,
    normalize_architecture,
)
from v1_model_utils.cuda_bkg_noise.build import LIBRARY as BKG_NOISE
from v1_model_utils.cuda_csr_external.build import LIBRARY as CSR_EXTERNAL
from v1_model_utils.cuda_csr_recurrent.build import LIBRARY as CSR_RECURRENT
from v1_model_utils.cuda_csr_resources.build import LIBRARY as CSR_RESOURCES
from v1_model_utils.cuda_frame_store.build import LIBRARY as FRAME_STORE
from v1_model_utils.cuda_glif_state.build import LIBRARY as GLIF_STATE
from v1_model_utils.cuda_lgn.build import LIBRARIES as LGN


# Longest cold build first, so the slowest libraries start immediately.
LIBRARIES = (
    CSR_EXTERNAL, CSR_RESOURCES, GLIF_STATE, CSR_RECURRENT, FRAME_STORE,
    BKG_NOISE, *LGN.values(),
)


# One library build peaks near 3 GB: cicc (~2.5 GB) next to its host compiler
# and its Python builder, measured on sm_120.
_BUILD_MEMORY_BYTES = 3 * 1024**3


def _memory_limit_bytes():
    """Host memory available to this process: MemAvailable, capped by its cgroup."""
    with open("/proc/meminfo", encoding="utf-8") as meminfo:
        limit = next(
            int(line.split()[1]) * 1024
            for line in meminfo
            if line.startswith("MemAvailable:")
        )
    with open("/proc/self/cgroup", encoding="utf-8") as cgroups:
        for line in cgroups:
            _, controllers, path = line.rstrip("\n").split(":", 2)
            if not controllers:
                candidate = Path("/sys/fs/cgroup", path.lstrip("/"), "memory.max")
            elif "memory" in controllers.split(","):
                candidate = Path(
                    "/sys/fs/cgroup/memory", path.lstrip("/"), "memory.limit_in_bytes"
                )
            else:
                continue
            try:
                value = candidate.read_text(encoding="utf-8").strip()
            except OSError:
                continue
            if value.isdigit():
                limit = min(limit, int(value))
    return limit


def default_jobs():
    """Concurrent library builds for the CPUs and memory this process may use.

    One build per CPU: a build's host object compiles in a fraction of its
    device object's time, so the CPUs stay busy without oversubscription (on 8
    CPUs this beat one build per two CPUs by 12 %). The builds may use at most
    half of the memory left to the process.
    """
    cpu_jobs = len(os.sched_getaffinity(0))
    memory_jobs = _memory_limit_bytes() // (2 * _BUILD_MEMORY_BYTES)
    return max(1, min(cpu_jobs, memory_jobs))


def prebuild(architecture=None, jobs=None, libraries=LIBRARIES):
    """Ensure every library is built; return ``{stem: error}`` for failures."""
    architecture = normalize_architecture(architecture or active_gpu_architecture())
    start = time.perf_counter()

    def ensure(library):
        library.ensure(architecture)
        print(f"CUDA operator {library.stem} ready after "
              f"{time.perf_counter() - start:.1f} s", flush=True)

    with ThreadPoolExecutor(max_workers=jobs or default_jobs()) as pool:
        futures = {library.stem: pool.submit(ensure, library) for library in libraries}
    return {
        stem: future.exception()
        for stem, future in futures.items()
        if future.exception() is not None
    }


def prebuild_for_run():
    """Prebuild at startup, leaving failures to each library's lazy build.

    A library that fails here fails the same way on first use, where its
    wrapper either reports it or falls back to the TensorFlow path.
    """
    if not tf.config.list_physical_devices("GPU"):
        return
    start = time.perf_counter()
    failures = prebuild()
    for stem, error in failures.items():
        print(f"CUDA operator prebuild failed for {stem}: {error}", flush=True)
    print(f"CUDA operators prebuilt in {time.perf_counter() - start:.1f} s", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--architecture", default=os.environ.get("V1_CUDA_ARCH"),
                        help="CUDA compute capability (default: active GPUs)")
    parser.add_argument("--jobs", type=int, default=None,
                        help="concurrent library builds (default: from CPUs and memory)")
    args = parser.parse_args()
    start = time.perf_counter()
    failures = prebuild(args.architecture, args.jobs)
    for stem, error in failures.items():
        print(f"FAILED {stem}: {error}", flush=True)
    print(f"Prebuilt {len(LIBRARIES) - len(failures)}/{len(LIBRARIES)} CUDA "
          f"operator libraries in {time.perf_counter() - start:.1f} s", flush=True)
    raise SystemExit(1 if failures else 0)


if __name__ == "__main__":
    main()
