"""Compare the current GLIF operator with an isolated packed-constant candidate.

The candidate keeps the op signature for a fair dispatch comparison. Its
syn_decay argument instead points to four interleaved coefficients; the three
now redundant inputs are retained solely for this experiment.
"""

import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time

import numpy as np
import tensorflow as tf


HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
OPS = REPO / "v1_model_utils" / "cuda_glif_state"
STEPS = 32
sys.path.insert(0, str(REPO))


def build(directory):
    from v1_model_utils.cuda_operator_cache import (
        active_gpu_architecture,
        normalize_architecture,
        resolve_cuda_build_toolchain,
    )

    arch = normalize_architecture(active_gpu_architecture())
    directory.mkdir(parents=True, exist_ok=True)
    toolchain = resolve_cuda_build_toolchain(arch)
    flags = tf.sysconfig.get_compile_flags()
    links = tf.sysconfig.get_link_flags()
    cuda_include = Path(tf.sysconfig.get_include()) / "third_party/gpus/cuda/include"
    sources = {}
    libraries = {}
    for label, name, cuda_source in (
        ("baseline", "BaselineGlifSingle", OPS / "glif_state_ops.cu.cc"),
        ("packed", "PackedGlifSingle", HERE / "packed_glif_state_ops.cu.cc"),
    ):
        cc = (OPS / "glif_state_ops.cc").read_text().replace("FusedGlifSingle", name)
        cu = cuda_source.read_text().replace("FusedGlifSingle", name)
        sources[label] = hashlib.sha256((cc + cu).encode()).hexdigest()
        stem = f"{label}.{arch}.{sources[label][:12]}"
        cc_path, cu_path = directory / f"{stem}.cc", directory / f"{stem}.cu.cc"
        cc_path.write_text(cc)
        cu_path.write_text(cu)
        cc_obj, cu_obj = directory / f"{stem}.cc.o", directory / f"{stem}.cu.o"
        shared = directory / f"{stem}.so"
        if not shared.exists():
            subprocess.run([toolchain.cxx, "-std=c++17", "-O3", "-w", "-fPIC", "-c",
                            cc_path, "-o", cc_obj, *flags], check=True)
            subprocess.run([toolchain.nvcc, "-std=c++17", "-O3", "-w", "-x", "cu", "-c",
                            cu_path, "-o", cu_obj, "-DGOOGLE_CUDA=1",
                            "--expt-relaxed-constexpr",
                            f"-gencode=arch=compute_{arch},code=sm_{arch}",
                            "-Xcompiler=-fPIC", f"-I{cuda_include}", *flags], check=True)
            subprocess.run([toolchain.cxx, "-shared", "-O3", cc_obj, cu_obj, "-o",
                            shared, *links, f"-L{toolchain.library_directory}",
                            "-l:libcudart.so.12",
                            f"-Wl,-rpath,{toolchain.library_directory}"], check=True)
        libraries[label] = tf.load_op_library(str(shared))
    return arch, sources, libraries


def inputs(n, batch, dtype):
    rng = np.random.default_rng(713)
    np_dtype = np.float16 if dtype == tf.float16 else np.float32

    def uniform(low, high, *shape):
        return tf.constant(rng.uniform(low, high, shape).astype(np_dtype))

    def normal(*shape):
        return tf.constant(rng.normal(0, 0.3, shape).astype(np_dtype))

    values = [
        uniform(0, 1, batch, n), normal(batch, n),
        tf.zeros((batch, n), tf.int8), normal(batch, n * 2),
        normal(batch, n * 4), normal(batch, n * 4), normal(batch, n * 4),
        uniform(.4, .95, n * 4), uniform(.5, 2.7, n * 4),
        uniform(.74, .997, n * 2), uniform(-.3, .3, n * 2),
        uniform(.83, .98, n), uniform(.001, .02, n * 4),
        tf.fill((n,), tf.constant(2, tf.int8)), tf.constant(1., dtype),
        tf.constant(0., dtype), uniform(.0005, .01, n * 4),
        uniform(.001, .02, n * 2), uniform(-1., -.8, n),
        uniform(-.03, .03, n),
    ]
    values[0] = tf.cast(values[0] < .02, dtype)
    packed = tf.reshape(tf.stack([values[i] for i in (7, 8, 12, 16)], axis=-1), [-1])
    upstream = [normal(batch, n), normal(batch, n * 2),
                normal(batch, n * 4), normal(batch, n * 4)]
    return values, packed, upstream


def call(library, label, values, packed, upstream, *, backward=False,
         hard_reset=False, detach_reset=True, detach_asc_reset=False):
    coeff = packed if label == "packed" else values[7]
    if backward:
        args = [values[i] for i in (0, 2, 3, 4)] + [coeff] + [values[i] for i in
            (8, 9, 11, 12, 13, 10, 14)] + list(upstream) + [values[i] for i in
            (16, 17, 18, 19)]
        return getattr(library, f"{label}_glif_single_backward")(
            *args, hard_reset=hard_reset, detach_reset=detach_reset,
            detach_asc_reset=detach_asc_reset)
    args = values.copy()
    args[7] = coeff
    return getattr(library, f"{label}_glif_single_forward")(
        *args, hard_reset=hard_reset)


def compare(libraries, values, packed, upstream):
    worst = 0.0
    for backward in (False, True):
        for hard_reset in (False, True):
            for detach_reset, detach_asc_reset in (
                ((True, False), (False, True)) if backward else ((True, False),)
            ):
                outputs = [call(libraries[label], label, values, packed, upstream,
                                backward=backward, hard_reset=hard_reset,
                                detach_reset=detach_reset,
                                detach_asc_reset=detach_asc_reset)
                           for label in ("baseline", "packed")]
                for expected, actual in zip(*outputs):
                    difference = np.max(np.abs(expected.numpy().astype(np.float32)
                                               - actual.numpy().astype(np.float32)))
                    worst = max(worst, float(difference))
                    tolerance = 0 if expected.dtype == tf.float16 else 1e-6
                    if difference > tolerance:
                        raise AssertionError(
                            f"output mismatch ({expected.dtype.name}): {difference}"
                        )
    return worst


def chain(library, label, values, packed, upstream, backward):
    if backward:
        @tf.function
        def run():
            gradients = upstream
            for _ in range(STEPS):
                out = call(library, label, values, packed, gradients, backward=True)
                gradients = out[1:5]
            return gradients[0]
    else:
        @tf.function
        def run():
            state = values[:6]
            for _ in range(STEPS):
                args = values.copy()
                args[:6] = [state[0], *state[1:]]
                out = call(library, label, args, packed, upstream)
                state = [state[0], *out]
            return state[1]
    return run


def measure(fn):
    for _ in range(3):
        fn().numpy()
    timings = []
    for _ in range(5):
        start = time.perf_counter()
        for _ in range(8):
            result = fn()
        result.numpy()
        timings.append((time.perf_counter() - start) * 1e6 / (8 * STEPS))
    return float(np.median(timings))


def main():
    check_only = "--check-only" in sys.argv[1:]
    packed_first = "--packed-first" in sys.argv[1:]
    timing_order = ("packed", "baseline") if packed_first else ("baseline", "packed")
    workspace = Path(tempfile.gettempdir()) / "glif_packed_cache"
    arch, sources, libraries = build(workspace)
    rows = []
    with tf.device("/GPU:0"):
        cases = ((13, 2),) if check_only else ((51978, 8), (51978, 32))
        for n, batch in cases:
            for dtype in (tf.float16, tf.float32):
                values, packed, upstream = inputs(n, batch, dtype)
                error = compare(libraries, values, packed, upstream)
                row = dict(n=n, batch=batch, dtype=dtype.name,
                           max_abs_difference=error)
                if check_only:
                    print(json.dumps(row), flush=True)
                    continue
                for backward, direction in ((False, "forward"), (True, "backward")):
                    functions = {label: chain(libraries[label], label, values,
                                              packed, upstream, backward)
                                 for label in libraries}
                    for label in timing_order:
                        row[f"{label}_{direction}_us"] = measure(functions[label])
                    row[f"{direction}_ratio"] = (row[f"packed_{direction}_us"] /
                                                 row[f"baseline_{direction}_us"])
                rows.append(row)
                print(json.dumps(row), flush=True)
    if check_only:
        return
    result = dict(gpu=tf.config.experimental.get_device_details(
        tf.config.list_physical_devices("GPU")[0]).get("device_name"),
        architecture=arch, tensorflow=tf.__version__, source_sha256=sources,
        steps=STEPS, timing_order=timing_order, rows=rows)
    filename = "packed_constants_results_reverse.json" if packed_first else "packed_constants_results.json"
    path = HERE / filename
    path.write_text(json.dumps(result, indent=2) + "\n")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
