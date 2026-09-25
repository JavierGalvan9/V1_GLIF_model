"""Per-timestep cost of the GLIF state kernels: legacy Euler vs exact propagator.

Dispatch overhead dominates a single eager call at these sizes, so each measured
function chains STEPS state transitions the way the RNN does - the output state
of one step is the input of the next - and reports the cost per step.
"""
import json, os, subprocess, sys, time
from pathlib import Path
import numpy as np, tensorflow as tf

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
OPS = REPO / "v1_model_utils" / "cuda_glif_state"
BUILD = Path(os.environ.get("GLIF_BENCH_DIR", "/tmp/glif_kernel_bench"))
LEGACY_REF = os.environ.get("GLIF_LEGACY_REF", "HEAD~1")
BASIS, STEPS = 4, 64

sys.path.insert(0, str(REPO))


def _build():
    """Compile the current operators and those of `LEGACY_REF` side by side.

    The legacy sources get their op names rewritten so both libraries can be
    loaded into one TensorFlow op registry.
    """
    from v1_model_utils.cuda_operator_cache import (
        active_gpu_architecture, normalize_architecture,
        resolve_cuda_build_toolchain)

    BUILD.mkdir(parents=True, exist_ok=True)
    for stem, ref in (("legacy_ops", LEGACY_REF), ("new_ops", None)):
        for suffix in (".cc", ".cu.cc"):
            name = f"glif_state_ops{suffix}"
            text = (
                subprocess.run(["git", "show", f"{ref}:v1_model_utils/"
                                f"cuda_glif_state/{name}"], cwd=REPO,
                               capture_output=True, text=True, check=True).stdout
                if ref else (OPS / name).read_text())
            if ref:
                text = text.replace("FusedGlifSingle", "LegacyGlifSingle")
            (BUILD / f"{stem}{suffix}").write_text(text)

    architecture = os.environ.get("GLIF_BENCH_ARCH") or normalize_architecture(
        active_gpu_architecture())
    toolchain = resolve_cuda_build_toolchain(architecture)
    compile_flags, link_flags = (tf.sysconfig.get_compile_flags(),
                                 tf.sysconfig.get_link_flags())
    cuda_include = Path(tf.sysconfig.get_include()) / "third_party/gpus/cuda/include"
    libraries = []
    for stem in ("legacy_ops", "new_ops"):
        shared = BUILD / f"{stem}.{architecture}.so"
        if not shared.exists():
            subprocess.run([toolchain.cxx, "-std=c++17", "-O3", "-fPIC", "-c",
                            BUILD / f"{stem}.cc", "-o", BUILD / f"{stem}.cc.o",
                            *compile_flags], check=True)
            subprocess.run([toolchain.nvcc, "-std=c++17", "-O3", "-x", "cu", "-c",
                            BUILD / f"{stem}.cu.cc", "-o", BUILD / f"{stem}.cu.o",
                            "-DGOOGLE_CUDA=1", "--expt-relaxed-constexpr",
                            f"-gencode=arch=compute_{architecture},"
                            f"code=sm_{architecture}", "-Xcompiler=-fPIC",
                            f"-I{cuda_include}", *compile_flags], check=True)
            subprocess.run([toolchain.cxx, "-shared", "-O3", BUILD / f"{stem}.cc.o",
                            BUILD / f"{stem}.cu.o", "-o", shared, *link_flags,
                            f"-L{toolchain.library_directory}", "-l:libcudart.so.12",
                            f"-Wl,-rpath,{toolchain.library_directory}"], check=True)
        libraries.append(tf.load_op_library(str(shared)))
    print(f"built for sm_{architecture}", flush=True)
    return libraries


legacy, new = _build()


def constants(n, batch, dt):
    rng = np.random.default_rng(3)
    np_dt = np.float16 if dt == tf.float16 else np.float32
    u = lambda lo, hi, *s: tf.constant(rng.uniform(lo, hi, s).astype(np_dt))
    f = lambda *s: tf.constant(rng.normal(0, .3, s).astype(np_dt))
    return dict(
        z=tf.cast(tf.constant((rng.random((batch, n)) < .02).astype(np.float32)), dt),
        v=f(batch, n), r=tf.zeros((batch, n), tf.int8), asc=f(batch, n * 2),
        rise=f(batch, n * BASIS), psc=f(batch, n * BASIS), inputs=f(batch, n * BASIS),
        syn_decay=u(.4, .95, n * BASIS), psc_initial=u(.5, 2.7, n * BASIS),
        asc_decay=u(.74, .997, n * 2), asc_amps=u(-.3, .3, n * 2),
        decay=u(.83, .98, n), cf=u(.001, .02, n), psc_factor=u(.001, .02, n * BASIS),
        rise_factor=u(.0005, .01, n * BASIS), asc_factor=u(.001, .02, n * 2),
        reset_coeff=u(-1., -.8, n), t_ref=tf.fill((n,), tf.constant(2, tf.int8)),
        dt=tf.constant(1., dt), v_reset=tf.constant(0., dt),
        gv=f(batch, n), ga=f(batch, n * 2), grise=f(batch, n * BASIS),
        gpsc=f(batch, n * BASIS))


def chains(kind, t):
    """Forward and backward drivers that chain STEPS transitions."""
    if kind == "legacy":
        def one(z, v, r, asc, rise, psc):
            return legacy.legacy_glif_single_forward(
                z, v, r, asc, rise, psc, t["inputs"], t["syn_decay"],
                t["psc_initial"], t["asc_decay"], t["asc_amps"], t["decay"],
                t["cf"], t["t_ref"], t["dt"], t["v_reset"], hard_reset=False)

        def one_back(asc, rise, gv, ga, grise, gpsc):
            out = legacy.legacy_glif_single_backward(
                t["z"], t["r"], asc, rise, t["syn_decay"], t["psc_initial"],
                t["asc_decay"], t["decay"], t["cf"], t["t_ref"], t["asc_amps"],
                t["dt"], gv, ga, grise, gpsc,
                hard_reset=False, detach_reset=True, detach_asc_reset=False)
            return out[1], out[2], out[3], out[4]
    else:
        def one(z, v, r, asc, rise, psc):
            return new.fused_glif_single_forward(
                z, v, r, asc, rise, psc, t["inputs"], t["syn_decay"],
                t["psc_initial"], t["asc_decay"], t["asc_amps"], t["decay"],
                t["psc_factor"], t["t_ref"], t["dt"], t["v_reset"],
                t["rise_factor"], t["asc_factor"], t["reset_coeff"], hard_reset=False)

        def one_back(asc, rise, gv, ga, grise, gpsc):
            out = new.fused_glif_single_backward(
                t["z"], t["r"], asc, rise, t["syn_decay"], t["psc_initial"],
                t["asc_decay"], t["decay"], t["psc_factor"], t["t_ref"],
                t["asc_amps"], t["dt"], gv, ga, grise, gpsc, t["rise_factor"],
                t["asc_factor"], t["reset_coeff"],
                hard_reset=False, detach_reset=True, detach_asc_reset=False)
            return out[1], out[2], out[3], out[4]

    @tf.function
    def forward():
        v, r, asc, rise, psc = t["v"], t["r"], t["asc"], t["rise"], t["psc"]
        for _ in range(STEPS):
            v, r, asc, rise, psc = one(t["z"], v, r, asc, rise, psc)
        return v

    @tf.function
    def backward():
        gv, ga, grise, gpsc = t["gv"], t["ga"], t["grise"], t["gpsc"]
        for _ in range(STEPS):
            gv, ga, grise, gpsc = one_back(t["asc"], t["rise"], gv, ga, grise, gpsc)
        return gv

    return forward, backward


def timeit(fn, reps=30):
    for _ in range(5):
        fn().numpy()
    best = min(_one(fn, reps) for _ in range(3))
    return best / STEPS * 1e6           # microseconds per timestep


def _one(fn, reps):
    t0 = time.perf_counter()
    for _ in range(reps):
        out = fn()
    out.numpy()
    return (time.perf_counter() - t0) / reps


if __name__ == "__main__":
    results = []
    for n, batch in [(51978, 8), (51978, 32), (230924, 8), (230924, 32)]:
        for dt, tag in [(tf.float16, "fp16"), (tf.float32, "fp32")]:
            with tf.device("/GPU:0"):
                t = constants(n, batch, dt)
                row = dict(n=n, batch=batch, dtype=tag)
                for kind in ("legacy", "new"):
                    fwd, bwd = chains(kind, t)
                    row[f"{kind}_fwd"], row[f"{kind}_bwd"] = timeit(fwd), timeit(bwd)
                w = 2 if tag == "fp16" else 4
                # psc_factor and rise_factor (N*B each) and asc_factor (2N)
                # replace current_factor (N); reset_coeff (N) is new.
                row["extra_constant_MB"] = w * n * (2 * BASIS + 2) / 1e6
                results.append(row)
                print(json.dumps(row), flush=True)
    (HERE / "benchmark_results.json").write_text(json.dumps(results, indent=1))
    print(f"wrote {HERE / 'benchmark_results.json'}")
