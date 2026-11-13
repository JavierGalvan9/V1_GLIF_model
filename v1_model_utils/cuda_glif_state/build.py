"""Build architecture-keyed TensorFlow CUDA state-transition operators."""

import argparse
import os
from pathlib import Path

import tensorflow as tf

from v1_model_utils.cuda_operator_cache import (
    OperatorLibrary,
    active_gpu_architecture,
    artifact_path,
    normalize_architecture,
    resolve_cuda_build_toolchain,
    run_build_commands,
    temporary_build_directory,
    write_build_metadata,
)


HERE = Path(__file__).resolve().parent
BUILD_FLAGS = ("--expt-relaxed-constexpr",)
# The tiled kernels' shape per target architecture (glif_state_ops.cu.cc,
# V1_GLIF_*); an architecture without an entry compiles the defaults. sm_100
# (B200, Experiments/talaia_b200_20261023/tracks/glif100): four samples per
# forward thread, two resident forward blocks per SM, early forward loads and
# 128-thread backward blocks, bit for bit the default results.
ARCHITECTURE_FLAGS = {
    "100": (
        "-DV1_GLIF_FORWARD_TILE=4", "-DV1_GLIF_FORWARD_MIN_BLOCKS=2",
        "-DV1_GLIF_FORWARD_EARLY_LOADS=1", "-DV1_GLIF_BACKWARD_THREADS=128",
    ),
}


def architecture_kernel_flags(architecture):
    """GLIF kernel compile flags that depend on the target compute capability."""
    return ARCHITECTURE_FLAGS.get(normalize_architecture(architecture), ())


def build_flags_for(architecture):
    """The complete compile-flag set for one target architecture."""
    return (*BUILD_FLAGS, *architecture_kernel_flags(architecture))


LIBRARY = OperatorLibrary(
    HERE,
    "glif_state_ops",
    sources=(HERE / "build.py", HERE / "glif_state_ops.cc", HERE / "glif_state_ops.cu.cc"),
    build_module="v1_model_utils.cuda_glif_state.build",
    build_flags=build_flags_for,
)


def _build(stem, architecture, toolchain, compile_flags, link_flags):
    cxx, nvcc = toolchain.cxx, toolchain.nvcc
    build_flags = build_flags_for(architecture)
    output = artifact_path(HERE, stem, architecture)
    cuda_include = Path(tf.sysconfig.get_include()) / "third_party/gpus/cuda/include"
    with temporary_build_directory(HERE, stem, architecture) as workspace:
        cc_object = workspace / f"{stem}.cc.o"
        cu_object = workspace / f"{stem}.cu.o"
        temporary = workspace / output.name
        run_build_commands([
            cxx, "-std=c++17", "-O3", "-fPIC", "-Wall", "-Wextra",
            "-Wno-unused-parameter", "-c", HERE / f"{stem}.cc",
            "-o", cc_object, *compile_flags,
        ], [
            nvcc, "-std=c++17", "-O3", "-x", "cu", "-c",
            HERE / f"{stem}.cu.cc", "-o", cu_object, "-DGOOGLE_CUDA=1",
            *build_flags,
            f"-gencode=arch=compute_{architecture},code=sm_{architecture}",
            f"-gencode=arch=compute_{architecture},code=compute_{architecture}",
            "-Xcompiler=-fPIC,-Wall,-Wextra,-Wno-unused-parameter",
            f"-I{cuda_include}", *compile_flags,
        ])
        run_build_commands([
            cxx, "-shared", "-O3", cc_object, cu_object, "-o", temporary,
            *link_flags, f"-L{toolchain.library_directory}", "-l:libcudart.so.12",
            f"-Wl,-rpath,{toolchain.library_directory}",
        ])
        temporary.replace(output)
    write_build_metadata(HERE, stem, architecture, build_flags=build_flags)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--architecture", default=os.environ.get("V1_CUDA_ARCH"))
    args = parser.parse_args()
    architecture = normalize_architecture(
        args.architecture or active_gpu_architecture()
    )
    toolchain = resolve_cuda_build_toolchain(architecture)
    _build(
        "glif_state_ops", architecture, toolchain,
        tf.sysconfig.get_compile_flags(), tf.sysconfig.get_link_flags(),
    )


if __name__ == "__main__":
    main()
