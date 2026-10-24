"""Build architecture-keyed TensorFlow CUDA LGN operators (drifting-grating and movie)."""

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
STEMS = ("lgn_grating_ops", "lgn_movie_ops")
LIBRARIES = {
    stem: OperatorLibrary(
        HERE,
        stem,
        sources=(HERE / "build.py", HERE / "lgn_common.cuh", HERE / f"{stem}.cc",
                 HERE / f"{stem}.cu.cc"),
        build_module="v1_model_utils.cuda_lgn.build",
        build_flags=BUILD_FLAGS,
        build_args=("--stem", stem),
    )
    for stem in STEMS
}


def _build(stem, architecture, toolchain, compile_flags, link_flags):
    cxx, nvcc = toolchain.cxx, toolchain.nvcc
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
            *BUILD_FLAGS,
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
    write_build_metadata(HERE, stem, architecture, build_flags=BUILD_FLAGS)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--architecture", default=os.environ.get("V1_CUDA_ARCH"))
    parser.add_argument("--stem", choices=STEMS, action="append",
                        help="operator library to build (repeatable; default: all)")
    args = parser.parse_args()
    architecture = normalize_architecture(
        args.architecture or active_gpu_architecture()
    )
    toolchain = resolve_cuda_build_toolchain(architecture)
    for stem in args.stem or STEMS:
        _build(
            stem, architecture, toolchain,
            tf.sysconfig.get_compile_flags(), tf.sysconfig.get_link_flags(),
        )


if __name__ == "__main__":
    main()
