"""Build architecture-keyed TensorFlow CUDA state-transition operators."""

import argparse
import os
from pathlib import Path
import shlex
import subprocess

import tensorflow as tf

from v1_model_utils.cuda_operator_cache import (
    active_gpu_architecture,
    artifact_path,
    normalize_architecture,
    resolve_cuda_build_toolchain,
    temporary_build_directory,
    write_build_metadata,
)


HERE = Path(__file__).resolve().parent
BUILD_FLAGS = ("--expt-relaxed-constexpr",)


def _run(command):
    print(shlex.join(str(part) for part in command), flush=True)
    subprocess.run([str(part) for part in command], check=True)


def _build(stem, architecture, toolchain, compile_flags, link_flags):
    cxx, nvcc = toolchain.cxx, toolchain.nvcc
    output = artifact_path(HERE, stem, architecture)
    cuda_include = Path(tf.sysconfig.get_include()) / "third_party/gpus/cuda/include"
    with temporary_build_directory(HERE, stem, architecture) as workspace:
        cc_object = workspace / f"{stem}.cc.o"
        cu_object = workspace / f"{stem}.cu.o"
        temporary = workspace / output.name
        _run([
            cxx, "-std=c++17", "-O3", "-fPIC", "-Wall", "-Wextra",
            "-Wno-unused-parameter", "-c", HERE / f"{stem}.cc",
            "-o", cc_object, *compile_flags,
        ])
        _run([
            nvcc, "-std=c++17", "-O3", "-x", "cu", "-c",
            HERE / f"{stem}.cu.cc", "-o", cu_object, "-DGOOGLE_CUDA=1",
            *BUILD_FLAGS,
            f"-gencode=arch=compute_{architecture},code=sm_{architecture}",
            f"-gencode=arch=compute_{architecture},code=compute_{architecture}",
            "-Xcompiler=-fPIC,-Wall,-Wextra,-Wno-unused-parameter",
            f"-I{cuda_include}", *compile_flags,
        ])
        _run([
            cxx, "-shared", "-O3", cc_object, cu_object, "-o", temporary,
            *link_flags, f"-L{toolchain.library_directory}", "-l:libcudart.so.12",
            f"-Wl,-rpath,{toolchain.library_directory}",
        ])
        temporary.replace(output)
    write_build_metadata(HERE, stem, architecture, build_flags=BUILD_FLAGS)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--architecture", default=os.environ.get("V1_CUDA_ARCH"))
    parser.add_argument(
        "--stem",
        choices=("glif_state_ops", "spike_history_ops"),
        help="Build only one operator; by default both are built.",
    )
    args = parser.parse_args()
    architecture = normalize_architecture(
        args.architecture or active_gpu_architecture()
    )
    toolchain = resolve_cuda_build_toolchain(architecture)
    compile_flags = tf.sysconfig.get_compile_flags()
    link_flags = tf.sysconfig.get_link_flags()
    stems = (args.stem,) if args.stem else ("glif_state_ops", "spike_history_ops")
    for stem in stems:
        _build(
            stem, architecture, toolchain, compile_flags, link_flags
        )


if __name__ == "__main__":
    main()
