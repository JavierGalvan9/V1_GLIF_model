"""Build architecture-keyed TensorFlow CUDA state-transition operators."""

import argparse
import os
from pathlib import Path
import shlex
import subprocess
import sys

import tensorflow as tf

from v1_model_utils.cuda_operator_cache import (
    active_gpu_architecture,
    artifact_path,
    normalize_architecture,
    temporary_build_directory,
    write_build_metadata,
)


HERE = Path(__file__).resolve().parent
BUILD_FLAGS = ("--expt-relaxed-constexpr",)


def _run(command):
    print(shlex.join(str(part) for part in command), flush=True)
    subprocess.run([str(part) for part in command], check=True)


def _build(stem, architecture, prefix, cxx, nvcc, compile_flags, link_flags):
    output = artifact_path(HERE, stem, architecture)
    cuda_include = Path(tf.sysconfig.get_include()) / "third_party/gpus/cuda/include"
    with temporary_build_directory(HERE, stem, architecture) as workspace:
        cc_object = workspace / f"{stem}.cc.o"
        cu_object = workspace / f"{stem}.cu.o"
        temporary = workspace / output.name
        _run([
            cxx, "-std=c++17", "-O3", "-fPIC", "-Wall", "-Wextra",
            "-Werror", "-Wno-unused-parameter", "-c", HERE / f"{stem}.cc",
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
            *link_flags, f"-L{prefix / 'lib'}", "-l:libcudart.so.12",
            f"-Wl,-rpath,{prefix / 'lib'}",
        ])
        temporary.replace(output)
    write_build_metadata(HERE, stem, architecture, build_flags=BUILD_FLAGS)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--architecture", default=os.environ.get("V1_CUDA_ARCH"))
    args = parser.parse_args()
    architecture = normalize_architecture(
        args.architecture or active_gpu_architecture()
    )
    prefix = Path(sys.prefix)
    nvcc = prefix / "bin/nvcc"
    if not nvcc.exists():
        raise FileNotFoundError(f"CUDA compiler not found at {nvcc}")
    cxx = os.environ.get("CXX", "g++-11")
    compile_flags = tf.sysconfig.get_compile_flags()
    link_flags = tf.sysconfig.get_link_flags()
    for stem in ("glif_state_ops", "spike_history_ops"):
        _build(
            stem, architecture, prefix, cxx, nvcc, compile_flags, link_flags
        )


if __name__ == "__main__":
    main()
