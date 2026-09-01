"""Build the GPU-local CSR resource operators for one CUDA architecture."""

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
PACKAGE = HERE.parent
BUILD_FLAGS = (
    "-DV1_THREADS=128", "-DV1_BATCH32_TILE=32", "-DV1_WARP_REDUCTION=0",
    "-DV1_HALF2_PROJECTION=1", "-DV1_WARP_PER_ROW=1", "-DV1_PREPROJECT=0",
    "-DV1_PAIR_PREPROJECT=0", "-DV1_FORWARD_THREADS=128",
    "-DV1_FORWARD_HALF2_ATOMICS=1", "-DV1_FORWARD_FLOAT_ACCUM=0",
    "-DV1_FORWARD_GROUPED=1", "--expt-relaxed-constexpr", "--use_fast_math",
)


def _run(command):
    print(shlex.join(str(part) for part in command), flush=True)
    subprocess.run([str(part) for part in command], check=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--architecture", default=os.environ.get("V1_CUDA_ARCH"))
    args = parser.parse_args()
    architecture = normalize_architecture(
        args.architecture or active_gpu_architecture()
    )
    prefix = Path(sys.prefix)
    nvcc = prefix / "bin/nvcc"
    cxx = os.environ.get("CXX", "g++-11")
    if not nvcc.exists():
        raise FileNotFoundError(f"CUDA compiler not found at {nvcc}")
    compile_flags = tf.sysconfig.get_compile_flags()
    link_flags = tf.sysconfig.get_link_flags()
    cuda_include = Path(tf.sysconfig.get_include()) / "third_party/gpus/cuda/include"
    output = artifact_path(HERE, "csr_resource_ops", architecture)
    with temporary_build_directory(
        HERE, "csr_resource_ops", architecture
    ) as workspace:
        registration_object = workspace / "csr_resource_ops.cc.o"
        kernels_object = workspace / "csr_resource_ops.cu.o"
        temporary_output = workspace / output.name
        _run([cxx, "-std=c++17", "-O3", "-fPIC", "-c", HERE / "csr_resource_ops.cc", "-o", registration_object, *compile_flags])
        _run([
            nvcc, "-std=c++17", "-O3", "-x", "cu", "-c",
            HERE / "csr_resource_ops.cu.cc", "-o", kernels_object,
            "-DGOOGLE_CUDA=1", *BUILD_FLAGS,
            f"-gencode=arch=compute_{architecture},code=sm_{architecture}",
            f"-gencode=arch=compute_{architecture},code=compute_{architecture}",
            "-Xcompiler=-fPIC", f"-I{cuda_include}", *compile_flags,
        ])
        _run([cxx, "-shared", "-O3", registration_object,
              kernels_object, "-o", temporary_output, *link_flags, f"-L{prefix / 'lib'}",
              "-l:libcudart.so.12", f"-Wl,-rpath,{prefix / 'lib'}"])
        temporary_output.replace(output)
    write_build_metadata(
        HERE, "csr_resource_ops", architecture, build_flags=BUILD_FLAGS
    )
    print(output)


if __name__ == "__main__":
    main()
