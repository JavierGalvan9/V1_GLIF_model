"""Build the external-current TensorFlow CUDA weight-backward operator."""

import argparse
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time

import tensorflow as tf


HERE = Path(__file__).resolve().parent


def _run(command):
    print(shlex.join(str(part) for part in command), flush=True)
    subprocess.run([str(part) for part in command], check=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--architecture",
        default=os.environ.get("V1_CUDA_ARCH", "120"),
        help="CUDA compute capability without a decimal point (default: 120)",
    )
    args = parser.parse_args()
    prefix = Path(sys.prefix)
    nvcc = prefix / "bin/nvcc"
    cxx = os.environ.get("CXX", "g++-11")
    if not nvcc.exists():
        raise FileNotFoundError(f"CUDA compiler not found at {nvcc}")
    compile_flags = tf.sysconfig.get_compile_flags()
    link_flags = tf.sysconfig.get_link_flags()
    cuda_include = Path(tf.sysconfig.get_include()) / "third_party/gpus/cuda/include"
    cc_object = HERE / "external_current_ops.cc.o"
    cu_object = HERE / "external_current_ops.cu.o"
    output = HERE / "_external_current_ops.so"
    started = time.perf_counter()
    _run(
        [
            cxx,
            "-std=c++17",
            "-O3",
            "-fPIC",
            "-c",
            HERE / "external_current_ops.cc",
            "-o",
            cc_object,
            *compile_flags,
        ]
    )
    _run(
        [
            nvcc,
            "-std=c++17",
            "-O3",
            "-x",
            "cu",
            "-c",
            HERE / "external_current_ops.cu.cc",
            "-o",
            cu_object,
            "-DGOOGLE_CUDA=1",
            "--expt-relaxed-constexpr",
            "--use_fast_math",
            f"-gencode=arch=compute_{args.architecture},code=sm_{args.architecture}",
            f"-gencode=arch=compute_{args.architecture},code=compute_{args.architecture}",
            "-Xcompiler=-fPIC",
            f"-I{cuda_include}",
            *compile_flags,
        ]
    )
    _run(
        [
            cxx,
            "-shared",
            "-O3",
            cc_object,
            cu_object,
            "-o",
            output,
            *link_flags,
            f"-L{prefix / 'lib'}",
            "-l:libcudart.so.12",
            f"-Wl,-rpath,{prefix / 'lib'}",
        ]
    )
    (HERE / "build_metadata.txt").write_text(
        f"tensorflow={tf.__version__}\narchitecture={args.architecture}\n"
        f"seconds={time.perf_counter() - started:.6f}\n"
    )
    print(output)


if __name__ == "__main__":
    main()
