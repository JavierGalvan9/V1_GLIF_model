"""Print reproducibility-critical TensorFlow, CUDA, and host GPU versions."""

import json
import platform
import subprocess
import sys

import keras
import numpy as np
import tensorflow as tf


def _command(*arguments):
    result = subprocess.run(arguments, check=False, capture_output=True, text=True)
    return result.stdout.strip() or result.stderr.strip()


def main():
    build = tf.sysconfig.get_build_info()
    devices = tf.config.list_physical_devices("GPU")
    report = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "tensorflow": tf.__version__,
        "keras": keras.__version__,
        "numpy": np.__version__,
        "tensorflow_cuda": build.get("cuda_version"),
        "tensorflow_cudnn": build.get("cudnn_version"),
        "tensorflow_compute_capabilities": build.get("cuda_compute_capabilities"),
        "nvcc": _command(sys.prefix + "/bin/nvcc", "--version"),
        "driver_and_gpus": _command(
            "nvidia-smi",
            "--query-gpu=driver_version,name,compute_cap,memory.total",
            "--format=csv,noheader",
        ),
        "visible_gpus": [
            tf.config.experimental.get_device_details(device) for device in devices
        ],
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
