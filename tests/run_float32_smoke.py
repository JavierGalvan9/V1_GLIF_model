import os
import shutil
import subprocess
import sys
import tempfile


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAINING_SCRIPT = os.path.join(REPO_ROOT, "multi_training_single_gpu_split.py")


def resolve_data_dir():
    candidates = [
        os.path.join(REPO_ROOT, "GLIF_network_nll_full"),
        os.path.join(REPO_ROOT, "GLIF_network_nll_core"),
        os.path.join(REPO_ROOT, "GLIF_network_nll"),
        os.path.join(REPO_ROOT, "GLIF_network"),
        os.path.join(REPO_ROOT, "biorealistic-v1-model", "tiny"),
    ]
    for candidate in candidates:
        if os.path.exists(os.path.join(candidate, "network", "v1_node_types.csv")):
            return candidate
    raise RuntimeError("Could not find a V1 network fixture for the smoke test.")


DATA_DIR = resolve_data_dir()


def resolve_python():
    candidates = [
        sys.executable,
        os.path.join(REPO_ROOT, ".venv", "bin", "python"),
    ]
    for candidate in candidates:
        if candidate and os.path.exists(candidate):
            return candidate
    raise RuntimeError("Could not find a Python interpreter for the smoke test.")


def main():
    python_bin = resolve_python()
    results_dir = tempfile.mkdtemp(prefix="v1_float32_smoke_")
    try:
        command = [
            python_bin,
            TRAINING_SCRIPT,
            "--neurons", "10",
            "--batch_size", "1",
            "--steps_per_epoch", "1",
            "--n_epochs", "1",
            "--seq_len", "500",
            "--dtype", "float32",
            "--n_output", "1",
            "--neurons_per_output", "1",
            "--results_dir", results_dir,
            "--data_dir", DATA_DIR,
            "--ckpt_dir", "",
        ]
        completed = subprocess.run(
            command,
            cwd=REPO_ROOT,
            check=False,
            text=True,
        )
        if completed.returncode != 0:
            raise SystemExit(completed.returncode)
    finally:
        shutil.rmtree(results_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
