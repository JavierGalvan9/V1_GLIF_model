import os
from unittest import mock

from v1_model_utils import cuda_caches


_MANAGED = (
    "CUDA_CACHE_MAXSIZE", "CUDA_DEVICE_ORDER", "XLA_FLAGS", "V1_CUDA_CACHE_DIR",
)


def _clean_env(tmp_path, **overrides):
    env = {
        key: value for key, value in os.environ.items() if key not in _MANAGED
    }
    env["V1_CUDA_CACHE_DIR"] = str(tmp_path)
    env.update(overrides)
    return env


def test_pins_device_order_and_jit_budget(tmp_path):
    with mock.patch.dict(os.environ, _clean_env(tmp_path), clear=True), \
            mock.patch.object(cuda_caches, "_visible_gpu_key", return_value="gpu"):
        cuda_caches.configure_cuda_caches()
        assert os.environ["CUDA_DEVICE_ORDER"] == "PCI_BUS_ID"
        assert os.environ["CUDA_CACHE_MAXSIZE"] == str(2 * 1024**3)


def test_first_run_only_dumps_then_later_runs_also_load(tmp_path):
    results = tmp_path / "xla_autotune" / "gpu.txtpb"
    with mock.patch.dict(os.environ, _clean_env(tmp_path), clear=True), \
            mock.patch.object(cuda_caches, "_visible_gpu_key", return_value="gpu"):
        cuda_caches.configure_cuda_caches()
        # Nothing measured yet: loading a missing file is an XLA error.
        assert os.environ["XLA_FLAGS"] == (
            f"--xla_gpu_dump_autotune_results_to={results}"
        )

    results.parent.mkdir(parents=True, exist_ok=True)
    results.write_text("version: 3\n")
    with mock.patch.dict(os.environ, _clean_env(tmp_path), clear=True), \
            mock.patch.object(cuda_caches, "_visible_gpu_key", return_value="gpu"):
        cuda_caches.configure_cuda_caches()
        assert os.environ["XLA_FLAGS"] == (
            f"--xla_gpu_load_autotune_results_from={results} "
            f"--xla_gpu_dump_autotune_results_to={results}"
        )


def test_cache_file_is_keyed_by_visible_gpu(tmp_path):
    listing = "NVIDIA L40S\nNVIDIA RTX PRO 6000\nNVIDIA L40S\n"
    completed = mock.Mock(stdout=listing)
    env = _clean_env(tmp_path, CUDA_VISIBLE_DEVICES="1")
    with mock.patch.dict(os.environ, env, clear=True), \
            mock.patch.object(cuda_caches.subprocess, "run", return_value=completed):
        cuda_caches.configure_cuda_caches()
        assert "NVIDIA-RTX-PRO-6000.txtpb" in os.environ["XLA_FLAGS"]
        assert "L40S" not in os.environ["XLA_FLAGS"]


def test_missing_nvidia_smi_falls_back(tmp_path):
    with mock.patch.dict(os.environ, _clean_env(tmp_path), clear=True), \
            mock.patch.object(cuda_caches.subprocess, "run", side_effect=OSError):
        cuda_caches.configure_cuda_caches()
        assert "unknown-gpu.txtpb" in os.environ["XLA_FLAGS"]


def test_caller_settings_win(tmp_path):
    env = _clean_env(
        tmp_path,
        CUDA_DEVICE_ORDER="FASTEST_FIRST",
        CUDA_CACHE_MAXSIZE="17",
        XLA_FLAGS="--xla_gpu_dump_autotune_results_to=/somewhere/else",
    )
    with mock.patch.dict(os.environ, env, clear=True):
        cuda_caches.configure_cuda_caches()
        assert os.environ["CUDA_DEVICE_ORDER"] == "FASTEST_FIRST"
        assert os.environ["CUDA_CACHE_MAXSIZE"] == "17"
        assert os.environ["XLA_FLAGS"] == (
            "--xla_gpu_dump_autotune_results_to=/somewhere/else"
        )
