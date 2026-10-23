"""Memory reporting follows TensorFlow GPU identity, not host NVML indices."""

import ctypes.util
from types import SimpleNamespace
from unittest import mock

import pytest

from v1_model_utils import callbacks


@pytest.fixture(autouse=True)
def clear_handle_cache():
    callbacks._nvml_device_handle.cache_clear()
    yield
    callbacks._nvml_device_handle.cache_clear()


def test_memory_uses_cuda_pci_identity(monkeypatch):
    monkeypatch.setattr(callbacks.tf.config, "get_visible_devices", lambda kind: [
        SimpleNamespace(name="/physical_device:GPU:3"),
    ])
    monkeypatch.setattr(callbacks.tf.config, "get_logical_device_configuration", lambda device: None)
    runtime = mock.Mock()

    def pci_id(buffer, size, ordinal):
        assert ordinal == 3
        buffer.value = b"0000:c5:00.0"
        return 0

    runtime.cudaDeviceGetPCIBusId.side_effect = pci_id
    # No GPU or NVML calls are made by this test.
    monkeypatch.setattr(ctypes.util, "find_library", lambda name: "fake-cudart")
    monkeypatch.setattr(ctypes, "CDLL", lambda path: runtime)
    monkeypatch.setattr(callbacks, "_nvml_device_count", lambda: 10)
    handle_by_bus = mock.Mock(return_value="correct-device")
    monkeypatch.setattr(callbacks.pynvml, "nvmlDeviceGetHandleByPciBusId", handle_by_bus)
    memory = mock.Mock(return_value=SimpleNamespace(used=5 * 2**30, free=7 * 2**30, total=12 * 2**30))
    monkeypatch.setattr(callbacks.pynvml, "nvmlDeviceGetMemoryInfo", memory)
    wrong_handle = mock.Mock(return_value="wrong-device")
    monkeypatch.setattr(callbacks.pynvml, "nvmlDeviceGetHandleByIndex", wrong_handle)

    assert callbacks.device_memory_gib(0) == (5, 7, 12)
    handle_by_bus.assert_called_once_with("0000:c5:00.0")
    memory.assert_called_once_with("correct-device")
    wrong_handle.assert_not_called()
    assert callbacks.device_memory_gib(0) == (5, 7, 12)
    handle_by_bus.assert_called_once()


@pytest.mark.parametrize("gpu_id", [-1, 0, 1])
def test_unavailable_logical_gpu_is_not_replaced_by_host_gpu_zero(monkeypatch, gpu_id):
    monkeypatch.setattr(callbacks.tf.config, "get_visible_devices", lambda kind: [])
    memory = mock.Mock()
    monkeypatch.setattr(callbacks.pynvml, "nvmlDeviceGetMemoryInfo", memory)
    assert callbacks.device_memory_gib(gpu_id) is None
    memory.assert_not_called()
