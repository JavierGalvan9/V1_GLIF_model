import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from v1_model_utils import cuda_operator_cache


class EnsureArtifactTest(unittest.TestCase):
    def test_resolves_conda_toolchain_and_records_cuda_version(self):
        with tempfile.TemporaryDirectory() as temporary:
            prefix = Path(temporary)
            nvcc = prefix / "bin" / "nvcc"
            cxx = prefix / "bin" / "x86_64-conda-linux-gnu-c++"
            runtime = prefix / "lib" / "libcudart.so.12"
            nvcc.parent.mkdir()
            nvcc.touch()
            cxx.touch()
            runtime.parent.mkdir()
            runtime.touch()

            with (
                mock.patch.object(cuda_operator_cache.sys, "prefix", str(prefix)),
                mock.patch.dict(os.environ, {}, clear=True),
                mock.patch.object(
                    cuda_operator_cache,
                    "_command_version",
                    return_value="Cuda compilation tools, release 12.9, V12.9.86",
                ),
                mock.patch.object(
                    cuda_operator_cache.tf.sysconfig,
                    "get_build_info",
                    return_value={"cuda_version": "12.5.1"},
                ),
            ):
                toolchain = cuda_operator_cache.resolve_cuda_build_toolchain("120")

            self.assertEqual(toolchain.nvcc, nvcc)
            self.assertEqual(toolchain.cxx, str(cxx))
            self.assertEqual(toolchain.library_directory, prefix / "lib")
            self.assertEqual(toolchain.cuda_version, "12.9")

    def test_rejects_pre_blackwell_compiler_for_sm120(self):
        with tempfile.TemporaryDirectory() as temporary:
            prefix = Path(temporary)
            nvcc = prefix / "bin" / "nvcc"
            nvcc.parent.mkdir()
            nvcc.touch()
            with (
                mock.patch.object(cuda_operator_cache.sys, "prefix", str(prefix)),
                mock.patch.dict(os.environ, {}, clear=True),
                mock.patch.object(
                    cuda_operator_cache,
                    "_command_version",
                    return_value="Cuda compilation tools, release 12.7, V12.7.0",
                ),
            ):
                with self.assertRaisesRegex(RuntimeError, "CUDA 12.8 or newer"):
                    cuda_operator_cache.resolve_cuda_build_toolchain("120")

    def test_passes_requested_target_to_multi_artifact_builder(self):
        with tempfile.TemporaryDirectory() as temporary:
            cache_root = Path(temporary) / "cache"
            source = Path(temporary) / "kernel.cc"
            source.write_text("kernel", encoding="utf-8")
            directory = Path(temporary) / "operator"
            directory.mkdir()

            def build(command, check):
                self.assertTrue(check)
                self.assertEqual(command[-2:], ["--stem", "spike_history_ops"])
                output = cuda_operator_cache.artifact_path(
                    directory, "spike_history_ops", "89"
                )
                output.write_bytes(b"library")

            environment = {
                "architecture": "89",
                "tensorflow": "test",
                "tensorflow_cuda": "test",
                "tensorflow_cudnn": "test",
                "compile_flags": [],
                "link_flags": [],
            }
            with (
                mock.patch.object(cuda_operator_cache, "cuda_cache_root", return_value=cache_root),
                mock.patch.object(cuda_operator_cache, "build_environment", return_value=environment),
                mock.patch.object(cuda_operator_cache.subprocess, "run", side_effect=build) as run,
            ):
                output = cuda_operator_cache.ensure_artifact(
                    directory,
                    "spike_history_ops",
                    sources=(source,),
                    build_module="test.builder",
                    architecture="89",
                    build_args=("--stem", "spike_history_ops"),
                )

            self.assertTrue(output.exists())
            self.assertEqual(run.call_count, 1)
            metadata = json.loads(
                output.with_name("spike_history_ops_metadata.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertIn("source_digest", metadata)


if __name__ == "__main__":
    unittest.main()
