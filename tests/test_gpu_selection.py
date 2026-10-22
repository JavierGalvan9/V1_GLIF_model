"""CPU-only checks of the PCIe-balanced GPU auto-selection and its launch hook.

``nvidia-smi`` and PCI sysfs are mocked throughout; nothing touches a GPU.
"""

from dataclasses import replace
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

from v1_model_utils import gpu_selection, training_orchestration
from v1_model_utils.gpu_selection import (
    GPU, GPUSelectionError, auto_select_gpus, normalize_bdf, pci_ancestry, select,
)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL = "NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition"
ROOTS = ("0000:40:01.1", "0000:c0:01.1")


def gpu(index, root, **changes):
    item = GPU(index, f"GPU-{index}", f"0000:{index:02x}:00.0", MODEL,
               97887, 97340, 0, "12.0", root, (root, f"0000:{index:02x}:00.0"), False)
    return replace(item, **changes)


def host():
    """This host: RTX PRO 6000 at 0,1,3 | 5,6,8,9 and L40S at 2,4 | 7."""
    l40s = {"name": "NVIDIA L40S", "compute_cap": "8.9", "total_mib": 46068,
            "free_mib": 45540}
    return [gpu(i, ROOTS[i >= 5], **(l40s if i in (2, 4, 7) else {})) for i in range(10)]


def unconstrained_environ():
    return {key: value for key, value in os.environ.items()
            if key not in (*gpu_selection.MASKS, *gpu_selection.SCHEDULERS)}


def indices(selected):
    return [item.index for item in selected]


class SelectionTests(unittest.TestCase):
    def test_four_balances_instead_of_first_four_three_plus_one(self):
        devices = [gpu(i, ROOTS[i >= 3]) for i in range(7)]
        self.assertEqual(indices(select(devices, 4)), [0, 3, 1, 4])

    def test_host_proposals_spread_replicas_and_skip_l40s(self):
        self.assertEqual(indices(select(host(), 1)), [0])
        self.assertEqual(indices(select(host(), 2)), [0, 5])
        self.assertEqual(indices(select(host(), 4)), [0, 5, 1, 6])
        self.assertEqual(indices(select(host(), 7)), [0, 5, 1, 6, 3, 8, 9])

    def test_uneven_capacity_uses_remaining_root(self):
        selected = select([gpu(0, "a"), *[gpu(i, "b") for i in range(1, 5)]], 4)
        self.assertEqual([item.root_port for item in selected].count("a"), 1)

    def test_busy_memory_utilization_and_other_models_excluded(self):
        devices = [gpu(0, "a", compute_busy=True), gpu(1, "a", utilization=1),
                   gpu(2, "a", free_mib=100), gpu(3, "a", name="NVIDIA L40S",
                                                  compute_cap="8.9"), gpu(4, "b")]
        self.assertEqual(indices(select(devices, 1)), [4])
        with self.assertRaisesRegex(GPUSelectionError, "found 1"):
            select(devices, 2)

    def test_too_few_target_model_never_falls_back_to_a_lesser_card(self):
        devices = [replace(item, compute_busy=item.name == MODEL) for item in host()]
        with self.assertRaisesRegex(GPUSelectionError, "Need 1 idle .*RTX PRO 6000.*found 0"):
            select(devices, 1)
        self.assertEqual(indices(select(devices, 1, model="NVIDIA L40S")), [2])

    def test_invalid_counts(self):
        for count in (0, -1):
            with self.assertRaises(ValueError):
                select([gpu(0, "a")], count)

    def test_ancestry_groups_different_switch_branches_under_same_root(self):
        with tempfile.TemporaryDirectory() as folder:
            base = Path(folder)
            sysfs = base / "devices"
            sysfs.mkdir()
            for bus, branch in (("45", "42"), ("4e", "4b")):
                target = base / "pci0000:40" / "0000:40:01.1" / f"0000:{branch}:00.0" / f"0000:{bus}:00.0"
                target.mkdir(parents=True)
                (sysfs / f"0000:{bus}:00.0").symlink_to(target)
            a = pci_ancestry("00000000:45:00.0", sysfs)
            b = pci_ancestry("00000000:4E:00.0", sysfs)
            self.assertEqual(a[0], b[0])
            self.assertNotEqual(a[1], b[1])

    def test_bdf_normalization_and_missing_topology(self):
        self.assertEqual(normalize_bdf("00000000:CE:00.0"), "0000:ce:00.0")
        with tempfile.TemporaryDirectory() as folder, self.assertRaises(ValueError):
            pci_ancestry("0000:00:00.0", Path(folder))

    def test_discover_parses_nvidia_smi_and_marks_compute_processes(self):
        with tempfile.TemporaryDirectory() as folder:
            base = Path(folder)
            sysfs = base / "devices"
            sysfs.mkdir()
            target = base / "0000:40:01.1" / "0000:45:00.0"
            target.mkdir(parents=True)
            (sysfs / "0000:45:00.0").symlink_to(target)
            outputs = {
                "--query-gpu": f"0, GPU-a, 00000000:45:00.0, {MODEL}, 97887, 97340, 0, 12.0\n",
                "--query-compute-apps": "GPU-a\n",
            }
            inventory = gpu_selection.discover(
                lambda arguments: outputs[arguments[0].split("=")[0]], sysfs)
        self.assertEqual(inventory, [GPU(0, "GPU-a", "0000:45:00.0", MODEL, 97887, 97340, 0,
                                         "12.0", "0000:40:01.1",
                                         ("0000:40:01.1", "0000:45:00.0"), True)])


class AutoSelectionTests(unittest.TestCase):
    def test_selection_exposes_uuids_in_pci_order(self):
        environ = {}
        with mock.patch.object(gpu_selection, "discover", return_value=host()):
            selected = auto_select_gpus(2, environ)
        self.assertEqual(indices(selected), [0, 5])
        self.assertEqual(environ, {"CUDA_DEVICE_ORDER": "PCI_BUS_ID",
                                   "CUDA_VISIBLE_DEVICES": "GPU-0,GPU-5"})

    def test_forced_link_sharing_is_reported(self):
        devices = [replace(item, compute_busy=item.index >= 5) for item in host()]
        with mock.patch.object(gpu_selection, "discover", return_value=devices), \
                mock.patch("builtins.print") as printed:
            self.assertEqual(indices(auto_select_gpus(2, {})), [0, 1])
        self.assertIn("share a PCIe root port", str(printed.call_args_list))
        with mock.patch.object(gpu_selection, "discover", return_value=host()), \
                mock.patch("builtins.print") as printed:
            auto_select_gpus(2, {})
        self.assertNotIn("WARNING", str(printed.call_args_list))

    def test_explicit_masks_and_scheduler_jobs_win_even_when_empty(self):
        for key in (*gpu_selection.MASKS, *gpu_selection.SCHEDULERS):
            for value in ("", "0,1", "GPU-0,GPU-1"):
                environ = {key: value}
                with self.subTest(key=key, value=value), \
                        mock.patch.object(gpu_selection, "discover") as discover:
                    self.assertIsNone(auto_select_gpus(1, environ))
                    discover.assert_not_called()
                    self.assertEqual(environ, {key: value})

    def test_missing_nvidia_smi_leaves_cpu_hosts_alone(self):
        environ = {}
        with mock.patch.object(gpu_selection, "discover", side_effect=FileNotFoundError):
            self.assertIsNone(auto_select_gpus(1, environ))
        self.assertEqual(environ, {})

    def test_unresolved_topology_fails_closed(self):
        with mock.patch.object(gpu_selection, "discover", side_effect=ValueError("sysfs")), \
                self.assertRaisesRegex(GPUSelectionError, "sysfs"):
            auto_select_gpus(1, {})


class LaunchHookTests(unittest.TestCase):
    def _launch(self, *arguments, environ):
        with mock.patch.object(gpu_selection, "discover", return_value=host()), \
                mock.patch.object(training_orchestration, "_query_compute_capabilities",
                                  side_effect=lambda devices: ("12.0",) * len(devices)):
            return training_orchestration.maybe_launch_training_workers(
                ("multi_training.py", *arguments), environ=environ)[1]

    def test_default_on_selects_n_gpus_before_the_plan(self):
        environ = {}
        plan = self._launch("--n_gpus=4", "--global_batch_size=8", environ=environ)
        self.assertEqual(plan.visible_devices, ("GPU-0", "GPU-5", "GPU-1", "GPU-6"))
        self.assertEqual(environ["CUDA_VISIBLE_DEVICES"], "GPU-0,GPU-5,GPU-1,GPU-6")

    def test_explicit_mask_disables_selection(self):
        environ = {"CUDA_VISIBLE_DEVICES": "3,5"}
        plan = self._launch("--n_gpus=2", environ=environ)
        self.assertEqual(plan.visible_devices, ("3", "5"))
        self.assertEqual(environ, {"CUDA_VISIBLE_DEVICES": "3,5"})

    def test_flag_disables_selection(self):
        for flag in ("--noauto_select_gpus", "--auto_select_gpus=false"):
            with self.subTest(flag=flag), \
                    mock.patch.object(gpu_selection, "auto_select_gpus") as selection:
                self._launch(flag, environ={"CUDA_VISIBLE_DEVICES": "0"})
                selection.assert_not_called()
        self.assertTrue(training_orchestration._auto_select_gpus_enabled(
            ("x", "--noauto_select_gpus", "--auto_select_gpus")))

    def test_workers_never_reselect(self):
        with mock.patch.object(gpu_selection, "auto_select_gpus") as selection:
            self._launch("--n_gpus=2", "--distributed_worker_index=1",
                         environ={"CUDA_VISIBLE_DEVICES": "GPU-5"})
        selection.assert_not_called()

    def test_hook_runs_before_tensorflow_is_imported(self):
        """Run multi_training's module prologue with selection mocked."""
        script = """
import runpy, sys, absl.app, absl.flags
from v1_model_utils import gpu_selection
from test_gpu_selection import host
def discover():
    assert "tensorflow" not in sys.modules, "TensorFlow imported before GPU selection"
    return host()
gpu_selection.discover = discover
def parse_only(main, argv=None):
    import os
    absl.flags.FLAGS(sys.argv)
    assert absl.flags.FLAGS.auto_select_gpus
    print("VISIBLE=" + os.environ["CUDA_VISIBLE_DEVICES"])
    raise SystemExit(0)
absl.app.run = parse_only
sys.argv = ["multi_training.py", "--n_gpus=1"]
runpy.run_path("multi_training.py", run_name="__main__")
"""
        with tempfile.TemporaryDirectory() as cache:
            environ = dict(unconstrained_environ(), V1_CUDA_CACHE_DIR=cache,
                           PYTHONPATH=os.pathsep.join((REPO_ROOT, os.path.join(REPO_ROOT, "tests"))))
            result = subprocess.run([sys.executable, "-c", script], cwd=REPO_ROOT, env=environ,
                                    capture_output=True, text=True, timeout=600)
        self.assertEqual(result.returncode, 0, result.stderr[-3000:])
        self.assertIn("VISIBLE=GPU-0", result.stdout)


if __name__ == "__main__":
    unittest.main()
