import json
import unittest
from unittest import mock

from v1_model_utils import training_orchestration


class TrainingOrchestrationTests(unittest.TestCase):
    def test_single_gpu_keeps_the_current_process_and_per_replica_batch(self):
        plan = training_orchestration.plan_training_launch(
            [
                "multi_training.py",
                "--batch_size=8",
                "--grating_batch_size=4",
                "--gray_batch_size=4",
            ],
            environ={"CUDA_VISIBLE_DEVICES": "3,5"},
            compute_capabilities=("8.6", "8.6"),
        )

        self.assertFalse(plan.launch_workers)
        self.assertEqual(plan.n_gpus, 1)
        self.assertEqual(plan.per_replica_batch_size, 8)

    def test_single_gpu_limits_tensorflow_to_the_first_visible_device(self):
        environ = {"CUDA_VISIBLE_DEVICES": "3,5"}

        result, plan = training_orchestration.maybe_launch_training_workers(
            ["multi_training.py", "--n_gpus=1"], environ=environ
        )

        self.assertIsNone(result)
        self.assertEqual(plan.visible_devices, ("3",))
        self.assertEqual(environ["CUDA_VISIBLE_DEVICES"], "3")

    def test_global_batch_is_split_and_preserves_stimulus_ratio(self):
        plan = training_orchestration.plan_training_launch(
            [
                "multi_training.py",
                "--n_gpus=2",
                "--global_batch_size=8",
                "--grating_batch_size=1",
                "--gray_batch_size=1",
            ],
            environ={"CUDA_VISIBLE_DEVICES": "3,5"},
            compute_capabilities=("8.6", "8.6"),
        )

        # Splitting the batch is independent of how the replicas are executed.
        self.assertFalse(plan.launch_workers)
        self.assertEqual(plan.per_replica_batch_size, 4)
        self.assertEqual(plan.grating_batch_size, 2)
        self.assertEqual(plan.gray_batch_size, 2)
        self.assertEqual(plan.visible_devices, ("3", "5"))

    def test_multi_gpu_stays_in_one_process_on_every_architecture(self):
        """Per-device CSR resources removed the reason to fork per GPU."""
        for capability in ("8.0", "8.6", "8.9", "12.0"):
            with self.subTest(compute_capability=capability):
                plan = training_orchestration.plan_training_launch(
                    ["multi_training.py", "--n_gpus=2", "--global_batch_size=8"],
                    environ={"CUDA_VISIBLE_DEVICES": "3,5"},
                    compute_capabilities=(capability, capability),
                )
                self.assertEqual(plan.mode, "mirrored")
                self.assertFalse(plan.launch_workers)

    def test_multi_worker_mode_forks_one_process_per_gpu(self):
        plan = training_orchestration.plan_training_launch(
            [
                "multi_training.py",
                "--n_gpus=2",
                "--global_batch_size=8",
                "--distributed_mode=multi_worker",
            ],
            environ={"CUDA_VISIBLE_DEVICES": "3,5"},
            compute_capabilities=("8.6", "8.6"),
        )

        self.assertTrue(plan.launch_workers)
        self.assertEqual(len(plan.worker_addresses), 2)

    def test_every_requested_gpu_stays_visible_in_one_process(self):
        environ = {"CUDA_VISIBLE_DEVICES": "3,5"}

        with mock.patch.object(
            training_orchestration,
            "_query_compute_capabilities",
            return_value=("12.0", "12.0"),
        ):
            result, plan = training_orchestration.maybe_launch_training_workers(
                ["multi_training.py", "--n_gpus=2"], environ=environ
            )

        self.assertIsNone(result)
        self.assertFalse(plan.launch_workers)
        self.assertEqual(environ["CUDA_VISIBLE_DEVICES"], "3,5")

    def test_global_batch_must_be_divisible_by_worker_count(self):
        with self.assertRaisesRegex(ValueError, "divisible"):
            training_orchestration.plan_training_launch(
                ["multi_training.py", "--n_gpus=2", "--global_batch_size=7"],
                environ={"CUDA_VISIBLE_DEVICES": "0,1"},
                compute_capabilities=("8.6", "8.6"),
            )

    def test_requested_gpu_count_cannot_exceed_visible_devices(self):
        with self.assertRaisesRegex(ValueError, "2 GPU.*visible"):
            training_orchestration.plan_training_launch(
                ["multi_training.py", "--n_gpus=2"],
                environ={"CUDA_VISIBLE_DEVICES": "4"},
            )

    def test_worker_commands_use_one_visible_gpu_and_shared_cluster(self):
        plan = training_orchestration.plan_training_launch(
            [
                "multi_training.py",
                "--n_gpus", "2",
                "--batch_size", "4",
                "--grating_batch_size", "2",
                "--gray_batch_size", "2",
            ],
            environ={"CUDA_VISIBLE_DEVICES": "4,7,8"},
            worker_addresses=("127.0.0.1:31001", "127.0.0.1:31002"),
            compute_capabilities=("8.6", "8.6"),
        )
        self.assertEqual(plan.compute_capability, "86")

        workers = training_orchestration.build_worker_processes(
            plan, base_environ={"EXAMPLE": "kept"}, enabled=True
        )
        self.assertEqual([worker.environ["CUDA_VISIBLE_DEVICES"] for worker in workers], ["4", "7"])
        self.assertEqual(workers[0].environ["NCCL_P2P_DISABLE"], "1")
        self.assertEqual(workers[0].environ["NCCL_DEBUG"], "WARN")
        self.assertEqual([worker.is_chief for worker in workers], [True, False])
        self.assertEqual(workers[0].argv.count("--distributed_worker_index=0"), 1)
        self.assertIn("--batch_size=4", workers[1].argv)
        cluster = json.loads(workers[1].environ["TF_CONFIG"])
        self.assertEqual(cluster["cluster"]["worker"], ["127.0.0.1:31001", "127.0.0.1:31002"])
        self.assertEqual(cluster["task"], {"type": "worker", "index": 1})

    def test_worker_mode_never_recursively_launches(self):
        plan = training_orchestration.plan_training_launch(
            ["multi_training.py", "--n_gpus=2", "--distributed_worker_index=1"],
            environ={"CUDA_VISIBLE_DEVICES": "7", "TF_CONFIG": "{}"},
        )

        self.assertFalse(plan.launch_workers)
        self.assertFalse(plan.is_chief)

    def test_launcher_propagates_failure_and_terminates_remaining_worker(self):
        plan = training_orchestration.plan_training_launch(
            ["multi_training.py", "--n_gpus=2"],
            environ={"CUDA_VISIBLE_DEVICES": "0,1"},
            worker_addresses=("127.0.0.1:31001", "127.0.0.1:31002"),
            compute_capabilities=("8.6", "8.6"),
        )
        failed = mock.Mock()
        failed.poll.side_effect = [None, 9]
        failed.wait.return_value = 9
        running = mock.Mock()
        running.poll.return_value = None
        running.wait.return_value = -15

        with mock.patch.object(
            training_orchestration.subprocess,
            "Popen",
            side_effect=[failed, running],
        ):
            exit_code = training_orchestration.launch_worker_processes(
                plan, enabled=True
            )

        self.assertEqual(exit_code, 9)
        running.terminate.assert_called_once_with()

    def test_mixed_compute_capabilities_are_rejected_before_launch(self):
        with self.assertRaisesRegex(ValueError, "same compute capability"):
            training_orchestration.plan_training_launch(
                ["multi_training.py", "--n_gpus=2"],
                environ={"CUDA_VISIBLE_DEVICES": "0,1"},
                compute_capabilities=("8.6", "8.9"),
            )


if __name__ == "__main__":
    unittest.main()
