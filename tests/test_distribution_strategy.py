import unittest
from unittest import mock

from v1_model_utils import tf_utils
from v1_model_utils import training_orchestration as orchestration


class DistributionStrategyTests(unittest.TestCase):
    def setUp(self):
        # create_distribution_strategy declares the connectivity backend in the
        # process environment, so keep that out of the rest of the suite.
        patcher = mock.patch.dict(tf_utils.os.environ, {}, clear=False)
        patcher.start()
        self.addCleanup(patcher.stop)
        tf_utils.os.environ.pop("V1_CSR_RESOURCE_MODE", None)

    def test_multi_gpu_uses_single_process_nccl(self):
        strategy = object()
        with mock.patch.object(
            tf_utils.tf.distribute,
            "NcclAllReduce",
            return_value="nccl",
        ), mock.patch.object(
            tf_utils.tf.distribute,
            "MirroredStrategy",
            return_value=strategy,
        ) as constructor:
            actual = tf_utils.create_distribution_strategy(
                devices=["GPU:0", "GPU:1"],
            )

        self.assertIs(actual, strategy)
        constructor.assert_called_once_with(cross_device_ops="nccl")

    def test_multi_gpu_does_not_depend_on_compute_capability(self):
        """Per-device CSR resources removed the Blackwell-only restriction."""
        with mock.patch.object(
            tf_utils.tf.config.experimental, "get_device_details"
        ) as details, mock.patch.object(
            tf_utils.tf.distribute, "NcclAllReduce", return_value="nccl"
        ), mock.patch.object(
            tf_utils.tf.distribute, "MirroredStrategy", return_value=object()
        ):
            tf_utils.create_distribution_strategy(
                devices=["GPU:0", "GPU:1"],
            )

        details.assert_not_called()

    def test_multi_gpu_declares_device_local_connectivity(self):
        """Replicas share one graph, so metadata must live per device."""
        with mock.patch.object(
            tf_utils.tf.distribute, "NcclAllReduce", return_value="nccl"
        ), mock.patch.object(
            tf_utils.tf.distribute, "MirroredStrategy", return_value=object()
        ):
            tf_utils.create_distribution_strategy(devices=["GPU:0", "GPU:1"])

        self.assertEqual(tf_utils.os.environ["V1_CSR_RESOURCE_MODE"], "1")

    def test_multi_gpu_respects_an_explicit_connectivity_setting(self):
        tf_utils.os.environ["V1_CSR_RESOURCE_MODE"] = "0"
        with mock.patch.object(
            tf_utils.tf.distribute, "NcclAllReduce", return_value="nccl"
        ), mock.patch.object(
            tf_utils.tf.distribute, "MirroredStrategy", return_value=object()
        ):
            tf_utils.create_distribution_strategy(devices=["GPU:0", "GPU:1"])

        self.assertEqual(tf_utils.os.environ["V1_CSR_RESOURCE_MODE"], "0")

    def test_single_gpu_leaves_the_tensor_connectivity_backend(self):
        with mock.patch.object(
            tf_utils.tf.distribute, "MirroredStrategy", return_value=object()
        ):
            tf_utils.create_distribution_strategy(devices=["GPU:0"])

        self.assertNotIn("V1_CSR_RESOURCE_MODE", tf_utils.os.environ)

    def test_single_gpu_uses_mirrored(self):
        strategy = object()
        with mock.patch.object(
            tf_utils.tf.distribute, "MirroredStrategy", return_value=strategy
        ) as constructor:
            actual = tf_utils.create_distribution_strategy(
                devices=["GPU:0"],
            )

        self.assertIs(actual, strategy)
        constructor.assert_called_once_with()

    def test_worker_uses_multi_worker_nccl(self):
        strategy = object()
        with mock.patch.object(
            tf_utils.tf.distribute,
            "MultiWorkerMirroredStrategy",
            return_value=strategy,
        ) as constructor:
            actual = tf_utils.create_distribution_strategy(multi_worker=True)

        self.assertIs(actual, strategy)
        options = constructor.call_args.kwargs["communication_options"]
        self.assertEqual(
            options.implementation,
            tf_utils.tf.distribute.experimental.CommunicationImplementation.NCCL,
        )


class LaunchPlanTests(unittest.TestCase):
    ARGV = ("multi_training.py", "--n_gpus", "2", "--global_batch_size", "8")

    def _plan(self, *extra, capabilities=("12.0", "12.0")):
        return orchestration.plan_training_launch(
            self.ARGV + extra,
            environ={"CUDA_VISIBLE_DEVICES": "0,1"},
            compute_capabilities=capabilities,
        )

    def test_mirrored_is_the_default_and_forks_nothing(self):
        plan = self._plan()
        self.assertEqual(plan.mode, "mirrored")
        self.assertFalse(plan.launch_workers)
        self.assertEqual(plan.worker_addresses, ())
        self.assertEqual(plan.per_replica_batch_size, 4)

    def test_multi_worker_is_opt_in_and_reserves_addresses(self):
        plan = self._plan("--distributed_mode", "multi_worker")
        self.assertTrue(plan.launch_workers)
        self.assertEqual(len(plan.worker_addresses), 2)

    def test_unknown_mode_is_rejected(self):
        with self.assertRaises(ValueError):
            self._plan("--distributed_mode", "hierarchical")

    def test_ampere_workers_disable_peer_access(self):
        plan = self._plan(
            "--distributed_mode", "multi_worker", capabilities=("8.6", "8.6")
        )
        workers = orchestration.build_worker_processes(plan, base_environ={})
        self.assertEqual(workers[0].environ["NCCL_P2P_DISABLE"], "1")

    def test_blackwell_workers_keep_peer_access(self):
        plan = self._plan("--distributed_mode", "multi_worker")
        workers = orchestration.build_worker_processes(plan, base_environ={})
        self.assertNotIn("NCCL_P2P_DISABLE", workers[0].environ)

    def test_non_chief_workers_are_quiet_and_redirected(self):
        plan = self._plan("--distributed_mode", "multi_worker")
        workers = orchestration.build_worker_processes(plan, base_environ={})
        self.assertNotIn("TF_CPP_MIN_LOG_LEVEL", workers[0].environ)
        self.assertEqual(workers[1].environ["TF_CPP_MIN_LOG_LEVEL"], "2")
        self.assertTrue(
            orchestration.worker_log_path(1).endswith("worker_1.log")
        )

    def test_mixed_architectures_are_rejected(self):
        with self.assertRaises(ValueError):
            self._plan(capabilities=("12.0", "8.9"))


if __name__ == "__main__":
    unittest.main()
