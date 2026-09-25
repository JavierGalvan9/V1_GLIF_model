import os
import pickle as pkl
import tempfile
import types
import unittest

from v1_model_utils import tf_utils


class RuntimeBookkeepingTests(unittest.TestCase):
    def test_extract_sim_name(self):
        logdir = "/tmp/Simulation_results/v1_10/b_demo"
        self.assertEqual(tf_utils.extract_sim_name(logdir), "b_demo")

    def test_resolve_performance_statistics_path_is_scoped_to_logdir(self):
        logdir = "/tmp/Simulation_results/v1_10/b_demo"
        expected = "/tmp/Simulation_results/v1_10/b_demo/performance_statistics.csv"
        self.assertEqual(tf_utils.resolve_performance_statistics_path(logdir), expected)

    def test_total_epoch_accounting_resume_within_planned_runs(self):
        flags = types.SimpleNamespace(n_runs=3, n_epochs=10, run_session=1)
        self.assertEqual(tf_utils.compute_total_epochs(flags, checkpoint_epochs=10), 30)
        self.assertEqual(tf_utils.current_run_start_epoch(flags, checkpoint_epochs=10), 10)

    def test_total_epoch_accounting_resume_beyond_original_plan(self):
        flags = types.SimpleNamespace(n_runs=1, n_epochs=10, run_session=0)
        self.assertEqual(tf_utils.compute_total_epochs(flags, checkpoint_epochs=50), 60)
        self.assertEqual(tf_utils.current_run_start_epoch(flags, checkpoint_epochs=50), 50)

    def test_infer_completed_epochs_and_report_epoch(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            payload = {
                "epoch_metric_values": {"train_loss": [1.0, 0.5, 0.25]},
                "min_val_loss": 0.25,
                "no_improve_epochs": 0,
            }
            with open(os.path.join(tmpdir, "train_end_data.pkl"), "wb") as handle:
                pkl.dump(payload, handle)

            self.assertEqual(tf_utils.infer_completed_epochs(tmpdir), 3)
            self.assertEqual(
                tf_utils.infer_report_epoch(tmpdir, "", run_session=4, n_epochs=10),
                3,
            )

    def test_append_performance_statistics(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            logdir = os.path.join(tmpdir, "Simulation_results", "v1_10", "b_stats")
            os.makedirs(logdir, exist_ok=True)
            stats_file = tf_utils.append_performance_statistics(
                logdir=logdir,
                n_neurons=10,
                n_edges=20,
                batch_size=1,
                seq_len=100,
                mean_rate=0.1,
                sem_rate=0.01,
                mean_step_time=1.5,
                sem_step_time=0.1,
                mean_gpu_memory=2.0,
                sem_gpu_memory=0.2,
                mode="train",
            )
            self.assertEqual(
                stats_file,
                os.path.join(logdir, "performance_statistics.csv"),
            )
            self.assertTrue(os.path.exists(stats_file))
            with open(stats_file, "r", encoding="utf-8") as handle:
                rows = handle.read().strip().splitlines()
            self.assertEqual(len(rows), 2)
            self.assertIn("b_stats,10,20,1,100", rows[1])
            self.assertTrue(rows[1].endswith('"train"'))


if __name__ == "__main__":
    unittest.main()
