"""Offload opt-in defaults through the real bootstrap and flag definitions."""

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from v1_model_utils import training_orchestration
from v1_model_utils.offload_options import normalize_offload_args


ROOT = Path(__file__).resolve().parents[1]
PARSE_ONLY = """
import json, runpy, sys
import absl.app, absl.flags
args = sys.argv[1:]
def parse_only(main, argv=None):
    absl.flags.FLAGS(sys.argv)
    print(json.dumps([absl.flags.FLAGS.voltage_offload, absl.flags.FLAGS.history_offload]))
    raise SystemExit(0)
absl.app.run = parse_only
sys.argv = ['multi_training.py', '--noauto_select_gpus', *args]
runpy.run_path('multi_training.py', run_name='__main__')
"""


@pytest.mark.parametrize("arguments,expected", [
    ([], ["none", "none"]),
    (["--voltage_offload"], ["adaptive", "none"]),
    (["--history_offload"], ["none", "adaptive"]),
    (["--voltage_offload", "--history_offload"], ["adaptive", "adaptive"]),
    (["--voltage_offload", "adaptive", "--history_offload", "adaptive"],
     ["adaptive", "adaptive"]),
    (["--voltage_offload", "raw", "--history_offload", "none"], ["raw", "none"]),
    (["--voltage_offload=compressed", "--history_offload=raw"], ["compressed", "raw"]),
])
def test_real_training_entrypoint_offload_options(arguments, expected):
    result = subprocess.run(
        [sys.executable, "-c", PARSE_ONLY, *arguments], cwd=ROOT,
        env=dict(os.environ, CUDA_VISIBLE_DEVICES="", TF_CPP_MIN_LOG_LEVEL="3"),
        capture_output=True, text=True, timeout=90,
    )
    assert result.returncode == 0, result.stderr[-4000:]
    assert json.loads(result.stdout.strip().splitlines()[-1]) == expected


def test_bare_offload_flags_reach_every_worker():
    plan = training_orchestration.plan_training_launch(
        ["multi_training.py", "--n_gpus=2", "--distributed_mode=multi_worker",
         "--voltage_offload", "--history_offload"],
        environ={"CUDA_VISIBLE_DEVICES": "0,1"}, compute_capabilities=("8.9", "8.9"),
        worker_addresses=("localhost:1234", "localhost:1235"),
    )
    workers = training_orchestration.build_worker_processes(plan, base_environ={})
    assert len(workers) == 2
    for worker in workers:
        assert "--voltage_offload=adaptive" in worker.argv
        assert "--history_offload=adaptive" in worker.argv


def test_offload_normalization_preserves_positional_arguments():
    argv = ["multi_training.py", "--voltage_offload", "--", "--history_offload"]
    assert normalize_offload_args(argv) == (
        "multi_training.py", "--voltage_offload=adaptive", "--", "--history_offload",
    )
    assert argv == ["multi_training.py", "--voltage_offload", "--", "--history_offload"]
