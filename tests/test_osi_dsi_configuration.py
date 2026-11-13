"""Evaluation must preserve the requested reset and weight configuration."""

import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest


@pytest.mark.parametrize("caching", [False, True])
@pytest.mark.parametrize("uniform_weights", [False, True])
def test_estimator_preserves_weight_initialization_and_loader_choice(
    tmp_path, monkeypatch, caching, uniform_weights,
):
    import tensorflow as tf

    import osi_dsi_estimator as estimator
    from v1_model_utils import cuda_prebuild, load_sparse, tf_utils

    options = SimpleNamespace(
        n_input=1, n_trials_per_angle=1, neurons=1000,
        data_dir=str(tmp_path), task_name="test", dtype="float32", seed=3000,
        caching=caching, uniform_weights=uniform_weights,
    )
    monkeypatch.setattr(estimator, "absl", SimpleNamespace(
        app=SimpleNamespace(flags=SimpleNamespace(FLAGS=options)),
    ))
    monkeypatch.setattr(tf_utils, "configure_gpu_memory_growth", lambda: None)
    monkeypatch.setattr(tf_utils, "print_tensorflow_runtime_info", lambda: None)
    monkeypatch.setattr(cuda_prebuild, "prebuild_for_run", lambda: None)
    monkeypatch.setattr(tf_utils, "configure_reproducibility", lambda seed: None)
    monkeypatch.setattr(tf_utils, "enable_tensorflow_optimizations", lambda **kwargs: None)
    monkeypatch.setattr(tf_utils, "configure_run_paths", lambda *a, **k: ("test", str(tmp_path), 0))
    monkeypatch.setattr(tf_utils, "configure_policy_and_dtype", lambda dtype: (None, tf.float32))
    monkeypatch.setattr(tf_utils, "create_distribution_strategy", lambda **k: tf.distribute.get_strategy())
    calls = []

    class LoadedNetwork(Exception):
        pass

    def loader(selected_cache):
        def load(flags, neurons, flag_str):
            calls.append((selected_cache, flags.uniform_weights))
            assert flags is options
            raise LoadedNetwork
        return load

    monkeypatch.setattr(load_sparse, "cached_load_v1", loader(True))
    monkeypatch.setattr(load_sparse, "load_v1", loader(False))
    with pytest.raises(LoadedNetwork):
        estimator.main(None)
    assert calls == [(caching, uniform_weights)]


def test_estimator_cli_defaults_to_hard_reset_and_allows_soft_reset():
    repo = Path(__file__).resolve().parents[1]
    code = """
import runpy
from absl import app, flags

def check(main):
    assert flags.FLAGS['hard_reset'].default is True
    flags.FLAGS(['osi_dsi_estimator.py'])
    assert flags.FLAGS.hard_reset is True
    flags.FLAGS(['osi_dsi_estimator.py', '--nohard_reset'])
    assert flags.FLAGS.hard_reset is False
    flags.FLAGS(['osi_dsi_estimator.py', '--hard_reset'])
    assert flags.FLAGS.hard_reset is True

app.run = check
runpy.run_path('osi_dsi_estimator.py', run_name='__main__')
"""
    result = subprocess.run(
        [sys.executable, "-c", code], cwd=repo,
        env=dict(os.environ, CUDA_VISIBLE_DEVICES=""),
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
