from types import SimpleNamespace
from unittest import mock

import pytest

from v1_model_utils import tf_utils


def test_configure_policy_rejects_bfloat16():
    with pytest.raises(ValueError, match="float16, float32"):
        tf_utils.configure_policy_and_dtype("bfloat16")


def test_explicit_restore_path_takes_precedence_over_run_checkpoint(tmp_path):
    run_dir = tmp_path / "run"
    intermediate_dir = run_dir / "Intermediate_checkpoints"
    explicit_dir = tmp_path / "explicit"
    intermediate_dir.mkdir(parents=True)
    explicit_dir.mkdir()
    flags = SimpleNamespace(
        ckpt_dir=str(run_dir),
        restore_from=str(explicit_dir),
    )
    optimizer = object()
    model = SimpleNamespace(trainable_variables=[])

    def latest_checkpoint(path):
        if path == str(explicit_dir):
            return str(explicit_dir / "ckpt-7")
        if path == str(intermediate_dir):
            return str(intermediate_dir / "ckpt-3")
        return None

    with (
        mock.patch.object(tf_utils.tf.train, "latest_checkpoint", side_effect=latest_checkpoint),
        mock.patch(
            "v1_model_utils.other_v1_utils.optimizers_match", return_value=True
        ),
        mock.patch.object(tf_utils.tf.train, "Checkpoint", return_value=object()),
        mock.patch.object(tf_utils, "restore_and_rebase") as restore,
    ):
        _, _, restored_path = tf_utils.restore_training_checkpoint(
            flags, model, optimizer, learning_rate=0.1
        )

    assert restored_path == str(explicit_dir / "ckpt-7")
    assert restore.call_args.args[1] == restored_path


def test_missing_explicit_restore_path_fails_instead_of_falling_back(tmp_path):
    run_dir = tmp_path / "run"
    (run_dir / "Intermediate_checkpoints").mkdir(parents=True)
    missing = tmp_path / "missing"
    flags = SimpleNamespace(ckpt_dir=str(run_dir), restore_from=str(missing))

    with pytest.raises(FileNotFoundError, match="restore_from"):
        tf_utils.restore_training_checkpoint(
            flags,
            SimpleNamespace(trainable_variables=[]),
            object(),
            learning_rate=0.1,
        )
