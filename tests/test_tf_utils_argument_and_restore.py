from types import SimpleNamespace
from unittest import mock

import pytest

from v1_model_utils import tf_utils


def test_configure_policy_rejects_bfloat16():
    with pytest.raises(ValueError, match="float16, float32"):
        tf_utils.configure_policy_and_dtype("bfloat16")


def test_untrained_evaluation_skips_checkpoint_restore(tmp_path):
    flags = SimpleNamespace(restore_from="", dtype="float32")
    with mock.patch.object(tf_utils, "resolve_checkpoint_directory") as resolve:
        checkpoint, logdir, epoch = tf_utils.restore_evaluation_checkpoint(
            flags, object(), lambda _: None, str(tmp_path), current_epoch=-10,
            evaluate_untrained=True,
        )

    resolve.assert_not_called()
    assert checkpoint is None
    assert logdir == str(tmp_path / "Initial_evaluation")
    assert (tmp_path / "Initial_evaluation").is_dir()
    assert epoch == 0

    flags.restore_from = "Best_model"
    with pytest.raises(ValueError, match="evaluate_untrained"):
        tf_utils.restore_evaluation_checkpoint(
            flags, object(), lambda _: None, str(tmp_path),
            evaluate_untrained=True,
        )


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


def test_core_only_evaluation_restores_through_a_full_model(tmp_path):
    # Matching dtypes, but the core-only readout cannot take the checkpoint's shape.
    flags = SimpleNamespace(
        restore_from=str(tmp_path), dtype="float32", restore_runtime_dtype_cast=True,
    )
    model = SimpleNamespace(variables=[tf_utils.tf.Variable(1.0)])
    ignored = {"projection_v1/kernel", "projection_v1/bias"}
    with (
        mock.patch.object(
            tf_utils, "resolve_checkpoint_directory",
            return_value=(str(tmp_path / "ckpt-1"), "path"),
        ),
        mock.patch.object(tf_utils, "infer_checkpoint_model_dtype", return_value="float32"),
        mock.patch.object(tf_utils, "restore_model_with_runtime_dtype_cast") as copy_restore,
        mock.patch.object(tf_utils, "restore_and_rebase") as direct_restore,
    ):
        tf_utils.restore_evaluation_checkpoint(
            flags, model, lambda _: None, str(tmp_path),
            runtime_cast_ignored_variables=ignored,
        )

    direct_restore.assert_not_called()
    assert copy_restore.call_args.kwargs["ignored_target_variable_names"] == ignored
    assert copy_restore.call_args.kwargs["checkpoint_dtype_name"] == "float32"


def test_host_numbered_spontaneous_seed_matches_the_stream():
    helper = tf_utils.DistributedSeedHelper(
        3000, tf_utils.tf.Variable(0, dtype=tf_utils.tf.int64),
        tf_utils.tf.Variable(0, dtype=tf_utils.tf.int64),
    )
    streamed = [helper.next_spontaneous_seed().numpy() for _ in range(3)]
    numbered = [helper.spontaneous_seed(tf_utils.tf.constant(draw)).numpy() for draw in (1, 2, 3)]
    assert [s.tolist() for s in streamed] == [s.tolist() for s in numbered]
    assert len({tuple(s) for s in streamed}) == 3


def test_memory_optimizer_can_be_turned_off_for_later_functions():
    import tensorflow as tf
    from tensorflow.core.protobuf import rewriter_config_pb2
    from tensorflow.python.eager import context

    tf_context = context.context()
    saved = tf_context._config
    try:
        tf_utils.enable_tensorflow_optimizations(enabled=False, memory_optimizer=False)
        no_mem_opt = rewriter_config_pb2.RewriterConfig.NO_MEM_OPT
        assert tf_context.config.graph_options.rewrite_options.memory_optimization == no_mem_opt
        assert tf.function(lambda: tf.constant(1))().numpy() == 1
        options = tf_context.function_call_options.config_proto_serialized
        called_with = type(tf_context.config).FromString(options)
        assert called_with.graph_options.rewrite_options.memory_optimization == no_mem_opt
    finally:
        tf_context._config = saved
        tf_context.function_call_options = None
