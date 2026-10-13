"""Guards against a failed restore that leaves the model at random init.

``expect_partial()`` forgives every unmatched value, so a checkpoint written by
a model with different shapes or names restores nothing while the code prints a
successful-looking message and trains on from random weights.
"""
import os
import tempfile

import pytest
import tensorflow as tf

from v1_model_utils import tf_utils
from v1_model_utils.optimizers import ExponentiatedAdam
from v1_model_utils.other_v1_utils import optimizers_match


def _dense_model(units=(4,)):
    layers = [tf.keras.layers.Input(shape=(3,))]
    layers += [tf.keras.layers.Dense(unit) for unit in units]
    model = tf.keras.Sequential(layers)
    model.build((None, 3))
    return model


def _saved_checkpoint(directory, model, optimizer=None):
    if optimizer is not None:
        optimizer.build(model.trainable_variables)
    checkpoint = tf_utils.make_checkpoint(model, optimizer)
    manager = tf.train.CheckpointManager(
        checkpoint, os.path.join(directory, 'Best_model'), max_to_keep=1
    )
    return manager.save()


def _model_with_projection(shared_units=4, projection_units=2, dtype='float32'):
    inputs = tf.keras.layers.Input(shape=(3,), dtype=dtype)
    shared = tf.keras.layers.Dense(
        shared_units, name='shared', dtype=dtype,
        kernel_initializer='ones', bias_initializer='ones',
    )(inputs)
    outputs = tf.keras.layers.Dense(
        projection_units, name='projection_v1', dtype=dtype,
    )(shared)
    return tf.keras.Model(inputs, outputs)


def test_restore_raises_when_the_model_does_not_match():
    with tempfile.TemporaryDirectory() as directory:
        path = _saved_checkpoint(directory, _dense_model())
        other = _dense_model(units=(4, 5))
        with pytest.raises(ValueError, match='random initialization'):
            tf_utils.restore_and_rebase(
                tf.train.Checkpoint(model=other), path, other
            )


def test_resuming_with_the_optimizer_raises_when_the_model_does_not_match():
    """A Keras 3 checkpoint shares Keras 2's `optimizer/_variables/` prefix,
    which must not route it through the lenient legacy-slot restore."""
    with tempfile.TemporaryDirectory() as directory:
        path = _saved_checkpoint(directory, _dense_model(), tf.keras.optimizers.Adam())
        other = _dense_model(units=(4, 5))
        optimizer = tf.keras.optimizers.Adam()
        optimizer.build(other.trainable_variables[:2])
        with pytest.raises(ValueError):
            tf_utils.restore_and_rebase(
                tf_utils.make_checkpoint(other, optimizer), path, other,
                optimizer, require_optimizer=True,
            )


def test_restore_without_the_optimizer_keeps_the_weights():
    with tempfile.TemporaryDirectory() as directory:
        model = _dense_model()
        path = _saved_checkpoint(directory, model, tf.keras.optimizers.Adam())
        expected = [variable.numpy() for variable in model.weights]

        restored = _dense_model()
        tf_utils.restore_and_rebase(
            tf.train.Checkpoint(model=restored), path, restored
        )
        for reference, variable in zip(expected, restored.weights):
            assert (reference == variable.numpy()).all()


def test_optimizer_identity_separates_adam_from_exponentiated_adam():
    with tempfile.TemporaryDirectory() as directory:
        model = _dense_model()
        path = _saved_checkpoint(directory, model, tf.keras.optimizers.Adam())
        assert tf_utils.checkpoint_optimizer_identity(path) == 'Adam'

        exponentiated = ExponentiatedAdam()
        exponentiated.build(model.trainable_variables)
        assert not optimizers_match(exponentiated, path)

        adam = tf.keras.optimizers.Adam()
        adam.build(model.trainable_variables)
        assert optimizers_match(adam, path)


def test_legacy_checkpoints_without_an_optimizer_id_still_compare_by_shape():
    with tempfile.TemporaryDirectory() as directory:
        model = _dense_model()
        optimizer = tf.keras.optimizers.Adam()
        optimizer.build(model.trainable_variables)
        # A checkpoint written the pre-identity way, as older runs left on disk.
        checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)
        manager = tf.train.CheckpointManager(
            checkpoint, os.path.join(directory, 'Best_model'), max_to_keep=1
        )
        path = manager.save()

        assert tf_utils.checkpoint_optimizer_identity(path) is None
        adam = tf.keras.optimizers.Adam()
        adam.build(model.trainable_variables)
        assert optimizers_match(adam, path)


def test_runtime_dtype_cast_can_ignore_a_different_evaluation_projection():
    with tempfile.TemporaryDirectory() as directory:
        source = _model_with_projection(projection_units=2, dtype='float16')
        path = _saved_checkpoint(directory, source)
        target = _model_with_projection(projection_units=1, dtype='float32')
        target.get_layer('shared').set_weights([
            tf.zeros((3, 4), dtype=tf.float32),
            tf.zeros((4,), dtype=tf.float32),
        ])

        tf_utils.restore_model_with_runtime_dtype_cast(
            target_model=target,
            build_model_fn=lambda _: _model_with_projection(
                projection_units=2, dtype='float16'
            ),
            checkpoint_directory=path,
            checkpoint_dtype_name='float16',
            target_dtype_name='float32',
            ignored_target_variable_names={
                'projection_v1/kernel',
                'projection_v1/bias',
            },
        )

        kernel, bias = target.get_layer('shared').get_weights()
        assert (kernel == 1).all()
        assert (bias == 1).all()


def _v1_model(network, lgn_input, bkg_input, *, track_voltage_penalty, train):
    from v1_model_utils import models

    return models.create_model(
        network, lgn_input, bkg_input, seq_len=4, n_input=lgn_input["n_inputs"],
        n_output=1, down_sample=2, batch_size=1, use_state_input=True,
        return_state=True, acceleration="tensorflow",
        track_voltage_penalty=track_voltage_penalty,
        return_voltage_sequences=not track_voltage_penalty,
        train_recurrent=train, train_recurrent_per_type=False, train_noise=train,
    )


def test_evaluation_model_must_share_the_training_state_structure(tmp_path):
    """The voltage-penalty state adds an input layer ahead of the cell, and
    checkpoints locate the cell by layer position: an evaluation model built
    without it restores nothing (silently, before the restore guard)."""
    from tests.network_fixtures import load_network

    network, lgn_input, bkg_input = load_network(10)
    trained = _v1_model(network, lgn_input, bkg_input, track_voltage_penalty=True, train=True)
    cell = trained.get_layer("rsnn").cell
    cell.recurrent_weight_values.assign(cell.recurrent_weight_values + 1.0)
    path = tf.train.Checkpoint(model=trained).save(str(tmp_path / "ckpt"))

    mismatched = _v1_model(network, lgn_input, bkg_input, track_voltage_penalty=False, train=False)
    with pytest.raises(ValueError, match="random initialization"):
        tf_utils.restore_and_rebase(tf.train.Checkpoint(model=mismatched), path, mismatched)

    evaluation = _v1_model(network, lgn_input, bkg_input, track_voltage_penalty=True, train=False)
    tf_utils.restore_and_rebase(tf.train.Checkpoint(model=evaluation), path, evaluation)
    restored = evaluation.get_layer("rsnn").cell.recurrent_weight_values
    assert (restored.numpy() == cell.recurrent_weight_values.numpy()).all()
    # A later model in the same process keeps the variable paths that the
    # runtime dtype-cast restore matches by name.
    assert {variable.path for variable in evaluation.variables} == {
        variable.path for variable in trained.variables
    }
