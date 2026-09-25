from types import SimpleNamespace

import numpy as np
import tensorflow as tf

from v1_model_utils.optimizers import (
    ExponentiatedAdam,
    clip_gradients_by_global_norm,
    create_optimizer,
    scale_loss_for_optimizer,
    unscale_gradients_for_optimizer,
)
from v1_model_utils.other_v1_utils import optimizers_match
from v1_model_utils.tf_utils import restore_and_rebase


def test_exponentiated_adam_dense_update_and_serialization():
    variable = tf.Variable([1.0, 2.0])
    optimizer = ExponentiatedAdam(learning_rate=0.01, jit_compile=False)

    optimizer.apply_gradients([(tf.constant([0.1, -0.2]), variable)])

    np.testing.assert_allclose(variable.numpy(), [0.9900501, 2.0201], rtol=1e-5)
    restored = ExponentiatedAdam.from_config(optimizer.get_config())
    assert restored.beta_1 == optimizer.beta_1
    assert restored.beta_2 == optimizer.beta_2


def test_exponentiated_adam_sparse_update_coalesces_duplicate_indices():
    variable = tf.Variable([[1.0], [2.0], [3.0]])
    optimizer = ExponentiatedAdam(learning_rate=0.01, jit_compile=False)
    gradient = tf.IndexedSlices(
        values=tf.constant([[0.1], [0.2]]),
        indices=tf.constant([1, 1]),
        dense_shape=tf.constant([3, 1]),
    )

    optimizer.apply_gradients([(gradient, variable)])

    np.testing.assert_array_equal(variable.numpy()[[0, 2]], [[1.0], [3.0]])
    assert variable.numpy()[1, 0] < 2.0


def test_loss_scale_adapter_uses_current_keras_contract():
    flags = SimpleNamespace(optimizer="adam", dtype="float16")
    variable = tf.Variable([1.0])
    optimizer = create_optimizer(flags, 0.01, [variable])
    loss = tf.constant(2.0)

    scaled_loss = scale_loss_for_optimizer(optimizer, loss)
    gradients = [tf.constant([4.0])]
    adapted = unscale_gradients_for_optimizer(optimizer, gradients)

    assert float(scaled_loss) > float(loss)
    if hasattr(optimizer, "scale_loss"):
        assert adapted is gradients
    else:
        assert float(adapted[0][0]) < float(gradients[0][0])


def test_optimizer_checkpoint_restores_iterations_and_slots(tmp_path):
    source_variable = tf.Variable([1.0, 2.0], name="weights")
    source_optimizer = ExponentiatedAdam(learning_rate=0.01, jit_compile=False)
    source_optimizer.apply_gradients(
        [(tf.constant([0.1, -0.2]), source_variable)]
    )
    source_model = tf.Module()
    source_model.weights = source_variable
    checkpoint_path = tf.train.Checkpoint(
        optimizer=source_optimizer, model=source_model
    ).save(str(tmp_path / "ckpt"))

    target_variable = tf.Variable([1.0, 2.0], name="weights")
    target_optimizer = ExponentiatedAdam(learning_rate=0.01, jit_compile=False)
    target_optimizer.build([target_variable])
    target_model = tf.Module()
    target_model.weights = target_variable
    target_checkpoint = tf.train.Checkpoint(
        optimizer=target_optimizer, model=target_model
    )

    assert optimizers_match(target_optimizer, checkpoint_path)
    restore_and_rebase(
        target_checkpoint,
        checkpoint_path,
        target_model,
        target_optimizer,
        require_optimizer=True,
    )

    assert int(target_optimizer.iterations) == int(source_optimizer.iterations)
    np.testing.assert_allclose(target_variable.numpy(), source_variable.numpy())
    source_optimizer.apply_gradients(
        [(tf.constant([0.2, 0.3]), source_variable)]
    )
    target_optimizer.apply_gradients(
        [(tf.constant([0.2, 0.3]), target_variable)]
    )
    np.testing.assert_allclose(
        target_variable.numpy(), source_variable.numpy(), rtol=1e-6
    )


def test_keras3_scaled_gradients_are_deferred_to_optimizer_clipping():
    class Keras3LossScaleOptimizerStub:
        # create_optimizer() marks the optimizers it configured to clip.
        clips_gradients_internally = True
        dynamic_scale = tf.constant(8.0)

        def scale_loss(self, loss):
            return loss * self.dynamic_scale

    gradients = [tf.constant([24.0, 32.0])]
    adapted, unscaled_norm = clip_gradients_by_global_norm(
        gradients, 1.0, optimizer=Keras3LossScaleOptimizerStub()
    )

    assert adapted is gradients
    np.testing.assert_allclose(unscaled_norm.numpy(), 5.0)
