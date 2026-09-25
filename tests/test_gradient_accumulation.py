"""``--sequential_stimuli`` must update the weights like the single-shot step.

The sequential step rolls the drifting-grating and the gray-screen halves of a
training step out one after the other to halve the activations the backward
pass keeps alive. The weights may only move once both halves have contributed,
so a ``GradientAccumulator`` parks the first half's gradients until the second
arrives and the sum is applied - the gradient the combined step computes in one
backward pass.

Keras 3 offers the same thing as ``gradient_accumulation_steps`` on the
optimizer. These tests pin down why this repository cannot use it: its update
branch all-reduces inside a ``tf.cond``, which no ``MirroredStrategy`` allows.
"""

from types import SimpleNamespace

import numpy as np
import pytest
import tensorflow as tf

from v1_model_utils import optimizers as optimizer_utils
from v1_model_utils.optimizers import (
    GradientAccumulator,
    clip_gradients_by_global_norm,
    create_optimizer,
)


requires_two_gpus = pytest.mark.skipif(
    len(tf.config.list_physical_devices("GPU")) < 2,
    reason="two CUDA GPUs required",
)


def make_flags(**overrides):
    flags = dict(
        optimizer="adam",
        dtype="float32",
        learning_rate=0.01,
        global_clipnorm=0.0,
        sequential_stimuli=False,
        lr_schedule="none",
    )
    flags.update(overrides)
    return SimpleNamespace(**flags)


def test_a_parked_substep_leaves_the_weights_alone():
    variable = tf.Variable([1.0, 2.0])
    optimizer = create_optimizer(make_flags(optimizer="sgd"), 0.1, [variable])
    accumulator = GradientAccumulator([variable])

    accumulator.accumulate([tf.constant([1.0, 1.0])])

    np.testing.assert_array_equal(variable.numpy(), [1.0, 2.0])
    assert int(optimizer.iterations) == 0


def test_draining_returns_the_sum_of_the_substeps():
    variable = tf.Variable([1.0, 2.0])
    accumulator = GradientAccumulator([variable])

    accumulator.accumulate([tf.constant([1.0, -1.0])])
    (total,) = accumulator.drain([tf.constant([3.0, 0.5])])

    np.testing.assert_allclose(total.numpy(), [4.0, -0.5], rtol=1e-6)


def test_draining_clears_the_buffers_for_the_next_step():
    variable = tf.Variable([1.0, 2.0])
    accumulator = GradientAccumulator([variable])

    accumulator.accumulate([tf.constant([1.0, 1.0])])
    accumulator.drain([tf.constant([1.0, 1.0])])

    accumulator.accumulate([tf.constant([2.0, 2.0])])
    (total,) = accumulator.drain([tf.constant([5.0, 5.0])])

    np.testing.assert_allclose(total.numpy(), [7.0, 7.0], rtol=1e-6)


def test_unreachable_variables_stay_unreachable():
    """A `None` gradient must not become a zero the optimizer acts on."""
    reached = tf.Variable([1.0])
    unreached = tf.Variable([1.0])
    accumulator = GradientAccumulator([reached, unreached])

    accumulator.accumulate([tf.constant([1.0]), None])
    totals = accumulator.drain([tf.constant([1.0]), None])

    np.testing.assert_allclose(totals[0].numpy(), [2.0], rtol=1e-6)
    assert totals[1] is None


def test_sparse_substep_gradients_accumulate_on_their_own_rows():
    variable = tf.Variable([[1.0], [2.0], [3.0]])
    accumulator = GradientAccumulator([variable])
    row_one = tf.IndexedSlices(
        values=tf.constant([[0.5]]),
        indices=tf.constant([1]),
        dense_shape=tf.constant([3, 1]),
    )

    accumulator.accumulate([row_one])
    (total,) = accumulator.drain([row_one])

    np.testing.assert_allclose(total.numpy(), [[0.0], [1.0], [0.0]], rtol=1e-6)


@pytest.mark.parametrize("optimizer_name", ["adam", "exp_adam", "sgd"])
def test_accumulated_substeps_match_the_single_shot_update(optimizer_name):
    evoked_gradient = tf.constant([0.3, -0.7])
    spontaneous_gradient = tf.constant([-0.1, 0.4])

    combined_variable = tf.Variable([1.0, 2.0])
    combined_optimizer = create_optimizer(
        make_flags(optimizer=optimizer_name), 0.01, [combined_variable]
    )
    combined_optimizer.apply_gradients(
        [(evoked_gradient + spontaneous_gradient, combined_variable)]
    )

    sequential_variable = tf.Variable([1.0, 2.0])
    sequential_optimizer = create_optimizer(
        make_flags(optimizer=optimizer_name, sequential_stimuli=True),
        0.01,
        [sequential_variable],
    )
    accumulator = GradientAccumulator([sequential_variable])
    accumulator.accumulate([evoked_gradient])
    sequential_optimizer.apply_gradients(
        zip(accumulator.drain([spontaneous_gradient]), [sequential_variable])
    )

    np.testing.assert_allclose(
        sequential_variable.numpy(), combined_variable.numpy(), rtol=1e-6
    )
    # One update per step keeps learning-rate schedules, which are driven by
    # the optimizer's step counter, in step between the two modes.
    assert int(sequential_optimizer.iterations) == int(combined_optimizer.iterations)


def test_the_optimizer_owns_clipping_so_the_train_step_defers():
    variable = tf.Variable([1.0, 2.0])
    optimizer = create_optimizer(
        make_flags(optimizer="sgd", global_clipnorm=1.0), 0.1, [variable]
    )
    assert optimizer.global_clipnorm == 1.0
    assert optimizer_utils.optimizer_clips_gradients(optimizer)

    gradients = [tf.constant([3.0, 4.0])]
    deferred, raw_norm = clip_gradients_by_global_norm(
        gradients, 1.0, optimizer=optimizer
    )

    assert deferred is gradients
    np.testing.assert_allclose(raw_norm.numpy(), 5.0)


def test_clipping_sees_the_summed_gradient_rather_than_a_substep():
    """Substeps that only overshoot together must still be clipped together."""
    variable = tf.Variable([0.0, 0.0])
    optimizer = create_optimizer(
        make_flags(optimizer="sgd", sequential_stimuli=True, global_clipnorm=1.0),
        1.0,
        [variable],
    )
    accumulator = GradientAccumulator([variable])

    # Each substep has norm 0.5 and would survive clipping on its own; their
    # sum has norm 1.0 and still does, so the update is the plain sum.
    accumulator.accumulate([tf.constant([0.3, 0.4])])
    optimizer.apply_gradients(
        zip(accumulator.drain([tf.constant([0.3, 0.4])]), [variable])
    )
    np.testing.assert_allclose(variable.numpy(), [-0.6, -0.8], rtol=1e-6)

    # The summed gradient has norm 10 and is scaled down to the 1.0 limit.
    variable.assign([0.0, 0.0])
    accumulator.accumulate([tf.constant([3.0, 4.0])])
    optimizer.apply_gradients(
        zip(accumulator.drain([tf.constant([3.0, 4.0])]), [variable])
    )
    np.testing.assert_allclose(variable.numpy(), [-0.6, -0.8], rtol=1e-6)


def test_float32_runs_clip_through_the_optimizer_too():
    """Keras 3 gives every optimizer ``scale_loss``, so it cannot select one."""
    variable = tf.Variable([0.0, 0.0])
    optimizer = create_optimizer(
        make_flags(optimizer="sgd", global_clipnorm=1.0), 1.0, [variable]
    )

    optimizer.apply_gradients([(tf.constant([6.0, 8.0]), variable)])

    np.testing.assert_allclose(variable.numpy(), [-0.6, -0.8], rtol=1e-6)


def test_loss_scaled_substeps_survive_the_round_trip():
    """A scaled sum unscales to the same gradient as the unscaled sum."""
    variable = tf.Variable([1.0, 2.0])
    optimizer = create_optimizer(
        make_flags(optimizer="sgd", dtype="float16"), 1.0, [variable]
    )
    scale = float(optimizer.dynamic_scale)
    accumulator = GradientAccumulator([variable])

    accumulator.accumulate([tf.constant([0.3, -0.7]) * scale])
    optimizer.apply_gradients(
        zip(
            accumulator.drain([tf.constant([-0.1, 0.4]) * scale]),
            [variable],
        )
    )

    np.testing.assert_allclose(variable.numpy(), [0.8, 2.3], rtol=1e-5)


@requires_two_gpus
def test_replicas_accumulate_their_own_half_and_apply_the_reduced_sum():
    strategy = tf.distribute.MirroredStrategy(devices=["GPU:0", "GPU:1"])
    with strategy.scope():
        variable = tf.Variable([0.0, 0.0])
        optimizer = create_optimizer(
            make_flags(optimizer="sgd"), 1.0, [variable]
        )
        accumulator = GradientAccumulator([variable])

    @tf.function
    def substep(gradient, closes_step):
        def replica_fn():
            replica_id = tf.cast(
                tf.distribute.get_replica_context().replica_id_in_sync_group,
                tf.float32,
            )
            local = [gradient * (1.0 + replica_id)]
            if closes_step:
                optimizer.apply_gradients(
                    zip(accumulator.drain(local), [variable])
                )
            else:
                accumulator.accumulate(local)

        strategy.run(replica_fn)

    substep(tf.constant([1.0, 0.0]), False)
    np.testing.assert_array_equal(variable.numpy(), [0.0, 0.0])

    substep(tf.constant([3.0, 0.0]), True)
    # Each replica sums its substeps, then the optimizer all-reduces:
    # (1 + 3) + (2 + 6) = 12.
    np.testing.assert_allclose(variable.numpy(), [-12.0, 0.0], rtol=1e-6)


@requires_two_gpus
def test_keras_native_accumulation_is_unusable_under_a_strategy():
    """Documents why the accumulator exists rather than the Keras option."""
    strategy = tf.distribute.MirroredStrategy(devices=["GPU:0", "GPU:1"])
    with strategy.scope():
        variable = tf.Variable([0.0, 0.0])
        optimizer = tf.keras.optimizers.SGD(
            learning_rate=1.0, gradient_accumulation_steps=2
        )
        optimizer.build([variable])

    @tf.function
    def substep():
        strategy.run(
            lambda: optimizer.apply_gradients([(tf.ones([2]), variable)])
        )

    with pytest.raises(RuntimeError, match="merge_call"):
        substep()
