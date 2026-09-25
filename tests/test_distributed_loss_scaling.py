"""Replicas must agree on gradient finiteness before the loss scale branches.

``LossScaleOptimizer`` tests finiteness on each replica's local gradients and
puts the gradient all-reduce and the variable update inside the finite branch.
Replicas that disagree therefore run different graphs, which corrupts both the
weights and the mirrored loss scale.
"""

import numpy as np
import pytest
import tensorflow as tf

from v1_model_utils import optimizers as optimizer_utils


requires_two_gpus = pytest.mark.skipif(
    len(tf.config.list_physical_devices("GPU")) < 2,
    reason="two CUDA GPUs required",
)


def test_no_op_outside_a_replica_context():
    gradients = [tf.constant([1.0, 2.0]), None]
    synchronized = optimizer_utils.synchronize_gradient_finiteness(gradients)
    assert synchronized is gradients


def test_all_none_gradients_pass_through():
    gradients = [None, None]
    assert optimizer_utils.synchronize_gradient_finiteness(gradients) is gradients


@requires_two_gpus
@pytest.mark.parametrize("overflowing_replica", [0, 1])
def test_one_replica_overflow_reaches_every_replica(overflowing_replica):
    strategy = tf.distribute.MirroredStrategy(
        cross_device_ops=tf.distribute.NcclAllReduce()
    )
    assert strategy.num_replicas_in_sync == 2

    @tf.function
    def step():
        def replica_fn():
            replica_id = tf.distribute.get_replica_context().replica_id_in_sync_group
            value = tf.where(
                tf.equal(replica_id, overflowing_replica),
                tf.constant(float("inf")),
                tf.constant(1.0),
            )
            gradients = [tf.fill([4], value), tf.ones([2])]
            synchronized = optimizer_utils.synchronize_gradient_finiteness(gradients)
            return tf.reduce_all(
                [tf.reduce_all(tf.math.is_finite(g)) for g in synchronized]
            )

        return strategy.run(replica_fn)

    verdicts = [
        bool(value.numpy())
        for value in strategy.experimental_local_results(step())
    ]
    assert verdicts == [False, False], (
        "every replica must see the step as non-finite once one replica "
        f"overflows, got {verdicts}"
    )


@requires_two_gpus
def test_finite_gradients_are_left_unchanged():
    strategy = tf.distribute.MirroredStrategy(
        cross_device_ops=tf.distribute.NcclAllReduce()
    )

    @tf.function
    def step():
        def replica_fn():
            gradients = [tf.constant([1.5, -2.5]), tf.constant([0.25])]
            return optimizer_utils.synchronize_gradient_finiteness(gradients)

        return strategy.run(replica_fn)

    first, second = step()
    for value in strategy.experimental_local_results(first):
        np.testing.assert_allclose(value.numpy(), [1.5, -2.5])
    for value in strategy.experimental_local_results(second):
        np.testing.assert_allclose(value.numpy(), [0.25])


@requires_two_gpus
def test_loss_scale_stays_identical_across_replicas():
    """The mirrored loss scale must take one value, not one per replica."""
    strategy = tf.distribute.MirroredStrategy(
        cross_device_ops=tf.distribute.NcclAllReduce()
    )
    with strategy.scope():
        variable = tf.Variable(tf.ones([4]))
        optimizer = tf.keras.mixed_precision.LossScaleOptimizer(
            tf.keras.optimizers.SGD(0.1)
        )
        optimizer.build([variable])

    @tf.function
    def step():
        def replica_fn():
            replica_id = tf.distribute.get_replica_context().replica_id_in_sync_group
            # Only the second replica overflows, as dynamic loss scaling makes
            # happen naturally while it searches for a workable scale.
            value = tf.where(
                tf.equal(replica_id, 1),
                tf.constant(float("inf")),
                tf.constant(1.0),
            )
            gradients = [tf.fill([4], value)]
            gradients = optimizer_utils.synchronize_gradient_finiteness(gradients)
            optimizer.apply_gradients(zip(gradients, [variable]))

        strategy.run(replica_fn)

    step()
    scales = {
        float(value.numpy())
        for value in strategy.experimental_local_results(
            optimizer.dynamic_scale.value
        )
    }
    assert len(scales) == 1, f"replicas disagree on the loss scale: {scales}"
    for value in strategy.experimental_local_results(variable):
        np.testing.assert_allclose(
            value.numpy(), np.ones(4), err_msg="a skipped step still updated"
        )


@requires_two_gpus
def test_sparse_gradients_still_carry_the_verdict():
    """A gradient list of only IndexedSlices must still synchronize.

    An empty sparse gradient would carry nothing, so the signal has to reach a
    row the gradient actually holds.
    """
    strategy = tf.distribute.MirroredStrategy(
        cross_device_ops=tf.distribute.NcclAllReduce()
    )

    @tf.function
    def step():
        def replica_fn():
            replica_id = tf.distribute.get_replica_context().replica_id_in_sync_group
            value = tf.where(
                tf.equal(replica_id, 1),
                tf.constant(float("inf")),
                tf.constant(1.0),
            )
            sparse = tf.IndexedSlices(
                tf.fill([2, 3], value),
                tf.constant([0, 2]),
                tf.constant([4, 3], tf.int32),
            )
            synchronized = optimizer_utils.synchronize_gradient_finiteness([sparse])
            return tf.reduce_all(tf.math.is_finite(synchronized[0].values))

        return strategy.run(replica_fn)

    verdicts = [
        bool(value.numpy())
        for value in strategy.experimental_local_results(step())
    ]
    assert verdicts == [False, False], verdicts
