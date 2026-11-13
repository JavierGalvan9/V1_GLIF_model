"""OSI/DSI EMD losses pool their estimates across MirroredStrategy replicas (two logical CPUs)."""
import os
import subprocess
import sys
from pathlib import Path

TWO_REPLICA_SCRIPT = """
import numpy as np
import tensorflow as tf
tf.config.set_logical_device_configuration(
    tf.config.list_physical_devices("CPU")[0],
    [tf.config.LogicalDeviceConfiguration()] * 2,
)
from tests.test_rolling_osi_loss import build_synthetic_rolling_loss, tuned_rates

strategy = tf.distribute.MirroredStrategy(["/CPU:0", "/CPU:1"])
angles = np.array([0.0, 90.0, 180.0, 270.0], dtype=np.float32)
angle = tf.constant(angles)
# Neuron 0 is below the 0.5 Hz preferred-rate cutoff in replica 0's trials alone
# and above it once both replicas' trials are pooled.
quiet = np.array([0.3, 1.0, 1.0, 1.0], dtype=np.float32)
replica_rates = [
    [tuned_rates(angles, [0.1, 0.2, 0.3, 0.4]) * quiet, tuned_rates(angles, [0.6, 0.5, 0.4, 0.2], 2.0)],
    [tuned_rates(angles, [0.3, 0.1, 0.5, 0.2]) * quiet, tuned_rates(angles, [0.2, 0.7, 0.1, 0.4], 1.5)],
    [tuned_rates(angles, [0.5, 0.4, 0.2, 0.6]) * quiet, tuned_rates(angles, [0.1, 0.3, 0.6, 0.3], 0.5)],
]


def build(scope, method):
    with scope:
        loss = build_synthetic_rolling_loss(decay=0.8)
        loss._method = method
        loss._rolling_warmup = True
        weight = tf.Variable(1.0)
    return loss, weight


def make_step(loss, weight):
    @tf.function
    def step(rates, angles_):
        def replica_fn(r, a):
            # Each replica's loss is divided by the replica count and the
            # gradients are summed, as in multi_training.
            with tf.GradientTape() as tape:
                value = loss.loss_from_rates(weight * r, a)
                scaled = value / strategy.num_replicas_in_sync
            return value, tape.gradient(scaled, weight)
        return strategy.run(replica_fn, args=(rates, angles_))
    return step


def distribute(values):
    return strategy.experimental_distribute_values_from_function(
        lambda ctx: tf.constant(values[ctx.replica_id_in_sync_group])
    )


def run(step, batch):
    values, grads = step(distribute(batch), distribute([angles, angles]))
    local = strategy.experimental_local_results
    return [float(v) for v in local(values)], sum(float(g) for g in local(grads))


def reference_step(loss, weight, batch):
    with tf.GradientTape() as tape:
        value = loss.loss_from_rates(
            weight * tf.constant(np.concatenate(batch)), tf.constant(np.concatenate([angles, angles]))
        )
    return float(value), float(tape.gradient(value, weight))


for method in ("rolling_osi_emd", "batch_osi_emd"):
    pooled, weight = build(strategy.scope(), method)
    step = make_step(pooled, weight)
    reference, ref_weight = build(tf.distribute.get_strategy().scope(), method)
    for batch in replica_rates:
        values, gradient = run(step, batch)
        ref_value, ref_gradient = reference_step(reference, ref_weight, batch)
        # Every replica computes the single-device loss of the concatenated
        # batch, and the summed gradient is that loss's gradient.
        np.testing.assert_allclose(values, [ref_value, ref_value], rtol=1e-5, err_msg=method)
        np.testing.assert_allclose(gradient, ref_gradient, rtol=1e-4, atol=1e-7, err_msg=method)
        assert ref_gradient != 0.0, method
    mask = pooled._preferred_direction_active(
        tf.constant(np.concatenate(replica_rates[0])), tf.constant(np.concatenate([angles, angles]))
    )
    assert bool(mask[0]) and not bool(pooled._preferred_direction_active(
        tf.constant(replica_rates[0][0]), angle)[0]), method

    if method == "rolling_osi_emd":
        # Identical copies equal to the single-device EMA of the concatenated batches.
        for key, variable in reference._rolling_state_variables().items():
            for copy in pooled._rolling_state_variables()[key].values:
                np.testing.assert_allclose(copy.numpy(), variable.numpy(), rtol=1e-6, atol=1e-9)
        # Restoring the saved state (also into one replica) continues bit-exactly.
        saved = pooled.get_rolling_state()
        resumed, resumed_weight = build(strategy.scope(), method)
        assert resumed.set_rolling_state(saved)
        extra = [replica_rates[1][1], replica_rates[2][0]]
        assert run(step, extra) == run(make_step(resumed, resumed_weight), extra)
        single, _ = build(tf.distribute.get_strategy().scope(), method)
        assert single.set_rolling_state(saved)
        np.testing.assert_array_equal(single._rolling_ori_real.numpy(), saved["ori_real"][0])
"""


def test_two_replicas_pool_into_the_single_device_loss_of_the_concatenated_batch():
    # A fresh interpreter: two logical CPUs can only be configured before
    # TensorFlow initializes, which an earlier test has usually done.
    result = subprocess.run(
        [sys.executable, "-c", TWO_REPLICA_SCRIPT],
        cwd=Path(__file__).resolve().parents[1],
        env={**os.environ, "CUDA_VISIBLE_DEVICES": ""},
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr[-4000:]
