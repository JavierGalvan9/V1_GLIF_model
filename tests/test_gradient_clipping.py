"""``--global_clipnorm`` clips the summed gradient, whatever the replica count.

The optimizers ``create_optimizer`` builds measure and clip the gradient in
Keras' private ``_all_reduce_sum_gradients`` hook, after the cross-replica sum.
These tests fail if Keras stops calling that hook: the stored norm would stay
at zero and the update would go unclipped.
"""

import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import tensorflow as tf

from v1_model_utils.optimizers import apply_gradients_and_measure, create_optimizer


def make_optimizer(clipnorm, dtype="float32"):
    flags = SimpleNamespace(optimizer="sgd", dtype=dtype, global_clipnorm=clipnorm)
    variable = tf.Variable([0.0, 0.0])
    return variable, create_optimizer(flags, 1.0, [variable])


def test_keras_clipping_is_not_used():
    _, optimizer = make_optimizer(1.0)

    assert optimizer.global_clipnorm is None
    assert optimizer.summed_clipnorm == 1.0


def test_single_replica_update_is_clipped_and_its_norm_stored():
    variable, optimizer = make_optimizer(1.0)

    norm = apply_gradients_and_measure(optimizer, [tf.constant([6.0, 8.0])], [variable])

    np.testing.assert_allclose(norm.numpy(), 10.0)
    np.testing.assert_allclose(optimizer.summed_gradient_norm.numpy(), 10.0)
    np.testing.assert_allclose(variable.numpy(), [-0.6, -0.8], rtol=1e-6)


def test_nonpositive_limit_measures_without_clipping():
    variable, optimizer = make_optimizer(0.0)

    norm = apply_gradients_and_measure(optimizer, [tf.constant([6.0, 8.0])], [variable])

    np.testing.assert_allclose(norm.numpy(), 10.0)
    np.testing.assert_allclose(variable.numpy(), [-6.0, -8.0], rtol=1e-6)


def test_loss_scaled_norm_and_limit_are_in_unscaled_units():
    variable, optimizer = make_optimizer(1.0, dtype="float16")
    scale = float(optimizer.dynamic_scale)
    assert scale != 1.0

    norm = apply_gradients_and_measure(
        optimizer, [tf.constant([6.0, 8.0]) * scale], [variable]
    )

    np.testing.assert_allclose(norm.numpy(), 10.0, rtol=1e-6)
    np.testing.assert_allclose(variable.numpy(), [-0.6, -0.8], rtol=1e-6)


def test_skipped_update_reports_nan():
    variable, optimizer = make_optimizer(1.0, dtype="float16")
    apply_gradients_and_measure(optimizer, [tf.constant([6.0, 8.0])], [variable])

    norm = apply_gradients_and_measure(
        optimizer, [tf.constant([np.inf, 0.0])], [variable]
    )

    assert np.isnan(norm.numpy())
    assert int(optimizer.iterations) == 1


TWO_REPLICA_SCRIPT = """
import numpy as np
import tensorflow as tf
tf.config.set_logical_device_configuration(
    tf.config.list_physical_devices("CPU")[0],
    [tf.config.LogicalDeviceConfiguration()] * 2,
)
from tests.test_gradient_clipping import make_optimizer
from v1_model_utils.optimizers import apply_gradients_and_measure

strategy = tf.distribute.MirroredStrategy(["/CPU:0", "/CPU:1"])
with strategy.scope():
    variable, optimizer = make_optimizer(1.0)

@tf.function
def step():
    def replica_fn():
        replica = tf.distribute.get_replica_context().replica_id_in_sync_group
        # Norms 1.25 and 3.75: clipping each share to 1.0 before the sum
        # would apply [-1.2, -1.6]; their sum [3, 4] clips to [-0.6, -0.8].
        gradient = tf.constant([0.75, 1.0]) * tf.cast(1 + 2 * replica, tf.float32)
        return apply_gradients_and_measure(optimizer, [gradient], [variable])
    return strategy.experimental_local_results(strategy.run(replica_fn))

norms = [norm.numpy() for norm in step()]
np.testing.assert_allclose(norms, [5.0, 5.0], rtol=1e-6)
np.testing.assert_allclose(variable.numpy(), [-0.6, -0.8], rtol=1e-6)
"""


def test_two_replicas_clip_the_sum_of_their_gradients():
    # A fresh interpreter: two logical CPUs can only be configured before
    # TensorFlow initializes, which an earlier test has usually done.
    result = subprocess.run(
        [sys.executable, "-c", TWO_REPLICA_SCRIPT],
        cwd=Path(__file__).resolve().parents[1],
        env={**os.environ, "CUDA_VISIBLE_DEVICES": ""},
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
