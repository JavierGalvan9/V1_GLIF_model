"""Replicated training must read connectivity from its own GPU's resource.

Two replicas in one process share the graph, so the CSR metadata has to be
resolved per device. These tests pin down both halves of that contract: the
resource is created once per visible GPU, and every replica reproduces the
single-GPU result exactly.
"""

import numpy as np
import pytest
import tensorflow as tf

from v1_model_utils import cuda_csr_resources
from v1_model_utils.cuda_csr_recurrent import (
    DIRECT_CSR,
    build_csr_connectivity,
    calculate_recurrent_csr_currents,
)


requires_two_gpus = pytest.mark.skipif(
    len(tf.config.list_physical_devices("GPU")) < 2,
    reason="two CUDA GPUs required",
)


def test_resource_is_created_once_per_visible_gpu(monkeypatch):
    created = []

    class FakeOps:
        @staticmethod
        def initialize_v1_csr_resource(*values, resource_name):
            created.append(resource_name)
            return tf.constant(True)

    monkeypatch.delenv("V1_DISTRIBUTED_WORKER", raising=False)
    monkeypatch.setattr(
        cuda_csr_resources.wrapper.tf.config,
        "list_logical_devices",
        lambda _: [
            tf.config.LogicalDevice("/device:GPU:0", "GPU"),
            tf.config.LogicalDevice("/device:GPU:3", "GPU"),
        ],
    )
    monkeypatch.setattr(cuda_csr_resources.wrapper, "load_ops", lambda: FakeOps)

    class Metadata:
        pass

    metadata = Metadata()
    for field in cuda_csr_resources.wrapper._METADATA_FIELDS:
        setattr(metadata, field, tf.zeros((2,), tf.int32))

    resource = cuda_csr_resources.wrapper.initialize_resource(metadata)

    # The kernel appends the executing device's ordinal to the base name, so
    # the ordinal in the name must come from the device, not from enumeration.
    assert created == [
        f"{resource.name}_gpu0",
        f"{resource.name}_gpu3",
    ]


@requires_two_gpus
def test_every_replica_matches_the_single_gpu_result(monkeypatch):
    indices = np.array([[0, 0], [1, 0], [0, 1], [1, 2]], np.int64)
    types = np.array([0, 1, 1, 0], np.int64)
    spikes = tf.constant([[1.0, 0.5, 0.25], [0.0, 1.0, 0.5]])
    basis = tf.constant([[1.0, 0.5, -0.5, 0.25], [0.5, -1.0, 0.25, 1.0]])
    upstream = tf.reshape(tf.range(16, dtype=tf.float32) / 7, (4, 4))

    monkeypatch.setenv("V1_CSR_RESOURCE_MODE", "0")
    tensor_connectivity = build_csr_connectivity(
        indices, types, 3, 2, weights_csr_ordered=DIRECT_CSR
    )
    weights = tf.Variable([0.2, -0.4, 0.7, 0.1])
    with tf.GradientTape() as tape:
        tape.watch(spikes)
        expected_currents = calculate_recurrent_csr_currents(
            spikes, weights, basis, 0.37, tensor_connectivity
        )
        loss = tf.reduce_sum(expected_currents * upstream)
    expected = (expected_currents,) + tape.gradient(loss, (spikes, weights))

    monkeypatch.setenv("V1_CSR_RESOURCE_MODE", "1")
    resource_connectivity = build_csr_connectivity(
        indices, types, 3, 2, weights_csr_ordered=DIRECT_CSR
    )
    assert resource_connectivity.resource_name is not None

    strategy = tf.distribute.MirroredStrategy(
        cross_device_ops=tf.distribute.NcclAllReduce()
    )
    assert strategy.num_replicas_in_sync >= 2
    with strategy.scope():
        replica_weights = tf.Variable([0.2, -0.4, 0.7, 0.1])

    @tf.function
    def replica_step():
        def forward():
            with tf.GradientTape() as replica_tape:
                replica_tape.watch(spikes)
                currents = calculate_recurrent_csr_currents(
                    spikes, replica_weights, basis, 0.37, resource_connectivity
                )
                replica_loss = tf.reduce_sum(currents * upstream)
            spike_grad, weight_grad = replica_tape.gradient(
                replica_loss, (spikes, replica_weights)
            )
            return currents, spike_grad, weight_grad

        return strategy.run(forward)

    actual = replica_step()
    for reference, per_replica in zip(expected, actual):
        for value in strategy.experimental_local_results(per_replica):
            np.testing.assert_allclose(
                value.numpy(), reference.numpy(), rtol=0, atol=2e-6
            )


def test_a_worker_only_addresses_its_own_gpu(monkeypatch):
    """One process per GPU must not place metadata on a peer worker's device.

    MultiWorkerMirroredStrategy makes every worker's devices visible in the
    eager context, and reaching one before its remote service is up fails.
    """
    monkeypatch.setenv("V1_DISTRIBUTED_WORKER", "1")
    monkeypatch.setattr(
        cuda_csr_resources.wrapper.tf.config,
        "list_logical_devices",
        lambda _: [
            tf.config.LogicalDevice("/job:worker/replica:0/task:0/device:GPU:0", "GPU"),
            tf.config.LogicalDevice("/job:worker/replica:0/task:1/device:GPU:0", "GPU"),
        ],
    )
    assert cuda_csr_resources.wrapper.replica_devices() == ("/device:GPU:0",)
