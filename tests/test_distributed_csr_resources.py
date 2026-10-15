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


def _recurrent_and_lgn(width=40, n_post=50, n_lgn=12, batch=3):
    """A three-slot recurrent history (rows never repeat a target) and an
    LGN-like input (rows do), in their operators' CSR orders, FP32."""
    rng = np.random.default_rng(5)
    flat = np.unique(rng.integers(0, 3 * width * n_post, 900))
    rec_indices = np.stack((flat % n_post, flat // n_post), axis=1)
    rec_types = rng.integers(0, 3, flat.size)
    order = np.lexsort((rec_types, rec_indices[:, 0], rec_indices[:, 1]))
    lgn_posts = rng.integers(0, n_post, 300)
    lgn_indices = np.stack((lgn_posts, rng.integers(0, n_lgn, 300)), axis=1)
    lgn_indices = np.concatenate((lgn_indices, lgn_indices[:40]))   # repeated targets
    lgn_order = np.lexsort((np.arange(len(lgn_indices)), lgn_indices[:, 0], lgn_indices[:, 1]))
    spikes = (rng.random((batch, 3 * width)) < 0.3).astype(np.float32)
    return dict(
        rec=(rec_indices[order], rec_types[order], 3 * width, n_post),
        lgn=(lgn_indices[lgn_order], rng.integers(0, 3, len(lgn_indices)), n_lgn, n_post),
        rec_weights=rng.normal(size=flat.size).astype(np.float32),
        lgn_weights=rng.normal(size=len(lgn_indices)).astype(np.float32),
        slots=tuple(tf.split(tf.constant(spikes), 3, axis=1)),
        activity=tf.constant((rng.random((batch, n_lgn)) < 0.4).astype(np.float32)),
        basis=tf.constant(rng.normal(size=(3, 4)).astype(np.float32)),
        upstream=tf.constant(rng.normal(size=(batch * n_post, 4)).astype(np.float32)),
    )


@requires_two_gpus
def test_every_replica_matches_the_single_gpu_fused_step(monkeypatch):
    """The model's recurrent + LGN step (one fused operator) reads both
    connectivities from each replica's own GPU resource and reproduces the
    single-GPU tensor-backend currents and gradients on every replica."""
    from v1_model_utils.cuda_csr_external import (
        build_csr_connectivity as build_external,
        calculate_recurrent_and_external_currents,
        fusable,
    )
    from v1_model_utils.cuda_csr_recurrent import spike_queue

    case = _recurrent_and_lgn()

    def connectivities():
        recurrent = build_csr_connectivity(*case["rec"], weights_csr_ordered=DIRECT_CSR)
        lgn = build_external(*case["lgn"], weights_csr_ordered=DIRECT_CSR)
        assert fusable(case["slots"], case["basis"], recurrent, (), case["activity"], lgn)
        return recurrent, lgn

    def step(recurrent, lgn, rec_weights, lgn_weights):
        slots, activity = case["slots"], case["activity"]
        with tf.GradientTape() as tape:
            tape.watch(slots)
            currents, _ = calculate_recurrent_and_external_currents(
                slots, rec_weights, case["basis"], 0.37, recurrent,
                tuple(spike_queue(slot) for slot in slots[1:]), activity, lgn_weights, lgn,
                compute_activity_gradient=False,
            )
            loss = tf.reduce_sum(currents * case["upstream"])
        return (currents, *tape.gradient(loss, (slots, rec_weights, lgn_weights)))

    monkeypatch.setenv("V1_CSR_RESOURCE_MODE", "0")
    expected = step(*connectivities(), tf.Variable(case["rec_weights"]),
                    tf.Variable(case["lgn_weights"]))

    monkeypatch.setenv("V1_CSR_RESOURCE_MODE", "1")
    recurrent, lgn = connectivities()
    assert recurrent.resource_name is not None and lgn.resource_name is not None
    strategy = tf.distribute.MirroredStrategy(
        cross_device_ops=tf.distribute.NcclAllReduce()
    )
    assert strategy.num_replicas_in_sync >= 2
    with strategy.scope():
        rec_weights = tf.Variable(case["rec_weights"])
        lgn_weights = tf.Variable(case["lgn_weights"])

    @tf.function
    def replica_step():
        return strategy.run(lambda: step(recurrent, lgn, rec_weights, lgn_weights))

    actual = replica_step()
    assert "V1CsrForwardFusedResource" in {
        op.type for op in replica_step.get_concrete_function().graph.get_operations()
    }
    for reference, per_replica in zip(tf.nest.flatten(expected), tf.nest.flatten(actual)):
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
