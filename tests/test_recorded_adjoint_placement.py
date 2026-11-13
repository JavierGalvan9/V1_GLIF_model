"""The recorded adjoint's loops stay on their replica's GPU under MirroredStrategy.

A loop invariant that is a bare variable read carries no device; its lowered
Enter node then lands on GPU:0 and every other replica copies it across GPUs on
each iteration (recorded_adjoint._on_replica_device). Only the runtime placement
shows this, so each case runs a tiny recorded-adjoint gradient step in a child
process with device-placement logging and reads the placer's log.
"""
import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import tensorflow as tf

ROOT = Path(__file__).resolve().parents[1]
BATCH, LENGTH = 4, 11
# Lowered while-loop control nodes of replica 1 that the placer put on GPU:0.
OFF_REPLICA = re.compile(
    r"placer\.cc:\d+\] (\S*replica_1\S*): \((Enter|Merge|Switch|NextIteration|Exit)\): \S*GPU:0$"
)


def _gpus():
    return tf.config.list_physical_devices("GPU")


def _run_child(mode, n_gpus):
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    devices = visible.split(",") if visible else [str(i) for i in range(len(_gpus()))]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=",".join(devices[:n_gpus]),
               V1_CSR_RESOURCE_MODE="1" if n_gpus > 1 else "0", TF_CPP_MIN_LOG_LEVEL="0")
    result = subprocess.run(
        [sys.executable, "-m", "tests.test_recorded_adjoint_placement", mode],
        cwd=ROOT, env=env, capture_output=True, text=True, timeout=1800,
    )
    assert result.returncode == 0, result.stdout[-3000:] + result.stderr[-3000:]
    return result.stdout, result.stderr


@pytest.mark.parametrize("bare", [False, True], ids=["fixed", "bare_reads"])
def test_two_replicas_keep_adjoint_loops_on_their_gpu(bare):
    """With the fix no replica-1 loop node is on GPU:0; with bare reads (the
    old code) some are, which shows the check can see the fault."""
    if len(_gpus()) < 2:
        pytest.skip("needs two GPUs")
    _, log = _run_child("bare" if bare else "fixed", 2)
    off = sorted({match.group(1) for line in log.splitlines()
                  if (match := OFF_REPLICA.search(line))})
    assert "placer.cc" in log, "no runtime placement log"
    if bare:
        assert off, "bare variable reads no longer put loop nodes on GPU:0; drop this case"
    else:
        assert not off, off[:20]


def test_one_gpu_fix_is_a_no_op():
    """On one replica the gradients are bitwise those of the bare reads."""
    if not _gpus():
        pytest.skip("needs a GPU")
    out, _ = _run_child("noop", 1)
    assert "NOOP_BITWISE True" in out, out[-3000:]


def test_two_replicas_roll_out_concurrently_with_offload():
    """No frame-store op joins the global chain into which TensorFlow's automatic
    control dependencies put every stateful op without a resource input, across
    replicas: there, replica 1's rollout (gated on its length check) would wait
    for replica 0's whole forward."""
    if len(_gpus()) < 2:
        pytest.skip("needs two GPUs")
    out, _ = _run_child("chain", 2)
    chain = next(line for line in out.splitlines() if line.startswith("CHAIN"))
    assert chain.split()[1:] == ["Assert", "Assert"], chain


def _child(mode):
    from tests.network_fixtures import load_network
    from v1_model_utils import cuda_csr_recurrent, models, recorded_adjoint, spatial_layout
    from v1_model_utils.recorded_adjoint import RecordedAdjointRunner, SpikeStatistics

    if mode != "noop":
        tf.debugging.set_log_device_placement(True)
    tf.keras.mixed_precision.set_global_policy("mixed_float16")
    strategy = tf.distribute.MirroredStrategy(cross_device_ops=tf.distribute.NcclAllReduce())
    network, lgn, bkg = load_network(1000)
    edge_orders = None
    if cuda_csr_recurrent.DIRECT_CSR:
        network, lgn, bkg, edge_orders = spatial_layout.apply_csr_edge_order(
            network, lgn, bkg, max_delay=3
        )
    n_input = int(lgn["n_inputs"])
    with strategy.scope():
        model = models.create_model(
            network, lgn, bkg, seq_len=LENGTH, n_input=n_input, n_output=2,
            dtype=tf.float16, batch_size=BATCH, max_delay=3, train_recurrent=True,
            train_noise=True, use_state_input=True, return_state=True, add_metric=False,
            seed=3, acceleration="cuda", edge_orders=edge_orders,
            track_voltage_penalty=True, return_voltage_sequences=False,
            recurrent_dampening_factor=0.1,
        )
        model.build((BATCH, LENGTH, n_input))
        cell = model.get_layer("rsnn").cell
        runner = RecordedAdjointRunner(
            models.build_sequence_and_state_model(model, model.get_layer("rsnn")), cell,
            sequence_length=LENGTH, chunk_size=LENGTH, differentiate_inputs=False,
            accumulated_variable=cell.accumulated_weight_gradient_variable,
            **(dict(voltage_offload="raw", history_offload="adaptive") if mode == "chain" else {}),
        )
    rng = np.random.default_rng(0)
    spikes = (rng.random((BATCH, LENGTH, n_input)) < 0.02).astype(np.float16)
    initial_spikes = (rng.random((BATCH, 3 * cell._n_neurons)) < 0.02).astype(np.float16)

    def gradients():
        def replica_step():
            # Built inside the replica, as the per-replica dataset batches are:
            # an eager tensor would live on GPU:0 for every replica.
            inputs = tf.constant(spikes)
            state = (tf.constant(initial_spikes), *cell.zero_state(BATCH, tf.float32)[1:])
            with tf.GradientTape() as tape:
                outputs = runner(inputs, state, SpikeStatistics((0, LENGTH), ()))
                loss = (tf.reduce_sum(tf.cast(outputs[0], tf.float32) * 1.3)
                        + tf.reduce_sum(outputs[2]))
            return tape.gradient(loss, model.trainable_variables)

        step = tf.function(lambda: strategy.run(replica_step))
        if mode == "chain":
            from tensorflow.python.framework import auto_control_deps
            graph = step.get_concrete_function().graph
            print("CHAIN", *(op.type for op in graph.get_operations()
                             if auto_control_deps.op_is_stateful(op)
                             and not any(t.dtype == tf.resource for t in op.inputs)))
            return []
        results = step()
        return [np.asarray(strategy.experimental_local_results(g)[0]) for g in results]

    fixed = recorded_adjoint._on_replica_device
    if mode == "bare":
        recorded_adjoint._on_replica_device = lambda value: value
    reference = gradients()
    if mode == "noop":
        recorded_adjoint._on_replica_device = lambda value: value
        bare = gradients()
        recorded_adjoint._on_replica_device = fixed
        print("NOOP_BITWISE", all(np.array_equal(a, b) for a, b in zip(reference, bare)))


if __name__ == "__main__":
    _child(sys.argv[1])
