"""Inspect executed, optimized history-offload graphs for host integer payloads.

Run in a fresh process for each frozen source tree (no Nsight required)::

    python tests/test_history_offload_placement.py --source /path/to/baseline --expect-host
    python tests/test_history_offload_placement.py --source /path/to/fixed

The actual recorded forward/gradient runs on a 1000-neuron CUDA model. Scalar
loop counters are excluded: only tensors descending from packed GLIF outputs
or packed frame-store reads are followed through structural control-flow ops.
"""

import argparse
import json
import os
import subprocess
from collections import defaultdict, deque
from pathlib import Path
import sys


STRUCTURAL = {"Identity", "Enter", "Exit", "Merge", "Switch", "NextIteration"}


def _edge(value):
    name, separator, port = value.rpartition(":")
    return (name, int(port)) if separator and port.isdigit() else (value, 0)


def _host_payloads(graph, packed_shape):
    """Find packed int32 data edges entering known GPU HostMemory kernels."""
    from tensorflow.core.framework import types_pb2

    consumers = defaultdict(list)
    pending = deque()
    for node in graph.node:
        for index, value in enumerate(node.input):
            if not value.startswith("^"):
                consumers[_edge(value)].append((node, index))
        if node.op == "FusedGlifStep":
            pending.extend(((node.name, 9), (node.name, 10)))
        elif node.op == "V1FrameStoreTake":
            shape = tuple(dim.size for dim in node.attr["shape"].shape.dim)
            if shape == packed_shape:
                pending.append((node.name, 0))
    seen, failures = set(), set()
    while pending:
        tensor = pending.popleft()
        if tensor in seen:
            continue
        seen.add(tensor)
        for node, index in consumers[tensor]:
            if node.op not in STRUCTURAL | {"Bitcast"}:
                continue
            # A Switch predicate is a scalar; only its data input propagates.
            if node.op == "Switch" and index != 0:
                continue
            if node.op in STRUCTURAL and node.attr["T"].type == types_pb2.DT_INT32:
                failures.add((node.name, node.op, node.device))
            pending.append((node.name, 0))
            if node.op == "Switch":
                pending.append((node.name, 1))
    return sorted(failures), len(seen)


def test_packed_history_stays_in_device_memory():
    import pytest
    import tensorflow as tf

    if not tf.config.list_physical_devices("GPU"):
        pytest.skip("CUDA GPU required")
    from tests.network_fixtures import network_path

    if not Path(network_path(1000)).is_file():
        pytest.skip("needs the cached 1000-neuron test network")
    root = Path(__file__).resolve().parents[1]
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=visible.split(",")[0],
               TF_CPP_MIN_LOG_LEVEL="1")
    result = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), "--source", str(root)],
        cwd=root, env=env, capture_output=True, text=True, timeout=1800,
    )
    assert result.returncode == 0, result.stdout[-8000:] + result.stderr[-3000:]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--expect-host", action="store_true")
    args = parser.parse_args()
    source = args.source.resolve()
    sys.path.insert(0, str(source))

    import numpy as np
    import tensorflow as tf
    from tensorflow.python.eager import context
    from tensorflow.python.framework import kernels

    from tests.test_recorded_adjoint import RecordedAdjointTest
    from v1_model_utils import recorded_adjoint
    from v1_model_utils.recorded_adjoint import SpikeStatistics

    assert Path(recorded_adjoint.__file__).resolve().is_relative_to(source)
    assert tf.config.list_physical_devices("GPU"), "a CUDA GPU is required"
    # Establish the runtime-specific premise rather than baking in a TF version.
    for op in STRUCTURAL:
        registry = kernels.get_registered_kernels_for_op(op)
        assert any(
            kernel.device_type == "GPU" and kernel.host_memory_arg
            and any(constraint.name == "T" and tf.int32.as_datatype_enum in
                    constraint.allowed_values.list.type for constraint in kernel.constraint)
            for kernel in registry.kernel
        ), f"{op}: integer HostMemory premise changed; inspect this regression test"

    tf.keras.mixed_precision.set_global_policy("mixed_float16")
    case = RecordedAdjointTest()
    model, layer, n_input = case._model()
    runner = case._runner(model, layer, 4, "raw", "raw")
    runner._PREFETCH, runner._COPY_GROUP = 3, 2
    rng = np.random.default_rng(17)
    inputs = tf.constant((rng.random((case.BATCH, case.SEQ_LEN, n_input)) < 0.02)
                         .astype(np.float16))
    initial = list(layer.cell.zero_state(case.BATCH, tf.float32))
    initial[0] = tf.constant((rng.random(initial[0].shape) < 0.02).astype(np.float16))

    @tf.function(autograph=False)
    def step():
        with tf.GradientTape() as tape:
            outputs = runner(inputs, tuple(initial), SpikeStatistics((0, case.SEQ_LEN)))
            loss = tf.reduce_sum(tf.cast(outputs[0], tf.float32)) + tf.reduce_sum(outputs[2])
        return tape.gradient(loss, model.trainable_variables)

    # Collection happens while the executor builds and runs the optimized graph,
    # after functional control flow has been lowered to runtime transport nodes.
    with context.collect_graphs() as graphs:
        gradients = step()
        for gradient in gradients:
            assert np.isfinite(gradient.numpy()).all()
    shape = (case.BATCH, -(-layer.cell._n_neurons // 31))
    assert any(
        node.op == "V1FrameStoreTake"
        and tuple(dim.size for dim in node.attr["shape"].shape.dim) == shape
        for graph in graphs for node in graph.node
    ), "history offload fell back to resident storage"
    findings, tracked = [], 0
    for graph in graphs:
        failures, count = _host_payloads(graph, shape)
        findings.extend(failures)
        tracked += count
    assert graphs and tracked, "no executed packed-history graph was collected"
    print(json.dumps({"source": str(source), "graphs": len(graphs),
                      "tracked_tensors": tracked, "host_payloads": findings}, indent=2))
    assert bool(findings) == args.expect_host, (
        "expected known baseline host payloads" if args.expect_host
        else "packed history traverses HostMemory control-flow kernels"
    )


if __name__ == "__main__":
    main()
