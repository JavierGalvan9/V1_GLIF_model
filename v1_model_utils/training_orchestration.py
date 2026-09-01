"""Process orchestration for one-process-per-GPU V1 training.

This module deliberately has no TensorFlow imports: ``multi_training.py`` calls
it before importing TensorFlow so every worker can expose exactly one GPU.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
import signal
import socket
import subprocess
import sys
import time


_WORKER_FLAG = "distributed_worker_index"
_MULTI_WORKER_NCCL_ARCHITECTURES = {"86"}


@dataclass(frozen=True)
class TrainingLaunchPlan:
    argv: tuple[str, ...]
    n_gpus: int
    worker_index: int | None
    compute_capability: str | None
    visible_devices: tuple[str, ...]
    worker_addresses: tuple[str, ...]
    per_replica_batch_size: int
    grating_batch_size: int
    gray_batch_size: int

    @property
    def launch_workers(self):
        return (
            self.n_gpus > 1
            and self.worker_index is None
            and self.compute_capability in _MULTI_WORKER_NCCL_ARCHITECTURES
        )

    @property
    def is_chief(self):
        return self.worker_index in (None, 0)


@dataclass(frozen=True)
class WorkerProcess:
    argv: tuple[str, ...]
    environ: dict[str, str]
    is_chief: bool


def _option(argv, name, default=None):
    option = f"--{name}"
    for index, value in enumerate(argv[1:], start=1):
        if value.startswith(option + "="):
            return value.split("=", 1)[1]
        if value == option:
            if index + 1 >= len(argv):
                raise ValueError(f"{option} requires a value")
            return argv[index + 1]
    return default


def _replace_options(argv, replacements, removals=()):
    names = set(replacements) | set(removals)
    result = [argv[0]]
    index = 1
    while index < len(argv):
        value = argv[index]
        matched = next(
            (name for name in names if value == f"--{name}" or value.startswith(f"--{name}=")),
            None,
        )
        if matched is None:
            result.append(value)
            index += 1
            continue
        if value == f"--{matched}":
            index += 2
        else:
            index += 1
    result.extend(f"--{name}={value}" for name, value in replacements.items())
    return tuple(result)


def _available_devices(environ):
    configured = environ.get("CUDA_VISIBLE_DEVICES")
    if configured is not None:
        return tuple(value.strip() for value in configured.split(",") if value.strip())
    try:
        output = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader,nounits"],
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return ()
    return tuple(line.strip() for line in output.splitlines() if line.strip())


def _reserve_worker_addresses(count):
    sockets = []
    try:
        for _ in range(count):
            listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            listener.bind(("127.0.0.1", 0))
            sockets.append(listener)
        return tuple(f"127.0.0.1:{item.getsockname()[1]}" for item in sockets)
    finally:
        for listener in sockets:
            listener.close()


def _query_compute_capabilities(devices):
    try:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,uuid,compute_cap",
                "--format=csv,noheader,nounits",
            ],
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError(
            "Could not query GPU compute capabilities before distributed launch."
        ) from exc
    capabilities = {}
    for line in output.splitlines():
        fields = tuple(field.strip() for field in line.split(","))
        if len(fields) != 3:
            continue
        index, uuid, capability = fields
        capabilities[index] = capability
        capabilities[uuid] = capability
    try:
        return tuple(capabilities[device] for device in devices)
    except KeyError as exc:
        raise RuntimeError(
            f"Could not resolve compute capability for visible GPU {exc.args[0]!r}."
        ) from exc


def _split_stimulus_batch(per_replica_batch, grating, gray):
    if grating <= 0 or gray <= 0:
        raise ValueError("All per-replica stimulus batch sizes must be positive.")
    total = grating + gray
    numerator = per_replica_batch * grating
    if numerator % total:
        raise ValueError(
            "global_batch_size cannot preserve the requested grating/gray split "
            "as integer per-replica batches."
        )
    resolved_grating = numerator // total
    resolved_gray = per_replica_batch - resolved_grating
    if resolved_grating <= 0 or resolved_gray <= 0:
        raise ValueError("The resolved grating/gray batches must both be positive.")
    return resolved_grating, resolved_gray


def plan_training_launch(
    argv, *, environ=None, worker_addresses=None, compute_capabilities=None
):
    """Validate CLI distribution settings and return an immutable launch plan."""
    argv = tuple(argv)
    environ = os.environ if environ is None else environ
    n_gpus = int(_option(argv, "n_gpus", 1))
    if n_gpus < 1:
        raise ValueError("n_gpus must be at least one.")
    worker_value = _option(argv, _WORKER_FLAG)
    worker_index = None if worker_value is None else int(worker_value)
    if worker_index is not None and not 0 <= worker_index < n_gpus:
        raise ValueError("distributed_worker_index is outside n_gpus.")

    batch = int(_option(argv, "batch_size", 2))
    grating = int(_option(argv, "grating_batch_size", 1))
    gray = int(_option(argv, "gray_batch_size", 1))
    global_value = _option(argv, "global_batch_size")
    global_batch = None if global_value is None else int(global_value)
    if global_batch not in (None, 0):
        if global_batch <= 0 or global_batch % n_gpus:
            raise ValueError("global_batch_size must be positive and divisible by n_gpus.")
        batch = global_batch // n_gpus
        grating, gray = _split_stimulus_batch(batch, grating, gray)
    elif batch <= 0 or grating <= 0 or gray <= 0 or grating + gray != batch:
        raise ValueError(
            "batch_size must be positive and equal grating_batch_size + gray_batch_size."
        )

    visible_devices = _available_devices(environ)
    architecture = None
    if worker_index is None and n_gpus > 1 and n_gpus > len(visible_devices):
        raise ValueError(
            f"Requested {n_gpus} GPU(s), but only {len(visible_devices)} are visible."
        )
    if worker_index is not None:
        visible_devices = tuple(filter(None, (environ.get("CUDA_VISIBLE_DEVICES", ""),)))
    else:
        visible_devices = visible_devices[:n_gpus]
    if n_gpus > 1 and worker_index is None:
        if compute_capabilities is None:
            compute_capabilities = _query_compute_capabilities(visible_devices)
        normalized_capabilities = {
            str(value).strip().lower().removeprefix("sm_").replace(".", "")
            for value in compute_capabilities
        }
        if len(normalized_capabilities) != 1:
            raise ValueError(
                "Multi-GPU training requires GPUs with the same compute capability."
            )
        architecture = normalized_capabilities.pop()
        if architecture not in {"80", "86", "89", "120"}:
            raise ValueError(
                f"Unsupported distributed GPU compute capability sm_{architecture}."
            )
    if worker_addresses is None and architecture in _MULTI_WORKER_NCCL_ARCHITECTURES:
        worker_addresses = _reserve_worker_addresses(n_gpus)

    rewritten = _replace_options(
        argv,
        {
            "batch_size": batch,
            "grating_batch_size": grating,
            "gray_batch_size": gray,
        },
        removals=("global_batch_size",),
    )
    return TrainingLaunchPlan(
        argv=rewritten,
        n_gpus=n_gpus,
        worker_index=worker_index,
        compute_capability=architecture,
        visible_devices=visible_devices,
        worker_addresses=tuple(worker_addresses or ()),
        per_replica_batch_size=batch,
        grating_batch_size=grating,
        gray_batch_size=gray,
    )


def build_worker_processes(plan, *, base_environ=None, enabled=False):
    """Build worker commands and environments without starting processes."""
    if not (plan.launch_workers or enabled):
        return ()
    if len(plan.worker_addresses) != plan.n_gpus:
        raise ValueError("One worker address is required per GPU.")
    base_environ = dict(os.environ if base_environ is None else base_environ)
    cluster = {"worker": list(plan.worker_addresses)}
    workers = []
    for index, device in enumerate(plan.visible_devices):
        environ = dict(base_environ)
        environ["CUDA_VISIBLE_DEVICES"] = device
        environ["V1_DISTRIBUTED_WORKER"] = "1"
        environ.setdefault("NCCL_P2P_DISABLE", "1")
        environ.setdefault("NCCL_DEBUG", "WARN")
        environ["TF_CONFIG"] = json.dumps(
            {"cluster": cluster, "task": {"type": "worker", "index": index}},
            separators=(",", ":"),
        )
        argv = _replace_options(
            plan.argv,
            {_WORKER_FLAG: index},
        )
        benchmark_output = _option(argv, "benchmark_output")
        if index and benchmark_output:
            root, extension = os.path.splitext(benchmark_output)
            argv = _replace_options(
                argv,
                {"benchmark_output": f"{root}.worker_{index}{extension}"},
            )
        profile_logdir = _option(argv, "profile_logdir")
        if index and profile_logdir:
            argv = _replace_options(
                argv,
                {"profile_logdir": worker_diagnostic_path(profile_logdir, index)},
            )
        workers.append(WorkerProcess((sys.executable, *argv), environ, index == 0))
    return tuple(workers)


def launch_worker_processes(plan, *, enabled=False):
    """Run all workers, propagate failures, and forward termination signals."""
    specs = build_worker_processes(plan, enabled=enabled)
    processes = [subprocess.Popen(spec.argv, env=spec.environ) for spec in specs]
    previous_handlers = {}

    def terminate_workers(_signum=None, _frame=None):
        for process in processes:
            if process.poll() is None:
                process.terminate()

    for signum in (signal.SIGINT, signal.SIGTERM):
        previous_handlers[signum] = signal.getsignal(signum)
        signal.signal(signum, terminate_workers)
    try:
        while True:
            statuses = [process.poll() for process in processes]
            failure = next((status for status in statuses if status not in (None, 0)), None)
            if failure is not None:
                for process, status in zip(processes, statuses):
                    if status is None:
                        process.terminate()
                for process in processes:
                    process.wait()
                return failure
            if all(status == 0 for status in statuses):
                return 0
            time.sleep(0.05)
    finally:
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)


def maybe_launch_training_workers(argv=None, environ=None):
    """Launch distributed workers from the parent, returning its exit code."""
    argv = tuple(sys.argv if argv is None else argv)
    target_environ = os.environ if environ is None else environ
    plan = plan_training_launch(argv, environ=target_environ)
    if not plan.launch_workers:
        if plan.worker_index is None and plan.visible_devices:
            target_environ["CUDA_VISIBLE_DEVICES"] = ",".join(plan.visible_devices)
        return None, plan
    return launch_worker_processes(plan), plan


def worker_diagnostic_path(path, worker_index):
    """Keep non-chief diagnostic output separate from chief artifacts."""
    if not path or worker_index in (None, 0):
        return path
    return os.path.join(path, "workers", f"worker_{worker_index}")


def current_process_gpu_memory_mib():
    """Return physical GPU memory attributed to this worker process."""
    try:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,used_gpu_memory",
                "--format=csv,noheader,nounits",
            ],
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    memory = 0.0
    found = False
    for line in output.splitlines():
        fields = tuple(field.strip() for field in line.split(","))
        if len(fields) != 2 or fields[0] != str(os.getpid()):
            continue
        try:
            memory += float(fields[1])
            found = True
        except ValueError:
            continue
    return memory if found else None
