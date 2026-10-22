"""Pick idle GPUs spread across PCIe root ports before CUDA initializes.

The GPUs behind one host PCIe root port share its uplink, which the recorded
adjoint's voltage offload saturates (~25 GB per replica and step), so replicas
are placed on the least occupied root port first. Selection only ever narrows
an unconstrained host: any explicit device mask or scheduler allocation wins,
and the module stays free of TensorFlow imports because it must run before
CUDA reads ``CUDA_VISIBLE_DEVICES``.
"""

import csv
from dataclasses import dataclass
import os
from pathlib import Path
import re
import subprocess

BDF = re.compile(r"^[0-9a-f]{4}:[0-9a-f]{2}:[0-9a-f]{2}\.[0-7]$")
MASKS = ("CUDA_VISIBLE_DEVICES", "NVIDIA_VISIBLE_DEVICES", "GPU_DEVICE_ORDINAL",
         "SLURM_JOB_GPUS", "SLURM_STEP_GPUS")
SCHEDULERS = ("SLURM_JOB_ID", "PBS_JOBID", "LSB_JOBID", "JOB_ID")
SYSFS = Path("/sys/bus/pci/devices")


class GPUSelectionError(RuntimeError):
    """No placement satisfies the selection rules; never fall back to sharing."""


@dataclass(frozen=True)
class GPU:
    index: int
    uuid: str
    bus_id: str
    name: str
    total_mib: int
    free_mib: int
    utilization: int
    compute_cap: str
    root_port: str
    ancestry: tuple[str, ...]
    compute_busy: bool


def normalize_bdf(value):
    """Map NVIDIA's eight-digit PCI domain onto the four-digit sysfs name."""
    domain, bus, device = value.lower().split(":")
    result = f"{int(domain, 16):04x}:{bus}:{device}"
    if not BDF.fullmatch(result):
        raise ValueError(f"Invalid PCI BDF: {value}")
    return result


def pci_ancestry(bus_id, sysfs=SYSFS):
    """Return the PCI path from the host root port down to the device."""
    device = sysfs / normalize_bdf(bus_id)
    if not device.exists():
        raise ValueError(f"Missing PCI sysfs device: {device}")
    ancestry = tuple(p.lower() for p in device.resolve().parts if BDF.fullmatch(p.lower()))
    if len(ancestry) < 2:
        raise ValueError(f"Cannot identify an upstream PCIe root port: {device}")
    return ancestry


def selection_blockers(environ):
    """Variables that hand device placement to the user or a scheduler.

    Present-but-empty masks count: an empty ``CUDA_VISIBLE_DEVICES`` is an
    explicit request for no GPU. Inside a scheduler job the visible devices are
    the allocation, and numeric masks there are not physical ``nvidia-smi``
    indices, so balancing is left to the scheduler.
    """
    return [key for key in (*MASKS, *SCHEDULERS) if key in environ]


def _query(arguments):
    return subprocess.run(["nvidia-smi", *arguments], check=True, text=True,
                          capture_output=True).stdout


def discover(query=_query, sysfs=SYSFS):
    """Read the physical GPU inventory from ``nvidia-smi`` and PCI sysfs."""
    rows = csv.reader(query([
        "--query-gpu=index,uuid,pci.bus_id,name,memory.total,memory.free,utilization.gpu,compute_cap",
        "--format=csv,noheader,nounits",
    ]).splitlines(), skipinitialspace=True)
    busy = set(query(["--query-compute-apps=gpu_uuid", "--format=csv,noheader,nounits"])
               .split())
    result = []
    for row in rows:
        index, uuid, bus, name, total, free, utilization, capability = (x.strip() for x in row)
        ancestry = pci_ancestry(bus, sysfs)
        result.append(GPU(int(index), uuid, normalize_bdf(bus), name, int(total),
                          int(free), int(utilization), capability, ancestry[0],
                          ancestry, uuid in busy))
    return result


def _capability_key(gpu):
    return tuple(int(part) for part in gpu.compute_cap.split(".")), gpu.total_mib


def select(gpus, count, model=None, min_free_fraction=0.9, max_utilization=0):
    """Choose ``count`` idle GPUs of one model, balanced across root ports.

    ``model`` defaults to the host's newest, largest GPU model (highest compute
    capability, then memory), so a lesser card is never mixed in nor silently
    substituted when too few of the target model are idle. A GPU is idle when it
    runs no compute process, its sampled utilization is at most
    ``max_utilization`` and at least ``min_free_fraction`` of its memory is free.
    Placement repeatedly takes the lowest-index GPU of the root port with the
    fewest selected GPUs (ties by root BDF), which minimizes the maximum
    occupancy of any uplink and is deterministic.
    """
    if count < 1 or not 0 <= min_free_fraction <= 1 or not 0 <= max_utilization <= 100:
        raise ValueError("Count must be positive; memory and utilization thresholds must be valid")
    if not gpus:
        raise GPUSelectionError("nvidia-smi reports no GPUs")
    if model is None:
        model = max(gpus, key=_capability_key).name
    candidates = [gpu for gpu in gpus if gpu.name == model and not gpu.compute_busy
                  and gpu.free_mib >= min_free_fraction * gpu.total_mib
                  and gpu.utilization <= max_utilization]
    if len(candidates) < count:
        raise GPUSelectionError(
            f"Need {count} idle {model} GPU(s); found {len(candidates)} "
            f"(indices {[gpu.index for gpu in candidates]}). Set CUDA_VISIBLE_DEVICES "
            "explicitly, or pass --noauto_select_gpus to keep the first visible GPUs."
        )
    groups = {}
    for gpu in sorted(candidates, key=lambda gpu: (gpu.index, gpu.uuid)):
        groups.setdefault(gpu.root_port, []).append(gpu)
    selected, occupancy = [], dict.fromkeys(groups, 0)
    for _ in range(count):
        root = min((root for root, queue in groups.items() if queue),
                   key=lambda root: (occupancy[root], root))
        selected.append(groups[root].pop(0))
        occupancy[root] += 1
    return selected


def auto_select_gpus(count, environ=None):
    """Expose ``count`` root-balanced idle GPUs by UUID; return them or None.

    Returns None, leaving the environment untouched, when the placement belongs
    to someone else (see ``selection_blockers``) or ``nvidia-smi`` cannot run
    (no GPU host). Any other failure, including unresolved PCI topology or too
    few idle GPUs, raises ``GPUSelectionError`` instead of sharing a link.
    """
    environ = os.environ if environ is None else environ
    blockers = selection_blockers(environ)
    if blockers:
        print(f"GPU auto-selection skipped: {', '.join(blockers)} set; "
              "the explicit mask or scheduler allocation decides placement.")
        return None
    try:
        inventory = discover()
    except OSError as error:
        print(f"GPU auto-selection skipped: nvidia-smi unavailable ({error}).")
        return None
    except (ValueError, subprocess.CalledProcessError) as error:
        raise GPUSelectionError(f"GPU auto-selection failed: {error}") from error
    selected = select(inventory, count)
    environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    environ["CUDA_VISIBLE_DEVICES"] = ",".join(gpu.uuid for gpu in selected)
    occupancy = {}
    for gpu in selected:
        occupancy.setdefault(gpu.root_port, []).append(gpu.index)
    print("GPU auto-selection (PCIe root port: physical indices): "
          + "; ".join(f"{root}: {indices}" for root, indices in sorted(occupancy.items())))
    roots = {gpu.root_port for gpu in inventory if gpu.name == selected[0].name}
    if max(map(len, occupancy.values())) > -(-count // len(roots)):
        print(f"WARNING: busy GPUs force replicas to share a PCIe root port ({len(roots)} "
              f"hold this model); expect slower voltage offload than a balanced placement.")
    print(f"CUDA_VISIBLE_DEVICES={environ['CUDA_VISIBLE_DEVICES']}")
    return selected
