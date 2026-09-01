"""Build and cache architecture-specific TensorFlow CUDA operators."""

from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile

import tensorflow as tf


def normalize_architecture(architecture):
    """Return a CUDA compute capability as digits, for example ``"86"``."""
    value = str(architecture).strip().lower().removeprefix("sm_").replace(".", "")
    if not value.isdigit():
        raise ValueError(f"Invalid CUDA architecture: {architecture!r}")
    return value


def active_gpu_architecture():
    """Return the common compute capability of all visible GPUs."""
    devices = tf.config.list_physical_devices("GPU")
    if not devices:
        raise RuntimeError("No GPU is visible; cannot select a CUDA operator.")
    capabilities = {
        tuple(tf.config.experimental.get_device_details(device)["compute_capability"])
        for device in devices
    }
    if len(capabilities) != 1:
        formatted = ", ".join(".".join(map(str, value)) for value in sorted(capabilities))
        raise RuntimeError(
            "All visible GPUs must have the same compute capability when using "
            f"custom CUDA operators; found: {formatted}."
        )
    major, minor = capabilities.pop()
    return f"{major}{minor}"


def _cache_component(value):
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", str(value)).strip("-") or "unknown"


def cuda_cache_root():
    """Return the persistent cache root, honoring standard cache overrides."""
    configured = os.environ.get("V1_CUDA_CACHE_DIR")
    if configured:
        return Path(configured).expanduser()
    xdg_cache = os.environ.get("XDG_CACHE_HOME")
    base = Path(xdg_cache).expanduser() if xdg_cache else Path.home() / ".cache"
    return base / "v1_glif" / "cuda"


def runtime_cache_key():
    """Return a readable key for the active TensorFlow/CUDA ABI."""
    build_info = tf.sysconfig.get_build_info()
    abi = {
        "tensorflow": tf.__version__,
        "tensorflow_cuda": str(build_info.get("cuda_version", "unknown")),
        "tensorflow_cudnn": str(build_info.get("cudnn_version", "unknown")),
        "compile_flags": list(tf.sysconfig.get_compile_flags()),
        "link_flags": list(tf.sysconfig.get_link_flags()),
    }
    digest = hashlib.sha256(
        json.dumps(abi, sort_keys=True).encode("utf-8")
    ).hexdigest()[:12]
    return (
        f"tf-{_cache_component(abi['tensorflow'])}_"
        f"cuda-{_cache_component(abi['tensorflow_cuda'])}_{digest}"
    )


def artifact_directory(directory, architecture):
    """Return the external cache directory for one operator module and GPU."""
    architecture = normalize_architecture(architecture)
    return (
        cuda_cache_root()
        / runtime_cache_key()
        / f"sm{architecture}"
        / Path(directory).resolve().name
    )


def artifact_path(directory, stem, architecture):
    """Return the cached shared-library path for an operator architecture."""
    return artifact_directory(directory, architecture) / f"_{stem}.so"


def metadata_path(directory, stem, architecture):
    """Return the environment metadata path for one cached operator."""
    return artifact_directory(directory, architecture) / f"{stem}_metadata.json"


def build_environment(architecture):
    """Describe ABI-sensitive inputs shared by every TensorFlow CUDA build."""
    build_info = tf.sysconfig.get_build_info()
    return {
        "architecture": normalize_architecture(architecture),
        "tensorflow": tf.__version__,
        "tensorflow_cuda": str(build_info.get("cuda_version", "unknown")),
        "tensorflow_cudnn": str(build_info.get("cudnn_version", "unknown")),
        "compile_flags": list(tf.sysconfig.get_compile_flags()),
        "link_flags": list(tf.sysconfig.get_link_flags()),
    }


def write_build_metadata(
    directory, stem, architecture, *, build_flags=(), source_digest=None
):
    """Atomically publish the ABI environment and operator-specific flags."""
    path = metadata_path(directory, stem, architecture)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = build_environment(architecture)
    payload["build_flags"] = list(build_flags)
    if source_digest is not None:
        payload["source_digest"] = source_digest
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)
    return path


@contextmanager
def temporary_build_directory(directory, stem, architecture):
    """Yield a cache-local workspace removed after the operator is linked."""
    parent = artifact_directory(directory, architecture)
    parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{stem}-build-", dir=parent
    ) as temporary:
        yield Path(temporary)


def _source_digest(sources):
    digest = hashlib.sha256()
    for source in sources:
        digest.update(source.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(source.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def ensure_artifact(
    directory,
    stem,
    *,
    sources,
    build_module,
    architecture=None,
    build_flags=(),
):
    """Return a fresh architecture-specific artifact, building it if needed.

    A per-architecture process lock makes this safe when multiple local workers
    start simultaneously. Build modules must publish their final library at the
    path returned by :func:`artifact_path`.
    """
    directory = Path(directory)
    architecture = normalize_architecture(
        architecture or active_gpu_architecture()
    )
    output = artifact_path(directory, stem, architecture)
    metadata = metadata_path(directory, stem, architecture)
    sources = tuple(Path(source) for source in sources)
    missing = [source for source in sources if not source.exists()]
    if missing:
        raise FileNotFoundError(
            "CUDA operator source does not exist: "
            + ", ".join(str(source) for source in missing)
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    lock_path = output.parent / f".{stem}.lock"
    command = [
        sys.executable,
        "-m",
        build_module,
        "--architecture",
        architecture,
    ]
    with lock_path.open("a", encoding="utf-8") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        expected_metadata = build_environment(architecture)
        expected_metadata["build_flags"] = list(build_flags)
        expected_metadata["source_digest"] = _source_digest(sources)
        try:
            cached_metadata = json.loads(metadata.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError):
            cached_metadata = None
        stale = (
            not output.exists()
            or cached_metadata != expected_metadata
        )
        if stale:
            try:
                subprocess.run(command, check=True)
            except subprocess.CalledProcessError as error:
                raise RuntimeError(
                    f"Failed to build CUDA operator for sm_{architecture}: "
                    f"{' '.join(command)}"
                ) from error
            if not output.exists():
                raise RuntimeError(
                    f"CUDA build for sm_{architecture} did not produce {output}; "
                    f"command: {' '.join(command)}"
                )
            write_build_metadata(
                directory,
                stem,
                architecture,
                build_flags=build_flags,
                source_digest=expected_metadata["source_digest"],
            )
            try:
                cached_metadata = json.loads(metadata.read_text(encoding="utf-8"))
            except (FileNotFoundError, json.JSONDecodeError) as error:
                raise RuntimeError(
                    f"CUDA build for sm_{architecture} did not produce valid "
                    f"metadata at {metadata}."
                ) from error
            if cached_metadata != expected_metadata:
                raise RuntimeError(
                    f"CUDA build metadata for sm_{architecture} does not match "
                    "the active TensorFlow/CUDA environment or build flags."
                )
    return output


def resolve_artifact(directory, stem):
    """Resolve the operator matching the active GPUs."""
    architecture = active_gpu_architecture()
    cached = artifact_path(directory, stem, architecture)
    if cached.exists():
        return cached
    raise FileNotFoundError(
        f"CUDA operator for sm_{architecture} is not built; run "
        f"`python -m v1_model_utils.{Path(directory).name}.build "
        f"--architecture {architecture}`. Expected: {cached}"
    )
