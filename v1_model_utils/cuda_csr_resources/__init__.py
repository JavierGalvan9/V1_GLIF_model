"""GPU-local opaque CSR connectivity resources for distributed workers."""

from .wrapper import (
    ResourceConnectivity,
    initialize_resource,
    load_ops,
    resource_mode_enabled,
)

__all__ = (
    "ResourceConnectivity",
    "initialize_resource",
    "load_ops",
    "resource_mode_enabled",
)
