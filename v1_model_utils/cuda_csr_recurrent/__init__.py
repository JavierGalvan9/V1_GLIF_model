"""Fused CUDA recurrent synaptic-current operator."""

from .wrapper import (
    SPECIALIZED_BATCH_SIZES,
    CsrConnectivity,
    build_csr_connectivity,
    calculate_recurrent_csr_currents,
    kernel_variant,
)

__all__ = (
    "SPECIALIZED_BATCH_SIZES",
    "CsrConnectivity",
    "build_csr_connectivity",
    "calculate_recurrent_csr_currents",
    "kernel_variant",
)
