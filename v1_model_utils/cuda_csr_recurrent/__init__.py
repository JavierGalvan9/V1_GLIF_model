"""Fused CUDA recurrent synaptic-current operator."""

from .wrapper import (
    DIRECT_CSR,
    SPECIALIZED_BATCH_SIZES,
    CsrConnectivity,
    build_csr_connectivity,
    calculate_recurrent_csr_currents,
    kernel_variant,
    to_csr_order,
    to_original_order,
)

__all__ = (
    "DIRECT_CSR",
    "SPECIALIZED_BATCH_SIZES",
    "CsrConnectivity",
    "build_csr_connectivity",
    "calculate_recurrent_csr_currents",
    "kernel_variant",
    "to_csr_order",
    "to_original_order",
)
