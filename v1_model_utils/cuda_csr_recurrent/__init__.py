"""Fused CUDA recurrent synaptic-current operator."""

from .wrapper import (
    DIRECT_CSR,
    MAX_SPIKE_SLOTS,
    CsrConnectivity,
    accumulate_recurrent_weight_gradient,
    build_csr_connectivity,
    calculate_recurrent_csr_currents,
    spike_queue,
    to_csr_order,
    to_original_order,
)

__all__ = (
    "DIRECT_CSR",
    "MAX_SPIKE_SLOTS",
    "CsrConnectivity",
    "accumulate_recurrent_weight_gradient",
    "build_csr_connectivity",
    "calculate_recurrent_csr_currents",
    "spike_queue",
    "to_csr_order",
    "to_original_order",
)
