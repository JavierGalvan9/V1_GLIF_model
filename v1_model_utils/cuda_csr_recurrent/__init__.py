"""Fused CUDA recurrent synaptic-current operator."""

from .wrapper import (
    DIRECT_CSR,
    MAX_SPIKE_SLOTS,
    CsrConnectivity,
    StepGradientLink,
    accumulate_recurrent_weight_gradient,
    build_csr_connectivity,
    calculate_recurrent_csr_currents,
    recurrent_backward,
    spike_activity_masks,
    spike_queue,
    to_csr_order,
    to_original_order,
)

__all__ = (
    "DIRECT_CSR",
    "MAX_SPIKE_SLOTS",
    "CsrConnectivity",
    "StepGradientLink",
    "accumulate_recurrent_weight_gradient",
    "build_csr_connectivity",
    "calculate_recurrent_csr_currents",
    "recurrent_backward",
    "spike_activity_masks",
    "spike_queue",
    "to_csr_order",
    "to_original_order",
)
