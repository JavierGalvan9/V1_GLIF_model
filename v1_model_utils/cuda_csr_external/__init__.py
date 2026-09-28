"""CUDA current accumulation for LGN and background connections."""

from .wrapper import (
    CsrConnectivity,
    SPECIALIZED_BATCH_SIZES,
    build_csr_connectivity,
    calculate_external_csr_currents,
    kernel_variant,
    uses_bkg_gather,
)

__all__ = (
    "CsrConnectivity",
    "SPECIALIZED_BATCH_SIZES",
    "build_csr_connectivity",
    "calculate_external_csr_currents",
    "kernel_variant",
    "uses_bkg_gather",
)
