"""Build-time contracts shared by all CSR CUDA operator libraries."""

# Kernels index weights directly by CSR position. Model construction and
# checkpoint translation use this same value, so the layouts cannot diverge.
DIRECT_CSR = True
