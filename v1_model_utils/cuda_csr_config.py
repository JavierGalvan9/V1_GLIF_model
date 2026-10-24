"""Build-time contracts shared by all CSR CUDA operator libraries."""

# Kernels index weights directly by CSR position. Model construction and
# checkpoint translation use this same value, so the layouts cannot diverge.
DIRECT_CSR = True


def architecture_kernel_flags(architecture):
    """CSR kernel compile flags that depend on the target compute capability.

    Every library that compiles these kernels has to pass the same values. The
    resource operators include the recurrent and external kernel sources
    verbatim, so a flag set for one library and not the other silently compiles
    two different kernels from one source file - which is how the resource
    backend lost the packed backward specialization and ran a third slower.
    """
    # sm_100 allows 227 KiB of shared memory per block (sm_120: 99 KiB), so its
    # tile sweep and tiled forward take the widest tile the 13-bit target key
    # allows: 8,192 targets x 4 FP32 = 128 KiB of accumulators.
    if str(architecture) == "100":
        return ("-DV1_SWEEP_TARGETS=8192", "-DV1_TILE_TARGETS=8192")
    return ()
