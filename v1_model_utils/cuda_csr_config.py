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
    #
    # Its spike-gradient row kernel runs fastest in 128-thread blocks of 32 rows
    # at eight blocks per SM (64 registers, 32 warps), and its event weight
    # gradient with 32-edge queue items (30 registers, 64 warps per SM, where
    # 256-edge items took 99 registers and 16 warps) on a grid of 16 blocks per
    # SM (2,368).
    # Launch shape only: every sum keeps its order, so results are bitwise
    # those of the default shape.
    if str(architecture) == "100":
        return ("-DV1_SWEEP_TARGETS=8192", "-DV1_TILE_TARGETS=8192",
                "-DV1_ROWTILE_WARPS=4", "-DV1_ROWTILE_MIN_BLOCKS=8",
                "-DV1_EVENT_CHUNK=32", "-DV1_EVENT_GRAD_BLOCKS=2368")
    return ()
