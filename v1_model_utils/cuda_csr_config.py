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
    architecture = int(str(architecture).lower().removeprefix("sm_").replace(".", ""))
    # The packed backward specialization is qualified on SM120 only; older
    # architectures keep the batch-lane path until they are measured.
    return (f"-DV1_PACKED_BACKWARD={int(architecture >= 120)}",)
