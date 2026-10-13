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
    # No kernel specializes by architecture at present (every architecture runs
    # the same backward); add any such flag here and nowhere else.
    del architecture
    return ()
