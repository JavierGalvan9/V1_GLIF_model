# GPU-local CSR resources

This internal production module stores recurrent, LGN, and background CSR
metadata in an opaque operator resource owned by a GPU rather than in the
graph. As ordinary tensors that metadata - 92 MiB for the recurrent matrix and
125 MiB for the LGN input - is placed on one device and copied to every other
replica on each use, 500 timesteps per training step, which both unbalances GPU
memory badly enough to reach 4x and serializes the step behind those copies.

`initialize_resource` creates one resource per GPU this process will place
replicas on, under a shared base name. The kernels resolve the name by
executing device (`DeviceResourceName` appends the GPU ordinal), so each
replica reads metadata that already lives on its own GPU. A one-process-per-GPU
worker addresses only `/device:GPU:0`: under `MultiWorkerMirroredStrategy` its
eager context also lists the other workers' devices, and reaching one before
its remote service is up fails.

Resource mode is declared, never inferred from how many GPUs happen to be
visible: `tf_utils.create_distribution_strategy` sets `V1_CSR_RESOURCE_MODE=1`
when it builds a multi-replica strategy, and the one-process-per-GPU launcher
sets `V1_DISTRIBUTED_WORKER`. `V1_CSR_RESOURCE_MODE` forces either backend for
tests and benchmarks. Single-replica training keeps the tensor-backed
operators.

The library provides recurrent forward/backward (including the in-place
accumulating backward, see `cuda_csr_recurrent/README.md`), the background
input's fixed-four forward gather (`BkgCsrForwardResource`, reading the
incoming CSR stored in the resource), and the external weight-only and
activity backward operations. Two contracts are shared with the tensor
backend and must not drift:

- **Compile flags.** `csr_resource_ops.cu.cc` `#include`s the recurrent and
  external kernel sources verbatim, so any flag set for one library and not the
  other compiles two different kernels from one source file.
  `cuda_csr_config.architecture_kernel_flags` is the single definition of the
  architecture-dependent flags for all three build modules.
- **Kernel selection.** Both backends call the same launchers: one
  pair-projected recurrent backward for every shape, and the shared
  event-driven weight gradient for the external inputs.
- **Operator interface.** Like the tensor backend, the forward finds its
  active rows on the device (there is no `active_indices` input) and every
  operator takes the synaptic basis in FP32. The resource stores the metadata
  in the same widths the tensor operators take (uint32 indices, uint8 synapse
  types), so the included kernel sources compile unchanged.

A connectivity declared without a backward (`needs_backward=False`, the LGN and
BKG inputs) uploads an empty pair projection. Initialization accepts that; the
backward operators refuse to run without it.

Missing or stale libraries are built automatically under a per-architecture
lock. They may also be prebuilt explicitly:

```bash
python -m v1_model_utils.cuda_csr_resources.build --architecture 120
```

Architecture-keyed binaries for `sm80`, `sm86`, `sm89`, and `sm120` coexist in
the cache and are ignored by Git.
