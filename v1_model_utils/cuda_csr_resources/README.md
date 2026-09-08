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
operators, whose background-input forward carries a gather specialization this
library does not implement - worth about 3% of a step at batch 32.

The library provides recurrent forward/backward and the external weight-only
and activity backward operations. Two contracts are shared with the tensor
backend and must not drift:

- **Compile flags.** `csr_resource_ops.cu.cc` `#include`s the recurrent and
  external kernel sources verbatim, so any flag set for one library and not the
  other compiles two different kernels from one source file.
  `cuda_csr_config.architecture_kernel_flags` is the single definition of the
  architecture-dependent flags for all three build modules.
- **Kernel selection.** The Python wrapper decides whether the compact
  pair-projected backward applies (`pair_projection_applies`) and passes the
  answer as the `pair_projected` attribute, so both backends read one gate.

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
