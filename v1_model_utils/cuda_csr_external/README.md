# LGN/background CUDA current operator

This package exposes the production external-current interface used by both
LGN and background connections. It shares the mature CSR forward and full
activity/weight backward kernels from `cuda_csr_recurrent`, and adds a
dedicated weight-only backward operator. The latter does not allocate or
calculate activity gradients and is the intended model configuration.

The basis dimension follows the recurrent implementation: four basis values
select the compile-time specialization, while every other positive dimension
uses a dynamic-basis kernel. Backward has static batch variants for 1, 2, 4,
8, 16, 32, 64, 128 and 256. Other batch sizes use runtime fallbacks or
generic pair-projected kernels. FP32 master-weight gradients retain the original checkpoint edge
order; CSR identifiers are `uint32` and synapse types are `uint8`. Every
operator takes the synaptic basis in FP32. The LGN forward goes through the
recurrent `V1CsrForward`, so it gets the device-built active-row queue and the
warp-aggregated scatter described in `cuda_csr_recurrent/README.md`.

Build both required libraries in the configured environment:

```bash
python -m v1_model_utils.cuda_csr_recurrent.build
python -m v1_model_utils.cuda_csr_external.build
```

Both commands detect the visible GPU architecture by default. Pass
`--architecture 86`, `--architecture 80`, `--architecture 89`, or
`--architecture 120` to prebuild caches for RTX 3090, A100, L40S, or RTX Pro
6000 GPUs respectively. Architecture-specific shared libraries coexist and are
selected automatically at runtime.

## CSR-ordered weights

This operator shares the recurrent forward kernel, so it follows the same
`DIRECT_CSR` contract: LGN and background weights must be in each population's
CSR edge order. Note that the external CSR sort key omits the synapse type,
unlike the recurrent one, so the two populations get different permutations.
`spatial_layout.apply_csr_edge_order` derives all three.

## Weight gradient

The connectivity's `sparse_activity` flag chooses between two kernels:

- **Sparse activity (LGN, the default): event driven.** The kernel pair in
  `cuda_csr_recurrent/event_weight_grad.cuh`, shared with the recurrent
  operator, visits only presynaptic rows with an active sample, one block per
  1,024 edges, and rebuilds each edge's projection in FP32 from the upstream
  gradient for just the active samples. It serves every batch size and basis
  dimension, writes each edge exactly once and is deterministic. At 20 Hz it is
  1.5-3.4x faster than the pair-projected kernel on LGN, and it does not read
  the per-edge pair projection, so a sparse connectivity builds that only when
  an activity gradient is requested (`needs_activity_backward`): 365 MiB of
  device metadata saved on the LGN input of the 203,816-neuron network.
- **Dense activity (BKG, `sparse_activity=False`): compact pair-projected.**
  The Poisson background is active in every row at every step, so an event path
  skips nothing and gathers the upstream gradient once per edge and active
  sample. Projecting each distinct `(postsynaptic neuron, synapse type)` pair
  once and running a packed row kernel over the result is cheaper there, and
  its projection metadata is small (about 4 MiB for BKG). `V1Column` builds the
  BKG input this way.

## Compact pair-projected gradients

With the four-column basis, the activity gradient and the dense weight gradient project each distinct
`(postsynaptic neuron, synapse type)` pair onto the basis once rather than once
per edge, then runs a packed batch-lane row kernel over the result. A lane holds
four consecutive FP32 batch samples, so an edge needs eight lanes, a warp covers
four edge slots, and one 16-byte load serves all of them. Everything stays FP32.

`blockIdx.y` splits a CSR row across blocks, which the BKG population needs: it
has 100 non-empty rows against a resident capacity of about 4,512 blocks, so one
row per block filled 2% of the GPU. LGN's 17,400 rows already exceed that
capacity and get a single split. Split blocks cannot each own the activity
output, so they accumulate into a float scratch that `CastActivityGradKernel`
narrows; with one split the scratch is skipped.

See `external_grad_analysis_20260905/REPORT.md` for its qualification.

For FP16 activity gradients with four basis columns, batches 3–31 that lack a
static specialization use the generic pair projection and 64-thread packed row
reduction when `RowSplitCount` returns one. This avoids the 256-thread direct
reduction on numerous short LGN rows. Inputs with fewer nonempty rows retain
the direct fallback, avoiding extra projection, split-row accumulation and
cast launches for small BKG batches. FP32 dispatch is unchanged.
