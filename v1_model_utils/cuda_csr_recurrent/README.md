# CUDA synaptic currents

This module is the default recurrent synaptic-current backend for `V1Column`.
It stores delayed connectivity as presynaptic CSR metadata, retains a permutation
to the trainable weights' original edge order, and fuses current and gradient
calculation in TensorFlow CUDA custom operations.

## CSR-ordered weights

`DIRECT_CSR` in `build.py` compiles the kernels to index weights by CSR
position instead of gathering through `edge_ids`. The caller must then supply
weights in CSR edge order: build the model from a network passed through
`spatial_layout.apply_csr_edge_order`, which reorders every edge-aligned array
at once, so weights, sign masks and the per-edge reference values inside the
weight regularizers all agree. `build_csr_connectivity(..., weights_csr_ordered=True)`
asserts that its own derived permutation is the identity, and
`require_csr_ordered_weights` refuses to run an undeclared caller rather than
silently pairing CSR positions with original-order weights.

Because nothing then dereferences `edge_ids`, `build_csr_connectivity` sends an
empty tensor for it instead of one dead `uint32` per edge — 321 MiB on the
recurrent connectivity of the 203,816-neuron network, 365 MiB on LGN. The
permutation stays on the host as `edge_order`, which is what `to_csr_order`,
`to_original_order` and the checkpoint translation use. The external operator
asserts the emptiness, so a caller that still uploads the permutation while the
kernels ignore it fails loudly instead of wasting device memory. Distributed
worker mode keeps the real tensor: its resource operator validates the length
against `post_ids`.

The `pair_*` projection metadata is only read by the backward kernels, so
`cuda_csr_external.build_csr_connectivity(..., needs_backward=False)` skips it
for a connectivity that is never differentiated — LGN input under
`--notrain_input` with no activity gradient.

Checkpoints are written in the network's original edge and neuron order, so
existing checkpoints stay loadable and per-edge tools keep working;
`V1Column.translate_checkpointed_layout` moves weights and the optimizer slots
that mirror them across that boundary.

## Compact pair-projected backward

At batch 32 with the four-column basis, the backward pass projects each distinct
`(postsynaptic neuron, synapse type)` pair onto the basis once rather than once
per edge (1,675,548 pairs for 84,132,910 edges in the 203,816-neuron network),
then runs a packed batch-lane row kernel over the result. Everything stays FP32:
the projection, the spike-gradient accumulation, and the butterfly that reduces
the weight gradient. `pair_projection_applies` gates the compact path strictly on
batch 32 and four basis columns; FP32 and every other batch size or basis
dimension retain the established batch-lane/general kernels. The build enables
the packed specialization for SM120 and newer; older architectures retain the
batch-lane path until independently benchmarked. It uses no tensor cores, so
that gate is a qualification boundary rather than a hardware requirement.

Three things make it fast, and none of them changes what the operation computes:

- **Packed batch lanes.** A lane holds four consecutive batch samples instead of
  one, so an edge needs eight lanes rather than 32 and one 16-byte load serves
  four edge slots at once. A 32-edge tile issues eight projection loads instead
  of 32. The bytes moved are unchanged.
- **Butterfly weight-gradient reduction.** The cross-lane sum costs about one
  shuffle per edge and no shared memory, replacing a shared-memory round trip
  through a tensor-core fragment whose `mma_sync` was fifteen-sixteenths wasted
  (every A row held the same spike vector). Shared memory per block falls from
  6,400 B to 256 B, so an SM runs its full 24 blocks instead of 16.
- **Tiled projection.** Writing the pair-major layout directly puts consecutive
  threads on current gradients 1.6 MiB apart, so a warp fetches 1 KiB to use
  256 B. Reading pair-contiguous and transposing through shared memory
  coalesces both halves.

The redundant 321 MiB `weight_grad` clear is also gone: both backward kernels
write every CSR position exactly once.

`BackwardPackedRowKernel` compiles to 44 registers and 256 bytes of shared
memory with no stack frame and no spills; `PreprojectPairsTiledKernel` to 35
registers and 4,352 bytes. Both leave an SM free to hold its full block
complement.

Measured on the 203,816-neuron network against the previous tensor-row
production kernel, on an otherwise idle SM120 GPU:

| | Previous | Current | Change |
|---|---:|---:|---:|
| Row kernel | 3.863 ms | 2.046 ms | -47.0% |
| Compact projection | 0.355 ms | 0.220 ms | -38.0% |
| Weight-gradient clear | 0.187 ms | removed | -100% |
| Total device time | 4.416 ms | 2.277 ms | **-48.4%** |
| Real training update | 5.811 s | 4.464 s | **-23.2%** |

It is also *more accurate* than the kernel it replaces. Against the FP32
batch-lane path, which never rounds anything to half, the previous tensor-row
kernel's weight gradient carried 5.4e-6 mean absolute error because it converted
the projected value to half for the tensor-core dot product; the packed kernel's
carries 3.5e-12, and its spike gradient matches the previous kernel's accuracy.
An FP16 projection would be a further 13 points faster but makes the spike
gradient about 865 times less accurate, so it was not taken. See
`wmma_recurrent_analysis_20260901/REPORT.md` for the full comparison, the
variants that were screened and rejected, and what remains unqualified.

## Neuron layout

`--neuron_layout morton` (the default) renumbers neurons along a space-filling
curve, cutting the distinct 128-byte sectors a CSR row's warp requests by about
23%. What that is worth depends on whether the postsynaptic currents array
(`batch * n_neurons * n_syn_basis` fp16) still fits in L2:

| | Batch 32 (49.8 MiB, fits 96 MiB L2) | Batch 64 (99.5 MiB, does not) |
|---|---:|---:|
| Recurrent forward | -15% | -20% |
| LGN forward | -21% | -27% |
| Training step, end to end | ~0.3%, inside noise | **-15.6%** |

Background input gains nothing either way: with about 100 active rows it is
launch-bound.

The two end-to-end figures differ by much more than the forward kernels do,
for two reasons. At batch 32 the layout *costs* the recurrent backward 4.4% -
the batch-lane kernel reads `projected[pair_ids[csr] * 32 + lane]`, and pair
ids inherit the `(post, type)` ordering, which Morton slightly widens - and
that almost exactly cancels the forward gain. At batch 64
`pair_projection_applies` gates that kernel off, so the penalty disappears and
the warp-per-row backward that replaces it benefits from the layout as well:
the forwards account for only about a sixth of the 15.6%.

The layout depends on CSR-ordered weights: with the `edge_ids` indirection still
in place it makes the weight gather about 5x more scattered and costs 16% end to
end. See `Benchmarks_metrics/morton_forward_layout_20260901/` and
`Benchmarks_metrics/batch64_layout_20260902/` for the measurements and the
calibration against a profiled training step.

## LGN row order

`--lgn_row_order retinotopic` ranks LGN rows by the mean postsynaptic index
they drive, so consecutive forward blocks scatter into nearby postsynaptic
memory. It is orthogonal to the neuron layout - one renumbers the presynaptic
side, the other the postsynaptic - and needs no kernel changes: the relabel is
applied before `apply_csr_edge_order`, so the CSR permutation carries it into
the forward, the weight backward and the checkpoint translation alike.

It **defaults to `original`**, because its measured value is small and depends
on the same L2 boundary:

| V1 layout | Batch 32 | Batch 64 |
|---|---:|---:|
| canonical | -2.0% of the LGN forward | -4.5% |
| morton | no measurable effect | -2.1% |

At batch 64 that is roughly 0.09% of a training step, and at batch 32 it is
nothing, against a per-timestep `[batch, n_inputs]` gather that pays for it
(`V1Column._permute_lgn_input`, needed because the spike stream stays in
canonical row order). A 672x improvement in the cross-row locality proxy buying
2% is the useful lesson here: for these kernels sector-count proxies track
performance and reuse-distance proxies do not.

It is implemented rather than skipped because the ordering also shapes the
projection gather that the packed backward kernel spends most of its time on,
which does not depend on the batch. See
`Benchmarks_metrics/batch64_layout_20260902/` and
`Benchmarks_metrics/lgn_row_order_20260902/`.

Build it in the project environment:

```bash
conda activate neuro_tf221
python -m v1_model_utils.cuda_csr_recurrent.build
```

The build uses the CUDA 12.9 `nvcc` and C++ compiler installed in the active
environment. Do not load a separate system CUDA toolkit before building.

The default architecture is detected from the visible GPU. Set `V1_CUDA_ARCH`
or pass `--architecture` to prebuild for another compute capability. Builds are
cached under their CUDA ABI and architecture directory (for example,
`sm86/cuda_csr_recurrent/_csr_recurrent_ops.so`), and runtime loading selects the exact match
for the visible GPUs.

The operator dispatches four basis columns to an unrolled specialization and
uses a runtime loop for every other positive basis dimension. Batch sizes
`1, 2, 4, 8, 16, 32, 64, 128, 256` have separate compiled forward and backward
kernels. Other positive batch sizes use the generic dispatch.
The generic backward path processes runtime batches in four-sample tiles and
skips zero weight-gradient writes, which keeps arbitrary batches efficient for
the model's sparse firing regime.

Training defaults to `--acceleration=auto`. Use `--acceleration=cuda` to
require the optimized kernels or `--acceleration=tensorflow` for the reference
implementation.

The CUDA kernels consume the FP32 recurrent master weights directly. Activations
and synaptic basis values still use the model compute dtype.
