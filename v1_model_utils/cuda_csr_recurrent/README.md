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

Checkpoints are written in the network's original edge and neuron order, so
existing checkpoints stay loadable and per-edge tools keep working;
`V1Column.translate_checkpointed_layout` moves weights and the optimizer slots
that mirror them across that boundary.

## Compact pair-projected backward

At batch 32 with the four-column basis, the backward pass projects each distinct
`(postsynaptic neuron, synapse type)` pair onto the basis once rather than once
per edge (1,675,548 pairs for 84,132,910 edges in the 203,816-neuron network),
and maps one batch sample per warp lane. `pair_projection_applies` gates this
strictly on that shape; every other batch size and basis dimension keeps the
general kernel. Measured 17.7 s -> 8.2 s per training update at batch 32.

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

It is implemented rather than skipped because a tensor-row WMMA kernel would
make the same ordering pay through fragment reuse instead of cache capacity,
which does not depend on the batch. See
`Benchmarks_metrics/batch64_layout_20260902/` and
`Benchmarks_metrics/lgn_row_order_20260902/`.

Build it in the project environment:

```bash
conda activate neuro_tf2151
python -m v1_model_utils.cuda_csr_recurrent.build
```

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
