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

## Forward

The forward is a scatter over the active `(batch, presynaptic row)` slots.
`BuildActiveQueue` finds them on the device with an ordered stream compaction
(`cub::DeviceSelect`), storing each flat `batch * n_pre + row` index, and keeps the count
on the device. The old host `tf.where` computed the same ordered list but copied
its count to the host to size the output, so every forward waited on that round
trip. The total `batch * n_pre` must fit a signed 32-bit index. The same queue
representation works at every firing rate and batch size within that limit.

With four basis columns and no repeated targets in a row, the forward prefers
shared-memory target tiles. It checks the compiled kernel's static allocation
and the device's opt-in shared-memory allowance first; devices that cannot fit
the tile use the scatter kernel. Unexpected CUDA API errors are reported.

A fixed 4,512-block grid consumes the queue, each block taking the next slots
from an atomic ticket (about 1,024 edges of work per ticket: one LGN row, or
seven recurrent ones). Consumption therefore follows the queue's row-major
order, so concurrent blocks scatter into the same few samples' currents. That
order matters once the currents outgrow L2: with an unordered queue the
recurrent forward was 2-4x slower from batch 64 up, and a static grid stride
over the ordered queue, which lets blocks drift apart, was 1.5x slower at batch
128 and 3.2x at 512 while gaining at most 0.02 ms at batch 32 or below.

For a connectivity whose rows repeat a target (LGN: 1.42 edges per target on
average, one per synapse type), `ForwardKernel` sums the edges that land on the
same postsynaptic neuron within a warp, in FP32 registers, before a single lane
issues the atomic (a `half2` one for FP16 currents and an even basis width).
That removes atomics and roundings, and relies on posts ascending within a CSR
row, which `build_csr_connectivity` guarantees. Recurrent rows never repeat a
target, so they take a plain one-thread-per-edge scatter with no warp
collectives. `csr_order.repeats_targets` decides this once when the
connectivity is built, and the wrappers pass it as the `aggregate_runs`
attribute. Four basis columns are unrolled; any other width loops.

The four-basis recurrent path uses `TiledForwardKernel` with a connectivity
table built on first use. Operators share this immutable table when the device,
CSR buffers, buffer lengths, postsynaptic count, and tile size match. This keeps
the separate gray and evoked validation graphs from retaining one table per
unrolled chunk. A weak registry releases the table with its last operator;
each operator keeps a strong reference for its warm path, avoiding registry
lookups or synchronization per timestep. The initial table fill completes before
publication so another CUDA stream can safely reuse it. CUDA kernel configuration
remains per operator because it depends on the kernel's dtype and library.

## Compact pair-projected backward

The backward projects each distinct `(postsynaptic neuron, synapse type)` pair
onto the basis once rather than once per edge (1,675,972 pairs for 84,145,692
edges in the 203,816-neuron network). One path serves every batch size, basis
dimension and spike dtype. The batch is processed in slices of
`min(64, next power of two)` samples (FP16; FP32 caps the slice at 32); the
projection is laid out
`[pair, batch_stride]` with the stride a whole number of slices, padding
samples project to zero, and the spike-gradient kernels run one grid row per
slice. The backward is split into two halves:

- **Spike gradient (SpMM).** With FP16 spikes the projection is stored as FP16,
  scaled by a power of two that `AbsMaxFiniteKernel` and `ProjectionScaleKernel`
  derive on the device from the largest finite `|current_grad|`. Without the
  scale these values sit near FP16's flush-to-zero floor. A power of two is
  exact in both directions, so the only error added is mantissa rounding, and
  the row kernel divides it back out after its FP32 accumulation.
  `PreprojectPairsKernel` gives each pair a thread that walks the batch, one
  vector load per sample and one 32-byte store per 16 samples. Sixteen-bit
  elements let a lane load eight samples in one 16-byte transaction, so at
  batch 64 each edge reads its pair's whole 128-byte line in a single pass and
  the edge descriptors are streamed once rather than once per 32-sample slice.
  FP32 spikes keep an FP32 projection (four samples per load), so a caller that
  chose FP32 keeps its precision. `BackwardRowTileKernel` gives each CSR row a
  warp; a block of eight warps owns 32 consecutive rows, takes them from a
  shared counter, and writes the tile out as `[batch, pre]` through shared
  memory, including the zeros of edgeless rows, so the output is never cleared
  first and no transpose pass is needed. Out-of-range lanes read an all-zero
  sentinel pair instead of branching. The 64-sample slice keeps two partial
  sums per lane so its sums are bitwise those of two 32-sample slices.
- **Weight gradient (event driven, `event_weight_grad.cuh`).** The spikes are
  sparse, so `EventRowQueueKernel` queues each row that fired, one item per
  256 edges so long rows spread over several blocks, with a 64-bit mask of the
  active samples when the batch is at most 64, and `EventWeightGradKernel`
  rebuilds the projection in FP32 from `current_grad` and the basis for the
  row's active samples only, in ascending order (compacted through shared
  memory above 64 samples). Each edge has one writer and a fixed summation order, so
  there are no atomics and the result is deterministic. Rows that never fired
  get an exact zero, so a non-finite upstream gradient no longer reaches the
  weight gradient of silent rows. Mixed-precision loss scaling still sees it
  through the spike gradient and the firing rows. The external operator uses the
  same kernels for the LGN weight gradient.

## In-place weight-gradient accumulation

Returned as a tensor, the recurrent weight gradient is a dense 336 MB value per
timestep that the RNN loop's gradient sums (`AddN`) across the whole sequence.
`V1CsrBackwardPairProjectedAccumulate` (and `V1CsrBackwardAccumulateResource`)
instead add it into an FP32 `[n_edges]` resource variable and return only the
spike gradient; the event kernel already adds per edge, so the only change is
that nothing clears the buffer first. A read of the variable that is still
alive makes the op copy on write.

`SegmentedRecomputeRunner` drives it when it is given
`V1Column.accumulated_weight_gradient_variable`: it zeroes its accumulator,
publishes the handle through `accumulate_recurrent_weight_gradient` for the
reverse pass, and reads the total once afterwards. The recurrent weight is
dropped from the recomputed chunks' tape targets, so no dense zero placeholder
or loop-carried gradient is built for it. The accumulator is a replica-local
(`ON_READ`) variable and the handle is published per thread, because
`MirroredStrategy` traces each replica in its own thread; each replica adds
only into the accumulator on its own GPU (the handle lookup refuses any other
device) and returns its own partial gradient, which the optimizer all-reduces
exactly as before.

Measured on the 203,816-neuron network at the production operating point, the
synaptic path per timestep went from 3.886 ms to 2.437 ms, and a real training
update from 3.557 s to 2.800 s. Against an FP64 reference the 25-step
accumulated weight gradient went from 2.09e-4 to 5.4e-8, LGN currents from
5.14e-3 to 1.45e-3, and recurrent currents from 3.33e-4 to 2.52e-4 (the FP16
store floor is 2.07e-4). See `kernel_opt_20260920/OPTIMIZATION_REPORT.md` for the
measurements, the ablations and the variants that were rejected.

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
uses a runtime loop for every other positive basis dimension. The forward does
not depend on the batch; the backward compiles one kernel set per slice width
(1, 2, 4, 8, 16, 32, 64) and covers any batch with them.
The raw operators accept at most eight equal-shape spike-slot tensors. The
public wrapper supports longer histories by concatenating them into one
generic operand. If carried queue records are supplied for a longer history,
it rebuilds the flattened queue and returns the newest slot's record; forward
and gradient semantics are preserved, but the older records are not reused.
Training defaults to `--acceleration=auto`. Use `--acceleration=cuda` to
require the optimized kernels or `--acceleration=tensorflow` for the reference
implementation.

The CUDA kernels consume the FP32 recurrent master weights directly, and take
the synaptic basis in FP32: `V1Column` builds it in FP32 for the CUDA backend,
because rounding those 360 constants to FP16 biases every contribution.
Activations and currents use the model compute dtype.
