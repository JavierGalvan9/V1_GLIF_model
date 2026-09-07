# LGN/background CUDA current operator

This package exposes the production external-current interface used by both
LGN and background connections. It shares the mature CSR forward and full
activity/weight backward kernels from `cuda_csr_recurrent`, and adds a
dedicated weight-only backward operator. The latter does not allocate or
calculate activity gradients and is the intended model configuration.

The basis dimension follows the recurrent implementation: four basis values
select the compile-time specialization, while every other positive dimension
uses a dynamic-basis kernel. Backward has static batch variants for 1, 2, 4,
8, 16, 32, 64, 128 and 256. Other batch sizes use a four-sample tiled runtime
fallback. FP32 master-weight gradients retain the original checkpoint edge
order; CSR identifiers are `uint32` and synapse types are `uint8`.

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

## Compact pair-projected gradients

At batch 32 with the four-column basis, both gradients project each distinct
`(postsynaptic neuron, synapse type)` pair onto the basis once rather than once
per edge, then run a packed batch-lane row kernel over the result. A lane holds
four consecutive FP32 batch samples, so an edge needs eight lanes, a warp covers
four edge slots, and one 16-byte load serves all of them. Everything stays FP32.

`blockIdx.y` splits a CSR row across blocks, which the BKG population needs: it
has 100 non-empty rows against a resident capacity of about 4,512 blocks, so one
row per block filled 2% of the GPU. LGN's 17,400 rows already exceed that
capacity and get a single split. Split blocks cannot each own the activity
output, so they accumulate into a float scratch that `CastActivityGradKernel`
narrows; with one split the scratch is skipped.

The weight-gradient clear is skipped for this path only. `UsesCompactWeightPath`
gates that on the same predicate `LaunchWeightBackward` selects on, because every
other weight-gradient path splits the batch across tiles and accumulates with
`atomicAdd`, which does need a zeroed buffer.

Measured against the previous tensor-row kernel on the real connectivity:

| Kernel | LGN before | LGN after | BKG before | BKG after |
|---|---:|---:|---:|---:|
| Activity gradient | 12,693 us | 4,133 us | 239.1 us | 24.1 us |
| Weight gradient | 6,162 us | 3,675 us | 356.5 us | 20.5 us |
| Compact projection | 163.3 us | 38.7 us | 36.5 us | 20.7 us |

It is also more accurate. Against a float64 host reference the previous weight
gradient carried up to 16% (LGN) and 65% (BKG) relative error on significant
entries, because it converted the projected value to half for its tensor-core
dot product; the packed kernel stays under 0.02%. The activity gradient's
accuracy is unchanged -- it was already FP32.

See `external_grad_analysis_20260905/REPORT.md` for the qualification.
