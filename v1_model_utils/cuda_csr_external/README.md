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
