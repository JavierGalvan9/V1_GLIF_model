# LGN/background CUDA current operator

This package exposes the production external-current interface used by both
LGN and background connections. It shares the mature CSR forward and full
activity/weight backward kernels from `cuda_synaptic_currents`, and adds a
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
python -m v1_model_utils.cuda_synaptic_currents.build
python -m v1_model_utils.cuda_external_currents.build
```
