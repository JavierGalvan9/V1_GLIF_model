# Differentiable CUDA state transition

This module is the production adapter for GLIF/PSC/ASC state evolution during
training and ordinary Keras simulation. It fuses the dense state update and the
spike/refractory/history update while preserving the
legacy `V1Column` output and state interface.

The CUDA backward kernel supports the `triangular`, `gaussian`, and `slayer`
surrogate gradients selected through `surrogate_gradient`. The legacy
`pseudo_gauss=True` option remains an alias for `surrogate_gradient="gaussian"`.

Builds are architecture-keyed and validated against the active TensorFlow/CUDA
environment:

```bash
conda run -n neuro_tf221 python -m v1_model_utils.cuda_glif_state.build
```

The build uses the CUDA 12.9 `nvcc` and C++ compiler installed in the Conda
environment. No system CUDA toolkit is required.
