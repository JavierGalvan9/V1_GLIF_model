# Differentiable CUDA state transition

This module is the production adapter for GLIF/PSC/ASC state evolution during
training and ordinary Keras simulation. It fuses the dense state update and the
triangular-surrogate spike/refractory/history update while preserving the
legacy `V1Column` output and state interface.

`acceleration="auto"` selects CUDA on a visible compatible GPU for the
supported non-Gaussian surrogate and otherwise uses the TensorFlow reference
path. Explicit `acceleration="cuda"` rejects `pseudo_gauss=True` rather than
silently changing its gradient.

Builds are architecture-keyed and validated against the active TensorFlow/CUDA
environment:

```bash
conda run -n neuro_tf2151 python -m v1_model_utils.cuda_glif_state.build
```
