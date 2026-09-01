# GPU-local CSR resources

This internal production module stores recurrent, LGN, and background CSR
metadata in an opaque resource owned by each distributed worker's local GPU.
It prevents TensorFlow from placing the full connectivity graph on one GPU or
copying it through distributed graph edges.

Resource mode is selected only for workers launched by `multi_training.py
--n_gpus N` with `N > 1`. Direct single-GPU training continues to use the
tensor-backed current operators. The resource library provides recurrent
forward/backward and the external weight-only backward operation; the complete
external activity-gradient path remains available for diagnostic callers.

Missing or stale libraries are built automatically under a per-architecture
lock. They may also be prebuilt explicitly:

```bash
python -m v1_model_utils.cuda_csr_resources.build --architecture 86
```

Architecture-keyed binaries for `sm80`, `sm86`, `sm89`, and `sm120` coexist in
this directory and are ignored by Git.
