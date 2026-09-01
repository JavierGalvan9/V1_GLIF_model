# CUDA synaptic currents

This module is the default recurrent synaptic-current backend for `V1Column`.
It stores delayed connectivity as presynaptic CSR metadata, retains a permutation
to the trainable weights' original edge order, and fuses current and gradient
calculation in TensorFlow CUDA custom operations.

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
