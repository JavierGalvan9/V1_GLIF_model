# Differentiable CUDA state transition

This module is the production adapter for GLIF/PSC/ASC state evolution during
training and ordinary Keras simulation. One op, `FusedGlifStep`, performs the
dense state update and the threshold and refractory mask, so the float32
membrane is written once and never read back by a separate op. The op reads
and writes no spike-history slot: `wrapper.py` shifts the history by
relabelling tensors. When the cell returns voltage sequences, the same op also
writes the compute-dtype copy of the membrane they expose. With the online
voltage loss it adds each step's neuron-mean voltage penalty to the carried
accumulator, and when the background input has its fixed four-edge layout it
also gathers the background current itself (`V1Column._fused_bkg_inputs`). The
`V1Column` output and state interface is unchanged.

Precision is split by state block, whatever the Keras policy: the membrane
and after-spike currents and every propagator constant are float32, while the
synaptic state (`psc_rise`, `psc`), its input currents and the spike buffers
follow the compute dtype. The four per-(neuron, basis) constants are packed as
`[syn_decay, psc_initial, psc_factor, psc_rise_factor]` and read with one
`float4` load.

The CUDA backward kernel supports the `triangular`, `gaussian`, and `slayer`
surrogate gradients selected through `surrogate_gradient`. The legacy
`pseudo_gauss=True` option remains an alias for `surrogate_gradient="gaussian"`.

The fused operators accept any positive synaptic basis dimension. Basis
dimensions 1 to 8 select the tiled kernels, compiled for each width: the
forward tiles two samples per neuron so they reuse its coefficient blocks, the
backward reads dense or bit-packed masks (`FusedGlifStepBackwardPacked`) and
can fuse the inputs gradient's abs-max bound, and both move the synaptic rows in
aligned accesses of up to 16 bytes where the widths allow. Wider rows use the
runtime-basis kernels (packed masks and the bound included). Every path gives
the same results bit for bit; only the voltage penalty accumulator of the
runtime-basis forward, which adds atomically, depends on the order. Dispatch
does not depend on batch size. The tiled kernels use 512 threads per block; the
fused BKG gather needs four bases. `build.ARCHITECTURE_FLAGS` reshapes them per
compute capability through the `V1_GLIF_*` macros: sm_100 (B200) tiles four
samples per forward thread, keeps two forward blocks per SM, loads the forward
state up front and runs 128-thread backward blocks, with the same results bit
for bit. Other architectures compile the defaults.

Builds are architecture-keyed and validated against the active TensorFlow/CUDA
environment:

```bash
conda run -n neuro_tf221 python -m v1_model_utils.cuda_glif_state.build
```

The build uses the CUDA 12.9 `nvcc` and C++ compiler installed in the Conda
environment. No system CUDA toolkit is required.
