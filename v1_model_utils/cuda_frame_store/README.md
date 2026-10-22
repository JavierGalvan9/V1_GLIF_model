# CUDA host frame store

`RecordedAdjointRunner` (`--backward_mode recorded`) keeps the float32 membrane
voltage of every step for its reverse pass. With `--voltage_offload raw` or
`compressed`, this package parks that record in pinned host memory: the
forward hands each frame to `offload_frame`, and the reverse loop gets it back
with `take_frame`, which also prefetches the next frames. The copies run on the
store's own streams and are ordered against the compute stream with CUDA events
only, so the host never synchronizes.

## Modes

- `raw`: frames are copied exactly, so the gradients are bitwise those of the
  GPU-resident record.
- `compressed`: frames go out raw while the PCIe link keeps up. The frames the
  link could not take by the end of the forward are encoded with TQ-pack
  (`tq_codec.cu.h`). TQ-pack quantizes to the grid 2^-16 (|error| <= 2^-17,
  about 7.6e-6, on the stored voltages only; the live forward voltage stays
  exact) and predicts each frame from the two later ones (about 0.4 of the raw
  bytes). Encoded frames are decoded in reverse order as the reverse loop takes
  them. Blocks holding a non-finite value or |x| >= 127 are stored raw.

## Admission and the adaptive mix

- The latest `_PREFETCH` frames always stay on the GPU, because the reverse
  loop needs them first.
- Copies enter the copy stream, earliest frames first, only while fewer than
  two copy groups are unfinished. Frames still queued when the forward ends
  stay on the GPU, so the offloaded share adapts to the bandwidth the copy
  engine gets (for example, when several GPUs share one PCIe root port).
- With `compressed`, the store projects whether the remaining frames fit
  through the link from its measured link rate and forward frame rate. It
  encodes just the share of frames needed to fit, spread evenly. A frame the
  encoder's ring arena cannot take stays raw; the forward never waits.

## Memory

The host buffer holds the whole record: 52 MB per step at batch 64 and 203,816
neurons, so about 26 GB of pinned host RAM per GPU at `seq_len` 500. It is
allocated once per (GPU, layout) and reused by every later rollout.

## Environment variables

- `V1_FRAME_STORE_STATS=1`: print one statistics line per rollout to stderr
  (copied, fetched, encoded, link rate, ring use).
- `V1_FRAME_STORE_LINK_GBPS=<GB/s>`: simulate the link rate the adaptive codec
  assumes. `0` encodes every offloaded frame.
- `V1_FRAME_STORE_RING_FRAMES=<n>`: the TQ ring arena size in worst-case
  frames (default 6). This is a test-only knob that forces a full ring.

The op builds into the architecture-keyed CUDA cache like the other `cuda_*`
packages (`python -m v1_model_utils.cuda_frame_store.build`).
