# Interleaved GLIF constants experiment

The production exact-integration kernel still reads `syn_decay`, `psc_initial`,
`psc_factor`, and `psc_rise_factor` from separate `[N, R]` arrays. This
experiment interleaves them as four values per `(neuron, basis)` and reads one
aligned `float4` for float32 or one 64-bit word for float16. Inspection of the
RTX PRO 6000 machine code confirmed `LDG.E.128` and `LDG.E.64` loads.

`packed_glif_state_ops.cu.cc` is an isolated candidate based on the current
production kernel. The benchmark compiles both kernels with distinct TensorFlow
op names. To keep TensorFlow dispatch comparable, the candidate retains the
existing op signature, passes the packed array through `syn_decay`, and leaves
the other three coefficient inputs unused. Packing is done once before timing;
the production integration path has not been changed.

## Reproduce

Use one RTX PRO 6000 GPU. The UUID below was the tested device; update it if
running on another host.

```bash
CUDA_VISIBLE_DEVICES=GPU-6100f4b3-d3a6-9529-0b54-d183a6fb693b \
  conda run -n neuro_tf221 python3 benchmark_packed_constants.py
CUDA_VISIBLE_DEVICES=GPU-6100f4b3-d3a6-9529-0b54-d183a6fb693b \
  conda run -n neuro_tf221 python3 benchmark_packed_constants.py --packed-first
```

The script checks all forward and backward outputs before timing each case.
It warms each function, then reports the median of five trials of eight calls;
each call chains 32 state transitions in a `tf.function`. Lower ratios are
faster. Compilation artifacts are cached under `/tmp/glif_packed_cache` by
source hash and GPU architecture.

## Results

RTX PRO 6000 Blackwell Max-Q, sm_120, TensorFlow 2.21.0; 51,978 neurons,
four synaptic bases. Each cell shows packed / baseline for the first run and a
second run with the timing order reversed. Full timings and source hashes are
in `packed_constants_results.json` and
`packed_constants_results_reverse.json`.

| Batch | Dtype | Forward ratio | Backward ratio | Forward + backward ratio |
|---:|---|---:|---:|---:|
| 8 | float16 | 1.005 / 1.010 | 0.971 / 0.975 | 0.986 / 0.990 |
| 8 | float32 | 0.938 / 0.941 | 0.983 / 0.991 | 0.962 / 0.968 |
| 32 | float16 | 1.003 / 1.000 | 0.969 / 0.969 | 0.984 / 0.983 |
| 32 | float32 | 0.969 / 0.978 | 0.985 / 0.980 | 0.978 / 0.979 |

Float16 outputs matched exactly in all measured cases. The largest absolute
float32 difference was `1.19e-7`. Small-case checks also covered hard and soft
reset plus two gradient detachment configurations.

Packing modestly improves the combined kernel time, especially in float32.
Float16 forward time is unchanged to within about 1%. These measurements omit
the one-time packing operation and retain three unused inputs in the candidate
op. They measure only this state kernel, not a training step.
