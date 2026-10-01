# Membrane-reset detachment and surrogate-scale factorial

## Final decision

For the current FP16 V1 GLIF training path, use:

```text
--detach_reset
--nodetach_asc_reset
--dampening_factor 0.1
--recurrent_dampening_factor 0.1
--voltage_gradient_dampening 0.0
```

The direct alternative proposed for testing—attached membrane reset with surrogate
amplitude `1.0`—is numerically stable and reasonably competitive, but it converged more
slowly in every matched production-horizon 10k experiment.

The confidence in this recommendation is now **moderately high for the tested training
regime**. It is not yet a final claim about complete 203k-model validation performance.

## Why the factorial was necessary

Reset treatment and surrogate amplitude interact in the temporal Jacobian. For a
subtractive reset,

\[
v_{t+1}=\alpha v_t + I_t-z_t,
\]

the local voltage derivative is approximately:

- attached reset: \(\alpha-g_t\);
- detached reset: \(\alpha\).

Increasing surrogate amplitude increases multiple spike-mediated recurrent paths. An
attached reset also introduces the negative \(-g_t\) term, which can provide strong
cancellation. Therefore, “attached with 1.0” cannot be inferred from results for
“attached with 0.1”; it must be measured directly.

## Experiment 1: full 2×2 screen at sequence length 100

Conditions:

| Condition | Membrane reset | Surrogate amplitude |
| --- | --- | ---: |
| A | detached | 0.1 |
| B | attached | 0.1 |
| C | detached | 1.0 |
| D | attached | 1.0 |

All conditions kept ASC attached, voltage dampening off, recurrent dampening at 0.1,
and clipping disabled. Three matched seeds used 10,000 neurons, FP16, sequence length
100, and 100 updates.

### Endpoint results

| Condition | Seed 3000 | Seed 3001 | Seed 3002 | Mean |
| --- | ---: | ---: | ---: | ---: |
| Detached, 0.1 | 7.4603 | 7.1892 | 7.7340 | 7.4612 |
| Attached, 0.1 | 7.4785 | 7.2139 | 7.7483 | 7.4802 |
| Detached, 1.0 | 7.5441 | 7.2152 | 7.7567 | 7.5053 |
| Attached, 1.0 | 7.4706 | 7.1907 | 7.7490 | 7.4701 |

Detached `0.1` beat attached `1.0` in all three seeds, but only by 0.0015--0.0150 at
this short temporal horizon. Attached `1.0` was clearly viable and better than attached
`0.1` on two seeds.

Detached `1.0` looked merely worse from terminal losses, but raw diagnostics revealed
that this interpretation was unsafe.

## Experiment 2: raw-gradient diagnostics

At sequence length 100, initial FP16 global gradient norms were:

| Condition | Evoked | Spontaneous | Finite? |
| --- | ---: | ---: | --- |
| Detached, 0.1 | 0.984 | 0.852 | yes |
| Attached, 0.1 | 0.575 | 0.482 | yes |
| Attached, 1.0 | 1.051 | 0.874 | yes |
| Detached, 1.0 | NaN | not reached | no |

Detached `1.0` produced 2,084,519 nonfinite elements on the first evoked update. An
FP32 repeat showed finite but enormous values:

- global norm: approximately 16.6 million evoked and 7.1 million spontaneous;
- maximum absolute element: approximately 9.8 million evoked and 3.9 million
  spontaneous.

This proves that detached `1.0` has an exploding temporal Jacobian. The FP16 loss-scale
optimizer can skip invalid updates, so its apparently ordinary benchmark loss is not
evidence of successful training.

Attached reset stabilizes the `1.0` surrogate through cancellation. That stabilization
is real, not merely theoretical.

## Experiment 3: intermediate detached amplitudes

At sequence length 100, detached amplitudes `0.125`, `0.15`, and `0.175` remained
finite initially. Amplitudes `0.2`, `0.3`, and `0.5` were already nonfinite in FP16.

Initial evoked norms grew rapidly:

| Amplitude | Global norm | Maximum absolute element |
| ---: | ---: | ---: |
| 0.100 | 0.984 | 0.125 |
| 0.125 | 1.228 | 0.244 |
| 0.150 | 1.573 | 0.474 |
| 0.175 | 2.207 | 0.930 |

Over 100 sequence-100 updates, stronger stable amplitudes improved endpoint loss in
every seed:

| Detached amplitude | Mean endpoint | Mean final-20 loss |
| ---: | ---: | ---: |
| 0.100 | 7.4612 | 7.4871 |
| 0.125 | 7.4273 | 7.4597 |
| 0.150 | 7.3877 | 7.4274 |
| 0.175 | 7.3547 | 7.3976 |

This initially suggested that `0.1` was conservative rather than optimal. However,
surrogate stability depends on the temporal horizon.

## Experiment 4: production temporal horizon

The model is normally trained with sequence length 500. One raw-gradient update was
therefore tested at sequence length 500:

| Candidate | Evoked norm | Spontaneous norm | Finite? |
| --- | ---: | ---: | --- |
| Detached, 0.1 | 0.839 | 0.823 | yes |
| Attached, 1.0 | 0.845 | 0.755 | yes |
| Detached, 0.175 | NaN | not reached | no |

The apparent sequence-100 optimum `0.175` does not transfer to the production horizon.
It must be rejected. This also shows why surrogate amplitude cannot be tuned using only
short sequences.

## Experiment 5: final sequence-500 comparison

The two viable candidates were trained for 25 updates at sequence length 500, with
three matched 10k-network seeds.

### Loss results

| Candidate | Seed 3000 | Seed 3001 | Seed 3002 | Mean endpoint | Mean final-10 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Detached, 0.1 | 6.6771 | 6.3999 | 6.9056 | 6.6609 | 6.7181 |
| Attached, 1.0 | 6.7009 | 6.4207 | 6.9318 | 6.6845 | 6.7386 |

Attached `1.0` minus detached `0.1` endpoint differences were:

- seed 3000: `+0.0238`;
- seed 3001: `+0.0209`;
- seed 3002: `+0.0263`.

Detached `0.1` also had lower mean loss over the complete measured curve and over the
last ten updates in every seed. This is the most relevant local result because it uses
the production temporal horizon.

### Resource results

Mean steady-state step times were close:

- detached `0.1`: 1.379 seconds;
- attached `1.0`: 1.393 seconds.

Mean TensorFlow peak memory was:

- detached `0.1`: approximately 751 MiB;
- attached `1.0`: approximately 693 MiB.

Thus, detached `0.1` was about 1% faster in this run but consumed about 58 MiB more
TensorFlow peak memory. Timing on a shared host should not be treated as a robust speed
claim. The memory difference is consistent with the extra useful gradient paths that
remain active without reset cancellation.

## Interpretation

The experiments support three distinct conclusions:

1. **Detached reset requires a small surrogate amplitude.** Without the negative reset
   derivative, amplitudes above the stable boundary produce explosive recurrent
   gradients.
2. **Attached reset permits a larger surrogate amplitude.** The reset derivative
   provides enough cancellation for `1.0` to remain finite.
3. **Permitting a larger surrogate does not make attached `1.0` better here.** At the
   production sequence length, detached `0.1` had consistently better early loss.

The value `0.1` is not a universal normalized-surrogate constant. It is an empirically
safe amplitude for this particular recurrent architecture, precision, sequence length,
and recurrent dampening. It should be reconsidered if those change.

## Confidence and limitations

Confidence that detached `0.1` is preferable to attached `1.0` for the tested 10k FP16
screen is high: both were finite and one condition won every paired loss comparison.

Confidence that detached `0.1` will give better final 203k-model validation is only
moderate because:

- the screening loss excluded OSI/DSI, synchronization, and recurrent regularization;
- the small network and batch differ from production;
- the final sequence-500 comparison covered only 25 updates;
- no validation or biological activity metrics were compared;
- only three seeds were used.

## Recommended next production step

Use detached `0.1` as the primary configuration and attached `1.0` as the only remaining
serious control. Keep ASC attached in both. Run a short complete-objective, matched
full-network pilot while recording raw gradients and loss components.

Do not test detached amplitudes near `0.175` on the full sequence: they are already
nonfinite on the 10k sequence-500 model. Do not use detached `1.0`; it is demonstrably
explosive.
