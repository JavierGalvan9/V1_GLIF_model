# Should the V1 GLIF voltage reset be detached?

## Decision

**Use a detached membrane-reset gradient as the default for the next full-model pilot,
provided that the spike surrogate amplitude remains at `0.1`, the voltage self-loop is
not dampened, and raw global gradient norms are monitored.**

This is a conditional engineering decision, not evidence that reset detachment is
universally superior. The current evidence favors detachment for this implementation,
but a definitive scientific conclusion about final convergence still requires matched
full-model, multi-seed validation.

## Question tested

The two conditions had identical forward dynamics:

\[
v_{t+1} = \alpha v_t + I_t - z_t.
\]

They differed only in the backward treatment of the reset:

- **detached:** `-stop_gradient(prev_z)`;
- **attached:** `-prev_z`.

The ASC spike-injection path remained differentiable in both conditions. Refractory
state gradients remained detached. The voltage self-loop retained its full derivative
in both conditions.

## Literature conclusion

Primary sources do not establish a universal rule:

- Reset detachment is a robust convention when surrogate derivatives are poorly
  scaled or repeatedly amplified by recurrent paths.
- With a normalized or sufficiently small surrogate, attached and detached resets can
  perform similarly.
- Reset-aware and exact-gradient methods provide counterexamples where propagating
  reset information is viable.
- The literature does not support constant strong dampening of the passive membrane
  derivative as the standard companion to reset detachment.

See [literature_review.md](literature_review.md) and
[literature_addendum.md](literature_addendum.md) for sources and model distinctions.

## Controlled 10k-neuron experiment

### Configuration

- 10,000 neurons;
- FP16 training;
- sequence length 100;
- segmented exact BPTT, chunk size 25;
- surrogate amplitude `dampening_factor=0.1`;
- recurrent-gradient dampening `0.1`;
- voltage-gradient dampening `0.0`;
- no global clipping;
- identical forward model and matched seeds 3000, 3001, and 3002;
- 100 optimizer updates per condition;
- simplified pilot loss with synchronization, OSI, and recurrent regularization costs
  disabled.

The simplified loss makes this a stability and early-optimization screen. It is not a
substitute for the complete training objective.

### Loss results

Lower loss favors the detached condition.

| Seed | Detached final | Attached final | Detached minus attached | Mean curve difference | Final-20 mean difference |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 3000 | 7.4603 | 7.4785 | -0.0182 | -0.0062 | -0.0181 |
| 3001 | 7.1892 | 7.2139 | -0.0247 | -0.0027 | -0.0178 |
| 3002 | 7.7340 | 7.7483 | -0.0143 | +0.0081 | -0.0003 |

The detached condition had a lower endpoint in all three seeds. The mean endpoint
advantage was approximately 0.019 loss units. However:

- the effect is small relative to differences between seeds;
- one seed had a higher detached loss over the complete measured curve;
- seed 3002 showed almost no difference over the final 20 updates;
- 100 updates do not establish long-run convergence or validation performance.

The correct interpretation is **weak but consistent endpoint evidence favoring
detachment**, not a decisive convergence advantage.

### Gradient results

One matched seed was run with gradient diagnostics before any clipping:

| Branch | Detached global norm | Attached global norm | Detached/attached |
| --- | ---: | ---: | ---: |
| Evoked | 0.984 | 0.575 | 1.71× |
| Spontaneous | 0.852 | 0.482 | 1.77× |

Maximum absolute gradients also increased:

- evoked: 0.0266 attached to 0.1255 detached;
- spontaneous: 0.0282 attached to 0.0815 detached.

All measured gradients were finite. This directly confirms the user's observation:
detaching the reset makes gradients larger. It also supports the mathematical
explanation. The attached reset contributes a negative surrogate term to the temporal
Jacobian, partially cancelling the membrane/recurrent gradient. Detachment removes
that cancellation.

At surrogate amplitude `0.1`, the increased gradients were not explosive in the 10k
screen and coincided with slightly better early endpoint loss. Therefore, the larger
gradient is not by itself a reason to restore the reset derivative.

## Performance observations

No meaningful memory difference was detected between reset modes: paired TensorFlow
peak-memory differences were within approximately 1 MiB. Runtime measurements favored
detachment in these runs, but the magnitude varied substantially and the runs shared a
busy multi-GPU host. They are not sufficiently controlled to claim a speed advantage.

The reset choice should therefore be based on optimization behavior, not this pilot's
runtime numbers.

## Why the recommendation is conditional

The 10k pilot differs from production training in important ways:

- 100 versus 500 timesteps;
- 10k versus 203k neurons;
- batch size 2 versus the larger production batch;
- simplified versus complete loss;
- 100 updates versus many epochs;
- no validation measurement;
- only three seeds.

Longer sequences and larger recurrent graphs can amplify the 1.7× initial norm increase.
Conversely, the full loss may change both scale and useful credit paths. The result
cannot safely determine a clipping threshold for production.

## Recommended full-model decision experiment

Run a short paired pilot using the production network and complete loss, with identical
initial weights, batches, and seeds:

1. detached reset, surrogate amplitude 0.1;
2. attached reset, surrogate amplitude 0.1.

For the initial diagnostic steps, use:

```text
--dampening_factor 0.1
--recurrent_dampening_factor 0.1
--voltage_gradient_dampening 0.0
--global_clipnorm 0.0
--debug_gradients
```

Record:

- pre-clip global gradient norm;
- maximum per-variable/group gradient norms;
- nonfinite or skipped mixed-precision updates;
- total and component losses;
- firing rates and near-threshold voltage fractions;
- validation loss;
- wall-clock time and memory.

After measuring the norm distribution, add clipping only as a safety rail. Select a
threshold that clips rare upper-tail updates, not nearly every update. Then continue a
matched multi-seed run long enough to compare validation convergence.

## Decision rule for changing the default

Keep reset detachment if the full-model pilot shows:

- finite gradients without persistent clipping;
- equal or better validation convergence;
- no pathological firing-rate or voltage behavior.

Restore the attached reset if detachment causes persistent norm growth, frequent
nonfinite/loss-scale-skipped updates, or worse matched validation performance even after
reasonable surrogate scaling and rare-event clipping.

Do not use strong voltage self-loop dampening as the first response. It suppresses
temporal membrane credit on every timestep and changes a separate part of the gradient
estimator.

## Present conclusion

For the tested V1 GLIF implementation, the evidence currently favors **detaching only
the membrane reset**. It produces larger gradients, exactly as expected, but those
gradients remain finite with a `0.1` surrogate amplitude and yield a small, consistent
100-update endpoint advantage across three 10k-network seeds. The advantage is not yet
large or mature enough to claim faster final convergence. The full-model paired pilot
is the remaining experiment needed for a clear production conclusion.
