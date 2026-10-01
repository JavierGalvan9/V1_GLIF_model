# Should the spike-triggered ASC update be detached?

## Decision

**No. Keep the spike-triggered after-spike-current update differentiable.**

For the current V1 GLIF implementation, the recommended combination is:

```text
membrane reset: detached
ASC spike increment: attached
surrogate amplitude: 0.1
voltage self-loop dampening: 0.0
```

This conclusion is better supported than the membrane-reset decision. It agrees with
the primary literature and produced substantially lower short-run loss in every matched
10k-network seed.

## The two pathways are different

The membrane update contains a discontinuous reset:

\[
v_{t+1} = \alpha v_t + I_t - z_t.
\]

The ASC update writes the spike into a slow biological state:

\[
a_{t+1} = \rho a_t + z_t A.
\]

Detaching the membrane reset removes the surrogate derivative of an instantaneous
state correction. Detaching the ASC increment removes credit assignment through a slow
adaptation pathway. These are not the same operation and should not share one flag.

## Literature evidence

Official LSNN and e-prop implementations preserve a differentiable local spike path
into adaptation even when they stop spike gradients through reset or recurrent paths.
Their analytical eligibility traces explicitly include adaptation feedback. This is a
mechanistic reason to keep the ASC increment attached: past spikes change a slow state,
and that slow state affects future voltage and firing.

I found no comparable primary evidence supporting ASC detachment as a general default.
The detailed source analysis is in
[asc_gradient_literature.md](asc_gradient_literature.md).

## Controlled 2×2 experiment

Four conditions were tested:

| Condition | Membrane reset | ASC increment |
| --- | --- | --- |
| A | detached | attached |
| B | attached | attached |
| C | detached | detached |
| D | attached | detached |

The experiment used three matched seeds, 10,000 neurons, FP16, sequence length 100,
100 optimizer updates, surrogate amplitude 0.1, recurrent-gradient dampening 0.1, no
voltage dampening, and no global clipping. Forward dynamics were identical across all
conditions.

### Final losses

| Membrane reset | ASC path | Seed 3000 | Seed 3001 | Seed 3002 | Mean |
| --- | --- | ---: | ---: | ---: | ---: |
| Detached | Attached | 7.4603 | 7.1892 | 7.7340 | 7.4612 |
| Attached | Attached | 7.4785 | 7.2139 | 7.7483 | 7.4802 |
| Detached | Detached | 7.6141 | 7.3386 | 7.8903 | 7.6143 |
| Attached | Detached | 7.5375 | 7.2792 | 7.8327 | 7.5498 |

Detaching ASC worsened final loss in every matched seed:

- with membrane reset detached: `+0.1538`, `+0.1494`, and `+0.1563`;
- with membrane reset attached: `+0.0590`, `+0.0652`, and `+0.0844`.

The mean ASC-detachment penalty was approximately:

- `+0.1531` when the membrane reset was detached;
- `+0.0695` when the membrane reset was attached.

This effect is larger and more consistent than the membrane-reset effect. The best
condition in every seed was detached membrane reset with attached ASC.

## Gradient behavior

For seed 3000, initial pre-clipping global norms were:

| Membrane reset | ASC path | Evoked | Spontaneous |
| --- | --- | ---: | ---: |
| Detached | Attached | 0.984 | 0.852 |
| Attached | Attached | 0.575 | 0.482 |
| Detached | Detached | 1.291 | 1.095 |
| Attached | Detached | 0.708 | 0.592 |

ASC detachment increased rather than decreased the global norm in both reset
conditions. The likely explanation is again cancellation: the attached ASC pathway can
contribute gradients with a sign that partially offsets other recurrent paths. Removing
the pathway does not simply remove gradient magnitude; it changes the complete
time-unrolled Jacobian.

All measured gradients remained finite. Nevertheless, ASC detachment simultaneously
increased the initial norm and worsened optimization, giving no practical reason to use
it in this regime.

## Scope and limitations

The experiment used the same simplified screening objective as the membrane-reset
study: firing-rate and voltage losses were active, while synchronization, OSI/DSI, and
recurrent regularization costs were disabled. It tested early training, not final
validation convergence.

Even with those limitations, the conclusion about ASC is reasonably clear because:

- the mechanism is supported by recurrent-SNN literature;
- all six paired comparisons favor attached ASC;
- the loss effect is substantially larger than the membrane-reset effect;
- ASC detachment did not improve gradient stability.

## Implementation status

The experiment introduced independent controls:

```text
--detach_reset / --nodetach_reset
--detach_asc_reset / --nodetach_asc_reset
```

The default remains `detach_reset=True` and `detach_asc_reset=False`. TensorFlow and
CUDA implement identical behavior. The state-transition and clipping suite passes all
17 tests, covering the independent paths, FP16/FP32, and soft/hard reset.

## Final recommendation

Proceed with:

```text
--detach_reset
--nodetach_asc_reset
--dampening_factor 0.1
--recurrent_dampening_factor 0.1
--voltage_gradient_dampening 0.0
```

The remaining uncertainty concerns membrane-reset detachment on the complete 203k
training objective. There is currently no comparable uncertainty about ASC: detaching
it would discard a literature-supported slow-state credit pathway and performed worse
in every local experiment.
