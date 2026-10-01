# Should the spike-triggered ASC/adaptation update be detached?

Date: 2026-08-31

## Bottom line

The available evidence supports **detaching the membrane reset while retaining the gradient through the spike-triggered adaptation/after-spike-current (ASC) increment**. These are not the same operation:

- the membrane reset is an instantaneous discontinuity used to restart voltage after a spike;
- the ASC/adaptation increment writes the spike into a slow dynamical state that is intended to carry information forward in time.

I found no primary source recommending that the spike contribution to an ASC or ALIF adaptation state be detached during ordinary surrogate-gradient BPTT. In contrast, primary RSNN implementations from Bellec and colleagues explicitly preserve this local adaptation path even when they stop other spike-mediated paths for e-prop. Therefore, ASC detachment should be treated as an experimental approximation or ablation, not as a consequence of reset detachment.

For the V1 GLIF model, the best initial configuration is:

1. detach `prev_z` in the membrane reset;
2. leave `prev_z * asc_amps` differentiable in the ASC update;
3. stabilize training with a properly scaled surrogate derivative, recurrent-gradient control, and global-norm clipping if needed;
4. compare ASC attachment/detachment only as a separate matched ablation.

## What the recurrent-SNN literature actually implements

### LSNN: the adaptation increment is differentiable

Bellec et al.'s LSNN is an explicitly recurrent spiking network trained with BPTT. Its adaptive threshold state obeys

\[
b_{t+1}=\rho b_t+(1-\rho)z_t,
\qquad B_t=b_0+\beta b_t.
\]

The paper presents neuronal adaptation as the mechanism that provides long, activity-silent memory, and applies surrogate derivatives to spikes during BPTT. It reports a dampened surrogate amplitude for stability over long unrolls; it does not prescribe detaching the adaptation increment. See [Bellec et al., NeurIPS 2018](https://papers.neurips.cc/paper/7359-long-short-term-memory-and-learning-to-learn-in-networks-of-spiking-neurons.pdf).

The authors' reference implementation is even more explicit. It computes

```python
new_b = decay_b * state.b + (1. - decay_b) * state.z
```

without `stop_gradient`, so BPTT differentiates through the spike-triggered write into adaptation. See the [official LSNN implementation](https://github.com/IGITUGraz/LSNN-official/blob/a9158a3540da92ae51c46a3b7abd4eae75a2bb86/lsnn/spiking_models.py#L425-L450).

This evidence is directly relevant to the layered-versus-recurrent concern: LSNN is an RSNN with explicit recurrent synaptic connections, not merely a feed-forward SNN unrolled over layers.

### e-prop: local adaptation remains attached even when other spike paths stop

The strongest evidence for distinguishing the two operations is the authors' e-prop implementation. To make automatic differentiation reproduce e-prop, it maintains two spike views:

- `z`, which can be stopped before recurrent transmission and membrane reset;
- `z_local`, which remains differentiable and updates adaptation.

The key update is:

```python
if use_stop_gradient:
    z = tf.stop_gradient(z)

new_b = decay_b * b + z_local
new_v = decay * v + recurrent_input(z) - reset(z)
```

The source comment states that the threshold update need not depend on stopped `z` because it is local. See the [official eligibility-propagation implementation](https://github.com/IGITUGraz/eligibility_propagation/blob/efd02e6879c01cda3fa9a7838e8e2fd08163c16e/Figure_3_and_S7_e_prop_tutorials/models.py#L291-L323).

Its analytical eligibility trace also contains the adaptation feedback term

\[
\epsilon^a_{t+1}=(\rho-\beta\psi_t)\epsilon^a_t+\psi_t\epsilon^v_t,
\]

where \(\psi_t\) is the surrogate spike derivative. This term only exists because spike generation and adaptation are differentiated locally. See [Bellec et al., Nature Communications 2020](https://www.nature.com/articles/s41467-020-17236-y) and the [reference eligibility-trace code](https://github.com/IGITUGraz/eligibility_propagation/blob/efd02e6879c01cda3fa9a7838e8e2fd08163c16e/Figure_3_and_S7_e_prop_tutorials/models.py#L326-L366).

This is an important conceptual result: even a learning rule deliberately truncating nonlocal recurrent paths preserves the spike-to-adaptation path as part of the neuron's local state Jacobian.

### Biological GLIF/AdEx equations do not answer the gradient question

Biophysical models define spike-triggered adaptation or after-spike currents as state jumps. For example, an adaptation variable is incremented at each spike and then decays. The [Neuronal Dynamics treatment](https://neuronaldynamics.epfl.ch/online/Ch6.S1.html) gives this forward model clearly. Teeter et al.'s GLIF family similarly uses post-spike state changes to reproduce diverse neuronal responses ([Nature Communications 2018](https://www.nature.com/articles/s41467-017-02717-4)).

These sources establish the forward dynamics, but they do not decide how a surrogate-gradient training algorithm should differentiate the discontinuity. That decision must come from learning literature and experiments. The closest primary RSNN learning evidence above keeps adaptation attached.

## Why membrane-reset detachment does not imply ASC detachment

Consider a simplified neuron with voltage \(v\), spike \(z=H(v-\theta)\), surrogate derivative \(h\), and an ASC/adaptation state \(a\):

\[
v_{t+1}=\alpha v_t+I_t-z_t,
\qquad
a_{t+1}=\rho a_t+A z_t.
\]

For the membrane reset, attaching the spike gives the direct local voltage derivative

\[
\frac{\partial v_{t+1}}{\partial v_t}\approx\alpha-h_t.
\]

Detaching the reset removes the artificial reset contribution and leaves approximately \(\alpha\). This is the reset-detachment choice investigated previously.

For the ASC update, attaching the spike instead provides

\[
\frac{\partial a_{t+1}}{\partial v_t}\approx A h_t.
\]

This derivative says: changing the pre-spike voltage can change whether an ASC is triggered, which then changes future voltage and spikes for the ASC decay timescale. It is precisely a temporal credit-assignment path through the slow state.

Detaching the ASC increment sets that cross-state derivative to zero. The gradient can still propagate along \(\partial a_{t+1}/\partial a_t=\rho\), but upstream parameters can no longer receive credit for causing the spike that wrote into the ASC. Thus ASC detachment is closer to truncating a recurrent connection than to suppressing a numerical reset artifact.

With multiple ASCs, the cross derivative is a vector \(\mathbf A h_t\). Positive and negative ASC amplitudes can create stabilizing or amplifying feedback depending on how each current enters later voltage dynamics. Consequently, the effect cannot be inferred from membrane-reset results alone.

## Expected stability trade-off

Keeping ASC attached adds another spike-mediated loop to the state Jacobian. In a long, explicitly recurrent simulation this may increase gradient variance or norm, particularly when:

- the surrogate derivative is too large;
- ASC amplitudes are large;
- ASC decay constants are close to one;
- recurrent weights already place the network near an unstable backward regime;
- several ASC components align constructively.

That is a legitimate reason to measure the path. It is not, by itself, evidence that it should be deleted. Bellec et al. address long-horizon RSNN stability by dampening the surrogate derivative (their examples commonly use `0.3`), while retaining adaptation eligibility. The adjustment targets every spike-mediated Jacobian contribution coherently instead of selectively erasing adaptation credit.

ASC detachment could still be useful as a pragmatic biased-gradient approximation if attached ASC gradients dominate and prevent optimization after surrogate scaling and clipping have been calibrated. If so, it should be described explicitly as truncated temporal credit assignment.

## Recommended experiment

Use identical forward passes, initial weights, stimulus batches, and seeds. The minimum informative factorial comparison is:

| Condition | Membrane reset | ASC increment |
|---|---|---|
| A | attached | attached |
| B | detached | attached |
| C | detached | detached |
| D | attached | detached |

Condition B is the literature-supported candidate. Conditions C and D identify whether any benefit comes specifically from suppressing ASC temporal credit rather than membrane reset.

For each condition, collect:

- full-objective training and held-out loss versus both update count and wall time;
- pre-clipping global gradient norm and the fraction of clipped updates;
- gradient norms grouped by recurrent weights, background input, and neuron parameters;
- nonfinite gradients and mixed-precision loss-scale skips;
- population and cell-type firing rates;
- voltage and ASC-state distributions;
- ASC-path-only cotangent contribution to `prev_z`, if instrumentable;
- final OSI/DSI and synchronization objectives, not only the simplified firing-rate loss.

First run a fast multi-seed 10k-network screen. Advance both B and the best competing condition to a matched short full-network pilot. A few early updates cannot establish convergence; use enough updates to separate transient gradient behavior from a persistent loss trend.

Also sweep surrogate amplitude before concluding that ASC attachment is unstable. Recommended starting values are `0.05`, `0.1`, `0.2`, and `0.3`, while keeping voltage self-loop dampening at zero. Calibrate global clipping from measured full-model norms rather than transferring the 10k threshold directly.

## Decision rule

Choose detached membrane reset with attached ASC unless one of the following is demonstrated across matched seeds:

1. attached ASC causes persistent nonfinite or extreme gradients after reasonable surrogate scaling and clipping; or
2. detached ASC gives reproducibly better held-out/full-objective convergence without damaging long-timescale task metrics or cell-type dynamics.

If ASC detachment only reduces gradient norms but does not improve convergence or validation behavior, retain the attached ASC path. Lower norms are not intrinsically better; they are valuable only when they produce more stable or effective optimization.

## Confidence and remaining gap

Confidence is **moderate** that ASC should remain attached. The primary RSNN/ALIF implementations consistently support that design, and the Jacobian argument is clear. Confidence is not high because I found no controlled published ablation specifically comparing attached versus detached spike-triggered ASC gradients in a large heterogeneous GLIF cortical model. The proposed project-specific factorial experiment is therefore necessary for a firm conclusion.
