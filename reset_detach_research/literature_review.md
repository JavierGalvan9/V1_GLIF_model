# Reset detachment and gradient stability in surrogate-gradient SNNs

## Executive conclusion

Detaching the spike from the voltage-reset branch is a common, defensible surrogate-gradient convention. The paper supplied for this review explicitly recommends it as a practical training heuristic, and its cited empirical basis shows that allowing surrogate gradients through reset can degrade learning when the surrogate derivative is poorly scaled. However, the evidence does **not** establish that reset detachment always improves learning, nor that adding a backward-only damping factor to the membrane self-loop is the standard remedy when detachment exposes large gradients.

For the V1 recurrent model, the most literature-aligned order of operations is:

1. detach only the reset occurrence in the voltage equation;
2. keep the forward neuron dynamics unchanged;
3. normalize or reduce the peak scale of the spike surrogate derivative;
4. monitor and clip the global gradient norm;
5. only then test backward-only voltage self-loop damping as a separate, explicitly biased gradient estimator.

The proposed voltage dampening is mathematically coherent and may be useful, but it shortens temporal credit assignment through the membrane on every non-reset step. It should therefore be tuned against the BPTT/recomputation horizon rather than assumed to be part of reset detachment.

## What the cited IEEE paper actually says

The linked paper is Eshraghian et al., *Training Spiking Neural Networks Using Lessons From Deep Learning*, Proceedings of the IEEE 111(9), 2023, DOI [10.1109/JPROC.2023.3308088](https://doi.org/10.1109/JPROC.2023.3308088); an open author version is available on [arXiv](https://arxiv.org/abs/2109.12894).

It writes a discrete soft-reset LIF update as

\[
U[t] = \beta U[t-1] + W X[t] - \theta S[t-1].
\]

In its practical-training guidance, the authors state that the surrogate derivative should not be copied into the reset branch; they describe ignoring that branch in the backward pass and note that snnTorch implements this by detaching the reset term. This is a tutorial/review recommendation, not a new controlled experiment in that paper. The cited empirical support is primarily Zenke and Vogels (2021), discussed below.

The IEEE article treats a spiking neuron itself as recurrent because its hidden state evolves over time. Thus, the recommendation is not restricted to purely static or single-timestep networks. However, much of the article's illustrative deep-learning discussion concerns layered SNNs, and reset detachment alone does not address the additional recurrent Jacobian introduced by learned recurrent synapses in an RSNN.

The review's wording is categorical, but it also notes two possible implementations: use the reset's original analytical derivative (zero almost everywhere), or detach the reset. Both prevent the surrogate derivative from entering that branch.

## Direct empirical evidence

### Zenke and Vogels (2021)

[Zenke and Vogels, *The Remarkable Robustness of Surrogate Gradient Learning for Instilling Complex Function in Spiking Neural Networks*](https://doi.org/10.1162/neco_a_01367) systematically varied surrogate shape and scale in feedforward and recurrent SNNs trained with BPTT. An open version is available from [bioRxiv](https://doi.org/10.1101/2020.06.29.176925), with accompanying [code](https://github.com/fzenke/randman).

Paper findings:

- Surrogate-gradient performance was relatively robust to surrogate **shape**, but sensitive to its **scale**.
- A surrogate whose peak grows with its sharpness performed poorly when gradients flowed through the spike reset. The degradation increased with network depth.
- With the reset detached, both their normalized and asymptotically scaled surrogate variants performed similarly in the tested condition.
- Explicit recurrent synapses created the same qualitative sensitivity: propagating a poorly scaled surrogate through learned recurrent connections could reduce performance to chance, even with one hidden layer.
- Their practical experiments subsequently used a normalized SuperSpike derivative with peak 1 and detached reset terms.

The important interpretation is not simply “detach reset.” It is that reset and recurrent synapses both create repeated surrogate-dependent paths, so the **magnitude of the surrogate derivative** controls the stability of the unrolled Jacobian.

There is also a model mismatch worth making explicit. Zenke and Vogels tested a multiplicative hard reset,

\[
U_{t+1}=(\beta U_t+(1-\beta)I_t)(1-S_t),
\]

whereas the V1 implementation uses subtractive reset. Their strongest adverse result was the combination of differentiable reset and an asymptotic/high-peak surrogate. Their standard normalized surrogate with differentiable reset did not exhibit the same catastrophic interaction. The paper therefore supports detachment as a cautious default, especially under recurrence, depth, or excessive surrogate scale; it does not prove that reset gradients are universally harmful.

### Yang (2020)

[Yang, *Temporal Surrogate Back-propagation for Spiking Neural Networks*](https://arxiv.org/abs/2011.09964) restored the temporal derivative contributed by reset in a multiplicative-reset LIF formulation. The reset-aware derivative improved robustness to learning-rate changes in a toy single-neuron task, but produced essentially unchanged results on N-MNIST, MNIST, and CIFAR-10. The author concluded that the benefit generally did not justify the extra cost.

This is useful counterevidence: dropping the reset derivative is an approximation, and including it can help in some conditions. The paper's larger benchmarks were convolutional/layered rather than a large biologically constrained RSNN, so it does not resolve the choice for this V1 model.

### Gygax and Zenke (2025)

[Gygax and Zenke, *Elucidating the Theoretical Underpinnings of Surrogate Gradient Learning in Spiking Neural Networks*](https://doi.org/10.1162/neco_a_01752) provides a more recent theoretical treatment; the [preprint is available on arXiv](https://arxiv.org/abs/2404.14964). They report that backpropagating through reset or detaching it gave similar performance when the surrogate derivative was normalized by its scale, whereas an unnormalized surrogate made reset backpropagation harmful. They also emphasize that surrogate gradients are generally not exact gradients of a surrogate loss.

This strengthens the conclusion that detachment and surrogate normalization are coupled design choices in the backward model.

## Why gradients can become larger after detaching reset

Consider the model's additive soft reset, suppressing current-state details:

\[
v_{t+1} = \alpha v_t + I_t - z_t, \qquad
z_t = H(v_t-\vartheta).
\]

Let \(g_t\) denote the chosen surrogate derivative for \(\partial z_t/\partial v_t\). With the reset differentiable, the direct temporal voltage Jacobian contains

\[
\frac{\partial v_{t+1}}{\partial v_t} = \alpha - g_t
\]

when the reset magnitude is one (more generally, \(\alpha-\theta g_t\)). With reset detachment it becomes

\[
\frac{\partial v_{t+1}}{\partial v_t} = \alpha.
\]

Therefore, detachment does not necessarily reduce gradient magnitude. Near threshold, the full-reset term can partially cancel the positive leak path. Removing it eliminates that cancellation and leaves a persistent self-loop. In a learned RSNN the full state Jacobian also includes recurrent spike feedback, approximately

\[
J_t \approx \alpha I + W_{\mathrm{rec}}D_t + J_{\mathrm{syn},t} + J_{\mathrm{ASC},t},
\]

where \(D_t=\mathrm{diag}(g_t)\). Products of these time-varying Jacobians determine whether gradients grow or decay. Even if \(\alpha<1\), recurrent weights, synaptic states, adaptation currents, and converging loss paths can make the product or accumulated gradient large.

This explanation is an inference from the model equations and standard BPTT, consistent with Zenke and Vogels' recurrence experiments. It is not a claim that their experiments used this V1 GLIF architecture.

## Assessment of backward-only voltage dampening

The contemplated update is effectively

```python
dampened_v = straight_through_dampen(v, d)
new_v = decay * dampened_v + current_factor * c1 - stop_gradient(prev_z)
```

If `straight_through_dampen` preserves `v` in the forward pass and multiplies only its backward derivative by a factor \(q\), then the direct self-loop becomes \(q\alpha\). This gives a simple bound on that one gradient path and leaves input/current gradients untouched.

Advantages:

- It preserves the exact forward dynamics and firing behavior.
- It directly weakens the long product of membrane self-loop Jacobians.
- It is more targeted than scaling the whole loss or every gradient.

Costs and risks:

- It is an additional biased backward rule, separate from reset detachment.
- It suppresses legitimate temporal credit on every step, not just at spikes.
- Its effective attenuation over \(K\) steps is roughly \(q^K\) on the pure voltage path. For example, \(q=0.5\) is extremely aggressive over a long BPTT segment.
- It does not control amplification through learned recurrent synapses, PSC states, or ASC feedback.
- Comparing values of `d` without accounting for segment length can be misleading.

I did not find a primary SNN source in this review that recommends a constant backward-only membrane self-loop factor as the canonical companion to reset detachment. It is best framed as a model-specific hypothesis worth testing, not as the literature-standard implementation.

## Better-supported stabilization controls

### 1. Normalize the surrogate derivative

This is the most directly supported intervention. Zenke and Vogels found recurrence-sensitive failures when the surrogate peak grew with sharpness and recommended appropriately normalized derivatives. A useful control is to keep

\[
\max_v |g(v)| \leq 1
\]

and tune width separately from peak amplitude. For a fast-sigmoid form, use the normalized derivative

\[
g(v)=\frac{1}{(1+\beta |v-\vartheta|)^2},
\]

whose peak remains one as \(\beta\) changes, rather than multiplying the numerator by \(\beta\).

### 2. Global gradient-norm clipping

Clipping is a safety mechanism rather than a cure for an unstable Jacobian, but it prevents rare batches from producing destructive optimizer steps. A recent recurrent-SNN study, [Bouanane et al., *Enhancing temporal learning in recurrent spiking networks for neuromorphic applications*](https://doi.org/10.1088/2634-4386/add293), used a peak-one surrogate plus global gradient-norm clipping specifically to mitigate explosion during BPTT. Its numeric clipping threshold is task- and normalization-dependent and should not be copied directly.

Log both the pre-clip global norm and the fraction of steps clipped. If nearly every step clips, the underlying scale or dynamics still need correction.

### 3. Recurrent-weight and activity initialization

[Rossbroich, Gygax, and Zenke, *Fluctuation-driven initialization for spiking neural network training*](https://doi.org/10.1088/2634-4386/ac97bb) derives data-dependent initialization intended to place LIF networks in a fluctuation-driven firing regime. The authors report benefits across fully connected, convolutional, recurrent, and Dale-constrained SNNs; [paper and code links](https://zenkelab.org/2022/06/fluctuation-driven-initialization-for-spiking-neural-network-training/) are public. This targets healthy membrane distributions and active surrogate support, rather than directly damping all temporal voltage gradients.

For this fixed-connectivity biological model, wholesale reinitialization may be inappropriate. The transferable lesson is to inspect membrane-potential distributions, firing rates, and recurrent effective gain at initialization before attributing large gradients solely to reset detachment.

### 4. Shorter truncation/recomputation horizons

Truncated BPTT bounds the number of Jacobian factors multiplied together and is a standard recurrent-training control. It also changes the learned temporal credit assignment, so it should be varied independently from reset treatment. This is especially relevant because the V1 model already uses segmented recomputation.

### 5. Learning rate and optimizer controls

Lower learning rate, warm-up, and adaptive optimization can keep high but finite gradients from creating unstable steps. They do not diagnose the source. They should be evaluated after recording raw gradient norms so optimizer adaptation does not hide an unstable backward model.

## Recurrent versus layered SNNs

A time-unrolled feedforward SNN is already recurrent at the neuron-state level: voltage, PSC, reset, and adaptation states connect adjacent timesteps. A learned RSNN adds explicit recurrence through \(W_{\mathrm{rec}}z_t\). Consequently:

- Results on layered SNNs are relevant to the implicit reset/voltage loop.
- They understate the number of feedback paths in this model.
- Zenke and Vogels explicitly tested learned recurrence and found surrogate-scale sensitivity could be even more severe there.
- The V1 GLIF model has additional slow state variables (PSC and ASC), so a single scalar voltage damping coefficient cannot characterize the stability of its complete state Jacobian.

This is why the experiment should measure component-wise gradients and not only total loss.

## Recommended experiment for this model

### Current-configuration observations

A read-only inspection of the current training path found three details that should be
verified against the exact command used for each experiment:

- `parallel_training_testing.py` currently defaults `dampening_factor` to `1` (a nearby
  comment says `0.1`) and forwards it to `multi_training.py`, overriding that script's
  lower default. Thus, runs launched through the wrapper may use a peak-one triangular
  surrogate rather than the intended lower-amplitude surrogate.
- The active shell command sets `recurrent_dampening_factor` to `0.1`, so explicit
  recurrent-current gradients are already attenuated separately.
- The optimizer path records gradient diagnostics when requested but does not currently
  apply global gradient-norm clipping.

These observations make the effective surrogate amplitude and raw global-norm
distribution the first quantities to confirm. They do not by themselves establish the
cause of the reported large gradients.

The current `voltage_gradient_dampening=0.5` convention gives a pure voltage-path
backward multiplier of `decay * (1 - 0.5)`. For the inspected network, membrane decay
values were approximately 0.83--0.98 (median 0.93), so this reduces that local
multiplier to approximately 0.42--0.49. Although the forward membrane time constants
are unchanged, this is a strong contraction of backward temporal memory and should not
be treated as mild stabilization.

Use identical initial weights, input batches, BPTT segment length, optimizer state, and random seed for all conditions. Run a small multi-seed pilot before long training.

### Minimum factorial comparison

1. **Full reset gradient**, normalized surrogate, no voltage damping.
2. **Detached voltage reset**, normalized surrogate, no voltage damping.
3. **Detached voltage reset**, normalized surrogate, mild backward voltage damping.
4. **Detached voltage reset**, reduced surrogate peak, no voltage damping.

If computationally affordable, separate whether ASC spike injection is detached. Do not call that “reset detachment”: it removes a distinct spike-to-adaptation credit path.

### Values to test

- Keep surrogate peak at 1 for the primary comparison, then test one lower peak such as 0.5.
- Parameterize the voltage backward multiplier directly as \(q\), avoiding ambiguity about whether “dampening = 0.5” means multiply by 0.5 or by \(1-0.5\).
- Start with mild \(q\) values tied to segment length, for example 1.0, 0.99, and 0.95. Include 0.5 only as an intentionally strong ablation.
- Use the existing global clipping threshold if one is already established; otherwise choose it from a short baseline distribution of unclipped gradient norms rather than importing a number from another paper.

### Log at every training step initially

- total pre-clip and post-clip gradient norm;
- fraction of clipped steps;
- gradient norms for recurrent weights, input weights, membrane-related parameters, ASC parameters, and readout weights;
- loss components, firing rates by cell type, membrane mean/variance and near-threshold fraction;
- NaN/Inf occurrence;
- wall-clock time and peak memory.

### Decision rule

Prefer reset detachment without voltage damping if normalized surrogate scaling and clipping make it stable. Add the smallest voltage damping that reliably controls the problematic norm only if instability persists. Judge convergence both by optimizer steps and wall-clock time, and reject settings that reduce training loss faster by collapsing firing rates or materially changing biologically relevant activity statistics.

## Bottom line

The literature supports trying reset detachment in this RSNN. It also explains the user's observation that detachment can increase gradients: removing the reset derivative can remove cancellation, while explicit recurrence remains. The best-supported first correction is surrogate-derivative normalization/scale control plus gradient-norm monitoring and clipping. Backward-only membrane damping is a reasonable experimental safeguard, but it should be mild, independently ablated, and interpreted as deliberately shortening voltage-path credit assignment.

## Primary sources

- Eshraghian, J. K., et al. (2023). [Training Spiking Neural Networks Using Lessons From Deep Learning](https://doi.org/10.1109/JPROC.2023.3308088). *Proceedings of the IEEE*, 111(9), 1016–1054. [Open author version](https://arxiv.org/abs/2109.12894).
- Zenke, F., & Vogels, T. P. (2021). [The Remarkable Robustness of Surrogate Gradient Learning for Instilling Complex Function in Spiking Neural Networks](https://doi.org/10.1162/neco_a_01367). *Neural Computation*, 33(4), 899–925. [Open preprint](https://doi.org/10.1101/2020.06.29.176925).
- Yang, Y. (2020). [Temporal Surrogate Back-propagation for Spiking Neural Networks](https://arxiv.org/abs/2011.09964).
- Gygax, J., & Zenke, F. (2025). [Elucidating the Theoretical Underpinnings of Surrogate Gradient Learning in Spiking Neural Networks](https://doi.org/10.1162/neco_a_01752). *Neural Computation*. [Preprint](https://arxiv.org/abs/2404.14964).
- Rossbroich, J., Gygax, J., & Zenke, F. (2022). [Fluctuation-driven initialization for spiking neural network training](https://doi.org/10.1088/2634-4386/ac97bb). *Neuromorphic Computing and Engineering*, 2, 044016.
- Bouanane, K., et al. (2025). [Enhancing temporal learning in recurrent spiking networks for neuromorphic applications](https://doi.org/10.1088/2634-4386/add293). *Neuromorphic Computing and Engineering*.
