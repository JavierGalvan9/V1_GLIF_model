# Reset-gradient detachment in recurrent SNNs: literature addendum

## Bottom line

The literature does **not** establish a universal rule that the reset must be detached. It supports a narrower conclusion:

- For ordinary surrogate-gradient BPTT, detaching reset is a robust engineering default when the surrogate derivative is not carefully normalized. The strongest controlled evidence shows that a large surrogate scale combined with reset recurrence or explicit synaptic recurrence can severely impair learning.
- When the surrogate derivative is properly scaled, recent controlled results report little or no performance difference between attached and detached reset.
- Other mathematically consistent training schemes explicitly retain reset effects and can outperform methods that omit them on temporally demanding tasks. Consequently, “attached” is not intrinsically wrong; its stability depends on the gradient formulation and scaling.
- There is no controlled published benchmark sufficiently close to this long-horizon, heterogeneous, explicitly recurrent GLIF cortical model to decide the question without a matched local ablation.

The defensible working hypothesis for this project is therefore: **detach the subtractive membrane-reset path as the initial candidate, remove the strong voltage-path dampening, normalize the surrogate scale, and compare it against the attached-reset control under matched gradient safeguards.** Final adoption should be determined by validation loss, gradient stability, firing statistics, and seed-to-seed robustness—not by training loss from one run.

## What the strongest controlled study actually tested

Zenke and Vogels systematically varied surrogate shape, scale, reset differentiation, network depth, and explicit recurrence. Their key result was not simply “detach reset”: normalized SuperSpike and an asymptotic, increasingly tall surrogate performed similarly when reset was detached, whereas the combination of the asymptotic surrogate and differentiable reset impaired performance, increasingly so with depth. With explicit recurrent synapses, the large-scale surrogate could reduce performance to chance even with one hidden layer. They concluded that an excessive surrogate scale is harmful in the presence of either implicit recurrence (reset) or explicit recurrence ([paper, especially Figures 4–5](https://research-explorer.ista.ac.at/download/8253/11131/2021_NeuralComputation_Zenke.pdf); [journal record](https://doi.org/10.1162/neco_a_01367)).

This is relevant to an RSNN: the paper separately tested explicitly recurrent connectivity, and its later benchmarks included recurrent networks on SHD (inputs lasting 0.6–1.4 s), Raw Heidelberg Digits, and Speech Commands. However, the decisive reset-detachment comparison was the controlled synthetic classification study, not a reset ablation across all of those speech benchmarks. The broader recurrent benchmarks used a normalized surrogate and detached reset. Thus, they demonstrate that the recommended configuration trains RSNNs, but do not prove that it is optimal for every RSNN.

Gygax and Zenke later supplied the most important qualification. In discrete-time LIF experiments, they found essentially no performance difference between backpropagating through reset and omitting that path when the surrogate derivative was normalized by its inverse width, while unnormalized derivatives made reset backpropagation harmful ([article, section 7.2 and supplementary Figure S2](https://direct.mit.edu/neco/article/37/5/886/128506/Elucidating-the-Theoretical-Underpinnings-of); [preprint](https://arxiv.org/abs/2404.14964)). This makes reset detachment and surrogate normalization interacting choices, not independent binary decisions.

## Why reset type matters

The gradient consequences differ between reset equations. Let `g_t` denote the surrogate derivative of the spike and `alpha` the membrane decay.

### Subtractive (soft) reset

For

```text
v[t+1] = alpha * v[t] + input[t] - theta * z[t]
```

the direct temporal Jacobian is approximately

```text
attached reset:  alpha - theta * g_t
detached reset:  alpha
```

Attaching reset can therefore either damp, reverse, or enlarge the local derivative depending on `g_t`. Detaching removes this surrogate-dependent term, but exposes the near-unit `alpha` path over long sequences. The user's observation that gradients grow after detachment is therefore plausible: the former `-theta*g_t` term may have provided cancellation, while explicit recurrent spike paths still inject surrogate-dependent feedback.

### Multiplicative (hard-to-baseline) reset

For a pre-reset voltage `h_t` and reset baseline `v_reset`,

```text
v[t+1] = (1 - z[t]) * h_t + z[t] * v_reset
```

the attached derivative with respect to `h_t` contains

```text
(1 - z[t]) + (v_reset - h_t) * g_t,
```

whereas detaching `z[t]` leaves only `(1-z[t])`. At a spike, the latter intentionally cuts the voltage-memory path. This is qualitatively different from subtractive reset, where detachment leaves `alpha` intact. Results for one reset convention should not be transferred mechanically to the other.

Yang explicitly restored the missing reset derivative for a multiplicative reset. It improved learning-rate robustness in a 100-step single-neuron toy problem, but produced virtually identical results on NMNIST (99.29 vs 99.28), MNIST (99.49 vs 99.50), and CIFAR-10 (88.96 vs 88.98); the author concluded that the gain usually did not justify the overhead ([paper and benchmark table](https://arxiv.org/abs/2011.09964)). Those convolutional benchmarks used only 5–30 time steps and are weak evidence for a long-horizon RSNN.

## Counterevidence: retaining reset can be useful

EXODUS was designed because SLAYER omitted the neuron's reset response in its gradient. EXODUS uses the implicit function theorem to include reset effects and reproduce BPTT-equivalent gradients in its spike-response formulation. It reported more stable layerwise gradients and comparable or higher validation accuracy than SLAYER across three temporal datasets, including reported comparisons of 92.8 ± 2.2% versus 87.8 ± 3.0% and 78.01% versus 70.58% in two settings. The difference grew for neurons with longer memory ([paper](https://arxiv.org/abs/2205.10242); [official implementation](https://github.com/synsense/sinabs-exodus)).

This result prevents a universal “reset gradients are bad” conclusion. It is not, however, a direct contradiction of the Zenke result: EXODUS compares complete temporal-gradient algorithms in a spike-response model, while Zenke and Vogels isolate surrogate scale and reset/recurrent graph paths under BPTT. Algorithmic correctness, gradient scale, and numerical conditioning are entangled differently.

## Temporal credit and the proposed voltage dampening

With reset detached, the passive voltage contribution over `k` steps is approximately `alpha^k`. Replacing its backward derivative by `alpha*(1-d)` changes that to `[alpha*(1-d)]^k` while leaving the forward neuron unchanged. This is a deliberate biased-gradient truncation of voltage memory, not reset detachment itself.

That can suppress exploding gradients, but it can also prevent losses at late times from assigning credit to earlier subthreshold voltages. A value of `d=0.5` is especially aggressive for this model: based on the previously inspected decay range, it approximately halves every one-step voltage derivative and collapses the effective backward voltage timescale to around 1–1.4 ms. None of the primary studies above recommends this as the standard companion to detached reset.

The literature instead points first to:

1. normalize or reduce the surrogate derivative amplitude;
2. control explicit recurrent gain and its surrogate-mediated gradient path;
3. use global gradient-norm clipping as a safety bound;
4. only then test mild backward voltage dampening as an explicit temporal-credit ablation.

These controls serve different purposes. Surrogate scaling reduces gradients passing through spikes, including explicit recurrent feedback. Global clipping bounds the aggregate update without continually shortening the temporal Jacobian. Voltage dampening changes credit assignment at every time step, even when no neuron spikes.

## What would constitute a clear local conclusion

A controlled comparison should use the same initial weights, batches, optimizer, learning-rate schedule, surrogate scale, clipping threshold, and seeds. At minimum compare:

| Condition | Reset path | Surrogate amplitude | Voltage dampening |
|---|---:|---:|---:|
| A | attached | normalized/tuned | 0 |
| B | detached | same as A | 0 |
| C | detached | same as A | mild, only if B is unstable |

Run the 10k network first to eliminate unstable settings, but use the production-size model for the scientific decision because recurrent spectral properties and aggregate gradient norms change with network size. Evaluate multiple seeds and report:

- validation loss against optimization steps **and wall-clock time**;
- pre-clipping global norm, clipping frequency, nonfinite steps, and mixed-precision loss-scale skips;
- gradient norms split by parameter group and loss component;
- firing-rate and membrane-voltage distributions;
- late-versus-early loss sensitivity or another long-lag credit diagnostic;
- final held-out task metrics rather than training loss alone.

The decision rule should be predeclared. Prefer detached reset only if it gives consistently better held-out convergence or stability across seeds without excessive clipping or degraded long-lag behavior. Prefer attached reset if normalized scaling makes it equally stable and it gives better validation performance or temporal credit. If they are statistically indistinguishable, detached reset is the simpler and more conventional engineering default, but the result should be described as equivalence under the tested configuration rather than a general advantage.

## Evidence limits

- The most direct reset/scale ablations use simpler LIF networks, not heterogeneous GLIF neurons with after-spike currents and refractory states.
- Published comparisons mix subtractive and multiplicative reset equations.
- Some papers use “reset gradient” to mean a temporal reset-response term in a spike-response formulation, which is not identical to differentiating `-theta*z[t]` in this implementation.
- No cited paper isolates ASC reset/injection gradients. Detaching the membrane reset does not logically imply detaching spike-dependent ASC updates; that requires its own ablation.

