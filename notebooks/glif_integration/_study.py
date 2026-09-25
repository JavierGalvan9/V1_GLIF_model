"""The analysis behind the integration-accuracy study.

Each function returns a results dictionary and nothing else - no plotting, no
printing - so the notebook, the LaTeX report and any regression check all read
exactly the same numbers.  Plotting lives in :mod:`_figures`; the drivers and
the ground truths live in :mod:`_simulation`.

Notation follows the thesis (Galvan Fraile, 2025), Sections 1.2 and 1.3:
``V`` is :math:`V_j`, the synaptic pair is :math:`(C^{syn}_{j,r}, I^{syn}_{j,r})`,
the after-spike currents are :math:`I^s_j`, and :math:`\\delta t = 1` ms.
"""
import json
import os

import numpy as np

import _simulation as S
from v1_model_utils.glif_propagators import membrane_coefficients

HERE = os.path.dirname(os.path.abspath(__file__))
SEED_OPEN, SEED_CLOSED = 0, 1
T_OPEN, T_CLOSED = 300, 800
SUBSTEPS = 128           # Richardson reference is built from M and 2M
CLOSED_SUBSTEPS = 4096   # plain sub-stepped Euler for the closed loop
# The mixed configuration: only the synaptic pair in float16; the membrane,
# the after-spike currents and the propagator constants stay float32.
MIXED = ("psc", "rise")
PICK = 100               # the cell type used for the single-neuron figure


def parameters():
    """Cell types, and the coefficients of both schemes."""
    p = S.load_cell_types()
    coeffs = {
        scheme: membrane_coefficients(p["tau_m"], p["g"], S.TAU_SYN, p["k"],
                                      p["asc_amps"], S.DT, scheme=scheme)
        for scheme in ("euler", "exact")
    }
    return p, coeffs


def drive(p):
    """The open-loop stimulus: Poisson synaptic input, and ~10 Hz output spikes."""
    rng = np.random.default_rng(SEED_OPEN)
    n = p["tau_m"].size
    inputs = S.poisson_drive(T_OPEN, n, rng)
    spikes = (rng.random((T_OPEN, n)) < 0.01).astype(float)
    return inputs, spikes


def propagator_identity(p, coeffs, stride=5):
    """Check the closed forms against a general matrix exponential."""
    from scipy.linalg import expm
    b = S.TAU_SYN.size
    worst = 0.0
    for i in range(0, p["tau_m"].size, stride):
        P = expm(S.system_matrix(p["tau_m"][i], p["g"][i], S.TAU_SYN, p["k"][i]) * S.DT)
        worst = max(worst,
                    np.abs(P[-1, b:2 * b] - coeffs["exact"]["A"][i]).max(),
                    np.abs(P[-1, :b] - coeffs["exact"]["B"][i]).max(),
                    np.abs(P[-1, 2 * b:2 * b + 2] - coeffs["exact"]["D"][i]).max())
    return dict(worst=float(worst))


def convergence(p, coeffs, inputs, spikes):
    """Sub-step Euler towards the propagator, against a Richardson truth."""
    truth = S.richardson_reference(p, inputs, spikes, substeps=SUBSTEPS)
    substeps = [1, 2, 4, 8, 16, 32, 64, 128]
    errors = [S.rel_rmse(S.reference(p, inputs, spikes, m, scheme="euler")["v"], truth)
              for m in substeps]
    return dict(
        substeps=substeps,
        errors=[float(e) for e in errors],
        exact=float(S.rel_rmse(S.simulate(coeffs["exact"], p, inputs, spikes)["v"],
                               truth)),
        legacy=float(S.rel_rmse(S.simulate(coeffs["euler"], p, inputs, spikes)["v"],
                                truth)),
    )


def errors(p, coeffs, inputs, spikes):
    """Relative error per regime and scheme, in float64 and in production.

    Production means what the CUDA kernel really does under a mixed-precision
    policy: arithmetic in float32, but *both* the recurrent state and the
    propagator constants stored in float16.  The intermediate column keeps the
    constants wide, which is what separates the two storage costs.
    """
    n = p["tau_m"].size
    cases = {"subthreshold": np.zeros((T_OPEN, n)), "spiking (~10 Hz)": spikes}
    half = dict(dtype=np.float32, state_dtype=np.float16)
    rows = []
    for case, z in cases.items():
        reference = S.richardson_reference(p, inputs, z, substeps=SUBSTEPS)
        for scheme in ("euler", "exact"):
            c = coeffs[scheme]
            rows.append(dict(
                case=case, scheme=scheme,
                fp64=float(S.rel_rmse(S.simulate(c, p, inputs, z)["v"], reference)),
                fp16_state=float(S.rel_rmse(
                    S.simulate(c, p, inputs, z, **half)["v"], reference)),
                fp16=float(S.rel_rmse(
                    S.simulate(c, p, inputs, z, coeff_dtype=np.float16,
                               **half)["v"], reference)),
                mixed=float(S.rel_rmse(
                    S.simulate(c, p, inputs, z, narrow=MIXED, **half)["v"],
                    reference)),
            ))
    return rows


def decomposition(p, coeffs, inputs, spikes):
    """How much of Euler's float64 error each of its two approximations causes.

    The Euler step holds the synaptic current over the step *and* applies the
    spike events (reset, ASC increment) at its end. The two live in disjoint
    constants, so crossing them gives one hybrid that fixes only the current
    and one that fixes only the events.
    """
    current, events = ("A", "B", "D"), ("reset_coeff", "asc_spike_factor", "asc_jump")

    def hybrid(for_current, for_events):
        return {**{k: coeffs[for_current][k] for k in current},
                **{k: coeffs[for_events][k] for k in events}}

    schemes = dict(euler=coeffs["euler"], events_only=hybrid("exact", "euler"),
                   current_only=hybrid("euler", "exact"), exact=coeffs["exact"])
    n = p["tau_m"].size
    out = {}
    for case, z in (("subthreshold", np.zeros((T_OPEN, n))), ("spiking", spikes)):
        reference = S.richardson_reference(p, inputs, z, substeps=SUBSTEPS)
        out[case] = {name: float(S.rel_rmse(S.simulate(c, p, inputs, z)["v"], reference))
                     for name, c in schemes.items()}
    return out


def psp(p, coeffs, length=40, arrival=2):
    """The postsynaptic potential one presynaptic spike produces."""
    one = {k: np.asarray(p[k])[[PICK]] for k in
           ("C", "g", "tau_m", "k", "asc_amps", "t_ref_steps")}
    one_coeffs = {s: {k: np.asarray(v)[[PICK]] for k, v in coeffs[s].items()}
                  for s in coeffs}
    inputs = np.zeros((length, 1, S.TAU_SYN.size))
    inputs[arrival, 0, 0] = 1.0                  # into the fastest basis
    quiet = np.zeros((length, 1))
    truth = S.richardson_reference(one, inputs, quiet, substeps=SUBSTEPS)[:, 0]
    traces = {s: S.simulate(one_coeffs[s], one, inputs, quiet)["v"][:, 0]
              for s in coeffs}
    return dict(ms=np.arange(length), truth=truth, traces=traces,
                peaks={s: float(v.max()) for s, v in traces.items()},
                tau_m=float(one["tau_m"][0]),
                underestimate=float(1 - traces["euler"].max() / truth.max()))


def _one_off(spikes, reference):
    """Reference spikes a run misses in their own ms but hits in a neighbour."""
    near = np.roll(spikes, 1, axis=0) | np.roll(spikes, -1, axis=0)
    return reference & ~spikes & near


def closed_loop(p, coeffs):
    """Let the neurons spike, reset and adapt on their own."""
    rng = np.random.default_rng(SEED_CLOSED)
    n = p["tau_m"].size
    inputs = S.poisson_drive(T_CLOSED, n, rng, rate=0.5, gain=0.3)
    # Ground truth by sub-stepping the OLD method: independent of the closed form.
    # Richardson extrapolation does not apply here: the neurons spike on their
    # own, so runs at M and 2M can diverge by a whole spike.
    reference_spikes, _ = S.reference_closed(p, inputs, substeps=CLOSED_SUBSTEPS,
                                             scheme="euler")
    rows = []
    for scheme in ("euler", "exact"):
        for tag, kw in [("float64", {}),
                        ("float16", dict(dtype=np.float32,
                                         state_dtype=np.float16,
                                         coeff_dtype=np.float16)),
                        ("mixed", dict(dtype=np.float32,
                                       state_dtype=np.float16, narrow=MIXED))]:
            s, _ = S.simulate_closed(coeffs[scheme], p, inputs, **kw)
            rows.append(dict(scheme=scheme, precision=tag,
                             match=float((s & reference_spikes).sum()
                                         / reference_spikes.sum()),
                             within_one=float(((s & reference_spikes).sum()
                                               + _one_off(s, reference_spikes).sum())
                                              / reference_spikes.sum()),
                             rate_err=float(abs(s.sum() - reference_spikes.sum())
                                            / reference_spikes.sum()),
                             spikes=s))
    # The same comparison against an 8x coarser reference: if the exact scheme's
    # agreement moves, what is left of its mismatch is the reference's error.
    coarse, _ = S.reference_closed(p, inputs, substeps=CLOSED_SUBSTEPS // 8,
                                   scheme="euler")
    exact64 = next(r["spikes"] for r in rows
                   if r["scheme"] == "exact" and r["precision"] == "float64")
    # How far the production float16 run's misses land from the truth.
    exact16 = next(r["spikes"] for r in rows
                   if r["scheme"] == "exact" and r["precision"] == "float16")
    missed = reference_spikes & ~exact16
    one_off = _one_off(exact16, reference_spikes)
    return dict(rows=rows, reference=reference_spikes,
                coarse_match=float((exact64 & coarse).sum() / coarse.sum()),
                half_missed=int(missed.sum()),
                half_one_off=float(one_off.sum() / missed.sum()),
                half_within_one=float(((reference_spikes & exact16).sum()
                                       + one_off.sum()) / reference_spikes.sum()),
                n_spikes=int(reference_spikes.sum()),
                rate=float(reference_spikes.mean() * 1000),
                by_scheme={r["scheme"]: r["spikes"] for r in rows
                           if r["precision"] == "float64"})


def precision(p, coeffs, inputs, spikes):
    """Which stored block sets the float16 floor of the exact scheme.

    The four state blocks scale with batch and sequence length; the propagator
    constants do not, so they are the cheapest of the five to widen and are
    ablated alongside them.
    """
    n = p["tau_m"].size
    widths = {"v": 1, "asc": 2, "psc": S.TAU_SYN.size, "rise": S.TAU_SYN.size}
    total = sum(widths.values())
    reference = S.richardson_reference(p, inputs, spikes, substeps=SUBSTEPS)

    def run(half, narrow_coeffs):
        c = S._quantise(coeffs["exact"],
                        np.float16 if narrow_coeffs else None, np.float32)
        aux = S._quantise(S.aux_constants(p),
                          np.float16 if narrow_coeffs else None, np.float32)
        state = S.zero_state(n, np.float32)
        out = np.empty((T_OPEN, n))
        for t in range(T_OPEN):
            state = S.step(state, c, aux, inputs[t].astype(np.float32),
                           spikes[t].astype(np.float32))
            state = {k: (v.astype(np.float16).astype(np.float32) if k in half else v)
                     for k, v in state.items()}
            out[t] = state["v"]
        return out

    # The symbols of the report: v is V_j, psc/rise are I^syn/C^syn, asc is I^s.
    symbol = {"v": "$V_j$", "asc": "$I^s_j$",
              "psc": r"$I^{\mathrm{syn}}$", "rise": r"$C^{\mathrm{syn}}$"}
    rows = []
    for half, narrow_coeffs in [((), False), (("psc", "rise"), False),
                                (("asc",), False), (("v",), False),
                                (("v", "asc"), False),
                                (("v", "psc", "rise", "asc"), False),
                                (("v", "psc", "rise", "asc"), True)]:
        narrow = sum(widths[k] for k in half)
        kept = [k for k in ("v", "asc", "psc", "rise") if k not in half]
        label = ", ".join(symbol[k] for k in half) or "nothing"
        if narrow_coeffs:
            label += ", constants"
        rows.append(dict(
            half=", ".join(half) + (", constants" if narrow_coeffs else "") or "nothing",
            kept=", ".join(kept) or "nothing",
            symbols=label,
            coeffs_narrow=narrow_coeffs,
            err=float(S.rel_rmse(run(set(half), narrow_coeffs), reference)),
            bytes=2 * narrow + 4 * (total - narrow)))
    euler = float(S.rel_rmse(
        S.simulate(coeffs["euler"], p, inputs, spikes, np.float32, np.float16,
                   coeff_dtype=np.float16)["v"], reference))
    return dict(rows=rows, euler_fp16=euler)


def benchmark():
    """The kernel timings, if benchmark_kernels.py has been run."""
    path = os.path.join(HERE, "benchmark_results.json")
    if not os.path.exists(path):
        return None
    with open(path) as handle:
        rows = json.load(handle)
    for r in rows:
        r["ratio"] = ((r["new_fwd"] + r["new_bwd"])
                      / (r["legacy_fwd"] + r["legacy_bwd"]))
    fp16 = [r["ratio"] for r in rows if r["dtype"] == "fp16"]
    return dict(rows=rows, fp16_mean=float(np.mean(fp16)),
                fp16_min=float(min(fp16)), fp16_max=float(max(fp16)))


def gradients():
    """The finite-difference gradient check, if verify_gradients.py has run."""
    path = os.path.join(HERE, "gradient_check.json")
    if not os.path.exists(path):
        return None
    with open(path) as handle:
        results = json.load(handle)
    worst = {k: max(v["rel_err"] for v in blocks.values())
             for k, blocks in results.items()}
    return dict(blocks=results, worst=worst, overall=max(worst.values()))


def run_all():
    """Every result in the study, in one pass."""
    p, coeffs = parameters()
    inputs, spikes = drive(p)
    return dict(
        p=p, coeffs=coeffs, inputs=inputs, spikes=spikes,
        identity=propagator_identity(p, coeffs),
        convergence=convergence(p, coeffs, inputs, spikes),
        errors=errors(p, coeffs, inputs, spikes),
        decomposition=decomposition(p, coeffs, inputs, spikes),
        psp=psp(p, coeffs),
        closed_loop=closed_loop(p, coeffs),
        precision=precision(p, coeffs, inputs, spikes),
        benchmark=benchmark(),
        gradients=gradients(),
    )
