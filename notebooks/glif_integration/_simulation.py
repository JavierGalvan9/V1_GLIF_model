"""Shared simulation machinery for the integration-accuracy study.

Everything here is a *measuring instrument*: the model itself lives in
``v1_model_utils.glif_propagators``, and this module only drives it and supplies
a sub-stepped float64 ground truth to compare against.

The state dictionary keeps the implementation's field names; report.tex and the
notebook use the symbols of the thesis (Galvan Fraile 2025, Sections 1.2-1.3):

    ``v``     V_j[t]            membrane potential, in units of dV = V_th - E_L
    ``rise``  C^syn_{j,r}[t]    rise variable of receptor basis r
    ``psc``   I^syn_{j,r}[t]    postsynaptic current of receptor basis r
    ``asc``   I^s_j[t]          after-spike current, s in {1, 2}
    ``z``     S_j[t]            spike indicator

and for the coefficients, ``A`` -> script-A, ``B`` -> script-B, ``D`` -> script-D,
``reset_coeff`` -> script-R, matching glif_propagators.membrane_coefficients.
"""

import glob
import json
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DT = 1.0
TAU_SYN = np.load(os.path.join(ROOT, "synaptic_data", "tau_basis.npy"))
CELL_MODELS = os.path.join(
    ROOT, "GLIF_network_nll_core", "components", "cell_models",
    "*_glif_lif_asc_config.json",
)

# Categorical slots 1-4 of the validated default palette, in fixed order.
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
INK = {"primary": "#0b0b0b", "secondary": "#52514e", "muted": "#8c8b85",
       "grid": "#e5e4df", "surface": "#fcfcfb"}


def load_cell_types():
    """The GLIF3 cell types of the V1 model, normalised the way models.py does."""
    C, g, k, amps, t_ref = [], [], [], [], []
    for path in sorted(glob.glob(CELL_MODELS)):
        with open(path) as handle:
            d = json.load(handle)
        scale = d["V_th"] - d["E_L"]            # models.py voltage_scale
        C.append(d["C_m"])
        g.append(d["g"])
        k.append(d["asc_decay"])
        amps.append(np.asarray(d["asc_amps"]) / scale)
        t_ref.append(d["t_ref"])
    C, g = np.array(C), np.array(g)
    return dict(
        C=C, g=g, tau_m=C / g, k=np.array(k), asc_amps=np.array(amps),
        t_ref_steps=np.maximum(np.ceil(np.array(t_ref) / DT), 1).astype(int),
    )


def zero_state(n, dtype=np.float64):
    b = TAU_SYN.size
    return dict(v=np.zeros(n, dtype), rise=np.zeros((n, b), dtype),
                psc=np.zeros((n, b), dtype), asc=np.zeros((n, 2), dtype))


def step(state, c, aux, inp, z, dt=DT):
    """One step of the generalised update, given propagator coefficients `c`.

    This is exactly what models.py._dense_update_impl and the CUDA kernel do.
    """
    d = aux["syn_decay"]
    new_psc = state["psc"] * d + dt * d * state["rise"]
    new_rise = state["rise"] * d + inp * aux["psc_initial"]
    new_asc = aux["asc_decay"] * state["asc"] + z[..., None] * c["asc_jump"]
    drive = (c["A"] * state["psc"] + c["B"] * state["rise"]).sum(-1) \
        + (c["D"] * state["asc"]).sum(-1)
    # A spike resets the membrane and injects an ASC that drives it across the
    # same step; the two constants are kept apart because the training flags
    # detach them separately, but the forward sum is the same either way.
    new_v = (aux["decay"] * state["v"] + drive
             + (c["reset_coeff"] + c["asc_spike_factor"]) * z)
    return dict(v=new_v, rise=new_rise, psc=new_psc, asc=new_asc)


def aux_constants(p, dt=DT):
    """The decays that both schemes share; already exact in the repo."""
    return dict(decay=np.exp(-dt / p["tau_m"]), syn_decay=np.exp(-dt / TAU_SYN),
                psc_initial=np.e / TAU_SYN, asc_decay=np.exp(-dt * p["k"]))


def _quantise(values, store_dtype, work_dtype):
    """Round to the storage dtype, then widen back for the arithmetic."""
    if store_dtype is None:
        return {k: np.asarray(v, work_dtype) for k, v in values.items()}
    return {k: np.asarray(v, store_dtype).astype(work_dtype)
            for k, v in values.items()}


def simulate(coeffs, p, inputs, z, dtype=np.float64, state_dtype=None,
             coeff_dtype=None, dt=DT, record=("v",), narrow=None):
    """Prescribed-spike ("open loop") run: every scheme sees the same drive.

    The production kernel evaluates the step in float32 but *stores* both the
    recurrent state and the propagator constants in the layer's compute dtype,
    which is float16 under a mixed-precision policy.  `state_dtype` and
    `coeff_dtype` model those two storage choices independently, because they
    cost very differently: the state scales with batch and sequence length, the
    constants do not.  Passing both as float16 reproduces production.
    `narrow` limits the state rounding to the named blocks (default: all).
    """
    n = inputs.shape[1]
    aux = _quantise(aux_constants(p, dt), coeff_dtype, dtype)
    c = _quantise(coeffs, coeff_dtype, dtype)
    state = zero_state(n, dtype)
    inputs, z = inputs.astype(dtype), z.astype(dtype)
    out = {k: np.empty((inputs.shape[0],) + state[k].shape) for k in record}
    for t in range(inputs.shape[0]):
        state = step(state, c, aux, inputs[t], z[t], dt)
        if state_dtype is not None:
            state = {k: (v.astype(state_dtype).astype(dtype)
                         if narrow is None or k in narrow else v)
                     for k, v in state.items()}
        for k in record:
            out[k][t] = state[k]
    return out


def reference(p, inputs, z, substeps=512, record=("v",), scheme="exact"):
    """float64 ground truth, by sub-stepping at dt/substeps.

    With ``scheme="euler"`` this is an *independent* convergent method: the
    Euler step is first-order accurate, so its answer approaches the true
    trajectory as 1/substeps and never borrows the closed form under test.  With
    ``scheme="exact"`` the result is invariant to `substeps` and costs one pass -
    use it once the convergence study has established the two agree.

    Events act at their own grid point: the reset and the ASC jump before the
    sub-step loop, the input spikes after it.
    """
    from v1_model_utils.glif_propagators import membrane_coefficients
    h = DT / substeps
    n = inputs.shape[1]
    c = membrane_coefficients(p["tau_m"], p["g"], TAU_SYN, p["k"], p["asc_amps"],
                              h, scheme=scheme)
    aux = aux_constants(p, h)
    psc_initial = np.e / TAU_SYN
    state = zero_state(n)
    blank_in, blank_z = np.zeros_like(inputs[0]), np.zeros(n)
    out = {k: np.empty((inputs.shape[0],) + state[k].shape) for k in record}
    for t in range(inputs.shape[0]):
        state["v"] = state["v"] - z[t]
        state["asc"] = state["asc"] + z[t][:, None] * p["asc_amps"]
        for _ in range(substeps):
            state = step(state, c, aux, blank_in, blank_z, h)
        state["rise"] = state["rise"] + inputs[t] * psc_initial
        for k in record:
            out[k][t] = state[k]
    return out


def simulate_closed(coeffs, p, inputs, dtype=np.float64, state_dtype=None,
                    coeff_dtype=None, v_th=1.0, narrow=None):
    """Self-consistent run: the neuron spikes, resets and adapts on its own."""
    T, n = inputs.shape[0], inputs.shape[1]
    aux = _quantise(aux_constants(p), coeff_dtype, dtype)
    c = _quantise(coeffs, coeff_dtype, dtype)
    state = zero_state(n, dtype)
    inputs = inputs.astype(dtype)
    z, r = np.zeros(n, dtype), np.zeros(n, int)
    spikes, volt = np.zeros((T, n), bool), np.empty((T, n))
    for t in range(T):
        state = step(state, c, aux, inputs[t], z)
        if state_dtype is not None:
            state = {k: (v.astype(state_dtype).astype(dtype)
                         if narrow is None or k in narrow else v)
                     for k, v in state.items()}
        r = np.maximum(r + z.astype(int) * p["t_ref_steps"] - 1, 0)
        z = ((state["v"] >= v_th) & (r == 0)).astype(dtype)
        spikes[t], volt[t] = z > 0, state["v"]
    return spikes, volt


def reference_closed(p, inputs, substeps=512, v_th=1.0, scheme="euler"):
    """Sub-stepped float64 closed loop; the threshold is still tested on the grid.

    Defaults to sub-stepping the Euler step, so the ground truth is reached
    by a method independent of the closed form under test.
    """
    from v1_model_utils.glif_propagators import membrane_coefficients
    h = DT / substeps
    T, n = inputs.shape[0], inputs.shape[1]
    c = membrane_coefficients(p["tau_m"], p["g"], TAU_SYN, p["k"], p["asc_amps"],
                              h, scheme=scheme)
    aux = aux_constants(p, h)
    psc_initial = np.e / TAU_SYN
    state = zero_state(n)
    blank_in, blank_z = np.zeros_like(inputs[0]), np.zeros(n)
    z, r = np.zeros(n), np.zeros(n, int)
    spikes, volt = np.zeros((T, n), bool), np.empty((T, n))
    for t in range(T):
        state["v"] = state["v"] - z
        state["asc"] = state["asc"] + z[:, None] * p["asc_amps"]
        for _ in range(substeps):
            state = step(state, c, aux, blank_in, blank_z, h)
        state["rise"] = state["rise"] + inputs[t] * psc_initial
        r = np.maximum(r + z.astype(int) * p["t_ref_steps"] - 1, 0)
        z = ((state["v"] >= v_th) & (r == 0)).astype(float)
        spikes[t], volt[t] = z > 0, state["v"]
    return spikes, volt


def poisson_drive(T, n, rng, rate=0.35, gain=0.04):
    """Poisson synaptic drive on the ms grid, gamma-distributed weights."""
    b = TAU_SYN.size
    return (rng.random((T, n, b)) < rate) * rng.gamma(2.0, 0.5, (T, n, b)) * gain


def rel_rmse(x, reference):
    return np.sqrt(((x - reference) ** 2).mean()) / np.sqrt((reference ** 2).mean())


def system_matrix(tau_m, g, tau_syn, k):
    """The generator of the linear subthreshold system, ordered

        [rise_0..rise_B, psc_0..psc_B, asc_0, asc_1, V]

    so that ``expm(M dt)`` is the propagator the closed forms claim to be.  This
    is the independent check on the algebra in glif_propagators: a general
    matrix exponential against hand-derived integrals.
    """
    b = len(tau_syn)
    n = 2 * b + 3
    M = np.zeros((n, n))
    C = g * tau_m
    for i in range(b):
        M[i, i] = -1.0 / tau_syn[i]                  # rise decays
        M[b + i, b + i] = -1.0 / tau_syn[i]          # psc decays
        M[b + i, i] = 1.0                            # rise drives psc
        M[-1, b + i] = 1.0 / C                       # psc drives V
    for j in range(2):
        M[2 * b + j, 2 * b + j] = -k[j]              # asc decays
        M[-1, 2 * b + j] = 1.0 / C                   # asc drives V
    M[-1, -1] = -1.0 / tau_m                         # V leaks
    return M


def richardson_reference(p, inputs, z, substeps=128):
    """Richardson-extrapolated Euler ground truth.

    The Euler step is first order, so its error is ``c/M + O(1/M^2)`` and the
    combination ``2 f(2M) - f(M)`` cancels the leading term.  This gives a
    reference that is accurate to ~1e-7 *without* using the closed form under
    test anywhere, so both schemes are measured against a third method.
    """
    coarse = reference(p, inputs, z, substeps, scheme="euler")["v"]
    fine = reference(p, inputs, z, 2 * substeps, scheme="euler")["v"]
    return 2.0 * fine - coarse
