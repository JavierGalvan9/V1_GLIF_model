"""The membrane propagator must be exact, and must still reproduce the legacy step."""

import numpy as np
import pytest

from v1_model_utils.glif_propagators import membrane_coefficients

DT = 1.0
TAU_SYN = np.array([0.98958772, 1.88019093, 3.62895928, 5.65340431])


def _params(n=6, seed=0):
    rng = np.random.default_rng(seed)
    tau_m = np.concatenate([rng.uniform(5.5, 52.0, n - 1), [TAU_SYN[-1]]])
    return dict(
        tau_m=tau_m,
        g=rng.uniform(2.0, 20.0, tau_m.size),
        k=np.stack([rng.uniform(0.003, 0.02, tau_m.size),
                    rng.uniform(0.02, 0.3, tau_m.size)], axis=1),
        asc_amps=rng.uniform(-8.0, 8.0, (tau_m.size, 2)),
    )


def _step(state, c, decay, inp, z, psc_initial, syn_decay, asc_decay, dt=DT):
    psc, rise, asc, v = state
    drive = (c["A"] * psc + c["B"] * rise).sum(-1) + (c["D"] * asc).sum(-1)
    return (psc * syn_decay + dt * syn_decay * rise,
            rise * syn_decay + inp * psc_initial,
            asc_decay * asc + z[:, None] * c["asc_jump"],
            decay * v + drive
            + (c["reset_coeff"] + c["asc_spike_factor"]) * z)


def _simulate(scheme, p, inputs, z, dt=DT, substeps=1):
    """Run the propagator at dt/substeps; events land on the coarse grid."""
    h = dt / substeps
    c = membrane_coefficients(p["tau_m"], p["g"], TAU_SYN, p["k"], p["asc_amps"],
                              h, scheme=scheme)
    decay, syn_decay = np.exp(-h / p["tau_m"]), np.exp(-h / TAU_SYN)
    asc_decay, psc_initial = np.exp(-h * p["k"]), np.e / TAU_SYN
    n, b = p["tau_m"].size, TAU_SYN.size
    state = (np.zeros((n, b)), np.zeros((n, b)), np.zeros((n, 2)), np.zeros(n))
    zero_in, zero_z = np.zeros((n, b)), np.zeros(n)
    out = np.empty((inputs.shape[0], n))
    for t in range(inputs.shape[0]):
        for m in range(substeps):
            last = m == substeps - 1
            state = _step(state, c, decay, inputs[t] if last else zero_in,
                          z[t] if m == 0 else zero_z,
                          psc_initial, syn_decay, asc_decay, h)
        out[t] = state[3]
    return out


@pytest.fixture(scope="module")
def drive():
    rng = np.random.default_rng(7)
    p = _params()
    n, b = p["tau_m"].size, TAU_SYN.size
    inputs = (rng.random((200, n, b)) < 0.4) * rng.gamma(2.0, 0.5, (200, n, b)) * 0.3
    z = (rng.random((200, n)) < 0.02).astype(float)
    return p, inputs, z


def test_zoh_reproduces_the_legacy_step(drive):
    p, inputs, z = drive
    c = membrane_coefficients(p["tau_m"], p["g"], TAU_SYN, p["k"], p["asc_amps"],
                              DT, scheme="euler")
    current_factor = (1.0 - np.exp(-DT / p["tau_m"])) / p["g"]
    np.testing.assert_allclose(c["A"], current_factor[:, None] * np.ones(TAU_SYN.size))
    np.testing.assert_allclose(c["D"], current_factor[:, None] * np.ones(2))
    assert not c["B"].any()
    np.testing.assert_array_equal(c["reset_coeff"], -np.ones(p["tau_m"].size))
    assert not c["asc_spike_factor"].any()
    np.testing.assert_array_equal(c["asc_jump"], p["asc_amps"])


def test_exact_scheme_is_invariant_to_the_step_it_is_built_for(drive):
    """Sub-stepping an exact propagator must not change the answer."""
    p, inputs, z = drive
    coarse = _simulate("exact", p, inputs, z)
    fine = _simulate("exact", p, inputs, z, substeps=64)
    assert np.abs(coarse - fine).max() < 1e-12 * max(np.abs(fine).max(), 1.0)


def test_exact_scheme_beats_the_legacy_one_against_a_substepped_reference(drive):
    p, inputs, z = drive
    reference = _simulate("euler", p, inputs, z, substeps=4096)
    scale = np.sqrt((reference ** 2).mean())
    err = {s: np.sqrt(((_simulate(s, p, inputs, z) - reference) ** 2).mean()) / scale
           for s in ("euler", "exact")}
    assert err["exact"] < err["euler"] / 100.0
    assert err["exact"] < 1e-4


def test_degenerate_time_constants_stay_finite():
    """tau_m == tau_syn is a removable singularity of the alpha propagator."""
    n = TAU_SYN.size
    c = membrane_coefficients(TAU_SYN, np.full(n, 5.0), TAU_SYN,
                              np.full((n, 2), 1.0 / TAU_SYN[0]),
                              np.zeros((n, 2)), DT)
    for key, value in c.items():
        assert np.isfinite(value).all(), key
    # On the diagonal the alpha coefficient collapses to dt^2/2 * e^{-dt/tau}/C.
    tau = TAU_SYN
    expected = 0.5 * DT ** 2 * np.exp(-DT / tau) / (5.0 * tau)
    np.testing.assert_allclose(np.diag(c["B"]), expected, rtol=1e-10)


def test_rejects_unknown_scheme():
    with pytest.raises(ValueError, match="scheme"):
        membrane_coefficients([10.0], [1.0], TAU_SYN, [[0.1, 0.2]], [[1.0, 1.0]],
                              DT, scheme="rk4")


def _expm_membrane_row(tau_m, g, tau_syn, k, dt):
    """The membrane row of exp(M dt), computed by scaling-and-squaring.

    An independent route to the same coefficients: it shares no code with the
    closed forms, so it catches algebra errors and cancellation alike.
    """
    from scipy.linalg import expm
    b = len(tau_syn)
    M = np.zeros((2 * b + 3, 2 * b + 3))
    C = g * tau_m
    for i in range(b):
        M[i, i] = M[b + i, b + i] = -1.0 / tau_syn[i]
        M[b + i, i] = 1.0
        M[-1, b + i] = 1.0 / C
    for j in range(2):
        M[2 * b + j, 2 * b + j] = -k[j]
        M[-1, 2 * b + j] = 1.0 / C
    M[-1, -1] = -1.0 / tau_m
    return expm(M * dt)[-1]


@pytest.mark.parametrize("scale", [1e-7, 1e-5, 1e-3, 5e-3, 7.5e-3, 1e-2, 0.1, 1.0])
def test_coefficients_hold_precision_through_the_near_degenerate_band(scale):
    """Sweep tau_m towards tau_syn and demand full precision all the way in.

    ``B`` is the delicate one: written directly, its factor
    ``(e^x (x-1) + 1) / x^2`` cancels two O(1) terms to make one of size
    ``x^2/2``, which silently costs eight digits around ``x = 1e-4``. The band
    is reachable in practice - the slowest synaptic basis sits inside the
    membrane time-constant range - so it is swept explicitly here.
    """
    tau_syn, g, k = TAU_SYN, np.array([8.0]), np.array([[0.003, 0.3]])
    for sign in (+1, -1):
        # x = (1/tau_m - 1/tau_r) dt; place tau_m so that x = sign * scale.
        target = 1.0 / tau_syn[-1] + sign * scale / DT
        tau_m = np.array([1.0 / target])
        c = membrane_coefficients(tau_m, g, tau_syn, k, np.zeros((1, 2)), DT)
        row = _expm_membrane_row(tau_m[0], g[0], tau_syn, k[0], DT)
        b = tau_syn.size
        for name, got, want in (("A", c["A"][0], row[b:2 * b]),
                                ("B", c["B"][0], row[:b]),
                                ("D", c["D"][0], row[2 * b:2 * b + 2])):
            rel = np.abs(got - want) / np.abs(want)
            assert rel.max() < 1e-11, (
                f"{name} lost precision at x = {sign * scale:+.1e}: "
                f"relative error {rel.max():.2e}"
            )
