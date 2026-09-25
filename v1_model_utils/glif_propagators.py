"""Step propagators for the GLIF3 membrane / alpha-synapse / ASC system.

Between spikes the GLIF3 neuron is a *linear* time-invariant system, so the
transition over one simulation step has a closed form (Rotter & Diesmann 1999;
Morrison et al. 2007) - the same propagator-matrix construction NEST uses for
``iaf_psc_alpha``.  In normalised units (E_L = 0, V_th = 1, V_reset = 0):

    rise_b' = -rise_b / tau_b
    psc_b'  = -psc_b / tau_b + rise_b
    asc_j'  = -k_j asc_j
    V'      = -V / tau_m + (sum_b psc_b + sum_j asc_j) / C

The repo has always propagated ``rise``, ``psc`` and ``asc`` exactly; the
membrane was the weak link, integrated with a *zero-order hold* that froze the
synaptic current at its start-of-step value.  With dt = 1 ms and a fastest
synaptic basis of tau ~ 0.99 ms, the current changes by an order of magnitude
inside a single step, so that hold is the dominant error of the simulation.

This module returns the per-neuron coefficients of the generalised step

    V(t+dt) = decay * V(t) + sum_b [A_b psc_b(t) + B_b rise_b(t)]
              + sum_j D_j asc_j(t) + reset_coeff * z(t) + asc_spike_factor * z(t)
    asc_j(t+dt) = asc_decay_j asc_j(t) + asc_jump_j z(t)

which covers both the historical scheme and the exact one - see
:func:`membrane_coefficients` - so the integrator is chosen entirely by which
constants are built at setup time.  No branch, no extra state, no runtime cost
beyond a handful of fused multiply-adds per neuron per step.
"""

import numpy as np

SCHEMES = ("euler", "exact")


def _expm1_ratio(x):
    """``(exp(x) - 1) / x``, series-continued through x = 0."""
    x = np.asarray(x, np.float64)
    out = np.empty_like(x)
    big = np.abs(x) >= 1e-6
    out[big] = np.expm1(x[big]) / x[big]
    xs = x[~big]
    out[~big] = 1.0 + xs / 2.0 + xs * xs / 6.0
    return out


def _phi1(a, b, dt):
    """``int_0^dt e^{-a(dt-s)} e^{-b s} ds``, stable as ``a -> b``.

    Closed form ``(e^{-b dt} - e^{-a dt}) / (a - b)``, rewritten through
    :func:`_expm1_ratio` so the removable singularity costs no accuracy.
    """
    a = np.asarray(a, np.float64)
    return np.exp(-a * dt) * dt * _expm1_ratio((a - np.asarray(b, np.float64)) * dt)


def _phi2_factor(x):
    """``f(x) = (e^x (x - 1) + 1) / x^2``, to full precision for every x.

    Written directly, the numerator cancels two O(1) terms to produce one of
    size ``x^2 / 2``, so it loses ``2 log10(1/|x|)`` digits - eight of them at
    ``x = 1e-4``.  The identity ``f = 1 + (x - 1) phi2(x)`` with
    ``phi2(x) = (expm1(x)/x - 1)/x`` cancels only against O(1/2) and costs one
    order of ``x`` instead of two.  Below the crossover the Taylor series
    ``sum_n x^n (n+1)/(n+2)!`` takes over; the cut-off is where the series
    truncation (``~x^5/420``) meets the recurrence's rounding (``~2 eps/x``),
    leaving a worst relative error of about 1e-13 over all x.
    """
    out = np.empty_like(x)
    small = np.abs(x) < 7.5e-3
    xs = x[small]
    out[small] = (0.5 + xs / 3.0 + xs * xs / 8.0 + xs ** 3 / 30.0
                  + xs ** 4 / 144.0)
    xb = x[~small]
    out[~small] = 1.0 + (xb - 1.0) * ((np.expm1(xb) / xb) - 1.0) / xb
    return out


def _phi2(a, b, dt):
    """``int_0^dt e^{-a(dt-s)} s e^{-b s} ds``, stable as ``a -> b``.

    Equal to ``dt^2 e^{-a dt} f(x)`` with ``x = (a - b) dt`` and ``f`` as in
    :func:`_phi2_factor`.  The degeneracy is not hypothetical here: the slowest
    synaptic basis (5.65 ms) sits inside the membrane time-constant range
    (5.5-51.8 ms), so some neurons land on ``a == b`` to within rounding.
    """
    a = np.asarray(a, np.float64)
    x = (a - np.asarray(b, np.float64)) * dt
    return _phi2_factor(x) * dt * dt * np.exp(-a * dt)


def membrane_coefficients(tau_m, g, tau_syn, k, asc_amps, dt, scheme="exact"):
    """Per-neuron coefficients of the generalised membrane step.

    Parameters
    ----------
    tau_m : (N,) membrane time constants, ms.
    g : (N,) leak conductances, in the same units the currents are expressed in.
    tau_syn : (B,) synaptic basis time constants, ms.
    k : (N, 2) ASC rate constants, 1/ms.
    asc_amps : (N, 2) ASC jump amplitudes, already divided by the voltage scale.
    dt : simulation step, ms.
    scheme : ``"exact"`` for the closed-form propagator, ``"euler"`` to reproduce
        the historical Euler scheme bit for bit.

    Returns
    -------
    dict with ``A`` (N, B), ``B`` (N, B), ``D`` (N, 2), ``reset_coeff`` (N,),
    ``asc_spike_factor`` (N,) and ``asc_jump`` (N, 2).

    A spike has two separate effects on the membrane within its own step, and
    they are returned separately because the training flags govern them
    separately: ``reset_coeff`` is the reset alone (``detach_reset``), and
    ``asc_spike_factor`` is the drive from the ASC that spike injects
    (``detach_asc_reset``).  Folding them into one constant would make
    ``detach_reset`` silently cut part of the ASC path as well.

    Both schemes share one evaluation path, so the only thing ``scheme``
    selects is the value of the constants:

    ``"euler"``   A_b = D_j = (1 - decay)/g, B_b = 0, reset_coeff = -1,
                asc_spike_factor = 0, asc_jump = asc_amps.  The current is held
                at its start-of-step value and spike events are applied at the
                *end* of the step, so the ASC a spike injects does not drive the
                membrane until the following step.
    ``"exact"`` A, B, D are the exact convolution integrals of the alpha and
                ASC kernels against the membrane kernel.  A spike emitted at
                grid point t resets V and injects its ASC jump *at* t, so both
                propagate across the step: the reset decays by ``decay`` and the
                fresh ASC drives V through ``D``, which is what
                ``asc_spike_factor`` carries.
    """
    if scheme not in SCHEMES:
        raise ValueError(f"scheme must be one of {SCHEMES}, got {scheme!r}")
    tau_m = np.asarray(tau_m, np.float64)
    g = np.asarray(g, np.float64)
    tau_syn = np.asarray(tau_syn, np.float64)
    k = np.asarray(k, np.float64)
    asc_amps = np.asarray(asc_amps, np.float64)
    decay = np.exp(-dt / tau_m)
    C = g * tau_m
    n_basis = tau_syn.size

    if scheme == "euler":
        current_factor = ((1.0 - decay) / g)[:, None]
        return dict(
            A=np.repeat(current_factor, n_basis, axis=1),
            B=np.zeros((tau_m.size, n_basis)),
            D=np.repeat(current_factor, 2, axis=1),
            reset_coeff=-np.ones_like(tau_m),
            asc_spike_factor=np.zeros_like(tau_m),
            asc_jump=asc_amps.copy(),
        )

    a = (1.0 / tau_m)[:, None]
    beta = (1.0 / tau_syn)[None, :]
    A = _phi1(*np.broadcast_arrays(a, beta), dt) / C[:, None]
    B = _phi2(*np.broadcast_arrays(a, beta), dt) / C[:, None]
    D = _phi1(*np.broadcast_arrays(a, k), dt) / C[:, None]
    return dict(
        A=A,
        B=B,
        D=D,
        # The reset acts at the spike's own grid point, so it decays across the
        # step.  V_th - E_L is 1 in the normalised units the model works in.
        reset_coeff=-decay,
        # The ASC that same spike injects drives the membrane across the step.
        asc_spike_factor=(D * asc_amps).sum(-1),
        # The jump lands at the spike's grid point and then decays.
        asc_jump=np.exp(-dt * k) * asc_amps,
    )
