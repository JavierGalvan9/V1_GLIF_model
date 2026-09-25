"""The figures of the integration-accuracy study.

One definition per figure, shared by the notebook and by the LaTeX report, so
the two can never drift.  Each function takes results that are already computed
and returns a matplotlib ``Figure``.

Symbols follow the thesis (Galvan Fraile, 2025): the membrane potential is
:math:`V_j`, the synaptic state pair is :math:`(C^{syn}_{j,r}, I^{syn}_{j,r})`,
the after-spike currents are :math:`I^s_j`, and the step is
:math:`\\delta t = 1` ms.
"""
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

# Categorical slots 1-4 of a CVD-validated palette, assigned in fixed order.
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
INK = {"primary": "#0b0b0b", "secondary": "#52514e", "muted": "#8c8b85",
       "grid": "#e5e4df", "truth": "#c9c8c1", "surface": "#fcfcfb"}
EULER, EXACT = PALETTE[1], PALETTE[0]
EXACT_HALF = PALETTE[2]

RC = {
    "figure.facecolor": INK["surface"], "axes.facecolor": INK["surface"],
    "savefig.facecolor": INK["surface"], "figure.dpi": 120,
    "axes.edgecolor": INK["grid"], "axes.labelcolor": INK["secondary"],
    "axes.titlecolor": INK["primary"], "axes.titlesize": 11,
    "axes.titleweight": "semibold", "axes.titlelocation": "left",
    "axes.spines.top": False, "axes.spines.right": False,
    "xtick.color": INK["secondary"], "ytick.color": INK["secondary"],
    "grid.color": INK["grid"], "grid.linewidth": 0.8, "font.size": 9,
    "legend.frameon": False, "lines.linewidth": 2.0,
    "mathtext.fontset": "cm", "pdf.fonttype": 42,
}


def use_style():
    mpl.rcParams.update(RC)


def timescales(tau_m, tau_syn, dt):
    """Where the synaptic time constants sit relative to the step, and what the
    Euler step puts in place of the alpha current."""
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.3), constrained_layout=True)

    ax = axes[0]
    ax.hist(tau_m, bins=np.logspace(np.log10(0.8), np.log10(60), 40),
            color=EXACT, alpha=0.9)
    top = ax.get_ylim()[1]
    ax.set_ylim(0, top * 1.30)
    # The basis markers stop below the label band so nothing overlaps the bars.
    for i, tau in enumerate(tau_syn):
        ax.axvline(tau, color=EULER, lw=2, ymax=0.77)
        ax.annotate(rf"$\tau_{i+1}$={tau:.2f}", xy=(tau, top * 1.00),
                    xytext=(tau, top * (1.27 - 0.085 * (i % 2))),
                    color=EULER, fontsize=7.5, va="bottom", ha="center",
                    arrowprops=dict(arrowstyle="-", color=EULER, lw=0.8,
                                    shrinkA=1, shrinkB=0))
    ax.axvline(dt, color=INK["primary"], lw=1.8, ls=":", ymax=0.77)
    ax.text(dt * 0.92, top * 0.55, r"$\delta t$", color=INK["primary"],
            fontsize=10, ha="right", va="center")
    ax.text(0.97, 0.62, "membrane $\\tau_m$ (blue)\nsynaptic bases $\\tau_r$ (orange)",
            transform=ax.transAxes, ha="right", va="top", fontsize=8,
            color=INK["secondary"])
    ax.set(xscale="log", xlabel=r"time constant (ms)", ylabel="cell types",
           title="The fastest synapse is as fast as the step")
    ax.grid(axis="y", alpha=0.5)

    ax = axes[1]
    tau = tau_syn[0]
    fine = np.linspace(0, 6, 800)
    alpha = (np.e / tau) * fine * np.exp(-fine / tau)
    grid = np.arange(0, 7)
    held = (np.e / tau) * grid * np.exp(-grid / tau)
    ax.plot(fine, alpha, color=EXACT, label=r"$I^{\mathrm{syn}}_{j,r}(t)$", zorder=3)
    ax.step(np.append(grid, 7), np.append(held, held[-1]), where="post",
            color=EULER, lw=2, label="what Euler integrates")
    ax.scatter(grid, held, s=22, color=EULER, zorder=4)
    ax.annotate("the spike's own step\nis integrated as zero",
                xy=(0.5, 0.02), xytext=(0.75, alpha.max() * 0.30), fontsize=8,
                color=INK["secondary"], ha="left",
                arrowprops=dict(arrowstyle="->", color=INK["muted"], lw=1,
                                connectionstyle="arc3,rad=0.25"))
    ax.set(xlabel="time since the presynaptic spike (ms)",
           ylabel="synaptic current (a.u.)", xlim=(0, 6),
           title=rf"Held constant for a whole step ($\tau_1$ = {tau:.2f} ms)")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(axis="y", alpha=0.5)
    return fig


def convergence(substeps, errors, exact, legacy):
    """The sub-stepped Euler step walking towards the propagator as 1/M."""
    fig, ax = plt.subplots(figsize=(5.8, 3.6), constrained_layout=True)
    ax.loglog(substeps, errors, "o-", color=EULER, ms=6,
              label="Euler, sub-stepped")
    ax.axhline(exact, color=EXACT, lw=2, label="exact propagator, one step")
    ax.scatter([1], [legacy], marker="s", s=60, color=EULER, zorder=4,
               label="the repo's scheme (Euler + end-of-step events)")
    ax.loglog(substeps, errors[0] / np.array(substeps), ls=":", lw=1.4,
              color=INK["muted"])
    ax.text(substeps[-4], errors[0] / substeps[-4] * 2.2, r"$\propto 1/M$",
            color=INK["muted"], fontsize=8)
    ax.text(1.15, exact * 2.2, f"exact: {exact:.0e}  (the reference's own floor)",
            color=EXACT, fontsize=8)
    ax.text(1.35, legacy, f"  the repo: {legacy:.0e}", color=EULER, fontsize=8,
            va="center")
    ax.set(xlabel="sub-steps per millisecond $M$",
           ylabel=r"relative RMS error in $V_j$",
           title="Euler converges to the propagator, from "
                 + f"{legacy / exact:,.0f}".replace(",", "\u2009") + "x away",
           ylim=(exact / 4, legacy * 5))
    ax.grid(alpha=0.6, which="both")
    ax.legend(loc="center left", fontsize=8)
    return fig


def error_bars(rows):
    """Relative error per regime and scheme, at the three storage choices."""
    fig, ax = plt.subplots(figsize=(7.4, 3.5), constrained_layout=True)
    x = np.arange(len(rows))
    series = [("fp64", EXACT, "float64"),
              ("fp16_state", PALETTE[2], "float16 state"),
              ("fp16", PALETTE[3], "float16 state + constants")]
    width = 0.26
    for i, (key, colour, name) in enumerate(series):
        values = [r[key] for r in rows]
        bars = ax.bar(x + (i - 1) * width, values, width=width * 0.92,
                      color=colour, label=name,
                      edgecolor=INK["surface"], linewidth=1.0)
        for bar, v in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, v * 1.3, f"{v:.0e}",
                    ha="center", fontsize=6.8, color=INK["secondary"], rotation=90)
    floor = min(min(r[k] for k, _, _ in series) for r in rows)
    ax.set(yscale="log", xticks=x, ylabel=r"relative RMS error in $V_j$",
           title="Storing the propagator constants in float16 costs more than "
                 "storing the state",
           ylim=(floor / 4, 8.0))
    ax.set_xticklabels([f"{r['case']}\n{r['scheme']}" for r in rows], fontsize=8)
    ax.grid(axis="y", alpha=0.6)
    ax.legend(loc="upper right", fontsize=8, ncols=3)
    return fig


def psp(ms, truth, traces, peaks):
    """The postsynaptic potential produced by one presynaptic spike."""
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.3), constrained_layout=True)
    for ax, (lo, hi, title) in zip(axes, [(0, 30, "Post-synaptic potential"),
                                          (1.5, 9.5, "The steps that matter")]):
        ax.plot(ms, truth, color=INK["truth"], lw=4.5, label="ground truth",
                solid_capstyle="round")
        for scheme, colour, name in [("euler", EULER, "Euler"),
                                     ("exact", EXACT, "exact propagator")]:
            ax.plot(ms, traces[scheme], color=colour, marker="o", ms=3.6, label=name)
        ax.set(xlabel="time (ms)", ylabel=r"$V_j$  ($\Delta V$ units)",
               title=title, xlim=(lo, hi))
        ax.grid(alpha=0.5)
    axes[1].legend(loc="lower right", fontsize=8)
    axes[0].text(0.97, 0.97,
                 f"peak PSP\n truth  {truth.max():.4f}\n exact  {peaks['exact']:.4f}"
                 f"\n Euler  {peaks['euler']:.4f}",
                 transform=axes[0].transAxes, ha="right", va="top", fontsize=8,
                 family="monospace", color=INK["secondary"])
    return fig


def spike_fidelity(closed, reference_spikes, by_scheme, window, n_neurons=14):
    """How many spikes land in the right millisecond, and what that looks like."""
    fig, axes = plt.subplots(1, 2, figsize=(9.8, 3.6), constrained_layout=True,
                             gridspec_kw=dict(width_ratios=[1, 1.65]))
    ax = axes[0]
    groups = ["float64", "float16"]
    x = np.arange(len(groups))
    for i, (scheme, colour, name) in enumerate([("euler", EULER, "Euler"),
                                                ("exact", EXACT, "exact propagator")]):
        values = [next(r["match"] for r in closed if r["scheme"] == scheme
                       and r["precision"] == g) * 100 for g in groups]
        bars = ax.bar(x + (i - 0.5) * 0.38, values, width=0.36, color=colour,
                      label=name, edgecolor=INK["surface"], linewidth=1.2)
        for bar, v in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, v + 3, f"{v:.1f}%",
                    ha="center", fontsize=8.5, color=INK["secondary"])
    ax.set(xticks=x, ylim=(0, 128), yticks=[0, 25, 50, 75, 100],
           ylabel="ground-truth spikes reproduced (%)",
           title="Spikes at exactly the right millisecond")
    ax.set_xticklabels(groups, fontsize=9)
    ax.grid(axis="y", alpha=0.5)
    ax.legend(loc="upper center", fontsize=8)

    ax = axes[1]
    exact_half = next(r["spikes"] for r in closed
                      if r["scheme"] == "exact" and r["precision"] == "float16")
    rasters = [(by_scheme["exact"], EXACT, 0.28), (exact_half, EXACT_HALF, 0.0),
               (by_scheme["euler"], EULER, -0.28)]
    busiest = np.argsort(-reference_spikes.sum(0))[:n_neurons]
    for row, neuron in enumerate(busiest):
        truth_t = np.nonzero(reference_spikes[window, neuron])[0] + window.start
        ax.eventplot(truth_t, lineoffsets=row, linelengths=0.84, linewidths=3.2,
                     colors=INK["truth"])
        for trace, colour, dy in rasters:
            times = np.nonzero(trace[window, neuron])[0] + window.start
            ax.eventplot(times, lineoffsets=row + dy, linelengths=0.26,
                         linewidths=1.5, colors=colour)
    ax.set(xlabel="time (ms)", yticks=[], ylabel="neuron",
           xlim=(window.start, window.stop), ylim=(-0.9, n_neurons + 0.8),
           title="A scheme is right when its tick lines up with the grey bar")
    ax.spines["left"].set_visible(False)
    handles = [mpl.lines.Line2D([], [], color=c, lw=3, label=l)
               for c, l in [(INK["truth"], "ground truth"),
                            (EXACT, "exact, float64 (top)"),
                            (EXACT_HALF, "exact, float16 (middle)"),
                            (EULER, "Euler, float64 (bottom)")]]
    ax.legend(handles=handles, loc="upper center", fontsize=7.5, ncols=4,
              bbox_to_anchor=(0.5, 1.005))
    return fig


def precision_ablation(ablation, euler_fp16):
    """What each stored block costs when it is kept in float16."""
    fig, ax = plt.subplots(figsize=(7.0, 3.4), constrained_layout=True)
    order = sorted(ablation, key=lambda a: -a["err"])
    ys = np.arange(len(order))
    errors = [a["err"] for a in order]
    floor = min(errors) / 3
    ax.hlines(ys, floor, errors, color=INK["grid"], lw=1.4)
    ax.scatter(errors, ys, s=80, color=EXACT, zorder=3,
               edgecolor=INK["surface"], lw=1.5)
    for y, a in zip(ys, order):
        note = (f"{a['bytes']} B/neuron/sample"
                + (" + shared constants" if a.get("coeffs_narrow") else ""))
        ax.text(a["err"] * 1.35, y, note, va="center",
                fontsize=8, color=INK["secondary"])
    ax.axvline(euler_fp16, color=EULER, ls="--", lw=1.6)
    ax.text(euler_fp16 * 0.85, len(order) - 0.35, "Euler,\nall float16  ",
            color=EULER, fontsize=8, ha="right", va="top")
    ax.set_yticks(ys)
    ax.set_yticklabels([f"float16: {a['symbols']}" for a in order], fontsize=8.5)
    ax.set(xscale="log", xlabel=r"relative RMS error in $V_j$",
           xlim=(floor, euler_fp16 * 22), ylim=(-0.7, len(order) - 0.3),
           title="What each stored block costs when it is kept in float16")
    ax.grid(axis="x", alpha=0.5, which="major")
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    return fig
