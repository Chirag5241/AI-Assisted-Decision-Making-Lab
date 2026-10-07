"""Experiment 1: one feature, two actions, feature always shown.

The human's beliefs h1, h2 are guaranteed to converge to u1, u2. The question is
WHEN the human's choice becomes right, i.e. when sign(h1 - h2) flips to
sign(u1 - u2), and what that costs in discounted regret.

    python -m experiments.exp01_single_feature_flip                  # all figures
    python -m experiments.exp01_single_feature_flip --u 2 0 --h0 0 1 --curve sigmoid
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

from simlab import CountLearner, analysis, curves, plotting, simulate
from simlab.plotting import INK, INK_2, SERIES

OUT = Path(__file__).resolve().parent.parent / "results" / "exp01"
DELTA = 0.95

CURVES = {
    "exponential": curves.exponential(0.1),
    "hyperbolic": curves.hyperbolic(5.0),
    "power law": curves.power_law(0.5),
    "sigmoid": curves.sigmoid(20.0, 4.0),
}
CURVE_LABELS = {
    "exponential": "exponential, rate 0.1",
    "hyperbolic": "hyperbolic, half-life 5",
    "power law": "power law, exponent 0.5",
    "sigmoid": "sigmoid, midpoint 20",
}


def col(*values):
    """K values -> a (K, 1) matrix for a one-feature world."""
    return np.asarray(values, dtype=float).reshape(-1, 1)


def describe(res, b=0):
    c = res.correct[:, [b]]
    return analysis.first_correct(c)[0], analysis.settle_time(c)[0], analysis.n_switches(c)[0]


def fmt_t(t):
    return "never" if np.isinf(t) else f"{int(t)}"


# --- figure 1: what a flip looks like ----------------------------------------

def fig_trajectories():
    exp = curves.exponential(0.1)
    scenarios = [
        ("A. Wrong start, large true gap", col(2.0, 0.0), col(0.0, 1.0), exp),
        ("B. Wrong start, small true gap", col(1.2, 1.0), col(0.0, 1.5), exp),
        ("C. Right from the start", col(2.0, 0.0), col(0.5, 0.0), exp),
        ("D. Action 1 learned 10x faster: right, wrong, right", col(1.0, 0.5), col(3.0, 2.8),
         curves.exponential(col(0.3, 0.03))),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
    print("\nFigure 1 scenarios (exponential learner)")
    print(f"  {'scenario':<52}{'du':>6}{'dh0':>7}{'first right':>13}{'settles':>9}{'switches':>10}")
    for ax, (title, U, H0, curve) in zip(axes.flat, scenarios):
        res = simulate(U, H0, CountLearner(curve), T=80)
        wrong = plotting.trajectory_panel(ax, res, title=title)
        first, settle, switches = describe(res)
        du, dh0 = (U[0] - U[1])[0], (H0[0] - H0[1])[0]
        print(f"  {title:<52}{du:>6.2f}{dh0:>7.2f}{fmt_t(first):>13}{fmt_t(settle):>9}{switches:>10}")

        ax.set_ylabel("utility weight")
        note = f"$\\Delta u$ = {du:+.1f},  $\\Delta h_0$ = {dh0:+.1f}:  "
        note += f"wrong on {int(wrong.sum())} rounds, right from $t$ = {fmt_t(settle)}" if wrong.any() else "never wrong"
        ax.set_title(title, pad=20)
        ax.annotate(note, (0, 1), xytext=(0, 5), xycoords="axes fraction", textcoords="offset points",
                    fontsize=9, color=INK_2)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    handles.append(Patch(color=plotting.WRONG_FILL))
    labels.append("rounds where the human picks the wrong action")
    axes[1, 0].legend(handles, labels, loc="center right", fontsize=9)
    fig.suptitle("Beliefs always converge; the choice is right only once $h_1 - h_2$ has the sign of $u_1 - u_2$",
                 x=0.01, ha="left", fontsize=12, fontweight="bold")
    fig.savefig(OUT / "fig1_trajectories.png")
    plt.close(fig)


# --- figure 2: flip time over the (du, dh0) plane ----------------------------

def fig_heatmaps(T=200, G=120, lim=3.0):
    # an even grid keeps du = 0 and dh0 = 0 (ties) off the lattice
    axis = np.linspace(-lim, lim, G)
    du, dh0 = np.meshgrid(axis, axis)
    U = np.stack([du.ravel() / 2, -du.ravel() / 2], axis=1)[:, :, None]
    H0 = np.stack([dh0.ravel() / 2, -dh0.ravel() / 2], axis=1)[:, :, None]
    cmap, norm, bounds, labels = plotting.flip_time_cmap(T)

    fig, axes = plt.subplots(1, 4, figsize=(14, 3.9), constrained_layout=True, sharey=True)
    print(f"\nFigure 2: flip time over a {G}x{G} grid of (du, dh0), horizon {T}")
    print(f"  {'curve':<14}{'median flip (wrong starts)':>28}{'never flips':>14}{'sim != closed form':>24}")
    for ax, (name, curve) in zip(axes, CURVES.items()):
        res = simulate(U, H0, CountLearner(curve), T=T)
        flip = analysis.first_correct(res.correct)
        theory = analysis.theory_flip_time(du.ravel(), dh0.ravel(), curve, T - 1)
        wrong_start = (du * dh0 < 0).ravel()
        # the only disagreements allowed are knife-edge worlds where h1_t == h2_t exactly at the
        # flip round, so the human's choice that round is a coin toss (off by one round)
        differ = np.flatnonzero(flip != theory)
        t_tie = np.minimum(flip, theory)[differ].astype(int)
        gap_at_tie = res.H[t_tie, differ, 0, 0] - res.H[t_tie, differ, 1, 0]
        assert np.all(np.abs(flip[differ] - theory[differ]) == 1) and np.all(np.abs(gap_at_tie) < 1e-12)
        print(f"  {name:<14}{np.median(flip[wrong_start]):>28.0f}{np.isinf(flip[wrong_start]).mean():>13.0%}"
              f"{f'{len(differ)} cells (exact ties)':>24}")

        img = np.where(np.isinf(flip), T, flip).reshape(G, G)
        mesh = ax.pcolormesh(axis, axis, img, cmap=cmap, norm=norm, shading="nearest", rasterized=True)
        ax.axhline(0, color=INK_2, lw=0.6)
        ax.axvline(0, color=INK_2, lw=0.6)
        ax.set_title(CURVE_LABELS[name])
        ax.set_xlabel("true gap  $\\Delta u = u_1 - u_2$")
        ax.set_aspect("equal")
        ax.grid(False)
    axes[0].set_ylabel("initial belief gap  $\\Delta h_0$")
    for x, y in [(0.75, 0.75), (0.25, 0.25)]:
        axes[0].text(x, y, "already\nright", transform=axes[0].transAxes, ha="center", va="center",
                     fontsize=8.5, color=INK_2)
    cbar = fig.colorbar(mesh, ax=axes, ticks=[(a + b) / 2 for a, b in zip(bounds[:-1], bounds[1:])],
                        shrink=0.85, pad=0.01)
    cbar.ax.set_yticklabels(labels)
    cbar.set_label("first round the human is right")
    cbar.outline.set_visible(False)
    fig.suptitle("Flip time depends only on the ratio $|\\Delta h_0| / |\\Delta u|$: constant along rays through the origin",
                 x=0.01, ha="left", fontsize=12, fontweight="bold")
    fig.savefig(OUT / "fig2_flip_time_heatmap.png")
    plt.close(fig)


# --- figure 3: flip time vs ratio, and what the delay costs ------------------

def fig_ratio_and_regret(T=400, n_seeds=400):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 4.6), constrained_layout=True)

    # left: flip time against r = |dh0| / |du| for a wrong-signed start
    r_line = np.logspace(-2, 2, 400)
    r_pts = np.logspace(-2, 2, 17)
    U = np.broadcast_to(col(0.5, -0.5), (len(r_pts), 2, 1))
    H0 = np.stack([-r_pts / 2, r_pts / 2], axis=1)[:, :, None]
    for (name, curve), color in zip(CURVES.items(), SERIES):
        theory = analysis.theory_flip_time(1.0, -r_line, curve, 100000)
        ax1.plot(r_line, theory, color=color, drawstyle="steps-post", label=CURVE_LABELS[name])
        sim = analysis.first_correct(simulate(U, H0, CountLearner(curve), T=T).correct)
        ax1.plot(r_pts, sim, "o", ms=6, color=color, mec=plotting.SURFACE, mew=1.2)
    ax1.axhline(T, color=INK_2, lw=0.8, ls=":")
    ax1.text(0.011, T * 1.12, f"simulation horizon ({T} rounds)", fontsize=8, color=INK_2)
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.set_ylim(0.8, 2e4)
    ax1.set_xlabel("how wrong vs how much it matters:  $|\\Delta h_0| \\,/\\, |\\Delta u|$")
    ax1.set_ylabel("flip time (rounds)")
    ax1.set_title("Flip time: lines = closed form, dots = simulation")
    ax1.legend(loc="upper left", fontsize=9)

    # right: expected discounted regret against |du| for a fixed wrong belief dh0 = -1
    du_line = np.logspace(-2, 1, 300)
    du_pts = np.logspace(-2, 1, 13)
    U = np.repeat(np.stack([du_pts / 2, -du_pts / 2], axis=1)[:, :, None], n_seeds, axis=0)
    H0 = np.broadcast_to(col(-0.5, 0.5), U.shape)
    for (name, curve), color in zip(CURVES.items(), SERIES):
        flip = analysis.theory_flip_time(du_line, -1.0, curve, 100000)
        ax2.plot(du_line, analysis.theory_discounted_regret(du_line, flip, DELTA), color=color,
                 label=CURVE_LABELS[name])
        res = simulate(U, H0, CountLearner(curve), T=T, seed=7)
        sim = res.discounted_regret(DELTA).reshape(len(du_pts), n_seeds).mean(axis=1)
        ax2.plot(du_pts, sim, "o", ms=6, color=color, mec=plotting.SURFACE, mew=1.2)
    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax2.set_xlabel("true gap  $|\\Delta u|$   (initial belief gap fixed at $\\Delta h_0 = -1$, wrong sign)")
    ax2.set_ylabel(f"expected discounted regret  ($\\delta$ = {DELTA})")
    ax2.set_title(f"Cost of the delay: lines = closed form, dots = mean of {n_seeds} runs")
    fig.savefig(OUT / "fig3_ratio_and_regret.png")
    plt.close(fig)


# --- one custom world from the command line ----------------------------------

def run_custom(u, h0, curve_name, T):
    curve = CURVES[curve_name]
    res = simulate(col(*u), col(*h0), CountLearner(curve), T=T)
    first, settle, switches = describe(res)
    print(f"u = {u}, h0 = {h0}, {CURVE_LABELS[curve_name]}")
    print(f"  first right at t = {fmt_t(first)}, right for good from t = {fmt_t(settle)}, {switches} switch(es)")
    if len(u) == 2:
        du, dh0 = u[0] - u[1], h0[0] - h0[1]
        print(f"  closed form: needs phi > {analysis.flip_threshold(du, dh0):.3f}, "
              f"flip at t = {fmt_t(analysis.theory_flip_time(du, dh0, curve, T - 1))}")
    print(f"  discounted regret (delta = {DELTA}, this draw of x): {res.discounted_regret(DELTA)[0]:.3f}")

    fig, ax = plt.subplots(figsize=(7, 4.2), constrained_layout=True)
    plotting.trajectory_panel(ax, res, title=f"u = {u}, h0 = {h0}  ({CURVE_LABELS[curve_name]})")
    ax.set_ylabel("utility weight")
    ax.legend(fontsize=9)
    path = OUT / "custom.png"
    fig.savefig(path)
    print(f"  wrote {path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--u", type=float, nargs="+", help="true weights, one per action")
    parser.add_argument("--h0", type=float, nargs="+", help="initial beliefs, one per action")
    parser.add_argument("--curve", choices=list(CURVES), default="exponential")
    parser.add_argument("--T", type=int, default=100)
    args = parser.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    plotting.use_style()
    if args.u or args.h0:
        if not (args.u and args.h0 and len(args.u) == len(args.h0)):
            parser.error("--u and --h0 must both be given, with one value per action")
        run_custom(args.u, args.h0, args.curve, args.T)
        return
    fig_trajectories()
    fig_heatmaps()
    fig_ratio_and_regret()
    print(f"\nfigures written to {OUT}")


if __name__ == "__main__":
    main()
