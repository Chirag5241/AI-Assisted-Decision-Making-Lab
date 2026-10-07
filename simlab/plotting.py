"""Shared figure style and reusable panels."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3de"
WRONG_FILL = "#ecebe6"
NEVER = "#a3a29b"

# categorical slots, assigned in fixed order and never cycled
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
# one-hue sequential ramp, light -> dark
BLUES = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]


def use_style():
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "font.size": 10, "text.color": INK,
        "axes.edgecolor": GRID, "axes.labelcolor": INK_2, "axes.titlesize": 10.5,
        "axes.titleweight": "bold", "axes.titlelocation": "left",
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
        "xtick.color": INK_2, "ytick.color": INK_2, "xtick.major.size": 0, "ytick.major.size": 0, "xtick.minor.size": 0, "ytick.minor.size": 0,
        "lines.linewidth": 2, "legend.frameon": False, "figure.dpi": 110, "savefig.dpi": 170,
    })


def _runs(mask):
    """[start, stop) index pairs of the True runs in a 1-D bool array."""
    edges = np.flatnonzero(np.diff(np.concatenate([[False], mask, [False]]).astype(int)))
    return list(zip(edges[::2], edges[1::2]))


def trajectory_panel(ax, res, b=0, title=""):
    """Beliefs h_k(t) for a 1-feature world, the truths u_k, and the rounds the human gets wrong."""
    T = res.regret.shape[0]
    t = np.arange(T)
    K = res.U.shape[1]
    wrong = ~res.correct[:, b]
    for start, stop in _runs(wrong):
        ax.axvspan(start - 0.5, stop - 0.5, color=WRONG_FILL, lw=0)
    for k in range(K):
        ax.axhline(res.U[b, k, 0], color=SERIES[k], lw=1, ls=(0, (4, 3)))
        ax.plot(t, res.H[:T, b, k, 0], color=SERIES[k], label=f"$h_{k + 1}$  (dashed: $u_{k + 1}$)")
    ax.set_xlim(-0.5, T - 0.5)
    ax.set_title(title)
    ax.set_xlabel("round $t$")
    return wrong


def flip_time_cmap(T):
    """Discrete sequential scale for flip times; 0 = right from the start, grey = never within T."""
    bounds = [b for b in [0, 1, 2, 5, 10, 20, 50, 100] if b < T] + [T, T + 1]
    colors = ["#f6f5f1"] + BLUES[: len(bounds) - 3] + [NEVER]
    labels = ["0"] + [
        f"{lo}" if hi - lo == 1 else f"{lo}–{hi - 1}" for lo, hi in zip(bounds[1:-2], bounds[2:-1])
    ] + [f"never (≥{T})"]
    return ListedColormap(colors), BoundaryNorm(bounds, len(colors)), bounds, labels
