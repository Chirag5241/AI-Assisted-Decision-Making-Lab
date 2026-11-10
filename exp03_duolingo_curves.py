"""Experiment 3: fit the lab's learning curves phi(m) to real Duolingo learners.

Data: Duolingo's spaced-repetition release (Settles & Meeder 2016), about 13M rows,
`learning_traces.13m.csv.gz` from Harvard Dataverse (linked from github.com/duolingo/halflife-regression).
Each row is one word practised in one session:
  history_seen  times the learner had seen the word before   -> our exposure count m
  p_recall      share of that session's exercises on the word answered correctly
  delta         seconds since the word was last practised     -> forgetting

Two fits:
  1. Learning: mean recall at exposure m, for short gaps only (delta < GAP), fit as
         recall(m) = floor + (ceiling - floor) * phi(m)
     for each of simlab's curve families. Tells us which phi real learners follow.
  2. Forgetting: at fixed m, recall as a function of delta, fit as recall = 2^(-delta / half_life)
     scaled by the recall right after practice: recall = a * 2^(-delta / half_life). Gives the
     decay the memory-decay extension needs.

    python -m experiments.exp03_duolingo_curves path/to/learning_traces.13m.csv.gz
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import least_squares

from simlab import curves, plotting
from simlab.plotting import INK_2, SERIES

OUT = Path(__file__).resolve().parent.parent / "results" / "exp03"
COLS = ["p_recall", "delta", "history_seen", "learning_language"]
M_MAX = 40          # exposure counts kept for the learning fit
GAP = 86400         # "short gap": practised again within a day
DAY = 86400.0


def aggregate(path, chunksize=1_000_000, language=None):
    """Stream the file once. Returns per-m learning table and per-(m bin, delta bin) forgetting table."""
    learn = {}
    forget = {}
    dbins = np.array([0, 1 / 24, 1 / 4, 1, 2, 4, 7, 14, 30, 60, 120]) * DAY
    mbins = [(1, 2), (3, 5), (6, 10), (11, 20), (21, 40)]
    rows = 0
    for ch in pd.read_csv(path, usecols=COLS, chunksize=chunksize):
        if language:
            ch = ch[ch.learning_language == language]
        rows += len(ch)
        short = ch[(ch.delta < GAP) & (ch.history_seen <= M_MAX)]
        g = short.groupby("history_seen").p_recall.agg(["sum", "count"])
        for m, r in g.iterrows():
            s, c = learn.get(int(m), (0.0, 0))
            learn[int(m)] = (s + r["sum"], c + int(r["count"]))
        di = np.digitize(ch.delta.values, dbins) - 1
        for bi, (lo, hi) in enumerate(mbins):
            sel = (ch.history_seen.values >= lo) & (ch.history_seen.values <= hi) & (di >= 0) & (di < len(dbins) - 1)
            if not sel.any():
                continue
            df = pd.DataFrame({"d": di[sel], "p": ch.p_recall.values[sel], "t": ch.delta.values[sel]})
            gg = df.groupby("d").agg(s=("p", "sum"), c=("p", "count"), ts=("t", "sum"))
            for d, r in gg.iterrows():
                key = (bi, int(d))
                s, c, ts = forget.get(key, (0.0, 0, 0.0))
                forget[key] = (s + r.s, c + int(r.c), ts + r.ts)
    m = np.array(sorted(learn))
    L = pd.DataFrame({"m": m, "recall": [learn[k][0] / learn[k][1] for k in m], "n": [learn[k][1] for k in m]})
    F = pd.DataFrame([dict(mbin=f"{mbins[b][0]}-{mbins[b][1]}", days=v[2] / v[1] / DAY, recall=v[0] / v[1], n=v[1])
                      for (b, d), v in sorted(forget.items())])
    return L, F, rows


FAMILIES = {
    # name: (curve factory, initial guess, bounds) for the curve's own parameters
    "exponential": (lambda p: curves.exponential(p[0]), [0.1], ([1e-4], [0.999])),
    "hyperbolic": (lambda p: curves.hyperbolic(p[0]), [5.0], ([1e-3], [1e3])),
    "power law": (lambda p: curves.power_law(p[0]), [0.5], ([1e-3], [10.0])),
    "sigmoid": (lambda p: curves.sigmoid(p[0], p[1]), [5.0, 2.0], ([0.0, 0.05], [100.0, 50.0])),
}


def fit_learning(L):
    m, y, w = L.m.values.astype(float), L.recall.values, np.sqrt(L.n.values)
    out = {}
    for name, (make, p0, (lo, hi)) in FAMILIES.items():
        def resid(theta):
            floor, ceil, *p = theta
            return w * (floor + (ceil - floor) * make(p)(m) - y)
        x0 = [y.min(), y.max(), *p0]
        r = least_squares(resid, x0, bounds=([0, 0, *lo], [1, 1, *hi]))
        floor, ceil, *p = r.x
        pred = floor + (ceil - floor) * make(p)(m)
        rmse = float(np.sqrt(np.average((pred - y) ** 2, weights=L.n.values)))
        k, N = len(r.x), int(L.n.sum())
        out[name] = dict(floor=float(floor), ceiling=float(ceil), params=[float(v) for v in p], rmse=rmse,
                         weighted_sse=float((r.fun ** 2).sum()), n_params=k)
    best = min(out, key=lambda k: out[k]["weighted_sse"])
    return out, best


def fit_forgetting(F):
    out = {}
    for mb, g in F.groupby("mbin", sort=False):
        t, y, w = g.days.values, g.recall.values, np.sqrt(g.n.values)
        r = least_squares(lambda th: w * (th[1] * 2 ** (-t / th[0]) - y), [10.0, y.max()], bounds=([1e-3, 0], [1e5, 1]))
        out[mb] = dict(half_life_days=float(r.x[0]), recall_at_0=float(r.x[1]), n=int(g.n.sum()))
    return out


def figures(L, F, fits, best, forget):
    plotting.use_style()
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    ax[0].scatter(L.m, L.recall, s=np.clip(L.n / L.n.max() * 120, 8, 120), color=INK_2, alpha=0.6, label="Duolingo learners")
    mm = np.linspace(L.m.min(), L.m.max(), 200)
    for i, (name, f) in enumerate(fits.items()):
        make = FAMILIES[name][0]
        ax[0].plot(mm, f["floor"] + (f["ceiling"] - f["floor"]) * make(f["params"])(mm), color=SERIES[i],
                   lw=2.6 if name == best else 1.4, label=f"{name} (RMSE {f['rmse']:.4f})")
    ax[0].set(title="Learning: recall after m exposures (gap < 1 day)", xlabel="times seen before, m", ylabel="share recalled")
    ax[0].legend(fontsize=8)
    for i, (mb, g) in enumerate(F.groupby("mbin", sort=False)):
        ax[1].plot(g.days, g.recall, "o", color=SERIES[i], ms=4)
        tt = np.linspace(0, g.days.max(), 200)
        ax[1].plot(tt, forget[mb]["recall_at_0"] * 2 ** (-tt / forget[mb]["half_life_days"]), color=SERIES[i], label=f"m {mb}: half-life {forget[mb]['half_life_days']:.0f} d")
    ax[1].set(title="Forgetting: recall vs days since last practice", xlabel="days since last seen", ylabel="share recalled")
    ax[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "fig1_duolingo_curves.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--language", default=None, help="e.g. es, fr, de; default all")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    L, F, rows = aggregate(a.path, language=a.language)
    fits, best = fit_learning(L)
    forget = fit_forgetting(F)
    summary = dict(rows=rows, language=a.language, best_learning_curve=best, learning=fits, forgetting=forget)
    (OUT / "fit.json").write_text(json.dumps(summary, indent=1))
    L.to_csv(OUT / "learning_table.csv", index=False)
    F.to_csv(OUT / "forgetting_table.csv", index=False)
    figures(L, F, fits, best, forget)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
