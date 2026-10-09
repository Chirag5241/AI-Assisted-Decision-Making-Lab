"""The lab website: a small Flask site on top of simlab.

    python -m webapp.app          # then open http://127.0.0.1:5050

Every number on the page is computed here in Python by simlab; the browser
only collects the inputs and draws what comes back.
"""
from __future__ import annotations

import itertools
import math
import os
import re
from functools import lru_cache
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import numpy as np
from flask import Flask, abort, jsonify, render_template, request, send_from_directory, url_for

from simlab import CountLearner, ScheduledMasks, analysis, correlated, curves, draws, plotting, simulate

from . import guide

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"

# The site's name and strapline live here only; every page reads them from the template context.
SITE_NAME = "AI-Assisted Decision Making Lab"
SITE_KICKER = "CS 598 · Helping a learning human pick the best move"

MAX_ACTIONS, MAX_FEATURES, N_PROBES, MAX_WINDOWS, MAX_SEED = 6, 6, 2000, 12, 999
AWARE_STREAM = 7919    # the random fallback of the state-aware policy draws from its own stream of the seed

app = Flask(__name__)
app.config["TEMPLATES_AUTO_RELOAD"] = True   # template edits show up on reload, no restart


@app.context_processor
def _static_versioning():
    """Stamp static URLs with the file's mtime so an edited script or stylesheet is never served stale."""
    def static_v(filename):
        mtime = int((Path(app.static_folder) / filename).stat().st_mtime)
        return url_for("static", filename=filename, v=mtime)
    # the standard figures, the equations and the benches are the same lists on every page (webapp/guide.py)
    bench = next((b for b in guide.BENCHES if b["endpoint"] == request.endpoint), None)
    return dict(static_v=static_v, site_name=SITE_NAME, site_kicker=SITE_KICKER, benches=guide.BENCHES, bench=bench,
                max_seed=MAX_SEED,
                standard=guide.STANDARD, standard_figures=guide.STANDARD_FIGURES, equations=guide.EQUATIONS,
                equation=guide.EQUATION)

# One "speed" parameter per curve. Learning speed belongs to the feature, not the action: an array
# with one speed per feature gives each feature its own curve.
CURVE_SPECS = {
    "exponential": dict(
        label="Exponential", param="rate", min=0.01, max=0.5, step=0.01, default=0.1,
        hint="closes this fraction of the remaining gap every round",
        make=lambda p: curves.exponential(p)),
    "hyperbolic": dict(
        label="Hyperbolic", param="half-life", min=1, max=50, step=0.5, default=5,
        hint="rounds needed to close half of the gap",
        make=lambda p: curves.hyperbolic(p)),
    "power law": dict(
        label="Power law", param="exponent", min=0.1, max=2, step=0.05, default=0.5,
        hint="remaining gap decays as (1 + t) to minus this power",
        make=lambda p: curves.power_law(p)),
    "sigmoid": dict(
        label="Sigmoid", param="midpoint", min=2, max=80, step=1, default=20,
        hint="round of the sudden 'aha'; little is learned before it",
        make=lambda p: curves.sigmoid(p, np.asarray(p) / 5.0)),
}

PRESETS = [
    dict(name="A", title="Wrong start, large true gap",
         u1=2.0, u2=0.0, h1=0.0, h2=1.0, curve="exponential", p1=0.1),
    dict(name="B", title="Wrong start, small true gap",
         u1=1.2, u2=1.0, h1=0.0, h2=1.5, curve="exponential", p1=0.1),
    dict(name="C", title="Right from the start",
         u1=2.0, u2=0.0, h1=0.5, h2=0.0, curve="exponential", p1=0.1),
]

EXP01_FIGURES = [
    ("fig1_trajectories.png", "What a flip looks like",
     "Four worlds under an exponential learner. Beliefs always converge, but the choice is right only "
     "while h1 - h2 has the sign of u1 - u2. In D each action has its own speed, so a human who starts "
     "right goes wrong for 50 rounds before coming back."),
    ("fig2_flip_time_heatmap.png", "Flip time over every (true gap, belief gap) pair",
     "With one shared learning curve the flip time depends only on the ratio of the two gaps, so it is "
     "constant along rays through the origin. Grey cells never flip within 200 rounds."),
    ("fig3_ratio_and_regret.png", "How the learning curve sets the flip time and its cost",
     "Left: flip time against the ratio of the gaps, closed form against simulation. Right: expected "
     "discounted regret for a fixed wrong belief as the true gap grows."),
]


def _num(name, default, lo, hi):
    try:
        value = float(request.args.get(name, default))
    except (TypeError, ValueError):
        return default
    return min(max(value, lo), hi) if math.isfinite(value) else default


def _round(t):
    return None if math.isinf(t) else int(t)


def _verdict(du, dh0, wrong, settle, T):
    """Headline numeral plus one plain sentence describing what the human does. Both actions share the
    feature's one learning curve, so the belief gap crosses zero at most once: a wrong start flips once."""
    gaps = f"belief gap {dh0:+.2f} against a true gap of {du:+.2f}"
    n_wrong = int(wrong.sum())
    if du == 0:
        return "0", "no flip needed", "Both actions are worth exactly the same, so whatever the human picks is right."
    if n_wrong == 0:
        return "0", "no flip needed", f"The human starts with the right sign ({gaps}) and never picks the wrong action."
    if math.isinf(settle):
        return "never", f"within {T} rounds", (
            f"With a {gaps}, the human is still picking the wrong action at the end of the horizon. "
            "Lengthen the horizon or speed up the learner to see the flip.")
    return str(int(settle)), "round of the flip", (
        f"The human starts with the wrong sign ({gaps}), picks the wrong action for "
        f"{n_wrong} round{'s' if n_wrong != 1 else ''}, and is right from round {int(settle)} on.")


def _settle_map(learner, u_mid, h_mid, du, dh0, T, G=60):
    """Round from which the human is right for good, binned, over a G x G grid of (du, dh0) worlds.

    The grid keeps the mean levels (u1 + u2)/2 and (h1 + h2)/2 of the world on the bench; with the one
    shared learning curve only the two gaps matter. Row 0 is the largest dh0.
    """
    lim = float(np.ceil(max(3.0, 1.15 * abs(du), 1.15 * abs(dh0))))
    axis = np.linspace(-lim, lim, G)   # even G keeps the tie lines du = 0, dh0 = 0 off the lattice
    gu, gh = (g.ravel() for g in np.meshgrid(axis, axis[::-1]))
    U = np.stack([u_mid + gu / 2, u_mid - gu / 2], axis=1)[:, :, None]
    H0 = np.stack([h_mid + gh / 2, h_mid - gh / 2], axis=1)[:, :, None]
    H = learner.trajectory(U, H0, T)
    settle = analysis.settle_time((H[:, :, 0, 0] - H[:, :, 1, 0]) * gu > 0)

    _, _, bounds, labels = plotting.flip_time_cmap(T)
    labels[0] = "0 (always right)"
    bins = np.searchsorted(bounds, np.where(np.isinf(settle), T, settle), side="right") - 1
    return dict(lim=lim, G=G, bins=bins.reshape(G, G).tolist(), labels=labels)


@lru_cache(maxsize=1024)
def run_world(u1, u2, h1, h2, curve_name, speed, T, delta, seed=0):
    spec = CURVE_SPECS[curve_name]
    learner = CountLearner(spec["make"](speed))     # one feature, so one curve for both actions
    du, dh0 = u1 - u2, h1 - h2

    # everything is shown every round, so the beliefs follow the learning curve in closed form
    H = learner.trajectory(np.array([[u1], [u2]]), np.array([[h1], [h2]]), T)[:, :, 0]   # (T, 2)
    gap = H[:, 0] - H[:, 1]
    wrong = np.zeros(T, bool) if du == 0 else ~(gap * du > 0)
    correct = ~wrong[:, None]
    settle = analysis.settle_time(correct)[0]
    switches = int(analysis.n_switches(correct)[0])

    # The state of each round is drawn, x_t ~ N(0, 1), and the loss is taken there, as in the proposal:
    # regret  max_k (U x_t)_k - (U x_t)_yhat  is |du| |x_t| on a wrong round and 0 on a right one.
    x = draws.states(seed, T, 1)[:, 0]
    regret_per_round = wrong * abs(du) * np.abs(x)
    # the objective weights round t by delta^t, with delta one constant in (0, 1): the "patience" of
    # Noti et al. (2025) and the discount factor of Guan et al. (2026), as in the proposal
    weights = delta ** np.arange(T)
    discounted_per_round = weights * regret_per_round
    regret = float(discounted_per_round.sum())
    # the same loss averaged over every state instead of the one drawn: |du| E|x| per wrong round
    wrong_cost = abs(du) * analysis.MEAN_ABS_STD_NORMAL
    mean_regret_per_round = wrong * wrong_cost
    mean_regret = float((weights * mean_regret_per_round).sum())
    still_wrong = bool(wrong[-1])

    # value gap  max_k (U x_t)_k - u_hat(y_hat): the true best value minus the value the human expects
    # from the action they pick, u_hat(y_hat) = max_k h_k x_t. For x > 0 that is (max u - max h) x and
    # for x < 0 it is (min u - min h) x, so over x ~ N(0, 1) it averages to (|du| - |h1 - h2|) E|x| / 2.
    value_gap = np.maximum(u1 * x, u2 * x) - (H * x[:, None]).max(axis=1)
    mean_value_gap = (abs(du) - np.abs(gap)) * analysis.MEAN_ABS_STD_NORMAL / 2

    # the best move y* = argmax_k u_k x_t and the choice y_hat = argmax_k h_k x_t at the drawn state: both
    # swap with the sign of x_t, and they differ exactly on the wrong rounds
    best = np.array([u1, u2])[None, :] * x[:, None]
    best = best.argmax(axis=1)
    choice = (H * x[:, None]).argmax(axis=1) if du == 0 else np.where(wrong, 1 - best, best)
    if du == 0:
        best = choice                                           # equally good actions: whichever the human picks

    numeral, numeral_label, sentence = _verdict(du, dh0, wrong, settle, T)
    if du == 0:
        needed = dict(value="—", note="the actions are equally good")
    else:
        phi_star = float(analysis.flip_threshold(du, dh0))
        needed = dict(value=f"{phi_star:.0%}", note="of the belief-to-truth gap must close before the sign is right")

    edges = np.concatenate([[0], np.flatnonzero(np.diff(wrong.astype(int))) + 1, [T]])
    runs = [dict(start=int(a), stop=int(b), wrong=bool(wrong[a])) for a, b in zip(edges[:-1], edges[1:])]

    return dict(
        du=du, dh0=dh0, T=T, u=[u1, u2], delta=delta,
        numeral=numeral, numeral_label=numeral_label, sentence=sentence,
        settle=_round(settle), runs=runs,
        stats=[
            dict(label="Learning needed", **needed),
            dict(label="Rounds wrong", value=str(int(wrong.sum())), note=f"out of {T} simulated"),
            dict(label="Choice changes", value=str(switches), note="right/wrong switches"),
            dict(label="Discounted regret on this draw", value=f"{regret:.2f}",
                 note=(f"{mean_regret:.2f} averaged over states; a lower bound, still wrong at round {T}" if still_wrong
                       else f"{mean_regret:.2f} averaged over states x ~ N(0, 1)")),
        ],
        # the browser draws the figures from these, so they can follow a slider as it moves; regret,
        # discounted, value_gap, choice and best are taken at the drawn states x, the *_mean ones averaged
        series=dict(h1=np.round(H[:, 0], 4).tolist(), h2=np.round(H[:, 1], 4).tolist(), x=np.round(x, 3).tolist(),
                    regret=np.round(regret_per_round, 4).tolist(), discounted=np.round(discounted_per_round, 4).tolist(),
                    value_gap=np.round(value_gap, 4).tolist(), choice=(choice + 1).tolist(), best=(best + 1).tolist(),
                    regret_mean=np.round(mean_regret_per_round, 4).tolist(),
                    value_gap_mean=np.round(mean_value_gap, 4).tolist()),
        seed=seed,
        map=_settle_map(learner, (u1 + u2) / 2, (h1 + h2) / 2, du, dh0, T),
    )


@app.route("/overview")
def overview():
    """The figures, the equations behind them and how to read both, with a live specimen of each figure."""
    return render_template("overview.html", default_world=DEFAULT_TIMED, page="overview")


@app.route("/")
def lab():
    specs = {name: {k: v for k, v in spec.items() if k != "make"} for name, spec in CURVE_SPECS.items()}
    return render_template("lab.html", curve_specs=specs, presets=PRESETS, page="lab")


@app.route("/api/run")
def api_run():
    curve_name = request.args.get("curve", "exponential")
    if curve_name not in CURVE_SPECS:
        abort(400, "unknown curve")
    spec = CURVE_SPECS[curve_name]
    return jsonify(run_world(
        _num("u1", 1.2, -3, 3), _num("u2", 1.0, -3, 3), _num("h1", 0.0, -3, 3), _num("h2", 1.5, -3, 3),
        curve_name, _num("p1", spec["default"], spec["min"], spec["max"]),
        int(_num("T", 80, 10, 400)), _num("delta", 0.95, 0.01, 0.99), _seed(),
    ))


# ---------------------------------------------------------------------------
# Larger worlds: K actions, n features, and a choice of what is shown: the same every round
# ("More actions and features"), or with windows of rounds in which one feature or action is
# held back ("Timed hiding")
# ---------------------------------------------------------------------------

DEFAULT_WORLD = dict(
    # the human is right about features 1 and 2 and badly wrong about feature 3
    U=[[1.0, 0.5, 0.0], [0.0, 1.0, 0.5], [-0.5, 0.0, 1.0]],
    H=[[1.0, 0.5, 1.5], [0.0, 1.0, 0.0], [-0.5, 0.0, -1.5]],
    F=[1, 1, 1], A=[1, 1, 1], curve="exponential", p1=0.05, T=100, delta=0.95,
)
# the same world, with the misjudged feature held back from round 20 to round 25
DEFAULT_TIMED = dict(DEFAULT_WORLD, W="f3:20-25")
# the same world again, with at most two features shown and the algorithm choosing which for each state
DEFAULT_AWARE = dict(DEFAULT_WORLD, C=2, judge="belief", prefer="margin")


def _human_list(items):
    items = [str(i) for i in items]
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]


def _describe(mask, noun):
    chosen = [i + 1 for i, on in enumerate(mask) if on]
    if len(chosen) == len(mask):
        return f"all {len(mask)} {noun}s" if len(mask) > 1 else f"the only {noun}"
    return f"{noun}{'s' if len(chosen) > 1 else ''} {_human_list(chosen)}"


def _matrix(name, K, n):
    """Parse 'a,b;c,d' into a K x n tuple of tuples, clamped to [-3, 3] like the one-feature sliders."""
    try:
        rows = [[float(v) for v in row.split(",")] for row in request.args.get(name, "").split(";")]
    except ValueError:
        abort(400, f"{name} must be numbers")
    if len(rows) != K or any(len(row) != n for row in rows) or not all(math.isfinite(v) for r in rows for v in r):
        abort(400, f"{name} must be a {K} x {n} matrix")
    return tuple(tuple(min(max(v, -3.0), 3.0) for v in row) for row in rows)


def _mask(name, size):
    mask = tuple(v == "1" for v in request.args.get(name, ",".join("1" * size)).split(","))
    if len(mask) != size or not any(mask):
        abort(400, f"{name} must switch on at least one of {size}")
    return mask


def _seed():
    """Which draw of the states to run on."""
    return int(_num("seed", 0, 0, MAX_SEED))


def _speeds(spec, n):
    """The learning speed of each feature: ps='0.05,0.1,0.2' gives every feature its own, otherwise all
    of them share p1. Each is clamped to the curve's range."""
    raw = request.args.get("ps", "").strip()
    if not raw:
        return (_num("p1", spec["default"], spec["min"], spec["max"]),) * n
    try:
        speeds = [float(v) for v in raw.split(",")]
    except ValueError:
        abort(400, "ps must be numbers")
    if len(speeds) != n or not all(math.isfinite(v) for v in speeds):
        abort(400, f"ps must be {n} speeds, one per feature")
    return tuple(min(max(v, spec["min"]), spec["max"]) for v in speeds)


def _windows(K, n, T):
    """Parse W='f3:20-25;a1:40-60': feature 3 hidden in rounds 20 to 25 and action 1 not offered in
    rounds 40 to 60, both ends included. Returns (kind, index, first, last) cut to the horizon."""
    raw = request.args.get("W", "").strip()
    if not raw:
        return ()
    parts = raw.split(";")
    if len(parts) > MAX_WINDOWS:
        abort(400, f"at most {MAX_WINDOWS} windows")
    windows = []
    for part in parts:
        match = re.fullmatch(r"([fa])(\d{1,4}):(\d{1,4})-(\d{1,4})", part.strip())
        if not match:
            abort(400, "W must look like f3:20-25;a1:40-60")
        kind, (index, first, last) = match[1], (int(v) for v in match.groups()[1:])
        if not 1 <= index <= (n if kind == "f" else K) or first > last:
            abort(400, "a window names an existing feature or action and a first round no later than its last")
        if first < T:
            windows.append((kind, index - 1, first, min(last, T - 1)))
    return tuple(windows)


def _schedule(base, windows, kind, T):
    """(T, size) bool: the base mask every round, switched off inside each window of this kind."""
    rounds = np.tile(np.array(base, bool), (T, 1))
    for window_kind, index, first, last in windows:
        if window_kind == kind:
            rounds[first:last + 1, index] = False
    return rounds


def _window_text(windows, F_base, A_base):
    """The windows that change something, as a clause for the headline sentence."""
    live = [w for w in windows if (F_base if w[0] == "f" else A_base)[w[1]]]
    if not live:
        return ""
    if len(live) > 2:
        return f", with {len(live)} timed windows"
    return ", with " + " and ".join(
        f"{'feature' if kind == 'f' else 'action'} {index + 1} {'hidden' if kind == 'f' else 'not offered'} in "
        + (f"round {first}" if first == last else f"rounds {first}\u2013{last}")
        for kind, index, first, last in live)


def _subsets(size):
    """Every non-empty on/off mask of the given size, as a bool array (2^size - 1, size)."""
    return np.array([m for m in itertools.product([False, True], repeat=size) if any(m)])


# ---- the state-aware policy: for each round's state, a feature subset on which the best move ranks first ----
#
# A subset F works in a state x when the best move a = argmax_k (U x)_k ranks first on the shown features alone:
#     sum_{i in F} w_ai x_i  >  sum_{i in F} w_ki x_i   for every other move k.
# Its margin is the left side minus the largest right side. Two sets of weights can judge it:
#   the human's own, w = H_t: the human, as they are now, picks a from F, so the round has no error;
#   the truth, w = U: the hidden features are not needed to see that a is best.
# Judged by the human's weights (the default), the policy shows a subset on which the human is right and,
# among those, prefers one on which the truth agrees, so the pick stays right as the human learns. Judged by
# the truth alone, the human may still be wrong on what is shown until they have learned it. Among equals the
# top-ranked is the widest margin (or the most features). When no subset works a random candidate is shown.
#
# The search is exhaustive over the subsets of at most `budget` features: sum_{c <= budget} C(n, c) of them,
# 2^n - 1 when the budget is n, so exponential in the number of features. With two moves it is not needed
# (rank the features by (w_ai - w_ki) x_i and take the largest positive ones), but with more moves finding a
# subset that beats every rival at once contains hitting set, so no shortcut is known in general. Here n <= 6.

def _aware_sets(n, budget):
    """The candidate feature subsets: every non-empty one of at most `budget` features."""
    sets = _subsets(n)
    return sets[sets.sum(axis=1) <= budget]


def _aware_margin(est, true):
    """The margin of each candidate: `est` (S, K, P) are the moves' scores under it, `true` (K, P) what the
    moves are really worth. Moves that are exactly equally good are all correct, so the margin is the best
    score among the best moves minus the best score among the rest. Returns (S, P)."""
    best = (true.max(axis=0) - true) <= 1e-12                                    # (K, P)
    return np.where(best, est, -np.inf).max(axis=1) - np.where(best, -np.inf, est).max(axis=1)


def _aware_pick(margin, sets, prefer, truth_margin=None):
    """Which candidates work and which one is shown, from their margins (S, P) by the judging weights. With
    `truth_margin` (the human's weights are judging), a candidate on which the truth agrees is preferred, and
    ranked by the truth's margin. Margins closer than 1e-9 count as equal (to zero, or to each other): then
    the first candidate in order is taken, so rounding never decides. Returns works (S, P), bool, and the
    index of the top-ranked working candidate (P,)."""
    works = margin > 1e-9
    pool, key = works, margin
    if truth_margin is not None:
        both = works & (truth_margin > 1e-9)
        agreed = both.any(axis=0)                           # states with a candidate the truth agrees on
        pool, key = np.where(agreed, both, works), np.where(agreed, truth_margin, margin)
    if prefer == "most":       # the most features rather than the widest margin
        key = sets.sum(axis=1)[:, None] + np.zeros_like(margin)
    rank = np.where(pool, key, -np.inf)
    return works, (rank >= rank.max(axis=0) - 1e-9).argmax(axis=0)


def _aware_schedule(U, H0, curve, x, sets, judge, prefer, seed):
    """Run the policy on the drawn states x (T, n): in each round show the top-ranked candidate that works, by
    the human's weights of that round or by the truth, or a random candidate when none works. Every action is
    offered, so a shown feature's weights all learn. Returns the masks shown (T, n) and how many candidates
    worked in each round (T,)."""
    T, n = x.shape
    N = np.zeros(n)                                     # rounds each feature has been shown
    lots = draws.uniforms(seed + AWARE_STREAM, T)       # one 32-bit number per round, for the random fallback
    shown, worked = np.zeros((T, n), bool), np.zeros(T, int)
    for t in range(T):
        true = (U @ x[t])[:, None]
        by_truth = _aware_margin(((U[None] * sets[:, None, :]) @ x[t])[:, :, None], true)
        if judge == "truth":
            works, chosen = _aware_pick(by_truth, sets, prefer)
        else:
            H = U + (H0 - U) * (1.0 - curve(N))
            by_human = _aware_margin(((H[None] * sets[:, None, :]) @ x[t])[:, :, None], true)
            works, chosen = _aware_pick(by_human, sets, prefer, by_truth)
        worked[t] = works.sum()
        pick = int(chosen[0]) if works.any() else (int(lots[t]) * len(sets)) >> 32
        shown[t] = sets[pick]
        N += sets[pick]
    return shown, worked


def _aware_scores(W, X, sets):
    """The moves' scores under each candidate for weights W (K, n) on the states X (P, n): (S, K, P)."""
    S, (K, n) = len(sets), W.shape
    return ((sets[:, None, :] * W[None]).reshape(S * K, n) @ X.T).reshape(S, K, len(X))


def _aware_average(H, U, X, sets, judge, prefer, truth_margin):
    """The policy's accuracy, regret and value gap for beliefs H (K, n), averaged over the probe states X. In
    each state it shows the candidate it would pick there and the human chooses from it with H; where no
    candidate works a random one is shown, so the three are averaged over the candidates. `truth_margin`
    (S, P) is the candidates' margin by the truth, which is the same every round."""
    cols = np.arange(len(X))
    est = _aware_scores(H, X, sets)
    true = U @ X.T
    best_value = true.max(axis=0)
    top = est.max(axis=1)                                                   # (S, P)
    tied = est == top[:, None, :]
    count = tied.sum(axis=1)
    regret = (tied * (best_value - true)[None]).sum(axis=1) / count         # the human's regret under each candidate
    # how often the pick is a best move: a human who cannot tell several moves apart picks among them evenly
    right = (tied & ((best_value - true) <= 1e-12)[None]).sum(axis=1) / count
    gap = best_value[None] - top
    works, chosen = (_aware_pick(truth_margin, sets, prefer) if judge == "truth"
                     else _aware_pick(_aware_margin(est, true), sets, prefer, truth_margin))
    found = works.any(axis=0)
    return tuple(float(np.where(found, of[chosen, cols], of.mean(axis=0)).mean()) for of in (right, regret, gap))


@lru_cache(maxsize=512)
def run_general(U_rows, H_rows, F_now, A_now, curve_name, speeds, T, delta, full=True, windows=(), seed=0, aware=None):
    """Score the chosen policy: the features and actions switched on, minus each window's rounds.
    `speeds` holds one learning speed per feature (all equal when the features share a curve).
    `seed` picks the draw of the states x_0 .. x_{T-1}: the loss of round t is taken at x_t, as in the
    proposal, and every policy in the batch is run on the same draw. The average of that loss over all
    states is kept beside it, as a reference.
    `full` also runs every feature subset and every action subset, each kept every round, for the
    ranking tables; without it only the chosen policy and the show-everything reference run, which
    is fast enough to follow a control while it moves.
    `aware` = (budget, judge, prefer) swaps the fixed choice for the state-aware policy: every action
    offered, and for each round's state a feature subset of at most `budget` on which the best move ranks
    first, judged by the truth or by the human's weights (see `_aware_schedule`). It is ranked against the
    fixed subsets within the same budget."""
    U, H0 = np.array(U_rows), np.array(H_rows)
    K, n = U.shape
    F_base, A_base = np.array(F_now), np.array(A_now)
    F_bench, A_bench = _schedule(F_base, windows, "f", T), _schedule(A_base, windows, "a", T)   # (T, n), (T, K)
    x_draw = draws.states(seed, T, n)                   # (T, n): the state of each round, the same for every policy
    curve = CURVE_SPECS[curve_name]["make"](np.array(speeds))                     # (n,): one curve per feature
    if aware:
        budget, judge, prefer = aware
        sets = _aware_sets(n, budget)
        F_bench, worked = _aware_schedule(U, H0, curve, x_draw, sets, judge, prefer, seed)

    def every(sets):
        """Fixed subsets, the same every round: (count, size) to (T, count, size)."""
        return np.broadcast_to(sets, (T,) + sets.shape)

    def along(rounds, count=1):
        """The bench's own schedule, for `count` worlds of the batch."""
        return np.repeat(rounds[:, None, :], count, axis=1)

    if full:
        # One batch holds every candidate policy: each feature subset with the bench's actions, each
        # action subset with the bench's features, then the bench itself and the world where nothing
        # is held back.
        # (the state-aware policy is set against the fixed subsets within its budget, and no action subsets)
        feature_sets, action_sets = (sets, np.zeros((0, K), bool)) if aware else (_subsets(n), _subsets(K))
        n_f, n_a = len(feature_sets), len(action_sets)
        F = np.concatenate([every(feature_sets), along(F_bench, n_a + 1), np.ones((T, 1, n), bool)], axis=1)
        A = np.concatenate([along(A_bench, n_f), every(action_sets), along(A_bench), np.ones((T, 1, K), bool)], axis=1)
    else:
        F = np.concatenate([along(F_bench), np.ones((T, 1, n), bool)], axis=1)
        A = np.concatenate([along(A_bench), np.ones((T, 1, K), bool)], axis=1)
    B = F.shape[1]
    now, everything = B - 2, B - 1
    learner = CountLearner(curve)
    res = simulate(np.broadcast_to(U, (B, K, n)), np.broadcast_to(H0, (B, K, n)), learner, T=T,
                   policy=ScheduledMasks(F, A))

    # the learner is deterministic, so H_t is exact; score it on a fixed set of probe states
    X = np.random.default_rng(0).standard_normal((N_PROBES, n))
    per_round = [analysis.evaluate(res.H[t], U, X, F[t], A[t], value_gap=True) for t in range(T)]
    acc, reg, value_gap = (np.array(series) for series in zip(*per_round))      # each (T, B)
    # once everything shown is learned: the last round's choice, kept up until every weight in use is right
    acc_limit, reg_limit = analysis.evaluate(res.U, U, X, F[-1], A[-1])
    if aware:
        # Averaged over states, the bench is not the subset it happened to show: in another state it would
        # have shown another. Score the policy itself, with the beliefs of each round, and with the truth.
        truth_margin = _aware_margin(_aware_scores(U, X, sets), U @ X.T)
        # the ceiling the budget sets: the share of states in which some candidate settles the best move by the
        # truth; in the rest a hidden feature is needed, whatever the human believes
        ceiling = float((truth_margin > 1e-9).any(axis=0).mean())
        for t in range(T):
            acc[t, now], reg[t, now], value_gap[t, now] = _aware_average(res.H[t, now], U, X, sets, judge, prefer, truth_margin)
        acc_limit[now], reg_limit[now], _ = _aware_average(U, U, X, sets, judge, prefer, truth_margin)
    # The round's own loss, at the state drawn for it: regret l(x_t, y_hat_t), value gap, the pick and the
    # best move, each (T, B). The objective is their discounted sum on this draw; `mean_discounted` is the
    # same sum of the averages over states.
    real_reg, real_gap, real_pick, real_best = analysis.realized(res.H[:T] * F[:, :, None, :], U, x_draw[:, None, :], A)
    weights = delta ** np.arange(T)
    discounted = (weights[:, None] * real_reg).sum(axis=0)
    mean_discounted = (weights[:, None] * reg).sum(axis=0)
    # Regret axis: fixed by the truth alone (the largest regret any probe state allows), so it stays put
    # while the policy, the beliefs, the learner and the draw change.
    payoff = U @ X.T
    regret_cap = float((payoff.max(axis=0) - payoff.min(axis=0)).max())
    # Value-gap axis, fixed the same way: by the largest value any probe state is worth, at the truth or
    # at the first beliefs.
    value_cap = float(max(payoff.max(axis=0).max(), (H0 @ X.T).max(axis=0).max()))

    def ranking(offset, sets, bench, noun):
        """Rank one family of fixed subsets by discounted regret; list the best 8 and the bench. The bench
        is one of them unless its windows change this family over the rounds: then it is an extra row."""
        fixed = offset + np.arange(len(sets))
        steady = bool((bench == bench[0]).all()) and not aware      # the state-aware policy is never one of them
        here = offset + int(np.flatnonzero((sets == bench[0]).all(axis=1))[0]) if steady else now
        ids = fixed if steady else np.append(fixed, now)
        order = ids[np.argsort(discounted[ids], kind="stable")]
        rank = {int(b): i + 1 for i, b in enumerate(order)}
        listed = sorted(set(order[:8].tolist()) | {here}, key=rank.get)
        rows = []
        for b in listed:
            scheduled = b == now and not steady
            rows.append(dict(
                rank=rank[b], mask=None if scheduled else [bool(v) for v in sets[b - offset]],
                label=("the policy on the bench" if aware else "the schedule on the bench") if scheduled
                else _describe(sets[b - offset], noun),
                discounted=float(discounted[b]), mean=float(mean_discounted[b]), start=float(acc[0, b]), end=float(acc[-1, b]),
                floor=float(reg_limit[b]), current=b == here, scheduled=scheduled))
        return dict(rows=rows, total=len(ids), rank=rank[here], best=int(fixed[discounted[fixed].argmin()]))

    rankings = None
    if full:
        by_feature = ranking(0, feature_sets, F_bench, "feature")
        by_action = None if aware else ranking(n_f, action_sets, A_bench, "action")

    # What the last round holds back decides the lasting loss. Usually that is what was held back all
    # along; a window that runs to the end of the horizon leaves something out that was in use before.
    F_end, A_end = F_bench[-1], A_bench[-1]
    settled = bool((F_bench.any(axis=0) == F_end).all() and (A_bench.any(axis=0) == A_end).all())
    scheduled = bool((F_bench != F_end).any() or (A_bench != A_end).any())
    hidden = []
    if not F_end.all():
        one = (~F_end).sum() == 1
        hidden.append(_describe(~F_end, "feature") + ((" stays" if one else " stay") + " hidden" if settled
                                                       else (" is" if one else " are") + " hidden at the end"))
    if not A_end.all():
        one = (~A_end).sum() == 1
        hidden.append(_describe(~A_end, "action") + (" is" if one else " are")
                      + (" never offered" if settled else " not offered at the end"))
    sentence = (f"Showing {_describe(F_base, 'feature')} and offering {_describe(A_base, 'action')}"
                f"{_window_text(windows, F_base, A_base)}, the human picks "
                f"the best move {acc[0, now]:.0%} of the time at round 0 and {acc[-1, now]:.0%} by round {T - 1}. ")
    if reg_limit[now] > 1e-9:
        sentence += ((f"Even after learning everything shown, {reg_limit[now]:.2f} of utility is lost per round" if settled
                      else f"If the last round's choice carried on until everything shown is learned, {reg_limit[now]:.2f} "
                           "of utility would still be lost per round")
                     + (f" because {_human_list(hidden)}. " if hidden else ". "))
    elif settled:
        sentence += "Once everything shown is learned, the human always picks the best move. "
    else:
        sentence += "Once everything shown at the end is learned, the human always picks the best move. "
    if full and not aware:
        best_f, best_a = by_feature["best"], by_action["best"]
        kept = " every round" if scheduled else ""       # a fixed subset, against a schedule that changes
        feature_helps = discounted[best_f] < discounted[now] - 1e-9
        if feature_helps:
            sentence += (f"On this draw, showing {_describe(feature_sets[best_f], 'feature')}{kept} instead would cut "
                         f"the discounted regret to {discounted[best_f]:.2f}. ")
        if discounted[best_a] < discounted[now] - 1e-9:
            offer = f"ffering {_describe(action_sets[best_a - n_f], 'action')}{kept} instead would cut"
            sentence += (f"Separately, o{offer} it to {discounted[best_a]:.2f}." if feature_helps
                         else f"On this draw, o{offer} the discounted regret to {discounted[best_a]:.2f}.")
        elif not feature_helps:
            sentence += ("On this draw, no fixed choice of the features alone, or of the actions alone, does better."
                         if scheduled else "On this draw, changing only the features, or only the actions, does no better.")
        for r in (by_feature, by_action):
            del r["best"]
        rankings = dict(features=by_feature, actions=by_action)

    if aware:
        # the state-aware policy has its own sentence: how often a subset that works exists, and how the human did
        found = int((worked > 0).sum())
        right = int((real_reg[:, now] <= 1e-9).sum())               # rounds in which the human picked a best move
        most = f"at most {budget} of the {n} features" if budget < n else (f"any of the {n} features" if n > 1 else "the one feature")
        rounds = f"every one of the {T} rounds" if found == T else f"{found} of {T} rounds"
        rest = "one" if T - found == 1 else str(T - found)
        sentence = f"Showing {most}, chosen for each round's state, "
        if judge == "truth":
            sentence += (f"the algorithm found a subset on which the true weights rank the best move first in {rounds}"
                         + ("" if found == T else f"; in the other {rest} no subset within the budget does, so a random one was shown")
                         + f". The human picked the best move in {right} of the {T} rounds. ")
        elif found == T:
            sentence += (f"the algorithm found a subset that makes the human pick the best move in {rounds}, so nothing "
                         "is lost on this draw. ")
        else:
            sentence += (f"the algorithm found a subset that makes the human pick the best move in {rounds}. In the other "
                         f"{rest} no subset within the budget does, so a random one was shown; all of the regret comes "
                         "from those rounds. ")
        if full:
            best_f = by_feature["best"]
            sentence += (f"The best fixed subset within the budget, {_describe(feature_sets[best_f], 'feature')}, "
                         f"costs {discounted[best_f]:.2f} on this draw.")
            del by_feature["best"]
            rankings = dict(features=by_feature, actions=None)

    held_back = not (F_bench.all() and A_bench.all())
    stats = [
        dict(label="Averaged over states", value=f"{mean_discounted[now]:.2f}",
             note="the same discounted regret, averaged over every state instead of the ones drawn"),
        dict(label=f"Best move picked at round {T - 1}", value=f"{acc[-1, now]:.0%}",
             note=f"of states; {acc[0, now]:.0%} at round 0, heading for {acc_limit[now]:.0%} once fully learned"),
        dict(label="Regret per round once learned", value=f"{reg_limit[now]:.2f}",
             note=("on average, the lasting price of what is held back" if reg_limit[now] > 1e-9
                   else "nothing held back matters")),
        dict(label="Against showing everything", value=f"{discounted[now] - discounted[everything]:+.2f}",
             note=(f"on this draw; showing and offering everything costs {discounted[everything]:.2f}"
                   if held_back else "this is the show-everything policy")),
    ]
    if aware:
        stats[0] = dict(label="Rounds with a subset that works", value=f"{found} of {T}",
                        note=(("judged by the truth" if judge == "truth" else "judged by the human's beliefs") + "; "
                              + ("a random subset is shown in the rest" if found < T else "no round needed the random fallback")))
        stats[1] = dict(label="Averaged over states", value=f"{mean_discounted[now]:.2f}",
                        note="the discounted regret with each round's beliefs, averaged over every state instead of the one drawn")
        stats[2] = dict(label="States a subset can settle", value=f"{ceiling:.0%}",
                        note=("by the truth, some subset within the budget ranks the best move first in every state"
                              if ceiling > 1 - 1e-9 else
                              f"by the truth, with at most {budget} feature{'s' if budget > 1 else ''}; the rest need a "
                              "hidden feature, whatever the human believes"))

    # The human's choice and the best move at the state drawn in each round. A round is missed when the
    # pick costs regret there.
    picks, best_moves, missed = real_pick[:, now], real_best[:, now], real_reg[:, now] > 1e-9
    return dict(
        K=K, n=n, T=T, seed=seed, numeral=f"{discounted[now]:.2f}", numeral_label="discounted regret on this draw",
        sentence=sentence.strip(), rankings=rankings, full=full, stats=stats,
        # The browser draws the figures from these. reg, discounted and value_gap are taken at the state
        # drawn in each round; the *_mean series are their averages over all states, and acc is the share
        # of states with the best move.
        series=dict(reg=np.round(real_reg[:, now], 4).tolist(), discounted=np.round(weights * real_reg[:, now], 4).tolist(),
                    value_gap=np.round(real_gap[:, now], 4).tolist(),
                    reg_mean=np.round(reg[:, now], 4).tolist(), value_gap_mean=np.round(value_gap[:, now], 4).tolist(),
                    acc=np.round(acc[:, now], 4).tolist(),
                    ref_acc=np.round(acc[:, everything], 4).tolist() if held_back else None),
        x=np.round(x_draw, 3).tolist(),                                           # the drawn states, [round][feature]
        # Beliefs over time as [feature][action][round], the same res.H the scores above use: a weight
        # on a hidden feature, or of an action that is not offered, is not learned in that round and
        # keeps its value (the page fades those flat stretches).
        beliefs=np.round(res.H[:T, now].transpose(2, 1, 0), 3).tolist(),
        U=U.tolist(), F=F_base.tolist(), A=A_base.tolist(), delta=delta,
        # what is shown and offered round by round, as [feature][round] and [action][round], and the
        # rounds at which that changes
        schedule=dict(F=F_bench.T.astype(int).tolist(), A=A_bench.T.astype(int).tolist()),
        changes=[t for t in range(1, T) if (F_bench[t] != F_bench[t - 1]).any() or (A_bench[t] != A_bench[t - 1]).any()],
        # the state-aware policy: its budget, how many candidate subsets there are, and how many of them made
        # the human right in each round (0 means a random one was shown)
        policy=dict(budget=budget, judge=judge, prefer=prefer, candidates=len(sets), worked=worked.tolist(),
                    ceiling=ceiling) if aware else None,
        choice=dict(picks=(picks + 1).tolist(), best=(best_moves + 1).tolist(), missed=missed.tolist()),
        acc_limit=float(acc_limit[now]), reg_limit=float(reg_limit[now]),
        regret_cap=regret_cap, value_cap=value_cap,
    )


def _bench_page(page, default_world, mode):
    """One bench, three pages: `mode` is "fixed" (the same subsets every round), "timed" (windows of rounds in
    which something is held back) or "aware" (the features are chosen for each round's state)."""
    specs = {name: {k: v for k, v in spec.items() if k != "make"} for name, spec in CURVE_SPECS.items()}
    return render_template("worlds.html", curve_specs=specs, default_world=default_world, page=page, mode=mode,
                           timed=mode == "timed", aware=mode == "aware",
                           max_actions=MAX_ACTIONS, max_features=MAX_FEATURES, max_windows=MAX_WINDOWS)


@app.route("/worlds")
def worlds():
    return _bench_page("worlds", DEFAULT_WORLD, "fixed")


@app.route("/timed")
def timed():
    """The same bench, with windows of rounds in which a feature or an action is held back."""
    return _bench_page("timed", DEFAULT_TIMED, "timed")


@app.route("/aware")
def aware():
    """The same bench again, with the features chosen for each round's state so that the human picks the best move."""
    return _bench_page("aware", DEFAULT_AWARE, "aware")


@app.route("/api/world")
def api_world():
    curve_name = request.args.get("curve", "exponential")
    if curve_name not in CURVE_SPECS:
        abort(400, "unknown curve")
    spec = CURVE_SPECS[curve_name]
    K, n = int(_num("K", 3, 2, MAX_ACTIONS)), int(_num("n", 3, 1, MAX_FEATURES))
    T = int(_num("T", 100, 10, 300))
    F, A, windows, aware = _mask("F", n), _mask("A", K), _windows(K, n, T), None
    if request.args.get("policy") == "aware":
        # the state-aware policy chooses the features itself, within a budget, and offers every action
        F, A, windows = (True,) * n, (True,) * K, ()
        aware = (int(_num("C", 2, 1, n)), "truth" if request.args.get("judge") == "truth" else "belief",
                 "most" if request.args.get("prefer") == "most" else "margin")
    for base, kind, problem in ((F, "f", "show no feature"), (A, "a", "offer no action")):
        empty = np.flatnonzero(~_schedule(base, windows, kind, T).any(axis=1))
        if len(empty):
            abort(400, f"round {empty[0]} would {problem}")
    return jsonify(run_general(
        _matrix("U", K, n), _matrix("H", K, n), F, A, curve_name, _speeds(spec, n), T, _num("delta", 0.95, 0.01, 0.99),
        full=request.args.get("rank", "1") != "0", windows=windows, seed=_seed(), aware=aware,
    ))


# ---------------------------------------------------------------------------
# Correlated features (Guan et al. 2026): three features, a budget of k per round
# ---------------------------------------------------------------------------

CORR_FEATURES, CORR_SCATTER = 3, 300

DEFAULT_CORRELATED = dict(
    # Features 1 and 2 are the pair to commit to, but feature 3 moves with both and is misjudged, so it is
    # filled in wrongly forever unless it is shown first: 12 rounds of exploring halve the discounted regret.
    U=[[-1.5, 0.5, 0.0], [0.5, 1.5, -1.0], [1.0, -1.5, 0.0]],
    H=[[-1.5, -1.0, 1.5], [-0.5, 0.5, 1.5], [1.5, -1.5, -0.5]],
    rho=[0.8, 0.8, 0.6], k=2, explore=12, C=[1, 1, 0], curve="exponential", p1=0.15, T=120, delta=0.97,
)


def _explore_grid(T, cycle, size=20):
    """Exploration lengths the ranking tries: whole rotations, at most `size` of them, within the horizon."""
    step = cycle * max(1, math.ceil(T / (cycle * size)))
    return list(range(step, T, step))


@lru_cache(maxsize=256)
def run_correlated(U_rows, H_rows, rho, k, explore, commit, curve_name, speeds, T, delta, full=True, seed=0):
    """Score explore-then-commit policies under correlated features. Every policy explores with the same
    rotation, so the exploration rounds are scored once and each policy only adds its committed rounds.
    `speeds` holds one learning speed per feature. `seed` picks the draw of the states x_t ~ N(0, Sigma): the
    loss of round t is taken at x_t and every policy is run on the same draw, with the average over states beside it."""
    U, H0 = np.array(U_rows), np.array(H_rows)
    K, n = U.shape
    Sigma = correlated.correlation_matrix(*rho)
    X = correlated.probes(Sigma, N_PROBES)
    score = correlated.Scorer(U, Sigma, X)
    curve = CURVE_SPECS[curve_name]["make"](np.array(speeds))
    commit = np.array(commit)

    cycle = correlated.rotation(n, k)
    rot_masks = correlated.explore_then_commit(n, k, T, cycle[0], T + 1)
    rot_counts = correlated.exposure(rot_masks)                       # (T + 1, n): counts before round t
    rot_H = correlated.beliefs(U, H0, curve, rot_counts[:T])
    rot_scores = score(rot_H, rot_masks[:T])
    # the state drawn for each round, and the loss there while exploring: regret, value gap, pick, best move
    x_draw = draws.states(seed, T, n) @ np.linalg.cholesky(Sigma).T               # (T, n)
    rot_real = analysis.realized(score.effective(rot_H, rot_masks[:T]), U, x_draw)

    fixed = [m for m in _subsets(n) if m.sum() <= k]
    policies = [(explore, tuple(commit))] + [(0, tuple(m)) for m in fixed]
    if full:
        policies += [(e, tuple(m)) for e in _explore_grid(T, len(cycle)) for m in fixed]
    policies = list(dict.fromkeys(policies))          # the bench policy may also be on the grid; keep it first

    # every committed round of every policy, scored in one batch
    counts, masks, owner, when = [], [], [], []
    for p, (e, m) in enumerate(policies):
        e = min(e, T)
        steps = np.arange(T - e)[:, None]
        counts.append(rot_counts[e] + steps * np.array(m))
        masks.append(np.broadcast_to(m, (T - e, n)))
        owner.append(np.full(T - e, p))
        when.append(np.arange(e, T))
    counts, masks, when = np.concatenate(counts), np.concatenate(masks), np.concatenate(when)
    committed_H = correlated.beliefs(U, H0, curve, counts)
    committed = score(committed_H, masks)
    committed_real = analysis.realized(score.effective(committed_H, masks), U, x_draw[when])

    # (T, policies): the averages over states, then the same four quantities at the drawn states
    acc, reg, gap = (np.tile(series[:, None], (1, len(policies))) for series in rot_scores)
    real_reg, real_gap, real_pick, real_best = (np.tile(series[:, None], (1, len(policies))) for series in rot_real)
    owner = np.concatenate(owner)
    for p, (e, _) in enumerate(policies):
        rounds = np.arange(min(e, T), T)
        for out, series in zip((acc, reg, gap, real_reg, real_gap, real_pick, real_best), committed + committed_real):
            out[rounds, p] = series[owner == p]
    weights = delta ** np.arange(T)
    discounted = (weights[:, None] * real_reg).sum(axis=0)          # the objective, on this draw
    mean_discounted = (weights[:, None] * reg).sum(axis=0)          # the same, averaged over states

    # once the committed features are learned: they reach the truth, the others stay where exploration left them
    policy_masks = np.array([m for _, m in policies])
    frozen = correlated.beliefs(U, H0, curve, np.array([rot_counts[min(e, T)] for e, _ in policies]))
    lim_H = np.where(policy_masks[:, None, :], U, frozen)
    acc_limit, reg_limit, _ = score(lim_H, policy_masks)

    # the axes of the regret and value-gap figures, fixed by the world as on the other benches
    payoff = U @ X.T
    regret_cap = float((payoff.max(axis=0) - payoff.min(axis=0)).max())
    value_cap = float(max(payoff.max(axis=0).max(), (H0 @ X.T).max(axis=0).max()))

    def label(e, m):
        shown = _describe(np.array(m), "feature")
        return f"show {shown} every round" if e == 0 else f"explore {e} rounds, then show {shown}"

    def row(p, rank=None):
        e, m = policies[p]
        return dict(rank=rank, explore=int(e), mask=[bool(v) for v in m], label=label(e, m),
                    discounted=float(discounted[p]), mean=float(mean_discounted[p]), floor=float(reg_limit[p]),
                    start=float(acc[0, p]), end=float(acc[-1, p]), current=p == 0)

    fixed_ids = [p for p, (e, _) in enumerate(policies) if e == 0]
    best_fixed = min(fixed_ids, key=lambda p: (discounted[p], p))
    bench_is_fixed = explore == 0

    sentence = (f"Exploring for {explore} round{'s' if explore != 1 else ''} and then showing "
                f"{_describe(commit, 'feature')}" if explore else f"Showing {_describe(commit, 'feature')} every round")
    sentence += (f", the human picks the best move {acc[0, 0]:.0%} of the time at round 0 and "
                 f"{acc[-1, 0]:.0%} by round {T - 1}. ")
    hidden = ~commit
    if reg_limit[0] > 1e-9 and hidden.any():
        sentence += (f"{_describe(hidden, 'feature').capitalize()} {'is' if hidden.sum() == 1 else 'are'} filled in "
                     f"from what is shown, using beliefs that are never corrected, so {reg_limit[0]:.2f} of utility "
                     "is lost every round for good. ")
    elif reg_limit[0] <= 1e-9:
        sentence += "Once the committed features are learned, the human always picks the best move. "

    ranking = None
    if full:
        order = np.argsort(discounted, kind="stable")
        rank = {int(p): i + 1 for i, p in enumerate(order)}
        best = int(order[0])
        listed = sorted(set(order[:10].tolist()) | {0, best_fixed}, key=rank.get)
        if best != 0 and discounted[best] < discounted[0] - 1e-9:
            sentence += f"On this draw, the best policy found is to {label(*policies[best])}, at {discounted[best]:.2f}. "
        if policies[best][0] == 0:
            sentence += "On this draw, no exploration beats the best fixed subset."
        else:
            sentence += (f"Never exploring costs at least {discounted[best_fixed]:.2f}, so the best fixed subset keeps "
                         f"only {discounted[best] / discounted[best_fixed]:.0%} of that performance.")
        # discounted regret against the exploration length, one curve per committed subset
        lengths = [0] + _explore_grid(T, len(cycle))
        where = {pol: p for p, pol in enumerate(policies)}
        sweep = [dict(mask=[bool(v) for v in m], label=_describe(m, "feature"),
                      discounted=[float(discounted[where[(e, tuple(m))]]) for e in lengths]) for m in fixed]
        ranking = dict(rows=[row(p, rank[p]) for p in listed], total=len(policies), rank=rank[0],
                       best=row(best, 1), best_fixed=row(best_fixed, rank[best_fixed]),
                       retained=float(discounted[best] / discounted[best_fixed]) if discounted[best_fixed] > 1e-12 else 1.0,
                       sweep=dict(lengths=lengths, curves=sweep))

    # the bench policy, round by round
    bench_masks = correlated.explore_then_commit(n, k, explore, commit, T)
    bench_H = correlated.beliefs(U, H0, curve, correlated.exposure(bench_masks))           # (T, K, n)
    effective = lambda t: score.effective(bench_H[t], bench_masks[t])
    truth_eff = score.effective(U, commit)

    # a sample of the probe states, with the best move and the human's pick at the first and last round
    Xs = X[:CORR_SCATTER]
    best_move = (U @ Xs.T).argmax(axis=0)
    pick = lambda t: (effective(t) @ Xs.T).argmax(axis=0)

    # the human's choice and the best move at the state drawn in each round, for the policy on the bench
    picks, best_moves, missed = real_pick[:, 0], real_best[:, 0], real_reg[:, 0] > 1e-9

    ref = None if bench_is_fixed and best_fixed == 0 else best_fixed
    return dict(
        K=K, n=n, T=T, k=k, delta=delta, rho=list(rho), Sigma=Sigma.round(4).tolist(),
        seed=seed, numeral=f"{discounted[0]:.2f}", numeral_label="discounted regret on this draw",
        sentence=sentence.strip(), full=full, ranking=ranking,
        stats=[
            dict(label="Averaged over states", value=f"{mean_discounted[0]:.2f}",
                 note="the same discounted regret, averaged over every state instead of the ones drawn"),
            dict(label="Regret per round once learned", value=f"{reg_limit[0]:.2f}",
                 note=("on average, what wrongly filled-in features keep costing" if reg_limit[0] > 1e-9
                       else "nothing is lost for good")),
            dict(label="Best fixed subset", value=f"{discounted[best_fixed]:.2f}",
                 note=f"on this draw, showing {_describe(np.array(policies[best_fixed][1]), 'feature')} every round"),
            dict(label="Against the best fixed subset", value=f"{discounted[0] - discounted[best_fixed]:+.2f}",
                 note="this is the best fixed subset" if ref is None else "on this draw; negative is better"),
        ],
        # reg, discounted and value_gap are taken at the state drawn in each round; the *_mean series are their
        # averages over all states, and acc is the share of states with the best move
        series=dict(reg=np.round(real_reg[:, 0], 4).tolist(), discounted=np.round(weights * real_reg[:, 0], 4).tolist(),
                    value_gap=np.round(real_gap[:, 0], 4).tolist(),
                    reg_mean=np.round(reg[:, 0], 4).tolist(), value_gap_mean=np.round(gap[:, 0], 4).tolist(),
                    acc=np.round(acc[:, 0], 4).tolist(),
                    ref_acc=None if ref is None else np.round(acc[:, ref], 4).tolist()),
        x=np.round(x_draw, 3).tolist(),                                                   # [round][feature]
        ref_label=None if ref is None else label(*policies[ref]),
        schedule=bench_masks.astype(int).T.tolist(),                                      # [feature][round]
        beliefs=np.round(bench_H.transpose(2, 1, 0), 3).tolist(),                          # [feature][action][round]
        U=U.tolist(), C=commit.tolist(), explore=explore,
        choice=dict(picks=(picks + 1).tolist(), best=(best_moves + 1).tolist(), missed=missed.tolist()),
        effective=dict(first=np.round(effective(0), 3).tolist(), last=np.round(effective(T - 1), 3).tolist(),
                       truth=np.round(truth_eff, 3).tolist(), first_mask=bench_masks[0].tolist()),
        scatter=dict(x=np.round(Xs, 3).tolist(), best=(best_move + 1).tolist(),
                     first=(pick(0) + 1).tolist(), last=(pick(T - 1) + 1).tolist()),
        acc_limit=float(acc_limit[0]), reg_limit=float(reg_limit[0]),
        regret_cap=regret_cap, value_cap=value_cap,
    )


def _rho():
    try:
        rho = tuple(float(v) for v in request.args.get("rho", "").split(","))
    except ValueError:
        abort(400, "rho must be three numbers")
    if len(rho) != 3 or not all(math.isfinite(r) and abs(r) <= 0.95 for r in rho):
        abort(400, "rho must be three correlations in [-0.95, 0.95]")
    if not correlated.is_valid(correlated.correlation_matrix(*rho)):
        abort(400, "these three correlations cannot occur together")
    return rho


@app.route("/correlated")
def correlated_page():
    specs = {name: {k: v for k, v in spec.items() if k != "make"} for name, spec in CURVE_SPECS.items()}
    return render_template("correlated.html", curve_specs=specs, default_world=DEFAULT_CORRELATED, page="correlated",
                           max_actions=MAX_ACTIONS)


@app.route("/api/correlated")
def api_correlated():
    curve_name = request.args.get("curve", "exponential")
    if curve_name not in CURVE_SPECS:
        abort(400, "unknown curve")
    spec = CURVE_SPECS[curve_name]
    K, n = int(_num("K", 3, 2, MAX_ACTIONS)), CORR_FEATURES
    k = int(_num("k", 2, 1, n - 1))
    T = int(_num("T", 120, 10, 300))
    commit = _mask("C", n)
    if sum(commit) > k:
        abort(400, f"at most {k} features can be shown per round")
    return jsonify(run_correlated(
        _matrix("U", K, n), _matrix("H", K, n), _rho(), k, int(_num("explore", 0, 0, T)), commit, curve_name,
        _speeds(spec, n), T, _num("delta", 0.95, 0.01, 0.99),
        full=request.args.get("rank", "1") != "0", seed=_seed(),
    ))


@app.route("/experiments")
def experiments():
    shown = [dict(file=f"exp01/{name}", title=title, text=text, exists=(RESULTS / "exp01" / name).exists())
             for name, title, text in EXP01_FIGURES]
    return render_template("experiments.html", figures=shown, page="experiments")


@app.route("/results/<path:filename>")
def results(filename):
    return send_from_directory(str(RESULTS), filename)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=int(os.environ.get("PORT", 5050)), debug=False, threaded=True)
