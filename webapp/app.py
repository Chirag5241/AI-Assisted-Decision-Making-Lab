"""The lab website: a small Flask site on top of simlab.

    python -m webapp.app          # then open http://127.0.0.1:5050

Every number on the page is computed here in Python by simlab; the browser
only collects the inputs and draws what comes back.
"""
from __future__ import annotations

import itertools
import math
import os
from functools import lru_cache
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import numpy as np
from flask import Flask, abort, jsonify, render_template, request, send_from_directory, url_for

from simlab import CountLearner, FixedMasks, analysis, curves, plotting, simulate

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"

# The site's name and strapline live here only; every page reads them from the template context.
SITE_NAME = "AI-Assisted Decision Making Lab"
SITE_KICKER = "CS 598 · Helping a learning human pick the best move"

MAX_ACTIONS, MAX_FEATURES, N_PROBES = 6, 6, 2000

app = Flask(__name__)
app.config["TEMPLATES_AUTO_RELOAD"] = True   # template edits show up on reload, no restart


@app.context_processor
def _static_versioning():
    """Stamp static URLs with the file's mtime so an edited script or stylesheet is never served stale."""
    def static_v(filename):
        mtime = int((Path(app.static_folder) / filename).stat().st_mtime)
        return url_for("static", filename=filename, v=mtime)
    return dict(static_v=static_v, site_name=SITE_NAME, site_kicker=SITE_KICKER)

# One "speed" parameter per curve; a (2, 1) array gives each action its own speed.
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
         u1=2.0, u2=0.0, h1=0.0, h2=1.0, curve="exponential", p1=0.1, p2=0.1, split=0),
    dict(name="B", title="Wrong start, small true gap",
         u1=1.2, u2=1.0, h1=0.0, h2=1.5, curve="exponential", p1=0.1, p2=0.1, split=0),
    dict(name="C", title="Right from the start",
         u1=2.0, u2=0.0, h1=0.5, h2=0.0, curve="exponential", p1=0.1, p2=0.1, split=0),
    dict(name="D", title="Action 1 learned 10x faster: right, wrong, right",
         u1=1.0, u2=0.5, h1=3.0, h2=2.8, curve="exponential", p1=0.3, p2=0.03, split=1),
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


def _verdict(du, dh0, wrong, settle, switches, T):
    """Headline numeral plus one plain sentence describing what the human does."""
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
    if switches <= 1:
        return str(int(settle)), "round of the flip", (
            f"The human starts with the wrong sign ({gaps}), picks the wrong action for "
            f"{n_wrong} round{'s' if n_wrong != 1 else ''}, and is right from round {int(settle)} on.")
    first_wrong = int(np.flatnonzero(wrong)[0])
    return str(int(settle)), "round of the final flip", (
        f"The choice changes {switches} times. The human {'starts right, ' if not wrong[0] else ''}"
        f"goes wrong at round {first_wrong}, is wrong on {n_wrong} rounds in total, "
        f"and is right for good from round {int(settle)}.")


def _settle_map(learner, u_mid, h_mid, du, dh0, T, G=60):
    """Round from which the human is right for good, binned, over a G x G grid of (du, dh0) worlds.

    The grid keeps the mean levels (u1 + u2)/2 and (h1 + h2)/2 of the world on the bench, so with
    per-action learning speeds the map is specific to those levels. Row 0 is the largest dh0.
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
def run_world(u1, u2, h1, h2, curve_name, p1, p2, split, T, delta):
    spec = CURVE_SPECS[curve_name]
    speed = np.array([[p1], [p2]]) if split else p1
    learner = CountLearner(spec["make"](speed))
    du, dh0 = u1 - u2, h1 - h2

    # everything is shown every round, so the beliefs follow the learning curve in closed form
    H = learner.trajectory(np.array([[u1], [u2]]), np.array([[h1], [h2]]), T)[:, :, 0]   # (T, 2)
    gap = H[:, 0] - H[:, 1]
    wrong = np.zeros(T, bool) if du == 0 else ~(gap * du > 0)
    correct = ~wrong[:, None]
    settle = analysis.settle_time(correct)[0]
    switches = int(analysis.n_switches(correct)[0])

    # regret loss  max_k (U x)_k - (U x)_yhat  is |du| |x| on a wrong round and 0 on a right one;
    # averaged over states x ~ N(0, 1) that is |du| E|x| per wrong round
    wrong_cost = abs(du) * analysis.MEAN_ABS_STD_NORMAL
    regret_per_round = wrong * wrong_cost
    # the objective weights round t by delta^t, with delta one constant in (0, 1): the "patience" of
    # Noti et al. (2025) and the discount factor of Guan et al. (2026), as in the proposal
    discounted_per_round = delta ** np.arange(T) * regret_per_round
    regret = float(discounted_per_round.sum())
    still_wrong = bool(wrong[-1])

    # value gap  max_k (U x)_k - u_hat(y_hat): the true best value minus the value the human expects
    # from the action they pick, u_hat(y_hat) = max_k h_k x. For x > 0 that is (max u - max h) x and
    # for x < 0 it is (min u - min h) x, so over x ~ N(0, 1) it averages to (|du| - |h1 - h2|) E|x| / 2.
    value_gap = (abs(du) - np.abs(gap)) * analysis.MEAN_ABS_STD_NORMAL / 2

    # the choice y_hat = argmax_k h_k x for a positive state (for x < 0 both it and the best move swap)
    best = None if du == 0 else (0 if du > 0 else 1)
    choice = H.argmax(axis=1) if best is None else np.where(wrong, 1 - best, best)

    numeral, numeral_label, sentence = _verdict(du, dh0, wrong, settle, switches, T)
    if split:
        needed = dict(value="varies", note="each action has its own curve, so no single threshold")
    elif du == 0:
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
            dict(label="Choice changes", value=str(switches),
                 note="more than one means the choice un-flipped" if switches > 1 else "right/wrong switches"),
            dict(label="Expected discounted regret", value=f"{regret:.2f}",
                 note=(f"lower bound: still wrong at round {T}" if still_wrong
                       else f"δ = {delta:g}, states x ~ N(0, 1)")),
        ],
        # the browser draws the figures from these, so they can follow a slider as it moves
        series=dict(h1=np.round(H[:, 0], 4).tolist(), h2=np.round(H[:, 1], 4).tolist(),
                    regret=np.round(regret_per_round, 4).tolist(), choice=(choice + 1).tolist(),
                    discounted=np.round(discounted_per_round, 4).tolist(),
                    value_gap=np.round(value_gap, 4).tolist()),
        wrong_cost=wrong_cost, best=None if best is None else best + 1,
        map=_settle_map(learner, (u1 + u2) / 2, (h1 + h2) / 2, du, dh0, T),
    )


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
    split = request.args.get("split", "0") == "1"
    p1 = _num("p1", spec["default"], spec["min"], spec["max"])
    p2 = _num("p2", spec["default"], spec["min"], spec["max"]) if split else p1
    split = split and p1 != p2  # equal speeds are one shared curve, so the closed form applies
    return jsonify(run_world(
        _num("u1", 1.2, -3, 3), _num("u2", 1.0, -3, 3), _num("h1", 0.0, -3, 3), _num("h2", 1.5, -3, 3),
        curve_name, p1, p2, split, int(_num("T", 80, 10, 400)), _num("delta", 0.95, 0.01, 0.99),
    ))


# ---------------------------------------------------------------------------
# Larger worlds: K actions, n features, and a fixed choice of what is shown
# ---------------------------------------------------------------------------

DEFAULT_WORLD = dict(
    # the human is right about features 1 and 2 and badly wrong about feature 3
    U=[[1.0, 0.5, 0.0], [0.0, 1.0, 0.5], [-0.5, 0.0, 1.0]],
    H=[[1.0, 0.5, 1.5], [0.0, 1.0, 0.0], [-0.5, 0.0, -1.5]],
    F=[1, 1, 1], A=[1, 1, 1], curve="exponential", p1=0.05, T=100, delta=0.95,
)


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


def _subsets(size):
    """Every non-empty on/off mask of the given size, as a bool array (2^size - 1, size)."""
    return np.array([m for m in itertools.product([False, True], repeat=size) if any(m)])


@lru_cache(maxsize=512)
def run_general(U_rows, H_rows, F_now, A_now, curve_name, speed, T, delta, full=True):
    """Score the chosen policy. `full` also runs every feature subset and every action subset for
    the ranking tables; without it only the chosen policy and the show-everything reference run,
    which is fast enough to follow a control while it moves."""
    U, H0 = np.array(U_rows), np.array(H_rows)
    K, n = U.shape
    F_now, A_now = np.array(F_now), np.array(A_now)

    if full:
        # One batch holds every candidate policy: each feature subset with the chosen actions, each
        # action subset with the chosen features, and last the world where nothing is held back.
        feature_sets, action_sets = _subsets(n), _subsets(K)
        n_f, n_a = len(feature_sets), len(action_sets)
        F = np.vstack([feature_sets, np.tile(F_now, (n_a, 1)), np.ones((1, n), bool)])
        A = np.vstack([np.tile(A_now, (n_f, 1)), action_sets, np.ones((1, K), bool)])
    else:
        F = np.vstack([F_now, np.ones(n, bool)])
        A = np.vstack([A_now, np.ones(K, bool)])
    B = len(F)
    learner = CountLearner(CURVE_SPECS[curve_name]["make"](speed))
    res = simulate(np.broadcast_to(U, (B, K, n)), np.broadcast_to(H0, (B, K, n)), learner, T=T,
                   policy=FixedMasks(F, A))

    # the learner is deterministic, so H_t is exact; score it on a fixed set of probe states
    X = np.random.default_rng(0).standard_normal((N_PROBES, n))
    per_round = [analysis.evaluate(res.H[t], U, X, F, A, value_gap=True) for t in range(T)]
    acc, reg, value_gap = (np.array(series) for series in zip(*per_round))      # each (T, B)
    acc_limit, reg_limit = analysis.evaluate(res.U, U, X, F, A)   # once everything shown is learned
    discounted = ((delta ** np.arange(T))[:, None] * reg).sum(axis=0)
    # Regret axis: fixed by the truth alone (the expected regret of always picking the worst action),
    # so it stays put while subsets, beliefs and the learner change.
    payoff = U @ X.T
    worst_regret = float((payoff.max(axis=0) - payoff.min(axis=0)).mean())
    # Value-gap axis, fixed the same way: the gap lies between minus the largest value the human can
    # expect (never more than with everything shown, at the first beliefs or at the truth) and the
    # true best value.
    value_bound = float(max(payoff.max(axis=0).mean(), (H0 @ X.T).max(axis=0).mean()))

    def ranking(offset, sets, chosen, noun):
        """Rank one family of subsets by discounted regret; list the best 8 and the current one."""
        ids = offset + np.arange(len(sets))
        order = ids[np.argsort(discounted[ids], kind="stable")]
        rank = {int(b): i + 1 for i, b in enumerate(order)}
        now = offset + int(np.flatnonzero((sets == chosen).all(axis=1))[0])
        listed = sorted(set(order[:8].tolist()) | {now}, key=rank.get)
        rows = [dict(rank=rank[b], mask=[bool(v) for v in sets[b - offset]],
                     label=_describe(sets[b - offset], noun), discounted=float(discounted[b]),
                     start=float(acc[0, b]), end=float(acc[-1, b]), floor=float(reg_limit[b]), current=b == now)
                for b in listed]
        return dict(rows=rows, total=len(sets), rank=rank[now], best=int(order[0]), now=now)

    now, everything, rankings = 0, B - 1, None
    if full:
        by_feature = ranking(0, feature_sets, F_now, "feature")
        by_action = ranking(n_f, action_sets, A_now, "action")
        now = by_feature["now"]

    hidden = []
    if not F_now.all():
        hidden.append(_describe(~F_now, "feature") + (" stays" if (~F_now).sum() == 1 else " stay") + " hidden")
    if not A_now.all():
        hidden.append(_describe(~A_now, "action") + (" is" if (~A_now).sum() == 1 else " are") + " never offered")
    sentence = (f"Showing {_describe(F_now, 'feature')} and offering {_describe(A_now, 'action')}, the human picks "
                f"the best move {acc[0, now]:.0%} of the time at round 0 and {acc[-1, now]:.0%} by round {T - 1}. ")
    if reg_limit[now] > 1e-9:
        sentence += (f"Even after learning everything shown, {reg_limit[now]:.2f} of utility is lost per round"
                     + (f" because {_human_list(hidden)}. " if hidden else ". "))
    else:
        sentence += "Once everything shown is learned, the human always picks the best move. "
    if full:
        best_f, best_a = by_feature["best"], by_action["best"]
        feature_helps = discounted[best_f] < discounted[now] - 1e-9
        if feature_helps:
            sentence += (f"Showing {_describe(feature_sets[best_f], 'feature')} instead would cut the discounted "
                         f"regret to {discounted[best_f]:.2f}. ")
        if discounted[best_a] < discounted[now] - 1e-9:
            offer = f"ffering {_describe(action_sets[best_a - n_f], 'action')} instead would cut"
            sentence += (f"Separately, o{offer} it to {discounted[best_a]:.2f}." if feature_helps
                         else f"O{offer} the discounted regret to {discounted[best_a]:.2f}.")
        elif not feature_helps:
            sentence += "Changing only the features, or only the actions, does no better over this horizon."
        for r in (by_feature, by_action):
            del r["best"], r["now"]
        rankings = dict(features=by_feature, actions=by_action)

    held_back = not (F_now.all() and A_now.all())

    # The human's choice for one reference state, every feature equal to +1 (the "positive state" of
    # the one-feature bench): y_hat = argmax over offered actions of the shown beliefs times x, against
    # the best move y* = argmax over all actions of U x. Ties go to the lowest-numbered action.
    x_ref = np.ones(n)
    scores = (res.H[:T, now] * F_now) @ x_ref                          # (T, K)
    picks = np.where(A_now, scores, -np.inf).argmax(axis=1)
    worth = U @ x_ref
    best_move = int(worth.argmax())
    missed = worth[picks] < worth.max() - 1e-9      # a pick that ties with the best move is not a miss
    return dict(
        K=K, n=n, T=T, numeral=f"{discounted[now]:.2f}", numeral_label="expected discounted regret",
        sentence=sentence.strip(), rankings=rankings, full=full,
        stats=[
            dict(label="Best move picked at round 0", value=f"{acc[0, now]:.0%}",
                 note="with the human's first beliefs"),
            dict(label=f"Best move picked at round {T - 1}", value=f"{acc[-1, now]:.0%}",
                 note=f"heading for {acc_limit[now]:.0%} once fully learned"),
            dict(label="Regret per round once learned", value=f"{reg_limit[now]:.2f}",
                 note="the lasting price of what is held back" if reg_limit[now] > 1e-9 else "nothing held back matters"),
            dict(label="Against showing everything", value=f"{discounted[now] - discounted[everything]:+.2f}",
                 note=(f"discounted regret; showing and offering everything costs {discounted[everything]:.2f}"
                       if held_back else "this is the show-everything policy")),
        ],
        # the browser draws the two figures from these
        series=dict(acc=np.round(acc[:, now], 4).tolist(), reg=np.round(reg[:, now], 4).tolist(),
                    discounted=np.round(delta ** np.arange(T) * reg[:, now], 4).tolist(),
                    value_gap=np.round(value_gap[:, now], 4).tolist(),
                    ref_acc=np.round(acc[:, everything], 4).tolist() if held_back else None),
        # Beliefs over time as [feature][action][round], drawn as the learning curve of every weight.
        # A hidden feature or an action that is not offered is never actually learned (the scores above
        # use res.H, where it stays put); the page fades those lines instead of flattening them.
        beliefs=np.round(learner.trajectory(U, H0, T).transpose(2, 1, 0), 3).tolist(),
        U=U.tolist(), F=F_now.tolist(), A=A_now.tolist(), delta=delta,
        choice=dict(x=x_ref.tolist(), picks=(picks + 1).tolist(), best=best_move + 1, missed=missed.tolist()),
        acc_limit=float(acc_limit[now]), reg_limit=float(reg_limit[now]),
        worst_regret=worst_regret, value_bound=value_bound,
    )


@app.route("/worlds")
def worlds():
    specs = {name: {k: v for k, v in spec.items() if k != "make"} for name, spec in CURVE_SPECS.items()}
    return render_template("worlds.html", curve_specs=specs, default_world=DEFAULT_WORLD, page="worlds",
                           max_actions=MAX_ACTIONS, max_features=MAX_FEATURES)


@app.route("/api/world")
def api_world():
    curve_name = request.args.get("curve", "exponential")
    if curve_name not in CURVE_SPECS:
        abort(400, "unknown curve")
    spec = CURVE_SPECS[curve_name]
    K, n = int(_num("K", 3, 2, MAX_ACTIONS)), int(_num("n", 3, 1, MAX_FEATURES))
    return jsonify(run_general(
        _matrix("U", K, n), _matrix("H", K, n), _mask("F", n), _mask("A", K), curve_name,
        _num("p1", spec["default"], spec["min"], spec["max"]),
        int(_num("T", 100, 10, 300)), _num("delta", 0.95, 0.01, 0.99),
        full=request.args.get("rank", "1") != "0",
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
