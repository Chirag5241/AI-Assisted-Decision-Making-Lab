"""Experiment 2: teaching a novice tic-tac-toe player, then letting her play alone.

World (simlab.tictactoe): K = 9 squares, 10 feature types per square. Geometry features
(centre, corner, edge, occupied) are always visible; tactical ones (win, block, fork, block
fork, threat) are hidden unless the AI reveals them, at most c per round (c = 2 by default).

Training: T rounds of simlab.simulate on real positions, one policy per run, CountLearner.
Graduation: on held-out positions the player looks for the c tactical features she believes
matter most (|h_f| * spread of f), then moves. We score
  - regret under the model's linear truth (the lab's loss), and
  - real mistakes: moves that lower the minimax value of the game.

    python -m experiments.exp02_tictactoe
"""
from __future__ import annotations

import itertools
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from simlab import CountLearner, curves, plotting, simulate
from simlab import tictactoe as ttt
from simlab.plotting import INK_2, SERIES

OUT = Path(__file__).resolve().parent.parent / "results" / "exp02"
B = 300                      # independent learners per run
TS = [0, 1, 2, 3, 5, 8, 12, 20, 30, 50]   # training lengths
SEED = 0
C = 2                        # features revealed per round in training, and checked per move after

U = ttt.block_matrix(ttt.W_TRUE)
H0 = ttt.block_matrix(ttt.H_NOVICE)
OBV = ttt.feature_mask(ttt.OBVIOUS)
TAC = {f: ttt.feature_mask([f]) for f in ttt.TACTICAL}
TI = [ttt.IDX[f] for f in ttt.TACTICAL]


def build_pools(seed=SEED):
    pos = ttt.decision_positions()
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(pos))
    cut = int(0.7 * len(pos))
    train = [pos[i] for i in order[:cut]]
    test = [pos[i] for i in order[cut:]]
    X = lambda ps: np.stack([ttt.state_vector(b) for b in ps])
    MV = lambda ps: np.array([[ttt.move_value(b, k) if b[k] == 0 else -9 for k in range(9)] for b in ps], float)
    return dict(train=X(train), test=X(test), mv_test=MV(test))


def spread(Xpool):
    """How much each feature type varies across legal moves: the player's sense of 'this matters'."""
    blocks = Xpool.reshape(-1, 9, ttt.NF)
    legal = blocks[:, :, ttt.IDX["occupied"]] == 0
    return blocks[legal].std(axis=0)


# ---------- policies (simlab interface: policy(t, H, x, rng) -> F (B, n), A (B, K)) ----------
class NoHints:
    name = "no hints"
    def __call__(self, t, H, x, rng):
        B_ = H.shape[0]
        return np.broadcast_to(OBV, (B_, OBV.size)), np.ones((B_, 9), bool)


def mask_of(fs):
    m = OBV.copy()
    for f in fs:
        m = m | TAC[f]
    return m


class Rotate:
    name = "take turns"
    def __call__(self, t, H, x, rng):
        B_ = H.shape[0]
        n = len(ttt.TACTICAL)
        F = mask_of([ttt.TACTICAL[(C * t + i) % n] for i in range(C)])
        return np.broadcast_to(F, (B_, F.size)), np.ones((B_, 9), bool)


class HelpNow:
    """Reveal whatever makes THIS move best (sees the position); random among ties."""
    name = "help now"
    def __call__(self, t, H, x, rng):
        B_ = H.shape[0]
        opts = [mask_of(S) for c in range(C + 1) for S in itertools.combinations(ttt.TACTICAL, c)]
        true = np.einsum("bkn,bn->bk", U[None].repeat(B_, 0), x)
        regs = []
        for m in opts:
            est = np.einsum("bkn,bn->bk", H, x * m)
            pick = est.argmax(axis=1)
            regs.append(true.max(axis=1) - true[np.arange(B_), pick])
        regs = np.stack(regs, axis=1)
        best = regs <= regs.min(axis=1, keepdims=True) + 1e-9
        choice = (best * rng.random(best.shape)).argmax(axis=1)
        F = np.stack([opts[c] for c in choice])
        return F, np.ones((B_, 9), bool)


class Schedule:
    """A fixed lesson plan chosen in advance (does not look at the position)."""
    def __init__(self, plan, name):
        self.plan, self.name = plan, name
    def __call__(self, t, H, x, rng):
        B_ = H.shape[0]
        F = mask_of(self.plan[t]) if t < len(self.plan) else OBV
        return np.broadcast_to(F, (B_, F.size)), np.ones((B_, 9), bool)


# ---------- graduation ----------
def graduate(Hb, Xtest, MV, sd):
    """Hb: (B, K, n) final beliefs. Returns per-learner linear regret, mistake rate, checked feature."""
    h = Hb[:, 0, :ttt.NF]                                   # every block holds the same weights
    checks = np.array(TI)[np.argsort(-(np.abs(h[:, TI]) * sd[TI]), axis=1, kind="stable")[:, :C]]   # (B, C)
    blocks = Xtest.reshape(-1, 9, ttt.NF)
    w = np.array([ttt.W_TRUE[f] for f in ttt.FEATURES])
    true = blocks @ w                                       # (P, 9)
    obv = np.zeros(ttt.NF, bool); obv[[ttt.IDX[f] for f in ttt.OBVIOUS]] = True
    reg, mis = np.empty(len(h)), np.empty(len(h))
    for b in range(len(h)):
        m = obv.copy(); m[list(checks[b])] = True
        pick = (blocks * m) @ h[b]
        k = pick.argmax(axis=1)
        rows = np.arange(len(k))
        reg[b] = (true.max(axis=1) - true[rows, k]).mean()
        mis[b] = (MV[rows, k] < MV.max(axis=1)).mean()
    return reg, mis, checks


def beliefs_from_counts(counts, T, curve):
    """Final H for a lesson plan: tactical feature f shown counts[f] times, obvious ones T times."""
    N = np.zeros(ttt.NF)
    for f in ttt.OBVIOUS:
        N[ttt.IDX[f]] = T
    for f, c in counts.items():
        N[ttt.IDX[f]] = c
    w = np.array([ttt.W_TRUE[f] for f in ttt.FEATURES]); h0 = np.array([ttt.H_NOVICE[f] for f in ttt.FEATURES])
    h = w + (h0 - w) * (1 - curve(N))
    return ttt.block_matrix(dict(zip(ttt.FEATURES, h)))[None]


def plan_curriculum(T, curve, Xplan, MVplan, sd):
    """Greedy lesson planner on the graduation objective, using only planning positions."""
    counts = {f: 0 for f in ttt.TACTICAL}
    plan = []
    for _ in range(T):
        scores = []
        for S in itertools.combinations(ttt.TACTICAL, C):
            c = dict(counts)
            for f in S:
                c[f] += 1
            reg, mis, _ = graduate(beliefs_from_counts(c, T, curve), Xplan, MVplan, sd)
            gap = sum(abs(ttt.W_TRUE[f] - ttt.H_NOVICE[f]) for f in S)
            scores.append((round(reg[0], 6), round(mis[0], 6), -gap, S))
        S = min(scores)[3]
        for f in S:
            counts[f] += 1
        plan.append(S)
    return plan


def run(curve, pools, Ts=TS, b=B, label=None):
    label = label or curve.name
    sd = spread(pools["train"])
    rng = np.random.default_rng(SEED)
    plan_idx = rng.choice(len(pools["train"]), size=600, replace=False)
    Xplan = pools["train"][plan_idx]
    MVplan = np.array([[ttt.move_value(_decode(x), k) if x[k * ttt.NF] == 0 else -9 for k in range(9)] for x in Xplan], float)
    sampler = lambda r, B_, n: pools["train"][r.integers(len(pools["train"]), size=B_)]
    rows = []
    for T in Ts:
        plan = plan_curriculum(T, curve, Xplan, MVplan, sd)
        pols = [NoHints(), HelpNow(), Rotate(), Schedule(plan, "planned curriculum")]
        for pol in pols:
            if T == 0:
                Hf = np.repeat(H0[None], b, 0); train_reg = np.nan
            else:
                res = simulate(U, np.repeat(H0[None], b, 0), CountLearner(curve), T, policy=pol, x_sampler=sampler, seed=SEED + T)
                Hf, train_reg = res.H[-1], res.regret.mean()
            reg, mis, checks = graduate(Hf, pools["test"], pools["mv_test"], sd)
            share = {ttt.FEATURES[i]: float((checks == i).any(axis=1).mean()) for i in TI}
            rows.append(dict(curve=label, T=T, policy=pol.name, grad_regret=float(reg.mean()),
                             grad_mistakes=float(mis.mean()), train_regret=float(train_reg),
                             checks=share, plan=[list(S) for S in plan] if pol.name == "planned curriculum" else None))
            print(f"{label:16} T={T:3} {pol.name:20} grad mistakes {mis.mean():.3f}  grad regret {reg.mean():7.2f}  train regret {train_reg:7.2f}")
    return rows


_DECODE = {}
def _decode(x):
    """Recover the board (us = 1, them = 2) from a state vector: needed only for minimax lookups."""
    key = x.tobytes()
    if key not in _DECODE:
        _DECODE[key] = _ALL.get(key)
    return _DECODE[key]


_ALL = {}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    pools = build_pools()
    for b in ttt.decision_positions():
        _ALL[ttt.state_vector(b).tobytes()] = b
    sd = spread(pools["train"])
    # reference points
    perfect = graduate(U[None], pools["test"], pools["mv_test"], sd)
    novice = graduate(H0[None], pools["test"], pools["mv_test"], sd)
    allseen_h0 = (pools["test"].reshape(-1, 9, ttt.NF) @ np.array([ttt.H_NOVICE[f] for f in ttt.FEATURES])).argmax(1)
    refs = dict(truth_optimal_share=ttt.validate_truth(ttt.decision_positions()),
                perfect_beliefs_mistakes=float(perfect[1][0]), novice_mistakes=float(novice[1][0]),
                novice_checks=[ttt.FEATURES[int(i)] for i in novice[2][0]], C=C, n_decision_positions=len(ttt.decision_positions()),
                n_test=len(pools["test"]))
    print(refs)
    rows = []
    for rate in [0.05, 0.2]:
        rows += run(curves.exponential(rate), pools, label=f"exponential {rate}")
    rows += run(curves.power_law(0.1), pools, Ts=[3, 12, 30], label="power law 0.1")
    rows += run(curves.sigmoid(midpoint=6.0, width=2.0), pools, Ts=[3, 12, 30], label="sigmoid mid 6")
    rtc = rounds_to_competence(rows, refs)
    print(json.dumps(rtc, indent=1))
    (OUT / "summary.json").write_text(json.dumps(dict(refs=refs, rounds_to_competence=rtc, rows=rows), indent=1))
    figures(rows, refs, "exponential 0.05")


def rounds_to_competence(rows, refs, tol=0.01):
    """Fewest training rounds after which graduation mistakes are within `tol` of perfect beliefs."""
    out = {}
    for r in sorted(rows, key=lambda r: r["T"]):
        key = (r["curve"], r["policy"])
        if key not in out and r["grad_mistakes"] <= refs["perfect_beliefs_mistakes"] + tol:
            out[key] = r["T"]
    return {f"{k[0]} | {k[1]}": v for k, v in out.items()}


def figures(rows, refs, curve="exponential 0.05"):
    plotting.use_style()
    pols = ["no hints", "help now", "take turns", "planned curriculum"]
    main = [r for r in rows if r["curve"] == curve]
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    for i, p in enumerate(pols):
        rs = sorted([r for r in main if r["policy"] == p], key=lambda r: r["T"])
        T = [r["T"] for r in rs]
        axes[0].plot(T, [100 * r["grad_mistakes"] for r in rs], "o-", color=SERIES[i], label=p)
        axes[1].plot(T[1:], [r["train_regret"] for r in rs][1:], "o-", color=SERIES[i], label=p)
    axes[0].axhline(100 * refs["perfect_beliefs_mistakes"], color=INK_2, ls=":", lw=1.2)
    axes[0].text(max(TS), 100 * refs["perfect_beliefs_mistakes"] + 1, "perfect beliefs", ha="right", color=INK_2, fontsize=8.5)
    axes[0].set(title="After graduation: real mistakes (% of moves)", xlabel="training rounds T", ylabel="% moves that worsen the game")
    axes[1].set(title="During training: regret per move", xlabel="training rounds T", ylabel="regret (model units)")
    axes[0].legend(fontsize=8.5)
    T20 = [r for r in main if r["T"] == 12]
    feats = list(T20[0]["checks"].keys())
    width = 0.8 / len(pols)
    for i, p in enumerate(pols):
        r = next(r for r in T20 if r["policy"] == p)
        axes[2].bar(np.arange(len(feats)) + i * width, [100 * r["checks"][f] for f in feats], width, color=SERIES[i], label=p)
    axes[2].set_xticks(np.arange(len(feats)) + 0.4 - width / 2, [f.replace("_", " ") for f in feats])
    axes[2].set(title="After 12 rounds: what she checks on her own", ylabel="% of learners who check it")
    axes[2].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "fig1_graduation.png")
    print("wrote", OUT / "fig1_graduation.png")


if __name__ == "__main__":
    main()
