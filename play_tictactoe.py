"""Teach yourself tic-tac-toe with the lab's model, and measure learning separately from following hints.

Parts, like the paper:
  1. Pre-test  : positions with no hints. Where you start.
  2. Training  : hinted positions, with an unhinted PROBE every few rounds. Probes are the only
                 training moves that show what you have learned, since there is no hint to follow.
  3. Graduation: new positions, no hints. What you learned.

Hint styles (--hints):
  presence  (default) says WHAT to look for, not where: "a move here blocks their win". You still
            have to find the square, which is the skill a beginner lacks.
  conflict  shows where two different features are, on positions where they point to DIFFERENT
            squares, so the hint cannot simply be followed: you have to know which matters more.
  after     no hint before you move. After you commit, every square's features and the best move
            are shown (commit-then-reveal), so every training move also measures learning.
  answer    the old style: names the square for each feature. Easy to follow, little to learn.

At the end you get accuracy on hinted moves, on probes and at graduation, the transfer gap
(hinted minus probe accuracy) and, for answer/conflict hints, how often you simply followed the hint.

    python -m experiments.play_tictactoe
    python -m experiments.play_tictactoe --hints conflict --train 15
    python -m experiments.play_tictactoe --hints answer --seed 4010     # compare with the old style
"""
from __future__ import annotations

import argparse
import csv
import itertools
import time
from pathlib import Path

import numpy as np

from simlab import curves
from simlab import tictactoe as ttt

OUT = Path(__file__).resolve().parent.parent / "results" / "play"
C = 2
SAYS = {
    "win": "you win right away",
    "block": "blocks their win",
    "fork": "makes two threats at once (a fork)",
    "block_fork": "takes the square where they could fork",
    "threat": "makes a two-in-a-row",
}
LOOK = {
    "win": "a move that wins right away",
    "block": "a move that blocks their win",
    "fork": "a move that makes two threats at once (a fork)",
    "block_fork": "a move that takes the square where they could fork",
    "threat": "a move that makes a two-in-a-row",
}
NAME = {"win": "WIN", "block": "BLOCK", "fork": "FORK", "block_fork": "BLOCK A FORK", "threat": "TWO-IN-A-ROW"}
RESULT = {1: "a win", 0: "a draw", -1: "a loss"}
TAC = ttt.TACTICAL


def draw(b):
    cell = lambda i: "X" if b[i] == 1 else "O" if b[i] == 2 else str(i + 1)
    rows = [f"  {cell(r)} | {cell(r + 1)} | {cell(r + 2)}" for r in (0, 3, 6)]
    return ("\n  --+---+--\n").join(rows)


def squares_with(b, f):
    return [k for k in range(9) if b[k] == 0 and ttt.square_features(b, k)[ttt.IDX[f]]]


def values(b):
    vals = {i: ttt.move_value(b, i) for i in range(9) if b[i] == 0}
    best = max(vals.values())
    return vals, best, [i for i, v in vals.items() if v == best]


def plan_lessons(T, seed):
    """The planned curriculum from Experiment 2, made for a typical beginner (the AI cannot see your beliefs)."""
    from experiments import exp02_tictactoe as e2
    e2.C = C
    pools = e2.build_pools(seed)
    sd = e2.spread(pools["train"])
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(pools["train"]), size=400, replace=False)
    lookup = {ttt.state_vector(b).tobytes(): b for b in ttt.decision_positions()}
    boards = [lookup[x.tobytes()] for x in pools["train"][idx]]
    MV = np.array([[ttt.move_value(b, k) if b[k] == 0 else -9 for k in range(9)] for b in boards], float)
    return [list(S) for S in e2.plan_curriculum(T, curves.exponential(0.05), pools["train"][idx], MV, sd)]


def conflict_pairs(b):
    """Feature pairs whose squares are non-empty and disjoint, with a best move among them."""
    _, _, best_sq = values(b)
    out = []
    for f1, f2 in itertools.combinations(TAC, 2):
        s1, s2 = set(squares_with(b, f1)), set(squares_with(b, f2))
        if s1 and s2 and not (s1 & s2) and (set(best_sq) & (s1 | s2)):
            out.append((f1, f2))
    return out


def ask_move(b):
    while True:
        try:
            raw = input("  Your move (1-9, q to quit): ").strip().lower()
        except EOFError:
            raise SystemExit("\nInput ended.")
        if raw == "q":
            raise SystemExit("Stopped. Nothing was saved.")
        if raw.isdigit() and 1 <= int(raw) <= 9 and b[int(raw) - 1] == 0:
            return int(raw) - 1
        print("  That square is taken or not 1-9. Try again.")


def why(b, k):
    f = ttt.square_features(b, k)
    tags = [SAYS[t] for t in TAC if f[ttt.IDX[t]]]
    return ", ".join(tags) if tags else "keeps your options open"


def show_hints(b, hints, style):
    """Print the hint; return the squares a pure hint-follower would pick (None if the hint names none)."""
    if style == "presence":
        print("  Look for:")
        for h in hints:
            print(f"    {LOOK[h][0].upper() + LOOK[h][1:]}: {'there is one on this board' if squares_with(b, h) else 'none on this board'}.")
        return None
    print(f"  Hints: {' + '.join(NAME[h] for h in hints)}")
    named, lines = [], []
    for h in hints:
        for k in squares_with(b, h):
            lines.append(f"    square {k + 1}: {SAYS[h]}")
            named.append(k)
    print("\n".join(lines) if lines else "    No square does that here.")
    return named or None


def reveal_after(b, k, vals, best, best_sq):
    good = vals[k] == best
    print(f"  {'Good move' if good else 'Not best'}: it leads to {RESULT[vals[k]]} with best play."
          + ("" if good else f" Best: square {' or '.join(str(i + 1) for i in best_sq)} ({RESULT[best]})."))
    print("  What each open square does:")
    for i in sorted(vals):
        print(f"    square {i + 1}: {why(b, i)}" + ("   <- best" if i in best_sq else ""))


def feedback(b, k, vals, best, best_sq):
    if vals[k] == best:
        print(f"  Good move. It keeps {RESULT[best]} with best play.")
    else:
        sq = " or ".join(str(i + 1) for i in best_sq)
        print(f"  Not best: your move leads to {RESULT[vals[k]]}; square {sq} keeps {RESULT[best]}.")
        print(f"  Square {best_sq[0] + 1}: {why(b, best_sq[0])}.")


def deciding_feature(b, best_sq):
    f = ttt.square_features(b, best_sq[0])
    for t in ["win", "block", "fork", "block_fork", "threat"]:
        if f[ttt.IDX[t]]:
            return t
    return "position"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hints", choices=["presence", "conflict", "after", "answer"], default="presence")
    ap.add_argument("--policy", choices=["planned", "turns"], default="planned", help="which features to hint (not used by --hints conflict)")
    ap.add_argument("--pre", type=int, default=6)
    ap.add_argument("--train", type=int, default=15)
    ap.add_argument("--probe-every", type=int, default=3, help="every Nth training move is an unhinted probe (0 = none)")
    ap.add_argument("--test", type=int, default=10)
    ap.add_argument("--seed", type=int, default=int(time.time()) % 10_000)
    a = ap.parse_args()

    rng = np.random.default_rng(a.seed)
    pos = ttt.decision_positions()
    order = list(rng.permutation(len(pos)))
    take = lambda n: [pos[order.pop()] for _ in range(n)]
    pre, tst = take(a.pre), take(a.test)
    probe_at = {i for i in range(a.train) if a.probe_every and (i + 1) % a.probe_every == 0}
    n_hinted = a.train - len(probe_at)

    trn, lessons = [], []
    if a.hints == "conflict":
        while len(trn) < n_hinted and order:
            b = pos[order.pop()]
            pairs = conflict_pairs(b)
            if pairs:
                trn.append(b)
                lessons.append(list(pairs[int(rng.integers(len(pairs)))]))
    else:
        trn = take(n_hinted)
        if a.policy == "planned" and a.hints != "after":
            print("Planning lessons for a typical beginner...")
            lessons = plan_lessons(n_hinted, a.seed)
        else:
            lessons = [[TAC[(C * t + i) % len(TAC)] for i in range(C)] for t in range(n_hinted)]
    probes = take(len(probe_at))

    print("\nYou are X and it is always your move. Squares are numbered 1-9, top row first.")
    print("Some positions come from games where you moved second, so O can have one more mark than you.")
    print(f"Hint style: {a.hints}. Seed {a.seed} (rerun with --seed {a.seed} --hints {a.hints} for the same positions).")
    log = []

    def play(part, b, idx, total, hints):
        print(f"\n[{part} {idx}/{total}]")
        print(draw(b))
        named = show_hints(b, hints, a.hints) if (hints and a.hints != "after") else None
        k = ask_move(b)
        vals, best, best_sq = values(b)
        if part == "training":
            if a.hints == "after":
                reveal_after(b, k, vals, best, best_sq)
            else:
                feedback(b, k, vals, best, best_sq)
        elif part == "probe":
            feedback(b, k, vals, best, best_sq)
        log.append(dict(part=part, hints=a.hints if hints else "", shown="+".join(hints), board="".join(map(str, b)),
                        move=k + 1, good=int(vals[k] == best), deciding=deciding_feature(b, best_sq),
                        followed="" if named is None else int(k in named)))

    print("\n=== Part 1: pre-test (no hints, no feedback) ===")
    for i, b in enumerate(pre):
        play("pre-test", b, i + 1, len(pre), [])
    print("\n=== Part 2: training (probes have no hint) ===")
    it_h, it_p = iter(zip(trn, lessons)), iter(probes)
    for i in range(a.train):
        if i in probe_at:
            print("\n  -- probe: no hint this time --")
            play("probe", next(it_p), i + 1, a.train, [])
        else:
            nxt = next(it_h, None)
            if nxt is None:
                continue
            b, hints = nxt
            play("training", b, i + 1, a.train, [] if a.hints == "after" else hints)
    print("\n=== Part 3: graduation (on your own) ===")
    for i, b in enumerate(tst):
        play("graduation", b, i + 1, len(tst), [])

    acc = lambda part: [r["good"] for r in log if r["part"] == part]
    pct = lambda xs: f"{100 * np.mean(xs):.0f}% ({sum(xs)}/{len(xs)})" if xs else "n/a"
    h, p = acc("training"), acc("probe")
    print("\n=== Results ===")
    print(f"  Pre-test:             {pct(acc('pre-test'))} best moves")
    print(f"  Training, {'commit-then-reveal' if a.hints == 'after' else 'with hints'}: {pct(h)}")
    print(f"  Training probes:      {pct(p)}")
    print(f"  Graduation:           {pct(acc('graduation'))}")
    if h and p and a.hints != "after":
        print(f"  Transfer gap (hinted minus probe): {100 * (np.mean(h) - np.mean(p)):+.0f} points."
              " A large gap means the hints were doing the work.")
    fol = [r["followed"] for r in log if r["followed"] != ""]
    if fol:
        print(f"  Moves on a square the hint named: {pct(fol)}")
    miss = [r["deciding"] for r in log if r["part"] == "graduation" and not r["good"]]
    if miss:
        print("  Graduation misses, by what the best move did: "
              + ", ".join(f"{NAME.get(t, t.upper())} x{miss.count(t)}" for t in sorted(set(miss), key=miss.count, reverse=True)))
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"session-{time.strftime('%Y%m%d-%H%M%S')}-{a.hints}.csv"
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(log[0]))
        w.writeheader()
        w.writerows(log)
    print(f"  Saved your session to {path}")


if __name__ == "__main__":
    main()
