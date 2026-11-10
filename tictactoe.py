"""Tic-tac-toe as a world for the lab's model.

A position is seen from the side to move ("us" = 1, "them" = 2). The K = 9 actions are the
squares. Each square k gets a feature vector f(k) of length NF describing what playing there
does; the state x stacks the nine vectors, x = [f(0), ..., f(8)], so x has n = 9 * NF entries.
U and H are block-diagonal: row k holds the weights w (or the human's h) in block k and zeros
elsewhere, so u_k . x = w . f(k), which is exactly simlab's "value of action k = u_k . x".

Showing feature type j means unmasking entry j in every block. Because every action is offered
every round, all nine blocks are always updated together and the human keeps one shared weight
per feature type, as a person would.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np

LINES = [(0, 1, 2), (3, 4, 5), (6, 7, 8), (0, 3, 6), (1, 4, 7), (2, 5, 8), (0, 4, 8), (2, 4, 6)]
CORNERS, EDGES, CENTER = (0, 2, 6, 8), (1, 3, 5, 7), 4
OPPOSITE = {0: 8, 2: 6, 6: 2, 8: 0}

# feature types, in block order
FEATURES = ["occupied", "win", "block", "fork", "block_fork", "threat", "center", "corner", "opp_corner", "edge"]
NF = len(FEATURES)
IDX = {f: i for i, f in enumerate(FEATURES)}
# features anyone sees at a glance (always shown, the s = Bx part) vs. ones that need looking for
OBVIOUS = ["occupied", "center", "corner", "opp_corner", "edge"]
TACTICAL = ["win", "block", "fork", "block_fork", "threat"]

# truth: the Newell-Simon priority order (win > block > fork > block fork > centre > opposite
# corner > corner > edge) written as weights; validated against minimax in `validate_truth`
W_TRUE = {"occupied": -1000.0, "win": 100.0, "block": 50.0, "fork": 20.0, "block_fork": 10.0,
          "threat": 0.5, "center": 4.0, "corner": 1.5, "opp_corner": 2.0, "edge": 1.0}
# a novice: likes the centre and making lines, half-sees her own wins, ignores the opponent's
H_NOVICE = {"occupied": -1000.0, "win": 3.0, "block": 0.5, "fork": 0.0, "block_fork": 0.0,
            "threat": 3.0, "center": 6.0, "corner": 2.0, "opp_corner": 0.0, "edge": 1.0}


def winner(b):
    for a, c, d in LINES:
        if b[a] and b[a] == b[c] == b[d]:
            return b[a]
    return 0


@lru_cache(maxsize=None)
def value(b):
    """Minimax value for the side to move (1): +1 win, 0 draw, -1 loss. `b` is a tuple of 9."""
    w = winner(b)
    if w:
        return 1 if w == 1 else -1
    empties = [i for i in range(9) if b[i] == 0]
    if not empties:
        return 0
    return max(-value(_swap(_play(b, i, 1))) for i in empties)


def move_value(b, k):
    """Minimax value of playing k, for the side to move."""
    nb = _play(b, k, 1)
    if winner(nb) == 1:
        return 1
    if 0 not in nb:
        return 0
    return -value(_swap(nb))


def _play(b, i, p):
    return b[:i] + (p,) + b[i + 1:]


def _swap(b):
    return tuple(0 if v == 0 else 3 - v for v in b)


def _threats(b, p):
    """Lines with two of p's marks and an empty third square."""
    return sum(1 for ln in LINES if sum(b[i] == p for i in ln) == 2 and sum(b[i] == 0 for i in ln) == 1)


def square_features(b, k):
    f = np.zeros(NF)
    if b[k] != 0:
        f[IDX["occupied"]] = 1.0
        return f
    mine, theirs = _play(b, k, 1), _play(b, k, 2)
    f[IDX["win"]] = float(winner(mine) == 1)
    f[IDX["block"]] = float(winner(theirs) == 2)
    new = _threats(mine, 1) - _threats(b, 1)
    f[IDX["threat"]] = float(max(new, 0))
    f[IDX["fork"]] = float(_threats(mine, 1) >= 2 and not f[IDX["win"]])
    f[IDX["block_fork"]] = float(_threats(theirs, 2) >= 2 and not f[IDX["block"]])
    f[IDX["center"]] = float(k == CENTER)
    f[IDX["corner"]] = float(k in CORNERS)
    f[IDX["opp_corner"]] = float(k in CORNERS and b[OPPOSITE[k]] == 2)
    f[IDX["edge"]] = float(k in EDGES)
    return f


def state_vector(b):
    return np.concatenate([square_features(b, k) for k in range(9)])


def block_matrix(weights: dict):
    w = np.array([weights[f] for f in FEATURES])
    M = np.zeros((9, 9 * NF))
    for k in range(9):
        M[k, k * NF:(k + 1) * NF] = w
    return M


def feature_mask(types):
    """(9 * NF,) bool mask that shows the given feature types in every block."""
    m = np.zeros(NF, dtype=bool)
    m[[IDX[t] for t in types]] = True
    return np.tile(m, 9)


def reachable_positions():
    """Every non-terminal position reachable in real play, seen from the side to move."""
    seen, out = set(), []

    def walk(b, p):
        key = b if p == 1 else _swap(b)
        if key in seen:
            return
        seen.add(key)
        if winner(b) or 0 not in b:
            return
        out.append(key)
        for i in range(9):
            if b[i] == 0:
                walk(_play(b, i, p), 3 - p)

    walk((0,) * 9, 1)
    return out


def decision_positions():
    """Positions where the choice matters: not every legal move has the same minimax value."""
    keep = []
    for b in reachable_positions():
        vals = {move_value(b, k) for k in range(9) if b[k] == 0}
        if len(vals) > 1:
            keep.append(b)
    return keep


def validate_truth(positions, weights=W_TRUE):
    """Share of positions where argmax of the linear truth is a minimax-optimal move."""
    U = block_matrix(weights)
    ok = 0
    for b in positions:
        k = int(np.argmax(U @ state_vector(b)))
        best = max(move_value(b, i) for i in range(9) if b[i] == 0)
        ok += move_value(b, k) == best
    return ok / len(positions)
