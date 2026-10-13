"""Losses for one round, as alternatives to regret. After Jiang et al. (2014), "Diverse Randomized
Agents Vote to Win", which scores an agent on its ranking of the actions and models its noise.

`scores` (K, ...) are the human's scores of the actions on the shown features and `true` (K, ...) what
the actions are worth; the first axis is the action. Every loss is zero-or-more and lower is better.

  regret        what the best move is worth minus what the pick is worth (the lab's loss)
  wrong         1 when the pick is not the best move (the paper's objective)
  pairs         the pairs of actions the human orders wrongly (Kendall tau, the Mallows distance)
  ranked        regret at every position of the human's ranking: the best action left minus the one placed
  noisy         regret when the pick is a Plackett-Luce draw from the scores, softmax(scores / tau)
  log           minus the log of the chance that draw is the best move (cross-entropy)
"""
from __future__ import annotations

import numpy as np

NAMES = ("regret", "wrong", "pairs", "ranked", "noisy", "log")


def of_round(scores, true, tau=1.0):
    """Every loss of a round, as a dict of arrays shaped like one action's scores. A human who cannot
    tell several actions apart picks among them evenly, and a tied pair counts as half wrong; the
    ranking behind `ranked` takes tied actions in their own order."""
    s, u = np.asarray(scores, float), np.asarray(true, float)
    K = len(u)
    gap = u.max(axis=0) - u                                       # each pick's regret
    top = s == s.max(axis=0)
    share = top / top.sum(axis=0)
    soft = np.exp((s - s.max(axis=0)) / tau)
    soft = soft / soft.sum(axis=0)
    best = gap <= 1e-12
    pairs = sum(0.5 * np.abs(np.sign(s[k] - s[l]) - np.sign(u[k] - u[l])) * (u[k] != u[l])
                for k in range(K) for l in range(k + 1, K))
    placed = np.take_along_axis(u, (-s).argsort(axis=0, kind="stable"), axis=0)       # values in the human's order
    left = np.maximum.accumulate(placed[::-1], axis=0)[::-1]                          # the best value not yet placed
    return dict(regret=(share * gap).sum(axis=0), wrong=1.0 - (share * best).sum(axis=0), pairs=pairs,
                ranked=(left - placed).sum(axis=0), noisy=(soft * gap).sum(axis=0),
                log=-np.log((soft * best).sum(axis=0)))
