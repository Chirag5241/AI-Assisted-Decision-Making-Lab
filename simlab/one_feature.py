"""One feature a round, two actions (the one-feature solution).

The algorithm names one feature j before the state is drawn and the human picks action 1 when g_j x_j > 0,
using that feature alone. With x ~ N(0, Sigma) and d = u_1 - u_2 the expected regret of the round is

    (sqrt(d' Sigma d) - sgn(g_j) (Sigma d)_j) / sqrt(2 pi),

so only the sign of the belief counts. For Sigma = I this is the solution's Proposition 1: each feature's cost
falls with its own count, and showing one feature forever is optimal, the one with the largest score
s_j = |d_j| (2 delta^tau_j - 1), tau_j the showings before g_j takes the sign of d_j.

With correlated features the good sign is that of (Sigma d)_j, and learning drives g_j to the sign of d_j. When
the two disagree the feature costs more the more it is shown, the costs no longer fall, and the best schedule
can switch features. `optimal` searches every schedule, so it holds either way.
"""
from __future__ import annotations

import itertools

import numpy as np

from .correlated import imputation

ROOT_2PI = np.sqrt(2.0 * np.pi)
TIE = 1e-9          # values closer than this count as equal: the lower-numbered feature is kept


def believed(d, g0, psi):
    """(n, M) the believed gap of each feature after m = 0 .. M-1 showings: d_j + psi(m) (g0_j - d_j)."""
    d, g0 = np.asarray(d, float), np.asarray(g0, float)
    return d[:, None] + np.asarray(psi, float)[None, :] * (g0 - d)[:, None]


def costs(d, g0, psi, Sigma=None):
    """(n, M) the expected regret of a round that shows feature j after it has been shown m times."""
    d = np.asarray(d, float)
    Sigma = np.eye(len(d)) if Sigma is None else np.asarray(Sigma, float)
    reach = Sigma @ d                                        # (Sigma d)_j = E[x_j d.x]
    return (np.sqrt(d @ Sigma @ d) - np.sign(believed(d, g0, psi)) * reach[:, None]) / ROOT_2PI


def flip_counts(d, g0, psi):
    """tau_j: the showings before feature j's belief has the sign of d_j for good (0 if it starts there)."""
    right = np.sign(believed(d, g0, psi)) == np.sign(np.asarray(d, float))[:, None]
    # the sign flips at most once, so tau_j is the first count at which it is right
    return np.where(right.any(axis=1), right.argmax(axis=1), right.shape[1])


def scores(d, g0, psi, delta):
    """The solution's score s_j = |d_j| (2 delta^tau_j - 1); show the feature with the largest, forever."""
    return np.abs(np.asarray(d, float)) * (2.0 * delta ** flip_counts(d, g0, psi) - 1.0)


def solution_value(d, g0, psi, delta):
    """V* = (|d| - max_j s_j) / (sqrt(2 pi) (1 - delta)): the solution's optimal discounted regret."""
    return (np.linalg.norm(d) - scores(d, g0, psi, delta).max()) / (ROOT_2PI * (1.0 - delta))


def constant_values(c, delta):
    """(n,) the discounted regret of showing feature j every round, given its costs (n, M), the last of which
    holds for every later round."""
    c = np.asarray(c, float)
    M = c.shape[1]
    return c @ delta ** np.arange(M) + c[:, -1] * delta ** M / (1.0 - delta)


def _settled(c):
    """Per feature, the count from which its cost no longer changes."""
    moving = np.abs(c - c[:, -1:]) > 1e-12
    return np.where(moving.any(axis=1), c.shape[1] - moving[:, ::-1].argmax(axis=1), 0)


def optimal(c, delta, T):
    """The least discounted regret over every schedule, by dynamic programming over how often each feature
    has been shown. `c` (n, M) are the costs, constant from each feature's settled count on. Returns the value
    and the first T features of a schedule that attains it."""
    c = np.asarray(c, float)
    n = len(c)
    cap = _settled(c)
    V = np.zeros(cap + 1)
    pick = np.zeros(cap + 1, int)
    stay = c[np.arange(n), cap] / (1.0 - delta)                   # a feature past its cap: shown forever from here
    # a state leads only to states with a larger count, so go through them from the largest down
    for m in itertools.product(*[range(k, -1, -1) for k in cap]):
        best = j_best = None
        for j in range(n):
            if m[j] == cap[j]:
                value = stay[j]
            else:
                value = c[j, m[j]] + delta * V[tuple(m[i] + (i == j) for i in range(n))]
            if best is None or value < best - TIE:
                best, j_best = value, j
        V[m], pick[m] = best, j_best
    m, schedule = np.zeros(n, int), []
    for _ in range(T):
        j = int(pick[tuple(np.minimum(m, cap))])
        schedule.append(j)
        m[j] += 1
    return float(V[(0,) * n]), np.array(schedule)


def schedule_costs(c, schedule):
    """(T,) the expected regret of each round of a schedule of feature indices."""
    c = np.asarray(c, float)
    m, out = np.zeros(len(c), int), []
    for j in schedule:
        out.append(c[j, min(m[j], c.shape[1] - 1)])
        m[j] += 1
    return np.array(out)


# ---------------------------------------------------------------------------
# A human who fills in the hidden features from the shown one, as on the Correlated features bench: x_hat_i =
# Sigma_ij x_j, so they pick action 1 when (Sigma g)_j x_j > 0. The cost of showing j now depends on the beliefs
# about every feature, and so on every count: teaching one feature changes what showing another costs.
# ---------------------------------------------------------------------------

def imputed_caps(d, g0, psi, Sigma, tol=1e-3):
    """Per feature, the count from which the rest of its belief's error moves no (Sigma g)_j by more than `tol`
    of the smallest |(Sigma d)_j|: past it the count is treated as no longer changing anything."""
    d, g0, Sigma = np.asarray(d, float), np.asarray(g0, float), np.asarray(Sigma, float)
    reach = np.abs(Sigma @ d).min()
    pull = np.abs(g0 - d) * np.abs(Sigma).max(axis=1)              # how far feature i can move any (Sigma g)_j
    small = np.asarray(psi, float)[None, :] * pull[:, None] <= tol * reach
    return np.where(small.any(axis=1), small.argmax(axis=1), small.shape[1] - 1)


def imputed_costs(d, g0, psi, Sigma, caps):
    """(n, *caps + 1) the expected regret of a round that shows feature j at count vector m, for the human who
    fills in the hidden features."""
    d, Sigma = np.asarray(d, float), np.asarray(Sigma, float)
    g = believed(d, g0, psi)
    grids = np.meshgrid(*[g[i, :caps[i] + 1] for i in range(len(d))], indexing="ij")
    perceived = np.einsum("i...,ij->j...", np.array(grids), Sigma)  # (Sigma g)_j at every count vector
    reach = Sigma @ d
    return (np.sqrt(d @ Sigma @ d) - np.sign(perceived) * reach.reshape((-1,) + (1,) * len(d))) / ROOT_2PI


def optimal_grid(C, delta, T):
    """`optimal` for costs that depend on every count: C (n, *shape), the last count of each axis absorbing.
    Goes through the count vectors from the largest total down, one total at a time."""
    C = np.asarray(C, float)
    n, shape = len(C), np.array(C.shape[1:])
    cap = shape - 1
    V, pick = np.zeros(C.shape[1:]), np.zeros(C.shape[1:], int)
    index = np.indices(C.shape[1:]).reshape(n, -1)
    order = np.argsort(index.sum(axis=0), kind="stable")
    layers = np.split(index[:, order], np.cumsum(np.bincount(index.sum(axis=0)))[:-1], axis=1)
    for m in reversed(layers):
        best, j_best = np.full(m.shape[1], np.inf), np.zeros(m.shape[1], int)
        for j in range(n):
            here = C[j][tuple(m)]
            ahead = m.copy()
            ahead[j] = np.minimum(m[j] + 1, cap[j])
            value = np.where(m[j] == cap[j], here / (1.0 - delta), here + delta * V[tuple(ahead)])
            better = value < best - TIE
            best, j_best = np.where(better, value, best), np.where(better, j, j_best)
        V[tuple(m)], pick[tuple(m)] = best, j_best
    m, schedule = np.zeros(n, int), []
    for _ in range(T):
        j = int(pick[tuple(np.minimum(m, cap))])
        schedule.append(j)
        m[j] += 1
    return float(V[(0,) * n]), np.array(schedule)


def grid_constant_values(C, delta):
    """(n,) `constant_values` for costs that depend on every count: feature j shown every round, the others never."""
    C = np.asarray(C, float)
    n = len(C)
    along = [C[j][tuple(slice(None) if i == j else 0 for i in range(n))] for j in range(n)]     # its own axis only
    return np.array([constant_values(row[None], delta)[0] for row in along])


def grid_schedule_costs(C, schedule):
    """(T,) the expected regret of each round of a schedule, for costs that depend on every count."""
    C = np.asarray(C, float)
    cap = np.array(C.shape[1:]) - 1
    m, out = np.zeros(len(C), int), []
    for j in schedule:
        out.append(C[j][tuple(np.minimum(m, cap))])
        m[j] += 1
    return np.array(out)


def simulate(d, g0, psi, schedule, runs, Sigma=None, seed=0, impute=False):
    """(T,) the regret of each round averaged over `runs` draws of the states, for a human who sees only the
    scheduled feature, picks action 1 when g_j x_j > 0 and guesses on a tie. With `impute` they first fill in the
    hidden features with their conditional mean (simlab/correlated.py) and score g . x_hat."""
    d = np.asarray(d, float)
    n, T = len(d), len(schedule)
    Sigma = np.eye(n) if Sigma is None else np.asarray(Sigma, float)
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((runs, T, n)) @ np.linalg.cholesky(Sigma).T
    g = believed(d, g0, psi)
    m, out = np.zeros(n, int), np.zeros(T)
    for t, j in enumerate(schedule):
        beliefs = g[np.arange(n), np.minimum(m, g.shape[1] - 1)]
        if impute:
            seen = imputation(Sigma, np.arange(n) == j) @ X[:, t].T            # (n, runs): x_hat
            perceived = beliefs @ seen
        else:
            perceived = beliefs[j] * X[:, t, j]
        margin = X[:, t] @ d
        wrong = np.where(perceived == 0, 0.5, (np.sign(perceived) != np.sign(margin)).astype(float))
        out[t] = (np.abs(margin) * wrong).mean()
        m[j] += 1
    return out
