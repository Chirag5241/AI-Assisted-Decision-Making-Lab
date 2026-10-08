"""Correlated features, after Guan, Raman & Fang (2026).

States are x ~ N(0, Sigma). The human sees only the shown features x_S and fills in the hidden ones
with their conditional mean, E[x_hidden | x_S] = Sigma_hS Sigma_SS^-1 x_S, then scores action k with
(H x_hat)_k. Because x_hat = P_S x for a fixed n x n matrix P_S, that is the same as scoring the full
state with the effective beliefs H P_S, so `analysis.evaluate` scores this world unchanged.

As in the paper, a belief is learned only on rounds where its feature is shown: the filled-in value
of a hidden feature teaches the human nothing about that feature's weight. Every action is offered
every round, so each feature has one exposure count m_j, and the belief used in round t is
U + (H0 - U) (1 - phi(m_j(t))) with m_j(t) the rounds before t in which feature j was shown.
"""
from __future__ import annotations

import itertools

import numpy as np

from . import analysis


def correlation_matrix(r12, r13, r23):
    return np.array([[1.0, r12, r13], [r12, 1.0, r23], [r13, r23, 1.0]])


def is_valid(Sigma, tol=1e-6):
    """A correlation matrix the states can actually have: symmetric and positive definite."""
    return bool(np.allclose(Sigma, Sigma.T) and np.linalg.eigvalsh(Sigma).min() > tol)


def probes(Sigma, count, seed=0):
    """Probe states x ~ N(0, Sigma), from the same standard normals the independent bench draws,
    so Sigma = I gives exactly that bench's states."""
    Z = np.random.default_rng(seed).standard_normal((count, len(Sigma)))
    return Z @ np.linalg.cholesky(Sigma).T


def imputation(Sigma, shown):
    """P with x_hat = P x: shown features pass through, hidden ones get their conditional mean."""
    shown = np.asarray(shown, bool)
    n = len(Sigma)
    P = np.zeros((n, n))
    S, h = np.flatnonzero(shown), np.flatnonzero(~shown)
    P[S, S] = 1.0
    if len(S) and len(h):
        P[np.ix_(h, S)] = Sigma[np.ix_(h, S)] @ np.linalg.inv(Sigma[np.ix_(S, S)])
    return P


def rotation(n, k):
    """The exploration phase: every size-k subset in turn, so each feature is shown equally often."""
    sets = list(itertools.combinations(range(n), k))
    masks = np.zeros((len(sets), n), bool)
    for i, s in enumerate(sets):
        masks[i, list(s)] = True
    return masks


def explore_then_commit(n, k, explore, commit, T):
    """(T, n) masks: the rotation for the first `explore` rounds, then the `commit` mask every round.
    explore = 0 is the stationary policy that shows `commit` from the start."""
    cycle = rotation(n, k)
    masks = np.empty((T, n), bool)
    e = min(explore, T)
    masks[:e] = cycle[np.arange(e) % len(cycle)]
    masks[e:] = np.asarray(commit, bool)
    return masks


def exposure(masks):
    """Rounds before t in which each feature was shown: (T, ..., n) masks to (T, ..., n) counts."""
    masks = np.asarray(masks, float)
    return np.concatenate([np.zeros_like(masks[:1]), np.cumsum(masks, axis=0)[:-1]])


def beliefs(U, H0, curve, counts):
    """Beliefs for exposure counts (..., n): each feature's weights close their gap by phi(m_j)."""
    phi = curve(counts)[..., None, :]
    return U + (H0 - U) * (1.0 - phi)


class Scorer:
    """Scores beliefs under feature masks on one fixed set of correlated probe states."""

    def __init__(self, U, Sigma, X):
        self.U, self.Sigma, self.X = np.asarray(U, float), Sigma, X
        n = len(Sigma)
        self.P = np.array([imputation(Sigma, m) for m in itertools.product([False, True], repeat=n)])
        self.code = 2 ** np.arange(n)[::-1]    # mask -> its row in self.P

    def effective(self, H, masks):
        """H P_S for each belief (..., K, n) and its mask (..., n)."""
        P = self.P[np.asarray(masks, bool) @ self.code]
        return H @ P

    def __call__(self, H, masks, chunk=256):
        """Accuracy, regret and value gap for beliefs (B, K, n) under masks (B, n), each (B,)."""
        Heff = self.effective(H, masks)
        parts = [analysis.evaluate(Heff[i:i + chunk], self.U, self.X, value_gap=True)
                 for i in range(0, len(Heff), chunk)]
        return tuple(np.concatenate(series) for series in zip(*parts))
