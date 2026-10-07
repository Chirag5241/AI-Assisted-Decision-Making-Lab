"""The interaction loop from the proposal, batched over B independent worlds.

Round t:
  1. the policy picks a feature mask F_t and an action mask A_t;
  2. the human plays y_hat = argmax_{k in A_t} sum_{j in F_t} (H_t)_kj x_j and
     incurs regret max_k (U x)_k - (U x)_{y_hat}  (max over ALL K actions);
  3. the learner updates H_t -> H_{t+1} on the shown entries.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


class ShowAll:
    """Every feature and every action is shown every round."""

    def __call__(self, t, H, x, rng):
        B, K, n = H.shape
        return np.ones((B, n), dtype=bool), np.ones((B, K), dtype=bool)


class FixedSubset:
    """The same feature subset and action subset every round (the stationary policy)."""

    def __init__(self, features, actions):
        self.features, self.actions = list(features), list(actions)

    def __call__(self, t, H, x, rng):
        B, K, n = H.shape
        F = np.zeros((B, n), dtype=bool)
        A = np.zeros((B, K), dtype=bool)
        F[:, self.features] = True
        A[:, self.actions] = True
        return F, A


class FixedMasks:
    """Stationary policy given as bool masks, F (B, n) and A (B, K): each world in the batch
    can show a different subset, so every candidate subset can be compared in one run."""

    def __init__(self, F, A):
        self.F, self.A = np.asarray(F, dtype=bool), np.asarray(A, dtype=bool)

    def __call__(self, t, H, x, rng):
        B, K, n = H.shape
        return np.broadcast_to(self.F, (B, n)), np.broadcast_to(self.A, (B, K))


def standard_normal(rng, B, n):
    return rng.standard_normal((B, n))


@dataclass
class Result:
    U: np.ndarray        # (B, K, n)
    H: np.ndarray        # (T+1, B, K, n); H[t] is the belief used in round t
    x: np.ndarray        # (T, B, n)
    y_hat: np.ndarray    # (T, B) human's move
    y_star: np.ndarray   # (T, B) optimal move
    regret: np.ndarray   # (T, B)

    @property
    def correct(self):
        """(T, B) bool: the human's move was optimal in that round."""
        return self.regret <= 1e-12

    def discounted_regret(self, delta):
        T = self.regret.shape[0]
        return ((delta ** np.arange(T))[:, None] * self.regret).sum(axis=0)


def _argmax_random_ties(scores, rng):
    best = scores == scores.max(axis=1, keepdims=True)
    return (best * rng.random(scores.shape)).argmax(axis=1)


def simulate(U, H0, learner, T, policy=None, x_sampler=standard_normal, seed=0) -> Result:
    """Run T rounds. U and H0 are (K, n) or (B, K, n) and broadcast against each other."""
    rng = np.random.default_rng(seed)
    U, H0 = np.broadcast_arrays(np.atleast_2d(np.asarray(U, float)), np.atleast_2d(np.asarray(H0, float)))
    if U.ndim == 2:
        U, H0 = U[None], H0[None]
    U, H = U.copy(), H0.copy()
    B, K, n = U.shape
    policy = policy or ShowAll()
    state = learner.init(U, H)
    rows = np.arange(B)

    H_hist = np.empty((T + 1, B, K, n))
    xs = np.empty((T, B, n))
    y_hat = np.empty((T, B), dtype=int)
    y_star = np.empty((T, B), dtype=int)
    regret = np.empty((T, B))
    H_hist[0] = H

    for t in range(T):
        x = x_sampler(rng, B, n)
        F, A = policy(t, H, x, rng)

        est = np.einsum("bkn,bn->bk", H, x * F)
        y = _argmax_random_ties(np.where(A, est, -np.inf), rng)
        true = np.einsum("bkn,bn->bk", U, x)

        xs[t], y_hat[t], y_star[t] = x, y, true.argmax(axis=1)
        regret[t] = true.max(axis=1) - true[rows, y]

        H = learner.update(H, U, x, F, A, state, rng)
        H_hist[t + 1] = H

    return Result(U=U, H=H_hist, x=xs, y_hat=y_hat, y_star=y_star, regret=regret)
