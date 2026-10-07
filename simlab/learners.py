"""Human learning rules: how H_t becomes H_{t+1} after a round.

All tensors carry a leading batch dimension B so a whole sweep runs at once:
U, H: (B, K, n);  x: (B, n);  F: (B, n) bool;  A: (B, K) bool.
Only entries (k, j) with k in A_t and j in F_t are updated.
"""
from __future__ import annotations

import numpy as np

from .curves import Curve


class CountLearner:
    """Noti-style learner: H_kj = U_kj + (H0_kj - U_kj) * (1 - phi(N_kj)).

    N_kj counts the rounds in which action k and feature j were shown together.
    Deterministic, and independent of the realised states x.
    """

    def __init__(self, curve: Curve):
        self.curve = curve
        self.name = curve.name

    def init(self, U, H0):
        return {"H0": H0.copy(), "N": np.zeros(H0.shape)}

    def update(self, H, U, x, F, A, state, rng):
        state["N"] += A[:, :, None] & F[:, None, :]
        return U + (state["H0"] - U) * (1.0 - self.curve(state["N"]))

    def trajectory(self, U, H0, T):
        """Beliefs H_0 .. H_{T-1} when every feature and action is shown every round.

        Same numbers as running `simulate` with the show-everything policy, without the loop,
        so a whole grid of worlds costs a few array operations. Returns (T, *U.shape).
        """
        U, H0 = np.broadcast_arrays(np.asarray(U, float), np.asarray(H0, float))
        # the curve depends on the round and on the (action, feature) entry, never on the world,
        # so evaluate it once per entry and let it broadcast across the batch
        counts = np.arange(T, dtype=float).reshape((T,) + (1,) * U.ndim) + np.zeros(U.shape[-2:])
        return U + (H0 - U) * (1.0 - self.curve(counts))


class DeltaRuleLearner:
    """Feedback-driven learner (LMS / delta rule).

    After the round the human sees the realised payoff of every shown action,
    y_k = u_k . x (+ noise), compares it to their own estimate from the shown
    features, and nudges the shown weights along the error. Hidden features act
    as unexplained noise in the payoff, so this rule is where feature
    correlation would start to bias learning.

    With `normalized=True` the step is eta / ||x_F||^2; for one feature and no
    noise this reduces exactly to the exponential curve with rate eta.
    `eta` may be an array of shape (K,) for per-action learning speeds.
    """

    def __init__(self, eta=0.1, noise=0.0, normalized=True):
        self.eta = np.asarray(eta, dtype=float)
        self.noise = noise
        self.normalized = normalized
        self.name = "delta rule"

    def init(self, U, H0):
        return {}

    def update(self, H, U, x, F, A, state, rng):
        xF = x * F
        payoff = np.einsum("bkn,bn->bk", U, x)
        if self.noise:
            payoff = payoff + self.noise * rng.standard_normal(payoff.shape)
        err = (payoff - np.einsum("bkn,bn->bk", H, xF)) * A
        step = self.eta * np.ones_like(err)
        if self.normalized:
            norm = (xF**2).sum(axis=1, keepdims=True)
            step = step / np.where(norm > 0, norm, 1.0)
        return H + (step * err)[:, :, None] * xF[:, None, :]
