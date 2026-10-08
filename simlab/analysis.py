"""Flip metrics, plus the closed form for the 1-feature / 2-action case.

Why the 1-feature, 2-action case is special: with a scalar state x the human
plays action 1 iff (h1 - h2) x > 0 and the optimum is action 1 iff
(u1 - u2) x > 0, so the human is right iff sign(h1 - h2) == sign(u1 - u2),
whatever x is. When both actions share one learning curve phi,

    h1_t - h2_t = (1 - phi(t)) * dh0 + phi(t) * du,

so a human who starts with the wrong sign flips exactly when
phi(t) > |dh0| / (|dh0| + |du|). Only the ratio |dh0| / |du| matters.
"""
from __future__ import annotations

import numpy as np

MEAN_ABS_STD_NORMAL = np.sqrt(2.0 / np.pi)


# --- metrics on simulated runs: `correct` is (T, B) bool ---------------------

def first_correct(correct):
    """First round in which the human is right; inf if never within the horizon."""
    t = correct.argmax(axis=0).astype(float)
    t[~correct.any(axis=0)] = np.inf
    return t


def settle_time(correct):
    """First round from which the human is right for the rest of the horizon; inf if wrong at the end."""
    T = correct.shape[0]
    last_wrong = T - 1 - (~correct)[::-1].argmax(axis=0)
    t = np.where(correct.all(axis=0), 0.0, last_wrong + 1.0)
    t[t >= T] = np.inf
    return t


def n_switches(correct):
    """Number of right<->wrong changes: >1 means the human's choice un-flipped at some point."""
    return (correct[1:] != correct[:-1]).sum(axis=0)


def evaluate(H, U, X, F=None, A=None, value_gap=False):
    """Accuracy and expected regret of beliefs H (..., K, n) against U over probe states X (P, n).

    F (..., n) and A (..., K) are optional bool masks for the features shown and the actions
    offered: the human scores only offered actions using only shown features, while regret is
    measured against the best of ALL K actions on the full state. Actions the human cannot
    tell apart are chosen uniformly at random (their regret is averaged).

    This is the n-feature generalisation of the flip indicator: for one feature and
    two actions the accuracy is exactly 0 before the flip and 1 after it.

    With `value_gap=True` a third array is returned: the mean of max_k (U x)_k - u_hat(y_hat),
    the true best value minus the value the human expects from the action they pick
    (u_hat(y_hat) is the largest of their own scores among the offered actions).
    """
    H = np.asarray(H, float)
    lead, (K, n) = H.shape[:-2], H.shape[-2:]
    seen = (H if F is None else H * np.asarray(F)[..., None, :]).reshape(-1, K, n)
    B, P = len(seen), len(X)
    offered = np.ones((B, K), bool) if A is None else np.broadcast_to(A, lead + (K,)).reshape(B, K)

    est = seen @ X.T                                     # (B, K, P)
    est[~offered] = -np.inf
    U = np.asarray(U, float)
    shared_truth = U.ndim == 2                           # one U for the whole batch: score it once
    true = U @ X.T if shared_truth else np.broadcast_to(U, H.shape).reshape(B, K, n) @ X.T
    regret_of = true.max(axis=-2, keepdims=True) - true  # (K, P) or (B, K, P)

    pick = est.argmax(axis=1)                            # (B, P)
    cols = np.arange(P)
    regret = regret_of[pick, cols] if shared_truth else regret_of[np.arange(B)[:, None], pick, cols]
    accuracy, mean_regret = (regret <= 1e-12).mean(axis=1), regret.mean(axis=1)

    # Two offered actions tie on a whole region of states only if the human's shown weights for
    # them are identical (e.g. beliefs that start at zero); only those worlds need tie-splitting.
    alike = (seen[:, :, None, :] == seen[:, None, :, :]).all(axis=-1) & offered[:, :, None] & offered[:, None, :]
    for b in np.flatnonzero((alike.sum(axis=-1) > 1).any(axis=-1)):
        tied = est[b] == est[b].max(axis=0, keepdims=True)
        weight = tied / tied.sum(axis=0, keepdims=True)
        r = regret_of if shared_truth else regret_of[b]
        accuracy[b] = (weight * (r <= 1e-12)).sum(axis=0).mean()
        mean_regret[b] = (weight * r).sum(axis=0).mean()
    if not value_gap:
        return accuracy.reshape(lead), mean_regret.reshape(lead)
    gap = (true.max(axis=-2) - est.max(axis=1)).mean(axis=-1)       # (P,) or (B, P) minus (B, P)
    return accuracy.reshape(lead), mean_regret.reshape(lead), gap.reshape(lead)


def realized(seen, U, x, offered=None):
    """Regret, value gap, pick and best move at one drawn state: the loss of a round, not its average.

    `seen` (..., K, n) are the weights the human scores with (hidden features already zeroed, or
    filled in), `x` (..., n) the state drawn for each world and `offered` (..., K) the actions on the
    table; `x` and `offered` broadcast against `seen`. Actions the human cannot tell apart at that
    state share the regret evenly, as in `evaluate`; the pick reported is the lowest-numbered of
    them. Regret is measured against the best of all K actions on the full state.
    """
    seen = np.asarray(seen, float)
    est = np.einsum("...kn,...n->...k", seen, x)
    if offered is not None:
        est = np.where(offered, est, -np.inf)
    true = np.einsum("kn,...n->...k", np.asarray(U, float), x)
    top = est.max(axis=-1, keepdims=True)
    tied = est == top
    regret_of = true.max(axis=-1, keepdims=True) - true
    regret = np.broadcast_to(tied * regret_of, est.shape).sum(axis=-1) / tied.sum(axis=-1)
    shape = regret.shape
    return (regret, np.broadcast_to(true.max(axis=-1), shape) - top[..., 0], tied.argmax(axis=-1),
            np.broadcast_to(true.argmax(axis=-1), shape))


# --- closed form, 1 feature / 2 actions / shared curve -----------------------

def flip_threshold(du, dh0):
    """Fraction of learning phi* a wrong-signed human needs: right iff phi(t) > phi*.

    Returns 0 where the signs already agree (the human is right from round 0).
    """
    du, dh0 = np.broadcast_arrays(np.asarray(du, float), np.asarray(dh0, float))
    with np.errstate(invalid="ignore"):
        phi = np.abs(dh0) / (np.abs(dh0) + np.abs(du))
    return np.where(du * dh0 > 0, 0.0, phi)


def theory_flip_time(du, dh0, curve, T):
    """Predicted flip round: smallest t <= T with sign(h1_t - h2_t) == sign(du); inf if none."""
    du, dh0 = np.broadcast_arrays(np.asarray(du, float), np.asarray(dh0, float))
    phi = curve(np.arange(T + 1)).reshape((T + 1,) + (1,) * du.ndim)
    aligned = ((1.0 - phi) * dh0 + phi * du) * du > 0
    return np.where(aligned.any(axis=0), aligned.argmax(axis=0), np.inf)


def theory_discounted_regret(du, flip_time, delta, mean_abs_x=MEAN_ABS_STD_NORMAL):
    """Expected discounted regret when the human is wrong on rounds 0..flip_time-1 and right after."""
    return np.abs(du) * mean_abs_x * (1.0 - delta ** np.asarray(flip_time, float)) / (1.0 - delta)
