"""Learning curves phi(m).

phi(m) is the fraction of the gap between the human's initial belief H0 and the
truth U that has been closed after m observations: phi(0) = 0, phi is
non-decreasing, and phi(m) -> 1 as m -> infinity (Noti et al.'s monotone phi).

Curve parameters may be arrays that broadcast against the count tensor of shape
(B, K, n), e.g. a rate of shape (K, 1) gives each action its own learning speed.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass(frozen=True)
class Curve:
    name: str
    fn: Callable[[np.ndarray], np.ndarray]

    def __call__(self, m):
        return self.fn(np.asarray(m, dtype=float))


def exponential(rate=0.1) -> Curve:
    """Constant-fraction learner: each observation closes `rate` of the remaining gap."""
    rate = np.asarray(rate, dtype=float)
    return Curve("exponential", lambda m: 1.0 - (1.0 - rate) ** m)


def hyperbolic(half=5.0) -> Curve:
    """Averaging-style learner: phi = m / (m + half); half the gap closes after `half` observations."""
    half = np.asarray(half, dtype=float)
    return Curve("hyperbolic", lambda m: m / (m + half))


def power_law(exponent=0.5) -> Curve:
    """Heavy-tailed learner: the remaining gap decays as (1 + m)^-exponent."""
    exponent = np.asarray(exponent, dtype=float)
    return Curve("power law", lambda m: 1.0 - (1.0 + m) ** (-exponent))


def sigmoid(midpoint=20.0, width=4.0) -> Curve:
    """Slow-start learner: little progress, then a rapid "aha" around `midpoint` observations."""
    midpoint = np.asarray(midpoint, dtype=float)
    width = np.asarray(width, dtype=float)

    def fn(m):
        s = 1.0 / (1.0 + np.exp(-(m - midpoint) / width))
        s0 = 1.0 / (1.0 + np.exp(midpoint / width))
        return (s - s0) / (1.0 - s0)

    return Curve("sigmoid", fn)
