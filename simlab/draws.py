"""The state of each round, x_t ~ N(0, I), as a reproducible draw.

The proposal's loss is taken at the state the human actually sees in round t, so a run needs one state
per round. A small counter-based generator (mulberry32, then Box-Muller) makes them, rather than
numpy's own, because the browser build of the site has to produce the very same numbers.
"""
from __future__ import annotations

import numpy as np

_MASK = np.uint64(0xFFFFFFFF)


def uniforms(seed, count):
    """`count` uniform numbers in [0, 1) from mulberry32, as 32-bit integers (uint64 array)."""
    a = (np.uint64(seed) + (np.arange(count, dtype=np.uint64) + np.uint64(1)) * np.uint64(0x6D2B79F5)) & _MASK
    t = ((a ^ (a >> np.uint64(15))) * (a | np.uint64(1))) & _MASK
    t = t ^ ((t + (((t ^ (t >> np.uint64(7))) * (t | np.uint64(61))) & _MASK)) & _MASK)
    return t ^ (t >> np.uint64(14))


def states(seed, T, n):
    """(T, n) independent standard normal states, one row per round; `seed` picks the draw."""
    r = uniforms(seed, 2 * T * n).astype(float)
    u1, u2 = (r[0::2] + 0.5) / 2.0**32, r[1::2] / 2.0**32
    return (np.sqrt(-2.0 * np.log(u1)) * np.cos(2.0 * np.pi * u2)).reshape(T, n)
