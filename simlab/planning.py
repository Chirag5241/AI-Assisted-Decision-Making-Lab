"""Planning what to show before the state is drawn, for two actions.

With two actions only the gap between their rows matters: d = u_1 - u_2 is the true gap and g the gap
the human believes, on the shown features (zero on a hidden one). The human picks action 1 when
g.x > 0 and it is the best move when d.x > 0. For x ~ N(0, I) the expected regret of a round is

    E[ |d.x| 1{sign(g.x) != sign(d.x)} ] = (|d| - g.d / |g|) / sqrt(2 pi),

so it depends on g only through its direction. A human with g = 0 cannot tell the actions apart and
guesses: |d| / sqrt(2 pi), half of the always-wrong |d| sqrt(2 / pi).

Every action is offered, so feature j's believed gap after m rounds shown is
d_j + (g0_j - d_j)(1 - phi_j(m)), and the state of the problem is how often each feature has been
shown. `plan` finds the schedule F_0 .. F_{T-1}, each of at most `budget` features, with the least
expected discounted regret, by dynamic programming over (round, counts):

    V_t(m) = min_F [ L(F, m) + delta V_{t+1}(m + 1_F) ],    V_T = 0.

Round t has (t + 1)^n count vectors, so this is for a few features only.
"""
from __future__ import annotations

import itertools

import numpy as np

ROOT_2PI = np.sqrt(2.0 * np.pi)
TIE = 1e-9          # losses closer than this count as equal: the first candidate in order is kept


def expected_regret(g, d):
    """Expected regret of a round over x ~ N(0, I): `g` (..., n) is the gap the human believes on the
    shown features (zero on a hidden one) and `d` (n,) the true gap."""
    g, d = np.asarray(g, float), np.asarray(d, float)
    size = (g * g).sum(axis=-1)
    with np.errstate(invalid="ignore", divide="ignore"):
        along = np.where(size > 1e-24, (g * d).sum(axis=-1) / np.sqrt(size), 0.0)
    return (np.sqrt((d * d).sum()) - along) / ROOT_2PI


def candidates(n, budget):
    """Every non-empty subset of at most `budget` of the n features, as bool masks: smaller subsets
    first, and among equals the one with the lower-numbered features."""
    sets = [m for m in itertools.product([False, True], repeat=n) if 0 < sum(m) <= budget]
    return np.array(sorted(sets, key=lambda m: (sum(m), [-on for on in m])))


def plan(d, g0, phi, budget, delta):
    """Schedules for T rounds, set before any state is drawn. `phi` (n, T) is each feature's learning
    curve at 0 .. T - 1 rounds shown. Returns a dict:

      sets      (S, n) the candidate subsets
      optimal   the schedule with the least expected discounted regret
      myopic    the schedule that shows, each round, the subset losing least in that round
      fixed     (S, T) the expected regret of each candidate kept every round

    `optimal` and `myopic` are dicts of `schedule` (T, n) bool, `loss` (T,) the expected regret of
    each round, and `gave_up` (T,) bool: the rounds that show a subset losing more in that round than
    another would at the same beliefs."""
    d, g0, phi = np.asarray(d, float), np.asarray(g0, float), np.asarray(phi, float)
    n, T = phi.shape
    sets = candidates(n, budget)
    believed = d[:, None] + (g0 - d)[:, None] * (1.0 - phi)                  # (n, T): feature j's gap after m rounds
    # each candidate's loss over the counts of its own features: T long on those axes, 1 on the others
    loss = []
    for mask in sets:
        g = np.zeros((T,) * int(mask.sum()) + (n,))
        for axis, j in enumerate(np.flatnonzero(mask)):
            g[..., j] = believed[j].reshape([T if a == axis else 1 for a in range(int(mask.sum()))])
        table = expected_regret(g, d)
        loss.append(table.reshape([T if on else 1 for on in mask]))
    at = lambda i, m: float(loss[i][tuple(m[j] if sets[i][j] else 0 for j in range(n))])

    def first_least(values):
        """The first candidate that no later one beats by more than TIE."""
        best = 0
        for i in range(1, len(values)):
            if values[i] < values[best] - TIE:
                best = i
        return best

    def follow(choose):
        """Run a rule forward from zero counts: choose(t, m) names the candidate to show."""
        m, shown, cost, gave_up = np.zeros(n, int), [], [], []
        for t in range(T):
            now = [at(i, m) for i in range(len(sets))]
            i = choose(t, m, now)
            shown.append(sets[i]); cost.append(now[i]); gave_up.append(now[i] > min(now) + TIE)
            m += sets[i]
        return dict(schedule=np.array(shown), loss=np.array(cost), gave_up=np.array(gave_up))

    # backwards over the rounds: in round t no feature has been shown more than t times
    V, picks = np.zeros((T + 1,) * n), []
    for t in range(T - 1, -1, -1):
        later, grid = delta * V, (slice(0, t + 1),) * n
        best = pick = None
        for i, mask in enumerate(sets):
            value = (loss[i][tuple(slice(0, t + 1) if on else slice(0, 1) for on in mask)]
                     + later[tuple(slice(int(on), int(on) + t + 1) for on in mask)])
            if best is None:
                best, pick = value.copy(), np.zeros(value.shape, np.uint8)
            else:
                better = value < best - TIE
                np.copyto(best, value, where=better)
                pick[better] = i
        V[grid] = best
        picks.append(pick)
    picks.reverse()

    every = np.arange(T)
    return dict(sets=sets,
                optimal=follow(lambda t, m, now: int(picks[t][tuple(m)])),
                myopic=follow(lambda t, m, now: first_least(now)),
                fixed=np.array([loss[i][tuple(every if on else 0 for on in mask)] for i, mask in enumerate(sets)]))
