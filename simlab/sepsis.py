"""Sepsis at the bedside: the two-action problem with Sigma, B and U fitted to the PhysioNet 2019 data.

The hidden state x is the labs; the symptoms s are the vitals and demographics, seen every round. Both are in
normal scores and (x, s) is taken as jointly N(0, Sigma), so s = Bx + eps with B = Sigma_sx Sigma_xx^-1 and the
noise eps carried by Sigma: nothing in B is assumed exact. A panel F of at most c labs is revealed and every other
lab is filled in, by one of two rules (World.fill):

  "symptoms"  from s alone, mu(s) = A s with A = Sigma_xs Sigma_ss^-1 (Problem Description);
  "shown"     from s and the shown labs, E[x_H | x_F, s] = A s + Gamma_HF Gamma_FF^-1 (x_F - A_F s)
              (Guan, Raman & Fang), where Gamma = Sigma_xx - A Sigma_sx is the spread of the labs s leaves.

The utility is on the labs only: d . x is the log-odds of sepsis less the log-odds of the treatment threshold,
set at the average patient's risk so that there is no offset. The human holds weights p on the labs and treats
("stay") when p . x_hat > 0; with the hidden labs filled in, that is q . z > 0 for z = (x, s) and weights q fixed
by the panel and the rule. A round's regret is |d . x| when the human's sign is wrong; d . x and q . z are jointly
normal, so its expectation is exact,

    (sqrt(D' Sigma D) - D' Sigma q / sqrt(q' Sigma q)) / sqrt(2 pi),    D = (d, 0)

(planning.expected_regret with Sigma = I). The human learns each lab's weight toward the truth on the rounds it is
shown, p_j = d_j + psi(m_j) (p0_j - d_j), psi(m) = r^m; s carries no utility of its own, so nothing about it is
learned.

At graduation the human orders the panel of greatest perceived value: the variance, given s, of the margin they
will compute once the panel is back (Autonomous Test Selection),

  "symptoms"  p_F' Gamma_FF p_F,
  "shown"     (Gamma p)_F' Gamma_FF^-1 (Gamma p)_F.

Neither depends on s, so neither does the panel, and every training policy here names its panel before the
round's state is drawn. Under "shown" with p = d the human's margin is E[d . x | x_F, s], so the true weights are
the best for any panel and the panel they value most is the one of least regret; under "symptoms" neither holds.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass
from functools import cached_property

import numpy as np

ROOT_2PI = np.sqrt(2.0 * np.pi)


@dataclass
class World:
    names: list           # every variable of z = (x, s): the labs, then the symptoms
    n_labs: int
    Sigma: np.ndarray     # (n, n) correlation of the normal scores of z
    d: np.ndarray         # (labs,) true weights: stay minus home
    fill: str = "symptoms"  # how hidden labs are filled in: "symptoms" or "shown"

    @classmethod
    def from_fit(cls, fit):
        return cls(names=list(fit["labs"]) + list(fit["symptoms"]), n_labs=len(fit["labs"]),
                   Sigma=np.asarray(fit["Sigma"], float), d=np.asarray(fit["d"], float))

    @property
    def L(self):
        return slice(0, self.n_labs)

    @property
    def V(self):
        """The symptoms' place in z."""
        return slice(self.n_labs, len(self.names))

    @cached_property
    def D(self):
        """(n,) the true weights on z: d on the labs, nothing on the symptoms."""
        return np.concatenate([self.d, np.zeros(len(self.names) - self.n_labs)])

    @cached_property
    def A(self):
        """(labs, symptoms): the labs' conditional mean given the symptoms, mu(s) = A s."""
        S = self.Sigma
        return S[self.L, self.V] @ np.linalg.inv(S[self.V, self.V])

    @cached_property
    def Gamma(self):
        """(labs, labs): the labs' covariance left once the symptoms are seen."""
        S = self.Sigma
        return S[self.L, self.L] - self.A @ S[self.V, self.L]

    def probes(self, count, seed=0):
        """`count` states z ~ N(0, Sigma), for checking the exact regret by sampling."""
        Z = np.random.default_rng(seed).standard_normal((count, len(self.names)))
        return Z @ np.linalg.cholesky(self.Sigma).T


def panels(n_labs, budget):
    """Every panel of at most `budget` labs, the empty one first, as (P, n_labs) bool masks: smaller first,
    and among equals the one with the lower-numbered labs."""
    sets = [m for size in range(budget + 1) for m in itertools.combinations(range(n_labs), size)]
    masks = np.zeros((len(sets), n_labs), bool)
    for i, s in enumerate(sets):
        masks[i, list(s)] = True
    return masks


def _through_shown(world, p, masks):
    """(P, labs) k = Gamma_FF^-1 (Gamma p_L)_F on each panel's labs, zero elsewhere: under the "shown" rule, the
    weight a shown lab carries once the hidden labs are filled in from it. Solved for all panels of a size at once."""
    g = world.Gamma @ p
    k = np.zeros(masks.shape)
    for rows, idx, inverse in _blocks(world, masks):
        k[rows[:, None], idx] = np.einsum("pij,pj->pi", inverse, g[idx])
    return k


_BLOCKS = {}


def _blocks(world, masks):
    """Per panel size: the panels' rows, their labs (P, size) and Gamma_FF^-1 (P, size, size). Kept for the last
    few (world, panels) pairs, since training asks for the same ones every round."""
    key = (id(world), masks.shape, masks.tobytes())
    if key not in _BLOCKS:
        if len(_BLOCKS) > 8:
            _BLOCKS.clear()
        out, sizes = [], masks.sum(axis=1)
        for size in np.unique(sizes[sizes > 0]):
            rows = np.flatnonzero(sizes == size)
            idx = np.nonzero(masks[rows])[1].reshape(len(rows), size)
            out.append((rows, idx, np.linalg.inv(world.Gamma[idx[:, :, None], idx[:, None, :]])))
        _BLOCKS[key] = (world, out)    # the world is kept so its id is not reused while the entry lives
    return _BLOCKS[key][1]


def effective(world, p, masks):
    """(P, n) the weights the human puts on z = (x, s) under each panel, from their weights p on the labs.
    "symptoms": p on a shown lab, zero on a hidden one, and on s what the hidden labs pass through mu(s).
    "shown": k on the shown labs, and on s A' p - A_F' k."""
    masks = np.atleast_2d(masks)
    q = np.zeros((len(masks), len(world.names)))
    if world.fill == "shown":
        k = _through_shown(world, p, masks)
        q[:, world.L] = k
        q[:, world.V] = p @ world.A - k @ world.A
    else:
        q[:, world.L] = p * masks
        q[:, world.V] = (p * ~masks) @ world.A
    return q


def perceived(world, p, masks):
    """(P,) the perceived value of each panel: p_F' Gamma_FF p_F, or (Gamma p)_F' Gamma_FF^-1 (Gamma p)_F."""
    if world.fill == "shown":
        return _through_shown(world, p, masks) @ (world.Gamma @ p)
    g = p * masks
    return ((g @ world.Gamma) * g).sum(axis=1)


def chosen(world, p, masks):
    """The human's own panel: the first of greatest perceived value."""
    v = perceived(world, p, masks)
    return masks[int(np.flatnonzero(v >= v.max() - 1e-12)[0])]


def regret(world, q):
    """(P,) expected regret of a round for a human weighing z by each row of q (P, n); q = 0 guesses."""
    q = np.atleast_2d(q)
    Sd = world.Sigma @ world.D
    size = np.sqrt(((q @ world.Sigma) * q).sum(axis=1))
    along = np.divide(q @ Sd, size, out=np.zeros(len(q)), where=size > 1e-12)
    return (np.sqrt(world.D @ Sd) - along) / ROOT_2PI


def sampled_regret(world, q, X):
    """(P,) the same, averaged over the states X (N, n) instead."""
    true = X @ world.D
    wrong = np.sign(X @ np.atleast_2d(q).T) != np.sign(true)[:, None]
    return (np.abs(true)[:, None] * wrong).mean(axis=0)


def believed(world, p0, counts, rate):
    """The weights on the labs after each has been shown `counts` times."""
    return world.d + rate ** np.asarray(counts, float) * (p0 - world.d)


def graduation(world, p, masks):
    """Regret per round once training stops: the human orders their own panel with weights p."""
    return float(regret(world, effective(world, p, chosen(world, p, masks)))[0])


# ---------------------------------------------------------------------------------------------------------------
# Training policies: each names the labs shown in a round from the human's current weights and counts.
# ---------------------------------------------------------------------------------------------------------------

def policy_self(world, p, counts, budget, masks, rng):
    """No teacher: the human orders their own panel."""
    return chosen(world, p, masks)


def policy_random(world, p, counts, budget, masks, rng):
    mask = np.zeros(world.n_labs, bool)
    mask[rng.choice(world.n_labs, budget, replace=False)] = True
    return mask


def policy_expert(world, p, counts, budget, masks, rng):
    """The panel an expert who knew d would order, every round."""
    return chosen(world, world.d, masks)


def policy_teacher(world, p, counts, budget, masks, rng):
    """Show the labs whose belief, once right, would cut the graduation regret most; when fewer than `budget`
    would cut it at all, fill up with the labs whose belief is furthest from the truth, in lab units."""
    now = graduation(world, p, masks)
    gain = np.empty(world.n_labs)
    for j in range(world.n_labs):
        fixed = p.copy()
        fixed[j] = world.d[j]
        gain[j] = now - graduation(world, fixed, masks)
    off = np.abs(p - world.d) * np.sqrt(np.diag(world.Gamma))
    order = np.lexsort((-off, -np.where(gain > 1e-9, gain, 0.0)))
    mask = np.zeros(world.n_labs, bool)
    mask[order[:budget]] = True
    return mask


POLICIES = {"self": policy_self, "random": policy_random, "expert": policy_expert, "teacher": policy_teacher}


def train(world, p0, policy, T, budget, rate, seed=0):
    """Run T training rounds. Returns the shown panels (T, labs), the expected regret of each training round,
    and the graduation regret had training stopped before each round and after the last (T + 1)."""
    rng = np.random.default_rng(seed)
    masks = panels(world.n_labs, budget)
    counts = np.zeros(world.n_labs)
    shown, during, after = [], [], []
    for t in range(T):
        p = believed(world, p0, counts, rate)
        after.append(graduation(world, p, masks))
        F = policy(world, p, counts, budget, masks, rng)
        during.append(float(regret(world, effective(world, p, F))[0]))
        shown.append(F)
        counts += F
    after.append(graduation(world, believed(world, p0, counts, rate), masks))
    return np.array(shown), np.array(during), np.array(after)


def decisions(q, X):
    """Stay (True) or home for each state z, for weights q (n,)."""
    return X @ q > 0
