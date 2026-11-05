"""The sepsis world (simlab/sepsis.py): exact regret, the symptoms-only fill-in, the human's panel, and training;
and the page that draws the fitted study."""
import dataclasses

import numpy as np
import pytest

from simlab import sepsis


def small_world(seed=0):
    """Three labs and two symptoms with a random correlation, and random true weights on the labs."""
    rng = np.random.default_rng(seed)
    G = rng.standard_normal((5, 8))
    S = G @ G.T
    sd = np.sqrt(np.diag(S))
    return sepsis.World(names=["L1", "L2", "L3", "S1", "S2"], n_labs=3, Sigma=S / np.outer(sd, sd), d=rng.standard_normal(3))


def test_exact_regret_matches_sampling():
    world = small_world()
    X = world.probes(400_000, seed=1)
    masks = sepsis.panels(3, 3)
    q = np.vstack([sepsis.effective(world, p, masks) for p in (world.d, np.random.default_rng(2).standard_normal(3))])
    np.testing.assert_allclose(sepsis.regret(world, q), sepsis.sampled_regret(world, q, X), rtol=0.01, atol=2e-3)
    # every lab at the true weights is the truth; no weights at all is a coin flip, half of always wrong
    assert sepsis.regret(world, world.D)[0] == pytest.approx(0, abs=1e-12)
    S = world.Sigma[:3, :3]
    assert sepsis.regret(world, np.zeros(5))[0] == pytest.approx(np.sqrt(world.d @ S @ world.d / (2 * np.pi)))
    # the symptoms carry no utility: the truth is d . x on the labs alone
    np.testing.assert_allclose(world.D, np.concatenate([world.d, [0, 0]]))


def test_hidden_labs_are_filled_in_from_the_symptoms_alone():
    world = small_world()
    p = np.random.default_rng(3).standard_normal(3)
    shown = np.array([True, False, True])
    q = sepsis.effective(world, p, shown)[0]
    X = world.probes(5, seed=4)
    s, xL = X[:, 3:], X[:, :3]
    filled = np.where(shown, xL, s @ world.A.T)        # mu_L(s) = A s on the hidden labs
    np.testing.assert_allclose(X @ q, filled @ p)
    # A is the regression of the labs on the symptoms, and Gamma what it leaves
    S = world.Sigma
    np.testing.assert_allclose(world.A, S[:3, 3:] @ np.linalg.inv(S[3:, 3:]))
    np.testing.assert_allclose(world.Gamma, S[:3, :3] - world.A @ S[3:, :3])


def test_hidden_labs_filled_in_from_the_shown_ones_too():
    world = dataclasses.replace(small_world(), fill="shown")
    p = np.random.default_rng(3).standard_normal(3)
    S = world.Sigma
    X = world.probes(5, seed=4)
    for shown in ([True, False, True], [False, True, False], [False, False, False]):
        shown = np.array(shown)
        q = sepsis.effective(world, p, shown)[0]
        seen = np.concatenate([shown, [True, True]])     # the shown labs and the symptoms
        o, h = np.flatnonzero(seen), np.flatnonzero(~seen)
        filled = X.copy()
        filled[:, h] = X[:, o] @ np.linalg.solve(S[np.ix_(o, o)], S[np.ix_(o, h)])    # E[x_H | x_F, s]
        np.testing.assert_allclose(X @ q, filled[:, :3] @ p, atol=1e-10)
        assert (q[:3][~shown] == 0).all()
        # the value of the panel: (Gamma p)_F' Gamma_FF^-1 (Gamma p)_F
        g, F = world.Gamma @ p, np.flatnonzero(shown)
        value = g[F] @ np.linalg.solve(world.Gamma[np.ix_(F, F)], g[F]) if len(F) else 0.0
        assert sepsis.perceived(world, p, shown[None])[0] == pytest.approx(value)
    # no lab shown: both rules fill in from the symptoms alone
    none = np.zeros((1, 3), bool)
    np.testing.assert_allclose(sepsis.effective(world, p, none), sepsis.effective(small_world(), p, none))


@pytest.mark.parametrize("seed", range(5))
def test_with_the_true_weights_the_valued_panel_is_the_best_only_when_shown_labs_fill_in(seed):
    world = dataclasses.replace(small_world(seed), fill="shown")
    for budget in (1, 2):
        masks = sepsis.panels(3, budget)
        every = sepsis.regret(world, sepsis.effective(world, world.d, masks))
        picked = sepsis.regret(world, sepsis.effective(world, world.d, sepsis.chosen(world, world.d, masks)))[0]
        assert picked == pytest.approx(every.min(), abs=1e-12)


def test_the_human_orders_the_panel_of_greatest_perceived_value():
    world = small_world()
    masks = sepsis.panels(3, 2)
    assert len(masks) == 1 + 3 + 3 and not masks[0].any()
    p = np.random.default_rng(5).standard_normal(3)
    best = max(masks, key=lambda m: p[m] @ world.Gamma[np.ix_(m, m)] @ p[m])
    assert (sepsis.chosen(world, p, masks) == best).all()


def test_training_teaches_what_is_shown():
    world = small_world()
    p0 = np.random.default_rng(6).standard_normal(3)
    shown, during, after = sepsis.train(world, p0, sepsis.policy_expert, 60, 1, 0.8)
    assert shown.shape == (60, 3) and (shown.sum(axis=1) == 1).all() and len(during) == 60 and len(after) == 61
    lab = int(np.flatnonzero(shown[0])[0])
    p = sepsis.believed(world, p0, shown.sum(axis=0), 0.8)
    # the shown lab is learned, the others keep their first belief
    np.testing.assert_allclose(p[lab], world.d[lab], atol=1e-4)
    others = [j for j in range(3) if j != lab]
    np.testing.assert_allclose(p[others], p0[others])
    assert after[-1] == pytest.approx(sepsis.graduation(world, p, sepsis.panels(3, 1)))


def test_teacher_fixes_what_graduation_needs():
    world = small_world(1)
    p0 = np.random.default_rng(7).standard_normal(3)
    for budget in (1, 2):
        *_, after = sepsis.train(world, p0, sepsis.policy_teacher, 40, budget, 0.7)
        assert after[-1] <= after[0] + 1e-12


def test_sepsis_page_draws_the_saved_study():
    from webapp.app import app
    page = app.test_client().get("/sepsis")
    assert page.status_code == 200
    html = page.data.decode()
    assert "Sepsis, from data" in html and "window.SEPSIS" in html and 'id="fig-graduation-symptoms"' in html and 'id="fig-graduation-shown"' in html
