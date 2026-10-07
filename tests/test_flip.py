import numpy as np
import pytest

from simlab import CountLearner, DeltaRuleLearner, FixedMasks, analysis, curves, simulate

CURVES = [curves.exponential(0.1), curves.hyperbolic(5.0), curves.power_law(0.5), curves.sigmoid(20.0, 4.0)]


def random_worlds(B, seed=1):
    rng = np.random.default_rng(seed)
    return rng.uniform(-3, 3, (B, 2, 1)), rng.uniform(-3, 3, (B, 2, 1))


@pytest.mark.parametrize("curve", CURVES, ids=lambda c: c.name)
def test_curve_is_a_learning_curve(curve):
    phi = curve(np.arange(5000))
    assert phi[0] == 0
    assert np.all(np.diff(phi) >= 0)
    assert phi[-1] <= 1 and curve(1e9) > 0.999


@pytest.mark.parametrize("curve", CURVES, ids=lambda c: c.name)
def test_simulated_flip_matches_closed_form(curve):
    T = 150
    U, H0 = random_worlds(2000)
    res = simulate(U, H0, CountLearner(curve), T=T)
    du, dh0 = (U[:, 0] - U[:, 1])[:, 0], (H0[:, 0] - H0[:, 1])[:, 0]

    theory = analysis.theory_flip_time(du, dh0, curve, T - 1)
    np.testing.assert_array_equal(analysis.first_correct(res.correct), theory)
    # a shared curve makes h1 - h2 monotone, so the flip is permanent
    np.testing.assert_array_equal(analysis.settle_time(res.correct), theory)
    assert analysis.n_switches(res.correct).max() <= 1

    # the threshold form: a wrong-signed human is right at round t  <=>  phi(t) > phi*
    phi_star = analysis.flip_threshold(du, dh0)
    already_right = du * dh0 > 0
    np.testing.assert_array_equal(res.correct, (curve(np.arange(T))[:, None] > phi_star[None, :]) | already_right)


def test_regret_is_gap_times_abs_x_until_the_flip():
    U, H0 = random_worlds(500)
    res = simulate(U, H0, CountLearner(curves.exponential(0.1)), T=60)
    du = (U[:, 0] - U[:, 1])[:, 0]
    expected = np.where(res.correct, 0.0, np.abs(du)[None, :] * np.abs(res.x[:, :, 0]))
    np.testing.assert_allclose(res.regret, expected, atol=1e-12)


def test_normalized_delta_rule_equals_exponential_curve_for_one_feature():
    U, H0 = random_worlds(200)
    a = simulate(U, H0, CountLearner(curves.exponential(0.2)), T=50)
    b = simulate(U, H0, DeltaRuleLearner(eta=0.2), T=50)
    np.testing.assert_allclose(a.H, b.H, atol=1e-9)


def test_per_action_speeds_can_unflip_a_correct_human():
    # starts right (h1 > h2, u1 > u2), but h1 falls to its truth much faster than h2 does
    curve = curves.exponential(np.array([[0.3], [0.03]]))
    res = simulate([[1.0], [0.5]], [[3.0], [2.8]], CountLearner(curve), T=100)
    assert res.correct[0, 0] and not res.correct[1, 0]
    assert analysis.n_switches(res.correct)[0] == 2
    assert analysis.settle_time(res.correct)[0] == 51


def test_evaluate_is_the_flip_indicator_in_one_dimension():
    U, H0 = random_worlds(50)
    res = simulate(U, H0, CountLearner(curves.hyperbolic(5.0)), T=40)
    acc, _ = analysis.evaluate(res.H[:-1], res.U, np.random.default_rng(0).standard_normal((64, 1)))
    np.testing.assert_array_equal(acc, res.correct.astype(float))


def test_evaluate_with_masks_and_ties():
    U = np.array([[1.0, 0.0], [0.0, 1.0]])
    X = np.random.default_rng(0).standard_normal((4000, 2))
    # knowing the truth on everything shown: perfect with both features, imperfect with one hidden
    assert analysis.evaluate(U, U, X)[0] == 1.0
    acc_hidden, regret_hidden = analysis.evaluate(U, U, X, F=np.array([True, False]))
    assert 0.5 < acc_hidden < 1.0 and regret_hidden > 0
    # a human with no opinion ties every action: a coin toss between two actions
    acc_tie, _ = analysis.evaluate(np.zeros((2, 2)), U, X)
    assert acc_tie == 0.5
    # an action that is never offered can never be picked
    acc_one, _ = analysis.evaluate(U, U, X, A=np.array([True, False]))
    assert abs(acc_one - 0.5) < 0.03


def test_hidden_entries_are_never_learned():
    U, H0 = np.ones((1, 2, 3)), np.zeros((1, 2, 3))
    F, A = np.array([[True, False, True]]), np.array([[True, False]])
    res = simulate(U, H0, CountLearner(curves.exponential(0.5)), T=30, policy=FixedMasks(F, A))
    np.testing.assert_allclose(res.H[-1, 0, 0], [1.0, 0.0, 1.0], atol=1e-6)   # shown action: shown features learned
    np.testing.assert_array_equal(res.H[-1, 0, 1], [0.0, 0.0, 0.0])            # action never offered


@pytest.mark.parametrize("curve", CURVES + [curves.exponential(np.array([[0.3], [0.03]]))], ids=lambda c: c.name)
def test_closed_form_trajectory_matches_the_simulation(curve):
    U, H0 = random_worlds(300)
    learner = CountLearner(curve)
    np.testing.assert_allclose(learner.trajectory(U, H0, 60), simulate(U, H0, learner, T=60).H[:60], atol=1e-12)
