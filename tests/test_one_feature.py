"""One feature a round, two actions (simlab/one_feature.py): the solution's single-feature schedule, and correlated
states that break it."""
import itertools

import numpy as np
import pytest

from simlab import one_feature

PSI = 0.6 ** np.arange(60)
D, G0 = np.array([1.0, 3.0]), np.array([2.0, -3.0])          # the solution's worked example


def test_the_worked_example_matches_the_solution():
    assert one_feature.flip_counts(D, G0, PSI).tolist() == [0, 2]
    for delta, scores, value, pick in [(0.9, [1, 1.86], 5.195, 1), (0.7, [1, -0.06], 2.875, 0)]:
        assert one_feature.scores(D, G0, PSI, delta) == pytest.approx(scores)
        assert one_feature.solution_value(D, G0, PSI, delta) == pytest.approx(value, abs=1e-3)
        best, schedule = one_feature.optimal(one_feature.costs(D, G0, PSI), delta, 20)
        assert best == pytest.approx(one_feature.solution_value(D, G0, PSI, delta)) and (schedule == pick).all()


def test_the_cost_of_a_round_is_the_average_over_states():
    Sigma = np.array([[1.0, 0.9], [0.9, 1.0]])
    for sigma in (None, Sigma):
        c = one_feature.costs([1.0, -0.5], [-1.0, 1.5], 0.8 ** np.arange(12), sigma)
        schedule = [1] * 8 + [0] * 4
        simulated = one_feature.simulate([1.0, -0.5], [-1.0, 1.5], 0.8 ** np.arange(12), schedule, 200_000, sigma)
        assert simulated == pytest.approx(one_feature.schedule_costs(c, schedule), abs=6e-3)


def test_the_search_is_the_best_of_every_schedule():
    rng = np.random.default_rng(3)
    for _ in range(5):
        d, g0, delta, Sigma = rng.standard_normal(3), rng.standard_normal(3), 0.8, np.full((3, 3), 0.6) + 0.4 * np.eye(3)
        c = one_feature.costs(d, g0, PSI, Sigma)
        best, _ = one_feature.optimal(c, delta, 1)
        # every schedule of 8 rounds, each continued with the best single feature from there
        least = np.inf
        for order in itertools.product(range(3), repeat=8):
            head = one_feature.schedule_costs(c, order) @ delta ** np.arange(8)
            m = np.bincount(order, minlength=3)
            tail = min(one_feature.constant_values(c[j:j + 1, min(m[j], c.shape[1] - 1):], delta)[0] for j in range(3))
            least = min(least, head + delta ** 8 * tail)
        assert best <= least + 1e-9


def test_independent_worlds_show_one_feature_forever():
    rng = np.random.default_rng(0)
    for _ in range(200):
        d, g0, delta = rng.standard_normal(3), rng.standard_normal(3), rng.uniform(0.6, 0.98)
        psi = rng.uniform(0.3, 0.8) ** np.arange(400)
        best, schedule = one_feature.optimal(one_feature.costs(d, g0, psi), delta, 40)
        assert best == pytest.approx(one_feature.solution_value(d, g0, psi, delta), rel=1e-12)
        assert (schedule == schedule[0]).all()


def test_a_proxy_feature_makes_the_plan_switch():
    d, g0, psi, delta = np.array([1.0, -0.5]), np.array([-1.0, 1.5]), 0.8 ** np.arange(60), 0.9
    c = one_feature.costs(d, g0, psi, np.array([[1.0, 0.9], [0.9, 1.0]]))
    # x2 starts with the sign of (Sigma d)_2 > 0 and learns the sign of d_2 < 0: its cost goes up
    assert c[1, 0] < c[1, -1]
    best, schedule = one_feature.optimal(c, delta, 12)
    assert schedule.tolist() == [1] * 7 + [0] * 5
    # the plan's own cost is the value the search reports
    assert one_feature.schedule_costs(c, schedule) @ delta ** np.arange(12) + c[0, -1] * delta ** 12 / (1 - delta) == pytest.approx(best)
    assert best < one_feature.constant_values(c, delta).min() - 0.4


# the human who fills in the hidden features from the shown one

SIGMA2 = np.array([[1.0, 0.8], [0.8, 1.0]])


def test_filling_in_with_independent_features_changes_nothing():
    rng = np.random.default_rng(1)
    for _ in range(20):
        d, g0, delta = rng.standard_normal(3), rng.standard_normal(3), 0.9
        C = one_feature.imputed_costs(d, g0, PSI, np.eye(3), one_feature.imputed_caps(d, g0, PSI, np.eye(3)))
        assert one_feature.optimal_grid(C, delta, 1)[0] == pytest.approx(one_feature.solution_value(d, g0, PSI, delta))


def test_the_cost_of_a_filled_in_round_is_the_average_over_states():
    d, g0 = [1.0, 0.2], [1.0, -3.0]
    C = one_feature.imputed_costs(d, g0, PSI, SIGMA2, one_feature.imputed_caps(d, g0, PSI, SIGMA2))
    schedule = [1, 1, 0, 0, 1, 0, 1, 1, 1, 0]
    simulated = one_feature.simulate(d, g0, PSI, schedule, 200_000, SIGMA2, impute=True)
    assert simulated == pytest.approx(one_feature.grid_schedule_costs(C, schedule), abs=6e-3)


def test_the_search_over_every_count_is_the_best_of_every_schedule():
    d, g0, delta = np.array([1.0, 0.3]), np.array([1.5, -2.5]), 0.9
    psi = 0.7 ** np.arange(60)
    C = one_feature.imputed_costs(d, g0, psi, SIGMA2, one_feature.imputed_caps(d, g0, psi, SIGMA2))
    best, plan = one_feature.optimal_grid(C, delta, 400)
    cost = lambda schedule: one_feature.grid_schedule_costs(C, schedule) @ delta ** np.arange(len(schedule))
    tail = delta ** 400 / (1 - delta) * one_feature.grid_schedule_costs(C, plan)[-1]
    assert cost(plan) + tail == pytest.approx(best)
    # every way to spend the first 10 rounds, each followed by the best from the counts it leaves
    least = np.inf
    for head in itertools.product(range(2), repeat=10):
        m = np.minimum(np.bincount(head, minlength=2), np.array(C.shape[1:]) - 1)
        least = min(least, cost(list(head)) + delta ** 10 * one_feature.optimal_grid(C[:, m[0]:, m[1]:], delta, 1)[0])
    assert least == pytest.approx(best)


def test_filling_in_a_backwards_feature_spoils_the_solutions_pick():
    d, g0, delta = np.array([1.0, 0.2]), np.array([1.0, -3.0]), 0.9
    C = one_feature.imputed_costs(d, g0, PSI, SIGMA2, one_feature.imputed_caps(d, g0, PSI, SIGMA2))
    assert one_feature.scores(d, g0, PSI, delta).argmax() == 0
    # x1 is never wrong itself, but x2 is never taught, so filling it in keeps the pick wrong
    J = one_feature.grid_constant_values(C, delta)
    assert J[0] == pytest.approx(C[0, 0, 0] / (1 - delta)) and C[0, 0, 0] > np.sqrt(d @ SIGMA2 @ d) / np.sqrt(2 * np.pi)
    best, plan = one_feature.optimal_grid(C, delta, 10)
    assert plan.tolist() == [1, 1] + [0] * 8 and best < J.min() - 1
