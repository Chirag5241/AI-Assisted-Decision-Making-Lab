"""Planning what to show before the state is drawn (simlab/planning.py): two actions, x ~ N(0, I)."""
import itertools

import numpy as np
import pytest

from simlab import planning

D, G0 = np.array([1.0, -0.5, 1.5]), np.array([1.0, -0.5, -1.5])        # the human has feature 3 backwards


def test_expected_regret_is_the_average_over_normal_states():
    x = np.random.default_rng(0).standard_normal((1_000_000, 3))
    for g in ([0.3, -1.0, 0.0], [1.0, -0.5, -1.5], [0.0, 0.0, 2.0], [-1.0, 0.5, -1.5]):
        wrong = np.sign(x @ g) != np.sign(x @ D)
        assert planning.expected_regret(g, D) == pytest.approx((np.abs(x @ D) * wrong).mean(), abs=4e-3), g
    always_wrong = np.sqrt(3.5 * 2 / np.pi)                               # E|d.x|
    assert planning.expected_regret(D, D) == pytest.approx(0) and planning.expected_regret(-D, D) == pytest.approx(always_wrong)
    # only the direction of the believed gap counts, and a human with none guesses
    assert planning.expected_regret(7 * np.array([0.3, -1, 0]), D) == pytest.approx(planning.expected_regret([0.3, -1, 0], D))
    assert planning.expected_regret(np.zeros(3), D) == pytest.approx(always_wrong / 2)
    # one feature: right sign or wrong sign, whatever the size
    assert planning.expected_regret([[0, 0, 0.01], [0, 0, -9]], D).tolist() == pytest.approx(
        [(np.sqrt(3.5) - 1.5) / np.sqrt(2 * np.pi), (np.sqrt(3.5) + 1.5) / np.sqrt(2 * np.pi)])


def test_candidates_are_ordered_by_size_then_by_feature():
    assert planning.candidates(3, 2).astype(int).tolist() == [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [1, 0, 1], [0, 1, 1]]
    assert len(planning.candidates(3, 3)) == 7 and len(planning.candidates(2, 1)) == 2


@pytest.mark.parametrize("budget,delta,rate", [(1, 0.9, 0.5), (2, 0.9, 0.5), (2, 0.6, 0.3), (3, 0.99, 0.4)])
def test_the_optimal_plan_matches_a_search_of_every_schedule(budget, delta, rate):
    T = 5
    phi = np.tile(1 - (1 - rate) ** np.arange(T), (3, 1))
    made = planning.plan(D, G0, phi, budget, delta)
    sets, weight = made["sets"], delta ** np.arange(T)

    def cost(order):
        m, total = np.zeros(3, int), 0.0
        for t, i in enumerate(order):
            believed = D + (G0 - D) * (1 - phi[np.arange(3), m])
            total += weight[t] * planning.expected_regret(believed * sets[i], D)
            m += sets[i]
        return total

    least = min(cost(order) for order in itertools.product(range(len(sets)), repeat=T))
    assert made["optimal"]["loss"] @ weight == pytest.approx(least, abs=1e-7)
    # each account is the schedule it goes with, and a fixed subset is one of the schedules searched
    for key in ("optimal", "myopic"):
        order = [int(np.flatnonzero((sets == shown).all(axis=1))[0]) for shown in made[key]["schedule"]]
        assert made[key]["loss"] @ weight == pytest.approx(cost(order))
    for i in range(len(sets)):
        assert made["fixed"][i] @ weight == pytest.approx(cost([i] * T)) and cost([i] * T) >= least - 1e-9
    assert not made["myopic"]["gave_up"].any() and made["myopic"]["loss"] @ weight >= least - 1e-9


def test_patience_decides_whether_the_backwards_feature_is_taught():
    T = 100
    phi = np.tile(1 - 0.9 ** np.arange(T), (3, 1))
    shown = lambda delta: planning.plan(D, G0, phi, 2, delta)["optimal"]
    hasty, patient = shown(0.8), shown(0.95)
    assert (hasty["schedule"] == [True, True, False]).all() and not hasty["gave_up"].any()
    assert patient["schedule"][:, 2].all() and patient["gave_up"][:7].all() and not patient["gave_up"][7:].any()
    # a round given up loses more than a guess would; afterwards the plan loses less than the myopic rule ever does
    assert patient["loss"][0] > planning.expected_regret(np.zeros(3), D) > hasty["loss"][0] > patient["loss"][-1]
