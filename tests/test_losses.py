"""The losses of a round (simlab/losses.py): alternatives to regret, after Jiang et al. (2014)."""
import itertools

import numpy as np
import pytest

from simlab import losses


def test_each_loss_on_one_round_worked_by_hand():
    true = np.array([3.0, 1.0, 0.0])
    # the human ranks the actions 2, 3, 1: the worst possible top, and two of the three pairs wrong
    out = losses.of_round(np.array([0.0, 2.0, 1.0]), true)
    assert out["regret"] == 2 and out["wrong"] == 1 and out["pairs"] == 2
    assert out["ranked"] == (3 - 1) + (3 - 0) + 0          # the best action left, minus the one placed, at each place
    p = np.exp([0.0, 2.0, 1.0]) / np.exp([0.0, 2.0, 1.0]).sum()
    assert out["noisy"] == pytest.approx(p @ (3 - true)) and out["log"] == pytest.approx(-np.log(p[0]))
    # the true ranking costs nothing by the first four, and the noisy two still charge for the margin
    right = losses.of_round(true, true)
    assert [right[key] for key in ("regret", "wrong", "pairs", "ranked")] == [0, 0, 0, 0]
    assert 0 < right["noisy"] < out["noisy"] and 0 < right["log"] < out["log"]
    # a wider margin for the best move lowers both; tau sets how fast
    assert losses.of_round(2 * true, true)["log"] < right["log"] < losses.of_round(true, true, tau=2.0)["log"]
    assert set(losses.NAMES) == set(out)


def test_ties_are_shared():
    out = losses.of_round(np.array([1.0, 1.0, 0.0]), np.array([3.0, 1.0, 0.0]))
    assert out["regret"] == 1 and out["wrong"] == 0.5 and out["pairs"] == 0.5


def test_pairs_is_the_kendall_tau_distance_and_ranked_regret_starts_with_regret():
    rng = np.random.default_rng(0)
    scores, true = rng.standard_normal((4, 500)), rng.standard_normal((4, 500))
    out = losses.of_round(scores, true)
    swapped = sum((scores[k] > scores[l]) != (true[k] > true[l]) for k, l in itertools.combinations(range(4), 2))
    np.testing.assert_array_equal(out["pairs"], swapped)
    assert (out["ranked"] >= out["regret"] - 1e-12).all() and (out["wrong"] == (out["regret"] > 0)).all()
    # with two actions the first three agree up to the size of the gap, and ranked regret is regret
    two = losses.of_round(scores[:2], true[:2])
    np.testing.assert_array_equal(two["wrong"], two["pairs"])
    np.testing.assert_allclose(two["ranked"], two["regret"])
    np.testing.assert_allclose(two["regret"], two["wrong"] * np.abs(true[0] - true[1]))
