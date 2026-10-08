import pytest

from webapp.app import app


@pytest.fixture
def client():
    return app.test_client()


def test_default_world_flips_at_round_21(client):
    data = client.get("/api/run?u1=1.2&u2=1&h1=0&h2=1.5&curve=exponential&p1=0.1&T=80").get_json()
    assert data["settle"] == 21 and data["numeral"] == "21"
    assert data["runs"] == [dict(start=0, stop=21, wrong=True), dict(start=21, stop=80, wrong=False)]
    assert data["stats"][0]["value"] == "88%"
    assert len(data["series"]["h1"]) == 80 and data["series"]["h1"][0] == 0 and data["series"]["h2"][0] == 1.5
    # regret loss: |du| E|x| = 0.2 * sqrt(2/pi) on each of the 21 wrong rounds, then zero
    assert data["series"]["regret"][:21] == [0.1596] * 21 and set(data["series"]["regret"][21:]) == {0.0}
    # Fig. 2 weights round t by delta^t (delta defaults to 0.95), and the headline sums those
    assert data["delta"] == 0.95
    assert data["series"]["discounted"][:3] == [0.1596, round(0.1596 * 0.95, 4), round(0.159577 * 0.95 ** 2, 4)]
    assert abs(sum(data["series"]["discounted"]) - float(data["stats"][3]["value"])) < 0.01
    # u1 > u2, so action 1 is best; the human picks action 2 until the flip
    assert data["best"] == 1 and data["series"]["choice"] == [2] * 21 + [1] * 59
    grid = data["map"]["bins"]
    assert len(grid) == 60 and len(grid[0]) == 60
    assert grid[0][-1] == 0 and grid[-1][0] == 0          # same-sign corners: right from the start
    assert grid[0][0] > 0 and grid[-1][-1] > 0            # opposite-sign corners: a flip is needed


def test_per_action_speeds_report_the_unflip(client):
    query = "u1=1&u2=0.5&h1=3&h2=2.8&curve=exponential&p1=0.3&p2=0.03&split=1&T=80"
    data = client.get("/api/run?" + query).get_json()
    assert data["settle"] == 51
    assert [run["wrong"] for run in data["runs"]] == [False, True, False]


def test_bad_input_is_rejected_or_clamped(client):
    assert client.get("/api/run?curve=bogus").status_code == 400
    assert client.get("/api/run?u1=abc&T=1e9&delta=nan").get_json()["T"] == 400


def test_pages_render(client):
    assert b"AI-Assisted Decision Making Lab" in client.get("/").data
    assert client.get("/experiments").status_code == 200


DEFAULT_3X3 = ("K=3&n=3&U=1,0.5,0;0,1,0.5;-0.5,0,1&H=1,0.5,1.5;0,1,0;-0.5,0,-1.5"
               "&A=1,1,1&curve=exponential&p1=0.05&T=100&delta=0.95")


def test_hiding_a_misjudged_feature_beats_showing_everything(client):
    data = client.get("/api/world?" + DEFAULT_3X3 + "&F=1,1,1").get_json()
    ranking = data["rankings"]["features"]
    best, now = ranking["rows"][0], next(r for r in ranking["rows"] if r["current"])
    assert ranking["total"] == 7
    # showing everything is the reference itself, so no reference line; the regret axis ceiling covers the data
    assert len(data["series"]["acc"]) == 100 and data["series"]["ref_acc"] is None
    assert max(data["series"]["reg"]) <= data["worst_regret"]
    assert best["mask"] == [True, True, False] and best["discounted"] < now["discounted"]
    assert now["floor"] == 0 and best["floor"] > 0          # hiding is cheaper now but costs forever
    assert now["end"] > now["start"] and best["end"] == best["start"]


def test_action_subsets_are_ranked_alongside_feature_subsets(client):
    data = client.get("/api/world?" + DEFAULT_3X3 + "&F=1,1,1").get_json()
    features, actions = data["rankings"]["features"], data["rankings"]["actions"]
    assert actions["total"] == 7 and len(actions["rows"]) == 7
    # the world on the bench is the same policy in both tables
    now_f = next(r for r in features["rows"] if r["current"])
    now_a = next(r for r in actions["rows"] if r["current"])
    assert now_a["mask"] == [True, True, True] and now_a["discounted"] == now_f["discounted"]
    # offering a single action leaves the human no choice, so nothing changes with learning
    for r in actions["rows"]:
        if sum(r["mask"]) == 1:
            assert r["start"] == r["end"] and r["floor"] > 0
    # dropping an action is applied when it is put on the bench
    two = client.get("/api/world?" + DEFAULT_3X3.replace("A=1,1,1", "A=1,1,0") + "&F=1,1,1").get_json()
    row = next(r for r in actions["rows"] if r["mask"] == [True, True, False])
    assert two["numeral"] == f"{row['discounted']:.2f}"
    assert "action 3 is never offered" in two["sentence"]


def test_larger_world_page_and_validation(client):
    assert client.get("/worlds").status_code == 200
    assert client.get("/api/world?K=2&n=2&U=1,2;3,4&H=0,0").status_code == 400      # wrong shape
    assert client.get("/api/world?K=2&n=1&U=1;0&H=0;1&F=0").status_code == 400        # nothing shown
    one_by_two = client.get("/api/world?K=2&n=1&U=1;0&H=0;1").get_json()
    assert one_by_two["rankings"]["features"]["total"] == 1
    assert one_by_two["stats"][0]["value"] == "0%"                                    # wrong sign at round 0


def test_value_gap_matches_a_monte_carlo_average(client):
    import numpy as np
    u, h = np.array([1.0, 0.5]), np.array([3.0, 2.8])
    data = client.get("/api/run?u1=1&u2=0.5&h1=3&h2=2.8&curve=exponential&p1=0.3&p2=0.03&split=1&T=40").get_json()
    x = np.random.default_rng(0).standard_normal(400_000)
    for t in (0, 5, 39):
        h_t = np.array([data["series"]["h1"][t], data["series"]["h2"][t]])
        # true best value minus the value the human expects from the action they pick
        sampled = (np.outer(x, u).max(axis=1) - np.outer(x, h_t).max(axis=1)).mean()
        assert abs(sampled - data["series"]["value_gap"][t]) < 0.01
    # a human whose two beliefs are as far apart as the truths, but in the wrong order: wrong choice, zero value gap
    flipped = client.get("/api/run?u1=1&u2=0&h1=0&h2=1&curve=exponential&p1=0.1&T=20").get_json()
    assert flipped["series"]["value_gap"][0] == 0 and flipped["series"]["regret"][0] > 0


def test_light_answer_matches_the_full_one_for_the_chosen_policy(client):
    query = "/api/world?" + DEFAULT_3X3.replace("A=1,1,1", "A=1,0,1") + "&F=1,1,0"
    full, light = client.get(query).get_json(), client.get(query + "&rank=0").get_json()
    assert full["full"] and not light["full"] and light["rankings"] is None
    for key in ("numeral", "series", "stats", "acc_limit", "reg_limit", "worst_regret", "value_bound", "beliefs", "choice"):
        assert light[key] == full[key], key
    assert full["sentence"].startswith(light["sentence"])


def test_larger_world_beliefs_value_gap_and_discounting(client):
    import numpy as np
    data = client.get("/api/world?" + DEFAULT_3X3 + "&F=1,1,0&rank=0").get_json()
    beliefs = np.array(data["beliefs"])                      # [feature][action][round]
    assert beliefs.shape == (3, 3, 100)
    np.testing.assert_allclose(beliefs[:, :, 0].T, [[1, 0.5, 1.5], [0, 1, 0], [-0.5, 0, -1.5]])
    # the shown features are learned; the hidden feature 3 never is, so its weights stay at the first beliefs
    assert data["F"] == [True, True, False]
    np.testing.assert_allclose(beliefs[:2, :, -1].T, np.array(data["U"])[:, :2], atol=0.02)
    np.testing.assert_array_equal(beliefs[2], np.repeat(beliefs[2][:, :1], 100, axis=1))
    # the same holds for an action that is not offered: none of its weights move
    dropped = np.array(client.get("/api/world?" + DEFAULT_3X3.replace("A=1,1,1", "A=1,0,1") + "&rank=0").get_json()["beliefs"])
    np.testing.assert_array_equal(dropped[:, 1], np.repeat(dropped[:, 1, :1], 100, axis=1))
    assert np.ptp(dropped[2, [0, 2]], axis=1).min() > 0.4     # feature 3 is the misjudged one, and is learned where offered
    # the scores treat feature 3 as hidden too: the human never gets everything right
    assert data["acc_limit"] < 0.9
    series = data["series"]
    np.testing.assert_allclose(series["discounted"], np.array(series["reg"]) * 0.95 ** np.arange(100), atol=1e-4)
    assert max(abs(v) for v in series["value_gap"]) <= data["value_bound"]


def test_value_gap_of_the_general_scorer_matches_the_one_feature_closed_form():
    import numpy as np
    from simlab import analysis
    X = np.random.default_rng(0).standard_normal((200_000, 1))
    U, H = np.array([[1.0], [0.5]]), np.array([[3.0], [2.8]])
    _, _, gap = analysis.evaluate(H, U, X, value_gap=True)
    assert abs(gap - (0.5 - 0.2) * analysis.MEAN_ABS_STD_NORMAL / 2) < 0.005


def test_choice_for_the_all_ones_state(client):
    import numpy as np
    data = client.get("/api/world?" + DEFAULT_3X3 + "&F=1,1,1&rank=0").get_json()
    U = np.array(data["U"])
    choice = data["choice"]
    assert choice["x"] == [1, 1, 1] and choice["best"] == int(U.sum(axis=1).argmax()) + 1 == 1
    # first beliefs sum to (3, 1, -2) per action, so the human starts on action 1, which is also best here
    beliefs = np.array(data["beliefs"])                                  # [feature][action][round]
    np.testing.assert_array_equal(choice["picks"], beliefs.sum(axis=0).argmax(axis=0) + 1)
    # with action 1 withheld the human can never pick the best move for this state
    without = client.get("/api/world?" + DEFAULT_3X3.replace("A=1,1,1", "A=0,1,1") + "&F=1,1,1&rank=0").get_json()
    assert without["choice"]["best"] == 1 and 1 not in without["choice"]["picks"]
