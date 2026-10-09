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
    # the regret of a round is taken at the state drawn for it: |du| |x_t| on each of the 21 wrong rounds, then zero
    import numpy as np
    x, regret = np.array(data["series"]["x"]), np.array(data["series"]["regret"])
    np.testing.assert_allclose(regret[:21], 0.2 * np.abs(x[:21]), atol=2e-4)
    assert set(regret[21:]) == {0.0} and len(set(regret[:21])) == 21
    # beside it, the same loss averaged over every state: |du| E|x| = 0.2 * sqrt(2/pi) on a wrong round
    assert data["series"]["regret_mean"][:21] == [0.1596] * 21 and set(data["series"]["regret_mean"][21:]) == {0.0}
    # Fig. 4 weights round t by delta^t (delta defaults to 0.95), and the tile sums those on this draw
    assert data["delta"] == 0.95
    np.testing.assert_allclose(data["series"]["discounted"], regret * 0.95 ** np.arange(80), atol=1e-4)
    assert abs(sum(data["series"]["discounted"]) - float(data["stats"][3]["value"])) < 0.01
    assert data["stats"][3]["note"].startswith("2.10 averaged over states")
    # u1 > u2, so action 1 is best in a positive state and action 2 in a negative one; the human picks the
    # other one until the flip and the best one after it
    best, choice = np.array(data["series"]["best"]), np.array(data["series"]["choice"])
    np.testing.assert_array_equal(best, np.where(x > 0, 1, 2))
    assert (choice[:21] != best[:21]).all() and (choice[21:] == best[21:]).all()
    # another draw moves the regret but not the beliefs, the flip or the average
    other = client.get("/api/run?u1=1.2&u2=1&h1=0&h2=1.5&curve=exponential&p1=0.1&T=80&seed=5").get_json()
    assert other["series"]["x"] != data["series"]["x"] and other["series"]["regret"] != data["series"]["regret"]
    for key in ("h1", "h2", "regret_mean", "value_gap_mean"):
        assert other["series"][key] == data["series"][key], key
    assert other["settle"] == 21 and other["runs"] == data["runs"]
    grid = data["map"]["bins"]
    assert len(grid) == 60 and len(grid[0]) == 60
    assert grid[0][-1] == 0 and grid[-1][0] == 0          # same-sign corners: right from the start
    assert grid[0][0] > 0 and grid[-1][-1] > 0            # opposite-sign corners: a flip is needed


def test_one_feature_has_one_curve_for_both_actions(client):
    import numpy as np
    # a per-action speed is no longer a thing: both beliefs close the same share of their gap each round
    query = "u1=1&u2=0.5&h1=3&h2=2.8&curve=exponential&p1=0.3&T=80"
    data = client.get("/api/run?" + query).get_json()
    assert data == client.get("/api/run?" + query + "&p2=0.03&split=1").get_json()
    h1, h2 = np.array(data["series"]["h1"]), np.array(data["series"]["h2"])
    np.testing.assert_allclose((h1 - 1) / 2, (h2 - 0.5) / 2.3, atol=1e-4)
    # so the gap never changes sign more than once: this world starts right and stays right
    assert data["settle"] == 0 and [run["wrong"] for run in data["runs"]] == [False]
    assert all(len(client.get("/api/run?u1=1&u2=0.5&h1=0&h2=2&curve=" + curve).get_json()["runs"]) <= 2
               for curve in ("exponential", "hyperbolic", "power%20law", "sigmoid"))
    page = client.get("/").data.decode()
    assert 'id="split"' not in page and 'data-key="p2"' not in page


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
    assert 0 <= min(data["series"]["reg"]) and max(data["series"]["reg"]) <= data["regret_cap"]
    # on this draw and on average over states alike
    assert best["mask"] == [True, True, False] and best["discounted"] < now["discounted"] and best["mean"] < now["mean"]
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
    assert one_by_two["series"]["acc"][0] == 0                                        # wrong sign at round 0


def test_value_gap_matches_a_monte_carlo_average(client):
    import numpy as np
    u, h = np.array([1.0, 0.5]), np.array([3.0, 2.8])
    data = client.get("/api/run?u1=1&u2=0.5&h1=3&h2=2.8&curve=exponential&p1=0.3&T=40").get_json()
    x = np.random.default_rng(0).standard_normal(400_000)
    for t in (0, 5, 39):
        h_t = np.array([data["series"]["h1"][t], data["series"]["h2"][t]])
        # true best value minus the value the human expects from the action they pick
        sampled = (np.outer(x, u).max(axis=1) - np.outer(x, h_t).max(axis=1)).mean()
        assert abs(sampled - data["series"]["value_gap_mean"][t]) < 0.01
    # a human whose two beliefs are as far apart as the truths, but in the wrong order: wrong choice, zero value gap
    flipped = client.get("/api/run?u1=1&u2=0&h1=0&h2=1&curve=exponential&p1=0.1&T=20").get_json()
    assert flipped["series"]["value_gap_mean"][0] == 0 and flipped["series"]["regret"][0] > 0


def test_light_answer_matches_the_full_one_for_the_chosen_policy(client):
    query = "/api/world?" + DEFAULT_3X3.replace("A=1,1,1", "A=1,0,1") + "&F=1,1,0"
    full, light = client.get(query).get_json(), client.get(query + "&rank=0").get_json()
    assert full["full"] and not light["full"] and light["rankings"] is None
    for key in ("numeral", "series", "stats", "acc_limit", "reg_limit", "regret_cap", "value_cap", "beliefs", "choice", "x"):
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
    assert max(abs(v) for v in series["value_gap_mean"]) <= data["value_cap"]


def test_value_gap_of_the_general_scorer_matches_the_one_feature_closed_form():
    import numpy as np
    from simlab import analysis
    X = np.random.default_rng(0).standard_normal((200_000, 1))
    U, H = np.array([[1.0], [0.5]]), np.array([[3.0], [2.8]])
    _, _, gap = analysis.evaluate(H, U, X, value_gap=True)
    assert abs(gap - (0.5 - 0.2) * analysis.MEAN_ABS_STD_NORMAL / 2) < 0.005


def test_regret_choice_and_value_gap_are_taken_at_the_state_drawn_each_round(client):
    import numpy as np
    from simlab import draws
    data = client.get("/api/world?" + DEFAULT_3X3 + "&F=1,1,0&seed=3").get_json()
    U, x = np.array(data["U"]), np.array(data["x"])                       # x: [round][feature]
    np.testing.assert_allclose(x, draws.states(3, 100, 3), atol=5e-4)
    beliefs = np.array(data["beliefs"])                                    # [feature][action][round]
    choice, series = data["choice"], data["series"]
    worth = x @ U.T                                                        # [round][action]: what each action is truly worth at x_t
    scores = np.einsum("jkt,tj->tk", beliefs[:2], x[:, :2])                # the human sees features 1 and 2 only
    np.testing.assert_array_equal(choice["best"], worth.argmax(axis=1) + 1)
    np.testing.assert_array_equal(choice["picks"], scores.argmax(axis=1) + 1)
    picks = np.array(choice["picks"]) - 1
    # the proposal's loss: max_k (U x_t)_k - (U x_t)_yhat, zero exactly when the pick is the best move
    regret = worth.max(axis=1) - worth[np.arange(100), picks]
    np.testing.assert_allclose(series["reg"], regret, atol=5e-3)
    assert choice["missed"] == (regret > 1e-9).tolist() and 0 < sum(choice["missed"]) < 100
    np.testing.assert_allclose(series["value_gap"], worth.max(axis=1) - scores.max(axis=1), atol=5e-3)
    # the headline is the discounted sum on this draw, and the bench's row of each table carries it
    assert data["numeral"] == f"{sum(series['discounted']):.2f}" == f"{(regret * 0.95 ** np.arange(100)).sum():.2f}"
    now = next(r for r in data["rankings"]["features"]["rows"] if r["current"])
    assert f"{now['discounted']:.2f}" == data["numeral"] and f"{now['mean']:.2f}" == data["stats"][0]["value"]
    # every policy is run on the same draw: the tables sort by that, and carry the average over states beside it
    rows = data["rankings"]["features"]["rows"]
    assert [r["discounted"] for r in rows] == sorted(r["discounted"] for r in rows) and all("mean" in r for r in rows)
    # another draw changes the loss of the rounds, not the beliefs or the averages over states
    other = client.get("/api/world?" + DEFAULT_3X3 + "&F=1,1,0&seed=4").get_json()
    assert other["series"]["reg"] != series["reg"] and other["numeral"] != data["numeral"]
    for key in ("reg_mean", "value_gap_mean", "acc"):
        assert other["series"][key] == series[key], key
    assert other["beliefs"] == data["beliefs"] and other["stats"][0] == data["stats"][0]
    # with action 1 withheld the human can never pick it, whatever the state
    without = client.get("/api/world?" + DEFAULT_3X3.replace("A=1,1,1", "A=0,1,1") + "&F=1,1,1&rank=0").get_json()
    assert 1 not in without["choice"]["picks"] and 1 in without["choice"]["best"]
    # beliefs that cannot tell the actions apart share the regret evenly
    blank = client.get("/api/world?K=2&n=1&U=1;0&H=0;0&T=10&rank=0").get_json()
    assert abs(blank["series"]["reg"][0] - abs(blank["x"][0][0]) / 2) < 1e-3 and blank["choice"]["picks"][0] == 1


def test_the_drawn_states_are_standard_normal_and_reproducible():
    import numpy as np
    from simlab import draws
    x = draws.states(0, 40_000, 3)
    assert abs(x.mean()) < 0.01 and abs(x.std() - 1) < 0.01
    assert np.abs(np.corrcoef(x.T) - np.eye(3)).max() < 0.02                       # features are independent
    assert abs(np.corrcoef(x[:-1, 0], x[1:, 0])[0, 1]) < 0.02                      # and so are rounds
    np.testing.assert_array_equal(draws.states(7, 5, 2), draws.states(7, 5, 2))
    assert not np.allclose(draws.states(7, 5, 2), draws.states(8, 5, 2))


def test_a_window_pauses_learning_and_it_resumes_where_it_stopped(client):
    import numpy as np
    plain = client.get("/api/world?" + DEFAULT_3X3).get_json()
    timed = client.get("/api/world?" + DEFAULT_3X3 + "&W=f3:20-25").get_json()
    was, now = np.array(plain["beliefs"]), np.array(timed["beliefs"])    # [feature][action][round]
    # feature 3 is hidden in rounds 20 to 25: its weights are the same from round 20 up to round 26 ...
    np.testing.assert_array_equal(now[2][:, 20:27], np.repeat(now[2][:, 20:21], 7, axis=1))
    # ... then carry on six rounds behind the world with no window; before the window nothing differs
    np.testing.assert_array_equal(now[2][:, 26:], was[2][:, 20:94])
    np.testing.assert_array_equal(now[:, :, :21], was[:, :, :21])
    assert timed["schedule"]["F"][2] == [1] * 20 + [0] * 6 + [1] * 74 and timed["changes"] == [20, 26]
    assert timed["schedule"]["A"] == [[1] * 100] * 3 and plain["changes"] == []
    # inside the window the human decides exactly as if feature 3 had never been shown
    hidden = client.get("/api/world?" + DEFAULT_3X3 + "&F=1,1,0").get_json()
    assert timed["series"]["reg"][20:26] == hidden["series"]["reg"][20:26]
    assert timed["series"]["reg"][:20] == plain["series"]["reg"][:20]
    assert "feature 3 hidden in rounds 20–25" in timed["sentence"]
    # the bench is no fixed feature subset, so it is ranked next to all seven of them
    features, actions = timed["rankings"]["features"], timed["rankings"]["actions"]
    bench = next(r for r in features["rows"] if r["current"])
    assert features["total"] == 8 and bench["scheduled"] and bench["mask"] is None
    assert f"{bench['discounted']:.2f}" == timed["numeral"]
    # no action window: among the action subsets the bench is still "all 3 actions"
    assert actions["total"] == 7 and next(r for r in actions["rows"] if r["current"])["mask"] == [True] * 3


def test_windows_over_the_whole_horizon_match_the_fixed_subset_and_bad_ones_are_refused(client):
    whole = client.get("/api/world?" + DEFAULT_3X3 + "&W=f3:0-99;a2:0-500").get_json()
    fixed = client.get("/api/world?" + DEFAULT_3X3.replace("A=1,1,1", "A=1,0,1") + "&F=1,1,0").get_json()
    for key in ("numeral", "series", "stats", "beliefs", "choice", "acc_limit", "reg_limit", "schedule"):
        assert whole[key] == fixed[key], key
    assert whole["rankings"]["features"]["total"] == 7 and whole["changes"] == []
    # a window that only reaches the end leaves the feature in use before it, and says so
    tail = client.get("/api/world?" + DEFAULT_3X3 + "&W=f3:80-99&rank=0").get_json()
    without = client.get("/api/world?" + DEFAULT_3X3 + "&F=1,1,0&rank=0").get_json()
    assert "feature 3 is hidden at the end" in tail["sentence"] and tail["reg_limit"] == without["reg_limit"] > 0
    # a window past the horizon changes nothing
    late = client.get("/api/world?" + DEFAULT_3X3 + "&W=a1:100-120&rank=0").get_json()
    assert late["series"] == client.get("/api/world?" + DEFAULT_3X3 + "&rank=0").get_json()["series"]
    for bad in ("f1:5-9;f2:5-9;f3:5-9", "a1:0-3;a2:2-2;a3:1-9", "f4:1-2", "f1:9-5", "f1", ";".join(["f1:1-2"] * 13)):
        assert client.get("/api/world?" + DEFAULT_3X3 + "&W=" + bad).status_code == 400, bad


def test_timed_page_and_the_page_menu(client):
    page = client.get("/timed").data.decode()
    assert 'id="add-window"' in page and 'id="fig-schedule"' in page and 'window.BENCH_MODE = "timed"' in page
    assert '"W": "f3:20-25"' in page
    assert '<a href="/timed" aria-current="page">Timed hiding</a>' in page
    plain = client.get("/worlds").data.decode()
    assert 'id="add-window"' not in plain and 'window.BENCH_MODE = "fixed"' in plain
    # every page offers the Overview and every experiment, and the menu names the experiment you are on
    for path, name in (("/", "1 feature, 2 actions"), ("/worlds", "More actions and features"), ("/timed", "Timed hiding"),
                       ("/aware", "State-aware subsets"), ("/correlated", "Correlated features"),
                       ("/experiments", "Experiment 1 figures")):
        html = client.get(path).data.decode()
        assert f'<summary class="current" title="Go to an experiment">{name}</summary>' in html
        assert f'<a href="{path}" aria-current="page"' in html and '<a href="/overview" >Overview</a>' in html
        for other in ("/", "/worlds", "/timed", "/aware", "/correlated", "/experiments"):
            assert f'<a href="{other}" ' in html
    overview = client.get("/overview").data.decode()
    assert '<a href="/overview" aria-current="page">Overview</a>' in overview
    assert '<summary  title="Go to an experiment">Experiments</summary>' in overview


def test_each_feature_can_have_its_own_learning_speed(client):
    import numpy as np
    shared = client.get("/api/world?" + DEFAULT_3X3).get_json()
    # the same speed named three times is the shared curve
    assert client.get("/api/world?" + DEFAULT_3X3 + "&ps=0.05,0.05,0.05").get_json() == shared
    data = client.get("/api/world?" + DEFAULT_3X3.replace("H=1,0.5,1.5;0,1,0;-0.5,0,-1.5", "H=0,0,0;0,0,0;0,0,0") + "&ps=0.02,0.1,0.4&rank=0").get_json()
    beliefs, U = np.array(data["beliefs"]), np.array(data["U"])              # [feature][action][round]
    t = np.arange(100)
    for j, rate in enumerate((0.02, 0.1, 0.4)):
        # every action's weight on feature j closes its gap at feature j's rate
        np.testing.assert_allclose(beliefs[j], U[:, j:j + 1] * (1 - (1 - rate) ** t), atol=1e-3)
    # the speeds are clamped to the curve's range, and there must be one per feature
    assert client.get("/api/world?" + DEFAULT_3X3 + "&ps=9,0.05,0.05&rank=0").get_json()["series"] == \
        client.get("/api/world?" + DEFAULT_3X3 + "&ps=0.5,0.05,0.05&rank=0").get_json()["series"]
    for bad in ("0.1,0.2", "0.1,x,0.2", "0.1,0.2,0.3,0.4"):
        assert client.get("/api/world?" + DEFAULT_3X3 + "&ps=" + bad).status_code == 400, bad
    # the correlated bench takes them too
    corr = ("/api/correlated?K=3&U=-1.5,0.5,0;0.5,1.5,-1;1,-1.5,0&H=0,0,0;0,0,0;0,0,0&rho=0.8,0.8,0.6&k=2&explore=12"
            "&C=1,1,0&curve=exponential&T=60&rank=0")
    assert client.get(corr + "&p1=0.15").get_json() == client.get(corr + "&ps=0.15,0.15,0.15").get_json()
    slow = np.array(client.get(corr + "&ps=0.15,0.15,0.01").get_json()["beliefs"])
    fast = np.array(client.get(corr + "&p1=0.15").get_json()["beliefs"])
    np.testing.assert_array_equal(slow[:2], fast[:2])
    assert np.abs(slow[2]).max() < np.abs(fast[2]).max()                      # feature 3 has barely moved
    for path in ("/worlds", "/timed", "/aware", "/correlated"):
        assert 'id="each-speed"' in client.get(path).data.decode()


def test_every_bench_lists_the_same_six_figures_under_the_same_numbers(client):
    import re
    from webapp import guide
    titles = [figure["title"] for figure in guide.STANDARD_FIGURES]
    assert [figure["n"] for figure in guide.STANDARD_FIGURES] == [1, 2, 3, 4, 5, 6]
    own = {"/": 2, "/worlds": 0, "/timed": 0, "/aware": 1, "/correlated": 3}
    for path, extra in own.items():
        html = client.get(path).data.decode()
        captions = re.findall(r"<figcaption><b>(?:<a [^>]*>)?Fig\. (\d+)(?:</a>)?</b>(.*?)(?:<span|</figcaption>)", html, re.S)
        numbers = [int(n) for n, _ in captions]
        # the six standard figures first, in order, each under its shared title; then the bench's own, numbered on
        assert numbers == list(range(1, 7 + extra)), path
        assert [title.strip() for _, title in captions[:6]] == titles, path
        assert html.count("Not drawn on this bench") == (1 if path == "/" else 0)
        for n in range(1, 7):
            assert f'href="/overview#fig-{n}"' in html
    # the one-feature bench has nothing to schedule: Fig. 1 keeps its number and title and says why
    lab = client.get("/").data.decode()
    assert 'id="fig-schedule"' not in lab and "both are in play in every" in lab and 'id="fig-accuracy"' in lab
    # the saved report is numbered as plates, so "Fig. 1" always means the schedule
    assert "Fig. 1" not in client.get("/experiments").data.decode()


def test_overview_sets_out_the_equations_and_the_standard_figures(client):
    from webapp import guide
    html = client.get("/overview").data.decode()
    for equation in guide.EQUATIONS:
        assert f'id="eq-{equation["id"]}"' in html and f"<b>Eq. {equation['n']}</b>" in html
    # no derivation is written yet; each equation says so where its link will go
    assert html.count("Derivation not written yet") == len(guide.EQUATIONS)
    for figure in guide.STANDARD_FIGURES:
        assert f'id="fig-{figure["n"]}"' in html and figure["title"] in html
        assert all(guide.EQUATION[e] for e in figure["equations"])
    # the specimens are drawn by the benches' own code, for the Timed hiding default
    assert 'id="fig-schedule"' in html and 'id="fig-beliefs"' in html and "overview.js" in html and '"W": "f3:20-25"' in html
    # a bench that does not draw a figure says why, here as on the bench itself
    assert guide.BENCHES[0]["absent"]["schedule"] in html
    # a derivation, once written, is linked
    guide.EQUATION["regret"]["derivation"] = "/derivations/regret"
    try:
        linked = client.get("/overview").data.decode()
        assert '<a class="derivation" href="/derivations/regret">Derivation</a>' in linked
        assert linked.count("Derivation not written yet") == len(guide.EQUATIONS) - 1
    finally:
        guide.EQUATION["regret"]["derivation"] = None


def test_correlated_bench_takes_the_loss_at_the_drawn_state(client):
    import numpy as np
    from simlab import draws
    base = ("/api/correlated?K=3&U=-1.5,0.5,0;0.5,1.5,-1;1,-1.5,0&H=-1.5,-1,1.5;-0.5,0.5,1.5;1.5,-1.5,-0.5"
            "&k=2&explore=0&C=1,1,0&curve=exponential&p1=0.15&T=60")
    data = client.get(base + "&rho=0,0,0&rank=0&seed=2").get_json()
    choice, U, x = data["choice"], np.array(data["U"]), np.array(data["x"])
    np.testing.assert_allclose(x, draws.states(2, 60, 3), atol=5e-4)          # independent features: the draw itself
    worth = x @ U.T
    np.testing.assert_array_equal(choice["best"], worth.argmax(axis=1) + 1)
    # with independent features a hidden one is filled in as zero, so the human scores with the shown ones alone
    beliefs = np.array(data["beliefs"])                                        # [feature][action][round]
    scores = np.einsum("jkt,tj->tk", beliefs[:2], x[:, :2])
    np.testing.assert_array_equal(choice["picks"], scores.argmax(axis=1) + 1)
    regret = worth.max(axis=1) - worth[np.arange(60), np.array(choice["picks"]) - 1]
    np.testing.assert_allclose(data["series"]["reg"], regret, atol=5e-3)
    assert data["numeral"] == f"{sum(data['series']['discounted']):.2f}" and choice["missed"] == (regret > 1e-9).tolist()
    # correlated states are the same standard normals, mixed: x_t = L z_t
    mixed = np.array(client.get(base + "&rho=0.8,0.8,0.6&rank=0&seed=2").get_json()["x"])
    L = np.linalg.cholesky(np.array([[1, 0.8, 0.8], [0.8, 1, 0.6], [0.8, 0.6, 1]]))
    np.testing.assert_allclose(mixed, draws.states(2, 60, 3) @ L.T, atol=5e-4)
    # the ranking is on this draw, with the average over states beside it
    ranked = client.get(base + "&rho=0.8,0.8,0.6&seed=2").get_json()["ranking"]
    assert [r["discounted"] for r in ranked["rows"]] == sorted(r["discounted"] for r in ranked["rows"])
    assert all(r["mean"] >= 0 for r in ranked["rows"])


def test_state_aware_policy_shows_a_subset_that_makes_the_human_right(client):
    import itertools
    import numpy as np
    # judged by the human's own weights: a subset works when the human, as they are now, picks the best move
    data = client.get("/api/world?" + DEFAULT_3X3 + "&policy=aware&judge=belief&prefer=most&C=2&seed=5").get_json()
    from simlab import draws
    U, x = np.array(data["U"]), draws.states(5, 100, 3)                                      # the states, unrounded
    shown = np.array(data["schedule"]["F"]).T.astype(bool)                                   # [round][feature]
    # the beliefs of each round, unrounded: a feature's weights close 5% of their gap each round it was shown
    H0 = np.array([[1, 0.5, 1.5], [0, 1, 0], [-0.5, 0, -1.5]])
    before = np.vstack([np.zeros(3), np.cumsum(shown, axis=0)[:-1]])                         # rounds shown before t
    beliefs = (U[:, :, None] + (H0 - U)[:, :, None] * (0.95 ** before.T)[None]).transpose(1, 0, 2)   # [feature][action][round]
    np.testing.assert_allclose(beliefs, data["beliefs"], atol=6e-4)
    policy, choice, reg = data["policy"], data["choice"], np.array(data["series"]["reg"])
    candidates = [set(c) for size in (1, 2) for c in itertools.combinations(range(3), size)]
    assert policy["budget"] == 2 and policy["candidates"] == len(candidates) == 6
    assert shown.sum(axis=1).min() >= 1 and shown.sum(axis=1).max() <= 2                     # never more than C features
    assert np.array(data["schedule"]["A"]).all()                                             # every action is offered

    def works(t, subset):
        """With the beliefs of round t, the best move scores strictly above every other on these features alone."""
        scores = sum(beliefs[i, :, t] * x[t, i] for i in subset)
        best = int((U @ x[t]).argmax())
        return all(scores[best] > scores[k] for k in range(3) if k != best)

    def by_truth(t, subset):
        """The same test with the true weights: the hidden features are not needed to see which move is best."""
        scores = sum(U[:, i] * x[t, i] for i in subset)
        best = int((U @ x[t]).argmax())
        return all(scores[best] > scores[k] for k in range(3) if k != best)

    for t in range(100):
        count = sum(works(t, c) for c in candidates)
        assert count == policy["worked"][t], t
        if count:
            # a subset that works was shown, so the human picks the best move and the round costs nothing
            assert works(t, set(np.flatnonzero(shown[t]))) and choice["picks"][t] == choice["best"][t] and reg[t] == 0
            # with several to choose from it is one the truth agrees on, if there is one, and then the largest
            agreed = [c for c in candidates if works(t, c) and by_truth(t, c)]
            pool = agreed or [c for c in candidates if works(t, c)]
            assert set(np.flatnonzero(shown[t])) in pool and shown[t].sum() == max(len(c) for c in pool)
        else:
            assert reg[t] > 0 and choice["picks"][t] != choice["best"][t]        # nothing works: a random subset, and a loss
    found = sum(1 for count in policy["worked"] if count)
    assert 0 < found < 100 and data["stats"][0]["value"] == f"{found} of 100"
    assert f"in {found} of 100 rounds" in data["sentence"] and "a random one was shown" in data["sentence"]
    # only what is shown is learned: a feature's weights move exactly after the rounds it was shown in
    moved = np.abs(np.diff(beliefs, axis=2)).max(axis=1) > 0                                 # [feature][round]
    assert not moved[~shown.T[:, :-1]].any()
    # it is ranked against the fixed subsets within the same budget, on the same draw, and beats them here
    rows = data["rankings"]["features"]["rows"]
    bench = next(r for r in rows if r["current"])
    assert data["rankings"]["actions"] is None and data["rankings"]["features"]["total"] == 7
    assert bench["scheduled"] and bench["rank"] == 1 and f"{bench['discounted']:.2f}" == data["numeral"]
    assert all(sum(r["mask"]) <= 2 for r in rows if not r["scheduled"])
    # a human with blank beliefs cannot be steered: no subset works, a random one is shown, and the pick is a guess
    blank = client.get("/api/world?K=3&n=3&U=1,0.5,0;0,1,0.5;-0.5,0,1&H=0,0,0;0,0,0;0,0,0&policy=aware&judge=belief&C=1&T=20&rank=0").get_json()
    assert blank["policy"]["worked"][0] == 0 and abs(blank["series"]["acc"][0] - 1 / 3) < 1e-3
    page = client.get("/aware").data.decode()
    assert 'window.BENCH_MODE = "aware"' in page and 'id="fig-worked"' in page and 'id="feature-chips"' not in page
    assert "If no subset works, it shows one at random, for now." in page


def test_equally_good_moves_both_count_as_correct(client):
    import numpy as np
    # actions 1 and 2 are worth the same in every state, so picking either is correct
    world = "K=3&n=3&U=1,1,0;1,1,0;0,0,1&H=1,0.5,1.5;0,1,0;-0.5,0,-1.5&curve=exponential&p1=0.05&T=300&delta=0.95"
    data = client.get("/api/world?" + world + "&policy=aware&judge=belief&C=2&rank=0").get_json()
    worked, reg, acc = np.array(data["policy"]["worked"]), np.array(data["series"]["reg"]), data["series"]["acc"]
    # once the beliefs have reached the truth, a subset that makes the human right exists in almost every state
    # (with at most 2 of 3 features a few states have none), and the policy stops falling back to a random one
    assert np.abs(np.array(data["beliefs"])[:, :, -1].T - np.array(data["U"])).max() < 0.05
    assert acc[-1] > 0.95 and acc[-1] > acc[0] and (worked[-100:] > 0).mean() > 0.9
    assert reg[(worked > 0)].max() == 0
    # the best move shown for a round is the human's pick whenever that pick is one of the equally good ones
    picks, best = np.array(data["choice"]["picks"]), np.array(data["choice"]["best"])
    np.testing.assert_array_equal(picks[reg == 0], best[reg == 0])
    assert (picks[reg > 0] != best[reg > 0]).all() and {1, 2} <= set(picks.tolist())
    # showing all three features, the human ends up right in every state on every bench that scores accuracy
    full = client.get("/api/world?" + world + "&rank=0").get_json()
    assert full["acc_limit"] == 1 and full["series"]["acc"][-1] > 0.99
    # one feature, two equally good actions: never wrong, and the two lines of Fig. 3 coincide
    flat = client.get("/api/run?u1=1&u2=1&h1=0.5&h2=2&curve=exponential&p1=0.1&T=30").get_json()
    assert flat["series"]["choice"] == flat["series"]["best"] and set(flat["series"]["regret"]) == {0.0}


def test_state_aware_policy_judged_by_the_truth_shows_the_widest_margin_and_has_a_ceiling(client):
    import itertools
    import numpy as np
    from simlab import draws
    # the world of the question: action 2 is best only when x1 < 0, by a hair, and needs x1 and x2 together to show it
    world = "K=3&n=3&U=1.1,1,0;1,1,0;0,0,1&H=0,0,1;1,1,0;1,0,0&curve=exponential&p1=0.3&T=140&delta=0.95"
    data = client.get("/api/world?" + world + "&policy=aware&judge=truth&C=2&seed=1").get_json()
    U, x, policy = np.array(data["U"]), draws.states(1, 140, 3), data["policy"]
    shown = np.array(data["schedule"]["F"]).T.astype(bool)
    assert policy["judge"] == "truth" and policy["prefer"] == "margin"
    candidates = [list(c) for size in (1, 2) for c in itertools.combinations(range(3), size)]
    for t in range(140):
        a = int((U @ x[t]).argmax())
        # by the truth, on the shown features alone: the best move's score minus the best rival's
        margins = [U[a, F] @ x[t, F] - max(U[k, F] @ x[t, F] for k in range(3) if k != a) for F in candidates]
        working = [m for m in margins if m > 1e-9]
        assert len(working) == policy["worked"][t], t
        if working:       # the top-ranked subset is shown: the widest margin
            assert abs(margins[candidates.index(np.flatnonzero(shown[t]).tolist())] - max(working)) < 1e-9, t
    # the truth decides what is shown, so the schedule does not depend on what the human believes ...
    other = client.get("/api/world?" + world.replace("H=0,0,1;1,1,0;1,0,0", "H=2,-1,0;0,0,0;1,1,1") + "&policy=aware&judge=truth&C=2&seed=1&rank=0").get_json()
    assert other["schedule"] == data["schedule"] and other["policy"]["worked"] == policy["worked"]
    # ... and the human can be wrong on a subset that works, until they have learned it
    reg, worked = np.array(data["series"]["reg"]), np.array(policy["worked"])
    assert reg[:10][worked[:10] > 0].max() > 0 and reg[60:][worked[60:] > 0].max() == 0
    assert "the true weights rank the best move first" in data["sentence"]
    # The ceiling. With one feature a state where action 2 is best can never be settled: x1 alone points to action 3
    # and x2 or x3 alone cannot tell action 2 from action 1. P(x1 < 0, x1 + x2 > x3) = 1/4 - arcsin(1/sqrt 3) / 2 pi.
    lost = 0.25 - np.arcsin(1 / np.sqrt(3)) / (2 * np.pi)
    for C, expected in ((1, 1 - lost), (3, 1.0)):
        for judge in ("truth", "belief"):        # the ceiling belongs to U and C, not to the rule or the beliefs
            got = client.get("/api/world?" + world + f"&policy=aware&C={C}&judge={judge}&rank=0").get_json()
            assert abs(got["policy"]["ceiling"] - expected) < 0.02, (C, judge)
            assert got["stats"][2]["label"] == "States a subset can settle"
    assert client.get("/api/world?" + world + "&policy=aware&C=3&rank=0").get_json()["acc_limit"] == 1
    # By default the human's own weights judge: the choice in Fig. 3 is then wrong only where no subset works,
    # whatever the budget, and the errors stop once the human has learned enough
    for C in (1, 2, 3):
        got = client.get("/api/world?" + world + f"&policy=aware&C={C}&rank=0").get_json()
        missed, worked = np.array(got["choice"]["missed"]), np.array(got["policy"]["worked"])
        assert got["policy"]["judge"] == "belief" and not missed[worked > 0].any() and missed[worked == 0].all()
        assert f"{int((worked > 0).sum())} of 140" in got["stats"][0]["value"]
    # the page says how the search scales and what the fallback is
    page = client.get("/aware").data.decode()
    assert "The search is exponential." in page and "hitting-set" in page and 'name="judge"' in page
