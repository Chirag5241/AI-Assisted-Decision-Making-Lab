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
    data = client.get("/api/run?u1=1&u2=0.5&h1=3&h2=2.8&curve=exponential&p1=0.3&T=40").get_json()
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
    assert 'id="add-window"' in page and 'id="fig-schedule"' in page and "window.TIMED_BENCH = true" in page
    assert '"W": "f3:20-25"' in page
    assert '<a href="/timed" aria-current="page">Timed hiding</a>' in page
    plain = client.get("/worlds").data.decode()
    assert 'id="add-window"' not in plain and "window.TIMED_BENCH = false" in plain
    # every page offers the Overview and every experiment, and the menu names the experiment you are on
    for path, name in (("/", "1 feature, 2 actions"), ("/worlds", "More actions and features"), ("/timed", "Timed hiding"),
                       ("/correlated", "Correlated features"), ("/experiments", "Experiment 1 figures")):
        html = client.get(path).data.decode()
        assert f'<summary class="current" title="Go to an experiment">{name}</summary>' in html
        assert f'<a href="{path}" aria-current="page"' in html and '<a href="/overview" >Overview</a>' in html
        for other in ("/", "/worlds", "/timed", "/correlated", "/experiments"):
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
    for path in ("/worlds", "/timed", "/correlated"):
        assert 'id="each-speed"' in client.get(path).data.decode()


def test_every_bench_lists_the_same_six_figures_under_the_same_numbers(client):
    import re
    from webapp import guide
    titles = [figure["title"] for figure in guide.STANDARD_FIGURES]
    assert [figure["n"] for figure in guide.STANDARD_FIGURES] == [1, 2, 3, 4, 5, 6]
    own = {"/": 2, "/worlds": 0, "/timed": 0, "/correlated": 3}
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


def test_correlated_bench_reports_the_choice_for_the_reference_state(client):
    import numpy as np
    data = client.get("/api/correlated?K=3&U=-1.5,0.5,0;0.5,1.5,-1;1,-1.5,0&H=-1.5,-1,1.5;-0.5,0.5,1.5;1.5,-1.5,-0.5"
                      "&rho=0,0,0&k=2&explore=0&C=1,1,0&curve=exponential&p1=0.15&T=60&rank=0").get_json()
    choice, U = data["choice"], np.array(data["U"])
    assert choice["x"] == [1, 1, 1] and choice["best"] == int(U.sum(axis=1).argmax()) + 1
    # with independent features a hidden one is filled in as zero, so the scores are the shown beliefs summed
    beliefs = np.array(data["beliefs"])                              # [feature][action][round]
    np.testing.assert_array_equal(choice["picks"], beliefs[:2].sum(axis=0).argmax(axis=0) + 1)
    assert choice["missed"] == [bool(U.sum(axis=1)[k - 1] < U.sum(axis=1).max() - 1e-9) for k in choice["picks"]]
