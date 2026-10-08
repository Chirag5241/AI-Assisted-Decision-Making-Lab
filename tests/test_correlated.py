import numpy as np
import pytest

from simlab import correlated
from webapp.app import DEFAULT_CORRELATED, app


@pytest.fixture
def client():
    return app.test_client()


def query(world=DEFAULT_CORRELATED, **changes):
    w = {**world, **changes}
    rows = lambda M: ";".join(",".join(str(v) for v in row) for row in M)
    return ("/api/correlated?" + f"K={len(w['U'])}&U={rows(w['U'])}&H={rows(w['H'])}&rho={','.join(map(str, w['rho']))}"
            f"&k={w['k']}&explore={w['explore']}&C={','.join(str(int(c)) for c in w['C'])}"
            f"&curve={w['curve']}&p1={w['p1']}&T={w['T']}&delta={w['delta']}")


def test_imputation_is_the_conditional_mean():
    Sigma = correlated.correlation_matrix(0.8, 0.5, 0.3)
    X = correlated.probes(Sigma, 400_000)
    np.testing.assert_allclose(np.cov(X.T), Sigma, atol=0.01)
    P = correlated.imputation(Sigma, [True, False, True])
    # filled-in x2 is the least-squares prediction of x2 from x1 and x3
    coef, *_ = np.linalg.lstsq(X[:, [0, 2]], X[:, 1], rcond=None)
    np.testing.assert_allclose(P[1, [0, 2]], coef, atol=0.01)
    np.testing.assert_array_equal(P[:, 1], 0)
    np.testing.assert_array_equal(P[[0, 2]][:, [0, 2]], np.eye(2))


def test_independent_features_give_the_independent_bench(client):
    world = dict(DEFAULT_CORRELATED, rho=[0, 0, 0], explore=0)
    mine = client.get(query(world) + "&rank=0").get_json()
    rows = lambda M: ";".join(",".join(str(v) for v in row) for row in M)
    theirs = client.get(f"/api/world?K=3&n=3&U={rows(world['U'])}&H={rows(world['H'])}&F=1,1,0&A=1,1,1"
                        f"&curve=exponential&p1={world['p1']}&T={world['T']}&delta={world['delta']}&rank=0").get_json()
    assert mine["numeral"] == theirs["numeral"]
    np.testing.assert_allclose(mine["series"]["acc"], theirs["series"]["acc"])
    np.testing.assert_allclose(mine["series"]["reg"], theirs["series"]["reg"])
    assert mine["reg_limit"] == pytest.approx(theirs["reg_limit"])


def test_exploring_first_beats_every_fixed_subset_in_the_default_world(client):
    data = client.get(query()).get_json()
    ranking = data["ranking"]
    best, fixed = ranking["best"], ranking["best_fixed"]
    # on the states drawn, and by a wide margin on average over all states
    assert best["explore"] > 0 and best["discounted"] < fixed["discounted"] and best["mean"] < 0.6 * fixed["mean"]
    assert ranking["retained"] == pytest.approx(best["discounted"] / fixed["discounted"])
    # committing to the same pair without exploring leaves feature 3 misjudged for good
    assert fixed["mask"] == best["mask"] == [True, True, False]
    assert fixed["floor"] > best["floor"] + 0.1
    # the bench is the ranked policy it claims to be, and its number is the ranking's
    now = next(r for r in ranking["rows"] if r["current"])
    assert now["explore"] == 12 and data["numeral"] == f"{now['discounted']:.2f}"
    assert f"{now['mean']:.2f}" == data["stats"][0]["value"]
    # the explored feature is shown during exploration only
    schedule = np.array(data["schedule"])
    assert schedule.shape == (3, 120) and schedule[:, :12].sum(axis=0).tolist() == [2] * 12
    assert schedule[2, 12:].sum() == 0 and schedule[:2, 12:].all()
    # and its weights stop moving once it is hidden
    beliefs = np.array(data["beliefs"])                       # [feature][action][round]
    assert np.ptp(beliefs[2, :, 12:], axis=1).max() == 0 and np.ptp(beliefs[2, :, :13], axis=1).min() > 0


def test_correlation_makes_a_hidden_misjudged_feature_cost(client):
    fixed = dict(DEFAULT_CORRELATED, explore=0)
    independent = client.get(query(fixed, rho=[0, 0, 0]) + "&rank=0").get_json()
    correlated_ = client.get(query(fixed) + "&rank=0").get_json()
    assert correlated_["reg_limit"] > independent["reg_limit"] + 0.1


def test_light_answer_matches_the_full_one(client):
    full, light = client.get(query()).get_json(), client.get(query() + "&rank=0").get_json()
    assert light["ranking"] is None and full["ranking"] is not None
    for key in ("numeral", "series", "schedule", "beliefs", "effective", "scatter", "reg_limit", "acc_limit"):
        assert light[key] == full[key], key
    assert full["sentence"].startswith(light["sentence"])


def test_validation(client):
    assert client.get(query(rho=[0.9, 0.9, -0.9])).status_code == 400          # not a correlation matrix
    assert client.get(query(rho=[0.99, 0, 0])).status_code == 400              # outside the sliders
    assert client.get(query(C=[1, 1, 1])).status_code == 400                   # over the budget of 2
    assert client.get(query(k=1, C=[1, 1, 0])).status_code == 400
    assert client.get(query(k=1, C=[0, 0, 1])).status_code == 200


def test_page_and_navigation(client):
    page = client.get("/correlated")
    assert page.status_code == 200 and b"correlated.js" in page.data and b'aria-current="page">Correlated' in page.data
    assert b'href="/correlated"' in client.get("/worlds").data
