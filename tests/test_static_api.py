"""The static build answers the data endpoints in JavaScript (webapp/static/static-api.js).
These tests run that script under Node and compare its answers with the Flask endpoints."""
import json
import shutil
import subprocess
from pathlib import Path

import pytest

from webapp.app import app

STATIC = Path(__file__).resolve().parent.parent / "webapp" / "static"
pytestmark = pytest.mark.skipif(shutil.which("node") is None or not (STATIC / "probes.js").exists(),
                                reason="needs Node and a built webapp/static/probes.js (python build_pages.py)")

W = "K=3&n=3&U=1,0.5,0;0,1,0.5;-0.5,0,1&H=1,0.5,1.5;0,1,0;-0.5,0,-1.5"
QUERIES = [
    "/api/run?u1=1.2&u2=1&h1=0&h2=1.5&curve=exponential&p1=0.1&T=80&delta=0.95",
    "/api/run?u1=1&u2=0.5&h1=3&h2=2.8&curve=exponential&p1=0.3&p2=0.03&split=1&T=80&delta=0.9",
    "/api/run?u1=2&u2=0&h1=0.5&h2=0&curve=sigmoid&p1=20&T=120",
    "/api/run?u1=-0.4&u2=1.1&h1=2.5&h2=-2&curve=power%20law&p1=0.5&T=400&delta=0.5",
    "/api/run?u1=1&u2=1&h1=0&h2=2&curve=hyperbolic&p1=5&T=40",
    f"/api/world?{W}&F=1,1,1&A=1,1,1&curve=exponential&p1=0.05&T=100&delta=0.95",
    f"/api/world?{W}&F=1,1,0&A=1,0,1&curve=exponential&p1=0.05&T=100&delta=0.95",
    f"/api/world?{W}&F=1,0,1&A=1,1,1&curve=sigmoid&p1=20&T=60&delta=0.8&rank=0",
    "/api/world?K=2&n=1&U=1;0&H=0;1&curve=hyperbolic&p1=5&T=50",
    "/api/world?K=4&n=2&U=1,2;-1,0.5;0,0;2,-2&H=0,0;0,0;0,0;0,0&F=1,1&A=1,1,0,1&curve=power%20law&p1=0.5&T=40",
    "/api/world?K=2&n=2&U=1,2;3,4&H=0,0",          # wrong shape: both sides refuse it
    "/api/run?curve=bogus",
    "/api/correlated?K=3&U=-1.5,0.5,0;0.5,1.5,-1;1,-1.5,0&H=-1.5,-1,1.5;-0.5,0.5,1.5;1.5,-1.5,-0.5"
    "&rho=0.8,0.8,0.6&k=2&explore=12&C=1,1,0&curve=exponential&p1=0.15&T=40&delta=0.97",
    "/api/correlated?K=2&U=1,0,0;0,1,0&H=0,0,0;0,0,0&rho=0,0,0&k=1&explore=0&C=0,0,1"
    "&curve=hyperbolic&p1=5&T=30&delta=0.9&rank=0",
    "/api/correlated?rho=0.9,0.9,-0.9",            # not a correlation matrix: both sides refuse it
]

DRIVER = """
require(process.argv[1] + "/probes.js");
const api = require(process.argv[1] + "/static-api.js");
(async () => {
  const out = [];
  for (const url of JSON.parse(process.argv[2])) {
    const [path, query] = url.split("?");
    try {
      const q = new URLSearchParams(query);
      out.push(path === "/api/run" ? api.run(q) : await (path === "/api/correlated" ? api.correlated(q) : api.world(q)));
    }
    catch (error) { if (error instanceof api.BadRequest) out.push({ status: 400 }); else throw error; }
  }
  console.log(JSON.stringify(out));
})();
"""


@pytest.fixture(scope="module")
def answers():
    done = subprocess.run(["node", "-e", DRIVER, str(STATIC), json.dumps(QUERIES)], capture_output=True, text=True, check=True)
    return json.loads(done.stdout)


def same(a, b, where=""):
    """Equal structure and text; numbers equal to what float32 probe states allow."""
    if isinstance(a, dict):
        assert isinstance(b, dict) and a.keys() == b.keys(), f"{where}: keys {sorted(a)} vs {sorted(b)}"
        for key in a:
            same(a[key], b[key], f"{where}.{key}")
    elif isinstance(a, list):
        assert isinstance(b, list) and len(a) == len(b), f"{where}: length {len(a)} vs {len(b)}"
        for i, (x, y) in enumerate(zip(a, b)):
            same(x, y, f"{where}[{i}]")
    elif isinstance(a, bool) or a is None or isinstance(a, str):
        assert a == b, f"{where}: {a!r} vs {b!r}"
    else:
        assert abs(a - b) <= 2e-3, f"{where}: {a} vs {b}"


@pytest.mark.parametrize("index", range(len(QUERIES)), ids=lambda i: QUERIES[i].split("&")[0][:40] + f" #{i}")
def test_browser_answer_matches_the_server(answers, index):
    response = app.test_client().get(QUERIES[index])
    if response.status_code == 400:
        assert answers[index] == {"status": 400}
    else:
        same(response.get_json(), answers[index], QUERIES[index][:24])
