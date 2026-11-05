"""The companion site's distributions (distributions/dist.js), checked under Node: every pdf has total mass 1,
every CDF is the running integral of its pdf, every sampler draws with the stated mean and spread,
and every arrow on the map joins two distributions the site knows."""
import json
import shutil
import subprocess
from pathlib import Path

import pytest

DIST = Path(__file__).resolve().parent.parent / "distributions" / "dist.js"
pytestmark = pytest.mark.skipif(shutil.which("node") is None, reason="needs Node")

CHECK = r"""
require(process.argv[1]);
const {list, cdf, defaults, rng, edges, byId} = DIST;
const out = {bad_edges: edges.filter(e => !byId[e.from] || !byId[e.to]).map(e => e.from + "->" + e.to),
             unlinked: list.filter(d => !edges.some(e => e.from === d.id || e.to === d.id)).map(d => d.id), rows: []};
for (const d of list) {
  const p = defaults(d); d.fix && d.fix(p);
  let mass = 0, cdfErr = 0;
  if (d.kind === "discrete") {
    for (let k = -200; k <= 400; k++) mass += d.pdf(k, p);
    for (let k = d.x[0]; k <= d.x[1]; k++) { let s = 0; for (let j = -200; j <= k; j++) s += d.pdf(j, p); cdfErr = Math.max(cdfErr, Math.abs(s - cdf(d, k, p))); }
  } else {
    // midpoint rule on a wide grid; probes sit inside the plotted range
    const lo = -400, hi = 400, n = 400000, h = (hi - lo) / n;
    const probes = [0.25, 0.5, 0.75].map(f => d.x[0] + f * (d.x[1] - d.x[0]));
    let acc = 0, pi = 0, below = 0;
    for (let i = 0; i < n; i++) {
      const x = lo + (i + 0.5) * h;
      while (pi < probes.length && x > probes[pi]) { below = Math.max(below, Math.abs(acc - cdf(d, probes[pi], p))); pi++; }
      const f = d.pdf(x, p);
      if (Number.isFinite(f)) acc += f * h;
    }
    mass = acc; cdfErr = below;
  }
  const r = rng(1), N = 200000; let s = 0, s2 = 0;
  for (let i = 0; i < N; i++) { const v = d.sample(r, p); s += v; s2 += v * v; }
  const m = s / N;
  out.rows.push({id: d.id, mass, cdfErr, mean: d.mean(p), sd: Math.sqrt(d.variance(p)), sampleMean: m, sampleSd: Math.sqrt(s2 / N - m * m)});
}
console.log(JSON.stringify(out));
"""

# mass that the integration grid cannot see: singular edges (arcsine) and tails past +-400 (Levy, Cauchy)
LOOSE = {"arcsine": 0.05, "levy": 0.05, "cauchy": 0.01, "pareto": 0.01}


@pytest.fixture(scope="module")
def report():
    run = subprocess.run(["node", "-e", CHECK, str(DIST)], capture_output=True, text=True, check=True)
    return json.loads(run.stdout)


def test_every_arrow_joins_two_known_distributions_and_none_is_left_alone(report):
    assert report["bad_edges"] == [] and report["unlinked"] == []
    assert len(report["rows"]) == 35


def test_every_pdf_has_mass_one_and_its_cdf_is_its_integral(report):
    for row in report["rows"]:
        tol = LOOSE.get(row["id"], 0.003)
        assert abs(row["mass"] - 1) < tol, row
        assert row["cdfErr"] < max(tol, 0.003), row


def test_every_sampler_draws_with_the_stated_mean_and_spread(report):
    for row in report["rows"]:
        if row["mean"] is not None and abs(row["mean"]) != float("inf"):       # Cauchy and Levy have no mean
            assert abs(row["sampleMean"] - row["mean"]) < 0.02 * max(1, row["sd"] or 1), row
        if row["sd"] is not None and row["sd"] != float("inf"):                 # NaN and infinity arrive as null
            assert abs(row["sampleSd"] - row["sd"]) < 0.03 * max(1, row["sd"]), row


PAIRS = r"""
require(process.argv[1]); require(process.argv[2]);
const {defaults, byId, rng} = DIST, Q = PAIRS;
const pop = (x, y, link, rho) => Q.population({xId: x, yId: y, px: defaults(byId[x]), py: defaults(byId[y]), link, rho});
const G = pop("normal", "normal", "straight", 0.6), U = pop("normal", "normal", "u", 0.9), C = pop("cauchy", "binomial", "straight", 0.7);
const sorted = Array.from(G.X).sort((a, b) => a - b), r = rng(5), means = [];
for (let k = 0; k < 2000; k++) means.push(Q.STATS.mean.est(G, Q.sample(G, r, 25)));
const err = (P, m, n) => Q.predictionError(P, m, 1.5, n, 200, 3);
console.log(JSON.stringify({
  keepsCurve: sorted.every((v, i) => v === G.xs[i]),
  r: G.r, rs: G.rs, eta2: G.eta2, band: Q.band(G, 1.5).mean,
  uR: U.r, uEta2: U.eta2, cauchyR: C.r, cauchyRs: C.rs,
  se: Q.summary(means).sd, rule: Q.STATS.mean.rule(G, 25),
  line10: err(U, "line", 10), line1000: err(U, "line", 1000), local1000: err(U, "local", 1000), straight1000: err(G, "line", 1000),
}));
"""


@pytest.fixture(scope="module")
def pairs():
    run = subprocess.run(["node", "-e", PAIRS, str(DIST), str(DIST.parent / "pairs-core.js")], capture_output=True, text=True, check=True)
    return json.loads(run.stdout)


def test_the_link_pairs_draws_up_without_changing_either_curve(pairs):
    assert pairs["keepsCurve"]
    # a straight link of strength 0.6 on two normals: r = 0.6, rank correlation (6/pi) asin(0.3), r^2 explained
    assert abs(pairs["r"] - 0.6) < 0.03 and abs(pairs["rs"] - 0.582) < 0.03 and abs(pairs["eta2"] - 0.36) < 0.03
    assert abs(pairs["band"] - 0.9) < 0.1
    # a U-shaped link has no straight-line correlation, yet X explains much of Y
    assert abs(pairs["uR"]) < 0.05 and pairs["uEta2"] > 0.4
    # heavy tails wreck r but not the rank correlation
    assert pairs["cauchyRs"] > 0.5 and abs(pairs["cauchyR"]) < pairs["cauchyRs"] - 0.3


def test_sampling_error_follows_one_over_root_n(pairs):
    assert abs(pairs["se"] / pairs["rule"] - 1) < 0.08


def test_specification_error_stays_put_while_sampling_error_shrinks(pairs):
    line10, line1000 = pairs["line10"], pairs["line1000"]
    assert line10["spec"] == line1000["spec"] and abs(line1000["spec"]) > 0.5      # the line misses the U by the same amount
    assert line1000["sampling"] < line10["sampling"] / 5                            # sampling error falls about like 1/sqrt(n)
    assert abs(pairs["local1000"]["spec"]) < 0.15                                   # a shape that can bend misses far less
    assert abs(pairs["straight1000"]["spec"]) < 0.1                                 # and a line is right when the truth is a line
