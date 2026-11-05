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
