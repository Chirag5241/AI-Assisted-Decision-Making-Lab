// One feature a round. Every number is in window.ONE_FEATURE, computed in Python; this file draws.
(function () {
  const study = window.ONE_FEATURE, $ = (id) => document.getElementById(id);
  const SUB = ["₁", "₂", "₃"];
  const CLS = ["c-s1", "c-s4", "c-ink"];        // x1 every round, x2 every round, the plan

  // Fig. 1: s_j against delta, with showing nothing at 0
  const score = study.score;
  Charts.line($("fig-score"), score.series[0].length, score.series.map((v, j) => ({ v, cls: CLS[j], name: "s" + SUB[j] })),
    [{ y: 0, label: "show nothing" }], [], {
      ylabel: "score s_j", xlabel: "discount δ", xstart: score.start, xscale: score.step, lo: -2, hi: 3,
      vlines: [{ t: (score.teach - score.start) / score.step + 0.5, label: "√(2/3)" }],
      alt: "Each feature's score as the discount goes from 0.5 to 0.99" });

  // Figs. 2 and 3: the simulated regret of each round, one line per schedule, on an axis set by the world
  function rounds(host, example, what) {
    Charts.line(host, study.T, example.policies.map((p, i) => ({
      v: p.sim, cls: CLS[i], name: p.name + (i === example.rule ? " (the solution)" : ""), dash: i === 2,
    })), [], [], { ylabel: "expected regret", lo: 0, hi: example.top, alt: "Simulated regret of each round, " + what,
      tipLines: (t) => example.switches ? ["plan shows x" + SUB[example.plan[t]]] : [] });
  }
  rounds($("fig-solved"), study.solved, "independent features");
  rounds($("fig-broken"), study.broken, "correlated features");
  rounds($("fig-imputed"), study.imputed, "hidden features filled in");

  // Figs. 4 and 6: the plan's schedule, one row per feature
  const schedule = (host, example) => Charts.schedule(host, study.T,
    example.d.map((_, j) => ({ name: "x" + SUB[j], on: example.plan.map((k) => k === j) })), [], "The feature the plan shows in each round");
  schedule($("fig-plan"), study.broken);
  schedule($("fig-imputed-plan"), study.imputed);

  // Fig. 7: how often the solution's rule, and any single feature forever, misses the optimum; dashed, for the
  // human who fills in the hidden features
  const sweep = study.sweep;
  Charts.line($("fig-sweep"), sweep.rule.length, [
    { v: sweep.rule, cls: "c-s4", name: "the solution's pick, as if independent, is not optimal" },
    { v: sweep.constant, cls: "c-ink", name: "the best schedule switches features" },
    { v: sweep.imputed_rule, cls: "c-s4", dash: true, name: "the same, filling in: the solution's pick is not optimal" },
    { v: sweep.imputed_constant, cls: "c-ink", dash: true, name: "the same, filling in: the best schedule switches" },
  ], [], [], { ylabel: "share of worlds", xlabel: "correlation ρ", xstart: sweep.start, xscale: sweep.step, lo: 0, hi: 0.6,
    yticks: [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6].map((v) => ({ v, label: Math.round(v * 100) + "%" })),
    fmt: (v) => Math.round(v * 100) + "%", alt: "Share of random worlds where each fails, as the correlation goes from 0 to 0.95" });
})();
