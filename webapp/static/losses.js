// Alternatives to the regret loss. Every number is in window.LOSS_STUDY, computed in Python; this file draws.
(function () {
  const study = window.LOSS_STUDY, $ = (id) => document.getElementById(id);
  const SUB = ["₁", "₂", "₃", "₄", "₅", "₆"];
  const make = (tag, cls, text) => { const e = document.createElement(tag); if (cls) e.className = cls; if (text !== undefined) e.textContent = text; return e; };
  const name = (shown) => (shown ? "{" + shown.map((i) => "x" + SUB[i - 1]).join(", ") + "} every round" : "State-aware subsets");

  // Fig. 1: with two actions regret, wrong pick, pairs out of order and ranked regret are one step
  const against = study.margin;
  Charts.line($("fig-margin"), against.series.regret.length, [
    { v: against.series.regret, cls: "c-ink", name: "regret, wrong pick, pairs out of order, ranked regret" },
    { v: against.series.noisy, cls: "c-s1", name: "noisy regret" },
    { v: against.series.log, cls: "c-s4", name: "log loss" },
  ], [], [], { ylabel: "loss of the round", xlabel: "the human's margin for the best move", xstart: against.start, xscale: against.step,
    lo: 0, hi: against.top, alt: "Each loss as the human's margin for the best move goes from minus 3 to 3" });

  // Fig. 2: one panel per loss, the same four policies in each, on an axis that is the same for every policy
  const drawn = [[null, "c-ink"], [[1, 2], "c-s1"], [[1, 2, 3], "c-s4"], [[3], "c-s5"]].map(([shown, cls]) =>
    ({ cls, policy: study.policies.find((p) => String(p.shown) === String(shown)) }));
  $("rounds-legend").replaceChildren(...drawn.map(({ cls, policy }) => {
    const item = make("span");
    item.append(Charts.swatch("line " + cls), name(policy.shown));
    return item;
  }));
  $("fig-rounds").replaceChildren(...study.losses.map((loss) => {
    const panel = make("div"), host = make("div", "chart");
    panel.append(make("p", "panel-title", loss.name), host);
    Charts.line(host, study.T, drawn.map(({ cls, policy }) => ({ v: policy.series[loss.key], cls, name: name(policy.shown) })), [], [],
      { ylabel: loss.name.toLowerCase(), lo: -0.03 * loss.top, hi: loss.top, pct: loss.key === "wrong", height: 300, noLegend: true,
        alt: loss.name + ", round by round, for four policies" });
    return panel;
  }));
})();
