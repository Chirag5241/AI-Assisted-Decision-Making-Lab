// The six standard figures, drawn the same way on every bench and on the Overview. Each bench hands over
// its answer in one shape (the shape /api/world returns) and this file places the marks, so Fig. 2 is the
// beliefs chart everywhere, with the same colours, keys and conventions. A figure whose host element is
// not on the page (a bench that lists it as not relevant) is skipped.
window.StandardFigures = (function () {
  const $ = (id) => document.getElementById(id);
  const SUB = ["₁", "₂", "₃", "₄", "₅", "₆"];
  // each feature's panel uses a slightly lighter shade of the action colours than the one before (as in matrix.js)
  const SHADES = ["100%", "88%", "76%", "66%", "57%", "50%"];
  const make = (tag, cls, text) => { const e = document.createElement(tag); if (cls) e.className = cls; if (text !== undefined) e.textContent = text; return e; };
  const key = (cls, text) => { const item = make("span"); item.append(make("i", cls), text); return item; };
  const lineKey = (cls, text) => { const item = make("span"); item.append(Charts.swatch(cls), text); return item; };   // drawn like its line
  const pct = (v) => Math.round(v * 100) + "%";
  const UNITS = [-3, -2, -1, 0, 1, 2, 3].map((v) => ({ v, label: v }));

  // data: {T, K, U[k][j], delta, beliefs[j][k][t], schedule: {F[j][t], A[k][t]}, changes: [t], choice: {picks, best, missed},
  //        series: {reg, discounted, reg_mean, value_gap, value_gap_mean, acc, ref_acc}, reg_limit, acc_limit}
  //   choice (picks and best, per round), reg, discounted and value_gap are taken at the state drawn in each round;
  //   reg_mean and value_gap_mean are their averages over all states and acc the share of states with the best move.
  //   ref_acc and the two limits are optional. x[t][j] is the state drawn for round t, one number per feature.
  // opts: {regret: {lo, hi, yticks}, value: {lo, hi, yticks}   the fixed axes of Figs. 4 and 5
  //        vlines: [{t, label}]   rules to draw instead of one at every change of the schedule
  //        refName   what the grey reference line is
  //        truthLabels   name each dashed truth on its line (the one-feature bench)}
  function draw(data, opts) {
    const T = data.T, plan = data.schedule;
    const n = data.beliefs.length, K = data.U.length;
    const vlines = opts.vlines || (data.changes || []).map((t) => ({ t }));
    // the grey in every figure: the share of states with the wrong action in each round
    const wrongShare = data.series.acc.map((a) => 1 - a);
    // the figures taken at the drawn state say in their tooltip what that state was: the vector x_t, feature by feature
    const number = (v) => (v < 0 ? "\u2212" : "") + Math.abs(v).toFixed(2);
    const names = n === 1 ? "x\u209c" : "x\u209c = (" + (n <= 3 ? plan.F.map((_, j) => "x" + SUB[j]).join(", ") : "x\u2081 \u2026 x" + SUB[n - 1]) + ")";
    const tipLines = (t) => [names + " = " + (n === 1 ? number(data.x[t][0]) : "(" + data.x[t].map(number).join(", ") + ")")];
    // Fig. 3 also says which features the human was shown in that round, and, on the state-aware bench, when
    // that subset was the random fallback
    const shownLine = (t) => "shown: " + plan.F.map((on, j) => (on[t] ? "x" + SUB[j] : null)).filter(Boolean).join(", ")
      + (data.policy && data.policy.worked[t] === 0 ? "  (random: no subset works)" : "");
    const choiceTip = (t) => tipLines(t).concat(n > 1 ? [shownLine(t)] : []);

    // Fig. 1: what is shown and offered in each round
    if ($("fig-schedule")) {
      Charts.schedule($("fig-schedule"), T,
        plan.F.map((on, j) => ({ name: "x" + SUB[j], on: on.map(Boolean) }))
          .concat(plan.A.map((on, k) => ({ name: "a" + SUB[k], on: on.map(Boolean), group: k === 0 }))),
        vlines.filter((v) => v.label), "Which features are shown and which actions are offered in each round");
    }

    // Fig. 2: one panel per feature (2.1, 2.2, ...); one line per action, heading for its dashed truth
    const host = $("fig-beliefs");
    if (host) {
      const fig = host.dataset.fig;
      $("beliefs-legend").replaceChildren(
        ...data.U.map((_, k) => lineKey("line c-a" + (k + 1), "a" + SUB[k])),
        lineKey("truth", "truth u"), key("start", "starting belief"), lineKey("line faded", "not in use: nothing learned"),
        key("box share", "grey: share of states with the wrong action"));
      host.classList.toggle("single", n === 1);
      host.replaceChildren(...data.beliefs.map((byAction, j) => {
        const panel = make("div"), title = make("p", "panel-title"), chart = make("div", "chart");
        if (n > 1) title.append(make("b", "", `Fig. ${fig}.${j + 1}`));
        title.append("x" + SUB[j]);
        const rounds = plan.F[j].filter((on) => !on).length;
        if (rounds) title.append(make("small", "", rounds === T ? "hidden" : `hidden for ${rounds} round${rounds === 1 ? "" : "s"}`));
        chart.style.setProperty("--shade", SHADES[j]);   // same action colours, a shade per feature
        panel.append(title, chart);
        // a weight is learned only in the rounds its feature is shown and its action offered; elsewhere its line is faded
        const off = (k) => plan.F[j].map((on, t) => !(on && plan.A[k][t]));
        const never = (k) => off(k).every(Boolean);
        Charts.line(chart, T,
          byAction.map((v, k) => ({ v, cls: "c-a" + (k + 1), name: (n === 1 ? "h" : "a") + SUB[k], dot: true, off: off(k), faded: never(k) })),
          data.U.map((row, k) => ({ y: row[j], cls: "truth c-a" + (k + 1) + (never(k) ? " faded" : ""),
            label: opts.truthLabels ? `u${SUB[k]} = ${row[j].toFixed(2)}` : undefined })), [],
          { ylabel: "weight on x" + SUB[j], width: n === 1 ? 1120 : 560, height: n === 1 ? 320 : 250, lo: -3.25, hi: 3.25,
            noLegend: true, dotRadius: 5, wrongShare, vlines, yticks: UNITS,
            alt: "Each action's believed weight on feature " + (j + 1) + " over the rounds, against the true weights" });
        return panel;
      }));
    }

    // Fig. 3: the choice at the state drawn in each round against the best move there
    if ($("fig-choice")) {
      const picks = data.choice.picks, missed = data.choice.missed, misses = [];
      for (let start = 0, t = 1; t <= T; t++) {
        if (t < T && missed[t] === missed[start]) continue;
        if (missed[start]) misses.push([start, t]);
        start = t;
      }
      Charts.line($("fig-choice"), T,
        [{ v: data.choice.best, cls: "c-ink", dash: true, name: "y* best move" }, { v: picks, cls: "c-ink", name: "\u0177 human's choice" }],
        [], misses, { ylabel: "action chosen", step: true, width: 1120, height: 96 + 26 * K, lo: 0.5, hi: K + 0.5, vlines, tipLines: choiceTip,
          yticks: Array.from({ length: K }, (_, k) => ({ v: k + 1, label: "a" + SUB[k], faded: !plan.A[k].some(Boolean) })), fmt: (v) => "action " + v,
          alt: "Which action the human chooses in each round at that round's state, against the best move there" });
    }

    // Fig. 4: the regret of each round at its drawn state, then weighted by delta^t, over its average across states
    if ($("fig-regret")) {
      const limit = data.reg_limit === undefined ? [] : [{ y: data.reg_limit, label: "average once fully learned: " + data.reg_limit.toFixed(2) }];
      Charts.line($("fig-regret"), T,
        [{ v: data.series.reg, cls: "c-ink", name: "at the drawn state x\u209c" },
          { v: data.series.discounted, cls: "c-ref", name: "discounted, \u03b4 = " + data.delta },
          { v: data.series.reg_mean, cls: "c-ink", dash: true, name: "averaged over states" }],
        limit, [], Object.assign({ ylabel: "regret per round", step: true, wrongShare, vlines, tipLines,
          alt: "Regret loss in each round at the state drawn for it, before and after discounting, with its average over states" }, opts.regret));
    }

    // Fig. 5: value gap at the drawn state, around zero, over its average across states
    if ($("fig-value")) {
      Charts.line($("fig-value"), T,
        [{ v: data.series.value_gap, cls: "c-ink", name: "at the drawn state x\u209c" },
          { v: data.series.value_gap_mean, cls: "c-ink", dash: true, name: "averaged over states" }],
        [{ y: 0, cls: "zero", label: "zero: expects exactly what the best move is worth" }], [],
        Object.assign({ ylabel: "value gap per round", step: true, wrongShare, vlines, tipLines,
          alt: "True best value minus the value the human expects from the chosen action, at each round's state" }, opts.value));
    }

    // Fig. 6: share of states with the best move picked, against the bench's reference policy
    if ($("fig-accuracy")) {
      Charts.line($("fig-accuracy"), T,
        (data.series.ref_acc ? [{ v: data.series.ref_acc, cls: "c-ref", name: opts.refName }] : [])
          .concat([{ v: data.series.acc, cls: "c-a1", name: "on the bench" }]),
        data.acc_limit === undefined ? [] : [{ y: data.acc_limit, label: "once fully learned: " + pct(data.acc_limit) }], [],
        { ylabel: "states with the best move picked", pct: true, lo: -0.03, hi: 1.08, width: 1120, height: 300, wrongShare, vlines,
          alt: "Share of states in which the human picks the best move, per round" });
    }
  }

  return { draw, SUB, SHADES };
})();
