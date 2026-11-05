// Sepsis, from data. Every number is in window.SEPSIS, computed in Python (experiments/sepsis_fit.py); this file draws.
(function () {
  const study = window.SEPSIS, $ = (id) => document.getElementById(id);
  const SVG = "http://www.w3.org/2000/svg";
  const svgEl = (tag, attrs, text) => {
    const e = document.createElementNS(SVG, tag);
    for (const k in attrs) e.setAttribute(k, attrs[k]);
    if (text !== undefined) e.textContent = text;
    return e;
  };
  const el = (tag, cls, text) => {
    const e = document.createElement(tag);
    if (cls) e.className = cls;
    if (text !== undefined) e.textContent = text;
    return e;
  };
  const names = study.labs.concat(study.symptoms), nL = study.labs.length;
  // the four training policies, each with its line style
  const POLICY = [
    { key: "teacher", cls: "c-ink" }, { key: "self", cls: "c-s1" },
    { key: "random", cls: "c-s5" }, { key: "expert", cls: "c-s4" },
  ];

  // Horizontal bars, one row per name and one bar per series, on a fixed axis [0, hi]; a rule at `mark`.
  function bars(host, rows, series, opts) {
    const W = 640, L = 130, R = 16, TOP = 6, band = 8 + 7 * series.length, gap = 4, pw = W - L - R;
    const ph = rows.length * (band + gap), X = (v) => L + (Math.min(Math.max(v, 0), opts.hi) / opts.hi) * pw;
    const svg = svgEl("svg", { viewBox: `0 0 ${W} ${TOP + ph + 40}`, role: "img", "aria-label": opts.alt });
    for (const v of opts.ticks) {
      svg.append(svgEl("line", { class: "grid", x1: X(v), x2: X(v), y1: TOP, y2: TOP + ph }),
        svgEl("text", { x: X(v), y: TOP + ph + 16, "text-anchor": "middle" }, opts.fmt(v)));
    }
    rows.forEach((row, i) => {
      const y = TOP + i * (band + gap);
      if (row.split) svg.append(svgEl("line", { class: "zero", x1: 0, x2: W - R, y1: y - gap / 2, y2: y - gap / 2 }));
      svg.append(svgEl("text", { x: L - 8, y: y + band / 2 + 4, "text-anchor": "end", class: row.faded ? "faded" : "" }, row.name));
      series.forEach((s, k) => {
        const bar = svgEl("rect", { class: "bar " + s.cls + (row.faded ? " faded" : ""), x: L, y: y + 4 + 7 * k, width: Math.max(0.5, X(row.values[k]) - L), height: 6 });
        bar.append(svgEl("title", {}, `${row.name}, ${s.name}: ${opts.fmt(row.values[k], true)}`));
        svg.append(bar);
      });
    });
    if (opts.mark !== undefined) svg.append(svgEl("line", { class: "vline", x1: X(opts.mark), x2: X(opts.mark), y1: TOP, y2: TOP + ph }));
    svg.append(svgEl("text", { x: L + pw / 2, y: TOP + ph + 34, "text-anchor": "middle" }, opts.xlabel));
    const legend = el("div", "legend");
    if (series.length > 1) for (const s of series) {
      const item = el("span"), key = el("i", "bar-key " + s.cls);
      item.append(key, s.name);
      legend.append(item);
    }
    host.replaceChildren(legend, svg);
  }

  // A correlation matrix: blue for +1, orange for -1, paper for 0; a rule between the labs and the symptoms.
  function matrix(host, M, labels, split) {
    const n = M.length, L = 112, TOP = 112, cell = 16, W = L + n * cell + 8;
    const svg = svgEl("svg", { viewBox: `0 0 ${W} ${TOP + n * cell + 34}`, role: "img", "aria-label": "Correlation matrix of the risk scores" });
    const mix = (v) => {
      const a = Math.min(Math.abs(v), 1), [r, g, b] = v >= 0 ? [42, 120, 214] : [235, 104, 52];
      return `rgb(${Math.round(252 + (r - 252) * a)},${Math.round(252 + (g - 252) * a)},${Math.round(251 + (b - 251) * a)})`;
    };
    for (let i = 0; i < n; i++) {
      svg.append(svgEl("text", { x: L - 6, y: TOP + i * cell + cell / 2 + 4, "text-anchor": "end" }, labels[i]),
        svgEl("text", { x: L + i * cell + cell / 2 + 4, y: TOP - 6, transform: `rotate(-90 ${L + i * cell + cell / 2 + 4} ${TOP - 6})` }, labels[i]));
      for (let j = 0; j < n; j++) {
        const rect = svgEl("rect", { x: L + j * cell, y: TOP + i * cell, width: cell, height: cell, fill: mix(M[i][j]) });
        rect.append(svgEl("title", {}, `${labels[i]} · ${labels[j]}: ${M[i][j].toFixed(2)}`));
        svg.append(rect);
      }
    }
    for (const at of [L + split * cell]) {
      svg.append(svgEl("line", { class: "zero", x1: at, x2: at, y1: TOP, y2: TOP + n * cell }),
        svgEl("line", { class: "zero", x1: L, x2: L + n * cell, y1: TOP + split * cell, y2: TOP + split * cell }));
    }
    // the scale, -1 to 1
    const key = TOP + n * cell + 12, kw = 200;
    for (let k = 0; k < 40; k++) svg.append(svgEl("rect", { x: L + (k * kw) / 40, y: key, width: kw / 40 + 0.5, height: 8, fill: mix(-1 + (2 * k + 1) / 40) }));
    svg.append(svgEl("text", { x: L, y: key + 20, "text-anchor": "middle" }, "−1"), svgEl("text", { x: L + kw / 2, y: key + 20, "text-anchor": "middle" }, "0"),
      svgEl("text", { x: L + kw, y: key + 20, "text-anchor": "middle" }, "1"));
    host.replaceChildren(svg);
  }

  const pct = (v) => Math.round(v * 100) + "%";

  // Fig. 1: coverage of every variable, septic against the rest; the dropped ones faded
  const kept = new Set(names);
  const all = study.all_labs.concat(study.all_symptoms);
  bars($("fig-coverage"), all.map((c, i) => ({
    name: c, values: [study.coverage[c].septic, study.coverage[c].other], faded: !kept.has(c), split: i === study.all_labs.length,
  })), [{ cls: "c-a2", name: "septic" }, { cls: "c-a1", name: "not septic" }], {
    hi: 1, ticks: [0, 0.2, 0.4, 0.6, 0.8, 1], fmt: (v, exact) => (exact ? (v * 100).toFixed(1) + "%" : pct(v)), mark: study.min_coverage,
    xlabel: "measured by the decision hour", alt: "Share of patients with each variable measured, septic and not" });

  // Fig. 2: Sigma
  matrix($("fig-sigma"), study.Sigma, names, nL);

  // Fig. 3: d, on the labs only
  bars($("fig-weights"), study.labs.map((c, i) => ({ name: c, values: [study.d[i]] })), [{ cls: "c-ink", name: "d" }], {
    hi: 0.4, ticks: [0, 0.1, 0.2, 0.3, 0.4], fmt: (v, exact) => v.toFixed(exact ? 3 : 1),
    xlabel: "weight on x_j (log-odds per standard deviation)", alt: "Logistic weight of each lab" });

  // Fig. 4: ROC curves, on the unit square
  const grid = study.roc.labs.map((_, i) => i / (study.roc.labs.length - 1));
  const auc = (k) => " (" + study.auroc[k].toFixed(3) + ")";
  Charts.line($("fig-roc"), grid.length, [
    { v: study.roc.labs, cls: "c-ink", name: "linear, labs: this U" + auc("labs") },
    { v: study.roc.symptoms, cls: "c-s1", name: "linear, symptoms only" + auc("symptoms") },
    { v: study.roc.all, cls: "c-s4", name: "linear, labs + symptoms" + auc("all") },
    { v: study.roc.gbm_filled, cls: "c-s6", dash: true, name: "boosted trees, same labs" + auc("gbm_filled") },
    { v: study.roc.gbm, cls: "c-s5", dash: true, name: "boosted trees, labs with gaps kept" + auc("gbm") },
    { v: grid, cls: "c-ref", dash: true, name: "chance" },
  ], [], [], { lo: 0, hi: 1, ylabel: "true-positive rate", xlabel: "false-positive rate", xstart: 0, xscale: 1 / (grid.length - 1),
    yticks: [0, 0.2, 0.4, 0.6, 0.8, 1].map((v) => ({ v, label: v })), alt: "ROC curves of each model on held-out patients" });

  // Fig. 5: what the symptoms say about each lab
  bars($("fig-explained"), study.labs.map((c, i) => ({ name: c, values: [study.explained[i]] })), [{ cls: "c-ink", name: "explained" }], {
    hi: 0.05, ticks: [0, 0.01, 0.02, 0.03, 0.04, 0.05], fmt: (v, exact) => (exact ? (v * 100).toFixed(1) + "%" : pct(v)),
    xlabel: "share of the lab's variance explained by the symptoms", alt: "Share of each lab's variance the symptoms explain" });

  // Figs. 6 to 8 and the table, for the chosen budget
  const lab = (j) => study.labs[j];
  const FILLS = ["symptoms", "shown"];            // left to right: the theory's fill-in, then Guan et al.'s
  const SHORT = { symptoms: "From the symptoms", shown: "From the symptoms and shown labs" };

  // a table with a header row, from rows of cells; a cell is text or [text, class]
  function table(host, head, rows, groups) {
    const thead = el("thead");
    if (groups) {
      const tr = el("tr");
      for (const [t, span] of groups) { const th = el("th", span > 1 ? "num" : "", t); th.scope = "colgroup"; th.colSpan = span; tr.append(th); }
      thead.append(tr);
    }
    const hr = el("tr");
    head.forEach((t, i) => { const th = el("th", i ? "num" : "", t); th.scope = "col"; hr.append(th); });
    thead.append(hr);
    const body = el("tbody");
    for (const row of rows) {
      const tr = el("tr"), th = el("th", "", row[0]);
      th.scope = "row";
      tr.append(th, ...row.slice(1).map((cell) => el("td", "num", cell)));
      body.append(tr);
    }
    host.replaceChildren(thead, body);
  }

  function draw(c) {
    const ref = study.reference, at = (fill) => study.by_fill[fill][c];
    const opts = { lo: 0, hi: 0.6, ylabel: "expected regret", yticks: [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6].map((v) => ({ v, label: v.toFixed(1) })),
      fmt: (v) => v.toFixed(3), width: 520 };
    for (const fill of FILLS) {
      const e = at(fill);
      const line = (key) => POLICY.map((p) => ({ v: e.policies[p.key][key], cls: p.cls, name: study.policies[p.key] }));
      const hlines = [
        { y: ref.symptoms_only, label: "truth, symptoms only" },
        { y: e.best_regret, label: "truth, best panel" },
        { y: ref.guess, label: "guess" },
      ];
      Charts.line($("fig-graduation-" + fill), study.T + 1, line("after"), hlines, [], Object.assign(
        { alt: "Graduation regret against the length of training, " + SHORT[fill].toLowerCase() }, opts));
      Charts.line($("fig-training-" + fill), study.T, line("during"), hlines.slice(0, 2), [], Object.assign(
        { alt: "Expected regret of each training round, " + SHORT[fill].toLowerCase() }, opts));
      const ex = e.policies.teacher.example, used = ex.map((row, j) => j).filter((j) => ex[j].some(Boolean));
      Charts.schedule($("fig-schedule-" + fill), study.T, used.map((j) => ({ name: lab(j), on: ex[j].map(Boolean) })), [],
        "The labs the teacher shows in each round for one human, " + SHORT[fill].toLowerCase());
    }

    table($("panels"), ["Fill-in", "Valued most", "Regret", "Least regret", "Regret"], FILLS.map((fill) => {
      const e = at(fill);
      return [SHORT[fill], e.expert.map(lab).join(", "), e.expert_regret.toFixed(3), e.best.map(lab).join(", "), e.best_regret.toFixed(3)];
    }));

    const cells = (r) => [pct(r.sensitivity), pct(r.specificity), r.cost.toFixed(3)];
    const rows = [
      ["Truth, every lab"].concat(...FILLS.map(() => cells(ref.clinical_every_lab))),
      ["Truth, symptoms only"].concat(...FILLS.map(() => cells(ref.clinical_symptoms_only))),
      ["Truth, panel valued most"].concat(...FILLS.map((f) => cells(at(f).clinical_expert))),
    ].concat(POLICY.map((p) => ["Graduate: " + study.policies[p.key]].concat(...FILLS.map((f) => cells(at(f).policies[p.key].clinical)))));
    table($("clinical"), ["Who decides"].concat(...FILLS.map(() => ["Sens.", "Spec.", "Cost"])), rows,
      [["", 1]].concat(FILLS.map((f) => [SHORT[f], 3])));
  }
  const radios = document.querySelectorAll('input[name="budget"]');
  radios.forEach((r) => r.addEventListener("change", () => draw(r.value)));
  draw(document.querySelector('input[name="budget"]:checked').value);
})();
