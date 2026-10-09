// Bench for three correlated features. The browser holds the inputs (two matrices, three correlations and
// an explore-then-commit policy); every number comes back from /api/correlated (simlab) and is drawn here.
(function () {
  const SPECS = window.CURVE_SPECS;
  const $ = (id) => document.getElementById(id);
  const bench = $("bench");
  const { SUB, SHADES } = MatrixEditor;
  const N = 3, PAIRS = [[0, 1], [0, 2], [1, 2]];

  // S: null while every feature learns at the one speed p1, otherwise one speed per feature
  const world = { K: 3, U: [], H: [], k: 2, C: [true, true, false], S: null };
  const REMEMBER = "decision-lab:correlated";

  const randomWeight = () => Math.round((Math.random() * 4 - 2) * 10) / 10;
  const clamp = (v, lo, hi) => Math.min(Math.max(v, lo), hi);
  const inputs = (key) => document.querySelectorAll(`[data-key="${key}"]`);
  const value = (key) => parseFloat(inputs(key)[0].value);
  const curve = () => document.querySelector('input[name="curve"]:checked').value;
  const rho = () => ["r12", "r13", "r23"].map(value);
  const make = (tag, cls, text) => { const e = document.createElement(tag); if (cls) e.className = cls; if (text !== undefined) e.textContent = text; return e; };
  const featureList = (mask) => mask.map((on, j) => (on ? "x" + SUB[j] : null)).filter(Boolean).join(", ");

  // ---- world state -------------------------------------------------------

  // Grow or shrink to K actions, keeping what is already there: new truth rows are random, new beliefs 0.
  function resize(K) {
    const grow = (M, fresh) => Array.from({ length: K }, (_, k) =>
      Array.from({ length: N }, (_, j) => (M[k] && M[k][j] !== undefined ? M[k][j] : fresh())));
    world.U = grow(world.U, randomWeight);
    world.H = grow(world.H, () => 0);
    world.K = K;
  }

  // The commit set may hold at most k features and at least one.
  function fitCommit() {
    let on = 0;
    world.C = world.C.map((v) => v && ++on <= world.k);
    if (!world.C.some(Boolean)) world.C[0] = true;
  }

  // 1 + 2 r12 r13 r23 - r12^2 - r13^2 - r23^2 > 0 (with |r| < 1) is what makes Sigma a correlation matrix
  function determinant([a, b, c]) { return 1 + 2 * a * b * c - a * a - b * b - c * c; }

  // ---- drawing the inputs ------------------------------------------------

  function drawChips() {
    const count = world.C.filter(Boolean).length;
    $("commit-chips").replaceChildren(...world.C.map((on, j) => {
      const chip = make("button", "chip", "x" + SUB[j]);
      chip.type = "button";
      chip.setAttribute("aria-pressed", String(on));
      chip.disabled = !on && count >= world.k;
      chip.title = chip.disabled ? `The budget allows ${world.k} feature${world.k > 1 ? "s" : ""}; switch one off first` : `feature ${j + 1}: ${on ? "on" : "off"}`;
      chip.addEventListener("click", () => {
        if (on && count === 1) return;   // at least one must stay on
        world.C[j] = !on;
        drawAll();
        run();
      });
      return chip;
    }));
    const explore = value("explore");
    $("policy-hint").textContent = (explore > 0
      ? `For ${explore} round${explore > 1 ? "s" : ""} it shows every set of ${world.k} in turn, then ${featureList(world.C)} for good.`
      : `It shows ${featureList(world.C)} every round, the fixed-subset policy.`) + " Regret always counts the full state.";
  }

  function drawSigma() {
    const r = rho(), det = determinant(r);
    const S = [[1, r[0], r[1]], [r[0], 1, r[2]], [r[1], r[2], 1]];
    $("sigma").replaceChildren(MatrixEditor.grid(S, { label: "The correlation matrix Sigma", rowPrefix: "x" }));
    const ok = det > 1e-6;
    $("sigma-det").textContent = ok ? `det Σ = ${det.toFixed(3)}` : `det Σ = ${det.toFixed(3)}: these three correlations cannot occur together. Move one toward the others.`;
    $("sigma-det").classList.toggle("error", !ok);
    return ok;
  }

  function drawAll() {
    document.querySelectorAll("[data-size]").forEach((el) => { el.value = world.K; });
    document.querySelector(`input[name="k"][value="${world.k}"]`).checked = true;
    drawChips();
    drawSigma();
    for (const name of ["U", "H"]) MatrixEditor.draw($("matrix-" + name), world, name, { onChange: schedule });
  }

  const speeds = Speeds.attach({
    state: world, count: () => N, spec: () => SPECS[curve()], shared: () => value("p1"),
    onChange: () => schedule(), onToggle: () => run(),
  });

  function applyCurve(resetValue) {
    const spec = SPECS[curve()];
    inputs("p1").forEach((el) => {
      el.min = spec.min; el.max = spec.max; el.step = spec.step;
      if (resetValue) el.value = spec.default;
    });
    $("p1-label").textContent = spec.param;
    $("curve-hint").textContent = spec.param + ": " + spec.hint;
    if (resetValue) speeds.reset(); else speeds.draw();
  }

  // the exploration phase can last at most the whole horizon
  function fitExplore() {
    const T = value("T");
    inputs("explore").forEach((el) => { el.max = T; el.value = clamp(parseFloat(el.value) || 0, 0, T); });
  }

  // ---- talking to Python -------------------------------------------------

  function query() {
    const rows = (M) => M.map((row) => row.join(",")).join(";");
    const q = new URLSearchParams({
      K: world.K, U: rows(world.U), H: rows(world.H), rho: rho().join(","), k: world.k,
      explore: value("explore"), C: world.C.map((on) => (on ? 1 : 0)).join(","),
      curve: curve(), p1: value("p1"), T: value("T"), delta: value("delta"), seed: value("seed"),
    });
    if (speeds.param()) q.set("ps", speeds.param());
    return q;
  }

  const stepArea = (share) => "polygon(0% 100%, " + share.map((w, t) => {
    const y = (100 * (1 - w)).toFixed(1);
    return `${((100 * t) / share.length).toFixed(2)}% ${y}%, ${((100 * (t + 1)) / share.length).toFixed(2)}% ${y}%`;
  }).join(", ") + ", 100% 100%)";

  const key = (cls, text) => { const item = make("span"), i = make("i", cls); item.append(i, text); return item; };
  let last = null;   // the latest answer, so the ring toggle can redraw the state clouds without asking Python again

  function drawScatter(data) {
    const round = document.querySelector('input[name="ring"]:checked').value;
    const picks = data.scatter[round], best = data.scatter.best;
    const missed = picks.filter((p, i) => p !== best[i]).length;
    $("scatter-legend").replaceChildren(
      ...data.U.map((_, k) => key("pt-key bg-a" + (k + 1), "best move a" + SUB[k])),
      key("ring-key", `human's pick is not the best (${missed} of ${best.length}, round ${round === "first" ? 0 : data.T - 1})`),
      (() => { const item = make("span"); item.append(Charts.swatch("fit"), "E[xⱼ | xᵢ] = ρᵢⱼ xᵢ"); return item; })());
    $("fig-scatter").replaceChildren(...PAIRS.map(([i, j], p) => {
      const panel = make("div"), title = make("p", "panel-title", `x${SUB[i]} and x${SUB[j]}`), host = make("div", "chart");
      title.append(make("small", "", "ρ = " + data.rho[p]));
      panel.append(title, host);
      Charts.scatter(host, data.scatter.x.map((x, s) => ({ x: x[i], y: x[j], cls: "c-a" + best[s], ring: picks[s] !== best[s] })),
        [{ slope: data.rho[p] }],
        { xname: "x" + SUB[i], yname: "x" + SUB[j], alt: `Probe states, feature ${i + 1} against feature ${j + 1}, coloured by the best move` });
      return panel;
    }));
  }

  function render(data) {
    last = data;
    $("numeral").textContent = data.numeral;
    $("numeral-label").textContent = data.numeral_label;
    $("sentence").textContent = data.sentence;
    $("sentence").classList.remove("error");

    $("tiles").replaceChildren(...data.stats.map((stat) => {
      const tile = make("div", "tile");
      for (const part of ["label", "value", "note"]) tile.append(make("div", part, stat[part]));
      return tile;
    }));

    const wrongShare = data.series.acc.map((a) => 1 - a);
    $("strip-wrong").style.clipPath = stepArea(wrongShare);
    $("strip-end").textContent = "round " + (data.T - 1);
    const committed = data.explore > 0 && data.explore < data.T ? [{ t: data.explore, label: "commit" }] : [];

    // Figs. 1 to 6, the standard ones. Every action is offered in every round here; only the features change.
    // One rule marks the end of exploring, instead of one at every turn of the rotation.
    const top = data.regret_cap || 1, bound = 1.05 * (data.value_cap || 1);
    StandardFigures.draw(Object.assign({}, data, { schedule: { F: data.schedule, A: data.U.map(() => Array(data.T).fill(1)) } }), {
      regret: { lo: -0.04 * top, hi: 1.04 * top }, value: { lo: -bound, hi: bound },
      vlines: committed, refName: data.ref_label ? "best fixed subset" : undefined,
    });

    drawScatter(data);

    const C = data.C, first = data.effective.first_mask;
    $("fig-effective").replaceChildren(...[
      ["Human at round 0", `shows ${featureList(first)}`, data.effective.first, first],
      [`Human at round ${data.T - 1}`, `shows ${featureList(C)}`, data.effective.last, C],
      ["Best possible", `with ${featureList(C)} and the truth`, data.effective.truth, C],
    ].map(([name, note, M, mask]) => {
      const panel = make("div"), title = make("p", "panel-title", name);
      title.append(make("small", "", note));
      panel.append(title, MatrixEditor.grid(M, { label: name + ": effective weights, rows are actions", featureOn: (j) => mask[j] }));
      return panel;
    }));

    if (data.ranking) {   // only the full answer runs the search
      renderSweep(data);
      renderRanking(data.ranking);
      for (const el of stale) el.classList.remove("stale");
    }
  }

  function renderSweep(data) {
    const { lengths, curves } = data.ranking.sweep;
    const step = lengths.length > 1 ? lengths[1] - lengths[0] : 1;
    const marks = data.explore <= lengths[lengths.length - 1] ? [{ t: data.explore / step + 0.5, label: "on the bench" }] : [];
    Charts.line($("fig-sweep"), lengths.length,
      curves.map((c, i) => {
        const onBench = c.mask.every((on, j) => on === data.C[j]);
        return { v: c.discounted, cls: onBench ? "c-ink" : "c-s" + i, name: "then " + featureList(c.mask) + (onBench ? " (on the bench)" : ""),
                 dash: c.mask.filter(Boolean).length < data.k };
      }),
      [], [], { ylabel: "discounted regret", xlabel: "rounds explored", xscale: step, width: 1120, height: 320, lo: 0, vlines: marks,
        alt: "Discounted regret of each explore-then-commit policy against the number of exploration rounds" });
  }

  function renderRanking(ranking) {
    const { rows, total } = ranking;
    $("ranking-sub").textContent = `${total} policies searched: every fixed subset within the budget, and exploring for a whole number ` +
      `of rotations before committing to each of them. The best ${Math.min(10, rows.length)} are listed with the one on the bench ` +
      `and the best fixed subset (rank ${ranking.best_fixed.rank}).`;
    const worst = Math.max(...rows.map((r) => r.discounted), 1e-9);
    const pct = (v) => Math.round(v * 100) + "%";
    $("ranking").replaceChildren(...rows.map((r) => {
      const tr = make("tr", r.current ? "current" : "");
      tr.tabIndex = 0;
      tr.title = r.current ? "On the bench now" : "Put this policy on the bench";
      const members = make("td");
      r.mask.forEach((on, j) => members.append(make("span", "feat" + (on ? " on" : ""), "x" + SUB[j])));
      const bar = make("td", "num bar-cell"), fill = make("span", "bar");
      fill.style.width = (r.discounted / worst) * 100 + "%";
      bar.append(fill, make("span", "", r.discounted.toFixed(2)));
      tr.append(make("td", "rank", r.rank + (r.current ? "  \u2190 now" : "")),
        make("td", "", r.explore ? r.explore + " rounds" : "no (fixed)"), members, bar, make("td", "num", r.mean.toFixed(2)),
        make("td", "num", `${pct(r.start)} \u2192 ${pct(r.end)}`), make("td", "num", r.floor.toFixed(2)));
      const apply = () => {
        world.C = r.mask.slice();
        inputs("explore").forEach((el) => { el.value = r.explore; });
        drawAll();
        run();
      };
      tr.addEventListener("click", apply);
      tr.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); apply(); } });
      return tr;
    }));
  }

  // Live updates, as on the larger-world bench: quick answers for the chosen policy while a control moves,
  // then the full answer, which also searches every policy, once the controls rest.
  const stale = [document.querySelector(".rankings"), $("fig-sweep")];
  let inFlight = false, moving = false, rested = false, restTimer = null, fullRequest = null;

  async function request(full) {
    if (!drawSigma()) {   // an impossible correlation matrix: say so instead of asking Python
      $("sentence").textContent = "These correlations cannot all hold at once, so there are no states to draw. Adjust ρ to continue.";
      $("sentence").classList.add("error");
      bench.setAttribute("aria-busy", "false");
      return;
    }
    const q = query();
    history.replaceState(null, "", "?" + q.toString());
    try { sessionStorage.setItem(REMEMBER, q.toString()); } catch (error) { /* storage unavailable: nothing to remember */ }
    if (!full) q.set("rank", "0");
    fullRequest = full ? new AbortController() : null;
    try {
      const response = await fetch("/api/correlated?" + q.toString(), full ? { signal: fullRequest.signal } : {});
      if (!response.ok) throw new Error("The server answered " + response.status);
      render(await response.json());
    } catch (error) {
      if (error.name === "AbortError") return;
      $("sentence").textContent = "Could not run this world: " + error.message + ". Is the Python server still running?";
      $("sentence").classList.add("error");
    }
    bench.setAttribute("aria-busy", "false");
  }

  async function pump() {
    if (inFlight) return;
    inFlight = true;
    while (moving || rested) {
      const full = !moving;
      moving = false;
      if (full) rested = false;
      await request(full);
    }
    inFlight = false;
  }

  function schedule() {
    moving = true;
    rested = false;
    if (fullRequest) fullRequest.abort();
    for (const el of stale) el.classList.add("stale");
    clearTimeout(restTimer);
    restTimer = setTimeout(() => { rested = true; pump(); }, 220);
    pump();
  }

  function run() {
    clearTimeout(restTimer);
    if (fullRequest) fullRequest.abort();
    moving = false;
    rested = true;
    for (const el of stale) el.classList.add("stale");
    pump();
  }

  // ---- wiring ------------------------------------------------------------

  document.querySelectorAll("[data-size]").forEach((el) => {
    el.addEventListener("input", () => {
      const v = parseInt(el.value, 10);
      if (isNaN(v)) return;
      resize(clamp(v, parseInt(el.min, 10), parseInt(el.max, 10)));
      drawAll();
      schedule();
    });
  });

  document.querySelectorAll("[data-key]").forEach((el) => {
    const sync = (fallback) => {
      const v = parseFloat(el.value);
      if (isNaN(v) && !fallback) return;
      const c = isNaN(v) ? inputs(el.dataset.key)[0].value : clamp(v, parseFloat(el.min), parseFloat(el.max));
      inputs(el.dataset.key).forEach((other) => { if (other !== el || fallback) other.value = c; });
      if (el.dataset.key === "T") fitExplore();
      if (el.dataset.key === "explore") drawChips();
      if (el.dataset.key.startsWith("r")) drawSigma();
      schedule();
    };
    el.addEventListener("input", () => sync(false));
    el.addEventListener("change", () => sync(true));
  });

  document.querySelectorAll('input[name="curve"]').forEach((el) =>
    el.addEventListener("change", () => { applyCurve(true); schedule(); }));

  document.querySelectorAll('input[name="k"]').forEach((el) =>
    el.addEventListener("change", () => { world.k = parseInt(el.value, 10); fitCommit(); drawAll(); run(); }));

  document.querySelectorAll('input[name="ring"]').forEach((el) =>
    el.addEventListener("change", () => { if (last) drawScatter(last); }));

  document.querySelectorAll("[data-rho]").forEach((button) =>
    button.addEventListener("click", () => {
      const r = rho(), mean = Math.round((20 * (r[0] + r[1] + r[2])) / 3) / 20;
      for (const name of ["r12", "r13", "r23"]) inputs(name).forEach((el) => { el.value = button.dataset.rho === "equal" ? mean : 0; });
      drawSigma();
      run();
    }));

  // a fresh draw of the states: another number, picked at random
  $("redraw").addEventListener("click", () => {
    const box = inputs("seed")[0], max = parseInt(box.max, 10);
    let next = Math.floor(Math.random() * (max + 1));
    if (String(next) === box.value) next = (next + 1) % (max + 1);
    inputs("seed").forEach((el) => { el.value = next; });
    schedule();
  });

  document.querySelectorAll("[data-fill]").forEach((button) =>
    button.addEventListener("click", () => {
      const [name, how] = button.dataset.fill.split(":");
      world[name] = world[name].map((row, k) => row.map((_, j) =>
        how === "random" ? randomWeight() : how === "zeros" ? 0 : world.U[k][j]));
      drawAll();
      run();
    }));

  // ---- first world: a shared link (?K=..&U=..) or the default ------------

  function load(s) {
    const matrix = (v) => (typeof v === "string" ? v.split(";").map((row) => row.split(",").map(Number)) : v);
    const U = matrix(s.U), H = matrix(s.H);
    const valid = (M) => Array.isArray(M) && M.length >= 2 && M.length <= 6 && M.every((row) => row.length === N && row.every(Number.isFinite));
    if (!valid(U) || !valid(H) || H.length !== U.length) return false;
    world.U = U; world.H = H; world.K = U.length;
    world.k = String(s.k) === "1" ? 1 : 2;
    const C = (typeof s.C === "string" ? s.C.split(",") : s.C || []).map((b) => String(b) === "1");
    world.C = C.length === N ? C : [true, true, false];
    fitCommit();
    const r = typeof s.rho === "string" ? s.rho.split(",").map(Number) : s.rho;
    if (Array.isArray(r) && r.length === 3 && r.every(Number.isFinite)) {
      ["r12", "r13", "r23"].forEach((name, i) => inputs(name).forEach((el) => { el.value = clamp(r[i], -0.95, 0.95); }));
    }
    if (s.curve in SPECS) document.querySelector(`input[name="curve"][value="${s.curve}"]`).checked = true;
    applyCurve(true);
    for (const name of ["p1", "T", "delta", "seed"]) {
      if (s[name] !== undefined && isFinite(s[name])) inputs(name).forEach((el) => { el.value = s[name]; });
    }
    speeds.load(s.ps);
    fitExplore();
    if (s.explore !== undefined && isFinite(s.explore)) inputs("explore").forEach((el) => { el.value = clamp(Number(s.explore), 0, value("T")); });
    return true;
  }

  document.querySelector('input[name="curve"]').checked = true;
  let remembered = "";
  try { remembered = sessionStorage.getItem(REMEMBER) || ""; } catch (error) { /* storage unavailable */ }
  const fromUrl = Object.fromEntries(new URLSearchParams(location.search.length > 1 ? location.search : remembered));
  if (!(fromUrl.U && load(fromUrl))) load(window.DEFAULT_WORLD);
  drawAll();
  run();
})();
