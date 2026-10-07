// Bench for K actions and n features. The browser holds the inputs (two matrices and
// two on/off masks); every number comes back from /api/world (simlab) and is drawn here.
(function () {
  const SPECS = window.CURVE_SPECS;
  const $ = (id) => document.getElementById(id);
  const bench = $("bench");
  const SUB = ["₁", "₂", "₃", "₄", "₅", "₆"];

  const world = { K: 3, n: 3, U: [], H: [], F: [], A: [] };
  const REMEMBER = "decision-lab:worlds";

  const randomWeight = () => Math.round((Math.random() * 4 - 2) * 10) / 10;
  const clamp = (v, lo, hi) => Math.min(Math.max(v, lo), hi);
  const inputs = (key) => document.querySelectorAll(`[data-key="${key}"]`);
  const curve = () => document.querySelector('input[name="curve"]:checked').value;

  // ---- world state -------------------------------------------------------

  // Grow or shrink to K x n, keeping what is already there. New truth cells are
  // random, new belief cells are 0, new features and actions start switched on.
  function resize(K, n) {
    const grow = (M, fresh) => Array.from({ length: K }, (_, k) =>
      Array.from({ length: n }, (_, j) => (M[k] && M[k][j] !== undefined ? M[k][j] : fresh())));
    world.U = grow(world.U, randomWeight);
    world.H = grow(world.H, () => 0);
    world.F = Array.from({ length: n }, (_, j) => (world.F[j] !== undefined ? world.F[j] : true));
    world.A = Array.from({ length: K }, (_, k) => (world.A[k] !== undefined ? world.A[k] : true));
    if (!world.F.some(Boolean)) world.F[0] = true;
    if (!world.A.some(Boolean)) world.A[0] = true;
    world.K = K;
    world.n = n;
  }

  // ---- drawing the inputs ------------------------------------------------

  // Each feature's panel and sliders use a slightly lighter shade of the action colours than the one before.
  const SHADES = ["100%", "88%", "76%", "66%", "57%", "50%"];

  // One weight: its symbol, a slider, and a number box with up and down arrows (hold to repeat).
  function entryRow(name, k, j) {
    const make = (tag, cls) => { const e = document.createElement(tag); if (cls) e.className = cls; return e; };
    const what = `${name === "U" ? "true" : "believed"} weight of feature ${j + 1} for action ${k + 1}`;
    const row = make("div", "row entry e-a" + (k + 1) + (world.F[j] && world.A[k] ? "" : " off"));

    const label = make("label"), sym = make("span", "sym dot"), letter = make("span"), index = make("sub");
    index.textContent = `${k + 1}${j + 1}`;
    letter.append(name === "U" ? "u" : "h", index);
    sym.append(letter);
    label.append(sym);

    const slider = make("input");
    slider.type = "range"; slider.min = "-3"; slider.max = "3"; slider.step = "0.1";
    slider.id = `s${name}-${k}-${j}`;
    slider.value = world[name][k][j];
    slider.setAttribute("aria-label", what);
    label.htmlFor = slider.id;

    const cell = make("div", "cell"), input = make("input");
    input.type = "number"; input.step = "0.1"; input.min = "-3"; input.max = "3";
    input.id = `m${name}-${k}-${j}`;
    input.value = world[name][k][j];
    input.setAttribute("aria-label", what + ", exact value");
    cell.append(input);

    const set = (value, from) => {
      world[name][k][j] = clamp(value, -3, 3);
      if (from !== slider) slider.value = world[name][k][j];
      if (from !== input) input.value = world[name][k][j];
      $(`v${name}-${k}-${j}`).textContent = shortNumber(world[name][k][j]);   // the snapshot follows the slider
      schedule();
    };
    slider.addEventListener("input", () => set(parseFloat(slider.value), slider));
    input.addEventListener("input", () => { const v = parseFloat(input.value); if (!isNaN(v)) set(v, input); });
    input.addEventListener("change", () => { input.value = world[name][k][j]; });
    for (const dir of [1, -1]) {
      const arrow = make("button", dir > 0 ? "up" : "down");
      arrow.type = "button";
      arrow.tabIndex = -1;   // the arrow keys already step the focused slider or box
      arrow.setAttribute("aria-label", (dir > 0 ? "Raise the " : "Lower the ") + what);
      const step = () => set(Math.round((world[name][k][j] + 0.1 * dir) * 10) / 10, null);
      let wait = null, repeat = null;
      const stop = () => { clearTimeout(wait); clearInterval(repeat); };
      arrow.addEventListener("pointerdown", (event) => {
        event.preventDefault();
        stop();
        step();
        wait = setTimeout(() => { repeat = setInterval(step, 70); }, 350);
      });
      for (const end of ["pointerup", "pointerleave", "pointercancel"]) arrow.addEventListener(end, stop);
      cell.append(arrow);
    }
    row.append(label, slider, cell);
    return row;
  }

  const shortNumber = (v) => String(Math.round(v * 100) / 100);

  // The whole matrix at a glance: one row per action, one column per feature. Clicking a value jumps to its slider.
  function snapshot(name) {
    const make = (tag, cls, text) => { const e = document.createElement(tag); e.className = cls; if (text !== undefined) e.textContent = text; return e; };
    const columns = `repeat(${world.n}, minmax(0, 1fr))`;
    const grid = make("div", "snapshot"), cols = make("div", "cols"), sides = make("div", "sides"), bracket = make("div", "bracket");
    grid.classList.toggle("dense", world.n >= 5);   // five or six columns: smaller figures so neighbours stay apart
    grid.setAttribute("role", "group");
    grid.setAttribute("aria-label", (name === "U" ? "The truth matrix U" : "The first-belief matrix H0") + ", rows are actions and columns are features");
    cols.style.gridTemplateColumns = columns;
    bracket.style.gridTemplateColumns = columns;
    world.F.forEach((shown, j) => cols.append(make("span", "head" + (shown ? "" : " off"), "x" + SUB[j])));
    world.A.forEach((offered, k) => {
      sides.append(make("span", "head" + (offered ? "" : " off"), "a" + SUB[k]));
      world.F.forEach((shown, j) => {
        const value = make("span", "value" + (shown && offered ? "" : " off"), shortNumber(world[name][k][j]));
        value.id = `v${name}-${k}-${j}`;
        value.title = "Go to this weight's slider";
        value.addEventListener("click", () => { const slider = $(`s${name}-${k}-${j}`); slider.scrollIntoView({ block: "center" }); slider.focus(); });
        bracket.append(value);
      });
    });
    grid.append(make("span", ""), cols, sides, bracket);
    return grid;
  }

  // One matrix: its snapshot, then its weights as sliders grouped by feature like the panels of the beliefs figure.
  function drawMatrix(name) {
    $("matrix-" + name).replaceChildren(snapshot(name), ...world.F.map((shown, j) => {
      const group = document.createElement("div"), head = document.createElement("p");
      group.className = "entry-group";
      group.style.setProperty("--shade", SHADES[j]);
      head.className = "group-label";
      head.textContent = "weights on x" + SUB[j] + (shown ? "" : "  (hidden)");
      group.append(head, ...world.A.map((_, k) => entryRow(name, k, j)));
      return group;
    }));
  }

  function drawChips(id, mask, prefix, noun) {
    $(id).replaceChildren(...mask.map((on, i) => {
      const chip = document.createElement("button");
      chip.type = "button";
      chip.className = "chip";
      chip.textContent = prefix + SUB[i];
      chip.setAttribute("aria-pressed", String(on));
      chip.title = `${noun} ${i + 1}: ${on ? "on" : "off"}`;
      chip.addEventListener("click", () => {
        if (on && mask.filter(Boolean).length === 1) return;   // at least one must stay on
        mask[i] = !on;
        drawAll();
        run();
      });
      return chip;
    }));
  }

  function drawAll() {
    document.querySelectorAll("[data-size]").forEach((el) => { el.value = world[el.dataset.size]; });
    drawChips("feature-chips", world.F, "x", "feature");
    drawChips("action-chips", world.A, "a", "action");
    drawMatrix("U");
    drawMatrix("H");
  }

  function applyCurve(resetValue) {
    const spec = SPECS[curve()];
    inputs("p1").forEach((el) => {
      el.min = spec.min; el.max = spec.max; el.step = spec.step;
      if (resetValue) el.value = spec.default;
    });
    $("p1-label").textContent = spec.param;
    $("curve-hint").textContent = spec.param + ": " + spec.hint;
  }

  // ---- talking to Python -------------------------------------------------

  function query() {
    const bits = (mask) => mask.map((on) => (on ? 1 : 0)).join(",");
    const rows = (M) => M.map((row) => row.join(",")).join(";");
    return new URLSearchParams({
      K: world.K, n: world.n, U: rows(world.U), H: rows(world.H), F: bits(world.F), A: bits(world.A),
      curve: curve(), p1: inputs("p1")[0].value, T: inputs("T")[0].value, delta: inputs("delta")[0].value,
    });
  }

  // CSS polygon for a step area: the hatched part of each round's column is that round's share
  const stepArea = (share) => "polygon(0% 100%, " + share.map((w, t) => {
    const y = (100 * (1 - w)).toFixed(1);
    return `${((100 * t) / share.length).toFixed(2)}% ${y}%, ${((100 * (t + 1)) / share.length).toFixed(2)}% ${y}%`;
  }).join(", ") + ", 100% 100%)";

  function render(data) {
    $("numeral").textContent = data.numeral;
    $("numeral-label").textContent = data.numeral_label;
    $("sentence").textContent = data.sentence;
    $("sentence").classList.remove("error");

    $("tiles").replaceChildren(...data.stats.map((stat) => {
      const tile = document.createElement("div");
      tile.className = "tile";
      for (const part of ["label", "value", "note"]) {
        const el = document.createElement("div");
        el.className = part;
        el.textContent = stat[part];
        tile.append(el);
      }
      return tile;
    }));

    // the bar at the top and the grey in every figure: the share of states with the wrong action each round
    const wrongShare = data.series.acc.map((a) => 1 - a);
    $("strip-wrong").style.clipPath = stepArea(wrongShare);
    $("strip-end").textContent = "round " + (data.T - 1);

    // Fig. 1: the one-feature beliefs chart, once per feature; one line per action, heading for its dashed truth
    const key = (cls, text) => { const item = document.createElement("span"); const i = document.createElement("i"); i.className = cls; item.append(i, text); return item; };
    $("beliefs-legend").replaceChildren(
      ...data.U.map((_, k) => key("bg-a" + (k + 1), "a" + SUB[k])),
      key("truth-key", "truth u"), key("start", "starting belief"), key("faded-key", "hidden or not offered"),
      key("box share", "grey: share of states with the wrong action"));
    $("fig-beliefs").replaceChildren(...data.beliefs.map((byAction, j) => {
      const panel = document.createElement("div"), title = document.createElement("p"), host = document.createElement("div");
      title.className = "panel-title";
      title.textContent = "x" + SUB[j];
      if (!data.F[j]) { const note = document.createElement("small"); note.textContent = "hidden"; title.append(note); }
      host.className = "chart";
      host.style.setProperty("--shade", SHADES[j]);   // same action colours, a shade per feature
      panel.append(title, host);
      const shown = (k) => data.F[j] && data.A[k];
      Charts.line(host, data.T,
        byAction.map((v, k) => ({ v, cls: "c-a" + (k + 1), name: "a" + SUB[k], dot: true, faded: !shown(k) })),
        data.U.map((row, k) => ({ y: row[j], cls: "truth c-a" + (k + 1) + (shown(k) ? "" : " faded") })), [],
        { ylabel: "weight on x" + SUB[j], height: 250, lo: -3.25, hi: 3.25, noLegend: true, dotRadius: 5, wrongShare,
          yticks: [-3, -2, -1, 0, 1, 2, 3].map((v) => ({ v, label: v })),
          alt: "Each action's believed weight on feature " + (j + 1) + " over the rounds, against the true weights" });
      return panel;
    }));

    // Fig. 2: regret weighted by delta^t; its axis is fixed by the truth alone
    const top = data.worst_regret || 1;
    Charts.line($("fig-regret"), data.T,
      [{ v: data.series.reg, cls: "c-ref", name: "before discounting" },
        { v: data.series.discounted, cls: "c-ink", name: "discounted, \u03b4 = " + data.delta }],
      [{ y: data.reg_limit, label: "before discounting, once fully learned: " + data.reg_limit.toFixed(2) }], [],
      { ylabel: "regret per round", step: true, lo: -0.04 * top, hi: 1.04 * top, wrongShare,
        alt: "Expected regret loss in each round, before and after discounting" });

    // Fig. 3: value gap; its axis is fixed by the truth and the first beliefs
    const bound = 1.1 * (data.value_bound || 1);
    Charts.line($("fig-value"), data.T, [{ v: data.series.value_gap, cls: "c-ink", name: "expected value gap" }],
      [{ y: 0, cls: "zero", label: "zero: expects exactly what the best move is worth" }], [],
      { ylabel: "value gap per round", lo: -bound, hi: bound, wrongShare,
        alt: "True best value minus the value the human expects from the chosen action, per round" });

    // Fig. 4: share of states with the best move picked, against showing and offering everything
    const pct = (v) => Math.round(v * 100) + "%";
    Charts.line($("fig-accuracy"), data.T,
      (data.series.ref_acc ? [{ v: data.series.ref_acc, cls: "c-ref", name: "everything shown and offered" }] : [])
        .concat([{ v: data.series.acc, cls: "c-a1", name: "what is shown now" }]),
      [{ y: data.acc_limit, label: "once fully learned: " + pct(data.acc_limit) }], [],
      { ylabel: "states with the best move picked", pct: true, lo: -0.03, hi: 1.08, width: 1120, height: 300, wrongShare,
        alt: "Share of states in which the human picks the best move, per round" });

    // Fig. 5: the choice for one reference state (every feature at +1) against the best move there
    const picks = data.choice.picks, bestMove = data.choice.best, missed = data.choice.missed, misses = [];
    for (let start = 0, t = 1; t <= data.T; t++) {
      if (t < data.T && missed[t] === missed[start]) continue;
      if (missed[start]) misses.push([start, t]);
      start = t;
    }
    Charts.line($("fig-choice"), data.T,
      [{ v: picks.map(() => bestMove), cls: "c-ink", dash: true, name: "y* best move" }, { v: picks, cls: "c-ink", name: "\u0177 human's choice" }],
      [], misses, { ylabel: "action chosen", step: true, width: 1120, height: 96 + 26 * data.K, lo: 0.5, hi: data.K + 0.5,
        yticks: Array.from({ length: data.K }, (_, k) => ({ v: k + 1, label: "a" + SUB[k], faded: !data.A[k] })), fmt: (v) => "action " + v,
        alt: "Which action the human chooses each round for the state with every feature at +1, against the best move" });

    if (data.rankings) {   // only the full answer ranks every subset
      renderRanking("features", data.rankings.features, "x", "feature", (mask) => { world.F = mask; });
      renderRanking("actions", data.rankings.actions, "a", "action", (mask) => { world.A = mask; });
      rankings.classList.remove("stale");
    }
  }

  function renderRanking(key, ranking, prefix, noun, choose) {
    const { rows, total } = ranking;
    const other = noun === "feature" ? "actions" : "features";
    $("ranking-" + key + "-sub").textContent = total === 1
      ? `With one ${noun} there is only one subset.`
      : `All ${total} ${noun} subsets, each run with the chosen ${other}.` +
        (rows.length < total ? ` The best ${Math.min(8, rows.length)} and the one on the bench are listed.` : "");
    const worst = Math.max(...rows.map((r) => r.discounted), 1e-9);
    const pct = (v) => Math.round(v * 100) + "%";
    $("ranking-" + key).replaceChildren(...rows.map((r) => {
      const tr = document.createElement("tr");
      if (r.current) tr.className = "current";
      tr.tabIndex = 0;
      tr.title = r.current ? "On the bench now" : "Put this subset on the bench";
      const cell = (text, cls) => {
        const td = document.createElement("td");
        if (cls) td.className = cls;
        td.textContent = text;
        return td;
      };
      const members = document.createElement("td");
      r.mask.forEach((on, j) => {
        const dot = document.createElement("span");
        dot.className = "feat" + (on ? " on" : "");
        dot.textContent = prefix + SUB[j];
        members.append(dot);
      });
      const bar = document.createElement("td");
      bar.className = "num bar-cell";
      const fill = document.createElement("span");
      fill.className = "bar";
      fill.style.width = (r.discounted / worst) * 100 + "%";
      const value = document.createElement("span");
      value.textContent = r.discounted.toFixed(2);
      bar.append(fill, value);
      tr.append(cell(r.rank + (r.current ? "  \u2190 now" : ""), "rank"), members, bar,
        cell(`${pct(r.start)} \u2192 ${pct(r.end)}`, "num"), cell(r.floor.toFixed(2), "num"));
      const apply = () => { choose(r.mask.slice()); drawAll(); run(); };
      tr.addEventListener("click", apply);
      tr.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); apply(); } });
      return tr;
    }));
  }

  // Live updates. While a control is moving, ask only for the chosen policy (a quick answer) and
  // draw each one as it arrives. Once the controls rest, ask for the full answer, which also runs
  // every feature subset and every action subset for the two ranking tables.
  const rankings = document.querySelector(".rankings");
  let inFlight = false;     // one request at a time
  let moving = false;       // inputs changed since the last quick answer was requested
  let rested = false;       // the controls have been still long enough to rank the subsets
  let restTimer = null;
  let fullRequest = null;

  async function request(full) {
    const q = query();
    history.replaceState(null, "", "?" + q.toString());
    try { sessionStorage.setItem(REMEMBER, q.toString()); } catch (error) { /* storage unavailable: nothing to remember */ }
    if (!full) q.set("rank", "0");
    fullRequest = full ? new AbortController() : null;
    try {
      const response = await fetch("/api/world?" + q.toString(), full ? { signal: fullRequest.signal } : {});
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

  // a control is being dragged, typed in or stepped
  function schedule() {
    moving = true;
    rested = false;
    if (fullRequest) fullRequest.abort();   // a ranking run for older inputs is no longer wanted
    rankings.classList.add("stale");
    clearTimeout(restTimer);
    restTimer = setTimeout(() => { rested = true; pump(); }, 220);
    pump();
  }

  // a one-off change (a subset picked, a matrix refilled): go straight to the full answer
  function run() {
    clearTimeout(restTimer);
    if (fullRequest) fullRequest.abort();
    moving = false;
    rested = true;
    rankings.classList.add("stale");
    pump();
  }

  // ---- wiring ------------------------------------------------------------

  document.querySelectorAll("[data-size]").forEach((el) => {
    el.addEventListener("input", () => {
      const v = parseInt(el.value, 10);
      if (isNaN(v)) return;
      const size = clamp(v, parseInt(el.min, 10), parseInt(el.max, 10));
      resize(el.dataset.size === "K" ? size : world.K, el.dataset.size === "n" ? size : world.n);
      drawAll();
      schedule();
    });
  });

  document.querySelectorAll("[data-key]").forEach((el) => {
    el.addEventListener("input", () => {
      const v = parseFloat(el.value);
      if (isNaN(v)) return;
      const c = clamp(v, parseFloat(el.min), parseFloat(el.max));
      inputs(el.dataset.key).forEach((other) => { if (other !== el) other.value = c; });
      schedule();
    });
    el.addEventListener("change", () => {
      const v = parseFloat(el.value);
      const c = isNaN(v) ? inputs(el.dataset.key)[0].value : clamp(v, parseFloat(el.min), parseFloat(el.max));
      inputs(el.dataset.key).forEach((other) => { other.value = c; });
      schedule();
    });
  });

  document.querySelectorAll('input[name="curve"]').forEach((el) =>
    el.addEventListener("change", () => { applyCurve(true); schedule(); }));

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
    const mask = (v) => (typeof v === "string" ? v.split(",") : v).map((b) => String(b) === "1");
    const U = matrix(s.U), H = matrix(s.H);
    const valid = (M) => Array.isArray(M) && M.length >= 2 && M.length <= 6 && M[0].length >= 1 && M[0].length <= 6
      && M.every((row) => row.length === M[0].length && row.every(Number.isFinite));
    if (!valid(U) || !valid(H) || H.length !== U.length || H[0].length !== U[0].length) return false;
    world.U = U; world.H = H; world.K = U.length; world.n = U[0].length;
    world.F = s.F ? mask(s.F) : []; world.A = s.A ? mask(s.A) : [];
    if (world.F.length !== world.n) world.F = [];
    if (world.A.length !== world.K) world.A = [];
    resize(world.K, world.n);
    if (s.curve in SPECS) document.querySelector(`input[name="curve"][value="${s.curve}"]`).checked = true;
    applyCurve(true);
    for (const key of ["p1", "T", "delta"]) {
      if (s[key] !== undefined && isFinite(s[key])) inputs(key).forEach((el) => { el.value = s[key]; });
    }
    return true;
  }

  document.querySelector('input[name="curve"]').checked = true;
  // A shared link wins; otherwise come back to what was on this page before moving to another one.
  let remembered = "";
  try { remembered = sessionStorage.getItem(REMEMBER) || ""; } catch (error) { /* storage unavailable */ }
  const fromUrl = Object.fromEntries(new URLSearchParams(location.search.length > 1 ? location.search : remembered));
  if (!(fromUrl.U && load(fromUrl))) load(window.DEFAULT_WORLD);
  drawAll();
  run();
})();
