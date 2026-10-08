// Bench for K actions and n features. The browser holds the inputs (two matrices and
// two on/off masks); every number comes back from /api/world (simlab) and is drawn here.
// The "Timed hiding" page is the same bench with one more input: windows of rounds in
// which one feature or one action is held back.
(function () {
  const SPECS = window.CURVE_SPECS;
  const $ = (id) => document.getElementById(id);
  const bench = $("bench");
  const SUB = ["₁", "₂", "₃", "₄", "₅", "₆"];

  const TIMED = Boolean(window.TIMED_BENCH), MAX_WINDOWS = window.MAX_WINDOWS || 0, LAST_ROUND = 299;
  // W: the windows, each {kind: "f" or "a", index, from, to}; hidden from round `from` to round `to`, both included
  // S: null while every feature learns at the one speed p1, otherwise one speed per feature
  const world = { K: 3, n: 3, U: [], H: [], F: [], A: [], W: [], S: null };
  const REMEMBER = TIMED ? "decision-lab:timed" : "decision-lab:worlds";

  const randomWeight = () => Math.round((Math.random() * 4 - 2) * 10) / 10;
  const clamp = (v, lo, hi) => Math.min(Math.max(v, lo), hi);
  const inputs = (key) => document.querySelectorAll(`[data-key="${key}"]`);
  const curve = () => document.querySelector('input[name="curve"]:checked').value;
  const horizon = () => parseInt(inputs("T")[0].value, 10);
  const span = (w) => [Math.min(w.from, w.to), Math.max(w.from, w.to)];   // either box may hold the earlier round

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
    world.W = world.W.filter((w) => w.index >= 0 && w.index < (w.kind === "f" ? n : K));   // windows on what is gone go with it
    world.K = K;
    world.n = n;
  }

  // ---- drawing the inputs ------------------------------------------------

  const speeds = Speeds.attach({
    state: world, count: () => world.n, spec: () => SPECS[curve()], shared: () => parseFloat(inputs("p1")[0].value),
    onChange: () => schedule(), onToggle: () => run(),
  });

  function drawMatrix(name) {
    MatrixEditor.draw($("matrix-" + name), world, name,
      { featureOn: (j) => world.F[j], actionOn: (k) => world.A[k], onChange: schedule });
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

  // One row per window: what it hides, its first and last round, and a button to remove it.
  function drawWindows() {
    if (!TIMED) return;
    $("windows").replaceChildren(...world.W.map((w, at) => {
      const row = document.createElement("div"), target = document.createElement("select");
      row.className = "window";
      target.setAttribute("aria-label", `Window ${at + 1}: what it hides`);
      for (const [kind, count, prefix] of [["f", world.n, "x"], ["a", world.K, "a"]]) {
        for (let i = 0; i < count; i++) target.add(new Option(prefix + SUB[i], kind + i, false, w.kind === kind && w.index === i));
      }
      target.addEventListener("change", () => { w.kind = target.value[0]; w.index = Number(target.value.slice(1)); run(); });
      const boxes = {};
      for (const [key, label] of [["from", "first"], ["to", "last"]]) {
        const box = boxes[key] = document.createElement("input");
        box.type = "number"; box.min = "0"; box.max = String(LAST_ROUND); box.step = "1";
        box.value = w[key];
        box.setAttribute("aria-label", `Window ${at + 1}: ${label} round`);
        box.addEventListener("input", () => {
          const v = parseInt(box.value, 10);
          if (isNaN(v)) return;
          w[key] = clamp(v, 0, LAST_ROUND);
          schedule();
        });
        // once typing stops, put the earlier round first
        box.addEventListener("change", () => { [w.from, w.to] = span(w); boxes.from.value = w.from; boxes.to.value = w.to; });
      }
      const to = document.createElement("span"), remove = document.createElement("button");
      to.textContent = "to";
      remove.type = "button";
      remove.className = "remove";
      remove.textContent = "\u00d7";
      remove.setAttribute("aria-label", `Remove window ${at + 1}`);
      remove.addEventListener("click", () => { world.W.splice(at, 1); drawWindows(); run(); });
      row.append(target, boxes.from, to, boxes.to, remove);
      return row;
    }));
    $("add-window").disabled = world.W.length >= MAX_WINDOWS;
  }

  // The first round the windows would leave with no feature to show or no action to offer, as a message.
  function windowProblem() {
    for (const [kind, mask, nothing] of [["f", world.F, "no feature to show"], ["a", world.A, "no action to offer"]]) {
      for (let t = 0, T = horizon(); t < T; t++) {
        const left = mask.some((on, i) => on && !world.W.some((w) => w.kind === kind && w.index === i && span(w)[0] <= t && t <= span(w)[1]));
        if (!left) return `Round ${t} would have ${nothing}. Shorten a window or switch something else on.`;
      }
    }
    return null;
  }

  function drawAll() {
    document.querySelectorAll("[data-size]").forEach((el) => { el.value = world[el.dataset.size]; });
    drawChips("feature-chips", world.F, "x", "feature");
    drawChips("action-chips", world.A, "a", "action");
    drawWindows();
    speeds.fit();
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
    if (resetValue) speeds.reset(); else speeds.draw();
  }

  // ---- talking to Python -------------------------------------------------

  function query() {
    const bits = (mask) => mask.map((on) => (on ? 1 : 0)).join(",");
    const rows = (M) => M.map((row) => row.join(",")).join(";");
    const q = new URLSearchParams({
      K: world.K, n: world.n, U: rows(world.U), H: rows(world.H), F: bits(world.F), A: bits(world.A),
      curve: curve(), p1: inputs("p1")[0].value, T: inputs("T")[0].value, delta: inputs("delta")[0].value,
    });
    if (speeds.param()) q.set("ps", speeds.param());
    if (world.W.length) q.set("W", world.W.map((w) => `${w.kind}${w.index + 1}:${span(w).join("-")}`).join(";"));
    return q;
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

    // Figs. 1 to 6, the standard ones. The regret axis is fixed by the truth alone and the value-gap axis by
    // the truth and the first beliefs, so neither moves while the policy or the learner changes.
    const top = data.worst_regret || 1, bound = 1.1 * (data.value_bound || 1);
    StandardFigures.draw(data, {
      regret: { lo: -0.04 * top, hi: 1.04 * top }, value: { lo: -bound, hi: bound },
      refName: "everything shown and offered",
    });

    if (data.rankings) {   // only the full answer ranks every subset
      // a fixed subset put on the bench replaces that family's windows as well as its chips
      const fixed = (name, kind) => (mask) => { world[name] = mask; world.W = world.W.filter((w) => w.kind !== kind); };
      renderRanking("features", data.rankings.features, "x", "feature", fixed("F", "f"));
      renderRanking("actions", data.rankings.actions, "a", "action", fixed("A", "a"));
      rankings.classList.remove("stale");
    }
  }

  function renderRanking(key, ranking, prefix, noun, choose) {
    const { rows, total } = ranking;
    const other = noun === "feature" ? "actions" : "features";
    const subsets = total - (rows.some((r) => r.scheduled) ? 1 : 0);   // the bench's own schedule can be an extra row
    $("ranking-" + key + "-sub").textContent = (subsets === 1
      ? `With one ${noun} there is only one subset.`
      : TIMED ? `All ${subsets} ${noun} subsets, each kept every round, with the ${other} as scheduled on the bench.`
        : `All ${subsets} ${noun} subsets, each run with the chosen ${other}.`)
      + (rows.length < total ? ` The best ${Math.min(8, rows.length)} and the one on the bench are listed.` : "");
    const worst = Math.max(...rows.map((r) => r.discounted), 1e-9);
    const pct = (v) => Math.round(v * 100) + "%";
    $("ranking-" + key).replaceChildren(...rows.map((r) => {
      const tr = document.createElement("tr");
      if (r.current) tr.className = "current";
      tr.tabIndex = 0;
      tr.title = r.current ? "On the bench now" : TIMED ? `Put this subset on the bench, in place of the ${noun} windows` : "Put this subset on the bench";
      const cell = (text, cls) => {
        const td = document.createElement("td");
        if (cls) td.className = cls;
        td.textContent = text;
        return td;
      };
      const members = document.createElement("td");
      if (r.scheduled) members.textContent = "the windows on the bench";
      else r.mask.forEach((on, j) => {
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
      const apply = () => { if (r.scheduled) return; choose(r.mask.slice()); drawAll(); run(); };
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
    if (TIMED) {
      const problem = windowProblem();
      $("window-problem").textContent = problem || "";
      $("window-problem").hidden = !problem;
      if (problem) return;     // the figures keep the last world that could be run
    }
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

  if (TIMED) {
    $("add-window").addEventListener("click", () => {
      // hide the last feature in use (or, with one feature, the last action) for a short stretch early on
      const last = (mask) => mask.lastIndexOf(true);
      const [kind, index] = world.F.filter(Boolean).length > 1 || world.A.filter(Boolean).length < 2 ? ["f", last(world.F)] : ["a", last(world.A)];
      const T = horizon();
      world.W.push({ kind, index, from: Math.round(0.2 * T), to: Math.round(0.25 * T) });
      drawWindows();
      run();
    });
  }

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
    world.W = [];
    for (const part of TIMED && typeof s.W === "string" ? s.W.split(";").slice(0, MAX_WINDOWS) : []) {
      const m = /^([fa])(\d{1,4}):(\d{1,4})-(\d{1,4})$/.exec(part);
      if (m) world.W.push({ kind: m[1], index: Number(m[2]) - 1, from: Math.min(Number(m[3]), LAST_ROUND), to: Math.min(Number(m[4]), LAST_ROUND) });
    }
    resize(world.K, world.n);
    if (s.curve in SPECS) document.querySelector(`input[name="curve"][value="${s.curve}"]`).checked = true;
    applyCurve(true);
    for (const key of ["p1", "T", "delta"]) {
      if (s[key] !== undefined && isFinite(s[key])) inputs(key).forEach((el) => { el.value = s[key]; });
    }
    speeds.load(s.ps);
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
