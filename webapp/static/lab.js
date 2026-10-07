// The browser only collects inputs and shows what Python sends back:
// every number and figure comes from /api/run (simlab on the server).
(function () {
  const SPECS = window.CURVE_SPECS;
  const PRESETS = window.PRESETS;
  const KEYS = ["u1", "u2", "h1", "h2", "p1", "p2", "T", "delta"];
  const $ = (id) => document.getElementById(id);
  const inputs = (key) => document.querySelectorAll(`[data-key="${key}"]`);
  const bench = $("bench");
  const split = $("split");

  const get = (key) => parseFloat(inputs(key)[0].value);
  const set = (key, value) => inputs(key).forEach((el) => { el.value = value; });
  const curve = () => document.querySelector('input[name="curve"]:checked').value;
  const signed = (v) => (v >= 0 ? "+" : "−") + Math.abs(v).toFixed(2);

  // Point the speed sliders at the chosen curve's parameter.
  function applyCurve(resetValues) {
    const spec = SPECS[curve()];
    for (const key of ["p1", "p2"]) {
      inputs(key).forEach((el) => {
        el.min = spec.min; el.max = spec.max; el.step = spec.step;
        if (resetValues) el.value = spec.default;
      });
    }
    $("p1-label").textContent = split.checked ? spec.param + " 1" : spec.param;
    $("p1-label").classList.toggle("a1", split.checked);
    $("p2-label").textContent = spec.param + " 2";
    $("p2-row").hidden = !split.checked;
    $("curve-hint").textContent = spec.param + ": " + spec.hint;
  }

  function state() {
    const s = { curve: curve(), split: split.checked ? 1 : 0 };
    for (const key of KEYS) s[key] = get(key);
    if (!s.split) s.p2 = s.p1;
    return s;
  }

  function load(s) {
    if (s.curve in SPECS) {
      document.querySelector(`input[name="curve"][value="${s.curve}"]`).checked = true;
    }
    if (s.split !== undefined) split.checked = String(s.split) === "1";
    applyCurve(true);
    for (const key of KEYS) {
      if (s[key] !== undefined && s[key] !== "" && isFinite(s[key])) set(key, s[key]);
    }
  }

  function render(data) {
    const numeral = $("numeral");
    numeral.textContent = data.numeral;
    numeral.classList.toggle("word", isNaN(Number(data.numeral)));
    $("numeral-label").textContent = data.numeral_label;
    $("sentence").textContent = data.sentence;
    $("sentence").classList.remove("error");

    const strip = $("strip");
    strip.replaceChildren(...data.runs.map((run) => {
      const cell = document.createElement("i");
      cell.className = run.wrong ? "wrong" : "right";
      cell.style.flexGrow = run.stop - run.start;
      cell.title = `rounds ${run.start}–${run.stop - 1}: ${run.wrong ? "wrong" : "right"} action`;
      return cell;
    }));
    $("strip-end").textContent = "round " + (data.T - 1);

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

    const shade = data.runs.filter((run) => run.wrong).map((run) => [run.start, run.stop]);
    Charts.line($("fig-beliefs"), data.T,
      [{ v: data.series.h1, cls: "c-a1", name: "h\u2081", dot: true }, { v: data.series.h2, cls: "c-a2", name: "h\u2082", dot: true }],
      [{ y: data.u[0], label: "u\u2081 = " + data.u[0].toFixed(2) }, { y: data.u[1], label: "u\u2082 = " + data.u[1].toFixed(2) }],
      // fixed to the sliders' range (plus a little room for the dots), so the axis never jumps
      shade, { ylabel: "utility weight", lo: -3.25, hi: 3.25, yticks: [-3, -2, -1, 0, 1, 2, 3].map((v) => ({ v, label: v })), dotLabel: "starting belief (the h sliders)", alt: "Beliefs h1 and h2 over the rounds, converging to the true weights u1 and u2" });
    Charts.line($("fig-regret"), data.T,
      [{ v: data.series.regret, cls: "c-ref", name: "before discounting" },
        { v: data.series.discounted, cls: "c-ink", name: "discounted, \u03b4 = " + data.delta }],
      [], shade, { ylabel: "regret per round", height: 205, step: true, noXLabel: true, lo: -0.2, hi: 5.2, yticks: [0, 1, 2, 3, 4, 5].map((v) => ({ v, label: v })),
        alt: "Expected regret loss in each round, before and after discounting" });
    Charts.line($("fig-choice"), data.T,
      (data.best ? [{ v: data.series.choice.map(() => data.best), cls: "c-ink", dash: true, name: "y* best move" }] : [])
        .concat([{ v: data.series.choice, cls: "c-ink", name: "\u0177 human's choice" }]),
      [], shade, { ylabel: "action chosen", height: 150, step: true, lo: 0.5, hi: 2.5, fmt: (v) => "action " + v,
        yticks: [{ v: 1, label: "a\u2081" }, { v: 2, label: "a\u2082" }],
        alt: "Which of the two actions the human chooses in each round, against the best move" });
    const gap = data.series.h1.map((v, t) => v - data.series.h2[t]);
    Charts.line($("fig-gap"), data.T, [{ v: gap, cls: "c-ink", name: "h\u2081 \u2212 h\u2082" }],
      [{ y: 0, cls: "zero", label: "zero: indifferent" }, { y: data.du, label: "true gap \u0394u = " + signed(data.du) }],
      shade, { ylabel: "belief gap  h\u2081 \u2212 h\u2082", height: 230, lo: -6.4, hi: 6.4, yticks: [-6, -4, -2, 0, 2, 4, 6].map((v) => ({ v, label: v })), alt: "The belief gap h1 minus h2 over the rounds, against zero and the true gap" });
    Charts.line($("fig-value"), data.T, [{ v: data.series.value_gap, cls: "c-ink", name: "expected value gap" }],
      [{ y: 0, cls: "zero", label: "zero: expects exactly what the best move is worth" }], shade, { ylabel: "value gap per round", lo: -2.6, hi: 2.6, yticks: [-2, -1, 0, 1, 2].map((v) => ({ v, label: v })), alt: "True best value minus the value the human expects from the chosen action, per round" });
    Charts.heat($("fig-map"), $("fig-map-key"), data.map, data.du, data.dh0,
      "Map of the round from which the human is right for good, over the true gap and the initial belief gap");
  }

  // One request at a time, always for the latest inputs: while a slider is moving, each answer
  // is drawn as soon as it arrives and the next request goes out straight away.
  let inFlight = false;
  let stale = false;

  async function run() {
    if (inFlight) { stale = true; return; }
    inFlight = true;
    do {
      stale = false;
      const s = state();
      $("du").textContent = signed(s.u1 - s.u2);
      $("dh0").textContent = signed(s.h1 - s.h2);
      document.querySelectorAll(".preset").forEach((button) => {
        const p = PRESETS[button.dataset.preset];
        button.classList.toggle("active", Object.keys(p).every(
          (k) => k === "name" || k === "title" || String(p[k]) === String(s[k]) || Number(p[k]) === Number(s[k])));
      });
      const query = new URLSearchParams(s).toString();
      history.replaceState(null, "", "?" + query);
      try { sessionStorage.setItem("decision-lab:one-feature", query); } catch (error) { /* storage unavailable: nothing to remember */ }
      try {
        const response = await fetch("/api/run?" + query);
        if (!response.ok) throw new Error("The server answered " + response.status);
        render(await response.json());
      } catch (error) {
        $("sentence").textContent = "Could not run this world: " + error.message + ". Is the Python server still running?";
        $("sentence").classList.add("error");
      }
      bench.setAttribute("aria-busy", "false");
    } while (stale);
    inFlight = false;
  }

  const schedule = run;

  document.querySelectorAll("[data-key]").forEach((el) => {
    el.addEventListener("input", () => {
      if (el.value === "" || isNaN(parseFloat(el.value))) return;
      const clamped = Math.min(Math.max(parseFloat(el.value), parseFloat(el.min)), parseFloat(el.max));
      inputs(el.dataset.key).forEach((other) => { if (other !== el) other.value = clamped; });
      schedule();
    });
    el.addEventListener("change", () => {
      if (el.type !== "number") return;
      const v = parseFloat(el.value);
      set(el.dataset.key, isNaN(v) ? inputs(el.dataset.key)[0].value
        : Math.min(Math.max(v, parseFloat(el.min)), parseFloat(el.max)));
      schedule();
    });
  });

  document.querySelectorAll('input[name="curve"]').forEach((el) =>
    el.addEventListener("change", () => { applyCurve(true); schedule(); }));

  split.addEventListener("change", () => {
    if (split.checked) set("p2", get("p1"));
    applyCurve(false);
    schedule();
  });

  document.querySelectorAll(".preset").forEach((button) =>
    button.addEventListener("click", () => { load(PRESETS[button.dataset.preset]); run(); }));

  // A shared link (?u1=...&curve=...) restores its world; otherwise start from scenario B.
  // ... or, with a bare link, what was on this page before moving to another one.
  let remembered = "";
  try { remembered = sessionStorage.getItem("decision-lab:one-feature") || ""; } catch (error) { /* storage unavailable */ }
  const fromUrl = Object.fromEntries(new URLSearchParams(location.search.length > 1 ? location.search : remembered));
  load(Object.keys(fromUrl).length ? fromUrl : PRESETS[1]);
  run();
})();
