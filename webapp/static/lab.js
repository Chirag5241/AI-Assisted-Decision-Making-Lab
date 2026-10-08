// The browser only collects inputs and shows what Python sends back:
// every number and figure comes from /api/run (simlab on the server).
(function () {
  const SPECS = window.CURVE_SPECS;
  const PRESETS = window.PRESETS;
  const KEYS = ["u1", "u2", "h1", "h2", "p1", "T", "delta", "seed"];
  const $ = (id) => document.getElementById(id);
  const inputs = (key) => document.querySelectorAll(`[data-key="${key}"]`);
  const bench = $("bench");

  const get = (key) => parseFloat(inputs(key)[0].value);
  const set = (key, value) => inputs(key).forEach((el) => { el.value = value; });
  const curve = () => document.querySelector('input[name="curve"]:checked').value;
  const signed = (v) => (v >= 0 ? "+" : "−") + Math.abs(v).toFixed(2);

  // Point the speed slider at the chosen curve's parameter. One feature, so one curve for both actions.
  function applyCurve(resetValues) {
    const spec = SPECS[curve()];
    inputs("p1").forEach((el) => {
      el.min = spec.min; el.max = spec.max; el.step = spec.step;
      if (resetValues) el.value = spec.default;
    });
    $("p1-label").textContent = spec.param;
    $("curve-hint").textContent = spec.param + ": " + spec.hint;
  }

  function state() {
    const s = { curve: curve() };
    for (const key of KEYS) s[key] = get(key);
    return s;
  }

  function load(s) {
    if (s.curve in SPECS) {
      document.querySelector(`input[name="curve"][value="${s.curve}"]`).checked = true;
    }
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

    const [u1, u2] = data.u, h1 = data.series.h1[0], h2 = data.series.h2[0];
    const regretTop = 4 * (Math.abs(data.du) || 0.25);
    const valueTop = 4 * Math.max(Math.abs(Math.max(u1, u2) - Math.max(h1, h2)), Math.abs(Math.min(u1, u2) - Math.min(h1, h2)), 0.25);

    // Figs. 1 to 6, the standard ones, from this bench's answer in the shape every bench hands over: one feature,
    // two actions, both in play every round, and the human right in every state or in none.
    const wrong = [];
    for (const run of data.runs) for (let t = run.start; t < run.stop; t++) wrong.push(run.wrong);
    const always = Array(data.T).fill(1);
    StandardFigures.draw({
      T: data.T, K: 2, U: [[data.u[0]], [data.u[1]]], delta: data.delta,
      beliefs: [[data.series.h1, data.series.h2]],
      schedule: { F: [always], A: [always, always] }, changes: [],
      choice: { picks: data.series.choice, best: data.series.best, missed: wrong },
      series: { reg: data.series.regret, discounted: data.series.discounted, reg_mean: data.series.regret_mean,
        value_gap: data.series.value_gap, value_gap_mean: data.series.value_gap_mean,
        acc: wrong.map((w) => (w ? 0 : 1)) },
    }, {
      // Fixed by the world, not by the draw or the learner. A wrong round costs |du| |x_t|, and a state beyond
      // 4 is as good as never drawn. The value gap is at most |x_t| times the larger of the two gaps between
      // the truth and the first beliefs (best against best, worst against worst).
      regret: { lo: -0.04 * regretTop, hi: 1.04 * regretTop }, value: { lo: -valueTop, hi: valueTop },
      truthLabels: true,
    });

    // Fig. 7: the belief gap, whose sign decides the choice
    const shade = data.runs.filter((run) => run.wrong).map((run) => [run.start, run.stop]);
    const gap = data.series.h1.map((v, t) => v - data.series.h2[t]);
    Charts.line($("fig-gap"), data.T, [{ v: gap, cls: "c-ink", name: "h\u2081 \u2212 h\u2082" }],
      [{ y: 0, cls: "zero", label: "zero: indifferent" }, { y: data.du, label: "true gap \u0394u = " + signed(data.du) }],
      shade, { ylabel: "belief gap  h\u2081 \u2212 h\u2082", lo: -6.4, hi: 6.4, yticks: [-6, -4, -2, 0, 2, 4, 6].map((v) => ({ v, label: v })), alt: "The belief gap h1 minus h2 over the rounds, against zero and the true gap" });
    // Fig. 8: the round of the flip in every nearby world
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

  document.querySelectorAll(".preset").forEach((button) =>
    button.addEventListener("click", () => { load(PRESETS[button.dataset.preset]); run(); }));

  // a fresh draw of the states: another number, picked at random
  $("redraw").addEventListener("click", () => {
    const box = inputs("seed")[0], max = parseInt(box.max, 10);
    let next = Math.floor(Math.random() * (max + 1));
    if (String(next) === box.value) next = (next + 1) % (max + 1);
    inputs("seed").forEach((el) => { el.value = next; });
    schedule();
  });


  // A shared link (?u1=...&curve=...) restores its world; otherwise start from scenario B.
  // ... or, with a bare link, what was on this page before moving to another one.
  let remembered = "";
  try { remembered = sessionStorage.getItem("decision-lab:one-feature") || ""; } catch (error) { /* storage unavailable */ }
  const fromUrl = Object.fromEntries(new URLSearchParams(location.search.length > 1 ? location.search : remembered));
  load(Object.keys(fromUrl).length ? fromUrl : PRESETS[1]);
  run();
})();
