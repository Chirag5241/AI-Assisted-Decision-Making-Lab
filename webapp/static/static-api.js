// The static build of the site (GitHub Pages) has no Python server. This file answers the
// data endpoints, /api/run, /api/world and /api/correlated, in the browser, with the same JSON
// that webapp/app.py returns, so the pages, stylesheet and scripts are exactly the ones the Flask
// site uses. Keep it in step with app.py; tests/test_static_api.py compares the two.
(function (root) {
  "use strict";

  const MEAN_ABS_X = Math.sqrt(2 / Math.PI);          // E|x| for x ~ N(0, 1)
  const MAX_ACTIONS = 6, MAX_FEATURES = 6, MAX_WINDOWS = 12, MAX_SEED = 999;
  const AWARE_STREAM = 7919;    // the random fallback of the state-aware policy draws from its own stream of the seed

  const sigmoid = (p, m) => {
    const w = p / 5, s = 1 / (1 + Math.exp(-(m - p) / w)), s0 = 1 / (1 + Math.exp(p / w));
    return (s - s0) / (1 - s0);
  };
  const CURVES = {
    "exponential": { min: 0.01, max: 0.5, def: 0.1, phi: (p, m) => 1 - Math.pow(1 - p, m) },
    "hyperbolic": { min: 1, max: 50, def: 5, phi: (p, m) => m / (m + p) },
    "power law": { min: 0.1, max: 2, def: 0.5, phi: (p, m) => 1 - Math.pow(1 + m, -p) },
    "sigmoid": { min: 2, max: 80, def: 20, phi: sigmoid },
  };

  // ---- small helpers that mirror the Python side ---------------------------

  class BadRequest extends Error {}

  function num(q, name, def, lo, hi) {
    const raw = q.get(name);
    if (raw === null || raw.trim() === "") return def;
    const v = Number(raw);
    return Number.isFinite(v) ? Math.min(Math.max(v, lo), hi) : def;
  }
  const round = (v, digits) => { const f = Math.pow(10, digits); return Math.round(v * f) / f; };
  const f2 = (v) => v.toFixed(2);
  const signed2 = (v) => (v < 0 ? "-" : "+") + Math.abs(v).toFixed(2);
  const pct = (v) => (v * 100).toFixed(0) + "%";
  const humanList = (items) => (items.length === 1 ? String(items[0])
    : items.slice(0, -1).join(", ") + " and " + items[items.length - 1]);
  function describe(mask, noun) {
    const chosen = [];
    mask.forEach((on, i) => { if (on) chosen.push(i + 1); });
    if (chosen.length === mask.length) return mask.length > 1 ? `all ${mask.length} ${noun}s` : `the only ${noun}`;
    return `${noun}${chosen.length > 1 ? "s" : ""} ${humanList(chosen)}`;
  }
  // first round from which every later round is right; Infinity if the last round is wrong
  function settleTime(wrong) {
    const T = wrong.length;
    let lastWrong = -1;
    for (let t = 0; t < T; t++) if (wrong[t]) lastWrong = t;
    return lastWrong === T - 1 ? Infinity : lastWrong + 1;
  }
  function flipTimeScale(T) {
    const bounds = [0, 1, 2, 5, 10, 20, 50, 100].filter((b) => b < T).concat([T, T + 1]);
    const labels = ["0 (always right)"];
    for (let i = 1; i < bounds.length - 2; i++) {
      const lo = bounds[i], hi = bounds[i + 1];
      labels.push(hi - lo === 1 ? String(lo) : `${lo}–${hi - 1}`);
    }
    labels.push(`never (≥${T})`);
    // index of the bin a settle round falls in (the last bin is "never")
    const bin = (settle) => {
      const v = Number.isFinite(settle) ? settle : T;
      let i = 0;
      while (i + 1 < bounds.length && bounds[i + 1] <= v) i++;
      return i;
    };
    return { labels, bin };
  }

  // The state of each round, x_t ~ N(0, I), as [round][feature]: mulberry32, then Box-Muller. These are
  // the very numbers simlab/draws.py makes, so a draw is the same on the Flask site and here.
  // the i-th 32-bit number of the stream `seed` (mulberry32, counter form)
  function uniform32(seed, i) {
    const a = (seed + (i + 1) * 0x6D2B79F5) >>> 0;
    let t = Math.imul(a ^ (a >>> 15), a | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return (t ^ (t >>> 14)) >>> 0;
  }
  function drawStates(seed, T, n) {
    const uniform = (i) => uniform32(seed, i);
    return Array.from({ length: T }, (_, t) => Array.from({ length: n }, (_, j) => {
      const m = t * n + j, u1 = (uniform(2 * m) + 0.5) / 4294967296, u2 = uniform(2 * m + 1) / 4294967296;
      return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
    }));
  }
  const seedOf = (q) => Math.trunc(num(q, "seed", 0, 0, MAX_SEED));

  // ---- /api/run: one feature, two actions ----------------------------------

  function run(q) {
    const curveName = q.has("curve") ? q.get("curve") : "exponential";
    if (!(curveName in CURVES)) throw new BadRequest("unknown curve");
    const spec = CURVES[curveName];
    const speed = num(q, "p1", spec.def, spec.min, spec.max);   // one feature, so one curve for both actions
    const u1 = num(q, "u1", 1.2, -3, 3), u2 = num(q, "u2", 1.0, -3, 3);
    const h1 = num(q, "h1", 0.0, -3, 3), h2 = num(q, "h2", 1.5, -3, 3);
    const T = Math.trunc(num(q, "T", 80, 10, 400)), delta = num(q, "delta", 0.95, 0.01, 0.99), seed = seedOf(q);
    const du = u1 - u2, dh0 = h1 - h2;

    // everything is shown every round, so the beliefs follow the learning curve in closed form
    const left = [], b1 = [], b2 = [], gap = [], wrong = [];
    for (let t = 0; t < T; t++) {
      left.push(1 - spec.phi(speed, t));
      b1.push(u1 + (h1 - u1) * left[t]); b2.push(u2 + (h2 - u2) * left[t]);
      gap.push(b1[t] - b2[t]);
      wrong.push(du !== 0 && !(gap[t] * du > 0));
    }
    const settle = settleTime(wrong);
    let switches = 0;
    for (let t = 1; t < T; t++) if (wrong[t] !== wrong[t - 1]) switches++;
    const nWrong = wrong.filter(Boolean).length;

    // The state of each round is drawn, x_t ~ N(0, 1), and the loss is taken there: |du| |x_t| on a wrong
    // round. Averaged over every state instead, that is |du| E|x| per wrong round.
    const x = drawStates(seed, T, 1).map((row) => row[0]);
    const regretPerRound = wrong.map((w, t) => (w ? Math.abs(du) * Math.abs(x[t]) : 0));
    const discounted = regretPerRound.map((r, t) => Math.pow(delta, t) * r);
    const regret = discounted.reduce((a, b) => a + b, 0);
    const wrongCost = Math.abs(du) * MEAN_ABS_X;
    const meanRegretPerRound = wrong.map((w) => (w ? wrongCost : 0));
    const meanRegret = meanRegretPerRound.reduce((sum, r, t) => sum + Math.pow(delta, t) * r, 0);
    // the value gap at x_t, max_k u_k x_t - max_k h_k x_t, and its average over states
    const valueGap = gap.map((_, t) => Math.max(u1 * x[t], u2 * x[t]) - Math.max(b1[t] * x[t], b2[t] * x[t]));
    const meanValueGap = gap.map((g) => ((Math.abs(du) - Math.abs(g)) * MEAN_ABS_X) / 2);
    // the best move and the choice at x_t: both swap with the sign of x_t, and differ exactly on the wrong rounds
    const choice = wrong.map((w, t) => {
      const first = u2 * x[t] > u1 * x[t] ? 1 : 0;
      return du === 0 ? (b2[t] * x[t] > b1[t] * x[t] ? 1 : 0) : (w ? 1 - first : first);
    });
    const best = x.map((v, t) => (du === 0 ? choice[t] : (u2 * v > u1 * v ? 1 : 0)));     // equally good actions: whichever the human picks

    const gaps = `belief gap ${signed2(dh0)} against a true gap of ${signed2(du)}`;
    let numeral, label, sentence;
    if (du === 0) {
      [numeral, label, sentence] = ["0", "no flip needed", "Both actions are worth exactly the same, so whatever the human picks is right."];
    } else if (nWrong === 0) {
      [numeral, label, sentence] = ["0", "no flip needed", `The human starts with the right sign (${gaps}) and never picks the wrong action.`];
    } else if (!Number.isFinite(settle)) {
      [numeral, label, sentence] = ["never", `within ${T} rounds`,
        `With a ${gaps}, the human is still picking the wrong action at the end of the horizon. Lengthen the horizon or speed up the learner to see the flip.`];
    } else {      // the shared curve lets the belief gap cross zero only once
      [numeral, label, sentence] = [String(settle), "round of the flip",
        `The human starts with the wrong sign (${gaps}), picks the wrong action for ${nWrong} round${nWrong !== 1 ? "s" : ""}, and is right from round ${settle} on.`];
    }

    let needed;
    if (du === 0) needed = { value: "—", note: "the actions are equally good" };
    else {
      const phiStar = du * dh0 > 0 ? 0 : Math.abs(dh0) / (Math.abs(dh0) + Math.abs(du));
      needed = { value: pct(phiStar), note: "of the belief-to-truth gap must close before the sign is right" };
    }

    const runs = [];
    for (let start = 0, t = 1; t <= T; t++) {
      if (t < T && wrong[t] === wrong[start]) continue;
      runs.push({ start, stop: t, wrong: wrong[start] });
      start = t;
    }

    // settle round over a grid of (du, dh0) worlds that keep this world's mean levels; row 0 is the largest dh0
    const G = 60, uMid = (u1 + u2) / 2, hMid = (h1 + h2) / 2;
    const lim = Math.ceil(Math.max(3, 1.15 * Math.abs(du), 1.15 * Math.abs(dh0)));
    const stepSize = (2 * lim) / (G - 1);   // the same arithmetic as numpy.linspace, so grid points match to the last bit
    const axis = Array.from({ length: G }, (_, i) => (i === G - 1 ? lim : -lim + i * stepSize));
    const scale = flipTimeScale(T), bins = [];
    for (let r = 0; r < G; r++) {
      const gh = axis[G - 1 - r], row = [];
      for (let c = 0; c < G; c++) {
        const gu = axis[c];
        const e1 = (hMid + gh / 2) - (uMid + gu / 2), e2 = (hMid - gh / 2) - (uMid - gu / 2);
        let lastWrong = -1;
        for (let t = 0; t < T; t++) {
          const g = (uMid + gu / 2 + e1 * left[t]) - (uMid - gu / 2 + e2 * left[t]);
          if (!(g * gu > 0)) lastWrong = t;
        }
        row.push(scale.bin(lastWrong === T - 1 ? Infinity : lastWrong + 1));
      }
      bins.push(row);
    }

    return {
      du, dh0, T, u: [u1, u2], delta,
      numeral, numeral_label: label, sentence,
      settle: Number.isFinite(settle) ? settle : null, runs,
      stats: [
        Object.assign({ label: "Learning needed" }, needed),
        { label: "Rounds wrong", value: String(nWrong), note: `out of ${T} simulated` },
        { label: "Choice changes", value: String(switches), note: "right/wrong switches" },
        { label: "Discounted regret on this draw", value: f2(regret),
          note: wrong[T - 1] ? `${f2(meanRegret)} averaged over states; a lower bound, still wrong at round ${T}`
            : `${f2(meanRegret)} averaged over states x ~ N(0, 1)` },
      ],
      series: {
        h1: b1.map((v) => round(v, 4)), h2: b2.map((v) => round(v, 4)), x: x.map((v) => round(v, 3)),
        regret: regretPerRound.map((v) => round(v, 4)), discounted: discounted.map((v) => round(v, 4)),
        value_gap: valueGap.map((v) => round(v, 4)), choice: choice.map((c) => c + 1), best: best.map((c) => c + 1),
        regret_mean: meanRegretPerRound.map((v) => round(v, 4)), value_gap_mean: meanValueGap.map((v) => round(v, 4)),
      },
      seed,
      map: { lim, G, bins, labels: scale.labels },
    };
  }

  // ---- /api/world: K actions, n features -----------------------------------

  // The probe states are the ones the Python site draws (numpy default_rng(0)), shipped as base64 float32.
  const probeCache = {};
  function probes(n) {
    if (probeCache[n]) return probeCache[n];
    const packed = root.LAB_PROBES && root.LAB_PROBES[n];
    if (!packed) throw new Error("probe states for " + n + " features are missing");
    const binary = typeof atob === "function" ? atob(packed) : Buffer.from(packed, "base64").toString("binary");
    const bytes = new Uint8Array(binary.length);
    for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
    return (probeCache[n] = new Float32Array(bytes.buffer));
  }

  function matrix(q, name, K, n) {
    const rows = (q.get(name) || "").split(";").map((row) => row.split(",").map((v) => (v.trim() === "" ? NaN : Number(v))));
    if (rows.length !== K || rows.some((row) => row.length !== n || row.some((v) => !Number.isFinite(v)))) {
      throw new BadRequest(`${name} must be a ${K} x ${n} matrix`);
    }
    return rows.map((row) => row.map((v) => Math.min(Math.max(v, -3), 3)));
  }
  function mask(q, name, size) {
    const m = (q.has(name) ? q.get(name) : Array(size).fill("1").join(",")).split(",").map((v) => v === "1");
    if (m.length !== size || !m.some(Boolean)) throw new BadRequest(`${name} must switch on at least one of ${size}`);
    return m;
  }
  // The learning speed of each feature: ps='0.05,0.1,0.2' gives every feature its own, otherwise all share p1.
  function speedsOf(q, spec, n) {
    const raw = (q.get("ps") || "").trim();
    if (!raw) return Array(n).fill(num(q, "p1", spec.def, spec.min, spec.max));
    const speeds = raw.split(",").map((v) => (v.trim() === "" ? NaN : Number(v)));
    if (speeds.some((v) => Number.isNaN(v))) throw new BadRequest("ps must be numbers");
    if (speeds.length !== n || !speeds.every(Number.isFinite)) throw new BadRequest(`ps must be ${n} speeds, one per feature`);
    return speeds.map((v) => Math.min(Math.max(v, spec.min), spec.max));
  }

  // every non-empty on/off mask, in the order of Python's itertools.product([False, True], repeat=size)
  const subsets = (size) => Array.from({ length: (1 << size) - 1 }, (_, m) =>
    Array.from({ length: size }, (_, j) => Boolean(((m + 1) >> (size - 1 - j)) & 1)));

  // W='f3:20-25;a1:40-60': feature 3 hidden in rounds 20 to 25, action 1 not offered in rounds 40 to 60,
  // both ends included; cut to the horizon.
  function parseWindows(q, K, n, T) {
    const raw = (q.get("W") || "").trim();
    if (!raw) return [];
    const parts = raw.split(";");
    if (parts.length > MAX_WINDOWS) throw new BadRequest(`at most ${MAX_WINDOWS} windows`);
    const windows = [];
    for (const part of parts) {
      const match = /^([fa])(\d{1,4}):(\d{1,4})-(\d{1,4})$/.exec(part.trim());
      if (!match) throw new BadRequest("W must look like f3:20-25;a1:40-60");
      const kind = match[1], index = Number(match[2]), first = Number(match[3]), last = Number(match[4]);
      if (index < 1 || index > (kind === "f" ? n : K) || first > last) {
        throw new BadRequest("a window names an existing feature or action and a first round no later than its last");
      }
      if (first < T) windows.push({ kind, index: index - 1, first, last: Math.min(last, T - 1) });
    }
    return windows;
  }

  async function world(q, signal) {
    const curveName = q.has("curve") ? q.get("curve") : "exponential";
    if (!(curveName in CURVES)) throw new BadRequest("unknown curve");
    const spec = CURVES[curveName];
    const K = Math.trunc(num(q, "K", 3, 2, MAX_ACTIONS)), n = Math.trunc(num(q, "n", 3, 1, MAX_FEATURES));
    const T = Math.trunc(num(q, "T", 100, 10, 300));
    let Fbase = mask(q, "F", n), Abase = mask(q, "A", K), windows = parseWindows(q, K, n, T);
    // the state-aware policy chooses the features itself, within a budget, and offers every action
    const aware = q.get("policy") === "aware"
      ? { budget: Math.trunc(num(q, "C", 2, 1, n)), judge: q.get("judge") === "truth" ? "truth" : "belief",
        prefer: q.get("prefer") === "most" ? "most" : "margin" } : null;
    if (aware) { Fbase = Array(n).fill(true); Abase = Array(K).fill(true); windows = []; }
    // what is shown and what is offered in each round: the base mask, switched off inside each window
    const scheduleOf = (base, kind) => Array.from({ length: T }, (_, t) =>
      base.map((on, i) => on && !windows.some((w) => w.kind === kind && w.index === i && w.first <= t && t <= w.last)));
    let Fbench = scheduleOf(Fbase, "f");
    const Abench = scheduleOf(Abase, "a");
    for (const [rounds, problem] of [[Fbench, "show no feature"], [Abench, "offer no action"]]) {
      const empty = rounds.findIndex((m) => !m.some(Boolean));
      if (empty >= 0) throw new BadRequest(`round ${empty} would ${problem}`);
    }
    const U = matrix(q, "U", K, n), H0 = matrix(q, "H", K, n);
    const speeds = speedsOf(q, spec, n);
    const delta = num(q, "delta", 0.95, 0.01, 0.99), seed = seedOf(q);
    const full = q.get("rank") !== "0";
    const xDraw = drawStates(seed, T, n);     // the state of each round; every policy is run on this draw

    const X = probes(n), P = X.length / n;
    // phis[j][m]: how much of feature j's gap is closed after m rounds in use
    const phis = speeds.map((speed) => Array.from({ length: T + 1 }, (_, m) => spec.phi(speed, m)));

    // what every action is truly worth in every probe state, and the regret of picking it
    const regretOf = new Float64Array(P * K), bestTrue = new Float64Array(P);
    let regretCap = 0, valueCap = -Infinity;      // the fixed axes: the largest regret, and value, any probe state allows
    for (let p = 0; p < P; p++) {
      let hi = -Infinity, lo = Infinity, believed = -Infinity;
      for (let k = 0; k < K; k++) {
        let v = 0, h = 0;
        for (let j = 0; j < n; j++) { v += U[k][j] * X[p * n + j]; h += H0[k][j] * X[p * n + j]; }
        regretOf[p * K + k] = v;
        if (v > hi) hi = v;
        if (v < lo) lo = v;
        if (h > believed) believed = h;
      }
      for (let k = 0; k < K; k++) regretOf[p * K + k] = hi - regretOf[p * K + k];
      bestTrue[p] = hi;
      regretCap = Math.max(regretCap, hi - lo); valueCap = Math.max(valueCap, hi, believed);
    }

    // The round's own loss, at the state x drawn for it and with the weights W[k][j] the human holds then:
    // regret, value gap, the pick and the best move. Actions that tie share the regret; the pick is the first.
    function realized(W, F, A, x) {
      const worth = U.map((row) => row.reduce((sum, u, j) => sum + u * x[j], 0));
      let hi = -Infinity, best = -1, top = -Infinity, sumR = 0, count = 0, pick = -1;
      worth.forEach((v, k) => { if (v > hi) { hi = v; best = k; } });
      for (let k = 0; k < K; k++) {
        if (!A[k]) continue;
        let v = 0;
        for (let j = 0; j < n; j++) if (F[j]) v += W[k][j] * x[j];
        if (v > top) { top = v; sumR = hi - worth[k]; count = 1; pick = k; }
        else if (v === top) { sumR += hi - worth[k]; count++; }
      }
      // when several moves are exactly equally good, the best move reported is the human's pick if it is one of them
      return { reg: sumR / count, gap: hi - top, pick, best: sumR / count <= 1e-12 ? pick : best };
    }

    // Every action's score in every probe state for one set of shown features, from the first beliefs
    // (e0) and from the truth (e1). A weight in use moves from H0 to U along its feature's curve, so while
    // an action's shown weights are all equally far along, its score is a blend of the two.
    const blends = new Map();
    function blend(F) {
      const key = F.map((on) => (on ? 1 : 0)).join("");
      if (!blends.has(key)) {
        const e0 = new Float64Array(P * K), e1 = new Float64Array(P * K);
        for (let p = 0; p < P; p++) for (let k = 0; k < K; k++) {
          let s0 = 0, s1 = 0;
          for (let j = 0; j < n; j++) if (F[j]) { const x = X[p * n + j]; s0 += H0[k][j] * x; s1 += U[k][j] * x; }
          e0[p * K + k] = s0; e1[p * K + k] = s1;
        }
        blends.set(key, { e0, e1 });
      }
      return blends.get(key);
    }

    // Accuracy, regret and value gap of one round: features F shown, actions A offered, and N[k][j]
    // rounds of learning behind each weight (no N: everything in use is fully learned).
    function evaluate(F, A, N) {
      const ks = [], js = [];
      A.forEach((on, k) => { if (on) ks.push(k); });
      F.forEach((on, j) => { if (on) js.push(j); });
      let even = !N || js.every((j) => speeds[j] === speeds[js[0]]);
      const f = ks.map((k) => {
        if (!N) return 1;
        if (js.some((j) => N[k][j] !== N[k][js[0]])) even = false;
        return phis[js[0]][N[k][js[0]]];
      });
      const mix = even ? blend(F) : null;
      // separate speeds, or windows, have left an action's shown weights at different stages: score from the weights themselves
      const weights = even ? null : ks.map((k) => js.map((j) => U[k][j] + (H0[k][j] - U[k][j]) * (1 - phis[j][N[k][j]])));
      let acc = 0, reg = 0, gap = 0;
      for (let p = 0; p < P; p++) {
        let top = -Infinity, sumR = 0, sumOk = 0, count = 0;
        for (let a = 0; a < ks.length; a++) {
          const i = p * K + ks[a], r = regretOf[i];
          let v = 0;
          if (even) v = mix.e0[i] + f[a] * (mix.e1[i] - mix.e0[i]);
          else for (let b = 0; b < js.length; b++) v += weights[a][b] * X[p * n + js[b]];
          if (v > top) { top = v; sumR = r; sumOk = r <= 1e-12 ? 1 : 0; count = 1; }
          else if (v === top) { sumR += r; sumOk += r <= 1e-12 ? 1 : 0; count++; }   // ties are split evenly
        }
        acc += sumOk / count; reg += sumR / count; gap += bestTrue[p] - top;
      }
      return [acc / P, reg / P, gap / P];
    }

    // One policy: the features shown and the actions offered in each round. `real` holds the loss of each
    // round at its drawn state and `disc` its discounted sum, the objective on this draw; acc, reg and vgap are
    // the averages over states and `mean` their discounted sum. With `keep`, the beliefs of every round are
    // returned too, as [round][action][feature].
    function score(Ft, At, keep) {
      const N = U.map((row) => row.map(() => 0));
      const acc = [], reg = [], vgap = [], beliefs = [], real = [];
      let disc = 0, mean = 0;
      for (let t = 0; t < T; t++) {
        const [a, r, g] = evaluate(Ft[t], At[t], N);
        acc.push(a); reg.push(r); vgap.push(g); mean += Math.pow(delta, t) * r;
        const W = U.map((row, k) => row.map((u, j) => u + (H0[k][j] - u) * (1 - phis[j][N[k][j]])));
        real.push(realized(W, Ft[t], At[t], xDraw[t]));
        disc += Math.pow(delta, t) * real[t].reg;
        if (keep) beliefs.push(W);
        for (let k = 0; k < K; k++) if (At[t][k]) for (let j = 0; j < n; j++) if (Ft[t][j]) N[k][j]++;
      }
      // once everything shown is learned: the last round's choice, kept up until every weight in use is right
      const [accLimit, regLimit] = evaluate(Ft[T - 1], At[T - 1], null);
      return { acc, reg, vgap, disc, mean, real, accLimit, regLimit, beliefs };
    }

    // Ranking every subset can take a while for large worlds: give the page a turn now and then,
    // and stop if it has moved on to newer inputs.
    let lastPause = Date.now();
    const breathe = async () => {
      if (signal && signal.aborted) throw aborted();
      if (Date.now() - lastPause < 24) return;
      await nextTask();
      lastPause = Date.now();
      if (signal && signal.aborted) throw aborted();
    };

    const same = (a, b) => a.every((v, i) => v === b[i]);
    const every = (set) => Array(T).fill(set);                    // a fixed subset, the same every round
    const steady = (rounds) => rounds.every((m) => same(m, rounds[0]));

    // ---- the state-aware policy: for each round's state, a feature subset on which the best move ranks first ----
    //
    // A subset F works in a state x when the best move a = argmax_k (U x)_k ranks first on the shown features alone:
    //     sum_{i in F} w_ai x_i  >  sum_{i in F} w_ki x_i   for every other move k.
    // Its margin is the left side minus the largest right side. Judged by the human's weights (w = H_t, the
    // default), the human picks a from F, so the round has no error; among such subsets one on which the truth
    // agrees is preferred. Judged by the truth alone (w = U), the human may still be wrong until they have learned
    // it. Among equals the top-ranked is the widest margin (or the most features). When no subset works a random
    // candidate is shown.
    //
    // The search is exhaustive over the subsets of at most `budget` features, 2^n - 1 of them when the budget is n,
    // so exponential in the number of features (see webapp/app.py). Here n <= 6.
    let candidates = null, members = null, worked = null;
    const isBest = new Uint8Array(K);      // which moves are best in the state being judged
    // the margin of candidate s by the weights W in the state xs[at + j]: the best score among the best moves minus
    // the best score among the rest (moves that are exactly equally good are all correct)
    function marginOf(s, W, xs, at) {
      const js = members[s];
      let topBest = -Infinity, topRest = -Infinity;
      for (let k = 0; k < K; k++) {
        let v = 0;
        for (let i = 0; i < js.length; i++) v += W[k][js[i]] * xs[at + js[i]];
        if (isBest[k]) { if (v > topBest) topBest = v; } else if (v > topRest) topRest = v;
      }
      return topBest - topRest;
    }
    // Judge every candidate in one state by the weights W: how many work, and the top-ranked of those (-1 if none).
    // With `agree` (the human's weights are judging), a candidate on which the truth agrees is preferred and ranked
    // by the truth's margin. Margins closer than 1e-9 count as equal (to zero, or to each other): then the first
    // candidate in order is taken, so rounding never decides.
    const margins = [], truthMargins = [];
    function judgeState(W, xs, at, agree) {
      const S = candidates.length;
      let count = 0, agreed = false;
      for (let s = 0; s < S; s++) {
        margins[s] = marginOf(s, W, xs, at);
        if (!(margins[s] > 1e-9)) continue;
        count++;
        if (agree) { truthMargins[s] = marginOf(s, U, xs, at); if (truthMargins[s] > 1e-9) agreed = true; }
      }
      if (!count) return [0, -1];
      const inPool = (s) => margins[s] > 1e-9 && (!agreed || truthMargins[s] > 1e-9);
      const key = (s) => (aware.prefer === "most" ? members[s].length : agreed ? truthMargins[s] : margins[s]);
      let best = -Infinity;
      for (let s = 0; s < S; s++) if (inPool(s) && key(s) > best) best = key(s);
      for (let s = 0; s < S; s++) if (inPool(s) && key(s) >= best - 1e-9) return [count, s];
      return [count, -1];
    }
    // The policy's accuracy, regret and value gap with the human's weights W[k][j], averaged over the probe states.
    // In each state it shows the candidate it would pick there and the human chooses from it; where no candidate
    // works a random one is shown, so the three are averaged over the candidates. `pickedAt[p]` is that pick when
    // the truth judges, which is the same every round; without it the human's own weights judge.
    function awareAverage(W, pickedAt) {
      const S = candidates.length, out = [0, 0, 0];
      // the human's choice under candidate s in probe state p, added to `out` with weight w
      const human = (s, p, w) => {
        const js = members[s];
        let top = -Infinity, sumR = 0, sumOk = 0, count = 0;
        for (let k = 0; k < K; k++) {
          let v = 0;
          for (let i = 0; i < js.length; i++) v += W[k][js[i]] * X[p * n + js[i]];
          const r = regretOf[p * K + k];
          if (v > top) { top = v; sumR = r; sumOk = r <= 1e-12 ? 1 : 0; count = 1; }
          else if (v === top) { sumR += r; sumOk += r <= 1e-12 ? 1 : 0; count++; }     // ties are split evenly
        }
        out[0] += (w * sumOk) / count; out[1] += (w * sumR) / count; out[2] += w * (bestTrue[p] - top);
      };
      for (let p = 0; p < P; p++) {
        let chosen;
        if (pickedAt) chosen = pickedAt[p];
        else {
          for (let k = 0; k < K; k++) isBest[k] = regretOf[p * K + k] <= 1e-12 ? 1 : 0;
          chosen = judgeState(W, X, p * n, true)[1];
        }
        if (chosen >= 0) human(chosen, p, 1);
        else for (let s = 0; s < S; s++) human(s, p, 1 / S);
      }
      return out.map((v) => v / P);
    }
    if (aware) {
      candidates = subsets(n).filter((set) => set.filter(Boolean).length <= aware.budget);
      members = candidates.map((set) => set.reduce((js, on, j) => (on ? js.concat(j) : js), []));
      const S = candidates.length;
      // Run it on the drawn states. Every action is offered, so a shown feature's weights all learn together.
      const shownRounds = Array(n).fill(0);
      Fbench = []; worked = [];
      for (let t = 0; t < T; t++) {
        const x = xDraw[t], worth = U.map((row) => row.reduce((sum, u, j) => sum + u * x[j], 0));
        const bestWorth = Math.max(...worth);
        worth.forEach((v, k) => { isBest[k] = bestWorth - v <= 1e-12 ? 1 : 0; });
        const W = aware.judge === "truth" ? U
          : U.map((row, k) => row.map((u, j) => u + (H0[k][j] - u) * (1 - phis[j][shownRounds[j]])));
        let [count, pick] = judgeState(W, x, 0, aware.judge !== "truth");
        // none works: one of the candidates at random
        if (pick < 0) pick = Math.floor((uniform32(seed + AWARE_STREAM, t) * S) / 4294967296);
        worked.push(count);
        Fbench.push(candidates[pick]);
        candidates[pick].forEach((on, j) => { if (on) shownRounds[j]++; });
      }
    }

    const now = score(Fbench, Abench, true);
    if (aware) {
      // Averaged over states, the bench is not the subset it happened to show: in another state it would have
      // shown another. Score the policy itself, with the beliefs of each round, and with the truth.
      const byTruth = new Int32Array(P);
      let settled = 0;
      for (let p = 0; p < P; p++) {
        for (let k = 0; k < K; k++) isBest[k] = regretOf[p * K + k] <= 1e-12 ? 1 : 0;
        byTruth[p] = judgeState(U, X, p * n, false)[1];
        if (byTruth[p] >= 0) settled++;
      }
      // the ceiling the budget sets: the share of states in which some candidate settles the best move by the
      // truth; in the rest a hidden feature is needed, whatever the human believes
      aware.ceiling = settled / P;
      const pickedAt = aware.judge === "truth" ? byTruth : null;
      now.mean = 0;
      for (let t = 0; t < T; t++) {
        [now.acc[t], now.reg[t], now.vgap[t]] = awareAverage(now.beliefs[t], pickedAt);
        now.mean += Math.pow(delta, t) * now.reg[t];
        await breathe();
      }
      [now.accLimit, now.regLimit] = awareAverage(U, pickedAt);
    }
    const heldBack = !(Fbench.every((m) => m.every(Boolean)) && Abench.every((m) => m.every(Boolean)));
    const everything = heldBack ? score(every(Array(n).fill(true)), every(Array(K).fill(true))) : now;

    // What the last round holds back decides the lasting loss. Usually that is what was held back all
    // along; a window that runs to the end of the horizon leaves something out that was in use before.
    const Fend = Fbench[T - 1], Aend = Abench[T - 1];
    const used = (rounds) => rounds[0].map((_, i) => rounds.some((m) => m[i]));
    const settled = same(used(Fbench), Fend) && same(used(Abench), Aend);
    const scheduled = !(steady(Fbench) && steady(Abench));
    const hidden = [], offF = Fend.map((v) => !v), offA = Aend.map((v) => !v);
    if (offF.some(Boolean)) {
      const one = offF.filter(Boolean).length === 1;
      hidden.push(describe(offF, "feature") + (settled ? (one ? " stays" : " stay") + " hidden" : (one ? " is" : " are") + " hidden at the end"));
    }
    if (offA.some(Boolean)) {
      hidden.push(describe(offA, "action") + (offA.filter(Boolean).length === 1 ? " is" : " are") + (settled ? " never offered" : " not offered at the end"));
    }
    const live = windows.filter((w) => (w.kind === "f" ? Fbase : Abase)[w.index]);
    const windowText = !live.length ? "" : live.length > 2 ? `, with ${live.length} timed windows`
      : ", with " + live.map((w) => `${w.kind === "f" ? "feature" : "action"} ${w.index + 1} ${w.kind === "f" ? "hidden" : "not offered"} in `
        + (w.first === w.last ? `round ${w.first}` : `rounds ${w.first}–${w.last}`)).join(" and ");
    let sentence = `Showing ${describe(Fbase, "feature")} and offering ${describe(Abase, "action")}${windowText}, the human picks the best move ${pct(now.acc[0])} of the time at round 0 and ${pct(now.acc[T - 1])} by round ${T - 1}. `;
    if (now.regLimit > 1e-9) {
      sentence += (settled ? `Even after learning everything shown, ${f2(now.regLimit)} of utility is lost per round`
        : `If the last round's choice carried on until everything shown is learned, ${f2(now.regLimit)} of utility would still be lost per round`)
        + (hidden.length ? ` because ${humanList(hidden)}. ` : ". ");
    } else {
      sentence += settled ? "Once everything shown is learned, the human always picks the best move. "
        : "Once everything shown at the end is learned, the human always picks the best move. ";
    }

    let rankings = null;
    if (full) {
      // Rank one family of fixed subsets; the bench is one of them unless its windows change this
      // family over the rounds, and then it is ranked as an extra row.
      const rank = async (sets, bench, noun, policy) => {
        const fixed = steady(bench) && !aware, results = [];     // the state-aware policy is never one of the fixed subsets
        for (const set of sets) { results.push(fixed && same(set, bench[0]) ? now : score(...policy(set))); await breathe(); }
        if (!fixed) results.push(now);
        const order = results.map((_, i) => i).sort((a, b) => results[a].disc - results[b].disc || a - b);
        const place = {};
        order.forEach((i, r) => { place[i] = r + 1; });
        const here = fixed ? sets.findIndex((set) => same(set, bench[0])) : sets.length;
        const listed = [...new Set(order.slice(0, 8).concat([here]))].sort((a, b) => place[a] - place[b]);
        const best = order.find((i) => i < sets.length);
        return {
          rows: listed.map((i) => {
            const extra = i === sets.length;
            return { rank: place[i], mask: extra ? null : sets[i],
              label: extra ? (aware ? "the policy on the bench" : "the schedule on the bench") : describe(sets[i], noun),
              discounted: results[i].disc, mean: results[i].mean, start: results[i].acc[0], end: results[i].acc[T - 1], floor: results[i].regLimit,
              current: i === here, scheduled: extra };
          }),
          total: results.length, rank: place[here], best: { set: sets[best], disc: results[best].disc },
        };
      };
      // (the state-aware policy is set against the fixed subsets within its budget, and no action subsets)
      const byFeature = await rank(aware ? candidates : subsets(n), Fbench, "feature", (set) => [every(set), Abench]);
      const byAction = aware ? null : await rank(subsets(K), Abench, "action", (set) => [Fbench, every(set)]);
      if (!aware) {
        const kept = scheduled ? " every round" : "";       // a fixed subset, against a schedule that changes
        const featureHelps = byFeature.best.disc < now.disc - 1e-9;
        if (featureHelps) sentence += `On this draw, showing ${describe(byFeature.best.set, "feature")}${kept} instead would cut the discounted regret to ${f2(byFeature.best.disc)}. `;
        if (byAction.best.disc < now.disc - 1e-9) {
          const offer = `ffering ${describe(byAction.best.set, "action")}${kept} instead would cut`;
          sentence += featureHelps ? `Separately, o${offer} it to ${f2(byAction.best.disc)}.` : `On this draw, o${offer} the discounted regret to ${f2(byAction.best.disc)}.`;
        } else if (!featureHelps) {
          sentence += scheduled ? "On this draw, no fixed choice of the features alone, or of the actions alone, does better."
            : "On this draw, changing only the features, or only the actions, does no better.";
        }
        delete byAction.best;
      }
      rankings = { features: byFeature, actions: byAction };
    }

    const found = aware ? worked.filter((count) => count > 0).length : 0;
    if (aware) {
      // the state-aware policy has its own sentence: how often a subset that works exists, and how the human did
      const right = now.real.filter((r) => r.reg <= 1e-9).length;       // rounds in which the human picked a best move
      const most = aware.budget < n ? `at most ${aware.budget} of the ${n} features` : (n > 1 ? `any of the ${n} features` : "the one feature");
      const rounds = found === T ? `every one of the ${T} rounds` : `${found} of ${T} rounds`;
      const rest = T - found === 1 ? "one" : String(T - found);
      sentence = `Showing ${most}, chosen for each round's state, `;
      if (aware.judge === "truth") {
        sentence += `the algorithm found a subset on which the true weights rank the best move first in ${rounds}`
          + (found === T ? "" : `; in the other ${rest} no subset within the budget does, so a random one was shown`)
          + `. The human picked the best move in ${right} of the ${T} rounds. `;
      } else if (found === T) {
        sentence += `the algorithm found a subset that makes the human pick the best move in ${rounds}, so nothing is lost on this draw. `;
      } else {
        sentence += `the algorithm found a subset that makes the human pick the best move in ${rounds}. In the other ${rest} no subset within the budget does, so a random one was shown; all of the regret comes from those rounds. `;
      }
      if (full) sentence += `The best fixed subset within the budget, ${describe(rankings.features.best.set, "feature")}, costs ${f2(rankings.features.best.disc)} on this draw.`;
    }
    if (rankings) delete rankings.features.best;

    const changed = (rounds, t) => !same(rounds[t], rounds[t - 1]);
    const changes = [];
    for (let t = 1; t < T; t++) if (changed(Fbench, t) || changed(Abench, t)) changes.push(t);

    const stats = [
      { label: "Averaged over states", value: f2(now.mean), note: "the same discounted regret, averaged over every state instead of the ones drawn" },
      { label: `Best move picked at round ${T - 1}`, value: pct(now.acc[T - 1]),
        note: `of states; ${pct(now.acc[0])} at round 0, heading for ${pct(now.accLimit)} once fully learned` },
      { label: "Regret per round once learned", value: f2(now.regLimit),
        note: now.regLimit > 1e-9 ? "on average, the lasting price of what is held back" : "nothing held back matters" },
      { label: "Against showing everything", value: signed2(now.disc - everything.disc),
        note: heldBack ? `on this draw; showing and offering everything costs ${f2(everything.disc)}` : "this is the show-everything policy" },
    ];
    if (aware) {
      stats[0] = { label: "Rounds with a subset that works", value: `${found} of ${T}`,
        note: (aware.judge === "truth" ? "judged by the truth" : "judged by the human's beliefs") + "; "
          + (found < T ? "a random subset is shown in the rest" : "no round needed the random fallback") };
      stats[1] = { label: "Averaged over states", value: f2(now.mean),
        note: "the discounted regret with each round's beliefs, averaged over every state instead of the one drawn" };
      stats[2] = { label: "States a subset can settle", value: pct(aware.ceiling),
        note: aware.ceiling > 1 - 1e-9 ? "by the truth, some subset within the budget ranks the best move first in every state"
          : `by the truth, with at most ${aware.budget} feature${aware.budget > 1 ? "s" : ""}; the rest need a hidden feature, whatever the human believes` };
    }

    return {
      K, n, T, seed, numeral: f2(now.disc), numeral_label: "discounted regret on this draw",
      sentence: sentence.trim(), rankings, full, stats,
      // reg, discounted and value_gap are taken at the state drawn in each round; the *_mean series are their
      // averages over all states, and acc is the share of states with the best move
      series: {
        reg: now.real.map((r) => round(r.reg, 4)), discounted: now.real.map((r, t) => round(Math.pow(delta, t) * r.reg, 4)),
        value_gap: now.real.map((r) => round(r.gap, 4)),
        reg_mean: now.reg.map((v) => round(v, 4)), value_gap_mean: now.vgap.map((v) => round(v, 4)),
        acc: now.acc.map((v) => round(v, 4)),
        ref_acc: heldBack ? everything.acc.map((v) => round(v, 4)) : null,
      },
      x: xDraw.map((row) => row.map((v) => round(v, 3))),
      // Every weight over time as [feature][action][round]. A weight on a hidden feature, or of an action that is
      // not offered, is not learned in that round and keeps its value (the page fades those flat stretches).
      beliefs: Array.from({ length: n }, (_, j) => U.map((_, k) => now.beliefs.map((H) => round(H[k][j], 3)))),
      U, F: Fbase, A: Abase, delta,
      // what is shown and offered round by round, as [feature][round] and [action][round], and the rounds at which that changes
      schedule: {
        F: Fbase.map((_, j) => Fbench.map((m) => (m[j] ? 1 : 0))), A: Abase.map((_, k) => Abench.map((m) => (m[k] ? 1 : 0))),
      },
      changes,
      // the state-aware policy: its budget, how many candidate subsets there are, and how many of them made the
      // human right in each round (0 means a random one was shown)
      policy: aware ? { budget: aware.budget, judge: aware.judge, prefer: aware.prefer, candidates: candidates.length, worked,
        ceiling: aware.ceiling } : null,
      choice: { picks: now.real.map((r) => r.pick + 1), best: now.real.map((r) => r.best + 1), missed: now.real.map((r) => r.reg > 1e-9) },
      acc_limit: now.accLimit, reg_limit: now.regLimit,
      regret_cap: regretCap, value_cap: valueCap,
    };
  }

  // ---- /api/correlated: three features drawn from N(0, Sigma) --------------------

  const CORR_N = 3, CORR_SCATTER = 300;

  // Lower-triangular L with L L^T = Sigma, the same factor numpy.linalg.cholesky returns.
  function cholesky3(S) {
    const L = [[0, 0, 0], [0, 0, 0], [0, 0, 0]];
    for (let i = 0; i < 3; i++) {
      for (let j = 0; j <= i; j++) {
        let s = S[i][j];
        for (let k = 0; k < j; k++) s -= L[i][k] * L[j][k];
        L[i][j] = i === j ? Math.sqrt(s) : s / L[j][j];
      }
    }
    return L;
  }

  // P with x_hat = P x: shown features pass through, hidden ones get E[x_hidden | x_shown].
  function imputation(Sigma, shown) {
    const P = [[0, 0, 0], [0, 0, 0], [0, 0, 0]];
    const S = [], hidden = [];
    shown.forEach((on, i) => (on ? S : hidden).push(i));
    S.forEach((i) => { P[i][i] = 1; });
    if (!S.length || !hidden.length) return P;
    const sub = (rows, cols) => rows.map((r) => cols.map((c) => Sigma[r][c]));
    const M = sub(S, S);
    let inv;
    if (S.length === 1) inv = [[1 / M[0][0]]];
    else {
      const d = M[0][0] * M[1][1] - M[0][1] * M[1][0];
      inv = [[M[1][1] / d, -M[0][1] / d], [-M[1][0] / d, M[0][0] / d]];
    }
    const cross = sub(hidden, S);                 // Sigma_hidden,shown @ inv(Sigma_shown)
    hidden.forEach((h, a) => S.forEach((s, b) => {
      let v = 0;
      for (let c = 0; c < S.length; c++) v += cross[a][c] * inv[c][b];
      P[h][s] = v;
    }));
    return P;
  }

  // Every size-k subset in turn, in itertools.combinations order.
  function rotation(k) {
    const sets = [];
    const walk = (start, chosen) => {
      if (chosen.length === k) { sets.push(chosen.slice()); return; }
      for (let i = start; i < CORR_N; i++) { chosen.push(i); walk(i + 1, chosen); chosen.pop(); }
    };
    walk(0, []);
    return sets.map((set) => { const m = [false, false, false]; set.forEach((i) => { m[i] = true; }); return m; });
  }

  // How often each feature was shown in the first t rounds of the rotation.
  function rotationCounts(cycle, t) {
    const C = cycle.length, full = Math.floor(t / C), rem = t % C, counts = [0, 0, 0];
    for (let j = 0; j < CORR_N; j++) {
      let per = 0, extra = 0;
      for (let i = 0; i < C; i++) if (cycle[i][j]) { per++; if (i < rem) extra++; }
      counts[j] = full * per + extra;
    }
    return counts;
  }

  function exploreGrid(T, cycle) {
    const step = cycle * Math.max(1, Math.ceil(T / (cycle * 20)));
    const lengths = [];
    for (let e = step; e < T; e += step) lengths.push(e);
    return lengths;
  }

  function maskCode(m) { return (m[0] ? 4 : 0) + (m[1] ? 2 : 0) + (m[2] ? 1 : 0); }
  const sameMask = (a, b) => a.every((v, i) => v === b[i]);

  function parseRho(q) {
    const raw = q.get("rho");
    const parts = raw === null ? [] : raw.split(",");
    const rho = parts.map((v) => (v.trim() === "" ? NaN : Number(v)));
    if (rho.length !== 3 || rho.some((r) => !Number.isFinite(r))) throw new BadRequest("rho must be three numbers");
    if (rho.some((r) => Math.abs(r) > 0.95)) throw new BadRequest("rho must be three correlations in [-0.95, 0.95]");
    const [a, b, c] = rho, det = 1 + 2 * a * b * c - a * a - b * b - c * c;
    if (!(1 - a * a > 1e-6 && det > 1e-6)) throw new BadRequest("these three correlations cannot occur together");
    return rho;
  }

  async function correlated(q, signal) {
    const curveName = q.has("curve") ? q.get("curve") : "exponential";
    if (!(curveName in CURVES)) throw new BadRequest("unknown curve");
    const spec = CURVES[curveName];
    const K = Math.trunc(num(q, "K", 3, 2, MAX_ACTIONS));
    const U = matrix(q, "U", K, CORR_N), H0 = matrix(q, "H", K, CORR_N);
    const rho = parseRho(q);
    const k = Math.trunc(num(q, "k", 2, 1, CORR_N - 1));
    const T = Math.trunc(num(q, "T", 120, 10, 300)), delta = num(q, "delta", 0.95, 0.01, 0.99), seed = seedOf(q);
    const explore = Math.trunc(num(q, "explore", 0, 0, T));
    const commit = mask(q, "C", CORR_N);
    if (commit.filter(Boolean).length > k) throw new BadRequest(`at most ${k} features can be shown per round`);
    const speeds = speedsOf(q, spec, CORR_N);
    const full = q.get("rank") !== "0";
    const phi = (m, j) => spec.phi(speeds[j], m);       // feature j's own curve

    const Sigma = [[1, rho[0], rho[1]], [rho[0], 1, rho[2]], [rho[1], rho[2], 1]];
    const L = cholesky3(Sigma);
    const Z = probes(CORR_N), P = Z.length / CORR_N;
    const X = new Float64Array(Z.length);
    for (let p = 0; p < P; p++) for (let j = 0; j < CORR_N; j++) {
      let s = 0;
      for (let c = 0; c < CORR_N; c++) s += Z[p * CORR_N + c] * L[j][c];   // Z @ L^T
      X[p * CORR_N + j] = s;
    }

    // one imputation matrix per on/off mask, indexed like itertools.product([False, True], repeat=3)
    const Pmats = Array.from({ length: 8 }, (_, code) => imputation(Sigma, [Boolean(code & 4), Boolean(code & 2), Boolean(code & 1)]));
    const matmul = (H, code) => H.map((row) => {
      const out = [0, 0, 0];
      for (let j = 0; j < CORR_N; j++) for (let i = 0; i < CORR_N; i++) out[j] += row[i] * Pmats[code][i][j];
      return out;
    });

    const pay = new Float64Array(P * K), bestTrue = new Float64Array(P);
    let regretCap = 0, valueCap = -Infinity;      // the fixed axes: the largest regret, and value, any probe state allows
    for (let p = 0; p < P; p++) {
      let hi = -Infinity, lo = Infinity, believed = -Infinity;
      for (let a = 0; a < K; a++) {
        let v = 0, h = 0;
        for (let j = 0; j < CORR_N; j++) { v += U[a][j] * X[p * CORR_N + j]; h += H0[a][j] * X[p * CORR_N + j]; }
        pay[p * K + a] = v;
        if (v > hi) hi = v;
        if (v < lo) lo = v;
        if (h > believed) believed = h;
      }
      bestTrue[p] = hi;
      regretCap = Math.max(regretCap, hi - lo); valueCap = Math.max(valueCap, hi, believed);
    }

    // The state drawn for each round, x_t ~ N(0, Sigma), from the same standard normals as the other benches.
    const xDraw = drawStates(seed, T, CORR_N).map((z) => [0, 1, 2].map((j) => z[0] * L[j][0] + z[1] * L[j][1] + z[2] * L[j][2]));
    // The round's own loss at its drawn state x, for effective beliefs E: regret, value gap, pick and best move.
    // Actions that tie at x share the regret; the pick is the first of them.
    function realizedE(E, x) {
      const worth = U.map((row) => row[0] * x[0] + row[1] * x[1] + row[2] * x[2]);
      let hi = -Infinity, best = -1, top = -Infinity, sumR = 0, count = 0, pick = -1;
      worth.forEach((v, a) => { if (v > hi) { hi = v; best = a; } });
      E.forEach((row, a) => {
        const v = row[0] * x[0] + row[1] * x[1] + row[2] * x[2];
        if (v > top) { top = v; sumR = hi - worth[a]; count = 1; pick = a; }
        else if (v === top) { sumR += hi - worth[a]; count++; }
      });
      return { reg: sumR / count, gap: hi - top, pick, best: sumR / count <= 1e-12 ? pick : best };
    }

    // Accuracy, regret and value gap of effective beliefs E (K x 3) on the probe states.
    // Actions the human cannot tell apart (identical effective weights) are split evenly.
    function scoreE(E) {
      const alike = E.map((row) => {
        const g = [];
        for (let b = 0; b < K; b++) if (E[b].every((v, j) => v === row[j])) g.push(b);
        return g;
      });
      let acc = 0, reg = 0, gap = 0;
      for (let p = 0; p < P; p++) {
        let pick = 0, top = -Infinity;
        const scores = new Array(K);
        for (let a = 0; a < K; a++) {
          let s = 0;
          for (let j = 0; j < CORR_N; j++) s += E[a][j] * X[p * CORR_N + j];
          scores[a] = s;
          if (s > top) { top = s; pick = a; }
        }
        const tied = alike[pick];
        let sumR = 0, sumOk = 0;
        for (const a of tied) {
          const r = bestTrue[p] - pay[p * K + a];
          sumR += r; sumOk += r <= 1e-12 ? 1 : 0;
        }
        acc += sumOk / tied.length; reg += sumR / tied.length; gap += bestTrue[p] - top;
      }
      return [acc / P, reg / P, gap / P];
    }

    const belief = (counts) => U.map((row, a) => counts.map((c, j) => row[j] + (H0[a][j] - row[j]) * (1 - phi(c, j))));
    // one round: its averages over states, and its loss at the drawn state of round t
    const at = (counts, m, t) => {
      const E = matmul(belief(counts), maskCode(m));
      return { mean: scoreE(E), real: realizedE(E, xDraw[t]) };
    };

    const cycle = rotation(k);
    const rot = [];
    for (let t = 0; t < T; t++) rot.push(at(rotationCounts(cycle, t), cycle[t % cycle.length], t));
    const rotAcc = rot.map((s) => s.mean[0]), rotReg = rot.map((s) => s.mean[1]), rotGap = rot.map((s) => s.mean[2]);

    const fixed = subsets(CORR_N).filter((m) => m.filter(Boolean).length <= k);
    let policies = [[explore, commit]].concat(fixed.map((m) => [0, m]));
    if (full) for (const e of exploreGrid(T, cycle.length)) for (const m of fixed) policies.push([e, m]);
    const seen = new Set();
    policies = policies.filter(([e, m]) => {
      const key = e + m.map((v) => (v ? 1 : 0)).join("");
      if (seen.has(key)) return false;
      seen.add(key); return true;
    });

    let lastPause = Date.now();
    const breathe = async () => {
      if (signal && signal.aborted) throw aborted();
      if (Date.now() - lastPause < 24) return;
      await nextTask();
      lastPause = Date.now();
      if (signal && signal.aborted) throw aborted();
    };

    // committed rounds: counts grow only for the features the policy keeps showing
    const acc = policies.map(() => rotAcc.slice()), reg = policies.map(() => rotReg.slice()), gap = policies.map(() => rotGap.slice());
    const real = policies.map(() => rot.map((s) => s.real));       // [policy][round]: the loss at the drawn state
    for (let p = 0; p < policies.length; p++) {
      const e = Math.min(policies[p][0], T), m = policies[p][1], base = rotationCounts(cycle, e);
      for (let t = e; t < T; t++) {
        const round = at(base.map((c, j) => c + (m[j] ? t - e : 0)), m, t);
        [acc[p][t], reg[p][t], gap[p][t]] = round.mean;
        real[p][t] = round.real;
      }
      if (p % 2 === 1) await breathe();
    }
    // the objective on this draw, and the same sum of the averages over states
    const discounted = real.map((series) => series.reduce((sum, r, t) => sum + Math.pow(delta, t) * r.reg, 0));
    const meanDiscounted = reg.map((series) => series.reduce((sum, r, t) => sum + Math.pow(delta, t) * r, 0));

    const limit = policies.map(([e, m]) => {
      const frozen = belief(rotationCounts(cycle, Math.min(e, T)));
      const H = frozen.map((row, a) => row.map((v, j) => (m[j] ? U[a][j] : v)));
      return scoreE(matmul(H, maskCode(m)));
    });

    const labelOf = (e, m) => (e === 0 ? `show ${describe(m, "feature")} every round`
      : `explore ${e} rounds, then show ${describe(m, "feature")}`);
    const rowOf = (p, rank) => ({ rank, explore: policies[p][0], mask: policies[p][1].slice(), label: labelOf(...policies[p]),
      discounted: discounted[p], mean: meanDiscounted[p], floor: limit[p][1], start: acc[p][0], end: acc[p][T - 1], current: p === 0 });

    const fixedIds = policies.reduce((ids, [e], p) => (e === 0 ? ids.concat(p) : ids), []);
    const bestFixed = fixedIds.reduce((best, p) => (discounted[p] < discounted[best] ? p : best));

    let sentence = (explore ? `Exploring for ${explore} round${explore === 1 ? "" : "s"} and then showing ${describe(commit, "feature")}`
      : `Showing ${describe(commit, "feature")} every round`)
      + `, the human picks the best move ${pct(acc[0][0])} of the time at round 0 and ${pct(acc[0][T - 1])} by round ${T - 1}. `;
    const hidden = commit.map((on) => !on);
    if (limit[0][1] > 1e-9 && hidden.some(Boolean)) {
      sentence += `${describe(hidden, "feature").replace(/^./, (c) => c.toUpperCase())} ${hidden.filter(Boolean).length === 1 ? "is" : "are"} filled in `
        + `from what is shown, using beliefs that are never corrected, so ${f2(limit[0][1])} of utility is lost every round for good. `;
    } else if (limit[0][1] <= 1e-9) {
      sentence += "Once the committed features are learned, the human always picks the best move. ";
    }

    let ranking = null;
    if (full) {
      const order = policies.map((_, p) => p).sort((a, b) => discounted[a] - discounted[b] || a - b);
      const place = {};
      order.forEach((p, i) => { place[p] = i + 1; });
      const best = order[0];
      const listed = [...new Set(order.slice(0, 10).concat([0, bestFixed]))].sort((a, b) => place[a] - place[b]);
      if (best !== 0 && discounted[best] < discounted[0] - 1e-9) sentence += `On this draw, the best policy found is to ${labelOf(...policies[best])}, at ${f2(discounted[best])}. `;
      sentence += policies[best][0] === 0 ? "On this draw, no exploration beats the best fixed subset."
        : `Never exploring costs at least ${f2(discounted[bestFixed])}, so the best fixed subset keeps only ${pct(discounted[best] / discounted[bestFixed])} of that performance.`;
      const lengths = [0].concat(exploreGrid(T, cycle.length));
      const where = {};
      policies.forEach(([e, m], p) => { where[e + m.map((v) => (v ? 1 : 0)).join("")] = p; });
      ranking = {
        rows: listed.map((p) => rowOf(p, place[p])), total: policies.length, rank: place[0],
        best: rowOf(best, 1), best_fixed: rowOf(bestFixed, place[bestFixed]),
        retained: discounted[bestFixed] > 1e-12 ? discounted[best] / discounted[bestFixed] : 1,
        sweep: { lengths, curves: fixed.map((m) => ({ mask: m.slice(), label: describe(m, "feature"),
          discounted: lengths.map((e) => discounted[where[e + m.map((v) => (v ? 1 : 0)).join("")]]) })) },
      };
    }

    const benchMask = (t) => (t < explore ? cycle[t % cycle.length] : commit);
    const benchCounts = (t) => (t < explore ? rotationCounts(cycle, t)
      : rotationCounts(cycle, explore).map((c, j) => c + (commit[j] ? t - explore : 0)));
    const roundM = (M, digits) => M.map((row) => row.map((v) => round(v, digits)));
    const E0 = matmul(belief(benchCounts(0)), maskCode(benchMask(0)));
    const ET = matmul(belief(benchCounts(T - 1)), maskCode(benchMask(T - 1)));
    const ETU = matmul(U, maskCode(commit));
    const argmax = (E, p) => {
      let pick = 0, top = -Infinity;
      for (let a = 0; a < K; a++) {
        let s = 0;
        for (let j = 0; j < CORR_N; j++) s += E[a][j] * X[p * CORR_N + j];
        if (s > top) { top = s; pick = a; }
      }
      return pick;
    };

    const ref = explore === 0 && bestFixed === 0 ? null : bestFixed;
    return {
      K, n: CORR_N, T, k, delta, rho, Sigma: Sigma.map((row) => row.map((v) => round(v, 4))),
      seed, numeral: f2(discounted[0]), numeral_label: "discounted regret on this draw",
      sentence: sentence.trim(), full, ranking,
      stats: [
        { label: "Averaged over states", value: f2(meanDiscounted[0]), note: "the same discounted regret, averaged over every state instead of the ones drawn" },
        { label: "Regret per round once learned", value: f2(limit[0][1]),
          note: limit[0][1] > 1e-9 ? "on average, what wrongly filled-in features keep costing" : "nothing is lost for good" },
        { label: "Best fixed subset", value: f2(discounted[bestFixed]),
          note: `on this draw, showing ${describe(policies[bestFixed][1], "feature")} every round` },
        { label: "Against the best fixed subset", value: signed2(discounted[0] - discounted[bestFixed]),
          note: ref === null ? "this is the best fixed subset" : "on this draw; negative is better" },
      ],
      // reg, discounted and value_gap are taken at the state drawn in each round; the *_mean series are their
      // averages over all states, and acc is the share of states with the best move
      series: {
        reg: real[0].map((r) => round(r.reg, 4)), discounted: real[0].map((r, t) => round(Math.pow(delta, t) * r.reg, 4)),
        value_gap: real[0].map((r) => round(r.gap, 4)),
        reg_mean: reg[0].map((v) => round(v, 4)), value_gap_mean: gap[0].map((v) => round(v, 4)),
        acc: acc[0].map((v) => round(v, 4)),
        ref_acc: ref === null ? null : acc[ref].map((v) => round(v, 4)),
      },
      x: xDraw.map((row) => row.map((v) => round(v, 3))),
      ref_label: ref === null ? null : labelOf(...policies[ref]),
      schedule: [0, 1, 2].map((j) => Array.from({ length: T }, (_, t) => (benchMask(t)[j] ? 1 : 0))),
      beliefs: [0, 1, 2].map((j) => U.map((row, a) => Array.from({ length: T }, (_, t) => round(row[j] + (H0[a][j] - row[j]) * (1 - phi(benchCounts(t)[j], j)), 3)))),
      U, C: commit.slice(), explore,
      choice: { picks: real[0].map((r) => r.pick + 1), best: real[0].map((r) => r.best + 1), missed: real[0].map((r) => r.reg > 1e-9) },
      effective: { first: roundM(E0, 3), last: roundM(ET, 3), truth: roundM(ETU, 3), first_mask: benchMask(0).slice() },
      scatter: {
        x: Array.from({ length: Math.min(CORR_SCATTER, P) }, (_, p) => [0, 1, 2].map((j) => round(X[p * CORR_N + j], 3))),
        best: Array.from({ length: Math.min(CORR_SCATTER, P) }, (_, p) => {
          let pick = 0;
          for (let a = 1; a < K; a++) if (pay[p * K + a] > pay[p * K + pick]) pick = a;
          return pick + 1;
        }),
        first: Array.from({ length: Math.min(CORR_SCATTER, P) }, (_, p) => argmax(E0, p) + 1),
        last: Array.from({ length: Math.min(CORR_SCATTER, P) }, (_, p) => argmax(ET, p) + 1),
      },
      acc_limit: limit[0][0], reg_limit: limit[0][1],
      regret_cap: regretCap, value_cap: valueCap,
    };
  }

  // Let the page handle input before carrying on. A message channel is used rather than a timer
  // because browsers slow timers down (to one a second in a background tab).
  function nextTask() {
    return new Promise((resolve) => {
      if (typeof MessageChannel !== "function") { setTimeout(resolve, 0); return; }
      const channel = new MessageChannel();
      channel.port1.onmessage = () => { channel.port1.close(); resolve(); };
      channel.port2.postMessage(null);
    });
  }

  // ---- stand in for the server ---------------------------------------------

  const aborted = () => (typeof DOMException === "function" ? new DOMException("Aborted", "AbortError")
    : Object.assign(new Error("Aborted"), { name: "AbortError" }));

  async function answer(url, options) {
    const signal = options && options.signal;
    if (signal && signal.aborted) throw aborted();
    const [path, query] = url.split("?");
    const q = new URLSearchParams(query || "");
    try {
      const data = path === "/api/run" ? run(q) : path === "/api/correlated" ? await correlated(q, signal) : await world(q, signal);
      return new Response(JSON.stringify(data), { status: 200, headers: { "Content-Type": "application/json" } });
    } catch (error) {
      if (error instanceof BadRequest) return new Response(JSON.stringify({ error: error.message }), { status: 400 });
      throw error;
    }
  }

  if (typeof module !== "undefined" && module.exports) {
    module.exports = { run, world, correlated, BadRequest };   // for the comparison test, under Node
  } else {
    const realFetch = root.fetch.bind(root);
    root.fetch = (url, options) => (typeof url === "string" && (url.startsWith("/api/run") || url.startsWith("/api/world") || url.startsWith("/api/correlated"))
      ? answer(url, options) : realFetch(url, options));
  }
})(typeof window !== "undefined" ? window : globalThis);
