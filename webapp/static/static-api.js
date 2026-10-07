// The static build of the site (GitHub Pages) has no Python server. This file answers the two
// data endpoints, /api/run and /api/world, in the browser, with the same JSON that webapp/app.py
// returns, so the pages, stylesheet and scripts are exactly the ones the Flask site uses.
// Keep it in step with app.py; tests/test_static_api.py compares the two.
(function (root) {
  "use strict";

  const MEAN_ABS_X = Math.sqrt(2 / Math.PI);          // E|x| for x ~ N(0, 1)
  const MAX_ACTIONS = 6, MAX_FEATURES = 6;

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

  // ---- /api/run: one feature, two actions ----------------------------------

  function run(q) {
    const curveName = q.has("curve") ? q.get("curve") : "exponential";
    if (!(curveName in CURVES)) throw new BadRequest("unknown curve");
    const spec = CURVES[curveName];
    let split = q.get("split") === "1";
    const p1 = num(q, "p1", spec.def, spec.min, spec.max);
    const p2 = split ? num(q, "p2", spec.def, spec.min, spec.max) : p1;
    split = split && p1 !== p2;           // equal speeds are one shared curve
    const u1 = num(q, "u1", 1.2, -3, 3), u2 = num(q, "u2", 1.0, -3, 3);
    const h1 = num(q, "h1", 0.0, -3, 3), h2 = num(q, "h2", 1.5, -3, 3);
    const T = Math.trunc(num(q, "T", 80, 10, 400)), delta = num(q, "delta", 0.95, 0.01, 0.99);
    const du = u1 - u2, dh0 = h1 - h2;

    // everything is shown every round, so the beliefs follow the learning curve in closed form
    const left1 = [], left2 = [], b1 = [], b2 = [], gap = [], wrong = [];
    for (let t = 0; t < T; t++) {
      left1.push(1 - spec.phi(p1, t)); left2.push(1 - spec.phi(p2, t));
      b1.push(u1 + (h1 - u1) * left1[t]); b2.push(u2 + (h2 - u2) * left2[t]);
      gap.push(b1[t] - b2[t]);
      wrong.push(du !== 0 && !(gap[t] * du > 0));
    }
    const settle = settleTime(wrong);
    let switches = 0;
    for (let t = 1; t < T; t++) if (wrong[t] !== wrong[t - 1]) switches++;
    const nWrong = wrong.filter(Boolean).length;

    // regret is |du| |x| on a wrong round; over x ~ N(0, 1) that averages to |du| E|x|
    const wrongCost = Math.abs(du) * MEAN_ABS_X;
    const regretPerRound = wrong.map((w) => (w ? wrongCost : 0));
    const discounted = regretPerRound.map((r, t) => Math.pow(delta, t) * r);
    const regret = discounted.reduce((a, b) => a + b, 0);
    const valueGap = gap.map((g) => ((Math.abs(du) - Math.abs(g)) * MEAN_ABS_X) / 2);
    const best = du === 0 ? null : (du > 0 ? 0 : 1);
    const choice = wrong.map((w, t) => (best === null ? (b1[t] >= b2[t] ? 0 : 1) : (w ? 1 - best : best)));

    const gaps = `belief gap ${signed2(dh0)} against a true gap of ${signed2(du)}`;
    let numeral, label, sentence;
    if (du === 0) {
      [numeral, label, sentence] = ["0", "no flip needed", "Both actions are worth exactly the same, so whatever the human picks is right."];
    } else if (nWrong === 0) {
      [numeral, label, sentence] = ["0", "no flip needed", `The human starts with the right sign (${gaps}) and never picks the wrong action.`];
    } else if (!Number.isFinite(settle)) {
      [numeral, label, sentence] = ["never", `within ${T} rounds`,
        `With a ${gaps}, the human is still picking the wrong action at the end of the horizon. Lengthen the horizon or speed up the learner to see the flip.`];
    } else if (switches <= 1) {
      [numeral, label, sentence] = [String(settle), "round of the flip",
        `The human starts with the wrong sign (${gaps}), picks the wrong action for ${nWrong} round${nWrong !== 1 ? "s" : ""}, and is right from round ${settle} on.`];
    } else {
      [numeral, label, sentence] = [String(settle), "round of the final flip",
        `The choice changes ${switches} times. The human ${wrong[0] ? "" : "starts right, "}goes wrong at round ${wrong.indexOf(true)}, is wrong on ${nWrong} rounds in total, and is right for good from round ${settle}.`];
    }

    let needed;
    if (split) needed = { value: "varies", note: "each action has its own curve, so no single threshold" };
    else if (du === 0) needed = { value: "—", note: "the actions are equally good" };
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
          const g = (uMid + gu / 2 + e1 * left1[t]) - (uMid - gu / 2 + e2 * left2[t]);
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
        { label: "Choice changes", value: String(switches), note: switches > 1 ? "more than one means the choice un-flipped" : "right/wrong switches" },
        { label: "Expected discounted regret", value: f2(regret),
          note: wrong[T - 1] ? `lower bound: still wrong at round ${T}` : `δ = ${delta}, states x ~ N(0, 1)` },
      ],
      series: {
        h1: b1.map((v) => round(v, 4)), h2: b2.map((v) => round(v, 4)),
        regret: regretPerRound.map((v) => round(v, 4)), choice: choice.map((c) => c + 1),
        discounted: discounted.map((v) => round(v, 4)), value_gap: valueGap.map((v) => round(v, 4)),
      },
      wrong_cost: wrongCost, best: best === null ? null : best + 1,
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
  // every non-empty on/off mask, in the order of Python's itertools.product([False, True], repeat=size)
  const subsets = (size) => Array.from({ length: (1 << size) - 1 }, (_, m) =>
    Array.from({ length: size }, (_, j) => Boolean(((m + 1) >> (size - 1 - j)) & 1)));

  async function world(q, signal) {
    const curveName = q.has("curve") ? q.get("curve") : "exponential";
    if (!(curveName in CURVES)) throw new BadRequest("unknown curve");
    const spec = CURVES[curveName];
    const K = Math.trunc(num(q, "K", 3, 2, MAX_ACTIONS)), n = Math.trunc(num(q, "n", 3, 1, MAX_FEATURES));
    const U = matrix(q, "U", K, n), H0 = matrix(q, "H", K, n);
    const Fnow = mask(q, "F", n), Anow = mask(q, "A", K);
    const speed = num(q, "p1", spec.def, spec.min, spec.max);
    const T = Math.trunc(num(q, "T", 100, 10, 300)), delta = num(q, "delta", 0.95, 0.01, 0.99);
    const full = q.get("rank") !== "0";

    const X = probes(n), P = X.length / n;
    const phis = Array.from({ length: T }, (_, t) => spec.phi(speed, t));

    // what every action is truly worth in every probe state, and the regret of picking it
    const regretOf = new Float64Array(P * K), bestTrue = new Float64Array(P);
    let worstRegret = 0, vTrue = 0, vBelief = 0;
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
      worstRegret += (hi - lo) / P; vTrue += hi / P; vBelief += believed / P;
    }

    // One fixed policy (features shown, actions offered). Shown weights move from H0 to U along the
    // curve, so the human's score in a probe state is a blend of two fixed scores.
    function score(F, A) {
      const ks = [];
      for (let k = 0; k < K; k++) if (A[k]) ks.push(k);
      const e0 = new Float64Array(P * K), e1 = new Float64Array(P * K);
      for (let p = 0; p < P; p++) for (const k of ks) {
        let s0 = 0, s1 = 0;
        for (let j = 0; j < n; j++) if (F[j]) { const x = X[p * n + j]; s0 += H0[k][j] * x; s1 += U[k][j] * x; }
        e0[p * K + k] = s0; e1[p * K + k] = s1;
      }
      const at = (f) => {
        let acc = 0, reg = 0, gap = 0;
        for (let p = 0; p < P; p++) {
          let top = -Infinity, sumR = 0, sumOk = 0, count = 0;
          for (const k of ks) {
            const i = p * K + k, v = e0[i] + f * (e1[i] - e0[i]), r = regretOf[i];
            if (v > top) { top = v; sumR = r; sumOk = r <= 1e-12 ? 1 : 0; count = 1; }
            else if (v === top) { sumR += r; sumOk += r <= 1e-12 ? 1 : 0; count++; }   // ties are split evenly
          }
          acc += sumOk / count; reg += sumR / count; gap += bestTrue[p] - top;
        }
        return [acc / P, reg / P, gap / P];
      };
      const acc = [], reg = [], vgap = [];
      let disc = 0;
      for (let t = 0; t < T; t++) {
        const [a, r, g] = at(phis[t]);
        acc.push(a); reg.push(r); vgap.push(g); disc += Math.pow(delta, t) * r;
      }
      const [accLimit, regLimit] = at(1);
      return { acc, reg, vgap, disc, accLimit, regLimit };
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

    const allF = Array(n).fill(true), allA = Array(K).fill(true);
    const now = score(Fnow, Anow);
    const heldBack = !(Fnow.every(Boolean) && Anow.every(Boolean));
    const everything = heldBack ? score(allF, allA) : now;

    const hidden = [], offF = Fnow.map((v) => !v), offA = Anow.map((v) => !v);
    if (offF.some(Boolean)) hidden.push(describe(offF, "feature") + (offF.filter(Boolean).length === 1 ? " stays" : " stay") + " hidden");
    if (offA.some(Boolean)) hidden.push(describe(offA, "action") + (offA.filter(Boolean).length === 1 ? " is" : " are") + " never offered");
    let sentence = `Showing ${describe(Fnow, "feature")} and offering ${describe(Anow, "action")}, the human picks the best move ${pct(now.acc[0])} of the time at round 0 and ${pct(now.acc[T - 1])} by round ${T - 1}. `;
    sentence += now.regLimit > 1e-9
      ? `Even after learning everything shown, ${f2(now.regLimit)} of utility is lost per round` + (hidden.length ? ` because ${humanList(hidden)}. ` : ". ")
      : "Once everything shown is learned, the human always picks the best move. ";

    let rankings = null;
    if (full) {
      const same = (a, b) => a.every((v, i) => v === b[i]);
      const rank = async (sets, chosen, noun, policy) => {
        const results = [];
        for (const set of sets) { results.push(same(set, chosen) ? now : score(...policy(set))); await breathe(); }
        const order = sets.map((_, i) => i).sort((a, b) => results[a].disc - results[b].disc || a - b);
        const place = {};
        order.forEach((i, r) => { place[i] = r + 1; });
        const current = sets.findIndex((set) => same(set, chosen));
        const listed = [...new Set(order.slice(0, 8).concat([current]))].sort((a, b) => place[a] - place[b]);
        return {
          rows: listed.map((i) => ({ rank: place[i], mask: sets[i], label: describe(sets[i], noun), discounted: results[i].disc,
            start: results[i].acc[0], end: results[i].acc[T - 1], floor: results[i].regLimit, current: i === current })),
          total: sets.length, rank: place[current], best: { set: sets[order[0]], disc: results[order[0]].disc },
        };
      };
      const byFeature = await rank(subsets(n), Fnow, "feature", (set) => [set, Anow]);
      const byAction = await rank(subsets(K), Anow, "action", (set) => [Fnow, set]);
      const featureHelps = byFeature.best.disc < now.disc - 1e-9;
      if (featureHelps) sentence += `Showing ${describe(byFeature.best.set, "feature")} instead would cut the discounted regret to ${f2(byFeature.best.disc)}. `;
      if (byAction.best.disc < now.disc - 1e-9) {
        const offer = `ffering ${describe(byAction.best.set, "action")} instead would cut`;
        sentence += featureHelps ? `Separately, o${offer} it to ${f2(byAction.best.disc)}.` : `O${offer} the discounted regret to ${f2(byAction.best.disc)}.`;
      } else if (!featureHelps) {
        sentence += "Changing only the features, or only the actions, does no better over this horizon.";
      }
      delete byFeature.best; delete byAction.best;
      rankings = { features: byFeature, actions: byAction };
    }

    // the choice for the state with every feature at +1: shown beliefs summed per offered action
    const worth = U.map((row) => row.reduce((a, b) => a + b, 0));
    const top = Math.max(...worth), bestMove = worth.indexOf(top);
    const picks = phis.map((f) => {
      let pick = -1, bestScore = -Infinity;
      for (let k = 0; k < K; k++) {
        if (!Anow[k]) continue;
        let v = 0;
        for (let j = 0; j < n; j++) if (Fnow[j]) v += U[k][j] + (H0[k][j] - U[k][j]) * (1 - f);
        if (v > bestScore) { bestScore = v; pick = k; }     // ties go to the lowest-numbered action
      }
      return pick;
    });

    return {
      K, n, T, numeral: f2(now.disc), numeral_label: "expected discounted regret",
      sentence: sentence.trim(), rankings, full,
      stats: [
        { label: "Best move picked at round 0", value: pct(now.acc[0]), note: "with the human's first beliefs" },
        { label: `Best move picked at round ${T - 1}`, value: pct(now.acc[T - 1]), note: `heading for ${pct(now.accLimit)} once fully learned` },
        { label: "Regret per round once learned", value: f2(now.regLimit),
          note: now.regLimit > 1e-9 ? "the lasting price of what is held back" : "nothing held back matters" },
        { label: "Against showing everything", value: signed2(now.disc - everything.disc),
          note: heldBack ? `discounted regret; showing and offering everything costs ${f2(everything.disc)}` : "this is the show-everything policy" },
      ],
      series: {
        acc: now.acc.map((v) => round(v, 4)), reg: now.reg.map((v) => round(v, 4)),
        discounted: now.reg.map((r, t) => round(Math.pow(delta, t) * r, 4)),
        value_gap: now.vgap.map((v) => round(v, 4)),
        ref_acc: heldBack ? everything.acc.map((v) => round(v, 4)) : null,
      },
      // every weight's learning curve as [feature][action][round]; the page fades the unused ones
      beliefs: Array.from({ length: n }, (_, j) => U.map((row, k) => phis.map((f) => round(row[j] + (H0[k][j] - row[j]) * (1 - f), 3)))),
      U, F: Fnow, A: Anow, delta,
      choice: { x: Array(n).fill(1), picks: picks.map((k) => k + 1), best: bestMove + 1, missed: picks.map((k) => worth[k] < top - 1e-9) },
      acc_limit: now.accLimit, reg_limit: now.regLimit,
      worst_regret: worstRegret, value_bound: Math.max(vTrue, vBelief),
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
      const data = path === "/api/run" ? run(q) : await world(q, signal);
      return new Response(JSON.stringify(data), { status: 200, headers: { "Content-Type": "application/json" } });
    } catch (error) {
      if (error instanceof BadRequest) return new Response(JSON.stringify({ error: error.message }), { status: 400 });
      throw error;
    }
  }

  if (typeof module !== "undefined" && module.exports) {
    module.exports = { run, world, BadRequest };            // for the comparison test, under Node
  } else {
    const realFetch = root.fetch.bind(root);
    root.fetch = (url, options) => (typeof url === "string" && (url.startsWith("/api/run") || url.startsWith("/api/world"))
      ? answer(url, options) : realFetch(url, options));
  }
})(typeof window !== "undefined" ? window : globalThis);
