// Two variables: a population of (X, Y) pairs with any two curves and a chosen link, and the estimates
// the page computes from it. No drawing here, so Node can test it (tests/test_distributions.py).
(function (root) {
  "use strict";
  const { byId, rng } = root.DIST;

  // Each axis gets its own curve; a hidden normal pair (a, w) sets who is high together. The X draws are
  // handed out in the order of a, the Y draws in the order of w, so each axis keeps its curve exactly
  // and the link only decides the pairing.
  function population({ xId, yId, px, py, link, rho, N = 20000, seed = 1 }) {
    const xd = byId[xId], yd = byId[yId], r = rng(seed * 7919 + 17);
    const xs = new Float64Array(N), ys = new Float64Array(N), a = new Float64Array(N), w = new Float64Array(N);
    for (let i = 0; i < N; i++) { xs[i] = xd.sample(r, px); ys[i] = yd.sample(r, py); }
    xs.sort(); ys.sort();
    const k = Math.sqrt(Math.max(0, 1 - rho * rho));
    for (let i = 0; i < N; i++) {
      a[i] = r.n();
      const e = r.n();
      w[i] = link === "u" ? rho * (a[i] * a[i] - 1) / Math.SQRT2 + k * e : rho * a[i] + k * e;
    }
    const byA = order(a), byW = order(w), X = new Float64Array(N), Y = new Float64Array(N);
    for (let j = 0; j < N; j++) { X[byA[j]] = xs[j]; Y[byW[j]] = ys[j]; }
    const P = { X, Y, N, xs, ys, xd, yd, px, py };
    P.mx = mean(X); P.my = mean(Y); P.sx = sd(X); P.sy = sd(Y);
    P.rx = robust(xs); P.ry = robust(ys);        // spreads that heavy tails cannot blow up, for axes and bands
    P.cov = cov(X, Y); P.r = P.cov / (P.sx * P.sy); P.rs = spearman(X, Y);
    P.slope = P.cov / (P.sx * P.sx); P.icpt = P.my - P.slope * P.mx;
    P.eta2 = eta2(X, Y, xd.kind === "discrete");
    return P;
  }

  function order(v) {
    const idx = new Uint32Array(v.length);
    for (let i = 0; i < v.length; i++) idx[i] = i;
    return idx.sort((i, j) => v[i] - v[j]);
  }
  const mean = (v) => { let s = 0; for (let i = 0; i < v.length; i++) s += v[i]; return s / v.length; };
  function sd(v) { const m = mean(v); let s = 0; for (let i = 0; i < v.length; i++) s += (v[i] - m) ** 2; return Math.sqrt(s / v.length); }
  function cov(x, y) { const mx = mean(x), my = mean(y); let s = 0; for (let i = 0; i < x.length; i++) s += (x[i] - mx) * (y[i] - my); return s / x.length; }
  const quantile = (sorted, q) => sorted[Math.min(sorted.length - 1, Math.max(0, Math.round(q * (sorted.length - 1))))];
  const robust = (sorted) => Math.max(1e-9, (quantile(sorted, 0.75) - quantile(sorted, 0.25)) / 1.349);

  function ranks(v) {
    const o = order(v), rk = new Float64Array(v.length);
    for (let i = 0; i < o.length;) {                // ties share their average rank
      let j = i;
      while (j + 1 < o.length && v[o[j + 1]] === v[o[i]]) j++;
      for (let t = i; t <= j; t++) rk[o[t]] = (i + j) / 2;
      i = j + 1;
    }
    return rk;
  }
  function spearman(x, y) { const rx = ranks(x), ry = ranks(y); return cov(rx, ry) / (sd(rx) * sd(ry)); }

  // share of Var(Y) that the band averages explain (the correlation ratio); bands hold equal counts of X
  function eta2(X, Y, discrete) {
    const o = order(X), N = X.length, my = mean(Y);
    let between = 0, total = 0;
    for (let i = 0; i < N; i++) total += (Y[i] - my) ** 2;
    const groups = [];
    if (discrete) { for (let i = 0; i < N;) { let j = i; while (j < N && X[o[j]] === X[o[i]]) j++; groups.push([i, j]); i = j; } }
    else { const G = 50; for (let g = 0; g < G; g++) groups.push([Math.floor(g * N / G), Math.floor((g + 1) * N / G)]); }
    for (const [i, j] of groups) {
      let s = 0;
      for (let t = i; t < j; t++) s += Y[o[t]];
      between += (j - i) * (s / (j - i) - my) ** 2;
    }
    return total > 0 ? between / total : 0;
  }

  // the average Y of the population at X near s (its spread too): the true conditional mean
  function band(P, s) {
    const h = P.xd.kind === "discrete" ? 0.5 : Math.max(0.15 * P.rx, 1e-6);
    let n = 0, s1 = 0, s2 = 0;
    for (let i = 0; i < P.N; i++) if (Math.abs(P.X[i] - s) <= h) { n++; s1 += P.Y[i]; s2 += P.Y[i] * P.Y[i]; }
    const m = n ? s1 / n : NaN;
    return { n, mean: m, sd: n > 1 ? Math.sqrt(Math.max(0, s2 / n - m * m)) : NaN, h };
  }

  // band averages and spreads along X, on a fixed grid of centres
  function profile(P, centres) { return centres.map((c) => ({ x: c, ...band(P, c) })); }

  // least squares polynomial of a degree in t = (x - centre) / scale; the population's centre and scale
  // are used for every sample, so fits are comparable
  function polyfit(P, idx, degree) {
    const p = degree + 1, A = Array.from({ length: p }, () => new Float64Array(p + 1));
    const c = quantile(P.xs, 0.5), sc = P.rx;
    for (let q = 0; q < idx.length; q++) {
      const i = idx[q], t = (P.X[i] - c) / sc, y = P.Y[i], pw = [1];
      for (let d = 1; d < p; d++) pw.push(pw[d - 1] * t);
      for (let r = 0; r < p; r++) { for (let k = 0; k < p; k++) A[r][k] += pw[r] * pw[k]; A[r][p] += pw[r] * y; }
    }
    for (let r = 0; r < p; r++) A[r][r] += 1e-9;
    for (let col = 0; col < p; col++) {            // Gaussian elimination with partial pivoting
      let piv = col;
      for (let r = col + 1; r < p; r++) if (Math.abs(A[r][col]) > Math.abs(A[piv][col])) piv = r;
      [A[col], A[piv]] = [A[piv], A[col]];
      if (Math.abs(A[col][col]) < 1e-12) return null;
      for (let r = 0; r < p; r++) {
        if (r === col) continue;
        const f = A[r][col] / A[col][col];
        for (let k = col; k <= p; k++) A[r][k] -= f * A[col][k];
      }
    }
    const beta = A.map((row, r) => row[p] / row[r]);
    return (x) => { const t = (x - c) / sc; let v = 0, pw = 1; for (const b of beta) { v += b * pw; pw *= t; } return v; };
  }
  // the models a sample can fit: polynomials by least squares, or a local average of the sample's
  // nearby points (window twice the truth band), which bends with the data but uses fewer points
  const MODELS = {
    line: { name: "Straight line", fit: (P, idx) => polyfit(P, idx, 1) },
    quad: { name: "Quadratic", fit: (P, idx) => polyfit(P, idx, 2) },
    cubic: { name: "Cubic", fit: (P, idx) => polyfit(P, idx, 3) },
    local: { name: "Local average", fit: (P, idx) => {
      const h = P.xd.kind === "discrete" ? 0.5 : 0.3 * P.rx;
      return (x) => { let n = 0, s = 0; for (const i of idx) if (Math.abs(P.X[i] - x) <= h) { n++; s += P.Y[i]; } return n ? s / n : NaN; };
    } },
  };
  const all = (P) => { const idx = new Uint32Array(P.N); for (let i = 0; i < P.N; i++) idx[i] = i; return idx; };

  // n patients drawn from the population, with replacement
  function sample(P, r, n) { const idx = new Uint32Array(n); for (let q = 0; q < n; q++) idx[q] = Math.floor(r.u() * P.N); return idx; }

  // what one sample estimates
  const STATS = {
    mean: { name: "Mean of Y", pop: (P) => P.my, est: (P, idx) => { let s = 0; for (const i of idx) s += P.Y[i]; return s / idx.length; },
      rule: (P, n) => P.sy / Math.sqrt(n), ruleText: "σ_Y / √n" },
    sd: { name: "SD of Y", pop: (P) => P.sy, est: (P, idx) => { let s = 0, s2 = 0; for (const i of idx) { s += P.Y[i]; s2 += P.Y[i] ** 2; } const m = s / idx.length; return Math.sqrt(Math.max(0, s2 / idx.length - m * m)); },
      rule: (P, n) => P.sy / Math.sqrt(2 * n), ruleText: "σ_Y / √(2n), exact for normal Y" },
    r: { name: "Correlation r", pop: (P) => P.r, est: (P, idx) => sampleR(P, idx),
      rule: (P, n) => (1 - P.r * P.r) / Math.sqrt(n), ruleText: "(1 − r²) / √n, for normal pairs" },
    slope: { name: "Slope of the line", pop: (P) => P.slope, est: (P, idx) => sampleSlope(P, idx),
      rule: (P, n) => P.sy * Math.sqrt(Math.max(0, 1 - P.r * P.r)) / (P.sx * Math.sqrt(n)), ruleText: "σ_residual / (σ_X √n)" },
  };
  function moments(P, idx) {
    let sx = 0, sy = 0, sxx = 0, syy = 0, sxy = 0;
    for (const i of idx) { const x = P.X[i], y = P.Y[i]; sx += x; sy += y; sxx += x * x; syy += y * y; sxy += x * y; }
    const n = idx.length, mx = sx / n, my = sy / n;
    return { vx: sxx / n - mx * mx, vy: syy / n - my * my, c: sxy / n - mx * my, mx, my };
  }
  function sampleR(P, idx) { const m = moments(P, idx); return m.vx > 0 && m.vy > 0 ? m.c / Math.sqrt(m.vx * m.vy) : NaN; }
  function sampleSlope(P, idx) { const m = moments(P, idx); return m.vx > 0 ? m.c / m.vx : NaN; }

  function summary(values) {
    const v = values.filter(Number.isFinite);
    const m = v.reduce((a, b) => a + b, 0) / v.length;
    return { mean: m, sd: Math.sqrt(v.reduce((a, b) => a + (b - m) ** 2, 0) / v.length), n: v.length };
  }

  // repeated samples of size n: how far one model's prediction at s lands from the truth there
  function predictionError(P, model, s, n, reps, seed) {
    const r = rng(seed), preds = [], fit = MODELS[model].fit;
    for (let k = 0; k < reps; k++) { const f = fit(P, sample(P, r, n)); if (f) { const v = f(s); if (Number.isFinite(v)) preds.push(v); } }
    const truth = band(P, s).mean, best = fit(P, all(P))(s), sm = summary(preds);
    return { truth, best, avg: sm.mean, spec: best - truth, sampling: sm.sd, rmse: Math.sqrt((sm.mean - truth) ** 2 + sm.sd ** 2), preds, missing: reps - preds.length };
  }

  root.PAIRS = { population, band, profile, polyfit, MODELS, all, sample, STATS, summary, predictionError, quantile, spearman, eta2 };
})(typeof window !== "undefined" ? window : globalThis);
