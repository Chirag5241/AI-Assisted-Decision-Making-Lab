// Two variables: one population, four questions about it (correlation, variance, sampling error, specification error).
(function () {
  "use strict";
  const { list, byId, defaults, rng } = DIST;
  const Q = PAIRS;
  const $ = (id) => document.getElementById(id);
  const C = { ink: "#0b0b0b", ink2: "#52514e", ink3: "#6b6a65", rule: "#e4e3de", rule2: "#c9c8c1", dot: "#b4b2a9",
    band: "#2a78d6", line: "#eb6834", truth: "#6250d6", samp: "#1baf7a", pos: "#1baf7a", neg: "#e34948", disc: "#2a78d6", cont: "#eb6834" };
  const NS = [5, 10, 25, 50, 100, 250, 500, 1000];
  const W = 760;

  const st = {
    x: "normal", y: "normal", px: {}, py: {}, link: "straight", rho: 0.6, s: 1, seed: 1,
    stat: "mean", sampN: 25, est: [], last: null, sampSeed: 1,
    model: "line", specN: 50,
  };
  let P = null;

  // ---------- helpers ----------
  function fmt(v) {
    if (Number.isNaN(v)) return "—";
    if (!Number.isFinite(v)) return "∞";
    const a = Math.abs(v);
    if (Number.isInteger(v) || a >= 1000) return Math.round(v).toLocaleString();
    if (a >= 100) return v.toFixed(1);
    if (a >= 1) return String(+v.toFixed(3));
    return String(+v.toFixed(3));
  }
  function ticks(lo, hi, count) {
    const raw = (hi - lo) / count, mag = 10 ** Math.floor(Math.log10(raw)), e = raw / mag;
    const step = (e >= 7.5 ? 10 : e >= 3.5 ? 5 : e >= 1.5 ? 2 : 1) * mag, out = [];
    for (let v = Math.ceil(lo / step - 1e-9) * step; v <= hi + 1e-9; v += step) out.push(+v.toFixed(10));
    return out;
  }
  const esc = (s) => String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;");
  const xRange = (d) => (d.kind === "discrete" ? [d.x[0] - 0.5, d.x[1] + 0.5] : d.x.slice());
  const axisTicks = (d, r) => (d.kind === "discrete" ? ticks(d.x[0], d.x[1], Math.min(8, d.x[1] - d.x[0])).filter(Number.isInteger) : ticks(r[0], r[1], 8));
  const tile = (k, v, s) => `<div class="stat"><div class="k">${k}</div><div class="v">${v}</div><div class="s">${s}</div></div>`;
  const sw = (color, kind) => kind === "line" ? `<i style="background:none;border-top:2px solid ${color};height:0;width:16px"></i>`
    : kind === "dash" ? `<i style="background:none;border-top:2px dashed ${color};height:0;width:16px"></i>`
    : `<i style="background:${color}"></i>`;
  const legend = (id, items) => { $(id).innerHTML = items.map(([c, k, t]) => `<span>${sw(c, k)}${t}</span>`).join(""); };

  function canvas(id, H) {
    const c = $(id), dpr = window.devicePixelRatio || 1;
    if (c.width !== W * dpr || c.height !== H * dpr) { c.width = W * dpr; c.height = H * dpr; c.style.aspectRatio = `${W} / ${H}`; }
    const ctx = c.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, W, H);
    ctx.font = "11px 'IBM Plex Mono', ui-monospace, monospace";
    ctx.lineJoin = ctx.lineCap = "round";
    return ctx;
  }
  // a plotting frame with fixed ranges; returns the maps from data to pixels
  function frame(ctx, R, xr, yr, xt, yt, opts = {}) {
    const X = opts.logx ? (v) => R.l + (Math.log(v) - Math.log(xr[0])) / (Math.log(xr[1]) - Math.log(xr[0])) * (R.r - R.l)
      : (v) => R.l + (v - xr[0]) / (xr[1] - xr[0]) * (R.r - R.l);
    const Y = (v) => R.b - (v - yr[0]) / (yr[1] - yr[0]) * (R.b - R.t);
    ctx.strokeStyle = C.rule; ctx.lineWidth = 1; ctx.fillStyle = C.ink3;
    ctx.textAlign = "right";
    for (const t of yt) { ctx.beginPath(); ctx.moveTo(R.l, Y(t)); ctx.lineTo(R.r, Y(t)); ctx.stroke(); ctx.fillText(fmt(t), R.l - 6, Y(t) + 4); }
    ctx.textAlign = "center";
    for (const t of xt) { if (opts.xgrid) { ctx.beginPath(); ctx.moveTo(X(t), R.t); ctx.lineTo(X(t), R.b); ctx.stroke(); } ctx.fillText(fmt(t), X(t), R.b + 15); }
    ctx.strokeStyle = C.rule2; ctx.strokeRect(R.l + 0.5, R.t + 0.5, R.r - R.l - 1, R.b - R.t - 1);
    if (opts.xlab) { ctx.fillStyle = C.ink2; ctx.fillText(opts.xlab, (R.l + R.r) / 2, R.b + 31); }
    if (opts.ylab) { ctx.save(); ctx.translate(opts.ylabX ?? R.l - 40, (R.t + R.b) / 2); ctx.rotate(-Math.PI / 2); ctx.fillStyle = C.ink2; ctx.fillText(opts.ylab, 0, 0); ctx.restore(); }
    return { X, Y };
  }
  const clip = (ctx, R) => { ctx.save(); ctx.beginPath(); ctx.rect(R.l, R.t, R.r - R.l, R.b - R.t); ctx.clip(); };
  function path(ctx, pts, color, width, dash) {
    ctx.strokeStyle = color; ctx.lineWidth = width; ctx.setLineDash(dash || []);
    ctx.beginPath();
    let pen = false;
    for (const [x, y] of pts) { if (!Number.isFinite(y)) { pen = false; continue; } if (pen) ctx.lineTo(x, y); else { ctx.moveTo(x, y); pen = true; } }
    ctx.stroke(); ctx.setLineDash([]);
  }
  function dot(ctx, x, y, r, fill) { ctx.fillStyle = fill; ctx.strokeStyle = "#fcfcfb"; ctx.lineWidth = 2; ctx.beginPath(); ctx.arc(x, y, r, 0, 2 * Math.PI); ctx.fill(); ctx.stroke(); }
  function points(ctx, X, Y, idx, color, size, alpha) {
    ctx.fillStyle = color; ctx.globalAlpha = alpha;
    for (const i of idx) ctx.fillRect(X(P.X[i]) - size / 2, Y(P.Y[i]) - size / 2, size, size);
    ctx.globalAlpha = 1;
  }
  const everyone = () => Q.all(P);
  // x positions to draw a curve along X: whole numbers for a count, a fine grid otherwise
  function grid(d, n = 120) {
    if (d.kind === "discrete") { const g = []; for (let k = d.x[0]; k <= d.x[1]; k++) g.push(k); return g; }
    const r = d.x; return Array.from({ length: n + 1 }, (_, i) => r[0] + (r[1] - r[0]) * i / n);
  }
  // band averages along X depend on the population only, not on s, so each grid is worked out once per population
  const profiles = new WeakMap();
  function profileOf(d, n) {
    if (!profiles.has(P)) profiles.set(P, new Map());
    const m = profiles.get(P), k = d.id + ":" + n;
    if (!m.has(k)) m.set(k, Q.profile(P, grid(d, n)));
    return m.get(k);
  }
  const curveOf = (d) => profileOf(d, 80).map((b) => [b.x, b.n >= 20 ? b.mean : NaN]);

  // ---------- population controls ----------
  function fillSelect(id, value) {
    const sel = $(id);
    sel.innerHTML = ["discrete", "continuous"].map((k) =>
      `<optgroup label="${k === "discrete" ? "Discrete" : "Continuous"}">${list.filter((d) => d.kind === k).map((d) => `<option value="${d.id}">${esc(d.name)}</option>`).join("")}</optgroup>`).join("");
    sel.value = value;
  }
  function paramControls(axis) {
    const d = byId[st[axis]], p = st["p" + axis], box = $(axis + "-params");
    box.innerHTML = d.params.length ? d.params.map(([k, label, min, max, step]) =>
      `<div class="ctl"><label for="${axis}-${k}">${esc(label)}</label><input type="range" id="${axis}-${k}" data-k="${k}" min="${min}" max="${max}" step="${step}" value="${p[k]}"><output></output></div>`).join("")
      : `<p class="note">No parameters.</p>`;
    box.querySelectorAll("input").forEach((inp) => {
      inp.nextElementSibling.textContent = fmt(p[inp.dataset.k]);
      inp.addEventListener("input", () => {
        p[inp.dataset.k] = +inp.value; d.fix && d.fix(p);
        box.querySelectorAll("input").forEach((o) => { o.value = p[o.dataset.k]; o.nextElementSibling.textContent = fmt(p[o.dataset.k]); });
        rebuild();
      });
    });
  }
  function setDist(axis, id) {
    const d = byId[id];
    st[axis] = id; st["p" + axis] = defaults(d); d.fix && d.fix(st["p" + axis]);
    paramControls(axis);
  }
  function sControl(reset) {
    const d = byId[st.x], s = $("s"), r = d.x, step = d.kind === "discrete" ? 1 : +((r[1] - r[0]) / 200).toPrecision(2);
    s.min = r[0]; s.max = r[1]; s.step = step;
    if (reset && P) st.s = Math.max(r[0], Math.min(r[1], Math.round(Q.quantile(P.xs, 0.95) / step) * step));
    st.s = +(+st.s).toFixed(6);
    s.value = st.s; $("s-out").textContent = fmt(st.s);
  }

  // ---------- 1. correlation ----------
  const SR = { l: 130, r: 750, t: 10, b: 390 }, XS = { t: 430, b: 500 }, YS = { l: 8, r: 78 };
  function drawScatter() {
    const H = 512, ctx = canvas("scatter", H), xd = byId[st.x], yd = byId[st.y];
    const xr = xRange(xd), yr = xRange(yd);
    const { X, Y } = frame(ctx, SR, xr, yr, axisTicks(xd, xr), axisTicks(yd, yr), { xlab: "X: " + xd.name, ylab: "Y: " + yd.name, ylabX: 92 });
    const b = Q.band(P, st.s), inBand = [], rest = [];
    for (let i = 0; i < P.N; i++) (Math.abs(P.X[i] - st.s) <= b.h ? inBand : rest).push(i);
    clip(ctx, SR);
    ctx.fillStyle = C.band; ctx.globalAlpha = 0.08;
    ctx.fillRect(X(st.s - b.h), SR.t, X(st.s + b.h) - X(st.s - b.h), SR.b - SR.t); ctx.globalAlpha = 1;
    if ($("show-cov").checked) {
      const pos = [], neg = [];
      for (const i of rest) ((P.X[i] - P.mx) * (P.Y[i] - P.my) >= 0 ? pos : neg).push(i);
      points(ctx, X, Y, pos, C.pos, 1.7, 0.3); points(ctx, X, Y, neg, C.neg, 1.7, 0.3);
      path(ctx, [[X(P.mx), SR.t], [X(P.mx), SR.b]], C.ink2, 1, [3, 3]);
      path(ctx, [[SR.l, Y(P.my)], [SR.r, Y(P.my)]], C.ink2, 1, [3, 3]);
    } else points(ctx, X, Y, rest, C.dot, 1.7, 0.35);
    points(ctx, X, Y, inBand, C.band, 2.2, 0.85);
    if ($("show-true").checked) path(ctx, curveOf(xd).map(([x, y]) => [X(x), Y(y)]), C.truth, 2.6);
    if ($("show-line").checked) path(ctx, [xr[0], xr[1]].map((x) => [X(x), Y(P.icpt + P.slope * x)]), C.line, 2, [7, 5]);
    ctx.restore();

    // each variable's own curve beside its axis, on that distribution's fixed density scale
    const pdfX = (v) => Math.min(1.04, xd.pdf(v, st.px) / xd.y), pdfY = (v) => Math.min(1.04, yd.pdf(v, st.py) / yd.y);
    const hX = XS.b - XS.t, wY = YS.r - YS.l;
    ctx.strokeStyle = C.rule2; ctx.beginPath(); ctx.moveTo(SR.l, XS.b + 0.5); ctx.lineTo(SR.r, XS.b + 0.5); ctx.moveTo(YS.l + 0.5, SR.t); ctx.lineTo(YS.l + 0.5, SR.b); ctx.stroke();
    if (xd.kind === "discrete") {
      ctx.fillStyle = C.disc; const bw = Math.max(1.5, (X(1) - X(0)) * 0.55);
      for (const k of grid(xd)) { const v = pdfX(k); if (v > 0.002) ctx.fillRect(X(k) - bw / 2, XS.b - v * hX, bw, v * hX); }
    } else {
      const g = grid(xd, 200).map((x) => [X(x), XS.b - (Number.isFinite(pdfX(x)) ? pdfX(x) : 1.04) * hX]);
      ctx.fillStyle = C.cont; ctx.globalAlpha = 0.2; ctx.beginPath(); ctx.moveTo(X(xr[0]), XS.b); g.forEach(([a, c]) => ctx.lineTo(a, c)); ctx.lineTo(X(xr[1]), XS.b); ctx.fill(); ctx.globalAlpha = 1;
      path(ctx, g, C.cont, 1.8);
    }
    const yAt = (v) => YS.l + (Number.isFinite(pdfY(v)) ? pdfY(v) : 1.04) * wY;
    if (yd.kind === "discrete") {
      ctx.fillStyle = C.disc; const bh = Math.max(1.5, (Y(0) - Y(1)) * 0.55);
      for (const k of grid(yd)) { const v = pdfY(k); if (v > 0.002) ctx.fillRect(YS.l, Y(k) - bh / 2, v * wY, bh); }
    } else {
      const g = grid(yd, 200).map((y) => [yAt(y), Y(y)]);
      ctx.fillStyle = C.cont; ctx.globalAlpha = 0.2; ctx.beginPath(); ctx.moveTo(YS.l, Y(yr[0])); g.forEach(([a, c]) => ctx.lineTo(a, c)); ctx.lineTo(YS.l, Y(yr[1])); ctx.fill(); ctx.globalAlpha = 1;
      path(ctx, g, C.cont, 1.8);
    }
    // the band's own Y, on the same density scale as Y's curve
    if ($("show-cond").checked && inBand.length > 5) {
      const B = yd.kind === "discrete" ? null : 60, ys = inBand.map((i) => P.Y[i]);
      ctx.save(); ctx.beginPath(); ctx.rect(YS.l, SR.t, wY + 6, SR.b - SR.t); ctx.clip();
      if (B) {
        const w = (yr[1] - yr[0]) / B, h = new Float64Array(B);
        for (const v of ys) { const j = Math.floor((v - yr[0]) / w); if (j >= 0 && j < B) h[j]++; }
        const pts = []; for (let j = 0; j < B; j++) { const dens = h[j] / (ys.length * w) / yd.y; pts.push([YS.l + Math.min(1.04, dens) * wY, Y(yr[0] + (j + 0.5) * w)]); }
        path(ctx, pts, C.band, 2);
      } else {
        const cnt = new Map(); for (const v of ys) cnt.set(v, (cnt.get(v) || 0) + 1);
        ctx.fillStyle = C.band; for (const [k, c] of cnt) { const v = Math.min(1.04, c / ys.length / yd.y); ctx.fillRect(YS.l, Y(k) - 1.5, v * wY, 3); }
      }
      ctx.restore();
    }
    // connectors from the band's average to both curves
    if (Number.isFinite(b.mean)) {
      const ax = X(st.s), ay = Y(b.mean), hx = XS.b - Math.min(1, pdfX(st.s) || 0) * hX, ly = yAt(b.mean);
      path(ctx, [[ax, ay], [ax, hx]], C.line, 1.4, [3, 3]);
      path(ctx, [[ax, ay], [Math.min(ly, YS.r), ay]], C.line, 1.4, [3, 3]);
      dot(ctx, ax, hx, 4.5, C.band); dot(ctx, Math.min(ly, YS.r), ay, 4.5, C.line); dot(ctx, ax, ay, 6.5, C.line);
    }

    legend("scatter-legend", [[C.band, "box", "X ≈ s"], [C.line, "box", "their average Y"], [C.truth, "line", "true band averages"], [C.line, "dash", "straight line"], [C.cont, "box", "each variable's own curve, beside its axis"]]
      .concat($("show-cov").checked ? [[C.pos, "box", "(X − X̄)(Y − Ȳ) > 0"], [C.neg, "box", "< 0"]] : []));
    const lineAt = P.icpt + P.slope * st.s;
    $("corr-tiles").innerHTML =
      tile("Correlation r", fmt(P.r), "straight-line fit, −1 to 1") +
      tile("Rank correlation", fmt(P.rs), "any steady rise or fall") +
      tile("Covariance", fmt(P.cov), `average (X − X̄)(Y − Ȳ) = r·σ_X·σ_Y`) +
      tile(`Average Y at X ≈ ${fmt(st.s)}`, fmt(b.mean), `the line says ${fmt(lineAt)} · ${b.n} patients`);
  }

  // ---------- 2. variance ----------
  function drawVariance() {
    const xd = byId[st.x], yd = byId[st.y], r2 = P.r * P.r, e2 = Math.max(P.eta2, r2), parts = [
      [r2, C.line, "the straight line explains"], [e2 - r2, C.truth, "a curve explains on top"], [1 - e2, C.dot, "left over at a fixed X"]];
    $("var-bar").innerHTML = parts.map(([v, c]) => `<span style="width:${(v * 100).toFixed(2)}%;background:${c}"></span>`).join("");
    $("var-legend").innerHTML = parts.map(([v, c, t]) => `<span><i style="background:${c}"></i>${t}: ${(v * 100).toFixed(1)}%</span>`).join("");
    const H = 230, ctx = canvas("spread", H), R = { l: 56, r: 750, t: 10, b: 190 }, xr = xRange(xd);
    const top = (yd.x[1] - yd.x[0]) / 4;
    const { X, Y } = frame(ctx, R, xr, [0, top], axisTicks(xd, xr), ticks(0, top, 4), { xlab: "X: " + xd.name, ylab: "SD of Y", ylabX: 16 });
    clip(ctx, R);
    path(ctx, [[R.l, Y(P.sy)], [R.r, Y(P.sy)]], C.ink2, 1.4, [5, 4]);
    path(ctx, profileOf(xd, 80).map((b) => [X(b.x), b.n >= 20 ? Y(b.sd) : NaN]), C.band, 2.4);
    const b = Q.band(P, st.s);
    path(ctx, [[X(st.s), R.t], [X(st.s), R.b]], C.ink3, 1, [2, 3]);
    if (Number.isFinite(b.sd)) dot(ctx, X(st.s), Y(b.sd), 5.5, C.band);
    ctx.restore();
    if (P.sy > top) { ctx.fillStyle = C.ink2; ctx.textAlign = "right"; ctx.fillText(`all patients: ${fmt(P.sy)} (off the top)`, R.r - 4, R.t + 12); }
    const inf = !Number.isFinite(yd.variance(st.py));
    $("var-tiles").innerHTML =
      tile("Var(Y), all patients", fmt(P.sy * P.sy), inf ? "infinite in theory; this is the 20,000" : `SD ${fmt(P.sy)}`) +
      tile(`Var(Y) at X ≈ ${fmt(st.s)}`, fmt(b.sd * b.sd), `SD ${fmt(b.sd)}`) +
      tile("Knowing X shrinks the SD by", Number.isFinite(b.sd) ? `${Math.round((1 - b.sd / P.sy) * 100)}%` : "—", "at this s; the curve shows every s") +
      tile("R² of the line", fmt(r2), "= r², the line's share of Var(Y)");
  }

  // ---------- 3. sampling error ----------
  // a spread for the axes that heavy tails cannot blow up: the rule with robust spreads at the smallest n
  function axisSE(stat, n) {
    const rr = Math.min(0.99, P.r * P.r);
    return { mean: P.ry / Math.sqrt(n), sd: P.ry / Math.sqrt(2 * n), r: (1 - rr) / Math.sqrt(n), slope: P.ry * Math.sqrt(1 - rr) / (P.rx * Math.sqrt(n)) }[stat];
  }
  function histRange() {
    const S = Q.STATS[st.stat], c = S.pop(P), h = 4 * axisSE(st.stat, NS[0]);
    if (st.stat === "r") return [-1, 1];
    if (st.stat === "sd") return [0, c + h];
    return [c - h, c + h];
  }
  function drawSamples(k) {
    const r = rng(st.sampSeed++ * 104729 + 3), S = Q.STATS[st.stat];
    for (let i = 0; i < k; i++) { const idx = Q.sample(P, r, st.sampN); st.last = idx; st.est.push(S.est(P, idx)); }
  }
  function drawSampling() {
    const xd = byId[st.x], yd = byId[st.y], S = Q.STATS[st.stat];
    // the latest sample
    let ctx = canvas("samp-scatter", 300);
    const R = { l: 56, r: 750, t: 10, b: 262 }, xr = xRange(xd), yr = xRange(yd);
    let F = frame(ctx, R, xr, yr, axisTicks(xd, xr), axisTicks(yd, yr), { xlab: "X: " + xd.name, ylab: "Y", ylabX: 16 });
    clip(ctx, R);
    points(ctx, F.X, F.Y, everyone(), C.dot, 1.4, 0.18);
    if (st.last) {
      points(ctx, F.X, F.Y, st.last, C.samp, st.sampN > 250 ? 3 : 5, 0.9);
      if (st.stat === "r" || st.stat === "slope") {
        const m = (() => { let sx = 0, sy = 0, sxx = 0, sxy = 0; for (const i of st.last) { sx += P.X[i]; sy += P.Y[i]; sxx += P.X[i] ** 2; sxy += P.X[i] * P.Y[i]; } const n = st.last.length, mx = sx / n, my = sy / n; const b = (sxy / n - mx * my) / (sxx / n - mx * mx); return { b, a: my - b * mx }; })();
        path(ctx, xr.map((x) => [F.X(x), F.Y(P.icpt + P.slope * x)]), C.line, 2, [7, 5]);
        if (Number.isFinite(m.b)) path(ctx, xr.map((x) => [F.X(x), F.Y(m.a + m.b * x)]), C.samp, 2.2);
      } else {
        const est = st.est[st.est.length - 1];
        const lines = st.stat === "mean" ? [[P.my, est]] : [[P.my + P.sy, Q.STATS.mean.est(P, st.last) + est], [P.my - P.sy, Q.STATS.mean.est(P, st.last) - est]];
        for (const [pop, smp] of lines) { path(ctx, [[R.l, F.Y(pop)], [R.r, F.Y(pop)]], C.line, 2, [7, 5]); path(ctx, [[R.l, F.Y(smp)], [R.r, F.Y(smp)]], C.samp, 2.2); }
      }
    }
    ctx.restore();
    const what = { mean: "mean", sd: "mean ± SD", r: "line", slope: "line" }[st.stat];
    legend("samp-legend", [[C.samp, "box", `latest sample (n = ${st.sampN})`], [C.samp, "line", `its ${what}`], [C.line, "dash", `the population's ${what}`]]);

    // every estimate so far
    ctx = canvas("samp-hist", 220);
    const HR = { l: 56, r: 750, t: 10, b: 180 }, hr = histRange(), B = 50, w = (hr[1] - hr[0]) / B, h = new Float64Array(B);
    let off = 0;
    for (const v of st.est) { const j = Math.floor((v - hr[0]) / w); if (j >= 0 && j < B) h[j]++; else off++; }
    const top = Math.max(0.25, ...Array.from(h).map((c) => c / Math.max(1, st.est.length)));
    const yTop = top <= 0.25 ? 0.25 : top <= 0.5 ? 0.5 : 1;
    F = frame(ctx, HR, hr, [0, yTop], ticks(hr[0], hr[1], 8), ticks(0, yTop, 4), { xlab: S.name, ylab: "share of samples", ylabX: 16 });
    ctx.fillStyle = C.samp; ctx.globalAlpha = 0.55;
    for (let j = 0; j < B; j++) if (h[j]) { const v = h[j] / st.est.length; ctx.fillRect(F.X(hr[0] + j * w) + 0.5, F.Y(v), F.X(w) - F.X(0) - 1, F.Y(0) - F.Y(v)); }
    ctx.globalAlpha = 1;
    clip(ctx, HR);
    const pop = S.pop(P), sm = Q.summary(st.est);
    path(ctx, [[F.X(pop), HR.t], [F.X(pop), HR.b]], C.line, 2.2, [6, 4]);
    if (st.est.length) { const lv = st.est[st.est.length - 1]; path(ctx, [[F.X(lv), HR.b - 14], [F.X(lv), HR.b]], C.ink, 2.5); }
    ctx.restore();
    if (off) { ctx.fillStyle = C.ink2; ctx.textAlign = "right"; ctx.fillText(`${off} estimate${off > 1 ? "s" : ""} off the axis`, HR.r - 4, HR.t + 12); }
    $("hist-label").textContent = `Estimates from ${st.est.length.toLocaleString()} samples of n = ${st.sampN}`;
    $("samp-tiles").innerHTML =
      tile("The population's value", fmt(pop), "what every sample is trying to hit") +
      tile("Latest sample", st.est.length ? fmt(st.est[st.est.length - 1]) : "—", "black tick on the histogram") +
      tile("Standard error", st.est.length > 1 ? fmt(sm.sd) : "—", `SD of the ${st.est.length.toLocaleString()} estimates`) +
      tile("Rule of thumb", fmt(S.rule(P, st.sampN)), S.ruleText);
  }
  let sweepCache = null;
  function drawSweep() {
    const key = [P.N, st.x, st.y, JSON.stringify(st.px), JSON.stringify(st.py), st.link, st.rho, st.seed, st.stat].join("|");
    if (!sweepCache || sweepCache.key !== key) {
      const S = Q.STATS[st.stat], r = rng(91);
      sweepCache = { key, se: NS.map((n) => { const v = []; for (let k = 0; k < 300; k++) v.push(S.est(P, Q.sample(P, r, n))); return Q.summary(v).sd; }) };
    }
    const S = Q.STATS[st.stat], ctx = canvas("samp-sweep", 220), R = { l: 56, r: 732, t: 10, b: 180 }, top = 1.8 * axisSE(st.stat, NS[0]);
    const F = frame(ctx, R, [NS[0], NS[NS.length - 1]], [0, top], NS, ticks(0, top, 4), { logx: true, xgrid: true, xlab: "n, patients per sample (log scale)", ylab: "standard error", ylabX: 16 });
    clip(ctx, R);
    const fine = Array.from({ length: 100 }, (_, i) => NS[0] * (NS[NS.length - 1] / NS[0]) ** (i / 99));
    path(ctx, fine.map((n) => [F.X(n), F.Y(S.rule(P, n))]), C.ink2, 1.6, [5, 4]);
    path(ctx, NS.map((n, i) => [F.X(n), F.Y(sweepCache.se[i])]), C.samp, 2.2);
    NS.forEach((n, i) => dot(ctx, F.X(n), F.Y(sweepCache.se[i]), n === st.sampN ? 6 : 4, n === st.sampN ? C.ink : C.samp));
    ctx.restore();
    legend("sweep-legend", [[C.samp, "line", "measured over 300 samples at each n"], [C.ink2, "dash", "rule of thumb: " + S.ruleText]]);
  }

  // ---------- 4. specification error ----------
  let specCache = null;
  function specSweep() {
    const key = [P.N, st.x, st.y, JSON.stringify(st.px), JSON.stringify(st.py), st.link, st.rho, st.seed, st.model, st.s].join("|");
    if (!specCache || specCache.key !== key) specCache = { key, rows: NS.map((n) => Q.predictionError(P, st.model, st.s, n, 200, 77 + n)) };
    return specCache.rows;
  }
  function drawSpec() {
    const xd = byId[st.x], yd = byId[st.y], M = Q.MODELS[st.model], xr = xRange(xd), yr = xRange(yd);
    let ctx = canvas("spec-plot", 340);
    const R = { l: 56, r: 750, t: 10, b: 300 };
    let F = frame(ctx, R, xr, yr, axisTicks(xd, xr), axisTicks(yd, yr), { xlab: "X: " + xd.name, ylab: "Y", ylabX: 16 });
    clip(ctx, R);
    points(ctx, F.X, F.Y, everyone(), C.dot, 1.4, 0.18);
    const g = grid(xd, 120), r = rng(st.seed * 31 + st.specN);
    for (let k = 0; k < 40; k++) {
      const f = M.fit(P, Q.sample(P, r, st.specN));
      if (f) { ctx.globalAlpha = 0.35; path(ctx, g.map((x) => [F.X(x), F.Y(f(x))]), C.samp, 1.2); ctx.globalAlpha = 1; }
    }
    path(ctx, curveOf(xd).map(([x, y]) => [F.X(x), F.Y(y)]), C.truth, 2.8);
    const best = M.fit(P, everyone());
    path(ctx, g.map((x) => [F.X(x), F.Y(best(x))]), C.line, 2.4, [7, 5]);
    path(ctx, [[F.X(st.s), R.t], [F.X(st.s), R.b]], C.ink3, 1, [2, 3]);
    const rows = specSweep(), row = rows[NS.indexOf(st.specN)];
    if (Number.isFinite(row.truth)) dot(ctx, F.X(st.s), F.Y(row.truth), 6, C.truth);
    if (Number.isFinite(row.best)) dot(ctx, F.X(st.s), F.Y(row.best), 5, C.line);
    ctx.restore();
    legend("spec-legend", [[C.truth, "line", "truth: band averages"], [C.line, "dash", `best ${M.name.toLowerCase()} (all 20,000)`], [C.samp, "line", `40 fits from samples of n = ${st.specN}`]]);
    // the same miss, averaged over every patient: root mean square of (best fit − truth) along X
    let ss = 0, w = 0;
    for (const b of profileOf(xd, 120)) if (b.n >= 20) { const d = best(b.x) - b.mean; if (Number.isFinite(d)) { ss += b.n * d * d; w += b.n; } }
    $("spec-tiles").innerHTML =
      tile(`Truth at X = ${fmt(st.s)}`, fmt(row.truth), "average Y of the population there") +
      tile("Specification error, all X", fmt(Math.sqrt(ss / w)), "root mean square over every patient") +
      tile("Specification error", fmt(row.spec), `best ${M.name.toLowerCase()} − truth; fixed in n`) +
      tile("Sampling error", fmt(row.sampling), `SD over 200 samples of n = ${st.specN}` + (row.missing ? ` · ${row.missing} had no point near s` : "")) +
      tile("Total error (RMSE)", fmt(row.rmse), "≈ √(specification² + sampling²)");

    ctx = canvas("spec-sweep", 220);
    const SRR = { l: 56, r: 732, t: 10, b: 180 }, top = Math.max(0.5, +(1.2 * P.ry).toPrecision(2));
    F = frame(ctx, SRR, [NS[0], NS[NS.length - 1]], [0, top], NS, ticks(0, top, 4), { logx: true, xgrid: true, xlab: "n, patients per sample (log scale)", ylab: "error at s", ylabX: 16 });
    clip(ctx, SRR);
    path(ctx, rows.map((w, i) => [F.X(NS[i]), F.Y(Math.abs(w.spec))]), C.line, 2.4, [7, 5]);
    path(ctx, rows.map((w, i) => [F.X(NS[i]), F.Y(w.sampling)]), C.samp, 2.2);
    path(ctx, rows.map((w, i) => [F.X(NS[i]), F.Y(w.rmse)]), C.ink, 2.2);
    rows.forEach((w, i) => { if (NS[i] === st.specN) { dot(ctx, F.X(NS[i]), F.Y(w.rmse), 5.5, C.ink); dot(ctx, F.X(NS[i]), F.Y(w.sampling), 5, C.samp); dot(ctx, F.X(NS[i]), F.Y(Math.abs(w.spec)), 5, C.line); } });
    ctx.restore();
    legend("spec-sweep-legend", [[C.line, "dash", "|specification error|"], [C.samp, "line", "sampling error"], [C.ink, "line", "total (RMSE)"]]);
  }

  // ---------- wiring ----------
  function seg(id, items, key, after) {
    const box = $(id);
    box.innerHTML = items.map(([v, t]) => `<button data-v="${v}">${t}</button>`).join("");
    const sync = () => box.querySelectorAll("button").forEach((b) => b.setAttribute("aria-pressed", b.dataset.v === String(st[key]) ? "true" : "false"));
    box.querySelectorAll("button").forEach((b) => b.addEventListener("click", () => {
      st[key] = typeof st[key] === "number" ? +b.dataset.v : b.dataset.v; sync(); after();
    }));
    sync();
  }
  function resetSamples() { st.est = []; st.last = null; drawSamples(500); }
  function renderAll() { drawScatter(); drawVariance(); drawSampling(); drawSweep(); drawSpec(); }
  let pending = 0, pendingReset = false;
  function rebuild(resetS) {
    cancelAnimationFrame(pending);
    pendingReset = pendingReset || !!resetS;     // a newer rebuild must not drop an older one's reset of s
    pending = requestAnimationFrame(() => {
      P = Q.population({ xId: st.x, yId: st.y, px: st.px, py: st.py, link: st.link, rho: st.rho, seed: st.seed });
      sControl(pendingReset);
      pendingReset = false;
      resetSamples();
      renderAll();
    });
  }
  let sTimer = 0;
  function moveS() {
    st.s = +$("s").value; $("s-out").textContent = fmt(st.s);
    drawScatter(); drawVariance();
    clearTimeout(sTimer); sTimer = setTimeout(drawSpec, 90);
  }

  fillSelect("x-dist", st.x); fillSelect("y-dist", st.y);
  setDist("x", st.x); setDist("y", st.y);
  $("x-dist").addEventListener("change", (e) => { setDist("x", e.target.value); rebuild(true); });
  $("y-dist").addEventListener("change", (e) => { setDist("y", e.target.value); rebuild(); });
  const linkBox = $("link");
  const syncLink = () => linkBox.querySelectorAll("button").forEach((b) => b.setAttribute("aria-pressed", b.dataset.v === st.link ? "true" : "false"));
  linkBox.querySelectorAll("button").forEach((b) => b.addEventListener("click", () => { st.link = b.dataset.v; syncLink(); rebuild(); }));
  syncLink();
  $("rho").addEventListener("input", () => { st.rho = +$("rho").value; $("rho-out").textContent = fmt(st.rho); rebuild(); });
  $("rho-out").textContent = fmt(st.rho);
  $("s").addEventListener("input", moveS);
  $("new-pop").addEventListener("click", () => { st.seed++; rebuild(); });
  ["show-line", "show-true", "show-cond", "show-cov"].forEach((id) => $(id).addEventListener("change", drawScatter));
  seg("stat", Object.entries(Q.STATS).map(([k, v]) => [k, v.name]), "stat", () => { resetSamples(); drawSampling(); drawSweep(); });
  seg("samp-n", NS.map((n) => [n, n.toLocaleString()]), "sampN", () => { resetSamples(); drawSampling(); drawSweep(); });
  seg("model", Object.entries(Q.MODELS).map(([k, v]) => [k, v.name]), "model", drawSpec);
  seg("spec-n", NS.map((n) => [n, n.toLocaleString()]), "specN", drawSpec);
  $("draw-1").addEventListener("click", () => { drawSamples(1); drawSampling(); });
  $("draw-500").addEventListener("click", () => { drawSamples(500); drawSampling(); });
  $("draw-clear").addEventListener("click", () => { st.est = []; st.last = null; drawSampling(); });
  rebuild(true);
})();
