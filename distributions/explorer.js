// The explorer: one distribution at a time, with sliders, simulated draws, a shaded probability, and its CDF.
(function () {
  "use strict";
  const { list, byId, edges, edgeTypes, defaults, cdf, rng } = DIST;
  const $ = (id) => document.getElementById(id);
  const COLOR = { discrete: "#2a78d6", continuous: "#eb6834" };
  const SAMP = "#1baf7a", MARK = "#ffe27a", INK = "#0b0b0b", INK2 = "#52514e", INK3 = "#898781", RULE = "#e4e3de", RULE2 = "#c9c8c1";

  const params = {};           // remembered slider values, per distribution
  const shade = {};            // remembered [a, b], per distribution
  let cur = null, n = 1000, seed = 1, draws = [];

  // ---------- helpers ----------
  function fmt(v) {
    if (Number.isNaN(v)) return "undefined";
    if (!Number.isFinite(v)) return "∞";
    const a = Math.abs(v);
    if (a >= 1000) return Math.round(v).toLocaleString();
    if (a >= 100) return v.toFixed(1);
    if (a >= 1) return String(+v.toFixed(3));
    return String(+v.toFixed(4));
  }
  function ticks(lo, hi, count) {
    const raw = (hi - lo) / count, mag = 10 ** Math.floor(Math.log10(raw)), e = raw / mag;
    const step = (e >= 7.5 ? 10 : e >= 3.5 ? 5 : e >= 1.5 ? 2 : 1) * mag, out = [];
    for (let v = Math.ceil(lo / step - 1e-9) * step; v <= hi + 1e-9; v += step) out.push(+v.toFixed(10));
    return out;
  }
  const esc = (s) => String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;");
  function frame(W, H, M, xr, yr, xt, yt, yfmt) {
    const X = (v) => M.l + (v - xr[0]) / (xr[1] - xr[0]) * (W - M.l - M.r);
    const Y = (v) => H - M.b - (v - yr[0]) / (yr[1] - yr[0]) * (H - M.t - M.b);
    let s = "";
    for (const t of yt) s += `<line x1="${M.l}" x2="${W - M.r}" y1="${Y(t)}" y2="${Y(t)}" stroke="${RULE}"/>` +
      `<text x="${M.l - 8}" y="${Y(t) + 4}" text-anchor="end" font-size="11" fill="${INK3}">${yfmt(t)}</text>`;
    for (const t of xt) s += `<text x="${X(t)}" y="${H - M.b + 16}" text-anchor="middle" font-size="11" fill="${INK3}">${fmt(t)}</text>`;
    s += `<line x1="${M.l}" x2="${W - M.r}" y1="${Y(yr[0])}" y2="${Y(yr[0])}" stroke="${RULE2}"/>`;
    return { X, Y, s };
  }
  const xRange = (d) => (d.kind === "discrete" ? [d.x[0] - 0.5, d.x[1] + 0.5] : d.x);
  const step = (d) => (d.kind === "discrete" ? 1 : +((d.x[1] - d.x[0]) / 200).toPrecision(2));
  const snap = (d, v) => { const s = step(d); return +(Math.round(v / s) * s).toFixed(6); };

  // ---------- picker ----------
  function buildPicker() {
    let html = "";
    for (const kind of ["discrete", "continuous"]) {
      html += `<h2>${kind === "discrete" ? "Discrete (counts)" : "Continuous (measurements)"}</h2>`;
      for (const d of list.filter((x) => x.kind === kind))
        html += `<a href="#${d.id}" data-id="${d.id}"><span class="sw" style="background:${COLOR[kind]}"></span>${esc(d.name)}</a>`;
    }
    $("picker").innerHTML = html;
  }

  // ---------- controls ----------
  function buildControls(d) {
    const p = params[d.id];
    if (!d.params.length) { $("controls").innerHTML = `<p class="noparams">No parameters: this one has a single fixed shape.</p>`; return; }
    $("controls").innerHTML = d.params.map(([k, label, min, max, st]) =>
      `<div class="ctl"><label for="p-${k}">${esc(label)}</label><input type="range" id="p-${k}" data-k="${k}" min="${min}" max="${max}" step="${st}" value="${p[k]}"><output id="o-${k}"></output></div>`).join("");
    $("controls").querySelectorAll("input").forEach((inp) => inp.addEventListener("input", () => {
      p[inp.dataset.k] = +inp.value;
      d.fix && d.fix(p);
      syncControls(d);
      resample();
      render();
    }));
    syncControls(d);
  }
  function syncControls(d) {
    const p = params[d.id];
    for (const [k] of d.params) { $(`p-${k}`).value = p[k]; $(`o-${k}`).textContent = fmt(p[k]); }
  }

  function setupShade(d) {
    const xr = d.kind === "discrete" ? d.x : d.x, st = step(d);
    for (const id of ["sa", "sb"]) { const el = $(id); el.min = xr[0]; el.max = xr[1]; el.step = st; }
    if (!shade[d.id]) {
      const p = params[d.id], m = d.mean(p), sd = Math.sqrt(d.variance(p));
      let a, b;
      if (Number.isFinite(m) && Number.isFinite(sd) && sd > 0) { a = m - sd; b = m + sd; }
      else { a = xr[0] + 0.3 * (xr[1] - xr[0]); b = xr[0] + 0.6 * (xr[1] - xr[0]); }
      a = Math.max(xr[0], Math.min(xr[1], snap(d, a)));
      b = Math.max(a, Math.min(xr[1], snap(d, b)));
      shade[d.id] = [a, b];
    }
    $("sa").value = shade[d.id][0];
    $("sb").value = shade[d.id][1];
  }

  // ---------- simulation ----------
  function resample() {
    const d = cur, p = params[d.id], r = rng(seed * 7919 + d.id.length);
    draws = new Float64Array(n);
    for (let i = 0; i < n; i++) draws[i] = d.sample(r, p);
  }

  // ---------- charts ----------
  function drawPdf(d) {
    const p = params[d.id], W = 760, H = 300, M = { l: 54, r: 16, t: 14, b: 30 };
    const xr = xRange(d), yr = [0, d.y];
    const xt = d.kind === "discrete" ? ticks(d.x[0], d.x[1], Math.min(10, d.x[1] - d.x[0])).filter(Number.isInteger) : ticks(xr[0], xr[1], 8);
    const F = frame(W, H, M, xr, yr, xt, ticks(0, d.y, 4), fmt), X = F.X, Y = F.Y;
    const [a, b] = shade[d.id], col = COLOR[d.kind], show = $("show-samples").checked;
    let s = `<defs><clipPath id="pc"><rect x="${M.l}" y="${M.t}" width="${W - M.l - M.r}" height="${H - M.t - M.b}"/></clipPath></defs>` + F.s;
    let clipped = false;
    s += `<g clip-path="url(#pc)">`;
    if (d.kind === "discrete") {
      s += `<rect x="${X(a - 0.5)}" y="${M.t}" width="${X(b + 0.5) - X(a - 0.5)}" height="${H - M.t - M.b}" fill="${MARK}" opacity="0.45"/>`;
      const bw = Math.max(1.5, (X(1) - X(0)) * 0.6);
      for (let k = d.x[0]; k <= d.x[1]; k++) {
        const v = d.pdf(k, p);
        if (v <= 0) continue;
        if (v > d.y) clipped = true;
        s += `<rect x="${X(k) - bw / 2}" y="${Y(Math.min(v, d.y * 1.02))}" width="${bw}" height="${Y(0) - Y(Math.min(v, d.y * 1.02))}" fill="${col}" opacity="${k >= a && k <= b ? 1 : 0.55}"/>`;
      }
      if (show && draws.length) {
        const cnt = new Map();
        for (const v of draws) cnt.set(v, (cnt.get(v) || 0) + 1);
        for (const [k, c] of cnt) {
          if (k < d.x[0] || k > d.x[1]) continue;
          const y = Y(Math.min(c / n, d.y * 1.02));
          s += `<line x1="${X(k) - bw * 0.75}" x2="${X(k) + bw * 0.75}" y1="${y}" y2="${y}" stroke="${SAMP}" stroke-width="2.5"/>`;
        }
      }
    } else {
      const N = 600, pts = [];
      for (let i = 0; i <= N; i++) {
        const x = xr[0] + (xr[1] - xr[0]) * i / N;
        let v = d.pdf(x, p);
        if (!Number.isFinite(v)) v = d.y * 2;
        if (v > d.y) clipped = true;
        pts.push([x, Math.min(v, d.y * 1.05)]);
      }
      if (show && draws.length) {
        const B = 60, w = (xr[1] - xr[0]) / B, h = new Float64Array(B);
        for (const v of draws) { const j = Math.floor((v - xr[0]) / w); if (j >= 0 && j < B) h[j]++; }
        for (let j = 0; j < B; j++) {
          if (!h[j]) continue;
          const dens = Math.min(h[j] / (n * w), d.y * 1.05);
          s += `<rect x="${X(xr[0] + j * w) + 0.5}" y="${Y(dens)}" width="${X(w) - X(0) - 1}" height="${Y(0) - Y(dens)}" fill="${SAMP}" opacity="0.32"/>`;
        }
      }
      const inside = pts.filter(([x]) => x >= a && x <= b);
      if (inside.length) {
        s += `<path d="M${X(a)},${Y(0)}L${X(a)},${Y(Math.min(d.pdf(a, p), d.y * 1.05) || 0)}` +
          inside.map(([x, v]) => `L${X(x).toFixed(1)},${Y(v).toFixed(1)}`).join("") +
          `L${X(b)},${Y(Math.min(d.pdf(b, p), d.y * 1.05) || 0)}L${X(b)},${Y(0)}Z" fill="${MARK}" opacity="0.85"/>`;
      }
      s += `<path d="${pts.map(([x, v], i) => (i ? "L" : "M") + X(x).toFixed(1) + "," + Y(v).toFixed(1)).join("")}" fill="none" stroke="${col}" stroke-width="2.4" stroke-linejoin="round"/>`;
    }
    const m = d.mean(p);
    if (Number.isFinite(m) && m >= xr[0] && m <= xr[1]) {
      s += `<line x1="${X(m)}" x2="${X(m)}" y1="${M.t}" y2="${Y(0)}" stroke="${INK}" stroke-dasharray="4 4" stroke-width="1.2"/>` +
        `<text x="${X(m) + 5}" y="${M.t + 12}" font-size="11" fill="${INK}">mean</text>`;
    }
    s += `</g>`;
    if (clipped) s += `<text x="${W - M.r - 4}" y="${M.t + 12}" text-anchor="end" font-size="11" fill="${INK2}">↑ peak runs off the fixed axis</text>`;
    $("pdf").innerHTML = s;
    $("pdf-label").textContent = d.kind === "discrete" ? "Probability of each value (PMF)" : "Density (PDF)";
    $("pdf-legend").innerHTML =
      `<span><i style="background:${col}"></i>${d.kind === "discrete" ? "P(X = k)" : "f(x)"}</span>` +
      `<span><i style="background:${SAMP};opacity:${d.kind === "discrete" ? 1 : 0.5}"></i>${n.toLocaleString()} simulated draws</span>` +
      `<span><i style="background:${MARK}"></i>shaded region</span>`;
  }

  function drawCdf(d) {
    const p = params[d.id], W = 760, H = 200, M = { l: 54, r: 16, t: 10, b: 30 };
    const xr = xRange(d);
    const xt = d.kind === "discrete" ? ticks(d.x[0], d.x[1], Math.min(10, d.x[1] - d.x[0])).filter(Number.isInteger) : ticks(xr[0], xr[1], 8);
    const F = frame(W, H, M, xr, [0, 1], xt, [0, 0.25, 0.5, 0.75, 1], (t) => String(t)), X = F.X, Y = F.Y;
    const col = COLOR[d.kind], [a, b] = shade[d.id];
    let s = F.s, path = "";
    if (d.kind === "discrete") {
      let prev = cdf(d, xr[0], p);
      path = `M${X(xr[0])},${Y(prev)}`;
      for (let k = d.x[0]; k <= d.x[1]; k++) { const c = cdf(d, k, p); path += `L${X(k)},${Y(prev)}L${X(k)},${Y(c)}`; prev = c; }
      path += `L${X(xr[1])},${Y(prev)}`;
    } else {
      const N = 400;
      for (let i = 0; i <= N; i++) { const x = xr[0] + (xr[1] - xr[0]) * i / N; path += (i ? "L" : "M") + X(x).toFixed(1) + "," + Y(cdf(d, x, p)).toFixed(1); }
    }
    s += `<path d="${path}" fill="none" stroke="${col}" stroke-width="2.4" stroke-linejoin="round"/>`;
    const fa = d.kind === "discrete" ? cdf(d, a - 1, p) : cdf(d, a, p), fb = cdf(d, b, p);
    const xa = d.kind === "discrete" ? a - 0.5 : a, xb = d.kind === "discrete" ? b + 0.5 : b;
    s += `<line x1="${X(xa)}" x2="${X(xb)}" y1="${Y(fa)}" y2="${Y(fa)}" stroke="${INK2}" stroke-dasharray="3 3"/>`;
    s += `<rect x="${X(xb) - 3}" y="${Y(fb)}" width="6" height="${Math.max(0, Y(fa) - Y(fb))}" fill="${MARK}" stroke="${INK}" stroke-width="0.8"/>`;
    s += `<circle cx="${X(xa)}" cy="${Y(fa)}" r="3.5" fill="${INK}"/><circle cx="${X(xb)}" cy="${Y(fb)}" r="3.5" fill="${INK}"/>`;
    s += `<text x="${X(xb) + 8}" y="${(Y(fa) + Y(fb)) / 2 + 4}" font-size="11" fill="${INK}">F(b) − F(a)</text>`;
    $("cdf").innerHTML = s;
  }

  function drawProb(d) {
    const p = params[d.id], [a, b] = shade[d.id];
    $("sa-out").textContent = fmt(a);
    $("sb-out").textContent = fmt(b);
    const prob = d.kind === "discrete" ? cdf(d, b, p) - cdf(d, a - 1, p) : cdf(d, b, p) - cdf(d, a, p);
    let inS = 0;
    for (const v of draws) if (v >= a && v <= b) inS++;
    $("prob").innerHTML = `P(${fmt(a)} ≤ X ≤ ${fmt(b)}) = <b>${(prob * 100).toFixed(1)}%</b> of the area` +
      (draws.length ? ` · ${(inS / draws.length * 100).toFixed(1)}% of the draws landed there` : "");
  }

  function drawStats(d) {
    const p = params[d.id], m = d.mean(p), v = d.variance(p);
    let s1 = 0, s2 = 0;
    for (const x of draws) { s1 += x; s2 += x * x; }
    const sm = s1 / draws.length, ssd = Math.sqrt(Math.max(0, s2 / draws.length - sm * sm));
    const tile = (k, val, sub) => `<div class="stat"><div class="k">${k}</div><div class="v">${val}</div><div class="s">${sub}</div></div>`;
    $("stats").innerHTML =
      tile("Mean", fmt(m), `draws: ${fmt(sm)}`) +
      tile("Standard deviation", fmt(Number.isNaN(v) ? NaN : Math.sqrt(v)), `draws: ${fmt(ssd)}`) +
      tile("Support", esc(d.support(p)), d.kind === "discrete" ? "whole numbers" : "a continuum");
  }

  function drawLinks(d) {
    const out = edges.filter((e) => e.from === d.id), inn = edges.filter((e) => e.to === d.id);
    const li = (other, e, arrow) => `<li>${arrow} <a href="#${other.id}">${esc(other.name)}</a> <span class="how">${esc(e.label)}</span> <span class="type">${edgeTypes[e.type].name.split(" (")[0]}</span></li>`;
    $("links").innerHTML =
      (inn.length ? `<h3>Comes from</h3><ul>${inn.map((e) => li(byId[e.from], e, "←")).join("")}</ul>` : "") +
      (out.length ? `<h3 style="margin-top:14px">Leads to</h3><ul>${out.map((e) => li(byId[e.to], e, "→")).join("")}</ul>` : "") +
      `<p style="margin:14px 0 0"><a href="map.html#${d.id}">See it on the map →</a></p>`;
  }

  function render() {
    const d = cur;
    drawPdf(d); drawProb(d); drawCdf(d); drawStats(d);
  }

  function select(id) {
    const d = byId[id] || byId.normal;
    cur = d;
    if (!params[d.id]) { params[d.id] = defaults(d); d.fix && d.fix(params[d.id]); }
    document.querySelectorAll("#picker a").forEach((a) => a.setAttribute("aria-current", a.dataset.id === d.id ? "true" : "false"));
    $("kind").textContent = d.kind;
    $("kind").className = "tag " + d.kind;
    $("name").textContent = d.name;
    $("story").textContent = d.story;
    $("formula").textContent = d.formula;
    $("gen").textContent = "How one draw is made: " + d.gen;
    document.title = d.name + " · Distribution Basics";
    buildControls(d);
    setupShade(d);
    drawLinks(d);
    resample();
    render();
  }

  // ---------- wiring ----------
  buildPicker();
  document.querySelectorAll(".sampler button[data-n]").forEach((b) => {
    b.setAttribute("aria-pressed", +b.dataset.n === n ? "true" : "false");
    b.addEventListener("click", () => {
      n = +b.dataset.n;
      document.querySelectorAll(".sampler button[data-n]").forEach((x) => x.setAttribute("aria-pressed", x === b ? "true" : "false"));
      resample(); render();
    });
  });
  $("redraw").addEventListener("click", () => { seed++; resample(); render(); });
  $("show-samples").addEventListener("change", render);
  $("sa").addEventListener("input", () => {
    const s = shade[cur.id]; s[0] = +$("sa").value; if (s[0] > s[1]) { s[1] = s[0]; $("sb").value = s[1]; } render();
  });
  $("sb").addEventListener("input", () => {
    const s = shade[cur.id]; s[1] = +$("sb").value; if (s[1] < s[0]) { s[0] = s[1]; $("sa").value = s[0]; } render();
  });
  window.addEventListener("hashchange", () => select(location.hash.slice(1)));
  select(location.hash.slice(1) || "normal");
})();
