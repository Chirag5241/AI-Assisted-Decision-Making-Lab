// Small SVG charts drawn in the browser, so a figure can follow a slider while it moves.
// The numbers always come from Python; this file only places marks.
window.Charts = (function () {
  const SVG = "http://www.w3.org/2000/svg";
  const clamp = (v, lo, hi) => Math.min(Math.max(v, lo), hi);
  const el = (tag, cls, text) => {
    const e = document.createElement(tag);
    if (cls) e.className = cls;
    if (text !== undefined) e.textContent = text;
    return e;
  };
  const svgEl = (tag, attrs, text) => {
    const e = document.createElementNS(SVG, tag);
    for (const k in attrs) e.setAttribute(k, attrs[k]);
    if (text !== undefined) e.textContent = text;
    return e;
  };

  function niceTicks(lo, hi, target) {
    const span = hi - lo || 1, raw = span / target, mag = Math.pow(10, Math.floor(Math.log10(raw)));
    const step = [1, 2, 5, 10].map((m) => m * mag).find((s) => span / s <= target) || 10 * mag;
    const ticks = [];
    for (let v = Math.ceil(lo / step - 1e-9) * step; v <= hi + 1e-9; v += step) ticks.push(Math.abs(v) < step * 1e-9 ? 0 : v);
    return { ticks, step };
  }
  const fmtTick = (v, step) => v.toFixed(step >= 1 ? 0 : step >= 0.1 ? 1 : 2);

  // series: [{v, cls, name, dash, faded, off: [bool per round]}]   hlines: [{y, label, cls}]   shade: [[start, stop)]
  // A series with `off` is faded from each round marked off to the next one: the stretch where its value is not in use.
  // opts: {ylabel, alt, height, step, lo, hi, pct, yticks: [{v, label}], fmt, noXLabel, dotLabel, width, noLegend, dotRadius, wrongShare,
  //        xlabel, xscale (point i sits at x = i * xscale), vlines: [{t, label}],
  //        tipLines(t): extra lines for the tooltip of round t, shown above the series' values}
  function line(host, T, series, hlines, shade, opts) {
    const W = opts.width || 560, H = opts.height || 340, L = 54, R = 12, TOP = 10, B = opts.noXLabel ? 22 : 40, pw = W - L - R, ph = H - TOP - B;
    let lo = opts.lo, hi = opts.hi;
    if (lo === undefined || hi === undefined) {
      let mn = Infinity, mx = -Infinity;
      for (const s of series) for (const v of s.v) { if (v < mn) mn = v; if (v > mx) mx = v; }
      for (const h of hlines) { mn = Math.min(mn, h.y); mx = Math.max(mx, h.y); }
      const pad = 0.1 * ((mx - mn) || 1);
      if (lo === undefined) lo = mn - pad;
      if (hi === undefined) hi = mx + pad;
    }
    // axes are fixed, so a value beyond one is drawn at the frame rather than outside it (the tooltip keeps the number)
    const X = (t) => L + (T === 1 ? 0 : (t / (T - 1)) * pw), Y = (v) => TOP + (1 - (clamp(v, lo, hi) - lo) / (hi - lo)) * ph;
    const edge = (t) => Math.min(Math.max(X(t), L), L + pw);
    const text = (attrs, s) => { const t = svgEl("text", attrs); t.textContent = s; return t; };

    const svg = svgEl("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": opts.alt });
    for (const [a, b] of shade) {
      const x0 = edge(a - 0.5), x1 = edge(b - 0.5);
      svg.append(svgEl("rect", { class: "shade", x: x0, y: TOP, width: Math.max(0, x1 - x0), height: ph }));
    }
    if (opts.wrongShare) {
      // grey behind the plot, darker where the human picks the wrong action in more of the states
      const level = (t) => Math.round(opts.wrongShare[t] * 10) / 10;
      for (let start = 0, t = 1; t <= T; t++) {
        if (t < T && level(t) === level(start)) continue;
        if (level(start) > 0) {
          const x0 = edge(start - 0.5), x1 = edge(t - 0.5);
          svg.append(svgEl("rect", { class: "shade-share", x: x0, y: TOP, width: Math.max(0, x1 - x0), height: ph, "fill-opacity": (0.2 * level(start)).toFixed(3) }));
        }
        start = t;
      }
    }
    const auto = opts.pct ? { ticks: [0, 0.2, 0.4, 0.6, 0.8, 1], step: 0.2 } : niceTicks(lo, hi, H < 250 ? 4 : 6);
    const yticks = opts.yticks || auto.ticks.map((v) => ({ v, label: opts.pct ? Math.round(v * 100) + "%" : fmtTick(v, auto.step) }));
    for (const tick of yticks) {
      svg.append(svgEl("line", { class: "grid", x1: L, x2: L + pw, y1: Y(tick.v), y2: Y(tick.v) }),
        text({ x: L - 8, y: Y(tick.v) + 4, "text-anchor": "end", class: tick.faded ? "faded" : "" }, tick.label));
    }
    const xscale = opts.xscale || 1, xname = opts.xlabel || "round t";
    for (const v of niceTicks(0, (T - 1) * xscale, 8).ticks) {
      svg.append(svgEl("line", { class: "grid", x1: X(v / xscale), x2: X(v / xscale), y1: TOP, y2: TOP + ph }),
        text({ x: X(v / xscale), y: TOP + ph + 16, "text-anchor": "middle" }, v));
    }
    if (!opts.noXLabel) svg.append(text({ x: L + pw / 2, y: H - 4, "text-anchor": "middle" }, xname));
    for (const v of opts.vlines || []) {
      svg.append(svgEl("line", { class: "vline", x1: edge(v.t - 0.5), x2: edge(v.t - 0.5), y1: TOP, y2: TOP + ph }));
      if (v.label) svg.append(text({ x: edge(v.t - 0.5) + 4, y: TOP + 12, class: "vline-label" }, v.label));
    }
    svg.append(text({ x: 12, y: TOP + ph / 2, "text-anchor": "middle", transform: `rotate(-90 12 ${TOP + ph / 2})` }, opts.ylabel));
    for (const h of hlines) {
      svg.append(svgEl("line", { class: h.cls || "hline", x1: L, x2: L + pw, y1: Y(h.y), y2: Y(h.y) }));
      if (h.label) {
        // label above its line, unless that would leave the plot or sit on a nearby line's label
        const crowded = hlines.some((o) => o !== h && o.label && Y(o.y) <= Y(h.y) && Y(h.y) - Y(o.y) < 15);
        const above = Y(h.y) - TOP > 18 && !crowded;
        svg.append(text({ x: L + pw - 2, y: Y(h.y) + (above ? -5 : 13), "text-anchor": "end" }, h.label));
      }
    }
    for (const s of series) {
      const cls = "line " + s.cls + (s.dash ? " dashed" : "");
      if (s.off && !opts.step) {
        // one path per run of rounds that are all in use or all off, each reaching the next run's first point
        for (let start = 0, t = 1; t < T; t++) {
          if (t < T - 1 && s.off[t] === s.off[start]) continue;
          let d = "";
          for (let i = start; i <= t; i++) d += `${i > start ? "L" : "M"}${X(i).toFixed(1)} ${Y(s.v[i]).toFixed(1)}`;
          svg.append(svgEl("path", { class: cls + (s.off[start] ? " faded" : ""), d }));
          start = t;
        }
        continue;
      }
      let d = "";
      for (let t = 0; t < T; t++) {
        const y = Y(s.v[t]).toFixed(1);
        // a per-round value is a step: flat across its own round, jumping between rounds
        d += opts.step ? `${t ? "L" : "M"}${edge(t - 0.5).toFixed(1)} ${y}L${edge(t + 0.5).toFixed(1)} ${y}`
          : `${t ? "L" : "M"}${X(t).toFixed(1)} ${y}`;
      }
      svg.append(svgEl("path", { class: cls + (s.faded ? " faded" : ""), d }));
    }
    // a large dot on the value at round 0: the starting point the user sets
    for (const s of series) if (s.dot) svg.append(svgEl("circle", { class: "dot " + s.cls + (s.faded ? " faded" : ""), cx: X(0), cy: Y(s.v[0]), r: opts.dotRadius || 6.5 }));
    const cross = svgEl("line", { class: "cross", y1: TOP, y2: TOP + ph, visibility: "hidden" });
    const hit = svgEl("rect", { x: L, y: TOP, width: pw, height: ph, fill: "transparent" });
    svg.append(cross, hit);

    const legend = el("div", "legend");
    for (const s of series) {
      const item = el("span");
      item.append(el("i", s.cls.replace("c-", "bg-") + (s.dash ? " dash" : "")), s.name);
      legend.append(item);
    }
    if (opts.dotLabel) { const item = el("span"); item.append(el("i", "start"), opts.dotLabel); legend.append(item); }
    if (shade.length) { const item = el("span"); item.append(el("i", "box"), "human picks the wrong action"); legend.append(item); }
    if (opts.wrongShare) { const item = el("span"); item.append(el("i", "box share"), "grey: share of states with the wrong action"); legend.append(item); }
    const tip = el("div", "tip");
    tip.hidden = true;
    host.replaceChildren(...(opts.noLegend ? [] : [legend]), svg, tip);

    const fmt = opts.fmt || ((v) => (opts.pct ? Math.round(v * 100) + "%" : v.toFixed(2)));
    hit.addEventListener("pointermove", (e) => {
      const box = svg.getBoundingClientRect(), sx = (e.clientX - box.left) * (W / box.width);
      const t = Math.min(Math.max(Math.round(((sx - L) / pw) * (T - 1)), 0), T - 1);
      cross.setAttribute("x1", X(t)); cross.setAttribute("x2", X(t)); cross.setAttribute("visibility", "visible");
      tip.replaceChildren(el("b", "", opts.xlabel ? xname + " " + t * xscale : "round " + t));
      for (const text of opts.tipLines ? opts.tipLines(t) : []) tip.append(document.createElement("br"), text);
      for (const s of series) tip.append(document.createElement("br"), s.name + "  " + fmt(s.v[t]));
      tip.hidden = false;
      const left = (X(t) / W) * box.width, onLeft = left < box.width / 2;
      tip.style.left = onLeft ? left + 12 + "px" : "";
      tip.style.right = onLeft ? "" : box.width - left + 12 + "px";
      tip.style.top = svg.offsetTop + 6 + "px";
    });
    hit.addEventListener("pointerleave", () => { cross.setAttribute("visibility", "hidden"); tip.hidden = true; });
  }

  // map: {lim, G, bins (G rows, top row = largest dh0), labels}; the ring marks (du, dh0)
  function heat(host, keyHost, map, du, dh0, alt) {
    const { lim, G, bins, labels } = map, W = 460, L = 54, TOP = 8, P = 380, cell = P / G, last = labels.length - 1;
    const half = lim / (G - 1);   // grid points are cell centres, so the plot reaches half a cell past +-lim
    const X = (v) => L + ((v + lim + half) / (2 * (lim + half))) * P, Y = (v) => TOP + ((lim + half - v) / (2 * (lim + half))) * P;
    const cls = (b) => "hb" + (b === last ? 8 : b);
    const svg = svgEl("svg", { viewBox: `0 0 ${W} ${TOP + P + 44}`, role: "img", "aria-label": alt });
    for (let r = 0; r < G; r++) {
      let start = 0;
      for (let c = 1; c <= G; c++) {
        if (c < G && bins[r][c] === bins[r][start]) continue;   // merge a run of equal cells into one mark
        const rect = svgEl("rect", { class: cls(bins[r][start]), x: L + start * cell, y: TOP + r * cell, width: (c - start) * cell + 0.5, height: cell + 0.5 });
        rect.append(svgEl("title", {}, "right for good from round " + labels[bins[r][start]]));
        svg.append(rect);
        start = c;
      }
    }
    svg.append(svgEl("line", { class: "zero", x1: X(0), x2: X(0), y1: TOP, y2: TOP + P }),
      svgEl("line", { class: "zero", x1: L, x2: L + P, y1: Y(0), y2: Y(0) }));
    for (let v = -lim; v <= lim; v += lim > 4 ? 2 : 1) {
      svg.append(svgEl("text", { x: X(v), y: TOP + P + 16, "text-anchor": "middle" }, v),
        svgEl("text", { x: L - 8, y: Y(v) + 4, "text-anchor": "end" }, v));
    }
    svg.append(svgEl("text", { x: L + P / 2, y: TOP + P + 36, "text-anchor": "middle" }, "true gap  Δu = u₁ − u₂"),
      svgEl("text", { x: 12, y: TOP + P / 2, "text-anchor": "middle", transform: `rotate(-90 12 ${TOP + P / 2})` }, "initial belief gap  Δh₀"));
    const cx = X(clamp(du, -lim, lim)), cy = Y(clamp(dh0, -lim, lim));
    svg.append(svgEl("circle", { class: "ring-halo", cx, cy, r: 8 }), svgEl("circle", { class: "ring", cx, cy, r: 8 }));
    host.replaceChildren(svg);

    keyHost.replaceChildren(el("span", "", "Right for good from round:"), ...labels.map((label, i) => {
      const item = el("span"), swatch = svgEl("svg", { viewBox: "0 0 14 10" });
      swatch.append(svgEl("rect", { class: cls(i), x: 0, y: 0, width: 14, height: 10 }));
      item.append(swatch, label);
      return item;
    }));
  }

  // points: [{x, y, cls, ring}] on a square plot over [-lim, lim]^2; lines: [{slope, cls, label}] through the origin
  function scatter(host, points, lines, opts) {
    const S = 300, L = 40, TOP = 8, P = S - L - 10, lim = opts.lim || 3;
    const X = (v) => L + ((clamp(v, -lim, lim) + lim) / (2 * lim)) * P, Y = (v) => TOP + ((lim - clamp(v, -lim, lim)) / (2 * lim)) * P;
    const svg = svgEl("svg", { viewBox: `0 0 ${S} ${TOP + P + 36}`, role: "img", "aria-label": opts.alt });
    for (const v of [-2, 0, 2]) {
      svg.append(svgEl("line", { class: v ? "grid" : "zero", x1: X(v), x2: X(v), y1: TOP, y2: TOP + P }),
        svgEl("line", { class: v ? "grid" : "zero", x1: L, x2: L + P, y1: Y(v), y2: Y(v) }),
        svgEl("text", { x: X(v), y: TOP + P + 14, "text-anchor": "middle" }, v),
        svgEl("text", { x: L - 6, y: Y(v) + 4, "text-anchor": "end" }, v));
    }
    for (const p of points) svg.append(svgEl("circle", { class: "pt " + p.cls, cx: X(p.x), cy: Y(p.y), r: 2.6 }));
    for (const p of points) if (p.ring) svg.append(svgEl("circle", { class: "pt-ring", cx: X(p.x), cy: Y(p.y), r: 4.6 }));
    for (const ln of lines) {
      // clip y = slope * x to the plot square
      const x1 = Math.abs(ln.slope) > 1 ? lim / Math.abs(ln.slope) : lim;
      svg.append(svgEl("line", { class: "fit " + (ln.cls || ""), x1: X(-x1), y1: Y(-x1 * ln.slope), x2: X(x1), y2: Y(x1 * ln.slope) }));
      if (ln.label) svg.append(svgEl("text", { x: L + 4, y: TOP + 12, class: "fit-label" }, ln.label));
    }
    svg.append(svgEl("text", { x: L + P / 2, y: TOP + P + 32, "text-anchor": "middle", class: "axis-name" }, opts.xname),
      svgEl("text", { x: 10, y: TOP + P / 2, "text-anchor": "middle", class: "axis-name", transform: `rotate(-90 10 ${TOP + P / 2})` }, opts.yname));
    host.replaceChildren(svg);
  }

  // rows: [{name, on: [bool per round], group}]; marks: [{t, label}] drawn as vertical rules before round t.
  // A row marked `group` starts a new group of rows and is set a little apart from the ones above it.
  function schedule(host, T, rows, marks, alt) {
    const W = 1120, L = 54, R = 12, TOP = 6, row = 22, gap = 6, apart = 10, pw = W - L - R;
    const tops = [];
    let ph = 0;
    rows.forEach((r, i) => { ph += i ? gap + (r.group ? apart : 0) : 0; tops.push(TOP + ph); ph += row; });
    const X = (t) => L + (t / T) * pw;
    const svg = svgEl("svg", { viewBox: `0 0 ${W} ${TOP + ph + 40}`, role: "img", "aria-label": alt });
    rows.forEach((r, i) => {
      const y = tops[i];
      svg.append(svgEl("rect", { class: "sched-off", x: L, y, width: pw, height: row }),
        svgEl("text", { x: L - 8, y: y + row / 2 + 4, "text-anchor": "end", class: "sched-name" }, r.name));
      for (let start = 0, t = 1; t <= T; t++) {
        if (t < T && r.on[t] === r.on[start]) continue;
        if (r.on[start]) svg.append(svgEl("rect", { class: "sched-on", x: X(start), y, width: Math.max(0.5, X(t) - X(start) - 0.6), height: row }));
        start = t;
      }
    });
    for (const v of niceTicks(0, T - 1, 8).ticks) svg.append(svgEl("text", { x: X(v + 0.5), y: TOP + ph + 16, "text-anchor": "middle" }, v));
    svg.append(svgEl("text", { x: L + pw / 2, y: TOP + ph + 34, "text-anchor": "middle" }, "round t"));
    for (const m of marks) {
      svg.append(svgEl("line", { class: "vline", x1: X(m.t), x2: X(m.t), y1: TOP - 4, y2: TOP + ph + 4 }));
    }
    host.replaceChildren(svg);
  }

  return { line, heat, scatter, schedule };
})();
