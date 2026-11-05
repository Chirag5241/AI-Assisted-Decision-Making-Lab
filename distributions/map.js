// The map: every distribution as a card with its shape, every known relationship as a labelled arrow.
(function () {
  "use strict";
  const { list, byId, edges, edgeTypes, defaults } = DIST;
  const $ = (id) => document.getElementById(id);
  const COLOR = { discrete: "#2a78d6", continuous: "#eb6834" };
  const NW = 150, NH = 92;
  const esc = (s) => String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;");

  // ---------- layout ----------
  // Grid cells [column, row], found by an annealing search that keeps arrows short, few crossings,
  // and no arrow through a card; then fixed here so the map never moves.
  const LAYOUT = {
    arcsine: [0, 0], erlang: [1, 0], exponential: [2, 0], weibull: [4, 0], rayleigh: [5, 0], chi: [6, 0], maxwell: [7, 0],
    powerfn: [0, 1], F: [4, 1], chi2: [5, 1], cauchy: [6, 1], halfnormal: [7, 1],
    pareto: [0, 2], laplace: [1, 2], beta: [3, 2], gamma: [4, 2], t: [5, 2], normal: [6, 2], lognormal: [7, 2],
    triangular: [0, 3], uniform: [2, 3], betabinom: [4, 3], levy: [7, 3],
    logistic: [0, 4], geometric: [2, 4], discUniform: [4, 4], binomial: [5, 4], poisson: [6, 4], invgamma: [7, 4],
    loglogistic: [0, 5], gumbel: [1, 5], negbin: [2, 5], bernoulli: [4, 5], hypergeom: [5, 5], skellam: [6, 5],
  };
  const CW = 205, RH = 150, PAD = 100;
  const at = (id) => ({ x: PAD + LAYOUT[id][0] * CW, y: PAD - 20 + LAYOUT[id][1] * RH });
  const W = PAD * 2 + 7 * CW, H = PAD * 2 - 40 + 5 * RH;

  // where the ray from a card's centre towards (tx, ty) leaves the card
  function exit(c, tx, ty, pad) {
    const dx = tx - c.x, dy = ty - c.y, hw = NW / 2 + pad, hh = NH / 2 + pad;
    const s = Math.min(Math.abs(dx) > 1e-9 ? hw / Math.abs(dx) : Infinity, Math.abs(dy) > 1e-9 ? hh / Math.abs(dy) : Infinity);
    return { x: c.x + dx * s, y: c.y + dy * s };
  }
  const bez = (a, c, b, t) => ({ x: (1 - t) ** 2 * a.x + 2 * (1 - t) * t * c.x + t * t * b.x, y: (1 - t) ** 2 * a.y + 2 * (1 - t) * t * c.y + t * t * b.y });
  function hitsCard(a, c, b, skip) {
    for (const d of list) {
      if (skip.includes(d.id)) continue;
      const n = at(d.id);
      for (let t = 0.04; t < 0.97; t += 0.04) {
        const q = bez(a, c, b, t);
        if (Math.abs(q.x - n.x) < NW / 2 + 6 && Math.abs(q.y - n.y) < NH / 2 + 6) return true;
      }
    }
    return false;
  }
  // bend each arrow just enough to clear the other cards; a pair of opposite arrows bows apart
  const ROUTES = edges.map((e) => {
    const a = at(e.from), b = at(e.to), mx = (a.x + b.x) / 2, my = (a.y + b.y) / 2;
    const len = Math.hypot(b.x - a.x, b.y - a.y), nx = -(b.y - a.y) / len, ny = (b.x - a.x) / len;
    const paired = edges.some((f) => f.from === e.to && f.to === e.from);
    const bends = paired ? [0.18, 0.3, 0.45] : [0, 0.12, -0.12, 0.22, -0.22, 0.34, -0.34, 0.48, -0.48, 0.65, -0.65];
    let c = { x: mx, y: my };
    for (const k of bends) {
      const cc = { x: mx + nx * k * len, y: my + ny * k * len };
      c = cc;
      if (!hitsCard(a, cc, b, [e.from, e.to])) break;
    }
    const p0 = exit(a, c.x, c.y, 0), p1 = exit(b, c.x, c.y, 3);
    return { p0, c, p1, mid: bez(p0, c, p1, 0.5) };
  });

  // a small picture of the shape at its default parameters, on its own fixed axes
  function spark(d, x0, y0, w, h) {
    const p = defaults(d);
    d.fix && d.fix(p);
    const col = COLOR[d.kind];
    if (d.kind === "discrete") {
      const k0 = d.x[0], k1 = d.x[1], span = k1 - k0 + 1, bw = Math.max(1.2, w / span * 0.55);
      let s = "";
      for (let k = k0; k <= k1; k++) {
        const v = Math.min(d.pdf(k, p) / d.y, 1);
        if (v <= 0.002) continue;
        const cx = x0 + (k - k0 + 0.5) / span * w;
        s += `<rect x="${(cx - bw / 2).toFixed(1)}" y="${(y0 + h - v * h).toFixed(1)}" width="${bw.toFixed(1)}" height="${(v * h).toFixed(1)}" fill="${col}"/>`;
      }
      return s;
    }
    const N = 90;
    let path = `M${x0},${y0 + h}`;
    for (let i = 0; i <= N; i++) {
      const x = d.x[0] + (d.x[1] - d.x[0]) * i / N;
      let v = d.pdf(x, p);
      if (!Number.isFinite(v)) v = d.y;
      path += `L${(x0 + w * i / N).toFixed(1)},${(y0 + h - Math.min(v / d.y, 1) * h).toFixed(1)}`;
    }
    return `<path d="${path}L${x0 + w},${y0 + h}Z" fill="${col}" fill-opacity="0.75" stroke="${col}" stroke-width="1"/>`;
  }

  // ---------- draw ----------
  let svg = `<defs>`;
  svg += `<marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0,0L10,5L0,10z" fill="#52514e"/></marker>`;
  svg += `<marker id="arrow-hot" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0,0L10,5L0,10z" fill="#0b0b0b"/></marker>`;
  svg += `</defs><g id="edges">`;
  edges.forEach((e, i) => {
    const r = ROUTES[i], dash = edgeTypes[e.type].dash;
    svg += `<g class="edge" data-from="${e.from}" data-to="${e.to}"><title>${esc(byId[e.from].name)} → ${esc(byId[e.to].name)}: ${esc(e.label)}</title>` +
      `<path d="M${r.p0.x.toFixed(1)},${r.p0.y.toFixed(1)}Q${r.c.x.toFixed(1)},${r.c.y.toFixed(1)} ${r.p1.x.toFixed(1)},${r.p1.y.toFixed(1)}" ${dash ? `stroke-dasharray="${dash}"` : ""} marker-end="url(#arrow)"/>` +
      `<text x="${r.mid.x.toFixed(1)}" y="${(r.mid.y + 4).toFixed(1)}" text-anchor="middle">${esc(e.label)}</text></g>`;
  });
  svg += `</g><g id="nodes">`;
  for (const d of list) {
    const n = at(d.id), x = n.x - NW / 2, y = n.y - NH / 2;
    svg += `<g class="node" data-id="${d.id}" tabindex="0" role="button" aria-label="${esc(d.name)}">` +
      `<rect class="box" x="${x}" y="${y}" width="${NW}" height="${NH}" rx="3"/>` +
      `<text class="nm" x="${n.x}" y="${y + 18}" text-anchor="middle">${esc(d.name)}</text>` +
      `<line x1="${x + 10}" x2="${x + NW - 10}" y1="${y + NH - 10}" y2="${y + NH - 10}" stroke="#c9c8c1"/>` +
      spark(d, x + 10, y + 26, NW - 20, NH - 37) + `</g>`;
  }
  svg += `</g>`;
  const el = $("map");
  el.setAttribute("viewBox", `0 0 ${W} ${H}`);
  el.innerHTML = svg;

  // ---------- legend ----------
  const sample = (dash) => `<svg width="34" height="10" aria-hidden="true"><line x1="0" x2="34" y1="5" y2="5" stroke="#52514e" stroke-width="1.6" ${dash ? `stroke-dasharray="${dash}"` : ""}/></svg>`;
  $("legend").innerHTML =
    Object.values(edgeTypes).map((t) => `<span>${sample(t.dash)}${esc(t.name)}</span>`).join("") +
    `<span><i style="background:${COLOR.discrete}"></i>discrete</span><span><i style="background:${COLOR.continuous}"></i>continuous</span>`;

  // ---------- zoom ----------
  let zoom = 1;
  const scroll = $("scroll");
  function applyZoom() { el.setAttribute("width", Math.round(W * zoom)); el.setAttribute("height", Math.round(H * zoom)); }
  function fit() { zoom = Math.min(1, (scroll.clientWidth - 2) / W); applyZoom(); }
  $("zoom-in").addEventListener("click", () => { zoom = Math.min(2, zoom * 1.25); applyZoom(); });
  $("zoom-out").addEventListener("click", () => { zoom = Math.max(0.25, zoom / 1.25); applyZoom(); });
  $("zoom-fit").addEventListener("click", fit);
  $("all-labels").addEventListener("change", (ev) => el.classList.toggle("all-labels", ev.target.checked));
  fit();

  // ---------- selection ----------
  let selected = null;
  function highlight(id) {
    const nodes = el.querySelectorAll(".node"), eds = el.querySelectorAll(".edge");
    if (!id) {
      nodes.forEach((n) => n.classList.remove("dim", "sel"));
      eds.forEach((e) => { e.classList.remove("dim", "hot"); e.querySelector("path").setAttribute("marker-end", "url(#arrow)"); });
      return;
    }
    const near = new Set([id]);
    eds.forEach((e) => {
      const hot = e.dataset.from === id || e.dataset.to === id;
      if (hot) { near.add(e.dataset.from); near.add(e.dataset.to); }
      e.classList.toggle("hot", hot);
      e.classList.toggle("dim", !hot);
      e.querySelector("path").setAttribute("marker-end", hot ? "url(#arrow-hot)" : "url(#arrow)");
    });
    nodes.forEach((n) => { n.classList.toggle("dim", !near.has(n.dataset.id)); n.classList.toggle("sel", n.dataset.id === id); });
  }

  function panel(id) {
    const side = $("side");
    if (!id) {
      side.innerHTML = `<p class="kicker">How to read it</p><p class="empty">Each arrow says how to get one distribution from another: transform the draws, fix a parameter, push a parameter to infinity, or make a parameter random.</p><p class="empty">Hover or click a card to see its connections and their labels.</p>`;
      return;
    }
    const d = byId[id], out = edges.filter((e) => e.from === id), inn = edges.filter((e) => e.to === id);
    const li = (o, e, arrow) => `<li>${arrow} <a data-go="${o.id}">${esc(o.name)}</a><br><span class="how">${esc(e.label)} · ${edgeTypes[e.type].name.split(" (")[0].toLowerCase()}</span></li>`;
    side.innerHTML =
      `<p class="kicker"><span class="tag ${d.kind}">${d.kind}</span></p><h2>${esc(d.name)}</h2><p>${esc(d.story)}</p>` +
      `<svg viewBox="0 0 280 110" width="100%" aria-hidden="true" style="margin:0 0 12px"><line x1="0" x2="280" y1="104" y2="104" stroke="#c9c8c1"/>${spark(d, 0, 4, 280, 100)}</svg>` +
      (inn.length ? `<p class="kicker">Comes from</p><ul>${inn.map((e) => li(byId[e.from], e, "←")).join("")}</ul>` : "") +
      (out.length ? `<p class="kicker">Leads to</p><ul>${out.map((e) => li(byId[e.to], e, "→")).join("")}</ul>` : "") +
      `<a class="go" href="index.html#${d.id}">Open in the explorer →</a>`;
    side.querySelectorAll("[data-go]").forEach((a) => a.addEventListener("click", () => select(a.dataset.go, true)));
  }

  function select(id, reveal) {
    selected = id && byId[id] ? id : null;
    highlight(selected);
    panel(selected);
    if (selected) {
      history.replaceState(null, "", "#" + selected);
      if (reveal) {
        const n = at(selected);
        scroll.scrollTo({ left: n.x * zoom - scroll.clientWidth / 2, top: n.y * zoom - scroll.clientHeight / 2, behavior: "smooth" });
      }
    } else history.replaceState(null, "", location.pathname);
  }

  el.querySelectorAll(".node").forEach((n) => {
    n.addEventListener("click", (ev) => { ev.stopPropagation(); select(n.dataset.id === selected ? null : n.dataset.id); });
    n.addEventListener("keydown", (ev) => { if (ev.key === "Enter" || ev.key === " ") { ev.preventDefault(); select(n.dataset.id); } });
    n.addEventListener("mouseenter", () => { if (!selected) highlight(n.dataset.id); });
    n.addEventListener("mouseleave", () => { if (!selected) highlight(null); });
  });
  el.addEventListener("click", () => select(null));
  select(location.hash.slice(1), true);
})();
