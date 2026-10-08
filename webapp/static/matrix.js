// The weight-matrix editor of the larger-world benches: the matrix at a glance, then one slider per
// weight grouped by feature. The page owns the numbers in world[name]; this file draws and edits them.
window.MatrixEditor = (function () {
  const $ = (id) => document.getElementById(id);
  const SUB = ["₁", "₂", "₃", "₄", "₅", "₆"];
  // Each feature's panel and sliders use a slightly lighter shade of the action colours than the one before.
  const SHADES = ["100%", "88%", "76%", "66%", "57%", "50%"];
  const clamp = (v, lo, hi) => Math.min(Math.max(v, lo), hi);
  const shortNumber = (v) => String(Math.round(v * 100) / 100);
  const make = (tag, cls, text) => { const e = document.createElement(tag); if (cls) e.className = cls; if (text !== undefined) e.textContent = text; return e; };

  // One weight: its symbol, a slider, and a number box with up and down arrows (hold to repeat).
  function entryRow(world, name, k, j, active, onChange) {
    const what = `${name === "U" ? "true" : "believed"} weight of feature ${j + 1} for action ${k + 1}`;
    const row = make("div", "row entry e-a" + (k + 1) + (active ? "" : " off"));

    const label = make("label"), sym = make("span", "sym dot"), letter = make("span"), index = make("sub");
    index.textContent = `${k + 1}${j + 1}`;
    letter.append(name === "U" ? "u" : "h", index);
    sym.append(letter);
    label.append(sym);

    const slider = make("input");
    slider.type = "range"; slider.min = "-3"; slider.max = "3"; slider.step = "0.1";
    slider.id = `s${name}-${k}-${j}`;
    slider.value = world[name][k][j];
    slider.setAttribute("aria-label", what);
    label.htmlFor = slider.id;

    const cell = make("div", "cell"), input = make("input");
    input.type = "number"; input.step = "0.1"; input.min = "-3"; input.max = "3";
    input.id = `m${name}-${k}-${j}`;
    input.value = world[name][k][j];
    input.setAttribute("aria-label", what + ", exact value");
    cell.append(input);

    const set = (value, from) => {
      world[name][k][j] = clamp(value, -3, 3);
      if (from !== slider) slider.value = world[name][k][j];
      if (from !== input) input.value = world[name][k][j];
      $(`v${name}-${k}-${j}`).textContent = shortNumber(world[name][k][j]);   // the snapshot follows the slider
      onChange();
    };
    slider.addEventListener("input", () => set(parseFloat(slider.value), slider));
    input.addEventListener("input", () => { const v = parseFloat(input.value); if (!isNaN(v)) set(v, input); });
    input.addEventListener("change", () => { input.value = world[name][k][j]; });
    for (const dir of [1, -1]) {
      const arrow = make("button", dir > 0 ? "up" : "down");
      arrow.type = "button";
      arrow.tabIndex = -1;   // the arrow keys already step the focused slider or box
      arrow.setAttribute("aria-label", (dir > 0 ? "Raise the " : "Lower the ") + what);
      const step = () => set(Math.round((world[name][k][j] + 0.1 * dir) * 10) / 10, null);
      let wait = null, repeat = null;
      const stop = () => { clearTimeout(wait); clearInterval(repeat); };
      arrow.addEventListener("pointerdown", (event) => {
        event.preventDefault();
        stop();
        step();
        wait = setTimeout(() => { repeat = setInterval(step, 70); }, 350);
      });
      for (const end of ["pointerup", "pointerleave", "pointercancel"]) arrow.addEventListener(end, stop);
      cell.append(arrow);
    }
    row.append(label, slider, cell);
    return row;
  }

  // A K x n matrix at a glance: one row per action, one column per feature, in brackets.
  // values[k][j] are the numbers shown; ids, when given, name each value so a slider can update it.
  function grid(values, opts) {
    const n = values[0].length, columns = `repeat(${n}, minmax(0, 1fr))`;
    const box = make("div", "snapshot"), cols = make("div", "cols"), sides = make("div", "sides"), bracket = make("div", "bracket");
    box.classList.toggle("dense", n >= 5);   // five or six columns: smaller figures so neighbours stay apart
    box.setAttribute("role", "group");
    box.setAttribute("aria-label", opts.label);
    cols.style.gridTemplateColumns = columns;
    bracket.style.gridTemplateColumns = columns;
    const featureOn = opts.featureOn || (() => true), actionOn = opts.actionOn || (() => true);
    for (let j = 0; j < n; j++) cols.append(make("span", "head" + (featureOn(j) ? "" : " off"), (opts.colPrefix || "x") + SUB[j]));
    values.forEach((row, k) => {
      sides.append(make("span", "head" + (actionOn(k) ? "" : " off"), (opts.rowPrefix || "a") + SUB[k]));
      row.forEach((v, j) => {
        const value = make("span", "value" + (featureOn(j) && actionOn(k) ? "" : " off"), shortNumber(v));
        if (opts.id) value.id = opts.id(k, j);
        if (opts.onPick) { value.title = "Go to this weight's slider"; value.addEventListener("click", () => opts.onPick(k, j)); }
        bracket.append(value);
      });
    });
    box.append(make("span", ""), cols, sides, bracket);
    return box;
  }

  // One matrix: its snapshot, then its weights as sliders grouped by feature like the panels of the beliefs figure.
  // opts: {featureOn(j), actionOn(k), onChange()}
  function draw(host, world, name, opts) {
    const featureOn = opts.featureOn || (() => true), actionOn = opts.actionOn || (() => true);
    const snapshot = grid(world[name], {
      label: (name === "U" ? "The truth matrix U" : "The first-belief matrix H0") + ", rows are actions and columns are features",
      featureOn, actionOn, id: (k, j) => `v${name}-${k}-${j}`,
      onPick: (k, j) => { const slider = $(`s${name}-${k}-${j}`); slider.scrollIntoView({ block: "center" }); slider.focus(); },
    });
    host.replaceChildren(snapshot, ...world[name][0].map((_, j) => {
      const group = make("div", "entry-group"), head = make("p", "group-label", "weights on x" + SUB[j] + (featureOn(j) ? "" : "  (hidden)"));
      group.style.setProperty("--shade", SHADES[j]);
      group.append(head, ...world[name].map((_, k) => entryRow(world, name, k, j, featureOn(j) && actionOn(k), opts.onChange)));
      return group;
    }));
  }

  return { draw, grid, SUB, SHADES, shortNumber };
})();
