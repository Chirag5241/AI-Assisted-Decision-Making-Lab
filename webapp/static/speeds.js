// The speed rows of the "How the human learns" card on the benches with several features. Learning speed
// belongs to the feature, not the action: every feature shares the one slider (p1), or, with the box
// ticked, each feature has a slider of its own. The page owns the numbers in state.S.
window.Speeds = (function () {
  const $ = (id) => document.getElementById(id);
  const SUB = ["₁", "₂", "₃", "₄", "₅", "₆"];
  const clamp = (v, lo, hi) => Math.min(Math.max(v, lo), hi);

  // opts: {state, count(), spec(), shared(), onChange(), onToggle()}
  //   state.S   null while the features share one speed, otherwise one speed per feature
  //   count()   how many features there are; spec() the chosen curve's parameter; shared() the value of p1
  //   onChange  a slider is moving; onToggle the box was ticked or cleared
  function attach(opts) {
    const state = opts.state, box = $("each-speed"), host = $("speed-rows"), sharedRow = $("p1").closest(".row");

    function draw() {
      const spec = opts.spec();
      box.closest("label").hidden = opts.count() < 2;     // one feature has one curve, whatever the actions
      box.checked = Boolean(state.S);
      sharedRow.hidden = Boolean(state.S);
      host.replaceChildren(...(state.S || []).map((speed, j) => {
        const row = document.createElement("div"), label = document.createElement("label"), name = document.createElement("span");
        const slider = document.createElement("input"), exact = document.createElement("input");
        row.className = "row";
        name.className = "sym";
        name.textContent = "x" + SUB[j];
        label.append(name);
        for (const input of [slider, exact]) { input.min = spec.min; input.max = spec.max; input.step = spec.step; input.value = speed; }
        slider.type = "range";
        slider.id = "speed-" + j;
        label.htmlFor = slider.id;
        slider.setAttribute("aria-label", `${spec.param} for feature ${j + 1}`);
        exact.type = "number";
        exact.setAttribute("aria-label", `${spec.param} for feature ${j + 1}, exact value`);
        const set = (from, other) => {
          const v = parseFloat(from.value);
          if (isNaN(v)) return;
          state.S[j] = clamp(v, spec.min, spec.max);
          other.value = state.S[j];
          opts.onChange();
        };
        slider.addEventListener("input", () => set(slider, exact));
        exact.addEventListener("input", () => set(exact, slider));
        exact.addEventListener("change", () => { exact.value = state.S[j]; });
        row.append(label, slider, exact);
        return row;
      }));
    }

    box.addEventListener("change", () => {
      state.S = box.checked ? Array(opts.count()).fill(opts.shared()) : null;
      draw();
      opts.onToggle();
    });

    return {
      draw,
      // a new curve: every speed back to that curve's default
      reset() { if (state.S) state.S = state.S.map(() => opts.spec().default); draw(); },
      // the world grew or shrank: new features start at the curve's default
      fit() {
        const n = opts.count();
        if (state.S) state.S = n < 2 ? null : Array.from({ length: n }, (_, j) => (state.S[j] !== undefined ? state.S[j] : opts.spec().default));
        draw();
      },
      // "0.05,0.1,0.2" from a shared link, if it names one speed per feature
      load(text) {
        const spec = opts.spec(), speeds = typeof text === "string" && text ? text.split(",").map(Number) : [];
        state.S = speeds.length === opts.count() && speeds.length > 1 && speeds.every(Number.isFinite)
          ? speeds.map((v) => clamp(v, spec.min, spec.max)) : null;
        draw();
      },
      // what to send as ps, or null while the features share p1
      param: () => (state.S ? state.S.join(",") : null),
    };
  }

  return { attach };
})();
