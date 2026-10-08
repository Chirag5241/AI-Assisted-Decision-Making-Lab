// The Overview's specimens: the six standard figures, drawn live for one small world by the same
// code the benches use, so the page that explains a figure shows exactly what a bench will show.
(function () {
  const world = window.DEFAULT_WORLD;
  const rows = (M) => M.map((row) => row.join(",")).join(";");
  const query = new URLSearchParams({
    K: world.U.length, n: world.U[0].length, U: rows(world.U), H: rows(world.H), F: world.F.join(","), A: world.A.join(","),
    curve: world.curve, p1: world.p1, T: world.T, delta: world.delta, W: world.W, rank: 0,
  });

  fetch("/api/world?" + query.toString())
    .then((response) => {
      if (!response.ok) throw new Error("The server answered " + response.status);
      return response.json();
    })
    .then((data) => {
      // the same fixed axes as on the bench: regret by the truth alone, value gap by the truth and the first beliefs
      const top = data.regret_cap || 1, bound = 1.05 * (data.value_cap || 1);
      StandardFigures.draw(data, {
        regret: { lo: -0.04 * top, hi: 1.04 * top }, value: { lo: -bound, hi: bound },
        refName: "everything shown and offered",
      });
    })
    .catch((error) => {
      const note = document.getElementById("specimen-problem");
      note.textContent = "The specimens could not be drawn: " + error.message + ". Is the Python server still running?";
      note.hidden = false;
    });
})();
