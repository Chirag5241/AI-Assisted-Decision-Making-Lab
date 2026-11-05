# AI-Assisted Decision Making Lab (CS 598)

Research site for a CS 598 project: an algorithm decides which **features** to show and which **actions** to offer a
human who is still learning the true utilities, so the human picks the best move now and becomes competent later.
Repo: `github.com/Chirag5241/AI-Assisted-Decision-Making-Lab`, served on GitHub Pages from `main` (root, `.nojekyll`).

## The model every page simulates

- State `x_t ~ N(0, I)` (or `N(0, Σ)` on the correlated / sepsis pages), one draw per round, reproducible by `seed`.
- Truth `U` (K actions × n features); action k is worth `(U x)_k`. Best move `y* = argmax_k (U x)_k`.
- Human beliefs `H_t`; they pick `ŷ = argmax_{k∈A_t} Σ_{j∈F_t} h_kj x_j` (only shown features, only offered actions).
- Learning (`CountLearner`): `h_kj,t = u_kj + (h_kj,0 − u_kj)(1 − φ_j(N_kj(t)))`, N = rounds feature j was shown with
  action k offered. Hidden/unoffered weights stay flat. Speed belongs to the **feature** (`ps=` gives one per feature).
- Curves φ: exponential (rate), hyperbolic (half-life), power law (exponent), sigmoid (midpoint) — `CURVE_SPECS` in
  `webapp/app.py`, mirrored by `CURVES` in `static-api.js`.
- Loss: regret at the **drawn** state, `max_k (Ux_t)_k − (Ux_t)_ŷ`. Objective: `Σ δ^t ℓ_t`. The average over states
  (2,000 probe states, `default_rng(0)`) is always reported beside it as the `*_mean` series.

## Layout

| Path | What it is |
|---|---|
| `simlab/` | Pure numpy science. `sim.py` (batched loop, policies), `learners.py`, `curves.py`, `draws.py` (mulberry32 + Box–Muller, **must match JS bit-for-bit**), `analysis.py` (flip/settle metrics, `evaluate`, `realized`), `correlated.py` (Guan et al. conditional-mean fill-in), `planning.py` (2-action DP over counts), `one_feature.py`, `losses.py` (regret alternatives, Jiang et al. 2014), `sepsis.py`, `plotting.py` (matplotlib style) |
| `webapp/app.py` | Flask site. All numbers computed here; routes + `/api/run`, `/api/world`, `/api/correlated`. `lru_cache`d runners (`run_world`, `run_general`, `run_correlated`, `_*_study`) |
| `webapp/guide.py` | Single source of truth for `EQUATIONS`, `STANDARD_FIGURES` (6 figures: schedule, beliefs, choice, regret, value, accuracy), `BENCHES`, `NOTES`, `MENU`. Titles/formulas are HTML here and nowhere else |
| `webapp/templates/` | Jinja. `base.html` (masthead + page menu from `MENU`), `lab.html`, `worlds.html` (one template for 4 benches via `mode`), `correlated.html`, `overview.html`, `_figures.html`, note pages |
| `webapp/static/` | Browser code only places marks: `charts.js` (SVG primitives), `figures.js` (the standard figures), `matrix.js` / `speeds.js` (editors), one script per page, `static-api.js` (JS port of the API), `probes.js` (generated), `style.css` |
| `*.html` at repo root | **Generated** by `build_pages.py` for GitHub Pages. Never hand-edit; edit the template and rebuild |
| `distributions/` | Separate hand-written static companion site (35 distributions: explorer + map) |
| `experiments/` | `exp01_single_feature_flip.py` (matplotlib → `results/exp01/`), `sepsis_data.py` (PhysioNet 2019 loader), `sepsis_fit.py` (writes `results/sepsis/fit.json`, `study.json`) |
| `results/` | Committed outputs the site reads. `data/` (raw PhysioNet) is gitignored |
| `theory/` | Obsidian notes (math), synced from a teammate's vault by `sync_theory.sh` (`rsync --delete` — it overwrites local edits). `<G>`/`<C>` prefixes are authorship tags; `Archive/` is superseded |
| `share/decision-lab.html` | Old single-file shareable build |
| `tests/` | pytest; `test_static_api.py` runs `static-api.js` under Node and diffs it against Flask |

## Pages (route → template → bench mode)

Benches (from `guide.BENCHES`): `/` 1 feature, 2 actions (`lab.html`, flip + settle map) · `/worlds` More actions
and features (mode `fixed`) · `/timed` Timed hiding (`W=f3:20-25;a1:40-60`) · `/planned` Planned subsets (K=2,
n≤3, T≤100; optimal DP vs myopic) · `/aware` State-aware subsets (budget `C`, `judge=belief|truth`,
`prefer=margin|most`) · `/correlated` Correlated features (3 features, ρ triple, explore-then-commit).
Notes (from `guide.NOTES`, placed before a bench in the menu): `/two-actions` (no figures), `/losses`,
`/one-feature`, `/sepsis`. Also `/overview` (figures + equations, live specimens) and `/experiments` (exp01 PNGs).

Study pages (`/losses`, `/one-feature`, `/sepsis`) compute everything once in Python and inject it as
`window.LOSS_STUDY` / `ONE_FEATURE` / `SEPSIS` via `tojson`, so the static build bakes the numbers in; they need no
JS port. Bench pages fetch `/api/*` live as controls move (`rank=0` = fast path without ranking tables).
Bench state is remembered per page in `sessionStorage` (`decision-lab:<page>`) and shareable via query string.

## Commands

```bash
python3 -m webapp.app            # Flask site at http://127.0.0.1:5050 (launch.json: "decision-lab")
python3 build_pages.py           # regenerate root *.html + webapp/static/probes.js for GitHub Pages
python3 -m http.server 5051      # preview the static build (launch.json: "pages-preview")
python3 -m pytest                # all tests; static-API parity tests need `node` and a built probes.js
python3 -m experiments.exp01_single_feature_flip
python3 -m experiments.sepsis_fit --study   # rerun the sepsis study from the saved fit
```

Deps: numpy, scipy, matplotlib, flask, pytest (scikit-learn only for the sepsis fit). No requirements file.

## Rules that keep the two sites in sync

1. **Change `app.py` API logic → change `static-api.js` identically** (same JSON, same strings, same rounding).
   Add the query to `QUERIES` in `tests/test_static_api.py`. Random draws go through `simlab/draws.py` /
   `uniform32` so Python and JS produce the same states; the aware fallback uses stream `seed + AWARE_STREAM`.
2. **Change a template, static file, or `guide.py` → run `build_pages.py` and commit the regenerated root
   `*.html` and `probes.js`.** New route? Add it to `PAGES` in `build_pages.py` (and `guide.BENCHES`/`NOTES`).
   Static links must be relative; the build asserts no `href="/` remains.
3. Static assets are cache-busted by mtime (`static_v()` in Flask, `versioned()` in the build).
4. Site name/kicker live only in `SITE_NAME` / `SITE_KICKER`; equations and figure titles only in `guide.py`.

## Conventions

- **Fixed chart axes, never auto-scaled.** Derive limits from parameter bounds (sliders are ±3) or from the world
  alone (`regret_cap`, `value_cap` from U and H0), so axes stay put while policies/learners/draws change.
- **Terse pages:** figures driven by simulation with one-line captions; no essay sections unless asked.
- Python computes every number; JS only draws. Server-built sentences (`sentence`, `stats`) are the page's prose.
- Code style: plain-English docstrings and comments in the project's voice ("the human", "the bench", "on this
  draw"); vectorised numpy over batch dim B; shapes noted in comments (`(T, B, n)`); ties broken by 1e-9 / 1e-12
  thresholds so rounding never decides.
- Visual design: paper/ink lab-notebook look; tokens on `:root` in `style.css` (`--a1..--a6` action colours,
  Fraunces / Newsreader / IBM Plex Mono). Action colours match line colours across figures.
- Limits: `MAX_ACTIONS = MAX_FEATURES = 6`, `N_PROBES = 2000`, `MAX_WINDOWS = 12`, `MAX_SEED = 999`, T ≤ 300.
