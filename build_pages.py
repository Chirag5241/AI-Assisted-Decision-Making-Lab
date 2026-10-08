"""Build the static copy of the site that GitHub Pages serves.

    python build_pages.py

GitHub Pages cannot run the Flask server, so this renders the site's own pages to index.html,
worlds.html and experiments.html at the repository root. They load the same stylesheet and
scripts as the Flask site (from webapp/static), so the two look and behave alike. Two extra
scripts stand in for the server:

  webapp/static/static-api.js   answers /api/run and /api/world in the browser
  webapp/static/probes.js       the probe states the Python site scores on (written here)

Run it again after changing a template, and commit the three pages and probes.js.
"""
from __future__ import annotations

import base64
from pathlib import Path

import numpy as np

from webapp.app import MAX_FEATURES, N_PROBES, app

ROOT = Path(__file__).resolve().parent
STATIC = ROOT / "webapp" / "static"
PAGES = {"/": "index.html", "/worlds": "worlds.html", "/experiments": "experiments.html"}
NOTE = "<!-- Built by build_pages.py from webapp/templates for GitHub Pages. Edit the template, then rebuild. -->\n"


def write_probes():
    """The same states app.py draws with numpy's default_rng(0), one block per feature count."""
    blocks = {n: base64.b64encode(np.random.default_rng(0).standard_normal((N_PROBES, n)).astype("<f4").tobytes()).decode()
              for n in range(1, MAX_FEATURES + 1)}
    body = ",\n".join(f'  {n}: "{data}"' for n, data in blocks.items())
    (STATIC / "probes.js").write_text(
        "// Probe states for the static build, base64 float32 (written by build_pages.py; do not edit).\n"
        f'(typeof window !== "undefined" ? window : globalThis).LAB_PROBES = {{\n{body},\n}};\n')


def versioned(name):
    return f"webapp/static/{name}?v={int((STATIC / name).stat().st_mtime)}"


def build():
    write_probes()
    app.config["STATIC_BUILD"] = True    # pages the static copy has no stand-in for drop out of the navigation
    client = app.test_client()
    for route, filename in PAGES.items():
        html = client.get(route).data.decode()
        # the Flask site is served from the server root; the static copy must work from any folder
        html = html.replace('href="/static/', 'href="webapp/static/').replace('src="/static/', 'src="webapp/static/')
        html = html.replace('src="/results/', 'src="results/')
        for target, page in PAGES.items():
            html = html.replace(f'<a href="{target}"', f'<a href="{page}"')
        # the stand-in for the server has to load before the page's own script asks for data
        stand_in = f'<script src="{versioned("static-api.js")}"></script>\n'
        html = html.replace('<script src="webapp/static/lab.js', stand_in + '<script src="webapp/static/lab.js')
        html = html.replace('<script src="webapp/static/worlds.js',
                            f'<script src="{versioned("probes.js")}"></script>\n' + stand_in + '<script src="webapp/static/worlds.js')
        assert 'href="/' not in html and 'src="/' not in html, f"{filename} still has a server-root link"
        (ROOT / filename).write_text(html.replace("<!doctype html>\n", "<!doctype html>\n" + NOTE, 1))
        print(f"wrote {filename} ({len(html) // 1024} KB)")


if __name__ == "__main__":
    build()
