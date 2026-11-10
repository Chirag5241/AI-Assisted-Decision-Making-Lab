"""Experiment 4: fit the weather world (rain tomorrow, Australian stations) and measure symptom strength.

Data: weatherAUS (Australian Bureau of Meteorology daily observations, as packaged in R's rattle
package and on Kaggle as "Rain in Australia"). The data is not stored in this repo; download
weatherAUS.csv and pass its path.

Mapping to the model:
  symptoms s (no instruments): Cloud3pm, Rainfall (today), WindGustSpeed, Temp3pm
  tests F (instruments):       Humidity3pm, Pressure3pm, PressureChange (3pm - 9am), Sunshine,
                               Humidity9am, Evaporation
  truth U: logistic regression of RainTomorrow on normal scores of all ten (as in sepsis)

Reports: linear vs boosted AUROC (how wrong a linear U is), true weights, symptom strength
(share of Var(d.x) explained by the symptoms) and the value of adding each single test.

    python -m experiments.exp04_weather path/to/weatherAUS.csv
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm, rankdata
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

OUT = Path(__file__).resolve().parent.parent / "results" / "exp04"
SYM = ["Cloud3pm", "Rainfall", "WindGustSpeed", "Temp3pm"]
TEST = ["Humidity3pm", "Pressure3pm", "PressureChange", "Sunshine", "Humidity9am", "Evaporation"]


def explained(S, w, keep):
    Sk, Sxk = S[np.ix_(keep, keep)], S[:, keep]
    G = S - Sxk @ np.linalg.solve(Sk, Sxk.T)
    return float(1 - (w @ G @ w) / (w @ S @ w))


def main(path):
    df = pd.read_csv(path)
    df["PressureChange"] = df.Pressure3pm - df.Pressure9am
    cols = SYM + TEST
    d = df.dropna(subset=cols + ["RainTomorrow"])
    Z = np.column_stack([norm.ppf((rankdata(d[c]) - 0.5) / len(d)) for c in cols])
    y = (d.RainTomorrow == "Yes").to_numpy().astype(int)
    idx = np.random.default_rng(0).permutation(len(y))
    tr, te = idx[: int(0.7 * len(y))], idx[int(0.7 * len(y)):]
    lr = LogisticRegression(max_iter=2000).fit(Z[tr], y[tr])
    gb = HistGradientBoostingClassifier(random_state=0).fit(Z[tr], y[tr])
    sym_only = LogisticRegression(max_iter=2000).fit(Z[tr][:, : len(SYM)], y[tr])
    w = lr.coef_[0]
    S = np.cov(Z.T)
    ns = len(SYM)
    out = dict(
        rows=int(len(d)), stations=int(d.Location.nunique()), dates=[str(d.Date.min()), str(d.Date.max())],
        rain_rate=float(y.mean()),
        auroc_linear=float(roc_auc_score(y[te], lr.decision_function(Z[te]))),
        auroc_boosted=float(roc_auc_score(y[te], gb.predict_proba(Z[te])[:, 1])),
        auroc_symptoms_only=float(roc_auc_score(y[te], sym_only.decision_function(Z[te][:, :ns]))),
        weights={c: float(v) for c, v in zip(cols, w)},
        symptom_strength=explained(S, w, list(range(ns))),
        with_one_test={c: explained(S, w, list(range(ns)) + [j]) for j, c in enumerate(cols) if j >= ns},
        Sigma=S.tolist(), names=cols, symptoms=SYM, tests=TEST,
    )
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "fit.json").write_text(json.dumps(out, indent=1))
    show = {k: v for k, v in out.items() if k not in ("Sigma",)}
    print(json.dumps(show, indent=1))


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: python -m experiments.exp04_weather path/to/weatherAUS.csv")
    main(sys.argv[1])
