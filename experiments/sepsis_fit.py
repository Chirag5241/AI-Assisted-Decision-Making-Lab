"""Fit Sigma, B and U of the two-action problem to the PhysioNet 2019 sepsis data, then run the study.

    python -m experiments.sepsis_fit            # fit (needs the raw data, see experiments/sepsis_data.py), then study
    python -m experiments.sepsis_fit --study    # the study only, from the saved fit

Writes results/sepsis/fit.json and results/sepsis/study.json, which the /sepsis page reads; the held-out patients
go to data/physionet2019/derived/ (not committed).

The fit, in order:
  1. one decision point per patient (sepsis_data.snapshots); keep the labs (x) and the vitals and demographics
     (s, the symptoms) measured by then in at least MIN_COVERAGE of the patients;
  2. risk scores: each variable's measured values cut into BINS quantile bins, each bin given its (smoothed)
     log-odds of sepsis, and those mapped through their empirical distribution to N(0, 1), all on the training
     patients. So x_j is "how septic lab j looks", monotone in risk even where the raw value is U-shaped
     (temperature, WBC), and revealing lab j reveals x_j;
  3. Sigma by EM for a Gaussian with missing values, rescaled to a correlation, so z = (x, s) ~ N(0, Sigma): the
     map from the labs to the symptoms, noise and all, is in Sigma_sx and Sigma_ss;
  4. every missing value filled in with its conditional mean given what was measured;
  5. U: an L2 logistic regression of sepsis on the labs x alone, the penalty picked by 5-fold cross-validated
     AUROC, so the right call is the one made knowing every lab. With two
     actions only d = u_stay - u_home counts: d is the fitted weight vector and d0 the intercept less the
     log-odds of the treatment threshold tau, so stay is best exactly when the fitted risk is above tau.
     tau is the risk of the average patient (x = 0), so d0 = 0 as in the theory: treating is worth it when a
     miss costs (1 - tau) / tau false alarms, about 16 here (the challenge's own scoring says 40).
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import warnings
from pathlib import Path

import numpy as np
from scipy.special import expit
from scipy.stats import norm

from simlab import sepsis

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "sepsis"
DERIVED = ROOT / "data" / "physionet2019" / "derived"

MIN_COVERAGE = 0.2       # a variable measured by the decision hour in fewer patients than this is left out
BINS, SMOOTH = 20, 20    # risk-score bins per variable, and the prior weight (in patients) pulling a bin to the base rate
TEST_SHARE = 0.2

# the study
BUDGETS = [1, 2, 3]
T = 40
RATE = 0.8               # psi(m) = RATE^m: each showing closes a fifth of the gap
HUMANS = 64              # random first beliefs, p0 ~ N(0, s^2 I) with s the root-mean-square of d
POLICY_NAMES = {"self": "No teacher: the human's own panel", "random": "Random panel",
                "expert": "The expert's panel, every round", "teacher": "Teacher: fix what graduation needs"}
FILL_NAMES = {"symptoms": "Hidden labs filled in from the symptoms", "shown": "Hidden labs filled in from the symptoms and the shown labs"}


def normal_scores(train, values):
    """Map `values` through the empirical distribution of `train` (its measured values) to N(0, 1); ties share
    their mid-rank. NaN stays NaN."""
    ref = np.sort(train[~np.isnan(train)])
    rank = (np.searchsorted(ref, values, "left") + np.searchsorted(ref, values, "right")) / 2.0
    return np.where(np.isnan(values), np.nan, norm.ppf((rank + 0.5) / (len(ref) + 1.0)))


def risk_scores(train, septic, values):
    """Each value's bin log-odds of sepsis, the bins being quantiles of `train` (measured values, with labels
    `septic`); NaN stays NaN. Also returns the bin edges and log-odds."""
    ok = ~np.isnan(train)
    edges = np.unique(np.quantile(train[ok], np.linspace(0, 1, BINS + 1)[1:-1]))
    which = np.searchsorted(edges, train[ok])
    base = septic.mean()
    rate = np.array([(septic[ok][which == k].sum() + SMOOTH * base) / ((which == k).sum() + SMOOTH)
                     for k in range(len(edges) + 1)])
    odds = np.log(rate / (1 - rate))
    out = np.full(len(values), np.nan)
    m = ~np.isnan(values)
    out[m] = odds[np.searchsorted(edges, values[m])]
    return out, edges, odds


def em(Z, iters=60, tol=1e-6):
    """Mean and covariance of a Gaussian from rows with missing values (NaN), by EM."""
    N, n = Z.shape
    miss = np.isnan(Z)
    mu, S = np.nanmean(Z, axis=0), np.diag(np.nanvar(Z, axis=0))
    patterns, which = np.unique(miss, axis=0, return_inverse=True)
    which = which.ravel()
    for _ in range(iters):
        total, outer = np.zeros(n), np.zeros((n, n))
        for k, pattern in enumerate(patterns):
            rows = Z[which == k]
            o, m = ~pattern, pattern
            filled = rows.copy()
            if m.any():
                if o.any():
                    B = S[np.ix_(m, o)] @ np.linalg.inv(S[np.ix_(o, o)])
                    filled[:, m] = mu[m] + (rows[:, o] - mu[o]) @ B.T
                    C = S[np.ix_(m, m)] - B @ S[np.ix_(o, m)]
                else:
                    filled[:, m] = mu[m]
                    C = S[np.ix_(m, m)]
                outer[np.ix_(m, m)] += len(rows) * C
            total += filled.sum(axis=0)
            outer += filled.T @ filled
        new_mu = total / N
        new_S = outer / N - np.outer(new_mu, new_mu)
        done = np.abs(new_S - S).max() < tol
        mu, S = new_mu, new_S
        if done:
            break
    return mu, S


def fill(X, Sigma):
    """Each missing value replaced by its conditional mean given the measured ones, under x ~ N(0, Sigma)."""
    X = X.copy()
    for i in range(len(X)):
        m = np.isnan(X[i])
        if m.any():
            o = ~m
            X[i, m] = Sigma[np.ix_(m, o)] @ np.linalg.solve(Sigma[np.ix_(o, o)], X[i, o]) if o.any() else 0.0
    return X


def roc(y, score, points=101):
    """AUROC and the curve at `points` evenly spaced false-positive rates."""
    from sklearn.metrics import roc_auc_score, roc_curve
    fpr, tpr, _ = roc_curve(y, score)
    grid = np.linspace(0, 1, points)
    return float(roc_auc_score(y, score)), np.round(np.interp(grid, fpr, tpr), 4).tolist()


def logistic(X, y):
    from sklearn.linear_model import LogisticRegressionCV
    return LogisticRegressionCV(Cs=np.logspace(-3, 1, 9), cv=5, scoring="roc_auc", max_iter=2000).fit(X, y)


def fit(seed=0):
    from sklearn.ensemble import HistGradientBoostingClassifier

    from . import sepsis_data
    values, septic, hospital, hour = sepsis_data.snapshots(seed)
    col = {c: i for i, c in enumerate(sepsis_data.COLUMNS)}
    every = sepsis_data.LABS + sepsis_data.VITALS + sepsis_data.DEMOGRAPHICS
    coverage = {c: float((~np.isnan(values[:, col[c]])).mean()) for c in every}
    labs, vitals, demographics = ([c for c in group if coverage[c] >= MIN_COVERAGE]
                                  for group in (sepsis_data.LABS, sepsis_data.VITALS, sepsis_data.DEMOGRAPHICS))
    names = labs + vitals + demographics
    raw = values[:, [col[c] for c in names]]

    rng = np.random.default_rng(seed)
    test = np.zeros(len(raw), bool)
    test[rng.permutation(len(raw))[: int(TEST_SHARE * len(raw))]] = True
    train = ~test

    y = septic.astype(int)
    risk, bins = [], {}
    for j, name in enumerate(names):
        r, edges, odds = risk_scores(raw[train, j], y[train], raw[:, j])
        risk.append(r)
        bins[name] = dict(edges=np.round(edges, 4).tolist(), log_odds=np.round(odds, 4).tolist())
    risk = np.column_stack(risk)
    Z = np.column_stack([normal_scores(risk[train, j], risk[:, j]) for j in range(len(names))])
    mu, S = em(Z[train])
    sd = np.sqrt(np.diag(S))
    X = (Z - mu) / sd                                   # z ~ N(0, Sigma) with unit variances
    Sigma = S / np.outer(sd, sd)
    Xf = fill(X, Sigma)                                 # a missing lab filled in from every measured variable

    nL = len(labs)
    L = slice(0, nL)
    full = logistic(Xf[train][:, L], y[train])          # U: the labs alone
    w, b = full.coef_[0], float(full.intercept_[0])
    models = {"labs": (full, L),
              "symptoms": (logistic(Xf[train][:, nL:], y[train]), slice(nL, None)),
              "all": (logistic(Xf[train], y[train]), slice(None))}
    scores = {key: roc(y[test], m.decision_function(Xf[test][:, cols])) for key, (m, cols) in models.items()}
    gbm = HistGradientBoostingClassifier(max_iter=400, learning_rate=0.05, early_stopping=True, random_state=seed)
    gbm.fit(raw[train][:, L], y[train])
    scores["gbm"] = roc(y[test], gbm.predict_proba(raw[test][:, L])[:, 1])
    # the same, given the labs with the gaps filled, as the linear model is: what is left is the cost of linearity
    gbm.fit(Xf[train][:, L], y[train])
    scores["gbm_filled"] = roc(y[test], gbm.predict_proba(Xf[test][:, L])[:, 1])
    # the same linear model fitted in one hospital system and scored in the other
    a, bset = hospital == "A", hospital == "B"
    transfer = {}
    for src, dst, key in ((a, bset, "A_to_B"), (bset, a, "B_to_A")):
        m = logistic(Xf[src][:, L], y[src])
        transfer[key] = roc(y[dst], m.decision_function(Xf[dst][:, L]))[0]

    DERIVED.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(DERIVED / "test.npz", X=Xf[test].astype(np.float32), measured=~np.isnan(X[test]), septic=septic[test])
    out = dict(
        source="PhysioNet/CinC Challenge 2019 training sets A and B (CC-BY 4.0)",
        patients=int(len(raw)), train=int(train.sum()), test=int(test.sum()),
        prevalence=float(septic.mean()), prevalence_A=float(septic[a].mean()), prevalence_B=float(septic[bset].mean()),
        onset_at_admission=float((hour[septic] == 0).mean()),
        coverage={c: dict(all=coverage[c], septic=float((~np.isnan(values[septic, col[c]])).mean()),
                          other=float((~np.isnan(values[~septic, col[c]])).mean()))
                  for c in every},
        labs=labs, vitals=vitals, demographics=demographics, symptoms=vitals + demographics,
        min_coverage=MIN_COVERAGE, bins=bins,
        Sigma=np.round(Sigma, 6).tolist(), d=np.round(w, 6).tolist(), d0=0.0,
        intercept=b, tau=float(expit(b)), cost_miss=float(np.exp(-b)), penalty_C=float(full.C_[0]),
        auroc={key: v[0] for key, v in scores.items()}, roc={key: v[1] for key, v in scores.items()}, transfer=transfer)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "fit.json").write_text(json.dumps(out, indent=1))
    return out


def clinical(world, q, X, septic, cost_miss):
    """On real patients: sensitivity, specificity, and the cost per patient in false alarms (a miss costs
    `cost_miss`) of staying when q.x > 0."""
    stay = sepsis.decisions(q, X)
    return dict(sensitivity=float(stay[septic].mean()), specificity=float((~stay[~septic]).mean()),
                cost=float((cost_miss * (septic & ~stay) + (~septic & stay)).mean()))


def study(fit_out, seed=0):
    """The teaching study, once for each way of filling in the hidden labs (FILL_NAMES)."""
    base = sepsis.World.from_fit(fit_out)
    held = np.load(DERIVED / "test.npz")
    Xr, septic = held["X"].astype(float), held["septic"]
    rng = np.random.default_rng(seed + 1)
    s = float(np.sqrt(np.mean(base.d ** 2)))
    nL = base.n_labs
    starts = s * rng.standard_normal((HUMANS, nL))
    miss = fit_out["cost_miss"]

    out = dict(T=T, rate=RATE, humans=HUMANS, budgets=BUDGETS, policies=POLICY_NAMES, fills=FILL_NAMES, belief_scale=s,
               names=base.names, n_labs=nL, A=np.round(base.A, 4).tolist(),
               explained=np.round(1 - np.diag(base.Gamma), 4).tolist(), by_fill={})
    # references, the same under either fill-in: the truth with every lab, with the symptoms alone, and guessing
    every = np.ones((1, nL), bool)
    none = np.zeros((1, nL), bool)
    out["reference"] = dict(
        every_lab=float(sepsis.regret(base, sepsis.effective(base, base.d, every))[0]),
        symptoms_only=float(sepsis.regret(base, sepsis.effective(base, base.d, none))[0]),
        guess=float(sepsis.regret(base, np.zeros(len(base.names)))[0]),
        clinical_every_lab=clinical(base, base.D, Xr, septic, miss),
        clinical_symptoms_only=clinical(base, sepsis.effective(base, base.d, none)[0], Xr, septic, miss))
    for fill in FILL_NAMES:
        world = dataclasses.replace(base, fill=fill)
        out["by_fill"][fill] = {}
        for c in BUDGETS:
            masks = sepsis.panels(nL, c)
            expert = sepsis.chosen(world, world.d, masks)
            q_expert = sepsis.effective(world, world.d, expert)
            # with the true weights, the panel of least regret: the expert's only when shown labs fill in
            every_panel = sepsis.regret(world, sepsis.effective(world, world.d, masks))
            best = masks[int(every_panel.argmin())]
            entry = dict(best=np.flatnonzero(best).tolist(), best_regret=float(every_panel.min()),
                         expert=np.flatnonzero(expert).tolist(),
                         expert_regret=float(sepsis.regret(world, q_expert)[0]),
                         clinical_expert=clinical(world, q_expert[0], Xr, septic, miss),
                         untrained=float(np.mean([sepsis.graduation(world, p0, masks) for p0 in starts])), policies={})
            for key, policy in sepsis.POLICIES.items():
                runs = [sepsis.train(world, p0, policy, T, c, RATE, seed=h) for h, p0 in enumerate(starts)]
                shown = np.array([r[0] for r in runs])                      # (humans, T, labs)
                during = np.array([r[1] for r in runs])
                after = np.array([r[2] for r in runs])
                # graduation, on real patients: each human's weights after T rounds and their own panel
                final = []
                for h, p0 in enumerate(starts):
                    p = sepsis.believed(world, p0, shown[h].sum(axis=0), RATE)
                    final.append(clinical(world, sepsis.effective(world, p, sepsis.chosen(world, p, masks))[0], Xr, septic, miss))
                entry["policies"][key] = dict(
                    during=np.round(during.mean(axis=0), 4).tolist(), after=np.round(after.mean(axis=0), 4).tolist(),
                    example=shown[0].T.astype(int).tolist(),                 # (labs, T) for the first human
                    clinical={k: float(np.mean([f[k] for f in final])) for k in final[0]})
            out["by_fill"][fill][str(c)] = entry
    (OUT / "study.json").write_text(json.dumps(out))
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--study", action="store_true", help="skip the fit and use results/sepsis/fit.json")
    args = parser.parse_args()
    warnings.filterwarnings("ignore")
    fitted = json.loads((OUT / "fit.json").read_text()) if args.study else fit()
    print({k: fitted[k] for k in ("patients", "prevalence", "auroc", "transfer", "tau", "labs")})
    study(fitted)
    print("wrote", OUT / "study.json")
