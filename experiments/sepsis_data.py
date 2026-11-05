"""Read the PhysioNet / CinC Challenge 2019 training data into one row per patient.

The data (40,336 ICU stays, one row per hour, CC-BY 4.0) is not in the repository. Fetch it once with

    aws s3 sync --no-sign-request s3://physionet-open/challenge-2019/1.0.0/training/ data/physionet2019/raw/

Each patient gives one decision point. A septic patient's is the first hour labelled 1, six hours before onset:
the last hour a decision is on time. A patient who never turns septic gives an hour drawn from the septic
patients' decision hours (cut to the stay), so time in the ICU does not tell the two apart. Each variable is its
last value at or before that hour (missing if never measured by then).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

RAW = Path(__file__).resolve().parent.parent / "data" / "physionet2019" / "raw"

VITALS = ["HR", "O2Sat", "Temp", "SBP", "MAP", "DBP", "Resp", "EtCO2"]
LABS = ["BaseExcess", "HCO3", "FiO2", "pH", "PaCO2", "SaO2", "AST", "BUN", "Alkalinephos", "Calcium", "Chloride",
        "Creatinine", "Bilirubin_direct", "Glucose", "Lactate", "Magnesium", "Phosphate", "Potassium",
        "Bilirubin_total", "TroponinI", "Hct", "Hgb", "PTT", "WBC", "Fibrinogen", "Platelets"]
COLUMNS = VITALS + LABS + ["Age", "Gender", "Unit1", "Unit2", "HospAdmTime", "ICULOS", "SepsisLabel"]
# the demographics that join the vitals in s; ICULOS is left out (the decision hour sets it) and so are the unit
# flags (administrative, missing for a third of the patients)
DEMOGRAPHICS = ["Age", "Gender", "HospAdmTime"]


def read(path):
    """(hours, 41) float array of one patient's file; NaN where nothing was measured."""
    with open(path) as f:
        header = f.readline().strip().split("|")
        assert header == COLUMNS, path
        return np.array([[float(v) for v in line.strip().split("|")] for line in f if line.strip()])


def carried(values):
    """Each column's last measured value at or before each hour."""
    out = values.copy()
    for t in range(1, len(out)):
        gap = np.isnan(out[t])
        out[t, gap] = out[t - 1, gap]
    return out


def snapshots(seed=0):
    """One row per patient at their decision hour. Returns (values (N, 41) with the clinical variables carried
    forward, septic (N,) bool, hospital (N,) 'A' or 'B', hour (N,))."""
    stays, hospital = [], []
    for setname, tag in (("training_setA", "A"), ("training_setB", "B")):
        for path in sorted((RAW / setname).glob("*.psv")):
            stays.append(read(path))
            hospital.append(tag)
    septic = np.array([(v[:, -1] == 1).any() for v in stays])
    onset = [int(np.argmax(v[:, -1] == 1)) for v, s in zip(stays, septic) if s]
    rng = np.random.default_rng(seed)
    hour = np.array([int(np.argmax(v[:, -1] == 1)) if s else min(int(rng.choice(onset)), len(v) - 1)
                     for v, s in zip(stays, septic)])
    rows = np.array([carried(v[: t + 1])[-1] for v, t in zip(stays, hour)])
    return rows, septic, np.array(hospital), hour
