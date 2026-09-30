"""PhysioNet/CinC Challenge 2012 (set-a) loader.

Each record is one ICU stay: ~48 h of irregularly sampled measurements of up to
37 temporal variables plus 4 static descriptors. We keep the raw observations
(time, variable, value) -- the event codec works on them directly -- and also
build a regular hourly grid (mean per hour + observation mask) for the patch
codecs.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np

STATIC_VARS = ["Age", "Gender", "Height", "ICUType"]
TS_VARS = [
    "Albumin", "ALP", "ALT", "AST", "Bilirubin", "BUN", "Cholesterol", "Creatinine",
    "DiasABP", "FiO2", "GCS", "Glucose", "HCO3", "HCT", "HR", "K", "Lactate", "Mg",
    "MAP", "MechVent", "Na", "NIDiasABP", "NIMAP", "NISysABP", "PaCO2", "PaO2", "pH",
    "Platelets", "RespRate", "SaO2", "SysABP", "Temp", "TroponinI", "TroponinT",
    "Urine", "WBC", "Weight",
]
TS_IDX = {v: i for i, v in enumerate(TS_VARS)}
ICU_TYPES = {1: "Coronary Care Unit", 2: "Cardiac Surgery Recovery Unit", 3: "Medical ICU", 4: "Surgical ICU"}
HOURS = 48

# Physiologically plausible ranges; values outside are recording errors and dropped.
VALID = {
    "pH": (6.5, 8.0), "Temp": (25, 45), "HR": (1, 300), "Weight": (20, 300), "Height": (100, 230),
    "DiasABP": (1, 300), "SysABP": (1, 300), "MAP": (1, 300), "NIDiasABP": (1, 300),
    "NISysABP": (1, 300), "NIMAP": (1, 300), "RespRate": (1, 100), "FiO2": (0.2, 1.0),
    "SaO2": (1, 100), "Urine": (0, 5000),
}


@dataclass
class Stay:
    record_id: str
    static: dict           # name -> float, NaN if missing
    obs: np.ndarray        # (n, 3) float32: minutes, var index, value; sorted by time
    death: int             # in-hospital mortality label
    split: str = "train"


def _minutes(hhmm: str) -> int:
    h, m = hhmm.split(":")
    return int(h) * 60 + int(m)


def _valid(name: str, v: float) -> bool:
    lo, hi = VALID.get(name, (-np.inf, np.inf))
    return lo <= v <= hi


def load_stays(root: str, limit: int | None = None) -> list[Stay]:
    outcomes = {}
    with open(os.path.join(root, "Outcomes-a.txt")) as f:
        next(f)
        for line in f:
            p = line.strip().split(",")
            outcomes[p[0]] = int(p[5])
    files = sorted(os.listdir(os.path.join(root, "set-a")))[:limit]
    stays = []
    for fn in files:
        rid = fn.split(".")[0]
        static = {k: np.nan for k in STATIC_VARS}
        obs = []
        with open(os.path.join(root, "set-a", fn)) as f:
            next(f)
            for line in f:
                t, name, val = line.strip().split(",")
                if not name or name == "RecordID":
                    continue
                v = float(val)
                if name in static:
                    if t == "00:00" and v >= 0 and _valid(name, v):
                        static[name] = v
                    continue
                if name not in TS_IDX or v < 0 or not _valid(name, v):
                    continue
                obs.append((_minutes(t), TS_IDX[name], v))
        obs = np.array(sorted(obs), dtype=np.float32).reshape(-1, 3)
        stays.append(Stay(rid, static, obs, outcomes[rid]))
    assign_splits(stays)
    return stays


def assign_splits(stays: list[Stay], seed: int = 0, frac=(0.7, 0.15, 0.15)) -> None:
    """Stratified train/val/test split, deterministic in seed."""
    rng = np.random.default_rng(seed)
    for label in (0, 1):
        idx = [i for i, s in enumerate(stays) if s.death == label]
        rng.shuffle(idx)
        n_tr = int(frac[0] * len(idx))
        n_va = int(frac[1] * len(idx))
        for j, i in enumerate(idx):
            stays[i].split = "train" if j < n_tr else "val" if j < n_tr + n_va else "test"


def to_grid(stay: Stay, hours: int = HOURS) -> tuple[np.ndarray, np.ndarray]:
    """Hourly mean grid (hours, V) and observation mask (hours, V)."""
    V = len(TS_VARS)
    s = np.zeros((hours, V), np.float64)
    c = np.zeros((hours, V), np.float64)
    if len(stay.obs):
        h = np.clip((stay.obs[:, 0] // 60).astype(int), 0, hours - 1)
        v = stay.obs[:, 1].astype(int)
        np.add.at(s, (h, v), stay.obs[:, 2])
        np.add.at(c, (h, v), 1)
    m = c > 0
    g = np.where(m, s / np.maximum(c, 1), 0.0)
    return g.astype(np.float32), m


class Normalizer:
    """Per-variable robust z-scoring fitted on the training split.

    Heavy-tailed variables (lab values, urine) are log1p-transformed first.
    """
    LOG_VARS = {"ALP", "ALT", "AST", "Bilirubin", "BUN", "Creatinine", "Glucose", "Lactate",
                "TroponinI", "TroponinT", "Urine", "WBC", "Platelets", "PaO2"}

    def __init__(self, mean=None, std=None):
        self.mean, self.std = mean, std
        self._log = np.array([v in self.LOG_VARS for v in TS_VARS])

    def _fwd(self, x, var):
        return np.where(self._log[var], np.log1p(np.maximum(x, 0)), x)

    def fit(self, stays: list[Stay]) -> "Normalizer":
        V = len(TS_VARS)
        self.mean, self.std = np.zeros(V), np.ones(V)
        allobs = np.concatenate([s.obs for s in stays if s.split == "train"])
        for v in range(V):
            x = self._fwd(allobs[allobs[:, 1] == v, 2], v)
            if len(x):
                self.mean[v], self.std[v] = x.mean(), max(x.std(), 1e-3)
        return self

    def transform(self, x, var):
        var = np.asarray(var, int)
        return ((self._fwd(x, var) - self.mean[var]) / self.std[var]).astype(np.float32)

    def inverse(self, z, var):
        var = np.asarray(var, int)
        y = z * self.std[var] + self.mean[var]
        return np.where(self._log[var], np.expm1(y), y).astype(np.float32)

    def grid(self, g: np.ndarray, m: np.ndarray) -> np.ndarray:
        var = np.broadcast_to(np.arange(g.shape[1]), g.shape)
        return np.where(m, self.transform(g, var), 0.0).astype(np.float32)

    def to_json(self):
        return {"vars": TS_VARS, "mean": self.mean.tolist(), "std": self.std.tolist(),
                "log_vars": sorted(self.LOG_VARS)}

    @classmethod
    def from_json(cls, d):
        return cls(np.array(d["mean"]), np.array(d["std"]))
