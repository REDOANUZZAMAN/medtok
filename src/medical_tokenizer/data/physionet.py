"""PhysioNet/CinC Challenge 2012, set-a (open access, ODC-By 1.0).

Each record is one ICU stay: up to 48 h of irregularly sampled measurements of
37 temporal variables plus 4 static descriptors, and an in-hospital mortality
label (Outcomes-a.txt). Cleaning: negative values (the dataset's missing-value
code) and physiologically impossible values are dropped. The raw observations
(time, variable, value) are kept for the event tokenizer; `to_grid` builds a
regular grid (mean per step + observation mask) at a configurable resolution
for the window autoencoder.
"""
from __future__ import annotations

import os
import tarfile
import urllib.request
from dataclasses import dataclass

import numpy as np

URL = "https://physionet.org/files/challenge-2012/1.0.0/{name}"
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
# Heavy-tailed variables are log1p-transformed before z-scoring.
LOG_VARS = {"ALP", "ALT", "AST", "Bilirubin", "BUN", "Creatinine", "Glucose", "Lactate",
            "TroponinI", "TroponinT", "Urine", "WBC", "Platelets", "PaO2"}
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
    obs: np.ndarray        # (n, 3) float32: minutes, variable index, value; time-sorted
    death: int             # in-hospital mortality
    split: str = "train"


def download(root: str) -> None:
    """Fetch set-a and its outcomes (about 13 MB) if not present."""
    os.makedirs(root, exist_ok=True)
    for name in ("set-a.tar.gz", "Outcomes-a.txt"):
        dst = os.path.join(root, name)
        if not os.path.exists(dst):
            print(f"downloading {name}")
            urllib.request.urlretrieve(URL.format(name=name), dst + ".part")
            os.replace(dst + ".part", dst)
    if not os.path.isdir(os.path.join(root, "set-a")):
        with tarfile.open(os.path.join(root, "set-a.tar.gz")) as t:
            t.extractall(root, filter="data")


def _minutes(hhmm: str) -> int:
    h, m = hhmm.split(":")
    return int(h) * 60 + int(m)


def _valid(name: str, v: float) -> bool:
    lo, hi = VALID.get(name, (-np.inf, np.inf))
    return lo <= v <= hi


def load_stays(root: str, limit: int | None = None, split_seed: int = 0) -> list[Stay]:
    outcomes = {}
    with open(os.path.join(root, "Outcomes-a.txt")) as f:
        next(f)
        for line in f:
            p = line.strip().split(",")
            outcomes[p[0]] = int(p[5])
    stays = []
    for fn in sorted(os.listdir(os.path.join(root, "set-a")))[:limit]:
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
        stays.append(Stay(rid, static, np.array(sorted(obs), np.float32).reshape(-1, 3), outcomes[rid]))
    assign_splits(stays, split_seed)
    return stays


def assign_splits(stays: list[Stay], seed: int = 0, frac=(0.7, 0.15, 0.15)) -> None:
    """Stratified (by outcome) patient-level train/val/test split, deterministic in seed."""
    rng = np.random.default_rng(seed)
    for label in sorted({s.death for s in stays}):
        idx = [i for i, s in enumerate(stays) if s.death == label]
        rng.shuffle(idx)
        n_tr, n_va = int(frac[0] * len(idx)), int(frac[1] * len(idx))
        for j, i in enumerate(idx):
            stays[i].split = "train" if j < n_tr else "val" if j < n_tr + n_va else "test"


def to_grid(obs: np.ndarray, steps: int = HOURS, resolution_min: float = 60.0,
            n_vars: int = len(TS_VARS)) -> tuple[np.ndarray, np.ndarray]:
    """Regular grid (steps, V) of per-step means and the observation mask (steps, V)."""
    s = np.zeros((steps, n_vars), np.float64)
    c = np.zeros((steps, n_vars), np.float64)
    if len(obs):
        h = np.clip((obs[:, 0] // resolution_min).astype(int), 0, steps - 1)
        v = obs[:, 1].astype(int)
        np.add.at(s, (h, v), obs[:, 2])
        np.add.at(c, (h, v), 1)
    m = c > 0
    return np.where(m, s / np.maximum(c, 1), 0.0).astype(np.float32), m


def fill_missing(z: np.ndarray, m: np.ndarray, strategy: str = "zero") -> np.ndarray:
    """Values at unobserved cells of a normalised grid: 'zero' (= train mean) or 'locf' (last observation
    carried forward, zero before the first one). The mask is always given to the model separately."""
    if strategy == "zero":
        return np.where(m, z, 0.0).astype(np.float32)
    if strategy != "locf":
        raise ValueError(f"unknown missing-value strategy {strategy!r}")
    out = np.zeros_like(z)
    last = np.zeros(z.shape[1], z.dtype)
    for h in range(z.shape[0]):
        last = np.where(m[h], z[h], last)
        out[h] = last
    return out


class Normalizer:
    """Per-variable z-scoring (after log1p for heavy-tailed variables), fitted on the training split."""

    def __init__(self, mean=None, std=None, variables=TS_VARS):
        self.variables = list(variables)
        self.mean = None if mean is None else np.asarray(mean, np.float64)
        self.std = None if std is None else np.asarray(std, np.float64)
        self._log = np.array([v in LOG_VARS for v in self.variables])

    def _fwd(self, x, var):
        return np.where(self._log[var], np.log1p(np.maximum(x, 0)), x)

    def fit(self, stays: list[Stay]) -> "Normalizer":
        V = len(self.variables)
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

    def grid(self, g: np.ndarray, m: np.ndarray, fill: str = "zero") -> np.ndarray:
        var = np.broadcast_to(np.arange(g.shape[1]), g.shape)
        return fill_missing(np.where(m, self.transform(g, var), 0.0), m, fill)

    def inverse_grid(self, z: np.ndarray) -> np.ndarray:
        return self.inverse(z, np.broadcast_to(np.arange(z.shape[-1]), z.shape))

    def to_json(self):
        return {"variables": self.variables, "mean": self.mean.tolist(), "std": self.std.tolist(),
                "log_vars": sorted(LOG_VARS & set(self.variables))}

    @classmethod
    def from_json(cls, d):
        return cls(d["mean"], d["std"], d.get("variables", TS_VARS))
