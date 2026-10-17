"""MIMIC-IV linked multimodal demo subset (credentialed PhysioNet data; never commit it or anything
patient-level derived from it -- only aggregate tables go to reports/).

Layout (supervisor's copy; the p10-p15 archives are unpacked into cxr/files):

    linked/studies.csv      one CXR study per patient: subject_id, study_id, cxr_time, n_images, n_ecg
    linked/cxr_images.csv   dicom_id, subject_id, study_id, ViewPosition
    linked/patients.csv     subject_id, gender, anchor_age, anchor_year, dod
    cxr/files/pXX/pSUBJECT/sSTUDY/DICOM_ID.jpg        MIMIC-CXR-JPG
    reports/files/pXX/pSUBJECT/sSTUDY.txt             free-text radiology report
    ecg/files/pXXXX/pSUBJECT/sECG/ECG.{hea,dat}       MIMIC-IV-ECG, 12 leads, 500 Hz, 10 s, WFDB format 16
    labs/labevents_7d.csv + d_labitems.csv            MIMIC-IV hosp labevents

ECG and labs cover the 7 days up to and including the CXR time; nothing after the radiograph.
The sample's reference time (t = 0) is therefore CXR time - 7 days, so every segment has t >= 0
(the protocol uses t = -1 for "no time") and the CXR / report sit at t = 10080 min.

Label `death`: death within 365 days of the radiograph (from patients.dod, same shifted timeline).
Patients are split 70/15/15 by subject, stratified by the label, as for PhysioNet 2012.
"""
from __future__ import annotations

import csv
import glob
import os
from dataclasses import dataclass, field
from datetime import datetime, timedelta

import numpy as np

from .physionet import assign_splits

WINDOW_MIN = 7 * 24 * 60
ECG_LEADS = ["I", "II", "III", "aVR", "aVF", "aVL", "V1", "V2", "V3", "V4", "V5", "V6"]
ECG_HZ = 500
FRONTAL = ("PA", "AP")
STATIC_VARS = ["Age", "Gender"]            # names shared with the PhysioNet static block


@dataclass
class Patient:
    subject_id: str
    study_id: str
    cxr_time: datetime
    static: dict                           # Age, Gender (1 = male, as in PhysioNet 2012)
    images: list[tuple[str, str]]          # (jpg path, ViewPosition), frontal views first
    report: str                            # path of the report .txt
    ecgs: list[tuple[str, float]]          # (record path without extension, minutes since t = 0)
    labs: np.ndarray = field(default_factory=lambda: np.zeros((0, 3), np.float32))  # (n,3) minutes, var idx, value
    death: int = 0
    days_to_death: float = float("nan")
    split: str = "train"

    @property
    def record_id(self) -> str:            # lets physionet.assign_splits / codecs treat it like a Stay
        return self.subject_id


def _dt(s: str, fmt: str = "%Y-%m-%d %H:%M:%S") -> datetime:
    return datetime.strptime(s, fmt)


def ecg_header_time(hea: str) -> datetime:
    """Record start from the WFDB header line: '<name> <nsig> <fs> <nsamp> HH:MM:SS DD/MM/YYYY'."""
    with open(hea) as f:
        p = f.readline().split()
    return _dt(f"{p[5]} {p[4]}", "%d/%m/%Y %H:%M:%S")


def load_cohort(root: str, split_seed: int = 0, horizon_days: int = 365) -> list[Patient]:
    rows = lambda f: list(csv.DictReader(open(os.path.join(root, f))))  # noqa: E731
    pats = {r["subject_id"]: r for r in rows("linked/patients.csv")}
    views: dict[str, list[tuple[str, str]]] = {}
    for r in rows("linked/cxr_images.csv"):
        views.setdefault(r["study_id"], []).append((r["dicom_id"], r["ViewPosition"]))
    out = []
    for r in rows("linked/studies.csv"):
        sid, st = r["subject_id"], r["study_id"]
        t_cxr = _dt(r["cxr_time"])
        t0 = t_cxr - timedelta(minutes=WINDOW_MIN)
        p = pats[sid]
        study_dir = os.path.join(root, "cxr", "files", f"p{sid[:2]}", f"p{sid}", f"s{st}")
        imgs = [(os.path.join(study_dir, f"{d}.jpg"), v) for d, v in views.get(st, [])]
        imgs = [x for x in imgs if os.path.exists(x[0])]
        imgs.sort(key=lambda x: (x[1] not in FRONTAL, x[1], x[0]))
        ecgs = []
        for hea in sorted(glob.glob(os.path.join(root, "ecg", "files", f"p{sid[:4]}", f"p{sid}", "*", "*.hea"))):
            ecgs.append((hea[:-4], (ecg_header_time(hea) - t0).total_seconds() / 60))
        ecgs.sort(key=lambda x: x[1])
        age = float(p["anchor_age"]) + (t_cxr.year - int(p["anchor_year"]))
        days = (_dt(p["dod"], "%Y-%m-%d") - t_cxr).total_seconds() / 86400 if p["dod"] else float("nan")
        out.append(Patient(sid, st, t_cxr, {"Age": age, "Gender": float(p["gender"] == "M")}, imgs,
                           os.path.join(root, "reports", "files", f"p{sid[:2]}", f"p{sid}", f"s{st}.txt"), ecgs,
                           death=int(days == days and days <= horizon_days), days_to_death=days))
    out.sort(key=lambda x: x.subject_id)
    assign_splits(out, split_seed)
    return out


# ------------------------------------------------------------------ labs
def lab_vocabulary(root: str, patients: list[Patient], top_k: int = 40) -> list[tuple[str, str]]:
    """The top_k most frequent numeric lab items on the TRAINING split -> [(itemid, 'lab:<label>')]."""
    train = {p.subject_id for p in patients if p.split == "train"}
    label = {r["itemid"]: r["label"] for r in csv.DictReader(open(os.path.join(root, "labs", "d_labitems.csv")))}
    count: dict[str, int] = {}
    for r in csv.DictReader(open(os.path.join(root, "labs", "labevents_7d.csv"))):
        if r["subject_id"] in train and r["valuenum"]:
            count[r["itemid"]] = count.get(r["itemid"], 0) + 1
    top = sorted(count, key=lambda k: (-count[k], k))[:top_k]
    names, seen = [], set()
    for k in top:
        n = f"lab:{label.get(k, k)}"
        n = n if n not in seen else f"{n} [{k}]"   # same label, different fluid / method
        seen.add(n)
        names.append((k, n))
    return names


def attach_labs(root: str, patients: list[Patient], vocab: list[tuple[str, str]]) -> None:
    """Fill Patient.labs with (minutes since t = 0, variable index, value) for in-vocabulary numeric results."""
    idx = {k: i for i, (k, _) in enumerate(vocab)}
    by_id = {p.subject_id: p for p in patients}
    obs: dict[str, list] = {s: [] for s in by_id}
    for r in csv.DictReader(open(os.path.join(root, "labs", "labevents_7d.csv"))):
        p = by_id.get(r["subject_id"])
        if p is None or not r["valuenum"] or r["itemid"] not in idx:
            continue
        t = (_dt(r["charttime"], "%m/%d/%Y %H:%M") - (p.cxr_time - timedelta(minutes=WINDOW_MIN))).total_seconds() / 60
        if 0 <= t <= WINDOW_MIN:
            obs[p.subject_id].append((t, idx[r["itemid"]], float(r["valuenum"])))
    for s, o in obs.items():
        by_id[s].labs = np.array(sorted(o), np.float32).reshape(-1, 3)


# ------------------------------------------------------------------ raw payloads
def load_image(path: str, size: int) -> np.ndarray:
    """Grayscale CXR -> centre square crop -> size x size, replicated to (size, size, 3) uint8
    (the MedMNIST tokenizers take 3 channels; raw_bits counts a replicated image once)."""
    from PIL import Image
    im = Image.open(path).convert("L")
    w, h = im.size
    s = min(w, h)
    im = im.crop(((w - s) // 2, (h - s) // 2, (w - s) // 2 + s, (h - s) // 2 + s)).resize((size, size), Image.BILINEAR)
    return np.repeat(np.asarray(im, np.uint8)[..., None], 3, axis=2)


def read_ecg(record: str) -> np.ndarray:
    """WFDB format-16 record -> (n_samples, 12) float32 in mV (gain and baseline from the header)."""
    with open(record + ".hea") as f:
        head = f.readline().split()
        sig = [f.readline().split() for _ in range(int(head[1]))]
    n = int(head[3])
    raw = np.fromfile(record + ".dat", "<i2", count=n * len(sig)).reshape(n, len(sig)).astype(np.float32)
    gain = np.array([float(s[2].split("(")[0].split("/")[0]) for s in sig], np.float32)
    base = np.array([float(s[2].split("(")[1].split(")")[0]) if "(" in s[2] else 0.0 for s in sig], np.float32)
    leads = [s[-1] for s in sig]
    x = (raw - base) / gain
    return x[:, [leads.index(lead) for lead in ECG_LEADS]]


def ecg_resample(x: np.ndarray, hz: int) -> np.ndarray:
    """Downsample by block averaging (500 Hz -> hz; hz must divide 500). NaN (WFDB missing) -> 0."""
    k = ECG_HZ // hz
    x = np.nan_to_num(x[: len(x) // k * k])
    return x.reshape(-1, k, x.shape[1]).mean(1).astype(np.float32)


def ecg_obs(x: np.ndarray, hz: int) -> np.ndarray:
    """(T,12) signal -> (T*12, 3) [minutes, lead index, mV] triples, the time-series codecs' payload.
    Sample k is stamped at the centre of its grid cell so float rounding cannot move it to a neighbour."""
    T, V = x.shape
    t = (np.arange(T) + 0.5) / hz / 60
    return np.stack([np.repeat(t, V), np.tile(np.arange(V), T), x.ravel()], 1).astype(np.float32)


def read_report(path: str) -> str:
    with open(path, encoding="utf-8") as f:
        return f.read().strip()
