"""Templated clinical text (the text modality's data source in this project).

None of the project's open datasets ships free-text notes (MIMIC notes are
credentialed-access and may not be redistributed). Short notes are therefore
RENDERED FROM STRUCTURED FIELDS with varied phrasing:
  * ICU admission notes use admission-time fields only (demographics, unit,
    first-hour vitals); nothing after hour 1 and no outcome enters the note,
    so text cannot leak the mortality label;
  * image-request notes state modality / body site only, never the diagnosis;
  * surgical-video notes state the procedure type only.
They exercise the text tokenizer and the multimodal layout; they are not
clinical language and no claim about clinical text is made from them.

ALL TEXT HERE IS SYNTHETIC. Every text segment carries the source tag
`synthetic-template:<kind>@v<TEMPLATE_VERSION>`; the generation rules and the
structured fields each kind reads are listed in TEMPLATE_RULES and are written
to data/processed/data_stats.json by `prepare`. Bump TEMPLATE_VERSION whenever
a template or rule changes (the text BPE must then be refitted).

Coverage (v2): wording variants for every sentence, units, missing values
("not recorded"), explicit negation (no ventilation / no vitals / GCS not
assessed) and times relative to admission ("at 00:12 after admission").

Records from different datasets are never joined into one sample, so no
sample claims to be a patient-matched multimodal record.
"""
from __future__ import annotations

import re

import numpy as np

from .physionet import ICU_TYPES, TS_IDX, Stay

_OPEN = [
    "{age}-year-old {sex} admitted to the {unit}.",
    "Patient is a {age} y/o {sex}, admitted to {unit}.",
    "Admission to {unit}: {sex}, age {age}.",
    "{Sex}, {age} years, transferred to the {unit}.",
]
_BODY = [
    "Height {height}, weight {weight}.",
    "Anthropometrics: height {height}; weight {weight}.",
    "Height {height}; admission weight {weight}.",
]
_VITALS = [
    "Initial vitals at {time} after admission: HR {hr}, MAP {map}, temp {temp}.",
    "At {time} heart rate {hr} bpm, mean arterial pressure {map} mmHg, temperature {temp}.",
    "First-hour observations ({time}) - HR {hr}; MAP {map}; T {temp}.",
]
_NEURO = [
    "GCS {gcs} on admission.",
    "Glasgow Coma Scale {gcs}.",
    "Neuro: GCS {gcs}.",
]
_VENT = ["Mechanically ventilated from {time} after admission.", "Intubated, on ventilator by {time}.",
         "Receiving mechanical ventilation (first charted {time} after admission)."]
_NO_VENT = ["Not mechanically ventilated in the first hour.", "No ventilatory support charted in hour 1.",
            "Not intubated on arrival."]
_NO_VITALS = ["No vital signs recorded in the first hour.", "First-hour vitals: none charted."]
_NO_GCS = ["GCS not assessed in the first hour.", "Neuro: no GCS recorded.", "Glasgow Coma Scale not documented."]

TEMPLATE_VERSION = 2
TEMPLATE_RULES = {
    "admission": {"source_dataset": "physionet2012",
                  "fields": ["Age", "Gender", "Height", "ICUType", "Weight", "HR", "MAP/NIMAP", "Temp", "GCS",
                             "MechVent"],
                  "window": "minutes 0-59 of the stay only; outcome and later observations never read",
                  "rules": "one variant per sentence drawn with rng seeded by record_id; missing value -> "
                           "'not recorded'; absent vitals / GCS / ventilation -> negated sentence; times are "
                           "HH:MM after admission of the first charted value"},
    "request": {"source_dataset": "medmnist", "fields": ["subset"],
                "rules": "modality / body-site sentence, variant = image index mod 3; never the class label"},
    "procedure": {"source_dataset": "cataract101", "fields": [],
                  "rules": "procedure-type sentence, variant = video id mod 3; never the phase labels"},
}


_FIELD_RE = re.compile(r"\d+(?:\.\d+)?(?:\s?(?:cm|kg|bpm|mmHg|C))?|\d{2}:\d{2}|not recorded|\bnot\b|\bno\b|\bnone\b",
                       re.IGNORECASE)


def note_fields(text: str) -> list[str]:
    """Values, units, times and negation cues of a note, in order (for field-preservation checks)."""
    return _FIELD_RE.findall(text)


def template_source(kind: str) -> str:
    """Source tag stored on every templated text Segment."""
    return f"synthetic-template:{kind}@v{TEMPLATE_VERSION}"


def _fmt(v, unit="", nd=0):
    if v is None or np.isnan(v):
        return "not recorded"
    return f"{v:.{nd}f}{unit}"


def _first_hour_obs(stay: Stay, var: str) -> tuple[float, float]:
    """(value, minute) of the first observation of `var` in hour 1, NaNs if none."""
    o = stay.obs
    sel = o[(o[:, 1] == TS_IDX[var]) & (o[:, 0] < 60)]
    return (float(sel[0, 2]), float(sel[0, 0])) if len(sel) else (np.nan, np.nan)


def _first_hour(stay: Stay, var: str) -> float:
    return _first_hour_obs(stay, var)[0]


def _clock(minute: float) -> str:
    m = int(minute)
    return f"{m // 60:02d}:{m % 60:02d}"


def stay_note(stay: Stay, seed: int | None = None) -> str:
    rng = np.random.default_rng(int(stay.record_id) if seed is None else seed)
    s = stay.static
    sex = {0.0: "female", 1.0: "male"}.get(s["Gender"], "patient")
    unit = ICU_TYPES.get(int(s["ICUType"]) if not np.isnan(s["ICUType"]) else 0, "ICU")
    age = _fmt(s["Age"])
    parts = [rng.choice(_OPEN).format(age=age, sex=sex, Sex=sex.capitalize(), unit=unit)]
    parts.append(rng.choice(_BODY).format(height=_fmt(s["Height"], " cm"),
                                          weight=_fmt(_first_hour(stay, "Weight"), " kg", 1)))
    (hr, t_hr), (mp, t_mp), (tp, t_tp) = (_first_hour_obs(stay, v) for v in ("HR", "MAP", "Temp"))
    if np.isnan(mp):
        mp, t_mp = _first_hour_obs(stay, "NIMAP")
    if not all(np.isnan([hr, mp, tp])):
        t0 = np.nanmin([t_hr, t_mp, t_tp])
        parts.append(rng.choice(_VITALS).format(time=_clock(t0), hr=_fmt(hr), map=_fmt(mp), temp=_fmt(tp, " C", 1)))
    else:
        parts.append(str(rng.choice(_NO_VITALS)))
    gcs = _first_hour(stay, "GCS")
    parts.append(rng.choice(_NEURO).format(gcs=_fmt(gcs)) if not np.isnan(gcs) else str(rng.choice(_NO_GCS)))
    vent, t_vent = _first_hour_obs(stay, "MechVent")
    parts.append(rng.choice(_VENT).format(time=_clock(t_vent)) if vent == 1 else str(rng.choice(_NO_VENT)))
    return " ".join(parts)


IMAGE_CONTEXT = {
    "bloodmnist": ["Peripheral blood smear, single-cell microscopy crop.",
                   "Blood film image of an individual cell for morphology review.",
                   "Microscopy: stained peripheral blood cell."],
    "pneumoniamnist": ["Pediatric chest radiograph, frontal view.",
                       "Chest X-ray (AP/PA) of a paediatric patient.",
                       "Frontal chest film, child."],
    "dermamnist": ["Dermatoscopic image of a pigmented skin lesion.",
                   "Dermoscopy photograph, skin lesion.",
                   "Skin lesion captured with a dermatoscope."],
}


def image_note(subset: str, idx: int) -> str:
    opts = IMAGE_CONTEXT[subset]
    return opts[idx % len(opts)]


VIDEO_CONTEXT = ["Cataract surgery, operating-microscope video.", "Cataract extraction procedure recording.",
                 "Ophthalmic surgery video, anterior segment."]


def video_note(idx: int) -> str:
    return VIDEO_CONTEXT[idx % len(VIDEO_CONTEXT)]
