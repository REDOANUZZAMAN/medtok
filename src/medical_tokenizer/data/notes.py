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
"""
from __future__ import annotations


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
    "Recorded height {height} and admission weight {weight}.",
]
_VITALS = [
    "Initial vitals: HR {hr}, MAP {map}, temp {temp}.",
    "On arrival heart rate {hr} bpm, mean arterial pressure {map} mmHg, temperature {temp}.",
    "First-hour observations - HR {hr}; MAP {map}; T {temp}.",
]
_NEURO = [
    "GCS {gcs} on admission.",
    "Glasgow Coma Scale {gcs}.",
    "Neuro: GCS {gcs}.",
]
_VENT = ["Mechanically ventilated on arrival.", "Intubated, on ventilator.", "Receiving mechanical ventilation."]


def _fmt(v, unit="", nd=0):
    if v is None or np.isnan(v):
        return "not recorded"
    return f"{v:.{nd}f}{unit}"


def _first_hour(stay: Stay, var: str) -> float:
    o = stay.obs
    sel = o[(o[:, 1] == TS_IDX[var]) & (o[:, 0] < 60)]
    return float(sel[0, 2]) if len(sel) else np.nan


def stay_note(stay: Stay, seed: int | None = None) -> str:
    rng = np.random.default_rng(int(stay.record_id) if seed is None else seed)
    s = stay.static
    sex = {0.0: "female", 1.0: "male"}.get(s["Gender"], "patient")
    unit = ICU_TYPES.get(int(s["ICUType"]) if not np.isnan(s["ICUType"]) else 0, "ICU")
    age = _fmt(s["Age"])
    parts = [rng.choice(_OPEN).format(age=age, sex=sex, Sex=sex.capitalize(), unit=unit)]
    parts.append(rng.choice(_BODY).format(height=_fmt(s["Height"], " cm"),
                                          weight=_fmt(_first_hour(stay, "Weight"), " kg", 1)))
    hr, mp, tp = (_first_hour(stay, v) for v in ("HR", "MAP", "Temp"))
    if np.isnan(mp):
        mp = _first_hour(stay, "NIMAP")
    if not all(np.isnan([hr, mp, tp])):
        parts.append(rng.choice(_VITALS).format(hr=_fmt(hr), map=_fmt(mp), temp=_fmt(tp, " C", 1)))
    gcs = _first_hour(stay, "GCS")
    if not np.isnan(gcs):
        parts.append(rng.choice(_NEURO).format(gcs=_fmt(gcs)))
    if _first_hour(stay, "MechVent") == 1:
        parts.append(str(rng.choice(_VENT)))
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


VIDEO_CONTEXT = ["Cataract surgery, operating-microscope video.", "Phacoemulsification cataract procedure recording.",
                 "Ophthalmic surgery video, anterior segment."]


def video_note(idx: int) -> str:
    return VIDEO_CONTEXT[idx % len(VIDEO_CONTEXT)]
