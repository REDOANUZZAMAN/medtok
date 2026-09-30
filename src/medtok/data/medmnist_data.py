"""MedMNIST v2 loader (28x28 npz files from Zenodo record 10519652).

All subsets are returned as uint8 RGB (N, 28, 28, 3) so one image codebook
serves grayscale and colour data alike.
"""
from __future__ import annotations

import os

import numpy as np

SUBSETS = ("bloodmnist", "pneumoniamnist", "dermamnist")
URL = "https://zenodo.org/records/10519652/files/{name}.npz"
CLASS_NAMES = {
    "bloodmnist": ["basophil", "eosinophil", "erythroblast", "immature granulocyte",
                   "lymphocyte", "monocyte", "neutrophil", "platelet"],
    "pneumoniamnist": ["normal", "pneumonia"],
    "dermamnist": ["actinic keratosis", "basal cell carcinoma", "benign keratosis", "dermatofibroma",
                   "melanoma", "melanocytic nevus", "vascular lesion"],
}


def load_subset(root: str, name: str) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    d = np.load(os.path.join(root, f"{name}.npz"))
    out = {}
    for split, key in (("train", "train"), ("val", "val"), ("test", "test")):
        x = d[f"{key}_images"]
        if x.ndim == 3:
            x = np.repeat(x[..., None], 3, axis=-1)
        out[split] = (x.astype(np.uint8), d[f"{key}_labels"].reshape(-1).astype(np.int64))
    return out


def load_all(root: str, subsets=SUBSETS) -> dict[str, dict]:
    return {s: load_subset(root, s) for s in subsets}
