"""MedMNIST v2 (CC BY 4.0; Yang et al., Scientific Data 2023), Zenodo record 10519652.

Official train/val/test splits are used unchanged. All subsets are returned as
uint8 RGB (N, S, S, 3) so one image codebook serves grayscale and colour data.
`size` 28 is the classic MedMNIST; 64/128/224 are the MedMNIST+ files.
"""
from __future__ import annotations

import os
import urllib.request

import numpy as np

URL = "https://zenodo.org/records/10519652/files/{file}"
CLASS_NAMES = {
    "bloodmnist": ["basophil", "eosinophil", "erythroblast", "immature granulocyte", "lymphocyte", "monocyte",
                   "neutrophil", "platelet"],
    "pneumoniamnist": ["normal", "pneumonia"],
    "dermamnist": ["actinic keratosis", "basal cell carcinoma", "benign keratosis", "dermatofibroma", "melanoma",
                   "melanocytic nevus", "vascular lesion"],
}


def _file(name: str, size: int) -> str:
    return f"{name}.npz" if size == 28 else f"{name}_{size}.npz"


def download(root: str, subsets, size: int = 28) -> None:
    os.makedirs(root, exist_ok=True)
    for s in subsets:
        dst = os.path.join(root, _file(s, size))
        if not os.path.exists(dst):
            print(f"downloading {_file(s, size)}")
            urllib.request.urlretrieve(URL.format(file=_file(s, size)), dst + ".part")
            os.replace(dst + ".part", dst)


def load_subset(root: str, name: str, size: int = 28) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    with np.load(os.path.join(root, _file(name, size))) as d:
        out = {}
        for split in ("train", "val", "test"):
            x = d[f"{split}_images"]
            if x.ndim == 3:
                x = np.repeat(x[..., None], 3, axis=-1)
            out[split] = (x.astype(np.uint8), d[f"{split}_labels"].reshape(-1).astype(np.int64))
    return out
