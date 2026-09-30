"""Synthetic fixtures: no medical dataset is needed to run the test suite."""
from __future__ import annotations

import os

import numpy as np
import pytest
import torch

from medical_tokenizer.data.physionet import STATIC_VARS, TS_VARS, Normalizer, Stay
from medical_tokenizer.tokenizers.text.codec import ClinicalTextTokenizer
from medical_tokenizer.tokenizers.timeseries.codec import GridSpec, ValueBinner
from medical_tokenizer.toolkit import build_lexicon

torch.set_num_threads(2)


def synthetic_stay(rid: int, rng: np.random.Generator, split: str = "train") -> Stay:
    obs = []
    for j in range(len(TS_VARS)):
        if rng.random() < 0.5:
            continue
        n = rng.integers(1, 12)
        t = np.sort(rng.uniform(0, 48 * 60 - 1, n))
        base = rng.uniform(1, 100)
        obs += [(float(tt), j, float(base + rng.normal(0, 5))) for tt in t]
    obs = np.array(sorted(obs), np.float32).reshape(-1, 3)
    obs[:, 2] = np.abs(obs[:, 2])
    static = {"Age": float(rng.integers(20, 90)), "Gender": float(rng.integers(0, 2)),
              "Height": float(rng.uniform(150, 190)) if rng.random() < 0.7 else np.nan, "ICUType": float(rng.integers(1, 5))}
    return Stay(str(100000 + rid), static, obs, int(rng.random() < 0.3), split)


def write_physionet(root: str, n: int = 40, seed: int = 0) -> None:
    """Files in the PhysioNet 2012 set-a text format."""
    rng = np.random.default_rng(seed)
    os.makedirs(os.path.join(root, "set-a"), exist_ok=True)
    with open(os.path.join(root, "Outcomes-a.txt"), "w") as out:
        out.write("RecordID,SAPS-I,SOFA,Length_of_stay,Survival,In-hospital_death\n")
        for i in range(n):
            s = synthetic_stay(i, rng)
            with open(os.path.join(root, "set-a", f"{s.record_id}.txt"), "w") as f:
                f.write("Time,Parameter,Value\n00:00,RecordID," + s.record_id + "\n")
                for k in STATIC_VARS:
                    f.write(f"00:00,{k},{-1 if np.isnan(s.static[k]) else s.static[k]}\n")
                for t, j, v in s.obs:
                    f.write(f"{int(t) // 60:02d}:{int(t) % 60:02d},{TS_VARS[int(j)]},{v:.2f}\n")
            out.write(f"{s.record_id},0,0,5,-1,{s.death}\n")


def write_medmnist(root: str, name: str, n=(64, 16, 16), n_classes: int = 3, gray: bool = False, seed: int = 0):
    """An npz in the MedMNIST v2 layout (class-dependent brightness so classes are learnable)."""
    rng = np.random.default_rng(seed)
    os.makedirs(root, exist_ok=True)
    arrays = {}
    for split, k in zip(("train", "val", "test"), n):
        y = rng.integers(0, n_classes, k)
        base = (40 + 60 * y)[:, None, None]
        x = np.clip(base + rng.normal(0, 20, (k, 28, 28)), 0, 255).astype(np.uint8)
        if not gray:
            x = np.repeat(x[..., None], 3, -1)
        arrays[f"{split}_images"], arrays[f"{split}_labels"] = x, y[:, None]
    np.savez(os.path.join(root, f"{name}.npz"), **arrays)


def write_video_cache(root: str, n_videos: int = 6, n_frames: int = 96, size: int = 32, fps: float = 2.0):
    from medical_tokenizer.data.video import save_cache, synthetic_video
    os.makedirs(root, exist_ok=True)
    for v in range(n_videos):
        frames, phase = synthetic_video(n_frames, size, 4, seed=v)
        phase = phase.astype(np.int64)
        phase[:3] = -1                      # unannotated lead-in, as in Cataract-101
        save_cache(os.path.join(root, f"{100 + v}.npz"), frames, np.arange(n_frames) / fps, phase,
                   {"dataset": "synthetic", "video_id": 100 + v, "fps": fps, "size": size})


@pytest.fixture(scope="session")
def stays():
    rng = np.random.default_rng(0)
    return [synthetic_stay(i, rng, "train" if i < 20 else "test") for i in range(24)]


@pytest.fixture(scope="session")
def lex():
    return build_lexicon()


@pytest.fixture(scope="session")
def spec():
    return GridSpec()


@pytest.fixture(scope="session")
def binner(stays, spec):
    return ValueBinner(16).fit(stays, spec)


@pytest.fixture(scope="session")
def norm(stays):
    return Normalizer().fit(stays)


@pytest.fixture(scope="session")
def text_tok():
    corpus = ["Patient admitted to the ICU with sepsis.", "Chest X-ray, frontal view.",
              "Heart rate 110 bpm, MAP 65 mmHg."] * 30
    return ClinicalTextTokenizer.train(corpus, vocab_size=400)
