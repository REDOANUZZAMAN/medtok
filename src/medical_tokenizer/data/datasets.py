"""Dataset adapters: config `data:` block -> training arrays and protocol Samples per split.

Every adapter is driven by the config, so adding a dataset means adding one
adapter here and a config file; tokenizers and downstream code are untouched.
"""
from __future__ import annotations

import glob
import os
import pickle

import numpy as np

from ..protocol.schema import Sample, Segment
from ..utils.experiment import REPO_ROOT
from . import medmnist, physionet
from .notes import image_note, stay_note, video_note
from .video import load_cache, video_splits


def abspath(p: str) -> str:
    return p if os.path.isabs(p) else os.path.join(REPO_ROOT, p)


# ------------------------------------------------------------------ images (MedMNIST)
def image_split(d: dict, split: str, subsets=None):
    """-> x (N,S,S,3) uint8, y (N,), subset name per sample (N,), index within subset split (N,)."""
    xs, ys, names, idx = [], [], [], []
    for s in subsets or d["subsets"]:
        x, y = medmnist.load_subset(abspath(d["root"]), s, d.get("image_size", 28))[split]
        xs.append(x)
        ys.append(y)
        names += [s] * len(x)
        idx.append(np.arange(len(x)))
    return np.concatenate(xs), np.concatenate(ys), np.array(names), np.concatenate(idx)


def subsample(n: int, frac: float | None, seed: int) -> np.ndarray:
    if not frac or frac >= 1:
        return np.arange(n)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(n, int(round(frac * n)), replace=False))


def image_samples(d: dict, split: str, subset: str, include_text: bool = False) -> list[Sample]:
    x, y, _, idx = image_split(d, split, [subset])
    out = []
    for i in range(len(x)):
        segs = [Segment("image", x[i], 0.0, subset)]
        if include_text:
            segs.insert(0, Segment("text", image_note(subset, int(idx[i])), 0.0, "template:request"))
        out.append(Sample(f"{subset}/{split}/{idx[i]}", segs, {"class": int(y[i])}, split, {"task": subset}))
    return out


# ------------------------------------------------------------------ time-series (PhysioNet 2012)
def stays(d: dict) -> list[physionet.Stay]:
    """Parsed stays with deterministic splits, cached in data/processed (git-ignored)."""
    seed = d.get("split_seed", 0)
    cache = abspath(os.path.join(d.get("processed", "data/processed"), f"physionet2012_stays_s{seed}.pkl"))
    if os.path.exists(cache):
        with open(cache, "rb") as f:
            return pickle.load(f)
    st = physionet.load_stays(abspath(d["root"]), split_seed=seed)
    os.makedirs(os.path.dirname(cache), exist_ok=True)
    with open(cache, "wb") as f:
        pickle.dump(st, f)
    return st


def icu_samples(d: dict, split: str, include_text: bool = False) -> list[Sample]:
    from ..tokenizers.timeseries.codec import stay_segment
    out = []
    for s in stays(d):
        if s.split != split:
            continue
        segs = [stay_segment(s)]
        if include_text:
            segs.insert(0, Segment("text", stay_note(s), 0.0, "template:admission"))
        out.append(Sample(f"physionet2012/{s.record_id}", segs, {"death": int(s.death)}, split, {"task": "mortality"}))
    return out


# ------------------------------------------------------------------ video (Cataract-101 frame cache)
def video_files(d: dict) -> dict[int, str]:
    return {int(os.path.basename(p)[:-4]): p for p in sorted(glob.glob(os.path.join(abspath(d["cache"]), "*.npz")))
            if not p.endswith(".part.npz")}


def video_split_map(d: dict) -> dict[int, str]:
    return video_splits(list(video_files(d)), d.get("split_seed", 0))


def load_video(path: str, fps: float) -> dict:
    """Load a cached video subsampled to `fps` (must divide the cache fps)."""
    v = load_cache(path)
    k = v["meta"]["fps"] / fps
    if abs(k - round(k)) > 1e-6 or k < 1:
        raise ValueError(f"target fps {fps} must divide the cache fps {v['meta']['fps']}")
    k = int(round(k))
    return {"frames": v["frames"][::k], "phase": v["phase"][::k], "t_sec": v["t_sec"][::k], "fps": fps,
            "meta": v["meta"]}


def video_clip_array(d: dict, split: str, max_videos: int | None = None, max_clips_per_video: int | None = None,
                     seed: int = 0) -> np.ndarray:
    """All non-overlapping clips (N, clip_len, S, S, 3) of the split's videos (optionally capped per video)."""
    from ..tokenizers.video.codec import make_clips
    files, smap = video_files(d), video_split_map(d)
    vids = [v for v in files if smap[v] == split][:max_videos]
    rng = np.random.default_rng(seed)
    out = []
    for v in vids:
        c = make_clips(load_video(files[v], d["fps"])["frames"], d["clip_len"])
        if max_clips_per_video and len(c) > max_clips_per_video:
            c = c[np.sort(rng.choice(len(c), max_clips_per_video, replace=False))]
        out.append(c)
    return np.concatenate(out)


def video_windows(d: dict, v: dict) -> list[tuple[int, int]]:
    """(first clip, label) for every window of `window_clips` clips, stepping `window_stride_clips`;
    the label is the phase of the window's last frame (online recognition); unannotated windows skipped."""
    L, W = d["clip_len"], d["window_clips"]
    n_clips = len(v["frames"]) // L
    out = []
    for c in range(0, n_clips - W + 1, d.get("window_stride_clips", 1)):
        lab = int(v["phase"][(c + W) * L - 1])
        if lab >= 0:
            out.append((c, lab))
    return out


def video_text_note(vid: int) -> str:
    return video_note(vid)
