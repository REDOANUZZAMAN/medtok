"""Surgical video: frame sampling into a compact per-video cache, and phase labels.

Cache format (one compressed .npz per video, git-ignored):
    frames   (N, S, S, 3) uint8   frames sampled at `fps`, centre-cropped to a square, resized to S
    t_sec    (N,) float32         timestamp of each sampled frame
    phase    (N,) int16           phase index at that frame; -1 = not annotated
    meta     JSON                 dataset, video id, source fps, phase names, ...
All experiments subsample this cache (e.g. a 2 fps cache also serves 1 and 0.5 fps).

Datasets
  Cataract-101  (CC BY-NC 4.0, openly downloadable; 101 cataract surgeries, 10 phases,
                 annotations = phase start frames, see its README). Used here as the
                 real-data surgical-video set.
  Cholec80      (CAMMA, data request required -- never downloaded automatically).
                 `cholec80_phases` + the same frame sampler work on a local copy.
"""
from __future__ import annotations

import csv
import json
import os
import tempfile
import zipfile
from concurrent.futures import ProcessPoolExecutor

import numpy as np

CATARACT101_URL = "http://ftp.itec.aau.at/datasets/ovid/cat-101/downloads/cataract-101.zip"
CATARACT101_PHASES = ["Incision", "Viscous agent injection", "Rhexis", "Hydrodissection", "Phacoemulsification",
                      "Irrigation and aspiration", "Capsule polishing", "Lens implant setting-up",
                      "Viscous agent removal", "Tonifying and antibiotics"]
CHOLEC80_PHASES = ["Preparation", "CalotTriangleDissection", "ClippingCutting", "GallbladderDissection",
                   "GallbladderPackaging", "CleaningCoagulation", "GallbladderRetraction"]
CHOLEC80_TOOLS = ["Grasper", "Bipolar", "Hook", "Scissors", "Clipper", "Irrigator", "SpecimenBag"]


# ------------------------------------------------------------------ frame sampling
def sample_frames(video_path: str, fps: float, size: int, max_frames: int | None = None):
    """Decode a video, keep frames at `fps`, centre-crop to square and resize to `size`.

    Returns (frames (N,size,size,3) uint8, t_sec (N,), source_fps, source_frame_index (N,)).
    """
    import av
    from PIL import Image

    frames, times, src_idx = [], [], []
    with av.open(video_path) as c:
        st = c.streams.video[0]
        st.thread_type = "AUTO"
        src_fps = float(st.average_rate)
        step = src_fps / fps
        nxt = 0.0
        for i, fr in enumerate(c.decode(st)):
            if i + 1e-6 < nxt:
                continue
            nxt += step
            img = fr.to_image()
            w, h = img.size
            s = min(w, h)
            img = img.crop(((w - s) // 2, (h - s) // 2, (w - s) // 2 + s, (h - s) // 2 + s))
            frames.append(np.asarray(img.resize((size, size), Image.Resampling.BILINEAR), np.uint8))
            times.append(i / src_fps)
            src_idx.append(i)
            if max_frames and len(frames) >= max_frames:
                break
    return np.stack(frames), np.array(times, np.float32), src_fps, np.array(src_idx)


def save_cache(path: str, frames, t_sec, phase, meta: dict) -> None:
    np.savez_compressed(path, frames=frames, t_sec=t_sec, phase=phase.astype(np.int16),
                        meta=np.frombuffer(json.dumps(meta).encode(), np.uint8))


def load_cache(path: str) -> dict:
    with np.load(path) as z:
        return {"frames": z["frames"], "t_sec": z["t_sec"], "phase": z["phase"].astype(np.int64),
                "meta": json.loads(bytes(z["meta"]).decode())}


# ------------------------------------------------------------------ Cataract-101
def cataract101_annotations(csv_text: str) -> dict[int, list[tuple[int, int]]]:
    """VideoID -> sorted [(start frame, phase 0..9)]."""
    out: dict[int, list] = {}
    for row in csv.DictReader(csv_text.splitlines(), delimiter=";"):
        out.setdefault(int(row["VideoID"]), []).append((int(row["FrameNo"]), int(row["Phase"]) - 1))
    return {k: sorted(v) for k, v in out.items()}


def phase_at(frame_idx: np.ndarray, starts: list[tuple[int, int]]) -> np.ndarray:
    """Phase of each frame: the most recent start annotation at or before it; -1 before the first."""
    s = np.array([a for a, _ in starts])
    p = np.array([b for _, b in starts])
    k = np.searchsorted(s, frame_idx, side="right") - 1
    return np.where(k >= 0, p[np.clip(k, 0, None)], -1)


def _c101_worker(args):
    zip_path, member, vid, starts, out, fps, size = args
    if os.path.exists(out):
        return vid, "cached"
    with zipfile.ZipFile(zip_path) as z, tempfile.NamedTemporaryFile(suffix=".mp4") as tmp:
        with z.open(member) as src:
            while chunk := src.read(1 << 22):
                tmp.write(chunk)
        tmp.flush()
        frames, t, src_fps, idx = sample_frames(tmp.name, fps, size)
    ph = phase_at(idx, starts)
    save_cache(out + ".part.npz", frames, t, ph, {"dataset": "cataract101", "video_id": vid, "fps": fps,
                                                  "size": size, "source_fps": src_fps,
                                                  "phase_names": CATARACT101_PHASES})
    os.replace(out + ".part.npz", out)
    return vid, f"{len(frames)} frames"


def prepare_cataract101(zip_path: str, out_dir: str, fps: float = 2.0, size: int = 64, workers: int = 4,
                        limit: int | None = None) -> list[int]:
    """Sample every video of the Cataract-101 zip into the frame cache (resumable)."""
    os.makedirs(out_dir, exist_ok=True)
    with zipfile.ZipFile(zip_path) as z:
        ann = cataract101_annotations(z.read("cataract-101/annotations.csv").decode())
        meta_rows = list(csv.DictReader(z.read("cataract-101/videos.csv").decode().splitlines(), delimiter=";"))
    with open(os.path.join(out_dir, "videos.json"), "w") as f:
        json.dump(meta_rows, f, indent=1)
    vids = sorted(ann)[:limit]
    jobs = [(zip_path, f"cataract-101/videos/case_{v}.mp4", v, ann[v], os.path.join(out_dir, f"{v}.npz"), fps, size)
            for v in vids]
    with ProcessPoolExecutor(workers) as ex:
        for vid, msg in ex.map(_c101_worker, jobs):
            print(f"  cataract101/{vid}: {msg}", flush=True)
    return vids


def video_splits(video_ids: list, seed: int = 0, frac=(0.7, 0.15, 0.15)) -> dict:
    """Video-level (= patient-level) split, deterministic in seed."""
    ids = sorted(video_ids)
    rng = np.random.default_rng(seed)
    rng.shuffle(ids)
    n_tr, n_va = int(round(frac[0] * len(ids))), int(round(frac[1] * len(ids)))
    return {v: ("train" if i < n_tr else "val" if i < n_tr + n_va else "test") for i, v in enumerate(ids)}


# ------------------------------------------------------------------ Cholec80 (local copy only)
def cholec80_phases(phase_file: str, sample_every: int = 25) -> np.ndarray:
    """Parse `videoXX-phase.txt` (one line per 25 fps frame) into phase indices every `sample_every` frames."""
    out = []
    with open(phase_file) as f:
        next(f)
        for line in f:
            frame, ph = line.split()
            if int(frame) % sample_every == 0:
                out.append(CHOLEC80_PHASES.index(ph))
    return np.array(out)


def cholec80_tools(tool_file: str) -> np.ndarray:
    """Parse `videoXX-tool.txt` (1 fps, 7 binary columns) -> (T,7) int8."""
    with open(tool_file) as f:
        next(f)
        return np.array([[int(x) for x in line.split()[1:]] for line in f], np.int8)


# ------------------------------------------------------------------ synthetic (tests only)
def synthetic_video(n_frames: int = 64, size: int = 32, n_phases: int = 4, seed: int = 0):
    """Moving blobs whose colour changes with a piecewise-constant 'phase' (for tests; no medical content)."""
    rng = np.random.default_rng(seed)
    cuts = np.sort(rng.choice(np.arange(4, n_frames - 4), n_phases - 1, replace=False))
    phase = np.searchsorted(cuts, np.arange(n_frames), side="right")
    colours = rng.integers(40, 255, (n_phases, 3))
    yy, xx = np.mgrid[:size, :size]
    frames = np.zeros((n_frames, size, size, 3), np.uint8)
    for t in range(n_frames):
        cx, cy = size / 2 + size / 4 * np.cos(t / 5), size / 2 + size / 4 * np.sin(t / 7)
        blob = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * (size / 8) ** 2))
        frames[t] = (blob[..., None] * colours[phase[t]]).astype(np.uint8)
    return frames, phase
