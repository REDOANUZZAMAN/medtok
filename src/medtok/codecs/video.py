"""Surgical-video codecs (interface level; Cholec80 needs a CAMMA data request).

SurgicalEventCodec  Exactly reversible event tokens from Cholec80-style
    annotations. Segment.data = {"phase": (T,) int phase index per second,
    "tools": (T, 7) {0,1} tool presence per second}. Output:
        <ev:phase:P> <ev:on:Tool>...  [<dt:k> <ev:...> ...]
    Every token carries its exact time (minutes, float32) in the `t` side
    channel; <dt:k> tokens give the model a coarse view of the gaps.

VideoFrameVQCodec   Clip -> per-frame VQ codes in the `video_vq` range, with a
    <frame> marker per frame and (frame, row, col) positions. Any frame
    tokenizer exposing codes(frames)->(T,h,w) and decode_codes works; the
    demo plugs in the MedMNIST image VQ-VAE on synthetic 28x28 frames.
"""
from __future__ import annotations

import numpy as np
import torch

from ..protocol.schema import Segment, TokenSpan
from ..protocol.vocab import RANGES, SPECIAL, Lexicon, dt_token
from .base import Codec

CHOLEC80_PHASES = ["Preparation", "CalotTriangleDissection", "ClippingCutting", "GallbladderDissection",
                   "GallbladderPackaging", "CleaningCoagulation", "GallbladderRetraction"]
CHOLEC80_TOOLS = ["Grasper", "Bipolar", "Hook", "Scissors", "Clipper", "Irrigator", "SpecimenBag"]
VIDEO_EVENTS = ([f"phase:{p}" for p in CHOLEC80_PHASES] + [f"on:{t}" for t in CHOLEC80_TOOLS] +
                [f"off:{t}" for t in CHOLEC80_TOOLS])


def load_cholec80_annotations(phase_file: str, tool_file: str, fps: int = 25) -> dict:
    """Parse Cholec80 `videoXX-phase.txt` (per frame @25 fps) and `videoXX-tool.txt` (@1 fps)."""
    phases = []
    with open(phase_file) as f:
        next(f)
        for line in f:
            frame, ph = line.split()
            if int(frame) % fps == 0:
                phases.append(CHOLEC80_PHASES.index(ph))
    tools = []
    with open(tool_file) as f:
        next(f)
        for line in f:
            tools.append([int(x) for x in line.split()[1:]])
    T = min(len(phases), len(tools))
    return {"phase": np.array(phases[:T]), "tools": np.array(tools[:T], np.int8)}


class SurgicalEventCodec(Codec):
    modality, rep, reversible = "video", "event", True

    def __init__(self, lexicon: Lexicon):
        self.lex = lexicon

    def alphabet_size(self) -> int:
        return len(VIDEO_EVENTS) + RANGES["dt"].size

    def raw_bits(self, seg: Segment) -> float:
        T = len(seg.data["phase"])
        return T * (3 + 7)  # 3-bit phase + 7 tool bits per second

    def encode(self, seg: Segment) -> TokenSpan:
        ph, tools = np.asarray(seg.data["phase"]), np.asarray(seg.data["tools"])
        ids, t = [], []
        prev_sec, prev_ph, prev_tools = None, None, np.zeros(tools.shape[1], np.int8)
        for s in range(len(ph)):
            evs = []
            if ph[s] != prev_ph:
                evs.append(f"phase:{CHOLEC80_PHASES[ph[s]]}")
            for j in np.flatnonzero(tools[s] != prev_tools):
                evs.append(f"{'on' if tools[s, j] else 'off'}:{CHOLEC80_TOOLS[j]}")
            if not evs:
                continue
            tm = seg.t_start + s / 60
            if prev_sec is not None:
                ids.append(dt_token((s - prev_sec) / 60))
                t.append(tm)
            for e in evs:
                ids.append(self.lex.event_id(e))
                t.append(tm)
            prev_sec, prev_ph, prev_tools = s, ph[s], tools[s]
        return TokenSpan.make(ids, "event", t=t, meta={"n_sec": len(ph), "t0": seg.t_start})

    def decode(self, span: TokenSpan) -> dict:
        T, t0 = span.meta["n_sec"], span.meta["t0"]
        ph = np.zeros(T, np.int64)
        tools = np.zeros((T, len(CHOLEC80_TOOLS)), np.int8)
        cur_ph, cur_tools, last = 0, np.zeros(len(CHOLEC80_TOOLS), np.int8), 0
        for gid, tm in zip(span.ids, span.t):
            if int(gid) not in RANGES["event"]:
                continue
            s = int(round((float(tm) - t0) * 60))
            ph[last:s], tools[last:s] = cur_ph, cur_tools
            last = s
            kind, name = self.lex.events[RANGES["event"].to_local(int(gid))].split(":")
            if kind == "phase":
                cur_ph = CHOLEC80_PHASES.index(name)
            else:
                cur_tools = cur_tools.copy()
                cur_tools[CHOLEC80_TOOLS.index(name)] = 1 if kind == "on" else 0
        ph[last:], tools[last:] = cur_ph, cur_tools
        return {"phase": ph, "tools": tools}


class VideoFrameVQCodec(Codec):
    modality, rep, reversible = "video", "vq", False

    def __init__(self, frame_model, grid: int = 7):
        if frame_model.vq.n_codes > RANGES["video_vq"].size:
            raise ValueError("frame codebook larger than video_vq range")
        self.model, self.grid = frame_model.eval(), grid

    def alphabet_size(self) -> int:
        return self.model.vq.n_codes + 1

    def raw_bits(self, seg: Segment) -> float:
        return np.asarray(seg.data).size * 8

    @torch.no_grad()
    def encode(self, seg: Segment) -> TokenSpan:
        frames = np.asarray(seg.data)  # (T,28,28,3) uint8
        fps = seg.meta.get("fps", 1.0)
        codes = self.model.vq.codes(self.model.encode_latent(self.model.to_input(torch.as_tensor(frames)))).numpy()
        ids, t, pos = [], [], []
        for f in range(len(frames)):
            tm = seg.t_start + f / fps / 60
            ids.append(SPECIAL["<frame>"])
            t.append(tm)
            pos.append([f, -1, -1])
            for r in range(self.grid):
                for c in range(self.grid):
                    ids.append(RANGES["video_vq"].to_global(int(codes[f, r, c])))
                    t.append(tm)
                    pos.append([f, r, c])
        return TokenSpan.make(ids, "vq", t=t, pos=pos, meta={"frames": len(frames), "grid": self.grid})

    @torch.no_grad()
    def decode(self, span: TokenSpan) -> np.ndarray:
        g = self.grid
        loc = [RANGES["video_vq"].to_local(int(i)) for i in span.ids if int(i) in RANGES["video_vq"]]
        idx = torch.as_tensor(np.array(loc).reshape(-1, g, g))
        return self.model.to_uint8(self.model.decode_codes(idx)).numpy()


def synthetic_cholec80(seconds: int = 1800, seed: int = 0) -> dict:
    """Plausible Cholec80-shaped annotation (phases in order, tools toggling)."""
    rng = np.random.default_rng(seed)
    cuts = np.sort(rng.choice(np.arange(60, seconds - 60), len(CHOLEC80_PHASES) - 1, replace=False))
    ph = np.searchsorted(cuts, np.arange(seconds), side="right")
    tools = np.zeros((seconds, len(CHOLEC80_TOOLS)), np.int8)
    for j in range(len(CHOLEC80_TOOLS)):
        s = 0
        while s < seconds:
            on = rng.integers(5, 120)
            off = rng.integers(10, 300) * (1 if j == 0 else 3)
            tools[s:s + on, j] = 1
            s += on + off
    return {"phase": ph, "tools": tools}
