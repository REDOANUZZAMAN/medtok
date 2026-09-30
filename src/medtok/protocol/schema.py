"""Unified sample schema and token-sequence container.

A *Sample* is one clinical case: an id, optional labels, and a list of
*Segments*. Each segment carries raw data for exactly one modality plus
timing / provenance metadata. Codecs turn a segment into a *TokenSpan*; the
serializer concatenates spans into one *TokenSequence*.

A TokenSequence is the protocol's wire format: parallel int arrays of equal
length. Besides the token id, every position carries
    mod   modality index (MODALITIES), -1 for structural tokens
    t     absolute time in minutes since the sample's reference time (-1: n/a)
    pos   (3,) spatial index: image (row, col, 0); video (frame, row, col);
          point cloud (x, y, z) voxel; time-series (patch, 0, 0); -1: n/a
The side channels feed time / spatial positional encodings in the model, so
the sequence itself stays short (no coordinate tokens per patch).
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from .vocab import MODALITIES, PROTOCOL_VERSION


@dataclass
class Segment:
    modality: str                    # one of MODALITIES
    data: Any                        # str | np.ndarray | dict (codec-specific raw payload)
    t_start: float = 0.0             # minutes relative to Sample reference time
    source: str = ""                 # dataset / file of origin
    meta: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.modality not in MODALITIES:
            raise ValueError(f"unknown modality {self.modality!r}")


@dataclass
class Sample:
    sample_id: str
    segments: list[Segment]
    labels: dict = field(default_factory=dict)
    split: str = "train"
    meta: dict = field(default_factory=dict)

    def modalities(self) -> list[str]:
        return [s.modality for s in self.segments]


@dataclass
class TokenSpan:
    """Output of one codec for one segment (without span delimiters)."""
    ids: np.ndarray                  # (n,) int64 global ids
    t: np.ndarray                    # (n,) float32 minutes, -1 if n/a
    pos: np.ndarray                  # (n, 3) int32, -1 if n/a
    rep: str                         # "bpe" | "vq" | "event" | "cont"
    cont: np.ndarray | None = None   # (k, d) continuous vectors when rep == "cont"
    meta: dict = field(default_factory=dict)  # what the decoder needs (shape, scale...)

    def __post_init__(self):
        n = len(self.ids)
        self.ids = np.asarray(self.ids, dtype=np.int64)
        self.t = np.asarray(self.t, dtype=np.float32).reshape(n)
        self.pos = np.asarray(self.pos, dtype=np.int32).reshape(n, 3)

    @staticmethod
    def make(ids, rep, t=None, pos=None, **kw) -> "TokenSpan":
        n = len(ids)
        t = np.full(n, -1.0) if t is None else t
        pos = np.full((n, 3), -1) if pos is None else pos
        return TokenSpan(np.asarray(ids), np.asarray(t), np.asarray(pos), rep, **kw)


@dataclass
class TokenSequence:
    ids: np.ndarray
    mod: np.ndarray
    t: np.ndarray
    pos: np.ndarray
    spans: list[dict]                # [{modality, rep, start, end, meta}] -- start/end index into ids (payload only)
    cont: list[np.ndarray] = field(default_factory=list)  # continuous payloads, in <cont>-slot order
    sample_id: str = ""
    labels: dict = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.ids)

    def to_npz_dict(self) -> dict:
        return {
            "ids": self.ids, "mod": self.mod, "t": self.t, "pos": self.pos,
            "header": np.frombuffer(json.dumps({
                "protocol_version": PROTOCOL_VERSION,
                "sample_id": self.sample_id,
                "labels": self.labels,
                "spans": self.spans,
            }, default=_json_default).encode(), dtype=np.uint8),
            **{f"cont_{i}": c for i, c in enumerate(self.cont)},
        }

    @classmethod
    def from_npz_dict(cls, d) -> "TokenSequence":
        h = json.loads(bytes(d["header"]).decode())
        if h["protocol_version"] != PROTOCOL_VERSION:
            raise ValueError("protocol version mismatch")
        conts = sorted((k for k in d.keys() if k.startswith("cont_")), key=lambda k: int(k[5:]))
        return cls(np.asarray(d["ids"]), np.asarray(d["mod"]), np.asarray(d["t"]), np.asarray(d["pos"]),
                   h["spans"], [np.asarray(d[k]) for k in conts], h["sample_id"], h["labels"])

    def save(self, path) -> None:
        np.savez_compressed(path, **self.to_npz_dict())

    @classmethod
    def load(cls, path) -> "TokenSequence":
        with np.load(path) as d:
            return cls.from_npz_dict(dict(d))


def _json_default(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))


def span_meta_json(meta: dict) -> dict:
    """Round-trip a span meta dict through JSON so it's safe for the header."""
    return json.loads(json.dumps(meta, default=_json_default))


def sample_to_manifest(s: Sample) -> dict:
    """Lightweight JSON view of a sample (no raw arrays) for the dataset manifest."""
    return {
        "sample_id": s.sample_id, "split": s.split, "labels": s.labels, "meta": s.meta,
        "segments": [{k: v for k, v in asdict(seg).items() if k != "data"} |
                     {"shape": list(np.shape(seg.data)) if isinstance(seg.data, np.ndarray) else None}
                     for seg in s.segments],
    }
