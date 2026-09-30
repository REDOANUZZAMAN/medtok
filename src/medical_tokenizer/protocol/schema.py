"""Unified sample schema and the columnar token-sequence container.

Sample       one case: id, labels, split and a list of Segments.
Segment      raw data of exactly one modality plus timing and provenance.
TokenSpan    what a codec produces for one Segment (payload only, no delimiters).
TokenSequence  the protocol's wire format: parallel arrays of equal length L

    ids   (L,)   int64    global token id in [0, VOCAB_SIZE)
    mod   (L,)   int8     modality index (vocab.MODALITIES), -1 for <BOS>/<EOS>
    t     (L,)   float32  time in minutes since the sample's reference time, -1 = n/a
    pos   (L,3)  int32    modality-specific index, -1 = n/a (see docs/token_protocol.md):
                            text (i,-1,-1)  image (row,col,-1)  video (t',row,col)
                            time-series (window,group,-1)  point cloud (x,y,z) voxel
    conf  (L,)   float32  optional confidence, NaN = n/a (None if no token has one)

plus `spans` (one dict per segment) and `cont` (continuous payloads, one array
per continuous span). A row-oriented view (one MedicalToken per position) is
in protocol/tokens.py; both views convert losslessly.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from .vocab import MODALITIES, PROTOCOL_VERSION, REPRESENTATIONS


@dataclass
class Segment:
    modality: str                    # one of MODALITIES
    data: Any                        # modality-specific raw payload, see docs/token_protocol.md
    t_start: float = 0.0             # minutes relative to the Sample's reference time
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


@dataclass
class TokenSpan:
    ids: np.ndarray                  # (n,) int64
    t: np.ndarray                    # (n,) float32
    pos: np.ndarray                  # (n,3) int32
    rep: str                         # one of REPRESENTATIONS
    cont: np.ndarray | None = None   # (k,d) float16 when rep == "continuous"
    meta: dict = field(default_factory=dict)   # whatever the decoder needs (shapes, scale, ...)
    conf: np.ndarray | None = None   # (n,) float32

    def __post_init__(self):
        n = len(self.ids)
        self.ids = np.asarray(self.ids, np.int64).reshape(n)
        self.t = np.asarray(self.t, np.float32).reshape(n)
        self.pos = np.asarray(self.pos, np.int32).reshape(n, 3)
        if self.conf is not None:
            self.conf = np.asarray(self.conf, np.float32).reshape(n)
        if self.rep not in REPRESENTATIONS:
            raise ValueError(f"unknown representation {self.rep!r}")

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
    spans: list[dict]                # [{modality, rep, start, end, meta}]; ids[start:end] is the payload
    cont: list[np.ndarray] = field(default_factory=list)
    sample_id: str = ""
    labels: dict = field(default_factory=dict)
    conf: np.ndarray | None = None

    def __len__(self) -> int:
        return len(self.ids)

    def header(self) -> dict:
        return {"protocol_version": PROTOCOL_VERSION, "sample_id": self.sample_id,
                "labels": self.labels, "spans": self.spans}

    def to_npz_dict(self) -> dict:
        d = {"ids": self.ids, "mod": self.mod, "t": self.t, "pos": self.pos,
             "header": np.frombuffer(json.dumps(self.header(), default=json_default).encode(), np.uint8),
             **{f"cont_{i}": c for i, c in enumerate(self.cont)}}
        if self.conf is not None:
            d["conf"] = self.conf
        return d

    @classmethod
    def from_npz_dict(cls, d) -> "TokenSequence":
        h = json.loads(bytes(d["header"]).decode())
        check_version(h["protocol_version"])
        conts = sorted((k for k in d if k.startswith("cont_")), key=lambda k: int(k[5:]))
        return cls(np.asarray(d["ids"]), np.asarray(d["mod"]), np.asarray(d["t"]), np.asarray(d["pos"]),
                   h["spans"], [np.asarray(d[k]) for k in conts], h["sample_id"], h["labels"],
                   np.asarray(d["conf"]) if "conf" in d else None)

    def save(self, path) -> None:
        np.savez_compressed(path, **self.to_npz_dict())

    @classmethod
    def load(cls, path) -> "TokenSequence":
        with np.load(path) as d:
            return cls.from_npz_dict(dict(d))


def check_version(v: str) -> None:
    if v != PROTOCOL_VERSION:
        raise ValueError(f"protocol version mismatch: data {v} vs code {PROTOCOL_VERSION}")


def json_default(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))


def jsonable(meta: dict) -> dict:
    """Round-trip a dict through JSON so it can live in a sequence header."""
    return json.loads(json.dumps(meta, default=json_default))


def sample_manifest(s: Sample) -> dict:
    """JSON view of a sample without raw payloads (for dataset manifests)."""
    return {
        "sample_id": s.sample_id, "split": s.split, "labels": s.labels, "meta": s.meta,
        "segments": [{k: v for k, v in asdict(seg).items() if k != "data"}
                     | {"shape": list(np.shape(seg.data)) if isinstance(seg.data, np.ndarray) else None}
                     for seg in s.segments],
    }
