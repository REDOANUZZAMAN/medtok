"""Unified token vocabulary (protocol v0.2).

All modalities share one flat id space split into fixed, non-overlapping
ranges. The ranges are part of the protocol, not learned, so any model trained
on protocol v0.2 sequences can read sequences produced by any v0.2 tokenizer
checkpoint. A codebook smaller than its range leaves the tail unused.

    [0, 256)        special / structural tokens (SPECIAL_TOKENS below)
    [256, 512)      time-delta tokens <DT:k> (exact seconds, minutes, hours, then log-spaced)
    [512, 768)      time-series variable tokens <VAR:name>
    [768, 1024)     value-bin tokens <VBIN:k> (per-variable quantile bins)
    [1024, 1280)    annotated event tokens <EV:name> (from dataset annotations only)
    [1280, 1536)    unsupervised segment-state tokens <SEG:k> (no semantic meaning)
    [1536, 9728)    text sub-words (8192)
    [9728, 10752)   image VQ codebook (1024)
    [10752, 11776)  time-series VQ codebook (1024)
    [11776, 12800)  video VQ codebook (1024)
    [12800, 13824)  point-cloud codebook (1024)
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field

PROTOCOL_VERSION = "0.2"


@dataclass(frozen=True)
class Range:
    name: str
    start: int
    size: int

    @property
    def end(self) -> int:
        return self.start + self.size

    def __contains__(self, i: int) -> bool:
        return self.start <= i < self.end

    def to_global(self, local: int) -> int:
        if not 0 <= local < self.size:
            raise ValueError(f"{self.name}: local id {local} outside [0,{self.size})")
        return self.start + local

    def to_local(self, gid: int) -> int:
        if gid not in self:
            raise ValueError(f"{self.name}: global id {gid} outside [{self.start},{self.end})")
        return gid - self.start


_LAYOUT = [
    ("special", 256), ("dt", 256), ("var", 256), ("vbin", 256), ("event", 256), ("segment", 256),
    ("text", 8192), ("image_vq", 1024), ("ts_vq", 1024), ("video_vq", 1024), ("pc_vq", 1024),
]
RANGES: dict[str, Range] = {}
_cursor = 0
for _name, _size in _LAYOUT:
    RANGES[_name] = Range(_name, _cursor, _size)
    _cursor += _size
VOCAB_SIZE = _cursor

# Modality index = position in this tuple (used in the `mod` side channel).
MODALITIES = ("text", "image", "timeseries", "video", "pointcloud")
MOD_INDEX = {m: i for i, m in enumerate(MODALITIES)}
# Representation types a codec can produce.
REPRESENTATIONS = ("subword", "vq", "event", "continuous")

SPAN_START = {"text": "<TEXT_START>", "image": "<IMAGE_START>", "timeseries": "<TIME_SERIES_START>",
              "video": "<VIDEO_START>", "pointcloud": "<POINT_CLOUD_START>"}
SPAN_END = {m: t.replace("_START>", "_END>") for m, t in SPAN_START.items()}
REP_MARK = {r: f"<REP:{r.upper()}>" for r in REPRESENTATIONS}

# Structural tokens. Ids are positional, so this list is append-only.
SPECIAL_TOKENS = [
    "<PAD>", "<BOS>", "<EOS>", "<UNK>", "<SEP>", "<MASK>",
    *[t for m in MODALITIES for t in (SPAN_START[m], SPAN_END[m])],
    *[REP_MARK[r] for r in REPRESENTATIONS],
    "<ROW>",      # image: start of a row of latent patches
    "<FRAME>",    # video: start of a frame
    "<CLIP>",     # video: start of a clip (group of frames encoded together)
    "<PATCH>",    # time-series: start of a temporal window
    "<MISSING>",  # time-series: window with no observation at all
    "<STATIC>",   # time-series: start of static (non-temporal) attributes
    "<CONT>",     # placeholder for one continuous vector (payload in TokenSequence.cont)
]
SPECIAL_TOKENS += [f"<RESERVED:{i}>" for i in range(len(SPECIAL_TOKENS), RANGES["special"].size)]
SPECIAL = {tok: i for i, tok in enumerate(SPECIAL_TOKENS)}
PAD, BOS, EOS, CONT = SPECIAL["<PAD>"], SPECIAL["<BOS>"], SPECIAL["<EOS>"], SPECIAL["<CONT>"]

# ---------------------------------------------------------------- time deltas
# 256 buckets for the gap between consecutive spans / events:
#   0          no gap
#   1..59      1..59 s, exact
#   60..118    1..59 min, exact
#   119..166   1..48 h, exact (a full ICU observation window)
#   167..255   log-spaced from 48 h to 30 days
DT_MAX_MIN = 30 * 24 * 60.0
_LOG_LO, _LOG_N = 48 * 60.0, 89
_DT_EDGES_MIN = ([0.0] + [s / 60 for s in range(1, 60)] + [float(m) for m in range(1, 60)]
                 + [h * 60.0 for h in range(1, 49)]
                 + [_LOG_LO * (DT_MAX_MIN / _LOG_LO) ** ((k + 1) / _LOG_N) for k in range(_LOG_N)])
assert len(_DT_EDGES_MIN) == 256


def dt_to_bucket(minutes: float) -> int:
    sec = minutes * 60
    if sec < 0.5:
        return 0
    if sec < 59.5:
        return int(round(sec))
    if minutes < 59.5:
        return 59 + max(int(round(minutes)), 1)
    if minutes < 48.5 * 60:
        return 118 + max(int(round(minutes / 60)), 1)
    k = math.ceil(math.log(minutes / _LOG_LO) / math.log(DT_MAX_MIN / _LOG_LO) * _LOG_N) - 1
    return 167 + min(max(k, 0), _LOG_N - 1)


def bucket_to_dt(bucket: int) -> float:
    """Representative gap in minutes (exact for whole seconds < 1 min, whole minutes < 1 h, whole hours <= 48 h)."""
    return _DT_EDGES_MIN[min(max(bucket, 0), 255)]


def dt_token(minutes: float) -> int:
    return RANGES["dt"].to_global(dt_to_bucket(minutes))


def range_of(gid: int) -> str:
    for r in RANGES.values():
        if gid in r:
            return r.name
    raise ValueError(f"id {gid} outside vocabulary (size {VOCAB_SIZE})")


# --------------------------------------------------------------- lexicon
@dataclass
class Lexicon:
    """Names for the enumerable ranges (var / vbin / event / segment).

    Codebook and sub-word ranges are addressed by local index and need no names.
    """
    variables: list[str] = field(default_factory=list)
    n_value_bins: int = 16
    events: list[str] = field(default_factory=list)
    n_segment_states: int = 64

    def __post_init__(self):
        for name, n in (("var", len(self.variables)), ("vbin", self.n_value_bins), ("event", len(self.events)),
                        ("segment", self.n_segment_states)):
            if n > RANGES[name].size:
                raise ValueError(f"{name}: {n} entries exceed range size {RANGES[name].size}")

    def var_id(self, name: str) -> int:
        return RANGES["var"].to_global(self.variables.index(name))

    def vbin_id(self, k: int) -> int:
        if not 0 <= k < self.n_value_bins:
            raise ValueError(f"value bin {k} outside [0,{self.n_value_bins})")
        return RANGES["vbin"].to_global(k)

    def event_id(self, name: str) -> int:
        return RANGES["event"].to_global(self.events.index(name))

    def segment_id(self, k: int) -> int:
        if not 0 <= k < self.n_segment_states:
            raise ValueError(f"segment state {k} outside [0,{self.n_segment_states})")
        return RANGES["segment"].to_global(k)

    def id_to_str(self, gid: int, text_decoder=None) -> str:
        r = RANGES[range_of(gid)]
        loc = gid - r.start
        if r.name == "special":
            return SPECIAL_TOKENS[loc]
        if r.name == "dt":
            return f"<DT:{loc}>"
        if r.name == "var":
            return f"<VAR:{self.variables[loc]}>" if loc < len(self.variables) else f"<VAR:?{loc}>"
        if r.name == "vbin":
            return f"<VBIN:{loc}>"
        if r.name == "event":
            return f"<EV:{self.events[loc]}>" if loc < len(self.events) else f"<EV:?{loc}>"
        if r.name == "segment":
            return f"<SEG:{loc}>"
        if r.name == "text" and text_decoder is not None:
            return text_decoder(loc)
        return f"<{r.name.upper()}:{loc}>"

    def to_json(self) -> dict:
        return {
            "protocol_version": PROTOCOL_VERSION,
            "vocab_size": VOCAB_SIZE,
            "ranges": {k: {"start": r.start, "size": r.size} for k, r in RANGES.items()},
            "modalities": list(MODALITIES),
            "representations": list(REPRESENTATIONS),
            "special_tokens": [t for t in SPECIAL_TOKENS if not t.startswith("<RESERVED")],
            "variables": self.variables,
            "n_value_bins": self.n_value_bins,
            "events": self.events,
            "n_segment_states": self.n_segment_states,
            "dt_bucket_minutes": _DT_EDGES_MIN,
        }

    def save(self, path) -> None:
        with open(path, "w") as f:
            json.dump(self.to_json(), f, indent=1)

    @classmethod
    def load(cls, path) -> "Lexicon":
        with open(path) as f:
            d = json.load(f)
        if d["protocol_version"] != PROTOCOL_VERSION:
            raise ValueError(f"protocol mismatch: file {d['protocol_version']} vs code {PROTOCOL_VERSION}")
        return cls(d["variables"], d["n_value_bins"], d["events"], d["n_segment_states"])
