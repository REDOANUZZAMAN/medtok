"""Unified token vocabulary.

All modalities share one flat id space, split into fixed, non-overlapping ranges.
Ranges are fixed at protocol level (not learned) so a model trained on one
checkpoint of the toolkit can read sequences from another as long as the
protocol version matches.

    [0, 256)            structural / special tokens (defined below)
    [256, 512)          time-delta tokens  <dt:k>   (exact min / exact h / log-spaced)
    [512, 768)          time-series variable tokens <var:NAME>
    [768, 1024)         value-bin tokens <vbin:k>   (per-variable quantiles)
    [1024, 1280)        surgical-video event tokens <ev:...>
    [1280, 9472)        text BPE sub-words          (8192)
    [9472, 10496)       image VQ codebook           (1024)
    [10496, 11008)      time-series VQ codebook     (512)
    [11008, 12032)      video VQ codebook           (1024, interface only)
    [12032, 13056)      point-cloud VQ codebook     (1024, interface only)

Codebooks smaller than their range simply leave the tail unused.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field

PROTOCOL_VERSION = "0.1"


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
            raise ValueError(f"{self.name}: global id {gid} outside range")
        return gid - self.start


RANGES: dict[str, Range] = {}
_cursor = 0
for _name, _size in [
    ("special", 256),
    ("dt", 256),
    ("var", 256),
    ("vbin", 256),
    ("event", 256),
    ("text", 8192),
    ("image_vq", 1024),
    ("ts_vq", 512),
    ("video_vq", 1024),
    ("pc_vq", 1024),
]:
    RANGES[_name] = Range(_name, _cursor, _size)
    _cursor += _size
VOCAB_SIZE = _cursor

MODALITIES = ("text", "image", "timeseries", "video", "pointcloud")

# Structural tokens. Order matters (ids are positional) -- append only.
SPECIAL_TOKENS = [
    "<pad>", "<bos>", "<eos>", "<unk>", "<sep>", "<mask>",
    # modality span delimiters
    "<txt>", "</txt>", "<img>", "</img>", "<ts>", "</ts>",
    "<vid>", "</vid>", "<pc>", "</pc>",
    # representation-type markers (which tokenizer produced the span)
    "<rep:bpe>", "<rep:vq>", "<rep:event>", "<rep:cont>",
    # layout markers
    "<row>",          # image: start of a new row of patch tokens
    "<frame>",        # video: start of a new frame / clip
    "<patch>",        # time-series: start of a new temporal patch
    "<missing>",      # time-series: patch or variable fully unobserved
    "<static>",       # start of static (non-temporal) attributes
    "<role:clin>",    # text role: clinician-written / templated note
    "<role:label>",   # text role: dataset label rendered as text
    "<cont>",         # placeholder: one continuous embedding slot
]
# Pad the reserved area with explicit reserved names so ids are stable.
SPECIAL_TOKENS += [f"<reserved:{i}>" for i in range(len(SPECIAL_TOKENS), RANGES["special"].size)]
SPECIAL = {tok: i for i, tok in enumerate(SPECIAL_TOKENS)}

SPAN_OPEN = {"text": "<txt>", "image": "<img>", "timeseries": "<ts>", "video": "<vid>", "pointcloud": "<pc>"}
SPAN_CLOSE = {m: t.replace("<", "</", 1) for m, t in SPAN_OPEN.items()}
REP_MARK = {"bpe": "<rep:bpe>", "vq": "<rep:vq>", "event": "<rep:event>", "cont": "<rep:cont>"}


# ---------------------------------------------------------------- time tokens
# 256 buckets for the gap between consecutive spans / events, in minutes:
#   0            same time
#   1..60        exact minutes (1-minute resolution up to 1 h)
#   61..107      exact hours 2 h .. 48 h (covers a full ICU observation window)
#   108..255     log-spaced from 48 h to 30 days
DT_MAX_MIN = 30 * 24 * 60
_DT_LOG_LO, _DT_LOG_N = 48 * 60, 148
_DT_EDGES = ([0.0] + [float(m) for m in range(1, 61)] + [h * 60.0 for h in range(2, 49)] +
             [_DT_LOG_LO * (DT_MAX_MIN / _DT_LOG_LO) ** ((k + 1) / _DT_LOG_N) for k in range(_DT_LOG_N)])
assert len(_DT_EDGES) == 256


def dt_to_bucket(minutes: float) -> int:
    if minutes <= 0.5:
        return 0
    if minutes <= 60:
        return int(round(minutes))
    if minutes <= _DT_LOG_LO:
        return 59 + int(round(minutes / 60))
    k = math.ceil(math.log(minutes / _DT_LOG_LO) / math.log(DT_MAX_MIN / _DT_LOG_LO) * _DT_LOG_N) - 1
    return 108 + min(max(k, 0), _DT_LOG_N - 1)


def bucket_to_dt(bucket: int) -> float:
    """Representative gap for a bucket, in minutes (exact for buckets < 108)."""
    return _DT_EDGES[min(bucket, len(_DT_EDGES) - 1)]


def dt_token(minutes: float) -> int:
    return RANGES["dt"].to_global(dt_to_bucket(minutes))


# ------------------------------------------------------------- lexicon object
@dataclass
class Lexicon:
    """Named entries for the enumerable ranges (var / vbin / event).

    Codebook and BPE ranges are addressed by local index and don't need names.
    """
    variables: list[str] = field(default_factory=list)
    n_value_bins: int = 16
    events: list[str] = field(default_factory=list)

    def var_id(self, name: str) -> int:
        return RANGES["var"].to_global(self.variables.index(name))

    def vbin_id(self, k: int) -> int:
        if not 0 <= k < self.n_value_bins:
            raise ValueError(k)
        return RANGES["vbin"].to_global(k)

    def event_id(self, name: str) -> int:
        return RANGES["event"].to_global(self.events.index(name))

    def id_to_str(self, gid: int, text_decoder=None) -> str:
        for r in RANGES.values():
            if gid in r:
                loc = gid - r.start
                if r.name == "special":
                    return SPECIAL_TOKENS[loc]
                if r.name == "dt":
                    return f"<dt:{loc}>"
                if r.name == "var":
                    return f"<var:{self.variables[loc]}>" if loc < len(self.variables) else f"<var:?{loc}>"
                if r.name == "vbin":
                    return f"<vbin:{loc}>"
                if r.name == "event":
                    return f"<ev:{self.events[loc]}>" if loc < len(self.events) else f"<ev:?{loc}>"
                if r.name == "text" and text_decoder is not None:
                    return text_decoder(loc)
                return f"<{r.name}:{loc}>"
        raise ValueError(f"id {gid} outside vocabulary (size {VOCAB_SIZE})")

    def to_json(self) -> dict:
        return {
            "protocol_version": PROTOCOL_VERSION,
            "vocab_size": VOCAB_SIZE,
            "ranges": {k: {"start": r.start, "size": r.size} for k, r in RANGES.items()},
            "special_tokens": [t for t in SPECIAL_TOKENS if not t.startswith("<reserved")],
            "variables": self.variables,
            "n_value_bins": self.n_value_bins,
            "events": self.events,
            "dt_bucket_edges_min": _DT_EDGES,
        }

    def save(self, path) -> None:
        with open(path, "w") as f:
            json.dump(self.to_json(), f, indent=1)

    @classmethod
    def load(cls, path) -> "Lexicon":
        d = json.load(open(path))
        if d["protocol_version"] != PROTOCOL_VERSION:
            raise ValueError(f"protocol mismatch: file {d['protocol_version']} vs code {PROTOCOL_VERSION}")
        return cls(variables=d["variables"], n_value_bins=d["n_value_bins"], events=d["events"])


def range_of(gid: int) -> str:
    for r in RANGES.values():
        if gid in r:
            return r.name
    raise ValueError(gid)
