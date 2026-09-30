"""Row-oriented view of the protocol: one MedicalToken per sequence position.

TokenSequence (schema.py) is the storage / model-input format (parallel
arrays). MedicalToken is the inspection / interchange format: every field of
one position in one typed record, serialisable to human-readable JSONL.
The two convert losslessly:

    tokens = to_tokens(seq, lexicon)          # list[MedicalToken]
    seq2   = from_tokens(tokens)              # == seq
    write_jsonl(seq, lexicon, "x.jsonl");  seq3 = read_jsonl("x.jsonl")

Span metadata (what a decoder needs, e.g. grid shape) travels on the span's
start token, so the JSONL file is self-describing.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from enum import Enum

import numpy as np

from .schema import TokenSequence, check_version, json_default
from .vocab import (MODALITIES, PROTOCOL_VERSION, RANGES, REP_MARK, SPAN_END, SPAN_START, SPECIAL_TOKENS,
                    Lexicon, range_of)


class TokenType(str, Enum):
    SEQUENCE_BOUNDARY = "sequence_boundary"   # <BOS> <EOS>
    MODALITY_START = "modality_start"         # <IMAGE_START> ...
    MODALITY_END = "modality_end"             # <IMAGE_END> ...
    REPRESENTATION = "representation"         # <REP:VQ> ...
    LAYOUT = "layout"                         # <ROW> <FRAME> <CLIP> <PATCH> <STATIC> <MISSING>
    SPECIAL = "special"                       # <PAD> <UNK> <SEP> <MASK>
    CONTINUOUS = "continuous"                 # <CONT> slot, vector in MedicalToken.embedding
    TIME_DELTA = "time_delta"                 # <DT:k>
    VARIABLE = "variable"                     # <VAR:name>
    VALUE_BIN = "value_bin"                   # <VBIN:k>
    EVENT = "event"                           # <EV:name>, annotation-derived
    SEGMENT = "segment"                       # <SEG:k>, unsupervised state
    SUBWORD = "subword"                       # text piece
    CODE = "code"                             # VQ codebook entry


_START = set(SPAN_START.values())
_END = set(SPAN_END.values())
_REP = set(REP_MARK.values())
_RANGE_TYPE = {"dt": TokenType.TIME_DELTA, "var": TokenType.VARIABLE, "vbin": TokenType.VALUE_BIN,
               "event": TokenType.EVENT, "segment": TokenType.SEGMENT, "text": TokenType.SUBWORD,
               "image_vq": TokenType.CODE, "ts_vq": TokenType.CODE, "video_vq": TokenType.CODE,
               "pc_vq": TokenType.CODE}


def token_type(gid: int) -> TokenType:
    r = range_of(gid)
    if r != "special":
        return _RANGE_TYPE[r]
    s = SPECIAL_TOKENS[gid]
    if s in ("<BOS>", "<EOS>"):
        return TokenType.SEQUENCE_BOUNDARY
    if s in _START:
        return TokenType.MODALITY_START
    if s in _END:
        return TokenType.MODALITY_END
    if s in _REP:
        return TokenType.REPRESENTATION
    if s == "<CONT>":
        return TokenType.CONTINUOUS
    if s in ("<ROW>", "<FRAME>", "<CLIP>", "<PATCH>", "<STATIC>", "<MISSING>"):
        return TokenType.LAYOUT
    return TokenType.SPECIAL


@dataclass
class MedicalToken:
    token_id: int                         # global id in the unified vocabulary
    token: str                            # human-readable form, e.g. "<IMAGE_VQ:17>"
    token_type: TokenType
    modality: str | None                  # None for <BOS>/<EOS>
    sample_id: str
    index: int                            # position in the sequence
    timestamp_min: float | None           # minutes since the sample's reference time
    position: tuple[int, int, int] | None # modality-specific spatial / structural index
    code: int | None = None               # local codebook / bin / state index
    embedding: list[float] | None = None  # continuous payload of a <CONT> slot
    confidence: float | None = None
    metadata: dict = field(default_factory=dict)   # span meta (on modality-start tokens)


def to_tokens(seq: TokenSequence, lex: Lexicon | None = None, text_decoder=None) -> list[MedicalToken]:
    lex = lex or Lexicon()
    start_meta = {sp["start"] - 2: {"rep": sp["rep"], "span_meta": sp["meta"]} for sp in seq.spans}
    # map every <CONT> position to its vector
    cont_at, ci = {}, 0
    for sp in seq.spans:
        if sp["rep"] == "continuous":
            where = [i for i in range(sp["start"], sp["end"]) if token_type(int(seq.ids[i])) == TokenType.CONTINUOUS]
            for j, i in enumerate(where):
                cont_at[i] = seq.cont[ci][j]
            ci += 1
    out = []
    for i, gid in enumerate(seq.ids.tolist()):
        tt = token_type(gid)
        r = RANGES[range_of(gid)]
        m = int(seq.mod[i])
        p = tuple(int(x) for x in seq.pos[i])
        c = None if seq.conf is None or np.isnan(seq.conf[i]) else float(seq.conf[i])
        out.append(MedicalToken(
            token_id=gid, token=lex.id_to_str(gid, text_decoder), token_type=tt,
            modality=MODALITIES[m] if m >= 0 else None, sample_id=seq.sample_id, index=i,
            timestamp_min=None if seq.t[i] < 0 else float(seq.t[i]),
            position=None if p == (-1, -1, -1) else p,
            code=gid - r.start if r.name != "special" else None,
            embedding=cont_at[i].astype(float).tolist() if i in cont_at else None,
            confidence=c, metadata=start_meta.get(i, {})))
    return out


def from_tokens(tokens: list[MedicalToken], labels: dict | None = None) -> TokenSequence:
    """Rebuild the columnar sequence; spans are re-derived from the boundary tokens."""
    n = len(tokens)
    ids = np.array([t.token_id for t in tokens], np.int64)
    mod = np.array([-1 if t.modality is None else MODALITIES.index(t.modality) for t in tokens], np.int8)
    t = np.array([-1.0 if x.timestamp_min is None else x.timestamp_min for x in tokens], np.float32)
    pos = np.array([(-1, -1, -1) if x.position is None else x.position for x in tokens], np.int32).reshape(n, 3)
    conf = None
    if any(x.confidence is not None for x in tokens):
        conf = np.array([np.nan if x.confidence is None else x.confidence for x in tokens], np.float32)
    spans, conts = [], []
    for i, tok in enumerate(tokens):
        if tok.token_type != TokenType.MODALITY_START:
            continue
        end = next(j for j in range(i + 1, n) if tokens[j].token_type == TokenType.MODALITY_END)
        rep = tok.metadata["rep"]
        spans.append({"modality": tok.modality, "rep": rep, "start": i + 2, "end": end,
                      "meta": tok.metadata.get("span_meta", {})})
        if rep == "continuous":
            conts.append(np.array([tokens[j].embedding for j in range(i + 2, end)
                                   if tokens[j].embedding is not None], np.float16))
    sid = tokens[0].sample_id if tokens else ""
    return TokenSequence(ids, mod, t, pos, spans, conts, sid, labels or {}, conf)


def write_jsonl(seq: TokenSequence, path, lex: Lexicon | None = None, text_decoder=None) -> None:
    with open(path, "w") as f:
        f.write(json.dumps({"protocol_version": PROTOCOL_VERSION, "sample_id": seq.sample_id,
                            "labels": seq.labels, "length": len(seq)}, default=json_default) + "\n")
        for tok in to_tokens(seq, lex, text_decoder):
            d = asdict(tok)
            d["token_type"] = tok.token_type.value
            f.write(json.dumps(d, default=json_default) + "\n")


def read_jsonl(path) -> TokenSequence:
    with open(path) as f:
        head = json.loads(f.readline())
        check_version(head["protocol_version"])
        toks = []
        for line in f:
            d = json.loads(line)
            d["token_type"] = TokenType(d["token_type"])
            d["position"] = None if d["position"] is None else tuple(d["position"])
            toks.append(MedicalToken(**d))
    return from_tokens(toks, head["labels"])
