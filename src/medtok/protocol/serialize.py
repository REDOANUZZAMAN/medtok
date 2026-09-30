"""Sample <-> TokenSequence serialization.

Layout of one serialized sample:

    <bos>
      <txt> <rep:bpe> ...text ids... </txt>
      <dt:k> <img> <rep:vq> <row> v v v <row> v v v ... </img>
      <dt:k> <ts> <rep:vq> <static> ... <patch> q q q q <patch> ... </ts>
    <eos>

Segments are ordered by t_start; a <dt:k> token before each span encodes the
gap since the previous segment (log-bucketed). Absolute time is also carried
in the `t` side channel.
"""
from __future__ import annotations

import numpy as np

from .schema import Sample, TokenSequence, TokenSpan, span_meta_json
from .vocab import MODALITIES, REP_MARK, SPAN_CLOSE, SPAN_OPEN, SPECIAL, dt_token

MOD_IDX = {m: i for i, m in enumerate(MODALITIES)}


def serialize(sample: Sample, codecs: dict[str, "Codec"]) -> TokenSequence:  # noqa: F821
    """Encode every segment with its modality's codec and lay the spans out."""
    encoded = []
    for seg in sample.segments:
        codec = codecs.get(seg.modality)
        if codec is None:
            raise KeyError(f"no codec registered for modality {seg.modality!r}")
        encoded.append(codec.encode(seg))
    return serialize_spans(sample, encoded)


def serialize_spans(sample: Sample, encoded: list[TokenSpan]) -> TokenSequence:
    """Layout step only: `encoded[i]` is the TokenSpan of `sample.segments[i]`.

    Lets callers encode many segments in one batched forward pass.
    """
    ids, mod, t, pos, spans, conts = [], [], [], [], [], []

    def push(tok_ids, m=-1, tt=None, pp=None):
        n = len(tok_ids)
        ids.append(np.asarray(tok_ids, np.int64))
        mod.append(np.full(n, m, np.int8))
        t.append(np.full(n, -1.0, np.float32) if tt is None else np.asarray(tt, np.float32))
        pos.append(np.full((n, 3), -1, np.int32) if pp is None else np.asarray(pp, np.int32))

    push([SPECIAL["<bos>"]])
    prev_t = None
    cursor = 1
    order = sorted(range(len(sample.segments)), key=lambda i: sample.segments[i].t_start)
    for i in order:
        seg, span = sample.segments[i], encoded[i]
        m = MOD_IDX[seg.modality]
        head = []
        if prev_t is not None:
            head.append(dt_token(seg.t_start - prev_t))
        head += [SPECIAL[SPAN_OPEN[seg.modality]], SPECIAL[REP_MARK[span.rep]]]
        push(head, m, np.full(len(head), seg.t_start))
        cursor += len(head)
        push(span.ids, m, span.t, span.pos)
        spans.append({"modality": seg.modality, "rep": span.rep, "start": cursor,
                      "end": cursor + len(span.ids), "meta": span_meta_json(span.meta)})
        cursor += len(span.ids)
        if span.cont is not None:
            conts.append(span.cont)
        push([SPECIAL[SPAN_CLOSE[seg.modality]]], m, [seg.t_start])
        cursor += 1
        prev_t = seg.t_start
    push([SPECIAL["<eos>"]])

    return TokenSequence(np.concatenate(ids), np.concatenate(mod), np.concatenate(t),
                         np.concatenate(pos), spans, conts, sample.sample_id, sample.labels)


def split_spans(seq: TokenSequence) -> list[TokenSpan]:
    """Recover per-segment TokenSpans from a sequence (inverse of the layout above)."""
    out, ci = [], 0
    for sp in seq.spans:
        s, e = sp["start"], sp["end"]
        cont = None
        if sp["rep"] == "cont":
            cont = seq.cont[ci]
            ci += 1
        out.append(TokenSpan(seq.ids[s:e], seq.t[s:e], seq.pos[s:e], sp["rep"], cont, sp["meta"]))
    return out


def deserialize(seq: TokenSequence, codecs: dict[str, "Codec"]) -> list[tuple[str, object]]:  # noqa: F821
    """Decode every span back to raw payloads: [(modality, data), ...]."""
    return [(sp["modality"], codecs[sp["modality"]].decode(span))
            for sp, span in zip(seq.spans, split_spans(seq))]


def validate(seq: TokenSequence) -> None:
    """Structural checks; raises AssertionError on a malformed sequence."""
    from .vocab import VOCAB_SIZE
    n = len(seq.ids)
    assert seq.mod.shape == (n,) and seq.t.shape == (n,) and seq.pos.shape == (n, 3)
    assert seq.ids[0] == SPECIAL["<bos>"] and seq.ids[-1] == SPECIAL["<eos>"]
    assert (seq.ids >= 0).all() and (seq.ids < VOCAB_SIZE).all()
    for sp in seq.spans:
        m = sp["modality"]
        assert seq.ids[sp["start"] - 1] == SPECIAL[REP_MARK[sp["rep"]]]
        assert seq.ids[sp["start"] - 2] == SPECIAL[SPAN_OPEN[m]]
        assert seq.ids[sp["end"]] == SPECIAL[SPAN_CLOSE[m]]
        assert (seq.mod[sp["start"]:sp["end"]] == MOD_IDX[m]).all()
