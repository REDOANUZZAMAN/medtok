"""Sample <-> TokenSequence layout.

    <BOS>
      <TEXT_START> <REP:SUBWORD> w w w ... <TEXT_END>
      <DT:k> <IMAGE_START> <REP:VQ> <ROW> v v v <ROW> v v v ... <IMAGE_END>
      <DT:k> <TIME_SERIES_START> <REP:EVENT> <STATIC> ... <TIME_SERIES_END>
    <EOS>

Segments are ordered by t_start (stable sort). A <DT:k> token before every span
but the first encodes the gap to the previous span; absolute time is also in
the `t` side channel.
"""
from __future__ import annotations

import numpy as np

from .schema import Sample, TokenSequence, TokenSpan, jsonable
from .vocab import BOS, EOS, MOD_INDEX, RANGES, REP_MARK, SPAN_END, SPAN_START, SPECIAL, VOCAB_SIZE, dt_token


def serialize(sample: Sample, codecs: dict) -> TokenSequence:
    """Encode every segment with its modality's codec, then lay the spans out."""
    spans = []
    for seg in sample.segments:
        if seg.modality not in codecs:
            raise KeyError(f"no codec registered for modality {seg.modality!r}")
        spans.append(codecs[seg.modality].encode(seg))
    return serialize_spans(sample, spans)


def serialize_spans(sample: Sample, encoded: list[TokenSpan]) -> TokenSequence:
    """Layout only: encoded[i] is the TokenSpan of sample.segments[i] (lets callers batch the encoding)."""
    ids, mod, t, pos, conf, spans, conts = [], [], [], [], [], [], []
    has_conf = any(s.conf is not None for s in encoded)

    def push(tok, m=-1, tt=None, pp=None, cc=None):
        n = len(tok)
        ids.append(np.asarray(tok, np.int64))
        mod.append(np.full(n, m, np.int8))
        t.append(np.full(n, -1.0, np.float32) if tt is None else np.asarray(tt, np.float32))
        pos.append(np.full((n, 3), -1, np.int32) if pp is None else np.asarray(pp, np.int32))
        conf.append(np.full(n, np.nan, np.float32) if cc is None else np.asarray(cc, np.float32))

    push([BOS])
    cursor, prev_t = 1, None
    for i in sorted(range(len(sample.segments)), key=lambda i: sample.segments[i].t_start):
        seg, span = sample.segments[i], encoded[i]
        m = MOD_INDEX[seg.modality]
        head = [] if prev_t is None else [dt_token(seg.t_start - prev_t)]
        head += [SPECIAL[SPAN_START[seg.modality]], SPECIAL[REP_MARK[span.rep]]]
        push(head, m, np.full(len(head), seg.t_start))
        cursor += len(head)
        push(span.ids, m, span.t, span.pos, span.conf)
        spans.append({"modality": seg.modality, "rep": span.rep, "start": cursor, "end": cursor + len(span.ids),
                      "meta": jsonable(span.meta)})
        cursor += len(span.ids)
        if span.cont is not None:
            conts.append(span.cont)
        push([SPECIAL[SPAN_END[seg.modality]]], m, [seg.t_start])
        cursor += 1
        prev_t = seg.t_start
    push([EOS])
    return TokenSequence(np.concatenate(ids), np.concatenate(mod), np.concatenate(t), np.concatenate(pos),
                         spans, conts, sample.sample_id, dict(sample.labels),
                         np.concatenate(conf) if has_conf else None)


def split_spans(seq: TokenSequence) -> list[TokenSpan]:
    """Recover the per-segment TokenSpans (inverse of the layout step)."""
    out, ci = [], 0
    for sp in seq.spans:
        s, e = sp["start"], sp["end"]
        cont = None
        if sp["rep"] == "continuous":
            cont, ci = seq.cont[ci], ci + 1
        conf = None if seq.conf is None else seq.conf[s:e]
        out.append(TokenSpan(seq.ids[s:e], seq.t[s:e], seq.pos[s:e], sp["rep"], cont, sp["meta"], conf))
    return out


def deserialize(seq: TokenSequence, codecs: dict) -> list[tuple[str, object]]:
    """Decode every span back to a raw payload: [(modality, data), ...] in sequence order."""
    return [(sp["modality"], codecs[sp["modality"]].decode(span)) for sp, span in zip(seq.spans, split_spans(seq))]


def validate(seq: TokenSequence) -> None:
    """Structural checks; raises ValueError on a malformed sequence."""
    n = len(seq.ids)

    def need(cond, msg):
        if not cond:
            raise ValueError(f"{seq.sample_id}: {msg}")

    need(seq.mod.shape == (n,) and seq.t.shape == (n,) and seq.pos.shape == (n, 3), "side-channel shapes")
    need(seq.conf is None or seq.conf.shape == (n,), "conf shape")
    need(n >= 2 and seq.ids[0] == BOS and seq.ids[-1] == EOS, "missing <BOS>/<EOS>")
    need(bool((seq.ids >= 0).all() and (seq.ids < VOCAB_SIZE).all()), "id outside vocabulary")
    n_cont = 0
    for sp in seq.spans:
        m, s, e = sp["modality"], sp["start"], sp["end"]
        need(seq.ids[s - 1] == SPECIAL[REP_MARK[sp["rep"]]], f"span {s}: missing rep marker")
        need(seq.ids[s - 2] == SPECIAL[SPAN_START[m]], f"span {s}: missing start token")
        need(seq.ids[e] == SPECIAL[SPAN_END[m]], f"span {s}: missing end token")
        need(bool((seq.mod[s - 2:e + 1] == MOD_INDEX[m]).all()), f"span {s}: modality channel mismatch")
        n_cont += sp["rep"] == "continuous"
    need(n_cont == len(seq.cont), "continuous payload count mismatch")


def drop_modality(seq: TokenSequence, modality: str) -> TokenSequence:
    """Remove every span of one modality (with its delimiters and preceding <DT>) -- missing-modality tests."""
    keep = np.ones(len(seq), bool)
    spans, conts, ci = [], [], 0
    for sp in seq.spans:
        is_cont = sp["rep"] == "continuous"
        if sp["modality"] == modality:
            a = sp["start"] - 2
            if a > 1 and int(seq.ids[a - 1]) in RANGES["dt"]:
                a -= 1
            keep[a:sp["end"] + 1] = False
        else:
            spans.append(dict(sp))
            if is_cont:
                conts.append(seq.cont[ci])
        ci += is_cont
    new_index = np.cumsum(keep) - 1
    for sp in spans:
        sp["start"], sp["end"] = int(new_index[sp["start"]]), int(new_index[sp["end"]])
    # the first remaining span must not start with a <DT> (it has no predecessor)
    ids = seq.ids[keep]
    out = TokenSequence(ids, seq.mod[keep], seq.t[keep], seq.pos[keep], spans, conts, seq.sample_id,
                        seq.labels, None if seq.conf is None else seq.conf[keep])
    if len(ids) > 1 and int(ids[1]) in RANGES["dt"]:
        out = _delete_positions(out, [1])
    return out


def _delete_positions(seq: TokenSequence, where: list[int]) -> TokenSequence:
    keep = np.ones(len(seq), bool)
    keep[where] = False
    shift = np.cumsum(~keep)
    spans = [dict(sp, start=int(sp["start"] - shift[sp["start"]]), end=int(sp["end"] - shift[sp["end"]]))
             for sp in seq.spans]
    return TokenSequence(seq.ids[keep], seq.mod[keep], seq.t[keep], seq.pos[keep], spans, seq.cont,
                         seq.sample_id, seq.labels, None if seq.conf is None else seq.conf[keep])
