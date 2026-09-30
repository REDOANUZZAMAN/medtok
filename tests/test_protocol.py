"""Vocabulary, schema, serialisation / deserialisation, MedicalToken view, shards."""
import numpy as np
import pytest

from medical_tokenizer.protocol.schema import Sample, Segment, TokenSequence, TokenSpan
from medical_tokenizer.protocol.serialize import (deserialize, drop_modality, serialize, serialize_spans,
                                                  split_spans, validate)
from medical_tokenizer.protocol.storage import load_shard, save_shard
from medical_tokenizer.protocol.tokens import TokenType, from_tokens, read_jsonl, to_tokens, token_type, write_jsonl
from medical_tokenizer.protocol.vocab import (BOS, CONT, EOS, MODALITIES, RANGES, SPAN_END, SPAN_START, SPECIAL,
                                              VOCAB_SIZE, Lexicon, bucket_to_dt, dt_to_bucket, range_of)


# ------------------------------------------------------------------ vocabulary
def test_ranges_disjoint_and_cover_vocab():
    rs = sorted(RANGES.values(), key=lambda r: r.start)
    assert rs[0].start == 0 and rs[-1].end == VOCAB_SIZE
    for a, b in zip(rs, rs[1:]):
        assert a.end == b.start


def test_required_special_tokens_exist():
    for name in ["<IMAGE_START>", "<IMAGE_END>", "<VIDEO_START>", "<VIDEO_END>", "<TIME_SERIES_START>",
                 "<TIME_SERIES_END>", "<TEXT_START>", "<TEXT_END>", "<POINT_CLOUD_START>", "<POINT_CLOUD_END>"]:
        assert name in SPECIAL
    assert set(SPAN_START) == set(MODALITIES) == set(SPAN_END)


def test_dt_buckets_exact_for_seconds_minutes_hours():
    for s in range(0, 60):
        assert bucket_to_dt(dt_to_bucket(s / 60)) == pytest.approx(s / 60)
    for m in range(1, 60):
        assert bucket_to_dt(dt_to_bucket(m)) == pytest.approx(m)
    for h in range(1, 49):
        assert bucket_to_dt(dt_to_bucket(h * 60)) == pytest.approx(h * 60)
    buckets = [dt_to_bucket(x) for x in np.geomspace(0.01, 40 * 24 * 60, 500)]
    assert buckets == sorted(buckets) and max(buckets) == 255


def test_range_conversion_errors():
    with pytest.raises(ValueError):
        RANGES["image_vq"].to_global(RANGES["image_vq"].size)
    with pytest.raises(ValueError):
        RANGES["image_vq"].to_local(0)
    with pytest.raises(ValueError):
        range_of(VOCAB_SIZE)


def test_lexicon_roundtrip(tmp_path, lex):
    lex.save(tmp_path / "lex.json")
    lex2 = Lexicon.load(tmp_path / "lex.json")
    assert lex2 == lex
    assert lex.id_to_str(lex.var_id("HR")) == "<VAR:HR>"
    assert lex.id_to_str(lex.segment_id(3)) == "<SEG:3>"
    assert lex.id_to_str(SPECIAL["<IMAGE_START>"]) == "<IMAGE_START>"


def test_lexicon_rejects_oversized():
    with pytest.raises(ValueError):
        Lexicon(variables=[f"v{i}" for i in range(300)])


# ------------------------------------------------------------------ serialisation
class DummyCodec:
    """Exact codec over small int arrays, for protocol-level tests."""

    def __init__(self, modality, rep="vq", rng_name="image_vq"):
        self.modality, self.rep, self.r = modality, rep, RANGES[rng_name]

    def encode(self, seg):
        a = np.asarray(seg.data)
        if self.rep == "continuous":
            return TokenSpan.make([CONT] * len(a), "continuous", t=[seg.t_start] * len(a),
                                  cont=np.stack([a, a * 2], 1).astype(np.float16))
        return TokenSpan.make(self.r.start + a, self.rep, t=[seg.t_start] * len(a),
                              pos=[[i, -1, -1] for i in range(len(a))], conf=np.linspace(0, 1, len(a)))

    def decode(self, span):
        if self.rep == "continuous":
            return span.cont[:, 0].astype(int)
        return span.ids - self.r.start


def make_sample():
    return Sample("s1", [Segment("timeseries", np.array([5, 6, 7]), 30.0),
                         Segment("image", np.array([1, 2, 3, 4]), 0.0),
                         Segment("video", np.array([9, 8]), 90.0)], labels={"y": 1})


CODECS = {"image": DummyCodec("image"), "timeseries": DummyCodec("timeseries", "continuous", "ts_vq"),
          "video": DummyCodec("video", "event", "segment")}


def test_serialize_layout_and_validate():
    seq = serialize(make_sample(), CODECS)
    validate(seq)
    assert seq.ids[0] == BOS and seq.ids[-1] == EOS
    # segments are sorted by time: image (t=0) first, then timeseries (30), then video (90)
    assert [sp["modality"] for sp in seq.spans] == ["image", "timeseries", "video"]
    first = seq.spans[0]
    assert seq.ids[first["start"] - 2] == SPECIAL["<IMAGE_START>"]
    assert int(seq.ids[seq.spans[1]["start"] - 3]) in RANGES["dt"]            # <DT> before the 2nd span
    assert seq.conf is not None and np.isnan(seq.conf[0])


def test_roundtrip_through_npz_and_deserialize(tmp_path):
    s = make_sample()
    seq = serialize(s, CODECS)
    seq.save(tmp_path / "x.npz")
    seq2 = TokenSequence.load(tmp_path / "x.npz")
    validate(seq2)
    np.testing.assert_array_equal(seq.ids, seq2.ids)
    np.testing.assert_array_equal(seq.pos, seq2.pos)
    out = dict(deserialize(seq2, CODECS))
    np.testing.assert_array_equal(out["image"], [1, 2, 3, 4])
    np.testing.assert_array_equal(out["timeseries"], [5, 6, 7])
    np.testing.assert_array_equal(out["video"], [9, 8])


def test_medical_token_view_roundtrip(tmp_path, lex):
    seq = serialize(make_sample(), CODECS)
    toks = to_tokens(seq, lex)
    assert toks[0].token_type == TokenType.SEQUENCE_BOUNDARY and toks[0].modality is None
    starts = [t for t in toks if t.token_type == TokenType.MODALITY_START]
    assert [t.modality for t in starts] == ["image", "timeseries", "video"]
    assert any(t.embedding is not None for t in toks)
    back = from_tokens(toks, seq.labels)
    validate(back)
    for f in ("ids", "mod", "t", "pos"):
        np.testing.assert_array_equal(getattr(back, f), getattr(seq, f))
    assert back.spans == seq.spans
    np.testing.assert_array_equal(back.cont[0], seq.cont[0])
    write_jsonl(seq, tmp_path / "x.jsonl", lex)
    j = read_jsonl(tmp_path / "x.jsonl")
    np.testing.assert_array_equal(j.ids, seq.ids)
    assert j.labels == {"y": 1}


def test_token_types_cover_every_id():
    kinds = {token_type(i) for i in range(VOCAB_SIZE)}
    assert TokenType.CODE in kinds and TokenType.SUBWORD in kinds and TokenType.CONTINUOUS in kinds


def test_shard_roundtrip(tmp_path):
    seqs = [serialize(make_sample(), CODECS) for _ in range(3)]
    seqs[1].sample_id = "other"
    save_shard(str(tmp_path / "s.npz"), seqs, {"k": 1})
    back, meta = load_shard(str(tmp_path / "s.npz"))
    assert meta == {"k": 1} and len(back) == 3 and back[1].sample_id == "other"
    for a, b in zip(seqs, back):
        np.testing.assert_array_equal(a.ids, b.ids)
        assert a.spans == b.spans
        np.testing.assert_array_equal(a.cont[0], b.cont[0])
        validate(b)


def test_drop_modality_keeps_valid_sequence():
    seq = serialize(make_sample(), CODECS)
    for m in ("image", "timeseries", "video"):
        out = drop_modality(seq, m)
        validate(out)
        assert m not in [sp["modality"] for sp in out.spans]
        assert int(out.ids[1]) not in RANGES["dt"]
    out = dict(deserialize(drop_modality(seq, "image"), CODECS))
    np.testing.assert_array_equal(out["video"], [9, 8])


def test_validate_rejects_corruption():
    seq = serialize(make_sample(), CODECS)
    seq.ids = seq.ids.copy()
    seq.ids[seq.spans[0]["end"]] = SPECIAL["<VIDEO_END>"]
    with pytest.raises(ValueError):
        validate(seq)


def test_protocol_version_mismatch(tmp_path):
    seq = serialize(make_sample(), CODECS)
    d = seq.to_npz_dict()
    import json
    h = json.loads(bytes(d["header"]).decode())
    h["protocol_version"] = "9.9"
    d["header"] = np.frombuffer(json.dumps(h).encode(), np.uint8)
    with pytest.raises(ValueError):
        TokenSequence.from_npz_dict(d)


def test_split_spans_inverse_of_layout():
    s = make_sample()
    spans = [CODECS[g.modality].encode(g) for g in s.segments]
    seq = serialize_spans(s, spans)
    got = split_spans(seq)
    order = sorted(range(3), key=lambda i: s.segments[i].t_start)
    for g, i in zip(got, order):
        np.testing.assert_array_equal(g.ids, spans[i].ids)
