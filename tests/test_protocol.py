import numpy as np
import pytest
import torch

from medtok.codecs.image import ImageContCodec, ImageVQCodec
from medtok.codecs.pointcloud import PointCloudVoxelCodec, chamfer, synthetic_organ_surface
from medtok.codecs.text import TextBPECodec
from medtok.codecs.timeseries import TSContCodec, TSEventCodec, TSVQCodec, ValueBinner, stay_segment
from medtok.codecs.video import SurgicalEventCodec, VideoFrameVQCodec, synthetic_cholec80
from medtok.data.physionet import STATIC_VARS, TS_IDX, TS_VARS, Normalizer, Stay, to_grid
from medtok.eval.downstream import UnifiedTransformer, collate, drop_modality
from medtok.protocol.schema import Sample, Segment, TokenSequence
from medtok.protocol.serialize import deserialize, serialize, validate
from medtok.protocol.vocab import RANGES, SPECIAL, VOCAB_SIZE, bucket_to_dt, dt_to_bucket
from medtok.quant.models import ImageVQVAE, TSPatchVQ
from medtok.toolkit import build_lexicon


# ------------------------------------------------------------------ fixtures
def _stay(seed=0, rid="100"):
    rng = np.random.default_rng(seed)
    obs = []
    for h in range(48):
        for v in ("HR", "MAP", "Temp"):
            obs.append((h * 60 + rng.integers(0, 60), TS_IDX[v], {"HR": 80, "MAP": 75, "Temp": 37}[v] + rng.normal(0, 3)))
        if h % 12 == 0:
            obs.append((h * 60 + 5, TS_IDX["Creatinine"], rng.uniform(0.5, 3)))
    static = {"Age": 60.0, "Gender": 1.0, "Height": np.nan, "ICUType": 3.0}
    return Stay(rid, static, np.array(sorted(obs), np.float32), int(seed % 2), "train")


@pytest.fixture(scope="module")
def stays():
    return [_stay(i, str(100 + i)) for i in range(20)]


@pytest.fixture(scope="module")
def lex():
    return build_lexicon()


@pytest.fixture(scope="module")
def binner(stays):
    return ValueBinner(16).fit(stays)


@pytest.fixture(scope="module")
def text_codec():
    return TextBPECodec.train(["Patient is a 60 y/o male admitted to the Medical ICU. GCS 15."] * 20, 300)


# ------------------------------------------------------------------- vocab
def test_ranges_disjoint_and_cover_vocab():
    rs = sorted(RANGES.values(), key=lambda r: r.start)
    assert rs[0].start == 0 and rs[-1].end == VOCAB_SIZE
    for a, b in zip(rs, rs[1:]):
        assert a.end == b.start


def test_dt_buckets_exact_for_minutes_and_hours():
    for m in list(range(1, 61)) + [h * 60 for h in range(2, 49)]:
        assert bucket_to_dt(dt_to_bucket(m)) == m
    assert dt_to_bucket(0) == 0
    assert dt_to_bucket(30 * 24 * 60) == 255
    b = [dt_to_bucket(x) for x in np.geomspace(1, 30 * 24 * 60, 500)]
    assert all(x <= y for x, y in zip(b, b[1:]))                     # monotone
    for x in np.geomspace(49 * 60, 29 * 24 * 60, 50):                  # log range: upper edge within 4%
        assert x <= bucket_to_dt(dt_to_bucket(x)) <= x * 1.04


def test_lexicon_roundtrip(tmp_path, lex):
    p = tmp_path / "lex.json"
    lex.save(p)
    lex2 = type(lex).load(p)
    assert lex2.variables == lex.variables and lex2.events == lex.events
    assert len(lex.variables) == len(STATIC_VARS) + len(TS_VARS) <= RANGES["var"].size
    assert lex.id_to_str(lex.var_id("HR")) == "<var:HR>"


# ------------------------------------------------------------------ codecs
def test_text_exact_roundtrip(text_codec):
    for s in ["Patient is a 60 y/o male.", "Unicode ok: 37.5 °C, µg/L – ± ✓", "   spaces\n\tand tabs  "]:
        span = text_codec.encode(Segment("text", s))
        assert all(int(i) in RANGES["text"] for i in span.ids)
        assert text_codec.decode(span) == s


def test_ts_event_roundtrip_matches_binned_grid(stays, lex, binner):
    c = TSEventCodec(lex, binner)
    s = stays[0]
    span = c.encode(stay_segment(s))
    out = c.decode(span)
    g, m = to_grid(s)
    assert (out["mask"] == m).all()
    for j, v in enumerate(TS_VARS):
        if m[:, j].any():
            np.testing.assert_allclose(out["grid"][m[:, j], j], binner.value(v, binner.bin(v, g[m[:, j], j])), rtol=1e-5)
    assert out["static"]["Age"] == pytest.approx(binner.value("Age", binner.bin("Age", 60.0)))
    assert np.isnan(out["static"]["Height"])


def test_ts_patch_codecs_shapes(stays, lex, binner):
    model = TSPatchVQ(n_codes=64)
    norm = Normalizer().fit(stays)
    for C in (TSVQCodec, TSContCodec):
        c = C(lex, binner, model, norm)
        span = c.encode(stay_segment(stays[1]))
        out = c.decode(span)
        assert out["grid"].shape == (48, len(TS_VARS))
    vq = TSVQCodec(lex, binner, model, norm).encode(stay_segment(stays[1]))
    assert (vq.ids == SPECIAL["<patch>"]).sum() == 12
    assert sum(int(i) in RANGES["ts_vq"] for i in vq.ids) == 48


def test_image_codecs():
    model = ImageVQVAE(n_codes=32, dim=8, hidden=16)
    img = np.random.default_rng(0).integers(0, 255, (28, 28, 3), dtype=np.uint8)
    vq = ImageVQCodec(model).encode(Segment("image", img))
    assert len(vq.ids) == 56 and (vq.ids == SPECIAL["<row>"]).sum() == 7
    assert ImageVQCodec(model).decode(vq).shape == (28, 28, 3)
    ct = ImageContCodec(model).encode(Segment("image", img))
    assert ct.cont.shape == (49, 8) and ImageContCodec(model).decode(ct).dtype == np.uint8


def test_surgical_event_exact_roundtrip(lex):
    ann = synthetic_cholec80(900, seed=3)
    c = SurgicalEventCodec(lex)
    span = c.encode(Segment("video", ann))
    out = c.decode(span)
    assert (out["phase"] == ann["phase"]).all() and (out["tools"] == ann["tools"]).all()
    assert c.raw_bits(Segment("video", ann)) > c.span_bits(span)


def test_video_frame_vq():
    model = ImageVQVAE(n_codes=32, dim=8, hidden=16)
    frames = np.zeros((3, 28, 28, 3), np.uint8)
    c = VideoFrameVQCodec(model)
    span = c.encode(Segment("video", frames, meta={"fps": 1}))
    assert (span.ids == SPECIAL["<frame>"]).sum() == 3 and all(int(i) in RANGES["video_vq"] for i in span.ids if i != SPECIAL["<frame>"])
    assert c.decode(span).shape == frames.shape


def test_pointcloud_codec():
    clouds = [synthetic_organ_surface(1024, seed=i) for i in range(6)]
    c = PointCloudVoxelCodec(res=8).fit(clouds, k=64)
    span = c.encode(Segment("pointcloud", clouds[0]))
    rec = c.decode(span)
    assert rec.shape == clouds[0].shape
    extent = float((clouds[0].max(0) - clouds[0].min(0)).max())
    assert chamfer(clouds[0], rec) < 0.05 * extent


# ------------------------------------------------------------ serialization
def _sample(stays, text):
    return Sample("s0", [Segment("text", text, 0.0), stay_segment(stays[0])], labels={"death": 1})


def test_serialize_validate_deserialize(tmp_path, stays, lex, binner, text_codec):
    codecs = {"text": text_codec, "timeseries": TSEventCodec(lex, binner)}
    text = "Patient is a 60 y/o male."
    seq = serialize(_sample(stays, text), codecs)
    validate(seq)
    assert seq.ids[0] == SPECIAL["<bos>"] and seq.ids[-1] == SPECIAL["<eos>"]
    dec = deserialize(seq, codecs)
    assert dec[0] == ("text", text)
    seq.save(tmp_path / "s.npz")
    seq2 = TokenSequence.load(tmp_path / "s.npz")
    assert (seq2.ids == seq.ids).all() and seq2.spans == seq.spans and seq2.labels == {"death": 1}


def test_drop_modality_keeps_valid_sequence(stays, lex, binner, text_codec):
    codecs = {"text": text_codec, "timeseries": TSEventCodec(lex, binner)}
    seq = serialize(_sample(stays, "Patient is a 60 y/o male."), codecs)
    for m in ("text", "timeseries"):
        d = drop_modality(seq, m)
        validate(d)
        assert [sp["modality"] for sp in d.spans] == [x for x in ("text", "timeseries") if x != m]


def test_unified_transformer_mixed_batch(stays, lex, binner, text_codec):
    model_ts = TSPatchVQ(n_codes=64)
    norm = Normalizer().fit(stays)
    seqs = [serialize(_sample(stays, "note"), {"text": text_codec, "timeseries": C(lex, binner, model_ts, norm)})
            for C in (TSVQCodec, TSContCodec)]
    seqs.append(serialize(_sample(stays, "note"), {"text": text_codec, "timeseries": TSEventCodec(lex, binner)}))
    b = collate(seqs, "death")
    assert b["cont_mask"].sum() == 48
    m = UnifiedTransformer({"mortality": 2}, {"timeseries": 16}, d=32, layers=1, heads=2)
    out = m(b, "mortality")
    assert out.shape == (3, 2) and torch.isfinite(out).all()
