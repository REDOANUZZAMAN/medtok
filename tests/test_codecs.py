"""Per-modality encode / decode, reversibility claims, positional encoding and metrics."""
import numpy as np
import pytest
import torch

from medical_tokenizer.codecs.base import MultimodalCodec
from medical_tokenizer.data.video import CATARACT101_PHASES, phase_at, synthetic_video
from medical_tokenizer.encoders.base import load_model, save_model
from medical_tokenizer.encoders.image import ImageAutoencoder
from medical_tokenizer.encoders.timeseries import TimeSeriesWindowAutoencoder
from medical_tokenizer.encoders.video import VideoClipAutoencoder
from medical_tokenizer.evaluation.compression import compression_summary, entropy_bits
from medical_tokenizer.evaluation.reconstruction import image_metrics, masked_ts_metrics
from medical_tokenizer.positional_encoding.unified import UnifiedPositionalEncoding
from medical_tokenizer.protocol.schema import Sample, Segment
from medical_tokenizer.protocol.serialize import validate
from medical_tokenizer.protocol.vocab import CONT, RANGES, SPECIAL
from medical_tokenizer.tokenizers.image.codec import ImageContinuousCodec, ImageVQCodec, extract_patches
from medical_tokenizer.tokenizers.pointcloud.codec import PointCloudTokenizer, chamfer, synthetic_surface
from medical_tokenizer.tokenizers.timeseries.codec import (TSContinuousCodec, TSEventCodec, TSVQCodec,
                                                           stay_segment)
from medical_tokenizer.tokenizers.video.codec import (AnnotatedPhaseEventCodec, VideoContinuousCodec,
                                                      VideoSegmentCodec, VideoVQCodec, make_clips)

VQ = {"type": "vq", "n_codes": 32}


# ------------------------------------------------------------------ text
def test_text_exact_roundtrip(text_tok):
    for s in ["Patient admitted with sepsis.", "Temp 38.5 °C; SpO₂ 92 % — unusual unicode", ""]:
        span = text_tok.encode(Segment("text", s))
        assert text_tok.decode(span) == s
        assert all(int(i) in RANGES["text"] for i in span.ids)
    assert text_tok.tokenize("Heart rate") and '"ids"' in text_tok.serialize("Heart rate")


# ------------------------------------------------------------------ image
@pytest.mark.parametrize("arch,p", [("conv", 4), ("conv", 2), ("patch", 7)])
def test_image_codecs(arch, p):
    m = ImageAutoencoder(patch_size=p, dim=8, hidden=16, arch=arch, quantizer=VQ).eval()
    vq, img = ImageVQCodec(m), np.random.default_rng(0).integers(0, 255, (28, 28, 3), dtype=np.uint8)
    g = 28 // p
    span = vq.encode(Segment("image", img))
    assert len(span.ids) == g * (g + 1) and (span.ids == SPECIAL["<ROW>"]).sum() == g
    rec = vq.decode(span)
    assert rec.shape == img.shape and rec.dtype == np.uint8
    np.testing.assert_array_equal(rec, vq.decode_codes(vq.codes(img[None]))[0])    # decode(tokens) == decode(ids)
    mc = ImageAutoencoder(patch_size=p, dim=8, hidden=16, arch=arch, quantizer=None).eval()
    cc = ImageContinuousCodec(mc)
    cs = cc.encode(Segment("image", img))
    assert (cs.ids == CONT).all() and cs.cont.shape == (g * g, 8)
    assert cc.decode(cs).shape == img.shape
    assert cc.payload_bits(cs) == g * g * 8 * 16
    assert extract_patches(img, p).shape == (g, g, p, p, 3)


def test_vq_codec_rejects_continuous_model():
    with pytest.raises(ValueError):
        ImageVQCodec(ImageAutoencoder(quantizer=None))


def test_model_save_load_roundtrip(tmp_path):
    m = ImageAutoencoder(dim=8, hidden=16, quantizer=VQ).eval()
    save_model(m, str(tmp_path / "m.pt"))
    m2, _ = load_model(str(tmp_path / "m.pt"))
    x = torch.randn(2, 3, 28, 28)
    torch.testing.assert_close(m.decode_codes(m.codes(x)), m2.decode_codes(m2.codes(x)))


# ------------------------------------------------------------------ time-series
def test_ts_event_obs_roundtrip_equals_binned_grid(stays, lex, binner, spec):
    ev = TSEventCodec(lex, binner, spec, "obs")
    for s in stays[:10]:
        span = ev.encode(stay_segment(s))
        out = ev.decode(span)
        g, m = spec.grid(s.obs)
        np.testing.assert_array_equal(out["mask"], m)
        for j, v in enumerate(spec.variables):
            if m[:, j].any():
                np.testing.assert_allclose(out["grid"][m[:, j], j], binner.value(v, binner.bin(v, g[m[:, j], j])))
        for k in ("Age", "Gender", "ICUType"):
            assert not np.isnan(out["static"][k])


def test_ts_event_change_mode_is_shorter_and_same_on_observed(stays, lex, binner, spec):
    obs_c, chg_c = TSEventCodec(lex, binner, spec, "obs"), TSEventCodec(lex, binner, spec, "change")
    for s in stays[:10]:
        seg = stay_segment(s)
        a, b = obs_c.encode(seg), chg_c.encode(seg)
        assert len(b.ids) <= len(a.ids)
        _, m = spec.grid(s.obs)
        np.testing.assert_allclose(chg_c.decode(b)["grid"][m], obs_c.decode(a)["grid"][m])


def test_ts_event_time_order_preserved(stays, lex, binner, spec):
    span = TSEventCodec(lex, binner, spec).encode(stay_segment(stays[0]))
    t = span.t[span.t >= 0]
    assert (np.diff(t) >= 0).all()


@pytest.mark.parametrize("window,stride", [(4, 4), (6, 3)])
def test_ts_window_codecs(stays, lex, binner, spec, norm, window, stride):
    m = TimeSeriesWindowAutoencoder(steps=48, window=window, stride=stride, groups=2, dim=4, hidden=32,
                                    quantizer=VQ).eval()
    W = m.n_windows
    vq = TSVQCodec(lex, binner, spec, m, norm)
    segs = [stay_segment(s) for s in stays[:4]]
    spans = vq.encode_batch(segs)
    n_static = len(vq.static_tokens(stays[0].static, 0)[0])
    assert len(spans[0].ids) == n_static + W * (1 + 2)
    out = vq.decode(spans[0])
    assert out["grid"].shape == (48, 37)
    mc = TimeSeriesWindowAutoencoder(steps=48, window=window, stride=stride, groups=2, dim=4, hidden=32).eval()
    cc = TSContinuousCodec(lex, binner, spec, mc, norm)
    cs = cc.encode(segs[0])
    assert cs.cont.shape == (W * 2, 4) and cc.decode(cs)["grid"].shape == (48, 37)


def test_ts_vq_missing_windows(lex, binner, spec, norm):
    m = TimeSeriesWindowAutoencoder(steps=48, window=4, groups=2, dim=4, hidden=32, quantizer=VQ).eval()
    vq = TSVQCodec(lex, binner, spec, m, norm, skip_empty=True)
    obs = np.array([[30.0, 14, 80.0], [35 * 60.0, 14, 90.0]], np.float32)        # only windows 0 and 8 observed
    seg = Segment("timeseries", {"obs": obs, "static": {"Age": 50.0}})
    span = vq.encode(seg)
    assert (span.ids == SPECIAL["<MISSING>"]).sum() == 10
    full = TSVQCodec(lex, binner, spec, m, norm, skip_empty=False)
    np.testing.assert_allclose(vq.decode(span)["grid"], full.decode(full.encode(seg))["grid"], atol=1e-5)


def test_window_overlap_add_identity():
    m = TimeSeriesWindowAutoencoder(steps=12, window=4, stride=2, groups=1, dim=4)
    x = torch.randn(2, 12, 37)
    w = m._windows(x)                                     # (B,W,window*V)
    # decoding the windows themselves through fold must return x (overlap-add of identical values)
    B = x.shape[0]
    cols = w.view(B, m.n_windows, m.window, m.n_vars).permute(0, 3, 2, 1).reshape(B, m.n_vars * m.window, m.n_windows)
    y = torch.nn.functional.fold(cols, (12, 1), (4, 1), stride=(2, 1))
    cnt = torch.nn.functional.fold(torch.ones(1, 4, m.n_windows), (12, 1), (4, 1), stride=(2, 1))
    torch.testing.assert_close((y / cnt).squeeze(-1).permute(0, 2, 1), x)


# ------------------------------------------------------------------ video
@pytest.fixture(scope="module")
def video_models():
    vq = VideoClipAutoencoder(frame_size=32, clip_len=8, patch_size=8, temporal_stride=4, dim=8, hidden=16,
                              quantizer=VQ).eval()
    ct = VideoClipAutoencoder(frame_size=32, clip_len=8, patch_size=8, temporal_stride=4, dim=8, hidden=16).eval()
    return vq, ct


def test_video_vq_and_continuous(video_models):
    vq_m, ct_m = video_models
    frames, _ = synthetic_video(32, 32)
    seg = Segment("video", {"frames": frames, "fps": 1.0})
    vq = VideoVQCodec(vq_m)
    span = vq.encode(seg)
    assert len(span.ids) == 4 * (1 + 2 * 4 * 4)                # 4 clips x (<CLIP> + t'=2 x 4 x 4)
    assert span.pos[:, 0].max() == 7                            # latent time index runs over the segment
    assert vq.decode(span)["frames"].shape == frames.shape
    cc = VideoContinuousCodec(ct_m)
    cs = cc.encode(seg)
    assert cs.cont.shape == (4 * 32, 8) and cc.decode(cs)["frames"].shape == frames.shape
    assert (np.diff(span.t) >= 0).all()


def test_video_segment_codec(video_models, lex):
    _, ct_m = video_models
    frames, _ = synthetic_video(64, 32)
    clips = make_clips(frames, 8)
    codec = VideoSegmentCodec(ct_m, lex).fit(clips, k=3)
    seg = Segment("video", {"frames": frames, "fps": 1.0})
    span = codec.encode(seg)
    states = codec.states(clips)
    np.testing.assert_array_equal(codec.states_from_span(span), states)      # state sequence exactly recovered
    n_runs = 1 + int((np.diff(states) != 0).sum())
    assert len(span.ids) == 2 * n_runs
    assert codec.decode(span)["frames"].shape == frames.shape


def test_annotated_phase_events_exact(lex):
    ph = np.array([-1] * 5 + [0] * 20 + [1] * 3 + [0] * 7 + [4] * 400 + [9] * 2)
    codec = AnnotatedPhaseEventCodec(lex, "cataract101", CATARACT101_PHASES)
    span = codec.encode(Segment("video", {"phase": ph, "fps": 1.0}))
    np.testing.assert_array_equal(codec.decode(span)["phase"], ph)
    assert (np.isin(span.ids, np.arange(RANGES["event"].start, RANGES["event"].end))).sum() == 6


def test_phase_at_semantics():
    starts = [(10, 0), (20, 1), (30, 1), (40, 5)]
    np.testing.assert_array_equal(phase_at(np.array([0, 9, 10, 25, 39, 40, 99]), starts), [-1, -1, 0, 1, 1, 5, 5])


# ------------------------------------------------------------------ point cloud
def test_pointcloud_tokenizer():
    train = [synthetic_surface(1024, seed=i) for i in range(5)]
    tok = PointCloudTokenizer(res=8).fit(train, k=32)
    pts = synthetic_surface(1024, seed=99)
    span = tok.encode(Segment("pointcloud", pts))
    rec = tok.decode(span)
    assert rec.shape == pts.shape
    assert chamfer(pts, rec) < 0.1 * np.ptp(pts, 0).max()
    assert all(int(i) in RANGES["pc_vq"] for i in span.ids)


# ------------------------------------------------------------------ positional encoding
def test_positional_encoding_modality_aware():
    for kind in ("learned", "sinusoidal"):
        pe = UnifiedPositionalEncoding(16, kind=kind)
        mod = torch.tensor([[1, 3, -1]])
        pos = torch.tensor([[[2, 3, -1], [2, 3, -1], [-1, -1, -1]]])
        t = torch.tensor([[0.0, 0.0, -1.0]])
        e = pe(mod, t, pos)
        assert e.shape == (1, 3, 16)
        assert not torch.allclose(e[0, 0], e[0, 1])          # same pos, different modality
        e2 = pe(mod, t, pos + torch.tensor([1, 0, 0]))
        assert not torch.allclose(e[0, 0], e2[0, 0])         # position changes the encoding


# ------------------------------------------------------------------ metrics
def test_compression_summary_arithmetic():
    s = compression_summary([800, 800], [80, 160], [10, 20], raw_elements=[100, 100],
                            discrete_ids=[np.array([1, 2]), np.array([1, 1])])
    assert s["compression_ratio"] == pytest.approx(1600 / 240)
    assert s["sequence_reduction"] == pytest.approx(200 / 30)
    assert s["token_entropy_bits"] == pytest.approx(entropy_bits([1, 2, 1, 1]))
    assert entropy_bits(np.array([5, 5, 5])) == 0.0


def test_image_metrics_known_values():
    x = np.zeros((2, 8, 8, 3), np.uint8)
    y = x.copy()
    y[:] = 10
    m = image_metrics(x, y)
    assert m["mse"] == pytest.approx(100) and m["mae"] == pytest.approx(10)
    assert m["psnr_db"] == pytest.approx(10 * np.log10(255 ** 2 / 100))
    assert image_metrics(x, x)["ssim"] == pytest.approx(1.0)


def test_masked_ts_metrics_ignore_unobserved():
    z = np.zeros((1, 4, 2))
    r = np.array([[[1.0, 100.0], [1.0, 100.0], [1.0, 100.0], [1.0, 100.0]]])
    m = np.zeros((1, 4, 2), bool)
    m[..., 0] = True
    out = masked_ts_metrics(z, r, m, ["a", "b"])
    assert out["mae_z"] == 1.0 and out["n_cells"] == 4 and set(out["mae_z_per_var"]) == {"a"}


# ------------------------------------------------------------------ multimodal integration
def test_multimodal_sample_to_tokens_and_back(stays, lex, binner, spec, norm, text_tok, video_models):
    """raw sample -> tokenizers -> one token sequence -> decoders, all five modalities."""
    rng = np.random.default_rng(0)
    img = rng.integers(0, 255, (28, 28, 3), dtype=np.uint8)
    frames, _ = synthetic_video(16, 32)
    pts = synthetic_surface(512)
    im_m = ImageAutoencoder(dim=8, hidden=16, quantizer=VQ).eval()
    ts_m = TimeSeriesWindowAutoencoder(steps=48, window=4, groups=2, dim=4, hidden=32, quantizer=VQ).eval()
    codecs = MultimodalCodec({
        "text": text_tok, "image": ImageVQCodec(im_m), "timeseries": TSVQCodec(lex, binner, spec, ts_m, norm),
        "video": VideoVQCodec(video_models[0]), "pointcloud": PointCloudTokenizer(res=4).fit([pts], k=16)})
    sample = Sample("case-1", [Segment("text", "Admitted to ICU."), Segment("image", img, 5.0),
                               stay_segment(stays[0]), Segment("video", {"frames": frames, "fps": 1.0}, 60.0),
                               Segment("pointcloud", pts, 61.0)], labels={"y": 0})
    seq = codecs.encode(sample)
    validate(seq)
    assert [sp["modality"] for sp in seq.spans] == ["text", "timeseries", "image", "video", "pointcloud"]
    out = dict(codecs.decode(seq))
    assert out["text"] == "Admitted to ICU."
    assert out["image"].shape == img.shape
    assert out["timeseries"]["grid"].shape == (48, 37)
    assert out["video"]["frames"].shape == frames.shape
    assert out["pointcloud"].shape == pts.shape
