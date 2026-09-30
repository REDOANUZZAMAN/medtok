"""End-to-end CLI pipeline on SYNTHETIC files laid out like the real datasets:

    train tokenizer -> evaluate -> encode shards -> decode -> downstream (unified Transformer)

for image (MedMNIST layout), time-series (PhysioNet 2012 layout) and video (frame cache).
"""
import json
import os

import numpy as np
import pytest

from medical_tokenizer.data.datasets import abspath
from medical_tokenizer.downstream.__main__ import run_downstream
from medical_tokenizer.encode import encode_task
from medical_tokenizer.evaluate import evaluate
from medical_tokenizer.protocol.serialize import split_spans, validate
from medical_tokenizer.toolkit import load_tokenizer
from medical_tokenizer.trainers import train
from medical_tokenizer.utils.config import load_config

from conftest import write_medmnist, write_physionet, write_video_cache

RUNS = {}
DS_TRAIN = ["train.epochs=1", "train.batch_size=16", "train.patience=1", "seeds=[0]",
            "model.d_model=32", "model.n_layers=1", "model.n_heads=2"]


@pytest.fixture(scope="module")
def work(tmp_path_factory):
    root = tmp_path_factory.mktemp("pipe")
    write_medmnist(str(root / "medmnist"), "bloodmnist", n_classes=3)
    write_medmnist(str(root / "medmnist"), "pneumoniamnist", n_classes=2, gray=True, seed=1)
    write_physionet(str(root / "physionet"), n=40)
    write_video_cache(str(root / "video"))
    return root


def cfg(path, work, *over):
    return load_config(abspath(path), [f"output_dir={work}/exp", "threads=2", *over])


def test_image_pipeline(work):
    d = [f"data.root={work}/medmnist", "data.subsets=[bloodmnist,pneumoniamnist]",
         "evaluate.subsets=[bloodmnist,pneumoniamnist]", "train.epochs=1", "train.batch_size=32",
         "model.hidden=16", "model.dim=8", "quantizer.n_codes=16"]
    runs = {}
    for c in ("configs/image_vq.yaml", "configs/image_continuous.yaml", "configs/image_vq_vanilla.yaml"):
        r = train(cfg(c, work, *d))
        assert os.path.exists(os.path.join(r.dir, "model.pt")) and os.path.exists(os.path.join(r.dir, "config.yaml"))
        runs[r.name] = r.dir
    RUNS["image_vq"] = runs["image_vq"]
    assert os.path.exists(os.path.join(runs["image_vq"], "codebook.npy"))
    m = evaluate(runs["image_vq"])
    assert 0 < m["bloodmnist"]["utilization"]["active_codes"] <= 16
    assert m["bloodmnist"]["compression"]["compression_ratio"] > 1
    assert "ssim" in m["bloodmnist"]["reconstruction"]
    mc = evaluate(runs["image_continuous"])
    assert "utilization" not in mc["bloodmnist"]
    shards = encode_task(runs["image_vq"], "bloodmnist")
    assert set(shards) == {"train", "val", "test"} and len(shards["test"]) == 16
    validate(shards["test"][0])
    img = load_tokenizer(runs["image_vq"]).decode(split_spans(shards["test"][0])[0])
    assert img.shape == (28, 28, 3)
    for rep in ("vq", "continuous"):
        r = run_downstream(cfg(f"configs/downstream/bloodmnist_{rep}.yaml", work, *DS_TRAIN,
                               f"tasks=[{{task: bloodmnist, tokenizer_run: {runs['image_' + rep]}}}]"))
        with open(os.path.join(r.dir, "metrics.json")) as f:
            res = json.load(f)
        assert 0 <= res["test"]["bloodmnist"]["acc"]["mean"] <= 1


def test_timeseries_pipeline(work):
    d = [f"data.root={work}/physionet", f"data.processed={work}/processed", "train.epochs=2", "train.eval_every=1",
         "train.batch_size=8", "model.hidden=32", "model.dim=4", "quantizer.n_codes=16"]
    runs = {}
    for c in ("timeseries_vq", "timeseries_continuous", "timeseries_event", "timeseries_event_change"):
        runs[c] = train(cfg(f"configs/{c}.yaml", work, *d)).dir
    RUNS["timeseries_vq"] = runs["timeseries_vq"]
    ev = evaluate(runs["timeseries_event"])
    assert ev["binned_values_exact_rate"] == 1.0                    # event codec is exact w.r.t. the bins
    ch = evaluate(runs["timeseries_event_change"])
    assert ch["compression"]["tokens_per_sample_mean"] <= ev["compression"]["tokens_per_sample_mean"]
    assert ch["reconstruction"]["mae_z"] == pytest.approx(ev["reconstruction"]["mae_z"], abs=1e-5)
    vq = evaluate(runs["timeseries_vq"])
    assert "utilization" in vq and vq["reconstruction"]["mae_z"] > 0
    evaluate(runs["timeseries_continuous"])
    r = run_downstream(cfg("configs/downstream/mortality_event.yaml", work, *DS_TRAIN,
                           f"tasks=[{{task: mortality, tokenizer_run: {runs['timeseries_event']}}}]"))
    assert os.path.exists(os.path.join(r.dir, "metrics.json"))


def test_video_pipeline(work):
    d = [f"data.cache={work}/video", "data.fps=1.0", "data.size=32", "data.clip_len=8", "data.window_clips=2",
         "data.max_clips_per_video=null", "train.epochs=1", "train.batch_size=4", "model.patch_size=8",
         "model.hidden=16", "model.dim=8", "quantizer.n_codes=16"]
    vq = train(cfg("configs/video_vq.yaml", work, *d)).dir
    ct = train(cfg("configs/video_continuous.yaml", work, *d)).dir
    seg = train(cfg("configs/video_segments.yaml", work, *d, f"segments.encoder_run={ct}", "segments.n_states=4")).dir
    for r in (vq, ct, seg):
        m = evaluate(r)
        assert m["reconstruction"]["psnr_db"] > 0
    m = evaluate(seg)
    assert m["annotated_phase_events"]["exact_roundtrip_rate"] == 1.0
    assert m["compression"]["tokens_per_minute"] < evaluate(vq)["compression"]["tokens_per_minute"]
    shards = encode_task(seg, "surgical_phase")
    lab = [s.labels["phase"] for s in shards["train"]]
    assert min(lab) >= 0
    r = run_downstream(cfg("configs/downstream/phase_vq.yaml", work, *DS_TRAIN,
                           f"tasks=[{{task: surgical_phase, tokenizer_run: {vq}}}]"))
    with open(os.path.join(r.dir, "metrics.json")) as f:
        assert np.isfinite(json.load(f)["test"]["surgical_phase"]["acc"]["mean"])
    RUNS["video_vq"] = vq


def test_multitask_one_model_three_modalities(work):
    """The same token format and ONE model serve tasks from image, time-series and video tokenizers."""
    tasks = ", ".join(f"{{task: {t}, tokenizer_run: {RUNS[k]}}}" for t, k in
                      (("bloodmnist", "image_vq"), ("mortality", "timeseries_vq"), ("surgical_phase", "video_vq")))
    r = run_downstream(cfg("configs/downstream/multitask_vq.yaml", work, *DS_TRAIN, f"tasks=[{tasks}]"))
    with open(os.path.join(r.dir, "metrics.json")) as f:
        res = json.load(f)
    assert set(res["test"]) == {"bloodmnist", "mortality", "surgical_phase"}
