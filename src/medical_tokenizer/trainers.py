"""Tokenizer training, one function per modality, all driven by the YAML config.

Every trainer writes into the Run directory:
    model.pt            autoencoder checkpoint (VQ or continuous)
    codebook.npy        (VQ) the learned codebook, row k = token k of the modality's range
    lexicon.json        protocol lexicon
    train_log.jsonl     per-epoch loss / validation reconstruction / codebook usage
    metrics.json/.csv   final validation metrics
plus modality assets (value bins, normaliser, segment states).
"""
from __future__ import annotations

import json
import math
import os
import shutil
import time

import numpy as np
import torch
import torch.nn.functional as F

from .data import datasets as D
from .data.physionet import Normalizer
from .encoders.base import LatentAutoencoder, load_model, save_model
from .encoders.image import ImageAutoencoder
from .encoders.timeseries import TimeSeriesWindowAutoencoder, masked_mse
from .encoders.video import VideoClipAutoencoder
from .evaluation.reconstruction import image_metrics
from .quantization.stats import CodeUsageTracker
from .toolkit import build_lexicon
from .utils.experiment import Run


def _optimizer(model, t: dict, steps: int):
    name = t.get("optimizer", "adamw").lower()
    opt_cls = {"adamw": torch.optim.AdamW, "adam": torch.optim.Adam}[name]
    opt = opt_cls(model.parameters(), lr=t["lr"], weight_decay=t.get("weight_decay", 0.0))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=t["lr"], total_steps=max(steps, 1),
                                                pct_start=t.get("warmup_frac", 0.05))
    return opt, sched


def _finish(run: Run, model: LatentAutoencoder, final: dict, lex=None) -> None:
    save_model(model, run.path("model.pt"))
    if model.is_discrete:
        np.save(run.path("codebook.npy"), model.quantizer.codebook.detach().numpy())
    (lex or build_lexicon()).save(run.path("lexicon.json"))
    final["params"] = int(sum(p.numel() for p in model.parameters()))
    run.save_metrics(final)


def _epoch_loop(run, model, cfg, n_train, train_step, validate):
    """Generic loop: train_step(batch_idx) -> (loss, QuantizerOutput); validate() -> dict."""
    t = cfg["train"]
    bs, epochs = t["batch_size"], t["epochs"]
    steps = epochs * (n_train // bs)
    opt, sched = _optimizer(model, t, steps)
    g = torch.Generator().manual_seed(int(cfg.get("seed", 0)))
    val = {}
    for ep in range(epochs):
        model.train()
        t0, tot, n, ppl, restarts = time.time(), 0.0, 0, [], 0
        perm = torch.randperm(n_train, generator=g)
        for i in range(0, n_train - bs + 1, bs):
            loss, q = train_step(perm[i:i + bs])
            opt.zero_grad()
            loss.backward()
            if t.get("grad_clip"):
                torch.nn.utils.clip_grad_norm_(model.parameters(), t["grad_clip"])
            opt.step()
            sched.step()
            tot += loss.item()
            n += 1
            if q.indices is not None:
                ppl.append(q.perplexity)
                restarts += q.restarted
        rec = {"epoch": ep + 1, "train_loss": tot / max(n, 1), "sec": time.time() - t0}
        if ppl:
            rec.update(batch_perplexity=float(np.mean(ppl)), codes_restarted=restarts)
        if (ep + 1) % t.get("eval_every", 1) == 0 or ep + 1 == epochs:
            model.eval()
            val = validate()
            rec.update({f"val_{k}": v for k, v in val.items()})
        run.log(rec)
    return val


# ------------------------------------------------------------------ image
def train_image(cfg: dict, run: Run) -> None:
    d = cfg["data"]
    xtr, _, _, _ = D.image_split(d, "train")
    xtr = xtr[D.subsample(len(xtr), d.get("train_fraction"), cfg.get("seed", 0))]
    xva, _, _, _ = D.image_split(d, "val")
    xva = xva[D.subsample(len(xva), d.get("val_fraction"), 1)]
    print(f"image train {xtr.shape} val {xva.shape}", flush=True)
    model = ImageAutoencoder(image_size=d.get("image_size", 28), quantizer=cfg.get("quantizer"), **cfg["model"])
    xtr_t, xva_t = torch.as_tensor(xtr), torch.as_tensor(xva)

    def step(idx):
        x = model.to_input(xtr_t[idx])
        xr, q = model(x)
        return F.mse_loss(xr, x) + q.loss, q

    @torch.no_grad()
    def validate():
        rec, tracker = [], CodeUsageTracker(max(model.n_codes, 1))
        for i in range(0, len(xva_t), 512):
            x = model.to_input(xva_t[i:i + 512])
            xr, q = model(x)
            rec.append(model.to_uint8(xr).numpy())
            if q.indices is not None:
                tracker.update(q.indices)
        m = image_metrics(xva, np.concatenate(rec), ssim=False)
        if model.is_discrete:
            s = tracker.stats()
            m.update(active_codes=s["active_codes"], utilization=s["utilization"], perplexity=s["perplexity"],
                     dead_pct=s["dead_pct"])
        return m

    final = _epoch_loop(run, model, cfg, len(xtr), step, validate)
    _finish(run, model, final)


# ------------------------------------------------------------------ time-series
def _ts_assets(cfg: dict, run: Run):
    from .tokenizers.timeseries.codec import GridSpec, ValueBinner
    d = cfg["data"]
    stays = D.stays(d)
    spec = GridSpec(d.get("steps", 48), d.get("resolution_min", 60.0))
    lex = build_lexicon(n_value_bins=cfg.get("event", {}).get("n_value_bins", 16))
    binner = ValueBinner(lex.n_value_bins).fit(stays, spec)
    norm = Normalizer().fit(stays)
    run.save_json("value_binner.json", binner.to_json())
    run.save_json("normalizer.json", norm.to_json())
    lex.save(run.path("lexicon.json"))
    return stays, spec, lex, norm


def train_timeseries(cfg: dict, run: Run) -> None:
    stays, spec, lex, norm = _ts_assets(cfg, run)
    d = cfg["data"]
    n_split = {s: sum(x.split == s for x in stays) for s in ("train", "val", "test")}
    if cfg["representation"] == "event":
        # nothing to learn beyond the value bins (fitted on the training split above)
        run.save_metrics({"fitted": "value_binner", "n_value_bins": lex.n_value_bins, "stays": n_split})
        print("event tokenizer: value bins fitted on", n_split["train"], "training stays", flush=True)
        return

    def tensors(split):
        xs, ms = [], []
        for s in stays:
            if s.split == split:
                g, m = spec.grid(s.obs)
                xs.append(norm.grid(g, m, d.get("fill", "zero")))
                ms.append(m)
        return torch.as_tensor(np.stack(xs)), torch.as_tensor(np.stack(ms))

    xtr, mtr = tensors("train")
    xva, mva = tensors("val")
    print(f"timeseries train {tuple(xtr.shape)} observed {mtr.float().mean():.4f}", flush=True)
    model = TimeSeriesWindowAutoencoder(n_vars=xtr.shape[2], steps=spec.steps, quantizer=cfg.get("quantizer"),
                                        **cfg["model"])
    hide = cfg["train"].get("hide_fraction", 0.2)
    g = torch.Generator().manual_seed(int(cfg.get("seed", 0)) + 1)

    def step(idx):
        # Denoising objective: a random `hide` fraction of observed cells is removed from the input
        # and still scored, so codes must capture temporal structure instead of copying cells.
        x, m = xtr[idx], mtr[idx]
        h = m & (torch.rand(m.shape, generator=g) < hide)
        vis = m & ~h
        xr, q = model((x * vis, vis))
        return masked_mse(xr, x, m) + q.loss, q

    @torch.no_grad()
    def validate():
        xr, q = model((xva, mva))
        out = {"masked_mse_z": masked_mse(xr, xva, mva).item(),
               "mae_z": float(((xr - xva).abs() * mva).sum() / mva.sum())}
        if q.indices is not None:
            tr = CodeUsageTracker(model.n_codes)
            tr.update(q.indices)
            s = tr.stats()
            out.update(active_codes=s["active_codes"], utilization=s["utilization"], perplexity=s["perplexity"],
                       dead_pct=s["dead_pct"])
        return out

    final = _epoch_loop(run, model, cfg, len(xtr), step, validate)
    _finish(run, model, final, lex)


# ------------------------------------------------------------------ video
def train_video(cfg: dict, run: Run) -> None:
    d = cfg["data"]
    if cfg["representation"] == "event":
        return _fit_video_segments(cfg, run)
    xtr = D.video_clip_array(d, "train", d.get("max_videos"), d.get("max_clips_per_video"), cfg.get("seed", 0))
    xva = D.video_clip_array(d, "val", d.get("max_videos"), d.get("val_clips_per_video", 8), 1)
    print(f"video clips train {xtr.shape} val {xva.shape}", flush=True)
    model = VideoClipAutoencoder(frame_size=xtr.shape[2], clip_len=d["clip_len"], quantizer=cfg.get("quantizer"),
                                 **cfg["model"])
    xtr_t, xva_t = torch.as_tensor(xtr), torch.as_tensor(xva)

    def step(idx):
        x = model.to_input(xtr_t[idx])
        xr, q = model(x)
        return F.mse_loss(xr, x) + q.loss, q

    @torch.no_grad()
    def validate():
        rec, tracker = [], CodeUsageTracker(max(model.n_codes, 1))
        for i in range(0, len(xva_t), 32):
            xr, q = model(model.to_input(xva_t[i:i + 32]))
            rec.append(model.to_uint8(xr).numpy())
            if q.indices is not None:
                tracker.update(q.indices)
        r = np.concatenate(rec)
        m = image_metrics(xva.reshape(-1, *xva.shape[2:]), r.reshape(-1, *r.shape[2:]), ssim=False)
        if model.is_discrete:
            s = tracker.stats()
            m.update(active_codes=s["active_codes"], utilization=s["utilization"], perplexity=s["perplexity"],
                     dead_pct=s["dead_pct"])
        return m

    final = _epoch_loop(run, model, cfg, len(xtr), step, validate)
    _finish(run, model, final)


def _fit_video_segments(cfg: dict, run: Run) -> None:
    from .tokenizers.video.codec import VideoSegmentCodec
    from .utils.experiment import resolve_run
    d, s = cfg["data"], cfg["segments"]
    enc_dir = resolve_run(s["encoder_run"])
    model, _ = load_model(os.path.join(enc_dir, "model.pt"))
    if model.clip_len != d["clip_len"]:
        raise ValueError("encoder clip_len differs from data.clip_len")
    shutil.copy(os.path.join(enc_dir, "model.pt"), run.path("encoder_model.pt"))
    lex = build_lexicon(n_segment_states=max(64, s["n_states"]))
    xtr = D.video_clip_array(d, "train", d.get("max_videos"), d.get("max_clips_per_video"), cfg.get("seed", 0))
    codec = VideoSegmentCodec(model, lex).fit(xtr, s["n_states"], iters=s.get("iters", 50), seed=cfg.get("seed", 0))
    np.savez(run.path("segment_states.npz"), centroids=codec.centroids, prototypes=codec.prototypes)
    lex.save(run.path("lexicon.json"))
    states = codec.states(xtr)
    counts = np.bincount(states, minlength=s["n_states"])
    run.save_metrics({"encoder_run": enc_dir, "n_states": s["n_states"], "train_clips": int(len(xtr)),
                      "states_used": int((counts > 0).sum())})
    with open(run.path("encoder_run.json"), "w") as f:
        json.dump({"encoder_run": enc_dir}, f)


TRAINERS = {"image": train_image, "timeseries": train_timeseries, "video": train_video}


def train(cfg: dict) -> Run:
    if cfg["modality"] not in TRAINERS:
        raise ValueError(f"no trainer for modality {cfg['modality']!r} (text / point cloud are interface-level)")
    run = Run(cfg)
    torch.set_num_threads(int(cfg.get("threads", os.cpu_count())))
    print(f"run dir: {run.dir}", flush=True)
    TRAINERS[cfg["modality"]](cfg, run)
    return run


def epochs_budget_note(cfg: dict, n: int) -> str:
    return f"{cfg['train']['epochs']} epochs x {math.ceil(n / cfg['train']['batch_size'])} steps"
