"""Tokenise a task's dataset with a trained tokenizer run -> token shards (the cross-modal sample set).

    python -m medical_tokenizer.encode --run experiments/image_vq/latest --task bloodmnist [--include-text]

Shards: <run>/tokens/<task>[_text_v<template version>][_<override-hash>]/{train,val,test}.npz (protocol/storage.py).
Tasks: bloodmnist | pneumoniamnist | dermamnist (image), mortality (timeseries), surgical_phase (video).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time

import numpy as np

from .codecs.base import MultimodalCodec
from .data import datasets as D
from .data.notes import TEMPLATE_VERSION, template_source
from .protocol.schema import Sample, Segment, TokenSequence
from .protocol.serialize import serialize_spans, validate
from .protocol.storage import load_shard, save_shard
from .toolkit import load_text_tokenizer, load_tokenizer
from .utils.experiment import load_run_config, resolve_run

TASK_MODALITY = {"bloodmnist": "image", "pneumoniamnist": "image", "dermamnist": "image",
                 "mortality": "timeseries", "surgical_phase": "video"}
SPLITS = ("train", "val", "test")


def shard_dir(run_dir: str, task: str, include_text: bool, overrides: dict | None) -> str:
    key = task + (f"_text_v{TEMPLATE_VERSION}" if include_text else "")    # new templates -> new shards
    if overrides:
        key += "_" + hashlib.sha1(json.dumps(overrides, sort_keys=True).encode()).hexdigest()[:6]
    return os.path.join(run_dir, "tokens", key)


def encode_task(run: str, task: str, include_text: bool = False, overrides: dict | None = None,
                force: bool = False) -> dict[str, list[TokenSequence]]:
    run_dir = resolve_run(run)
    cfg = load_run_config(run_dir)
    if TASK_MODALITY[task] != cfg["modality"]:
        raise ValueError(f"task {task} needs a {TASK_MODALITY[task]} tokenizer, run is {cfg['modality']}")
    out_dir = shard_dir(run_dir, task, include_text, overrides)
    if not force and all(os.path.exists(os.path.join(out_dir, f"{s}.npz")) for s in SPLITS):
        return {s: load_shard(os.path.join(out_dir, f"{s}.npz"))[0] for s in SPLITS}
    codec = load_tokenizer(run_dir)
    codecs = {cfg["modality"]: codec}
    if include_text:
        codecs["text"] = load_text_tokenizer()
    d = {**cfg["data"], **(overrides or {})}
    out = {}
    for split in SPLITS:
        t0 = time.time()
        if cfg["modality"] == "video":
            seqs = _encode_video_split(d, split, codec, codecs.get("text"))
        else:
            samples = (D.icu_samples(d, split, include_text) if task == "mortality"
                       else D.image_samples(d, split, task, include_text))
            mc = MultimodalCodec(codecs)
            seqs = [s for i in range(0, len(samples), 512) for s in mc.encode_batch(samples[i:i + 512])]
        for s in seqs[:200]:
            validate(s)
        meta = {"run": run_dir, "task": task, "split": split, "include_text": include_text, "overrides": overrides,
                "codecs": MultimodalCodec(codecs).describe()}
        save_shard(os.path.join(out_dir, f"{split}.npz"), seqs, meta)
        lens = [len(s) for s in seqs]
        print(f"  {task}/{split}: {len(seqs)} sequences, length mean {np.mean(lens):.1f} max {max(lens)} "
              f"({time.time() - t0:.0f}s)", flush=True)
        out[split] = seqs
    return out


def _encode_video_split(d: dict, split: str, codec, text_codec) -> list[TokenSequence]:
    """Encode each video's clips once, then cut windows (spans built from slices, window-relative time)."""
    from .tokenizers.video.codec import make_clips
    files, smap = D.video_files(d), D.video_split_map(d)
    W, fps = d["window_clips"], d["fps"]
    seqs = []
    for v in (v for v in files if smap[v] == split):
        vid = D.load_video(files[v], fps)
        wins = D.video_windows(d, vid)
        if not wins:
            continue
        clips = make_clips(vid["frames"], d["clip_len"])
        if codec.rep == "vq":
            per_clip = codec.codes(clips)
            make = lambda a: codec.span_from_codes(a, fps)                                  # noqa: E731
        elif codec.rep == "continuous":
            per_clip = codec.clip_latents(clips).numpy()
            make = lambda a: codec.span_from_latent(a, fps)                                 # noqa: E731
        else:
            per_clip = codec.states(clips)
            make = lambda a: codec.span_from_states(a, fps)                                 # noqa: E731
        for c, lab in wins:
            segs = [Segment("video", {"clips": [c, c + W], "fps": fps}, 0.0, "cataract101")]
            spans = [make(per_clip[c:c + W])]
            if text_codec is not None:
                segs.insert(0, Segment("text", D.video_text_note(v), 0.0, template_source("procedure")))
                spans.insert(0, text_codec.encode(segs[0]))
            sample = Sample(f"cataract101/{v}/{c}", segs, {"phase": lab, "video": v}, split)
            seqs.append(serialize_spans(sample, spans))
    return seqs


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Encode a task dataset into token shards.")
    ap.add_argument("--run", required=True)
    ap.add_argument("--task", required=True, choices=sorted(TASK_MODALITY))
    ap.add_argument("--include-text", action="store_true")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)
    encode_task(args.run, args.task, args.include_text, force=args.force)


if __name__ == "__main__":
    main()
