"""Build the lexicon and load trained tokenizers from experiment run directories.

    from medical_tokenizer.toolkit import load_tokenizer, build_lexicon, MultimodalCodec
    image = load_tokenizer("experiments/image_vq/latest")
    ts    = load_tokenizer("experiments/timeseries_event/latest")
    codec = MultimodalCodec({"image": image, "timeseries": ts})
    seq   = codec.encode(sample)            # TokenSequence
    back  = codec.decode(seq)               # [(modality, payload), ...]
"""
from __future__ import annotations

import json
import os

import numpy as np

from .codecs.base import Codec, MultimodalCodec  # noqa: F401  (re-export)
from .data.physionet import STATIC_VARS, TS_VARS, Normalizer
from .data.video import CATARACT101_PHASES, CHOLEC80_PHASES, CHOLEC80_TOOLS
from .encoders.base import load_model
from .protocol.vocab import Lexicon
from .utils.experiment import load_run_config, resolve_run


def annotated_events() -> list[str]:
    ev = [f"cataract101:phase:{p}" for p in CATARACT101_PHASES] + ["cataract101:phase:none"]
    ev += [f"cholec80:phase:{p}" for p in CHOLEC80_PHASES] + ["cholec80:phase:none"]
    ev += [f"cholec80:tool_on:{t}" for t in CHOLEC80_TOOLS] + [f"cholec80:tool_off:{t}" for t in CHOLEC80_TOOLS]
    return ev


def build_lexicon(n_value_bins: int = 16, n_segment_states: int = 64) -> Lexicon:
    return Lexicon(STATIC_VARS + TS_VARS, n_value_bins, annotated_events(), n_segment_states)


def load_tokenizer(run: str) -> Codec:
    """Reconstruct the Codec a `train` run produced, from its config and saved artifacts."""
    run_dir = resolve_run(run)
    cfg = load_run_config(run_dir)
    p = lambda f: os.path.join(run_dir, f)  # noqa: E731
    mod, rep = cfg["modality"], cfg["representation"]
    if mod == "image":
        from .tokenizers.image.codec import ImageContinuousCodec, ImageVQCodec
        model, _ = load_model(p("model.pt"))
        return ImageVQCodec(model) if rep == "vq" else ImageContinuousCodec(model)
    if mod == "timeseries":
        from .tokenizers.timeseries.codec import (GridSpec, TSContinuousCodec, TSEventCodec, TSVQCodec,
                                                  ValueBinner)
        d = cfg["data"]
        lex = Lexicon.load(p("lexicon.json"))
        spec = GridSpec(d.get("steps", 48), d.get("resolution_min", 60.0))
        with open(p("value_binner.json")) as f:
            binner = ValueBinner.from_json(json.load(f))
        if rep == "event":
            return TSEventCodec(lex, binner, spec, mode=cfg.get("event", {}).get("mode", "obs"))
        with open(p("normalizer.json")) as f:
            norm = Normalizer.from_json(json.load(f))
        model, _ = load_model(p("model.pt"))
        fill = d.get("fill", "zero")
        if rep == "vq":
            return TSVQCodec(lex, binner, spec, model, norm, fill=fill, skip_empty=cfg.get("tokens", {}).get("skip_empty", False))
        return TSContinuousCodec(lex, binner, spec, model, norm, fill=fill)
    if mod == "video":
        from .tokenizers.video.codec import VideoContinuousCodec, VideoSegmentCodec, VideoVQCodec
        if rep == "event":
            model, _ = load_model(p("encoder_model.pt"))
            s = np.load(p("segment_states.npz"))
            return VideoSegmentCodec(model, Lexicon.load(p("lexicon.json")), s["centroids"], s["prototypes"])
        model, _ = load_model(p("model.pt"))
        return VideoVQCodec(model) if rep == "vq" else VideoContinuousCodec(model)
    raise ValueError(f"no tokenizer loader for modality={mod!r} representation={rep!r}")


def load_text_tokenizer(assets_dir: str = "data/processed/assets"):
    from .data.datasets import abspath
    from .tokenizers.text.codec import ClinicalTextTokenizer
    return ClinicalTextTokenizer.load(os.path.join(abspath(assets_dir), "text_bpe.json"))
