"""One-stop loading of the lexicon and trained codecs from an artifacts dir.

    artifacts/
      lexicon.json        protocol ranges, special tokens, variables, events
      text_bpe.json       byte-level BPE (tokenizers format)
      ts_binner.json      per-variable quantile bins
      ts_norm.json        per-variable normaliser for the patch autoencoder
      image_vq.pt         ImageVQVAE checkpoint
      ts_vq.pt            TSPatchVQ checkpoint
"""
from __future__ import annotations

import json
import os

from .codecs.image import ImageContCodec, ImageVQCodec
from .codecs.text import TextBPECodec
from .codecs.timeseries import TSContCodec, TSEventCodec, TSVQCodec, ValueBinner
from .codecs.video import VIDEO_EVENTS, SurgicalEventCodec
from .data.physionet import STATIC_VARS, TS_VARS, Normalizer
from .protocol.vocab import Lexicon
from .quant.models import load_model

DEFAULT_ARTIFACTS = os.path.join(os.path.dirname(__file__), "..", "..", "artifacts")


def build_lexicon(n_value_bins: int = 16) -> Lexicon:
    return Lexicon(variables=STATIC_VARS + TS_VARS, n_value_bins=n_value_bins, events=list(VIDEO_EVENTS))


class Toolkit:
    def __init__(self, artifacts: str = DEFAULT_ARTIFACTS):
        self.dir = os.path.abspath(artifacts)
        p = lambda f: os.path.join(self.dir, f)  # noqa: E731
        self.lex = Lexicon.load(p("lexicon.json"))
        self.text = TextBPECodec.load(p("text_bpe.json"))
        self.binner = ValueBinner.from_json(json.load(open(p("ts_binner.json"))))
        self.norm = Normalizer.from_json(json.load(open(p("ts_norm.json"))))
        self.image_model, _ = load_model(p("image_vq.pt"))
        self.ts_model, _ = load_model(p("ts_vq.pt"))
        self.codecs = {
            ("text", "bpe"): self.text,
            ("image", "vq"): ImageVQCodec(self.image_model),
            ("image", "cont"): ImageContCodec(self.image_model),
            ("timeseries", "event"): TSEventCodec(self.lex, self.binner),
            ("timeseries", "vq"): TSVQCodec(self.lex, self.binner, self.ts_model, self.norm),
            ("timeseries", "cont"): TSContCodec(self.lex, self.binner, self.ts_model, self.norm),
            ("video", "event"): SurgicalEventCodec(self.lex),
        }

    def codec_set(self, image: str = "vq", timeseries: str = "vq") -> dict:
        """Modality -> codec mapping for serialize()/deserialize()."""
        return {"text": self.codecs[("text", "bpe")], "image": self.codecs[("image", image)],
                "timeseries": self.codecs[("timeseries", timeseries)],
                "video": self.codecs[("video", "event")]}
