"""Build cross-modal Samples and their TokenSequences (batched encoding).

ICU sample      = admission note (text) + 48 h vitals/labs (timeseries),
                  label `death` (in-hospital mortality).
MedMNIST sample = image-request note (text) + image, label `class`.
"""
from __future__ import annotations

import numpy as np

from ..codecs.timeseries import stay_segment
from ..protocol.schema import Sample, Segment, TokenSequence
from ..protocol.serialize import serialize_spans
from .notes import image_note, stay_note
from .physionet import Stay


def icu_sample(stay: Stay) -> Sample:
    return Sample(f"physionet2012/{stay.record_id}",
                  [Segment("text", stay_note(stay), 0.0, "template:admission"), stay_segment(stay)],
                  labels={"death": stay.death}, split=stay.split, meta={"task": "mortality"})


def image_sample(subset: str, split: str, i: int, img: np.ndarray, label: int) -> Sample:
    return Sample(f"{subset}/{split}/{i}",
                  [Segment("text", image_note(subset, i), 0.0, "template:request"),
                   Segment("image", img, 0.0, subset)],
                  labels={"class": int(label)}, split=split, meta={"task": subset})


def encode_icu(stays: list[Stay], tk, rep: str, batch: int = 256) -> list[TokenSequence]:
    codec = tk.codecs[("timeseries", rep)]
    out = []
    for b in range(0, len(stays), batch):
        chunk = stays[b:b + batch]
        if rep == "vq":
            codes = codec.codes([s.obs for s in chunk])
            ts_spans = [codec.span_from_codes(s.static, c) for s, c in zip(chunk, codes)]
        elif rep == "cont":
            z = codec.latents([s.obs for s in chunk]).numpy()
            ts_spans = [codec.span_from_latent(s.static, zz) for s, zz in zip(chunk, z)]
        else:
            ts_spans = [codec.encode(stay_segment(s)) for s in chunk]
        for s, span in zip(chunk, ts_spans):
            smp = icu_sample(s)
            out.append(serialize_spans(smp, [tk.text.encode(smp.segments[0]), span]))
    return out


def encode_images(subset: str, split: str, imgs: np.ndarray, labels: np.ndarray, tk, rep: str,
                  batch: int = 512) -> list[TokenSequence]:
    codec = tk.codecs[("image", rep)]
    out = []
    for b in range(0, len(imgs), batch):
        x = imgs[b:b + batch]
        if rep == "vq":
            spans = [codec.span_from_codes(c) for c in codec.codes(x)]
        else:
            spans = [codec.span_from_latent(z) for z in codec.latents(x).numpy()]
        for j, span in enumerate(spans):
            i = b + j
            smp = image_sample(subset, split, i, x[j], labels[i])
            out.append(serialize_spans(smp, [tk.text.encode(smp.segments[0]), span]))
    return out
