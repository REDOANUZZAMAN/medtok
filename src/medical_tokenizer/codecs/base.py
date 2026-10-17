"""Codec interface every modality tokenizer implements, and the sample-level codec.

A Codec maps one Segment payload to a TokenSpan (encode) and back (decode).
`reversibility` states what decode(encode(x)) gives back:
    "exact"  the original payload, bit for bit (text sub-words, annotation events)
    "lossy"  an approximation; the loss is measured by evaluation/reconstruction.py
"""
from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import Any

from ..protocol.schema import Sample, Segment, TokenSequence, TokenSpan
from ..protocol.serialize import serialize_spans, split_spans


class Codec(ABC):
    modality: str
    rep: str
    reversibility: str = "lossy"

    @abstractmethod
    def encode(self, seg: Segment) -> TokenSpan: ...

    @abstractmethod
    def decode(self, span: TokenSpan) -> Any:
        """Reconstruct a payload of the same type as Segment.data (see docs/token_protocol.md)."""

    def encode_batch(self, segs: list[Segment]) -> list[TokenSpan]:
        """Neural codecs override this with one batched forward pass."""
        return [self.encode(s) for s in segs]

    def alphabet_size(self) -> int:
        """Number of distinct ids this codec can emit inside a span (for fixed-length bit accounting)."""
        raise NotImplementedError

    def raw_bits(self, seg: Segment) -> float:
        """Storage size of the raw payload in bits (definition documented per codec)."""
        raise NotImplementedError

    def payload_bits(self, span: TokenSpan) -> float:
        """Storage size of the span: discrete ids at log2(alphabet) bits, continuous vectors at fp16.

        <CONT> placeholder ids carry no information and are not counted.
        """
        from ..protocol.vocab import CONT
        n_disc = int((span.ids != CONT).sum())
        bits = n_disc * math.log2(max(self.alphabet_size(), 2)) if n_disc else 0.0
        if span.cont is not None:
            bits += span.cont.size * 16
        return bits


class RoutedCodec(Codec):
    """Several codecs for ONE modality, chosen by Segment.source (e.g. MIMIC-IV ECG waveforms and lab
    events are both time-series). The route key is written into the span meta, so decode needs no
    side information and the sequence stays self-describing."""

    def __init__(self, routes: dict[str, Codec]):
        mods = {c.modality for c in routes.values()}
        if len(mods) != 1:
            raise ValueError(f"routed codecs must share one modality, got {sorted(mods)}")
        self.routes, self.modality = routes, mods.pop()
        self.rep = "/".join(sorted({c.rep for c in routes.values()}))
        self.reversibility = "exact" if all(c.reversibility == "exact" for c in routes.values()) else "lossy"

    def _route(self, source: str) -> Codec:
        if source not in self.routes:
            raise KeyError(f"no {self.modality} codec routed for source {source!r}")
        return self.routes[source]

    def encode(self, seg: Segment) -> TokenSpan:
        return self.encode_batch([seg])[0]

    def encode_batch(self, segs: list[Segment]) -> list[TokenSpan]:
        out: list[TokenSpan | None] = [None] * len(segs)
        for src in dict.fromkeys(s.source for s in segs):
            idx = [i for i, s in enumerate(segs) if s.source == src]
            for i, span in zip(idx, self._route(src).encode_batch([segs[i] for i in idx])):
                span.meta = {**span.meta, "source": src}
                out[i] = span
        return out

    def decode(self, span: TokenSpan) -> Any:
        return self._route(span.meta["source"]).decode(span)

    def raw_bits(self, seg: Segment) -> float:
        return self._route(seg.source).raw_bits(seg)


class MultimodalCodec:
    """Sample-level encode/decode: one codec per modality, one TokenSequence per Sample."""

    def __init__(self, codecs: dict[str, Codec]):
        for m, c in codecs.items():
            if c.modality != m:
                raise ValueError(f"codec for {m!r} reports modality {c.modality!r}")
        self.codecs = codecs

    def encode(self, sample: Sample) -> TokenSequence:
        return self.encode_batch([sample])[0]

    def encode_batch(self, samples: list[Sample]) -> list[TokenSequence]:
        """Groups segments by modality so each neural codec runs one batched pass."""
        by_mod: dict[str, list[tuple[int, int]]] = {}
        for i, s in enumerate(samples):
            for j, seg in enumerate(s.segments):
                by_mod.setdefault(seg.modality, []).append((i, j))
        spans: dict[tuple[int, int], TokenSpan] = {}
        for m, keys in by_mod.items():
            if m not in self.codecs:
                raise KeyError(f"no codec registered for modality {m!r}")
            out = self.codecs[m].encode_batch([samples[i].segments[j] for i, j in keys])
            spans.update(zip(keys, out))
        return [serialize_spans(s, [spans[(i, j)] for j in range(len(s.segments))]) for i, s in enumerate(samples)]

    def decode(self, seq: TokenSequence) -> list[tuple[str, Any]]:
        return [(sp["modality"], self.codecs[sp["modality"]].decode(span))
                for sp, span in zip(seq.spans, split_spans(seq))]

    def describe(self) -> dict:
        return {m: {"class": type(c).__name__, "rep": c.rep, "reversibility": c.reversibility}
                for m, c in self.codecs.items()}
