"""Codec interface every modality tokenizer implements."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from ..protocol.schema import Segment, TokenSpan


class Codec(ABC):
    modality: str       # one of MODALITIES
    rep: str            # "bpe" | "vq" | "event" | "cont"
    reversible: bool    # True: decode(encode(x)) == x exactly; False: approximate

    @abstractmethod
    def encode(self, seg: Segment) -> TokenSpan: ...

    @abstractmethod
    def decode(self, span: TokenSpan) -> Any:
        """Reconstruct the raw payload (same type/shape as Segment.data)."""

    def raw_bits(self, seg: Segment) -> float:
        """Size of the raw payload in bits, used for the compression ratio."""
        raise NotImplementedError

    def span_bits(self, span: TokenSpan, vocab_bits: float | None = None) -> float:
        """Size of the token payload in bits.

        Discrete tokens: n * log2(range size actually used by this codec).
        Continuous slots: n_vectors * d * 16 (fp16 storage).
        """
        import math
        bits = len(span.ids) * (vocab_bits if vocab_bits is not None else math.log2(self.alphabet_size()))
        if span.cont is not None:
            bits += span.cont.size * 16
        return bits

    def alphabet_size(self) -> int:
        """Number of distinct ids this codec can emit (for bit accounting)."""
        raise NotImplementedError
