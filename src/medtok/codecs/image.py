"""2D-image codecs (MedMNIST, 28x28 RGB uint8).

ImageVQCodec    7x7 VQ codes, one <row> marker per latent row -> 56 tokens.
ImageContCodec  same encoder, pre-quantisation latents as 49 continuous slots
                (<cont> placeholder tokens + side payload), i.e. the
                "continuous projection" baseline used by most multimodal LLMs.
Both are approximately reversible through the VQ-VAE decoder.
"""
from __future__ import annotations

import numpy as np
import torch

from ..protocol.schema import Segment, TokenSpan
from ..protocol.vocab import RANGES, SPECIAL
from .base import Codec

ROW = SPECIAL["<row>"]
CONT = SPECIAL["<cont>"]


class _ImageBase(Codec):
    modality = "image"
    reversible = False

    def __init__(self, model):
        self.model = model.eval()
        self.grid = 7

    def raw_bits(self, seg: Segment) -> float:
        return np.asarray(seg.data).size * 8

    @torch.no_grad()
    def latents(self, imgs: np.ndarray) -> torch.Tensor:
        x = self.model.to_input(torch.as_tensor(imgs))
        return self.model.encode_latent(x)


class ImageVQCodec(_ImageBase):
    rep = "vq"

    def alphabet_size(self) -> int:
        return self.model.vq.n_codes + 1  # codes + <row>

    @torch.no_grad()
    def codes(self, imgs: np.ndarray) -> np.ndarray:
        """Batch helper: (B,28,28,3) uint8 -> (B,7,7) local code ids."""
        return self.model.vq.codes(self.latents(imgs)).numpy()

    def span_from_codes(self, codes: np.ndarray, t0: float = 0.0) -> TokenSpan:
        g = self.grid
        ids, pos = [], []
        for r in range(g):
            ids.append(ROW)
            pos.append([r, -1, 0])
            for c in range(g):
                ids.append(RANGES["image_vq"].to_global(int(codes[r, c])))
                pos.append([r, c, 0])
        return TokenSpan.make(ids, "vq", t=[t0] * len(ids), pos=pos, meta={"grid": [g, g]})

    def encode(self, seg: Segment) -> TokenSpan:
        return self.span_from_codes(self.codes(np.asarray(seg.data)[None])[0], seg.t_start)

    def codes_from_span(self, span: TokenSpan) -> np.ndarray:
        ids = span.ids[span.ids != ROW]
        return np.array([RANGES["image_vq"].to_local(int(i)) for i in ids]).reshape(self.grid, self.grid)

    @torch.no_grad()
    def decode(self, span: TokenSpan) -> np.ndarray:
        idx = torch.as_tensor(self.codes_from_span(span))[None]
        return self.model.to_uint8(self.model.decode_codes(idx))[0].numpy()


class ImageContCodec(_ImageBase):
    rep = "cont"

    def alphabet_size(self) -> int:
        return 1

    def span_from_latent(self, z: np.ndarray, t0: float = 0.0) -> TokenSpan:
        g = self.grid
        pos = [[r, c, 0] for r in range(g) for c in range(g)]
        return TokenSpan.make([CONT] * (g * g), "cont", t=[t0] * (g * g), pos=pos,
                              cont=z.reshape(g * g, -1).astype(np.float16), meta={"grid": [g, g]})

    def encode(self, seg: Segment) -> TokenSpan:
        return self.span_from_latent(self.latents(np.asarray(seg.data)[None])[0].numpy(), seg.t_start)

    @torch.no_grad()
    def decode(self, span: TokenSpan) -> np.ndarray:
        z = torch.as_tensor(span.cont.astype(np.float32)).view(1, self.grid, self.grid, -1)
        return self.model.to_uint8(self.model.decode_latent(z))[0].numpy()

    def span_bits(self, span, vocab_bits=None) -> float:
        return span.cont.size * 16  # the placeholder ids carry no information
