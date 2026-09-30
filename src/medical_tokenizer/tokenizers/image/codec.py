"""2D image tokenizers. Payload: uint8 array (H, W, C).

ImageVQCodec          g x g latent grid -> g x g codebook ids, one <ROW> before each row
                      -> g*(g+1) tokens; positions (row, col).
ImageContinuousCodec  the same grid kept continuous: g*g <CONT> slots whose vectors
                      travel in TokenSequence.cont (continuous-projection baseline).
Both reconstruct through the autoencoder decoder, so both are lossy.
Event tokens are not defined for a single static image.
"""
from __future__ import annotations

import numpy as np
import torch

from ...codecs.base import Codec
from ...encoders.image import ImageAutoencoder
from ...protocol.schema import Segment, TokenSpan
from ...protocol.vocab import CONT, RANGES, SPECIAL

ROW = SPECIAL["<ROW>"]
R = RANGES["image_vq"]


def extract_patches(img: np.ndarray, p: int) -> np.ndarray:
    """(H,W,C) -> (H/p, W/p, p, p, C): the pixel region each token covers (for inspection / docs)."""
    H, W, C = img.shape
    return img.reshape(H // p, p, W // p, p, C).transpose(0, 2, 1, 3, 4)


class _ImageCodec(Codec):
    modality = "image"
    reversibility = "lossy"

    def __init__(self, model: ImageAutoencoder, batch_size: int = 512):
        self.model, self.bs = model.eval(), batch_size
        self.g = model.grid

    def raw_bits(self, seg: Segment) -> float:
        return float(np.asarray(seg.data).size * 8)

    @torch.no_grad()
    def latents(self, imgs: np.ndarray) -> torch.Tensor:
        out = [self.model.continuous_latent(self.model.to_input(torch.as_tensor(imgs[i:i + self.bs])))
               for i in range(0, len(imgs), self.bs)]
        return torch.cat(out)

    def _grid_pos(self):
        return [[r, c, -1] for r in range(self.g) for c in range(self.g)]


class ImageVQCodec(_ImageCodec):
    rep = "vq"

    def __init__(self, model, batch_size=512):
        if not model.is_discrete:
            raise ValueError("ImageVQCodec needs a model with a VectorQuantizer")
        if model.n_codes > R.size:
            raise ValueError(f"codebook {model.n_codes} larger than the image_vq range {R.size}")
        super().__init__(model, batch_size)

    def alphabet_size(self) -> int:
        return self.model.n_codes + 1           # codes + <ROW>

    @torch.no_grad()
    def codes(self, imgs: np.ndarray) -> np.ndarray:
        """(B,H,W,C) uint8 -> (B,g,g) local code ids."""
        return self.model.quantizer.codes(self.latents(imgs)).numpy()

    def span_from_codes(self, codes: np.ndarray, t0: float = 0.0) -> TokenSpan:
        ids, pos = [], []
        for r in range(self.g):
            ids.append(ROW)
            pos.append([r, -1, -1])
            ids += (R.start + codes[r]).tolist()
            pos += [[r, c, -1] for c in range(self.g)]
        return TokenSpan.make(ids, "vq", t=[t0] * len(ids), pos=pos, meta={"grid": [self.g, self.g]})

    def encode(self, seg: Segment) -> TokenSpan:
        return self.encode_batch([seg])[0]

    def encode_batch(self, segs):
        codes = self.codes(np.stack([np.asarray(s.data) for s in segs]))
        return [self.span_from_codes(c, s.t_start) for c, s in zip(codes, segs)]

    def codes_from_span(self, span: TokenSpan) -> np.ndarray:
        return np.array([R.to_local(int(i)) for i in span.ids if i != ROW]).reshape(self.g, self.g)

    @torch.no_grad()
    def decode_codes(self, codes: np.ndarray) -> np.ndarray:
        """(B,g,g) ids -> (B,H,W,C) uint8."""
        return self.model.to_uint8(self.model.decode_codes(torch.as_tensor(codes))).numpy()

    def decode(self, span: TokenSpan) -> np.ndarray:
        return self.decode_codes(self.codes_from_span(span)[None])[0]


class ImageContinuousCodec(_ImageCodec):
    rep = "continuous"

    def alphabet_size(self) -> int:
        return 1

    def span_from_latent(self, z: np.ndarray, t0: float = 0.0) -> TokenSpan:
        n = self.g * self.g
        return TokenSpan.make([CONT] * n, "continuous", t=[t0] * n, pos=self._grid_pos(),
                              cont=z.reshape(n, -1).astype(np.float16), meta={"grid": [self.g, self.g]})

    def encode(self, seg: Segment) -> TokenSpan:
        return self.encode_batch([seg])[0]

    def encode_batch(self, segs):
        z = self.latents(np.stack([np.asarray(s.data) for s in segs])).numpy()
        return [self.span_from_latent(zz, s.t_start) for zz, s in zip(z, segs)]

    @torch.no_grad()
    def decode_latents(self, z: np.ndarray) -> np.ndarray:
        return self.model.to_uint8(self.model.decode_latent(torch.as_tensor(z, dtype=torch.float32))).numpy()

    def decode(self, span: TokenSpan) -> np.ndarray:
        return self.decode_latents(span.cont.astype(np.float32).reshape(1, self.g, self.g, -1))[0]
