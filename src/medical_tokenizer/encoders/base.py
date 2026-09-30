"""Shared autoencoder skeleton: encoder -> quantiser (VQ or identity) -> decoder.

Every modality model implements encode_latent / decode_latent on a latent grid
of shape (B, *grid, dim). The same class therefore yields
  * the VQ tokenizer        (quantizer = VectorQuantizer), and
  * the continuous baseline (quantizer = IdentityQuantizer),
so the two differ only in the bottleneck.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from ..quantization.vq import QuantizerOutput, VectorQuantizer, build_quantizer

MODEL_REGISTRY: dict[str, type] = {}


def register(cls):
    MODEL_REGISTRY[cls.__name__] = cls
    return cls


class LatentAutoencoder(nn.Module):
    def __init__(self, dim: int, quantizer: dict | None):
        super().__init__()
        self.dim = dim
        self.quantizer_cfg = quantizer
        self.quantizer = build_quantizer(quantizer, dim)

    @property
    def is_discrete(self) -> bool:
        return isinstance(self.quantizer, VectorQuantizer)

    @property
    def n_codes(self) -> int:
        return self.quantizer.n_codes

    def encode_latent(self, x: torch.Tensor) -> torch.Tensor:        # -> (B, *grid, dim)
        raise NotImplementedError

    def decode_latent(self, z: torch.Tensor) -> torch.Tensor:        # (B, *grid, dim) -> x_hat
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, QuantizerOutput]:
        q = self.quantizer(self.encode_latent(x))
        return self.decode_latent(q.z_q), q

    @torch.no_grad()
    def continuous_latent(self, x: torch.Tensor) -> torch.Tensor:
        """Latents as the decoder sees them without discretisation (normalised if the quantiser normalises)."""
        z = self.encode_latent(x)
        return torch.nn.functional.normalize(z, dim=-1) if getattr(self.quantizer, "normalize", False) else z

    @torch.no_grad()
    def codes(self, x: torch.Tensor) -> torch.Tensor:
        return self.quantizer.codes(self.encode_latent(x))

    @torch.no_grad()
    def decode_codes(self, idx: torch.Tensor) -> torch.Tensor:
        return self.decode_latent(self.quantizer.lookup(idx))


def save_model(model: LatentAutoencoder, path: str, **extra) -> None:
    torch.save({"cls": type(model).__name__, "cfg": model.cfg, "state": model.state_dict(), **extra}, path)


def load_model(path: str, map_location: str = "cpu") -> tuple[LatentAutoencoder, dict]:
    ck = torch.load(path, map_location=map_location, weights_only=False)
    m = MODEL_REGISTRY[ck["cls"]](**ck["cfg"])
    m.load_state_dict(ck["state"])
    m.eval()
    return m, ck
