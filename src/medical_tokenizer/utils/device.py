"""Compute device for training loops.

MEDTOK_DEVICE=cpu|cuda overrides; default is CUDA when available. Only the training loops (tokenizer
autoencoders, downstream Transformer) use it. Models are moved back to the CPU before they are saved,
and evaluation / token encoding always runs on the CPU, so checkpoints and token shards do not depend
on the device.
"""
from __future__ import annotations

import os

import torch


def device() -> torch.device:
    name = os.environ.get("MEDTOK_DEVICE") or ("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def to_device(batch: dict, dev: torch.device) -> dict:
    return {k: v.to(dev, non_blocking=True) if torch.is_tensor(v) else v for k, v in batch.items()}
