"""Reconstruction metrics, applied only where they are meaningful.

images / video frames (uint8):  MSE, MAE (pixel units 0-255), PSNR (dB, peak 255), SSIM
                                (per image, channel-averaged, 7x7 Gaussian-free window as in
                                scikit-image defaults for small images)
time-series (partially observed): MAE / RMSE in per-variable z-units on OBSERVED cells only
                                (PSNR/SSIM are not defined for sparse irregular series)
point clouds:                   symmetric Chamfer distance (tokenizers/pointcloud)
text / annotation events:       exact-match rate (the codecs are exactly reversible)
"""
from __future__ import annotations

import numpy as np


def image_metrics(x: np.ndarray, xr: np.ndarray, ssim: bool = True) -> dict:
    """x, xr: (N,H,W,C) uint8. Returns means over images (PSNR averaged per image)."""
    a, b = x.astype(np.float64), xr.astype(np.float64)
    mse = ((a - b) ** 2).reshape(len(a), -1).mean(1)
    out = {"mse": float(mse.mean()), "mae": float(np.abs(a - b).mean()),
           "psnr_db": float(np.mean(10 * np.log10(255.0 ** 2 / np.maximum(mse, 1e-10))))}
    if ssim:
        out["ssim"] = float(np.mean([ssim_uint8(x[i], xr[i]) for i in range(len(x))]))
    return out


def ssim_uint8(a: np.ndarray, b: np.ndarray) -> float:
    from skimage.metrics import structural_similarity
    win = min(7, a.shape[0] - (1 - a.shape[0] % 2), a.shape[1] - (1 - a.shape[1] % 2))
    return float(structural_similarity(a, b, channel_axis=-1, data_range=255, win_size=win))


def masked_ts_metrics(z_true: np.ndarray, z_rec: np.ndarray, mask: np.ndarray, variables: list[str] | None = None) -> dict:
    """z_*: (N,T,V) normalised values; mask: (N,T,V) observed cells."""
    d = (z_rec - z_true)[mask]
    out = {"mae_z": float(np.abs(d).mean()), "rmse_z": float(np.sqrt((d ** 2).mean())), "n_cells": int(mask.sum())}
    if variables is not None:
        var = np.broadcast_to(np.arange(z_true.shape[-1]), mask.shape)[mask]
        out["mae_z_per_var"] = {variables[v]: float(np.abs(d[var == v]).mean())
                                for v in range(len(variables)) if (var == v).any()}
    return out


def exact_match_rate(pairs) -> float:
    pairs = list(pairs)
    return float(np.mean([np.array_equal(np.asarray(a), np.asarray(b)) if not isinstance(a, str) else a == b
                          for a, b in pairs])) if pairs else float("nan")
