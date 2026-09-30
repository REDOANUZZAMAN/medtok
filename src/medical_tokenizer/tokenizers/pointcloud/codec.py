"""3D point-cloud tokenizer -- INTERFACE-LEVEL implementation. Payload: float32 (N, 3).

No point-cloud dataset is trained on in this project (SCARED needs a
grand-challenge account). The pipeline is implemented end to end with a
non-learned feature encoder so that the protocol, serialisation and
evaluation paths are exercised; it is demonstrated on synthetic surfaces only.

    normalise   centre + isotropic scale into the unit cube (kept in span meta)
    group       voxel grid of resolution R (<= R^3 groups); a learned grouping
                (farthest-point sampling + kNN, as in PointNet++) can replace it
    features    per-group descriptor: mean offset from the voxel centre and per-axis
                std (6-d). Swap in a learned encoder via `feature_fn`
    codebook    k-means over training descriptors -> ids in the `pc_vq` range
    tokens      one per occupied voxel, Morton (z-order) sorted, voxel coordinate in `pos`

Decoding samples points from an axis-aligned Gaussian per voxel: lossy, measured
with the symmetric Chamfer distance. SCARED depth maps become point clouds with
`depth_to_points`.
"""
from __future__ import annotations

from typing import Callable

import numpy as np

from ...codecs.base import Codec
from ...protocol.schema import Segment, TokenSpan
from ...protocol.vocab import RANGES

R = RANGES["pc_vq"]


def morton3(x: int, y: int, z: int, bits: int = 5) -> int:
    code = 0
    for b in range(bits):
        code |= ((x >> b) & 1) << (3 * b) | ((y >> b) & 1) << (3 * b + 1) | ((z >> b) & 1) << (3 * b + 2)
    return code


def depth_to_points(depth: np.ndarray, fx: float, fy: float, cx: float, cy: float) -> np.ndarray:
    v, u = np.nonzero(depth > 0)
    z = depth[v, u]
    return np.stack([(u - cx) * z / fx, (v - cy) * z / fy, z], 1).astype(np.float32)


def normalise(pts: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    lo, hi = pts.min(0), pts.max(0)
    center, scale = (lo + hi) / 2, float((hi - lo).max() / 2 + 1e-9)
    return np.clip((pts - center) / scale * 0.5 + 0.5, 0, 1 - 1e-6), center, scale


def voxel_groups(u: np.ndarray, res: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """Unit-cube points -> [(voxel coord (3,), member offsets in voxel units in [-.5,.5])]."""
    vox = (u * res).astype(int)
    keys = (vox[:, 0] * res + vox[:, 1]) * res + vox[:, 2]
    out = []
    for k in np.unique(keys):
        sel = keys == k
        v = vox[sel][0]
        out.append((v, u[sel] * res - v - 0.5))
    return out


def offset_stats(offsets: np.ndarray) -> np.ndarray:
    return np.concatenate([offsets.mean(0), offsets.std(0)]).astype(np.float32)


def kmeans(x: np.ndarray, k: int, iters: int = 30, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    c = x[rng.choice(len(x), k, replace=len(x) < k)].copy()
    for _ in range(iters):
        a = ((x[:, None] - c[None]) ** 2).sum(-1).argmin(1)
        for j in range(k):
            sel = x[a == j]
            c[j] = sel.mean(0) if len(sel) else x[rng.integers(len(x))]
    return c


class PointCloudTokenizer(Codec):
    modality, rep, reversibility = "pointcloud", "vq", "lossy"

    def __init__(self, codebook: np.ndarray | None = None, res: int = 8,
                 feature_fn: Callable[[np.ndarray], np.ndarray] = offset_stats):
        if res > 32:
            raise ValueError("res > 32 overflows the 5-bit Morton key")
        self.codebook, self.res, self.feature_fn = codebook, res, feature_fn

    def alphabet_size(self) -> int:
        return len(self.codebook)

    def raw_bits(self, seg: Segment) -> float:
        return float(np.asarray(seg.data).size * 32)

    def fit(self, clouds, k: int = 256, seed: int = 0) -> "PointCloudTokenizer":
        feats = [self.feature_fn(off) for c in clouds for _, off in voxel_groups(normalise(c)[0], self.res)]
        self.codebook = kmeans(np.stack(feats), min(k, R.size), seed=seed)
        return self

    def encode(self, seg: Segment) -> TokenSpan:
        pts = np.asarray(seg.data, np.float32)
        u, center, scale = normalise(pts)
        groups = sorted(voxel_groups(u, self.res), key=lambda g: morton3(*g[0]))
        ids, pos = [], []
        for v, off in groups:
            d = self.feature_fn(off)
            ids.append(R.start + int(((self.codebook - d) ** 2).sum(1).argmin()))
            pos.append(v.tolist())
        return TokenSpan.make(ids, "vq", t=[seg.t_start] * len(ids), pos=pos,
                              meta={"center": center.tolist(), "scale": scale, "n_points": len(pts), "res": self.res})

    def decode(self, span: TokenSpan, seed: int = 0) -> np.ndarray:
        """Requires the default 6-d offset_stats features (mean, std) to render points."""
        rng = np.random.default_rng(seed)
        m = span.meta
        per = int(np.ceil(m["n_points"] / max(len(span.ids), 1)))
        pts = [v + 0.5 + self.codebook[R.to_local(int(g))][:3]
               + rng.standard_normal((per, 3)) * self.codebook[R.to_local(int(g))][3:6]
               for g, v in zip(span.ids, span.pos)]
        u = np.concatenate(pts) / m["res"]
        return (((u - 0.5) / 0.5) * m["scale"] + np.array(m["center"])).astype(np.float32)[:m["n_points"]]


def chamfer(a: np.ndarray, b: np.ndarray) -> float:
    """Symmetric mean nearest-neighbour distance (same units as the input)."""
    d = ((a[:, None] - b[None]) ** 2).sum(-1)
    return float(np.sqrt(d.min(1)).mean() + np.sqrt(d.min(0)).mean()) / 2


def synthetic_surface(n: int = 2048, seed: int = 0) -> np.ndarray:
    """Deformed ellipsoidal cap in mm: a SYNTHETIC stand-in for a tissue surface, for tests and demos only."""
    rng = np.random.default_rng(seed)
    th, ph = rng.uniform(0, np.pi / 2, n), rng.uniform(0, 2 * np.pi, n)
    a, b, c = rng.uniform(30, 60, 3)
    r = 1 + 0.1 * np.sin(3 * ph + rng.uniform(0, 6)) * np.sin(2 * th)
    pts = np.stack([a * r * np.sin(th) * np.cos(ph), b * r * np.sin(th) * np.sin(ph), c * r * np.cos(th)], 1)
    return (pts + rng.normal(0, 0.3, pts.shape)).astype(np.float32)
