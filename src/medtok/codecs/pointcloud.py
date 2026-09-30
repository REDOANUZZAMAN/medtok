"""3D point-cloud codec (interface level; SCARED needs a grand-challenge account).

PointCloudVoxelCodec
    1. normalise the cloud into the unit cube (centre + scale kept in meta);
    2. voxelise on an R^3 grid (R=8 -> <= 512 voxels);
    3. describe each occupied voxel by (mean offset from the voxel centre,
       per-axis std) and quantise that 6-d descriptor with a k-means codebook
       (in the `pc_vq` range);
    4. emit one token per occupied voxel in Morton (z-order) order, with the
       voxel coordinate in the `pos` side channel.
Decoding samples points from an axis-aligned Gaussian per voxel. The codec is
approximately reversible; quality is measured with the Chamfer distance.
SCARED depth maps become point clouds by back-projecting with the camera
intrinsics (see `depth_to_points`).
"""
from __future__ import annotations

import numpy as np

from ..protocol.schema import Segment, TokenSpan
from ..protocol.vocab import RANGES
from .base import Codec


def _morton(x, y, z, bits=4):
    code = 0
    for b in range(bits):
        code |= ((x >> b) & 1) << (3 * b) | ((y >> b) & 1) << (3 * b + 1) | ((z >> b) & 1) << (3 * b + 2)
    return code


def depth_to_points(depth: np.ndarray, fx: float, fy: float, cx: float, cy: float) -> np.ndarray:
    v, u = np.nonzero(depth > 0)
    z = depth[v, u]
    return np.stack([(u - cx) * z / fx, (v - cy) * z / fy, z], 1).astype(np.float32)


def kmeans(x: np.ndarray, k: int, iters: int = 30, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    c = x[rng.choice(len(x), k, replace=len(x) < k)].copy()
    for _ in range(iters):
        a = ((x[:, None] - c[None]) ** 2).sum(-1).argmin(1)
        for j in range(k):
            sel = x[a == j]
            c[j] = sel.mean(0) if len(sel) else x[rng.integers(len(x))]
    return c


class PointCloudVoxelCodec(Codec):
    modality, rep, reversible = "pointcloud", "vq", False

    def __init__(self, codebook: np.ndarray | None = None, res: int = 8):
        self.res, self.codebook = res, codebook

    def alphabet_size(self) -> int:
        return len(self.codebook)

    def raw_bits(self, seg: Segment) -> float:
        return np.asarray(seg.data).size * 32

    def _voxel_desc(self, pts: np.ndarray):
        lo, hi = pts.min(0), pts.max(0)
        center, scale = (lo + hi) / 2, float((hi - lo).max() / 2 + 1e-9)
        u = np.clip((pts - center) / scale * 0.5 + 0.5, 0, 1 - 1e-6)     # unit cube
        vox = (u * self.res).astype(int)
        keys = vox[:, 0] * self.res ** 2 + vox[:, 1] * self.res + vox[:, 2]
        out = []
        for k in np.unique(keys):
            sel = u[keys == k] * self.res - vox[keys == k][0] - 0.5            # in voxel units, [-.5,.5]
            out.append((vox[keys == k][0], np.concatenate([sel.mean(0), sel.std(0)])))
        return out, center, scale

    def fit(self, clouds, k: int = 256) -> "PointCloudVoxelCodec":
        descs = np.concatenate([np.stack([d for _, d in self._voxel_desc(c)[0]]) for c in clouds])
        self.codebook = kmeans(descs.astype(np.float32), min(k, RANGES["pc_vq"].size))
        return self

    def encode(self, seg: Segment) -> TokenSpan:
        pts = np.asarray(seg.data, np.float32)
        vox, center, scale = self._voxel_desc(pts)
        vox.sort(key=lambda vd: _morton(*vd[0]))
        ids, pos = [], []
        for v, d in vox:
            ids.append(RANGES["pc_vq"].to_global(int(((self.codebook - d) ** 2).sum(1).argmin())))
            pos.append(v.tolist())
        return TokenSpan.make(ids, "vq", t=[seg.t_start] * len(ids), pos=pos,
                              meta={"center": center.tolist(), "scale": scale, "n_points": len(pts), "res": self.res})

    def decode(self, span: TokenSpan, seed: int = 0) -> np.ndarray:
        rng = np.random.default_rng(seed)
        m = span.meta
        per = int(np.ceil(m["n_points"] / max(len(span.ids), 1)))
        out = []
        for gid, v in zip(span.ids, span.pos):
            d = self.codebook[RANGES["pc_vq"].to_local(int(gid))]
            p = v + 0.5 + d[:3] + rng.standard_normal((per, 3)) * d[3:]
            out.append(p)
        u = np.concatenate(out) / self.res
        return (((u - 0.5) / 0.5) * m["scale"] + np.array(m["center"])).astype(np.float32)[:m["n_points"]]


def chamfer(a: np.ndarray, b: np.ndarray) -> float:
    d = ((a[:, None] - b[None]) ** 2).sum(-1)
    return float(np.sqrt(d.min(1)).mean() + np.sqrt(d.min(0)).mean()) / 2


def synthetic_organ_surface(n: int = 2048, seed: int = 0) -> np.ndarray:
    """Deformed ellipsoidal patch, a stand-in for a SCARED liver/tissue surface (mm)."""
    rng = np.random.default_rng(seed)
    th = rng.uniform(0, np.pi / 2, n)
    ph = rng.uniform(0, 2 * np.pi, n)
    a, b, c = rng.uniform(30, 60, 3)
    r = 1 + 0.1 * np.sin(3 * ph + rng.uniform(0, 6)) * np.sin(2 * th)
    pts = np.stack([a * r * np.sin(th) * np.cos(ph), b * r * np.sin(th) * np.sin(ph), c * r * np.cos(th)], 1)
    return (pts + rng.normal(0, 0.3, pts.shape)).astype(np.float32)
