"""Physiological time-series codecs (PhysioNet 2012 ICU stays).

Segment.data for this modality is a dict:
    {"obs": (n,3) float32 [minutes, var_idx, value], "static": {name: value}}

Three representation types, all prefixed by the same static block
    <static> <var:Age> <vbin:k> <var:Gender> <vbin:k> ...

TSEventCodec  one event per observed (hour, variable) cell of the hourly grid:
              [<dt:k>] <var:NAME> <vbin:q>.  <dt> appears only when the hour
              changes. Values are per-variable quantile bins; decoding returns
              the bin's training median. Irregular sampling is preserved at
              hourly resolution and missingness costs nothing.
TSVQCodec     hourly grid -> 12 x 4 h patches -> G=4 product-quantised codes
              per patch: <patch> q q q q (x12). Fixed length regardless of
              sampling density.
TSContCodec   the same patch encoder, continuous latents as 12*G <cont> slots.
"""
from __future__ import annotations

import math

import numpy as np
import torch

from ..data.physionet import HOURS, STATIC_VARS, TS_VARS, Normalizer, Stay, to_grid
from ..protocol.schema import Segment, TokenSpan
from ..protocol.vocab import RANGES, SPECIAL, Lexicon, bucket_to_dt, dt_to_bucket

STATIC = SPECIAL["<static>"]
PATCH = SPECIAL["<patch>"]
CONT = SPECIAL["<cont>"]
MISSING = SPECIAL["<missing>"]


def stay_segment(stay: Stay) -> Segment:
    return Segment("timeseries", {"obs": stay.obs, "static": stay.static}, 0.0, "physionet2012",
                   {"record_id": stay.record_id})


class ValueBinner:
    """Per-variable quantile bins (static and temporal), fitted on train."""

    def __init__(self, n_bins: int = 16, edges: dict | None = None, centers: dict | None = None):
        self.n_bins = n_bins
        self.edges = edges or {}
        self.centers = centers or {}

    def fit(self, stays: list[Stay]) -> "ValueBinner":
        tr = [s for s in stays if s.split == "train"]
        vals = {v: [] for v in STATIC_VARS + TS_VARS}
        for s in tr:
            for k in STATIC_VARS:
                if not np.isnan(s.static[k]):
                    vals[k].append(s.static[k])
            g, m = to_grid(s)
            for j, v in enumerate(TS_VARS):
                vals[v].extend(g[m[:, j], j].tolist())
        for v, x in vals.items():
            x = np.asarray(x, np.float64)
            qs = np.quantile(x, np.linspace(0, 1, self.n_bins + 1)[1:-1]) if len(x) else np.zeros(self.n_bins - 1)
            e = np.unique(qs)  # discrete variables (Gender, GCS, MechVent) collapse to fewer bins
            b = np.digitize(x, e)
            self.edges[v] = e.tolist()
            self.centers[v] = [float(np.median(x[b == k])) if (b == k).any() else float("nan")
                               for k in range(len(e) + 1)]
        return self

    def bin(self, var: str, x) -> np.ndarray:
        return np.digitize(np.asarray(x, np.float64), self.edges[var])

    def value(self, var: str, k) -> np.ndarray:
        return np.asarray(self.centers[var])[np.asarray(k)]

    def to_json(self):
        return {"n_bins": self.n_bins, "edges": self.edges, "centers": self.centers}

    @classmethod
    def from_json(cls, d):
        return cls(d["n_bins"], d["edges"], d["centers"])


class _TSBase:
    modality = "timeseries"
    reversible = False

    def __init__(self, lexicon: Lexicon, binner: ValueBinner):
        self.lex, self.binner = lexicon, binner

    # ---- static block (shared by all three) ----
    def static_tokens(self, static: dict, t0: float):
        ids, t, pos = [STATIC], [t0], [[-1, -1, -1]]
        for k in STATIC_VARS:
            v = static.get(k, np.nan)
            if v is None or np.isnan(v):
                continue
            ids += [self.lex.var_id(k), self.lex.vbin_id(int(self.binner.bin(k, v)))]
            t += [t0, t0]
            pos += [[-1, -1, -1]] * 2
        return ids, t, pos

    def decode_static(self, ids) -> tuple[dict, int]:
        """Parse the static block; returns (static dict, index after it)."""
        assert ids[0] == STATIC
        out, i = {k: np.nan for k in STATIC_VARS}, 1
        while i + 1 < len(ids) and int(ids[i]) in RANGES["var"] and int(ids[i + 1]) in RANGES["vbin"]:
            name = self.lex.variables[RANGES["var"].to_local(int(ids[i]))]
            if name not in STATIC_VARS:
                break
            out[name] = float(self.binner.value(name, RANGES["vbin"].to_local(int(ids[i + 1]))))
            i += 2
        return out, i

    def raw_bits(self, seg: Segment) -> float:
        # raw record = (time, variable, value) triples at 16+8+32 bits, plus 4 static float32
        return len(seg.data["obs"]) * (16 + 8 + 32) + len(STATIC_VARS) * 32


class TSEventCodec(_TSBase):
    rep = "event"

    def alphabet_size(self) -> int:
        return 1 + len(self.lex.variables) + self.lex.n_value_bins + RANGES["dt"].size

    def encode(self, seg: Segment) -> TokenSpan:
        ids, t, pos = self.static_tokens(seg.data["static"], seg.t_start)
        g, m = to_grid(Stay("", {}, seg.data["obs"], 0))
        prev_h = 0
        for h in range(g.shape[0]):
            vs = np.flatnonzero(m[h])
            if not len(vs):
                continue
            if h != prev_h:
                ids.append(RANGES["dt"].to_global(dt_to_bucket((h - prev_h) * 60)))
                t.append(seg.t_start + h * 60)
                pos.append([h, -1, -1])
                prev_h = h
            for j in vs:
                name = TS_VARS[j]
                ids += [self.lex.var_id(name), self.lex.vbin_id(int(self.binner.bin(name, g[h, j])))]
                t += [seg.t_start + h * 60] * 2
                pos += [[h, j, -1]] * 2
        return TokenSpan.make(ids, "event", t=t, pos=pos)

    def decode(self, span: TokenSpan) -> dict:
        """-> {"static": {...}, "grid": (48,37) values, "mask": (48,37)}."""
        ids = span.ids
        static, i = self.decode_static(ids)
        g = np.zeros((HOURS, len(TS_VARS)), np.float32)
        m = np.zeros_like(g, bool)
        h = 0
        while i < len(ids):
            gid = int(ids[i])
            if gid in RANGES["dt"]:
                h += int(round(bucket_to_dt(RANGES["dt"].to_local(gid)) / 60))
                i += 1
                continue
            name = self.lex.variables[RANGES["var"].to_local(gid)]
            k = RANGES["vbin"].to_local(int(ids[i + 1]))
            j = TS_VARS.index(name)
            g[min(h, HOURS - 1), j] = self.binner.value(name, k)
            m[min(h, HOURS - 1), j] = True
            i += 2
        return {"static": static, "grid": g, "mask": m}


class _TSPatchBase(_TSBase):
    def __init__(self, lexicon, binner, model, normalizer: Normalizer):
        super().__init__(lexicon, binner)
        self.model, self.norm = model.eval(), normalizer
        self.P, self.G = model.n_patches, model.groups

    def grid_tensors(self, obs_list) -> tuple[torch.Tensor, torch.Tensor]:
        xs, ms = [], []
        for obs in obs_list:
            g, m = to_grid(Stay("", {}, obs, 0))
            xs.append(self.norm.grid(g, m))
            ms.append(m)
        return torch.as_tensor(np.stack(xs)), torch.as_tensor(np.stack(ms))

    @torch.no_grad()
    def latents(self, obs_list) -> torch.Tensor:  # (B,P,G,dim)
        return self.model.encode_latent(*self.grid_tensors(obs_list))

    def _grid_out(self, z_norm: torch.Tensor) -> np.ndarray:
        var = np.broadcast_to(np.arange(len(TS_VARS)), z_norm.shape)
        return self.norm.inverse(z_norm.numpy(), var)


class TSVQCodec(_TSPatchBase):
    rep = "vq"

    def alphabet_size(self) -> int:
        return self.model.vq.n_codes + 1

    @torch.no_grad()
    def codes(self, obs_list) -> np.ndarray:  # (B,P,G)
        return self.model.vq.codes(self.latents(obs_list)).numpy()

    def span_from_codes(self, static: dict, codes: np.ndarray, t0=0.0) -> TokenSpan:
        ids, t, pos = self.static_tokens(static, t0)
        plen = HOURS // self.P * 60
        for p in range(self.P):
            ids.append(PATCH)
            t.append(t0 + p * plen)
            pos.append([p, -1, -1])
            for gi in range(self.G):
                ids.append(RANGES["ts_vq"].to_global(int(codes[p, gi])))
                t.append(t0 + p * plen)
                pos.append([p, gi, -1])
        return TokenSpan.make(ids, "vq", t=t, pos=pos, meta={"patches": self.P, "groups": self.G})

    def encode(self, seg: Segment) -> TokenSpan:
        return self.span_from_codes(seg.data["static"], self.codes([seg.data["obs"]])[0], seg.t_start)

    @torch.no_grad()
    def decode(self, span: TokenSpan) -> dict:
        static, i = self.decode_static(span.ids)
        rest = span.ids[i:]
        codes = np.array([RANGES["ts_vq"].to_local(int(x)) for x in rest if x != PATCH]).reshape(1, self.P, self.G)
        z = self.model.decode_codes(torch.as_tensor(codes))[0]
        return {"static": static, "grid": self._grid_out(z), "mask": None}


class TSContCodec(_TSPatchBase):
    rep = "cont"

    def alphabet_size(self) -> int:
        return 1 + len(self.lex.variables) + self.lex.n_value_bins  # static block only

    def span_from_latent(self, static: dict, z: np.ndarray, t0=0.0) -> TokenSpan:
        ids, t, pos = self.static_tokens(static, t0)
        n_static = len(ids)
        plen = HOURS // self.P * 60
        for p in range(self.P):
            for gi in range(self.G):
                ids.append(CONT)
                t.append(t0 + p * plen)
                pos.append([p, gi, -1])
        return TokenSpan.make(ids, "cont", t=t, pos=pos, cont=z.reshape(self.P * self.G, -1).astype(np.float16),
                              meta={"patches": self.P, "groups": self.G, "n_static": n_static})

    def encode(self, seg: Segment) -> TokenSpan:
        return self.span_from_latent(seg.data["static"], self.latents([seg.data["obs"]])[0].numpy(), seg.t_start)

    @torch.no_grad()
    def decode(self, span: TokenSpan) -> dict:
        static, _ = self.decode_static(span.ids)
        z = torch.as_tensor(span.cont.astype(np.float32)).view(1, self.P, self.G, -1)
        return {"static": static, "grid": self._grid_out(self.model.decode_latent(z)[0]), "mask": None}

    def span_bits(self, span, vocab_bits=None) -> float:
        n_static = span.meta["n_static"]
        return n_static * math.log2(self.alphabet_size()) + span.cont.size * 16
