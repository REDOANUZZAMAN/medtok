"""Physiological time-series tokenizers. Payload (dict):

    {"obs": (n,3) float32 [minutes, variable index, value], "static": {name: value}}

All three representations start with the same static block
    <STATIC> <VAR:Age> <VBIN:k> <VAR:Gender> <VBIN:k> ...
so that comparisons isolate the temporal part.

TSEventCodec  measurement events on the regular grid, in time order:
                [<DT:k>] <VAR:name> <VBIN:q> ...   (<DT> only when the grid step changes; a gap
                                                    over 48 h is a chain of exact <DT> tokens)
              mode="obs"     one event per observed (step, variable) cell
              mode="change"  only when a variable's bin differs from its last emitted
                             bin (run-length / piecewise-constant segmentation)
              Values are per-variable quantile bins (fitted on train); decoding
              returns the bin's training median. These events are generic
              measurement / change events, not clinical events.
TSVQCodec     grid -> windows -> `groups` codebook ids per window:
                <PATCH> q q q q  (one block per window; <MISSING> for an empty
                window when skip_empty=True)
TSContinuousCodec  the same window encoder, latents kept as <CONT> slots.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import torch

from ...codecs.base import Codec
from ...data.physionet import STATIC_VARS, TS_VARS, Normalizer, Stay, to_grid
from ...encoders.timeseries import TimeSeriesWindowAutoencoder
from ...protocol.schema import Segment, TokenSpan
from ...protocol.vocab import CONT, RANGES, SPECIAL, Lexicon, bucket_to_dt, dt_chain

STATIC = SPECIAL["<STATIC>"]
PATCH = SPECIAL["<PATCH>"]
MISSING = SPECIAL["<MISSING>"]
R_VQ = RANGES["ts_vq"]


@dataclass
class GridSpec:
    """How irregular observations are resampled before tokenisation."""
    steps: int = 48
    resolution_min: float = 60.0
    variables: list[str] = field(default_factory=lambda: list(TS_VARS))
    static_vars: list[str] = field(default_factory=lambda: list(STATIC_VARS))

    def grid(self, obs: np.ndarray):
        return to_grid(obs, self.steps, self.resolution_min, len(self.variables))


def stay_segment(stay: Stay) -> Segment:
    return Segment("timeseries", {"obs": stay.obs, "static": stay.static}, 0.0, "physionet2012",
                   {"record_id": stay.record_id})


class ValueBinner:
    """Per-variable quantile bins for static and gridded temporal values, fitted on the training split."""

    def __init__(self, n_bins: int = 16, edges: dict | None = None, centers: dict | None = None):
        self.n_bins, self.edges, self.centers = n_bins, edges or {}, centers or {}

    def fit(self, stays: list[Stay], spec: GridSpec) -> "ValueBinner":
        vals = {v: [] for v in spec.static_vars + spec.variables}
        for s in stays:
            if s.split != "train":
                continue
            for k in spec.static_vars:
                if not np.isnan(s.static[k]):
                    vals[k].append(s.static[k])
            g, m = spec.grid(s.obs)
            for j, v in enumerate(spec.variables):
                vals[v].extend(g[m[:, j], j].tolist())
        for v, x in vals.items():
            x = np.asarray(x, np.float64)
            qs = np.quantile(x, np.linspace(0, 1, self.n_bins + 1)[1:-1]) if len(x) else np.zeros(0)
            e = np.unique(qs)            # discrete variables (Gender, GCS, MechVent) collapse to fewer bins
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


class _TSBase(Codec):
    modality = "timeseries"
    reversibility = "lossy"

    def __init__(self, lexicon: Lexicon, binner: ValueBinner, spec: GridSpec):
        self.lex, self.binner, self.spec = lexicon, binner, spec

    def static_tokens(self, static: dict, t0: float):
        ids = [STATIC]
        for k in self.spec.static_vars:
            v = static.get(k, np.nan)
            if v is None or np.isnan(v):
                continue
            ids += [self.lex.var_id(k), self.lex.vbin_id(int(self.binner.bin(k, v)))]
        return ids, [t0] * len(ids), [[-1, -1, -1]] * len(ids)

    def decode_static(self, ids) -> tuple[dict, int]:
        """Parse the static block; returns (static dict, index just after it)."""
        if int(ids[0]) != STATIC:
            raise ValueError("time-series span must start with <STATIC>")
        out, i = {k: np.nan for k in self.spec.static_vars}, 1
        while i + 1 < len(ids) and int(ids[i]) in RANGES["var"] and int(ids[i + 1]) in RANGES["vbin"]:
            name = self.lex.variables[RANGES["var"].to_local(int(ids[i]))]
            if name not in self.spec.static_vars:
                break
            out[name] = float(self.binner.value(name, RANGES["vbin"].to_local(int(ids[i + 1]))))
            i += 2
        return out, i

    def raw_bits(self, seg: Segment) -> float:
        # raw record: (time, variable, value) triples at 16 + 8 + 32 bits, plus the static values as float32
        return float(len(seg.data["obs"]) * (16 + 8 + 32) + len(self.spec.static_vars) * 32)


class TSEventCodec(_TSBase):
    rep = "event"

    def __init__(self, lexicon, binner, spec, mode: str = "obs"):
        super().__init__(lexicon, binner, spec)
        if mode not in ("obs", "change"):
            raise ValueError(f"mode must be 'obs' or 'change', got {mode!r}")
        self.mode = mode
        self._var_ids = np.array([lexicon.var_id(v) for v in spec.variables])

    def alphabet_size(self) -> int:
        return 1 + len(self.lex.variables) + self.lex.n_value_bins + RANGES["dt"].size

    def encode(self, seg: Segment) -> TokenSpan:
        ids, t, pos = self.static_tokens(seg.data["static"], seg.t_start)
        g, m = self.spec.grid(seg.data["obs"])
        bins = np.zeros(m.shape, np.int64)
        for j, v in enumerate(self.spec.variables):
            if m[:, j].any():
                bins[m[:, j], j] = self.binner.bin(v, g[m[:, j], j])
        last = np.full(len(self.spec.variables), -1)
        res, prev_h = self.spec.resolution_min, 0
        for h in range(self.spec.steps):
            vs = np.flatnonzero(m[h])
            if self.mode == "change":
                vs = vs[bins[h, vs] != last[vs]]
            if not len(vs):
                continue
            th = seg.t_start + h * res
            if h != prev_h:
                dts = dt_chain((h - prev_h) * res, res)      # > 48 h gaps: several exact <DT> tokens
                ids += dts
                t += [th] * len(dts)
                pos += [[h, -1, -1]] * len(dts)
                prev_h = h
            for j in vs:
                ids += [int(self._var_ids[j]), self.lex.vbin_id(int(bins[h, j]))]
                t += [th, th]
                pos += [[h, j, -1]] * 2
            last[vs] = bins[h, vs]
        return TokenSpan.make(ids, "event", t=t, pos=pos, meta={"mode": self.mode})

    def decode(self, span: TokenSpan) -> dict:
        """-> {"static", "grid" (steps,V) values, "mask" (steps,V) emitted cells}.

        In "change" mode each variable's value is carried forward from its last event
        (the grid is filled from the first event on), matching the segmentation.
        """
        ids = span.ids
        static, i = self.decode_static(ids)
        V, steps, res = len(self.spec.variables), self.spec.steps, self.spec.resolution_min
        g = np.zeros((steps, V), np.float32)
        m = np.zeros((steps, V), bool)
        h = 0
        while i < len(ids):
            gid = int(ids[i])
            if gid in RANGES["dt"]:
                h += int(round(bucket_to_dt(RANGES["dt"].to_local(gid)) / res))
                i += 1
                continue
            name = self.lex.variables[RANGES["var"].to_local(gid)]
            k = RANGES["vbin"].to_local(int(ids[i + 1]))
            j = self.spec.variables.index(name)
            g[min(h, steps - 1), j] = self.binner.value(name, k)
            m[min(h, steps - 1), j] = True
            i += 2
        if span.meta.get("mode", self.mode) == "change":
            filled = g.copy()
            for j in range(V):
                last = np.nan
                for hh in range(steps):
                    last = g[hh, j] if m[hh, j] else last
                    filled[hh, j] = 0.0 if np.isnan(last) else last
            g = filled
        return {"static": static, "grid": g, "mask": m}


class _TSWindowBase(_TSBase):
    def __init__(self, lexicon, binner, spec, model: TimeSeriesWindowAutoencoder, normalizer: Normalizer,
                 fill: str = "zero", batch_size: int = 256):
        super().__init__(lexicon, binner, spec)
        if model.steps != spec.steps or model.n_vars != len(spec.variables):
            raise ValueError("model grid does not match GridSpec")
        self.model, self.norm, self.fill, self.bs = model.eval(), normalizer, fill, batch_size
        self.W, self.G = model.n_windows, model.groups
        self.win_min = model.stride * spec.resolution_min

    def tensors(self, obs_list) -> tuple[torch.Tensor, torch.Tensor]:
        xs, ms = [], []
        for obs in obs_list:
            g, m = self.spec.grid(obs)
            xs.append(self.norm.grid(g, m, self.fill))
            ms.append(m)
        return torch.as_tensor(np.stack(xs)), torch.as_tensor(np.stack(ms))

    @torch.no_grad()
    def latents(self, obs_list) -> torch.Tensor:          # (B,W,G,dim)
        out = []
        for i in range(0, len(obs_list), self.bs):
            out.append(self.model.continuous_latent(self.tensors(obs_list[i:i + self.bs])))
        return torch.cat(out)

    def window_observed(self, obs) -> np.ndarray:
        _, m = self.spec.grid(obs)
        w = torch.as_tensor(m).float()[None].unfold(1, self.model.window, self.model.stride)
        return (w.sum((2, 3)) > 0)[0].numpy()

    @torch.no_grad()
    def decode_grid(self, z: torch.Tensor) -> np.ndarray:
        """Latents (B,W,G,dim) -> value grids in original units (B,steps,V)."""
        return self.norm.inverse_grid(self.model.decode_latent(z).numpy())


class TSVQCodec(_TSWindowBase):
    rep = "vq"

    def __init__(self, *a, skip_empty: bool = False, **kw):
        super().__init__(*a, **kw)
        if not self.model.is_discrete:
            raise ValueError("TSVQCodec needs a model with a VectorQuantizer")
        if self.model.n_codes > R_VQ.size:
            raise ValueError("codebook larger than the ts_vq range")
        self.skip_empty = skip_empty
        self._empty_codes = None

    def alphabet_size(self) -> int:
        return self.model.n_codes + 2                     # codes + <PATCH> + <MISSING>

    @torch.no_grad()
    def codes(self, obs_list) -> np.ndarray:              # (B,W,G)
        return self.model.quantizer.codes(self.latents(obs_list)).numpy()

    def empty_codes(self) -> np.ndarray:
        """Codes of a fully unobserved series (what <MISSING> windows decode to)."""
        if self._empty_codes is None:
            self._empty_codes = self.codes([np.zeros((0, 3), np.float32)])[0]
        return self._empty_codes

    def span_from_codes(self, static: dict, codes: np.ndarray, observed: np.ndarray | None = None,
                        t0: float = 0.0) -> TokenSpan:
        ids, t, pos = self.static_tokens(static, t0)
        for w in range(self.W):
            tw = t0 + w * self.win_min
            if self.skip_empty and observed is not None and not observed[w]:
                ids.append(MISSING)
                t.append(tw)
                pos.append([w, -1, -1])
                continue
            ids.append(PATCH)
            t.append(tw)
            pos.append([w, -1, -1])
            ids += (R_VQ.start + codes[w]).tolist()
            t += [tw] * self.G
            pos += [[w, gi, -1] for gi in range(self.G)]
        return TokenSpan.make(ids, "vq", t=t, pos=pos, meta={"windows": self.W, "groups": self.G})

    def encode(self, seg: Segment) -> TokenSpan:
        return self.encode_batch([seg])[0]

    def encode_batch(self, segs):
        codes = self.codes([s.data["obs"] for s in segs])
        return [self.span_from_codes(s.data["static"], c,
                                     self.window_observed(s.data["obs"]) if self.skip_empty else None, s.t_start)
                for s, c in zip(segs, codes)]

    def codes_from_span(self, span: TokenSpan) -> tuple[dict, np.ndarray]:
        static, i = self.decode_static(span.ids)
        rest, codes, w = span.ids[i:].tolist(), np.zeros((self.W, self.G), np.int64), 0
        j = 0
        while j < len(rest):
            if rest[j] == MISSING:
                codes[w] = self.empty_codes()[w]
                j += 1
            else:                                          # <PATCH> q*G
                codes[w] = [R_VQ.to_local(x) for x in rest[j + 1:j + 1 + self.G]]
                j += 1 + self.G
            w += 1
        return static, codes

    def decode(self, span: TokenSpan) -> dict:
        static, codes = self.codes_from_span(span)
        z = self.model.quantizer.lookup(torch.as_tensor(codes)[None])
        return {"static": static, "grid": self.decode_grid(z)[0], "mask": None}


class TSContinuousCodec(_TSWindowBase):
    rep = "continuous"

    def alphabet_size(self) -> int:
        return 1 + len(self.lex.variables) + self.lex.n_value_bins      # the discrete static block

    def span_from_latent(self, static: dict, z: np.ndarray, t0: float = 0.0) -> TokenSpan:
        ids, t, pos = self.static_tokens(static, t0)
        for w in range(self.W):
            ids += [CONT] * self.G
            t += [t0 + w * self.win_min] * self.G
            pos += [[w, gi, -1] for gi in range(self.G)]
        return TokenSpan.make(ids, "continuous", t=t, pos=pos, cont=z.reshape(self.W * self.G, -1).astype(np.float16),
                              meta={"windows": self.W, "groups": self.G})

    def encode(self, seg: Segment) -> TokenSpan:
        return self.encode_batch([seg])[0]

    def encode_batch(self, segs):
        z = self.latents([s.data["obs"] for s in segs]).numpy()
        return [self.span_from_latent(s.data["static"], zz, s.t_start) for s, zz in zip(segs, z)]

    def decode(self, span: TokenSpan) -> dict:
        static, _ = self.decode_static(span.ids)
        z = torch.as_tensor(span.cont.astype(np.float32)).view(1, self.W, self.G, -1)
        return {"static": static, "grid": self.decode_grid(z)[0], "mask": None}

    def payload_bits(self, span: TokenSpan) -> float:
        n_disc = int((span.ids != CONT).sum())
        return n_disc * math.log2(self.alphabet_size()) + span.cont.size * 16
