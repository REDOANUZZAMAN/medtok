"""Surgical-video tokenizers.

Frame payload (VQ / continuous / segment codecs):
    {"frames": (T, S, S, 3) uint8 sampled frames, "fps": sampling rate}
    T must be a multiple of the model's clip_len; clips do not overlap.
Annotation payload (AnnotatedPhaseEventCodec):
    {"phase": (T,) int phase index per sampled frame (-1 = unannotated), "fps": ...}

VideoVQCodec          per clip: <CLIP> then t' x h x w codebook ids; pos = (t', row, col)
                      with t' counted from the start of the segment.
VideoContinuousCodec  the same latent grid as <CONT> slots.
VideoSegmentCodec     UNSUPERVISED segment tokens: each clip's pooled latent is
                      assigned to the nearest of K state centroids (k-means on
                      training clips); consecutive clips in the same state merge
                      into one segment:  <SEG:k> <DT:duration>. The states are
                      clusters of visual appearance and carry NO surgical
                      meaning; decoding renders each clip from its state's mean
                      latent grid (very lossy by design).
AnnotatedPhaseEventCodec  events from expert annotations (phase transitions):
                      <EV:dataset:phase:name> <DT:gap> ... -- exactly reversible
                      at the sampled-frame resolution. Uses the label, so it is
                      never an input to phase recognition.
"""
from __future__ import annotations

import numpy as np
import torch

from ...codecs.base import Codec
from ...encoders.video import VideoClipAutoencoder
from ...protocol.schema import Segment, TokenSpan
from ...protocol.vocab import CONT, RANGES, SPECIAL, Lexicon, dt_token

CLIP = SPECIAL["<CLIP>"]
R_VQ = RANGES["video_vq"]


def make_clips(frames: np.ndarray, clip_len: int) -> np.ndarray:
    """(T,S,S,3) -> (T//clip_len, clip_len, S, S, 3); trailing frames that do not fill a clip are dropped."""
    n = len(frames) // clip_len
    return frames[:n * clip_len].reshape(n, clip_len, *frames.shape[1:])


class _VideoFrameCodec(Codec):
    modality = "video"
    reversibility = "lossy"

    def __init__(self, model: VideoClipAutoencoder, batch_size: int = 64):
        self.model, self.bs = model.eval(), batch_size
        self.L = model.clip_len
        self.gt, self.gh, self.gw = model.grid
        self.frames_per_latent = self.L // self.gt

    def raw_bits(self, seg: Segment) -> float:
        f = np.asarray(seg.data["frames"])
        return float(make_clips(f, self.L).size * 8)

    @torch.no_grad()
    def clip_latents(self, clips: np.ndarray) -> torch.Tensor:          # (N,t,h,w,dim)
        out = [self.model.continuous_latent(self.model.to_input(torch.as_tensor(clips[i:i + self.bs])))
               for i in range(0, len(clips), self.bs)]
        return torch.cat(out) if out else torch.zeros(0, self.gt, self.gh, self.gw, self.model.dim)

    @torch.no_grad()
    def decode_latents(self, z: torch.Tensor) -> np.ndarray:            # (N,t,h,w,dim) -> (N*L,S,S,3)
        out = [self.model.to_uint8(self.model.decode_latent(z[i:i + self.bs])) for i in range(0, len(z), self.bs)]
        x = torch.cat(out).numpy()
        return x.reshape(-1, *x.shape[2:])

    def _latent_times(self, n_clips: int, fps: float, t0: float) -> np.ndarray:
        """Start time (minutes) of every latent time step."""
        k = np.arange(n_clips * self.gt)
        return t0 + k * self.frames_per_latent / fps / 60


class VideoVQCodec(_VideoFrameCodec):
    rep = "vq"

    def __init__(self, model, batch_size=64):
        if not model.is_discrete:
            raise ValueError("VideoVQCodec needs a model with a VectorQuantizer")
        if model.n_codes > R_VQ.size:
            raise ValueError("codebook larger than the video_vq range")
        super().__init__(model, batch_size)

    def alphabet_size(self) -> int:
        return self.model.n_codes + 1

    @torch.no_grad()
    def codes(self, clips: np.ndarray) -> np.ndarray:                  # (N,t,h,w)
        return self.model.quantizer.codes(self.clip_latents(clips)).numpy()

    def span_from_codes(self, codes: np.ndarray, fps: float, t0: float = 0.0) -> TokenSpan:
        times = self._latent_times(len(codes), fps, t0)
        ids, t, pos = [], [], []
        for c in range(len(codes)):
            ids.append(CLIP)
            t.append(times[c * self.gt])
            pos.append([c * self.gt, -1, -1])
            for k in range(self.gt):
                tk = c * self.gt + k
                ids += (R_VQ.start + codes[c, k].ravel()).tolist()
                t += [times[tk]] * (self.gh * self.gw)
                pos += [[tk, r, q] for r in range(self.gh) for q in range(self.gw)]
        return TokenSpan.make(ids, "vq", t=t, pos=pos, meta={"clips": len(codes), "grid": list(self.model.grid),
                                                             "fps": fps})

    def encode(self, seg: Segment) -> TokenSpan:
        clips = make_clips(np.asarray(seg.data["frames"]), self.L)
        return self.span_from_codes(self.codes(clips), seg.data["fps"], seg.t_start)

    def codes_from_span(self, span: TokenSpan) -> np.ndarray:
        loc = [R_VQ.to_local(int(i)) for i in span.ids if i != CLIP]
        return np.array(loc, np.int64).reshape(-1, self.gt, self.gh, self.gw)

    def decode(self, span: TokenSpan) -> dict:
        z = self.model.quantizer.lookup(torch.as_tensor(self.codes_from_span(span)))
        return {"frames": self.decode_latents(z), "fps": span.meta["fps"]}


class VideoContinuousCodec(_VideoFrameCodec):
    rep = "continuous"

    def alphabet_size(self) -> int:
        return 1

    def span_from_latent(self, z: np.ndarray, fps: float, t0: float = 0.0) -> TokenSpan:
        n_clips = len(z)
        times = self._latent_times(n_clips, fps, t0)
        n_lat = n_clips * self.gt
        per = self.gh * self.gw
        t = np.repeat(times, per)
        pos = [[k, r, q] for k in range(n_lat) for r in range(self.gh) for q in range(self.gw)]
        return TokenSpan.make([CONT] * (n_lat * per), "continuous", t=t, pos=pos,
                              cont=z.reshape(n_lat * per, -1).astype(np.float16),
                              meta={"clips": n_clips, "grid": list(self.model.grid), "fps": fps})

    def encode(self, seg: Segment) -> TokenSpan:
        clips = make_clips(np.asarray(seg.data["frames"]), self.L)
        return self.span_from_latent(self.clip_latents(clips).numpy(), seg.data["fps"], seg.t_start)

    def decode(self, span: TokenSpan) -> dict:
        z = torch.as_tensor(span.cont.astype(np.float32)).view(-1, self.gt, self.gh, self.gw, self.model.dim)
        return {"frames": self.decode_latents(z), "fps": span.meta["fps"]}


class VideoSegmentCodec(_VideoFrameCodec):
    rep = "event"

    def __init__(self, model, lexicon: Lexicon, centroids: np.ndarray | None = None,
                 prototypes: np.ndarray | None = None, batch_size: int = 64):
        super().__init__(model, batch_size)
        self.lex = lexicon
        self.centroids = centroids        # (K, dim) in pooled-latent space
        self.prototypes = prototypes      # (K, t, h, w, dim) mean latent grid per state

    @property
    def n_states(self) -> int:
        return len(self.centroids)

    def alphabet_size(self) -> int:
        return self.n_states + RANGES["dt"].size

    @torch.no_grad()
    def fit(self, clips: np.ndarray, k: int, iters: int = 50, seed: int = 0) -> "VideoSegmentCodec":
        """k-means on pooled clip latents of training clips (k <= lexicon.n_segment_states)."""
        if k > self.lex.n_segment_states:
            raise ValueError(f"k={k} exceeds the lexicon's {self.lex.n_segment_states} segment states")
        z = self.clip_latents(clips)
        pooled = z.mean((1, 2, 3))
        g = torch.Generator().manual_seed(seed)
        c = pooled[torch.randperm(len(pooled), generator=g)[:k]].clone()
        for _ in range(iters):
            a = torch.cdist(pooled, c).argmin(1)
            for j in range(k):
                sel = a == j
                c[j] = pooled[sel].mean(0) if sel.any() else pooled[torch.randint(len(pooled), (1,), generator=g)][0]
        a = torch.cdist(pooled, c).argmin(1)
        protos = torch.stack([z[a == j].mean(0) if (a == j).any() else z.mean(0) for j in range(k)])
        self.centroids, self.prototypes = c.numpy(), protos.numpy()
        return self

    @torch.no_grad()
    def states(self, clips: np.ndarray) -> np.ndarray:
        pooled = self.clip_latents(clips).mean((1, 2, 3))
        return torch.cdist(pooled, torch.as_tensor(self.centroids)).argmin(1).numpy()

    def span_from_states(self, states: np.ndarray, fps: float, t0: float = 0.0) -> TokenSpan:
        clip_min = self.L / fps / 60
        ids, t, pos = [], [], []
        c = 0
        while c < len(states):
            e = c
            while e + 1 < len(states) and states[e + 1] == states[c]:
                e += 1
            n = e - c + 1
            ts = t0 + c * clip_min
            ids += [self.lex.segment_id(int(states[c])), dt_token(n * clip_min)]
            t += [ts, ts]
            pos += [[c, -1, -1], [c, -1, -1]]
            c = e + 1
        return TokenSpan.make(ids, "event", t=t, pos=pos, meta={"clips": len(states), "fps": fps})

    def encode(self, seg: Segment) -> TokenSpan:
        clips = make_clips(np.asarray(seg.data["frames"]), self.L)
        return self.span_from_states(self.states(clips), seg.data["fps"], seg.t_start)

    def states_from_span(self, span: TokenSpan) -> np.ndarray:
        """Per-clip state sequence, exact: segment boundaries come from the `pos` channel (start clip);
        the <DT> duration token is a (possibly rounded) summary for the sequence model."""
        n_total = span.meta["clips"]
        starts = [(int(span.pos[i][0]), RANGES["segment"].to_local(int(span.ids[i])))
                  for i in range(0, len(span.ids), 2)]
        out = np.zeros(n_total, np.int64)
        for j, (c, k) in enumerate(starts):
            out[c:starts[j + 1][0] if j + 1 < len(starts) else n_total] = k
        return out

    def decode(self, span: TokenSpan) -> dict:
        z = torch.as_tensor(self.prototypes[self.states_from_span(span)])
        return {"frames": self.decode_latents(z), "fps": span.meta["fps"]}


class AnnotatedPhaseEventCodec(Codec):
    modality, rep, reversibility = "video", "event", "exact"

    def __init__(self, lexicon: Lexicon, dataset: str, phase_names: list[str]):
        self.lex, self.dataset, self.names = lexicon, dataset, phase_names
        self._ids = [lexicon.event_id(self.event_name(p)) for p in phase_names]
        self._none = lexicon.event_id(f"{dataset}:phase:none")

    def event_name(self, phase: str) -> str:
        return f"{self.dataset}:phase:{phase}"

    def alphabet_size(self) -> int:
        return len(self.names) + 1 + RANGES["dt"].size

    def raw_bits(self, seg: Segment) -> float:
        return float(len(seg.data["phase"]) * np.ceil(np.log2(len(self.names) + 1)))

    def encode(self, seg: Segment) -> TokenSpan:
        ph, fps = np.asarray(seg.data["phase"]), seg.data["fps"]
        ids, t, pos, prev, prev_i = [], [], [], None, 0
        for i, p in enumerate(ph):
            if p == prev:
                continue
            ti = seg.t_start + i / fps / 60
            if prev is not None:
                ids.append(dt_token((i - prev_i) / fps / 60))
                t.append(ti)
                pos.append([i, -1, -1])
            ids.append(self._none if p < 0 else self._ids[int(p)])
            t.append(ti)
            pos.append([i, -1, -1])
            prev, prev_i = p, i
        return TokenSpan.make(ids, "event", t=t, pos=pos, meta={"n_frames": len(ph), "fps": fps,
                                                                "t0": seg.t_start})

    def decode(self, span: TokenSpan) -> dict:
        """Frame indices come from the exact `pos` channel, so decoding is exact for any gap length."""
        n = span.meta["n_frames"]
        ph = np.full(n, -1, np.int64)
        evs = [(int(p[0]), int(g)) for g, p in zip(span.ids, span.pos) if int(g) in RANGES["event"]]
        for k, (i, g) in enumerate(evs):
            end = evs[k + 1][0] if k + 1 < len(evs) else n
            ph[i:end] = -1 if g == self._none else self._ids.index(g)
        return {"phase": ph, "fps": span.meta["fps"]}
