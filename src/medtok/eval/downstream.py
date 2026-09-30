"""Downstream model over unified token sequences.

UnifiedTransformer reads any TokenSequence, whatever mix of modalities and
representation types it contains:
    x_i = E_tok[id_i] + E_mod[mod_i] + TimeEnc(t_i) + E_pos0[pos_i0] + E_pos1[pos_i1] + E_pos2[pos_i2]
At <cont> slots the token embedding is replaced by a per-modality linear
projection of the continuous payload, so continuous and discrete inputs share
the same backbone. The time / spatial side channels realise the protocol's
positional encodings. A pre-LN Transformer encoder is followed by masked mean
pooling and one linear head per task.
"""
from __future__ import annotations

import math
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score

from ..protocol.schema import TokenSequence
from ..protocol.vocab import MODALITIES, SPECIAL, VOCAB_SIZE

CONT = SPECIAL["<cont>"]
PAD = SPECIAL["<pad>"]
MAX_POS = 64


class TimeEncoding(nn.Module):
    """Sinusoidal encoding of absolute time (hours), learned vector for 'n/a'."""

    def __init__(self, d):
        super().__init__()
        self.register_buffer("freq", torch.exp(torch.arange(0, d, 2).float() * (-math.log(1000.0) / d)))
        self.na = nn.Parameter(torch.zeros(d))
        self.proj = nn.Linear(d, d)

    def forward(self, t_min):
        h = (t_min / 60.0).clamp(min=0)[..., None] * self.freq
        enc = self.proj(torch.cat([h.sin(), h.cos()], -1))
        return torch.where((t_min < 0)[..., None], self.na.expand_as(enc), enc)


class UnifiedTransformer(nn.Module):
    def __init__(self, tasks: dict[str, int], cont_dims: dict[str, int] | None = None, d=128, layers=4,
                 heads=4, dropout=0.1):
        super().__init__()
        self.tok = nn.Embedding(VOCAB_SIZE, d, padding_idx=PAD)
        self.mod = nn.Embedding(len(MODALITIES) + 1, d)
        self.time = TimeEncoding(d)
        self.pos = nn.ModuleList([nn.Embedding(MAX_POS + 1, d) for _ in range(3)])
        self.cont = nn.ModuleDict({m: nn.Linear(k, d) for m, k in (cont_dims or {}).items()})
        layer = nn.TransformerEncoderLayer(d, heads, 4 * d, dropout, batch_first=True, norm_first=True,
                                           activation="gelu")
        self.enc = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d)
        self.heads = nn.ModuleDict({k: nn.Linear(d, n) for k, n in tasks.items()})

    def embed(self, b):
        x = self.tok(b["ids"]) + self.mod(b["mod"] + 1) + self.time(b["t"])
        for k in range(3):
            x = x + self.pos[k]((b["pos"][..., k] + 1).clamp(0, MAX_POS))
        for m, proj in self.cont.items():
            sel = b["cont_mask"] & (b["mod"] == MODALITIES.index(m))
            if sel.any():
                x = x.clone()
                x[sel] = x[sel] - self.tok.weight[CONT] + proj(b["cont"][sel][:, :proj.in_features])
        return x

    def forward(self, b, task):
        pad = b["ids"] == PAD
        h = self.norm(self.enc(self.embed(b), src_key_padding_mask=pad))
        keep = (~pad).float()[..., None]
        pooled = (h * keep).sum(1) / keep.sum(1)
        return self.heads[task](pooled)


def collate(seqs: list[TokenSequence], label_key: str, max_len: int = 1536, cont_dim: int = 32) -> dict:
    L = min(max(len(s) for s in seqs), max_len)
    B = len(seqs)
    ids = np.full((B, L), PAD, np.int64)
    mod = np.full((B, L), -1, np.int64)
    t = np.full((B, L), -1.0, np.float32)
    pos = np.full((B, L, 3), -1, np.int64)
    cont = np.zeros((B, L, cont_dim), np.float32)
    for i, s in enumerate(seqs):
        n = min(len(s), L)
        ids[i, :n], mod[i, :n], t[i, :n], pos[i, :n] = s.ids[:n], s.mod[:n], s.t[:n], s.pos[:n]
        ci = 0
        for sp in s.spans:
            if sp["rep"] != "cont":
                continue
            c = s.cont[ci].astype(np.float32)
            ci += 1
            st = sp["start"]
            where = np.flatnonzero(s.ids[st:sp["end"]] == CONT) + st
            where = where[where < n]
            cont[i, where, :c.shape[1]] = c[:len(where)]
    y = np.array([s.labels[label_key] for s in seqs])
    return {"ids": torch.as_tensor(ids), "mod": torch.as_tensor(mod), "t": torch.as_tensor(t),
            "pos": torch.as_tensor(pos), "cont": torch.as_tensor(cont),
            "cont_mask": torch.as_tensor(ids == CONT), "y": torch.as_tensor(y)}


def drop_modality(seq: TokenSequence, modality: str) -> TokenSequence:
    """Remove every span of one modality (with its delimiters and dt token) from a sequence."""
    from ..protocol.vocab import RANGES
    keep = np.ones(len(seq), bool)
    spans, conts, ci = [], [], 0
    for sp in seq.spans:
        is_cont = sp["rep"] == "cont"
        if sp["modality"] == modality:
            a = sp["start"] - 2
            if a > 0 and int(seq.ids[a - 1]) in RANGES["dt"]:
                a -= 1
            keep[a:sp["end"] + 1] = False
        else:
            spans.append(dict(sp))
            if is_cont:
                conts.append(seq.cont[ci])
        ci += is_cont
    new_index = np.cumsum(keep) - 1
    for sp in spans:
        sp["start"], sp["end"] = int(new_index[sp["start"]]), int(new_index[sp["end"] - 1] + 1)
    return TokenSequence(seq.ids[keep], seq.mod[keep], seq.t[keep], seq.pos[keep], spans, conts,
                         seq.sample_id, seq.labels)


def metrics(task_kind: str, y: np.ndarray, logits: np.ndarray) -> dict:
    p = torch.softmax(torch.as_tensor(logits), -1).numpy()
    if task_kind == "binary":
        return {"auroc": float(roc_auc_score(y, p[:, 1])), "auprc": float(average_precision_score(y, p[:, 1])),
                "acc": float((p.argmax(1) == y).mean())}
    out = {"acc": float((p.argmax(1) == y).mean())}
    try:
        out["auroc_macro_ovr"] = float(roc_auc_score(y, p, multi_class="ovr"))
    except ValueError:
        pass
    return out


@torch.no_grad()
def predict(model, seqs, task, label_key, bs=128, max_len=1536):
    model.eval()
    order = np.argsort([len(s) for s in seqs], kind="stable")  # length-sorted batches, restored below
    out, ys = [], []
    for i in range(0, len(seqs), bs):
        b = collate([seqs[j] for j in order[i:i + bs]], label_key, max_len)
        out.append(model(b, task).numpy())
        ys.append(b["y"].numpy())
    inv = np.argsort(order)
    return np.concatenate(ys)[inv], np.concatenate(out)[inv]


def train(model, tasks: dict, epochs=20, bs=64, lr=5e-4, max_len=1536, patience=5, seed=0, log=print):
    """tasks: name -> dict(train=[seqs], val=[seqs], test=[seqs], label=str, kind=binary|multiclass, weight=None).

    Multi-task: each epoch interleaves minibatches of every task (sampled
    proportionally to size). Model selection on mean val score across tasks.
    """
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    n_steps = epochs * sum(math.ceil(len(t["train"]) / bs) for t in tasks.values())
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=n_steps, pct_start=0.1)
    key = lambda k, m: m["auroc"] if tasks[k]["kind"] == "binary" else m["acc"]  # noqa: E731
    best, best_state, bad, hist = -1, None, 0, []
    for ep in range(epochs):
        model.train()
        t0 = time.time()
        plan = []
        for k, t in tasks.items():
            # length-bucketed batches: shuffle, sort within windows of 20 batches, then split
            perm = rng.permutation(len(t["train"]))
            win = bs * 20
            for w in range(0, len(perm), win):
                chunk = sorted(perm[w:w + win], key=lambda i: len(t["train"][i]))
                plan += [(k, chunk[i:i + bs]) for i in range(0, len(chunk), bs)]
        rng.shuffle(plan)
        tot = 0.0
        for k, idx in plan:
            t = tasks[k]
            b = collate([t["train"][i] for i in idx], t["label"], max_len)
            w = t.get("weight")
            loss = F.cross_entropy(model(b, k), b["y"], weight=None if w is None else torch.as_tensor(w))
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            tot += loss.item()
        val = {k: metrics(t["kind"], *predict(model, t["val"], k, t["label"], max_len=max_len)) for k, t in tasks.items()}
        score = float(np.mean([key(k, m) for k, m in val.items()]))
        hist.append({"epoch": ep + 1, "loss": tot / len(plan), "val": val, "sec": time.time() - t0})
        log(f"  ep {ep + 1:2d} loss {tot / len(plan):.4f} val {score:.4f} "
            + " ".join(f"{k}:{key(k, m):.4f}" for k, m in val.items()) + f" ({time.time() - t0:.0f}s)")
        if score > best:
            best, bad = score, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    test = {k: metrics(t["kind"], *predict(model, t["test"], k, t["label"], max_len=max_len)) for k, t in tasks.items()}
    return {"best_val": best, "test": test, "history": hist,
            "params": sum(p.numel() for p in model.parameters())}
