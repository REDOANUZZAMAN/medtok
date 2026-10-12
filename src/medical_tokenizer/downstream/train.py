"""Training / evaluation loop for the UnifiedTokenTransformer (single- or multi-task)."""
from __future__ import annotations

import math
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..utils.device import device, to_device
from .dataset import UnifiedTokenDataset, length_bucketed_batches
from .tasks import DownstreamTask, balanced_class_weight, metrics, pick_threshold


@torch.no_grad()
def predict(model, ds: UnifiedTokenDataset, task: str, bs: int = 128) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    dev = next(model.parameters()).device
    order = np.argsort(ds.lengths(), kind="stable")
    ys, out = [], []
    for i in range(0, len(ds), bs):
        b = ds.collate([ds[j] for j in order[i:i + bs]])
        out.append(model(to_device(b, dev), task).float().cpu().numpy())
        ys.append(b["y"].numpy())
    inv = np.argsort(order)
    return np.concatenate(ys)[inv], np.concatenate(out)[inv]


def fit(model, data: dict[str, dict], tasks: dict[str, DownstreamTask], epochs: int = 20, batch_size: int = 64,
        lr: float = 5e-4, weight_decay: float = 0.01, patience: int = 5, seed: int = 0,
        class_weight: str | None = None, threshold: str | None = None, log=print) -> dict:
    """data: task -> {"train": ds, "val": ds, "test": ds}. Minibatches of all tasks are interleaved each epoch.
    Model selection: mean validation select_metric across tasks; the best state is restored before testing.
    class_weight="balanced": cross-entropy weighted by inverse training-class frequency (overrides task.class_weight).
    threshold="val_balanced_acc": binary tasks are tested at the cut-off that maximises validation balanced accuracy."""
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    dev = device()
    model.to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    steps = epochs * sum(math.ceil(len(d["train"]) / batch_size) for d in data.values())
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=max(steps, 1), pct_start=0.1)
    if class_weight not in (None, "balanced"):
        raise ValueError(f"class_weight must be null or 'balanced', got {class_weight!r}")
    if threshold not in (None, "val_balanced_acc"):
        raise ValueError(f"threshold must be null or 'val_balanced_acc', got {threshold!r}")
    weights = {k: balanced_class_weight(d["train"].labels(), tasks[k].n_classes) if class_weight == "balanced"
               else tasks[k].class_weight for k, d in data.items()}
    best, best_state, bad, hist = -math.inf, None, 0, []
    for ep in range(epochs):
        model.train()
        t0 = time.time()
        plan = [(k, idx) for k, d in data.items()
                for idx in length_bucketed_batches(d["train"].lengths(), batch_size, rng)]
        rng.shuffle(plan)
        tot = 0.0
        for k, idx in plan:
            ds = data[k]["train"]
            b = to_device(ds.collate([ds[i] for i in idx]), dev)
            w = weights[k]
            loss = F.cross_entropy(model(b, k), b["y"],
                                   weight=None if w is None else torch.tensor(w, dtype=torch.float32, device=dev))
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            tot += loss.item()
        val = {k: metrics(tasks[k], *predict(model, d["val"], k)) for k, d in data.items()}
        score = float(np.mean([val[k].get(tasks[k].select_metric, float("nan")) for k in data]))
        rec = {"epoch": ep + 1, "train_loss": tot / max(len(plan), 1), "val": val, "val_score": score,
               "sec": time.time() - t0}
        hist.append(rec)
        log(f"  ep {ep + 1:2d} loss {rec['train_loss']:.4f} val {score:.4f} ({rec['sec']:.0f}s)")
        if score > best:
            best, bad = score, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    thr = {k: pick_threshold(tasks[k], *predict(model, d["val"], k)) if threshold else None for k, d in data.items()}
    test = {k: metrics(tasks[k], *predict(model, d["test"], k), threshold=thr[k]) for k, d in data.items()}
    model.cpu()
    return {"best_val_score": best, "test": test, "history": hist, "epochs_run": len(hist),
            "class_weight": weights, "threshold": thr,
            "params": int(sum(p.numel() for p in model.parameters()))}
