"""Downstream task definitions and metrics.

A DownstreamTask names a label key in TokenSequence.labels, its type and class
count, and the metric used for model selection. Tasks know nothing about the
modality that produced the tokens.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from sklearn.metrics import average_precision_score, balanced_accuracy_score, f1_score, roc_auc_score


@dataclass
class DownstreamTask:
    name: str
    label_key: str
    kind: str                     # "binary" | "multiclass"
    n_classes: int
    select_metric: str            # key of metrics() used for early stopping / model selection
    class_weight: list | None = None
    description: str = ""


TASKS = {
    "mortality": DownstreamTask("mortality", "death", "binary", 2, "auroc",
                                description="PhysioNet 2012 in-hospital mortality from the first 48 h"),
    "bloodmnist": DownstreamTask("bloodmnist", "class", "multiclass", 8, "auroc",
                                 description="BloodMNIST blood-cell type (8 classes)"),
    "pneumoniamnist": DownstreamTask("pneumoniamnist", "class", "binary", 2, "auroc",
                                     description="PneumoniaMNIST pneumonia vs normal"),
    "dermamnist": DownstreamTask("dermamnist", "class", "multiclass", 7, "auroc",
                                 description="DermaMNIST skin-lesion type (7 classes)"),
    "surgical_phase": DownstreamTask("surgical_phase", "phase", "multiclass", 10, "macro_f1",
                                     description="Cataract-101 phase of the last frame of a window (online)"),
}


def metrics(task: DownstreamTask, y: np.ndarray, logits: np.ndarray) -> dict:
    p = torch.softmax(torch.as_tensor(logits, dtype=torch.float32), -1).numpy()
    pred = p.argmax(1)
    out = {"acc": float((pred == y).mean()), "balanced_acc": float(balanced_accuracy_score(y, pred)),
           "macro_f1": float(f1_score(y, pred, average="macro", labels=np.arange(task.n_classes), zero_division=0)),
           "n": int(len(y))}
    if task.kind == "binary":
        if len(np.unique(y)) == 2:
            out["auroc"] = float(roc_auc_score(y, p[:, 1]))
            out["auprc"] = float(average_precision_score(y, p[:, 1]))
    else:
        # macro one-vs-rest AUROC; equals sklearn's multi_class="ovr" when every class is present and
        # otherwise averages over the classes that occur (needed for model selection on small splits)
        per = [roc_auc_score(y == c, p[:, c]) for c in np.unique(y)] if len(np.unique(y)) > 1 else []
        if per:
            out["auroc"] = float(np.mean(per))
    return out
