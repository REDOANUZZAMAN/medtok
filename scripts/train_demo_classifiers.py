"""Train and save one image classifier per MedMNIST task for the demo app (VQ tokens -> class).

    .venv/bin/python scripts/train_demo_classifiers.py [--tasks bloodmnist pneumoniamnist dermamnist]

The thesis experiments (experiments/ds_*) report mean ± std over 3 seeds and do not keep weights. The demo
needs one model it can run, so this script repeats the ds_<task>_vq setup (configs/downstream/<task>_vq.yaml,
same tokenizer run, splits, model and training budget) for seed 0 only and saves
checkpoints/demo/<task>_vq.pt = {"cfg", "state", "test", "config"}. Test metrics of this single model are
stored with it; the thesis numbers stay the 3-seed ones in reports/.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

import torch                                                        # noqa: E402

from medical_tokenizer.downstream.dataset import UnifiedTokenDataset  # noqa: E402
from medical_tokenizer.downstream.model import UnifiedTokenTransformer  # noqa: E402
from medical_tokenizer.downstream.tasks import TASKS                 # noqa: E402
from medical_tokenizer.downstream.train import fit                   # noqa: E402
from medical_tokenizer.encode import encode_task                     # noqa: E402
from medical_tokenizer.utils.config import load_config               # noqa: E402
from medical_tokenizer.utils.experiment import set_seed               # noqa: E402

OUT = os.path.join(ROOT, "checkpoints", "demo")


def train(task_name: str, seed: int = 0) -> None:
    cfg = load_config(os.path.join(ROOT, "configs", "downstream", f"{task_name}_vq.yaml"))
    entry = cfg["tasks"][0]
    task = TASKS[task_name]
    shards = encode_task(entry["tokenizer_run"], task_name)
    data = {task_name: {s: UnifiedTokenDataset(v, task.label_key, cfg.get("max_len", 2048)) for s, v in shards.items()}}
    set_seed(seed)
    model = UnifiedTokenTransformer({task_name: task.n_classes}, {}, **cfg.get("model", {}))
    tr = cfg.get("train", {})
    t0 = time.time()
    res = fit(model, data, {task_name: task}, epochs=tr.get("epochs", 20), batch_size=tr.get("batch_size", 64),
              lr=tr.get("lr", 5e-4), weight_decay=tr.get("weight_decay", 0.01), patience=tr.get("patience", 5),
              seed=seed, log=lambda s: print(s, flush=True))
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, f"{task_name}_vq.pt")
    torch.save({"cfg": model.cfg, "state": model.state_dict(), "test": res["test"][task_name], "seed": seed,
                "epochs_run": res["epochs_run"], "config": cfg["_config_path"],
                "tokenizer_run": entry["tokenizer_run"]}, path)
    m = res["test"][task_name]
    print(f"{task_name}: saved {path} | test auroc {m.get('auroc', float('nan')):.4f} macro_f1 {m['macro_f1']:.4f} "
          f"({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--tasks", nargs="+", default=["bloodmnist", "pneumoniamnist", "dermamnist"])
    ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    for t in a.tasks:
        train(t)
