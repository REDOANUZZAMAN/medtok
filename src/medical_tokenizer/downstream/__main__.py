"""python -m medical_tokenizer.downstream --config configs/downstream/mortality_vq.yaml

Trains the UnifiedTokenTransformer on token shards of one or more (task, tokenizer run)
pairs -- several pairs train ONE shared model jointly -- for every seed in `seeds`,
and writes per-seed and aggregated (mean, std) test metrics.
"""
from __future__ import annotations

import os

import numpy as np
import torch

from ..encode import encode_task
from ..protocol.serialize import drop_modality
from ..utils.config import arg_parser, load_config
from ..utils.experiment import Run, set_seed
from .dataset import UnifiedTokenDataset, cont_dim_of
from .model import UnifiedTokenTransformer
from .tasks import TASKS
from .train import fit


def run_downstream(cfg: dict) -> Run:
    run = Run(cfg, kind="downstream")
    torch.set_num_threads(int(cfg.get("threads", os.cpu_count())))
    max_len = cfg.get("max_len", 2048)
    data, tasks, cont_dims, info = {}, {}, {}, {}
    for entry in cfg["tasks"]:
        name = entry["task"]
        task = TASKS[name]
        shards = encode_task(entry["tokenizer_run"], name, entry.get("include_text", False), entry.get("data_overrides"))
        if entry.get("drop_modality"):
            shards = {s: [drop_modality(q, entry["drop_modality"]) for q in v] for s, v in shards.items()}
        frac = entry.get("train_fraction")
        if frac:
            rng = np.random.default_rng(0)
            tr = shards["train"]
            shards["train"] = [tr[i] for i in np.sort(rng.choice(len(tr), int(frac * len(tr)), replace=False))]
        data[name] = {s: UnifiedTokenDataset(v, task.label_key, max_len) for s, v in shards.items()}
        tasks[name] = task
        cd = cont_dim_of(shards["train"])
        mod = next(sp["modality"] for sp in shards["train"][0].spans if sp["rep"] == "continuous") if cd else None
        if mod:
            cont_dims[mod] = max(cont_dims.get(mod, 0), cd)
        lens = data[name]["train"].lengths()
        info[name] = {"tokenizer_run": entry["tokenizer_run"], "n": {s: len(v) for s, v in shards.items()},
                      "seq_len_mean": float(lens.mean()), "seq_len_max": int(lens.max()),
                      "truncated_frac": float((lens > max_len).mean())}
        print(name, info[name], flush=True)
    per_seed = []
    for seed in cfg.get("seeds", [cfg.get("seed", 0)]):
        set_seed(seed)
        model = UnifiedTokenTransformer({k: t.n_classes for k, t in tasks.items()}, cont_dims, **cfg.get("model", {}))
        tr = cfg.get("train", {})
        res = fit(model, data, tasks, epochs=tr.get("epochs", 20), batch_size=tr.get("batch_size", 64),
                  lr=tr.get("lr", 5e-4), weight_decay=tr.get("weight_decay", 0.01), patience=tr.get("patience", 5),
                  seed=seed, log=lambda s: print(s, flush=True))
        run.log({"seed": seed, "test": res["test"], "best_val_score": res["best_val_score"],
                 "epochs_run": res["epochs_run"], "params": res["params"]}, file="seeds.jsonl")
        run.save_json(f"history_seed{seed}.json", res["history"])
        per_seed.append(res)
    agg = {}
    for name in tasks:
        keys = sorted({k for r in per_seed for k in r["test"][name] if k != "n"})
        agg[name] = {k: {"mean": float(np.mean([r["test"][name][k] for r in per_seed if k in r["test"][name]])),
                         "std": float(np.std([r["test"][name][k] for r in per_seed if k in r["test"][name]])),
                         "values": [r["test"][name].get(k) for r in per_seed]} for k in keys}
    run.save_metrics({"tasks": info, "test": agg, "seeds": cfg.get("seeds", [cfg.get("seed", 0)]),
                      "params": per_seed[0]["params"]})
    for name in tasks:
        m = tasks[name].select_metric
        print(f"{name}: test {m} {agg[name][m]['mean']:.4f} ± {agg[name][m]['std']:.4f}", flush=True)
    return run


def main(argv=None) -> None:
    ap = arg_parser("Train and evaluate the unified-token downstream model.")
    args = ap.parse_args(argv)
    if not args.config:
        ap.error("--config is required")
    run = run_downstream(load_config(args.config, args.set))
    print("done:", run.dir)


if __name__ == "__main__":
    main()
