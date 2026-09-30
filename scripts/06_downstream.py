"""Downstream validation of the unified tokens.

Experiments (select with --exp, results merged into artifacts/results/downstream.json):
  single     one UnifiedTransformer per (task, representation):
             mortality x {event, vq, cont}, bloodmnist x {vq, cont}, pneumoniamnist x {vq, cont}
  ablation   mortality (event tokens) with text only / time-series only
  multitask  ONE model, one vocabulary, trained jointly on mortality + bloodmnist + pneumoniamnist (VQ/event)
  baselines  non-token references: logistic regression on hand-crafted ICU features,
             small CNN on raw MedMNIST pixels
"""
import argparse
import json
import os
import pickle
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from medtok.eval.downstream import UnifiedTransformer, drop_modality, metrics, train  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")
PROC = os.path.join(ROOT, "data", "processed")
OUT = os.path.join(ROOT, "artifacts", "results", "downstream.json")
ap = argparse.ArgumentParser()
ap.add_argument("--exp", nargs="+", default=["single", "ablation", "multitask", "baselines"])
ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
ap.add_argument("--epochs", type=int, default=15)
ap.add_argument("--d", type=int, default=96)
ap.add_argument("--layers", type=int, default=3)
ap.add_argument("--threads", type=int, default=os.cpu_count())
ap.add_argument("--only", default=None, help="substring filter on run names")
args = ap.parse_args()
torch.set_num_threads(args.threads)

CONT_DIMS = {"image": 32, "timeseries": 16}
KIND = {"mortality": ("death", "binary", 2, [1.0, 3.0]), "bloodmnist": ("class", "multiclass", 8, None),
        "pneumoniamnist": ("class", "multiclass", 2, None)}
EPOCHS = {"mortality": args.epochs, "bloodmnist": args.epochs, "pneumoniamnist": args.epochs}


def load(dataset, rep, transform=None):
    name = "physionet" if dataset == "mortality" else dataset
    seqs = pickle.load(open(os.path.join(PROC, f"seqs_{name}_{rep}.pkl"), "rb"))
    if dataset == "mortality":
        stays = pickle.load(open(os.path.join(PROC, "stays.pkl"), "rb"))
        splits = [s.split for s in stays]
    else:
        splits = [s.sample_id.split("/")[1] for s in seqs]
    if transform:
        seqs = [transform(s) for s in seqs]
    lab, kind, n_cls, w = KIND[dataset]
    return {sp: [s for s, x in zip(seqs, splits) if x == sp] for sp in ("train", "val", "test")} | \
        {"label": lab, "kind": kind, "n_classes": n_cls, "weight": w}


results = json.load(open(OUT)) if os.path.exists(OUT) else {}


def save():
    json.dump(results, open(OUT, "w"), indent=1)


def run(name, task_specs, seeds, epochs):
    if args.only and args.only not in name:
        return
    runs = []
    for seed in seeds:
        print(f"== {name} seed {seed}", flush=True)
        model = UnifiedTransformer({k: t["n_classes"] for k, t in task_specs.items()}, CONT_DIMS, d=args.d,
                                   layers=args.layers)
        t0 = time.time()
        r = train(model, task_specs, epochs=epochs, seed=seed)
        r["train_sec"] = time.time() - t0
        print(f"   test {json.dumps(r['test'])}", flush=True)
        runs.append(r)
    agg = {}
    for k in runs[0]["test"]:
        for m in runs[0]["test"][k]:
            v = [r["test"][k][m] for r in runs]
            agg[f"{k}/{m}"] = {"mean": float(np.mean(v)), "std": float(np.std(v)), "values": v}
    results[name] = {"tasks": list(task_specs), "seeds": seeds, "agg": agg, "params": runs[0]["params"],
                     "train_sec_mean": float(np.mean([r["train_sec"] for r in runs])),
                     "history": [r["history"] for r in runs]}
    save()


if "single" in args.exp:
    for rep in ("event", "vq", "cont"):
        run(f"single/mortality/{rep}", {"mortality": load("mortality", rep)}, args.seeds, EPOCHS["mortality"])
    for ds in ("pneumoniamnist", "bloodmnist"):
        for rep in ("vq", "cont"):
            run(f"single/{ds}/{rep}", {ds: load(ds, rep)}, args.seeds, EPOCHS[ds])

if "ablation" in args.exp:
    run("ablation/mortality/event/text_only", {"mortality": load("mortality", "event", lambda s: drop_modality(s, "timeseries"))},
        args.seeds, EPOCHS["mortality"])
    run("ablation/mortality/event/ts_only", {"mortality": load("mortality", "event", lambda s: drop_modality(s, "text"))},
        args.seeds, EPOCHS["mortality"])

if "multitask" in args.exp:
    run("multitask/vq", {"mortality": load("mortality", "vq"), "bloodmnist": load("bloodmnist", "vq"),
                         "pneumoniamnist": load("pneumoniamnist", "vq")}, args.seeds, EPOCHS["bloodmnist"])
    run("multitask/event+vq", {"mortality": load("mortality", "event"), "bloodmnist": load("bloodmnist", "vq"),
                               "pneumoniamnist": load("pneumoniamnist", "vq")}, args.seeds, EPOCHS["bloodmnist"])

if "baselines" in args.exp:
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    from medtok.data.medmnist_data import load_subset
    from medtok.data.physionet import STATIC_VARS, TS_VARS

    stays = pickle.load(open(os.path.join(PROC, "stays.pkl"), "rb"))

    def feats(s):
        f = [np.nan_to_num(s.static[k], nan=-1) for k in STATIC_VARS]
        for v in range(len(TS_VARS)):
            x = s.obs[s.obs[:, 1] == v, 2]
            f += [x.mean(), x.min(), x.max(), x[-1], len(x)] if len(x) else [0, 0, 0, 0, 0]
        return f

    X = {sp: np.array([feats(s) for s in stays if s.split == sp]) for sp in ("train", "test")}
    Y = {sp: np.array([s.death for s in stays if s.split == sp]) for sp in ("train", "test")}
    lr = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=2000, class_weight="balanced"))
    lr.fit(X["train"], Y["train"])
    p = lr.predict_proba(X["test"])
    results["baseline/mortality/logreg_handcrafted"] = {"test": metrics("binary", Y["test"], np.log(p + 1e-9))}
    print("logreg", results["baseline/mortality/logreg_handcrafted"])

    import torch.nn as nn
    import torch.nn.functional as F
    for ds, n_cls in (("pneumoniamnist", 2), ("bloodmnist", 8)):
        d = load_subset(os.path.join(ROOT, "data", "raw", "medmnist"), ds)
        tens = {sp: (torch.as_tensor(d[sp][0]).permute(0, 3, 1, 2).float() / 127.5 - 1, torch.as_tensor(d[sp][1]))
                for sp in d}
        vals = []
        for seed in args.seeds:
            torch.manual_seed(seed)
            cnn = nn.Sequential(nn.Conv2d(3, 32, 3, padding=1), nn.GELU(), nn.MaxPool2d(2),
                                nn.Conv2d(32, 64, 3, padding=1), nn.GELU(), nn.MaxPool2d(2),
                                nn.Conv2d(64, 128, 3, padding=1), nn.GELU(), nn.AdaptiveAvgPool2d(1),
                                nn.Flatten(), nn.Linear(128, n_cls))
            opt = torch.optim.AdamW(cnn.parameters(), 1e-3)
            best, best_state = -1, None
            for ep in range(15):
                cnn.train()
                x, y = tens["train"]
                perm = torch.randperm(len(x))
                for i in range(0, len(x), 128):
                    loss = F.cross_entropy(cnn(x[perm[i:i + 128]]), y[perm[i:i + 128]])
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
                cnn.eval()
                with torch.no_grad():
                    acc = (cnn(tens["val"][0]).argmax(1) == tens["val"][1]).float().mean().item()
                if acc > best:
                    best, best_state = acc, {k: v.clone() for k, v in cnn.state_dict().items()}
            cnn.load_state_dict(best_state)
            with torch.no_grad():
                vals.append(metrics("multiclass", tens["test"][1].numpy(), cnn(tens["test"][0]).numpy()))
            print(ds, "cnn seed", seed, vals[-1], flush=True)
        results[f"baseline/{ds}/cnn_raw_pixels"] = {
            "agg": {f"{ds}/{m}": {"mean": float(np.mean([v[m] for v in vals])), "std": float(np.std([v[m] for v in vals]))}
                    for m in vals[0]}}
    save()
print("wrote", OUT)
