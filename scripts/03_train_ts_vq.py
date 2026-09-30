"""Train the PhysioNet patch VQ autoencoder on the training split.

Loss = masked MSE on observed cells (normalised units) + commitment. During
training a random 20% of observed cells is additionally hidden from the
encoder and scored, so the codes learn temporal structure rather than
copying inputs one-to-one.
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

from medtok.data.physionet import Normalizer, to_grid  # noqa: E402
from medtok.quant.models import TSPatchVQ, masked_mse, save_model  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")
ap = argparse.ArgumentParser()
ap.add_argument("--codes", type=int, default=512)
ap.add_argument("--groups", type=int, default=4)
ap.add_argument("--dim", type=int, default=16)
ap.add_argument("--epochs", type=int, default=150)
ap.add_argument("--bs", type=int, default=64)
ap.add_argument("--hide", type=float, default=0.2)
ap.add_argument("--threads", type=int, default=os.cpu_count())
ap.add_argument("--out", default=os.path.join(ROOT, "artifacts", "ts_vq.pt"))
args = ap.parse_args()

torch.manual_seed(0)
torch.set_num_threads(args.threads)
stays = pickle.load(open(os.path.join(ROOT, "data", "processed", "stays.pkl"), "rb"))
norm = Normalizer.from_json(json.load(open(os.path.join(ROOT, "artifacts", "ts_norm.json"))))


def tensors(split):
    xs, ms = [], []
    for s in stays:
        if s.split == split:
            g, m = to_grid(s)
            xs.append(norm.grid(g, m))
            ms.append(m)
    return torch.as_tensor(np.stack(xs)), torch.as_tensor(np.stack(ms))


xtr, mtr = tensors("train")
xva, mva = tensors("val")
print("train", tuple(xtr.shape), "observed frac", round(mtr.float().mean().item(), 4))

model = TSPatchVQ(n_codes=args.codes, groups=args.groups, dim=args.dim)
opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
steps = args.epochs * (len(xtr) // args.bs)
sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=1e-3, total_steps=steps, pct_start=0.05)
log = []
for ep in range(args.epochs):
    model.train()
    perm = torch.randperm(len(xtr))
    tot, n = 0.0, 0
    for i in range(0, len(perm) - args.bs + 1, args.bs):
        x, m = xtr[perm[i:i + args.bs]], mtr[perm[i:i + args.bs]]
        hide = m & (torch.rand(m.shape) < args.hide)
        vis = m & ~hide
        xr, _, commit, st = model(x * vis, vis)
        loss = masked_mse(xr, x, m) + commit
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        tot += loss.item()
        n += 1
    if (ep + 1) % 10 == 0 or ep == 0:
        model.eval()
        with torch.no_grad():
            xr, idx, _, _ = model(xva, mva)
            mse = masked_mse(xr, xva, mva).item()
            used = torch.unique(idx).numel()
        rec = {"epoch": ep + 1, "train_loss": tot / n, "val_masked_mse": mse, "val_codes_used": used,
               "batch_ppl": st["perplexity"]}
        log.append(rec)
        print(json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in rec.items()}), flush=True)

save_model(model, args.out, log=log)
print("saved", args.out)
