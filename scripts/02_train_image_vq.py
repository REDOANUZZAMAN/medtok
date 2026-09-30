"""Train the MedMNIST image VQ-VAE on the union of the training splits.

usage: 02_train_image_vq.py [--codes 512] [--epochs 15] [--out artifacts/image_vq.pt]
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from medtok.data.medmnist_data import SUBSETS, load_all  # noqa: E402
from medtok.quant.models import ImageVQVAE, save_model  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")
ap = argparse.ArgumentParser()
ap.add_argument("--codes", type=int, default=512)
ap.add_argument("--dim", type=int, default=32)
ap.add_argument("--epochs", type=int, default=15)
ap.add_argument("--bs", type=int, default=128)
ap.add_argument("--out", default=os.path.join(ROOT, "artifacts", "image_vq.pt"))
args = ap.parse_args()

torch.manual_seed(0)
torch.set_num_threads(os.cpu_count())
data = load_all(os.path.join(ROOT, "data", "raw", "medmnist"))
xtr = torch.as_tensor(np.concatenate([data[s]["train"][0] for s in SUBSETS]))
xva = torch.as_tensor(np.concatenate([data[s]["val"][0] for s in SUBSETS]))
print("train images", tuple(xtr.shape), "val", tuple(xva.shape))

model = ImageVQVAE(n_codes=args.codes, dim=args.dim)
opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
steps = args.epochs * (len(xtr) // args.bs)
sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=2e-3, total_steps=steps, pct_start=0.05)
log = []
for ep in range(args.epochs):
    model.train()
    perm = torch.randperm(len(xtr))
    t0, tot, n, ppl = time.time(), 0.0, 0, []
    for i in range(0, len(perm) - args.bs + 1, args.bs):
        x = model.to_input(xtr[perm[i:i + args.bs]])
        xr, _, commit, st = model(x)
        loss = F.mse_loss(xr, x) + commit
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        tot += loss.item() * len(x)
        n += len(x)
        ppl.append(st["perplexity"])
    model.eval()
    with torch.no_grad():
        idx_all, mse = [], 0.0
        for i in range(0, len(xva), 512):
            x = model.to_input(xva[i:i + 512])
            xr, idx, _, _ = model(x)
            mse += F.mse_loss(xr.clamp(-1, 1), x, reduction="sum").item()
            idx_all.append(idx.flatten())
        mse /= xva.numel()
        used = torch.unique(torch.cat(idx_all)).numel()
    psnr = 10 * np.log10(4 / mse)  # range [-1,1] -> peak-to-peak 2
    rec = {"epoch": ep + 1, "train_loss": tot / n, "val_mse": mse, "val_psnr": psnr,
           "batch_ppl": float(np.mean(ppl)), "val_codes_used": used, "sec": time.time() - t0}
    log.append(rec)
    print(json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in rec.items()}), flush=True)

save_model(model, args.out, log=log)
print("saved", args.out)
