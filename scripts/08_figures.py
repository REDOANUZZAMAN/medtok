"""Report figures -> docs/figures/*.png (reads artifacts/results/*)."""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

from medtok.data.physionet import TS_VARS  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")
RES = os.path.join(ROOT, "artifacts", "results")
FIG = os.path.join(ROOT, "docs", "figures")
os.makedirs(FIG, exist_ok=True)

# Colour follows the representation type, identically in every figure.
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
REP_COLOR = {"vq": "#2a78d6", "cont": "#eb6834", "event": "#1baf7a", "baseline": "#9a9893", "bpe": "#4a3aa7"}
REP_LABEL = {"vq": "VQ tokens", "cont": "Continuous projection", "event": "Event tokens", "baseline": "Non-token baseline"}
DIVERGING = LinearSegmentedColormap.from_list("bgr", ["#1c5cab", "#f0efec", "#c0302f"])
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 9, "axes.titlesize": 10,
    "axes.titleweight": "bold", "axes.axisbelow": True, "legend.frameon": False,
})
comp = json.load(open(os.path.join(RES, "compression.json")))


def save(fig, name):
    fig.savefig(os.path.join(FIG, name), dpi=160, bbox_inches="tight")
    plt.close(fig)
    print("wrote", name)


# 1. image reconstructions
for subset in ("bloodmnist", "pneumoniamnist"):
    r = np.load(os.path.join(RES, f"recon_{subset}.npy"))  # (3, 8, 28, 28, 3)
    fig, axes = plt.subplots(3, 8, figsize=(8, 3.3))
    for i, row in enumerate(("Original", "VQ (56 tokens)", "Continuous (49×32)")):
        for j in range(8):
            ax = axes[i, j]
            ax.imshow(r[i, j], interpolation="nearest")
            ax.set_xticks([]), ax.set_yticks([])
            ax.grid(False)
            for s in ax.spines.values():
                s.set_visible(False)
        axes[i, 0].set_ylabel(row, rotation=0, ha="right", va="center", color=INK2)
    fig.suptitle(f"{subset}: test-set reconstructions", fontweight="bold", x=0.02, ha="left")
    save(fig, f"recon_{subset}.png")

# 2. time-series reconstruction of one test stay (normalised units)
ts = np.load(os.path.join(RES, "recon_ts.npz"))
k = 0
panels = [("Observed (hourly grid)", np.where(ts["m"][k], ts["x"][k], np.nan)),
          ("Event tokens (bin medians)", np.where(ts["m"][k], ts["ev"][k], np.nan)),
          ("VQ tokens", ts["vq"][k]), ("Continuous projection", ts["cont"][k])]
fig, axes = plt.subplots(1, 4, figsize=(11, 4.2), sharey=True)
for ax, (title, g) in zip(axes, panels):
    im = ax.imshow(g.T, aspect="auto", cmap=DIVERGING, vmin=-3, vmax=3, interpolation="nearest")
    ax.set_title(title, loc="left")
    ax.set_xlabel("hour")
    ax.grid(False)
axes[0].set_yticks(range(len(TS_VARS)))
axes[0].set_yticklabels(TS_VARS, fontsize=6)
cb = fig.colorbar(im, ax=axes, shrink=0.8, pad=0.01)
cb.set_label("z-score (per variable)")
cb.outline.set_visible(False)
save(fig, "recon_timeseries.png")

# 3. codebook utilisation: rank-frequency of code usage on the test split
fig, axes = plt.subplots(1, 2, figsize=(9, 3.2))
for ax, key, title in ((axes[0], "image/bloodmnist/vq", "Image codebook (BloodMNIST test)"),
                       (axes[1], "timeseries/vq", "Time-series codebook (PhysioNet test)")):
    c = np.sort(np.array(comp[key]["code_counts"]))[::-1]
    p = c / c.sum()
    ax.plot(np.arange(1, len(p) + 1), np.maximum(p, 1e-7), color=REP_COLOR["vq"], lw=2)
    ax.axhline(1 / len(p), color=INK2, lw=1, ls="--")
    ax.text(len(p) * 0.98, 1 / len(p) * 1.3, "uniform", ha="right", color=INK2, fontsize=8)
    cb = comp[key]["codebook"]
    ax.set_title(title, loc="left")
    ax.text(0.98, 0.95, f"used {cb['used']}/{cb['n_codes']}\nperplexity {cb['perplexity']:.0f}",
            transform=ax.transAxes, ha="right", va="top", color=INK)
    ax.set_yscale("log")
    ax.set_xlabel("code rank")
    ax.set_ylabel("share of tokens")
save(fig, "codebook_usage.png")

# 4. compression vs reconstruction error (one panel per modality, each with its own error metric)
fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
pts = [(axes[0], "image/bloodmnist", ("vq", "cont"), "psnr_db", "PSNR (dB, higher is better)"),
       (axes[1], "timeseries", ("event", "vq", "cont"), "mae_z", "MAE on observed cells (z, lower is better)")]
for ax, base, reps, metric, ylab in pts:
    for rep in reps:
        r = comp[f"{base}/{rep}"]
        ax.scatter(r["compression_ratio"], r[metric], s=70, color=REP_COLOR[rep], edgecolor=SURFACE, lw=2, zorder=3)
        ax.annotate(f"{REP_LABEL[rep]}\n{r['tokens_per_sample']:.0f} tokens", (r["compression_ratio"], r[metric]),
                    xytext=(8, 0), textcoords="offset points", va="center", fontsize=8, color=INK)
    ax.set_xscale("log")
    ax.set_xlabel("compression ratio (raw bits / token bits, log)")
    ax.set_ylabel(ylab)
    ax.set_title("Image (BloodMNIST)" if "image" in base else "Time-series (PhysioNet 2012)", loc="left")
    ax.margins(x=0.6, y=0.25)
save(fig, "rate_distortion.png")

# 5. downstream performance
dpath = os.path.join(RES, "downstream.json")
if os.path.exists(dpath):
    d = json.load(open(dpath))
    specs = [("mortality", "mortality/auroc", "In-hospital mortality — AUROC", ("event", "vq", "cont"),
              "baseline/mortality/logreg_handcrafted"),
             ("bloodmnist", "bloodmnist/acc", "BloodMNIST — accuracy", ("vq", "cont"), "baseline/bloodmnist/cnn_raw_pixels"),
             ("pneumoniamnist", "pneumoniamnist/acc", "PneumoniaMNIST — accuracy", ("vq", "cont"),
              "baseline/pneumoniamnist/cnn_raw_pixels")]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
    for ax, (task, metric, title, reps, base) in zip(axes, specs):
        names, vals, errs, cols = [], [], [], []
        for rep in reps:
            r = d.get(f"single/{task}/{rep}")
            if r:
                names.append(REP_LABEL[rep].replace(" ", "\n", 1))
                vals.append(r["agg"][metric]["mean"])
                errs.append(r["agg"][metric]["std"])
                cols.append(REP_COLOR[rep])
        mt = d.get("multitask/event+vq") if task == "mortality" else d.get("multitask/vq")
        if mt and metric in mt["agg"]:
            names.append("Unified\nmulti-task")
            vals.append(mt["agg"][metric]["mean"])
            errs.append(mt["agg"][metric]["std"])
            cols.append(REP_COLOR["event" if task == "mortality" else "vq"])
        if base in d:
            b = d[base]
            v = b["test"]["auroc"] if "test" in b else b["agg"][metric]["mean"]
            names.append("Baseline\n(" + ("log. reg." if "logreg" in base else "CNN, pixels") + ")")
            vals.append(v)
            errs.append(0 if "test" in b else b["agg"][metric]["std"])
            cols.append(REP_COLOR["baseline"])
        x = np.arange(len(vals))
        ax.bar(x, vals, width=0.6, color=cols, yerr=errs, error_kw={"ecolor": INK2, "lw": 1, "capsize": 3},
               edgecolor=SURFACE, lw=2)
        for xi, v in zip(x, vals):
            ax.text(xi, v + 0.005, f"{v:.3f}", ha="center", va="bottom", fontsize=8, color=INK)
        lo = min(vals) - 0.08
        ax.set_ylim(max(0, lo), min(1.0, max(vals) + 0.05))
        ax.set_xticks(x)
        ax.set_xticklabels(names, fontsize=7.5)
        ax.set_title(title, loc="left")
        ax.grid(axis="x", visible=False)
    fig.text(0.01, -0.04, "Bars: mean over 3 seeds, whiskers ±1 s.d. Y-axes are truncated to show differences.",
             color=INK2, fontsize=8)
    save(fig, "downstream.png")
