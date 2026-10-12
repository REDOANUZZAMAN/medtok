"""Aggregate experiment outputs into research tables (Markdown + CSV) and figures (PNG).

    python -m medical_tokenizer.report [--out reports]

Reads ONLY files written by train / evaluate / downstream runs (the `latest` run of each
experiment). Missing experiments are listed as missing, never filled in.
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os

import numpy as np

from .data.datasets import abspath
from .utils.experiment import REPO_ROOT

# representation -> fixed colour (entity-bound, never by rank); reference palette slots 1-3
REP_COLOR = {"continuous": "#2a78d6", "vq": "#eb6834", "event": "#1baf7a"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def _load(path):
    with open(path) as f:
        return json.load(f)


def latest(name: str, file: str, root: str = "experiments"):
    p = os.path.join(abspath(root), name, "latest", file)
    return _load(p) if os.path.exists(p) else None


def _fmt(v, nd=3):
    if v is None:
        return "–"
    if isinstance(v, float):
        return f"{v:.{nd}f}" if abs(v) < 1e5 else f"{v:.3g}"
    return str(v)


class Table:
    def __init__(self, title: str, cols: list[str], note: str = ""):
        self.title, self.cols, self.rows, self.note = title, cols, [], note

    def add(self, *vals):
        self.rows.append(list(vals))

    def markdown(self) -> str:
        out = [f"### {self.title}", ""]
        if self.note:
            out += [self.note, ""]
        out.append("| " + " | ".join(self.cols) + " |")
        out.append("|" + "|".join("---" for _ in self.cols) + "|")
        out += ["| " + " | ".join(_fmt(v) for v in r) + " |" for r in self.rows]
        return "\n".join(out) + "\n"

    def write_csv(self, path):
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(self.cols)
            w.writerows(self.rows)


# ------------------------------------------------------------------ tables
def tokenizer_tables(missing: list) -> list[Table]:
    tabs = []
    t = Table("Images (MedMNIST test sets): reconstruction, compression, utilisation",
              ["experiment", "rep", "subset", "positions", "vocab tokens", "cont. vectors", "PSNR dB", "SSIM", "MAE",
               "ratio (fixed)", "ratio (+overhead)", "ratio (entropy)", "active codes", "util", "perplexity"],
              "Continuous rows are the continuous baseline: their positions carry fp16 vectors and are NOT "
              "vocabulary tokens. '+overhead' adds span delimiters and decoding metadata (docs/token_protocol.md §7).")
    for name in ("image_continuous", "image_vq", "image_vq_vanilla"):
        m = latest(name, "eval/metrics.json")
        if m is None:
            missing.append(name)
            continue
        for sub in ("bloodmnist", "pneumoniamnist", "dermamnist"):
            if sub not in m:
                continue
            r, c, u = m[sub]["reconstruction"], m[sub]["compression"], m[sub].get("utilization", {})
            t.add(name, m["representation"], sub, c["tokens_per_sample_mean"], c.get("discrete_tokens_per_sample"),
                  c.get("continuous_vectors_per_sample"), r["psnr_db"], r["ssim"], r["mae"], c["compression_ratio"],
                  c.get("compression_ratio_with_overhead"), c.get("entropy_coded_ratio"), u.get("active_codes"),
                  u.get("utilization"), u.get("perplexity"))
    tabs.append(t)
    t = Table("Physiological time-series (PhysioNet 2012 test stays)",
              ["experiment", "rep", "positions mean", "positions max", "cont. vectors", "MAE (z)", "RMSE (z)",
               "ratio (fixed)", "ratio (+overhead)", "ratio (entropy)", "seq. reduction vs obs", "active codes / bins",
               "util"],
              "Errors on observed cells of the hourly grid, per-variable z-units. Gridding error (raw -> hourly "
              "grid, shared by all representations) is listed separately. " + "Continuous rows are the continuous baseline: their positions carry fp16 vectors and are NOT "
              "vocabulary tokens. '+overhead' adds span delimiters and decoding metadata (docs/token_protocol.md §7).")
    grid_err = None
    for name in ("timeseries_continuous", "timeseries_vq", "timeseries_event", "timeseries_event_change"):
        m = latest(name, "eval/metrics.json")
        if m is None:
            missing.append(name)
            continue
        r, c = m["reconstruction"], m["compression"]
        u = m.get("utilization", {})
        u = u.get("value_bins", u)
        grid_err = m.get("gridding_mae_z", grid_err)
        t.add(name, m["representation"], c["tokens_per_sample_mean"], c["tokens_per_sample_max"],
              c.get("continuous_vectors_per_sample"), r["mae_z"], r["rmse_z"], c["compression_ratio"],
              c.get("compression_ratio_with_overhead"), c.get("entropy_coded_ratio"), c.get("sequence_reduction"),
              u.get("active_codes"), u.get("utilization"))
    if grid_err is not None:
        t.note += f" Gridding MAE (z) = {grid_err:.3f}."
    tabs.append(t)
    t = Table("Surgical video (Cataract-101 test videos)",
              ["experiment", "rep", "positions/min", "PSNR dB", "SSIM", "ratio (fixed)", "ratio (+overhead)",
               "ratio (entropy)", "active codes / states", "util"],
              "Cataract-101 (10 phases, video-level split) replaces Cholec80; results are not comparable with "
              "Cholec80 results. The annotated-phase-event row encodes ground-truth labels: it measures annotation "
              "encoding, not phase recognition from raw video. " + "Continuous rows are the continuous baseline: their positions carry fp16 vectors and are NOT "
              "vocabulary tokens. '+overhead' adds span delimiters and decoding metadata (docs/token_protocol.md §7).")
    for name in ("video_continuous", "video_vq", "video_segments"):
        m = latest(name, "eval/metrics.json")
        if m is None:
            missing.append(name)
            continue
        r, c, u = m["reconstruction"], m["compression"], m.get("utilization", {})
        t.add(name, m["representation"], c["tokens_per_minute"], r["psnr_db"], r["ssim"], c["compression_ratio"],
              c.get("compression_ratio_with_overhead"), c.get("entropy_coded_ratio"), u.get("active_codes"),
              u.get("utilization"))
        if "annotated_phase_events" in m:
            a = m["annotated_phase_events"]
            t.add("annotated phase events (ground-truth labels; annotation encoding only)", "event",
                  a.get("tokens_per_minute"), None, None, a["compression_ratio"],
                  a.get("compression_ratio_with_overhead"), a.get("entropy_coded_ratio"), None, None)
    tabs.append(t)
    return tabs


# headline metrics: AUROC + macro-F1 for images, AUROC + AUPRC for ICU mortality
DS = [("bloodmnist", ["ds_bloodmnist_continuous", "ds_bloodmnist_vq"], ["auroc", "macro_f1", "acc"]),
      ("pneumoniamnist", ["ds_pneumoniamnist_continuous", "ds_pneumoniamnist_vq"], ["auroc", "macro_f1", "acc"]),
      ("dermamnist", ["ds_dermamnist_continuous", "ds_dermamnist_vq"], ["auroc", "macro_f1", "acc"]),
      ("mortality", ["ds_mortality_continuous", "ds_mortality_vq", "ds_mortality_event", "ds_mortality_event_change",
                     "ds_mortality_event_text"], ["auroc", "auprc", "balanced_acc"]),
      ("surgical_phase", ["ds_phase_continuous", "ds_phase_vq", "ds_phase_segments"], ["macro_f1", "acc", "balanced_acc"])]


def downstream_table(missing: list) -> Table:
    t = Table("Downstream tasks (test set, mean ± std over seeds)",
              ["task", "experiment", "seq len (train mean)", "metric 1", "metric 2", "metric 3", "seeds"],
              "Fixed splits and an identical model / training budget per task. Images: AUROC and macro-F1; "
              "ICU mortality: AUROC and AUPRC. Text in ds_mortality_event_text is synthetic (admission-time fields "
              "only, no outcome). Surgical phase uses Cataract-101 and is not comparable with Cholec80 results.")
    for task, names, keys in DS + [("multitask", ["ds_multitask_vq"], None)]:
        for name in names:
            m = latest(name, "metrics.json")
            if m is None:
                missing.append(name)
                continue
            for tk, res in m["test"].items():
                ks = keys or [TASK_KEY[tk]]
                cells = [f"{k} {res[k]['mean']:.3f} ± {res[k]['std']:.3f}" if k in res else "–" for k in ks]
                cells += ["–"] * (3 - len(cells))
                t.add(tk if task == "multitask" else task, name + (" (joint)" if task == "multitask" else ""),
                      m["tasks"][tk]["seq_len_mean"], *cells, len(m["seeds"]))
    return t


TASK_KEY = {"bloodmnist": "auroc", "pneumoniamnist": "auroc", "dermamnist": "auroc", "mortality": "auroc",
            "surgical_phase": "macro_f1"}


def ablation_tables(missing: list) -> list[Table]:
    groups = [
        ("Image codebook size (5 epochs)", ["abl_image_k64", "abl_image_k128", "abl_image_k256", "abl_image_k512",
                                            "abl_image_k1024"]),
        ("Image token dimension (5 epochs)", ["abl_image_dim8", "abl_image_k512", "abl_image_dim64"]),
        ("Image patch size (patch-embedding architecture, 5 epochs)", ["abl_image_patch2", "abl_image_patch4",
                                                                       "abl_image_patch7"]),
        ("Image codebook-collapse remedies (K=512, 5 epochs)", ["abl_image_k512", "abl_image_norestart",
                                                               "abl_image_nocosine", "abl_image_lossupdate",
                                                               "abl_image_vanilla"]),
    ]
    tabs = []
    for title, names in groups:
        t = Table(title, ["experiment", "tokens", "PSNR dB (blood)", "SSIM (blood)", "active codes", "util",
                          "perplexity", "dead %", "ratio (fixed)"])
        for n in names:
            m = latest(n, "eval/metrics.json")
            if m is None:
                missing.append(n)
                continue
            b, u = m["bloodmnist"], m.get("all_subsets_utilization", {})
            t.add(n, b["compression"]["tokens_per_sample_mean"], b["reconstruction"]["psnr_db"],
                  b["reconstruction"]["ssim"], u.get("active_codes"), u.get("utilization"), u.get("perplexity"),
                  u.get("dead_pct"), b["compression"]["compression_ratio"])
        tabs.append(t)
    ts_groups = [
        ("Time-series codebook size", ["abl_ts_k64", "abl_ts_k128", "timeseries_vq", "abl_ts_k1024"]),
        ("Time-series window size / stride (4 tokens per window)", ["abl_ts_win2", "timeseries_vq", "abl_ts_win8",
                                                                     "abl_ts_win8_stride4"]),
        ("Time-series sampling interval (4 h windows)", ["abl_ts_res30", "timeseries_vq", "abl_ts_res120"]),
        ("Time-series missing values and collapse", ["timeseries_vq", "abl_ts_locf", "abl_ts_vanilla"]),
    ]
    for title, names in ts_groups:
        t = Table(title, ["experiment", "tokens mean", "MAE (z)", "RMSE (z)", "gridding MAE (z)", "active codes",
                          "util", "perplexity", "dead %"])
        for n in names:
            m = latest(n, "eval/metrics.json")
            if m is None:
                missing.append(n)
                continue
            u = m.get("utilization", {})
            t.add(n, m["compression"]["tokens_per_sample_mean"], m["reconstruction"]["mae_z"],
                  m["reconstruction"]["rmse_z"], m.get("gridding_mae_z"), u.get("active_codes"), u.get("utilization"),
                  u.get("perplexity"), u.get("dead_pct"))
        tabs.append(t)
    t = Table("Video temporal sampling rate and codebook size (5 epochs)",
              ["experiment", "tokens/min", "PSNR dB", "SSIM", "active codes", "util", "perplexity"])
    for n in ["abl_video_fps0.5", "abl_video_fps1", "abl_video_fps2", "abl_video_k128", "abl_video_k1024"]:
        m = latest(n, "eval/metrics.json")
        if m is None:
            missing.append(n)
            continue
        u = m.get("utilization", {})
        t.add(n, m["compression"]["tokens_per_minute"], m["reconstruction"]["psnr_db"], m["reconstruction"]["ssim"],
              u.get("active_codes"), u.get("utilization"), u.get("perplexity"))
    tabs.append(t)
    return tabs


def interface_table(missing: list) -> Table | None:
    m = latest("interfaces", "metrics.json")
    if m is None:
        missing.append("interfaces")
        return None
    t = Table("Interface modalities",
              ["modality", "setting", "exact round-trip / Chamfer", "field preservation", "tokens mean", "ratio",
               "ratio (+overhead)"],
              "Text: SYNTHETIC templated notes (not clinical free text); accepted on lossless round-trip, field "
              "preservation and protocol compatibility only, no claim about clinical language understanding. "
              "Point clouds: synthetic surfaces, non-learned features; Chamfer relative to object extent.")
    tx = m["text"]
    t.add("text", f"BPE vocab {tx['vocab_size']}, templates v{tx.get('template_version', 1)}",
          tx["exact_roundtrip_rate"], tx.get("field_preservation_rate"), tx["compression"]["tokens_per_sample_mean"],
          tx["compression"]["compression_ratio"], tx["compression"].get("compression_ratio_with_overhead"))
    for k, v in m["pointcloud_synthetic"].items():
        if isinstance(v, dict):
            t.add("pointcloud", k, v["chamfer_rel_extent"], None, v["compression"]["tokens_per_sample_mean"],
                  v["compression"]["compression_ratio"], v["compression"].get("compression_ratio_with_overhead"))
    return t


# ------------------------------------------------------------------ figures
def _style(ax, title=None, xlabel=None, ylabel=None):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK2)
        ax.spines[s].set_linewidth(0.8)
    ax.tick_params(colors=INK2, labelsize=8, width=0.8)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    if title:
        ax.set_title(title, fontsize=10, color=INK, loc="left")
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=8, color=INK2)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=8, color=INK2)


def figures(out: str) -> list[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    made = []

    def save(fig, name):
        p = os.path.join(out, name)
        fig.savefig(p, dpi=160, bbox_inches="tight", facecolor="#fcfcfb")
        plt.close(fig)
        made.append(p)

    # 1. original vs reconstructed images (VQ and continuous)
    ev_vq = os.path.join(abspath("experiments"), "image_vq", "latest", "eval", "recon_examples.npz")
    ev_ct = os.path.join(abspath("experiments"), "image_continuous", "latest", "eval", "recon_examples.npz")
    if os.path.exists(ev_vq) and os.path.exists(ev_ct):
        a, b = np.load(ev_vq), np.load(ev_ct)
        subs = [s for s in a.files]
        fig, axes = plt.subplots(3 * len(subs), 8, figsize=(8, 3 * len(subs)))
        for si, s in enumerate(subs):
            for j in range(8):
                for r, (img, lab) in enumerate(((a[s][0][j], "original"), (a[s][1][j], "VQ"), (b[s][1][j], "continuous"))):
                    ax = axes[3 * si + r, j]
                    ax.imshow(img, cmap="gray" if img.shape[-1] == 1 else None)
                    ax.set_xticks([])
                    ax.set_yticks([])
                    if j == 0:
                        ax.set_ylabel(f"{s}\n{lab}" if r == 0 else lab, fontsize=7, color=INK2)
        fig.suptitle("MedMNIST test images: original vs reconstruction", fontsize=10, x=0.02, ha="left")
        save(fig, "recon_images.png")

    # 2. original vs reconstructed time-series (one test stay, a few variables)
    reps = {r: os.path.join(abspath("experiments"), f"timeseries_{r}", "latest", "eval", "recon_examples.npz")
            for r in ("continuous", "vq", "event")}
    if all(os.path.exists(p) for p in reps.values()):
        from .data.physionet import TS_VARS
        d = {r: np.load(p, allow_pickle=True) for r, p in reps.items()}
        zt, mk = d["vq"]["z_true"][0], d["vq"]["mask"][0]
        vars_ = [v for v in ("HR", "MAP", "Temp", "GCS") if mk[:, TS_VARS.index(v)].sum() > 3][:3]
        fig, axes = plt.subplots(len(vars_), 1, figsize=(7, 1.8 * len(vars_)), sharex=True)
        for ax, v in zip(np.atleast_1d(axes), vars_):
            j = TS_VARS.index(v)
            h = np.flatnonzero(mk[:, j])
            ax.plot(h, zt[h, j], "o", color=INK, ms=4, label="observed", zorder=3)
            for r in ("continuous", "vq", "event"):
                zr = d[r]["z_rec"][0]
                ax.plot(h, zr[h, j], "-", lw=1.5, color=REP_COLOR[r], label=r)
            _style(ax, None, None, f"{v} (z)")
        axes[0].legend(fontsize=7, frameon=False, ncol=4, loc="upper left", bbox_to_anchor=(0, 1.35))
        axes[-1].set_xlabel("hour since admission", fontsize=8, color=INK2)
        fig.suptitle("PhysioNet test stay: observed vs reconstructed (observed cells)", fontsize=10, x=0.02,
                     ha="left", y=1.02)
        save(fig, "recon_timeseries.png")

    # 3. codebook usage histograms (sorted frequency, log scale)
    fig, ax = plt.subplots(figsize=(7, 3))
    any_ = False
    for name, lab, col in (("image_vq", "image VQ (EMA + cosine + restarts)", REP_COLOR["vq"]),
                           ("image_vq_vanilla", "image VQ-VAE vanilla", INK2),
                           ("timeseries_vq", "time-series VQ", REP_COLOR["continuous"]),
                           ("video_vq", "video VQ", REP_COLOR["event"])):
        p = os.path.join(abspath("experiments"), name, "latest", "eval", "code_counts.npy")
        if not os.path.exists(p):
            continue
        c = np.sort(np.load(p))[::-1].astype(float)
        c = c / c.sum()
        ax.plot(np.arange(1, len(c) + 1), np.maximum(c, 1e-7), lw=2, color=col, label=lab)
        any_ = True
    if any_:
        ax.set_yscale("log")
        _style(ax, "Codebook usage (test set), codes sorted by frequency", "code rank", "share of tokens (log)")
        ax.legend(fontsize=7, frameon=False)
        save(fig, "codebook_usage.png")
    else:
        plt.close(fig)

    # 4. compression vs reconstruction per modality (one small panel per modality, one axis each)
    pts = []
    for name, mod in (("image_continuous", "image"), ("image_vq", "image"), ("timeseries_continuous", "timeseries"),
                      ("timeseries_vq", "timeseries"), ("timeseries_event", "timeseries"),
                      ("timeseries_event_change", "timeseries"), ("video_continuous", "video"), ("video_vq", "video"),
                      ("video_segments", "video")):
        m = latest(name, "eval/metrics.json")
        if m is None:
            continue
        if mod == "image":
            c, r = m["bloodmnist"]["compression"], m["bloodmnist"]["reconstruction"]["psnr_db"]
            tok = c["tokens_per_sample_mean"]
        elif mod == "timeseries":
            c, r = m["compression"], m["reconstruction"]["mae_z"]
            tok = c["tokens_per_sample_mean"]
        else:
            c, r = m["compression"], m["reconstruction"]["psnr_db"]
            tok = c["tokens_per_minute"]
        pts.append((mod, name, m["representation"], c["compression_ratio"], r, tok))
    mods = [mm for mm in ("image", "timeseries", "video") if any(p[0] == mm for p in pts)]
    if mods:
        fig, axes = plt.subplots(2, len(mods), figsize=(3.4 * len(mods), 5.2), squeeze=False)
        for k, mod in enumerate(mods):
            sel = [p for p in pts if p[0] == mod]
            labels = [p[1].replace(f"{mod}_", "") for p in sel]
            cols = [REP_COLOR[p[2]] for p in sel]
            ax = axes[0, k]
            ax.bar(labels, [p[3] for p in sel], color=cols, width=0.6, edgecolor="#fcfcfb", linewidth=2)
            for i, p in enumerate(sel):
                ax.annotate(f"{p[3]:.1f}x", (i, p[3]), ha="center", va="bottom", fontsize=7, color=INK)
            ax.set_yscale("log")
            _style(ax, f"{mod}: storage compression", None, "raw bits / token bits (log)")
            ax.tick_params(axis="x", labelrotation=30, labelsize=7)
            ax = axes[1, k]
            ax.bar(labels, [p[5] for p in sel], color=cols, width=0.6, edgecolor="#fcfcfb", linewidth=2)
            for i, p in enumerate(sel):
                ax.annotate(f"{p[5]:.0f}", (i, p[5]), ha="center", va="bottom", fontsize=7, color=INK)
            _style(ax, f"{mod}: sequence length", None, "tokens per minute" if mod == "video" else "tokens per sample")
            ax.tick_params(axis="x", labelrotation=30, labelsize=7)
        fig.tight_layout()
        save(fig, "compression_sequence_length.png")
        fig, axes = plt.subplots(1, len(mods), figsize=(3.4 * len(mods), 2.8), squeeze=False)
        for k, mod in enumerate(mods):
            sel = [p for p in pts if p[0] == mod]
            ax = axes[0, k]
            labels = [p[1].replace(f"{mod}_", "") for p in sel]
            ax.bar(labels, [p[4] for p in sel], color=[REP_COLOR[p[2]] for p in sel], width=0.6,
                   edgecolor="#fcfcfb", linewidth=2)
            for i, p in enumerate(sel):
                ax.annotate(f"{p[4]:.2f}", (i, p[4]), ha="center", va="bottom", fontsize=7, color=INK)
            _style(ax, f"{mod}: reconstruction", None,
                   "MAE, z-units (lower is better)" if mod == "timeseries" else "PSNR dB (higher is better)")
            ax.tick_params(axis="x", labelrotation=30, labelsize=7)
        fig.tight_layout()
        save(fig, "reconstruction_error.png")

    # 5. downstream comparison (mean ± std per task, one panel per task)
    rows = []
    for task, names, keys in DS:
        for n in names:
            m = latest(n, "metrics.json")
            if m is None or task not in m["test"]:
                continue
            k = TASK_KEY[task]
            rep = "continuous" if "continuous" in n else "vq" if n.endswith("_vq") else "event"
            rows.append((task, n.replace(f"ds_{task.replace('surgical_', '')}_", "").replace("ds_", ""), rep,
                         m["test"][task][k]["mean"], m["test"][task][k]["std"], k))
    tasks = [t for t, _, _ in DS if any(r[0] == t for r in rows)]
    if tasks:
        fig, axes = plt.subplots(1, len(tasks), figsize=(2.9 * len(tasks), 3), squeeze=False)
        for k, t in enumerate(tasks):
            sel = [r for r in rows if r[0] == t]
            ax = axes[0, k]
            x = np.arange(len(sel))
            ax.bar(x, [r[3] for r in sel], yerr=[r[4] for r in sel], color=[REP_COLOR[r[2]] for r in sel], width=0.6,
                   edgecolor="#fcfcfb", linewidth=2, error_kw={"elinewidth": 1, "ecolor": INK2, "capsize": 2})
            for i, r in enumerate(sel):
                ax.annotate(f"{r[3]:.3f}", (i, r[3] + r[4]), ha="center", va="bottom", fontsize=7, color=INK)
            ax.set_xticks(x, [r[1] for r in sel], rotation=30, fontsize=7)
            lo = min(r[3] - r[4] for r in sel)
            ax.set_ylim(max(0, lo - 0.1), min(1.0, max(r[3] + r[4] for r in sel) + 0.06))
            _style(ax, t, None, f"test {sel[0][5]}")
        fig.tight_layout()
        save(fig, "downstream.png")

    # 6. codebook size ablation: utilisation and PSNR vs K (two panels, one axis each)
    ks, util, psnr = [], [], []
    for k in (64, 128, 256, 512, 1024):
        m = latest(f"abl_image_k{k}", "eval/metrics.json")
        if m:
            ks.append(k)
            util.append(m["all_subsets_utilization"]["perplexity"])
            psnr.append(m["bloodmnist"]["reconstruction"]["psnr_db"])
    if len(ks) >= 2:
        fig, axes = plt.subplots(1, 2, figsize=(7, 2.8))
        axes[0].plot(ks, ks, "--", color=GRID, lw=1.5, label="all codes equally used")
        axes[0].plot(ks, util, "-o", color=REP_COLOR["vq"], lw=2, ms=5, label="perplexity (effective codes)")
        axes[0].set_xscale("log", base=2)
        axes[0].set_yscale("log", base=2)
        _style(axes[0], "Effective codebook size", "codebook size K", "codes")
        axes[0].legend(fontsize=7, frameon=False)
        axes[1].plot(ks, psnr, "-o", color=REP_COLOR["vq"], lw=2, ms=5)
        axes[1].set_xscale("log", base=2)
        _style(axes[1], "Reconstruction (BloodMNIST)", "codebook size K", "PSNR dB")
        fig.tight_layout()
        save(fig, "ablation_codebook_size.png")

    # 7. t-SNE of learned image representations (continuous latents, pooled), coloured by class
    made += _tsne(out, plt)
    return made


def _tsne(out, plt) -> list[str]:
    run = os.path.join(abspath("experiments"), "image_continuous", "latest")
    if not os.path.exists(os.path.join(run, "model.pt")):
        return []
    from sklearn.manifold import TSNE
    from .data import datasets as D
    from .toolkit import load_tokenizer
    from .utils.experiment import load_run_config
    cfg = load_run_config(os.path.realpath(run))
    codec = load_tokenizer(run)
    x, y, _, _ = D.image_split(cfg["data"], "test", ["bloodmnist"])
    idx = np.random.default_rng(0).choice(len(x), min(1500, len(x)), replace=False)
    z = codec.latents(x[idx]).numpy().reshape(len(idx), -1)
    emb = TSNE(2, random_state=0, init="pca", perplexity=30).fit_transform(z)
    from .data.medmnist import CLASS_NAMES
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    cmap = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
    for c in range(8):
        s = y[idx] == c
        ax.scatter(emb[s, 0], emb[s, 1], s=6, color=cmap[c], label=CLASS_NAMES["bloodmnist"][c], linewidths=0)
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title("t-SNE of continuous image latents (BloodMNIST test, 1,500 cells)", fontsize=9, loc="left")
    ax.legend(fontsize=6, frameon=False, markerscale=2, loc="center left", bbox_to_anchor=(1, 0.5))
    p = os.path.join(out, "tsne_image_latents.png")
    fig.savefig(p, dpi=160, bbox_inches="tight", facecolor="#fcfcfb")
    plt.close(fig)
    return [p]


def token_sequence_figure(out, plt) -> list[str]:
    """Visualise one real multimodal token sequence: token type per position."""
    from .protocol.storage import load_shard
    from .protocol.tokens import token_type
    from .data.notes import TEMPLATE_VERSION
    paths = sorted(glob.glob(os.path.join(abspath("experiments"), "timeseries_event", "latest", "tokens",
                                          f"mortality_text_v{TEMPLATE_VERSION}", "test.npz")))
    if not paths:
        return []
    seq = load_shard(paths[0])[0][0]
    types = [token_type(int(i)).value for i in seq.ids]
    names = sorted(set(types), key=types.index)
    pal = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
    fig, ax = plt.subplots(figsize=(8, 1.6))
    for k, n in enumerate(names):
        xs = [i for i, t in enumerate(types) if t == n]
        ax.scatter(xs, [0] * len(xs), marker="|", s=300, color=pal[k % len(pal)] if k < 8 else INK2, label=n,
                   linewidths=1)
    ax.set_yticks([])
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.set_xlabel("position in sequence", fontsize=8, color=INK2)
    ax.set_title(f"Token types along one test sequence (templated note + ICU events, {len(seq)} tokens)",
                 fontsize=9, loc="left")
    ax.legend(fontsize=6, frameon=False, ncol=4, loc="upper left", bbox_to_anchor=(0, -0.45))
    p = os.path.join(out, "token_sequence.png")
    fig.savefig(p, dpi=160, bbox_inches="tight", facecolor="#fcfcfb")
    plt.close(fig)
    return [p]


def build_report(out: str = "reports") -> str:
    out = abspath(out)
    os.makedirs(os.path.join(out, "tables"), exist_ok=True)
    os.makedirs(os.path.join(out, "figures"), exist_ok=True)
    missing: list[str] = []
    tabs = tokenizer_tables(missing) + [downstream_table(missing)] + ablation_tables(missing)
    it = interface_table(missing)
    if it:
        tabs.append(it)
    md = ["# Results (generated)", "",
          "Generated by `python -m medical_tokenizer.report` from the `latest` run of each experiment. "
          "Do not edit by hand; rerun the command.", ""]
    for i, t in enumerate(tabs):
        if not t.rows:
            continue
        md.append(t.markdown())
        t.write_csv(os.path.join(out, "tables", f"table_{i:02d}.csv"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figs = figures(os.path.join(out, "figures")) + token_sequence_figure(os.path.join(out, "figures"), plt)
    md += ["## Figures", ""] + [f"![{os.path.basename(f)}](figures/{os.path.basename(f)})" for f in figs]
    if missing:
        md += ["", "## Not yet run", "", ", ".join(sorted(set(missing)))]
    with open(os.path.join(out, "RESULTS.md"), "w") as f:
        f.write("\n".join(md) + "\n")
    print(f"report: {os.path.relpath(out, REPO_ROOT)}/RESULTS.md, {len(figs)} figures, "
          f"{len(set(missing))} experiments missing")
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Build result tables and figures from experiment outputs.")
    ap.add_argument("--out", default="reports")
    args = ap.parse_args(argv)
    build_report(args.out)


if __name__ == "__main__":
    main()
