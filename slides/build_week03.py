"""Week 3 supervisor update deck (physiological time series). Run: .venv/bin/python slides/build_week03.py

Every result number is read from reports/tables/ (see deck_common.py); nothing is typed by hand.
"""
from pathlib import Path

from pptx.util import Inches

from deck_common import (AMBER, GREY, LIGHT, NAVY, RED, TEAL, Deck, banner, box, card, f, figure, load, n,
                         roadmap, row, table, text)

OUT = Path(__file__).with_name("week03_progress.pptx")

img = load(0)             # images (recap)
ts = load(1)              # time series: reconstruction, compression, utilisation
ts_k = load(8)            # time-series codebook size (gridding MAE column)
ts_miss = load(11)        # missing values and collapse

LABEL = {"timeseries_continuous": "A. continuous", "timeseries_vq": "B. VQ",
         "timeseries_event": "C. event (every obs.)", "timeseries_event_change": "C. event (changes only)"}


def t(exp):
    return row(ts, experiment=exp)


grid_mae = f(row(ts_k, experiment="timeseries_vq")["gridding MAE (z)"])
ev, evc, vq, ct = (t("timeseries_event"), t("timeseries_event_change"), t("timeseries_vq"),
                   t("timeseries_continuous"))

d = Deck(3, "Oct 22", OUT)

# 1 — title
d.title_slide("Week 3: physiological time series on PhysioNet 2012", "October 22, 2026")

# 2 — recap
s = d.slide("Recap: Week 2 (images)", source="reports/tables/table_00.csv")
roadmap(s, 3)
b_vq = row(img, experiment="image_vq", subset="bloodmnist")
b_ct = row(img, experiment="image_continuous", subset="bloodmnist")
b_va = row(img, experiment="image_vq_vanilla", subset="bloodmnist")
card(s, Inches(6.8), Inches(1.3), Inches(6.0), Inches(3.0), "Image results (BloodMNIST test)", [
    f"PSNR: continuous {f(b_ct['PSNR dB'])} dB, VQ {f(b_vq['PSNR dB'])} dB",
    f"VQ ratio (+overhead): {f(b_vq['ratio (+overhead)'])}×, {n(b_vq['vocab tokens'])} tokens per image",
    f"Active codes: {b_vq['active codes']} with remedies vs {b_va['active codes']} for vanilla VQ-VAE",
], size=14)
card(s, Inches(6.8), Inches(4.5), Inches(6.0), Inches(1.95), "Week-2 checkpoint passed", [
    "Data readable, compute enough (GPU re-run), both downstream tasks run",
    "No scope reduction needed",
], accent=AMBER, size=14)
text(s, Inches(0.5), Inches(5.2), Inches(6.0), Inches(1.3),
     "This week: the second fully implemented modality, ICU time series.", size=15, color=GREY)

# 3 — data and the two hard parts
s = d.slide("Work done: data, missing values, time", source="docs/datasets.md; gridding MAE from "
            "reports/tables/table_08.csv")
card(s, Inches(0.5), Inches(1.3), Inches(4.0), Inches(5.6), "Data", [
    "PhysioNet 2012 set-a, open access (ODC-By 1.0)",
    "First 48 h of each ICU stay; 37 variables + 4 static values (age, gender, height, ICU type)",
    "Patient-level split, fixed seed; normalisation and bins fitted on training stays only",
    "Task: in-hospital mortality",
], size=14)
card(s, Inches(4.7), Inches(1.3), Inches(4.0), Inches(5.6), "Missing values", [
    "Most variables are measured only a few times a day",
    "Hourly grid + an 'observed' mask; empty cells are filled with 0 for the encoder",
    "Errors are scored on observed cells only, never on filled cells",
    "Training hides a random part of the observed cells (denoising), so codes cannot just copy cells",
], accent=AMBER, size=14)
card(s, Inches(8.9), Inches(1.3), Inches(3.9), Inches(5.6), "Time encoding", [
    "<DT:k> tokens give a coarse time gap between events",
    "The exact time (minutes) is kept in a side channel on every token",
    f"Moving raw data onto the hourly grid already costs MAE {grid_mae} (z); this is shared by all methods",
], accent=NAVY, size=14)

# 4 — three representations
s = d.slide("Work done: three representations")
reps = [
    ("A. Continuous", "Window autoencoder over the hourly grid; each window gives fp16 latent "
     "vectors. Baseline only: not vocabulary tokens.", TEAL),
    ("B. Vector quantisation", f"Same autoencoder; each latent → nearest of {vq['active codes / bins']} codebook "
     "entries. Per window: <PATCH> + 4 codes.", TEAL),
    ("C. Event tokens", f"Each measurement → <VAR:name> <VBIN:k> ({ev['active codes / bins']} quantile bins per "
     "variable). Variant "
     "'changes only' writes a value only when its bin changes.", TEAL),
]
for i, (tt, dsc, col) in enumerate(reps):
    x = Inches(0.5 + i * 4.15)
    box(s, x, Inches(1.35), Inches(3.95), Inches(3.0), fill=LIGHT)
    box(s, x, Inches(1.35), Inches(3.95), Inches(0.08), fill=col)
    text(s, x + Inches(0.2), Inches(1.55), Inches(3.6), Inches(0.5), tt, size=19, bold=True, color=col)
    text(s, x + Inches(0.2), Inches(2.15), Inches(3.6), Inches(2.1), dsc, size=15)
card(s, Inches(0.5), Inches(4.6), Inches(12.3), Inches(2.3), "Same for all three", [
    "The same discrete static block (age, gender, height, ICU type) starts every sequence",
    "Same splits, same scoring (MAE / RMSE in per-variable z-units on observed cells)",
    "Event tokens learn nothing except the bins, so they are the simplest and most readable option",
], accent=GREY, size=15)

# 5 — results table
s = d.slide("Results: error, length, compression", source="reports/tables/table_01.csv")
rows = [["Representation", "positions mean", "max", "cont. vectors", "MAE (z)", "RMSE (z)", "ratio (fixed)",
         "ratio (+overhead)"]]
for exp in LABEL:
    x = t(exp)
    rows.append([LABEL[exp], f(x["positions mean"]), x["positions max"], n(x["cont. vectors"]), f(x["MAE (z)"]),
                 f(x["RMSE (z)"]), f(x["ratio (fixed)"]), f(x["ratio (+overhead)"])])
table(s, Inches(0.5), Inches(1.3), Inches(12.3), rows, col_w=[2.6, 1.5, 0.9, 1.4, 1.2, 1.2, 1.5, 2.0], size=14,
      row_h=Inches(0.5))
banner(s, Inches(3.95), f"Continuous rows: {n(ct['cont. vectors'])} positions carry fp16 vectors, NOT vocabulary "
       "tokens. ratio (+overhead) also counts span delimiters and decoding metadata.")
card(s, Inches(0.5), Inches(4.7), Inches(6.0), Inches(2.25), "Best reconstruction", [
    f"Event tokens: MAE {f(ev['MAE (z)'])} (z), close to the grid error {grid_mae}",
    "Both event variants decode to the same values on observed cells",
], size=14)
card(s, Inches(6.8), Inches(4.7), Inches(6.0), Inches(2.25), "Shortest and most compact", [
    f"VQ: {f(vq['positions mean'])} positions, ratio (+overhead) {f(vq['ratio (+overhead)'])}×",
    f"but the highest error: MAE {f(vq['MAE (z)'])} (z)",
], accent=AMBER, size=14)

# 6 — figure
s = d.slide("Results: one test stay, decoded three ways", source="reports/figures/recon_timeseries.png")
figure(s, "recon_timeseries", Inches(0.4), Inches(1.15), Inches(7.4), Inches(5.85))
card(s, Inches(8.0), Inches(1.3), Inches(4.8), Inches(5.55), "Reading the plot", [
    "Black dots = real measurements; lines = decoded values",
    "Event (green) passes through the dots: it stores each measurement as a bin",
    "Continuous (blue) follows the trend closely",
    "VQ (orange) is smoother and misses some peaks: 4 codes must describe a whole window",
    "GCS is almost constant here, so small wiggles look big on this scale",
], size=14)

# 7 — missing values decision
s = d.slide("Results: how to fill missing cells", source="reports/tables/table_11.csv")
MISS = {"timeseries_vq": "zero fill + mask (main)", "abl_ts_locf": "last value carried forward (LOCF)",
        "abl_ts_vanilla": "vanilla VQ-VAE (no remedies)"}
rows = [["VQ setting", "MAE (z)", "RMSE (z)", "active codes", "util", "perplexity", "dead %"]]
for exp, lab in MISS.items():
    x = row(ts_miss, experiment=exp)
    rows.append([lab, f(x["MAE (z)"]), f(x["RMSE (z)"]), x["active codes"], f(x["util"]), f(x["perplexity"]), f(x["dead %"])])
table(s, Inches(0.5), Inches(1.3), Inches(12.3), rows, col_w=[3.9, 1.3, 1.3, 1.6, 1.2, 1.6, 1.4], size=15,
      row_h=Inches(0.55))
locf = row(ts_miss, experiment="abl_ts_locf")
card(s, Inches(0.5), Inches(3.8), Inches(12.3), Inches(2.6), "Decision: keep zero fill + mask", [
    f"LOCF almost doubles the error (MAE {f(locf['MAE (z)'])} vs {f(vq['MAE (z)'])}), although the mask is "
    "given in both cases",
    f"LOCF also uses the codebook less evenly (perplexity {f(locf['perplexity'])} vs "
    f"{f(row(ts_miss, experiment='timeseries_vq')['perplexity'])}): copied values seem to waste codes",
    "The codebook remedies also matter here: vanilla VQ-VAE leaves codes dead",
    "Other time settings (window length, sampling interval) are ablations in Week 5",
], size=15)

# 8 — issues and decisions
s = d.slide("Issues and decisions")
cards = [
    ("Event sequences are long", f"Mean {f(ev['positions mean'])} positions, max {ev['positions max']}. "
     f"Decision: add the 'changes only' variant (mean {f(evc['positions mean'])}).", RED),
    ("VQ loses detail", f"VQ has the highest MAE ({f(vq['MAE (z)'])}). It is the most compact; Week 5 "
     "checks whether the task still works.", AMBER),
    ("Fair compression", "Same raw-bit scale for all methods; fp16 for continuous values; delimiters and "
     "metadata counted in ratio (+overhead).", NAVY),
    ("Grid error is shared", f"The hourly grid itself costs MAE {grid_mae} (z). It is reported separately so "
     "no method is blamed for it.", GREY),
]
for i, (tt, dsc, col) in enumerate(cards):
    x = Inches(0.5 + (i % 2) * 6.3)
    y = Inches(1.35 + (i // 2) * 2.85)
    box(s, x, y, Inches(6.0), Inches(2.6), fill=LIGHT)
    box(s, x, y, Inches(0.08), Inches(2.6), fill=col)
    text(s, x + Inches(0.25), y + Inches(0.15), Inches(5.6), Inches(0.5), tt, size=19, bold=True, color=col)
    text(s, x + Inches(0.25), y + Inches(0.75), Inches(5.6), Inches(1.75), dsc, size=17)

# 9 — next week + questions
s = d.slide("Next week and questions")
card(s, Inches(0.5), Inches(1.4), Inches(6.0), Inches(4.6), "Plan for Week 4: integration", [
    "Synthetic templated text (v2): round-trip and field-preservation tests",
    "Cataract-101 surgical video: VQ, continuous and phase-event tokens",
    "Point-cloud interface on synthetic surfaces",
    "Integration tests: all modalities in one sequence and one model",
])
card(s, Inches(6.8), Inches(1.4), Inches(6.0), Inches(4.6), "Questions for supervisor", [
    "Is zero fill + mask acceptable for missing values, or should I also try an interpolation baseline?",
    "Is the 48 h window with an hourly grid fine for the mortality task?",
    "For the thesis, should the main time-series method be event tokens (best error) or VQ (shortest)?",
], accent=AMBER)

d.save()
