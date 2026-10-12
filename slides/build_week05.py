"""Week 5 supervisor update deck (downstream tasks and ablations). Run: .venv/bin/python slides/build_week05.py

Every result number is read from reports/tables/ (see deck_common.py); nothing is typed by hand.
Downstream metrics are copied verbatim from the 'metric k' strings of table_03 (e.g. 'auroc 0.983 ± 0.002').
"""
from pathlib import Path

from pptx.util import Inches

from deck_common import (AMBER, GREY, LIGHT, NAVY, RED, TEAL, Deck, banner, box, card, f, figure, load, n,
                         roadmap, row, table, text)

OUT = Path(__file__).with_name("week05_progress.pptx")

vid = load(2)             # video (recap)
iface = load(13)          # text / point cloud (recap)
ds = load(3)              # downstream
abl_img_k = load(4)       # image codebook size
abl_ts_k = load(8)        # time-series codebook size
abl_ts_win = load(9)      # time-series window / stride
abl_ts_res = load(10)     # time-series sampling interval
SEEDS = ds[0]["seeds"]
assert all(r["seeds"] == SEEDS for r in ds)


def metric(exp, k):
    """('auroc', '0.983 ± 0.002') from the verbatim 'metric k' cell."""
    name, val = row(ds, experiment=exp)[f"metric {k}"].split(" ", 1)
    return name, val


def m(exp, k):
    return metric(exp, k)[1]


d = Deck(5, "Nov 5", OUT)

# 1 — title
d.title_slide("Week 5: downstream tasks and ablations", "November 5, 2026")

# 2 — recap
s = d.slide("Recap: Week 4 (integration)", source="reports/tables/table_02.csv, table_13.csv")
roadmap(s, 5)
txt = row(iface, modality="text")
vvq = row(vid, experiment="video_vq")
card(s, Inches(6.8), Inches(1.3), Inches(6.0), Inches(3.4), "Done in Week 4", [
    f"Synthetic text v2: exact round-trip {f(txt['exact round-trip / Chamfer'])}, field preservation "
    f"{f(txt['field preservation'])}",
    f"Cataract-101 video VQ: PSNR {f(vvq['PSNR dB'])} dB, util {f(vvq['util'])}",
    "Point-cloud interface tested on synthetic surfaces",
    "Integration tests: one model reads image, ICU and video tokens",
], size=14)
banner(s, Inches(4.95), "Text is synthetic; Cataract-101 is not comparable with Cholec80.", x=Inches(6.8),
       w=Inches(6.0), size=13)

# 3 — setup
s = d.slide("Work done: one model, fixed budget")
card(s, Inches(0.5), Inches(1.3), Inches(6.0), Inches(3.5), "Same model for every run", [
    "UnifiedTokenTransformer, trained from scratch on the token shards",
    "It sees only protocol fields, never raw data",
    "Same size and training budget for every representation of a task",
    "Early stopping on validation; best checkpoint tested once",
    f"{SEEDS} seeds; numbers are mean ± std",
], size=14)
rows = [["Task", "Data", "Headline metrics"],
        ["Image classification", "Blood / Pneumonia / Derma MNIST", "AUROC, macro-F1"],
        ["ICU mortality", "PhysioNet 2012", "AUROC, AUPRC"],
        ["Surgical phase (extra)", "Cataract-101", "macro-F1"],
        ["Multi-task (extra)", "all three, one model", "per task"]]
table(s, Inches(6.8), Inches(1.3), Inches(6.0), rows, col_w=[2.2, 2.3, 1.5], size=13, row_h=Inches(0.62))
card(s, Inches(0.5), Inches(5.0), Inches(12.3), Inches(1.9), "Why these metrics", [
    "AUROC: ranking quality, does not depend on a threshold",
    "macro-F1 (images) and AUPRC (mortality) show how the rare classes do; mortality is an imbalanced task",
], accent=GREY, size=14)

# 4 — downstream figure
s = d.slide("Results: all downstream tasks", source="reports/figures/downstream.png, reports/tables/table_03.csv")
figure(s, "downstream", Inches(0.3), Inches(1.2), Inches(12.7), Inches(2.6))
text(s, Inches(0.5), Inches(4.0), Inches(12.3), Inches(0.5),
     f"Test AUROC (surgical phase: macro-F1), mean ± std over {SEEDS} seeds. Colours: blue continuous, orange VQ, "
     "green event / segments.", size=14, color=GREY)
card(s, Inches(0.5), Inches(4.65), Inches(6.0), Inches(2.25), "Images", [
    "Continuous is best on all three subsets",
    "VQ is a little lower, but still clearly learns the task",
], size=14)
card(s, Inches(6.8), Inches(4.65), Inches(6.0), Inches(2.25), "ICU mortality", [
    "Event tokens reach about the same AUROC as continuous, with real discrete tokens",
    "VQ is the weakest on mortality",
], accent=AMBER, size=14)

# 5 — image classification table
s = d.slide("Results: image classification", source="reports/tables/table_03.csv")
rows = [["Subset", "Method", "seq len", "AUROC", "macro-F1", "accuracy"]]
for sub, name in (("bloodmnist", "BloodMNIST"), ("pneumoniamnist", "PneumoniaMNIST"), ("dermamnist", "DermaMNIST")):
    for rep in ("continuous", "vq"):
        exp = f"ds_{sub}_{rep}"
        assert [metric(exp, k)[0] for k in (1, 2, 3)] == ["auroc", "macro_f1", "acc"]
        rows.append([name if rep == "continuous" else "", "continuous" if rep == "continuous" else "VQ",
                     n(row(ds, experiment=exp)["seq len (train mean)"]), m(exp, 1), m(exp, 2), m(exp, 3)])
table(s, Inches(0.5), Inches(1.3), Inches(12.3), rows, col_w=[2.4, 1.8, 1.4, 2.2, 2.2, 2.3], size=15,
      row_h=Inches(0.5))
card(s, Inches(0.5), Inches(5.0), Inches(12.3), Inches(1.9), "Reading", [
    "The gap between VQ and continuous is small in AUROC and larger in macro-F1",
    "DermaMNIST macro-F1 is low for both: some lesion classes are very rare",
], size=15)

# 6 — mortality + extra tasks
s = d.slide("Results: ICU mortality and extra tasks", source="reports/tables/table_03.csv")
MORT = {"ds_mortality_continuous": "A. continuous", "ds_mortality_vq": "B. VQ",
        "ds_mortality_event": "C. event (every obs.)", "ds_mortality_event_change": "C. event (changes only)",
        "ds_mortality_event_text": "C. event + synthetic note"}
rows = [["Representation", "seq len", "AUROC", "AUPRC"]]
for exp, lab in MORT.items():
    assert [metric(exp, k)[0] for k in (1, 2)] == ["auroc", "auprc"]
    rows.append([lab, f(row(ds, experiment=exp)["seq len (train mean)"]), m(exp, 1), m(exp, 2)])
table(s, Inches(0.5), Inches(1.3), Inches(7.3), rows, col_w=[2.7, 1.2, 1.7, 1.7], size=13, row_h=Inches(0.5))
banner(s, Inches(4.5), "The note is SYNTHETIC (admission fields only, no outcome): a layout test, not a clinical "
       "text result.", w=Inches(7.3), h=Inches(0.65), size=13)
rows = [["Extra task", "Method", "macro-F1"]]
for exp, lab in (("ds_phase_continuous", "continuous"), ("ds_phase_vq", "VQ"), ("ds_phase_segments", "segments")):
    assert metric(exp, 1)[0] == "macro_f1"
    rows.append(["Surgical phase" if lab == "continuous" else "", lab, m(exp, 1)])
table(s, Inches(8.1), Inches(1.3), Inches(4.7), rows, col_w=[1.8, 1.3, 1.6], size=13, row_h=Inches(0.45))
mt = [r for r in ds if r["experiment"] == "ds_multitask_vq (joint)"]
rows = [["Multi-task (one VQ model)", "metric"]]
for r in mt:
    rows.append([r["task"], r["metric 1"]])
table(s, Inches(8.1), Inches(3.3), Inches(4.7), rows, col_w=[2.3, 2.4], size=13, row_h=Inches(0.45))
banner(s, Inches(5.3), "Surgical phase: Cataract-101, not comparable with Cholec80. Scores are low; it stays an "
       "extra task.", x=Inches(8.1), w=Inches(4.7), h=Inches(0.85), size=12)
text(s, Inches(0.5), Inches(5.3), Inches(7.3), Inches(1.6), [
    "Event tokens (every observation) come close to continuous AUROC but need much longer sequences",
    "One multi-task model reaches close to the single-task VQ scores",
], size=14, bullet=True)

# 7 — codebook-size ablation
s = d.slide("Ablation: codebook size", source="reports/tables/table_04.csv, table_08.csv, "
            "reports/figures/ablation_codebook_size.png")
figure(s, "ablation_codebook_size", Inches(0.3), Inches(1.15), Inches(7.2), Inches(2.9))
rows = [["Image K", "PSNR (blood)", "perplexity", "ratio (fixed)"]]
for r in abl_img_k:
    rows.append([r["experiment"].removeprefix("abl_image_k"), f(r["PSNR dB (blood)"]),
                 f(r["perplexity"]), f(r["ratio (fixed)"])])
table(s, Inches(7.7), Inches(1.25), Inches(5.1), rows, col_w=[1.1, 1.4, 1.3, 1.3], size=12, row_h=Inches(0.4))
rows = [["Time-series run", "MAE (z)", "active codes", "util"]]
for r in abl_ts_k:
    rows.append([r["experiment"], f(r["MAE (z)"]), r["active codes"], f(r["util"])])
table(s, Inches(0.5), Inches(4.25), Inches(6.9), rows, col_w=[2.7, 1.4, 1.6, 1.2], size=12, row_h=Inches(0.4))
card(s, Inches(7.7), Inches(4.0), Inches(5.1), Inches(2.9), "Reading", [
    "Bigger codebook: better reconstruction, every code still used",
    "But fewer bits saved per token (lower ratio)",
    "Time series: error also falls as K grows",
    "Image runs here use a short 5-epoch budget",
], size=13)

# 8 — time-granularity ablation
s = d.slide("Ablation: time granularity", source="reports/tables/table_09.csv, table_10.csv")
WIN = {"abl_ts_win2": "2 h windows", "timeseries_vq": "4 h windows (main)", "abl_ts_win8": "8 h windows",
       "abl_ts_win8_stride4": "8 h windows, stride 4 h"}
rows = [["Window", "tokens mean", "MAE (z)", "RMSE (z)"]]
for exp, lab in WIN.items():
    r = row(abl_ts_win, experiment=exp)
    rows.append([lab, f(r["tokens mean"]), f(r["MAE (z)"]), f(r["RMSE (z)"])])
table(s, Inches(0.5), Inches(1.3), Inches(6.0), rows, col_w=[2.6, 1.2, 1.1, 1.1], size=13, row_h=Inches(0.5))
RES = {"abl_ts_res30": "30 min grid", "timeseries_vq": "60 min grid (main)", "abl_ts_res120": "120 min grid"}
rows = [["Sampling interval", "MAE (z)", "gridding MAE (z)"]]
for exp, lab in RES.items():
    r = row(abl_ts_res, experiment=exp)
    rows.append([lab, f(r["MAE (z)"]), f(r["gridding MAE (z)"])])
table(s, Inches(6.8), Inches(1.3), Inches(6.0), rows, col_w=[2.6, 1.5, 1.9], size=13, row_h=Inches(0.5))
card(s, Inches(0.5), Inches(4.3), Inches(6.0), Inches(2.6), "Window length", [
    "Shorter windows: lower error, but more tokens",
    "Overlapping 8 h windows recover part of the loss",
    "4 h kept as the balance point",
], size=14)
card(s, Inches(6.8), Inches(4.3), Inches(6.0), Inches(2.6), "Sampling interval", [
    "A coarse grid looks easier for the tokenizer (lower MAE) ...",
    "... but the grid itself loses more (higher gridding MAE)",
    "So both errors must be reported together; 60 min kept",
], accent=AMBER, size=14)

# 9 — issues and decisions
s = d.slide("Issues and decisions")
cards = [
    ("VQ below continuous", "On every image task and on mortality. Discrete tokens cost some accuracy; "
     "reported honestly as the price of a shared vocabulary.", RED),
    ("Event tokens are long", "About the same AUROC as continuous on mortality, but much longer sequences, so "
     "attention cost is higher. Both are reported.", AMBER),
    ("Surgical phase is weak", "Low macro-F1 on Cataract-101 with 64 px frames. Kept as an extra task, "
     "not a main result.", AMBER),
    ("Fair budgets", f"Same model, fixed splits, same epochs and early stopping, {SEEDS} seeds for every run.",
     GREY),
]
for i, (tt, dsc, col) in enumerate(cards):
    x = Inches(0.5 + (i % 2) * 6.3)
    y = Inches(1.35 + (i // 2) * 2.85)
    box(s, x, y, Inches(6.0), Inches(2.6), fill=LIGHT)
    box(s, x, y, Inches(0.08), Inches(2.6), fill=col)
    text(s, x + Inches(0.25), y + Inches(0.15), Inches(5.6), Inches(0.5), tt, size=19, bold=True, color=col)
    text(s, x + Inches(0.25), y + Inches(0.75), Inches(5.6), Inches(1.75), dsc, size=17)

# 10 — next week + questions
s = d.slide("Next week and questions")
card(s, Inches(0.5), Inches(1.4), Inches(6.0), Inches(4.6), "Plan for Week 6: delivery", [
    "Summarise compression, reconstruction, utilisation and task metrics",
    "Check that one script reproduces every table and figure",
    "Finish code, protocol specification, report, demo and notebook",
    "Start the thesis draft from these results",
])
card(s, Inches(6.8), Inches(1.4), Inches(6.0), Inches(4.6), "Questions for supervisor", [
    "Is the accuracy gap of VQ vs continuous acceptable as a finding, or should I try to close it?",
    "Is the ablation set (codebook size, time granularity) enough for the thesis?",
    "Should event tokens be the recommended representation for ICU data?",
], accent=AMBER)

d.save()
