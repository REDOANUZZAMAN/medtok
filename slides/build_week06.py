"""Week 6 supervisor update deck (reproducibility and delivery). Run: .venv/bin/python slides/build_week06.py

Every result number is read from reports/tables/ (see deck_common.py); nothing is typed by hand.
"""
from pathlib import Path

from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.util import Inches

from deck_common import (AMBER, CATARACT_CITE, DARK, GREY, LIGHT, NAVY, RED, TEAL, WHITE, Deck, banner, box,
                         card, f, figure, load, n, roadmap, row, table, text)

OUT = Path(__file__).with_name("week06_progress.pptx")

img, ts, vid, ds, iface = load(0), load(1), load(2), load(3), load(13)
SEEDS = ds[0]["seeds"]


def metric(exp, k):
    return row(ds, experiment=exp)[f"metric {k}"].split(" ", 1)[1]


d = Deck(6, "Nov 12", OUT)

# 1 — title
d.title_slide("Week 6: results summary, reproducibility and delivery", "November 12, 2026")

# 2 — recap
s = d.slide("Recap: Week 5 (tasks and ablations)", source="reports/tables/table_03.csv")
roadmap(s, 6)
card(s, Inches(6.8), Inches(1.3), Inches(6.0), Inches(3.6), "Downstream results (test set)", [
    f"BloodMNIST AUROC: continuous {metric('ds_bloodmnist_continuous', 1)}, VQ {metric('ds_bloodmnist_vq', 1)}",
    f"Mortality AUROC: continuous {metric('ds_mortality_continuous', 1)}, event {metric('ds_mortality_event', 1)}, "
    f"VQ {metric('ds_mortality_vq', 1)}",
    "Ablations: bigger codebooks reconstruct better; time granularity trades error for length",
], size=14)
text(s, Inches(6.8), Inches(5.1), Inches(6.0), Inches(1.4),
     "This week: put everything together, make it reproducible, and deliver code, specification, report "
     "and demo.", size=15, color=GREY)

# 3 — compression and sequence length
s = d.slide("Summary: compression and length", source="reports/figures/compression_sequence_length.png "
            "(ratio (fixed)); reports/tables/table_00-02")
figure(s, "compression_sequence_length", Inches(0.3), Inches(1.15), Inches(8.3), Inches(4.85))
card(s, Inches(8.8), Inches(1.3), Inches(4.0), Inches(4.7), "Take-aways", [
    "VQ gives the biggest storage saving for images and time series",
    "Event tokens are readable and exact on bins, but long",
    "Video segments are the shortest sequences",
    "Continuous saves little for images and ICU data: it is the baseline",
], size=14)
banner(s, Inches(6.2), "Continuous positions carry fp16 vectors, NOT vocabulary tokens. Tables also report "
       "ratio (+overhead), which counts span delimiters and decoding metadata.", h=Inches(0.7))

# 4 — reconstruction and utilisation
s = d.slide("Summary: reconstruction, code use", source="reports/figures/reconstruction_error.png; "
            "reports/tables/table_00.csv, table_01.csv, table_02.csv")
figure(s, "reconstruction_error", Inches(0.3), Inches(1.15), Inches(12.7), Inches(3.4))
b = row(img, experiment="image_vq", subset="bloodmnist")
t = row(ts, experiment="timeseries_vq")
v = row(vid, experiment="video_vq")
rows = [["VQ codebook", "active codes", "util"],
        ["Image (BloodMNIST)", b["active codes"], f(b["util"])],
        ["Time series", t["active codes / bins"], f(t["util"])],
        ["Video (Cataract-101)", v["active codes / states"], f(v["util"])]]
table(s, Inches(0.5), Inches(4.75), Inches(6.0), rows, col_w=[2.8, 1.7, 1.5], size=14, row_h=Inches(0.48))
card(s, Inches(6.8), Inches(4.75), Inches(6.0), Inches(2.2), "Take-aways", [
    "Continuous reconstructs best for images and video",
    "Event tokens reconstruct ICU data best",
    "With the three remedies almost every code is used: no collapse",
], size=14)

# 5 — headline table (the two fully implemented modalities)
s = d.slide("Summary: the two main modalities", source="reports/tables/table_00.csv, table_01.csv, table_03.csv")
rows = [["Modality", "Method", "positions", "reconstruction", "ratio (+overhead)", "task metric 1",
         "task metric 2"]]
for rep in ("continuous", "vq"):
    x = row(img, experiment=f"image_{rep}", subset="bloodmnist")
    rows.append(["Image (BloodMNIST)" if rep == "continuous" else "", "continuous" if rep == "continuous" else "VQ",
                 n(x["positions"]), f"PSNR {f(x['PSNR dB'])} dB", f(x["ratio (+overhead)"]),
                 "AUROC " + metric(f"ds_bloodmnist_{rep}", 1), "macro-F1 " + metric(f"ds_bloodmnist_{rep}", 2)])
TS = (("continuous", "continuous"), ("vq", "VQ"), ("event", "event"), ("event_change", "event (changes)"))
for i, (rep, lab) in enumerate(TS):
    x = row(ts, experiment=f"timeseries_{rep}")
    rows.append(["ICU (PhysioNet 2012)" if i == 0 else "", lab, f(x["positions mean"]), f"MAE {f(x['MAE (z)'])} (z)",
                 f(x["ratio (+overhead)"]), "AUROC " + metric(f"ds_mortality_{rep}", 1),
                 "AUPRC " + metric(f"ds_mortality_{rep}", 2)])
table(s, Inches(0.5), Inches(1.3), Inches(12.3), rows, col_w=[1.9, 1.4, 1.15, 1.75, 1.4, 2.35, 2.35], size=12,
      row_h=Inches(0.55))
banner(s, Inches(5.35), "Continuous 'positions' are fp16 vectors, not vocabulary tokens. Images: positions per "
       "image; ICU: mean positions per stay.", h=Inches(0.55), size=13)
text(s, Inches(0.5), Inches(6.05), Inches(12.3), Inches(0.9), [
    f"Task metrics: test set, mean ± std over {SEEDS} seeds, same model and budget per task. "
    "Surgical video and text are reported separately (adapted modalities)."], size=13, color=GREY)

# 6 — reproducibility
s = d.slide("Reproducibility")
steps = [
    ("1  prepare", "download open data, clean, fixed splits"),
    ("2  tokenizers", "train + evaluate every tokenizer"),
    ("3  interfaces", "text and point-cloud checks"),
    ("4  downstream", f"all tasks, {SEEDS} seeds"),
    ("5  ablations", "codebook, window, grid, remedies"),
    ("6  report", "RESULTS.md, tables, figures"),
]
text(s, Inches(0.5), Inches(1.2), Inches(12.3), Inches(0.5), "One command: scripts/run_all.sh", size=20, bold=True,
     color=NAVY)
for i, (tt, dsc) in enumerate(steps):
    x = Inches(0.5 + i * 2.07)
    box(s, x, Inches(1.8), Inches(1.9), Inches(1.6), fill=TEAL if i == 5 else LIGHT,
        shape=MSO_SHAPE.ROUNDED_RECTANGLE)
    text(s, x + Inches(0.1), Inches(1.9), Inches(1.75), Inches(0.5), tt, size=13, bold=True,
         color=WHITE if i == 5 else NAVY)
    text(s, x + Inches(0.1), Inches(2.4), Inches(1.7), Inches(1.0), dsc, size=12,
         color=WHITE if i == 5 else DARK)
card(s, Inches(0.5), Inches(3.7), Inches(6.0), Inches(3.2), "What makes it repeatable", [
    "Fixed seeds and fixed splits; learned preprocessing fitted on training data only",
    "Each run stores its config, library versions, git commit, logs and metrics",
    "--skip-existing: safe to re-run after a crash",
    "Unit + integration tests on synthetic data, no download needed",
], size=14)
card(s, Inches(6.8), Inches(3.7), Inches(6.0), Inches(3.2), "Numbers are never typed", [
    "python -m medical_tokenizer.report writes every table and figure",
    "These slides read the same CSV files",
    "GPU re-run on Modal repeated all downstream runs on the same hardware",
    "Small differences can remain from multi-threaded CPU kernels",
], accent=AMBER, size=14)

# 7 — deliverables
s = d.slide("Deliverables")
items = [
    ("Code", "Python package medical_tokenizer: protocol, codecs, tokenizers, downstream model, CLI"),
    ("Specification", "docs/token_protocol.md: id ranges, special tokens, wire format, positions, bit accounting"),
    ("Report", "reports/RESULTS.md + tables + figures, generated from the experiment outputs"),
    ("Demo", "browser demo: data → tokens → data for each modality, plus a mixed sequence as JSONL"),
    ("Notebook", "notebooks/explore_tokens.ipynb: explore real token sequences"),
    ("Documentation", "architecture, datasets (licences, splits), experiment protocol and limits"),
]
for i, (tt, dsc) in enumerate(items):
    y = Inches(1.3 + i * 0.93)
    box(s, Inches(0.6), y + Inches(0.12), Inches(0.5), Inches(0.5), fill=TEAL, shape=MSO_SHAPE.OVAL)
    text(s, Inches(0.6), y + Inches(0.12), Inches(0.5), Inches(0.5), "✓", size=18, bold=True, color=WHITE,
         align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
    text(s, Inches(1.35), y, Inches(2.6), Inches(0.75), tt, size=19, bold=True, color=NAVY,
         anchor=MSO_ANCHOR.MIDDLE)
    text(s, Inches(4.0), y, Inches(8.8), Inches(0.75), dsc, size=16, anchor=MSO_ANCHOR.MIDDLE)

# 8 — limitations (all conditions from WEEKLY_PLAN.md)
s = d.slide("Limitations to state in the thesis")
lims = [
    ("Continuous is a baseline", "Its fp16 vectors are never counted as vocabulary tokens; ratios include "
     "fp16 precision, index bits, delimiters and metadata."),
    ("Cataract-101 ≠ Cholec80", "Different surgery and labels: results are not comparable. Ground-truth phase "
     "events = annotation encoding only."),
    ("Synthetic text", "Templated notes: accepted on round-trip, field preservation and protocol "
     "compatibility only; they never replace the downstream tasks."),
    ("Small inputs, small model", "28×28 images, 64×64 frames, a 2-layer Transformer: chosen to compare many "
     "settings fairly, not to beat the state of the art."),
]
for i, (tt, dsc) in enumerate(lims):
    x = Inches(0.5 + (i % 2) * 6.3)
    y = Inches(1.3 + (i // 2) * 2.45)
    box(s, x, y, Inches(6.0), Inches(2.25), fill=LIGHT)
    box(s, x, y, Inches(0.08), Inches(2.25), fill=RED if i < 3 else GREY)
    text(s, x + Inches(0.25), y + Inches(0.12), Inches(5.6), Inches(0.5), tt, size=18, bold=True,
         color=RED if i < 3 else GREY)
    text(s, x + Inches(0.25), y + Inches(0.65), Inches(5.6), Inches(1.55), dsc, size=15)
banner(s, Inches(6.3), CATARACT_CITE + " Point clouds: synthetic surfaces only.", accent=NAVY)

# 9 — issues and decisions
s = d.slide("Issues and decisions")
cards = [
    ("No single winner", "VQ: most compact (images, ICU). Continuous: best task scores. Events: best ICU reconstruction. "
     "Decision: the thesis presents trade-offs, not one winner.", AMBER),
    ("Scope kept", "Images + time series fully done with two tasks; video, text and point clouds adapted, as "
     "agreed at the Week-2 checkpoint.", TEAL),
    ("Extensions", "Better video encoders, point-cloud training, real paired clinical text (needs authorised "
     "data), Cholec80 if access is granted.", GREY),
    ("Licences", "Only open data in the repo; Cataract-101 non-commercial; no patient-level data committed.",
     NAVY),
]
for i, (tt, dsc, col) in enumerate(cards):
    x = Inches(0.5 + (i % 2) * 6.3)
    y = Inches(1.35 + (i // 2) * 2.85)
    box(s, x, y, Inches(6.0), Inches(2.6), fill=LIGHT)
    box(s, x, y, Inches(0.08), Inches(2.6), fill=col)
    text(s, x + Inches(0.25), y + Inches(0.15), Inches(5.6), Inches(0.5), tt, size=19, bold=True, color=col)
    text(s, x + Inches(0.25), y + Inches(0.75), Inches(5.6), Inches(1.75), dsc, size=17)

# 10 — next steps + questions
s = d.slide("Next steps and questions")
card(s, Inches(0.5), Inches(1.4), Inches(6.0), Inches(4.6), "After Week 6: thesis writing", [
    "Write the thesis chapters from RESULTS.md and the specification",
    "Use these weekly decks as the outline for the defence",
    "Freeze the code at a tagged commit for the examiners",
    "Fix anything the supervisor asks for in this review",
])
card(s, Inches(6.8), Inches(1.4), Inches(6.0), Inches(4.6), "Questions for supervisor", [
    "Is the 'trade-offs, no single winner' conclusion the right framing?",
    "Is the set of deliverables complete for submission?",
    "Which results should be in the main text, and which in the appendix?",
    "When can I send you the first full thesis draft?",
], accent=AMBER)

d.save()
