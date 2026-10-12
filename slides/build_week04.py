"""Week 4 supervisor update deck (adaptation and integration). Run: .venv/bin/python slides/build_week04.py

Every result number is read from reports/tables/ (see deck_common.py); nothing is typed by hand.
"""
from pathlib import Path

from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.util import Inches

from deck_common import (AMBER, CATARACT_CITE, GREY, LIGHT, NAVY, RED, TEAL, WHITE, Deck, banner, box, card,
                         f, figure, load, roadmap, row, table, text)

OUT = Path(__file__).with_name("week04_progress.pptx")

ts = load(1)              # time series (recap)
vid = load(2)             # surgical video
iface = load(13)          # text + point-cloud interfaces

GT = [r for r in vid if r["experiment"].startswith("annotated phase events")][0]
txt = row(iface, modality="text")
pcs = [r for r in iface if r["modality"] == "pointcloud"]

d = Deck(4, "Oct 29", OUT)

# 1 — title
d.title_slide("Week 4: adaptation and integration (text, surgical video, point clouds)", "October 29, 2026")

# 2 — recap
s = d.slide("Recap: Week 3 (time series)", source="reports/tables/table_01.csv")
roadmap(s, 4)
ev, vq = row(ts, experiment="timeseries_event"), row(ts, experiment="timeseries_vq")
evc = row(ts, experiment="timeseries_event_change")
card(s, Inches(6.8), Inches(1.3), Inches(6.0), Inches(3.3), "Time-series results (test set)", [
    f"Event tokens: lowest error, MAE {f(ev['MAE (z)'])} (z), but {f(ev['positions mean'])} positions",
    f"'Changes only' events: {f(evc['positions mean'])} positions, same error",
    f"VQ: {f(vq['positions mean'])} positions, ratio (+overhead) {f(vq['ratio (+overhead)'])}×, "
    f"MAE {f(vq['MAE (z)'])}",
    "Missing values: zero fill + mask kept (LOCF was worse)",
], size=14)
text(s, Inches(6.8), Inches(4.85), Inches(6.0), Inches(1.6),
     "Images and time series are now the two fully implemented modalities. This week adds the "
     "three 'adapted' modalities to the same protocol.", size=15, color=GREY)

# 3 — overview of what each modality is promised
s = d.slide("Scope of this week (as agreed)")
rows = [["Modality", "Data", "What is promised", "What is NOT claimed"],
        ["Clinical text", "Synthetic templated notes (v2)", "Lossless round-trip, field preservation, "
         "protocol compatibility", "Clinical language understanding"],
        ["Surgical video", "Cataract-101 (10 phases)", "VQ / continuous / segment tokens; phase events",
         "Comparison with Cholec80 results"],
        ["3D point cloud", "Synthetic surfaces", "Interface + tests: normalise → voxel groups → codes",
         "A trained point-cloud model"]]
table(s, Inches(0.5), Inches(1.35), Inches(12.3), rows, col_w=[2.0, 3.0, 4.0, 3.3], size=15,
      row_h=Inches(0.8))
banner(s, Inches(4.85), CATARACT_CITE, accent=NAVY)
text(s, Inches(0.5), Inches(5.6), Inches(12.3), Inches(1.3), [
    "All three use the same vocabulary ranges, the same wire format and the same validator as images and time series",
    "Advanced video encoders and point-cloud training stay 'extensions', as agreed at the Week-2 checkpoint",
], size=15, bullet=True)

# 4 — synthetic text
s = d.slide("Text: synthetic templated notes (v2)", source="reports/tables/table_13.csv")
rows = [["Setting", "exact round-trip", "field preservation", "tokens mean", "ratio", "ratio (+overhead)"],
        [txt["setting"], f(txt["exact round-trip / Chamfer"]), f(txt["field preservation"]), f(txt["tokens mean"]),
         f(txt["ratio"]), f(txt["ratio (+overhead)"])]]
table(s, Inches(0.5), Inches(1.3), Inches(12.3), rows, col_w=[3.3, 1.8, 1.9, 1.6, 1.4, 2.3], size=15,
      row_h=Inches(0.55))
banner(s, Inches(2.6), "SYNTHETIC text: accepted on round-trip, field preservation and protocol compatibility "
       "only. It never replaces the downstream tasks.", accent=RED)
card(s, Inches(0.5), Inches(3.4), Inches(6.0), Inches(3.5), "How the notes are made", [
    "Rendered from structured fields: several wordings, units, missing values, negation, times",
    "Each note carries the tag synthetic-template:<kind>@v2",
    "No label leakage: ICU notes read only the first hour and never the outcome",
    "Byte-level BPE trained on the training split only",
], size=14)
card(s, Inches(6.8), Inches(3.4), Inches(6.0), Inches(3.5), "What the numbers mean", [
    "exact round-trip = share of notes that decode to exactly the same text",
    "field preservation = share of values, units, times and negations that survive",
    "ratio (+overhead) also counts the span delimiters and metadata",
], accent=GREY, size=14)

# 5 — Cataract-101
s = d.slide("Surgical video: Cataract-101", source="reports/tables/table_02.csv")
VID = {"video_continuous": "A. continuous (fp16 vectors)", "video_vq": "B. VQ",
       "video_segments": "C. segment events (k-means states)"}
rows = [["Representation", "positions/min", "PSNR dB", "SSIM", "ratio (fixed)", "ratio (+overhead)",
         "util"]]
for exp, lab in VID.items():
    x = row(vid, experiment=exp)
    rows.append([lab, f(x["positions/min"]), f(x["PSNR dB"]), f(x["SSIM"]), f(x["ratio (fixed)"]),
                 f(x["ratio (+overhead)"]), f(x["util"])])
rows.append(["Annotated phase events (ground truth)", f(GT["positions/min"]), "–", "–", f(GT["ratio (fixed)"]),
             f(GT["ratio (+overhead)"]), "–"])
table(s, Inches(0.5), Inches(1.3), Inches(12.3), rows, col_w=[4.0, 1.5, 1.2, 1.0, 1.6, 1.9, 1.1], size=14,
      row_h=Inches(0.5))
banner(s, Inches(3.95), "Not comparable with Cholec80 results (other surgery, other 10-phase label set).",
       accent=RED, h=Inches(0.5))
banner(s, Inches(4.55), "Ground-truth phase events = annotation encoding only (label → token → label), "
       "NOT phase recognition from video.", accent=RED, h=Inches(0.5))
text(s, Inches(0.5), Inches(5.2), Inches(12.3), Inches(1.4), [
    "Continuous positions carry fp16 vectors and are not vocabulary tokens",
    "Segment events are very short (a new token only when the scene changes), but reconstruct the frames worst",
    "Phase recognition from video tokens is a downstream task in Week 5",
], size=15, bullet=True)
text(s, Inches(0.5), Inches(6.65), Inches(12.3), Inches(0.4), CATARACT_CITE, size=12, color=GREY)

# 6 — point clouds
s = d.slide("Point clouds: interface and tests", source="reports/tables/table_13.csv")
rows = [["Voxel resolution", "Chamfer (rel. to object size)", "tokens mean", "ratio", "ratio (+overhead)"]]
for x in pcs:
    rows.append([x["setting"], f(x["exact round-trip / Chamfer"]), f(x["tokens mean"]), f(x["ratio"]),
                 f(x["ratio (+overhead)"])])
table(s, Inches(0.5), Inches(1.3), Inches(12.3), rows, col_w=[2.3, 3.4, 2.0, 2.0, 2.6], size=15,
      row_h=Inches(0.55))
banner(s, Inches(3.75), "Synthetic surfaces and non-learned features: this shows the interface works, "
       "not a trained point-cloud tokenizer.")
card(s, Inches(0.5), Inches(4.5), Inches(6.0), Inches(2.4), "Pipeline", [
    "Normalise → group points into voxels → features → codebook ids",
    "depth_to_points ready for SCARED depth maps if access is granted",
], size=14)
card(s, Inches(6.8), Inches(4.5), Inches(6.0), Inches(2.4), "Trade-off", [
    "Finer voxels: lower Chamfer error, but more tokens and less compression",
    "Same pattern as image patch size: detail costs sequence length",
], accent=GREY, size=14)

# 7 — integration
s = d.slide("Integration: all modalities in one format", source="tests/test_pipeline_integration.py, "
            "reports/figures/token_sequence.png")
figure(s, "token_sequence", Inches(0.4), Inches(1.15), Inches(6.6), Inches(2.9))
text(s, Inches(0.5), Inches(4.1), Inches(6.4), Inches(0.9),
     "One real test sequence: a synthetic note (sub-word tokens) followed by ICU event tokens.", size=14,
     color=GREY)
tests = [
    ("Image pipeline", "train → evaluate → encode shards → decode → downstream"),
    ("Time-series pipeline", "all three representations; event codec exact on bins"),
    ("Video pipeline", "VQ / continuous / segments; phase events round-trip exactly"),
    ("One model, three modalities", "multi-task Transformer reads image, ICU and video tokens"),
]
for i, (tt, dsc) in enumerate(tests):
    y = Inches(1.3 + i * 1.0)
    box(s, Inches(7.3), y + Inches(0.12), Inches(0.5), Inches(0.5), fill=TEAL, shape=MSO_SHAPE.OVAL)
    text(s, Inches(7.3), y + Inches(0.12), Inches(0.5), Inches(0.5), "✓", size=18, bold=True, color=WHITE,
         align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
    text(s, Inches(8.0), y, Inches(4.8), Inches(0.4), tt, size=17, bold=True, color=NAVY)
    text(s, Inches(8.0), y + Inches(0.4), Inches(4.8), Inches(0.5), dsc, size=13)
card(s, Inches(0.5), Inches(5.2), Inches(12.3), Inches(1.7), "How the tests work", [
    "They run on small SYNTHETIC files laid out exactly like the real datasets, so they need no download",
    "Every sequence passes the validator; malformed input is rejected; JSONL / NPZ round-trips are lossless",
], accent=GREY, size=14)

# 8 — issues and decisions
s = d.slide("Issues and decisions")
cards = [
    ("No open clinical text", "MIMIC notes need credentialed access. Decision: synthetic, versioned "
     "templates; claims limited to round-trip and field preservation.", RED),
    ("Cholec80 not accessible", "Archive needs a CAMMA request. Decision: Cataract-101, clearly marked "
     "as not comparable; cited; non-commercial licence respected.", AMBER),
    ("Labels vs recognition", "Phase events from ground truth only show the protocol can store "
     "annotations. Recognition is measured from video tokens only.", AMBER),
    ("Point clouds stay small", "Synthetic surfaces only; no training. Kept as an extension, as agreed "
     "at the checkpoint.", GREY),
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
card(s, Inches(0.5), Inches(1.4), Inches(6.0), Inches(4.6), "Plan for Week 5: tasks and ablations", [
    "Image classification (AUROC, macro-F1) on the three MedMNIST subsets",
    "ICU mortality (AUROC, AUPRC) for every time-series representation",
    "One Transformer, fixed splits, same budget, " + load(3)[0]["seeds"] + " seeds",
    "Ablations: codebook size and time granularity",
])
card(s, Inches(6.8), Inches(1.4), Inches(6.0), Inches(4.6), "Questions for supervisor", [
    "Are the text acceptance criteria (round-trip, field preservation, compatibility) enough?",
    "Is it OK to present Cataract-101 phase recognition as an extra task, not a main result?",
    "Should the point-cloud interface stay in the thesis body or move to an appendix?",
], accent=AMBER)

d.save()
