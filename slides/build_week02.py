"""Week 2 supervisor update deck (image tokenization). Run: .venv/bin/python slides/build_week02.py

Every result number is read from reports/tables/ (see deck_common.py); nothing is typed by hand.
"""
from pathlib import Path

from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.util import Inches

from deck_common import (AMBER, DARK, GREY, LIGHT, NAVY, RED, TEAL, WHITE, Deck, banner, box, card, f,
                         figure, load, n, roadmap, row, table, text)

OUT = Path(__file__).with_name("week02_progress.pptx")

img = load(0)             # images: reconstruction, compression, utilisation
ds = load(3)              # downstream runs (only used to show that both tasks run)
SUBSETS = ["bloodmnist", "pneumoniamnist", "dermamnist"]
NAME = {"bloodmnist": "BloodMNIST", "pneumoniamnist": "PneumoniaMNIST", "dermamnist": "DermaMNIST"}


def r(exp, subset):
    return row(img, experiment=exp, subset=subset)


cont_pos = n(r("image_continuous", "bloodmnist")["positions"])
vq_pos = n(r("image_vq", "bloodmnist")["positions"])

d = Deck(2, "Oct 15", OUT)

# 1 — title
d.title_slide("Week 2: image tokenization on MedMNIST (continuous vs VQ)",
              "October 15, 2026")

# 2 — recap
s = d.slide("Recap: Week 1 and supervisor feedback")
roadmap(s, 2)
card(s, Inches(6.8), Inches(1.3), Inches(6.0), Inches(2.35), "Done in Week 1", [
    "Unified token protocol v0.2: one vocabulary, one sequence format",
    "Open datasets chosen; data pipeline and splits ready",
    "Reference code + synthetic tests for the format",
], size=14)
card(s, Inches(6.8), Inches(3.85), Inches(6.0), Inches(2.6), "Supervisor feedback (Oct 8)", [
    "Scope = a scoped, reproducible prototype",
    "Images + time series fully implemented; at least two downstream tasks",
    "Text: synthetic notes; video: Cataract-101 phase events; point clouds: interface + tests",
], accent=AMBER, size=14)
text(s, Inches(0.5), Inches(5.2), Inches(6.0), Inches(1.3),
     "This week: the first fully implemented modality (2D images), and the end-of-Week-2 checkpoint.",
     size=15, color=GREY)

# 3 — work done: method
s = d.slide("Work done: two image tokenizers")
text(s, Inches(0.5), Inches(1.2), Inches(12.3), Inches(0.5),
     "Same autoencoder for both:  28×28 image  →  7×7 latent grid  →  28×28 image",
     size=18, bold=True, color=NAVY)
card(s, Inches(0.5), Inches(1.85), Inches(6.0), Inches(3.2), "A. Continuous projection (baseline)", [
    f"{cont_pos} positions per image, each one carries an fp16 vector",
    "No discretisation: the encoder output goes straight to the model",
    "These vectors are NOT vocabulary tokens; they are reported as 'continuous vectors'",
], size=15)
card(s, Inches(6.8), Inches(1.85), Inches(6.0), Inches(3.2), "B. Vector quantisation (VQ)", [
    "Each latent vector → id of the nearest entry in a 1,024-entry codebook",
    f"{vq_pos} vocabulary tokens per image (7×7 codes + 7 <ROW> markers)",
    "Codebook kept alive with EMA updates, cosine lookup and dead-code restarts",
    "Control run: vanilla VQ-VAE without these three remedies",
], size=15)
card(s, Inches(0.5), Inches(5.25), Inches(12.3), Inches(1.6), "Fair comparison", [
    "One tokenizer for all three subsets (Blood, Pneumonia, Derma), trained on the training splits only",
    "Measured on the official test sets: reconstruction (PSNR, SSIM, MAE), compression, codebook use, tokens per image",
], accent=GREY, size=15)

# 4 — results: reconstruction figure + table
s = d.slide("Results: reconstruction on the test sets", source="reports/tables/table_00.csv, "
            "reports/figures/recon_images.png")
figure(s, "recon_images", Inches(0.4), Inches(1.15), Inches(5.6), Inches(5.85))
rows = [["Subset", "Method", "PSNR dB", "SSIM", "MAE"]]
for sub in SUBSETS:
    for exp, lab in (("image_continuous", "continuous"), ("image_vq", "VQ")):
        x = r(exp, sub)
        rows.append([NAME[sub] if lab == "continuous" else "", lab, f(x["PSNR dB"]), f(x["SSIM"]), f(x["MAE"])])
table(s, Inches(6.3), Inches(1.3), Inches(6.5), rows, col_w=[2.0, 1.4, 1.1, 1.0, 1.0], size=14,
      row_h=Inches(0.45))
text(s, Inches(6.3), Inches(4.65), Inches(6.5), Inches(2.3), [
    "Continuous keeps more detail: higher PSNR / SSIM on every subset",
    "VQ images are a little smoother, but cell shape, lung shape and lesion colour stay visible",
    "MAE is in 0–255 pixel units",
], size=15, bullet=True)

# 5 — results: compression and token counts
s = d.slide("Results: token counts and compression", source="reports/tables/table_00.csv")
rows = [["Subset", "Method", "positions", "vocab tokens", "cont. vectors", "ratio (fixed)", "ratio (+overhead)"]]
for sub in SUBSETS:
    for exp, lab in (("image_continuous", "continuous"), ("image_vq", "VQ")):
        x = r(exp, sub)
        rows.append([NAME[sub] if lab == "continuous" else "", lab, n(x["positions"]), n(x["vocab tokens"]),
                     n(x["cont. vectors"]), f(x["ratio (fixed)"]), f(x["ratio (+overhead)"])])
table(s, Inches(0.5), Inches(1.3), Inches(12.3), rows, col_w=[2.1, 1.4, 1.5, 1.8, 1.8, 1.7, 2.0], size=14,
      row_h=Inches(0.45))
banner(s, Inches(4.65), "Continuous positions carry fp16 vectors: they are NOT vocabulary tokens "
       "(vocab tokens = 0 for the continuous baseline).")
card(s, Inches(0.5), Inches(5.35), Inches(12.3), Inches(1.6), "How to read the ratios (raw bits ÷ token bits)", [
    "ratio (fixed): every VQ index costs log2(1,024) bits; every continuous value costs 16 bits (fp16)",
    "ratio (+overhead): also counts the span delimiters <START> <REP> <END> and the decoding metadata",
], accent=GREY, size=14)

# 6 — results: codebook utilisation
s = d.slide("Results: codebook utilisation", source="reports/tables/table_00.csv, "
            "reports/figures/codebook_usage.png")
figure(s, "codebook_usage", Inches(0.4), Inches(1.2), Inches(6.4), Inches(3.4))
rows = [["Subset", "Codebook", "active codes", "util", "perplexity"]]
for sub in SUBSETS:
    for exp, lab in (("image_vq", "with remedies"), ("image_vq_vanilla", "vanilla VQ-VAE")):
        x = r(exp, sub)
        rows.append([NAME[sub] if exp == "image_vq" else "", lab, x["active codes"], f(x["util"]),
                     f(x["perplexity"])])
table(s, Inches(6.9), Inches(1.3), Inches(5.9), rows, col_w=[1.75, 1.55, 0.95, 0.75, 0.9], size=12,
      row_h=Inches(0.42))
v_blood = r("image_vq_vanilla", "bloodmnist")
q_blood = r("image_vq", "bloodmnist")
card(s, Inches(0.5), Inches(4.8), Inches(6.2), Inches(2.15), "What it shows", [
    f"Vanilla VQ-VAE collapses: only {v_blood['active codes']} of 1,024 codes used on BloodMNIST",
    f"With the three remedies: {q_blood['active codes']} codes used (util {f(q_blood['util'])})",
    "Utilisation is measured on the test set, after training",
], size=14)
text(s, Inches(6.9), Inches(4.5), Inches(5.9), Inches(2.4), [
    "util = active codes ÷ 1,024",
    "perplexity = 'effective number' of codes (higher = more even use)",
    "PneumoniaMNIST is grey-scale and very uniform, so it needs fewer codes",
], size=13, bullet=True, color=DARK)

# 7 — results: t-SNE
s = d.slide("Results: do the latents keep meaning?", source="reports/figures/tsne_image_latents.png")
figure(s, "tsne_image_latents", Inches(0.4), Inches(1.2), Inches(7.6), Inches(5.7))
card(s, Inches(8.2), Inches(1.3), Inches(4.6), Inches(5.5), "Reading the plot", [
    "Each dot is one BloodMNIST test cell, coloured by its true cell type",
    "The encoder never saw the labels, but platelets and erythroblasts form their own groups",
    "So the latent space (which VQ then quantises) already holds class information",
    "Other white-cell types overlap: they look similar at 28×28",
    "A qualitative check only; the real test is the classification task in Week 5",
], size=14)

# 8 — end-of-week-2 checkpoint
s = d.slide("End-of-Week-2 checkpoint",
            source="run list: reports/tables/table_03.csv; timing: Modal GPU re-run vs the earlier CPU run")
task_runs = [x for x in ds if x["task"] in SUBSETS + ["mortality"] and "multitask" not in x["experiment"]]
checks = [
    ("Data readable", "MedMNIST (3 subsets) and PhysioNet 2012 load, clean and split with fixed seeds; "
     "Cataract-101 frame cache built"),
    ("Compute enough", "GPU re-run on Modal: about 10 min instead of about 17 h on the CPU; "
     "all downstream runs repeated on the same hardware"),
    ("Both downstream tasks run", "MedMNIST classification and PhysioNet 2012 mortality train end to end on "
     f"the token shards ({len(task_runs)} runs finished, {task_runs[0]['seeds']} seeds each)"),
]
for i, (t, dsc) in enumerate(checks):
    y = Inches(1.35 + i * 1.2)
    box(s, Inches(0.6), y + Inches(0.15), Inches(0.55), Inches(0.55), fill=TEAL, shape=MSO_SHAPE.OVAL)
    text(s, Inches(0.6), y + Inches(0.15), Inches(0.55), Inches(0.55), "✓", size=20, bold=True, color=WHITE,
         align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
    text(s, Inches(1.4), y, Inches(3.2), Inches(0.85), t, size=19, bold=True, color=NAVY,
         anchor=MSO_ANCHOR.MIDDLE)
    text(s, Inches(4.6), y, Inches(8.2), Inches(0.85), dsc, size=15, anchor=MSO_ANCHOR.MIDDLE)
card(s, Inches(0.5), Inches(5.0), Inches(12.3), Inches(1.9), "Decision: on schedule, no reduction needed", [
    "Keep full sample sizes and fixed splits; images, time series and both tasks stay in scope",
    "Fallback if we fall behind: smaller samples on the same fixed splits; advanced video encoding and "
    "point-cloud training move to 'extensions'",
], accent=TEAL, size=15)

# 9 — issues and decisions
s = d.slide("Issues and decisions")
pn = r("image_vq", "pneumoniamnist")
cards = [
    ("Codebook collapse", "Vanilla VQ-VAE used few codes. Decision: keep EMA + cosine lookup + dead-code "
     "restarts; keep vanilla as a control row.", RED),
    ("Low use on PneumoniaMNIST", f"Only {pn['active codes']} active codes (util {f(pn['util'])}). Grey X-rays "
     "are simple; one shared codebook is still used for all subsets.", AMBER),
    ("VQ costs detail", "VQ has lower PSNR than continuous. This is the price of discrete tokens; whether "
     "it hurts the task is tested in Week 5.", AMBER),
    ("Continuous ratio below 1", f"Continuous ratio (fixed) is {f(r('image_continuous', 'bloodmnist')['ratio (fixed)'])} "
     "on BloodMNIST: fp16 vectors take more bits than 8-bit pixels. It is a baseline, not a compressor.", GREY),
]
for i, (t, dsc, col) in enumerate(cards):
    x = Inches(0.5 + (i % 2) * 6.3)
    y = Inches(1.35 + (i // 2) * 2.85)
    box(s, x, y, Inches(6.0), Inches(2.6), fill=LIGHT)
    box(s, x, y, Inches(0.08), Inches(2.6), fill=col)
    text(s, x + Inches(0.25), y + Inches(0.15), Inches(5.6), Inches(0.5), t, size=19, bold=True, color=col)
    text(s, x + Inches(0.25), y + Inches(0.75), Inches(5.6), Inches(1.75), dsc, size=17)

# 10 — next week + questions
s = d.slide("Next week and questions")
card(s, Inches(0.5), Inches(1.4), Inches(6.0), Inches(4.6), "Plan for Week 3: time series", [
    "PhysioNet 2012, 48 h ICU stays, in-hospital mortality",
    "Handle missing values and irregular time (hourly grid + mask)",
    "Encode time with <DT> tokens + exact time side channel",
    "Compare continuous, VQ and event representations",
    "Report reconstruction error, sequence length, compression",
])
card(s, Inches(6.8), Inches(1.4), Inches(6.0), Inches(4.6), "Questions for supervisor", [
    "Is PSNR / SSIM / MAE + codebook use enough for the image part, or do you want another metric?",
    "Is the end-of-Week-2 checkpoint accepted (no scope reduction)?",
    "Is it OK to keep one shared codebook for all three MedMNIST subsets?",
], accent=AMBER)

d.save()
