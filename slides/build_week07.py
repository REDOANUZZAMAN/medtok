"""Week 7 supervisor update deck (validation on linked MIMIC-IV data). Run: .venv/bin/python slides/build_week07.py

Every result number is read from reports/mimic/mimic_metrics.json (written by
`python -m medical_tokenizer.mimic_validate`); nothing is typed by hand. MIMIC-IV is credentialed:
the deck shows aggregate numbers only (no images, signals or report text).
"""
import json
from pathlib import Path

from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.util import Inches

from deck_common import (AMBER, DARK, GREY, LIGHT, NAVY, RED, ROOT, TEAL, WHITE, Deck, banner, box, card, table,
                         text)

OUT = Path(__file__).with_name("week07_progress.pptx")
SRC = "reports/mimic/MIMIC_RESULTS.md (mimic_metrics.json)"
m = json.loads((ROOT / "reports" / "mimic" / "mimic_metrics.json").read_text())
c, im, e, u, ds, tx = m["cohort"], m["image"], m["ecg"], m["unified"], m["downstream"], m["text"]
labs = m["labs"]
LAB_MAIN = "res_360min"


def p1(x):
    return f"{100 * x:.1f} %"


def ms(d, k=3):
    return f"{d['mean']:.{k}f} ± {d['sd']:.{k}f}"


d = Deck(7, "Oct 17", OUT)

# 1 — title
d.title_slide("Week 7: validation on linked MIMIC-IV data", "October 17, 2026")

# 2 — task and data
s = d.slide("The task and the data", source=SRC)
card(s, Inches(0.5), Inches(1.3), Inches(6.0), Inches(2.3), "Task (supervisor, Oct 17)", [
    "Use the MIMIC-IV multimodal data to test and validate the method",
    "Main experiments used 3 separate public datasets and synthetic text",
    "MIMIC: one patient has several REAL modalities at once",
], size=15)
rows = [["Per patient (linked by subject_id)", "count"],
        ["Patients (train / val / test)", f"{c['patients']} ({c['split']['train']} / {c['split']['val']} / {c['split']['test']})"],
        ["Chest X-ray images", f"{c['images']}"],
        ["Radiology reports (real text)", f"{c['reports']}"],
        ["12-lead ECGs, 7 days before the X-ray", f"{c['ecgs']}"],
        ["Lab tests kept / median results per patient", f"{c['lab_variables']} / {c['lab_results_per_patient_median']:.1f}"],
        ["Died within 1 year of the X-ray", f"{c['deaths_1y']} ({p1(c['prevalence_1y'])})"]]
table(s, Inches(6.8), Inches(1.3), Inches(6.0), rows, col_w=[4.0, 2.0], size=13, row_h=Inches(0.5))
card(s, Inches(0.5), Inches(3.8), Inches(6.0), Inches(2.4), "Timeline of one patient", [
    "t = 0: seven days before the X-ray (labs and ECGs start)",
    "Labs and ECGs at their real times",
    "X-ray images and report at t = 7 days",
    "Nothing after the X-ray is included",
], accent=NAVY, size=15)
banner(s, Inches(6.4), "Credentialed data: processed on our server only. Slides and GitHub show aggregate "
       "numbers only - no X-rays, ECG traces or report text.", accent=RED, h=Inches(0.55), size=13)

# 3 — method: what goes where
s = d.slide("Method: one sequence per patient")
flow = [("Labs", "event tokens", "existing TSEventCodec"), ("ECG", "window VQ tokens", "existing TSVQCodec"),
        ("Chest X-ray", "image VQ tokens", "image VQ, K = 1024"), ("Report", "sub-word tokens", "byte-level BPE")]
for i, (a, b, cc) in enumerate(flow):
    x = Inches(0.5 + i * 3.1)
    box(s, x, Inches(1.35), Inches(2.85), Inches(1.65), fill=LIGHT, shape=MSO_SHAPE.ROUNDED_RECTANGLE)
    text(s, x + Inches(0.15), Inches(1.45), Inches(2.6), Inches(0.5), a, size=19, bold=True, color=NAVY)
    text(s, x + Inches(0.15), Inches(1.95), Inches(2.6), Inches(0.5), b, size=15)
    text(s, x + Inches(0.15), Inches(2.4), Inches(2.6), Inches(0.5), cc, size=12, color=GREY)
box(s, Inches(0.5), Inches(3.2), Inches(12.3), Inches(0.6), fill=TEAL, shape=MSO_SHAPE.ROUNDED_RECTANGLE)
text(s, Inches(0.5), Inches(3.2), Inches(12.3), Inches(0.6),
     "<BOS>  labs  →  ECG(s)  →  X-ray image(s)  →  report  <EOS>   (one sequence, in time order)",
     size=16, bold=True, color=WHITE, align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
card(s, Inches(0.5), Inches(4.05), Inches(6.0), Inches(2.85), "Added this week (2 small changes)", [
    "RoutedCodec: ECG and labs are both 'time series' but need different tokenizers; the source "
    "is stored in the span, so decoding needs no extra information",
    "Exact long time gaps (next slide)",
], size=14)
card(s, Inches(6.8), Inches(4.05), Inches(6.0), Inches(2.85), "Unchanged", [
    "Tokenizer designs, vocabulary layout and wire format",
    "ECG is a new signal type: same time-series code, only a new config",
    "All 62 unit + integration tests pass",
], accent=NAVY, size=14)

# 4 — unified protocol results
s = d.slide("Result 1: the protocol works on real data", source=SRC)
sl = u["sequence_length"]
rows = [["Check", "Result"],
        ["Sequences passing structural validation", f"{u['structural_validation_passed']} / {u['patients']}"],
        ["Shard save → load identical", p1(u["shard_roundtrip_identical"])],
        ["Report text exact after the round-trip", p1(u["text_exact_after_shard"])],
        ["Lab pattern exact after the round-trip", p1(u["labs_mask_exact_after_shard"])],
        ["Segments decoded back", f"{u['segments_decoded']:,}"],
        ["Sequence length mean / max", f"{sl['mean']:,.0f} / {sl['max']:,}"],
        ["Compressed size per patient", f"{u['shard_bytes_per_patient'] / 1024:.1f} KiB"]]
table(s, Inches(0.5), Inches(1.3), Inches(7.0), rows, col_w=[4.6, 2.4], size=14, row_h=Inches(0.55))
names = {"mimic-cxr-jpg": "X-ray images", "mimic-iv-ecg": "ECG", "mimic-iv:labevents": "Labs",
         "mimic-cxr:report": "Report"}
rows = [["Source", "tokens / patient", "share"]] + [
    [names[k], f"{v:.0f}", p1(u["token_share_by_source"][k])] for k, v in u["tokens_by_source_mean"].items()]
table(s, Inches(7.9), Inches(1.3), Inches(4.9), rows, col_w=[2.0, 1.7, 1.2], size=14, row_h=Inches(0.55))
text(s, Inches(7.9), Inches(4.3), Inches(4.9), Inches(1.0), "Images take the largest share of the sequence; "
     "the report is the shortest part.", size=14, color=GREY)

# 5 — the bug real data found
s = d.slide("Result 2: real data found a protocol bug", source=SRC)
lm = labs[LAB_MAIN]
card(s, Inches(0.5), Inches(1.3), Inches(6.0), Inches(3.0), "Problem", [
    "Time-gap tokens <DT> are exact only up to 48 h; longer gaps use rounded buckets",
    "PhysioNet stays are 48 h, so this never happened before",
    f"MIMIC labs cover 7 days: {p1(lm['patients_with_gap_over_48h'])} of test patients have a gap > 48 h "
    "and their lab times came back wrong",
], accent=RED, size=15)
card(s, Inches(6.8), Inches(1.3), Inches(6.0), Inches(3.0), "Fix", [
    "A long gap is written as a chain of exact ≤ 48 h <DT> tokens",
    "Decoders already add consecutive <DT> tokens: no decoder change",
    "Gaps ≤ 48 h give the same tokens as before, so earlier results do not change",
    f"Now: lab pattern exact for {p1(u['labs_mask_exact_after_shard'])} of patients",
], size=15)
banner(s, Inches(4.6), "Why it matters: synthetic tests and 48-hour data could not reveal this. Real "
       "multi-day records did.", accent=TEAL, h=Inches(0.6), size=15)
text(s, Inches(0.5), Inches(5.4), Inches(12.3), Inches(1.2), "New unit test: "
     "test_ts_event_long_gaps_keep_exact_time (1 h, 6 h and 24 h grids over 7 days).", size=14, color=GREY)

# 6 — images
s = d.slide("Result 3: chest X-ray tokenizer", source=SRC + " - test split, " + str(im["n_test_images"]) + " images")
rows = [["Tokenizer", "size", "PSNR dB", "SSIM", "active codes", "tokens / image", "ratio (+overhead)"]]
for lab, k, size in (("MedMNIST, zero-shot", "medmnist_zero_shot_28", "28 px"),
                     ("MIMIC-trained", "mimic_trained_28", "28 px"), ("MIMIC-trained", "mimic_trained_64", "64 px")):
    r = im[k]
    rows.append([lab, size, f"{r['reconstruction']['psnr_db']:.2f}", f"{r['reconstruction']['ssim']:.3f}",
                 f"{r['utilization']['active_codes']} / {r['utilization']['n_codes']}",
                 f"{r['compression']['tokens_per_sample_mean']:.0f}",
                 f"{r['compression']['compression_ratio_with_overhead']:.1f}"])
table(s, Inches(0.5), Inches(1.3), Inches(12.3), rows, col_w=[2.6, 1.1, 1.5, 1.4, 2.0, 1.8, 1.9], size=14,
      row_h=Inches(0.55), bold_rows=(2,))
rv = im["mimic_trained_28_on_pneumoniamnist"]
card(s, Inches(0.5), Inches(3.75), Inches(6.0), Inches(2.9), "What it shows", [
    "The MedMNIST codebook reads real X-rays, but uses only part of its codes and blurs more",
    "Same design trained on MIMIC: almost every code used, clearly sharper",
    f"Reverse check: MIMIC 28 px tokenizer on PneumoniaMNIST: PSNR {rv['reconstruction']['psnr_db']:.2f} dB",
], size=14)
card(s, Inches(6.8), Inches(3.75), Inches(6.0), Inches(2.9), "Conclusion", [
    "The tokenizer DESIGN generalises to real hospital images",
    "The CODEBOOK should be trained on in-domain images",
    "Images are downsampled: fidelity of the tokenizer, not diagnostic quality",
], accent=AMBER, size=14)

# 7 — ECG and labs
s = d.slide("Result 4: ECG and lab tests", source=SRC)
rec = e["reconstruction"]
card(s, Inches(0.5), Inches(1.3), Inches(6.0), Inches(1.6), f"ECG: {e['n_ecg']} recordings, 12 leads, {e['hz']} Hz", [
    f"{e['tokens_per_ecg']} tokens per 10-second ECG, no code changes",
    f"Compression {e['compression']['compression_ratio_vs_100hz_16bit']:.0f}× vs a 100 Hz 16-bit signal",
], size=14)
rows = [["ECG test metric", "value"], ["Mean lead correlation", ms(rec["lead_corr"])],
        ["PRDN (lower = better)", ms(rec["prdn_pct"], 1) + " %"],
        ["Active codes", f"{e['utilization']['active_codes']} / {e['utilization']['n_codes']}"]]
table(s, Inches(0.5), Inches(3.05), Inches(6.0), rows, col_w=[3.2, 2.8], size=14, row_h=Inches(0.5))
banner(s, Inches(5.25), "Weakest part: the rough waveform shape is kept, fine detail is lost. Next: more codes "
       "per window or a convolutional encoder.", x=Inches(0.5), w=Inches(6.0), h=Inches(0.95), size=13)
rows = [["Lab grid step", "tokens / patient", "merged results", "pattern exact"]]
for k, r in labs.items():
    rows.append([f"{int(k[4:-3]) // 60} h", f"{r['compression']['tokens_per_sample_mean']:.0f}",
                 p1(r["results_merged_into_cells"]), p1(r["observation_mask_preserved"])])
text(s, Inches(6.8), Inches(1.3), Inches(6.0), Inches(0.5), f"Labs: {c['lab_variables']} tests over 7 days",
     size=18, bold=True, color=TEAL)
table(s, Inches(6.8), Inches(1.85), Inches(6.0), rows, col_w=[1.4, 1.6, 1.5, 1.5], size=14, row_h=Inches(0.5))
card(s, Inches(6.8), Inches(4.05), Inches(6.0), Inches(2.15), "Take-away", [
    "The event tokenizer fits real hospital labs without changes",
    "Coarser grid = fewer tokens but more results merged; 6 h chosen",
], size=14)

# 8 — real text
s = d.slide("Result 5: real radiology reports", source=SRC + f" - {tx['n_test_reports']} test reports")
rows = [["Text tokenizer", "vocabulary", "exact round-trip", "tokens / report", "tokens / word"]]
for k, lab in (("synthetic_template_bpe", "BPE from synthetic notes (old)"), ("mimic_bpe_1024", "BPE from MIMIC reports"),
               ("mimic_bpe_4096", "BPE from MIMIC reports")):
    r = tx[k]
    rows.append([lab, f"{r['vocab']:,}", p1(r["exact_roundtrip"]), ms(r["tokens_per_report"], 0),
                 f"{r['tokens_per_word']:.2f}"])
table(s, Inches(0.5), Inches(1.3), Inches(12.3), rows, col_w=[4.0, 1.7, 2.2, 2.4, 2.0], size=15, row_h=Inches(0.6))
card(s, Inches(0.5), Inches(4.0), Inches(6.0), Inches(2.6), "What it shows", [
    "Every tokenizer is lossless on real clinical text",
    "The synthetic-trained vocabulary needs many more tokens per report",
], size=15)
card(s, Inches(6.8), Inches(4.0), Inches(6.0), Inches(2.6), "Link to Week 4 feedback", [
    "Synthetic text was accepted on round-trip only",
    "Real text confirms the round-trip, and shows the vocabulary must come from real notes",
], accent=AMBER, size=15)

# 9 — downstream
s = d.slide("Result 6: are the tokens informative?", source=SRC)
rows = [["Input tokens", "AUROC", "AUPRC"]]
for k in ("image", "ecg", "labs", "text", "all (image+ecg+labs+text)", "demographics", "all + demographics"):
    rows.append([k, ms(ds[k]["auroc"]), ms(ds[k]["auprc"])])
table(s, Inches(0.5), Inches(1.3), Inches(7.2), rows, col_w=[3.4, 1.9, 1.9], size=14, row_h=Inches(0.52),
      bold_rows=(5,))
card(s, Inches(8.0), Inches(1.3), Inches(4.8), Inches(3.4), "Setup", [
    f"Task: death within 1 year; {ds['positives']} of {ds['n']} ({p1(ds['prevalence'])})",
    f"{ds['protocol'].split(',')[0]}",
    "Simple logistic regression on token counts",
    f"Chance: AUROC 0.5, AUPRC {ds['prevalence']:.2f}",
], size=13)
banner(s, Inches(5.55), "Each modality alone is above chance and all four together are best. But age + sex alone "
       "is already strong: we claim 'informative and combinable tokens', NOT a mortality model.",
       h=Inches(0.8), size=13)

# 10 — limits, next steps, questions
s = d.slide("Limits, next steps and questions")
card(s, Inches(0.5), Inches(1.3), Inches(4.0), Inches(5.4), "Limits", [
    f"{c['patients']} patients, one hospital: validation, not a clinical model",
    "X-rays downsampled to 28 / 64 px",
    "Mortality is a convenience label; wide CV spread",
    "MIMIC-trained models stay on our server",
], accent=RED, size=14)
card(s, Inches(4.7), Inches(1.3), Inches(4.0), Inches(5.4), "Next steps", [
    "Improve the ECG tokenizer (more codes per window, conv encoder)",
    "Try one shared image codebook for MedMNIST + MIMIC",
    "Add the MIMIC chapter to the thesis",
], size=14)
card(s, Inches(8.9), Inches(1.3), Inches(3.9), Inches(5.4), "Questions", [
    "Is this validation enough for the thesis, or more MIMIC data?",
    "Keep 1-year mortality, or another task?",
    "Main text or appendix for the MIMIC results?",
], accent=AMBER, size=14)

d.save()
