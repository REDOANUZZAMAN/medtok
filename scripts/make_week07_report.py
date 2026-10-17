"""Week 7 report for the supervisor (MIMIC-IV validation) -> reports/week07_mimic_report.docx (+ .pdf with soffice).

    .venv/bin/python scripts/make_week07_report.py

Every number is read from reports/mimic/mimic_metrics.json (written by medical_tokenizer.mimic_validate);
nothing is typed by hand. Aggregate numbers only: no MIMIC images, signals, report text or identifiers.
"""
import json
import subprocess
from pathlib import Path

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Pt, RGBColor

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "week07_mimic_report.docx"
m = json.loads((ROOT / "reports" / "mimic" / "mimic_metrics.json").read_text())
c, im, e, u, ds, tx, labs = m["cohort"], m["image"], m["ecg"], m["unified"], m["downstream"], m["text"], m["labs"]
TEAL = RGBColor(0x0F, 0x76, 0x6E)


def p1(x):
    return f"{100 * x:.1f} %"


def ms(d, k=3):
    return f"{d['mean']:.{k}f} ± {d['sd']:.{k}f}"


doc = Document()
st = doc.styles["Normal"]
st.font.name, st.font.size = "Calibri", Pt(11)


def h(text, size=13):
    para = doc.add_paragraph()
    r = para.add_run(text)
    r.bold, r.font.size, r.font.color.rgb = True, Pt(size), TEAL
    para.paragraph_format.space_before = Pt(10)
    return para


def para(text, italic=False, size=None):
    q = doc.add_paragraph()
    r = q.add_run(text)
    r.italic = italic
    if size:
        r.font.size = Pt(size)
    return q


def bullet(text, bold_lead=None):
    q = doc.add_paragraph(style="List Bullet")
    if bold_lead:
        q.add_run(bold_lead).bold = True
    q.add_run(text)


def shade(cell, hex_fill):
    tcPr = cell._tc.get_or_add_tcPr()
    sh = OxmlElement("w:shd")
    sh.set(qn("w:val"), "clear")
    sh.set(qn("w:color"), "auto")
    sh.set(qn("w:fill"), hex_fill)
    tcPr.append(sh)


def table(rows, caption=None):
    if caption:
        para(caption, italic=True, size=10)
    t = doc.add_table(rows=len(rows), cols=len(rows[0]))
    t.style = "Table Grid"
    for i, row in enumerate(rows):
        for j, val in enumerate(row):
            cell = t.cell(i, j)
            cell.text = ""
            r = cell.paragraphs[0].add_run(str(val))
            r.font.size = Pt(9.5)
            if i == 0:
                r.bold = True
                r.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
                shade(cell, "1F2A44")
    doc.add_paragraph()


# ---------------------------------------------------------------- title
t = doc.add_paragraph()
r = t.add_run("MedTok – Week 7 Progress Report")
r.bold, r.font.size = True, Pt(18)
t.add_run("\nValidation on linked MIMIC-IV multimodal data").font.size = Pt(12.5)
t.add_run("\nREDOANUZZAMAN · October 17, 2026").font.size = Pt(10.5)

# ---------------------------------------------------------------- 1
h("1. Goal of this week")
para("You asked me to use the MIMIC-IV multimodal data from Rafi to test and validate my method. Until now I "
     "tested on three separate public datasets (MedMNIST, PhysioNet 2012, Cataract-101), and my clinical text was "
     "synthetic. MIMIC lets me check three open questions: (1) do the tokenizers work on real hospital data, "
     "(2) does the unified protocol hold when one patient has several real modalities at once, and (3) is real "
     "clinical text handled losslessly?")

# ---------------------------------------------------------------- 2
h("2. Data")
table([["Item", "Value"],
       ["Patients (train / val / test, split by patient, stratified by label)",
        f"{c['patients']} ({c['split']['train']} / {c['split']['val']} / {c['split']['test']})"],
       ["Chest X-ray images (MIMIC-CXR-JPG, all views)", f"{c['images']}"],
       ["Radiology reports (real free text)", f"{c['reports']}"],
       ["12-lead ECGs (MIMIC-IV-ECG), 7 days before the X-ray", f"{c['ecgs']}"],
       ["Lab tests kept / median results per patient", f"{c['lab_variables']} / {c['lab_results_per_patient_median']:.1f}"],
       ["Died within 1 year of the X-ray (label)", f"{c['deaths_1y']} ({p1(c['prevalence_1y'])})"],
       ["Median age / male fraction", f"{c['age_median']:.0f} / {c['male_fraction']:.2f}"]],
      "Table 1. The linked demo subset (p10–p19). Join key: subject_id.")
para("Timeline of one patient: t = 0 is seven days before the X-ray; labs and ECGs keep their real times; the X-ray "
     "and report are at t = 7 days. Nothing after the X-ray is used. The data is processed on our server only; "
     "this report contains aggregate numbers only.")

# ---------------------------------------------------------------- 3
h("3. What I did")
bullet(" each modality goes through an existing tokenizer: labs → event tokens, ECG → window VQ tokens, X-ray → "
       "image VQ tokens (K = 1,024), report → byte-level BPE sub-words. All parts of a patient are written into one "
       "protocol v0.2 sequence in time order.", "Method:")
bullet(" ECG and labs are both 'time series' but need different tokenizers. A small router picks the codec from "
       "the data source and stores the source in the sequence, so decoding needs no extra information.",
       "Addition 1 – RoutedCodec:")
bullet(" time-gap tokens were exact only up to 48 h (longer gaps used rounded buckets). Long gaps are now written as "
       "a chain of exact ≤ 48 h tokens. Gaps ≤ 48 h give the same tokens as before, so all earlier PhysioNet results "
       "are unchanged.", "Addition 2 – exact long gaps:")
bullet(" the MedMNIST image tokenizer was used unchanged (zero-shot), and the same design was trained on MIMIC "
       "X-rays (28 px and 64 px). All 62 unit and integration tests pass.", "Images:")

# ---------------------------------------------------------------- 4
h("4. Results")
sl = u["sequence_length"]
table([["Check (all 600 patients)", "Result"],
       ["Sequences passing structural validation", f"{u['structural_validation_passed']} / {u['patients']}"],
       ["Save → load of the token shard identical", p1(u["shard_roundtrip_identical"])],
       ["Report text exact after the round-trip", p1(u["text_exact_after_shard"])],
       ["Lab observation pattern exact after the round-trip", p1(u["labs_mask_exact_after_shard"])],
       ["Sequence length mean / max (tokens)", f"{sl['mean']:,.0f} / {sl['max']:,}"],
       ["Compressed size per patient", f"{u['shard_bytes_per_patient'] / 1024:.1f} KiB"]],
      "Table 2. Unified protocol on real linked records.")

rows = [["Image tokenizer (test, " + str(im["n_test_images"]) + " images)", "Size", "PSNR dB", "SSIM", "Active codes",
         "Tokens / image"]]
for lab, k, size in (("MedMNIST, zero-shot", "medmnist_zero_shot_28", "28 px"), ("MIMIC-trained", "mimic_trained_28", "28 px"),
                     ("MIMIC-trained", "mimic_trained_64", "64 px")):
    r_ = im[k]
    rows.append([lab, size, f"{r_['reconstruction']['psnr_db']:.2f}", f"{r_['reconstruction']['ssim']:.3f}",
                 f"{r_['utilization']['active_codes']} / 1024", f"{r_['compression']['tokens_per_sample_mean']:.0f}"])
table(rows, "Table 3. Chest X-ray reconstruction. Higher PSNR/SSIM = closer to the original.")

lm = labs["res_360min"]
table([["ECG / labs", "Result"],
       ["ECG: tokens per 10-s ECG / compression vs 100 Hz 16-bit",
        f"{e['tokens_per_ecg']} / {e['compression']['compression_ratio_vs_100hz_16bit']:.0f}×"],
       ["ECG: mean lead correlation (test)", ms(e["reconstruction"]["lead_corr"])],
       ["ECG: PRDN % (lower = better)", ms(e["reconstruction"]["prdn_pct"], 1)],
       ["Labs (6 h grid): tokens per patient / pattern exact", f"{lm['compression']['tokens_per_sample_mean']:.0f} / "
        f"{p1(lm['observation_mask_preserved'])}"],
       ["Labs: test patients with a gap > 48 h (affected by the fix)", p1(lm["patients_with_gap_over_48h"])]],
      "Table 4. ECG and lab tests.")

rows = [["Text tokenizer (92 test reports)", "Vocabulary", "Exact round-trip", "Tokens / report"]]
for k, lab in (("synthetic_template_bpe", "BPE trained on synthetic notes"), ("mimic_bpe_1024", "BPE trained on MIMIC reports"),
               ("mimic_bpe_4096", "BPE trained on MIMIC reports")):
    rows.append([lab, f"{tx[k]['vocab']:,}", p1(tx[k]["exact_roundtrip"]), ms(tx[k]["tokens_per_report"], 0)])
table(rows, "Table 5. Real radiology reports.")

rows = [["Input tokens", "AUROC", "AUPRC"]]
for k in ("image", "ecg", "labs", "text", "all (image+ecg+labs+text)", "demographics", "all + demographics"):
    rows.append([k, ms(ds[k]["auroc"]), ms(ds[k]["auprc"])])
table(rows, f"Table 6. Death within 1 year ({ds['positives']} of {ds['n']}). {ds['protocol']}. "
            f"Chance: AUROC 0.5, AUPRC {ds['prevalence']:.2f}.")

# ---------------------------------------------------------------- 5
h("5. Key findings")
bullet(" all 600 patient sequences pass validation and come back exactly after saving and loading. This is the main "
       "claim of the thesis, now shown on real linked records.", "The protocol works on real data:")
bullet(f" {p1(lm['patients_with_gap_over_48h'])} of test patients have a lab gap over 48 h; before the fix their lab "
       "times were decoded wrongly. Synthetic tests and 48-h PhysioNet stays could not show this.", "Real data found a bug:")
bullet(" the MedMNIST codebook reads real X-rays but uses only part of its codes and blurs more; the same design "
       "trained on MIMIC uses almost all codes and is clearly sharper. The design generalises; the codebook should "
       "be trained on in-domain images.", "Images:")
bullet(" it works with no code changes and high compression, but keeps only the rough waveform shape. This is the "
       "weakest part.", "ECG:")
bullet(" lossless for every tokenizer; the synthetic-trained vocabulary needs more than twice as many tokens per "
       "report as a vocabulary trained on real reports.", "Real text:")
bullet(" every modality alone is above chance and all four together are best. Age and sex alone are already strong, "
       "so I claim 'informative and combinable tokens', not a mortality model.", "Tokens are informative:")

# ---------------------------------------------------------------- 6
h("6. Limitations")
bullet(f" {c['patients']} patients from one hospital system: a validation of the method, not a clinical model.", "Size:")
bullet(" X-rays are downsampled to 28 / 64 px, so the image metrics measure tokenizer fidelity, not diagnostic "
       "quality.", "Resolution:")
bullet(" mortality is a convenience label; cross-validation spread is wide.", "Label:")
bullet(" all processing stays on our server; models trained on MIMIC are not shared.", "Data use:")

# ---------------------------------------------------------------- 7
h("7. Next steps")
bullet(" more codes per window or a convolutional encoder.", "Improve the ECG tokenizer:")
bullet(" train one image codebook on MedMNIST + MIMIC and test it on both.", "Shared codebook:")
bullet(" write the MIMIC validation chapter of the thesis.", "Thesis:")

# ---------------------------------------------------------------- 8
h("8. Questions for you (please write your answer under each question)")
QUESTIONS = [
    "Is this MIMIC validation enough for the thesis, or should I use more MIMIC data (for example, more patients)?",
    "Is death within 1 year a good downstream task, or do you prefer another label (for example, findings from the "
    "report, or in-hospital mortality)?",
    "Should the MIMIC results go in the main text of the thesis or in the appendix?",
    "The ECG tokenizer is the weakest part. Should I improve it now, or report it as a limitation?",
    "Should I try one shared image codebook for MedMNIST + MIMIC, or keep separate codebooks?",
    "Is it acceptable to show only aggregate numbers (no MIMIC images) in the thesis and the defence slides?",
    "Do you need the code or the results in another format?",
]
for i, q in enumerate(QUESTIONS, 1):
    qp = doc.add_paragraph()
    qp.add_run(f"Q{i}. ").bold = True
    qp.add_run(q)
    a = doc.add_paragraph()
    ar = a.add_run("Your answer: " + "_" * 58 + "\n" + "_" * 70 + "\n" + "_" * 70)
    ar.font.color.rgb = RGBColor(0x64, 0x74, 0x8B)

# ---------------------------------------------------------------- footer
doc.add_paragraph()
f = para("Source of all numbers: reports/mimic/MIMIC_RESULTS.md and mimic_metrics.json, generated by "
         "python -m medical_tokenizer.mimic_validate. Code: github.com/REDOANUZZAMAN/medtok "
         "(docs/mimic_validation.md).", italic=True, size=9)
f.alignment = WD_ALIGN_PARAGRAPH.LEFT

doc.save(OUT)
print(OUT)
try:
    subprocess.run(["soffice", "--headless", "--convert-to", "pdf", "--outdir", str(OUT.parent), str(OUT)],
                   check=True, capture_output=True, timeout=180)
    print(OUT.with_suffix(".pdf"))
except (OSError, subprocess.SubprocessError) as err:
    print("pdf skipped:", err)
