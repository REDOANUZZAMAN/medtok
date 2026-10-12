"""Week 1 supervisor update deck. Run: .venv/bin/python slides/build_week01.py"""
from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.util import Inches, Pt

OUT = Path(__file__).with_name("week01_progress.pptx")

NAVY = RGBColor(0x1F, 0x2A, 0x44)
TEAL = RGBColor(0x0F, 0x76, 0x6E)
LIGHT = RGBColor(0xF1, 0xF5, 0xF9)
GREY = RGBColor(0x64, 0x74, 0x8B)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
DARK = RGBColor(0x1E, 0x29, 0x3B)
FONT = "Calibri"

prs = Presentation()
prs.slide_width, prs.slide_height = Inches(13.333), Inches(7.5)
BLANK = prs.slide_layouts[6]
W = prs.slide_width


def box(slide, x, y, w, h, fill=None, line=None, shape=MSO_SHAPE.RECTANGLE):
    s = slide.shapes.add_shape(shape, x, y, w, h)
    if fill is None:
        s.fill.background()
    else:
        s.fill.solid()
        s.fill.fore_color.rgb = fill
    if line is None:
        s.line.fill.background()
    else:
        s.line.color.rgb = line
    s.shadow.inherit = False
    return s


def text(slide, x, y, w, h, lines, size=18, color=DARK, bold=False, align=PP_ALIGN.LEFT,
         anchor=MSO_ANCHOR.TOP, font=FONT, bullet=False, spacing=6):
    tb = slide.shapes.add_textbox(x, y, w, h)
    tf = tb.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = anchor
    if isinstance(lines, str):
        lines = [lines]
    for i, ln in enumerate(lines):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        p.space_after = Pt(spacing)
        # a line may be (text, overrides) to vary size / colour / boldness
        t, o = (ln, {}) if isinstance(ln, str) else ln
        r = p.add_run()
        r.text = ("•  " if bullet and not o.get("nobullet") else "") + t
        r.font.name = o.get("font", font)
        r.font.size = Pt(o.get("size", size))
        r.font.bold = o.get("bold", bold)
        r.font.color.rgb = o.get("color", color)
    return tb


def header(slide, title, n):
    box(slide, 0, 0, W, Inches(1.0), fill=NAVY)
    text(slide, Inches(0.5), Inches(0.12), Inches(11), Inches(0.8), title, size=30, color=WHITE,
         bold=True, anchor=MSO_ANCHOR.MIDDLE)
    text(slide, Inches(10.3), Inches(0.12), Inches(2.6), Inches(0.8), "Week 1  ·  Oct 8, 2026",
         size=14, color=RGBColor(0xCB, 0xD5, 0xE1), align=PP_ALIGN.RIGHT, anchor=MSO_ANCHOR.MIDDLE)
    text(slide, Inches(12.2), Inches(7.0), Inches(0.9), Inches(0.4), str(n), size=12, color=GREY,
         align=PP_ALIGN.RIGHT)


def card(slide, x, y, w, h, title, body, accent=TEAL, size=15):
    box(slide, x, y, w, h, fill=LIGHT)
    box(slide, x, y, Inches(0.08), h, fill=accent)
    text(slide, x + Inches(0.25), y + Inches(0.12), w - Inches(0.35), Inches(0.5), title, size=18,
         bold=True, color=accent)
    text(slide, x + Inches(0.25), y + Inches(0.62), w - Inches(0.35), h - Inches(0.7), body,
         size=size, bullet=True, spacing=4)


def table(slide, x, y, w, rows, col_w, size=14, row_h=Inches(0.45)):
    t = slide.shapes.add_table(len(rows), len(rows[0]), x, y, w, row_h * len(rows)).table
    for j, cw in enumerate(col_w):
        t.columns[j].width = Inches(cw)
    for i, row in enumerate(rows):
        for j, val in enumerate(row):
            c = t.cell(i, j)
            c.fill.solid()
            c.fill.fore_color.rgb = NAVY if i == 0 else (LIGHT if i % 2 else WHITE)
            tf = c.text_frame
            tf.paragraphs[0].text = ""
            r = tf.paragraphs[0].add_run()
            r.text = val
            r.font.name, r.font.size = FONT, Pt(size)
            r.font.bold = i == 0
            r.font.color.rgb = WHITE if i == 0 else DARK
    return t


n = 0

# 1 — title
n += 1
s = prs.slides.add_slide(BLANK)
box(s, 0, 0, W, prs.slide_height, fill=NAVY)
box(s, Inches(0.8), Inches(2.0), Inches(0.12), Inches(2.6), fill=TEAL)
text(s, Inches(1.2), Inches(1.9), Inches(11), Inches(0.5), "UNDERGRADUATE THESIS · WEEKLY PROGRESS REPORT 1",
     size=16, color=RGBColor(0x5E, 0xEA, 0xD4), bold=True)
text(s, Inches(1.2), Inches(2.4), Inches(11.3), Inches(1.8),
     "Design of a Medical Multimodal-Specific Tokenizer and a Unified Data Protocol", size=40,
     color=WHITE, bold=True)
text(s, Inches(1.2), Inches(4.6), Inches(11), Inches(0.6),
     "Week 1: problem definition, dataset selection and the unified token protocol", size=20,
     color=RGBColor(0xCB, 0xD5, 0xE1))
text(s, Inches(1.2), Inches(5.7), Inches(11), Inches(1.0),
     [("REDOANUZZAMAN", {"bold": True, "size": 20, "color": WHITE}),
      ("Sichuan University  ·  October 8, 2026", {"size": 16, "color": RGBColor(0xCB, 0xD5, 0xE1)})])

# 2 — problem & goal
n += 1
s = prs.slides.add_slide(BLANK)
header(s, "Problem and goal", n)
card(s, Inches(0.5), Inches(1.4), Inches(6.0), Inches(2.6), "Problem", [
    "Medical data is heterogeneous: images, ICU vital signs, surgical video, clinical text, 3D scans",
    "Each modality usually gets its own model and its own input format",
    "A single Transformer cannot read a mix of them without a shared format",
], accent=RGBColor(0xB9, 0x1C, 0x1C))
card(s, Inches(6.8), Inches(1.4), Inches(6.0), Inches(2.6), "Goal", [
    "One discrete token vocabulary for all medical modalities",
    "One sequence format (wire format) with time and position information",
    "A fair comparison of tokenisation strategies on real medical tasks",
])
text(s, Inches(0.5), Inches(4.35), Inches(12), Inches(0.5), "Three representation strategies to compare",
     size=20, bold=True, color=NAVY)
for i, (t, d) in enumerate([
    ("A. Continuous projection", "Encoder output vectors are passed directly to the model (no discretisation)"),
    ("B. Vector quantisation", "Encoder vectors are mapped to the nearest entry of a learned codebook"),
    ("C. Event-based tokens", "Measurements and annotations become symbolic tokens: <VAR:HR> <VBIN:5> …"),
]):
    x = Inches(0.5 + i * 4.15)
    box(s, x, Inches(4.95), Inches(3.95), Inches(1.9), fill=WHITE, line=RGBColor(0xCB, 0xD5, 0xE1))
    text(s, x + Inches(0.2), Inches(5.05), Inches(3.6), Inches(0.5), t, size=17, bold=True, color=TEAL)
    text(s, x + Inches(0.2), Inches(5.55), Inches(3.6), Inches(1.3), d, size=14)

# 3 — roadmap
n += 1
s = prs.slides.add_slide(BLANK)
header(s, "Six-week plan", n)
plan = [
    ("Week 1", "Problem definition, datasets, unified token protocol", True),
    ("Week 2", "Image tokenizer on MedMNIST: VQ vs continuous", False),
    ("Week 3", "Time-series tokenizers on PhysioNet 2012: event / VQ / continuous", False),
    ("Week 4", "Surgical-video tokenizers on Cataract-101; text and point-cloud interfaces", False),
    ("Week 5", "Downstream comparison: one Transformer, all strategies and tasks", False),
    ("Week 6", "Ablation studies, final results, thesis draft", False),
]
for i, (wk, d, cur) in enumerate(plan):
    y = Inches(1.35 + i * 0.95)
    box(s, Inches(0.8), y, Inches(1.6), Inches(0.75), fill=TEAL if cur else LIGHT,
        shape=MSO_SHAPE.ROUNDED_RECTANGLE)
    text(s, Inches(0.8), y, Inches(1.6), Inches(0.75), wk, size=18, bold=True,
         color=WHITE if cur else NAVY, align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
    text(s, Inches(2.7), y, Inches(8.5), Inches(0.75), d, size=19, bold=cur,
         color=DARK if cur else GREY, anchor=MSO_ANCHOR.MIDDLE)
    if cur:
        text(s, Inches(10.6), y, Inches(2.2), Inches(0.75), "◀  this week", size=16, bold=True,
             color=TEAL, anchor=MSO_ANCHOR.MIDDLE)

# 4 — datasets
n += 1
s = prs.slides.add_slide(BLANK)
header(s, "Datasets selected (open-access only)", n)
table(s, Inches(0.5), Inches(1.35), Inches(12.3), [
    ["Modality", "Dataset", "Size", "Task", "Licence"],
    ["2D images", "MedMNIST v2 (Blood / Pneumonia / Derma), 28×28",
     "23,674 train images", "8 / 2 / 7-class classification", "CC BY 4.0"],
    ["ICU time-series", "PhysioNet 2012 set-a, 37 variables + 4 static",
     "4,000 stays (2,799 / 599 / 602)", "In-hospital mortality", "ODC-By 1.0"],
    ["Surgical video", "Cataract-101, sampled at 2 fps, 64×64",
     "101 surgeries (71 / 15 / 15)", "10-phase recognition", "CC BY-NC 4.0"],
    ["Clinical text", "Templated notes from structured fields", "—", "Interface only", "—"],
    ["3D point cloud", "Synthetic surfaces", "—", "Interface only", "—"],
], col_w=[1.8, 4.1, 2.6, 2.4, 1.4], size=14, row_h=Inches(0.55))
card(s, Inches(0.5), Inches(4.8), Inches(6.0), Inches(2.4), "Why Cataract-101, not Cholec80", [
    "Cholec80 archive returns HTTP 403 (needs CAMMA request)",
    "Cataract-101: closest open surgical video with phase labels",
    "Cholec80 loaders already written, ready if access granted",
], accent=RGBColor(0xB4, 0x53, 0x09), size=14)
card(s, Inches(6.8), Inches(4.8), Inches(6.0), Inches(2.4), "Data-handling rules", [
    "Patient / video-level splits, fixed seed, no leakage",
    "Normalisation and value bins fitted on the training split only",
    "Request-only data (MIMIC, SCARED) is never downloaded by code",
], size=14)

# 5 — vocabulary
n += 1
s = prs.slides.add_slide(BLANK)
header(s, "Unified token protocol: 13,824-id vocabulary", n)
ranges = [  # (name, size, colour) from docs/token_protocol.md §3
    ("special", 256, RGBColor(0x47, 0x55, 0x69)), ("dt", 256, RGBColor(0x16, 0xA3, 0x4A)),
    ("var", 256, RGBColor(0xDC, 0x26, 0x26)), ("vbin", 256, RGBColor(0x7C, 0x2D, 0x12)),
    ("event", 256, RGBColor(0x93, 0x33, 0xEA)), ("segment", 256, RGBColor(0xDB, 0x27, 0x77)),
    ("text", 8192, RGBColor(0xD9, 0x77, 0x06)), ("image_vq", 1024, RGBColor(0x25, 0x63, 0xEB)),
    ("ts_vq", 1024, RGBColor(0x08, 0x91, 0xB2)), ("video_vq", 1024, RGBColor(0x0D, 0x94, 0x88)),
    ("pc_vq", 1024, RGBColor(0x4F, 0x46, 0xE5)),
]
total = sum(r[1] for r in ranges)
x0, bar_w = Inches(0.5), Inches(12.3)
x = x0
for name, size, col in ranges:
    w = int(bar_w * size / total)
    box(s, x, Inches(1.5), w, Inches(0.7), fill=col, line=WHITE)
    if size >= 1024:
        text(s, x, Inches(1.5), w, Inches(0.7), name, size=11, bold=True, color=WHITE,
             align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
    x += w
text(s, x0, Inches(2.25), Inches(4), Inches(0.4), "0", size=12, color=GREY)
text(s, x0 + bar_w - Inches(2), Inches(2.25), Inches(2), Inches(0.4), "13,823", size=12, color=GREY,
     align=PP_ALIGN.RIGHT)
text(s, x0, Inches(2.25), Inches(3.2), Inches(0.4), "  ← six 256-id structural ranges", size=12, color=GREY)
table(s, Inches(0.5), Inches(2.9), Inches(7.4), [
    ["Range", "Ids", "Meaning"],
    ["special", "0–255", "<BOS> <EOS>, modality start/end, <REP:…>"],
    ["dt", "256–511", "<DT:k> time gap (seconds → 30 days)"],
    ["var / vbin", "512–1023", "time-series variable + quantile bin"],
    ["event", "1024–1279", "expert-annotated events (e.g. phases)"],
    ["segment", "1280–1535", "unsupervised clip states"],
    ["text", "1536–9727", "8,192 sub-word pieces"],
    ["*_vq", "9728–13823", "4 codebooks × 1,024 (image/ts/video/pc)"],
], col_w=[1.4, 1.6, 4.4], size=13, row_h=Inches(0.48))
card(s, Inches(8.2), Inches(2.9), Inches(4.6), Inches(3.85), "Design choices", [
    "Fixed, non-overlapping ranges: never learned, so any tokenizer can be swapped",
    "Event vs segment kept apart: an annotated phase is never confused with a cluster id",
    "Exact time is kept in a side channel; <DT> tokens are a coarse view of it",
], size=14)

# 6 — sequence format
n += 1
s = prs.slides.add_slide(BLANK)
header(s, "Sequence format + real example", n)
text(s, Inches(0.5), Inches(1.25), Inches(12), Inches(0.5), "Wire format", size=20, bold=True, color=NAVY)
box(s, Inches(0.5), Inches(1.75), Inches(12.3), Inches(1.75), fill=DARK)
text(s, Inches(0.7), Inches(1.82), Inches(12), Inches(1.65), [
    "<BOS>",
    "  <TEXT_START> <REP:SUBWORD> w w w … <TEXT_END>",
    "  <DT:k> <IMAGE_START> <REP:VQ> <ROW> v v v … <ROW> … <IMAGE_END>",
    "  <DT:k> <TIME_SERIES_START> <REP:EVENT> <STATIC> <VAR:Age> <VBIN:9> … <TIME_SERIES_END>",
    "<EOS>",
], size=14, font="Consolas", color=RGBColor(0xE2, 0xE8, 0xF0), spacing=0)
text(s, Inches(0.5), Inches(3.7), Inches(12), Inches(0.5),
     "Real output: short note + 12 ICU measurements (40 tokens)",
     size=20, bold=True, color=NAVY)
box(s, Inches(0.5), Inches(4.2), Inches(12.3), Inches(1.45), fill=LIGHT)
text(s, Inches(0.7), Inches(4.27), Inches(11.9), Inches(1.35),
     "<BOS> <TEXT_START> <REP:SUBWORD> <TEXT:40> <TEXT:305> <TEXT:326> <TEXT:11> <TEXT:388> <TEXT:334> "
     "<TEXT:306> <TEXT:13> <TEXT_END> <DT:0> <TIME_SERIES_START> <REP:EVENT> <STATIC> <VAR:Age> <VBIN:4> "
     "<VAR:Gender> <VBIN:1> <VAR:ICUType> <VBIN:4> <VAR:GCS> <VBIN:9> <VAR:HR> <VBIN:4> <VAR:NIDiasABP> "
     "<VBIN:10> <VAR:NIMAP> <VBIN:13> <VAR:NISysABP> <VBIN:14> <VAR:RespRate> <VBIN:7> <VAR:Temp> <VBIN:0> "
     "<VAR:Urine> <VBIN:15> <TIME_SERIES_END> <EOS>", size=13, font="Consolas", spacing=0)
text(s, Inches(0.5), Inches(5.85), Inches(12.3), Inches(1.3), [
    "Each token also carries side channels: modality id, exact time (minutes), 3-D position, optional confidence",
    "Lossless containers: .npz shards for training and human-readable JSONL (one token per line); a validator checks every sequence",
], size=15, bullet=True)

# 7 — done this week
n += 1
s = prs.slides.add_slide(BLANK)
header(s, "Work completed this week", n)
done = [
    ("Problem scope", "Defined the 5 modalities, 3 strategies and the evaluation metrics: compression, reconstruction, codebook use, sequence length, downstream score"),
    ("Data pipeline", "Download / cleaning / splitting for MedMNIST and PhysioNet 2012; Cataract-101 frame cache at 2 fps, 64×64"),
    ("Protocol spec v0.2", "Vocabulary ranges, special tokens, time-delta buckets, wire format, side channels, storage formats"),
    ("Reference code", "Python package medical_tokenizer: protocol, serializer, validator, JSONL export"),
    ("Tests", "Unit + end-to-end tests on synthetic data, so the format is checked without downloads"),
]
for i, (t, d) in enumerate(done):
    y = Inches(1.35 + i * 1.12)
    box(s, Inches(0.6), y + Inches(0.15), Inches(0.5), Inches(0.5), fill=TEAL, shape=MSO_SHAPE.OVAL)
    text(s, Inches(0.6), y + Inches(0.15), Inches(0.5), Inches(0.5), "✓", size=18, bold=True, color=WHITE,
         align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
    text(s, Inches(1.35), y, Inches(3.0), Inches(0.8), t, size=19, bold=True, color=NAVY,
         anchor=MSO_ANCHOR.MIDDLE)
    text(s, Inches(4.3), y, Inches(8.5), Inches(0.8), d, size=16, anchor=MSO_ANCHOR.MIDDLE)

# 8 — next week + questions
n += 1
s = prs.slides.add_slide(BLANK)
header(s, "Next week and questions", n)
card(s, Inches(0.5), Inches(1.4), Inches(6.0), Inches(4.6), "Plan for Week 2: image tokenizer", [
    "Train a patch autoencoder on MedMNIST (Blood / Pneumonia / Derma)",
    "B: vector-quantised version with a 1,024-entry codebook",
    "A: continuous version with the same encoder, as the baseline",
    "Measure reconstruction error, codebook utilisation, tokens per image",
    "Show reconstructed images side by side with the originals",
])
card(s, Inches(6.8), Inches(1.4), Inches(6.0), Inches(4.6), "Questions for supervisor", [
    "Is Cataract-101 acceptable in place of Cholec80, or should I file the CAMMA data request?",
    "Is templated text OK for the text modality, or is MIMIC credentialed access expected?",
    "Is the six-week plan and its weekly scope acceptable?",
], accent=RGBColor(0xB4, 0x53, 0x09))

prs.save(OUT)
print(OUT)
