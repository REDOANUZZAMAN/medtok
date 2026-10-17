"""MedTok demo: see medical data -> tokens -> data again, in the browser.

    .venv/bin/python demo/server.py [--port 8093]      # then open http://<host>:8093

Only the Python standard library + the project's own code (no web framework). All models are the
`latest` runs in experiments/ and run on the CPU. Every sample shown comes from a TEST split.
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import math
import os
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

import numpy as np                                                  # noqa: E402
import torch                                                        # noqa: E402
import yaml                                                         # noqa: E402
from PIL import Image                                               # noqa: E402

from medical_tokenizer.data import datasets as D                    # noqa: E402
from medical_tokenizer.data.notes import note_fields, stay_note, template_source  # noqa: E402
from medical_tokenizer.data.physionet import TS_VARS                # noqa: E402
from medical_tokenizer.data.video import CATARACT101_PHASES         # noqa: E402
from medical_tokenizer.downstream.dataset import collate            # noqa: E402
from medical_tokenizer.downstream.model import UnifiedTokenTransformer  # noqa: E402
from medical_tokenizer.evaluation.reconstruction import image_metrics  # noqa: E402
from medical_tokenizer.protocol import split_spans, to_tokens       # noqa: E402
from medical_tokenizer.protocol.schema import Sample, Segment       # noqa: E402
from medical_tokenizer.protocol.serialize import serialize          # noqa: E402
from medical_tokenizer.protocol.vocab import CONT, PROTOCOL_VERSION, VOCAB_SIZE  # noqa: E402
from medical_tokenizer.tokenizers.timeseries.codec import stay_segment  # noqa: E402
from medical_tokenizer.tokenizers.video.codec import AnnotatedPhaseEventCodec, make_clips  # noqa: E402
from medical_tokenizer.protocol.vocab import RANGES                 # noqa: E402
from medical_tokenizer.toolkit import build_lexicon, load_text_tokenizer, load_tokenizer  # noqa: E402
from medical_tokenizer.utils.experiment import resolve_run          # noqa: E402

torch.set_num_threads(4)
os.chdir(ROOT)

CLASSES = {
    "bloodmnist": ["basophil", "eosinophil", "erythroblast", "immature granulocyte", "lymphocyte", "monocyte",
                   "neutrophil", "platelet"],
    "pneumoniamnist": ["normal", "pneumonia"],
    "dermamnist": ["actinic keratosis", "basal cell carcinoma", "benign keratosis", "dermatofibroma", "melanoma",
                   "melanocytic nevus", "vascular lesion"],
}
SHOW_VARS = ["HR", "MAP", "Temp", "GCS", "RespRate", "SysABP"]


# ------------------------------------------------------------------ loading
class State:
    def __init__(self):
        with open("configs/data.yaml") as f:
            self.data = yaml.safe_load(f)
        self.lex = build_lexicon()
        self.text = load_text_tokenizer()
        self.img = {r: load_tokenizer(f"experiments/image_{r}") for r in ("vq", "continuous")}
        self.ts = {"event": load_tokenizer("experiments/timeseries_event"),
                   "vq": load_tokenizer("experiments/timeseries_vq"),
                   "continuous": load_tokenizer("experiments/timeseries_continuous")}
        self.vid = {r: load_tokenizer(f"experiments/video_{r}") for r in ("vq", "continuous")}
        with open(os.path.join(resolve_run("experiments/video_vq"), "config.yaml")) as f:
            self.vcfg = yaml.safe_load(f)["data"]
        self.phase_codec = AnnotatedPhaseEventCodec(self.lex, "cataract101", CATARACT101_PHASES)
        self.images = {s: D.image_split(self.data["medmnist"], "test", [s])[:2] for s in CLASSES}
        self.stays = [s for s in D.stays(self.data["physionet2012"]) if s.split == "test"]
        self.videos = {v: p for v, p in D.video_files(self.vcfg).items()
                       if D.video_split_map(self.vcfg)[v] == "test"}
        self.k_img = self.img["vq"].model.n_codes
        print(f"loaded: image K={self.k_img}, {len(self.stays)} test stays, {len(self.videos)} test videos",
              flush=True)


S: State | None = None


# ------------------------------------------------------------------ helpers
def png(a: np.ndarray, scale: int = 1) -> str:
    im = Image.fromarray(np.asarray(a, np.uint8))
    if scale > 1:
        im = im.resize((im.width * scale, im.height * scale), Image.NEAREST)
    buf = io.BytesIO()
    im.save(buf, "PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def token_list(seq, limit: int | None = None) -> list[dict]:
    toks = to_tokens(seq, S.lex, S.text.piece)
    return [{"s": t.token, "type": t.token_type.value, "mod": t.modality} for t in toks[:limit]]


def bits_summary(codec, seg, span) -> dict:
    raw, pay = codec.raw_bits(seg), codec.payload_bits(span)
    return {"positions": int(len(span.ids)), "vocab_tokens": int((span.ids != CONT).sum()),
            "cont_vectors": int((span.ids == CONT).sum()), "raw_bits": raw, "token_bits": pay,
            "ratio": raw / max(pay, 1e-9)}


def clean(o):
    if isinstance(o, dict):
        return {k: clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return None if not math.isfinite(float(o)) else float(o)
    if isinstance(o, np.integer):
        return int(o)
    return o


_CLF: dict = {}


def classifier(task: str):
    """Demo classifier from scripts/train_demo_classifiers.py, reloaded when the checkpoint file changes."""
    path = os.path.join(ROOT, "checkpoints", "demo", f"{task}_vq.pt")
    if not os.path.exists(path):
        return None
    mt = os.path.getmtime(path)
    if task not in _CLF or _CLF[task][0] != mt:
        ck = torch.load(path, map_location="cpu", weights_only=False)
        m = UnifiedTokenTransformer(**ck["cfg"])
        m.load_state_dict(ck["state"])
        _CLF[task] = (mt, m.eval(), ck)
    return _CLF[task][1:]


def classify(x: np.ndarray, task: str) -> dict | None:
    """Image -> VQ token sequence (same serialisation as the training shards) -> class probabilities."""
    c = classifier(task)
    if c is None:
        return None
    m, ck = c
    seq = serialize(Sample("demo", [Segment("image", x)]), {"image": S.img["vq"]})
    with torch.no_grad():
        p = torch.softmax(m(collate([seq], None), task), -1)[0].numpy()
    thr = ck.get("threshold")   # binary tasks: cut-off chosen on validation (same rule as the thesis runs)
    top = int(p[1] >= thr) if thr is not None and len(p) == 2 else int(p.argmax())
    return {"task": task, "predicted": CLASSES[task][top], "threshold": thr, "probs": [{"label": CLASSES[task][k], "p": float(p[k])} for k in np.argsort(-p)],
            "test": {k: ck["test"].get(k) for k in ("auroc", "macro_f1", "acc")}, "n_test": ck["test"].get("n"),
            "seed": ck["seed"], "n_tokens": len(seq)}


def read_image(data: str, task: str | None = None) -> np.ndarray:
    """Base64 / data-URL picture -> centre square -> 28x28 RGB (grey replicated for the chest X-ray task)."""
    im = Image.open(io.BytesIO(base64.b64decode(data.split(",", 1)[-1])))
    im = im.convert("L").convert("RGB") if task == "pneumoniamnist" else im.convert("RGB")
    w, h = im.size
    s = min(w, h)
    im = im.crop(((w - s) // 2, (h - s) // 2, (w - s) // 2 + s, (h - s) // 2 + s)).resize((28, 28), Image.LANCZOS)
    return np.asarray(im, np.uint8)


# ------------------------------------------------------------------ endpoints
def api_info(q):
    return {"image_counts": {s: len(v[0]) for s, v in S.images.items()}, "n_stays": len(S.stays),
            "videos": sorted(S.videos), "image_codebook": S.k_img, "vocab_size": VOCAB_SIZE}


def _image_payload(x: np.ndarray, label: str | None, task: str | None = None):
    seg = Segment("image", x)
    out = {"label": label, "original": png(x, 6), "prediction": classify(x, task) if task else None}
    for rep, codec in S.img.items():
        span = codec.encode(seg)
        rec = codec.decode(span)
        m = image_metrics(x[None], rec[None], ssim=True)
        r = {"recon": png(rec, 6), "psnr": m["psnr_db"], "ssim": m["ssim"], **bits_summary(codec, seg, span)}
        if rep == "vq":
            r["codes"] = codec.codes_from_span(span).tolist()
            seq = serialize(Sample("demo", [seg]), {"image": codec})
            r["tokens"] = token_list(seq)
        out[rep] = r
    return out


def api_image(q):
    sub = q.get("subset", "bloodmnist")
    x, y = S.images[sub]
    i = int(q.get("i", np.random.randint(len(x)))) % len(x)
    out = _image_payload(x[i], CLASSES[sub][int(y[i])], sub)
    out.update(subset=sub, index=i)
    return out


def api_image_upload(body):
    task = body.get("task") if body.get("task") in CLASSES else None
    return _image_payload(read_image(body["png"], task), None, task)


def api_icu(q):
    i = int(q.get("i", np.random.randint(len(S.stays)))) % len(S.stays)
    st = S.stays[i]
    seg = stay_segment(st)
    ev = S.ts["event"]
    g, m = ev.spec.grid(st.obs)
    series = {}
    out = {"index": i, "record": st.record_id, "died": int(st.death), "note": stay_note(st), "reps": {}}
    for rep, codec in S.ts.items():
        span = codec.encode(seg)
        dec = codec.decode(span)
        out["reps"][rep] = bits_summary(codec, seg, span)
        for v in SHOW_VARS:
            j = TS_VARS.index(v)
            series.setdefault(v, {"hours": list(range(ev.spec.steps)),
                                  "original": [float(g[h, j]) if m[h, j] else None for h in range(ev.spec.steps)]})
            dg, dm = dec["grid"], dec["mask"]
            series[v][rep] = [float(dg[h, j]) if (dm is None and m[h, j]) or (dm is not None and dm[h, j]) else None
                              for h in range(ev.spec.steps)]
    out["series"] = series
    seq = serialize(Sample("demo", [seg]), {"timeseries": ev})
    out["tokens"] = token_list(seq, 160)
    out["n_tokens"] = len(seq)
    return out


def api_text(q):
    i = int(q.get("i", np.random.randint(len(S.stays)))) % len(S.stays)
    note = stay_note(S.stays[i])
    span = S.text.encode(Segment("text", note))
    dec = S.text.decode(span)
    pieces = [S.text.piece(RANGES["text"].to_local(int(t))) for t in span.ids]
    return {"index": i, "note": note, "source": template_source("admission"), "pieces": pieces,
            "ids": [int(t) for t in span.ids], "decoded": dec, "exact": dec == note,
            "fields": note_fields(note), "fields_ok": note_fields(dec) == note_fields(note),
            "chars": len(note), "n_tokens": len(span.ids)}


def api_video(q):
    vid = int(q.get("v", sorted(S.videos)[0]))
    v = D.load_video(S.videos[vid], S.vcfg["fps"])
    L = S.vcfg["clip_len"]
    clips = make_clips(v["frames"], L)
    c = int(q.get("clip", len(clips) // 2)) % len(clips)
    seg = Segment("video", {"frames": clips[c], "fps": S.vcfg["fps"]})
    out = {"video": vid, "clip": c, "n_clips": len(clips), "minutes": len(v["frames"]) / S.vcfg["fps"] / 60,
           "phase_now": CATARACT101_PHASES[v["phase"][c * L]] if v["phase"][c * L] >= 0 else "(before first phase)",
           "original": [png(f, 3) for f in clips[c]]}
    for rep, codec in S.vid.items():
        span = codec.encode(seg)
        dec = codec.decode(span)["frames"]
        out[rep] = {"recon": [png(f, 3) for f in dec], **bits_summary(codec, seg, span)}
    ph = Segment("video", {"phase": v["phase"], "fps": S.vcfg["fps"]})
    pspan = S.phase_codec.encode(ph)
    seq = serialize(Sample("demo", [ph]), {"video": S.phase_codec})
    out["phase_tokens"] = token_list(seq)
    out["phase_roundtrip"] = bool((S.phase_codec.decode(pspan)["phase"] == v["phase"]).all())
    n = len(v["phase"])
    out["timeline"] = [{"phase": int(p), "name": CATARACT101_PHASES[p] if p >= 0 else "none",
                        "start": s / n, "end": e / n} for p, s, e in _runs(v["phase"])]
    out["clip_pos"] = c * L / n
    return out


def _runs(a):
    s = 0
    for i in range(1, len(a) + 1):
        if i == len(a) or a[i] != a[s]:
            yield int(a[s]), s, i
            s = i


def api_sequence(q):
    i = int(q.get("i", np.random.randint(len(S.stays)))) % len(S.stays)
    st = S.stays[i]
    smp = Sample(f"physionet2012/{st.record_id}",
                 [Segment("text", stay_note(st), 0.0, template_source("admission")), stay_segment(st)])
    seq = serialize(smp, {"text": S.text, "timeseries": S.ts["event"]})
    toks = to_tokens(seq, S.lex, S.text.piece)
    counts = {}
    for t in toks:
        counts[t.token_type.value] = counts.get(t.token_type.value, 0) + 1
    jsonl = [json.dumps({"id": t.token_id, "token": t.token, "type": t.token_type.value, "modality": t.modality,
                         "t_min": t.timestamp_min, "pos": t.position}, default=str) for t in toks[:12]]
    return {"index": i, "n": len(toks), "spans": [{"modality": sp["modality"], "rep": sp["rep"]}
                                                  for sp in seq.spans],
            "tokens": [{"s": t.token, "type": t.token_type.value} for t in toks[:400]], "type_counts": counts,
            "jsonl": jsonl}


def api_results(q):
    p = os.path.join(ROOT, "reports", "RESULTS.md")
    return {"markdown": open(p).read() if os.path.exists(p) else "(no report yet)"}


def api_mimic_results(q):
    """Aggregate MIMIC-IV validation tables only (reports/mimic/). No patient-level data is served."""
    p = os.path.join(ROOT, "reports", "mimic", "MIMIC_RESULTS.md")
    return {"markdown": open(p).read() if os.path.exists(p) else "(no MIMIC report yet)"}


# ------------------------------------------------------------------ public JSON API (v1)
def v1_index(q):
    return {"name": "MedTok API", "protocol_version": PROTOCOL_VERSION, "vocab_size": VOCAB_SIZE,
            "image_codebook": S.k_img, "classify_tasks": {t: CLASSES[t] for t in CLASSES if classifier(t)},
            "endpoints": {"GET /api/v1": "this index",
                          "POST /api/v1/tokenize/image": "{image: base64 PNG/JPEG, rep?: vq|continuous}",
                          "POST /api/v1/tokenize/text": "{text: string}",
                          "POST /api/v1/classify/image": "{image: base64 PNG/JPEG, task: bloodmnist|pneumoniamnist|dermamnist}"}}


def v1_tokenize_image(body):
    rep = body.get("rep", "vq")
    if rep not in S.img:
        raise ValueError("rep must be 'vq' or 'continuous'")
    x = read_image(body["image"], body.get("task"))
    codec = S.img[rep]
    seg = Segment("image", x)
    seq = serialize(Sample(body.get("sample_id", "api"), [seg]), {"image": codec})
    toks = to_tokens(seq, S.lex, S.text.piece)
    rec = codec.decode(codec.encode(seg))
    m = image_metrics(x[None], rec[None], ssim=True)
    return {"rep": rep, "ids": [int(t) for t in seq.ids], "n_tokens": len(toks),
            "tokens": [{"id": t.token_id, "token": t.token, "type": t.token_type.value} for t in toks],
            "jsonl": [json.dumps({"id": t.token_id, "token": t.token, "type": t.token_type.value,
                                  "modality": t.modality, "t_min": t.timestamp_min, "pos": t.position}, default=str)
                      for t in toks],
            "reconstruction": {"png": png(rec), "psnr_db": m["psnr_db"], "ssim": m["ssim"]},
            **{k: v for k, v in bits_summary(codec, seg, codec.encode(seg)).items() if k != "positions"}}


def v1_tokenize_text(body):
    text = str(body["text"])
    if len(text) > 20000:
        raise ValueError("text longer than 20,000 characters")
    span = S.text.encode(Segment("text", text))
    dec = S.text.decode(span)
    return {"ids": [int(t) for t in span.ids], "pieces": [S.text.piece(RANGES["text"].to_local(int(t))) for t in span.ids],
            "n_tokens": len(span.ids), "decoded": dec, "exact_roundtrip": dec == text}


def v1_classify_image(body):
    task = body.get("task", "bloodmnist")
    if task not in CLASSES:
        raise ValueError(f"task must be one of {list(CLASSES)}")
    out = classify(read_image(body["image"], task), task)
    if out is None:
        raise ValueError(f"no demo classifier for {task}; run scripts/train_demo_classifiers.py")
    return {**out, "disclaimer": "research demo on 28x28 MedMNIST-style images; not a medical device"}


GET = {"/api/v1": v1_index, "/api/info": api_info, "/api/image": api_image, "/api/icu": api_icu, "/api/text": api_text,
       "/api/video": api_video, "/api/sequence": api_sequence, "/api/results": api_results,
       "/api/mimic_results": api_mimic_results}
POST = {"/api/image_upload": api_image_upload, "/api/v1/tokenize/image": v1_tokenize_image,
        "/api/v1/tokenize/text": v1_tokenize_text, "/api/v1/classify/image": v1_classify_image}


class Handler(BaseHTTPRequestHandler):
    def _send(self, code, body: bytes, ctype: str):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _json(self, fn, arg):
        try:
            self._send(200, json.dumps(clean(fn(arg))).encode(), "application/json")
        except Exception as e:                                   # show the error in the page, keep serving
            self._send(500, json.dumps({"error": f"{type(e).__name__}: {e}"}).encode(), "application/json")

    def do_GET(self):
        u = urlparse(self.path)
        if u.path in GET:
            return self._json(GET[u.path], {k: v[0] for k, v in parse_qs(u.query).items()})
        if u.path in ("/", "/index.html"):
            with open(os.path.join(ROOT, "demo", "index.html"), "rb") as f:
                return self._send(200, f.read(), "text/html; charset=utf-8")
        if u.path.startswith("/media/"):                         # walkthrough video (demo/record_walkthrough.js)
            name = os.path.basename(u.path)
            f = os.path.join(ROOT, "demo", "media", name)
            types = {".mp4": "video/mp4", ".webm": "video/webm"}
            if os.path.splitext(name)[1] in types and os.path.isfile(f):
                with open(f, "rb") as fh:
                    return self._send(200, fh.read(), types[os.path.splitext(name)[1]])
        self._send(404, b"not found", "text/plain")

    def do_POST(self):
        u = urlparse(self.path)
        if u.path not in POST:
            return self._send(404, b"not found", "text/plain")
        n = int(self.headers.get("Content-Length", 0))
        if n > 10_000_000:
            return self._send(413, b"too large", "text/plain")
        self._json(POST[u.path], json.loads(self.rfile.read(n)))

    def log_message(self, *a):
        pass


def main():
    global S
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8093)
    ap.add_argument("--host", default="0.0.0.0")
    a = ap.parse_args()
    S = State()
    print(f"MedTok demo on http://{a.host}:{a.port}", flush=True)
    ThreadingHTTPServer((a.host, a.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
