"""Exercise the interface-level modalities (surgical video, 3D point clouds) on
synthetic data in the real dataset formats, and serialise one sample that mixes
all five modalities into a single TokenSequence.

Writes artifacts/results/interfaces.json and data/samples_demo/ (synthetic,
safe to commit).
"""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np  # noqa: E402

from medtok.codecs.pointcloud import PointCloudVoxelCodec, chamfer, synthetic_organ_surface  # noqa: E402
from medtok.codecs.timeseries import stay_segment  # noqa: E402
from medtok.codecs.video import SurgicalEventCodec, VideoFrameVQCodec, synthetic_cholec80  # noqa: E402
from medtok.data.medmnist_data import load_subset  # noqa: E402
from medtok.protocol.schema import Sample, Segment  # noqa: E402
from medtok.protocol.serialize import deserialize, serialize, validate  # noqa: E402
from medtok.toolkit import Toolkit  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")
DEMO = os.path.join(ROOT, "data", "samples_demo")
os.makedirs(DEMO, exist_ok=True)
tk = Toolkit(os.path.join(ROOT, "artifacts"))
res = {}

# ---- surgical video: event tokens (exact) on 20 synthetic 30-min Cholec80 annotations
ev = SurgicalEventCodec(tk.lex)
exact, ratios, lens = [], [], []
for seed in range(20):
    ann = synthetic_cholec80(1800, seed)
    seg = Segment("video", ann)
    span = ev.encode(seg)
    out = ev.decode(span)
    exact.append(bool((out["phase"] == ann["phase"]).all() and (out["tools"] == ann["tools"]).all()))
    ratios.append(ev.raw_bits(seg) / ev.span_bits(span))
    lens.append(len(span.ids))
res["video/event"] = {"n": 20, "exact_roundtrip_rate": float(np.mean(exact)), "compression_ratio": float(np.mean(ratios)),
                      "tokens_per_30min": float(np.mean(lens))}

# ---- surgical video: frame VQ using the image codebook on a synthetic "clip" (MedMNIST frames drifting)
x, _ = load_subset(os.path.join(ROOT, "data", "raw", "medmnist"), "dermamnist")["test"]
clip = np.stack([np.roll(x[0], k, axis=1) for k in range(8)])
fv = VideoFrameVQCodec(tk.image_model)
span = fv.encode(Segment("video", clip, meta={"fps": 1}))
rec = fv.decode(span)
mse = float(((rec.astype(float) - clip) ** 2).mean())
res["video/frame_vq"] = {"frames": 8, "tokens": len(span.ids), "psnr_db": float(10 * np.log10(255 ** 2 / mse)),
                         "compression_ratio": fv.raw_bits(Segment("video", clip)) / fv.span_bits(span)}

# ---- point clouds: voxel + k-means codebook
train = [synthetic_organ_surface(2048, seed=i) for i in range(40)]
test = [synthetic_organ_surface(2048, seed=1000 + i) for i in range(10)]
pc = PointCloudVoxelCodec(res=8).fit(train, k=256)
np.save(os.path.join(ROOT, "artifacts", "pc_codebook.npy"), pc.codebook)
cds, rat, ntok, ext = [], [], [], []
for c in test:
    seg = Segment("pointcloud", c)
    sp = pc.encode(seg)
    r = pc.decode(sp)
    cds.append(chamfer(c, r))
    ext.append(float((c.max(0) - c.min(0)).max()))
    rat.append(pc.raw_bits(seg) / pc.span_bits(sp))
    ntok.append(len(sp.ids))
res["pointcloud/voxel_vq"] = {"n": 10, "points": 2048, "chamfer_mm": float(np.mean(cds)),
                              "chamfer_rel_extent": float(np.mean(np.array(cds) / np.array(ext))),
                              "compression_ratio": float(np.mean(rat)), "tokens": float(np.mean(ntok))}

# ---- one five-modality sample through the full protocol
from medtok.data.physionet import TS_IDX, Stay  # noqa: E402

rng = np.random.default_rng(0)  # synthetic stay: this directory is committed, so no real patient data
obs = sorted((h * 60 + int(rng.integers(0, 60)), TS_IDX[v], mu + rng.normal(0, sd))
             for h in range(48) for v, mu, sd in (("HR", 85, 8), ("MAP", 78, 6), ("Temp", 37.2, 0.4)))
stay = Stay("synthetic", {"Age": 64.0, "Gender": 1.0, "Height": 172.0, "ICUType": 4.0},
            np.array(obs, np.float32), 0)
ann = synthetic_cholec80(600, 7)
smp = Sample("demo/five-modality", [
    Segment("text", "Pre-operative note: 64-year-old admitted for laparoscopic cholecystectomy.", 0.0),
    Segment("image", x[1], 30.0, "dermamnist"),
    Segment("video", ann, 120.0, "synthetic-cholec80"),
    Segment("pointcloud", test[0], 125.0, "synthetic-scared"),
    Segment("timeseries", stay_segment(stay).data, 180.0, "synthetic-physionet-format"),
], labels={"demo": 1})
codecs = tk.codec_set("vq", "event") | {"pointcloud": pc}
seq = serialize(smp, codecs)
validate(seq)
dec = deserialize(seq, codecs)
seq.save(os.path.join(DEMO, "five_modality.npz"))
readable = [tk.lex.id_to_str(int(i), text_decoder=tk.text.piece) for i in seq.ids]
with open(os.path.join(DEMO, "five_modality.tokens.txt"), "w") as f:
    f.write(" ".join(readable) + "\n")
res["five_modality_sample"] = {"length": len(seq), "spans": [{k: v for k, v in s.items() if k != "meta"} for s in seq.spans],
                               "text_exact": dec[0][1] == smp.segments[0].data,
                               "video_exact": bool((dec[2][1]["phase"] == ann["phase"]).all())}
print(json.dumps(res, indent=1))
json.dump(res, open(os.path.join(ROOT, "artifacts", "results", "interfaces.json"), "w"), indent=1)
