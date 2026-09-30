"""Compression ratio, reconstruction error and token utilisation per modality x representation.

All numbers are computed on the TEST split. Writes artifacts/results/compression.json.

Bit accounting (payload span only; delimiters are shared protocol overhead and
reported separately as sequence length):
  discrete   n_tokens * log2(alphabet the codec can emit)       -> "fixed-length"
             n_tokens * H(empirical token distribution)          -> "entropy-coded"
  continuous n_vectors * dim * 16 (fp16)
"""
import json
import os
import pickle
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from medtok.codecs.timeseries import stay_segment  # noqa: E402
from medtok.data.medmnist_data import load_subset  # noqa: E402
from medtok.data.notes import stay_note  # noqa: E402
from medtok.data.physionet import TS_VARS, to_grid  # noqa: E402
from medtok.protocol.schema import Segment  # noqa: E402
from medtok.protocol.vocab import RANGES  # noqa: E402
from medtok.quant.vq import usage_stats  # noqa: E402
from medtok.toolkit import Toolkit  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")
OUT = os.path.join(ROOT, "artifacts", "results")
os.makedirs(OUT, exist_ok=True)
tk = Toolkit(os.path.join(ROOT, "artifacts"))
torch.set_num_threads(os.cpu_count())
res = {}


def entropy_bits(token_lists):
    ids = np.concatenate([np.asarray(t) for t in token_lists])
    _, c = np.unique(ids, return_counts=True)
    p = c / c.sum()
    return float(-(p * np.log2(p)).sum())


def summarise(name, raw_bits, tok_counts, fixed_bits, extra):
    raw, fixed = float(np.sum(raw_bits)), float(np.sum(fixed_bits))
    r = {"n_samples": len(raw_bits), "tokens_per_sample": float(np.mean(tok_counts)),
         "raw_bits_per_sample": raw / len(raw_bits), "token_bits_per_sample": fixed / len(raw_bits),
         "compression_ratio": raw / fixed, **extra}
    res[name] = r
    print(name, json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in r.items()
                            if not isinstance(v, (list, dict))}), flush=True)


# ------------------------------------------------------------------ images
for subset in ("bloodmnist", "pneumoniamnist"):
    x, _ = load_subset(os.path.join(ROOT, "data", "raw", "medmnist"), subset)["test"]
    vq, cont = tk.codecs[("image", "vq")], tk.codecs[("image", "cont")]
    with torch.no_grad():
        z = vq.latents(x)
        codes = tk.image_model.vq.codes(z)
        rec_vq = tk.image_model.to_uint8(tk.image_model.decode_codes(codes)).numpy()
        rec_ct = tk.image_model.to_uint8(tk.image_model.decode_latent(z)).numpy()
    codes = codes.numpy()

    def img_err(rec):
        mse = ((rec.astype(np.float64) - x) ** 2).mean(axis=(1, 2, 3))
        return {"mse_uint8": float(mse.mean()), "psnr_db": float(np.mean(10 * np.log10(255 ** 2 / np.maximum(mse, 1e-9)))),
                "mae_uint8": float(np.abs(rec.astype(np.float64) - x).mean())}

    raw = np.full(len(x), x[0].size * 8.0)
    spans = [vq.span_from_codes(c) for c in codes[:200]]
    n_tok = len(spans[0].ids)
    u = usage_stats([codes], tk.image_model.vq.n_codes)
    H = u["entropy_bits"]
    summarise(f"image/{subset}/vq", raw, [n_tok] * len(x), [vq.span_bits(spans[0])] * len(x),
              {**img_err(rec_vq), "codebook": {k: v for k, v in u.items() if k != "counts"},
               "entropy_coded_ratio": float(raw[0] / (49 * H)), "code_counts": u["counts"]})
    cs = cont.span_from_latent(z[0].numpy())
    summarise(f"image/{subset}/cont", raw, [len(cs.ids)] * len(x), [cont.span_bits(cs)] * len(x), img_err(rec_ct))
    np.save(os.path.join(OUT, f"recon_{subset}.npy"), np.stack([x[:8], rec_vq[:8], rec_ct[:8]]))

# ------------------------------------------------------------- time-series
stays = pickle.load(open(os.path.join(ROOT, "data", "processed", "stays.pkl"), "rb"))
test = [s for s in stays if s.split == "test"]
grids = [to_grid(s) for s in test]
norm = tk.norm
zg = np.stack([norm.grid(g, m) for g, m in grids])
mk = np.stack([m for _, m in grids])
std = norm.std


def ts_err(rec_grids):
    """Error on observed cells of the hourly grid, in normalised (z) units and as per-variable nMAE."""
    rz = np.stack([norm.grid(r, mk[i]) for i, r in enumerate(rec_grids)])
    d = np.abs(rz - zg)[mk]
    var = np.broadcast_to(np.arange(len(TS_VARS)), mk.shape)[mk]
    per_var = {TS_VARS[v]: float(d[var == v].mean()) for v in range(len(TS_VARS)) if (var == v).any()}
    return {"mae_z": float(d.mean()), "rmse_z": float(np.sqrt((d ** 2).mean())),
            "mae_z_per_var": per_var}


# raw irregular obs -> hourly grid: information lost before any tokenizer
raw_vals, grid_vals = [], []
for s, (g, m) in zip(test, grids):
    h = np.clip((s.obs[:, 0] // 60).astype(int), 0, 47)
    v = s.obs[:, 1].astype(int)
    raw_vals.append(norm.transform(s.obs[:, 2], v))
    grid_vals.append(norm.transform(g[h, v], v))
res["timeseries/gridding"] = {"mae_z_raw_to_hourly": float(np.abs(np.concatenate(raw_vals) - np.concatenate(grid_vals)).mean()),
                              "obs_per_stay": float(np.mean([len(s.obs) for s in test])),
                              "observed_cells_per_stay": float(mk.sum((1, 2)).mean())}
print("timeseries/gridding", res["timeseries/gridding"])
raw = np.array([tk.codecs[("timeseries", "event")].raw_bits(stay_segment(s)) for s in test])

ev = tk.codecs[("timeseries", "event")]
ev_spans = [ev.encode(stay_segment(s)) for s in test]
ev_rec = [ev.decode(sp)["grid"] for sp in ev_spans]
ids_ev = [sp.ids for sp in ev_spans]
vb = [np.array([RANGES["vbin"].to_local(int(i)) for i in ids if int(i) in RANGES["vbin"]]) for ids in ids_ev]
ub = usage_stats(vb, tk.lex.n_value_bins)
He = entropy_bits(ids_ev)
summarise("timeseries/event", raw, [len(i) for i in ids_ev], [ev.span_bits(sp) for sp in ev_spans],
          {**ts_err(ev_rec), "value_bins": {k: v for k, v in ub.items() if k != "counts"},
           "entropy_coded_ratio": float(raw.sum() / sum(len(i) * He for i in ids_ev))})

vqc = tk.codecs[("timeseries", "vq")]
codes = vqc.codes([s.obs for s in test])
with torch.no_grad():
    rec = tk.ts_model.decode_codes(torch.as_tensor(codes)).numpy()
    zc = vqc.latents([s.obs for s in test])
    rec_c = tk.ts_model.decode_latent(zc).numpy()
var = np.broadcast_to(np.arange(len(TS_VARS)), rec.shape[1:])
vq_rec = [norm.inverse(r, var) for r in rec]
ct_rec = [norm.inverse(r, var) for r in rec_c]
vq_spans = [vqc.span_from_codes(s.static, c) for s, c in zip(test, codes)]
u = usage_stats([codes], tk.ts_model.vq.n_codes)
summarise("timeseries/vq", raw, [len(sp.ids) for sp in vq_spans], [vqc.span_bits(sp) for sp in vq_spans],
          {**ts_err(vq_rec), "codebook": {k: v for k, v in u.items() if k != "counts"},
           "entropy_coded_ratio": float(raw.sum() / (codes.size * u["entropy_bits"])), "code_counts": u["counts"]})
ctc = tk.codecs[("timeseries", "cont")]
ct_spans = [ctc.span_from_latent(s.static, zz) for s, zz in zip(test, zc.numpy())]
summarise("timeseries/cont", raw, [len(sp.ids) for sp in ct_spans], [ctc.span_bits(sp) for sp in ct_spans], ts_err(ct_rec))
np.savez(os.path.join(OUT, "recon_ts.npz"), x=zg[:4], m=mk[:4],
         ev=np.stack([norm.grid(r, np.ones_like(mk[0])) for r in ev_rec[:4]]),
         vq=rec[:4], cont=rec_c[:4])

# -------------------------------------------------------------------- text
notes = [stay_note(s) for s in test]
spans = [tk.text.encode(Segment("text", n)) for n in notes]
exact = float(np.mean([tk.text.decode(sp) == n for sp, n in zip(spans, notes)]))
raw = np.array([len(n.encode()) * 8 for n in notes])
summarise("text/bpe", raw, [len(sp.ids) for sp in spans], [tk.text.span_bits(sp) for sp in spans],
          {"exact_roundtrip_rate": exact, "chars_per_token": float(np.mean([len(n) / len(sp.ids) for n, sp in zip(notes, spans)])),
           "vocab_used": int(len(np.unique(np.concatenate([sp.ids for sp in spans])))),
           "vocab_size": tk.text.alphabet_size()})

json.dump(res, open(os.path.join(OUT, "compression.json"), "w"), indent=1)
print("wrote", os.path.join(OUT, "compression.json"))
