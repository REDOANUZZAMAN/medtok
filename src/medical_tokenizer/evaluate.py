"""Evaluate a trained tokenizer run on its test split.

    python -m medical_tokenizer.evaluate --run experiments/image_vq/latest
    python -m medical_tokenizer.evaluate --interfaces      # text + point-cloud interface checks

Writes <run>/eval/metrics.json + metrics.csv (reconstruction, compression, sequence
length, token utilisation), code_counts.npy and recon_examples.npz.
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import torch

from .data import datasets as D
from .evaluation.compression import compression_summary, transformer_input_bytes
from .evaluation.reconstruction import exact_match_rate, image_metrics, masked_ts_metrics
from .protocol.schema import Segment
from .protocol.vocab import CONT, RANGES
from .quantization.stats import usage_stats
from .toolkit import load_tokenizer
from .utils.experiment import Run, load_run_config, resolve_run


class EvalWriter:
    def __init__(self, run_dir: str):
        self.dir = os.path.join(run_dir, "eval")
        os.makedirs(self.dir, exist_ok=True)

    def path(self, f):
        return os.path.join(self.dir, f)

    def save(self, metrics: dict):
        import json
        import csv
        from .protocol.schema import json_default
        from .utils.experiment import flatten
        with open(self.path("metrics.json"), "w") as f:
            json.dump(metrics, f, indent=1, default=json_default)
        with open(self.path("metrics.csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["metric", "value"])
            for k, v in flatten(metrics).items():
                w.writerow([k, v])


def _span_accounting(codec, segs, spans, raw_elements, d_model=128):
    disc = [s.ids[s.ids != CONT] for s in spans]
    cont_vals = [0 if s.cont is None else s.cont.size for s in spans]
    n_tok = [len(s.ids) for s in spans]
    out = compression_summary([codec.raw_bits(g) for g in segs], [codec.payload_bits(s) for s in spans], n_tok,
                              raw_elements, disc, cont_vals, spans)
    out["transformer_input_kib_per_sample"] = transformer_input_bytes(np.mean(n_tok), d_model) / 1024
    out["continuous_values_per_sample"] = float(np.mean(cont_vals))
    return out


# ------------------------------------------------------------------ image
def eval_image(run_dir: str, cfg: dict, w: EvalWriter) -> dict:
    codec = load_tokenizer(run_dir)
    d = cfg["data"]
    res, examples, all_counts = {}, {}, None
    for subset in cfg.get("evaluate", {}).get("subsets", d["subsets"]):
        x, _, _, _ = D.image_split(d, "test", [subset])
        segs = [Segment("image", xi) for xi in x]
        if codec.rep == "vq":
            codes = codec.codes(x)
            rec = codec.decode_codes(codes)
            spans = [codec.span_from_codes(c) for c in codes]
            counts = np.bincount(codes.ravel(), minlength=codec.model.n_codes)
            all_counts = counts if all_counts is None else all_counts + counts
            util = usage_stats(counts)
        else:
            z = codec.latents(x).numpy()
            rec = codec.decode_latents(z)
            spans = [codec.span_from_latent(zz) for zz in z]
            util = None
        r = {"reconstruction": image_metrics(x, rec),
             "compression": _span_accounting(codec, segs, spans, [xi.shape[0] * xi.shape[1] for xi in x])}
        if util:
            r["utilization"] = util
        res[subset] = r
        examples[subset] = np.stack([x[:16], rec[:16]])
        print(subset, {k: round(v, 3) for k, v in r["reconstruction"].items()},
              "ratio", round(r["compression"]["compression_ratio"], 2), flush=True)
    if all_counts is not None:
        np.save(w.path("code_counts.npy"), all_counts)
        res["all_subsets_utilization"] = usage_stats(all_counts)
    np.savez_compressed(w.path("recon_examples.npz"), **examples)
    return res


# ------------------------------------------------------------------ time-series
def eval_timeseries(run_dir: str, cfg: dict, w: EvalWriter) -> dict:
    import json
    from .data.physionet import Normalizer, TS_VARS
    from .tokenizers.timeseries.codec import stay_segment
    codec = load_tokenizer(run_dir)
    with open(os.path.join(run_dir, "normalizer.json")) as f:
        norm = Normalizer.from_json(json.load(f))
    d = cfg["data"]
    test = [s for s in D.stays(d) if s.split == "test"]
    segs = [stay_segment(s) for s in test]
    spec = codec.spec
    grids = [spec.grid(s.obs) for s in test]
    mk = np.stack([m for _, m in grids])
    z_true = np.stack([norm.grid(g, m) for g, m in grids])

    def to_z(rec_grids):
        return np.stack([norm.grid(r, mk[i]) for i, r in enumerate(rec_grids)])

    spans = codec.encode_batch(segs)
    out = {}
    if codec.rep == "event":
        dec = [codec.decode(sp) for sp in spans]
        rec = [x["grid"] for x in dec]
        # exactness w.r.t. the binned grid (what the event codec promises to preserve)
        binned = []
        for (g, m) in grids:
            b = np.zeros_like(g)
            for j, v in enumerate(spec.variables):
                if m[:, j].any():
                    b[m[:, j], j] = codec.binner.value(v, codec.binner.bin(v, g[m[:, j], j]))
            binned.append(b)
        out["binned_values_exact_rate"] = float(np.mean([np.allclose(r[m], b[m]) for r, b, (_, m)
                                                         in zip(rec, binned, grids)]))
        vb = [np.array([RANGES["vbin"].to_local(int(i)) for i in sp.ids if int(i) in RANGES["vbin"]]) for sp in spans]
        out["utilization"] = {"value_bins": usage_stats(np.bincount(np.concatenate(vb), minlength=codec.lex.n_value_bins))}
    elif codec.rep == "vq":
        _, codes = zip(*[codec.codes_from_span(sp) for sp in spans])
        codes = np.stack(codes)
        with torch.no_grad():
            rec = codec.decode_grid(codec.model.quantizer.lookup(torch.as_tensor(codes)))
        counts = np.bincount(codes.ravel(), minlength=codec.model.n_codes)
        np.save(w.path("code_counts.npy"), counts)
        out["utilization"] = usage_stats(counts)
    else:
        z = torch.as_tensor(np.stack([sp.cont.astype(np.float32) for sp in spans])).view(len(spans), codec.W, codec.G, -1)
        rec = codec.decode_grid(z)
    out["reconstruction"] = masked_ts_metrics(z_true, to_z(rec), mk, TS_VARS)
    out["compression"] = _span_accounting(codec, segs, spans, [len(s.obs) for s in test])
    # information already lost by resampling the irregular record onto the grid (same for every representation)
    rv, gv = [], []
    for s, (g, m) in zip(test, grids):
        h = np.clip((s.obs[:, 0] // spec.resolution_min).astype(int), 0, spec.steps - 1)
        v = s.obs[:, 1].astype(int)
        rv.append(norm.transform(s.obs[:, 2], v))
        gv.append(norm.transform(g[h, v], v))
    out["gridding_mae_z"] = float(np.abs(np.concatenate(rv) - np.concatenate(gv)).mean())
    np.savez_compressed(w.path("recon_examples.npz"), z_true=z_true[:8], mask=mk[:8], z_rec=to_z(rec[:8]) if len(rec) else None)
    print({k: round(v, 4) for k, v in out["reconstruction"].items() if not isinstance(v, dict)},
          "tokens", round(out["compression"]["tokens_per_sample_mean"], 1),
          "ratio", round(out["compression"]["compression_ratio"], 2), flush=True)
    return out


# ------------------------------------------------------------------ video
def eval_video(run_dir: str, cfg: dict, w: EvalWriter) -> dict:
    from .tokenizers.video.codec import make_clips
    codec = load_tokenizer(run_dir)
    d = cfg["data"]
    files, smap = D.video_files(d), D.video_split_map(d)
    test_vids = [v for v in files if smap[v] == "test"]
    ssim_every = cfg.get("evaluate", {}).get("ssim_every_nth_frame", 8)
    xs, rs, spans, segs, n_frames, counts = [], [], [], [], [], None
    for v in test_vids:
        vid = D.load_video(files[v], d["fps"])
        clips = make_clips(vid["frames"], d["clip_len"])
        seg = Segment("video", {"frames": clips.reshape(-1, *clips.shape[2:]), "fps": d["fps"]})
        if codec.rep == "vq":
            c = codec.codes(clips)
            sp = codec.span_from_codes(c, d["fps"])
            with torch.no_grad():
                r = codec.decode_latents(codec.model.quantizer.lookup(torch.as_tensor(c)))
            k = np.bincount(c.ravel(), minlength=codec.model.n_codes)
            counts = k if counts is None else counts + k
        elif codec.rep == "continuous":
            z = codec.clip_latents(clips)
            sp = codec.span_from_latent(z.numpy(), d["fps"])
            r = codec.decode_latents(z)
        else:
            st = codec.states(clips)
            sp = codec.span_from_states(st, d["fps"])
            r = codec.decode_latents(torch.as_tensor(codec.prototypes[st]))
            k = np.bincount(st, minlength=codec.n_states)
            counts = k if counts is None else counts + k
        x = clips.reshape(-1, *clips.shape[2:])
        xs.append(x[::ssim_every])
        rs.append(r[::ssim_every])
        spans.append(sp)
        segs.append(seg)
        n_frames.append(len(x))
    x_all, r_all = np.concatenate(xs), np.concatenate(rs)
    out = {"test_videos": len(test_vids), "frames_evaluated": int(len(x_all)),
           "reconstruction": image_metrics(x_all, r_all),
           "compression": _span_accounting(codec, segs, spans,
                                           [n * d.get("size", segs[0].data["frames"].shape[1]) ** 2 for n in n_frames])}
    minutes = np.array(n_frames) / d["fps"] / 60
    out["compression"]["tokens_per_minute"] = float(np.sum([len(s.ids) for s in spans]) / minutes.sum())
    if counts is not None:
        np.save(w.path("code_counts.npy"), counts)
        out["utilization"] = usage_stats(counts)
    if codec.rep == "event":
        out["annotated_phase_events"] = eval_annotated_phase_events(d, test_vids, files)
    np.savez_compressed(w.path("recon_examples.npz"), x=x_all[:32], r=r_all[:32])
    print({k: round(v, 3) for k, v in out["reconstruction"].items()},
          "tokens/min", round(out["compression"]["tokens_per_minute"], 1), flush=True)
    return out


def eval_annotated_phase_events(d: dict, vids, files) -> dict:
    """Reference numbers for annotation-derived event tokens (exact codec; uses labels, so never a model input)."""
    from .data.video import CATARACT101_PHASES
    from .tokenizers.video.codec import AnnotatedPhaseEventCodec
    from .toolkit import build_lexicon
    codec = AnnotatedPhaseEventCodec(build_lexicon(), "cataract101", CATARACT101_PHASES)
    segs = [Segment("video", {"phase": D.load_video(files[v], d["fps"])["phase"], "fps": d["fps"]}) for v in vids]
    spans = [codec.encode(s) for s in segs]
    exact = exact_match_rate((codec.decode(sp)["phase"], s.data["phase"]) for sp, s in zip(spans, segs))
    out = compression_summary([codec.raw_bits(s) for s in segs], [codec.payload_bits(s) for s in spans],
                              [len(s.ids) for s in spans], [len(s.data["phase"]) for s in segs], spans=spans)
    out["exact_roundtrip_rate"] = exact
    minutes = sum(len(s.data["phase"]) for s in segs) / d["fps"] / 60
    out["tokens_per_minute"] = float(sum(len(s.ids) for s in spans) / minutes)
    return out


# ------------------------------------------------------------------ interfaces (text, point cloud)
def eval_interfaces(out_root: str | None = None) -> str:
    from .data.datasets import stays
    from .data.notes import TEMPLATE_VERSION, note_fields, stay_note
    from .toolkit import load_text_tokenizer
    from .tokenizers.pointcloud.codec import PointCloudTokenizer, chamfer, synthetic_surface
    cfg = {"name": "interfaces", "seed": 0, "data": {"root": "data/raw/physionet2012"}}
    run = Run(cfg, root=out_root)
    text = load_text_tokenizer()
    notes = [stay_note(s) for s in stays(cfg["data"]) if s.split == "test"]
    segs = [Segment("text", n) for n in notes]
    spans = [text.encode(s) for s in segs]
    decoded = [text.decode(sp) for sp in spans]
    res = {"text": {"exact_roundtrip_rate": exact_match_rate(zip(decoded, notes)),
                    "field_preservation_rate": float(np.mean([note_fields(a) == note_fields(b)
                                                              for a, b in zip(decoded, notes)])),
                    "fields_per_note": float(np.mean([len(note_fields(n)) for n in notes])),
                    "chars_per_token": float(np.mean([len(n) / len(sp.ids) for n, sp in zip(notes, spans)])),
                    "vocab_size": text.alphabet_size(),
                    "compression": compression_summary([text.raw_bits(s) for s in segs],
                                                       [text.payload_bits(s) for s in spans],
                                                       [len(s.ids) for s in spans], [len(n) for n in notes],
                                                       spans=spans),
                    "source": "SYNTHETIC templated notes rendered from structured admission fields (not clinical free text)",
                    "template_version": TEMPLATE_VERSION,
                    "acceptance": "lossless round-trip, field preservation, protocol compatibility only"}}
    train_c = [synthetic_surface(2048, seed=i) for i in range(40)]
    test_c = [synthetic_surface(2048, seed=1000 + i) for i in range(20)]
    pc = {}
    for res_ in (4, 8, 16):
        tok = PointCloudTokenizer(res=res_).fit(train_c, k=256)
        segs = [Segment("pointcloud", c) for c in test_c]
        sp = [tok.encode(s) for s in segs]
        cd = [chamfer(c, tok.decode(s)) for c, s in zip(test_c, sp)]
        extent = np.mean([np.ptp(c, 0).max() for c in test_c])
        pc[f"res{res_}"] = {"chamfer_mm": float(np.mean(cd)), "chamfer_rel_extent": float(np.mean(cd) / extent),
                            "compression": compression_summary([tok.raw_bits(s) for s in segs],
                                                               [tok.payload_bits(s) for s in sp],
                                                               [len(s.ids) for s in sp], [len(c) for c in test_c],
                                                               spans=sp)}
    res["pointcloud_synthetic"] = pc | {"note": "synthetic surfaces only; interface-level, no learned encoder"}
    run.save_metrics(res)
    print("interfaces:", run.dir)
    return run.dir


EVALUATORS = {"image": eval_image, "timeseries": eval_timeseries, "video": eval_video}


def evaluate(run: str) -> dict:
    run_dir = resolve_run(run)
    cfg = load_run_config(run_dir)
    torch.set_num_threads(int(cfg.get("threads", os.environ.get("MEDTOK_THREADS", os.cpu_count()))))
    w = EvalWriter(run_dir)
    res = {"run": run_dir, "name": cfg.get("name"), "modality": cfg["modality"],
           "representation": cfg["representation"], **EVALUATORS[cfg["modality"]](run_dir, cfg, w)}
    w.save(res)
    print("wrote", w.path("metrics.json"))
    return res


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Evaluate tokenizer runs (reconstruction, compression, utilisation).")
    ap.add_argument("--run", action="append", default=[], help="run dir or experiment dir (uses latest)")
    ap.add_argument("--interfaces", action="store_true", help="evaluate the text / point-cloud interfaces")
    args = ap.parse_args(argv)
    for r in args.run:
        evaluate(r)
    if args.interfaces:
        eval_interfaces()
    if not args.run and not args.interfaces:
        ap.error("nothing to evaluate: pass --run and/or --interfaces")


if __name__ == "__main__":
    main()
