"""MIMIC-IV linked multimodal validation of the tokenizers and the unified protocol.

python -m medical_tokenizer.mimic_validate --config configs/mimic_validation.yaml [--set image.train.epochs=2]

Real, linked data from one hospital: per patient one chest X-ray study (MIMIC-CXR-JPG), its free-text
radiology report, the 12-lead ECGs (MIMIC-IV-ECG) and the lab results (MIMIC-IV hosp) of the 7 days
before the radiograph. Everything runs on CPU on this machine. The data is credentialed, so it is not
sent to a cloud GPU, and only aggregate numbers leave experiments/ (reports/mimic/).

Sections (each writes one block of metrics.json):
  cohort      counts, label prevalence, missingness
  image       MedMNIST tokenizer applied zero-shot vs the same design trained on MIMIC CXR (28 / 64 px)
  ecg         12-lead ECG through the existing window VQ tokenizer (new signal type, no code changes)
  labs        lab events through the existing event tokenizer at 1 h / 6 h / 24 h resolution
  text        real reports: synthetic-template BPE vs BPE retrained on MIMIC reports; exact round-trip
  unified     one protocol sequence per patient (4 sources, 2 sharing the time-series modality),
              shard save -> load -> decode, structural validation, token budget per modality
  downstream  1-year mortality from the tokens: bag-of-tokens logistic regression per modality and fused,
              repeated stratified 5-fold cross-validation (600 patients is too few for a fixed test split)
"""
from __future__ import annotations

import os
import time
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F

from .codecs.base import MultimodalCodec, RoutedCodec
from .data import mimic as M
from .data.datasets import abspath
from .data.physionet import STATIC_VARS as PN_STATIC
from .data.physionet import TS_VARS, Normalizer
from .encoders.image import ImageAutoencoder
from .encoders.timeseries import TimeSeriesWindowAutoencoder
from .evaluation.compression import compression_summary
from .evaluation.reconstruction import image_metrics
from .protocol.schema import Sample, Segment
from .protocol.serialize import split_spans, validate
from .protocol.storage import load_shard, save_shard
from .protocol.vocab import RANGES, Lexicon
from .quantization.stats import CodeUsageTracker, usage_stats
from .tokenizers.image.codec import ImageVQCodec
from .tokenizers.text.codec import ClinicalTextTokenizer
from .tokenizers.timeseries.codec import GridSpec, TSEventCodec, TSVQCodec, ValueBinner
from .toolkit import annotated_events, load_tokenizer
from .trainers import DEV, _epoch_loop
from .utils.config import arg_parser, load_config
from .utils.experiment import Run

SRC = {"labs": "mimic-iv:labevents", "ecg": "mimic-iv-ecg", "image": "mimic-cxr-jpg", "text": "mimic-cxr:report"}
ECG_VARS = [f"ecg:{lead}" for lead in M.ECG_LEADS]


def stays_view(pats, attr="labs"):
    """Patients as the Stay-like objects ValueBinner / Normalizer fit on (obs, static, split)."""
    return [SimpleNamespace(obs=getattr(p, attr), static=p.static, split=p.split) for p in pats]


def mean_sd(a) -> dict:
    a = np.asarray(a, np.float64)
    return {"mean": float(a.mean()), "sd": float(a.std(ddof=1)) if len(a) > 1 else 0.0}


# ------------------------------------------------------------------ cohort
def cohort_section(pats, lab_vars) -> dict:
    n = len(pats)
    split = {s: sum(p.split == s for p in pats) for s in ("train", "val", "test")}
    return {
        "patients": n, "split": split,
        "deaths_1y": int(sum(p.death for p in pats)), "prevalence_1y": float(np.mean([p.death for p in pats])),
        "deaths_1y_by_split": {s: int(sum(p.death for p in pats if p.split == s)) for s in split},
        "images": int(sum(len(p.images) for p in pats)),
        "images_per_patient": mean_sd([len(p.images) for p in pats]),
        "patients_without_frontal_view": int(sum(p.images[0][1] not in M.FRONTAL for p in pats)),
        "ecgs": int(sum(len(p.ecgs) for p in pats)), "ecgs_per_patient": mean_sd([len(p.ecgs) for p in pats]),
        "lab_variables": len(lab_vars),
        "lab_results_per_patient_median": float(np.median([len(p.labs) for p in pats])),
        "patients_without_labs": int(sum(len(p.labs) == 0 for p in pats)),
        "reports": n, "report_chars_median": float(np.median([len(M.read_report(p.report)) for p in pats])),
        "age_median": float(np.median([p.static["Age"] for p in pats])),
        "male_fraction": float(np.mean([p.static["Gender"] for p in pats])),
    }


# ------------------------------------------------------------------ image
def image_arrays(pats, size):
    x, owner = [], []
    for i, p in enumerate(pats):
        for path, _ in p.images:
            x.append(M.load_image(path, size))
            owner.append(i)
    return np.stack(x), np.array(owner)


def image_eval(codec: ImageVQCodec, x: np.ndarray) -> dict:
    codes = codec.codes(x)
    rec = codec.decode_codes(codes)
    segs = [Segment("image", im) for im in x]
    spans = [codec.span_from_codes(c) for c in codes]
    tr = CodeUsageTracker(codec.model.n_codes)
    tr.update(codes)
    return {"reconstruction": image_metrics(x, rec),
            "compression": compression_summary([codec.raw_bits(s) for s in segs],
                                               [codec.payload_bits(s) for s in spans],
                                               [len(s.ids) for s in spans],
                                               raw_elements=[x.shape[1] * x.shape[2]] * len(x), spans=spans),
            "utilization": tr.stats()}


def train_image_vq(cfg, run, x_tr, x_va, size) -> ImageAutoencoder:
    c = cfg["image"]
    sub = {**cfg, "train": c["train"]}
    model = ImageAutoencoder(image_size=size, quantizer=c["quantizer"], **c["model"])
    xtr_t, xva_t = torch.as_tensor(x_tr), torch.as_tensor(x_va)

    def step(idx):
        x = model.to_input(xtr_t[idx].to(DEV))
        xr, q = model(x)
        return F.mse_loss(xr, x) + q.loss, q

    @torch.no_grad()
    def val():
        xr, q = model(model.to_input(xva_t.to(DEV)))
        m = image_metrics(x_va, model.to_uint8(xr).cpu().numpy(), ssim=False)
        m.update(active_codes=usage_stats(np.bincount(q.indices.cpu().numpy().ravel(), minlength=model.n_codes))["active_codes"])
        return m

    print(f"[image {size}px] train {x_tr.shape} val {x_va.shape}", flush=True)
    sub_run = SimpleNamespace(log=lambda r: run.log({"section": f"image_{size}", **r}))
    _epoch_loop(sub_run, model, sub, len(x_tr), step, val)
    torch.save({"cfg": model.cfg, "state": model.state_dict()}, run.path(f"image_vq_{size}.pt"))
    return model.eval()


def image_section(cfg, run, pats):
    c, out, codecs = cfg["image"], {}, {}
    split = np.array([p.split for p in pats])
    pre = load_tokenizer(c["pretrained"])
    x28, own28 = image_arrays(pats, 28)
    out["medmnist_zero_shot_28"] = image_eval(pre, x28[split[own28] == "test"])
    for size in c["sizes"]:
        x, own = (x28, own28) if size == 28 else image_arrays(pats, size)
        model = train_image_vq(cfg, run, x[split[own] == "train"], x[split[own] == "val"], size)
        codecs[size] = ImageVQCodec(model)
        out[f"mimic_trained_{size}"] = image_eval(codecs[size], x[split[own] == "test"])
        if size == 28:   # the reverse direction: does the MIMIC tokenizer still work on MedMNIST chest X-rays?
            out["mimic_trained_28_on_pneumoniamnist"] = pneumoniamnist_eval(codecs[28])
        if size == c["unified_size"]:
            k = min(8, int((split[own] == "test").sum()))
            xt = x[split[own] == "test"][:k]
            np.savez_compressed(run.path("figures_local", f"cxr_recon_{size}.npz"), x=xt,
                                rec=codecs[size].decode_codes(codecs[size].codes(xt)))
    out["n_test_images"] = int((split[own28] == "test").sum())
    return out, codecs


def pneumoniamnist_eval(codec) -> dict:
    from .data.datasets import image_split
    x, _, _, _ = image_split({"root": "data/raw/medmnist", "image_size": 28}, "test", ["pneumoniamnist"])
    r = image_eval(codec, x)
    return {"reconstruction": r["reconstruction"], "utilization": {k: r["utilization"][k] for k in
                                                                  ("active_codes", "utilization", "perplexity")}}


# ------------------------------------------------------------------ ECG
def ecg_signals(pats, hz):
    """All ECGs at `hz`: list of (patient index, minutes since t=0, (T,12) mV)."""
    return [(i, t, M.ecg_resample(M.read_ecg(rec), hz)) for i, p in enumerate(pats) for rec, t in p.ecgs]


def ecg_section(cfg, run, pats, lex):
    c, hz = cfg["ecg"], cfg["ecg"]["hz"]
    sig = ecg_signals(pats, hz)
    T = sig[0][2].shape[0]
    split = np.array([pats[i].split for i, _, _ in sig])
    spec = GridSpec(T, 1.0 / hz / 60, ECG_VARS, [])
    obs = [M.ecg_obs(s, hz) for _, _, s in sig]
    norm = Normalizer(variables=ECG_VARS).fit([SimpleNamespace(obs=o, split=s) for o, s in zip(obs, split)])
    z = np.stack([norm.grid(*spec.grid(o)) for o in obs])
    m = np.ones_like(z, bool)
    zt = {s: torch.as_tensor(z[split == s]) for s in ("train", "val", "test")}
    mt = {s: torch.as_tensor(m[split == s]) for s in ("train", "val", "test")}
    model = TimeSeriesWindowAutoencoder(n_vars=len(ECG_VARS), steps=T, quantizer=c["quantizer"], **c["model"])

    def step(idx):
        x, mm = zt["train"][idx].to(DEV), mt["train"][idx].to(DEV)
        xr, q = model((x, mm))
        return F.mse_loss(xr, x) + q.loss, q

    @torch.no_grad()
    def val():
        xv = zt["val"].to(DEV)
        xr, q = model((xv, mt["val"].to(DEV)))
        return {"mse_z": F.mse_loss(xr, xv).item(),
                "active_codes": usage_stats(np.bincount(q.indices.cpu().numpy().ravel(), minlength=model.n_codes))["active_codes"]}

    print(f"[ecg] train {tuple(zt['train'].shape)} val {tuple(zt['val'].shape)}", flush=True)
    _epoch_loop(SimpleNamespace(log=lambda r: run.log({"section": "ecg", **r})), model, {**cfg, "train": c["train"]},
                len(zt["train"]), step, val)
    torch.save({"cfg": model.cfg, "state": model.state_dict()}, run.path("ecg_vq.pt"))
    run.save_json("ecg_normalizer.json", norm.to_json())
    codec = TSVQCodec(lex, ValueBinner(cfg["labs"]["n_value_bins"]), spec, model.eval(), norm)

    # test: through the protocol codec, errors in mV against the 100 Hz signal the tokenizer sees
    te = [k for k in range(len(sig)) if split[k] == "test"]
    segs = [Segment("timeseries", {"obs": obs[k], "static": {}}, 0.0, SRC["ecg"]) for k in te]
    spans = codec.encode_batch(segs)
    rec = np.stack([codec.decode(s)["grid"] for s in spans])
    x = np.stack([sig[k][2] for k in te])
    err = rec - x
    rmse = np.sqrt((err ** 2).mean((1, 2)))
    prd = 100 * np.sqrt((err ** 2).sum((1, 2)) / np.maximum(((x - x.mean(1, keepdims=True)) ** 2).sum((1, 2)), 1e-9))
    corr = [np.mean([np.corrcoef(x[i, :, j], rec[i, :, j])[0, 1] for j in range(12) if x[i, :, j].std() > 1e-6])
            for i in range(len(x))]
    codes = np.stack([codec.codes_from_span(s)[1] for s in spans])
    tr = CodeUsageTracker(model.n_codes)
    tr.update(codes)
    raw500 = M.ECG_HZ * 10 * 12 * 16.0                     # WFDB format 16 as stored
    comp = compression_summary([raw500] * len(spans), [codec.payload_bits(s) for s in spans], [len(s.ids) for s in spans],
                               raw_elements=[x.shape[1] * 12] * len(spans), spans=spans)
    comp["compression_ratio_vs_100hz_16bit"] = float(T * 12 * 16.0 / np.mean([codec.payload_bits(s) for s in spans]))
    np.savez_compressed(run.path("figures_local", "ecg_recon.npz"), x=x[:4], rec=rec[:4])
    out = {"n_ecg": len(sig), "n_test_ecg": len(te), "hz": hz, "steps": T,
           "tokens_per_ecg": int(len(spans[0].ids)),
           "reconstruction": {"rmse_mv": mean_sd(rmse), "prdn_pct": mean_sd(prd), "lead_corr": mean_sd(corr)},
           "compression": comp, "utilization": tr.stats()}
    return out, codec


# ------------------------------------------------------------------ labs
def labs_codec(cfg, pats, lex, lab_vars, res):
    spec = GridSpec(int(M.WINDOW_MIN // res), float(res), lab_vars, list(M.STATIC_VARS))
    binner = ValueBinner(cfg["labs"]["n_value_bins"]).fit(stays_view(pats), spec)
    return TSEventCodec(lex, binner, spec), spec


def labs_section(cfg, run, pats, lex, lab_vars):
    out, codecs = {}, {}
    test = [p for p in pats if p.split == "test"]
    lab_norm = Normalizer(variables=lab_vars).fit(stays_view(pats))
    for res in cfg["labs"]["resolutions_min"]:
        codec, spec = labs_codec(cfg, pats, lex, lab_vars, res)
        codecs[res] = codec
        segs = [Segment("timeseries", {"obs": p.labs, "static": p.static}, 0.0, SRC["labs"]) for p in test]
        spans = [codec.encode(s) for s in segs]
        z_err, mask_ok, idem, merged = [], [], [], []
        for p, sp in zip(test, spans):
            g, mk = spec.grid(p.labs)
            dec = codec.decode(sp)
            mask_ok.append(bool((dec["mask"] == mk).all()))
            var = np.broadcast_to(np.arange(len(lab_vars)), g.shape)
            if mk.any():
                z_err.append(np.abs(lab_norm.transform(dec["grid"][mk], var[mk]) - lab_norm.transform(g[mk], var[mk])))
            again = codec.encode(Segment("timeseries", {"obs": grid_to_obs(dec["grid"], dec["mask"], res),
                                                        "static": dec["static"]}, 0.0, SRC["labs"]))
            idem.append(bool(np.array_equal(again.ids, sp.ids)))
            merged.append(1 - mk.sum() / max(len(p.labs), 1) if len(p.labs) else 0.0)
        z_err = np.concatenate(z_err)
        out[f"res_{int(res)}min"] = {
            "steps": spec.steps,
            "compression": compression_summary([codec.raw_bits(s) for s in segs], [codec.payload_bits(s) for s in spans],
                                               [len(s.ids) for s in spans], raw_elements=[len(p.labs) for p in test],
                                               spans=spans),
            "mae_z_observed_cells": float(z_err.mean()),
            "observation_mask_preserved": float(np.mean(mask_ok)),
            "reencode_identical": float(np.mean(idem)),
            "results_merged_into_cells": float(np.mean(merged)),
            "patients_with_gap_over_48h": long_gap_patients(test, res),
        }
    return out, codecs


def long_gap_patients(pats, res: float) -> float:
    """Fraction of patients with > 48 h between two consecutive lab grid steps: the cases a single <DT>
    token could not encode exactly (they need the dt_chain fix)."""
    hit = 0
    for p in pats:
        steps = np.unique(np.clip(p.labs[:, 0] // res, 0, M.WINDOW_MIN // res - 1)) if len(p.labs) else []
        hit += bool(len(steps) > 1 and (np.diff(steps) * res > 48 * 60).any())
    return hit / len(pats)


def grid_to_obs(g, m, res):
    h, v = np.nonzero(m)
    return np.stack([(h + 0.5) * res, v, g[h, v]], 1).astype(np.float32)


# ------------------------------------------------------------------ text
def text_section(cfg, run, pats):
    c, out, toks = cfg["text"], {}, {}
    tr = [M.read_report(p.report) for p in pats if p.split == "train"]
    te = [M.read_report(p.report) for p in pats if p.split == "test"]
    toks["synthetic_template_bpe"] = ClinicalTextTokenizer.load(abspath(c["pretrained"]))
    for v in c["vocab_sizes"]:
        toks[f"mimic_bpe_{v}"] = ClinicalTextTokenizer.train(tr, v)
        toks[f"mimic_bpe_{v}"].save(run.path(f"text_bpe_mimic_{v}.json"))
    words = np.array([len(s.split()) for s in te])
    chars = np.array([len(s) for s in te])
    for name, tok in toks.items():
        ids = [tok.encode_ids(s) for s in te]
        n = np.array([len(i) for i in ids])
        out[name] = {"vocab": tok.alphabet_size(),
                     "exact_roundtrip": float(np.mean([tok.decode_ids(i) == s for i, s in zip(ids, te)])),
                     "tokens_per_report": mean_sd(n), "tokens_per_word": float(n.sum() / words.sum()),
                     "chars_per_token": float(chars.sum() / n.sum()),
                     "compression_ratio_vs_utf8": float(sum(len(s.encode()) * 8 for s in te)
                                                        / (n.sum() * np.log2(tok.alphabet_size())))}
    out["n_train_reports"], out["n_test_reports"] = len(tr), len(te)
    return out, toks


# ------------------------------------------------------------------ unified protocol
def patient_sample(p, img_size, hz) -> Sample:
    segs = [Segment("timeseries", {"obs": p.labs, "static": p.static}, 0.0, SRC["labs"])]
    segs += [Segment("timeseries", {"obs": M.ecg_obs(M.ecg_resample(M.read_ecg(rec), hz), hz), "static": {}}, t,
                     SRC["ecg"]) for rec, t in p.ecgs]
    segs += [Segment("image", M.load_image(path, img_size), float(M.WINDOW_MIN), SRC["image"], {"view": view})
             for path, view in p.images]
    segs.append(Segment("text", M.read_report(p.report), float(M.WINDOW_MIN), SRC["text"]))
    return Sample(f"mimic-iv/{p.subject_id}", segs, {"death_1y": p.death}, p.split, {"subject_id": p.subject_id})


def unified_section(cfg, run, pats, mc: MultimodalCodec):
    samples = [patient_sample(p, cfg["image"]["unified_size"], cfg["ecg"]["hz"]) for p in pats]
    t0 = time.time()
    seqs = mc.encode_batch(samples)
    enc_sec = time.time() - t0
    for s in seqs:
        validate(s)
    shard = run.path("tokens", "mimic_unified.npz")
    save_shard(shard, seqs, {"source": "mimic-iv linked demo", "protocol_note": "credentialed - do not share"})
    back, _ = load_shard(shard)
    same = [np.array_equal(a.ids, b.ids) and np.array_equal(a.t, b.t) and np.array_equal(a.pos, b.pos)
            and a.spans == b.spans and a.labels == b.labels for a, b in zip(seqs, back)]
    text_ok, labs_ok, n_dec = [], [], 0
    for smp, seq in zip(samples, back):
        dec = mc.decode(seq)
        n_dec += len(dec)
        order = sorted(range(len(smp.segments)), key=lambda i: smp.segments[i].t_start)
        for i, (mod, payload) in zip(order, dec):
            seg = smp.segments[i]
            if mod == "text":
                text_ok.append(payload == seg.data)
            elif seg.source == SRC["labs"]:
                labs_ok.append(bool((payload["mask"] == mc.codecs["timeseries"].routes[SRC["labs"]].spec.grid(
                    seg.data["obs"])[1]).all()))
    by_src: dict[str, list[int]] = {}
    for seq in seqs:
        per = {}
        for sp in seq.spans:
            k = sp["meta"].get("source") or SRC[sp["modality"]]
            per[k] = per.get(k, 0) + sp["end"] - sp["start"] + 3
        for k in SRC.values():
            by_src.setdefault(k, []).append(per.get(k, 0))
    lens = np.array([len(s) for s in seqs])
    return {
        "patients": len(seqs), "encode_seconds": enc_sec,
        "structural_validation_passed": len(seqs),
        "shard_roundtrip_identical": float(np.mean(same)),
        "segments_decoded": n_dec,
        "text_exact_after_shard": float(np.mean(text_ok)), "labs_mask_exact_after_shard": float(np.mean(labs_ok)),
        "sequence_length": {"mean": float(lens.mean()), "median": float(np.median(lens)), "max": int(lens.max()),
                            "over_4096": int((lens > 4096).sum()), "over_8192": int((lens > 8192).sum())},
        "tokens_by_source_mean": {k: float(np.mean(v)) for k, v in sorted(by_src.items())},
        "token_share_by_source": {k: float(np.sum(v) / lens.sum()) for k, v in sorted(by_src.items())},
        "shard_bytes_per_patient": float(os.path.getsize(shard) / len(seqs)),
        "routes": {k: type(c).__name__ for k, c in mc.codecs["timeseries"].routes.items()},
    }, seqs


# ------------------------------------------------------------------ downstream
def features(seqs, pats, lex, lab_vars):
    """Bag-of-tokens per source (log1p counts); labs as (variable, value-bin) pairs; static block as age/sex."""
    nb = lex.n_value_bins
    lab_idx = {lex.var_id(v): j for j, v in enumerate(lab_vars)}
    F_ = {k: [] for k in ("image", "ecg", "labs", "text", "demographics")}
    for seq, p in zip(seqs, pats):
        img = np.zeros(RANGES["image_vq"].size)
        ecg = np.zeros(RANGES["ts_vq"].size)
        txt = np.zeros(RANGES["text"].size)
        lab = np.zeros(len(lab_vars) * nb)
        for sp, span in zip(seq.spans, split_spans(seq)):
            ids = span.ids
            src = sp["meta"].get("source")
            if sp["modality"] == "image":
                v = ids[np.isin(ids, np.arange(RANGES["image_vq"].start, RANGES["image_vq"].end))]
                np.add.at(img, v - RANGES["image_vq"].start, 1)
            elif sp["modality"] == "text":
                np.add.at(txt, ids - RANGES["text"].start, 1)
            elif src == SRC["ecg"]:
                v = ids[np.isin(ids, np.arange(RANGES["ts_vq"].start, RANGES["ts_vq"].end))]
                np.add.at(ecg, v - RANGES["ts_vq"].start, 1)
            elif src == SRC["labs"]:
                for a, b in zip(ids[:-1], ids[1:]):
                    if int(a) in lab_idx and int(b) in RANGES["vbin"]:
                        lab[lab_idx[int(a)] * nb + RANGES["vbin"].to_local(int(b))] += 1
        F_["image"].append(img / max(len(p.images), 1))
        F_["ecg"].append(ecg / max(len(p.ecgs), 1))
        F_["text"].append(txt)
        F_["labs"].append(lab)
        F_["demographics"].append([p.static["Age"], p.static["Gender"]])
    return {k: np.log1p(np.asarray(v, np.float64)) if k != "demographics" else np.asarray(v, np.float64)
            for k, v in F_.items()}


def downstream_section(cfg, feats, y):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import average_precision_score, roc_auc_score
    from sklearn.model_selection import RepeatedStratifiedKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    c = cfg["downstream"]
    sets = {"demographics": ["demographics"], "image": ["image"], "ecg": ["ecg"], "labs": ["labs"],
            "text": ["text"], "image+ecg+labs": ["image", "ecg", "labs"],
            "all (image+ecg+labs+text)": ["image", "ecg", "labs", "text"],
            "all + demographics": ["image", "ecg", "labs", "text", "demographics"]}
    cv = RepeatedStratifiedKFold(n_splits=c["folds"], n_repeats=c["repeats"], random_state=cfg.get("seed", 0))
    out = {"task": f"death within {cfg['data']['horizon_days']} days of the radiograph", "n": int(len(y)),
           "positives": int(y.sum()), "prevalence": float(y.mean()),
           "protocol": f"{c['repeats']}x repeated stratified {c['folds']}-fold CV, logistic regression "
                       f"(L2, C={c['C']}, balanced class weights) on standardised log1p token counts"}
    for name, parts in sets.items():
        X = np.concatenate([feats[p] for p in parts], 1)
        X = X[:, X.std(0) > 0]
        au, ap = [], []
        for tr, te in cv.split(X, y):
            clf = make_pipeline(StandardScaler(), LogisticRegression(C=c["C"], class_weight="balanced", max_iter=5000))
            clf.fit(X[tr], y[tr])
            s = clf.predict_proba(X[te])[:, 1]
            au.append(roc_auc_score(y[te], s))
            ap.append(average_precision_score(y[te], s))
        out[name] = {"features": int(X.shape[1]), "auroc": mean_sd(au), "auprc": mean_sd(ap)}
        print(f"[downstream] {name:28s} AUROC {np.mean(au):.3f}±{np.std(au):.3f}  AUPRC {np.mean(ap):.3f}", flush=True)
    return out


# ------------------------------------------------------------------ main
def run_validation(cfg: dict) -> Run:
    run = Run(cfg)
    torch.set_num_threads(int(cfg.get("threads", os.environ.get("MEDTOK_THREADS", os.cpu_count()))))
    print(f"run dir: {run.dir}", flush=True)
    d = cfg["data"]
    root = abspath(d["root"])
    pats = M.load_cohort(root, d.get("split_seed", 0), d.get("horizon_days", 365))
    vocab = M.lab_vocabulary(root, pats, d.get("lab_top_k", 40))
    M.attach_labs(root, pats, vocab)
    lab_vars = [n for _, n in vocab]
    # one lexicon for every source: PhysioNet names first (ids unchanged), then ECG leads, then lab tests
    lex = Lexicon(PN_STATIC + TS_VARS + ECG_VARS + lab_vars, cfg["labs"]["n_value_bins"], annotated_events(), 64)
    lex.save(run.path("lexicon.json"))
    metrics = {"cohort": cohort_section(pats, lab_vars), "lab_variables": lab_vars}
    run.save_metrics(metrics)
    print("[cohort]", metrics["cohort"], flush=True)

    metrics["text"], toks = text_section(cfg, run, pats)
    run.save_metrics(metrics)
    metrics["labs"], lab_codecs = labs_section(cfg, run, pats, lex, lab_vars)
    run.save_metrics(metrics)
    metrics["image"], img_codecs = image_section(cfg, run, pats)
    run.save_metrics(metrics)
    metrics["ecg"], ecg_codec = ecg_section(cfg, run, pats, lex)
    run.save_metrics(metrics)

    mc = MultimodalCodec({
        "text": toks[f"mimic_bpe_{cfg['text']['unified_vocab']}"],
        "image": img_codecs[cfg["image"]["unified_size"]],
        "timeseries": RoutedCodec({SRC["labs"]: lab_codecs[cfg["labs"]["unified_resolution_min"]],
                                   SRC["ecg"]: ecg_codec}),
    })
    metrics["unified"], seqs = unified_section(cfg, run, pats, mc)
    run.save_metrics(metrics)
    print("[unified]", {k: v for k, v in metrics["unified"].items() if not isinstance(v, dict)}, flush=True)

    feats = features(seqs, pats, lex, lab_vars)
    metrics["downstream"] = downstream_section(cfg, feats, np.array([p.death for p in pats]))
    run.save_metrics(metrics)
    return run


# ------------------------------------------------------------------ report (aggregates only)
def _f(x, nd=3):
    return "-" if x is None else f"{x:,.{nd}f}" if isinstance(x, float) else f"{x:,}" if isinstance(x, int) else str(x)


def _ms(d, nd=3):
    return f"{d['mean']:.{nd}f} ± {d['sd']:.{nd}f}"


def write_report(run_dir: str, out_dir: str = "reports/mimic") -> str:
    """reports/mimic/MIMIC_RESULTS.md + mimic_metrics.json from a run's metrics.json (no patient-level data)."""
    import json
    import shutil
    from .utils.experiment import resolve_run
    run_dir = resolve_run(run_dir)
    with open(os.path.join(run_dir, "metrics.json")) as f:
        m = json.load(f)
    out_dir = abspath(out_dir)
    os.makedirs(out_dir, exist_ok=True)
    shutil.copy(os.path.join(run_dir, "metrics.json"), os.path.join(out_dir, "mimic_metrics.json"))
    c, L = m["cohort"], []
    L += ["# MIMIC-IV linked multimodal validation", "",
          f"Generated by `python -m medical_tokenizer.mimic_validate` from `{os.path.relpath(run_dir, abspath('.'))}`. "
          "Do not edit numbers by hand. Credentialed data: this file holds aggregate numbers only.", "",
          "## Cohort", "",
          "| item | value |", "|---|---|",
          f"| patients (train / val / test) | {c['patients']} ({c['split']['train']} / {c['split']['val']} / {c['split']['test']}) |",
          f"| died within 1 year of the radiograph | {c['deaths_1y']} ({100 * c['prevalence_1y']:.1f} %) |",
          f"| chest X-ray images (per patient) | {c['images']} ({_ms(c['images_per_patient'], 2)}) |",
          f"| 12-lead ECGs, 7 days before the CXR (per patient) | {c['ecgs']} ({_ms(c['ecgs_per_patient'], 2)}) |",
          f"| lab variables kept / median results per patient / patients with no labs | {c['lab_variables']} / "
          f"{_f(c['lab_results_per_patient_median'], 1)} / {c['patients_without_labs']} |",
          f"| radiology reports (median characters) | {c['reports']} ({_f(c['report_chars_median'], 0)}) |",
          f"| median age / male fraction | {_f(c['age_median'], 0)} / {_f(c['male_fraction'], 2)} |", ""]
    im = m["image"]
    L += ["## Chest X-ray images (VQ, K = 1024)", "",
          f"Test split, {im['n_test_images']} images. \"Zero-shot\" = the MedMNIST tokenizer used unchanged.", "",
          "| tokenizer | size | PSNR dB | SSIM | active codes | perplexity | tokens / image | ratio | ratio (+overhead) |",
          "|---|---|---|---|---|---|---|---|---|"]
    rows = [("MedMNIST, zero-shot", "medmnist_zero_shot_28", 28)] + [
        (f"MIMIC-trained", f"mimic_trained_{s}", s) for s in sorted(int(k.split("_")[-1]) for k in im
                                                                       if k.startswith("mimic_trained_") and k[-1].isdigit())]
    for name, k, s in rows:
        r = im[k]
        L.append(f"| {name} | {s} px | {_f(r['reconstruction']['psnr_db'], 2)} | {_f(r['reconstruction']['ssim'])} | "
                 f"{r['utilization']['active_codes']} | {_f(r['utilization']['perplexity'], 1)} | "
                 f"{_f(r['compression']['tokens_per_sample_mean'], 0)} | {_f(r['compression']['compression_ratio'], 1)} | "
                 f"{_f(r['compression']['compression_ratio_with_overhead'], 1)} |")
    if "mimic_trained_28_on_pneumoniamnist" in im:
        r = im["mimic_trained_28_on_pneumoniamnist"]
        L += ["", f"Reverse transfer: the MIMIC-trained 28 px tokenizer on the PneumoniaMNIST test set gives PSNR "
                  f"{_f(r['reconstruction']['psnr_db'], 2)} dB, SSIM {_f(r['reconstruction']['ssim'])}, "
                  f"{r['utilization']['active_codes']} active codes."]
    e = m["ecg"]
    L += ["", "## 12-lead ECG (existing window VQ tokenizer, K = 1024)", "",
          f"{e['n_ecg']} ECGs, {e['n_test_ecg']} in test. 500 Hz -> {e['hz']} Hz, {e['steps']} steps x 12 leads, "
          f"{e['tokens_per_ecg']} tokens per ECG. Errors are against the {e['hz']} Hz signal.", "",
          "| RMSE (mV) | PRDN % | mean lead correlation | active codes | perplexity | ratio vs stored 500 Hz | ratio vs 100 Hz 16-bit |",
          "|---|---|---|---|---|---|---|",
          f"| {_ms(e['reconstruction']['rmse_mv'], 4)} | {_ms(e['reconstruction']['prdn_pct'], 1)} | "
          f"{_ms(e['reconstruction']['lead_corr'])} | {e['utilization']['active_codes']} | "
          f"{_f(e['utilization']['perplexity'], 1)} | {_f(e['compression']['compression_ratio'], 1)} | "
          f"{_f(e['compression']['compression_ratio_vs_100hz_16bit'], 1)} |"]
    L += ["", "## Lab events (existing event tokenizer, 16 quantile bins)", "",
          "| grid step | steps in 7 days | tokens / patient (mean / max) | raw results / patient | results merged into a cell | "
          "MAE (z) | mask preserved | re-encode identical | ratio | patients with a gap > 48 h |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for k, r in m["labs"].items():
        cc = r["compression"]
        L.append(f"| {k[4:-3]} min | {r['steps']} | {_f(cc['tokens_per_sample_mean'], 0)} / {cc['tokens_per_sample_max']} | "
                 f"{_f(cc['raw_elements_per_sample'], 1)} | {100 * r['results_merged_into_cells']:.1f} % | "
                 f"{_f(r['mae_z_observed_cells'])} | {100 * r['observation_mask_preserved']:.1f} % | "
                 f"{100 * r['reencode_identical']:.1f} % | {_f(cc['compression_ratio'], 1)} | "
                 f"{100 * r['patients_with_gap_over_48h']:.1f} % |" if "patients_with_gap_over_48h" in r else
                 f"{100 * r['reencode_identical']:.1f} % | {_f(cc['compression_ratio'], 1)} | - |")
    t = m["text"]
    L += ["", "## Radiology reports (real clinical text)", "",
          f"BPE retrained on {t['n_train_reports']} training reports, evaluated on {t['n_test_reports']} test reports.", "",
          "| tokenizer | vocabulary | exact round-trip | tokens / report | tokens / word | characters / token |",
          "|---|---|---|---|---|---|"]
    for k, r in t.items():
        if isinstance(r, dict):
            L.append(f"| {k} | {r['vocab']} | {100 * r['exact_roundtrip']:.1f} % | {_ms(r['tokens_per_report'], 0)} | "
                     f"{_f(r['tokens_per_word'], 2)} | {_f(r['chars_per_token'], 2)} |")
    u = m["unified"]
    L += ["", "## Unified protocol: one token sequence per patient", "",
          "| check | result |", "|---|---|",
          f"| sequences passing structural validation | {u['structural_validation_passed']} / {u['patients']} |",
          f"| shard save -> load identical (ids, time, position, spans, labels) | {100 * u['shard_roundtrip_identical']:.1f} % |",
          f"| segments decoded back from the shard | {u['segments_decoded']} |",
          f"| report text exact after the shard round-trip | {100 * u['text_exact_after_shard']:.1f} % |",
          f"| lab observation pattern exact after the shard round-trip | {100 * u['labs_mask_exact_after_shard']:.1f} % |",
          f"| two time-series sources routed in one sequence | {', '.join(f'{k} -> {v}' for k, v in u['routes'].items())} |",
          f"| sequence length mean / median / max | {_f(u['sequence_length']['mean'], 0)} / "
          f"{_f(u['sequence_length']['median'], 0)} / {u['sequence_length']['max']} |",
          f"| sequences over 4,096 / 8,192 tokens | {u['sequence_length']['over_4096']} / {u['sequence_length']['over_8192']} |",
          f"| compressed shard size per patient | {_f(u['shard_bytes_per_patient'] / 1024, 1)} KiB |", "",
          "| source | tokens per patient (incl. delimiters) | share of all tokens |", "|---|---|---|"]
    for k, v in u["tokens_by_source_mean"].items():
        L.append(f"| {k} | {_f(v, 1)} | {100 * u['token_share_by_source'][k]:.1f} % |")
    d = m["downstream"]
    L += ["", "## Downstream: 1-year mortality from the tokens", "",
          f"{d['task']}; n = {d['n']}, {d['positives']} positive ({100 * d['prevalence']:.1f} %). {d['protocol']}. "
          f"AUPRC of a random classifier = prevalence.", "",
          "| input | features | AUROC | AUPRC |", "|---|---|---|---|"]
    for k, r in d.items():
        if isinstance(r, dict):
            L.append(f"| {k} | {r['features']} | {_ms(r['auroc'])} | {_ms(r['auprc'])} |")
    path = os.path.join(out_dir, "MIMIC_RESULTS.md")
    with open(path, "w") as f:
        f.write("\n".join(L) + "\n")
    return path


def main(argv=None) -> None:
    ap = arg_parser("Validate the tokenizers and the unified protocol on MIMIC-IV linked multimodal data.")
    ap.add_argument("--report", metavar="RUN", help="only regenerate reports/mimic/ from an existing run")
    args = ap.parse_args(argv)
    if args.report:
        print(write_report(args.report))
        return
    if not args.config:
        ap.error("--config is required")
    run = run_validation(load_config(args.config, args.set))
    print(write_report(run.dir))
    print(f"done: {run.dir}")


if __name__ == "__main__":
    main()
