"""python -m medical_tokenizer.data.prepare --config configs/data.yaml

1. downloads the OPEN datasets that are missing (MedMNIST v2 subsets, PhysioNet 2012 set-a);
   controlled / request-only datasets (MIMIC, Cholec80, SCARED) are never downloaded;
2. samples Cataract-101 videos into the frame cache, if the zip has been placed in
   data/raw/cataract101/ (download it yourself after accepting its licence; see docs/datasets.md);
3. parses PhysioNet stays (cached) and fits the text BPE on training-split templated notes;
4. writes data/processed/assets/{lexicon.json,text_bpe.json} and data/processed/data_stats.json,
   the dataset statistics computed from the files actually present.
"""
from __future__ import annotations

import json
import os

import numpy as np

from ..tokenizers.text.codec import ClinicalTextTokenizer
from ..toolkit import build_lexicon
from ..utils.config import arg_parser, load_config
from . import datasets as D
from . import medmnist, physionet
from .notes import IMAGE_CONTEXT, VIDEO_CONTEXT, stay_note
from .video import prepare_cataract101


def prepare(cfg: dict) -> dict:
    stats = {}
    im = cfg.get("medmnist")
    if im:
        medmnist.download(D.abspath(im["root"]), im["subsets"], im.get("image_size", 28))
        for s in im["subsets"]:
            sp = medmnist.load_subset(D.abspath(im["root"]), s, im.get("image_size", 28))
            stats[f"medmnist/{s}"] = {k: {"n": int(len(v[1])), "shape": list(v[0].shape[1:]),
                                          "class_counts": np.bincount(v[1]).tolist()} for k, v in sp.items()}
    ts = cfg.get("physionet2012")
    if ts:
        physionet.download(D.abspath(ts["root"]))
        st = D.stays(ts)
        for split in ("train", "val", "test"):
            ss = [s for s in st if s.split == split]
            stats.setdefault("physionet2012", {})[split] = {
                "stays": len(ss), "mortality_rate": float(np.mean([s.death for s in ss])),
                "obs_per_stay_mean": float(np.mean([len(s.obs) for s in ss])),
                "obs_per_stay_median": float(np.median([len(s.obs) for s in ss]))}
    vd = cfg.get("cataract101")
    if vd:
        zp = D.abspath(vd["zip"])
        if os.path.exists(zp) and zipfile_complete(zp):
            prepare_cataract101(zp, D.abspath(vd["cache"]), vd.get("fps", 2.0), vd.get("size", 64),
                                vd.get("workers", 4), vd.get("limit"))
            files = D.video_files({"cache": vd["cache"]})
            smap = D.video_split_map({"cache": vd["cache"], "split_seed": vd.get("split_seed", 0)})
            per = {}
            for v, p in files.items():
                with np.load(p) as z:
                    ph = z["phase"]
                    s = per.setdefault(smap[v], {"videos": 0, "frames": 0, "unannotated_frames": 0,
                                                 "phase_counts": np.zeros(10, np.int64)})
                    s["videos"] += 1
                    s["frames"] += int(len(ph))
                    s["unannotated_frames"] += int((ph < 0).sum())
                    s["phase_counts"] += np.bincount(ph[ph >= 0], minlength=10)
            stats["cataract101"] = {k: {**v, "phase_counts": v["phase_counts"].tolist(), "cache_fps": vd.get("fps", 2.0)}
                                    for k, v in per.items()}
        else:
            print(f"cataract101: {zp} missing or incomplete -- skipped (see docs/datasets.md)")
    assets = D.abspath(cfg.get("assets", "data/processed/assets"))
    os.makedirs(assets, exist_ok=True)
    build_lexicon().save(os.path.join(assets, "lexicon.json"))
    if ts:
        corpus = [stay_note(s) for s in D.stays(ts) if s.split == "train"]
        corpus += [t for opts in IMAGE_CONTEXT.values() for t in opts] * 50 + VIDEO_CONTEXT * 50
        bpe = ClinicalTextTokenizer.train(corpus, vocab_size=cfg.get("text", {}).get("vocab_size", 1024))
        bpe.save(os.path.join(assets, "text_bpe.json"))
        stats["text_bpe"] = {"vocab_size": bpe.alphabet_size(), "train_notes": len(corpus)}
    with open(D.abspath("data/processed/data_stats.json"), "w") as f:
        json.dump(stats, f, indent=1)
    print(json.dumps(stats, indent=1)[:3000])
    return stats


def zipfile_complete(path: str) -> bool:
    import zipfile
    try:
        with zipfile.ZipFile(path) as z:
            return len(z.namelist()) > 0
    except zipfile.BadZipFile:
        return False


def main(argv=None) -> None:
    ap = arg_parser("Download open datasets and build protocol assets.")
    args = ap.parse_args(argv)
    prepare(load_config(args.config or D.abspath("configs/data.yaml"), args.set))


if __name__ == "__main__":
    main()
