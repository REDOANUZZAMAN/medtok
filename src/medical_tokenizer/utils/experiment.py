"""Experiment run directories: one immutable folder per run.

    experiments/<name>/<YYYYmmdd-HHMMSS>_<config-hash>/
        config.yaml        resolved config (after inheritance and overrides)
        env.json           python / torch / numpy versions, git commit + dirty flag, CPU threads
        train_log.jsonl    one JSON record per logged step / epoch
        metrics.json/.csv  final metrics (flat keys in the CSV)
        ...                checkpoints, codebooks, token shards, eval outputs
    experiments/<name>/latest  -> symlink to the newest run

A run id is never reused: if the folder exists, a numeric suffix is added.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import platform
import random
import subprocess
import sys
import time

import numpy as np
import torch
import yaml

from ..protocol.schema import json_default

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))


def set_seed(seed: int, deterministic: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)


def config_hash(cfg: dict) -> str:
    clean = {k: v for k, v in cfg.items() if not k.startswith("_")}
    return hashlib.sha1(json.dumps(clean, sort_keys=True, default=str).encode()).hexdigest()[:8]


def git_state() -> dict:
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, stderr=subprocess.DEVNULL).decode().strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"], cwd=REPO_ROOT,
                                             stderr=subprocess.DEVNULL).decode().strip())
        return {"commit": commit, "dirty": dirty}
    except (subprocess.CalledProcessError, FileNotFoundError):
        return {"commit": None, "dirty": None}


def environment() -> dict:
    return {"python": sys.version.split()[0], "platform": platform.platform(), "torch": torch.__version__,
            "numpy": np.__version__, "torch_threads": torch.get_num_threads(), "cuda": torch.cuda.is_available(),
            "git": git_state(), "argv": sys.argv}


def flatten(d: dict, prefix: str = "") -> dict:
    out = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(flatten(v, key + "."))
        elif not isinstance(v, (list, tuple)):
            out[key] = v
    return out


class Run:
    def __init__(self, cfg: dict, kind: str = "run", root: str | None = None):
        self.cfg = cfg
        root = root or os.path.join(REPO_ROOT, cfg.get("output_dir", "experiments"))
        name = cfg.get("name", kind)
        base = os.path.join(root, name, f"{time.strftime('%Y%m%d-%H%M%S')}_{config_hash(cfg)}")
        path, i = base, 1
        while os.path.exists(path):
            path, i = f"{base}-{i}", i + 1
        os.makedirs(path)
        self.dir, self.name = path, name
        with open(self.path("config.yaml"), "w") as f:
            yaml.safe_dump(cfg, f, sort_keys=False)
        self.save_json("env.json", environment())
        latest = os.path.join(root, name, "latest")
        if os.path.islink(latest):
            os.remove(latest)
        if not os.path.exists(latest):
            os.symlink(os.path.basename(path), latest)
        set_seed(int(cfg.get("seed", 0)), cfg.get("deterministic", True))
        if "threads" in cfg:
            torch.set_num_threads(int(cfg["threads"]))

    def path(self, *parts) -> str:
        p = os.path.join(self.dir, *parts)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        return p

    def log(self, record: dict, file: str = "train_log.jsonl", echo: bool = True) -> None:
        with open(self.path(file), "a") as f:
            f.write(json.dumps(record, default=json_default) + "\n")
        if echo:
            print(json.dumps({k: round(v, 5) if isinstance(v, float) else v for k, v in record.items()
                              if not isinstance(v, (dict, list))}, default=json_default), flush=True)

    def save_json(self, name: str, obj) -> None:
        with open(self.path(name), "w") as f:
            json.dump(obj, f, indent=1, default=json_default)

    def save_metrics(self, metrics: dict, name: str = "metrics") -> None:
        self.save_json(f"{name}.json", metrics)
        flat = flatten(metrics)
        with open(self.path(f"{name}.csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["metric", "value"])
            for k, v in flat.items():
                w.writerow([k, v])


def resolve_run(path: str) -> str:
    """Accept a run dir, an experiment dir (uses its `latest`), or a path relative to the repo root."""
    for p in (path, os.path.join(REPO_ROOT, path)):
        if os.path.isdir(p):
            if os.path.exists(os.path.join(p, "config.yaml")):
                return os.path.realpath(p)
            if os.path.exists(os.path.join(p, "latest")):
                return os.path.realpath(os.path.join(p, "latest"))
    raise FileNotFoundError(f"no run found at {path!r}")


def load_run_config(run_dir: str) -> dict:
    with open(os.path.join(run_dir, "config.yaml")) as f:
        return yaml.safe_load(f)
