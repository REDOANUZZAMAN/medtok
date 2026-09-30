"""YAML experiment configs with inheritance and command-line overrides.

    base: image_vq.yaml          # optional; path relative to this file, merged first
    name: image_vq_k64
    quantizer: {n_codes: 64}     # deep-merged over the base

Overrides:  --set quantizer.n_codes=128 --set train.epochs=2   (values parsed as YAML)
"""
from __future__ import annotations

import argparse
import copy
import os

import yaml


def deep_merge(a: dict, b: dict) -> dict:
    out = copy.deepcopy(a)
    for k, v in b.items():
        out[k] = deep_merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else copy.deepcopy(v)
    return out


def load_config(path: str, overrides: list[str] | None = None) -> dict:
    with open(path) as f:
        cfg = yaml.safe_load(f) or {}
    if "base" in cfg:
        base = load_config(os.path.join(os.path.dirname(os.path.abspath(path)), cfg.pop("base")))
        cfg = deep_merge(base, cfg)
    for o in overrides or []:
        key, _, val = o.partition("=")
        node = cfg
        parts = key.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = yaml.safe_load(val)
    cfg["_config_path"] = os.path.relpath(os.path.abspath(path))
    return cfg


def arg_parser(description: str) -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=description)
    ap.add_argument("--config", required=False, help="YAML experiment config")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="override a config value")
    return ap


def get(cfg: dict, dotted: str, default=None):
    node = cfg
    for p in dotted.split("."):
        if not isinstance(node, dict) or p not in node:
            return default
        node = node[p]
    return node
