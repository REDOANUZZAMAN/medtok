"""Run an experiment plan: a YAML list of train / evaluate / downstream / interfaces / report steps.

    python -m medical_tokenizer.pipeline --config configs/experiment_comparison.yaml [--skip-existing] [--only image]

Step forms (paths relative to the repo root):
    - train: configs/image_vq.yaml            # optional: name: ..., set: [key=value, ...]
    - evaluate: experiments/image_vq          # evaluates the experiment's latest run
    - downstream: configs/downstream/bloodmnist_vq.yaml   # optional: name, set
    - interfaces: true
    - report: true
Every step creates new run directories; --skip-existing skips train/downstream steps whose
experiment already has a finished run, and evaluate steps whose run already has eval/metrics.json.
Each step can carry `tags: [...]`; --only TAG runs only steps with that tag.
"""
from __future__ import annotations

import argparse
import os
import time
import traceback

import yaml

from .data.datasets import abspath
from .utils.config import load_config


def _finished(exp_dir: str, marker: str = "metrics.json") -> bool:
    latest = os.path.join(exp_dir, "latest")
    return os.path.exists(os.path.join(latest, marker))


def run_plan(plan: dict, skip_existing: bool = False, only: str | None = None, keep_going: bool = True) -> list:
    out_root = plan.get("output_dir", "experiments")
    status = []
    for i, step in enumerate(plan["steps"]):
        if only and only not in step.get("tags", []):
            continue
        kind = next(k for k in ("train", "evaluate", "downstream", "interfaces", "report") if k in step)
        t0 = time.time()
        label = f"[{i + 1}/{len(plan['steps'])}] {kind} {step.get('name') or step[kind]}"
        try:
            if kind in ("train", "downstream"):
                over = list(step.get("set", []))
                if step.get("name"):
                    over.append(f"name={step['name']}")
                cfg = load_config(abspath(step[kind]), over)
                if skip_existing and _finished(abspath(os.path.join(cfg.get("output_dir", out_root), cfg["name"]))):
                    print(f"{label}: exists, skipped", flush=True)
                    status.append((label, "skipped", 0))
                    continue
                print(f"\n===== {label}", flush=True)
                if kind == "train":
                    from .trainers import train
                    train(cfg)
                else:
                    from .downstream.__main__ import run_downstream
                    run_downstream(cfg)
            elif kind == "evaluate":
                if skip_existing and _finished(abspath(step[kind]), os.path.join("eval", "metrics.json")):
                    print(f"{label}: exists, skipped", flush=True)
                    status.append((label, "skipped", 0))
                    continue
                print(f"\n===== {label}", flush=True)
                from .evaluate import evaluate
                evaluate(step[kind])
            elif kind == "interfaces":
                print(f"\n===== {label}", flush=True)
                from .evaluate import eval_interfaces
                eval_interfaces()
            else:
                print(f"\n===== {label}", flush=True)
                from .report import build_report
                build_report()
            status.append((label, "ok", time.time() - t0))
        except Exception as e:                                   # keep the plan going; failures are reported
            traceback.print_exc()
            status.append((label, f"FAILED: {type(e).__name__}: {e}", time.time() - t0))
            if not keep_going:
                break
    print("\n===== summary")
    for label, st, sec in status:
        print(f"{st:>10}  {sec / 60:6.1f} min  {label}")
    return status


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Run an experiment plan.")
    ap.add_argument("--config", required=True)
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument("--only", help="run only steps carrying this tag")
    ap.add_argument("--stop-on-error", action="store_true")
    args = ap.parse_args(argv)
    with open(abspath(args.config)) as f:
        plan = yaml.safe_load(f)
    run_plan(plan, args.skip_existing, args.only, keep_going=not args.stop_on_error)


if __name__ == "__main__":
    main()
