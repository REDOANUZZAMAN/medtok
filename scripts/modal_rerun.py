"""Run experiment plans on Modal (GPU for training, CPU containers for evaluation-only plans).

    # once: upload data + current experiments (tarballs on the `medtok` volume)
    modal run scripts/modal_rerun.py::upload
    # run the four plans of configs/rerun_gpu/ in parallel, then pull the results back
    modal run scripts/modal_rerun.py::rerun
    python scripts/modal_rerun.py merge          # untar logs/modal_results/*.tar over experiments/
    # downstream-only re-run on the existing tokenizers, then merge just those results
    modal run scripts/modal_rerun.py::rerun --only ds_image,ds_mortality,ds_phase
    python scripts/modal_rerun.py merge ds_image ds_mortality ds_phase

Each job unpacks data + experiments onto the container's local disk, runs one plan with
`medical_tokenizer.pipeline`, and writes a tarball of every file it created or changed
(new run dirs, `latest` symlinks, eval/ folders, token shards) to the volume.
Training uses CUDA when present (utils/device.py); evaluation and token encoding stay on the CPU.
"""
from __future__ import annotations

import os
import subprocess
import sys

import modal

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VOL = modal.Volume.from_name("medtok", create_if_missing=True)
JOBS = {"image": "gpu", "mortality": "gpu", "phase": "gpu", "evaluate": "cpu",   # plan -> container kind
        "ds_image": "gpu", "ds_mortality": "gpu", "ds_phase": "gpu"}               # downstream-only (no tokenizer training)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("libgl1")
    .pip_install("torch==2.14.0", "numpy==2.5.3", "scipy==1.18.1", "scikit-learn==1.9.1", "scikit-image==0.26.0",
                 "pyyaml==6.0.3", "tokenizers==0.23.2", "matplotlib==3.11.2", "pillow==12.3.0", "av==19.0.0")
    .add_local_dir(os.path.join(ROOT, "src"), "/repo/src")
    .add_local_dir(os.path.join(ROOT, "configs"), "/repo/configs")
    .add_local_file(os.path.join(ROOT, "pyproject.toml"), "/repo/pyproject.toml")
)
app = modal.App("medtok-rerun", image=image)


def _sh(cmd: str) -> None:
    print("+", cmd, flush=True)
    subprocess.run(cmd, shell=True, check=True)


def _run_plan(plan: str, threads: int) -> str:
    os.chdir("/repo")
    _sh("tar -xf /vol/data.tar && tar -xf /vol/experiments.tar")
    _sh("touch /tmp/start && sleep 1")
    env = {**os.environ, "PYTHONPATH": "/repo/src", "MEDTOK_THREADS": str(threads), "PYTHONUNBUFFERED": "1"}
    p = subprocess.run([sys.executable, "-m", "medical_tokenizer.pipeline", "--config", f"configs/rerun_gpu/{plan}.yaml"],
                       env=env)
    os.makedirs("/vol/results", exist_ok=True)
    out = f"/vol/results/{plan}.tar"
    _sh(f"find experiments -newer /tmp/start \\( -type f -o -type l \\) -print0 | tar --null -cf {out} -T -")
    VOL.commit()
    return f"{plan}: pipeline exit {p.returncode}, results {os.path.getsize(out) / 1e6:.1f} MB"


def _fn(gpu, cpu):
    return app.function(gpu=gpu, cpu=cpu, memory=32768, timeout=6 * 3600, volumes={"/vol": VOL})


@_fn("L4", 8)
def run_gpu(plan: str) -> str:
    _sh("nvidia-smi --query-gpu=name,memory.total --format=csv")
    return _run_plan(plan, 8)


@_fn(None, 16)
def run_cpu(plan: str) -> str:
    return _run_plan(plan, 16)


@app.local_entrypoint()
def upload():
    _sh(f"cd {ROOT} && tar -cf /tmp/medtok_data.tar data/processed data/raw/medmnist data/raw/physionet2012")
    _sh(f"cd {ROOT} && tar -cf /tmp/medtok_experiments.tar experiments")
    with VOL.batch_upload(force=True) as b:
        b.put_file("/tmp/medtok_data.tar", "/data.tar")
        b.put_file("/tmp/medtok_experiments.tar", "/experiments.tar")
    print("uploaded")


@app.local_entrypoint()
def rerun(only: str = ""):
    plans = [p for p in JOBS if p in only.split(",")] if only else [p for p in JOBS if not p.startswith("ds_")]
    calls = [(run_gpu if JOBS[p] == "gpu" else run_cpu).spawn(p) for p in plans]
    for p, c in zip(plans, calls):
        try:
            print(c.get(), flush=True)
        except Exception as e:                        # report and keep collecting the other jobs
            print(f"{p}: FAILED {type(e).__name__}: {e}", flush=True)
    os.makedirs(os.path.join(ROOT, "logs", "modal_results"), exist_ok=True)
    for p in plans:
        try:
            with open(os.path.join(ROOT, "logs", "modal_results", f"{p}.tar"), "wb") as f:
                for chunk in VOL.read_file(f"/results/{p}.tar"):
                    f.write(chunk)
            print("downloaded", p, flush=True)
        except Exception as e:
            print(f"{p}: no results ({type(e).__name__}: {e})", flush=True)


def merge(plans: list[str] | None = None) -> None:
    d = os.path.join(ROOT, "logs", "modal_results")
    for p in plans or JOBS:
        f = os.path.join(d, f"{p}.tar")
        if os.path.exists(f):
            _sh(f"cd {ROOT} && tar -xf {f}")
            print("merged", p)


if __name__ == "__main__" and sys.argv[1:2] == ["merge"]:
    merge(sys.argv[2:])          # e.g. `merge ds_image ds_mortality ds_phase`; no names = every plan
