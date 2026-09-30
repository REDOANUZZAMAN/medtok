#!/usr/bin/env bash
# Full reproduction, sequential (CPU): data -> tokenizers -> downstream -> ablations -> report.
# Safe to re-run: --skip-existing skips experiments that already have a finished run.
set -u
cd "$(dirname "$0")/.."
PY=${PY:-.venv/bin/python}
mkdir -p logs
$PY -m medical_tokenizer.data.prepare --config configs/data.yaml                               2>&1 | tee -a logs/prepare.log
$PY -m medical_tokenizer.pipeline --config configs/experiment_comparison.yaml --skip-existing --only tokenizers 2>&1 | tee -a logs/tokenizers.log
$PY -m medical_tokenizer.pipeline --config configs/experiment_comparison.yaml --skip-existing --only interfaces 2>&1 | tee -a logs/interfaces.log
$PY -m medical_tokenizer.pipeline --config configs/experiment_comparison.yaml --skip-existing --only downstream 2>&1 | tee -a logs/downstream.log
$PY -m medical_tokenizer.pipeline --config configs/ablations.yaml --skip-existing                2>&1 | tee -a logs/ablations.log
$PY -m medical_tokenizer.report                                                                2>&1 | tee -a logs/report.log
