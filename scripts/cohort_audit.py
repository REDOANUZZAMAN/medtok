"""Week 1-2 cohort audit of the linked MIMIC-IV subset: who is in it, what each patient has, how it is split.

Writes aggregate numbers only (credentialed data) to reports/mimic/COHORT_AUDIT.md.
    python scripts/cohort_audit.py [--root data/raw/mimic_demo_5gb/mimic_demo_5gb]
"""
import argparse
import csv
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from medical_tokenizer.data.mimic import load_cohort  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/raw/mimic_demo_5gb/mimic_demo_5gb")
    ap.add_argument("--out", default="reports/mimic/COHORT_AUDIT.md")
    a = ap.parse_args()

    pats = load_cohort(a.root)
    labs = list(csv.DictReader(open(os.path.join(a.root, "labs", "labevents_7d.csv"))))
    split = Counter(p.split for p in pats)
    pos = Counter(p.split for p in pats if p.death)
    n_ecg = [len(p.ecgs) for p in pats]
    reports = sum(os.path.exists(p.report) for p in pats)
    frontal = sum(any(v in ("PA", "AP") for _, v in p.images) for p in pats)
    hadm = sum(bool(r["hadm_id"]) for r in labs)

    L = ["# Cohort audit (week 1-2)", "",
         "Aggregate numbers only. Source: Rafi's linked subset of MIMIC-IV / MIMIC-CXR / MIMIC-IV-ECG "
         "(not the official 100-patient demo). Paper results must be re-run on cohorts built from the "
         "author's own credentialed access.", "",
         "| Item | Value |", "|---|---|",
         f"| distinct patients (subject_id) | {len({p.subject_id for p in pats})} |",
         f"| CXR studies (one per patient) | {len(pats)} |",
         f"| patients with a frontal (PA/AP) view | {frontal} |",
         f"| patients with a report file | {reports} |",
         f"| ECG records, total / per patient min-median-max | {sum(n_ecg)} / {min(n_ecg)}-{sorted(n_ecg)[len(n_ecg)//2]}-{max(n_ecg)} |",
         f"| lab rows (7 days up to the CXR) | {len(labs)} |",
         f"| lab rows with hadm_id | {hadm} ({hadm / len(labs):.0%}); no admissions table is linked, so in-hospital death cannot be derived |",
         f"| label now: death within 365 d of the CXR | {sum(pos.values())} of {len(pats)} |", "",
         "## Current split (70/15/15 by subject, stratified)", "",
         "| split | patients | deaths |", "|---|---|---|"]
    L += [f"| {s} | {split[s]} | {pos[s]} |" for s in ("train", "val", "test")]
    L += ["", "## Gaps against the MICCAI plan", "",
          f"- Test split has {pos['test']} positives; the plan wants about 200 (so about 5000 patients).",
          "- Label is 1-year death after the CXR, not in-hospital death after a 24 h landmark.",
          "- Lab window is 7 days; the plan's main window is 24 h (6 h and 48 h as sensitivity).",
          "- No CheXpert / report-derived disease labels are in the subset (needed for the main task E2).",
          "- Selection rule of the 600 patients is unknown (ask Rafi / supervisor)."]
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    open(a.out, "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
