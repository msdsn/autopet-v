#!/usr/bin/env python3
"""Turn a finished `interactive_eval.py` run into one `run.json` record and copy the
small artifacts to durable storage.

    python3 finalize_run.py --run_dir /content/work/runs/A0_baseline_20260826 \
        --run_id A0_baseline_20260826 --label "A0 baseline nnU-Net, 6 iters, strategy=all" \
        --drive /content/drive/MyDrive/autoPET/runs
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import time

import numpy as np


def _median(xs):
    xs = [x for x in xs if x is not None]
    return round(float(np.median(xs)), 2) if xs else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--run_id", default=None)
    ap.add_argument("--label", default="")
    ap.add_argument("--drive", default="/content/drive/MyDrive/autoPET/runs")
    ap.add_argument("--no_drive", action="store_true")
    args = ap.parse_args()

    run_id = args.run_id or os.path.basename(os.path.normpath(args.run_dir))
    summary = json.load(open(os.path.join(args.run_dir, "summary.json")))
    case_info = json.load(open(os.path.join(args.run_dir, "case_info.json")))
    auc = json.load(open(os.path.join(args.run_dir, "metric_scores_AUC.json")))

    # timing, split by tracer and by whether the model actually ran
    per_iter, per_case = {}, {}
    for tag, info in case_info.items():
        tr = "fdg" if tag.lower().startswith("fdg") else "psma" if tag.lower().startswith("psma") else "other"
        secs = [s for s, r in zip(info.get("iter_seconds", []), info.get("reused", []))
                if r is None and s > 0]
        per_iter.setdefault(tr, []).extend(secs)
        per_case.setdefault(tr, []).append(info.get("case_seconds"))

    gpu = ""
    try:
        gpu = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=20).stdout.strip()
    except Exception:
        pass

    record = {
        "run_id": run_id,
        "label": args.label,
        "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "gpu": gpu,
        "args": summary.get("args"),
        "predictor": summary.get("predictor"),
        "n_cases": summary.get("n_cases"),
        "max_iters": summary.get("max_iters"),
        "max_auc": summary.get("max_auc"),
        "results": {
            "mean_auc_dice": summary.get("mean_auc_dice"),
            "mean_auc_dmm": summary.get("mean_auc_dmm"),
            "final_score_50_50": summary.get("final_score_50_50"),
            "mean_dice_per_iteration": summary.get("mean_dice_per_iteration"),
            "mean_dmm_per_iteration": summary.get("mean_dmm_per_iteration"),
            "by_lesion_status": summary.get("by_lesion_status"),
            "by_tracer": summary.get("by_tracer"),
        },
        "empty_error_region_exposure": {
            k: v for k, v in (summary.get("empty_error_region_exposure") or {}).items()
            if k != "n_iters_with_zero_fp" and k != "n_iters_with_zero_fn"},
        "determinism_iter0_mismatches": summary.get("determinism_iter0_mismatches"),
        "timing": {
            "total_seconds": summary.get("total_seconds"),
            "median_iteration_seconds": {k: _median(v) for k, v in per_iter.items()},
            "median_case_seconds": {k: _median(v) for k, v in per_case.items()},
            "n_model_iterations": {k: len(v) for k, v in per_iter.items()},
        },
        "cache": summary.get("cache"),
        "worst_cases_auc_dice": sorted(auc, key=lambda t: auc[t]["auc_dice"])[:10],
    }

    out = os.path.join(args.run_dir, "run.json")
    with open(out, "w") as f:
        json.dump(record, f, indent=2)
    print(f"wrote {out}")

    if not args.no_drive:
        dst = os.path.join(args.drive, run_id)
        os.makedirs(dst, exist_ok=True)
        for name in ("run.json", "summary.json", "metric_scores.json",
                     "metric_scores_AUC.json", "case_info.json", "run.log"):
            src = os.path.join(args.run_dir, name)
            if os.path.isfile(src):
                shutil.copy2(src, os.path.join(dst, name))
        print(f"copied artifacts to {dst}")
    print(json.dumps(record["results"], indent=2)[:1500])


if __name__ == "__main__":
    main()
