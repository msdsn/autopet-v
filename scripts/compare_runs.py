"""Paired comparison of two evaluation rows on the cases they share.

A screening row and its control are run on the same case list with the same scribble
strategy per case, so the honest statistic is the **paired** mean difference, not the
difference of the two pooled means: per-case AUC varies far more between cases than
between models, and pairing removes that variance.

    python scripts/compare_runs.py <variant_run_dir> <control_run_dir> [--label X]

Reads `metric_scores_AUC.json` from each. AUC-DMM is NaN on lesion-free cases (it is a
nanmean over positives officially), so the DMM comparison is restricted to the cases
where both rows have a number; the Dice comparison uses every shared case.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from typing import Dict, List, Tuple


def load(run_dir: str) -> Dict[str, dict]:
    with open(os.path.join(run_dir, "metric_scores_AUC.json")) as f:
        return json.load(f)


def paired(a: Dict[str, dict], b: Dict[str, dict], key: str) -> Tuple[List[float], List[str]]:
    diffs, names = [], []
    for case in sorted(set(a) & set(b)):
        va, vb = a[case].get(key), b[case].get(key)
        if va is None or vb is None:
            continue
        if isinstance(va, float) and math.isnan(va):
            continue
        if isinstance(vb, float) and math.isnan(vb):
            continue
        diffs.append(float(va) - float(vb))
        names.append(case)
    return diffs, names


def report(name: str, diffs: List[float], names: List[str], top: int = 3) -> float:
    n = len(diffs)
    if n == 0:
        print(f"  {name:<10} no shared cases")
        return 0.0
    mean = sum(diffs) / n
    var = sum((d - mean) ** 2 for d in diffs) / (n - 1) if n > 1 else 0.0
    sem = math.sqrt(var / n) if n > 1 else 0.0
    wins = sum(1 for d in diffs if d > 1e-9)
    losses = sum(1 for d in diffs if d < -1e-9)
    t = mean / sem if sem > 0 else float("nan")
    print(f"  {name:<10} n={n:3d}  paired Δ = {mean:+.4f} ± {sem:.4f} (sem)  "
          f"t={t:+.2f}  better/worse/tied {wins}/{losses}/{n - wins - losses}")
    order = sorted(range(n), key=lambda i: diffs[i])
    if top:
        for tag, idxs in (("worst", order[:top]), ("best", order[-top:][::-1])):
            for i in idxs:
                print(f"      {tag:<5} {diffs[i]:+.3f}  {names[i][:64]}")
    return mean


def pooled(run_dir: str) -> dict:
    with open(os.path.join(run_dir, "run.json")) as f:
        return json.load(f)["results"]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("variant")
    ap.add_argument("control")
    ap.add_argument("--top", type=int, default=3, help="how many extreme cases to list")
    args = ap.parse_args()

    va, ca = load(args.variant), load(args.control)
    vp, cp = pooled(args.variant), pooled(args.control)
    print(f"variant  {args.variant}")
    print(f"control  {args.control}")
    print(f"  pooled AUC-Dice {cp['mean_auc_dice']:.4f} -> {vp['mean_auc_dice']:.4f}  "
          f"({vp['mean_auc_dice'] - cp['mean_auc_dice']:+.4f})")
    print(f"  pooled AUC-DMM  {cp['mean_auc_dmm']:.4f} -> {vp['mean_auc_dmm']:.4f}  "
          f"({vp['mean_auc_dmm'] - cp['mean_auc_dmm']:+.4f})")
    print(f"  pooled 50/50    {cp['final_score_50_50']:.4f} -> {vp['final_score_50_50']:.4f}  "
          f"({vp['final_score_50_50'] - cp['final_score_50_50']:+.4f})")
    print("paired, on the shared cases:")
    d_dice, n_dice = paired(va, ca, "auc_dice")
    d_dmm, n_dmm = paired(va, ca, "auc_dmm")
    m_dice = report("AUC-Dice", d_dice, n_dice, args.top)
    m_dmm = report("AUC-DMM", d_dmm, n_dmm, args.top)
    print(f"  50/50 proxy of the paired means: {(m_dice + m_dmm) / 2:+.4f}")

    # the adoption gate: DMM or Dice up by >0.05 with the other not worse
    ok = (m_dmm > 0.05 and m_dice >= 0) or (m_dice > 0.05 and m_dmm >= 0)
    print(f"  screen gate (Δ>+0.05 on one metric, other not worse): "
          f"{'PASS' if ok else 'FAIL'}")


if __name__ == "__main__":
    main()
