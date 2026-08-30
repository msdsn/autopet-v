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


def load_iterations(run_dir: str) -> Dict[str, Dict[int, dict]]:
    """`metric_scores.json` as {case: {iteration: {"dice": ..., "dmm": ...}}}."""
    with open(os.path.join(run_dir, "metric_scores.json")) as f:
        raw = json.load(f)
    out: Dict[str, Dict[int, dict]] = {}
    for case, rows in raw.items():
        out[case] = {int(r["iteration"]): r for r in rows}
    return out


def report_iterations(variant: str, control: str, top: int = 0) -> None:
    """Paired mean Dice and DMM at every iteration both rows have.

    The AUC is a summary of this curve; when a lever moves only one end of it -- and
    iteration 0 is the end the GC preliminary leaderboard ranks -- the AUC hides that.
    """
    a, b = load_iterations(variant), load_iterations(control)
    shared = sorted(set(a) & set(b))
    its = sorted({i for c in shared for i in a[c]} & {i for c in shared for i in b[c]})
    print(f"per iteration, paired over {len(shared)} shared case(s):")
    print(f"  {'iter':<5}{'n':>4}  {'control':>9} {'variant':>9} {'paired d':>10} "
          f"{'sem':>8}   {'control':>9} {'variant':>9} {'paired d':>10} {'sem':>8}")
    print(f"  {'':<5}{'':>4}  {'--- Dice':>9} {'':>9} {'':>10} {'':>8}   "
          f"{'--- DMM':>9} {'':>9} {'':>10} {'':>8}")
    for it in its:
        row = [f"  {it:<5}"]
        n_shown = None
        for key in ("dice", "dmm"):
            va, vb, dif = [], [], []
            for c in shared:
                ra, rb = a[c].get(it), b[c].get(it)
                if ra is None or rb is None:
                    continue
                x, y = ra.get(key), rb.get(key)
                if x is None or y is None:
                    continue
                if isinstance(x, float) and math.isnan(x):
                    continue
                if isinstance(y, float) and math.isnan(y):
                    continue
                va.append(float(x))
                vb.append(float(y))
                dif.append(float(x) - float(y))
            if not dif:
                row.append(f"{'':>9} {'':>9} {'':>10} {'':>8}")
                continue
            n = len(dif)
            m = sum(dif) / n
            var = sum((d - m) ** 2 for d in dif) / (n - 1) if n > 1 else 0.0
            sem = math.sqrt(var / n) if n > 1 else 0.0
            if n_shown is None:
                n_shown = n
                row.append(f"{n:>4} ")
            row.append(f" {sum(vb)/n:>9.4f} {sum(va)/n:>9.4f} {m:>+10.4f} {sem:>8.4f}  ")
        print("".join(row))


def pooled(run_dir: str) -> dict:
    with open(os.path.join(run_dir, "run.json")) as f:
        return json.load(f)["results"]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("variant")
    ap.add_argument("control")
    ap.add_argument("--top", type=int, default=3, help="how many extreme cases to list")
    ap.add_argument("--per_iteration", action="store_true",
                    help="also print the paired Dice/DMM curve, iteration by iteration; "
                         "iteration 0 is what the GC preliminary leaderboard ranks")
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
    if args.per_iteration:
        print()
        report_iterations(args.variant, args.control, args.top)


if __name__ == "__main__":
    main()
