"""Find cases whose Dice oscillates across the interaction iterations.

Scans a harness run's metric_scores.json for the largest drop between consecutive
iterations. Pass --baseline (the same cases without post-processing) to tell the
model's own instability from ours.

    python src/postproc/tools/oscillation_scan.py --run <dir> [--baseline <dir>]
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List, Tuple


def load_dice(run_dir: str) -> Dict[str, List[float]]:
    path = os.path.join(run_dir, "metric_scores.json")
    with open(path) as fh:
        data = json.load(fh)
    out = {}
    for tag, recs in data.items():
        dice = [r["dice"] for r in sorted(recs, key=lambda r: r["iteration"])]
        if len(dice) >= 2:
            out[tag] = dice
    return out


def worst_drop(dice: List[float]) -> float:
    return max(dice[i] - dice[i + 1] for i in range(len(dice) - 1))


def scan(run_dir: str, threshold: float) -> List[Tuple[float, str, List[float]]]:
    rows = [(worst_drop(d), tag, d) for tag, d in load_dice(run_dir).items()]
    rows.sort(reverse=True)
    return [r for r in rows if r[0] > threshold]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="harness output dir to check")
    ap.add_argument("--baseline", default=None,
                    help="the same cases without post-processing, to separate our "
                         "instability from the model's")
    ap.add_argument("--threshold", type=float, default=0.15,
                    help="report drops larger than this")
    args = ap.parse_args()

    bad = scan(args.run, args.threshold)
    base_bad = {tag for _, tag, _ in scan(args.baseline, args.threshold)} if args.baseline else set()

    total = len(load_dice(args.run))
    print(f"{os.path.basename(args.run)}: {len(bad)}/{total} cases drop more than "
          f"{args.threshold:.2f} Dice between consecutive iterations")
    if args.baseline:
        ours = [r for r in bad if r[1] not in base_bad]
        print(f"  of those, {len(ours)} do not oscillate in the baseline run -> ours")
    for drop, tag, dice in bad:
        flag = "" if tag in base_bad else "  <- not in baseline"
        print(f"  drop {drop:.3f}  " + " ".join(f"{x:.3f}" for x in dice) + f"  {tag[:44]}{flag}")

    if bad and not args.baseline:
        print("\npass --baseline to separate the model's own instability from ours")


if __name__ == "__main__":
    main()
