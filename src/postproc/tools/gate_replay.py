"""Replay the negative gate over the cached iteration-0 outputs of a real run.

negative_analysis.py stores the sparse iteration-0 prediction per case, which is all
is_probably_negative reads, so the gate can be re-run without a GPU. Reports how many
lesion-free and lesion-present cases it fires on and the change in mean AUC-Dice.

    python3 postproc/tools/gate_replay.py --out_dir /content/work/negan/<model> \
        [--set max_total_volume_ml=2.0] [--expect_neg 10 --expect_pos 3]
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

import sys
_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.dirname(os.path.dirname(_HERE))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from postproc.config import NegativeGateConfig                      # noqa: E402
from postproc.negative_gate import is_probably_negative             # noqa: E402
from postproc.tools.gate_sweep import build_table, exact_dice0_from_bundles, delta_auc  # noqa: E402


def rebuild(bundle_path: str):
    """Rebuild full-size (mask, pet, prob, spacing) holding the predicted voxels only.

    Everything the case-level gate reads lives inside the mask, so zeros outside are
    faithful.
    """
    with np.load(bundle_path) as z:
        shape = tuple(int(x) for x in z["shape"])
        idx = z["idx"]
        pet_v = z["pet"] if "pet" in z else np.zeros(0, np.float32)
        prob_v = z["prob"] if "prob" in z else np.zeros(0, np.float32)
        spacing = tuple(float(x) for x in z["spacing"])
    mask = np.zeros(shape, dtype=np.uint8)
    pet = np.zeros(shape, dtype=np.float32)
    prob = np.zeros(shape, dtype=np.float32)
    if idx.shape[0]:
        t = tuple(idx.T.astype(np.int64))
        mask[t] = 1
        pet[t] = pet_v
        prob[t] = prob_v
    has_prob = bool(idx.shape[0] and np.isfinite(prob_v).all())
    return mask, pet, (prob if has_prob else None), spacing


def parse_set(pairs: Optional[Sequence[str]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for kv in pairs or []:
        k, _, v = kv.partition("=")
        out[k] = json.loads(v)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--set", nargs="*", default=None, metavar="KEY=JSON",
                    help="override NegativeGateConfig fields, e.g. max_total_volume_ml=2.0")
    ap.add_argument("--expect_neg", type=int, default=None,
                    help="expected number of lesion-free cases the gate fires on")
    ap.add_argument("--expect_pos", type=int, default=None,
                    help="expected number of lesion-present cases the gate fires on")
    ap.add_argument("--iterations", type=int, default=6,
                    help="re-run the gate this many times per case with no scribbles; a "
                         "lesion-free case must be emptied at every one of them")
    args = ap.parse_args(argv)

    cfg = NegativeGateConfig(**parse_set(args.set))
    print("config:", json.dumps({k: v for k, v in cfg.__dict__.items()}, default=str))

    table = build_table(args.out_dir)
    exact_dice0_from_bundles(table, args.out_dir)
    bdir = os.path.join(args.out_dir, "bundles")

    fires: List[bool] = []
    rows = []
    for r in table:
        p = os.path.join(bdir, str(r["case"]).replace("/", "_") + ".npz")
        if not os.path.isfile(p):
            fires.append(False)
            continue
        mask, pet, prob, spacing = rebuild(p)
        per_iter = []
        for it in range(args.iterations):
            f, feats = is_probably_negative(mask, pet, prob, cfg, spacing=spacing,
                                            iteration=it, tracer=r["tracer"],
                                            return_features=True)
            per_iter.append(bool(f))
        if len(set(per_iter)) != 1:
            raise AssertionError(f"{r['case']}: gate is not iteration-stable: {per_iter}")
        fires.append(per_iter[0])
        rows.append((r, per_iter[0], feats))
        del mask, pet, prob

    d = delta_auc(table, fires)
    neg_fired = [r["case"] for r, f, _ in rows if f and r["empty_gt"]]
    pos_fired = [(r["case"], round(r["dice0"], 3)) for r, f, _ in rows if f and not r["empty_gt"]]
    rescued = [r["case"] for r, f, _ in rows if f and r["empty_gt"] and r["pred_voxels"] > 0]
    missed = [(r["case"], round(r["total_volume_ml"], 3), feats["blocked_by"])
              for r, f, feats in rows if (not f) and r["empty_gt"] and r["pred_voxels"] > 0]

    print(f"\nlesion-free cases the gate fires on : {len(neg_fired)} / "
          f"{sum(1 for r in table if r['empty_gt'])}")
    print(f"  of which rescued (were non-empty) : {len(rescued)}")
    print(f"lesion-present cases emptied        : {len(pos_fired)} / "
          f"{sum(1 for r in table if not r['empty_gt'])}")
    for c, dc in pos_fired:
        print(f"    {c[:60]:60s} Dice@0={dc}")
    if missed:
        print("non-empty lesion-free cases the gate MISSES:")
        for c, v, why in missed:
            print(f"    {c[:60]:60s} vol={v} mL  blocked_by={why}")
    print(f"\nexpected change in mean AUC-Dice     : {d['delta_mean_auc']:+.4f} "
          f"(gain {d['gain']:.1f}, cost {d['cost']:.3f}, over {len(table)} cases)")

    ok = True
    if args.expect_neg is not None and len(rescued) != args.expect_neg:
        print(f"FAIL: expected {args.expect_neg} rescued negatives, got {len(rescued)}")
        ok = False
    if args.expect_pos is not None and len(pos_fired) != args.expect_pos:
        print(f"FAIL: expected {args.expect_pos} emptied positives, got {len(pos_fired)}")
        ok = False
    print("REPLAY OK" if ok else "REPLAY MISMATCH")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
