#!/usr/bin/env python3
"""
Check that a Tier-A store from build_store.py works as an nnUNet_preprocessed folder.

Two passes, because reading every case in full means pulling the whole store back off
Drive. The light pass covers all cases from metadata only (file triplet, pkl with
class_locations, matching shapes and channel count, dtypes); the deep pass reads
--sample N cases in full and pulls one real nnUNetDataLoader batch.

    python verify_store.py --store <...>/nnUNetPlans_3d_fullres --sample 12
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np

DRIVE = Path("/content/drive/MyDrive/autoPET")
PLANS = DRIVE / "weights/nnUNet_results/Dataset998_AutoPETV/nnUNetTrainer__nnUNetPlans__3d_fullres/plans.json"
DSJSON = DRIVE / "weights/nnUNet_results/Dataset998_AutoPETV/nnUNetTrainer__nnUNetPlans__3d_fullres/dataset.json"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", required=True)
    ap.add_argument("--sample", type=int, default=12,
                    help="cases to read in full (0 = light pass only)")
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--expect", type=int, default=None,
                    help="fail if the store does not hold exactly this many cases")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from nnunetv2.training.dataloading.nnunet_dataset import infer_dataset_class
    from nnunetv2.training.dataloading.data_loader import nnUNetDataLoader
    from nnunetv2.utilities.plans_handling.plans_handler import PlansManager

    store = Path(args.store)
    cls = infer_dataset_class(str(store))
    print(f"1. infer_dataset_class -> {cls.__name__}")

    pm = PlansManager(str(PLANS))
    cfg = pm.get_configuration("3d_fullres")
    dsj = json.loads(DSJSON.read_text())
    lm = pm.get_label_manager(dsj)
    n_ch = len(dsj["channel_names"])

    ds = cls(str(store))
    ids = sorted(ds.identifiers)
    print(f"2. {len(ids)} identifiers "
          f"({sum(i.startswith('fdg_') for i in ids)} FDG / "
          f"{sum(i.startswith('psma_') for i in ids)} PSMA)")

    problems: list[str] = []
    t0 = time.time()
    shapes, mvox = [], []
    for k, i in enumerate(ids, 1):
        try:
            data, seg, _, props = ds.load_case(i)
            if data.shape[0] != n_ch:
                problems.append(f"{i}: {data.shape[0]} channels, expected {n_ch}")
            if tuple(data.shape[1:]) != tuple(np.asarray(seg).shape[1:]):
                problems.append(f"{i}: data {data.shape[1:]} vs seg {seg.shape[1:]}")
            if "class_locations" not in props:
                problems.append(f"{i}: no class_locations in properties")
            if str(data.dtype) != "float16":
                problems.append(f"{i}: data dtype {data.dtype}")
            shapes.append(tuple(data.shape[1:]))
            mvox.append(float(np.prod(data.shape[1:])) / 1e6)
        except Exception as e:  # noqa: BLE001
            problems.append(f"{i}: {type(e).__name__}: {e}")
        if k % 250 == 0:
            print(f"   light pass {k}/{len(ids)}  {time.time() - t0:.0f}s", flush=True)
    print(f"3. light pass over {len(ids)} cases in {time.time() - t0:.0f}s: "
          f"{len(problems)} problem(s)")
    for p in problems[:20]:
        print(f"   ! {p}")
    if mvox:
        print(f"   megavoxels: mean {np.mean(mvox):.1f}  p50 {np.percentile(mvox, 50):.1f}  "
              f"p95 {np.percentile(mvox, 95):.1f}  max {np.max(mvox):.1f}")

    if args.sample:
        rng = random.Random(args.seed)
        pick = rng.sample(ids, min(args.sample, len(ids)))
        print(f"4. deep pass on {len(pick)} cases")
        for i in pick:
            data, seg, _, props = ds.load_case(i)
            a = np.asarray(data)
            s = np.asarray(seg)
            bad = []
            if not np.isfinite(a.astype(np.float32)).all():
                bad.append("non-finite")
            if a.shape[0] > 2 and (a[2:] != 0).any():
                bad.append("guidance channels not zero")
            if s.max() > 1:
                bad.append(f"seg max {s.max()}")
            print(f"   {i[:52]:52s} {tuple(a.shape)} ch0[{a[0].min():.2f},{a[0].max():.2f}] "
                  f"ch1[{a[1].min():.2f},{a[1].max():.2f}] fg={int((s > 0).sum())} "
                  f"{'OK' if not bad else 'BAD: ' + ', '.join(bad)}")
            problems += [f"{i}: {x}" for x in bad]

        dl = nnUNetDataLoader(cls(str(store), identifiers=pick), args.batch_size,
                              cfg.patch_size, cfg.patch_size, lm,
                              oversample_foreground_percent=0.33,
                              sampling_probabilities=None, pad_sides=None)
        b = next(dl)
        d, t = b["data"], b["target"]
        print(f"5. batch data {tuple(d.shape)} {d.dtype}  target {tuple(t.shape)} {t.dtype}  "
              f"fg voxels {int((t > 0).sum())}")
        if str(d.dtype) != "torch.float32":
            problems.append(f"batch dtype {d.dtype}")
        if tuple(d.shape) != (args.batch_size, n_ch, *cfg.patch_size):
            problems.append(f"batch shape {tuple(d.shape)}")

    if args.expect is not None and len(ids) != args.expect:
        problems.append(f"expected {args.expect} cases, found {len(ids)}")

    if problems:
        print(f"\nFAILED with {len(problems)} problem(s)")
        return 1
    print("\nOK - this folder can be used as "
          "nnUNet_preprocessed/<Dataset>/nnUNetPlans_3d_fullres")
    return 0


if __name__ == "__main__":
    sys.exit(main())
