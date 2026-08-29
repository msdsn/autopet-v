"""Measure the positive-cell fraction the N1 presence head has to predict.

The auxiliary BCE of N1 asks a coarse grid (encoder stage 3, one cell = 8x8x8 voxels
= 6.4 mL at the plans spacing) whether a lesion is present in each cell. A fixed
``pos_weight`` would be a guess; this script measures the actual class balance on the
training sampler and prints ``pos_weight = (1 - p) / p``, which is the value that
makes the positive and the negative half of the BCE contribute equally.

    python -m train.measure_n1_prior --preprocessed <prep>/Dataset998_AutoPETV \
        --plans nnUNetPlans_interactive --batches 200

Patches are drawn through the real ``nnUNetDataLoader`` at the configured
``oversample_foreground_percent``, so the label-empty patches (about half of them)
count exactly as they will during training. Spatial augmentation is not applied; it
moves lesions around inside a patch but barely changes how many cells contain one.
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch
import torch.nn.functional as F

from nnunetv2.training.dataloading.data_loader import nnUNetDataLoader
from nnunetv2.training.dataloading.nnunet_dataset import infer_dataset_class
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--preprocessed", required=True)
    ap.add_argument("--plans", default="nnUNetPlans_interactive")
    ap.add_argument("--configuration", default="3d_fullres")
    ap.add_argument("--batches", type=int, default=200)
    ap.add_argument("--downsample", type=int, default=8,
                    help="voxels per gate cell per axis (encoder stage 3 = 8)")
    ap.add_argument("--oversample", type=float, default=None,
                    help="default: the plans oversample_foreground_percent")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    np.random.seed(args.seed)
    plans = PlansManager(os.path.join(args.preprocessed, args.plans + ".json"))
    with open(os.path.join(args.preprocessed, "dataset.json")) as f:
        dataset_json = json.load(f)
    cfg = plans.get_configuration(args.configuration)
    label_manager = plans.get_label_manager(dataset_json)
    folder = os.path.join(args.preprocessed, cfg.data_identifier)

    ds_cls = infer_dataset_class(folder)
    ids = sorted(ds_cls.get_identifiers(folder))
    ids = [i for i in ids
           if all(os.path.isfile(os.path.join(folder, i + s))
                  and os.path.getsize(os.path.join(folder, i + s)) > 0
                  for s in (".b2nd", "_seg.b2nd", ".pkl"))]
    if not ids:
        raise SystemExit(f"no complete case in {folder}")
    ds = ds_cls(folder, ids)

    patch = list(cfg.patch_size)
    grid = [p // args.downsample for p in patch]
    oversample = args.oversample if args.oversample is not None else cfg.configuration.get(
        "oversample_foreground_percent", 0.33)
    bs = int(cfg.batch_size)
    dl = nnUNetDataLoader(ds, bs, patch, patch, label_manager,
                          oversample_foreground_percent=float(oversample))

    n_cells = 0
    n_pos = 0
    n_patch = 0
    n_empty = 0
    per_patch = []
    for _ in range(args.batches):
        batch = dl.generate_train_batch()
        seg = batch["target"]
        if isinstance(seg, list):
            seg = seg[0]
        fg = (torch.as_tensor(seg)[:, :1] > 0).float()
        cells = F.adaptive_max_pool3d(fg, tuple(grid))
        n_cells += cells.numel()
        n_pos += int(cells.sum().item())
        for b in range(cells.shape[0]):
            n_patch += 1
            v = float(cells[b].mean().item())
            per_patch.append(v)
            if v == 0.0:
                n_empty += 1

    p = n_pos / max(1, n_cells)
    sp = cfg.spacing
    ml = args.downsample ** 3 * sp[0] * sp[1] * sp[2] / 1000.0
    print(f"cases {len(ids)}   batches {args.batches}   batch size {bs}   "
          f"patches {n_patch}   oversample_fg {oversample}")
    print(f"patch {patch} -> gate grid {grid} ({args.downsample}^3 voxels = {ml:.2f} mL per cell)")
    print(f"positive cells        {n_pos} / {n_cells} = {100 * p:.4f} %")
    print(f"label-empty patches   {n_empty} / {n_patch} = {100 * n_empty / max(1, n_patch):.1f} %")
    print(f"per-patch positive-cell fraction: mean {np.mean(per_patch) * 100:.4f} %  "
          f"median {np.median(per_patch) * 100:.4f} %  max {np.max(per_patch) * 100:.2f} %")
    if p <= 0:
        raise SystemExit("no positive cell seen -- raise --batches")
    print(f"\nN1_AUX_POS_WEIGHT={(1 - p) / p:.2f}")


if __name__ == "__main__":
    main()
