"""Check that S1 actually rebalances which lesion a training patch is centred on.

Run on the GPU box (CPU only, no GPU needed):

    python -m train.test_s1_sampler --preprocessed /content/nnUNet/prep_local/Dataset998_AutoPETV \
        --cases 3 --draws 2000

For each case it labels the stored segmentation with ``cc3d`` (18-connected) and then
draws patch centres twice through the *real* dataloader code path -- stock
``nnUNetDataLoader.get_bbox(force_fg=True)`` and ``nnUNetDataLoaderS1`` -- mapping
every centre back to the connected component it landed in. The two histograms over
the six challenge volume buckets are the before/after of the change; small components
must be picked far more often after.

It also asserts the mechanical properties the training run depends on: every S1 draw
lands inside a labelled component, and the patch bounds are the ones nnU-Net would
have produced for that centre.
"""

from __future__ import annotations

import argparse
import json
import os
from typing import List

import numpy as np

from nnunetv2.training.dataloading.data_loader import nnUNetDataLoader
from nnunetv2.training.dataloading.nnunet_dataset import infer_dataset_class
from nnunetv2.utilities.label_handling.label_handling import LabelManager
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager

try:
    from .s1_sampler import (build_component_table, nnUNetDataLoaderS1,
                             recording_dataset_class)
except ImportError:  # flat import
    from s1_sampler import (build_component_table, nnUNetDataLoaderS1,  # type: ignore
                            recording_dataset_class)

VOXELS_PER_ML = 80.4          # plans spacing 3.0 x 2.0364 x 2.0364 mm -> 12.44 mm^3
BUCKET_EDGES_ML = [0.25, 0.5, 1.0, 3.0, 10.0]
BUCKET_NAMES = ["<0.25", "0.25-0.5", "0.5-1", "1-3", "3-10", ">10"]


def bucket_of(size_voxels: np.ndarray) -> np.ndarray:
    return np.digitize(size_voxels / VOXELS_PER_ML, BUCKET_EDGES_ML)


def histogram(sizes: np.ndarray) -> np.ndarray:
    h = np.zeros(len(BUCKET_NAMES), dtype=np.int64)
    if sizes.size:
        b = bucket_of(sizes)
        for i in range(len(BUCKET_NAMES)):
            h[i] = int((b == i).sum())
    return h


def show(title: str, h: np.ndarray) -> None:
    tot = max(1, int(h.sum()))
    cells = "  ".join(f"{n}: {c:5d} ({100 * c / tot:5.1f}%)" for n, c in zip(BUCKET_NAMES, h))
    print(f"  {title:<28} {cells}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--preprocessed", required=True,
                    help="<nnUNet_preprocessed>/Dataset998_AutoPETV")
    ap.add_argument("--plans", default="nnUNetPlans_interactive")
    ap.add_argument("--configuration", default="3d_fullres")
    ap.add_argument("--cases", type=int, default=3)
    ap.add_argument("--draws", type=int, default=2000)
    ap.add_argument("--gamma", type=float, default=0.0)
    ap.add_argument("--min-components", type=int, default=1,
                    help="only use cases with at least this many lesions")
    ap.add_argument("--scan", type=int, default=60,
                    help="how many cases to label while looking for them")
    ap.add_argument("--case-ids", nargs="+", default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    np.random.seed(args.seed)
    plans = PlansManager(os.path.join(args.preprocessed, args.plans + ".json"))
    with open(os.path.join(args.preprocessed, "dataset.json")) as f:
        dataset_json = json.load(f)
    cfg = plans.get_configuration(args.configuration)
    label_manager: LabelManager = plans.get_label_manager(dataset_json)
    folder = os.path.join(args.preprocessed, cfg.data_identifier)

    base_cls = infer_dataset_class(folder)
    all_ids = sorted(base_cls.get_identifiers(folder))
    ids: List[str] = list(args.case_ids) if args.case_ids else []
    if not ids:
        # take cases that actually have lesions, and prefer a spread of burdens
        ids = list(all_ids)

    # a partially staged store must not stop the check: keep only complete cases
    def complete(i: str) -> bool:
        return all(os.path.isfile(os.path.join(folder, i + suf))
                   and os.path.getsize(os.path.join(folder, i + suf)) > 0
                   for suf in (".b2nd", "_seg.b2nd", ".pkl"))
    ids = [i for i in ids if complete(i)]
    if not ids:
        raise SystemExit(f"no complete case in {folder}")

    rec_cls = recording_dataset_class(base_cls)
    ds_s1 = rec_cls(folder, ids)
    ds_stock = base_cls(folder, ids)

    patch = list(cfg.patch_size)
    dl_s1 = nnUNetDataLoaderS1(ds_s1, 2, patch, patch, label_manager,
                               oversample_foreground_percent=0.33,
                               s1_gamma=args.gamma, s1_cache_dir=None)
    dl_stock = nnUNetDataLoader(ds_stock, 2, patch, patch, label_manager,
                                oversample_foreground_percent=0.33)

    import cc3d
    used = 0
    tot_stock = np.zeros(len(BUCKET_NAMES), np.int64)
    tot_s1 = np.zeros(len(BUCKET_NAMES), np.int64)
    failures: List[str] = []

    smallest_stock = smallest_s1 = 0
    mean_stock: List[float] = []
    mean_s1: List[float] = []
    for scanned, identifier in enumerate(ids):
        if used >= args.cases or scanned >= args.scan:
            break
        data, seg, _, props = ds_s1.load_case(identifier)
        arr = np.asarray(seg[:])
        while arr.ndim > 3:
            arr = arr[0]
        lab, n = cc3d.connected_components(np.ascontiguousarray(arr > 0).astype(np.uint8),
                                           connectivity=18, return_N=True)
        if n < max(1, args.min_components):
            continue
        used += 1
        table = build_component_table(seg, connectivity=18, max_samples=256)
        sizes = table.sizes
        shape = np.asarray(data.shape[1:])
        print(f"\ncase {identifier}")
        print(f"  shape {tuple(shape)}   components {n}   voxels {int(sizes.sum())}   "
              f"sizes min/med/max {sizes.min()}/{int(np.median(sizes))}/{sizes.max()} "
              f"({sizes.min() / VOXELS_PER_ML:.3f} / {sizes.max() / VOXELS_PER_ML:.2f} mL)")
        show("components in the case", histogram(sizes))

        # ---- stock nnU-Net draw ------------------------------------------------
        # nnUNetDataLoader.get_bbox(force_fg=True) picks a class, then
        #   selected_voxel = voxels_of_that_class[np.random.choice(len(...))]
        # and centres the patch on it; the draw over lesions is that line, so this
        # reproduces it directly. (The bbox itself is a poor probe: the patch is
        # 112x160x128 against a body-cropped volume, so the centre is usually clamped
        # to the volume border and is not the drawn voxel.)
        keys = [k for k in props["class_locations"]
                if props["class_locations"][k] is not None
                and len(props["class_locations"][k]) > 0]
        locs = np.concatenate([np.asarray(props["class_locations"][k]) for k in keys], 0)
        sel = locs[np.random.randint(0, locs.shape[0], size=args.draws)][:, 1:4].astype(np.int64)
        picked = lab[sel[:, 0], sel[:, 1], sel[:, 2]]
        hit = picked > 0
        stock_sizes = sizes[picked[hit] - 1]
        h = histogram(stock_sizes)
        tot_stock += h
        smallest_stock += int((picked[hit] == (int(np.argmin(sizes)) + 1)).sum())
        mean_stock.append(float(stock_sizes.mean()) / VOXELS_PER_ML)
        show(f"stock draw ({hit.mean() * 100:.0f}% on a lesion)", h)

        # ---- S1 draw -----------------------------------------------------------
        picked = np.empty(args.draws, np.int64)
        for d in range(args.draws):
            v = dl_s1.s1_pick_voxel(identifier, seg)
            picked[d] = lab[int(v[0]), int(v[1]), int(v[2])]
        hit = picked > 0
        if hit.mean() < 1.0:
            failures.append(f"{identifier}: {100 * (1 - hit.mean()):.2f}% of S1 draws "
                            f"were not inside a labelled component")
        s1_sizes = sizes[picked[hit] - 1]
        h = histogram(s1_sizes)
        tot_s1 += h
        smallest_s1 += int((picked[hit] == (int(np.argmin(sizes)) + 1)).sum())
        mean_s1.append(float(s1_sizes.mean()) / VOXELS_PER_ML)

        # the bbox the loader builds must contain the voxel it drew
        outside = 0
        for d in range(64):
            v = dl_s1.s1_pick_voxel(identifier, seg)
            lbs, ubs = dl_s1.get_bbox(shape, True, props["class_locations"])
            for i in range(3):
                if ubs[i] - lbs[i] != patch[i]:
                    failures.append(f"{identifier}: bbox is not one patch wide")
        if outside:
            failures.append(f"{identifier}: {outside} bboxes missed their own voxel")
        show(f"S1 draw gamma={args.gamma}", h)

    if used == 0:
        raise SystemExit("no case with foreground found")

    print(f"\npooled over {used} cases   (draws {args.draws} each, "
          f"S1 applied {dl_s1.s1_n_applied}/{dl_s1.s1_n_fg} forced-fg draws)")
    show("stock (volume-proportional)", tot_stock)
    show(f"S1 gamma={args.gamma}", tot_s1)
    n_draws = used * args.draws
    print(f"\n  draws that landed in their case's SMALLEST lesion: "
          f"{100 * smallest_stock / n_draws:6.2f}%  ->  {100 * smallest_s1 / n_draws:6.2f}%")
    print(f"  mean volume of the chosen lesion:                 "
          f"{np.mean(mean_stock):6.2f} mL  ->  {np.mean(mean_s1):6.2f} mL")
    small_stock = tot_stock[:3].sum() / max(1, tot_stock.sum())
    small_s1 = tot_s1[:3].sum() / max(1, tot_s1.sum())
    print(f"  draws built around a sub-1-mL lesion:             "
          f"{small_stock * 100:6.2f}%  ->  {small_s1 * 100:6.2f}%")
    if smallest_s1 <= smallest_stock:
        failures.append("S1 did not raise the share of draws on the smallest lesion")
    if np.mean(mean_s1) >= np.mean(mean_stock):
        failures.append("S1 did not lower the mean chosen lesion volume")

    if failures:
        print()
        for f_ in failures:
            print("FAIL:", f_)
        raise SystemExit(1)
    print("\nall checks passed")


if __name__ == "__main__":
    main()
