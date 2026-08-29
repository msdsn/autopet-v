"""Build a tiny fake nnU-Net preprocessed dataset to exercise training end to end.

Writes dataset.json, plans, splits and per-case data under $nnUNet_preprocessed,
laid out like the real store (blosc2 fp16 data, uint8 seg; --format npz gives the
numpy variant). PET holds a few bright lesions plus some unlabelled bright blobs
standing in for physiological uptake, so the corruption model has somewhere to
put its hallucinated false positives.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
from typing import List, Tuple

import numpy as np


def _blob(shape, centre, radius, rng) -> np.ndarray:
    grids = np.meshgrid(*[np.arange(s) - c for s, c in zip(shape, centre)], indexing="ij")
    radii = np.maximum(1.0, radius * rng.uniform(0.6, 1.4, size=len(shape)))
    d2 = sum((g / r) ** 2 for g, r in zip(grids, radii))
    return d2 <= 1.0


def make_case(shape: Tuple[int, int, int], rng: np.random.Generator,
              n_lesions: int = 3, n_uptake: int = 4, empty: bool = False,
              n_channels: int = 2):
    z, y, x = shape
    ct = rng.normal(0.0, 1.0, size=shape).astype(np.float32)
    # a smooth body-ish structure so CT is not pure noise
    gz, gy, gx = np.meshgrid(np.linspace(-1, 1, z), np.linspace(-1, 1, y),
                             np.linspace(-1, 1, x), indexing="ij")
    body = ((gy ** 2 + gx ** 2) < 0.8).astype(np.float32)
    ct = ct * 0.3 + body * 1.5

    pet = np.abs(rng.normal(0.0, 0.25, size=shape)).astype(np.float32) + body * 0.4
    seg = np.zeros(shape, dtype=np.int8)

    if not empty:
        for _ in range(int(rng.integers(1, n_lesions + 1))):
            c = [int(rng.integers(6, s - 6)) for s in shape]
            r = float(rng.integers(3, 8))
            m = _blob(shape, c, r, rng)
            seg[m] = 1
            pet[m] += rng.uniform(3.0, 8.0)

    # physiological uptake: bright but NOT labelled (bladder/kidney analogue)
    for _ in range(int(rng.integers(1, n_uptake + 1))):
        c = [int(rng.integers(6, s - 6)) for s in shape]
        r = float(rng.integers(3, 9))
        m = _blob(shape, c, r, rng) & (seg == 0)
        pet[m] += rng.uniform(2.0, 6.0)

    chans = [ct, pet.astype(np.float32)]
    # the real store keeps 4 channels to match the baseline layout: ch2/ch3 are
    # all-zero fg/bg guidance placeholders that the trainer overwrites on the fly
    while len(chans) < n_channels:
        chans.append(np.zeros(shape, dtype=np.float32))
    data = np.stack(chans, axis=0).astype(np.float32)
    return data, seg[None]


def class_locations_from_seg(seg: np.ndarray, max_per_class: int = 5000,
                             rng: np.random.Generator = None):
    """Reproduce nnU-Net's properties['class_locations'] (N x (1+ndim) arrays)."""
    rng = rng or np.random.default_rng(0)
    out = {}
    for cls in (1,):
        idx = np.argwhere(seg[0] == cls)
        if len(idx) == 0:
            out[cls] = np.zeros((0, seg.ndim), dtype=np.int64)
            continue
        if len(idx) > max_per_class:
            idx = idx[rng.choice(len(idx), max_per_class, replace=False)]
        out[cls] = np.concatenate([np.zeros((len(idx), 1), dtype=np.int64), idx], axis=1)
    return out


def write_case(folder: str, name: str, data: np.ndarray, seg: np.ndarray,
               spacing: List[float], rng: np.random.Generator,
               fmt: str = "npz", patch_size=None):
    props = {
        "spacing": list(spacing),
        "shape_before_cropping": list(data.shape[1:]),
        "bbox_used_for_cropping": [[0, int(s)] for s in data.shape[1:]],
        "shape_after_cropping_and_before_resampling": list(data.shape[1:]),
        "sitk_stuff": {"spacing": tuple(spacing[::-1]),
                       "origin": (0.0, 0.0, 0.0),
                       "direction": (1., 0., 0., 0., 1., 0., 0., 0., 1.)},
        "class_locations": class_locations_from_seg(seg, rng=rng),
    }
    if fmt == "b2nd":
        # mirrors the real store: blosc2, fp16 data, uint8 seg, never unpacked
        from nnunetv2.training.dataloading.nnunet_dataset import nnUNetDatasetBlosc2
        from nnunetv2.training.dataloading.nnunet_dataset import comp_blosc2_params
        d = data.astype(np.float16)
        s = seg.astype(np.uint8)
        blocks, chunks = comp_blosc2_params(d.shape, tuple(patch_size or d.shape[1:]))
        blocks_seg, chunks_seg = comp_blosc2_params(s.shape, tuple(patch_size or s.shape[1:]))
        nnUNetDatasetBlosc2.save_case(d, s, props, os.path.join(folder, name),
                                      chunks=chunks, blocks=blocks,
                                      chunks_seg=chunks_seg, blocks_seg=blocks_seg)
        return
    np.savez_compressed(os.path.join(folder, name + ".npz"), data=data, seg=seg)
    with open(os.path.join(folder, name + ".pkl"), "wb") as f:
        pickle.dump(props, f)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--preprocessed-root", required=True, help="$nnUNet_preprocessed")
    ap.add_argument("--dataset-name", default="Dataset999_InteractiveSynth")
    ap.add_argument("--data-identifier", default="nnUNetPlans_3d_fullres")
    ap.add_argument("--plans-name", default="nnUNetPlans_interactive")
    ap.add_argument("--baseline-plans", required=True)
    ap.add_argument("--baseline-dataset-json", required=True)
    ap.add_argument("--n-cases", type=int, default=8)
    ap.add_argument("--shape", nargs=3, type=int, default=[80, 96, 96])
    ap.add_argument("--patch-size", nargs=3, type=int, default=[64, 64, 64])
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--n-empty", type=int, default=2, help="how many lesion-free cases")
    ap.add_argument("--n-channels", type=int, default=4,
                    help="channels on disk (real store = 4: CT, PET, zero, zero)")
    ap.add_argument("--format", choices=["npz", "b2nd"], default="b2nd")
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    try:
        from .make_plans import build_dataset_json, build_plans
    except ImportError:
        from make_plans import build_dataset_json, build_plans  # type: ignore

    base = os.path.join(args.preprocessed_root, args.dataset_name)
    data_dir = os.path.join(base, args.data_identifier)
    os.makedirs(data_dir, exist_ok=True)

    with open(args.baseline_plans) as f:
        base_plans = json.load(f)
    with open(args.baseline_dataset_json) as f:
        base_dj = json.load(f)

    plans = build_plans(base_plans, args.plans_name, ["3d_fullres"],
                        data_identifier=args.data_identifier,
                        patch_size=args.patch_size, batch_size=args.batch_size,
                        dataset_name=args.dataset_name)
    # median image size must not be silly relative to our tiny cases
    plans["configurations"]["3d_fullres"]["median_image_size_in_voxels"] = [float(s) for s in args.shape]
    plans["original_median_shape_after_transp"] = [int(s) for s in args.shape]
    dj = build_dataset_json(base_dj, num_training=args.n_cases)

    with open(os.path.join(base, f"{args.plans_name}.json"), "w") as f:
        json.dump(plans, f, indent=2)
    with open(os.path.join(base, "dataset.json"), "w") as f:
        json.dump(dj, f, indent=2)
    with open(os.path.join(base, "dataset_fingerprint.json"), "w") as f:
        json.dump({"note": "synthetic placeholder"}, f)

    spacing = plans["configurations"]["3d_fullres"]["spacing"]
    rng = np.random.default_rng(args.seed)
    names = []
    for i in range(args.n_cases):
        name = f"case_{i:03d}"
        # interleave the lesion-free cases so both the train and the val split
        # contain one (a val split of only empty cases makes pseudo-Dice a constant 0)
        empty = args.n_empty > 0 and (i % max(1, args.n_cases // args.n_empty)) == 0
        data, seg = make_case(tuple(args.shape), rng, empty=empty,
                              n_channels=args.n_channels)
        write_case(data_dir, name, data, seg, spacing, rng,
                   fmt=args.format, patch_size=args.patch_size)
        names.append(name)
        print(f"{name}: shape={data.shape} label_vox={int(seg.sum())} empty={empty}")

    n_val = max(1, args.n_cases // 4)
    splits = [{"train": names[:-n_val], "val": names[-n_val:]} for _ in range(5)]
    with open(os.path.join(base, "splits_final.json"), "w") as f:
        json.dump(splits, f, indent=2)

    print("\npreprocessed dataset written to", base)
    print("data_identifier folder:", data_dir)
    print("plans:", os.path.join(base, f"{args.plans_name}.json"))


if __name__ == "__main__":
    main()
