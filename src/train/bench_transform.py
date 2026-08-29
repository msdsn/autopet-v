"""Benchmark and correctness check for InteractionSimulationTransform.

Runs at the real 112x160x128 patch size, on synthetic patches or on real
preprocessed ones with --dataset. Checks the channel ranges, that tumor strokes
land inside the label and background strokes outside it, that both sit on the
matching error region, and that no stroke spans more than one axial slice.

    python -m train.bench_transform --patch 112 160 128 --n 30
    python -m train.bench_transform --dataset Dataset998_AutoPETV --p-independent 0
"""

from __future__ import annotations

import argparse
import time
from collections import Counter

import numpy as np
import torch


def make_patch(shape, rng, n_lesions=3, empty=False):
    ct = rng.normal(0, 1, shape).astype(np.float32)
    pet = np.abs(rng.normal(0, 0.3, shape)).astype(np.float32)
    seg = np.zeros(shape, dtype=np.int16)
    if not empty:
        for _ in range(int(rng.integers(1, n_lesions + 1))):
            c = [int(rng.integers(8, s - 8)) for s in shape]
            r = rng.integers(3, 10, size=3)
            g = np.meshgrid(*[np.arange(s) - cc for s, cc in zip(shape, c)], indexing="ij")
            m = sum((gi / ri) ** 2 for gi, ri in zip(g, r)) <= 1.0
            seg[m] = 1
            pet[m] += rng.uniform(3, 8)
    for _ in range(int(rng.integers(1, 5))):  # physiological uptake
        c = [int(rng.integers(8, s - 8)) for s in shape]
        r = rng.integers(3, 10, size=3)
        g = np.meshgrid(*[np.arange(s) - cc for s, cc in zip(shape, c)], indexing="ij")
        m = (sum((gi / ri) ** 2 for gi, ri in zip(g, r)) <= 1.0) & (seg == 0)
        pet[m] += rng.uniform(2, 6)
    img = torch.from_numpy(np.stack([ct, pet], 0))
    return img, torch.from_numpy(seg[None])


# ---------------------------------------------------------------------------
# real patches straight out of the preprocessed store
# ---------------------------------------------------------------------------

def real_patch_source(dataset_name: str, plans_name: str, config: str, fold: int,
                      split: str = "train", trainer_name: str = "nnUNetTrainer_Interactive"):
    """Yield (image, segmentation) tensors of one real preprocessed patch each.

    transforms=None keeps the crop, padding and foreground oversampling that
    training uses, without any augmentation or interaction simulation.
    """
    from nnunetv2.training.dataloading.data_loader import nnUNetDataLoader
    from nnunetv2.training.dataloading.nnunet_dataset import infer_dataset_class

    try:
        from .viz import build_trainer
    except ImportError:
        from viz import build_trainer  # type: ignore

    tr = build_trainer(dataset_name, plans_name, config, fold, trainer_name, device="cpu")
    tr.dataset_class = infer_dataset_class(tr.preprocessed_dataset_folder)
    ds_tr, ds_val = tr.get_tr_and_val_datasets()
    ds = ds_tr if split == "train" else ds_val
    patch = tr.configuration_manager.patch_size
    dl = nnUNetDataLoader(ds, 1, patch, patch, tr.label_manager,
                          oversample_foreground_percent=tr.oversample_foreground_percent,
                          sampling_probabilities=None, pad_sides=None, transforms=None,
                          probabilistic_oversampling=tr.probabilistic_oversampling)
    print(f"real patch source: {dataset_name} fold {fold} {split} "
          f"({len(ds.identifiers)} cases), patch {list(patch)}")
    while True:
        b = next(dl)
        yield b["data"][0], b["target"][0]


def check_sample(out_image, seg, slice_axis: int, strict_label: bool, strict_error: bool) -> dict:
    """Verify one transformed sample. Returns a dict of counters; raises on failure.

    The two strict_* flags turn containment into assertions. They are off where the
    design breaks them on purpose: stroke perturbation can push a voxel off its
    region, and an independent draw uses a different prediction than ch4.
    """
    img = out_image.numpy() if isinstance(out_image, torch.Tensor) else np.asarray(out_image)
    s = seg[0].numpy() if isinstance(seg, torch.Tensor) else np.asarray(seg[0])
    assert img.shape[0] == 5, img.shape
    fg, bg, prev = img[2], img[3], img[4]
    assert 0.0 <= float(fg.min()) and float(fg.max()) <= 1.0 + 1e-6
    assert 0.0 <= float(bg.min()) and float(bg.max()) <= 1.0 + 1e-6
    assert set(np.unique(prev).tolist()) <= {0.0, 1.0}

    L = s > 0
    P = prev > 0.5
    fg_seed = fg >= 1.0 - 1e-6           # the stroke voxels themselves
    bg_seed = bg >= 1.0 - 1e-6
    res = {"n_fg_vox": int(fg_seed.sum()), "n_bg_vox": int(bg_seed.sum()),
           "fg_in_fn": 0, "bg_in_fp": 0, "fg_in_label": 0, "bg_out_label": 0,
           "stroke_groups": 0, "multi_slice_groups": 0}

    # absolute-annotation correctness: holds in BOTH interaction modes
    res["fg_in_label"] = int((fg_seed & L).sum())
    res["bg_out_label"] = int((bg_seed & ~L).sum())
    if strict_label:
        assert res["fg_in_label"] == res["n_fg_vox"], "a tumor stroke voxel is outside the label"
        assert res["bg_out_label"] == res["n_bg_vox"], "a background stroke voxel is inside the label"

    # error containment: only when the scribbles come from the ch4 prediction
    res["fg_in_fn"] = int((fg_seed & ~P & L).sum())
    res["bg_in_fp"] = int((bg_seed & P & ~L).sum())
    if strict_error:
        assert res["fg_in_fn"] == res["n_fg_vox"], "a tumor stroke voxel is not on a FN region"
        assert res["bg_in_fp"] == res["n_bg_vox"], "a background stroke voxel is not on a FP region"

    # axis convention: each connected stroke group lives on exactly one slice
    strokes = fg_seed | bg_seed
    if strokes.any():
        try:
            import cc3d
            cc = cc3d.connected_components(strokes.astype(np.uint8), connectivity=26)
        except Exception:
            from scipy.ndimage import label as ndlabel
            cc, _ = ndlabel(strokes)
        for cid in range(1, int(cc.max()) + 1):
            idx = np.argwhere(cc == cid)
            if not len(idx):
                continue
            res["stroke_groups"] += 1
            if len(np.unique(idx[:, slice_axis])) != 1:
                res["multi_slice_groups"] += 1
        # two strokes drawn on adjacent slices can merge into one group, so this is
        # only an assertion in the controlled (unperturbed) verification run
        if strict_label:
            assert res["multi_slice_groups"] == 0, "a stroke spans more than one axial slice"
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--patch", nargs=3, type=int, default=[112, 160, 128])
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--radius", type=float, default=10.0)
    ap.add_argument("--slice-axis", type=int, default=0)
    ap.add_argument("--empty-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--p-independent", type=float, default=None,
                    help="override p_independent_scribbles; 0 makes the error-containment "
                         "assertion strict, 1 exercises only the Category-2 path")
    ap.add_argument("--p-perturb", type=float, default=None,
                    help="override the ScribblePrompt stroke perturbation probability; "
                         "0 makes the label-containment assertion strict")
    ap.add_argument("--dataset", default=None,
                    help="pull REAL patches from $nnUNet_preprocessed/<dataset> instead of "
                         "generating synthetic ones")
    ap.add_argument("--plans", default="nnUNetPlans_interactive")
    ap.add_argument("--config", default="3d_fullres")
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--split", default="train", choices=["train", "val"])
    ap.add_argument("--trainer", default="nnUNetTrainer_Interactive")
    args = ap.parse_args()

    try:
        from .interaction_transform import InteractionSimulationTransform
    except ImportError:
        from interaction_transform import InteractionSimulationTransform  # type: ignore

    kw = {}
    if args.p_independent is not None:
        kw["p_independent_scribbles"] = float(args.p_independent)
    if args.p_perturb is not None:
        kw["p_perturb"] = float(args.p_perturb)
    t = InteractionSimulationTransform(radius=args.radius, slice_axis=args.slice_axis,
                                       seed=args.seed, **kw)
    rng = np.random.default_rng(args.seed)
    shape = tuple(args.patch)

    if args.dataset:
        src = real_patch_source(args.dataset, args.plans, args.config, args.fold,
                                args.split, args.trainer)
        next_patch = lambda: next(src)                       # noqa: E731
    else:
        def next_patch():
            return make_patch(shape, rng, empty=rng.random() < args.empty_frac)

    # perturbed strokes may leave the region they were drawn on: only assert
    # containment when perturbation is explicitly switched off
    strict_label = (args.p_perturb is not None) and (args.p_perturb == 0.0)

    times, ks, iters, kinds = [], [], [], Counter()
    n_indep = 0
    tot = Counter()
    # warm up (kernel cache, cc3d import, official simulator import)
    img, seg = next_patch()
    t(**{"image": img.clone(), "segmentation": seg.clone()})
    shape = tuple(img.shape[1:])

    for i in range(args.n):
        img, seg = next_patch()
        d = {"image": img, "segmentation": seg.clone()}
        t0 = time.perf_counter()
        out = t(**d)
        times.append(time.perf_counter() - t0)
        info = t.last_info
        ks.append(info["k"]); iters.append(info["n_iters"])
        kinds["fg>0" if info["n_fg"] else "fg=0"] += 1
        kinds["bg>0" if info["n_bg"] else "bg=0"] += 1
        n_indep += int(info.get("independent", False))
        # a per-sample "independent" draw relaxes error containment for THAT sample only
        chk = check_sample(out["image"], seg, args.slice_axis,
                           strict_label=strict_label,
                           strict_error=strict_label and not bool(info.get("independent", False)))
        for k, v in chk.items():
            tot[k] += v

    a = np.array(times) * 1000
    print(f"patch={shape}  n={args.n}  source={'real:' + args.dataset if args.dataset else 'synthetic'}")
    print(f"transform time  mean={a.mean():.1f} ms  median={np.median(a):.1f} ms  "
          f"p90={np.percentile(a, 90):.1f} ms  max={a.max():.1f} ms")
    print("k histogram (k=0..5):", np.bincount(ks, minlength=6).tolist())
    print("realised interaction iterations:", np.bincount(iters, minlength=6).tolist())
    print(f"independent (Category-2) scribble samples: {n_indep}/{args.n}")
    print("scribble presence:", dict(kinds))
    print(f"tumor stroke voxels      {tot['n_fg_vox']}  "
          f"inside label {tot['fg_in_label']}  on a FN region {tot['fg_in_fn']}")
    print(f"background stroke voxels {tot['n_bg_vox']}  "
          f"outside label {tot['bg_out_label']}  on a FP region {tot['bg_in_fp']}")
    print(f"stroke groups {tot['stroke_groups']}, spanning >1 axial slice: "
          f"{tot['multi_slice_groups']}")
    mem = np.prod(shape) * 4 * 3 / 1e6
    print(f"extra memory per sample: {mem:.1f} MB (3 float32 channels)")


if __name__ == "__main__":
    main()
