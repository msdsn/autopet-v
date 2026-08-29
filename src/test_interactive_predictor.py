#!/usr/bin/env python3
"""Check that `InteractiveNNUNetPredictor` feeds the network what training produced.

Covers the guidance encoding against `InteractionSimulationTransform`, the mapping from
nibabel scribble indices to the preprocessed grid against nnU-Net's own resampler, the
five channels' ranges, `cache_state_key`, and two evalset cases through the real loop.

    python3 test_interactive_predictor.py --model_folder <ckpt dir> \
        --input_cases /content/drive/MyDrive/autoPET/evalset \
        --image_dir .../imagesTr --label_dir .../labelsTr --repo /content/autoPETV
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time

import numpy as np
import nibabel as nib

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
for _v in ("nnUNet_raw", "nnUNet_preprocessed", "nnUNet_results"):
    os.environ.setdefault(_v, "/tmp/" + _v)

import interactive_eval as ie                                    # noqa: E402
from predictor import InteractiveNNUNetPredictor                  # noqa: E402

OK, FAIL = [], []


def check(cond, msg):
    (OK if cond else FAIL).append(msg)
    print(("  ok   " if cond else "  FAIL ") + msg, flush=True)


# ---------------------------------------------------------------------------
def test_encoding_matches_training_transform(radius: float):
    """Drive the real training transform with known coordinates and compare."""
    print("\n=== 1. encoding vs the training transform ===")
    import torch
    from train.interaction_transform import InteractionSimulationTransform
    from train import interaction_transform as itmod
    from train.guidance import guidance_map_from_coords

    shape = (24, 40, 32)                      # a stand-in patch on the plans grid
    fg = [[3, 5, 7], [3, 6, 7], [4, 6, 8], [20, 35, 28]]
    bg = [[10, 10, 10], [11, 12, 13]]
    prev = np.zeros(shape, dtype=np.uint8)
    prev[5:9, 10:16, 12:18] = 1

    class _Inter:                              # what simulate_interaction_sequence returns
        fg_coords, bg_coords, n_iters = fg, bg, 3

    orig_sim, orig_prev = itmod.simulate_interaction_sequence, itmod.make_fake_prev_prediction
    itmod.simulate_interaction_sequence = lambda *a, **k: _Inter()
    itmod.make_fake_prev_prediction = lambda *a, **k: prev
    try:
        tr = InteractionSimulationTransform(radius=radius, k_probs=(0.0, 1.0), seed=0)
        image = torch.zeros((2, *shape), dtype=torch.float32)
        seg = torch.zeros((1, *shape), dtype=torch.float32)
        seg[0, 6, 12, 14] = 1
        out = tr.apply({"image": image, "segmentation": seg})["image"].numpy()
    finally:
        itmod.simulate_interaction_sequence = orig_sim
        itmod.make_fake_prev_prediction = orig_prev

    check(out.shape[0] == 5, f"training transform produces 5 channels (got {out.shape[0]})")
    check(tr.radius == radius and tr.spacing is None,
          f"transform uses radius={tr.radius} voxels, spacing={tr.spacing} (isotropic)")

    mine_fg = guidance_map_from_coords(shape, fg, radius)
    mine_bg = guidance_map_from_coords(shape, bg, radius)
    check(np.array_equal(out[2], mine_fg),
          f"tumor guidance identical to the training transform (max {out[2].max():.4f}, "
          f"{int((out[2] > 0).sum())} nonzero voxels)")
    check(np.array_equal(out[3], mine_bg), "background guidance identical")
    check(np.array_equal(out[4], prev.astype(out.dtype)), "previous-prediction channel identical")
    check(float(out[2].max()) == 1.0 and float(out[2].min()) == 0.0,
          "guidance is in [0,1] and reaches 1 at a scribble voxel")


# ---------------------------------------------------------------------------
def test_grid_mapping(pred, case, radius):
    """Compare our index mapping against nnU-Net's own resampling of the same point."""
    print("\n=== 2. coordinate mapping onto the preprocessed grid ===")
    p = pred._ensure_predictor()
    cm = p.configuration_manager
    pet_img = nib.load(case.pet)
    ct = np.asanyarray(nib.load(case.ct).dataobj)
    pet = np.asanyarray(pet_img.dataobj)
    gt = np.asanyarray(nib.load(case.label).dataobj)
    spacing = tuple(float(z) for z in pet_img.header.get_zooms()[:3])

    fg = np.argwhere(gt > 0)
    pts = [[int(a), int(b), int(c)] for a, b, c in fg[:: max(1, len(fg) // 6)][:6]] \
        if len(fg) else [[int(s // 2) for s in gt.shape]]
    geom = pred._geometry(ct, pet, spacing, {"tumor": pts, "background": []},
                          case.tag, include_clicks_in_bbox=False)
    mapped, dropped = pred.map_coords_to_grid(pts, geom["bbox"], geom["shape_after_crop"],
                                              geom["new_shape"])
    check(dropped == 0, f"all {len(pts)} scribble voxels land inside the preprocessed grid")
    print(f"       crop {geom['shape_after_crop']} -> grid {geom['new_shape']}  "
          f"(spacing {geom['orig_spacing']} -> {list(geom['target_spacing'])})")

    # independent reference: one-hot the point in the ORIGINAL grid, crop, and let
    # nnU-Net's segmentation resampler move it to the preprocessed grid
    worst = 0
    for pt, m in zip(pts, mapped):
        one = np.zeros(tuple(gt.shape[::-1]), dtype=np.uint8)
        one[pt[2], pt[1], pt[0]] = 1
        one_c = one[geom["slicer"]]
        if tuple(one_c.shape) == tuple(geom["new_shape"]):
            ref = np.argwhere(one_c > 0)
        else:
            r = cm.resampling_fn_seg(one_c[None], geom["new_shape"], geom["orig_spacing"],
                                     geom["target_spacing"])[0]
            ref = np.argwhere(r > 0)
        if len(ref) == 0:
            continue                       # a single voxel can vanish when downsampling
        d = int(np.abs(ref - np.asarray(m)).max(axis=1).min())
        worst = max(worst, d)
    check(worst <= 1,
          f"our mapped index agrees with nnU-Net's resampling of the same point "
          f"(worst disagreement {worst} voxel)")
    return geom, pts, ct, pet, gt, spacing


# ---------------------------------------------------------------------------
def test_channels(pred, geom, pts, ct, pet, gt, spacing, case, radius):
    print("\n=== 3. the tensor handed to the network ===")
    built = {}
    real_infer = pred._infer_and_export

    def spy(data, g, rp, p_, pm, cm):
        built["data"] = np.array(data, copy=True)
        return real_infer(data, g, rp, p_, pm, cm)

    pred._infer_and_export = spy
    try:
        prev = (gt > 0).astype(np.uint8)      # stand in for a previous final mask
        mask = pred.predict(ct, pet, spacing, {"tumor": pts, "background": []},
                            prev_pred=prev, case_name=case.tag,
                            affine=nib.load(case.pet).affine)
    finally:
        pred._infer_and_export = real_infer

    d = built["data"]
    check(d.shape[0] == 5, f"5 input channels (got {d.shape[0]}), grid {d.shape[1:]}")
    check(0.0 <= d[2].min() and d[2].max() <= 1.0 and d[2].max() > 0.99,
          f"tumor guidance in [0,1], peak {d[2].max():.4f}")
    check(d[3].max() == 0.0, "background guidance is all zero when there are no bg scribbles")
    check(set(np.unique(d[4]).tolist()) <= {0.0, 1.0},
          f"previous-prediction channel is binary (values {np.unique(d[4])[:4]})")
    check(d[4].sum() > 0, f"previous-prediction channel is populated ({int(d[4].sum())} voxels)")
    check(mask.shape == pet.shape and mask.dtype == np.uint8,
          f"mask {mask.shape} uint8, {int(mask.sum())} foreground voxels")
    print(f"       guidance info: {pred.last_guidance_info}")
    print(f"       timings: {pred.last_timings}")

    # and with no previous mask at all
    pred._infer_and_export = spy
    try:
        pred.predict(ct, pet, spacing, {"tumor": [], "background": []},
                     prev_pred=None, case_name=case.tag, affine=nib.load(case.pet).affine)
    finally:
        pred._infer_and_export = real_infer
    d0 = built["data"]
    check(d0[2].max() == 0 and d0[3].max() == 0 and d0[4].max() == 0,
          "iteration 0 (no scribbles, no previous mask) -> channels 2-4 all zero")


# ---------------------------------------------------------------------------
def test_cache_key(pred, shape):
    print("\n=== 4. cache state key ===")
    a = np.zeros(shape, dtype=np.uint8); a[2:5, 2:5, 2:5] = 1
    b = np.zeros(shape, dtype=np.uint8); b[2:5, 2:5, 2:6] = 1
    check(pred.cache_state_key(None) == pred.cache_state_key(np.zeros(shape, np.uint8)) == "prev0",
          "no previous mask and an empty previous mask share the key 'prev0'")
    check(pred.cache_state_key(a) == pred.cache_state_key(a.copy()),
          "the key is stable for the same mask")
    check(pred.cache_state_key(a) != pred.cache_state_key(b),
          "the key separates two previous masks that differ by one voxel")
    check(pred.stateless is True and pred.cache_state_key(a) != "prev0",
          "the predictor is a pure function whose key covers prev_pred")


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_folder", default=None)
    ap.add_argument("--checkpoint", default="checkpoint_best.pth")
    ap.add_argument("--input_cases", default="/content/drive/MyDrive/autoPET/evalset")
    ap.add_argument("--image_dir", default=None)
    ap.add_argument("--label_dir", default=None)
    ap.add_argument("--repo", default="/content/autoPETV")
    ap.add_argument("--out_dir", default="/content/work/interactive_test")
    ap.add_argument("--n_cases", type=int, default=2)
    ap.add_argument("--radius", type=float, default=None)
    ap.add_argument("--skip_loop", action="store_true")
    args = ap.parse_args()

    radius = float(args.radius if args.radius is not None
                   else os.environ.get("nnUNet_interactive_radius", 10.0))
    test_encoding_matches_training_transform(radius)

    cases = ie.discover_cases(args.input_cases, args.image_dir, args.label_dir)
    cases = cases[:args.n_cases]
    print(f"\ncases: {[c.tag[:40] for c in cases]}")

    pred = InteractiveNNUNetPredictor(model_folder=args.model_folder,
                                      checkpoint_name=args.checkpoint,
                                      guidance_radius=radius)
    t0 = time.time()
    pred.warmup()
    print(f"model loaded in {time.time() - t0:.1f}s from {pred.model_folder}")
    p = pred._ensure_predictor()
    print(f"  normalization: {p.configuration_manager.normalization_schemes}")
    print(f"  channels: {p.dataset_json['channel_names']}  "
          f"mirror axes: {getattr(p, 'allowed_mirroring_axes', None)}  "
          f"use_mirroring: {p.use_mirroring}")


    geom, pts, ct, pet, gt, spacing = test_grid_mapping(pred, cases[0], radius)
    # `orig_spacing` comes from a float32 NIfTI header and `target_spacing` from the
    # plans as float64, so they must be compared with a tolerance, not by rounding.
    if np.allclose(geom["orig_spacing"], geom["target_spacing"], atol=1e-3):
        # that case was already at the plans spacing, so the mapping was the identity and
        # proves nothing. Find one that nnU-Net actually has to resample.
        tgt = np.asarray(geom["target_spacing"], dtype=np.float64)
        alt = None
        for c in ie.discover_cases(args.input_cases, args.image_dir, args.label_dir):
            z = np.asarray(list(nib.load(c.pet).header.get_zooms()[:3])[::-1], dtype=np.float64)
            if not np.allclose(z, tgt, atol=1e-3):
                alt = c
                break
        if alt is None:
            print("  (no case with a different spacing available -- mapping untested "
                  "under real resampling)")
        else:
            print(f"\n=== 2b. same check on a case that IS resampled: "
                  f"{alt.tag[:44]} ===")
            test_grid_mapping(pred, alt, radius)
    test_channels(pred, geom, pts, ct, pet, gt, spacing, cases[0], radius)
    test_cache_key(pred, (8, 8, 8))
    pred.close()

    if not args.skip_loop:
        print(f"\n=== 5. end to end: {len(cases)} cases x 6 iterations ===")
        shutil.rmtree(args.out_dir, ignore_errors=True)
        argv = ["--input_cases", args.input_cases, "--out_dir", args.out_dir,
                "--repo", args.repo, "--predictor", "interactive_nnunet",
                "--checkpoint", args.checkpoint, "--strategy", "all",
                "--save_predictions", "none", "--cases"] + [c.tag for c in cases]
        if args.image_dir:
            argv += ["--image_dir", args.image_dir]
        if args.label_dir:
            argv += ["--label_dir", args.label_dir]
        if args.model_folder:
            argv += ["--model_folder", args.model_folder]
        if args.radius is not None:
            argv += ["--guidance_radius", str(args.radius)]
        t0 = time.time()
        s = ie.main(argv)
        el = time.time() - t0
        sc = json.load(open(os.path.join(args.out_dir, "metric_scores.json")))
        for tag, recs in sc.items():
            print(f"  {tag[:44]:46s} dice {[round(r['dice'], 4) for r in recs]}")
            print(f"  {'':46s} dmm  {[round(r['dmm'], 4) for r in recs]}")
        ci = json.load(open(os.path.join(args.out_dir, "case_info.json")))
        its = [x for v in ci.values() for x, r in zip(v["iter_seconds"], v["reused"])
               if r is None and x > 0]
        check(all(len(v) == 6 for v in sc.values()), "6 iterations per case")
        check(len(its) > 0, f"median model iteration {np.median(its):.1f}s "
                            f"(total {el:.0f}s for {len(sc)} cases)")
        print(f"  AUC-Dice {s['mean_auc_dice']:.4f}  AUC-DMM {s['mean_auc_dmm']:.4f} (max 5.0)")

    print(f"\n{len(OK)} passed, {len(FAIL)} failed")
    if FAIL:
        for f in FAIL:
            print("  FAILED: " + f)
        raise SystemExit(1)
    print("ALL INTERACTIVE PREDICTOR TESTS PASSED")


if __name__ == "__main__":
    main()
