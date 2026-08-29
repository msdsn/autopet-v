#!/usr/bin/env python3
"""Break one baseline nnU-Net iteration into its stages and time each of them.

Covers writing and reading the input NIfTIs, CPU preprocessing, the sliding window, the
export back to the original spacing, and the segmentation round trip.  Also checks the
assumptions the fast path relies on: SimpleITKIO's array order, the crop bbox being
independent of the guidance channels, and per-channel normalization and resampling.

  python3 profile_predictor.py --input_cases /content/work/testcase \
      --case psma_0198cdca94fbb95f_2020-05-09 --out /content/work/profile.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from contextlib import contextmanager

import numpy as np
import nibabel as nib

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
for var in ("nnUNet_raw", "nnUNet_preprocessed", "nnUNet_results"):
    os.environ.setdefault(var, "/tmp/" + var)

import interactive_eval as ie
from predictor import BaselineNNUNetPredictor, generate_gaussian_heatmap

T = {}


@contextmanager
def timed(name, sync=False):
    if sync:
        import torch
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    yield
    if sync:
        import torch
        torch.cuda.synchronize()
    T[name] = round(time.perf_counter() - t0, 3)
    print(f"    [{name}] {T[name]:.3f}s", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_cases", default="/content/work/testcase")
    ap.add_argument("--case", default=None, help="case stem (no _0000)")
    ap.add_argument("--model_folder", default=None)
    ap.add_argument("--work", default="/content/work/profile_tmp")
    ap.add_argument("--out", default="/content/work/profile.json")
    ap.add_argument("--n_clicks", type=int, default=40)
    args = ap.parse_args()

    import torch
    from nnunetv2.imageio.simpleitk_reader_writer import SimpleITKIO
    from nnunetv2.preprocessing.preprocessors.default_preprocessor import DefaultPreprocessor
    from nnunetv2.inference.export_prediction import (
        convert_predicted_logits_to_segmentation_with_correct_shape,
    )

    cases = ie.discover_cases(args.input_cases)
    if args.case:
        cases = [c for c in cases if c.stem == args.case or c.tag == args.case]
    case = cases[0]
    print(f"case: {case.tag}")

    ct_img, pet_img = nib.load(case.ct), nib.load(case.pet)
    ct = np.asanyarray(ct_img.dataobj)
    pet = np.asanyarray(pet_img.dataobj)
    gt = np.asanyarray(nib.load(case.label).dataobj)
    shape = pet.shape
    print(f"shape (nibabel) {shape}  zooms {pet_img.header.get_zooms()[:3]}")

    fg_pts = [[int(a), int(b), int(c)] for a, b, c in np.argwhere(gt > 0)[:: max(1, int(gt.sum()) // args.n_clicks)][:args.n_clicks]]
    bg_pts = [[int(a), int(b), int(c)] for a, b, c in np.argwhere((gt == 0) & (pet > 2))[:args.n_clicks]]
    print(f"clicks: {len(fg_pts)} fg / {len(bg_pts)} bg")

    os.makedirs(args.work, exist_ok=True)
    files = [os.path.join(args.work, f"case_000{i}.nii.gz") for i in range(4)]

    pred = BaselineNNUNetPredictor(model_folder=args.model_folder)
    with timed("model_load"):
        p = pred._ensure_predictor()

    # ---------------- (a) write 4 NIfTIs -------------------------------
    aff = pet_img.affine
    with timed("a_write_4_niftis"):
        for arr, f in ((ct, files[0]), (pet, files[1])):
            a = np.asarray(arr)
            nib.save(nib.Nifti1Image(a.astype(np.float32) if a.dtype == np.float64 else a, aff), f)
        nib.save(nib.Nifti1Image(generate_gaussian_heatmap(fg_pts, shape, 0.0), aff), files[2])
        nib.save(nib.Nifti1Image(generate_gaussian_heatmap(bg_pts, shape, 0.0), aff), files[3])

    # ---------------- (b) nnU-Net reads them ---------------------------
    rw = SimpleITKIO()
    with timed("b_sitk_read_4"):
        data_file, props_file = rw.read_images(tuple(files))
    print(f"    sitk array {data_file.shape} dtype {data_file.dtype} props_spacing {props_file['spacing']}")

    # ---- assumption 1: sitk array == nibabel array transposed ---------
    ok_axes = np.array_equal(data_file[0], np.asarray(ct, dtype=data_file.dtype).transpose(2, 1, 0))
    ok_axes &= np.allclose(data_file[1], np.asarray(pet, dtype=data_file.dtype).transpose(2, 1, 0))
    print(f"    ASSUMPTION sitk==nib.transpose(2,1,0): {ok_axes}")
    ok_spacing = np.allclose(props_file["spacing"], list(pet_img.header.get_zooms()[:3])[::-1])
    print(f"    ASSUMPTION spacing == reversed nibabel zooms: {ok_spacing}")

    # ---------------- (c) preprocessing --------------------------------
    pp = DefaultPreprocessor(verbose=False)
    props = dict(props_file)
    with timed("c_preprocess_total"):
        data_pp, seg_pp, props = pp.run_case_npy(data_file, None, props, p.plans_manager,
                                          p.configuration_manager, p.dataset_json)
    print(f"    preprocessed {data_pp.shape} ({data_pp.nbytes/1e6:.0f} MB)  bbox {props['bbox_used_for_cropping']}")

    # ---- component breakdown of (c), replicated step by step ----------
    from nnunetv2.preprocessing.cropping.cropping import crop_to_nonzero
    from nnunetv2.preprocessing.resampling.default_resampling import compute_new_shape
    d = data_file.astype(np.float32)
    with timed("c1_crop_to_nonzero"):
        d_c, seg_c, bbox = crop_to_nonzero(d, None)
    orig_spacing = list(props_file["spacing"])
    target_spacing = p.configuration_manager.spacing
    new_shape = compute_new_shape(d_c.shape[1:], orig_spacing, target_spacing)
    with timed("c2_normalize_4ch"):
        d_n = pp._normalize(d_c, seg_c, p.configuration_manager,
                            p.plans_manager.foreground_intensity_properties_per_channel)
    with timed("c3_resample_data_4ch"):
        d_r = p.configuration_manager.resampling_fn_data(d_n, new_shape, orig_spacing, target_spacing)
    with timed("c4_resample_seg_1ch"):
        _ = p.configuration_manager.resampling_fn_seg(seg_c, new_shape, orig_spacing, target_spacing)
    with timed("c3b_resample_data_2ch_only"):
        _ = p.configuration_manager.resampling_fn_data(d_n[2:], new_shape, orig_spacing, target_spacing)
    print(f"    resample {tuple(d_c.shape[1:])} -> {tuple(new_shape)}  fn={p.configuration_manager.resampling_fn_data}")
    print(f"    step-by-step == run_case_npy: {np.allclose(d_r, data_pp)}")

    # ---- assumption 2: bbox independent of the guidance channels ------
    _, _, bbox_no_clicks = crop_to_nonzero(np.stack([d[0], d[1], np.zeros_like(d[0]), np.zeros_like(d[0])]), None)
    print(f"    ASSUMPTION bbox(with clicks) == bbox(no clicks): {bbox == bbox_no_clicks}  {bbox} vs {bbox_no_clicks}")
    _, _, bbox_ctpet = crop_to_nonzero(d[:2], None)
    print(f"    ASSUMPTION bbox(4ch) == bbox(CT,PET only): {bbox == bbox_ctpet}")

    # ---- alternative resamplers ---------------------------------------
    from concurrent.futures import ThreadPoolExecutor
    fn = p.configuration_manager.resampling_fn_data
    with timed("c5_resample_4ch_threaded"):
        with ThreadPoolExecutor(max_workers=4) as ex:
            outs = list(ex.map(lambda c: fn(d_n[c:c + 1], new_shape, orig_spacing, target_spacing), range(4)))
        d_thr = np.concatenate(outs, 0)
    print(f"    threaded == serial: {np.array_equal(d_thr, d_r)}")

    from nnunetv2.preprocessing.resampling.resample_torch import resample_torch_fornnunet
    for dev in ("cpu", "cuda"):
        with timed(f"c6_resample_4ch_torch_{dev}", sync=(dev == "cuda")):
            d_t = resample_torch_fornnunet(torch.from_numpy(d_n), new_shape, orig_spacing, target_spacing,
                                           is_seg=False, device=torch.device(dev))
        d_t = d_t.cpu().numpy() if hasattr(d_t, "cpu") else np.asarray(d_t)
        print(f"    torch({dev}) vs scipy: max|diff| CT {np.abs(d_t[0]-d_r[0]).max():.4f} "
              f"PET {np.abs(d_t[1]-d_r[1]).max():.4f} FG {np.abs(d_t[2]-d_r[2]).max():.4f}")
        del d_t

    # ---- is "build the guidance channel directly on the resampled grid" valid? ----
    # It is not: the guidance channel is a sparse spike map, and nnU-Net resamples it with
    # an order-3 spline, which spreads and rings around every spike.  Placing the spikes
    # on the resampled grid instead gives a very different tensor.
    import scipy.ndimage as ndi
    scale = np.array(new_shape, dtype=float) / np.array(d_c.shape[1:], dtype=float)
    direct = np.zeros(tuple(int(x) for x in new_shape), dtype=np.float32)
    for pt in fg_pts:                                   # nibabel (x,y,z) -> nnU-Net (z,y,x)
        z, y, x = pt[2], pt[1], pt[0]
        idx = tuple(int(round((v + 0.5) * sc - 0.5)) for v, sc in zip((z, y, x), scale))
        if all(0 <= idx[i] < direct.shape[i] for i in range(3)):
            direct[idx] = 1.0
    # NB `_normalize` works in place, so d_c[2] is already normalised here; recover the
    # raw indicator statistics from the click count instead.
    raw = np.zeros(d_c.shape[1:], dtype=np.float32)
    for pt in fg_pts:
        raw[pt[2], pt[1], pt[0]] = 1.0
    m, sd = float(raw.mean()), float(raw.std())
    direct_norm = (direct - m) / max(sd, 1e-8)
    proper = d_r[2]
    thr = 0.01 * float(np.abs(proper).max())
    print(f"    GUIDANCE on-grid vs properly resampled: max|diff| {np.abs(direct_norm-proper).max():.1f}"
          f" | peak: on-grid {direct_norm.max():.1f} vs resampled {proper.max():.1f}"
          f" | voxels above 1%% of peak: on-grid {int((np.abs(direct_norm-direct_norm.min())>thr).sum())}"
          f" vs resampled {int((np.abs(proper-proper.min())>thr).sum())}"
          f" (clicks: {len(fg_pts)})")
    del direct, direct_norm, raw

    # ---------------- (d) network forward ------------------------------
    with timed("d_network_sliding_window", sync=True):
        logits = p.predict_logits_from_preprocessed_data(torch.from_numpy(data_pp)).cpu()
    print(f"    logits {tuple(logits.shape)} dtype {logits.dtype}")

    with timed("d2_network_with_mirroring_tta", sync=True):
        p.use_mirroring = True
        _ = p.predict_logits_from_preprocessed_data(torch.from_numpy(data_pp)).cpu()
        p.use_mirroring = False
    print(f"    mirroring axes: {p.allowed_mirroring_axes}  "
          f"-> {T['d2_network_with_mirroring_tta']/max(T['d_network_sliding_window'],1e-9):.1f}x the forward cost")

    # ---------------- (e) export ---------------------------------------
    with timed("e_export_resample_back"):
        seg_out = convert_predicted_logits_to_segmentation_with_correct_shape(
            logits, p.plans_manager, p.configuration_manager, p.label_manager, props,
            return_probabilities=False)
    print(f"    seg {seg_out.shape} fg {int((seg_out > 0).sum())}")

    # ---- export alternatives ------------------------------------------
    fnp = p.configuration_manager.resampling_fn_probabilities
    with timed("e1_resample_logits_scipy"):
        lg = fnp(logits, props["shape_after_cropping_and_before_resampling"],
                 p.configuration_manager.spacing, list(props["spacing"]))
    seg_scipy = p.label_manager.convert_logits_to_segmentation(lg)
    del lg
    with timed("e2_resample_logits_torch_cuda", sync=True):
        lg = resample_torch_fornnunet(logits, props["shape_after_cropping_and_before_resampling"],
                                      p.configuration_manager.spacing, list(props["spacing"]),
                                      is_seg=False, device=torch.device("cuda"))
    seg_torch = p.label_manager.convert_logits_to_segmentation(lg.cpu() if hasattr(lg, "cpu") else lg)
    del lg
    sa = np.asarray(seg_scipy.cpu() if hasattr(seg_scipy, "cpu") else seg_scipy).astype(np.uint8)
    sb = np.asarray(seg_torch.cpu() if hasattr(seg_torch, "cpu") else seg_torch).astype(np.uint8)
    inter = int((sa & sb).sum()); tot = int(sa.sum()) + int(sb.sum())
    print(f"    torch-resampled logits -> mask: identical={np.array_equal(sa, sb)}  "
          f"Dice(scipy,torch)={2*inter/max(tot,1):.6f}  fg {int(sa.sum())} vs {int(sb.sum())}")

    # ---------------- (f) write + read back ----------------------------
    out_seg = os.path.join(args.work, "seg.nii.gz")
    with timed("f_write_and_read_seg"):
        rw.write_seg(seg_out, out_seg, props_file)
        back = np.asanyarray(nib.load(out_seg).dataobj).astype(np.uint8)
    print(f"    round trip equal: {np.array_equal(back, seg_out.transpose(2,1,0).astype(np.uint8))}")

    total = sum(v for k, v in T.items() if k.startswith(("a_", "b_", "c_", "d_", "e_", "f_")))
    T["TOTAL_abcdef"] = round(total, 3)
    print("\n=== BREAKDOWN ===")
    for k in ("a_write_4_niftis", "b_sitk_read_4", "c_preprocess_total", "d_network_sliding_window",
              "e_export_resample_back", "f_write_and_read_seg", "TOTAL_abcdef"):
        print(f"  {k:28s} {T[k]:7.2f}s  {100*T[k]/total:5.1f}%")
    print("  --- inside c ---")
    for k in ("c1_crop_to_nonzero", "c2_normalize_4ch", "c3_resample_data_4ch", "c4_resample_seg_1ch",
              "c3b_resample_data_2ch_only"):
        print(f"  {k:28s} {T[k]:7.2f}s")
    with open(args.out, "w") as f:
        json.dump({"case": case.tag, "shape": list(shape), "preprocessed_shape": list(data_pp.shape),
                   "timings": T}, f, indent=2)
    print(f"\nwritten to {args.out}")


if __name__ == "__main__":
    main()
