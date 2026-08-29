#!/usr/bin/env python3
"""End-to-end tests for the autoPET V evaluation harness.

Test 1 (CPU) runs the full interactive loop over a tiny synthetic dataset with
`ThresholdPredictor`.  Test 2 (GPU) runs `BaselineNNUNetPredictor` in-process on one
case.  Test 3 (CPU) replays the reference run in `autoPETV/test/final_output/`.

    python smoke_test.py --work_dir /content/work/smoke                # test 1 only
    python smoke_test.py --work_dir /content/work/smoke --nnunet       # 1 + 2
    python smoke_test.py --work_dir /content/work/smoke --nnunet --only_nnunet
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
import time

import numpy as np
import nibabel as nib

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import interactive_eval as ie                       # noqa: E402
from predictor import (  # noqa: E402
    BaselineNNUNetPredictor, FastBaselineNNUNetPredictor, ThresholdPredictor, affine_from_spacing,
)

SHAPE = (64, 64, 40)          # nibabel index space: (x, y, z), z = axial slices
SPACING = (2.04, 2.04, 3.0)   # mm, matches the autoPET geometry (plans: 3.0/2.036/2.036)
THR = 2.5                     # SUV threshold used by ThresholdPredictor


# ---------------------------------------------------------------------------
# synthetic data
# ---------------------------------------------------------------------------
def _ball(shape, center, radius):
    xx, yy, zz = np.ogrid[: shape[0], : shape[1], : shape[2]]
    # anisotropic voxels -> use mm distances so the blob is round in world space
    d2 = (
        ((xx - center[0]) * SPACING[0]) ** 2
        + ((yy - center[1]) * SPACING[1]) ** 2
        + ((zz - center[2]) * SPACING[2]) ** 2
    )
    return d2 <= radius ** 2


def _base_volumes(rng):
    """CT + PET background: an elliptical 'body' with low uptake, air outside."""
    xx, yy = np.ogrid[: SHAPE[0], : SHAPE[1]]
    body2d = ((xx - 32) / 26.0) ** 2 + ((yy - 32) / 22.0) ** 2 <= 1.0
    body = np.repeat(body2d[:, :, None], SHAPE[2], axis=2)

    ct = np.full(SHAPE, -1000.0, dtype=np.float32)
    ct[body] = 20.0 + rng.normal(0, 15, size=int(body.sum())).astype(np.float32)

    pet = np.zeros(SHAPE, dtype=np.float32)
    pet[body] = 0.9 + np.abs(rng.normal(0, 0.15, size=int(body.sum()))).astype(np.float32)
    return ct, pet, body


def make_synthetic_dataset(root: str, seed: int = 0):
    """4 cases exercising the interesting branches of the official loop."""
    rng = np.random.default_rng(seed)
    img_dir = os.path.join(root, "images")
    lab_dir = os.path.join(root, "labels")
    shutil.rmtree(root, ignore_errors=True)
    os.makedirs(img_dir)
    os.makedirs(lab_dir)
    affine = affine_from_spacing(SPACING)

    def save(stem, ct, pet, label):
        for arr, suffix in ((ct, "_0000"), (pet, "_0001")):
            img = nib.Nifti1Image(arr.astype(np.float32), affine)
            img.header.set_zooms(SPACING)
            nib.save(img, os.path.join(img_dir, f"{stem}{suffix}.nii.gz"))
        lab = nib.Nifti1Image(label.astype(np.uint8), affine)
        lab.header.set_zooms(SPACING)
        lab.set_data_dtype(np.uint8)
        nib.save(lab, os.path.join(lab_dir, f"{stem}.nii.gz"))

    cases = {}

    # (a) two hot blobs, only the first one is a lesion, and the lesion is slightly
    #     bigger than the hot region -> both over- and under-segmentation exist.
    ct, pet, body = _base_volumes(rng)
    b1 = _ball(SHAPE, (26, 30, 14), 9.0)
    b2 = _ball(SHAPE, (42, 36, 27), 7.0)
    pet[b1] = 8.0
    pet[b2] = 6.0
    ct[b1] = 45.0
    ct[b2] = 40.0
    label = _ball(SHAPE, (26, 30, 14), 12.0) & body
    save("synth_a_normal", ct, pet, label)
    cases["synth_a_normal"] = "over+under segmentation"

    # (b) negative case: empty GT, and the threshold predictor does fire -> Dice 0
    ct, pet, body = _base_volumes(rng)
    hot = _ball(SHAPE, (30, 34, 20), 6.0)
    pet[hot] = 5.0
    ct[hot] = 30.0
    save("synth_b_emptygt_fp", ct, pet, np.zeros(SHAPE, dtype=np.uint8))
    cases["synth_b_emptygt_fp"] = "empty GT, non-empty prediction"

    # (c) GT == exactly what the threshold predictor returns -> both error masks empty
    #     -> the official loop crashes the iteration (ValueError on the 3-way unpack).
    ct, pet, body = _base_volumes(rng)
    b = _ball(SHAPE, (30, 30, 18), 8.0)
    pet[b] = 7.0
    ct[b] = 45.0
    save("synth_c_exact", ct, pet, (pet > THR).astype(np.uint8))
    cases["synth_c_exact"] = "perfect prediction -> empty error masks"

    # (d) negative case with an empty prediction -> Dice 1.0 (denom == 0 rule)
    ct, pet, body = _base_volumes(rng)
    save("synth_d_emptygt_clean", ct, pet, np.zeros(SHAPE, dtype=np.uint8))
    cases["synth_d_emptygt_clean"] = "empty GT, empty prediction"

    return root, cases


# ---------------------------------------------------------------------------
# test 1: full loop with the trivial predictor
# ---------------------------------------------------------------------------
def check(cond, msg):
    if not cond:
        raise AssertionError(msg)
    print(f"  ok  {msg}")


def test_loop(work_dir: str, repo: str | None) -> dict:
    print("\n=== TEST 1: interactive loop with ThresholdPredictor ===")
    data_dir = os.path.join(work_dir, "synth_data")
    out_dir = os.path.join(work_dir, "out_threshold")
    shutil.rmtree(out_dir, ignore_errors=True)
    make_synthetic_dataset(data_dir)

    common = [
        "--input_cases", data_dir,
        "--predictor", "threshold",
        "--threshold", str(THR),
        "--strategy", "all",
    ] + (["--repo", repo] if repo else [])
    NIT = 6                      # organizers: iteration 0 + 5 corrections
    t0 = time.time()
    summary = ie.main(common + ["--out_dir", out_dir])
    print(f"  loop finished in {time.time() - t0:.1f}s")

    scores = json.load(open(os.path.join(out_dir, "metric_scores.json")))
    auc = json.load(open(os.path.join(out_dir, "metric_scores_AUC.json")))

    check(len(scores) == 4, f"4 cases scored (got {len(scores)})")
    check(all(len(v) == NIT for v in scores.values()),
          f"default is {NIT} iterations per case (iteration 0 + 5 corrections)")
    check(summary["max_auc"] == NIT - 1, f"max AUC is {NIT - 1}.0")
    check(all(set(r) == {"iteration", "dice", "dmm"} for v in scores.values() for r in v),
          "official record schema {iteration, dice, dmm}")
    check(all(set(v) == {"auc_dice", "auc_dmm"} for v in auc.values()),
          "AUC file has auc_dice + auc_dmm")

    a = scores["synth_a_normal_0000"]
    check(a[0]["dice"] > 0.3, f"case a Dice@0 = {a[0]['dice']:.3f} > 0.3")
    check(all(r["dice"] == a[0]["dice"] for r in a),
          "ThresholdPredictor ignores scribbles -> constant Dice")
    scr = [json.load(open(os.path.join(out_dir, "synth_a_normal_0000", f"iter_{k}_scribbles.json")))
           for k in range(NIT)]
    npts = [len(s["tumor"]) + len(s["background"]) for s in scr]
    check(npts[0] == 0, "iter 0 has no scribbles")
    check(all(npts[k + 1] > npts[k] for k in range(NIT - 1)),
          f"scribbles accumulate strictly: {npts}")
    check(all(isinstance(p, list) and len(p) == 3 and all(isinstance(v, int) for v in p)
              for p in scr[-1]["tumor"] + scr[-1]["background"]),
          "scribble points are [x, y, z] int triples")
    gt = np.asanyarray(nib.load(os.path.join(data_dir, "labels", "synth_a_normal.nii.gz")).dataobj)
    for p in scr[-1]["tumor"]:
        check_in = gt[p[0], p[1], p[2]] > 0
        if not check_in:
            raise AssertionError(f"tumor scribble {p} is outside the GT")
    print("  ok  every tumor scribble voxel lies inside the GT (nibabel index space)")

    b = scores["synth_b_emptygt_fp_0000"]
    check(b[0]["dice"] == 0.0, "empty GT + FP prediction -> Dice 0")
    check(all(math.isnan(r["dmm"]) for r in b), "empty GT -> DMM is NaN (official)")
    check(math.isnan(auc["synth_b_emptygt_fp_0000"]["auc_dmm"]), "NaN DMM propagates into AUC")

    d = scores["synth_d_emptygt_clean_0000"]
    check(all(r["dice"] == 1.0 for r in d), "empty GT + empty prediction -> Dice 1.0")

    c = scores["synth_c_exact_0000"]
    check(all(r["dice"] == 1.0 for r in c),
          f"perfect prediction is propagated to all {NIT} iterations (Dice 1.0)")
    check(all(os.path.isfile(os.path.join(out_dir, "synth_c_exact_0000", f"iter_{k}{ext}"))
              for k in range(NIT) for ext in (".nii.gz", "_scribbles.json")),
          "propagated iterations still write predictions and scribbles")

    # ... and the same case with the published loop's crash behaviour
    out_crash = os.path.join(work_dir, "out_threshold_crash")
    shutil.rmtree(out_crash, ignore_errors=True)
    ie.main(common + ["--out_dir", out_crash, "--official_empty_error_crash", "--quiet",
                      "--cases", "synth_c_exact"])
    cc = json.load(open(os.path.join(out_crash, "metric_scores.json")))["synth_c_exact_0000"]
    check(cc[0]["dice"] == 1.0 and all(r["dice"] == 0.0 for r in cc[1:]),
          "--official_empty_error_crash reproduces the published ValueError behaviour "
          "(1.0 then 0.0 for every later iteration)")

    # replayed (category-2 style) scribbles
    out_replay = os.path.join(work_dir, "out_replay")
    shutil.rmtree(out_replay, ignore_errors=True)
    ie.main(common + ["--out_dir", out_replay, "--replay_scribbles_dir", out_dir, "--quiet"])
    rep = json.load(open(os.path.join(out_replay, "metric_scores.json")))
    same_scr = all(
        json.load(open(os.path.join(out_replay, t, f"iter_{k}_scribbles.json")))
        == json.load(open(os.path.join(out_dir, t, f"iter_{k}_scribbles.json")))
        for t in rep for k in range(NIT))
    check(same_scr, "--replay_scribbles_dir feeds back exactly the recorded scribbles")

    # prediction cache
    cdir = os.path.join(work_dir, "cache")
    shutil.rmtree(cdir, ignore_errors=True)
    out_c1 = os.path.join(work_dir, "out_cache1")
    out_c2 = os.path.join(work_dir, "out_cache2")
    for o in (out_c1, out_c2):
        shutil.rmtree(o, ignore_errors=True)
    s1 = ie.main(common + ["--out_dir", out_c1, "--cache_dir", cdir, "--quiet"])
    s2 = ie.main(common + ["--out_dir", out_c2, "--cache_dir", cdir, "--quiet"])
    # the two lesion-free cases never get a scribble, so their 6 iterations share one
    # cache entry -> the very first run already serves them from cache
    check(s1["cache"]["hits"] == 2 * (NIT - 1),
          f"lesion-free cases reuse one inference for all {NIT} iterations "
          f"({s1['cache']['hits']} within-run hits)")
    check(s2["cache"]["misses"] == 0
          and s2["cache"]["hits"] == s1["cache"]["hits"] + s1["cache"]["misses"],
          f"second run served entirely from --cache_dir ({s2['cache']['hits']} hits, "
          f"{s1['cache']['misses']} inferences saved)")
    check(json.load(open(os.path.join(out_c1, "metric_scores.json")))
          == json.load(open(os.path.join(out_c2, "metric_scores.json"))),
          "cached run reproduces the uncached scores exactly")

    # resume from a previous run
    out_resume = os.path.join(work_dir, "out_resume")
    shutil.rmtree(out_resume, ignore_errors=True)
    t0 = time.time()
    ie.main(common + ["--out_dir", out_resume, "--prev_pred_dir", out_dir, "--quiet"])
    resumed = json.load(open(os.path.join(out_resume, "metric_scores.json")))
    same = all(
        (r1["dice"] == r2["dice"]) and
        (math.isnan(r1["dmm"]) and math.isnan(r2["dmm"]) or r1["dmm"] == r2["dmm"])
        for t in scores for r1, r2 in zip(scores[t], resumed[t])
    )
    check(same, f"--prev_pred_dir reproduces identical scores ({time.time() - t0:.1f}s)")

    print("  summary: " + json.dumps({k: summary[k] for k in (
        "n_cases", "max_auc", "mean_auc_dice", "mean_auc_dmm",
        "mean_auc_dmm_nan_propagating", "n_cases_with_nan_dmm")}))
    return summary


# ---------------------------------------------------------------------------
# test 2: nnU-Net in-process
# ---------------------------------------------------------------------------
def test_nnunet(work_dir: str, model_folder: str | None, real_case_dir: str | None,
                repo: str | None, full_loop: bool) -> None:
    """Run the in-process nnU-Net path and compare the fast predictor's mask and timings
    against the file-based reference one."""
    print("\n=== TEST 2: nnU-Net in-process (reference vs fast predictor) ===")
    data_dir = os.path.join(work_dir, "synth_data")
    if not os.path.isdir(data_dir):
        make_synthetic_dataset(data_dir)

    use_real = False
    if real_case_dir and os.path.isdir(os.path.join(real_case_dir, "images")):
        imgs = sorted(f for f in os.listdir(os.path.join(real_case_dir, "images")) if "_0000" in f)
        real = [f for f in imgs if os.path.getsize(os.path.join(real_case_dir, "images", f)) > 10_000]
        if real:
            data_dir = real_case_dir
            use_real = True
    print(f"  data: {data_dir} ({'REAL case' if use_real else 'synthetic case'})")

    cases = ie.discover_cases(data_dir, logger=None)
    case = cases[0]
    ct_img, pet_img = nib.load(case.ct), nib.load(case.pet)
    ct = np.asanyarray(ct_img.dataobj)
    pet = np.asanyarray(pet_img.dataobj)
    gt = np.asanyarray(nib.load(case.label).dataobj)
    spacing = tuple(float(z) for z in pet_img.header.get_zooms()[:3])
    print(f"  case={case.tag} shape={pet.shape} spacing={spacing} gt_voxels={int(gt.sum())}")

    ref = BaselineNNUNetPredictor(model_folder=model_folder)
    fast = FastBaselineNNUNetPredictor(model_folder=model_folder)
    t0 = time.time()
    ref.warmup()
    fast._predictor = ref._predictor          # one resident model for both
    fast.last_timings = {}
    t_load = time.time() - t0
    print(f"  model load: {t_load:.1f}s  ({ref.model_folder})")

    cache = os.path.join(work_dir, "nnunet_cache")
    shutil.rmtree(cache, ignore_errors=True)
    os.makedirs(cache, exist_ok=True)
    kw = dict(case_cache_dir=cache, affine=pet_img.affine, ct_path=case.ct,
              pet_path=case.pet, case_name=case.tag)

    # ---- iteration 0: no scribbles ----------------------------------------
    empty = {"tumor": [], "background": []}
    t0 = time.time(); m_ref0 = ref.predict(ct, pet, spacing, empty, **kw); t_ref0 = time.time() - t0
    t0 = time.time(); m_fast0 = fast.predict(ct, pet, spacing, empty, **kw); t_fast0 = time.time() - t0
    check(m_fast0.shape == pet.shape and m_fast0.dtype == np.uint8,
          f"fast mask shape {m_fast0.shape} dtype uint8")
    check(np.array_equal(m_ref0, m_fast0),
          f"iter0: fast predictor == reference predictor voxel for voxel "
          f"({int(m_ref0.sum())} fg)")
    print(f"  iter0  reference {t_ref0:6.1f}s   fast {t_fast0:6.1f}s   "
          f"speedup {t_ref0/max(t_fast0,1e-9):.2f}x")
    print(f"         fast breakdown {fast.last_timings}")
    print(f"         Dice vs GT: {ie.dice_score(m_fast0, gt):.4f}")

    # ---- iteration 1: with scribbles + softmax ----------------------------
    fg = np.argwhere(gt > 0)
    tumor_pts = [[int(a), int(b), int(c)] for a, b, c in fg[:: max(1, len(fg) // 5)][:5]]
    bg_pts = [[int(a), int(b), int(c)] for a, b, c in np.argwhere((gt == 0) & (pet > 1))[:3]]
    scr = {"tumor": tumor_pts, "background": bg_pts}
    t0 = time.time(); m_ref1 = ref.predict(ct, pet, spacing, scr, **kw); t_ref1 = time.time() - t0
    t0 = time.time()
    m_fast1, probs = fast.predict(ct, pet, spacing, scr, return_probabilities=True, **kw)
    t_fast1 = time.time() - t0
    check(np.array_equal(m_ref1, m_fast1),
          f"iter1 (with {len(tumor_pts)} tumor / {len(bg_pts)} bg scribbles): "
          f"fast == reference ({int(m_ref1.sum())} fg)")
    check(probs.shape == (2,) + pet.shape, f"softmax shape {probs.shape}")
    check(np.array_equal(probs.argmax(0).astype(np.uint8), m_fast1),
          "softmax argmax == mask (axis mapping correct)")
    print(f"  iter1  reference {t_ref1:6.1f}s   fast {t_fast1:6.1f}s   "
          f"speedup {t_ref1/max(t_fast1,1e-9):.2f}x")
    print(f"         fast breakdown {fast.last_timings}")

    # ---- iteration 2: CT/PET preprocessing now comes from the case cache --
    scr2 = {"tumor": tumor_pts + [[int(a), int(b), int(c)] for a, b, c in fg[5:9]],
            "background": bg_pts}
    t0 = time.time(); m_fast2 = fast.predict(ct, pet, spacing, scr2, **kw); t_fast2 = time.time() - t0
    check(fast.last_timings["ct_pet_cached"],
          f"iter2 reused the cached CT/PET preprocessing "
          f"({fast.last_timings['ct_pet_preproc_s']:.2f}s instead of "
          f"{t_fast0 and fast.last_timings['network_s']:.2f}s of network time)")
    print(f"  iter2  fast {t_fast2:6.1f}s   breakdown {fast.last_timings}")

    shutil.rmtree(cache, ignore_errors=True)
    print(f"  TIMING: model load {t_load:.1f}s | reference {t_ref0:.0f}/{t_ref1:.0f}s "
          f"| fast {t_fast0:.0f}/{t_fast1:.0f}/{t_fast2:.0f}s "
          f"| projected 6 iters: reference {t_ref0 + 5 * t_ref1:.0f}s vs "
          f"fast {t_fast0 + t_fast1 + 4 * t_fast2:.0f}s")

    if full_loop:
        print("\n  --- full 6-iteration loop with the fast baseline predictor ---")
        out_dir = os.path.join(work_dir, "out_nnunet")
        shutil.rmtree(out_dir, ignore_errors=True)
        argv = ["--input_cases", data_dir, "--out_dir", out_dir,
                "--predictor", "fast_baseline_nnunet", "--strategy", "centerline",
                "--cases", case.tag]
        if model_folder:
            argv += ["--model_folder", model_folder]
        if repo:
            argv += ["--repo", repo]
        t0 = time.time()
        s = ie.main(argv)
        print(f"  full loop {time.time() - t0:.1f}s -> "
              f"AUC-Dice {s['mean_auc_dice']:.4f}, AUC-DMM {s['mean_auc_dmm']:.4f}")


# ---------------------------------------------------------------------------
# test 3: replay the organizers' own reference run
# ---------------------------------------------------------------------------
def test_official_reference(repo: str) -> bool:
    """Replay the reference run `autoPETV/test/` ships.

    From its GT and prediction k, re-derive scribble file k+1 and the published Dice
    values.  The images themselves are LFS pointers, so this runs off the labels alone.
    """
    print("\n=== TEST 3: replay the official reference run in autoPETV/test ===")
    tag = "psma_ffcaa75377465b37_2018-03-04"
    fo = os.path.join(repo, "test", "final_output", tag + "_0000")
    lab = os.path.join(repo, "test", "labels", tag + ".nii.gz")
    ref_scores = os.path.join(repo, "test", "final_output", "dice_scores.json")
    if not (os.path.isdir(fo) and os.path.isfile(lab) and os.path.getsize(lab) > 1000):
        print("  SKIP: reference run not present (git-lfs pointers?)")
        return False

    sim, _ = ie.load_official(repo)
    gt = np.asanyarray(nib.load(lab).dataobj)
    preds, scribs = [], []
    for k in range(5):      # the reference run predates the 6-iteration protocol
        p = os.path.join(fo, f"iter_{k}.nii.gz")
        if not os.path.isfile(p) or os.path.getsize(p) < 1000:
            print("  SKIP: reference predictions are git-lfs pointers")
            return False
        preds.append(np.asanyarray(nib.load(p).dataobj))
        scribs.append(json.load(open(os.path.join(fo, f"iter_{k}_scribbles.json"))))

    # -- Dice ---------------------------------------------------------------
    published = json.load(open(ref_scores))[tag + "_0000"]
    ours = [ie.dice_score(pr, gt) for pr in preds]
    for k, (o, ref) in enumerate(zip(ours, published)):
        check(abs(o - ref["dice"]) < 1e-9,
              f"Dice@{k} reproduces the published value ({o:.10f} vs {ref['dice']:.10f})")

    # -- scribbles ----------------------------------------------------------
    matches = {}
    for strategy in ie.STRATEGIES:
        ok = True
        for k in range(4):
            data = {"tumor": list(scribs[k]["tumor"]), "background": list(scribs[k]["background"])}
            overseg = (preds[k] == 1) & (gt == 0)
            underseg = (preds[k] == 0) & (gt == 1)
            sbg, _, fp = sim.simulate_scribble_from_label(overseg, strategy)
            sfg, _, fn = sim.simulate_scribble_from_label(underseg, strategy)
            if fp <= fn:
                data["tumor"] += sfg
            else:
                data["background"] += sbg
            if data != scribs[k + 1]:
                ok = False
                break
        matches[strategy] = ok
    print(f"  strategy match: {matches}")
    check(any(matches.values()),
          "our scribble derivation reproduces the reference iter_k_scribbles.json exactly "
          f"(matching strategy: {[s for s, v in matches.items() if v]})")
    return True


# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--work_dir", default="/content/work/smoke")
    p.add_argument("--repo", default=None)
    p.add_argument("--model_folder", default=None)
    p.add_argument("--real_case_dir", default="/content/work/testcase")
    p.add_argument("--nnunet", action="store_true", help="also run the nnU-Net test")
    p.add_argument("--only_nnunet", action="store_true")
    p.add_argument("--full_loop", action="store_true",
                   help="in test 2, also run all 5 iterations through the harness")
    args = p.parse_args()

    os.makedirs(args.work_dir, exist_ok=True)
    if not args.only_nnunet:
        test_loop(args.work_dir, args.repo)
        test_official_reference(ie.resolve_repo(args.repo))
    if args.nnunet or args.only_nnunet:
        test_nnunet(args.work_dir, args.model_folder, args.real_case_dir, args.repo,
                    args.full_loop)
    print("\nALL SMOKE TESTS PASSED")


if __name__ == "__main__":
    main()
