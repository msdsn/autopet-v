"""Verify the output of a simulated container run.

Checks the output filename, dtype and geometry, agreement with the harness prediction
for the same case and iteration, and that nothing was written outside output/, state/
and cache/.

Usage:
    python -m submission.tests.check_gc_sim \
        --sim /content/work/gc_sim \
        --reference /content/work/eval_real/<case>_0000/iter_1.nii.gz
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np


def geom(img) -> dict:
    return {
        "size": list(img.GetSize()),
        "spacing": [round(float(s), 9) for s in img.GetSpacing()],
        "origin": [round(float(o), 9) for o in img.GetOrigin()],
        "direction": [round(float(d), 9) for d in img.GetDirection()],
    }


def dice(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(bool)
    b = b.astype(bool)
    s = a.sum() + b.sum()
    return 1.0 if s == 0 else float(2.0 * np.logical_and(a, b).sum() / s)


def main() -> int:
    import SimpleITK as sitk

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sim", required=True)
    ap.add_argument("--reference", default=None,
                    help="harness prediction (.nii.gz) for the SAME case and iteration")
    args = ap.parse_args()

    ok = True

    with open(os.path.join(args.sim, "sim_meta.json")) as f:
        meta = json.load(f)
    out_dir = os.path.join(args.sim, "output", "images", "tumor-lesion-segmentation")

    # 1. filename ----------------------------------------------------------
    files = sorted(n for n in os.listdir(out_dir) if not n.startswith("."))
    print(f"[1] output dir contains {len(files)} file(s): {files}")
    expected = meta["ct_uuid"] + ".mha"
    if files != [expected]:
        print(f"    FAIL expected exactly ['{expected}']")
        ok = False
    else:
        print(f"    PASS filename == CT uuid (PET uuid was {meta['pet_uuid']})")

    out_path = os.path.join(out_dir, files[0]) if files else None
    if out_path is None:
        return 1

    # 2. dtype -------------------------------------------------------------
    out_img = sitk.ReadImage(out_path)
    out_arr_sitk = sitk.GetArrayFromImage(out_img)          # (k, j, i)
    out_arr = np.transpose(out_arr_sitk, (2, 1, 0))         # (i, j, k), nibabel space
    print(f"[2] dtype={out_arr.dtype} shape={out_arr.shape} "
          f"unique={np.unique(out_arr)[:5]} positives={int(out_arr.sum())} "
          f"bytes={os.path.getsize(out_path)}")
    if out_arr.dtype != np.uint8:
        print("    FAIL not uint8")
        ok = False
    else:
        print("    PASS uint8")

    # 3. geometry ----------------------------------------------------------
    ct_dir = os.path.join(args.sim, "input", "images", "ct")
    ct_in = os.path.join(ct_dir, os.listdir(ct_dir)[0])
    g_in, g_out = geom(sitk.ReadImage(ct_in)), geom(out_img)
    if g_in == g_out:
        print(f"[3] PASS geometry identical to the input CT: size={g_out['size']} "
              f"spacing={g_out['spacing']} origin={g_out['origin']}")
    else:
        print("[3] FAIL geometry mismatch")
        for k in g_in:
            if g_in[k] != g_out[k]:
                print(f"    {k}: in={g_in[k]}  out={g_out[k]}")
        ok = False

    # 4. equality with the harness ----------------------------------------
    if args.reference:
        import nibabel as nib

        ref = np.asanyarray(nib.load(args.reference).dataobj).astype(np.uint8)
        print(f"[4] reference {args.reference}: shape={ref.shape} positives={int(ref.sum())}")
        if ref.shape != out_arr.shape:
            print("    FAIL shape mismatch vs reference")
            ok = False
        else:
            d = dice(out_arr, ref)
            identical = bool(np.array_equal(out_arr, ref))
            diff = int(np.logical_xor(out_arr.astype(bool), ref.astype(bool)).sum())
            print(f"    Dice(container, harness) = {d:.6f}   "
                  f"byte-identical={identical}   differing voxels={diff}")
            if d < 1.0:
                print("    FAIL Dice != 1.0 -- the container is not reproducing the harness")
                ok = False
            else:
                print("    PASS Dice == 1.0")
    else:
        print("[4] SKIP no --reference given")

    # 5. stray files -------------------------------------------------------
    strays = []
    for root, _d, fs in os.walk(args.sim):
        rel = os.path.relpath(root, args.sim)
        top = rel.split(os.sep)[0]
        if top in ("output", "cache", "tmp", "."):
            continue
        if top == "input":
            continue
        for fn in fs:
            strays.append(os.path.join(rel, fn))
    print(f"[5] files outside input/output/cache/tmp: {strays if strays else 'none'}")
    if strays:
        ok = False

    print("\n=== state/cache content after the run ===")
    for sub in ("output/state", "cache"):
        d = os.path.join(args.sim, sub)
        if os.path.isdir(d):
            for fn in sorted(os.listdir(d)):
                print(f"  {sub}/{fn}  ({os.path.getsize(os.path.join(d, fn))} B)")
        else:
            print(f"  {sub}/  (absent)")
    marker = os.path.join(args.sim, "output", "state", "autopetv_calls.json")
    if os.path.isfile(marker):
        with open(marker) as f:
            print("  marker:", json.dumps(json.load(f), indent=1)[:2000])

    print("\nRESULT:", "ALL CHECKS PASSED" if ok else "FAILURES ABOVE")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
