"""Build a fake Grand Challenge container filesystem from a real autoPET case.

Reproduces the mount layout on disk so `submission/process.py` can be run end-to-end
without Docker, by pointing the AUTOPETV_* env vars at it:

    <sim>/input/images/ct/<random-uuid>.mha
    <sim>/input/images/pet/<other-random-uuid>.mha      # different uuid on purpose:
    <sim>/input/lesion-clicks.json                      #   proves we use the CT's
    <sim>/output/images/tumor-lesion-segmentation/      # created by the container
    <sim>/output/state/
    <sim>/cache/

Usage:
    python -m submission.tests.build_gc_sim \
        --images_dir /content/work/testcase/images \
        --case psma_0198cdca94fbb95f_2020-05-09 \
        --out /content/work/gc_sim \
        --scribbles /content/work/eval_real/psma_0198cdca94fbb95f_2020-05-09_0000/iter_1_scribbles.json
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import time
import uuid as uuidlib


def convert(src_nii: str, dst_mha: str) -> None:
    import SimpleITK as sitk

    t0 = time.time()
    img = sitk.ReadImage(src_nii)
    sitk.WriteImage(img, dst_mha, True)
    print(f"  {os.path.basename(src_nii)} -> {os.path.basename(dst_mha)} "
          f"size={img.GetSize()} spacing={tuple(round(s, 4) for s in img.GetSpacing())} "
          f"({time.time() - t0:.1f} s)")


def scribbles_to_gc(scribbles: dict) -> dict:
    """Scribble dict -> the GC "Multiple points" JSON the challenge loop writes."""
    gc = {"version": {"major": 1, "minor": 0}, "type": "Multiple points", "points": []}
    for p in scribbles.get("tumor", []):
        gc["points"].append({"point": list(p), "name": "tumor"})
    for p in scribbles.get("background", []):
        gc["points"].append({"point": list(p), "name": "background"})
    return gc


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images_dir", required=True,
                    help="nnU-Net-format image dir holding <case>_0000.nii.gz / _0001.nii.gz")
    ap.add_argument("--case", required=True, help="case stem WITHOUT the _000X suffix")
    ap.add_argument("--out", required=True, help="simulation root")
    ap.add_argument("--scribbles", default=None,
                    help="iter_k_scribbles.json ({'tumor':[...], 'background':[...]}); "
                         "omit for iteration 0")
    ap.add_argument("--seed", type=int, default=None,
                    help="make the random uuids reproducible")
    ap.add_argument("--clean", action="store_true", help="wipe <out> first")
    args = ap.parse_args()

    if args.clean and os.path.isdir(args.out):
        shutil.rmtree(args.out)

    ct_src = os.path.join(args.images_dir, f"{args.case}_0000.nii.gz")
    pet_src = os.path.join(args.images_dir, f"{args.case}_0001.nii.gz")
    for p in (ct_src, pet_src):
        if not os.path.isfile(p):
            raise SystemExit(f"missing {p}")

    if args.seed is not None:
        import random

        rnd = random.Random(args.seed)
        ct_uuid = str(uuidlib.UUID(int=rnd.getrandbits(128), version=4))
        pet_uuid = str(uuidlib.UUID(int=rnd.getrandbits(128), version=4))
    else:
        ct_uuid, pet_uuid = str(uuidlib.uuid4()), str(uuidlib.uuid4())

    ct_dir = os.path.join(args.out, "input", "images", "ct")
    pet_dir = os.path.join(args.out, "input", "images", "pet")
    for d in (ct_dir, pet_dir,
              os.path.join(args.out, "output", "images", "tumor-lesion-segmentation"),
              os.path.join(args.out, "output", "state"),
              os.path.join(args.out, "cache"),
              os.path.join(args.out, "tmp")):
        os.makedirs(d, exist_ok=True)

    print(f"[sim] case={args.case}")
    convert(ct_src, os.path.join(ct_dir, ct_uuid + ".mha"))
    convert(pet_src, os.path.join(pet_dir, pet_uuid + ".mha"))

    scribbles = {"tumor": [], "background": []}
    if args.scribbles:
        with open(args.scribbles) as f:
            scribbles = json.load(f)
    gc = scribbles_to_gc(scribbles)
    with open(os.path.join(args.out, "input", "lesion-clicks.json"), "w") as f:
        json.dump(gc, f)
    print(f"[sim] lesion-clicks.json: {len(gc['points'])} point(s) "
          f"(tumor={len(scribbles.get('tumor', []))}, "
          f"background={len(scribbles.get('background', []))})")

    meta = {"case": args.case, "ct_uuid": ct_uuid, "pet_uuid": pet_uuid,
            "ct_src": ct_src, "pet_src": pet_src, "scribbles": args.scribbles}
    with open(os.path.join(args.out, "sim_meta.json"), "w") as f:
        json.dump(meta, f, indent=1)
    print(f"[sim] ct_uuid  = {ct_uuid}   <- expected output filename")
    print(f"[sim] pet_uuid = {pet_uuid}   (deliberately different)")
    print(f"[sim] root     = {args.out}")


if __name__ == "__main__":
    main()
