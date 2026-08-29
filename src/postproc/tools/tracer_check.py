"""Measure (and re-fit) the FDG/PSMA tracer heuristic on a labelled case list.

Ground truth is the case-name prefix (fdg_... / psma_...); the container only sees a
UUID, so the heuristic has to work from the images alone. Writes one row of tracer
features per case to --out_csv and prints the accuracy of guess_tracer with every
misclassified case. Features use the CT body extent, not the raw array axis: a scan
that stops at the neck has no "top 15 % of the array" that is a head.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from typing import Any, Dict, List, Optional, Sequence

import cc3d
import nibabel as nib
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.dirname(os.path.dirname(_HERE))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from postproc.tracer_classifier import guess_tracer, superior_axis, tracer_features  # noqa: E402
from postproc.utils import voxel_volume_ml                                           # noqa: E402

BODY_HU = -500.0


def body_slab_features(ct, pet, spacing, affine, step=2) -> Dict[str, Any]:
    """Head/body SUV statistics measured against the CT body extent."""
    axis, sign = superior_axis(affine)
    ct = np.asarray(ct)
    pet = np.asarray(pet, dtype=np.float32)
    other = tuple(a for a in range(3) if a != axis)
    sl: List[Any] = [slice(None, None, 4)] * 3
    sl[axis] = slice(None)
    occ = (np.asarray(ct[tuple(sl)], dtype=np.float32) > BODY_HU).sum(axis=other)
    nz = np.nonzero(occ > 0)[0]
    lo, hi = (int(nz[0]), int(nz[-1])) if nz.size else (0, ct.shape[axis] - 1)
    span = max(hi - lo, 1)

    def slab(a: float, b: float):
        """The part of the body between fractions a..b of the caudal->cranial extent."""
        if sign > 0:
            i0, i1 = lo + int(a * span), lo + int(b * span)
        else:
            i0, i1 = hi - int(b * span), hi - int(a * span)
        idx: List[Any] = [slice(None, None, step)] * 3
        idx[axis] = slice(max(i0, 0), min(i1 + 1, pet.shape[axis]))
        return pet[tuple(idx)]

    vox_ml = voxel_volume_ml(spacing) * step * step   # the slab keeps the superior axis

    def blob(arr, thr):
        hot = np.ascontiguousarray(arr >= thr).view(np.uint8)
        if not hot.any():
            return 0.0, 0.0
        lab = cc3d.connected_components(hot, connectivity=26)
        cnt = cc3d.statistics(lab)["voxel_counts"]
        if len(cnt) <= 1:
            return 0.0, 0.0
        b = int(np.argmax(cnt[1:]) + 1)
        return float(cnt[b] * vox_ml), float(arr[lab == b].max())

    head = slab(0.85, 1.0)          # skull vault down to roughly the skull base
    upper = slab(0.75, 1.0)
    trunk = slab(0.15, 0.75)
    pelvis = slab(0.0, 0.25)
    head_blob_ml, head_blob_suv = blob(head, 4.0)

    def p(a, q):
        return float(np.percentile(a, q)) if a.size else 0.0

    return {
        "superior_axis": int(axis), "superior_sign": int(sign),
        "body_lo": lo, "body_hi": hi, "body_span": span,
        "head_blob_ml": head_blob_ml, "head_blob_suv_max": head_blob_suv,
        "head_suv_max": float(head.max()) if head.size else 0.0,
        "head_suv_p999": p(head, 99.9), "head_suv_p99": p(head, 99),
        "head_suv_mean": float(head.mean()) if head.size else 0.0,
        "upper_suv_p999": p(upper, 99.9),
        "trunk_suv_p999": p(trunk, 99.9), "trunk_suv_max": float(trunk.max()) if trunk.size else 0.0,
        "pelvis_suv_p999": p(pelvis, 99.9), "pelvis_suv_max": float(pelvis.max()) if pelvis.size else 0.0,
        "global_suv_max": float(pet.max()),
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--image_dir", required=True)
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args(argv)

    cts = sorted(os.path.join(args.image_dir, f) for f in os.listdir(args.image_dir) if "_0000" in f)
    pets = sorted(os.path.join(args.image_dir, f) for f in os.listdir(args.image_dir) if "_0001" in f)
    pairs = list(zip(cts, pets))[:args.limit]

    rows: List[Dict[str, Any]] = []
    for n, (ctp, petp) in enumerate(pairs, 1):
        tag = os.path.basename(ctp).replace(".nii.gz", "")
        truth = "fdg" if tag.lower().startswith("fdg") else ("psma" if tag.lower().startswith("psma") else "?")
        ct_img, pet_img = nib.load(ctp), nib.load(petp)
        ct = np.asanyarray(ct_img.dataobj)
        pet = np.asanyarray(pet_img.dataobj)
        affine = pet_img.affine
        spacing = tuple(float(z) for z in pet_img.header.get_zooms()[:3])

        # Both paths are measured: the container always has the CT, but the PET-only
        # fallback is what runs if a caller does not pass it, so it must be right too.
        guess, feats = guess_tracer(pet, spacing, ct=ct, affine=affine, return_features=True)
        guess_noct = guess_tracer(pet, spacing, affine=affine)
        row: Dict[str, Any] = {"case": tag, "truth": truth, "guess": guess,
                               "guess_no_ct": guess_noct,
                               "confidence": feats["confidence"], "reason": feats["reason"]}
        row.update({("cur_" + k): v for k, v in feats.items()
                    if k not in ("tracer", "confidence", "reason", "placeholder")})
        row.update(body_slab_features(ct, pet, spacing, affine))
        rows.append(row)
        flag = "" if guess == truth else "   <-- WRONG"
        print(f"[{n}/{len(pairs)}] {tag[:46]:46s} truth={truth:4s} guess={guess:4s}"
              f" headblob={row['head_blob_ml']:8.1f}mL headSUV={row['head_suv_p999']:7.2f}"
              f" trunk={row['trunk_suv_p999']:8.2f} pelvis={row['pelvis_suv_p999']:8.2f}{flag}",
              flush=True)
        keys: List[str] = []
        for r in rows:
            for k in r:
                if k not in keys:
                    keys.append(k)
        with open(args.out_csv + ".tmp", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)
        os.replace(args.out_csv + ".tmp", args.out_csv)
        del ct, pet

    ok = sum(1 for r in rows if r["guess"] == r["truth"])
    ok_noct = sum(1 for r in rows if r["guess_no_ct"] == r["truth"])
    conf = [r["confidence"] for r in rows]
    print(f"\nguess_tracer accuracy (with CT): {ok}/{len(rows)}")
    print(f"guess_tracer accuracy (PET only): {ok_noct}/{len(rows)}")
    print(f"confidence: min={min(conf):.3f} p5={np.percentile(conf, 5):.3f} "
          f"median={np.median(conf):.3f}; cases below 0.60: "
          f"{sum(1 for c in conf if c < 0.60)}")
    for r in rows:
        if r["guess"] != r["truth"] or r["guess_no_ct"] != r["truth"]:
            print(f"  WRONG {r['case'][:50]:50s} truth={r['truth']} guess={r['guess']} "
                  f"no_ct={r['guess_no_ct']} conf={r['confidence']:.2f} ({r['reason']}) "
                  f"head_blob={r['cur_head_blob_ml']:.1f}mL "
                  f"head_p99={r['head_suv_p99']:.2f} trunk_p999={r['trunk_suv_p999']:.2f}")
    print("wrote", args.out_csv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
