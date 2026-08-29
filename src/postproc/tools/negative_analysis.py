"""Iteration-0 component statistics for the lesion-free ("negative") gate study.

Runs one iteration (no scribbles, no previous prediction) per case of a labelled set
and writes the table the gate thresholds are fitted on: components.csv (one row per
predicted component, with whether it matches a GT lesion under the challenge's
18-connectivity/IoU>=0.1 rule), cases.csv, and bundles/<tag>.npz, a sparse replay
bundle so a threshold sweep never needs the GPU again. Both CSVs are rewritten after
every case, so a killed run is still usable.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional, Sequence

import cc3d
import nibabel as nib
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.dirname(os.path.dirname(_HERE))          # .../src
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from postproc.tracer_classifier import superior_axis            # noqa: E402
from postproc.utils import foreground_prob, voxel_volume_ml     # noqa: E402

CONNECTIVITY = 18          # metrics.MetricEvaluator(connectivity=18)
IOU_MATCH = 0.1            # metrics matching threshold
BODY_HU = -500.0           # CT threshold separating patient from air/table


# ----------------------------------------------------------------------------- helpers
def tracer_from_name(tag: str) -> str:
    t = tag.lower()
    if t.startswith("fdg") or "fdg" in t:
        return "fdg"
    if t.startswith("psma") or "psma" in t:
        return "psma"
    return "unknown"


def body_extent(ct: np.ndarray, axis: int, stride: int = 4) -> tuple:
    """(lo, hi) index of the patient along `axis`, from a strided CT > BODY_HU pass."""
    sl = [slice(None, None, stride)] * 3
    sl[axis] = slice(None)
    sub = np.asarray(ct[tuple(sl)], dtype=np.float32)
    other = tuple(a for a in range(3) if a != axis)
    occupied = (sub > BODY_HU).sum(axis=other)
    nz = np.nonzero(occupied > 0)[0]
    if nz.size == 0:
        return 0, ct.shape[axis] - 1
    return int(nz[0]), int(nz[-1])


def gt_components(gt: np.ndarray) -> tuple:
    m = np.ascontiguousarray(np.asarray(gt) > 0).view(np.uint8)
    if not m.any():
        return np.zeros(m.shape, dtype=np.int32), np.zeros(0, dtype=np.int64)
    lab = cc3d.connected_components(m, connectivity=CONNECTIVITY)
    counts = cc3d.statistics(lab)["voxel_counts"]
    return lab, counts


def shell_suv(pet: np.ndarray, idx: np.ndarray, shape, radius_vox: int = 4) -> float:
    """Max SUV in a small box around the component, excluding the component itself.

    A false positive on the rim of a hot organ (bladder, kidney, brain) has a much
    hotter neighbourhood than a genuine small lesion in cool tissue.
    """
    if idx.shape[0] == 0:
        return 0.0
    lo = np.maximum(idx.min(0) - radius_vox, 0)
    hi = np.minimum(idx.max(0) + radius_vox + 1, np.asarray(shape))
    box = np.asarray(pet[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]], dtype=np.float32)
    local = np.ones(box.shape, dtype=bool)
    local[tuple((idx - lo).T)] = False
    return float(box[local].max()) if local.any() else 0.0


# ------------------------------------------------------------------------ per-case work
def analyse_case(tag, ct, pet, gt, prob_fg, mask, spacing, affine) -> Dict[str, Any]:
    vox_ml = voxel_volume_ml(spacing)
    axis, sign = superior_axis(affine)
    lo_b, hi_b = body_extent(ct, axis)
    span = max(hi_b - lo_b, 1)

    m = np.ascontiguousarray(np.asarray(mask) > 0)
    n_vox = int(m.sum())
    gt_lab, gt_counts = gt_components(gt)
    gt_vol_ml = float((np.asarray(gt) > 0).sum() * vox_ml)

    case: Dict[str, Any] = {
        "case": tag,
        "tracer": tracer_from_name(tag),
        "empty_gt": int(gt_vol_ml == 0.0),
        "gt_volume_ml": gt_vol_ml,
        "n_gt_lesions": int(max(len(gt_counts) - 1, 0)),
        "shape": list(map(int, m.shape)),
        "spacing": [float(s) for s in spacing],
        "voxel_ml": vox_ml,
        "superior_axis": int(axis),
        "superior_sign": int(sign),
        "body_lo": lo_b,
        "body_hi": hi_b,
        "pred_voxels": n_vox,
        "total_volume_ml": n_vox * vox_ml,
        "prob_max_global": float(prob_fg.max()) if prob_fg is not None else float("nan"),
        "pet_suv_max_global": float(np.asarray(pet, dtype=np.float32).max()),
    }

    comps: List[Dict[str, Any]] = []
    if n_vox == 0:
        case.update(n_components=0, largest_component_ml=0.0, suv_max_in_mask=0.0,
                    prob_max_in_mask=0.0, suv_max_largest=0.0, prob_max_largest=0.0,
                    n_tp_components=0, n_fp_components=0, tp_volume_ml=0.0)
        return {"case": case, "components": comps, "bundle": None}

    lab = cc3d.connected_components(m.view(np.uint8), connectivity=CONNECTIVITY)
    counts = cc3d.statistics(lab)["voxel_counts"]
    n_comp = int(np.count_nonzero(counts[1:]))

    idx_all = np.argwhere(m).astype(np.int32)
    lab_all = lab[tuple(idx_all.T)].astype(np.int32)
    pet_all = np.asarray(pet, dtype=np.float32)[tuple(idx_all.T)]
    ct_all = np.asarray(ct, dtype=np.float32)[tuple(idx_all.T)]
    prob_all = (prob_fg[tuple(idx_all.T)].astype(np.float32) if prob_fg is not None
                else np.full(idx_all.shape[0], np.nan, dtype=np.float32))
    gtl_all = gt_lab[tuple(idx_all.T)].astype(np.int32) if gt_lab.any() else np.zeros(idx_all.shape[0], np.int32)

    n_tp = n_fp = 0
    tp_ml = 0.0
    for cid in range(1, len(counts)):
        sel = lab_all == cid
        k = int(sel.sum())
        if k == 0:
            continue
        idx = idx_all[sel]
        p = prob_all[sel]
        s = pet_all[sel]
        cen = idx.mean(0)
        zc = float(cen[axis])
        zf = (zc - lo_b) / span
        if sign < 0:
            zf = 1.0 - zf

        # --- overlap with GT lesions under the challenge's matching rule -------------
        best_iou, best_gt = 0.0, 0
        if gt_lab.any():
            hit = gtl_all[sel]
            for g in np.unique(hit[hit > 0]):
                inter = int((hit == g).sum())
                union = k + int(gt_counts[g]) - inter
                iou = inter / union if union else 0.0
                if iou > best_iou:
                    best_iou, best_gt = iou, int(g)
        is_tp = int(best_iou >= IOU_MATCH)
        n_tp += is_tp
        n_fp += 1 - is_tp
        if is_tp:
            tp_ml += k * vox_ml

        comps.append({
            "case": tag,
            "tracer": case["tracer"],
            "empty_gt": case["empty_gt"],
            "comp_id": int(cid),
            "n_voxels": k,
            "volume_ml": k * vox_ml,
            "suv_max": float(s.max()),
            "suv_mean": float(s.mean()),
            "suv_p90": float(np.percentile(s, 90)),
            "prob_max": float(np.nanmax(p)) if p.size else float("nan"),
            "prob_mean": float(np.nanmean(p)) if p.size else float("nan"),
            "ct_mean_hu": float(ct_all[sel].mean()),
            "centroid_i": float(cen[0]), "centroid_j": float(cen[1]), "centroid_k": float(cen[2]),
            "z_frac": float(zf),
            "shell_suv_max": shell_suv(pet, idx, m.shape),
            "n_components_case": n_comp,
            "best_gt_iou": float(best_iou),
            "gt_lesion": best_gt,
            "is_tp": is_tp,
        })

    comps.sort(key=lambda c: -c["volume_ml"])
    largest = comps[0] if comps else None
    case.update(
        n_components=n_comp,
        largest_component_ml=largest["volume_ml"] if largest else 0.0,
        suv_max_in_mask=float(pet_all.max()),
        prob_max_in_mask=float(np.nanmax(prob_all)) if prob_all.size else float("nan"),
        suv_max_largest=largest["suv_max"] if largest else 0.0,
        prob_max_largest=largest["prob_max"] if largest else 0.0,
        n_tp_components=n_tp,
        n_fp_components=n_fp,
        tp_volume_ml=tp_ml,
    )

    bundle = {
        "idx": idx_all, "labels": lab_all, "pet": pet_all, "ct": ct_all,
        "prob": prob_all, "gt_label": gtl_all,
        "shape": np.asarray(m.shape, dtype=np.int32),
        "spacing": np.asarray(spacing, dtype=np.float32),
        "affine": np.asarray(affine, dtype=np.float64),
        "gt_counts": np.asarray(gt_counts, dtype=np.int64),
        "prob_max_global": np.float32(case["prob_max_global"]),
        "body": np.asarray([lo_b, hi_b, axis, sign], dtype=np.int32),
    }
    return {"case": case, "components": comps, "bundle": bundle}


# --------------------------------------------------------------------------------- main
def build_model(name: str, args):
    from predictor import FastBaselineNNUNetPredictor, InteractiveNNUNetPredictor
    common = dict(folds=(0,), checkpoint_name=args.checkpoint, device=args.device,
                  disable_tta=True, tile_step_size=0.5,
                  num_processes_preprocessing=3, num_processes_segmentation_export=3)
    if name == "fast_baseline_nnunet":
        return FastBaselineNNUNetPredictor(model_folder=args.model_folder, **common)
    if name == "interactive_nnunet":
        kw = dict(common)
        if args.model_folder:
            kw["model_folder"] = args.model_folder
        return InteractiveNNUNetPredictor(**kw)
    raise ValueError(name)


def write_csv(path: str, rows: Sequence[Dict[str, Any]]) -> None:
    if not rows:
        return
    keys: List[str] = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    tmp = path + ".tmp"
    with open(tmp, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: (json.dumps(v) if isinstance(v, list) else v) for k, v in r.items()})
    os.replace(tmp, path)


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--image_dir", required=True)
    ap.add_argument("--label_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--model", default="fast_baseline_nnunet",
                    choices=["fast_baseline_nnunet", "interactive_nnunet"])
    ap.add_argument("--model_folder", default=None)
    ap.add_argument("--checkpoint", default="checkpoint_final.pth")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--cases", nargs="*", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--negatives_first", action="store_true",
                    help="process lesion-free cases first")
    ap.add_argument("--skip_existing", action="store_true",
                    help="skip cases whose replay bundle already exists")
    ap.add_argument("--no_bundles", action="store_true")
    args = ap.parse_args(argv)

    os.makedirs(args.out_dir, exist_ok=True)
    bdir = os.path.join(args.out_dir, "bundles")
    os.makedirs(bdir, exist_ok=True)

    cts = sorted(os.path.join(args.image_dir, f) for f in os.listdir(args.image_dir) if "_0000" in f)
    pets = sorted(os.path.join(args.image_dir, f) for f in os.listdir(args.image_dir) if "_0001" in f)
    labels = sorted(os.path.join(args.label_dir, f) for f in os.listdir(args.label_dir))
    cases = []
    for c, p, l in zip(cts, pets, labels):
        tag = os.path.basename(c).replace(".nii.gz", "")
        stem = tag[:-5] if tag.endswith("_0000") else tag
        assert os.path.basename(p).replace(".nii.gz", "") == stem + "_0001", (c, p)
        assert os.path.basename(l).replace(".nii.gz", "") == stem, (c, l)
        cases.append((tag, c, p, l))
    if args.cases:
        want = set(args.cases)
        cases = [c for c in cases if c[0] in want or c[0][:-5] in want]

    if args.negatives_first:
        t0 = time.time()
        neg = []
        for tag, c, p, l in cases:
            g = np.asanyarray(nib.load(l).dataobj)
            neg.append(not bool((g > 0).any()))
            del g
        order = np.argsort([0 if n else 1 for n in neg], kind="stable")
        cases = [cases[i] for i in order]
        print(f"[order] {sum(neg)} lesion-free cases first ({time.time()-t0:.0f}s to read labels)",
              flush=True)
    if args.limit:
        cases = cases[:args.limit]

    model = build_model(args.model, args)
    comp_rows: List[Dict[str, Any]] = []
    case_rows: List[Dict[str, Any]] = []
    comp_csv = os.path.join(args.out_dir, "components.csv")
    case_csv = os.path.join(args.out_dir, "cases.csv")

    for n, (tag, ctp, petp, labp) in enumerate(cases, 1):
        bpath = os.path.join(bdir, tag.replace("/", "_") + ".npz")
        if args.skip_existing and os.path.isfile(bpath):
            print(f"[{n}/{len(cases)}] skip (bundle exists) {tag}", flush=True)
            continue
        t0 = time.time()
        ct_img, pet_img, lab_img = nib.load(ctp), nib.load(petp), nib.load(labp)
        ct = np.asanyarray(ct_img.dataobj)
        pet = np.asanyarray(pet_img.dataobj)
        gt = np.asanyarray(lab_img.dataobj)
        affine = pet_img.affine
        spacing = tuple(float(z) for z in pet_img.header.get_zooms()[:3])

        out = model.predict(ct, pet, spacing, {"tumor": [], "background": []},
                            prev_pred=None, case_cache_dir=None, affine=affine,
                            ct_path=ctp, pet_path=petp, case_name=tag,
                            return_probabilities=True)
        mask, prob = out
        prob_fg = foreground_prob(prob, mask.shape)

        res = analyse_case(tag, ct, pet, gt, prob_fg, mask, spacing, affine)
        res["case"]["model"] = args.model
        res["case"]["seconds"] = round(time.time() - t0, 1)
        case_rows.append(res["case"])
        comp_rows.extend(res["components"])
        if res["bundle"] is not None and not args.no_bundles:
            np.savez_compressed(bpath, **res["bundle"])
        elif not args.no_bundles:
            np.savez_compressed(bpath, idx=np.zeros((0, 3), np.int32),
                                shape=np.asarray(mask.shape, np.int32),
                                spacing=np.asarray(spacing, np.float32),
                                prob_max_global=np.float32(res["case"]["prob_max_global"]))
        write_csv(comp_csv, comp_rows)
        write_csv(case_csv, case_rows)
        c = res["case"]
        print(f"[{n}/{len(cases)}] {tag[:48]:48s} empty_gt={c['empty_gt']} "
              f"vol={c['total_volume_ml']:.3f}mL ncomp={c['n_components']} "
              f"suvmax={c['suv_max_in_mask']:.1f} pmax={c['prob_max_in_mask']:.3f} "
              f"tp={c['n_tp_components']} fp={c['n_fp_components']} ({c['seconds']}s)",
              flush=True)
        del ct, pet, gt, mask, prob, prob_fg

    print("DONE", len(case_rows), "cases ->", args.out_dir, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
