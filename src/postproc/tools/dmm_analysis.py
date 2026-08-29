"""Component / lesion dataset for the detection-metric (DMM) rule fit.

Writes components.csv (one row per case, threshold and predicted component), cases.csv
(counts, TP/FP/FN, F1) and lesions.csv (one row per GT lesion, with the threshold at
which it is first hit). It re-binarises the cached foreground softmax at several
thresholds, since IoU >= 0.1 matching makes recall cheap. Nothing here runs a network:
it reads the cache from `interactive_eval.py --cache_probabilities`, so it is CPU-only.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
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

from postproc.constraints import ConstraintState                # noqa: E402
from postproc.tracer_classifier import superior_axis            # noqa: E402
from postproc.utils import voxel_volume_ml                      # noqa: E402

CONNECTIVITY = 18          # metrics.MetricEvaluator(connectivity=18)
IOU_MATCH = 0.1            # metrics matching threshold
BODY_HU = -500.0           # CT threshold separating patient from air/table
MAX_FG_VOXELS = 20_000_000  # guard: a threshold that floods the volume is not a rule


def tracer_from_name(tag: str) -> str:
    t = tag.lower()
    if "fdg" in t:
        return "fdg"
    if "psma" in t:
        return "psma"
    return "unknown"


def scribble_hash(data: Dict[str, Any]) -> str:
    """Mirror of PredictionCache.scribble_hash; the two must yield the same stem."""
    payload = json.dumps(
        {"tumor": [list(map(int, p)) for p in data.get("tumor", [])],
         "background": [list(map(int, p)) for p in data.get("background", [])]},
        sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(payload.encode()).hexdigest()[:16]


def empty_scribble_hash() -> str:
    """The cache stem of iteration 0 (no scribbles), as PredictionCache writes it."""
    return scribble_hash({"tumor": [], "background": []})


def iteration_scribble_hash(run_dir: str, tag: str, it: int, shape, spacing):
    """The cache stem of iteration `it` of a --predictor postproc run.

    The layer hashes the rebuilt ConstraintState lists, not the raw json, so this
    rebuilds them the same way (de-duplicated, out-of-bounds dropped, tumor winning
    over background at a shared voxel).
    """
    path = os.path.join(run_dir, tag, f"iter_{it}_scribbles.json")
    if not os.path.isfile(path):
        return None, None
    with open(path) as fh:
        data = json.load(fh)
    st = ConstraintState.from_scribbles(data, shape, spacing)
    fg = st.tumor_array(shape)
    bg = st.background_array(shape)
    return scribble_hash({"tumor": fg.tolist(), "background": bg.tolist()}), (fg, bg)


def load_cached(cache_root: str, tag: str, stem: Optional[str] = None):
    """(mask, prob_fg) from one entry of the prediction cache."""
    path = os.path.join(cache_root, tag, (stem or empty_scribble_hash()) + ".npz")
    if not os.path.isfile(path):
        return None, None
    with np.load(path) as z:
        mask = z["mask"].astype(np.uint8)
        if "prob_fg" not in z.files:
            return mask, None
        q = z["prob_fg"]
        prob = q.astype(np.float32) / 255.0 if q.dtype == np.uint8 else q.astype(np.float32)
    return mask, prob


def body_frame(ct: np.ndarray, spacing, axis: int, sign: int, stride: int = 2):
    """Superior-inferior body extent and a coarse distance-to-body-surface map.

    The EDT runs on a `stride`-downsampled body mask; a few millimetres of resolution is
    plenty for a threshold rule and it keeps the transform off the full volume.
    """
    from scipy import ndimage

    body = np.asarray(ct[::stride, ::stride, ::stride], dtype=np.float32) > BODY_HU
    if body.any():
        lab, n = cc3d.connected_components(body.view(np.uint8), connectivity=26,
                                           return_N=True)
        if n > 1:
            counts = np.bincount(lab.ravel())
            counts[0] = 0
            body = lab == int(counts.argmax())
    sampling = [float(s) * stride for s in spacing]
    dist = ndimage.distance_transform_edt(body, sampling=sampling).astype(np.float32)

    other = tuple(a for a in range(3) if a != axis)
    occupied = body.sum(axis=other)
    nz = np.nonzero(occupied > 0)[0]
    if nz.size == 0:
        lo, hi = 0, body.shape[axis] - 1
    else:
        lo, hi = int(nz[0]) * stride, int(nz[-1]) * stride
    return dist, stride, lo, hi


def gt_components(gt: np.ndarray):
    m = np.ascontiguousarray(np.asarray(gt) > 0).view(np.uint8)
    if not m.any():
        return np.zeros(m.shape, dtype=np.int32), np.zeros(1, dtype=np.int64)
    lab = cc3d.connected_components(m, connectivity=CONNECTIVITY)
    counts = cc3d.statistics(lab)["voxel_counts"]
    return lab, counts


def _points_per_label(lab: np.ndarray, pts) -> Dict[int, int]:
    out: Dict[int, int] = {}
    if pts is None or len(pts) == 0:
        return out
    p = np.asarray(pts, dtype=np.int64)
    vals = lab[p[:, 0], p[:, 1], p[:, 2]]
    for v in vals[vals > 0]:
        out[int(v)] = out.get(int(v), 0) + 1
    return out


def close_mask(fg: np.ndarray, n: int) -> np.ndarray:
    """Binary closing by `n` voxels, restricted to the mask's bounding box.

    Merging components is at worst neutral for the detection metric, which does not
    punish multi-assignment; it only loses if the merged component's IoU with a small
    lesion falls under 0.1.
    """
    from scipy import ndimage

    if n <= 0 or not fg.any():
        return fg
    nz = np.argwhere(fg)
    lo = np.maximum(nz.min(0) - (n + 1), 0)
    hi = np.minimum(nz.max(0) + (n + 2), np.asarray(fg.shape))
    sl = tuple(slice(int(a), int(b)) for a, b in zip(lo, hi))
    sub = fg[sl]
    st = ndimage.generate_binary_structure(3, 3)
    out = fg.copy()
    out[sl] = ndimage.binary_erosion(
        ndimage.binary_dilation(sub, st, iterations=n), st, iterations=n,
        border_value=1)
    return out


def analyse_threshold(tag, tracer, thr, fg, base_mask, pet, ct, prob, gt_lab, gt_counts,
                      spacing, vox_ml, axis, sign, lo_b, span, dist, dstride,
                      iteration=0, fg_pts=None, bg_pts=None, closing=0):
    """Component rows for one binarisation threshold."""
    m = np.ascontiguousarray(fg)
    rows: List[Dict[str, Any]] = []
    n_fg = int(m.sum())
    if n_fg == 0:
        return rows
    if n_fg > MAX_FG_VOXELS:
        # A threshold that lights up a quarter of the body is not a candidate rule;
        # drop the row rather than spend minutes labelling noise.
        return rows
    lab = cc3d.connected_components(m.view(np.uint8), connectivity=CONNECTIVITY)
    counts = cc3d.statistics(lab)["voxel_counts"]

    idx = np.argwhere(m).astype(np.int32)
    t = tuple(idx.T)
    lab_v = lab[t].astype(np.int32)
    pet_v = np.asarray(pet, dtype=np.float32)[t]
    ct_v = np.asarray(ct, dtype=np.float32)[t]
    prob_v = prob[t].astype(np.float32)
    base_v = base_mask[t].astype(bool)
    gt_v = gt_lab[t].astype(np.int32)
    dist_v = dist[tuple((idx // dstride).T)]

    n_fg_in = _points_per_label(lab, fg_pts)
    n_bg_in = _points_per_label(lab, bg_pts)

    order = np.argsort(lab_v, kind="stable")
    lab_s = lab_v[order]
    starts = np.searchsorted(lab_s, np.arange(1, len(counts)), side="left")
    ends = np.searchsorted(lab_s, np.arange(1, len(counts)), side="right")

    for cid in range(1, len(counts)):
        sel = order[starts[cid - 1]:ends[cid - 1]]
        k = sel.size
        if k == 0:
            continue
        p = prob_v[sel]
        s = pet_v[sel]
        cen = idx[sel].mean(0)
        zf = (float(cen[axis]) - lo_b) / span
        if sign < 0:
            zf = 1.0 - zf

        matches: List[int] = []
        best_iou = 0.0
        hit = gt_v[sel]
        for g in np.unique(hit[hit > 0]):
            inter = int((hit == g).sum())
            union = k + int(gt_counts[g]) - inter
            iou = inter / union if union else 0.0
            best_iou = max(best_iou, iou)
            if iou >= IOU_MATCH:
                matches.append(int(g))

        rows.append({
            "case": tag,
            "tracer": tracer,
            "iteration": int(iteration),
            "threshold": thr,
            "closing": int(closing),
            "comp_id": int(cid),
            "n_tumor_points_inside": int(n_fg_in.get(cid, 0)),
            "n_bg_points_inside": int(n_bg_in.get(cid, 0)),
            "n_voxels": k,
            "volume_ml": k * vox_ml,
            "suv_max": float(s.max()),
            "suv_mean": float(s.mean()),
            "suv_p90": float(np.percentile(s, 90)),
            "prob_max": float(p.max()),
            "prob_mean": float(p.mean()),
            "prob_p90": float(np.percentile(p, 90)),
            "ct_mean_hu": float(ct_v[sel].mean()),
            "z_frac": float(zf),
            "dist_surface_mm_max": float(dist_v[sel].max()),
            "dist_surface_mm_mean": float(dist_v[sel].mean()),
            "n_base_voxels": int(base_v[sel].sum()),
            "n_gt_voxels_inside": int((hit > 0).sum()),
            "best_gt_iou": float(best_iou),
            "n_gt_matches": len(matches),
            "gt_matches": matches,
            "is_tp": int(len(matches) > 0),
        })
    return rows


def analyse_case(tag, ct, pet, gt, prob, base_mask, spacing, affine, thresholds,
                 iteration=0, fg_pts=None, bg_pts=None, geom=None, gt_pack=None,
                 mask_override=None, closings=(0,)):
    vox_ml = voxel_volume_ml(spacing)
    axis, sign = superior_axis(affine)
    if geom is None:
        geom = body_frame(ct, spacing, axis, sign)
    dist, dstride, lo_b, hi_b = geom
    span = max(hi_b - lo_b, 1)

    gt_lab, gt_counts = gt_pack if gt_pack is not None else gt_components(gt)
    n_gt = int(max(len(gt_counts) - 1, 0))

    comp_rows: List[Dict[str, Any]] = []
    case_rows: List[Dict[str, Any]] = []
    for thr, closing in ((t, c) for t in thresholds for c in closings):
        if mask_override is not None:
            # scored-mask mode: components come from the run's own output while
            # base_mask stays the model's, so n_base_voxels == 0 marks a component the
            # interaction layer created.
            fg = mask_override.astype(bool)
        else:
            fg = base_mask.astype(bool) if abs(thr - 0.5) < 1e-9 else (prob >= thr)
        if closing:
            fg = close_mask(np.ascontiguousarray(fg), int(closing))
        n_fg = int(fg.sum())
        rows = analyse_threshold(tag, tracer_from_name(tag), thr, fg, base_mask, pet, ct,
                                 prob, gt_lab, gt_counts, spacing, vox_ml, axis, sign,
                                 lo_b, span, dist, dstride,
                                 iteration=iteration, fg_pts=fg_pts, bg_pts=bg_pts,
                                 closing=closing)
        comp_rows.extend(rows)
        matched = set()
        for r in rows:
            matched.update(r["gt_matches"])
        tp = len(matched)
        fp = sum(1 for r in rows if not r["gt_matches"])
        fn = n_gt - tp
        f1 = float("nan") if (tp + fn) == 0 else (0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn))
        n_pred_vox = sum(r["n_voxels"] for r in rows)
        n_inter = sum(r["n_gt_voxels_inside"] for r in rows)
        n_gt_vox = int((np.asarray(gt) > 0).sum())
        case_rows.append({
            "case": tag, "tracer": tracer_from_name(tag), "iteration": int(iteration),
            "threshold": thr, "closing": int(closing),
            "n_gt_lesions": n_gt, "n_components": len(rows),
            "volume_ml": float(sum(r["volume_ml"] for r in rows)),
            "tp": tp, "fp": fp, "fn": fn, "f1": f1,
            "n_fg_voxels": n_fg, "flooded": int(n_fg > MAX_FG_VOXELS),
            "n_pred_voxels": int(n_pred_vox), "n_gt_voxels": n_gt_vox,
            "n_inter_voxels": int(n_inter), "voxel_ml": vox_ml,
            "dice": (float("nan") if n_gt_vox == 0 else
                     (2.0 * n_inter / (n_pred_vox + n_gt_vox) if (n_pred_vox + n_gt_vox) else 0.0)),
        })

    # --- ground-truth lesion rows ------------------------------------------------
    lesion_rows: List[Dict[str, Any]] = []
    if n_gt:
        gidx = np.argwhere(gt_lab > 0)
        gt_v = gt_lab[tuple(gidx.T)]
        pet_g = np.asarray(pet, dtype=np.float32)[tuple(gidx.T)]
        prob_g = prob[tuple(gidx.T)].astype(np.float32)
        dist_g = dist[tuple((gidx // dstride).T)]
        cen_axis = gidx[:, axis].astype(np.float32)
        detected = {thr: set() for thr in thresholds}
        for r in comp_rows:
            if r["closing"] == 0:
                detected[r["threshold"]].update(r["gt_matches"])
        for g in range(1, n_gt + 1):
            sel = gt_v == g
            k = int(sel.sum())
            zf = (float(cen_axis[sel].mean()) - lo_b) / span
            if sign < 0:
                zf = 1.0 - zf
            row = {
                "case": tag, "tracer": tracer_from_name(tag), "gt_id": g,
                "n_voxels": k, "volume_ml": k * vox_ml,
                "suv_max": float(pet_g[sel].max()), "suv_mean": float(pet_g[sel].mean()),
                "prob_max": float(prob_g[sel].max()),
                "prob_p90": float(np.percentile(prob_g[sel], 90)),
                "prob_mean": float(prob_g[sel].mean()),
                "frac_above_050": float((prob_g[sel] >= 0.5).mean()),
                "frac_above_030": float((prob_g[sel] >= 0.3).mean()),
                "z_frac": float(zf),
                "dist_surface_mm_mean": float(dist_g[sel].mean()),
            }
            for thr in thresholds:
                row[f"det_{thr:g}"] = int(g in detected[thr])
            lesion_rows.append(row)
    return comp_rows, case_rows, lesion_rows


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
            w.writerow({k: (json.dumps(v) if isinstance(v, list) else v)
                        for k, v in r.items()})
    os.replace(tmp, path)


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--image_dir", required=True)
    ap.add_argument("--label_dir", required=True)
    ap.add_argument("--cache_root", required=True,
                    help="<cache_dir>/<predictor_key>, the namespace written by "
                         "interactive_eval.py --cache_dir --cache_probabilities")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--thresholds", type=float, nargs="+",
                    default=[0.5, 0.4, 0.3, 0.2, 0.1, 0.05, 0.02])
    ap.add_argument("--positives_only", action="store_true")
    ap.add_argument("--cases", nargs="*", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--run_dir", default=None,
                    help="out_dir of a --predictor postproc run: analyse the base model "
                         "prediction of every iteration, keyed by that iteration's "
                         "accumulated scribble set, instead of iteration 0 only")
    ap.add_argument("--iterations", type=int, nargs="+", default=[0, 1, 2, 3, 4, 5])
    ap.add_argument("--stem_suffix", default="",
                    help="appended to the cache stem, e.g. `_prev0` for a stateful "
                         "predictor whose key folds in a hash of the previous mask")
    ap.add_argument("--closings", type=int, nargs="+", default=[0],
                    help="also emit rows for the mask closed by this many voxels")
    ap.add_argument("--final_masks", action="store_true",
                    help="with --run_dir: take the components from the run's scored mask "
                         "(<tag>/iter_k.nii.gz, after compliance) instead of the base "
                         "model's; the softmax still comes from the cache")
    args = ap.parse_args(argv)

    os.makedirs(args.out_dir, exist_ok=True)
    cts = sorted(os.path.join(args.image_dir, f) for f in os.listdir(args.image_dir)
                 if "_0000" in f)
    pets = sorted(os.path.join(args.image_dir, f) for f in os.listdir(args.image_dir)
                  if "_0001" in f)
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
    if args.limit:
        cases = cases[:args.limit]

    comp_rows: List[Dict[str, Any]] = []
    case_rows: List[Dict[str, Any]] = []
    lesion_rows: List[Dict[str, Any]] = []
    n_missing = 0
    iterations = args.iterations if args.run_dir else [0]
    for n, (tag, ctp, petp, labp) in enumerate(cases, 1):
        t0 = time.time()
        gt = np.asanyarray(nib.load(labp).dataobj)
        if args.positives_only and not (gt > 0).any():
            print(f"[{n}/{len(cases)}] skip lesion-free {tag[:56]}", flush=True)
            continue
        probe, _ = load_cached(args.cache_root, tag,
                               empty_scribble_hash() + args.stem_suffix)
        if probe is None:
            n_missing += 1
            print(f"[{n}/{len(cases)}] MISSING cache entry {tag[:56]}", flush=True)
            continue
        ct_img, pet_img = nib.load(ctp), nib.load(petp)
        ct = np.asanyarray(ct_img.dataobj)
        pet = np.asanyarray(pet_img.dataobj)
        spacing = tuple(float(z) for z in pet_img.header.get_zooms()[:3])
        axis, sign = superior_axis(pet_img.affine)
        geom = body_frame(ct, spacing, axis, sign)
        gt_pack = gt_components(gt)

        got = 0
        for it in iterations:
            if args.run_dir is None:
                stem, pts = empty_scribble_hash() + args.stem_suffix, (None, None)
            else:
                stem, pts = iteration_scribble_hash(args.run_dir, tag, it, ct.shape, spacing)
                if stem is None:
                    continue
                stem += args.stem_suffix
            base_mask, prob = load_cached(args.cache_root, tag, stem)
            if base_mask is None or prob is None:
                continue
            override = None
            if args.final_masks:
                fpath = os.path.join(args.run_dir, tag, f"iter_{it}.nii.gz")
                if not os.path.isfile(fpath):
                    continue
                override = (np.asanyarray(nib.load(fpath).dataobj) > 0).astype(np.uint8)
            got += 1
            c, k, l = analyse_case(tag, ct, pet, gt, prob, base_mask, spacing,
                                   pet_img.affine, args.thresholds, iteration=it,
                                   fg_pts=pts[0], bg_pts=pts[1], geom=geom,
                                   gt_pack=gt_pack, mask_override=override,
                                   closings=args.closings)
            comp_rows.extend(c)
            case_rows.extend(k)
            if it == iterations[0]:
                lesion_rows.extend(l)
        if got == 0:
            n_missing += 1
            print(f"[{n}/{len(cases)}] MISSING every iteration {tag[:46]}", flush=True)
            continue
        write_csv(os.path.join(args.out_dir, "components.csv"), comp_rows)
        write_csv(os.path.join(args.out_dir, "cases.csv"), case_rows)
        write_csv(os.path.join(args.out_dir, "lesions.csv"), lesion_rows)
        base = [r for r in case_rows if r["case"] == tag and r["iteration"] == iterations[0]
                and abs(r["threshold"] - 0.5) < 1e-9 and r["closing"] == 0]
        b = base[0] if base else {"n_gt_lesions": -1, "n_components": -1, "tp": -1,
                                  "fp": -1, "f1": float("nan")}
        print(f"[{n}/{len(cases)}] {tag[:42]:42s} its={got} gt={b['n_gt_lesions']:3d} "
              f"ncomp@.5={b['n_components']:3d} tp={b['tp']} fp={b['fp']} "
              f"f1={b['f1']:.3f} ({time.time() - t0:.1f}s)", flush=True)
        del ct, pet, gt, geom, gt_pack

    print(f"DONE cases={len(set(r['case'] for r in case_rows))} missing={n_missing} "
          f"-> {args.out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
