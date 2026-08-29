"""Run the interaction layer on real evalset cases and score it against ground truth.

Replays the evaluator's inner loop, including the official simulate_scribble_from_label,
against a stand-in predictor, and reports per iteration Dice, lesion-level F1
(18-connectivity, IoU >= 0.1) and an audit of everything a background scribble deleted.
Needs the dataset, so it lives outside tests/. Runs on CPU:

    python src/postproc/tools/real_case_check.py \
        --data /content/drive/MyDrive/autoPET/evalset --n 6 --iters 6
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.dirname(os.path.dirname(_HERE))
_REPO = os.path.dirname(_SRC)
for _p in (_SRC, os.path.join(_REPO, "autoPETV"), os.path.join(_REPO, "autoPETV", "interactive")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import cc3d  # noqa: E402
import nibabel as nib  # noqa: E402

from postproc.compliance import check_constraints  # noqa: E402
from postproc.pipeline import PostProcPredictor  # noqa: E402

import simulate_scribbles as sim  # noqa: E402


# ---------------------------------------------------------------------------
def dice(pred, gt):
    pred, gt = pred > 0, gt > 0
    denom = int(pred.sum()) + int(gt.sum())
    return 1.0 if denom == 0 else 2.0 * int((pred & gt).sum()) / denom


def lesion_f1(pred, gt, iou_threshold=0.1, connectivity=18):
    """The scorer's detection metric: 18-connectivity, IoU >= 0.1, multi-assignment ok."""
    pred = np.ascontiguousarray(pred > 0).view(np.uint8)
    gt = np.ascontiguousarray(gt > 0).view(np.uint8)
    if gt.sum() == 0:
        return float("nan")
    pl = cc3d.connected_components(pred, connectivity=connectivity)
    gl = cc3d.connected_components(gt, connectivity=connectivity)
    n_pred, n_gt = int(pl.max()), int(gl.max())
    if n_gt == 0:
        return float("nan")
    if n_pred == 0:
        return 0.0
    pv = np.bincount(pl.ravel(), minlength=n_pred + 1)
    gv = np.bincount(gl.ravel(), minlength=n_gt + 1)
    both = (pl > 0) & (gl > 0)
    matched_gt, matched_pred = set(), set()
    if both.any():
        pairs = np.stack([gl[both].ravel(), pl[both].ravel()], axis=1)
        uniq, cnt = np.unique(pairs, axis=0, return_counts=True)
        for (g, p), inter in zip(uniq, cnt):
            iou = inter / (gv[g] + pv[p] - inter)
            if iou >= iou_threshold:
                matched_gt.add(int(g))
                matched_pred.add(int(p))
    tp = len(matched_gt)
    fn = n_gt - tp
    fp = n_pred - len(matched_pred)
    return 0.0 if tp == 0 else (2 * tp) / (2 * tp + fp + fn)


def official_scribble(error_mask, strategy):
    out = sim.simulate_scribble_from_label(error_mask.astype(np.uint8), strategy)
    if len(out) == 2:  # the empty-region 2-tuple the reference loop unpacks as 3
        return [], 0
    coords, _cls, size = out
    return coords, int(size)


class SuvThreshold:
    """Crude stand-in: a global SUV threshold.

    On whole-body FDG this segments brain, heart, kidneys, bladder and liver, so Dice is
    near zero. Useful for stressing the compliance rules on real anatomy and for timing;
    use CorruptedGroundTruth to score the interaction itself.
    """

    name = "suv_threshold"

    def __init__(self, thr=3.0):
        self.thr = thr

    def predict(self, ct, pet, spacing, scribbles=None, prev_pred=None,
                case_cache_dir=None, *, return_probabilities=False, **kw):
        pet = np.asarray(pet)
        mask = (pet >= self.thr).astype(np.uint8)
        if return_probabilities:
            p = np.clip((pet - self.thr) / (2 * self.thr) + 0.5, 0.0, 1.0).astype(np.float32)
            return mask, p
        return mask


class CorruptedGroundTruth:
    """Stand-in model with a realistic error profile: the GT, deliberately damaged.

    Seeded per case, it drops the `drop` largest lesions (false negatives), dilates the
    rest by `dilate` voxels (boundary over-segmentation) and pastes `n_fp` blobs at the
    hottest non-lesion locations (false positives). The probability map is 0.95 on the
    surviving core, 0.60 on the rim and 0.55 on the pasted blobs, so the
    split-not-delete rule has a confidence signal to work with.
    """

    name = "corrupted_gt"

    def __init__(self, gt, pet, spacing, drop=1, dilate=1, n_fp=2, fp_radius_mm=12.0, seed=0):
        from scipy import ndimage

        rng = np.random.default_rng(seed)
        gt = np.ascontiguousarray(gt > 0)
        labels = cc3d.connected_components(gt.view(np.uint8), connectivity=18)
        counts = cc3d.statistics(labels)["voxel_counts"]
        order = np.argsort(counts[1:])[::-1] + 1          # largest first, deterministic
        dropped = set(int(v) for v in order[:drop])
        kept = gt & ~np.isin(labels, list(dropped)) if dropped else gt

        core = kept.copy()
        grown = ndimage.binary_dilation(kept, iterations=int(dilate)) if dilate else kept
        rim = grown & ~core

        fp = np.zeros(gt.shape, dtype=bool)
        hot = np.asarray(pet, dtype=np.float32).copy()
        hot[ndimage.binary_dilation(gt, iterations=6)] = -np.inf   # never on a real lesion
        rad = np.maximum((fp_radius_mm / np.asarray(spacing)).astype(int), 1)
        for _ in range(int(n_fp)):
            idx = np.unravel_index(int(np.argmax(hot)), hot.shape)
            sl = tuple(
                slice(max(int(c) - int(r), 0), min(int(c) + int(r) + 1, int(n)))
                for c, r, n in zip(idx, rad, gt.shape)
            )
            fp[sl] = True
            hot[sl] = -np.inf

        self.mask = (core | rim | fp).astype(np.uint8)
        self.prob = np.zeros(gt.shape, dtype=np.float32)
        self.prob[fp] = 0.55
        self.prob[rim] = 0.60
        self.prob[core] = 0.95
        self.parts = {"core": core, "rim": rim, "fp": fp, "dropped": gt & ~kept}
        self.rng = rng

    def predict(self, ct, pet, spacing, scribbles=None, prev_pred=None,
                case_cache_dir=None, *, return_probabilities=False, **kw):
        m = self.mask.copy()
        return (m, self.prob) if return_probabilities else m


# ---------------------------------------------------------------------------
def build_predictor(kind, gt, pet, spacing, args):
    if kind == "suv":
        return SuvThreshold(thr=args.suv_threshold)
    return CorruptedGroundTruth(
        gt, pet, spacing, drop=args.drop, dilate=args.dilate, n_fp=args.n_fp
    )


def run_case(ct_path, pet_path, gt_path, strategy, iters, cfg, args, verbose=True):
    ct_img, pet_img, gt_img = (nib.load(p) for p in (ct_path, pet_path, gt_path))
    ct = np.asarray(ct_img.dataobj, dtype=np.float32)
    pet = np.asarray(pet_img.dataobj, dtype=np.float32)
    gt = np.asarray(gt_img.dataobj, dtype=np.uint8)
    spacing = tuple(float(z) for z in pet_img.header.get_zooms()[:3])

    base = build_predictor(args.predictor, gt, pet, spacing, args)
    pp = PostProcPredictor(base, cfg)
    data = {"tumor": [], "background": []}
    rows, pred = [], None
    audits = []

    for it in range(iters):
        if it > 0 and gt.sum() > 0 and pred is not None:
            overseg = (pred == 1) & (gt == 0)
            underseg = (pred == 0) & (gt == 1)
            s_bg, n_fp = official_scribble(overseg, strategy)
            s_fg, n_fn = official_scribble(underseg, strategy)
            if n_fp <= n_fn:
                data["tumor"] = data["tumor"] + s_fg
            else:
                data["background"] = data["background"] + s_bg
            data = sim.gc_to_swfastedit_format(
                json.loads(json.dumps(sim.scribbles_to_gc_format(data)))
            )

        t0 = time.perf_counter()
        pred = pp.predict(ct, pet, spacing, data, None, None, affine=pet_img.affine, gt=gt)
        dt = time.perf_counter() - t0
        info = pp.last_info

        # An iteration with zero false positives (or zero false negatives) makes the
        # reference loop raise before it writes the scribble json, and every remaining
        # iteration of that case then scores 0.
        zero_fp = bool(gt.sum() > 0 and not ((pred == 1) & (gt == 0)).any())
        zero_fn = bool(gt.sum() > 0 and not ((pred == 0) & (gt == 1)).any())

        ok = check_constraints(pred, data["tumor"], data["background"])
        assert ok["ok"], f"constraint violated at iteration {it}: {ok}"
        if "gt_audit" in info.get("bg_compliance", {}):
            audits.append(info["bg_compliance"]["gt_audit"])

        rows.append(
            {
                "iteration": it,
                "dice": dice(pred, gt),
                "f1": lesion_f1(pred, gt),
                "n_tumor": info["n_tumor"],
                "n_background": info["n_background"],
                "gate": info["negative_gate_fired"],
                "t": dt,
                "t_post": dt - info.get("t_base_predict", 0.0),
                "vol_ml": info["final_volume_ml"],
                "zero_fp": zero_fp,
                "zero_fn": zero_fn,
                "empty_without_gate": info["empty_without_gate"],
            }
        )
        if verbose:
            r = rows[-1]
            print(
                f"    it{it}: dice={r['dice']:.3f} f1={r['f1']:.3f} "
                f"fg={r['n_tumor']:3d} bg={r['n_background']:3d} "
                f"vol={r['vol_ml']:7.1f}mL t={r['t']:5.1f}s (post {r['t_post']:4.1f}s)"
                + ("  GATE" if r["gate"] else "")
                + ("  ZERO-FP!" if r["zero_fp"] else "")
                + ("  ZERO-FN!" if r["zero_fn"] else ""),
                flush=True,
            )
    return rows, audits, gt.sum() == 0


def auc(values):
    v = np.asarray([x for x in values], dtype=float)
    return float(np.trapezoid(v, np.arange(len(v)))) if hasattr(np, "trapezoid") else float(
        np.trapz(v, np.arange(len(v)))
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/content/drive/MyDrive/autoPET/evalset")
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--iters", type=int, default=6)
    ap.add_argument("--strategy", default="centerline",
                    choices=["centerline", "boundary", "random"])
    ap.add_argument("--predictor", default="corrupt", choices=["corrupt", "suv"],
                    help="corrupt: GT with a realistic error profile (default); "
                         "suv: a global SUV threshold (anatomy/timing stress test only)")
    ap.add_argument("--suv-threshold", type=float, default=3.0)
    ap.add_argument("--drop", type=int, default=1, help="largest lesions to drop")
    ap.add_argument("--dilate", type=int, default=1, help="voxels of over-segmentation")
    ap.add_argument("--n-fp", type=int, default=2, help="false-positive blobs to paste")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    img_dir = os.path.join(args.data, "imagesTr")
    lab_dir = os.path.join(args.data, "labelsTr")
    stems = sorted(f[: -len("_0000.nii.gz")] for f in os.listdir(img_dir) if f.endswith("_0000.nii.gz"))
    stems = stems[: args.n]

    cfg = {"tracer": "auto"}
    results = {}
    for stem in stems:
        ct = os.path.join(img_dir, f"{stem}_0000.nii.gz")
        pet = os.path.join(img_dir, f"{stem}_0001.nii.gz")
        gt = os.path.join(lab_dir, f"{stem}.nii.gz")
        if not os.path.exists(gt):
            continue
        print(f"\n[{stem[:44]}] strategy={args.strategy} predictor={args.predictor}", flush=True)
        rows, audits, negative = run_case(
            ct, pet, gt, args.strategy, args.iters, cfg, args
        )
        results[stem] = {"rows": rows, "audits": audits, "negative": negative}
        print(
            f"    AUC-Dice={auc([r['dice'] for r in rows]):.3f} "
            f"AUC-F1={auc([r['f1'] for r in rows]):.3f} "
            f"(max {args.iters - 1}.0)"
            + ("   [lesion-free case]" if negative else ""),
            flush=True,
        )

    # -- summary ---------------------------------------------------------
    print("\n================ summary ================")
    print(f"predictor                          : {args.predictor}  strategy={args.strategy}")
    dmg = [a for r in results.values() for a in r["audits"]]
    over_half = sum(a["n_gt_lesions_over_half_removed"] for a in dmg)
    print(f"cases                              : {len(results)}")
    print(f"background-scribble deletions audited: {len(dmg)}")
    print(f"GT lesions >50% removed by a deletion: {over_half}")
    if dmg:
        print(f"worst single-lesion fraction removed : {max(a['worst_fraction_removed'] for a in dmg):.3f}")
        print(f"total GT volume removed              : {sum(a['removed_gt_ml'] for a in dmg):.2f} mL")
    rows_all = [r for v in results.values() for r in v["rows"]]
    n_zero_fp = sum(1 for r in rows_all if r["zero_fp"])
    n_zero_fn = sum(1 for r in rows_all if r["zero_fn"])
    n_empty = sum(1 for r in rows_all if r["empty_without_gate"])
    print(f"iterations with ZERO false positives  : {n_zero_fp}/{len(rows_all)}"
          "   (each one zeroes the rest of that case under the shipped evaluator)")
    print(f"iterations with ZERO false negatives  : {n_zero_fn}/{len(rows_all)}")
    print(f"empty output without the gate deciding: {n_empty}/{len(rows_all)}")
    times = [r["t_post"] for v in results.values() for r in v["rows"]]
    if times:
        print(f"post-processing per call             : "
              f"mean {np.mean(times):.2f}s  max {np.max(times):.2f}s")
    pos = [v for v in results.values() if not v["negative"]]
    if pos:
        a0 = np.mean([v["rows"][0]["dice"] for v in pos])
        aN = np.mean([v["rows"][-1]["dice"] for v in pos])
        f0 = np.nanmean([v["rows"][0]["f1"] for v in pos])
        fN = np.nanmean([v["rows"][-1]["f1"] for v in pos])
        print(f"positives: mean dice {a0:.3f} -> {aN:.3f}   mean F1 {f0:.3f} -> {fN:.3f}")
        print(f"positives: mean AUC-Dice {np.mean([auc([r['dice'] for r in v['rows']]) for v in pos]):.3f}"
              f"  mean AUC-F1 {np.mean([auc([r['f1'] for r in v['rows']]) for v in pos]):.3f}"
              f"  (max {args.iters - 1}.0)")
    neg = [v for v in results.values() if v["negative"]]
    if neg:
        emptied = sum(1 for v in neg if all(r["vol_ml"] == 0.0 for r in v["rows"]))
        print(f"lesion-free cases emptied at every iteration: {emptied}/{len(neg)}")
    for name, v in results.items():
        d = [r["dice"] for r in v["rows"]]
        print(f"  {name[:40]:42s} dice {d[0]:.3f} -> {d[-1]:.3f}"
              + ("  [neg]" if v["negative"] else ""))

    if args.out:
        with open(args.out, "w") as f:
            json.dump(results, f, indent=1, default=float)
        print(f"\nwritten: {args.out}")


if __name__ == "__main__":
    main()
