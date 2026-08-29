"""Iteration-0 study driven entirely from the cached base predictions (no GPU).

The base model's iteration-0 prediction is a fixed function of the case (empty scribble
set, no previous mask), so it is in the evaluation cache for every case.  Everything the
post-processing layer does at iteration 0 can therefore be re-scored offline, which is
what makes a parameter sweep affordable: only the layer runs, never the network.

What it reports per configuration: Dice@0, the lesion-level F1@0 (18-connectivity,
IoU >= 0.1, the scorer's rule), and how many *swallowed* lesions become matched -- a
swallowed lesion being one that a predicted component covers while being far larger than
it, so the union's IoU falls below 0.1 and both sides count as errors.
"""
from __future__ import annotations

import argparse, glob, json, os, sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import nibabel as nib
import cc3d

sys.path.insert(0, "/content/autopet/src")
from postproc.config import PostProcConfig            # noqa: E402
from postproc.pipeline import PostProcPredictor       # noqa: E402

IT0_STEM = "be2ff1e96e596758_prev0.npz"


def lesion_f1(pred, gt, iou=0.1, conn=18):
    pred = np.ascontiguousarray(pred > 0).view(np.uint8)
    gt = np.ascontiguousarray(gt > 0).view(np.uint8)
    if gt.sum() == 0:
        return float("nan"), 0, 0, 0
    pl = cc3d.connected_components(pred, connectivity=conn)
    gl = cc3d.connected_components(gt, connectivity=conn)
    npd, ngt = int(pl.max()), int(gl.max())
    if ngt == 0:
        return float("nan"), 0, 0, 0
    if npd == 0:
        return 0.0, 0, 0, ngt
    pv = np.bincount(pl.ravel(), minlength=npd + 1)
    gv = np.bincount(gl.ravel(), minlength=ngt + 1)
    both = (pl > 0) & (gl > 0)
    mg, mp = set(), set()
    if both.any():
        pairs = np.stack([gl[both].ravel(), pl[both].ravel()], 1)
        u, c = np.unique(pairs, axis=0, return_counts=True)
        for (g, p), inter in zip(u, c):
            if inter / (gv[g] + pv[p] - inter) >= iou:
                mg.add(int(g)); mp.add(int(p))
    tp = len(mg); fn = ngt - tp; fp = npd - len(mp)
    return (0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn)), tp, fp, fn


def dice(a, b):
    a, b = a > 0, b > 0
    d = int(a.sum()) + int(b.sum())
    return 1.0 if d == 0 else 2 * int((a & b).sum()) / d


def swallowed(pred, gt, conn=18, ratio=3.0, cover=0.5):
    """GT lesions that a predicted component covers but is ``ratio`` times larger than.

    Returns the set of GT ids that are unmatched *and* mostly inside such a component --
    exactly the failure the split operator targets.
    """
    gl = cc3d.connected_components(np.ascontiguousarray(gt > 0).view(np.uint8), connectivity=conn)
    pl = cc3d.connected_components(np.ascontiguousarray(pred > 0).view(np.uint8), connectivity=conn)
    ngt, npd = int(gl.max()), int(pl.max())
    if ngt == 0 or npd == 0:
        return set()
    gv = np.bincount(gl.ravel(), minlength=ngt + 1)
    pv = np.bincount(pl.ravel(), minlength=npd + 1)
    out = set()
    both = (pl > 0) & (gl > 0)
    if not both.any():
        return out
    pairs = np.stack([gl[both].ravel(), pl[both].ravel()], 1)
    u, c = np.unique(pairs, axis=0, return_counts=True)
    best = {}
    for (g, p), inter in zip(u, c):
        iou = inter / (gv[g] + pv[p] - inter)
        rec = inter / gv[g]
        cur = best.get(int(g))
        if cur is None or inter > cur[0]:
            best[int(g)] = (inter, int(p), iou, rec)
    for g, (inter, p, iou, rec) in best.items():
        if iou < 0.1 and rec >= cover and pv[p] >= ratio * gv[g]:
            out.add(g)
    return out


class Cached:
    """Serves the cached iteration-0 base prediction."""
    name = "cached_it0"

    def __init__(self, mask, prob):
        self.mask, self.prob = mask, prob

    def predict(self, ct, pet, spacing, scribbles=None, prev_pred=None, case_cache_dir=None,
                *, return_probabilities=False, **kw):
        return (self.mask.copy(), self.prob) if return_probabilities else self.mask.copy()


def one_case(job):
    tag, cachedir, img, lab, base_cfg, variants = job
    f = os.path.join(cachedir, tag, IT0_STEM)
    if not os.path.exists(f):
        return None
    z = np.load(f)
    mask0 = np.ascontiguousarray(z["mask"])
    prob0 = np.ascontiguousarray(z["prob_fg"]).astype(np.float32) / 255.0
    stem = tag[:-5] if tag.endswith("_0000") else tag
    need = [os.path.join(img, tag + ".nii.gz"),
            os.path.join(img, tag.replace("_0000", "_0001") + ".nii.gz"),
            os.path.join(lab, stem + ".nii.gz")]
    if not all(os.path.exists(x) for x in need):
        return None
    pet_img = nib.load(os.path.join(img, tag.replace("_0000", "_0001") + ".nii.gz"))
    pet = np.asarray(pet_img.dataobj, dtype=np.float32)
    ct = np.asarray(nib.load(os.path.join(img, tag + ".nii.gz")).dataobj, dtype=np.float32)
    gt = np.asarray(nib.load(os.path.join(lab, stem + ".nii.gz")).dataobj, dtype=np.uint8)
    spacing = tuple(float(v) for v in pet_img.header.get_zooms()[:3])

    rows = {}
    for name, over in variants:
        cfg = PostProcConfig.from_dict({**base_cfg, **over})
        pp = PostProcPredictor(Cached(mask0, prob0), cfg)
        out = pp.predict(ct, pet, spacing, {"tumor": [], "background": []}, None, None,
                         affine=pet_img.affine, case_name=tag)
        f1, tp, fp, fn = lesion_f1(out, gt)
        rows[name] = dict(dice=dice(out, gt), f1=f1, tp=tp, fp=fp, fn=fn,
                          vol_ml=float(int(out.sum()) * float(np.prod(spacing)) / 1000.0),
                          swallowed=sorted(swallowed(out, gt)),
                          swallowed10=sorted(swallowed(out, gt, ratio=10.0)),
                          split=(pp.last_info.get("cleanup", {}) or {}).get("split", {}))
    return tag, rows, bool(gt.sum() == 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="/content/work/cache/interactive_nnunet_d3fda4c702")
    ap.add_argument("--images", default="/content/work/evalset/imagesTr")
    ap.add_argument("--labels", default="/content/work/evalset/labelsTr")
    ap.add_argument("--config", default="/content/autopet/submission/postproc_config.json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--mode", default="a12", choices=["a12", "a12strict", "gate"])
    args = ap.parse_args()

    def _strip(d):
        # the shipped json carries documentation keys the dataclass does not accept
        return {k: (_strip(v) if isinstance(v, dict) else v)
                for k, v in d.items() if not k.startswith("_")}
    base_cfg = _strip(json.load(open(args.config))) if os.path.exists(args.config) else {}
    tags = sorted(os.path.basename(d) for d in glob.glob(os.path.join(args.cache, "*"))
                  if os.path.isdir(d))
    tags = [t for t in tags if os.path.exists(os.path.join(args.cache, t, IT0_STEM))]
    if args.limit:
        tags = tags[:args.limit]

    if args.mode == "a12":
        variants = [("base", {})]
        for x in (5.0, 10.0, 20.0):
            for h in (0.5, 1.0, 2.0):
                variants.append((f"X{x:g}_h{h:g}", {"cleanup": {
                    "split_large_components": True,
                    "split_min_volume_ml": x, "split_h_depth_suv": h}}))
    elif args.mode == "a12strict":
        # the permissive sweep showed FP growing 2.5x faster than TP: a watershed on a
        # big heterogeneous component yields many fragments, most matching nothing.
        # Test the opposite end -- cut in two only, into substantial pieces.
        variants = [("base", {})]
        for h in (2.0, 3.0):
            for mf in (2.0, 5.0):
                variants.append((f"strict_h{h:g}_f{mf:g}", {"cleanup": {
                    "split_large_components": True, "split_min_volume_ml": 20.0,
                    "split_h_depth_suv": h, "split_max_fragments": 2,
                    "split_min_fragment_ml": mf}}))
    else:
        variants = [("base", {})]

    jobs = [(t, args.cache, args.images, args.labels, base_cfg, variants) for t in tags]
    res = {}
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for i, r in enumerate(ex.map(one_case, jobs), 1):
            if r is None:
                continue
            tag, rows, neg = r
            res[tag] = {"rows": rows, "negative": neg}
            if i % 10 == 0:
                print(f"  {i}/{len(jobs)}", flush=True)
    json.dump(res, open(args.out, "w"))
    print("cases:", len(res), "->", args.out)


if __name__ == "__main__":
    main()
