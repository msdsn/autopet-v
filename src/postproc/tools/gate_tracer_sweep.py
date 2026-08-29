"""A13 -- per-tracer lesion-free gate thresholds, fitted and cross-validated offline.

The gate only acts while no scribble has been seen, so it acts at iteration 0, and its
decision is a function of the *base* prediction alone (it runs before cleanup).  Both are
in the evaluation cache, so the whole fit needs no GPU.

Scoring model, which is exact for negatives and a bound for positives:

* a lesion-free case never receives a scribble, so its six iterations are identical:
  AUC-Dice = 5.0 if the gate empties it, 0.0 otherwise, and it is excluded from AUC-DMM;
* a positive case that the gate empties loses only iteration 0, which carries weight 0.5
  in the trapezoid, and the next iteration delivers a tumor scribble: it gives up
  0.5 * Dice@0 and 0.5 * DMM@0 relative to leaving it alone.

The reported objective is the 50/50 rank score (mean AUC-Dice + mean nan-mean AUC-DMM),
which is what the challenge ranks on -- the earlier sweep maximised AUC-Dice alone, in
which DMM@0 has no weight at all.
"""
from __future__ import annotations

import argparse, glob, json, os, sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import nibabel as nib

sys.path.insert(0, "/content/autopet/src")
from postproc.config import PostProcConfig                    # noqa: E402
from postproc.negative_gate import negative_gate_features     # noqa: E402
from postproc.pipeline import PostProcPredictor               # noqa: E402
from postproc.tools.iter0_sweep import Cached, dice, lesion_f1, IT0_STEM   # noqa: E402


def one_case(job):
    tag, cachedir, img, lab, base_cfg = job
    f = os.path.join(cachedir, tag, IT0_STEM)
    stem = tag[:-5] if tag.endswith("_0000") else tag
    need = [f, os.path.join(img, tag + ".nii.gz"),
            os.path.join(img, tag.replace("_0000", "_0001") + ".nii.gz"),
            os.path.join(lab, stem + ".nii.gz")]
    if not all(os.path.exists(x) for x in need):
        return None
    z = np.load(f)
    mask0 = np.ascontiguousarray(z["mask"])
    prob0 = np.ascontiguousarray(z["prob_fg"]).astype(np.float32) / 255.0
    pet_img = nib.load(need[2])
    pet = np.asarray(pet_img.dataobj, dtype=np.float32)
    ct = np.asarray(nib.load(need[1]).dataobj, dtype=np.float32)
    gt = np.asarray(nib.load(need[3]).dataobj, dtype=np.uint8)
    spacing = tuple(float(v) for v in pet_img.header.get_zooms()[:3])

    # gate features come from the base mask -- the gate runs before cleanup
    feats = negative_gate_features(mask0, pet, prob0, spacing)

    # the pipeline's output with the gate switched off
    cfg = PostProcConfig.from_dict({**base_cfg, "negative_gate": {"enabled": False}})
    pp = PostProcPredictor(Cached(mask0, prob0), cfg)
    out = pp.predict(ct, pet, spacing, {"tumor": [], "background": []}, None, None,
                     affine=pet_img.affine, case_name=tag)
    f1, tp, fp, fn = lesion_f1(out, gt)
    return tag, dict(
        tracer=pp.last_info.get("tracer"),
        negative=bool(gt.sum() == 0),
        gate_volume_ml=float(feats["total_volume_ml"]),
        dice_ungated=dice(out, gt),
        dmm_ungated=(None if not np.isfinite(f1) else float(f1)),
    )


def score(rows, t_fdg, t_psma):
    """Mean AUC-Dice and nan-mean AUC-DMM under the thresholds, over the given rows."""
    d, m = [], []
    for r in rows:
        lim = t_fdg if r["tracer"] == "fdg" else t_psma
        fires = r["gate_volume_ml"] < lim
        if r["negative"]:
            d.append(5.0 if fires else 0.0)          # no scribble ever arrives
        else:
            # emptied at iteration 0 costs that iteration's half-weight; the rest of the
            # trajectory is unchanged, so only the iteration-0 terms differ
            d.append(-0.5 * r["dice_ungated"] if fires else 0.0)
            if r["dmm_ungated"] is not None:
                m.append(-0.5 * r["dmm_ungated"] if fires else 0.0)
    return float(np.mean(d)), (float(np.mean(m)) if m else 0.0)


def objective(rows, t_fdg, t_psma):
    a, b = score(rows, t_fdg, t_psma)
    return a + b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="/content/work/cache/interactive_nnunet_d3fda4c702")
    ap.add_argument("--images", default="/content/work/evalset/imagesTr")
    ap.add_argument("--labels", default="/content/work/evalset/labelsTr")
    ap.add_argument("--config", default="/content/autopet/submission/postproc_config.json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()

    def _strip(d):
        return {k: (_strip(v) if isinstance(v, dict) else v)
                for k, v in d.items() if not k.startswith("_")}
    base_cfg = _strip(json.load(open(args.config)))

    tags = sorted(os.path.basename(d) for d in glob.glob(os.path.join(args.cache, "*"))
                  if os.path.isdir(d))
    jobs = [(t, args.cache, args.images, args.labels, base_cfg) for t in tags]
    rows = {}
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for i, r in enumerate(ex.map(one_case, jobs), 1):
            if r:
                rows[r[0]] = r[1]
            if i % 20 == 0:
                print(f"  {i}/{len(jobs)}", flush=True)
    json.dump(rows, open(args.out, "w"), indent=1)

    R = list(rows.values())
    grid = [0.0, 0.5, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 20.0, 40.0, 1e9]
    print("\ncases %d  negatives %d (fdg %d / psma %d)  positives %d" % (
        len(R), sum(r["negative"] for r in R),
        sum(r["negative"] and r["tracer"] == "fdg" for r in R),
        sum(r["negative"] and r["tracer"] != "fdg" for r in R),
        sum(not r["negative"] for r in R)))

    shipped = objective(R, 6.0, 6.0)
    best, bt = max(((objective(R, a, b), (a, b)) for a in grid for b in grid),
                   key=lambda x: x[0])
    print("shipped (6.0, 6.0): objective %+.4f   [dice %+.4f, dmm %+.4f]" %
          (shipped, *score(R, 6.0, 6.0)))
    print("best in-sample %s: objective %+.4f   [dice %+.4f, dmm %+.4f]  Delta %+.4f" %
          (bt, best, *score(R, *bt), best - shipped))

    # leave-one-case-out: fit on the other 99, score the held-out case
    loo_new = loo_ship = 0.0
    for i in range(len(R)):
        tr = R[:i] + R[i + 1:]
        _, pair = max(((objective(tr, a, b), (a, b)) for a in grid for b in grid),
                      key=lambda x: x[0])
        loo_new += objective([R[i]], *pair)
        loo_ship += objective([R[i]], 6.0, 6.0)
    print("LOO: shipped %+.4f  per-tracer %+.4f  Delta %+.4f"
          % (loo_ship / len(R), loo_new / len(R), (loo_new - loo_ship) / len(R)))

    print("\ngrid (objective, relative to shipped):")
    print("  t_psma ->  " + "  ".join(f"{b:>6g}" for b in grid))
    for a in grid:
        print(f"  fdg {a:>6g}  " +
              "  ".join(f"{objective(R, a, b) - shipped:+6.3f}" for b in grid))


if __name__ == "__main__":
    main()
