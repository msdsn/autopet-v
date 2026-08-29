"""Fit and leave-one-out evaluate component-pruning rules against the detection metric.

Reads the tables written by dmm_analysis.py and searches one rule family,
`prune <=> f1 < a AND f2 < b AND f3 < c`, over at most three features. The guards from
cleanup.py are reproduced here: a component holding a tumor scribble is never pruned,
and the pass never empties a non-empty prediction. Selection is leave-one-out over
cases with a Dice floor on the fitting fold; the AUC objective is off-policy, since the
scribbles came from the run that produced the tables.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

FEATURES = ("volume_ml", "suv_max", "suv_mean", "suv_p90", "prob_max", "prob_mean",
            "prob_p90", "dist_surface_mm_max", "dist_surface_mm_mean", "z_frac",
            "ct_mean_hu", "n_voxels")

DEFAULT_GRIDS: Dict[str, Sequence[float]] = {
    "volume_ml": (0.0, 0.1, 0.2, 0.3, 0.5, 1.0, 2.0, 5.0, math.inf),
    "n_voxels": (0, 5, 10, 20, 40, 80, 160, math.inf),
    "suv_max": (0.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 15.0, math.inf),
    "suv_mean": (0.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, math.inf),
    "suv_p90": (0.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, math.inf),
    "prob_max": (0.0, 0.7, 0.8, 0.9, 0.95, 0.99, 1.01),
    "prob_mean": (0.0, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 1.01),
    "prob_p90": (0.0, 0.7, 0.8, 0.9, 0.95, 0.99, 1.01),
    "dist_surface_mm_max": (0.0, 10.0, 20.0, 30.0, 50.0, math.inf),
    "dist_surface_mm_mean": (0.0, 10.0, 20.0, 30.0, 50.0, math.inf),
    "z_frac": (0.0, 0.1, 0.2, 0.85, 0.95, 1.01),
    "ct_mean_hu": (-1000.0, -100.0, 0.0, 40.0, 100.0, math.inf),
}


def read_csv(path: str) -> List[Dict[str, Any]]:
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def _f(x) -> float:
    if x is None or x == "":
        return float("nan")
    if x in ("Infinity", "inf"):
        return math.inf
    return float(x)


class Frame:
    """One (case, iteration) prediction: the components and what they matched."""

    __slots__ = ("case", "tracer", "iteration", "n_gt", "gt_voxels", "feat", "masks",
                 "n_vox", "inter", "rank", "n_comp", "protected")

    def __init__(self, case, tracer, iteration, n_gt, gt_voxels, comps):
        self.case = case
        self.tracer = tracer
        self.iteration = int(iteration)
        self.n_gt = int(n_gt)
        self.gt_voxels = int(gt_voxels)
        self.n_comp = len(comps)
        self.feat = {k: np.array([c.get(k, np.nan) for c in comps], dtype=np.float64)
                     for k in FEATURES}
        self.masks = [c["_mask"] for c in comps]
        self.n_vox = np.array([c["n_voxels"] for c in comps], dtype=np.int64)
        self.inter = np.array([c["n_gt_voxels_inside"] for c in comps], dtype=np.int64)
        # G2: a component holding a tumor scribble is untouchable
        self.protected = np.array([c["n_tumor_points_inside"] > 0 for c in comps],
                                  dtype=bool)
        self.rank = sorted(range(len(comps)),
                           key=lambda i: (-comps[i]["suv_max"], -comps[i]["n_voxels"], i))

    def score(self, keep: np.ndarray, min_components_kept: int = 1):
        if self.n_comp and int(keep.sum()) < min_components_kept:
            keep = keep.copy()
            for i in self.rank[:min_components_kept]:
                keep[i] = True
        union = 0
        fp = n_vox = inter = 0
        for i in np.nonzero(keep)[0]:
            m = self.masks[i]
            if m:
                union |= m
            else:
                fp += 1
            n_vox += int(self.n_vox[i])
            inter += int(self.inter[i])
        tp = bin(union).count("1")
        fn = self.n_gt - tp
        f1 = float("nan") if (tp + fn) == 0 else (0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn))
        denom = n_vox + self.gt_voxels
        dice = float("nan") if self.gt_voxels == 0 else (2.0 * inter / denom if denom else 0.0)
        return f1, dice


class CaseSeq:
    """The frames of one case, ordered by iteration."""

    __slots__ = ("case", "tracer", "frames")

    def __init__(self, case, tracer, frames):
        self.case = case
        self.tracer = tracer
        self.frames = sorted(frames, key=lambda f: f.iteration)

    def auc(self, values: Sequence[float]) -> float:
        if len(values) < 2:
            return float(values[0]) if values else float("nan")
        return float(np.trapezoid(values, [f.iteration for f in self.frames]))


def load(out_dir: str, threshold: float, iterations: Optional[Sequence[int]] = None,
         positives_only: bool = True, closing: int = 0) -> List[CaseSeq]:
    comps = read_csv(os.path.join(out_dir, "components.csv"))
    cases = read_csv(os.path.join(out_dir, "cases.csv"))
    by_key: Dict[Tuple[str, int], List[Dict[str, Any]]] = {}
    for r in comps:
        if abs(float(r["threshold"]) - threshold) > 1e-9:
            continue
        if int(float(r.get("closing", 0) or 0)) != closing:
            continue
        it = int(float(r.get("iteration", 0) or 0))
        if iterations is not None and it not in iterations:
            continue
        d = {k: _f(r.get(k)) for k in FEATURES}
        d["n_voxels"] = int(float(r["n_voxels"]))
        d["n_gt_voxels_inside"] = int(float(r["n_gt_voxels_inside"]))
        d["n_tumor_points_inside"] = int(float(r.get("n_tumor_points_inside", 0) or 0))
        d["n_bg_points_inside"] = int(float(r.get("n_bg_points_inside", 0) or 0))
        d["suv_max"] = _f(r["suv_max"])
        m = 0
        for g in json.loads(r["gt_matches"]):
            m |= 1 << int(g)
        d["_mask"] = m
        by_key.setdefault((r["case"], it), []).append(d)

    per_case: Dict[str, List[Frame]] = {}
    tracer: Dict[str, str] = {}
    for r in cases:
        if abs(float(r["threshold"]) - threshold) > 1e-9:
            continue
        if int(float(r.get("closing", 0) or 0)) != closing:
            continue
        it = int(float(r.get("iteration", 0) or 0))
        if iterations is not None and it not in iterations:
            continue
        n_gt = int(float(r["n_gt_lesions"]))
        if positives_only and n_gt == 0:
            continue
        tracer[r["case"]] = r["tracer"]
        per_case.setdefault(r["case"], []).append(
            Frame(r["case"], r["tracer"], it, n_gt, int(float(r["n_gt_voxels"])),
                  by_key.get((r["case"], it), [])))
    want = None if iterations is None else set(int(i) for i in iterations)
    out = []
    dropped = []
    n_full = max((len(fr) for fr in per_case.values()), default=0)
    for c, fr in sorted(per_case.items()):
        have = {f.iteration for f in fr}
        # An AUC over a truncated iteration list is not comparable to one over the full
        # list, so a case the run has not finished is dropped rather than averaged in.
        if (want is not None and not want.issubset(have)) or (want is None and len(fr) < n_full):
            dropped.append(c)
            continue
        out.append(CaseSeq(c, tracer[c], fr))
    if dropped:
        print(f"[load] dropped {len(dropped)} case(s) with an incomplete iteration list",
              flush=True)
    return out


def keep_mask(fr: Frame, feats: Sequence[str], thr: Sequence[float],
              silence_decay: float = 1.0) -> np.ndarray:
    if fr.n_comp == 0:
        return np.zeros(0, dtype=bool)
    prune = np.ones(fr.n_comp, dtype=bool)
    scale = silence_decay ** fr.iteration
    for f, t in zip(feats, thr):
        if t is None or (isinstance(t, float) and math.isinf(t) and t > 0):
            continue
        # A component that survived k rounds without attracting a background scribble
        # is more likely to be real, so shrink the size threshold with the iteration.
        tt = t * scale if (silence_decay != 1.0 and f in ("volume_ml", "n_voxels")) else t
        prune &= fr.feat[f] < tt
    prune &= ~fr.protected
    return ~prune


def grid_points(feats: Sequence[str], grids):
    import itertools
    return list(itertools.product(*[grids[f] for f in feats]))


def evaluate(cases: Sequence[CaseSeq], feats, points, objective="auc",
             min_components_kept=1, silence_decay=1.0):
    """(score, dice) matrices of shape (n_points, n_cases)."""
    sc = np.full((len(points), len(cases)), np.nan)
    dc = np.full((len(points), len(cases)), np.nan)
    for gi, pt in enumerate(points):
        for ci, cs in enumerate(cases):
            f1s, dcs = [], []
            for fr in cs.frames:
                a, b = fr.score(keep_mask(fr, feats, pt, silence_decay),
                                min_components_kept)
                f1s.append(a)
                dcs.append(b)
            if objective == "auc":
                sc[gi, ci] = cs.auc(f1s)
                dc[gi, ci] = cs.auc(dcs)
            else:
                sc[gi, ci] = f1s[0]
                dc[gi, ci] = dcs[0]
    return sc, dc


def _best(sc_m: np.ndarray, dc_m: np.ndarray, floor: float) -> int:
    ok = dc_m >= floor
    cand = np.where(ok)[0] if ok.any() else np.arange(len(sc_m))
    return int(cand[np.lexsort((cand, -dc_m[cand], -sc_m[cand]))[0]])


def loo(sc: np.ndarray, dc: np.ndarray, idx: Sequence[int], slack: float, ctrl: int):
    idx = list(idx)
    picks, s_out, d_out = [], [], []
    for held in idx:
        fold = [j for j in idx if j != held]
        m_sc = np.nanmean(sc[:, fold], axis=1)
        m_dc = np.nanmean(dc[:, fold], axis=1)
        best = _best(m_sc, m_dc, float(m_dc[ctrl]) - slack)
        picks.append(best)
        s_out.append(sc[best, held])
        d_out.append(dc[best, held])
    return float(np.nanmean(s_out)), float(np.nanmean(d_out)), picks


def _fmt(feats, pt):
    return {f: (None if (isinstance(v, float) and math.isinf(v) and v > 0) else v)
            for f, v in zip(feats, pt)}


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in_dir", required=True, help="output dir of dmm_analysis.py")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--features", nargs="+", default=["volume_ml", "suv_max", "prob_mean"])
    ap.add_argument("--objective", choices=["auc", "iter0"], default="auc")
    ap.add_argument("--iterations", type=int, nargs="+", default=None)
    ap.add_argument("--closing", type=int, default=0,
                    help="use the rows produced with this closing radius (component "
                         "merging); 0 = the plain mask")
    ap.add_argument("--per_tracer", action="store_true")
    ap.add_argument("--dice_slack", type=float, default=0.01,
                    help="how far the fitting fold's mean Dice (AUC-Dice with "
                         "--objective auc) may fall below the unpruned control")
    ap.add_argument("--min_components_kept", type=int, default=1)
    ap.add_argument("--silence_decay", type=float, default=1.0,
                    help="multiply the size threshold by this per iteration "
                         "(<1 = prune less as a component survives more rounds)")
    ap.add_argument("--grid", action="append", default=None, metavar="FEATURE=v1,v2,...",
                    help="override the candidate values of one feature; `none` means "
                         "the criterion is disabled. Give a single value to evaluate a "
                         "pre-specified rule instead of fitting one")
    ap.add_argument("--out_json", default=None)
    args = ap.parse_args(argv)

    grids = dict(DEFAULT_GRIDS)
    for spec in (args.grid or []):
        name, _, vals = spec.partition("=")
        if name not in grids:
            raise SystemExit(f"no such feature {name!r}")
        grids[name] = tuple(math.inf if v.strip().lower() in ("none", "inf") else float(v)
                            for v in vals.split(","))

    cases = load(args.in_dir, args.threshold, args.iterations, closing=args.closing)
    feats = list(args.features)
    for f in feats:
        if f not in grids:
            raise SystemExit(f"no grid for feature {f!r}")
    pts = grid_points(feats, grids)
    sc, dc = evaluate(cases, feats, pts, args.objective, args.min_components_kept,
                      args.silence_decay)
    # Append the "prune nothing" control rather than looking it up: a pre-specified
    # rule (--grid with one value per feature) has no such point in its grid.
    ctrl_pt = tuple(0.0 for _ in feats)
    sc0, dc0 = evaluate(cases, feats, [ctrl_pt], args.objective,
                        args.min_components_kept, args.silence_decay)
    sc = np.vstack([sc, sc0])
    dc = np.vstack([dc, dc0])
    pts = list(pts) + [ctrl_pt]
    ctrl = len(pts) - 1

    groups = {"all": list(range(len(cases)))}
    if args.per_tracer:
        for tr in sorted({c.tracer for c in cases}):
            groups[tr] = [i for i, c in enumerate(cases) if c.tracer == tr]

    unit = "AUC-F1" if args.objective == "auc" else "F1@0"
    result: Dict[str, Any] = {"features": feats, "threshold": args.threshold,
                              "closing": args.closing,
                              "objective": args.objective, "n_cases": len(cases),
                              "silence_decay": args.silence_decay, "groups": {}}
    for name, idx in groups.items():
        base_s = float(np.nanmean(sc[ctrl, idx]))
        base_d = float(np.nanmean(dc[ctrl, idx]))
        loo_s, loo_d, picks = loo(sc, dc, idx, args.dice_slack, ctrl)
        m_sc = np.nanmean(sc[:, idx], axis=1)
        m_dc = np.nanmean(dc[:, idx], axis=1)
        full = _best(m_sc, m_dc, base_d - args.dice_slack)
        uniq = sorted({tuple(pts[p]) for p in picks})
        g = {
            "n": len(idx),
            "control_score": base_s, "control_dice": base_d,
            "loo_score": loo_s, "loo_dice": loo_d,
            "insample_score": float(m_sc[full]), "insample_dice": float(m_dc[full]),
            "chosen": _fmt(feats, pts[full]),
            "loo_n_distinct_rules": len(uniq),
            "loo_rules": [_fmt(feats, r) for r in uniq[:8]],
        }
        result["groups"][name] = g
        print(f"[{name:6s} n={len(idx):3d}] control {unit}={base_s:.4f} dice={base_d:.4f} "
              f"| LOO {unit}={loo_s:.4f} ({loo_s - base_s:+.4f}) dice={loo_d:.4f} "
              f"({loo_d - base_d:+.4f}) | in-sample {m_sc[full]:.4f} "
              f"| rule={g['chosen']} | {len(uniq)} distinct LOO rules", flush=True)

    if args.out_json:
        with open(args.out_json, "w") as fh:
            json.dump(result, fh, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
