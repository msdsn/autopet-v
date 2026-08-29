"""Fit and score lesion-free-gate rules on the iteration-0 tables of negative_analysis.py.

The objective is the change in mean AUC-Dice, not accuracy, because the two errors are
asymmetric: rescuing a non-empty lesion-free case is worth +5.0, while emptying a
positive costs only iteration 0 (weight 0.5) since iteration 1 delivers a tumor
scribble and turns the gate off for good. So
`dAUC_mean = (5.0 * n_neg_rescued - 0.5 * sum(Dice@0 of emptied positives)) / n_cases`.
Every rule is reported both in-sample and leave-one-out cross-validated.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

W0 = 0.5          # trapezoid weight of iteration 0
GAIN = 5.0        # AUC-Dice of a rescued lesion-free case


# ------------------------------------------------------------------ loading
def read_csv(path: str) -> List[Dict[str, Any]]:
    with open(path) as fh:
        rows = list(csv.DictReader(fh))
    out = []
    for r in rows:
        d: Dict[str, Any] = {}
        for k, v in r.items():
            if v == "" or v is None:
                d[k] = None
                continue
            try:
                d[k] = int(v)
            except ValueError:
                try:
                    d[k] = float(v)
                except ValueError:
                    d[k] = v
        out.append(d)
    return out


def build_table(out_dir: str) -> List[Dict[str, Any]]:
    """One dict per case with every feature a gate may look at, plus Dice@0."""
    cases = read_csv(os.path.join(out_dir, "cases.csv"))
    comps = read_csv(os.path.join(out_dir, "components.csv"))
    by_case: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for c in comps:
        by_case[c["case"]].append(c)

    table = []
    for c in cases:
        cs = sorted(by_case.get(c["case"], []), key=lambda x: -x["volume_ml"])
        vox_ml = float(c["voxel_ml"])
        n_pred = int(c["pred_voxels"])
        n_gt = int(round(float(c["gt_volume_ml"]) / vox_ml)) if vox_ml else 0
        inter = int(round(float(c.get("tp_volume_ml") or 0.0) / vox_ml)) if vox_ml else 0
        # Approximate Dice@0 from the stored voxel counts (tp_volume_ml counts whole
        # TP components). The cost model only needs ~1e-2; exact_dice0_from_bundles
        # replaces this with the exact value when the bundles are available.
        if n_pred == 0 and n_gt == 0:
            dice0 = 1.0
        elif n_pred == 0 or n_gt == 0:
            dice0 = 0.0
        else:
            dice0 = 2.0 * inter / (n_pred + n_gt)
        d = dict(c)
        d["dice0_est"] = dice0
        d["n_comp"] = len(cs)
        for key, default in (("volume_ml", 0.0), ("suv_max", 0.0), ("suv_mean", 0.0),
                             ("prob_max", 0.0), ("prob_mean", 0.0), ("z_frac", 0.0),
                             ("shell_suv_max", 0.0), ("ct_mean_hu", 0.0)):
            d["largest_" + key] = cs[0][key] if cs else default
            d["hottest_" + key] = (max(cs, key=lambda x: x["suv_max"])[key] if cs else default)
        d["max_comp_suv_max"] = max((x["suv_max"] for x in cs), default=0.0)
        d["max_comp_prob_max"] = max((x["prob_max"] for x in cs), default=0.0)
        d["mean_prob_in_mask"] = (float(np.average([x["prob_mean"] for x in cs],
                                                   weights=[x["n_voxels"] for x in cs]))
                                  if cs else 0.0)
        # soft volume: volume weighted by confidence, so a speck the model is unsure
        # about counts for almost nothing.
        d["soft_volume_ml"] = sum(x["prob_mean"] * x["volume_ml"] for x in cs)
        table.append(d)
    return table


def exact_dice0_from_bundles(table, out_dir: str) -> None:
    """Replace the estimated Dice@0 by the exact one, using the replay bundles."""
    bdir = os.path.join(out_dir, "bundles")
    for r in table:
        p = os.path.join(bdir, str(r["case"]).replace("/", "_") + ".npz")
        vox_ml = float(r["voxel_ml"])
        n_gt = int(round(float(r["gt_volume_ml"]) / vox_ml)) if vox_ml else 0
        n_pred = int(r["pred_voxels"])
        if n_pred == 0:
            r["dice0"] = 1.0 if n_gt == 0 else 0.0
            continue
        if not os.path.isfile(p):
            r["dice0"] = r["dice0_est"]
            continue
        with np.load(p) as z:
            gtl = z["gt_label"] if "gt_label" in z else None
        inter = int((gtl > 0).sum()) if gtl is not None else 0
        r["dice0"] = (2.0 * inter / (n_pred + n_gt)) if (n_pred + n_gt) else 1.0


# ------------------------------------------------------------------ scoring
def delta_auc(table: Sequence[Dict[str, Any]], fires: Sequence[bool],
              dice_key: str = "dice0") -> Dict[str, float]:
    """Expected change in mean AUC-Dice (and the breakdown) for a firing pattern."""
    gain = cost = 0.0
    neg_fire = neg_rescued = pos_fire = pos_fire_real = 0
    for r, f in zip(table, fires):
        if not f:
            continue
        if r["empty_gt"]:
            neg_fire += 1
            if r["pred_voxels"] > 0:
                gain += GAIN
                neg_rescued += 1
        else:
            pos_fire += 1
            # Firing on a positive that already predicted empty is a no-op; only a
            # non-empty prediction is a real intervention.
            if r["pred_voxels"] > 0:
                pos_fire_real += 1
            cost += W0 * float(r.get(dice_key, r["dice0_est"]))
    n = len(table)
    return {
        "neg_fired": neg_fire, "neg_rescued": neg_rescued, "pos_emptied": pos_fire,
        "pos_emptied_nonempty": pos_fire_real,
        "gain": gain, "cost": cost, "delta_total": gain - cost,
        "delta_mean_auc": (gain - cost) / n if n else 0.0,
    }


def margin_table(table, feature="total_volume_ml", grid=None) -> None:
    """Print what a chosen threshold does, next to the gap between the two classes.

    Leave-one-out punishes a threshold sitting exactly on the largest observed negative,
    so the threshold wants to go inside the gap; that needs the gap printed rather than
    a fitted number.
    """
    vals = sorted(float(r[feature]) for r in table)
    if grid is None:
        grid = [0.5, 0.75, 1.0, 1.05, 1.25, 1.5, 2.0, 3.0, 5.0, 8.0, 12.0, 17.0, 25.0, 50.0]
    print(f"\n--- what a chosen threshold on {feature} does "
          f"(neg rescued / positives actually emptied / dAUC) ---")
    print("{:>8s} {:>12s} {:>14s} {:>12s} {:>12s}".format(
        "V (mL)", "neg/36", "pos!=0 emptied", "Dice@0 lost", "dAUC mean"))
    for v in grid:
        fires = [float(r[feature]) < v for r in table]
        d = delta_auc(table, fires)
        print("{:>8.2f} {:>12d} {:>14d} {:>12.3f} {:>+12.4f}".format(
            v, d["neg_rescued"], d["pos_emptied_nonempty"], d["cost"] / W0,
            d["delta_mean_auc"]))
    neg_max = max((float(r[feature]) for r in table
                   if r["empty_gt"] and r["pred_voxels"] > 0), default=0.0)
    pos_above = sorted(float(r[feature]) for r in table
                       if not r["empty_gt"] and float(r[feature]) > neg_max)
    print(f"largest {feature} over the non-empty negatives : {neg_max:.4f}")
    print(f"the four smallest positives above it            : "
          f"{[round(x, 3) for x in pos_above[:4]]}")


# ------------------------------------------------------------------ rule shapes
class Rule:
    """A gate rule: a name, a fitted-parameter dict, and a per-case predicate."""

    def __init__(self, name: str, fit: Callable, apply: Callable, grid: Callable):
        self.name, self._fit, self._apply, self._grid = name, fit, apply, grid

    def candidates(self, table):
        return self._grid(table)

    def fires(self, table, params):
        return [bool(self._apply(r, params)) for r in table]

    def fit(self, table):
        best, best_d = None, -1e18
        for p in self.candidates(table):
            d = delta_auc(table, self.fires(table, p))["delta_total"]
            if d > best_d:
                best, best_d = p, d
        return best


def _quantile_grid(vals, k=40):
    v = sorted(set(float(x) for x in vals if x is not None and not math.isnan(float(x))))
    if not v:
        return [0.0]
    if len(v) <= k:
        cand = v
    else:
        cand = list(np.quantile(v, np.linspace(0, 1, k)))
    # thresholds sit between observed values, plus one above the maximum
    out = sorted(set([c + 1e-9 for c in cand] + [max(v) * 1.001 + 1e-6]))
    return out


def rule_threshold(name: str, feature: str, direction: str = "lt") -> Rule:
    def apply(r, p):
        x = r.get(feature)
        if x is None or (isinstance(x, float) and math.isnan(x)):
            return False
        return (x < p[feature]) if direction == "lt" else (x > p[feature])
    return Rule(name,
                None,
                apply,
                lambda t: [{feature: v} for v in _quantile_grid([r.get(feature) for r in t])])


def rule_pair(name: str, f1: str, f2: str, n1=18, n2=18) -> Rule:
    def apply(r, p):
        a, b = r.get(f1), r.get(f2)
        if a is None or b is None:
            return False
        return a < p[f1] and b < p[f2]

    def grid(t):
        g1 = _quantile_grid([r.get(f1) for r in t], n1)
        g2 = _quantile_grid([r.get(f2) for r in t], n2)
        return [{f1: a, f2: b} for a in g1 for b in g2]
    return Rule(name, None, apply, grid)


def rule_tracer_threshold(name: str, feature: str, n=25) -> Rule:
    """A per-tracer threshold on one feature (FDG and PSMA fitted independently)."""
    def apply(r, p):
        x = r.get(feature)
        if x is None:
            return False
        return x < p.get(r["tracer"], p.get("unknown", 0.0))

    def grid(t):
        out = []
        gs = {tr: _quantile_grid([r.get(feature) for r in t if r["tracer"] == tr], n)
              for tr in sorted({r["tracer"] for r in t})}
        keys = sorted(gs)
        def rec(i, acc):
            if i == len(keys):
                out.append(dict(acc))
                return
            for v in gs[keys[i]]:
                acc[keys[i]] = v
                rec(i + 1, acc)
        rec(0, {})
        return out
    return Rule(name, None, apply, grid)


# ------------------------------------------------------------------ LOO
def loo_score(rule: Rule, table) -> Dict[str, float]:
    """Leave-one-out: fit the threshold on 99 cases, apply it to the held-out one."""
    fires = []
    for i in range(len(table)):
        train = table[:i] + table[i + 1:]
        p = rule.fit(train)
        fires.append(bool(rule._apply(table[i], p)) if p is not None else False)
    d = delta_auc(table, fires)
    d["fires"] = fires
    return d


# ------------------------------------------------------------------ learned models
FEATURES = ["total_volume_ml", "largest_volume_ml", "n_components", "suv_max_in_mask",
            "largest_suv_max", "largest_suv_mean", "prob_max_in_mask", "largest_prob_max",
            "largest_prob_mean", "mean_prob_in_mask", "soft_volume_ml", "prob_max_global",
            "largest_z_frac", "largest_shell_suv_max", "max_comp_suv_max"]


def feature_matrix(table, features=FEATURES):
    X = np.zeros((len(table), len(features)), dtype=np.float64)
    for i, r in enumerate(table):
        for j, f in enumerate(features):
            v = r.get(f)
            X[i, j] = 0.0 if v is None or (isinstance(v, float) and math.isnan(v)) else float(v)
    # log1p the heavy-tailed volume/SUV features
    for j, f in enumerate(features):
        if "volume" in f or "suv" in f:
            X[:, j] = np.log1p(np.maximum(X[:, j], 0.0))
    return X


def learned_rules(table, seed=0):
    """Logistic regression and a depth-limited tree, both LOO-cross-validated.

    The target is "lesion-free", but the decision threshold on the predicted probability
    is chosen to maximise the AUC objective on the training fold, which is what exposes
    the asymmetric cost to the model.
    """
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.tree import DecisionTreeClassifier, export_text
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import make_pipeline
    except Exception as e:                                    # pragma: no cover
        return {"error": f"sklearn unavailable: {e}"}

    X = feature_matrix(table)
    y = np.array([r["empty_gt"] for r in table], dtype=int)
    # sample weight = |AUC at stake|: 5.0 for a rescuable negative, 0.5*Dice@0 for a
    # positive, ~0 for a negative that is already empty.
    w = np.array([GAIN if (r["empty_gt"] and r["pred_voxels"] > 0)
                  else (0.05 if r["empty_gt"] else max(W0 * float(r.get("dice0", 0.0)), 1e-3))
                  for r in table])

    out = {}
    for name, mk in (("logreg", lambda: make_pipeline(StandardScaler(),
                                                      LogisticRegression(max_iter=2000, C=1.0))),
                     ("tree3", lambda: DecisionTreeClassifier(max_depth=3, min_samples_leaf=5,
                                                              random_state=seed)),
                     ("tree2", lambda: DecisionTreeClassifier(max_depth=2, min_samples_leaf=5,
                                                              random_state=seed))):
        fires = []
        for i in range(len(table)):
            tr = np.ones(len(table), dtype=bool)
            tr[i] = False
            m = mk()
            m.fit(X[tr], y[tr], **({"logisticregression__sample_weight": w[tr]}
                                   if name == "logreg" else {"sample_weight": w[tr]}))
            ptr = m.predict_proba(X[tr])[:, 1]
            # pick the threshold that maximises the objective on the training fold
            best_t, best_d = 1.1, -1e18
            for t in np.unique(np.round(ptr, 4)):
                f = [bool(v >= t) for v in ptr]
                d = delta_auc([table[k] for k in np.where(tr)[0]], f)["delta_total"]
                if d > best_d:
                    best_t, best_d = float(t), d
            fires.append(bool(m.predict_proba(X[i:i + 1])[0, 1] >= best_t))
        out[name] = delta_auc(table, fires)
        out[name]["fires"] = fires
        full = mk()
        full.fit(X, y, **({"logisticregression__sample_weight": w}
                          if name == "logreg" else {"sample_weight": w}))
        if name.startswith("tree"):
            out[name]["model"] = export_text(full, feature_names=FEATURES)
        else:
            coefs = full[-1].coef_[0]
            out[name]["model"] = json.dumps(
                {f: round(float(c), 3) for f, c in sorted(zip(FEATURES, coefs),
                                                          key=lambda kv: -abs(kv[1]))})
    return out


# ------------------------------------------------------------------ report
def default_rules() -> List[Rule]:
    return [
        rule_threshold("(a) total_volume_ml < V", "total_volume_ml"),
        rule_threshold("(b) prob_max_in_mask < P", "prob_max_in_mask"),
        rule_threshold("(b') mean_prob_in_mask < P", "mean_prob_in_mask"),
        rule_threshold("(c) suv_max_in_mask < S", "suv_max_in_mask"),
        rule_threshold("(c') largest_suv_max < S", "largest_suv_max"),
        rule_threshold("soft_volume_ml < Q", "soft_volume_ml"),
        rule_tracer_threshold("(c'') largest_suv_max < S[tracer]", "largest_suv_max"),
        rule_tracer_threshold("total_volume_ml < V[tracer]", "total_volume_ml"),
        rule_pair("(d1) volume<V & largest_prob_mean<P", "total_volume_ml", "largest_prob_mean"),
        rule_pair("(d2) volume<V & suv_max<S", "total_volume_ml", "suv_max_in_mask"),
        rule_pair("(d3) volume<V & mean_prob<P", "total_volume_ml", "mean_prob_in_mask"),
        rule_pair("(d4) soft_volume<Q & suv_max<S", "soft_volume_ml", "suv_max_in_mask"),
        rule_pair("(d5) largest_volume<V & mean_prob<P", "largest_volume_ml", "mean_prob_in_mask"),
    ]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out_dir", required=True, help="a negative_analysis.py output dir")
    ap.add_argument("--json_out", default=None)
    ap.add_argument("--no_learned", action="store_true")
    ap.add_argument("--exact_dice", action="store_true", default=True)
    args = ap.parse_args(argv)

    table = build_table(args.out_dir)
    if args.exact_dice:
        exact_dice0_from_bundles(table, args.out_dir)
    for r in table:
        r.setdefault("dice0", r["dice0_est"])

    neg = [r for r in table if r["empty_gt"]]
    pos = [r for r in table if not r["empty_gt"]]
    nonempty_neg = [r for r in neg if r["pred_voxels"] > 0]
    print(f"cases={len(table)}  negatives={len(neg)} (non-empty prediction: {len(nonempty_neg)})"
          f"  positives={len(pos)}")
    print(f"headroom = {GAIN*len(nonempty_neg)/len(table):.3f} mean AUC-Dice\n")

    print("--- the non-empty negatives (what the gate has to catch) ---")
    hdr = ("case", "tracer", "vol_mL", "ncomp", "SUVmax", "pmax", "pmean", "zfrac", "shellSUV")
    print("{:<46s} {:>5s} {:>8s} {:>5s} {:>7s} {:>6s} {:>6s} {:>6s} {:>8s}".format(*hdr))
    for r in sorted(nonempty_neg, key=lambda r: -r["total_volume_ml"]):
        print("{:<46s} {:>5s} {:>8.3f} {:>5d} {:>7.2f} {:>6.3f} {:>6.3f} {:>6.2f} {:>8.1f}".format(
            str(r["case"])[:46], str(r["tracer"]), r["total_volume_ml"], int(r["n_components"]),
            r["suv_max_in_mask"], r["prob_max_in_mask"], r["mean_prob_in_mask"],
            r["largest_z_frac"], r["largest_shell_suv_max"]))

    print("\n--- positives: distribution of the same features ---")
    for f in ("total_volume_ml", "suv_max_in_mask", "prob_max_in_mask", "mean_prob_in_mask",
              "n_components", "soft_volume_ml", "largest_volume_ml"):
        a = np.array([float(r.get(f) or 0.0) for r in pos])
        b = np.array([float(r.get(f) or 0.0) for r in nonempty_neg])
        q = lambda v: " ".join(f"{np.quantile(v, x):.3f}" for x in (0, .05, .25, .5, .75, 1))
        print(f"{f:>22s}  pos[min p5 p25 p50 p75 max]= {q(a)}")
        print(f"{'':>22s}  neg[min p5 p25 p50 p75 max]= {q(b)}")

    n_pos_dice0 = sum(1 for r in pos if r["dice0"] == 0.0)
    n_pos_empty = sum(1 for r in pos if r["pred_voxels"] == 0)
    print(f"\npositives whose iteration-0 prediction is already empty: {n_pos_empty} "
          f"(emptying them is a no-op); positives with Dice@0 == 0: {n_pos_dice0}")

    margin_table(table)

    results = []
    for rule in default_rules():
        p = rule.fit(table)
        ins = delta_auc(table, rule.fires(table, p))
        cv = loo_score(rule, table)
        results.append({"rule": rule.name, "params": p, "in_sample": ins, "loo": cv})

    if not args.no_learned:
        for name, d in (learned_rules(table) or {}).items():
            if name == "error":
                print("learned rules skipped:", d)
                continue
            results.append({"rule": f"(e) {name}", "params": d.pop("model", None),
                            "in_sample": None, "loo": d})

    print("\n{:<44s} {:>28s} {:>28s}".format("rule", "in-sample", "LOO cross-validated"))
    print("{:<44s} {:>28s} {:>28s}".format("", "neg/36 pos/64  dAUC", "neg/36 pos/64  dAUC"))
    for r in sorted(results, key=lambda x: -(x["loo"]["delta_mean_auc"])):
        ins = r["in_sample"]
        s_ins = ("{:>4d} {:>4d}  {:+.4f}".format(ins["neg_rescued"], ins["pos_emptied_nonempty"],
                                                 ins["delta_mean_auc"]) if ins else "  --")
        cv = r["loo"]
        s_cv = "{:>4d} {:>4d}  {:+.4f}".format(cv["neg_rescued"], cv["pos_emptied_nonempty"],
                                               cv["delta_mean_auc"])
        print("{:<44s} {:>28s} {:>28s}".format(r["rule"][:44], s_ins, s_cv))
        if r["params"] is not None and not isinstance(r["params"], str):
            print("{:<44s}   params: {}".format("", json.dumps(
                {k: (round(v, 4) if isinstance(v, float) else v) for k, v in r["params"].items()})))

    for r in results:
        if isinstance(r["params"], str):
            print(f"\n--- {r['rule']} model ---\n{r['params']}")

    if args.json_out:
        with open(args.json_out, "w") as fh:
            json.dump({"n_cases": len(table), "results": [
                {k: v for k, v in r.items() if k != "loo"} | {
                    "loo": {k: v for k, v in r["loo"].items() if k != "fires"}}
                for r in results]}, fh, indent=1, default=str)
        print("\nwrote", args.json_out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
