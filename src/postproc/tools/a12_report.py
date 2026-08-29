"""Aggregate the A12 iteration-0 sweep into the row's numbers."""
import json, os, sys
import numpy as np

src = sys.argv[1] if len(sys.argv) > 1 else "/content/work/runs/A12/iter0_sweep.json"
d = json.load(open(src))
names = list(next(iter(d.values()))["rows"].keys())
pos = [t for t, v in d.items() if not v["negative"]]
neg = [t for t, v in d.items() if v["negative"]]

def agg(name):
    dice = [d[t]["rows"][name]["dice"] for t in d]
    f1 = [d[t]["rows"][name]["f1"] for t in pos]
    f1 = [x for x in f1 if x is not None and np.isfinite(x)]
    tp = sum(d[t]["rows"][name]["tp"] for t in pos)
    fp = sum(d[t]["rows"][name]["fp"] for t in pos)
    fn = sum(d[t]["rows"][name]["fn"] for t in pos)
    sw = sum(len(d[t]["rows"][name]["swallowed"]) for t in pos)
    sp = sum((d[t]["rows"][name].get("split") or {}).get("n_split", 0) for t in d)
    rm = sum((d[t]["rows"][name].get("split") or {}).get("removed_ml", 0.0) for t in d)
    return dict(dice=float(np.mean(dice)), f1=float(np.mean(f1)), tp=tp, fp=fp, fn=fn,
                pooled_f1=(2 * tp / (2 * tp + fp + fn) if tp else 0.0),
                swallowed=sw, n_split=sp, removed_ml=rm)

base = agg("base")
base_sw = {t: set(d[t]["rows"]["base"]["swallowed"]) for t in pos}
print("cases %d (%d positive, %d lesion-free)   iteration 0, B10 + shipped post-processing"
      % (len(d), len(pos), len(neg)))
print("%-12s %7s %7s %8s %6s %6s %6s %9s %7s %8s" %
      ("variant", "Dice@0", "F1@0", "pooledF1", "TP", "FP", "FN", "swallowed", "splits", "cut_mL"))
for n in names:
    a = agg(n)
    rec = sum(len(base_sw[t] - set(d[t]["rows"][n]["swallowed"])) for t in pos)
    tag = "" if n == "base" else "  recovered %d/%d" % (rec, base["swallowed"])
    print("%-12s %7.4f %7.4f %8.4f %6d %6d %6d %9d %7d %8.1f%s" %
          (n, a["dice"], a["f1"], a["pooled_f1"], a["tp"], a["fp"], a["fn"],
           a["swallowed"], a["n_split"], a["removed_ml"], tag))

print("\ndeltas vs base (per-case F1@0 mean, Dice@0 mean):")
for n in names:
    if n == "base":
        continue
    a = agg(n)
    print("  %-12s dF1 %+0.4f   dDice %+0.4f   dPooledF1 %+0.4f" %
          (n, a["f1"] - base["f1"], a["dice"] - base["dice"],
           a["pooled_f1"] - base["pooled_f1"]))
