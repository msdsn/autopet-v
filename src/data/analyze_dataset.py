#!/usr/bin/env python3
"""
Dataset analysis for autoPET V (Dataset998_AutoPETV, 1611 FDG+PSMA cases).

Reads the extracted metadata and every label in labelsTr, and reports per tracer:
spacing and shape distribution, number of empty-label (negative) cases, lesion count
and volume distribution (cc3d 26-connectivity), and the fold-0 split composition.
Writes label_stats.json and prints Markdown tables for docs/data_pipeline.md.

    python analyze_dataset.py --labels /content/work/labelsTr \
        --meta /content/drive/MyDrive/autoPET/meta \
        --out /content/drive/MyDrive/autoPET/meta/label_stats.json
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

CONNECTIVITY = 26


def tracer_of(case: str) -> str:
    return "fdg" if case.startswith("fdg_") else "psma"


# ----------------------------------------------------------------- workers --

def _one_label(path_str: str) -> dict:
    import SimpleITK as sitk
    import cc3d

    path = Path(path_str)
    img = sitk.ReadImage(path_str)
    arr = sitk.GetArrayFromImage(img)                 # (z, y, x)
    spacing = tuple(float(x) for x in reversed(img.GetSpacing()))  # -> (z, y, x)
    voxel_cc = float(np.prod(spacing)) / 1000.0       # mm^3 -> cm^3 (cc)

    mask = arr > 0
    n_vox = int(mask.sum())
    rec = {
        "case": path.name[: -len(".nii.gz")],
        "shape": [int(s) for s in arr.shape],
        "spacing": [round(s, 6) for s in spacing],
        "voxel_cc": voxel_cc,
        "n_tumor_voxels": n_vox,
        "tumor_volume_cc": n_vox * voxel_cc,
        "n_lesions": 0,
        "lesion_volumes_cc": [],
    }
    if n_vox:
        lab, n = cc3d.connected_components(mask, connectivity=CONNECTIVITY,
                                           return_N=True)
        counts = np.bincount(lab.reshape(-1))[1:]
        rec["n_lesions"] = int(n)
        rec["lesion_volumes_cc"] = sorted(
            (float(c) * voxel_cc for c in counts), reverse=True)
    return rec


# ------------------------------------------------------------- aggregation --

def pct(vals, qs=(0, 5, 25, 50, 75, 95, 100)):
    if not len(vals):
        return {f"p{q}": None for q in qs}
    a = np.asarray(vals, dtype=float)
    return {f"p{q}": round(float(np.percentile(a, q)), 4) for q in qs}


LESION_BUCKETS = [(0, 0), (1, 1), (2, 3), (4, 10), (11, 30), (31, 10 ** 9)]
BUCKET_NAMES = ["0 (negative)", "1", "2-3", "4-10", "11-30", "31+"]


def bucket_of(n: int) -> str:
    for name, (lo, hi) in zip(BUCKET_NAMES, LESION_BUCKETS):
        if lo <= n <= hi:
            return name
    return BUCKET_NAMES[-1]


def summarize(recs: list[dict]) -> dict:
    out = {}
    for tr in ("fdg", "psma", "all"):
        sel = [r for r in recs if tr == "all" or tracer_of(r["case"]) == tr]
        if not sel:
            continue
        neg = [r for r in sel if r["n_lesions"] == 0]
        pos = [r for r in sel if r["n_lesions"] > 0]
        all_les = [v for r in pos for v in r["lesion_volumes_cc"]]
        spac = Counter(tuple(r["spacing"]) for r in sel)
        shapes = np.array([r["shape"] for r in sel], dtype=float)
        out[tr] = {
            "n_cases": len(sel),
            "n_negative": len(neg),
            "n_positive": len(pos),
            "frac_negative": round(len(neg) / len(sel), 4),
            "spacing_unique": [{"spacing": list(k), "n": v}
                               for k, v in spac.most_common(10)],
            "n_distinct_spacings": len(spac),
            "shape_z": pct(shapes[:, 0]),
            "shape_y": pct(shapes[:, 1]),
            "shape_x": pct(shapes[:, 2]),
            "voxels_per_case_millions": pct(
                [np.prod(r["shape"]) / 1e6 for r in sel]),
            "lesions_per_case": pct([r["n_lesions"] for r in sel]),
            "lesions_per_positive_case": pct([r["n_lesions"] for r in pos]),
            "total_lesions": int(sum(r["n_lesions"] for r in sel)),
            "tumor_volume_cc_per_case": pct([r["tumor_volume_cc"] for r in sel]),
            "tumor_volume_cc_positive_only": pct(
                [r["tumor_volume_cc"] for r in pos]),
            "single_lesion_volume_cc": pct(all_les),
            "mean_single_lesion_volume_cc": round(float(np.mean(all_les)), 4) if all_les else None,
            "lesion_count_buckets": dict(Counter(bucket_of(r["n_lesions"]) for r in sel)),
        }
    return out


def fold_composition(recs: list[dict], splits: list[dict]) -> dict:
    by_case = {r["case"]: r for r in recs}
    out = {}
    for i, sp in enumerate(splits):
        f = {}
        for part in ("train", "val"):
            cases = sp[part]
            d = {"n": len(cases)}
            for tr in ("fdg", "psma"):
                sub = [c for c in cases if tracer_of(c) == tr]
                known = [by_case[c] for c in sub if c in by_case]
                d[tr] = {
                    "n": len(sub),
                    "n_negative": sum(1 for r in known if r["n_lesions"] == 0),
                    "n_positive": sum(1 for r in known if r["n_lesions"] > 0),
                    "n_labels_seen": len(known),
                }
            d["n_negative"] = sum(d[tr]["n_negative"] for tr in ("fdg", "psma"))
            d["n_positive"] = sum(d[tr]["n_positive"] for tr in ("fdg", "psma"))
            d["lesion_count_buckets"] = dict(Counter(
                bucket_of(by_case[c]["n_lesions"]) for c in cases if c in by_case))
            f[part] = d
        f["overlap_train_val"] = len(set(sp["train"]) & set(sp["val"]))
        out[f"fold_{i}"] = f
    return out


def check_fingerprint(recs: list[dict], fp: dict) -> dict:
    """Check the fingerprint's shape list against the labels.

    The fingerprint stores parallel lists with no case ids; nnU-Net builds them in
    sorted-identifier order, which is what we compare against.
    """
    shapes = fp.get("shapes_after_crop", [])
    if len(shapes) != len(recs):
        return {"matched": False, "reason": f"len {len(shapes)} vs {len(recs)}"}
    ordered = sorted(recs, key=lambda r: r["case"])
    hits = sum(1 for r, s in zip(ordered, shapes) if list(r["shape"]) == list(s))
    return {"matched": hits == len(recs), "agree": hits, "of": len(recs),
            "order": "sorted(case_id)"}


# ------------------------------------------------------------------ report --

def md_tables(stats: dict, folds: dict, extra: dict) -> str:
    L = []
    A = L.append
    A("### Cohort overview\n")
    A("| tracer | cases | negatives | % neg | total lesions | median lesions/pos case | median tumour vol (cc, pos only) |")
    A("|---|---:|---:|---:|---:|---:|---:|")
    for tr in ("fdg", "psma", "all"):
        s = stats.get(tr)
        if not s:
            continue
        A(f"| {tr.upper()} | {s['n_cases']} | {s['n_negative']} | "
          f"{100 * s['frac_negative']:.1f}% | {s['total_lesions']} | "
          f"{s['lesions_per_positive_case']['p50']:.0f} | "
          f"{s['tumor_volume_cc_positive_only']['p50']:.2f} |")

    A("\n### Geometry (labels are on the PET grid = the image grid)\n")
    A("| tracer | spacing (z,y,x) mm | n | shape z p5/p50/p95 | shape y/x p50 | Mvoxels p50 | Mvoxels p95 |")
    A("|---|---|---:|---|---|---:|---:|")
    for tr in ("fdg", "psma"):
        s = stats.get(tr)
        if not s:
            continue
        top = s["spacing_unique"][0]
        sp = ", ".join(f"{v:.4f}" for v in top["spacing"])
        A(f"| {tr.upper()} | {sp} | {top['n']}/{s['n_cases']} | "
          f"{s['shape_z']['p5']:.0f}/{s['shape_z']['p50']:.0f}/{s['shape_z']['p95']:.0f} | "
          f"{s['shape_y']['p50']:.0f}/{s['shape_x']['p50']:.0f} | "
          f"{s['voxels_per_case_millions']['p50']:.1f} | "
          f"{s['voxels_per_case_millions']['p95']:.1f} |")
    for tr in ("fdg", "psma"):
        s = stats.get(tr)
        if s:
            A(f"\n- {tr.upper()}: {s['n_distinct_spacings']} distinct spacing(s).")

    A("\n### Lesion burden distribution\n")
    A("| tracer | lesions/case p50 | p75 | p95 | max | single-lesion vol cc p50 | p95 | max |")
    A("|---|---:|---:|---:|---:|---:|---:|---:|")
    for tr in ("fdg", "psma"):
        s = stats.get(tr)
        if not s:
            continue
        lc, lv = s["lesions_per_case"], s["single_lesion_volume_cc"]
        A(f"| {tr.upper()} | {lc['p50']:.0f} | {lc['p75']:.0f} | {lc['p95']:.0f} | "
          f"{lc['p100']:.0f} | {lv['p50']:.3f} | {lv['p95']:.2f} | {lv['p100']:.1f} |")

    A("\n### Lesion-count buckets (all cases)\n")
    keys = BUCKET_NAMES
    A("| tracer | " + " | ".join(keys) + " |")
    A("|---" * (len(keys) + 1) + "|")
    for tr in ("fdg", "psma"):
        s = stats.get(tr)
        if not s:
            continue
        b = s["lesion_count_buckets"]
        A(f"| {tr.upper()} | " + " | ".join(str(b.get(k, 0)) for k in keys) + " |")

    f0 = folds.get("fold_0", {})
    if f0:
        A("\n### Fold 0 composition\n")
        A("| part | total | FDG | FDG neg | PSMA | PSMA neg | negatives total |")
        A("|---|---:|---:|---:|---:|---:|---:|")
        for part in ("train", "val"):
            d = f0[part]
            A(f"| {part} | {d['n']} | {d['fdg']['n']} | {d['fdg']['n_negative']} | "
              f"{d['psma']['n']} | {d['psma']['n_negative']} | {d['n_negative']} |")
        A(f"\n- train/val overlap: {f0['overlap_train_val']} cases.")
        A("\n| fold-0 part | " + " | ".join(keys) + " |")
        A("|---" * (len(keys) + 1) + "|")
        for part in ("train", "val"):
            b = f0[part]["lesion_count_buckets"]
            A(f"| {part} | " + " | ".join(str(b.get(k, 0)) for k in keys) + " |")

    if extra.get("fingerprint_check"):
        c = extra["fingerprint_check"]
        A(f"\n- `dataset_fingerprint.json` shape list matches labels in "
          f"{c.get('order')} order: {c.get('agree')}/{c.get('of')}.")
    return "\n".join(L)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default="/content/work/labelsTr")
    ap.add_argument("--meta", default="/content/drive/MyDrive/autoPET/meta")
    ap.add_argument("--out", default="/content/drive/MyDrive/autoPET/meta/label_stats.json")
    ap.add_argument("--md", default="/content/work/label_stats.md")
    ap.add_argument("--procs", type=int, default=12)
    args = ap.parse_args()

    meta = Path(args.meta)
    files = sorted(str(p) for p in Path(args.labels).glob("*.nii.gz"))
    print(f"{len(files)} labels in {args.labels}")

    with mp.Pool(args.procs) as pool:
        recs = []
        for i, r in enumerate(pool.imap_unordered(_one_label, files, chunksize=8), 1):
            recs.append(r)
            if i % 200 == 0:
                print(f"  {i}/{len(files)}", flush=True)
    recs.sort(key=lambda r: r["case"])

    stats = summarize(recs)
    splits = json.loads((meta / "splits_final.json").read_text())
    folds = fold_composition(recs, splits)
    fp = json.loads((meta / "dataset_fingerprint.json").read_text())
    extra = {
        "fingerprint_check": check_fingerprint(recs, fp),
        "fingerprint_foreground_intensity": fp["foreground_intensity_properties_per_channel"],
        "dataset_json": json.loads((meta / "dataset.json").read_text()),
        "n_labels_analyzed": len(recs),
        "connectivity": CONNECTIVITY,
    }

    payload = {
        "summary": stats,
        "folds": folds,
        "meta": extra,
        "per_case": {r["case"]: {k: r[k] for k in
                                 ("shape", "spacing", "n_lesions",
                                  "tumor_volume_cc", "n_tumor_voxels")}
                     for r in recs},
        "per_case_lesion_volumes": {r["case"]: [round(v, 5) for v in r["lesion_volumes_cc"]]
                                    for r in recs if r["n_lesions"]},
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(payload, indent=1))
    print(f"wrote {args.out}")

    md = md_tables(stats, folds, extra)
    Path(args.md).write_text(md)
    print(f"wrote {args.md}\n")
    print(md)


if __name__ == "__main__":
    main()
