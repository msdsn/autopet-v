#!/usr/bin/env python3
"""Build RESULTS.md from the per-row run.json files under results/.

The script runs on the machine that holds the evaluation output tree (a data root with
``results/``, ``runs/`` and ``ckpt/`` next to each other). What is checked into this
repository is the curated output of that run: ``results/RESULTS.md`` together with the
per-row folders it was generated from.

Usage (from the directory that holds the results tree):

    python3 results_index.py                       # rewrite ./RESULTS.md
    python3 results_index.py --root /path/to/results-root
    python3 results_index.py --no-hash             # skip the checkpoint sha256 pass

Reads   <root>/results/<ROW>/run.json      (one folder per named ablation row; falls back to
                                            summary.json + label.txt when a run was launched
                                            without the wrapper that writes run.json)
        <root>/results/<ROW>/note.txt      (optional caveat, rendered as a table footnote)
        <root>/results/<ROW>/subset.txt    (optional: marks a partial-case-list row and names
                                            the control it is paired against)
        <root>/results/curves_rows.txt     (optional: which rows get per-iteration curves)
        <root>/results/subset_notes.md     (optional prose under the subset table)
        <root>/results/runs_disposition.json (optional {run dir: reason} for the audit sections)
        <root>/ckpt/Dataset998_AutoPETV/*  (model folders, read-only)
        <root>/runs/*                      (to report runs still in flight, and finished runs
                                            that have not been curated into results/)
Writes  <root>/RESULTS.md
        <root>/results/model_sha256.json   (hash cache, so re-runs are cheap)

Subset rows are scored against their control on the control's own metric_scores.json restricted
to the subset's case list, so the reported delta is paired. The control's full-set numbers are
recomputed by the same code and checked against its reported aggregate; any mismatch is printed
as a warning in the generated file.

Nothing is moved or deleted; every path is opened read-only except the two outputs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time

ROW_ORDER = ["A0", "A1", "A2", "A3", "A4", "A5", "A5a", "A5b", "A5c",
             "A6", "A7", "A8", "A9", "A9b", "A10",
             "B0", "B0b", "B0b_sub30", "B3", "B3nostate_sub39", "B3g", "B3g25",
             "B6", "B6g", "B9",
             "B10", "B10g", "B10gbest", "B10gbest_sub30", "B10g9_ship",
             "B10g9nostate_sub39", "B11", "B12", "B12g",
             "E1_sub30", "X1_sub30", "X1B10_sub30"]

# rows whose per-iteration curves are worth printing in full; one row id per line in
# results/curves_rows.txt (blank lines and #-comments ignored). Absent file -> every row.
CURVES_FILE = "curves_rows.txt"

# a row folder containing subset.txt is a partial-case-list row: it is kept out of the main
# table (its numbers are not comparable to the 100-case rows) and shown in its own section
# next to the same-cases numbers of the control row named on subset.txt's first line.
SUBSET_FILE = "subset.txt"
SUBSET_NOTES = "subset_notes.md"        # free prose appended under the subset table
DISPOSITION = "runs_disposition.json"   # {run dir: why it will never become a row}

# run dirs that are work areas rather than named rows
NON_ROW_DIRS = {"_worksync", "_prehistory", "cache", "B_results", "dmm_analysis",
                "failure_analysis", "failure_analysis (1)", "negative_analysis"}


def row_sort_key(name):
    return (ROW_ORDER.index(name) if name in ROW_ORDER else len(ROW_ORDER), name)


def fmt(x, nd=3):
    if x is None:
        return "-"
    if isinstance(x, float) and math.isnan(x):
        return "n/a"
    return f"{x:.{nd}f}"


def get(d, *path, default=None):
    cur = d
    for p in path:
        if not isinstance(cur, dict) or p not in cur:
            return default
        cur = cur[p]
    return cur


def load_rows(results_dir):
    """One dict per row folder.

    `run.json` is the preferred source: it carries the label, the finish time and the
    `results` block. Some rows were launched by hand rather than through the run wrapper and
    therefore never got a `run.json`; for those we fall back to `summary.json`, which the loop
    itself writes and which holds the same aggregation under top-level keys. The fallback is
    recorded in `source` and surfaced in the generated table, so the two are never confused.
    An optional `label.txt` in the row folder supplies the one-line description that only
    `run.json` would otherwise carry.
    """
    rows = []
    for name in sorted(os.listdir(results_dir), key=row_sort_key):
        folder = os.path.join(results_dir, name)
        if not os.path.isdir(folder):
            continue
        rj = os.path.join(folder, "run.json")
        sj = os.path.join(folder, "summary.json")
        if os.path.isfile(rj):
            with open(rj) as fh:
                run = json.load(fh)
            res = run.get("results", {})
            source = "run.json"
            finished = run.get("finished_utc", "")
        elif os.path.isfile(sj):
            with open(sj) as fh:
                run = json.load(fh)
            res = run                        # summary.json is flat: the metrics are top level
            source = "summary.json"
            finished = time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                     time.gmtime(os.stat(sj).st_mtime)) + " (file mtime)"
        else:
            continue
        label = run.get("label", "")
        lf = os.path.join(folder, "label.txt")
        if not label and os.path.isfile(lf):
            label = open(lf).read().strip()
        nf = os.path.join(folder, "note.txt")
        note = open(nf).read().strip() if os.path.isfile(nf) else ""
        subset_ctl, subset_desc = "", ""
        pf = os.path.join(folder, SUBSET_FILE)
        if os.path.isfile(pf):
            lines = [x.strip() for x in open(pf) if x.strip() and not x.startswith("#")]
            if lines:
                subset_ctl = lines[0].split()[0]
                subset_desc = " ".join(lines[1:]).strip()
        dice_it = res.get("mean_dice_per_iteration") or []
        rows.append({
            "row": name,
            "label": label,
            "predictor": run.get("predictor", ""),
            "n": run.get("n_cases"),
            "auc_dice": res.get("mean_auc_dice"),
            "auc_dmm": res.get("mean_auc_dmm"),
            "score": res.get("final_score_50_50"),
            "dice0": dice_it[0] if len(dice_it) > 0 else None,
            "dice5": dice_it[-1] if dice_it else None,
            "neg_dice": get(res, "by_lesion_status", "lesion_absent", "auc_dice"),
            "n_neg": get(res, "by_lesion_status", "lesion_absent", "n"),
            "pos_dice": get(res, "by_lesion_status", "lesion_present", "auc_dice"),
            "pos_dmm": get(res, "by_lesion_status", "lesion_present", "auc_dmm"),
            "n_pos": get(res, "by_lesion_status", "lesion_present", "n"),
            "fdg_dice": get(res, "by_tracer", "fdg", "all", "auc_dice"),
            "fdg_dmm": get(res, "by_tracer", "fdg", "all", "auc_dmm"),
            "psma_dice": get(res, "by_tracer", "psma", "all", "auc_dice"),
            "psma_dmm": get(res, "by_tracer", "psma", "all", "auc_dmm"),
            "finished": finished,
            "source": source,
            "note": note,
            "subset_control": subset_ctl,
            "subset_desc": subset_desc,
            "dir": folder,
            "dice_iter": dice_it,
            "dmm_iter": res.get("mean_dmm_per_iteration") or [],
        })
    return rows


def trapezoid(vals):
    """Same as np.trapezoid over unit spacing: ends carry half weight."""
    return sum((a + b) / 2.0 for a, b in zip(vals, vals[1:]))


def score_cases(scores, cases=None):
    """Aggregate metric_scores.json over a case list, exactly as the harness does.

    AUC per case is the trapezoid over its six iterations; AUC-Dice is the plain mean and
    AUC-DMM the nanmean (DMM is NaN on lesion-free cases, which the organizers drop).
    """
    keys = sorted(scores) if cases is None else sorted(cases)
    dices, dmms, it0, it5 = [], [], [], []
    for k in keys:
        seq = sorted(scores[k], key=lambda r: r["iteration"])
        d = [r["dice"] for r in seq]
        m = [r["dmm"] for r in seq]
        dices.append(trapezoid(d))
        if not any(math.isnan(x) for x in m):
            dmms.append(trapezoid(m))
        it0.append(d[0])
        it5.append(d[-1])
    n = len(keys)
    mean = lambda xs: (sum(xs) / len(xs)) if xs else float("nan")
    ad, am = mean(dices), mean(dmms)
    return {"n": n, "auc_dice": ad, "auc_dmm": am,
            "score": (ad + am) / 2.0 if not math.isnan(am) else float("nan"),
            "dice0": mean(it0), "dice5": mean(it5)}


def sha256_file(path, chunk=8 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            b = fh.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def checkpoint_meta(path):
    """epoch / trainer / best-EMA out of an nnU-Net checkpoint, without pulling the weights."""
    try:
        import torch
    except Exception as exc:                                    # torch not importable
        return {"error": f"torch unavailable ({exc})"}
    for kwargs in ({"mmap": True}, {}):
        try:
            ck = torch.load(path, map_location="cpu", weights_only=False, **kwargs)
            return {"epoch": ck.get("current_epoch"),
                    "trainer": ck.get("trainer_name"),
                    "best_ema": ck.get("_best_ema")}
        except Exception as exc:
            last = exc
    return {"error": str(last)}


def scan_models(ckpt_root, hash_cache, do_hash=True):
    models = []
    if not os.path.isdir(ckpt_root):
        return models
    for name in sorted(os.listdir(ckpt_root)):
        folder = os.path.join(ckpt_root, name)
        fold0 = os.path.join(folder, "fold_0")
        if not os.path.isdir(fold0):
            continue
        entry = {"folder": name, "files": {}}
        for fn in ("checkpoint_final.pth", "checkpoint_best.pth", "checkpoint_latest.pth"):
            p = os.path.join(fold0, fn)
            if os.path.isfile(p):
                st = os.stat(p)
                entry["files"][fn] = {"size": st.st_size, "mtime": st.st_mtime}
        prog = os.path.join(fold0, "progress.txt")
        entry["progress_lines"] = sum(1 for _ in open(prog)) if os.path.isfile(prog) else None
        final = os.path.join(fold0, "checkpoint_final.pth")
        if os.path.isfile(final):
            entry["meta"] = checkpoint_meta(final)
            if do_hash:
                key = f"{name}/checkpoint_final.pth"
                st = entry["files"]["checkpoint_final.pth"]
                cached = hash_cache.get(key)
                if cached and cached.get("size") == st["size"] and cached.get("mtime") == st["mtime"]:
                    entry["sha256"] = cached["sha256"]
                else:
                    entry["sha256"] = sha256_file(final)
                    hash_cache[key] = {"sha256": entry["sha256"], "size": st["size"],
                                       "mtime": st["mtime"]}
        best = os.path.join(fold0, "checkpoint_best.pth")
        if os.path.isfile(best):
            entry["best_meta"] = checkpoint_meta(best)
        models.append(entry)
    return models


def run_identity(path):
    """(label, finished_utc) of a run.json — enough to tell two runs apart."""
    try:
        with open(path) as fh:
            d = json.load(fh)
        return (d.get("label", ""), d.get("finished_utc", ""))
    except Exception:
        return None


def survey_runs(runs_root, rows, results_dir):
    """Split runs/ into 'still running' and 'finished but not curated into results/'.

    The second half matters: once a run writes its run.json it stops looking unfinished, so a
    completed run that nobody copied into results/ would otherwise vanish from this report
    entirely. Identity is (label, finished_utc) read out of the run.json itself, so renaming a
    row folder does not create a phantom.
    """
    pending, uncurated = [], []
    if not os.path.isdir(runs_root):
        return pending, uncurated
    known = set()
    for r in rows:
        rj = os.path.join(results_dir, r["row"], "run.json")
        ident = run_identity(rj) if os.path.isfile(rj) else None
        if ident:
            known.add(ident)
    for name in sorted(os.listdir(runs_root)):
        p = os.path.join(runs_root, name)
        if not os.path.isdir(p) or name in NON_ROW_DIRS:
            continue
        rj = os.path.join(p, "run.json")
        if os.path.isfile(rj):
            ident = run_identity(rj)
            if ident and ident not in known:
                label = ident[0] or "(no label)"
                uncurated.append((name, label, ident[1]))
            continue
        # a ladder dir holding one sub-run per row is fine, look one level down
        try:
            subs = [x for x in sorted(os.listdir(p))
                    if os.path.isfile(os.path.join(p, x, "run.json"))]
        except OSError:
            subs = []
        if subs:
            continue
        pending.append((name, "no run.json yet (running or aborted)"))
    return pending, uncurated


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.path.dirname(os.path.abspath(__file__)),
                    help="autoPET Drive root (default: the directory of this script)")
    ap.add_argument("--out", default=None, help="output markdown (default <root>/RESULTS.md)")
    ap.add_argument("--no-hash", action="store_true", help="skip the checkpoint sha256 pass")
    args = ap.parse_args()

    root = os.path.abspath(args.root)
    results_dir = os.path.join(root, "results")
    out_path = args.out or os.path.join(root, "RESULTS.md")
    cache_path = os.path.join(results_dir, "model_sha256.json")

    all_rows = load_rows(results_dir)
    rows = [r for r in all_rows if not r["subset_control"]]
    subset_rows = [r for r in all_rows if r["subset_control"]]
    hash_cache = {}
    if os.path.isfile(cache_path):
        try:
            hash_cache = json.load(open(cache_path))
        except Exception:
            hash_cache = {}
    models = scan_models(os.path.join(root, "ckpt", "Dataset998_AutoPETV"), hash_cache,
                         do_hash=not args.no_hash)
    pending, uncurated = survey_runs(os.path.join(root, "runs"), all_rows, results_dir)

    L = []
    A = L.append
    A("# autoPET V — results index")
    A("")
    A(f"Generated by `results_index.py` on {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} "
      f"from `results/<row>/run.json`. Do not edit by hand — re-run the script.")
    A("")
    A("Protocol for every row below: fixed 100-case validation subset `docs/valset_v1.txt` "
      "(63 FDG / 37 PSMA, 64 lesion-bearing / 36 lesion-free), 6 iterations, all three scribble "
      "strategies round-robin over the sorted case list, `seed=42`, `--eval fixed`, official "
      "metrics from the challenge repo. AUC is `trapezoid` over iterations 0..5, max **5.0**. "
      "AUC-DMM is a nanmean, so lesion-free cases are excluded from it; a lesion-free case scores "
      "AUC-Dice 5.0 or 0.0 and nothing in between.")
    A("")
    A("## Rows")
    A("")
    A("| row | n | AUC-Dice | AUC-DMM | 50/50 | Dice@0 | Dice@5 | neg AUC-Dice | pos AUC-Dice | "
      "pos AUC-DMM | FDG AUC-Dice | FDG AUC-DMM | PSMA AUC-Dice | PSMA AUC-DMM |")
    A("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    noted = [r for r in rows if r["note"]]
    marks = {r["row"]: f" [^{i + 1}]" for i, r in enumerate(noted)}
    for r in rows:
        A("| `{row}`{mk} | {n} | {ad} | {am} | {sc} | {d0} | {d5} | {nd} | {pd} | {pm} | "
          "{fd} | {fm} | {sd} | {sm} |".format(
              row=r["row"], mk=marks.get(r["row"], ""),
              n=r["n"] if r["n"] is not None else "-",
              ad=fmt(r["auc_dice"]), am=fmt(r["auc_dmm"]), sc=fmt(r["score"]),
              d0=fmt(r["dice0"]), d5=fmt(r["dice5"]), nd=fmt(r["neg_dice"]),
              pd=fmt(r["pos_dice"]), pm=fmt(r["pos_dmm"]),
              fd=fmt(r["fdg_dice"]), fm=fmt(r["fdg_dmm"]),
              sd=fmt(r["psma_dice"]), sm=fmt(r["psma_dmm"])))
    A("")
    for i, r in enumerate(noted):
        A(f"[^{i + 1}]: **`{r['row']}`** — {r['note']}")
    if noted:
        A("")
    A("`Dice@0` / `Dice@5` are pooled over all 100 cases (a lesion-free case with an empty "
      "prediction contributes 1.0), which is why they sit above the mean Dice over the "
      "lesion-bearing cases alone.")
    A("")
    A("### What each row is")
    A("")
    A("| row | variant | predictor | finished (UTC) | numbers read from |")
    A("|---|---|---|---|---|")
    for r in rows:
        A(f"| `{r['row']}` | {r['label']} | `{r['predictor']}` | {r['finished']} | "
          f"`{r['source']}` |")
    A("")
    if any(r["source"] != "run.json" for r in rows):
        A("Rows marked `summary.json` were launched directly rather than through the run wrapper, "
          "so no `run.json` was ever written for them. `summary.json` is written by the evaluation "
          "loop itself and carries the same aggregation, the same `args` and the same "
          "`postproc_config`; what is missing is only the wrapper's provenance header (git commit, "
          "explicit finish time, run label — the label comes from `label.txt` instead). Nothing in "
          "the table is reconstructed or estimated.")
        A("")

    if subset_rows:
        A("## Robustness / subset rows")
        A("")
        A("Rows evaluated on a **stratified subset** of the 100 cases. Their numbers are not "
          "comparable to the table above — a 30- or 39-case subset has a different positive/negative "
          "and FDG/PSMA mix, so its AUC sits wherever that mix puts it. Each row is therefore shown "
          "against its control **recomputed on exactly the same case list** from the control's own "
          "`metric_scores.json`, which makes `Δ` a paired comparison and the only number in this "
          "section worth reading.")
        A("")
        A("| row | n | variant | AUC-Dice | AUC-DMM | 50/50 | Dice@0 | Dice@5 |")
        A("|---|---:|---|---:|---:|---:|---:|---:|")
        warnings = []
        for r in subset_rows:
            ctl_name = r["subset_control"]
            ctl = next((c for c in all_rows if c["row"] == ctl_name), None)
            try:
                sub_scores = json.load(open(os.path.join(r["dir"], "metric_scores.json")))
            except Exception as exc:
                warnings.append(f"`{r['row']}`: cannot read metric_scores.json ({exc})")
                continue
            cases = set(sub_scores)
            own = score_cases(sub_scores)
            A(f"| `{r['row']}` | {own['n']} | {r['subset_desc'] or r['label']} | "
              f"**{fmt(own['auc_dice'])}** | **{fmt(own['auc_dmm'])}** | {fmt(own['score'])} | "
              f"{fmt(own['dice0'])} | {fmt(own['dice5'])} |")
            if ctl is None:
                warnings.append(f"`{r['row']}`: control row `{ctl_name}` is not in results/")
                continue
            ctl_scores = json.load(open(os.path.join(ctl["dir"], "metric_scores.json")))
            missing = cases - set(ctl_scores)
            if missing:
                warnings.append(f"`{r['row']}`: {len(missing)} of its {len(cases)} cases are absent "
                                f"from control `{ctl_name}` — Δ omitted")
                continue
            # self-check: the control's own full-set numbers must fall out of the same code
            full = score_cases(ctl_scores)
            if ctl["auc_dice"] is not None and abs(full["auc_dice"] - ctl["auc_dice"]) > 1e-6:
                warnings.append(f"control `{ctl_name}`: recomputed AUC-Dice {full['auc_dice']:.6f} "
                                f"!= reported {ctl['auc_dice']:.6f}")
            paired = score_cases(ctl_scores, cases)
            A(f"| ⤷ control `{ctl_name}` | {paired['n']} | *same {paired['n']} cases* | "
              f"{fmt(paired['auc_dice'])} | {fmt(paired['auc_dmm'])} | {fmt(paired['score'])} | "
              f"{fmt(paired['dice0'])} | {fmt(paired['dice5'])} |")
            A(f"| ⤷ **Δ** | | *{r['row']} − {ctl_name}* | "
              f"**{own['auc_dice'] - paired['auc_dice']:+.3f}** | "
              f"**{own['auc_dmm'] - paired['auc_dmm']:+.3f}** | "
              f"{own['score'] - paired['score']:+.3f} | "
              f"{own['dice0'] - paired['dice0']:+.3f} | {own['dice5'] - paired['dice5']:+.3f} |")
        A("")
        for w in warnings:
            A(f"* ⚠️ {w}")
        if warnings:
            A("")
        notes_file = os.path.join(results_dir, SUBSET_NOTES)
        if os.path.isfile(notes_file):
            A(open(notes_file).read().strip())
            A("")

    curves_file = os.path.join(results_dir, CURVES_FILE)
    if os.path.isfile(curves_file):
        wanted = [ln.strip() for ln in open(curves_file)
                  if ln.strip() and not ln.lstrip().startswith("#")]
    else:
        wanted = [r["row"] for r in rows]
    by_row = {r["row"]: r for r in rows}
    curved = [by_row[w] for w in wanted if w in by_row and by_row[w]["dice_iter"]]
    if curved:
        A("## Per-iteration curves")
        A("")
        A("Mean Dice and mean DMM at each of the six iterations, pooled over all cases "
          "(the AUC columns above are `trapezoid` over exactly these values, so iterations 0 and 5 "
          "carry half the weight of the others). Iteration 0 has no scribbles; each later iteration "
          "adds one scribble simulated from the previous prediction's largest error. `Δ` is "
          "iteration 5 minus iteration 0, and `monotone?` says whether the curve ever goes down.")
        A("")
        A("| row | metric | it0 | it1 | it2 | it3 | it4 | it5 | Δ | monotone? |")
        A("|---|---|---:|---:|---:|---:|---:|---:|---:|---|")
        for r in curved:
            for metric, vals in (("Dice", r["dice_iter"]), ("DMM", r["dmm_iter"])):
                if not vals:
                    continue
                mono = all(b >= a - 1e-9 for a, b in zip(vals, vals[1:]))
                A("| `{row}` | {m} | {v} | {d} | {mo} |".format(
                    row=r["row"], m=metric,
                    v=" | ".join(fmt(x) for x in vals),
                    d=fmt(vals[-1] - vals[0]),
                    mo="yes" if mono else "**NO**"))
        A("")
        if len(wanted) < len(rows) and os.path.isfile(curves_file):
            A(f"Which rows appear here is set by `results/{CURVES_FILE}`.")
            A("")

    A("## Models")
    A("")
    A("Checkpoint folders under `ckpt/Dataset998_AutoPETV/`. They are written by the running "
      "`ckptsync` mirror — read them, never move them.")
    A("")
    A("| model folder | trainer | epochs | best vs final | checkpoint_final.pth sha256 | size |")
    A("|---|---|---|---|---|---|")
    for m in models:
        meta = m.get("meta", {})
        bmeta = m.get("best_meta", {})
        trainer = meta.get("trainer") or m["folder"].split("__")[0]
        ep = meta.get("epoch")
        # nnU-Net stores current_epoch already incremented, i.e. epochs completed
        ep = "-" if ep is None else str(ep)
        if "checkpoint_final.pth" not in m["files"]:
            note = "**no checkpoint_final.pth — training not finished**"
        elif bmeta.get("epoch") is None or meta.get("epoch") is None:
            note = "unknown"
        elif bmeta["epoch"] == meta["epoch"]:
            note = f"best == final (both epoch {bmeta['epoch']}, EMA {fmt(bmeta.get('best_ema'), 5)})"
        else:
            note = (f"**best is epoch {bmeta['epoch']}**, final is epoch {meta['epoch']} "
                    f"(EMA {fmt(bmeta.get('best_ema'), 5)}) — real best-vs-final choice")
        sha = m.get("sha256", "-")
        size = m["files"].get("checkpoint_final.pth", {}).get("size")
        A(f"| `{m['folder']}` | `{trainer}` | {ep} | {note} | `{sha}` | "
          f"{'-' if size is None else f'{size/1e6:.1f} MB'} |")
    A("")
    stale = []
    for m in models:
        f = m["files"]
        if "checkpoint_latest.pth" in f and "checkpoint_final.pth" in f and \
                f["checkpoint_latest.pth"]["mtime"] < f["checkpoint_final.pth"]["mtime"]:
            stale.append(m["folder"])
    if stale:
        A("**Stale `checkpoint_latest.pth`.** nnU-Net deletes `checkpoint_latest.pth` when training "
          "ends, but the mirror only ever adds, so an older one survives on Drive in: "
          + ", ".join(f"`{s}`" for s in stale) +
          ". Always name `checkpoint_final.pth` explicitly at evaluation time. Do not delete them.")
        A("")

    disp = {}
    dfile = os.path.join(results_dir, DISPOSITION)
    if os.path.isfile(dfile):
        try:
            disp = json.load(open(dfile))
        except Exception:
            disp = {}

    if pending:
        A("## In flight / incomplete")
        A("")
        A("| run dir | state |")
        A("|---|---|")
        for name, why in pending:
            A(f"| `runs/{name}` | {disp.get(name, why)} |")
        A("")

    if uncurated:
        A("## Finished but not in `results/`")
        A("")
        A("These run directories have a `run.json` — the run completed — but no matching row was "
          "copied into `results/`, so their numbers appear nowhere above. Most are deliberately "
          "partial (a `_sub30` / `_sub39` subset is not comparable to the 100-case rows and should "
          "not be curated as if it were); the rest are simply waiting to be added.")
        A("")
        A("| run dir | label | finished (UTC) | disposition |")
        A("|---|---|---|---|")
        for name, label, fin in uncurated:
            A(f"| `runs/{name}` | {label} | {fin} | {disp.get(name, 'not yet curated')} |")
        A("")

    with open(out_path, "w") as fh:
        fh.write("\n".join(L) + "\n")
    try:
        with open(cache_path, "w") as fh:
            json.dump(hash_cache, fh, indent=2, sort_keys=True)
    except Exception as exc:
        print(f"warning: could not write {cache_path}: {exc}")
    print(f"wrote {out_path} ({len(rows)} rows, {len(subset_rows)} subset rows, "
          f"{len(models)} models)")


if __name__ == "__main__":
    main()
