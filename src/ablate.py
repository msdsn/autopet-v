#!/usr/bin/env python3
"""Run a named ladder of interactive-evaluation variants and tabulate them.

One variant is one `interactive_eval.py` run: base-predictor flags plus `--postproc_set`
overrides, defined in configs/ablations.json.  Every variant sees the same case list,
strategy assignment, seed and `--cache_dir`, so the rows are comparable and the shared
cache keeps the sweep affordable.  The results table is written as markdown to
`<out_root>/<RUN_ID>/results.md` and printed.

Typical use -- the 5-case integration test:

    python3 src/ablate.py --config configs/ablations.json \
        --run_id 20260826-1200_A0-A7_postproc_ladder \
        --variants A0 A1 A2 A3 A4 A5 \
        --input_cases /content/drive/MyDrive/autoPET/evalset \
        --image_dir /content/drive/MyDrive/autoPET/evalset/imagesTr \
        --label_dir /content/drive/MyDrive/autoPET/evalset/labelsTr \
        --repo /content/autoPETV --cache_dir /content/work/cache \
        --out_root /content/work/runs --cases_file cases5.txt

Reusing an already-finished control (the full A0 sweep runs on its own box):

    python3 src/ablate.py ... --variants A1 A2 A3 A4 A5 A5a A5b A5c A6 A7 \
        --include_run A0=/content/work/runs/A0_baseline_20260826
"""
from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from typing import Dict, List, Optional, Sequence

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


# =============================================================================
# config
# =============================================================================
def load_config(path: str) -> Dict:
    with open(path) as f:
        text = f.read()
    if path.lower().endswith((".yaml", ".yml")):
        import yaml
        cfg = yaml.safe_load(text)
    else:
        cfg = json.loads(text)
    if not isinstance(cfg, dict) or "variants" not in cfg:
        raise ValueError(f"{path}: expected a mapping with a 'variants' list")
    seen = set()
    for v in cfg["variants"]:
        if "id" not in v:
            raise ValueError(f"{path}: a variant has no 'id'")
        if v["id"] in seen:
            raise ValueError(f"{path}: duplicate variant id {v['id']!r}")
        seen.add(v["id"])
    return cfg


def select_variants(cfg: Dict, wanted: Optional[Sequence[str]]) -> List[Dict]:
    by_id = {v["id"]: v for v in cfg["variants"]}
    if not wanted:
        return list(cfg["variants"])
    missing = [w for w in wanted if w not in by_id]
    if missing:
        raise SystemExit(f"unknown variant id(s) {missing}; known: {sorted(by_id)}")
    return [by_id[w] for w in wanted]


# =============================================================================
# one variant
# =============================================================================
def variant_argv(variant: Dict, defaults: Dict, args, out_dir: str) -> List[str]:
    argv = [
        sys.executable, os.path.join(HERE, "interactive_eval.py"),
        "--input_cases", args.input_cases,
        "--out_dir", out_dir,
        "--strategy", str(defaults.get("strategy", "all")),
        "--max_iters", str(defaults.get("max_iters", 6)),
        "--eval", str(defaults.get("eval", "fixed")),
        "--save_predictions", str(args.save_predictions
                                  or defaults.get("save_predictions", "all")),
    ]
    if args.image_dir:
        argv += ["--image_dir", args.image_dir]
    if args.label_dir:
        argv += ["--label_dir", args.label_dir]
    if args.repo:
        argv += ["--repo", args.repo]
    if args.cache_dir:
        argv += ["--cache_dir", args.cache_dir]
    if args.strategy_order:
        argv += ["--strategy_order", args.strategy_order]
    if args.iter_budget_s is not None:
        argv += ["--iter_budget_s", str(args.iter_budget_s)]
    cases = case_list(args)
    if cases:
        argv += ["--cases"] + cases
    if args.limit is not None:
        argv += ["--limit", str(args.limit)]

    argv += list(defaults.get("args") or [])
    argv += list(variant.get("args") or [])
    if variant.get("postproc_config"):
        argv += ["--postproc_config", variant["postproc_config"]]
    overrides = list(defaults.get("postproc_set") or []) + list(variant.get("postproc_set") or [])
    if overrides:
        argv += ["--postproc_set"] + overrides
    for extra in args.extra_arg or []:
        argv += shlex.split(extra)
    return argv


def case_list(args) -> List[str]:
    if args.cases_file:
        with open(args.cases_file) as f:
            return [ln.rstrip("\n") for ln in f if ln.strip()]
    return list(args.cases or [])


def run_variant(variant: Dict, defaults: Dict, args, run_root: str) -> Dict:
    vid = variant["id"]
    out_dir = os.path.join(run_root, vid)
    summary_path = os.path.join(out_dir, "summary.json")
    argv = variant_argv(variant, defaults, args, out_dir)
    printable = " ".join(shlex.quote(a) for a in argv)

    if args.skip_existing and os.path.isfile(summary_path):
        print(f"[{vid}] summary.json exists -> skipped", flush=True)
        return {"id": vid, "label": variant.get("label", vid), "out_dir": out_dir,
                "command": printable, "skipped": True}

    print(f"\n=== {vid}  {variant.get('label', '')}\n{printable}\n", flush=True)
    if args.dry_run:
        return {"id": vid, "label": variant.get("label", vid), "out_dir": out_dir,
                "command": printable, "dry_run": True}

    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "command.txt"), "w") as f:
        f.write(printable + "\n")
    t0 = time.time()
    proc = subprocess.run(argv, cwd=HERE)
    wall = time.time() - t0
    if proc.returncode != 0:
        print(f"[{vid}] FAILED with exit code {proc.returncode} after {wall:.0f}s",
              flush=True)
        if args.stop_on_error:
            raise SystemExit(proc.returncode)
        return {"id": vid, "label": variant.get("label", vid), "out_dir": out_dir,
                "command": printable, "failed": True, "wall_seconds": round(wall, 1)}

    finalize(out_dir, f"{args.run_id}/{vid}", variant.get("label", vid), args)
    return {"id": vid, "label": variant.get("label", vid), "out_dir": out_dir,
            "command": printable, "wall_seconds": round(wall, 1)}


def finalize(out_dir: str, run_id: str, label: str, args) -> None:
    argv = [sys.executable, os.path.join(HERE, "finalize_run.py"),
            "--run_dir", out_dir, "--run_id", run_id, "--label", label]
    if args.no_drive:
        argv += ["--no_drive"]
    else:
        argv += ["--drive", args.drive]
    subprocess.run(argv, cwd=HERE)


# =============================================================================
# table
# =============================================================================
def _median_seconds(case_info: Dict) -> Optional[float]:
    """Median wall time of a scored iteration, propagated iterations excluded.

    Post-processing time is included, so a rung served from the base cache still shows
    what its own stages cost.
    """
    xs: List[float] = []
    for info in case_info.values():
        secs = info.get("iter_seconds") or []
        reused = info.get("reused") or [None] * len(secs)
        for s, r in zip(secs, reused):
            if r == "propagated":
                continue
            if s is not None and s > 0:
                xs.append(float(s))
    return round(float(np.median(xs)), 1) if xs else None


def collect(run: Dict) -> Dict:
    """Everything the table needs, read back from a finished run directory."""
    out = dict(run)
    try:
        summary = json.load(open(os.path.join(run["out_dir"], "summary.json")))
    except Exception as e:
        out["error"] = f"no summary.json ({e})"
        return out
    try:
        case_info = json.load(open(os.path.join(run["out_dir"], "case_info.json")))
    except Exception:
        case_info = {}

    dice_it = summary.get("mean_dice_per_iteration") or []
    status = summary.get("by_lesion_status") or {}
    cache = summary.get("cache") or {}
    guar = summary.get("guarantees") or {}
    out.update(
        n_cases=summary.get("n_cases"),
        auc_dice=summary.get("mean_auc_dice"),
        auc_dmm=summary.get("mean_auc_dmm"),
        score=summary.get("final_score_50_50"),
        dice_0=dice_it[0] if dice_it else None,
        dice_last=dice_it[-1] if dice_it else None,
        neg_auc_dice=(status.get("lesion_absent") or {}).get("auc_dice"),
        n_neg=(status.get("lesion_absent") or {}).get("n"),
        pos_auc_dice=(status.get("lesion_present") or {}).get("auc_dice"),
        pos_auc_dmm=(status.get("lesion_present") or {}).get("auc_dmm"),
        n_pos=(status.get("lesion_present") or {}).get("n"),
        n_zero_fp=(summary.get("empty_error_region_exposure") or {}).get(
            "total_iters_with_zero_fp"),
        s_per_iter=_median_seconds(case_info),
        cache_hits=cache.get("hits"),
        cache_misses=cache.get("misses"),
        model_calls=cache.get("model_calls"),
        prob_upgrades=cache.get("prob_upgrades"),
        guarantee_violations=guar.get("n_iterations_violating"),
        guarantee_checked=guar.get("n_iterations_checked"),
        guarantee_enforced=guar.get("enforced"),
        fg_outside=guar.get("n_iterations_fg_outside"),
        bg_inside=guar.get("n_iterations_bg_inside"),
        iters_with_fg=guar.get("n_iterations_with_fg_points"),
        iters_with_bg=guar.get("n_iterations_with_bg_points"),
        total_seconds=summary.get("total_seconds"),
    )
    # `model_calls` is only populated by the base-level cache (postproc runs).  For a
    # plain run the loop-level cache reports misses, which is the same quantity.
    if out["model_calls"] is None:
        out["model_calls"] = out["cache_misses"]
    return out


def _f(x, nd=4):
    if x is None:
        return "-"
    try:
        if isinstance(x, float) and np.isnan(x):
            return "nan"
        return f"{float(x):.{nd}f}"
    except (TypeError, ValueError):
        return str(x)


def _i(x):
    return "-" if x is None else str(int(x))


def _guarantee_cell(r: Dict, which: str, bad_key: str, total_key: str) -> str:
    """`ok`, `N/M bad` or `(N/M)` when this rung does not claim the guarantee."""
    if r.get("guarantee_checked") in (None, 0):
        return "-"
    bad, total = r.get(bad_key), r.get(total_key)
    if bad is None:
        return "-"
    enforced = which in (r.get("guarantee_enforced") or [])
    if not enforced:
        return f"off ({bad}/{total})" if total else "off"
    return "ok" if not bad else f"BROKEN {bad}/{total}"


def results_table(rows: List[Dict], run_id: str, header_note: str = "") -> str:
    cols = ("variant", "AUC-Dice", "AUC-DMM", "Dice@0", "Dice@5", "neg AUC-Dice",
            "pos AUC-Dice", "pos AUC-DMM", "n_zero_fp", "s/iter", "fwd passes",
            "G1 bg", "G2 fg")
    lines = [f"### {run_id}", ""]
    if header_note:
        lines += [header_note, ""]
    lines += ["| " + " | ".join(cols) + " |",
              "|" + "|".join(["---"] * len(cols)) + "|"]
    for r in rows:
        if r.get("error"):
            lines.append(f"| {r['id']} | {r['error']} |" + " |" * (len(cols) - 2))
            continue
        lines.append("| " + " | ".join([
            r["id"], _f(r["auc_dice"]), _f(r["auc_dmm"]), _f(r["dice_0"]),
            _f(r["dice_last"]), _f(r["neg_auc_dice"]), _f(r["pos_auc_dice"]),
            _f(r["pos_auc_dmm"]), _i(r["n_zero_fp"]),
            "-" if r["s_per_iter"] is None else f"{r['s_per_iter']:.1f}",
            _i(r["model_calls"]),
            _guarantee_cell(r, "bg", "bg_inside", "iters_with_bg"),
            _guarantee_cell(r, "fg", "fg_outside", "iters_with_fg"),
        ]) + " |")
    lines += ["", "Legend: AUC over iterations 0..5, max 5.0. `neg AUC-Dice` is the "
              "lesion-free subset (5.0 or 0.0 per case, excluded from DMM); "
              "`pos ...` is the lesion-bearing subset. `n_zero_fp` counts iterations "
              "whose prediction had no false-positive voxel (the empty-error-region "
              "corner). `s/iter` is the median wall time of a scored iteration, "
              "post-processing included. `fwd passes` is the number of network calls "
              "the variant actually paid for; the rest came from the shared base cache. "
              "`G1 bg` = no background scribble inside the scored mask, `G2 fg` = every "
              "tumor scribble inside it, checked at every iteration; `off (n/m)` means "
              "the rung does not run that compliance stage and n of m iterations that "
              "had such a point did not satisfy it.", ""]
    for r in rows:
        lines.append(f"* **{r['id']}** — {r.get('label', '')}"
                     + (f"  (n={r['n_cases']}, "
                        f"{r.get('n_pos')} positive / {r.get('n_neg')} lesion-free, "
                        f"{_i(r.get('cache_hits'))} cache hits, "
                        f"{_i(r.get('prob_upgrades'))} prob upgrades, "
                        f"total {r.get('total_seconds')}s)" if not r.get("error") else ""))
    return "\n".join(lines) + "\n"


# =============================================================================
# CLI
# =============================================================================
def make_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default=os.path.join(HERE, "..", "configs", "ablations.json"))
    p.add_argument("--run_id", default=None,
                   help="<UTC yyyymmdd-HHMM>_<expid>_<slug>; defaults to a generated one")
    p.add_argument("--variants", nargs="*", default=None,
                   help="ids to run, in order; default = all of them")
    p.add_argument("--out_root", default="/content/work/runs")
    p.add_argument("--drive", default="/content/drive/MyDrive/autoPET/runs")
    p.add_argument("--no_drive", action="store_true")

    p.add_argument("--input_cases", required=True)
    p.add_argument("--image_dir", default=None)
    p.add_argument("--label_dir", default=None)
    p.add_argument("--repo", default=None)
    p.add_argument("--cache_dir", default=None,
                   help="shared prediction cache; point it at an existing one to reuse "
                        "the base inference of a run that has already happened")
    p.add_argument("--cases", nargs="*", default=None)
    p.add_argument("--cases_file", default=None,
                   help="one case tag per line (handles the FDG names with spaces)")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--strategy_order", default=None)
    p.add_argument("--save_predictions", default=None)
    p.add_argument("--iter_budget_s", type=float, default=None)
    p.add_argument("--extra_arg", action="append", default=None, metavar="'--flag value'",
                   help="one shell-quoted string appended verbatim to every "
                        "interactive_eval.py invocation; repeatable")

    p.add_argument("--include_run", nargs="*", default=None, metavar="ID=PATH",
                   help="add a finished run directory to the table without re-running it")
    p.add_argument("--skip_existing", action="store_true",
                   help="do not re-run a variant whose out_dir already has summary.json")
    p.add_argument("--stop_on_error", action="store_true")
    p.add_argument("--dry_run", action="store_true",
                   help="print the commands and the resolved run layout, run nothing")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = make_parser().parse_args(argv)
    cfg = load_config(os.path.abspath(args.config))
    defaults = cfg.get("defaults") or {}
    variants = select_variants(cfg, args.variants)

    run_id = args.run_id or (time.strftime("%Y%m%d-%H%M", time.gmtime())
                             + "_A0-A7_postproc_ladder")
    run_root = os.path.join(args.out_root, run_id)
    os.makedirs(run_root, exist_ok=True)
    print(f"RUN_ID   {run_id}\nrun root {run_root}\n"
          f"cache    {args.cache_dir}\nvariants {[v['id'] for v in variants]}", flush=True)
    with open(os.path.join(run_root, "sweep.json"), "w") as f:
        json.dump({"run_id": run_id, "config": os.path.abspath(args.config),
                   "variants": [v["id"] for v in variants],
                   "cases": case_list(args), "args": vars(args),
                   "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())},
                  f, indent=2)

    runs: List[Dict] = []
    for item in args.include_run or []:
        vid, _, path = item.partition("=")
        if not path:
            raise SystemExit(f"--include_run expects ID=PATH, got {item!r}")
        runs.append({"id": vid, "label": f"{vid} (imported from {path})",
                     "out_dir": path, "command": "(imported)"})

    for variant in variants:
        runs.append(run_variant(variant, defaults, args, run_root))

    if args.dry_run:
        return 0

    rows = [collect(r) for r in runs]
    note = (f"Cache: `{args.cache_dir}` shared by every variant. "
            f"Cases: {len(case_list(args)) or 'all'}; "
            f"strategy `{defaults.get('strategy', 'all')}`, "
            f"{defaults.get('max_iters', 6)} iterations.")
    table = results_table(rows, run_id, note)
    md_path = os.path.join(run_root, "results.md")
    with open(md_path, "w") as f:
        f.write(table)
    with open(os.path.join(run_root, "results.json"), "w") as f:
        json.dump(rows, f, indent=2)
    if not args.no_drive:
        dst = os.path.join(args.drive, run_id)
        try:
            os.makedirs(dst, exist_ok=True)
            for name in ("results.md", "results.json", "sweep.json"):
                with open(os.path.join(run_root, name)) as src, \
                        open(os.path.join(dst, name), "w") as out:
                    out.write(src.read())
            print(f"copied results to {dst}")
        except OSError as e:
            print(f"could not copy results to Drive: {e}")

    print("\n" + table)
    print(f"wrote {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
