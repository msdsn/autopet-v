"""Docker-free acceptance suite for the v0.2 submission container.

Reproduces the container's mount layout on disk (`build_gc_sim.py`) and drives it
through the AUTOPETV_* env vars, checking agreement with the offline harness per
iteration, determinism across a state-directory restore, the persistence regimes
(either state root missing, both missing, state disabled), wall time and peak RSS
against 1200 s / 30 GB, and the output interface.  Results go to `<out>/report.json`.

Usage (on the GPU box; ~40 min for 3 cases x 3 iterations):

    python -m submission.tests.run_v02_suite \
      --images_dir /content/drive/MyDrive/autoPET/evalset/imagesTr \
      --labels_dir /content/drive/MyDrive/autoPET/evalset/labelsTr \
      --cases "psma_..." "fdg_..." \
      --model_folder /content/work/ft_model \
      --repo /content/autoPETV \
      --out /content/work/v02_suite
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
from typing import Dict, List, Optional, Sequence

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

#: the configuration the image ships; the harness is driven with this same file
SHIPPED_POSTPROC_CONFIG = os.path.join(REPO_ROOT, "submission", "postproc_config.json")

#: the B3 ablation rung, for reporting how far the shipped config has moved from it
B3_POSTPROC_SET = [
    "enable_bg_compliance=true",
    "enable_fg_compliance=true",
    "enable_cleanup=true",
    "cleanup.suv_floor_mode=component",
    "negative_gate.enabled=true",
    "negative_gate.max_prob=0.60",
    "monotone.mode=minmax",
    "pass_cached_prev_pred=true",
]


def md5(path: str) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def run(cmd: List[str], env: Optional[Dict[str, str]] = None, log: Optional[str] = None,
        cwd: Optional[str] = None) -> Dict[str, object]:
    t0 = time.time()
    e = dict(os.environ)
    if env:
        e.update(env)
    proc = subprocess.run(cmd, env=e, cwd=cwd or REPO_ROOT,
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    out = proc.stdout.decode("utf-8", "replace")
    wall = time.time() - t0
    if log:
        os.makedirs(os.path.dirname(log), exist_ok=True)
        with open(log, "w") as f:
            f.write(out)
    return {"rc": proc.returncode, "wall_s": round(wall, 1), "log": out, "log_path": log}


def parse_container_log(text: str) -> Dict[str, object]:
    """Pull the numbers the container prints about itself out of its own log."""
    info: Dict[str, object] = {}
    m = re.search(r"\[mem\] peak RSS self=([\d.]+) GB children=([\d.]+) GB", text)
    if m:
        info["rss_self_gb"] = float(m.group(1))
        info["rss_children_gb"] = float(m.group(2))
    m = re.search(r"\[timing\] (\{.*\})", text)
    if m:
        try:
            info["timings"] = json.loads(m.group(1))
        except Exception:
            pass
    m = re.search(r"\[predict\] guidance: (\{.*\})", text)
    if m:
        try:
            g = json.loads(m.group(1))
            info["prev_pred_source"] = g.get("prev_pred_source")
            info["prev_pred_voxels"] = g.get("prev_pred_voxels")
            info["n_tumor_mapped"] = g.get("n_tumor_mapped")
            info["n_background_mapped"] = g.get("n_background_mapped")
        except Exception:
            pass
    m = re.search(r"\[postproc\] (\{.*\})", text)
    if m:
        try:
            info["postproc"] = json.loads(m.group(1))
        except Exception:
            pass
    m = re.search(r"\[state\] reading case state from (\S+)", text)
    if m:
        info["state_read_from"] = m.group(1)
    m = re.search(r"mirrored \S+ -> (\[.*?\]|\(nothing to mirror\))", text)
    if m:
        info["state_mirrored_to"] = m.group(1)
    m = re.search(r"status=(\S+)", text)
    if m:
        info["status"] = m.group(1)
    info["geometry_verified"] = "geometry verified identical to the input CT" in text
    info["empty_mask_fallback"] = "EMPTY-MASK-FALLBACK" in text
    return info


# --------------------------------------------------------------------------- #
def build_sim(images_dir: str, case: str, out: str, seed: int) -> dict:
    """One `<sim>/input` tree per case (iteration 0: no clicks file yet)."""
    cmd = [sys.executable, "-m", "submission.tests.build_gc_sim",
           "--images_dir", images_dir, "--case", case, "--out", out,
           "--seed", str(seed), "--clean"]
    r = run(cmd, log=os.path.join(out, "..", "logs", os.path.basename(out) + "_build.log"))
    if r["rc"] != 0:
        raise SystemExit(f"build_gc_sim failed for {case}:\n{r['log']}")
    with open(os.path.join(out, "sim_meta.json")) as f:
        return json.load(f)


def write_clicks(sim: str, scribbles_json: Optional[str]) -> int:
    """Replace `<sim>/input/lesion-clicks.json` with the harness's scribbles."""
    sys.path.insert(0, os.path.join(REPO_ROOT, "submission", "tests"))
    from submission.tests.build_gc_sim import scribbles_to_gc

    path = os.path.join(sim, "input", "lesion-clicks.json")
    if scribbles_json is None:
        if os.path.exists(path):
            os.remove(path)
        return 0
    with open(scribbles_json) as f:
        scr = json.load(f)
    gc = scribbles_to_gc(scr)
    with open(path, "w") as f:
        json.dump(gc, f)
    return len(gc["points"])


#: the two candidate state roots, in the container's own read-preference order
def state_roots(sim: str) -> List[str]:
    return [os.path.join(sim, "cache", "state"), os.path.join(sim, "output", "state")]


def wipe_roots(sim: str, which: Sequence[str]) -> None:
    """Simulate a root that does NOT persist between the 6 calls of a case."""
    for name in which:
        shutil.rmtree(os.path.join(sim, name, "state"), ignore_errors=True)


def container_env(sim: str, model_folder: str, extra: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    env = {
        "AUTOPETV_INPUT_DIR": os.path.join(sim, "input"),
        "AUTOPETV_OUTPUT_DIR": os.path.join(sim, "output", "images", "tumor-lesion-segmentation"),
        # both roots, exactly as the image leaves them (AUTOPETV_STATE_DIR unset)
        "AUTOPETV_STATE_DIRS": os.pathsep.join(state_roots(sim)),
        "AUTOPETV_CACHE_DIR": os.path.join(sim, "cache"),
        "AUTOPETV_OUTPUT_ROOT": os.path.join(sim, "output"),
        "AUTOPETV_TMP_DIR": os.path.join(sim, "tmp"),
        "AUTOPETV_MODEL_FOLDER": model_folder,
        "AUTOPETV_PREDICTOR": "interactive_postproc",
        "AUTOPETV_NPP": "1",
        "AUTOPETV_NPS": "1",
        # the image sets these; the simulation must not inherit whatever the shell has
        "AUTOPETV_POSTPROC_CONFIG": os.path.join(REPO_ROOT, "submission", "postproc_config.json"),
        "AUTOPETV_ENABLE_TTA": "0",
        "AUTOPETV_MIRROR_AXES": "",
        "PYTHONPATH": os.pathsep.join([REPO_ROOT, os.path.join(REPO_ROOT, "src")]),
    }
    if extra:
        env.update(extra)
    return env


def out_file(sim: str) -> Optional[str]:
    d = os.path.join(sim, "output", "images", "tumor-lesion-segmentation")
    if not os.path.isdir(d):
        return None
    names = sorted(n for n in os.listdir(d) if not n.startswith("."))
    return os.path.join(d, names[0]) if len(names) == 1 else None


def compare_to_harness(out_mha: str, ref_nii: str) -> dict:
    import numpy as np
    import nibabel as nib
    import SimpleITK as sitk

    img = sitk.ReadImage(out_mha)
    arr = np.transpose(sitk.GetArrayFromImage(img), (2, 1, 0)).astype(np.uint8)
    ref = np.asanyarray(nib.load(ref_nii).dataobj).astype(np.uint8)
    if arr.shape != ref.shape:
        return {"shape_match": False, "container": list(arr.shape), "harness": list(ref.shape)}
    inter = int(np.logical_and(arr > 0, ref > 0).sum())
    s = int((arr > 0).sum()) + int((ref > 0).sum())
    return {
        "shape_match": True,
        "identical": bool(np.array_equal(arr, ref)),
        "differing_voxels": int(np.logical_xor(arr > 0, ref > 0).sum()),
        "dice": 1.0 if s == 0 else round(2.0 * inter / s, 6),
        "voxels_container": int((arr > 0).sum()),
        "voxels_harness": int((ref > 0).sum()),
        "dtype_uint8": bool(arr.dtype == np.uint8),
    }


def geometry_ok(sim: str) -> dict:
    import SimpleITK as sitk

    def g(img):
        return {"size": list(img.GetSize()),
                "spacing": [round(float(s), 9) for s in img.GetSpacing()],
                "origin": [round(float(o), 9) for o in img.GetOrigin()],
                "direction": [round(float(d), 9) for d in img.GetDirection()]}

    ct_dir = os.path.join(sim, "input", "images", "ct")
    ct = os.path.join(ct_dir, os.listdir(ct_dir)[0])
    o = out_file(sim)
    if o is None:
        return {"ok": False, "reason": "output file is not unique"}
    gi, go = g(sitk.ReadImage(ct)), g(sitk.ReadImage(o))
    return {"ok": gi == go, "size": go["size"], "spacing": go["spacing"]}


def strays(sim: str) -> List[str]:
    """Files the container left outside its three mounts.

    `state_snapshot/` is this script's own copy of the state directory, kept so the
    determinism re-run sees the same input, and is not counted.
    """
    out = []
    for root, _d, fs in os.walk(sim):
        rel = os.path.relpath(root, sim)
        top = rel.split(os.sep)[0]
        if top in ("output", "cache", "tmp", "input", "state_snapshot", "."):
            continue
        out += [os.path.join(rel, f) for f in fs]
    return out


# --------------------------------------------------------------------------- #
def check_shipped_config_matches_b3(report: dict) -> None:
    """Record whether the shipped config still equals the `B3` rung.

    Informational: the harness is driven with the shipped file itself, so this only
    makes the drift visible.
    """
    sys.path.insert(0, os.path.join(REPO_ROOT, "src"))
    from postproc import PostProcConfig

    raw = json.load(open(os.path.join(REPO_ROOT, "submission", "postproc_config.json")))
    shipped = PostProcConfig.from_dict({k: v for k, v in raw.items() if not k.startswith("_")})

    b3 = PostProcConfig()
    for item in B3_POSTPROC_SET:
        key, _, value = item.partition("=")
        obj = b3
        parts = key.split(".")
        for p in parts[:-1]:
            obj = getattr(obj, p)
        try:                       # values are JSON, with a bare string as the fallback
            parsed = json.loads(value)
        except json.JSONDecodeError:
            parsed = value
        setattr(obj, parts[-1], parsed)

    same = shipped.to_dict() == b3.to_dict()
    report["shipped_config_is_B3"] = same
    report["shipped_config"] = json.load(open(SHIPPED_POSTPROC_CONFIG))
    if not same:
        report["shipped_config_diff_vs_B3"] = sorted(
            k for k, v in shipped.to_dict().items() if v != b3.to_dict()[k])
    print(f"[config] harness and container both read {SHIPPED_POSTPROC_CONFIG}")
    print(f"[config] shipped config == the original B3 rung: {same}"
          + ("" if same else f"  (moved on: {report['shipped_config_diff_vs_B3']})"))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images_dir", required=True)
    ap.add_argument("--labels_dir", required=True)
    ap.add_argument("--cases", nargs="+", required=True)
    ap.add_argument("--model_folder", required=True)
    ap.add_argument("--repo", default=os.environ.get("AUTOPETV_REPO", "/content/autoPETV"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--max_iters", type=int, default=3)
    ap.add_argument("--harness_dir", default=None,
                    help="reuse an existing harness run instead of running it again")
    ap.add_argument("--skip_container", action="store_true")
    ap.add_argument("--seed", type=int, default=20260828)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    report: dict = {"cases": {}, "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "max_iters": args.max_iters, "model_folder": args.model_folder,
                    "cases_requested": args.cases}
    check_shipped_config_matches_b3(report)

    # ---- 1. the offline harness (the reference) --------------------------
    harness = args.harness_dir or os.path.join(args.out, "harness")
    if args.harness_dir is None:
        cmd = [sys.executable, os.path.join(REPO_ROOT, "src", "interactive_eval.py"),
               "--input_cases", args.out,          # unused: both dirs are explicit
               "--image_dir", args.images_dir, "--label_dir", args.labels_dir,
               "--out_dir", harness, "--repo", args.repo,
               "--cases", *args.cases,
               "--max_iters", str(args.max_iters),
               "--predictor", "postproc", "--base_predictor", "interactive_nnunet",
               "--model_folder", args.model_folder, "--checkpoint", "checkpoint_final.pth",
               "--keep_state", "--save_predictions", "all",
               "--npp", "1", "--nps", "1",
               # the shipped file itself, not a copy of its values, so that the harness
               # and the container cannot disagree about the configuration
               "--postproc_config", SHIPPED_POSTPROC_CONFIG]
        print("[harness]", " ".join(f'"{c}"' if " " in c else c for c in cmd), flush=True)
        r = run(cmd, log=os.path.join(args.out, "logs", "harness.log"))
        report["harness"] = {"rc": r["rc"], "wall_s": r["wall_s"]}
        print(f"[harness] rc={r['rc']} wall={r['wall_s']}s -> {harness}", flush=True)
        if r["rc"] != 0:
            print(r["log"][-4000:])
            return 1
    else:
        report["harness"] = {"reused": harness}

    if args.skip_container:
        json.dump(report, open(os.path.join(args.out, "report.json"), "w"), indent=1)
        return 0

    # ---- 2. the container, case by case ----------------------------------
    for ci, case in enumerate(args.cases):
        tag = case + "_0000"
        hdir = os.path.join(harness, tag)
        if not os.path.isdir(hdir):
            print(f"[sim] no harness output for {tag} -- skipping")
            continue
        sim = os.path.join(args.out, "sim", f"case{ci}")
        meta = build_sim(args.images_dir, case, sim, args.seed + ci)
        crec: dict = {"tag": tag, "sim": sim, "ct_uuid": meta["ct_uuid"],
                      "pet_uuid": meta["pet_uuid"], "iterations": []}
        report["cases"][case] = crec

        for it in range(args.max_iters):
            scr = os.path.join(hdir, f"iter_{it}_scribbles.json")
            n_pts = write_clicks(sim, scr if (it > 0 and os.path.isfile(scr)) else None)
            state_case_dirs = []
            for sd in state_roots(sim):
                if os.path.isdir(sd):
                    state_case_dirs += [os.path.join(os.path.basename(os.path.dirname(sd)), d)
                                        for d in sorted(os.listdir(sd)) if d.startswith("case_")]
            # Snapshot the state so the determinism re-run sees the same input.  Both
            # roots have to be captured: the container reads whichever carries the most
            # calls, so restoring one would leave the other holding this iteration's
            # own output.
            snap = os.path.join(sim, "state_snapshot")
            shutil.rmtree(snap, ignore_errors=True)
            for sd in state_roots(sim):
                if os.path.isdir(sd):
                    shutil.copytree(sd, os.path.join(snap, os.path.basename(os.path.dirname(sd))))

            log = os.path.join(args.out, "logs", f"case{ci}_iter{it}.log")
            r = run([sys.executable, "-m", "submission.process"],
                    env=container_env(sim, args.model_folder), log=log)
            info = parse_container_log(r["log"])
            o = out_file(sim)
            rec = {"iteration": it, "n_click_points": n_pts, "rc": r["rc"],
                   "wall_s": r["wall_s"], "state_dirs_before": state_case_dirs,
                   "md5": md5(o) if o else None, **info}
            ref = os.path.join(hdir, f"iter_{it}.nii.gz")
            if o and os.path.isfile(ref):
                rec["vs_harness"] = compare_to_harness(o, ref)
            rec["geometry"] = geometry_ok(sim)
            rec["strays"] = strays(sim)
            crec["iterations"].append(rec)
            v = rec.get("vs_harness", {})
            print(f"[sim] {tag} iter{it}: rc={r['rc']} wall={r['wall_s']}s "
                  f"rss={info.get('rss_self_gb')}GB prev={info.get('prev_pred_source')} "
                  f"identical={v.get('identical')} dice={v.get('dice')} "
                  f"vox={v.get('voxels_container')}/{v.get('voxels_harness')}", flush=True)

            # -- determinism: restore the pre-run state, run the same call again
            if it == 1 and ci == 0:
                for sd in state_roots(sim):
                    shutil.rmtree(sd, ignore_errors=True)
                    src = os.path.join(snap, os.path.basename(os.path.dirname(sd)))
                    if os.path.isdir(src):
                        shutil.copytree(src, sd)
                r2 = run([sys.executable, "-m", "submission.process"],
                         env=container_env(sim, args.model_folder),
                         log=os.path.join(args.out, "logs", f"case{ci}_iter{it}_rerun.log"))
                o2 = out_file(sim)
                crec["determinism_rerun"] = {
                    "iteration": it, "rc": r2["rc"], "wall_s": r2["wall_s"],
                    "md5_first": rec["md5"], "md5_rerun": md5(o2) if o2 else None,
                    "identical": bool(o2 and md5(o2) == rec["md5"]),
                }
                print(f"[det] {tag} iter{it} rerun: "
                      f"md5-identical={crec['determinism_rerun']['identical']}", flush=True)
            shutil.rmtree(snap, ignore_errors=True)

        # ---- 3. the four persistence regimes -----------------------------
        # We do not get to know which of /cache and /output survives the six calls of a
        # case, so the container writes both and reads whichever carried this case's
        # state.  Each regime wipes the root(s) that "do not persist" before every call,
        # which is what a non-persistent mount looks like from inside the container.
        if ci == 0:
            regimes = {
                # name          wiped before every call     masks must match the harness
                "cache_only":  (["output"],                 True),
                "output_only": (["cache"],                  True),
                "neither":     (["cache", "output"],        False),
            }
            crec["regimes"] = {}
            for name, (wiped, must_match) in regimes.items():
                rows = []
                # start each regime from a clean slate, so it cannot inherit the
                # state the "both" run above left behind
                wipe_roots(sim, ["cache", "output"])
                for it in range(args.max_iters):
                    scr = os.path.join(hdir, f"iter_{it}_scribbles.json")
                    write_clicks(sim, scr if (it > 0 and os.path.isfile(scr)) else None)
                    wipe_roots(sim, wiped)
                    r = run([sys.executable, "-m", "submission.process"],
                            env=container_env(sim, args.model_folder),
                            log=os.path.join(args.out, "logs", f"case{ci}_iter{it}_{name}.log"))
                    info = parse_container_log(r["log"])
                    o = out_file(sim)
                    ref = os.path.join(hdir, f"iter_{it}.nii.gz")
                    rec = {"iteration": it, "rc": r["rc"], "wall_s": r["wall_s"],
                           "md5": md5(o) if o else None, **info}
                    if o and os.path.isfile(ref):
                        rec["vs_harness"] = compare_to_harness(o, ref)
                    rec["geometry"] = geometry_ok(sim)
                    rec["must_match_harness"] = bool(must_match)
                    rec["ok"] = bool(
                        r["rc"] == 0 and rec["geometry"]["ok"]
                        and (rec.get("vs_harness", {}).get("identical") or not must_match)
                    )
                    rows.append(rec)
                    print(f"[{name}] {tag} iter{it}: rc={r['rc']} wall={r['wall_s']}s "
                          f"read={os.path.basename(os.path.dirname(str(info.get('state_read_from'))))}"
                          f"/{os.path.basename(str(info.get('state_read_from')))} "
                          f"prev={info.get('prev_pred_source')} status={info.get('status')} "
                          f"identical={rec.get('vs_harness', {}).get('identical')} "
                          f"(required={must_match}) -> {'OK' if rec['ok'] else 'FAIL'}",
                          flush=True)
                crec["regimes"][name] = rows

            # and once with state switched off entirely: must equal the "neither" run
            it = args.max_iters - 1
            scr = os.path.join(hdir, f"iter_{it}_scribbles.json")
            write_clicks(sim, scr if os.path.isfile(scr) else None)
            wipe_roots(sim, ["cache", "output"])
            r = run([sys.executable, "-m", "submission.process"],
                    env=container_env(sim, args.model_folder,
                                      {"AUTOPETV_STATE_ENABLED": "0"}),
                    log=os.path.join(args.out, "logs", f"case{ci}_state_disabled.log"))
            o = out_file(sim)
            neither = crec["regimes"]["neither"]
            crec["state_disabled"] = {
                "iteration": it, "rc": r["rc"], "wall_s": r["wall_s"],
                "md5": md5(o) if o else None,
                "roots_created": [d for d in state_roots(sim) if os.path.isdir(d)],
                "same_as_neither": bool(o and neither and md5(o) == neither[it]["md5"]),
                **parse_container_log(r["log"]),
            }
            print(f"[state=0] {tag} iter{it}: rc={r['rc']} "
                  f"same_as_neither={crec['state_disabled']['same_as_neither']} "
                  f"roots_created={crec['state_disabled']['roots_created']}", flush=True)

        json.dump(report, open(os.path.join(args.out, "report.json"), "w"), indent=1)

    # ---- summary ---------------------------------------------------------
    print("\n" + "=" * 100)
    print(f"{'case':<28} {'it':>2} {'wall_s':>7} {'rss_gb':>7} {'prev':>9} "
          f"{'identical':>10} {'dice':>9} {'vox_c':>8} {'vox_h':>8}")
    worst_wall, worst_rss, all_identical = 0.0, 0.0, True
    for case, crec in report["cases"].items():
        for rec in crec["iterations"]:
            v = rec.get("vs_harness", {})
            worst_wall = max(worst_wall, float(rec["wall_s"]))
            worst_rss = max(worst_rss, float(rec.get("rss_self_gb") or 0))
            all_identical &= bool(v.get("identical"))
            print(f"{case[:28]:<28} {rec['iteration']:>2} {rec['wall_s']:>7} "
                  f"{str(rec.get('rss_self_gb')):>7} {str(rec.get('prev_pred_source')):>9} "
                  f"{str(v.get('identical')):>10} {str(v.get('dice')):>9} "
                  f"{str(v.get('voxels_container')):>8} {str(v.get('voxels_harness')):>8}")
    regimes_ok = True
    for case, crec in report["cases"].items():
        for name, rows in (crec.get("regimes") or {}).items():
            bad = [r["iteration"] for r in rows if not r["ok"]]
            regimes_ok &= not bad
            worst_wall = max([worst_wall] + [float(r["wall_s"]) for r in rows])
            print(f"{('regime ' + name):<28} {'':>2} "
                  f"{'OK' if not bad else 'FAIL at iteration(s) ' + str(bad)}"
                  f"   (masks must match harness: {rows[0]['must_match_harness']})")
        sd = crec.get("state_disabled")
        if sd:
            print(f"{'AUTOPETV_STATE_ENABLED=0':<28} {'':>2} "
                  f"{'OK' if sd['same_as_neither'] and not sd['roots_created'] else 'FAIL'}"
                  f"   (same mask as the no-persistence regime, no state root created)")
            regimes_ok &= bool(sd["same_as_neither"] and not sd["roots_created"])
        det = crec.get("determinism_rerun")
        if det:
            print(f"{'determinism re-run':<28} {'':>2} "
                  f"{'OK' if det['identical'] else 'FAIL'}   (md5 of two runs of the same "
                  f"call against the same restored state)")
            regimes_ok &= bool(det["identical"])
    report["worst_wall_s"] = worst_wall
    report["worst_rss_gb"] = worst_rss
    report["all_masks_identical_to_harness"] = all_identical
    report["persistence_regimes_ok"] = regimes_ok
    print("=" * 100)
    print(f"worst wall {worst_wall:.0f} s ({100 * worst_wall / 1200:.1f} % of the 1200 s budget), "
          f"worst peak RSS {worst_rss:.2f} GB ({100 * worst_rss / 30:.1f} % of 30 GB)")
    print(f"every container mask identical to the harness: {all_identical}")
    print(f"every persistence regime behaved as required:  {regimes_ok}")
    json.dump(report, open(os.path.join(args.out, "report.json"), "w"), indent=1)
    print(f"report -> {os.path.join(args.out, 'report.json')}")
    return 0 if (all_identical and regimes_ok) else 1


if __name__ == "__main__":
    raise SystemExit(main())
