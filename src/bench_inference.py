"""Seconds per iteration for one inference configuration, measured back to back.

`run.json`'s `median_iteration_seconds` comes from a whole evaluation, so two rows
measured on different boxes -- or on the same box with a training job on the GPU -- are
not comparable, and the ratio we actually need (what does TTA cost? what does a second
member cost?) is buried under that. This script times the configurations of one row
**in one process, on the same cases, back to back**, so the ratio between them is
measured under whatever contention the box happens to have.

    python3 bench_inference.py --cases 3 \
        --config plain:/content/work/models/b10 \
        --config tta8:/content/work/models/b10 \
        --config ens:/content/work/models/b10,/content/work/models/re40

A config is `<name>:<spec>`; `<spec>` is one or more `<model_folder>[:<ckpt>]` separated
by commas (more than one = a probability-level ensemble).  `tta` in the name, or
`--tta_for <name>`, turns 8-way mirroring on for that config.  Every configuration is
run at iteration 0 (no scribbles, no previous mask), which is the iteration the GC
preliminary leaderboard ranks and the most expensive one per unit of information.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ensemble_predictor import EnsembleInteractivePredictor, parse_member_spec  # noqa: E402


def load_case(case_dir, tag):
    import nibabel as nib
    img = os.path.join(case_dir, "imagesTr")
    ct = nib.load(os.path.join(img, f"{tag}_0000.nii.gz"))
    pet = nib.load(os.path.join(img, f"{tag}_0001.nii.gz"))
    return (np.asarray(ct.dataobj, dtype=np.float32),
            np.asarray(pet.dataobj, dtype=np.float32),
            [float(x) for x in ct.header.get_zooms()[:3]], ct.affine)


def build(spec, tta, device, tile_step_size):
    from predictor import InteractiveNNUNetPredictor
    common = dict(device=device, folds=(0,), tile_step_size=tile_step_size,
                  disable_tta=not tta,
                  force_mirror_axes=(0, 1, 2) if tta else None,
                  resample_channels="scipy", resample_logits="torch",
                  num_resample_threads=4, deterministic=True)
    parts = [parse_member_spec(s) for s in spec.split(",")]
    members = [InteractiveNNUNetPredictor(model_folder=p["model_folder"],
                                          checkpoint_name=p["checkpoint"], **common)
               for p in parts]
    if len(members) == 1:
        return members[0]
    return EnsembleInteractivePredictor(members)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", action="append", required=True,
                    help="<name>:<model_folder>[:<ckpt>][,<model_folder>[:<ckpt>]]")
    ap.add_argument("--case_dir", default="/content/work/evalset")
    ap.add_argument("--cases_file", default=None)
    ap.add_argument("--cases", type=int, default=3, help="how many cases to time")
    ap.add_argument("--repeats", type=int, default=1, help="passes over the case list")
    ap.add_argument("--tta_for", nargs="*", default=None,
                    help="config names to run with 8-way mirroring; the default is "
                         "every name containing 'tta'")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--tile_step_size", type=float, default=0.5)
    ap.add_argument("--out", default=None, help="write the table as json here")
    a = ap.parse_args()

    img = os.path.join(a.case_dir, "imagesTr")
    if a.cases_file:
        with open(a.cases_file) as f:
            names = [ln.strip() for ln in f if ln.strip() and not ln.startswith("#")]
        names = [n[:-5] if n.endswith("_0000") else n for n in names]
    else:
        names = sorted(f[:-12] for f in os.listdir(img) if f.endswith("_0000.nii.gz"))
    names = names[:a.cases]
    print(f"{len(names)} case(s) from {img}")

    configs = []
    for c in a.config:
        name, _, spec = c.partition(":")
        tta = (name in a.tta_for) if a.tta_for is not None else ("tta" in name.lower())
        configs.append((name, spec, tta))

    rows = {}
    for name, spec, tta in configs:
        p = build(spec, tta, a.device, a.tile_step_size)
        if hasattr(p, "warmup"):
            p.warmup()
        tot, net = [], []
        for _ in range(a.repeats):
            for tag in names:
                ct, pet, sp, aff = load_case(a.case_dir, tag)
                t0 = time.perf_counter()
                p.predict(ct, pet, sp, {"tumor": [], "background": []}, prev_pred=None,
                          affine=aff, case_name=tag, return_probabilities=False)
                tot.append(time.perf_counter() - t0)
                lt = getattr(p, "last_timings", {}) or {}
                net.append(float(lt.get("network_s", float("nan"))))
                del ct, pet
        rows[name] = {
            "spec": spec, "tta8": tta, "n": len(tot),
            "median_total_s": round(statistics.median(tot), 2),
            "mean_total_s": round(sum(tot) / len(tot), 2),
            "median_network_s": round(statistics.median(net), 2),
            "per_case_total_s": [round(x, 2) for x in tot],
        }
        print(f"{name:<10} tta8={tta}  median {rows[name]['median_total_s']:7.2f} s "
              f"(network {rows[name]['median_network_s']:7.2f} s)  "
              f"per case {rows[name]['per_case_total_s']}")
        for m in (getattr(p, "members", None) or [p]):
            if hasattr(m, "close"):
                m.close()
        import gc
        gc.collect()
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass

    base = configs[0][0]
    print(f"\nratios against '{base}':")
    for name in rows:
        r = rows[name]["median_total_s"] / rows[base]["median_total_s"]
        rows[name]["ratio_vs_base"] = round(r, 3)
        print(f"  {name:<10} x{r:.2f}")
    if a.out:
        with open(a.out, "w") as f:
            json.dump({"base": base, "cases": names, "rows": rows}, f, indent=2)
        print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
