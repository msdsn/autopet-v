"""Per-tracer confirmation of the `pet_renorm` mechanism. Inference only, ~2 min.

The RE screen was better on FDG and worse on PSMA in 4 of 4 comparisons, and measuring
the store gives a reason: `pet_renorm="ctnorm"` inverts the store's z-score with ONE
pair of cohort constants, while PSMA's per-case `sd` is 1.9x FDG's. This runs the
**unmodified 2-channel LesionTracer** on real store lesion patches under four PET
representations and splits the result by tracer -- the cheapest possible test of that
claim, and it needs no training.

    python -m train.test_pet_renorm --plans <nnUNetPlans_re.json> \
        --init <re_init_5ch.pth> --cuda --per-tracer 6

| representation | mu, sd used for SUV = z*sd + mu |
|---|---|
| `none`   | no inversion; the store's per-case z-score goes in raw |
| `pooled` | 0.1088 / 0.6249 -- the shipped constants (B17's, 120 cases) |
| `tracer` | FDG 0.0899 / 0.5168, PSMA 0.1441 / 0.9856 (medians over all 1611 cases) |
| `case`   | this case's own `mu_full` / `sd_full` from its store `.pkl` -- exact |

All four then apply LesionTracer's own channel-1 normalisation,
`(clip(SUV, 1.0433, 51.211) - 7.0638) / 7.9604`.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import pickle

import numpy as np
import torch

try:
    from .networks_re import STEM_CONV_KEYS, PET_LO, PET_HI, PET_MEAN, PET_STD
    from .test_networks_re import build, STOCK
except ImportError:  # flat import
    from networks_re import STEM_CONV_KEYS, PET_LO, PET_HI, PET_MEAN, PET_STD  # type: ignore
    from test_networks_re import build, STOCK  # type: ignore

# cohort medians of the store's per-case pet_norm_correction (train.pet_stats, n=1611)
POOLED = (0.1088, 0.6249)          # the constants nnUNetPlans_re shipped
BY_TRACER = {"fdg": (0.0899, 0.5168), "psma": (0.1441, 0.9856)}


def ctnorm(z: torch.Tensor, mu: float, sd: float) -> torch.Tensor:
    return ((z * sd + mu).clamp(PET_LO, PET_HI) - PET_MEAN) / PET_STD


def dice(pred: np.ndarray, gt: np.ndarray) -> float:
    s = pred.sum() + gt.sum()
    return 1.0 if s == 0 else float(2.0 * np.logical_and(pred, gt).sum() / s)


def lesion_patch(store: str, case: str, patch):
    import blosc2
    seg = np.asarray(blosc2.open(os.path.join(store, case + "_seg.b2nd"), mode="r")[:])[0]
    if seg.max() == 0:
        return None
    idx = np.argwhere(seg > 0)
    c = idx[len(idx) // 2]
    data = blosc2.open(os.path.join(store, case + ".b2nd"), mode="r")
    sl, pad = [], []
    for a in range(3):
        lo = max(0, min(int(c[a]) - patch[a] // 2, seg.shape[a] - patch[a]))
        hi = min(seg.shape[a], lo + patch[a])
        sl.append(slice(lo, hi))
        pad.append((0, patch[a] - (hi - lo)))
    img = np.asarray(data[(slice(0, 2), *sl)]).astype(np.float32)
    lab = (seg[tuple(sl)] > 0)
    if any(p[1] for p in pad):
        img = np.pad(img, [(0, 0)] + pad)
        lab = np.pad(lab, pad)
    return (img, lab) if lab.sum() else None


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--plans", required=True)
    ap.add_argument("--init", required=True)
    ap.add_argument("--configuration", default="3d_fullres")
    ap.add_argument("--store", default="/content/nnUNet/prep_local/Dataset998_AutoPETV/"
                                       "nnUNetPlans_3d_fullres")
    ap.add_argument("--per-tracer", type=int, default=6, help="lesion cases per tracer")
    ap.add_argument("--cuda", action="store_true")
    args = ap.parse_args()

    device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")
    plans = json.load(open(args.plans))
    cfg = plans["configurations"][args.configuration]
    arch, patch = cfg["architecture"], [int(p) for p in cfg["patch_size"]]

    net = build(arch, 2, 2, False, STOCK, drop_kwargs=("pet_renorm",))
    sd = torch.load(args.init, map_location="cpu", weights_only=False)["network_weights"]
    net.load_state_dict({k: (v[:, :2].contiguous() if k in STEM_CONV_KEYS else v)
                         for k, v in sd.items()}, strict=True)
    net = net.to(device).eval()
    print(f"stock 2-channel LesionTracer, patch {patch}, store {args.store}")

    picked = {"fdg": [], "psma": []}
    for f in sorted(glob.glob(os.path.join(args.store, "*_seg.b2nd"))):
        case = os.path.basename(f)[: -len("_seg.b2nd")]
        t = "fdg" if case.startswith("fdg") else "psma" if case.startswith("psma") else None
        if t is None or len(picked[t]) >= args.per_tracer:
            continue
        if all(len(v) >= args.per_tracer for v in picked.values()):
            break
        picked[t].append(case)

    modes = ["none", "pooled", "tracer", "case"]
    results = {t: {m: [] for m in modes} for t in picked}
    for tracer, cases in picked.items():
        for case in cases:
            got = lesion_patch(args.store, case, patch)
            if got is None:
                continue
            img, lab = got
            with open(os.path.join(args.store, case + ".pkl"), "rb") as fh:
                corr = pickle.load(fh).get("pet_norm_correction")
            if not isinstance(corr, dict):
                continue
            per_case = (corr["mu_full"], corr["sd_full"])
            x = torch.from_numpy(img)[None]
            row = []
            for mode in modes:
                xi = x.clone()
                if mode == "pooled":
                    xi[:, 1] = ctnorm(xi[:, 1], *POOLED)
                elif mode == "tracer":
                    xi[:, 1] = ctnorm(xi[:, 1], *BY_TRACER[tracer])
                elif mode == "case":
                    xi[:, 1] = ctnorm(xi[:, 1], *per_case)
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16,
                                                     enabled=device.type == "cuda"):
                    logits = net(xi.to(device))
                d = dice(logits.argmax(1)[0].cpu().numpy() > 0, lab)
                results[tracer][mode].append(d)
                row.append(d)
            print(f"  {tracer:<5} {case[:30]:<32} mu={per_case[0]:.4f} sd={per_case[1]:.4f}  "
                  + "  ".join(f"{m}={v:.4f}" for m, v in zip(modes, row)))

    print("\n" + "=" * 78)
    print(f"{'':<8}" + "".join(f"{m:>13}" for m in modes))
    for tracer in ("fdg", "psma"):
        v = results[tracer]
        if not v[modes[0]]:
            continue
        print(f"{tracer.upper():<8}" + "".join(
            f"{np.mean(v[m]):>13.4f}" for m in modes) + f"   (n={len(v[modes[0]])})")
    allv = {m: [d for t in results for d in results[t][m]] for m in modes}
    if allv[modes[0]]:
        print(f"{'ALL':<8}" + "".join(f"{np.mean(allv[m]):>13.4f}" for m in modes)
              + f"   (n={len(allv[modes[0]])})")


if __name__ == "__main__":
    main()
