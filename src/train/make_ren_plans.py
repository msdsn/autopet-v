"""Write ``nnUNetPlans_ren.json`` from the RE plans by swapping in the gated network.

Everything else -- spacing, 192^3 patch, batch size, data identifier, normalization
schemes and ``pet_renorm`` -- is copied verbatim, which is what keeps RE-N comparable
with plain RE at the same epoch count and keeps the RE tensors loadable.

    python -m train.make_ren_plans --base-plans <prep>/nnUNetPlans_re.json \
        --out <prep>/nnUNetPlans_ren.json
"""

from __future__ import annotations

import argparse
import json
import os
from copy import deepcopy


def build_plans(base_plans: dict, plans_name: str, configuration: str = "3d_fullres",
                gate_stage: int = 3, gate_class: int = 1) -> dict:
    plans = deepcopy(base_plans)
    plans["plans_name"] = plans_name
    arch = plans["configurations"][configuration]["architecture"]
    arch["network_class_name"] = "train.networks_ren.ResEncPresenceGateUNet"
    arch["arch_kwargs"]["gate_stage"] = int(gate_stage)
    arch["arch_kwargs"]["gate_class"] = int(gate_class)
    return plans


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-plans", required=True, help="nnUNetPlans_re.json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--configuration", default="3d_fullres")
    ap.add_argument("--plans-name", default=None)
    ap.add_argument("--gate-stage", type=int, default=3)
    ap.add_argument("--gate-class", type=int, default=1)
    args = ap.parse_args()

    with open(args.base_plans) as f:
        base = json.load(f)
    name = args.plans_name or os.path.basename(args.out)[:-len(".json")]
    plans = build_plans(base, name, args.configuration, args.gate_stage, args.gate_class)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(plans, f, indent=2, sort_keys=False)

    cfg = plans["configurations"][args.configuration]
    kw = cfg["architecture"]["arch_kwargs"]
    grid = [int(p) for p in cfg["patch_size"]]
    cell = [1, 1, 1]
    for i, st in enumerate(kw["strides"]):
        st = [st] * 3 if isinstance(st, int) else list(st)
        grid = [g // s for g, s in zip(grid, st)]
        cell = [c * s for c, s in zip(cell, st)]
        if i == args.gate_stage:
            break
    sp = cfg["spacing"]
    ml = cell[0] * cell[1] * cell[2] * sp[0] * sp[1] * sp[2] / 1000.0
    print(f"wrote {args.out}")
    print(f"  plans_name         {plans['plans_name']}")
    print(f"  network_class_name {cfg['architecture']['network_class_name']}")
    print(f"  pet_renorm         {kw.get('pet_renorm')} (unchanged)")
    print(f"  gate_stage         {args.gate_stage}  grid {grid}  "
          f"C={kw['features_per_stage'][args.gate_stage]}  cell {cell} vox = {ml:.3f} mL")
    print(f"  data_identifier    {cfg['data_identifier']} (unchanged)")


if __name__ == "__main__":
    main()
