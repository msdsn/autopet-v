"""Derive the N1 plans from the interactive plans by swapping in ``PresenceGateUNet``.

nnU-Net builds the network from ``configurations.<cfg>.architecture``:
``network_class_name`` is resolved with ``pydoc.locate`` and ``arch_kwargs`` is handed
to the constructor. An architecture variant is therefore a plans file, not a code path
in the trainer. Spacing, patch size, batch size, data identifier and normalization
schemes are copied unchanged, which is what keeps N1 comparable with the control and
the B10 tensors loadable.

    python -m train.make_n1_plans --base-plans <prep>/nnUNetPlans_interactive.json \
        --out <prep>/nnUNetPlans_n1.json
"""

from __future__ import annotations

import argparse
import json
import os
from copy import deepcopy
from typing import List, Sequence


def stage_grid(patch_size: Sequence[int], strides: Sequence, stage: int) -> List[int]:
    """Spatial size of an encoder stage's output for a given patch size."""
    size = [int(p) for p in patch_size]
    sizes = []
    for st in strides:
        st = [st] * len(size) if isinstance(st, int) else list(st)
        size = [s // t for s, t in zip(size, st)]
        sizes.append(list(size))
    return sizes[stage]


def build_plans(base_plans: dict, plans_name: str, configuration: str = "3d_fullres",
                gate_stage: int = 3, gate_class: int = 1) -> dict:
    plans = deepcopy(base_plans)
    plans["plans_name"] = plans_name
    cfg = plans["configurations"][configuration]
    arch = cfg["architecture"]
    arch["network_class_name"] = "train.networks_n1.PresenceGateUNet"
    arch["arch_kwargs"]["gate_stage"] = int(gate_stage)
    arch["arch_kwargs"]["gate_class"] = int(gate_class)
    return plans


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-plans", required=True, help="nnUNetPlans_interactive.json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--configuration", default="3d_fullres")
    ap.add_argument("--plans-name", default=None,
                    help="default: basename of --out without .json")
    ap.add_argument("--gate-stage", type=int, default=3,
                    help="encoder stage the presence head reads (3 = 8x downsampled)")
    ap.add_argument("--gate-class", type=int, default=1,
                    help="logit channel the log-odds map is added to")
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
    grid = stage_grid(cfg["patch_size"], kw["strides"], args.gate_stage)
    sp = cfg["spacing"]
    cell = [p // g for p, g in zip(cfg["patch_size"], grid)]
    ml = cell[0] * cell[1] * cell[2] * sp[0] * sp[1] * sp[2] / 1000.0
    print(f"wrote {args.out}")
    print(f"  plans_name         {plans['plans_name']}")
    print(f"  network_class_name {cfg['architecture']['network_class_name']}")
    print(f"  gate_stage         {args.gate_stage}  grid {grid}  "
          f"cell {cell} voxels = {ml:.2f} mL")
    print(f"  gate_class         {args.gate_class}")
    print(f"  data_identifier    {cfg['data_identifier']} (unchanged)")


if __name__ == "__main__":
    main()
