"""Derive the B13/B14 plans from the interactive plans by swapping the architecture.

nnU-Net builds the network from ``configurations.<cfg>.architecture`` --
``network_class_name`` resolved with ``pydoc.locate`` and ``arch_kwargs`` passed to
the constructor -- so an architecture variant is a plans file, not a code path in the
trainer. Everything else (spacing, patch size, batch size, normalization schemes) is
copied unchanged, which is what keeps the two variants comparable to the source model
and the pretrained tensors loadable.
"""

from __future__ import annotations

import argparse
import json
import os
from copy import deepcopy
from typing import Dict, List, Sequence

DEFAULT_EDIT_FEATURES: List[int] = [16, 16, 24, 24, 32, 32]


def bottleneck_grid(patch_size: Sequence[int], strides: Sequence[Sequence[int]],
                    stage: int = -1) -> List[int]:
    """Spatial size of an encoder stage's output for a given patch size."""
    size = [int(p) for p in patch_size]
    sizes = []
    for st in strides:
        st = [st] * len(size) if isinstance(st, int) else list(st)
        size = [s // t for s, t in zip(size, st)]
        sizes.append(list(size))
    return sizes[stage]


def variant_architecture(base_arch: dict, variant: str, patch_size: Sequence[int],
                         context_layers: int = 2, context_heads: int = 8,
                         context_dim: int = 128, context_mlp_ratio: float = 4.0,
                         context_type: str = "attention",
                         edit_features: Sequence[int] = None,
                         n_guidance_channels: int = 3) -> dict:
    arch = deepcopy(base_arch)
    kw = arch["arch_kwargs"]
    if variant == "b13":
        arch["network_class_name"] = "train.networks.GlobalContextUNet"
        kw["context_type"] = context_type
        kw["context_stage"] = -1
        kw["context_layers"] = int(context_layers)
        kw["context_heads"] = int(context_heads)
        kw["context_dim"] = int(context_dim)
        kw["context_mlp_ratio"] = float(context_mlp_ratio)
    elif variant == "b14":
        arch["network_class_name"] = "train.networks.EditBranchUNet"
        ef = list(edit_features) if edit_features else list(DEFAULT_EDIT_FEATURES)
        if len(ef) != int(kw["n_stages"]):
            raise ValueError(f"--edit-features needs {kw['n_stages']} entries, got {len(ef)}")
        kw["edit_features_per_stage"] = [int(f) for f in ef]
        kw["n_conv_per_edit_stage"] = 2
        kw["n_guidance_channels"] = int(n_guidance_channels)
    else:
        raise ValueError(f"unknown variant {variant!r}")
    return arch


def build_plans(base_plans: dict, variant: str, plans_name: str, configuration: str = "3d_fullres",
                **kwargs) -> dict:
    plans = deepcopy(base_plans)
    plans["plans_name"] = plans_name
    cfg = plans["configurations"][configuration]
    cfg["architecture"] = variant_architecture(cfg["architecture"], variant,
                                               cfg["patch_size"], **kwargs)
    return plans


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-plans", required=True, help="nnUNetPlans_interactive.json")
    ap.add_argument("--variant", required=True, choices=["b13", "b14"])
    ap.add_argument("--out", required=True, help="path of the plans file to write")
    ap.add_argument("--configuration", default="3d_fullres")
    ap.add_argument("--plans-name", default=None, help="default: basename of --out without .json")
    ap.add_argument("--context-type", default="attention", choices=["attention", "mamba"])
    ap.add_argument("--context-layers", type=int, default=2)
    ap.add_argument("--context-heads", type=int, default=8)
    ap.add_argument("--context-dim", type=int, default=128)
    ap.add_argument("--context-mlp-ratio", type=float, default=4.0)
    ap.add_argument("--edit-features", nargs="+", type=int, default=None)
    args = ap.parse_args()

    with open(args.base_plans) as f:
        base = json.load(f)
    name = args.plans_name or os.path.basename(args.out)[:-len(".json")]
    extra: Dict = {}
    if args.variant == "b13":
        extra = dict(context_type=args.context_type, context_layers=args.context_layers,
                     context_heads=args.context_heads, context_dim=args.context_dim,
                     context_mlp_ratio=args.context_mlp_ratio)
    else:
        extra = dict(edit_features=args.edit_features)
    plans = build_plans(base, args.variant, name, args.configuration, **extra)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(plans, f, indent=2, sort_keys=False)
    arch = plans["configurations"][args.configuration]["architecture"]
    print(f"wrote {args.out}")
    print(f"  plans_name        {plans['plans_name']}")
    print(f"  network_class_name {arch['network_class_name']}")
    for k in ("context_type", "context_stage", "context_layers", "context_dim",
              "edit_features_per_stage", "n_guidance_channels"):
        if k in arch["arch_kwargs"]:
            print(f"  {k:<22} {arch['arch_kwargs'][k]}")


if __name__ == "__main__":
    main()
