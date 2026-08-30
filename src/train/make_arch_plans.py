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
DEFAULT_EVA_IMG_SIZE: List[int] = [224, 182]      # 16 x 13 tokens at patch 14
DEFAULT_EVA_FUSE_STAGES: List[int] = [3]          # (256, 14, 20, 16) at the plans patch


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
                         context_type: str = "attention", context_rope_theta=10000.0,
                         edit_features: Sequence[int] = None,
                         n_guidance_channels: int = 3,
                         eva_img_size: Sequence[int] = None,
                         eva_fuse_stages: Sequence[int] = None,
                         eva_z_stride: int = 1, eva_freeze_blocks: int = 4,
                         eva_chunk: int = 0, eva_grad_checkpointing: bool = True,
                         eva_interact_slab: int = 4) -> dict:
    arch = deepcopy(base_arch)
    kw = arch["arch_kwargs"]
    if variant in ("b13", "b13b"):
        arch["network_class_name"] = "train.networks.GlobalContextUNet"
        kw["context_type"] = context_type
        kw["context_stage"] = -1
        kw["context_layers"] = int(context_layers)
        kw["context_heads"] = int(context_heads)
        kw["context_dim"] = int(context_dim)
        kw["context_mlp_ratio"] = float(context_mlp_ratio)
        # "grid": one theta per axis, equal to that axis's extent at the plans patch.
        # lambda_i = 2*pi*theta^(i/n), so on the 7-voxel z axis the six bands sweep
        # 6.00 down to 1.19 rad across the axis -- a usable ladder, all of it within
        # one cycle. A single theta = 10000 gives 6.00 / 0.28 / 0.013 rad instead, i.e.
        # two of three bands carry no position at all.
        kw["context_rope_theta"] = (
            [float(g) for g in bottleneck_grid(patch_size, kw["strides"], -1)]
            if context_rope_theta == "grid" else context_rope_theta)
    elif variant in ("b17", "b18"):
        # B17: the trainable 2.5D EVA-02-B branch. The token volume is resized to the
        # fused stages' grids, so the only geometry the plans have to carry is the
        # per-slice input size (a multiple of 14 in both directions).
        arch["network_class_name"] = "train.networks_eva.EVAFusionUNet"
        kw["eva_img_size"] = [int(v) for v in (eva_img_size or DEFAULT_EVA_IMG_SIZE)]
        kw["eva_fuse_stages"] = [int(v) for v in (eva_fuse_stages or DEFAULT_EVA_FUSE_STAGES)]
        kw["eva_z_stride"] = int(eva_z_stride)
        kw["eva_freeze_blocks"] = int(eva_freeze_blocks)
        kw["eva_chunk"] = int(eva_chunk)
        kw["eva_grad_checkpointing"] = bool(eva_grad_checkpointing)
        # false on purpose: these plans ship next to fold_0/ and the predictor rebuilds
        # the class from them inside a container started with --network=none. The
        # pretrained weights are loaded once at surgery time by the B17 trainer and
        # travel in our own checkpoint from then on.
        kw["eva_pretrained"] = False
        if variant == "b18":
            # B18 differs from B17 in one block: the ViT is also given the three
            # interaction channels, through a zero-init patch embedding of its own.
            # `eva_interact_slab` is the +/- half-width in slices over which a
            # scribble is spread before the 2D backbone sees it (0 = no slab).
            arch["network_class_name"] = "train.networks_eva.EVAInteractiveFusionUNet"
            kw["eva_interact_slab"] = int(eva_interact_slab)
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
    ap.add_argument("--variant", required=True, choices=["b13", "b13b", "b14", "b17", "b18"])
    ap.add_argument("--out", required=True, help="path of the plans file to write")
    ap.add_argument("--configuration", default="3d_fullres")
    ap.add_argument("--plans-name", default=None, help="default: basename of --out without .json")
    ap.add_argument("--context-type", default="attention", choices=["attention", "mamba"])
    ap.add_argument("--context-layers", type=int, default=2)
    ap.add_argument("--context-heads", type=int, default=8)
    ap.add_argument("--context-dim", type=int, default=128)
    ap.add_argument("--context-mlp-ratio", type=float, default=4.0)
    ap.add_argument("--context-rope-theta", default=10000.0,
                    help="rotary base: a number, or 'grid' for one theta per axis equal "
                         "to that axis's extent on the bottleneck grid")
    ap.add_argument("--edit-features", nargs="+", type=int, default=None)
    ap.add_argument("--eva-img-size", nargs=2, type=int, default=None,
                    help="per-slice input size, a multiple of 14 (default 224 182 = 16x13 tokens)")
    ap.add_argument("--eva-fuse-stages", nargs="+", type=int, default=None)
    ap.add_argument("--eva-z-stride", type=int, default=1)
    ap.add_argument("--eva-freeze-blocks", type=int, default=4)
    ap.add_argument("--eva-chunk", type=int, default=0,
                    help="slices per EVA forward chunk (0 = the whole batch at once)")
    ap.add_argument("--eva-interact-slab", type=int, default=4,
                    help="B18: +/- slices a scribble is spread over before the 2D "
                         "backbone sees it (0 = the raw slice)")
    args = ap.parse_args()

    with open(args.base_plans) as f:
        base = json.load(f)
    name = args.plans_name or os.path.basename(args.out)[:-len(".json")]
    extra: Dict = {}
    if args.variant in ("b13", "b13b"):
        theta = args.context_rope_theta
        if theta != "grid":
            theta = float(theta)
        extra = dict(context_type=args.context_type, context_layers=args.context_layers,
                     context_heads=args.context_heads, context_dim=args.context_dim,
                     context_mlp_ratio=args.context_mlp_ratio, context_rope_theta=theta)
    elif args.variant in ("b17", "b18"):
        extra = dict(eva_img_size=args.eva_img_size, eva_fuse_stages=args.eva_fuse_stages,
                     eva_z_stride=args.eva_z_stride, eva_freeze_blocks=args.eva_freeze_blocks,
                     eva_chunk=args.eva_chunk, eva_interact_slab=args.eva_interact_slab)
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
              "context_heads", "context_rope_theta",
              "edit_features_per_stage", "n_guidance_channels",
              "eva_img_size", "eva_fuse_stages", "eva_z_stride", "eva_freeze_blocks",
              "eva_chunk", "eva_grad_checkpointing", "eva_pretrained",
              "eva_interact_slab"):
        if k in arch["arch_kwargs"]:
            print(f"  {k:<22} {arch['arch_kwargs'][k]}")


if __name__ == "__main__":
    main()
