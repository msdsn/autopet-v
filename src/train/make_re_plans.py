"""Write ``nnUNetPlans_re.json``: our interactive plans, LesionTracer's architecture.

The file is *our* ``nnUNetPlans_interactive.json`` with exactly three fields replaced
from the LesionTracer plans (Zenodo 14007247, ``nnUNetResEncUNetLPlansMultiTalent``,
configuration ``3d_fullres``):

* ``architecture`` -- the ResEncL description, with ``network_class_name`` pointed at
  ``train.networks_re.ResEncInteractiveUNet`` and ``pet_renorm`` added to
  ``arch_kwargs``,
* ``patch_size``,
* ``batch_size``.

Everything else stays ours on purpose, and each of those is load-bearing:

* ``spacing`` is ``[3.0, 2.0364201068878174, 2.0364201068878174]`` on **both** sides,
  so the pretrained filters see the same millimetres per voxel they were trained at
  and the existing store needs no re-preprocessing. This is the fact that makes the
  whole row affordable.
* ``data_identifier`` keeps pointing at ``nnUNetPlans_3d_fullres``, the staged store.
* ``normalization_schemes`` stays the interactive five (``CTNormalization``,
  ``ZScoreNormalization``, ``NoNormalization`` x3): they describe how the *store on
  disk* was built and how the predictor must preprocess at inference. Their PET
  scheme is reproduced inside the network instead (``pet_renorm``), which is the only
  place it can be applied without rebuilding the 39.5 GB store.
* ``foreground_intensity_properties_per_channel`` stays ours -- and for channels 0
  and 1 it is byte-identical to theirs anyway, both fingerprints coming from the same
  autoPET cohort.
* ``batch_dice`` stays ours (``true``; theirs is ``false``), so the loss is the B10
  recipe the C0 control runs and the row measures the backbone, not the loss.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from copy import deepcopy
from typing import Sequence

RE_CLASS = "train.networks_re.ResEncInteractiveUNet"
STOCK_CLASS = "dynamic_network_architectures.architectures.unet.ResidualEncoderUNet"


def total_stride(strides: Sequence) -> list:
    tot = None
    for st in strides:
        st = list(st) if isinstance(st, (list, tuple)) else [st] * 3
        tot = st if tot is None else [a * b for a, b in zip(tot, st)]
    return tot


def build_plans(base_plans: dict, lt_plans: dict, plans_name: str = "nnUNetPlans_re",
                configuration: str = "3d_fullres", lt_configuration: str = "3d_fullres",
                patch_size: Sequence[int] = None, batch_size: int = None,
                pet_renorm: str = "ctnorm", network_class: str = RE_CLASS) -> dict:
    plans = deepcopy(base_plans)
    plans["plans_name"] = plans_name
    plans["configurations"] = {configuration: plans["configurations"][configuration]}
    cfg = plans["configurations"][configuration]

    lt_cfg = lt_plans["configurations"][lt_configuration]
    for k in ("inherits_from",):
        if k in lt_cfg:
            raise ValueError(f"{lt_configuration} inherits from {lt_cfg[k]}; resolve it first "
                             f"(use --lt-configuration 3d_fullres)")

    # the spacing must match, or the pretrained filters are looking at the wrong scale
    if [round(float(s), 6) for s in lt_cfg["spacing"]] != [round(float(s), 6) for s in cfg["spacing"]]:
        raise ValueError(f"spacing mismatch: ours {cfg['spacing']} vs LesionTracer "
                         f"{lt_cfg['spacing']} -- the warm start is only valid at equal spacing")

    arch = deepcopy(lt_cfg["architecture"])
    arch["network_class_name"] = network_class
    if network_class == RE_CLASS:
        arch["arch_kwargs"]["pet_renorm"] = str(pet_renorm)
    cfg["architecture"] = arch

    ps = [int(p) for p in (patch_size if patch_size is not None else lt_cfg["patch_size"])]
    tot = total_stride(arch["arch_kwargs"]["strides"])
    bad = [(p, t) for p, t in zip(ps, tot) if p % t]
    if bad:
        raise ValueError(f"patch {ps} is not divisible by the total stride {tot} (offenders {bad})")
    cfg["patch_size"] = ps
    cfg["batch_size"] = int(batch_size if batch_size is not None else lt_cfg["batch_size"])
    return plans


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-plans", required=True, help="our nnUNetPlans_interactive.json")
    ap.add_argument("--lesiontracer-plans", required=True,
                    help="plans.json from the Zenodo 14007247 model folder")
    ap.add_argument("--out", required=True)
    ap.add_argument("--plans-name", default=None, help="default: basename of --out without .json")
    ap.add_argument("--configuration", default="3d_fullres")
    ap.add_argument("--lt-configuration", default="3d_fullres")
    ap.add_argument("--patch-size", nargs=3, type=int, default=None,
                    help="default: LesionTracer's 192 192 192")
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--pet-renorm", choices=["none", "ctnorm"], default="ctnorm")
    ap.add_argument("--stock-class", action="store_true",
                    help="emit stock ResidualEncoderUNet instead (for the zero-shot reference)")
    args = ap.parse_args()

    with open(args.base_plans) as f:
        base = json.load(f)
    with open(args.lesiontracer_plans) as f:
        lt = json.load(f)
    name = args.plans_name or os.path.basename(args.out)[: -len(".json")]
    plans = build_plans(base, lt, name, args.configuration, args.lt_configuration,
                        args.patch_size, args.batch_size, args.pet_renorm,
                        STOCK_CLASS if args.stock_class else RE_CLASS)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(plans, f, indent=2, sort_keys=False)

    cfg = plans["configurations"][args.configuration]
    kw = cfg["architecture"]["arch_kwargs"]
    print(f"wrote {args.out}")
    print(f"  plans_name         {plans['plans_name']}")
    print(f"  network_class_name {cfg['architecture']['network_class_name']}")
    print(f"  patch_size         {cfg['patch_size']}  ({math.prod(cfg['patch_size'])/1e6:.2f} M voxels)")
    print(f"  batch_size         {cfg['batch_size']}   batch_dice {cfg['batch_dice']}")
    print(f"  spacing            {cfg['spacing']}")
    print(f"  data_identifier    {cfg['data_identifier']}")
    print(f"  normalization      {cfg['normalization_schemes']}")
    print(f"  pet_renorm         {kw.get('pet_renorm')}")
    print(f"  features_per_stage {kw['features_per_stage']}")
    print(f"  n_blocks_per_stage {kw['n_blocks_per_stage']}")
    print(f"  total stride       {total_stride(kw['strides'])}")


if __name__ == "__main__":
    main()
