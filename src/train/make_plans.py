"""Derive nnUNetPlans_interactive.json and a matching dataset.json from the baseline.

Spacing, patch size and architecture are left exactly as the baseline has them, so
the weight surgery in init_from_baseline.py stays valid; only the normalization
schemes, use_mask_for_norm and the per-channel intensity properties grow to 5
channels. data_identifier still points at the 2-channel preprocessed folder.
"""

from __future__ import annotations

import argparse
import json
import os
from copy import deepcopy
from typing import Dict, List

CHANNEL_NAMES: Dict[str, str] = {
    "0": "CT",
    "1": "PET",
    # 'noNorm' is the token nnU-Net's channel-name -> normalization map understands.
    "2": "noNorm",
    "3": "noNorm",
    "4": "noNorm",
}

CHANNEL_SEMANTICS: Dict[str, str] = {
    "0": "CT",
    "1": "PET_SUV",
    "2": "tumor_scribble_guidance_clippedEDT",
    "3": "background_scribble_guidance_clippedEDT",
    "4": "previous_prediction_binary",
}

# In the plans these are resolved as class names, so it must be "NoNormalization";
# "noNorm" above is the dataset.json channel-name token for the same class.
NORMALIZATION_SCHEMES: List[str] = [
    "CTNormalization",
    "ZScoreNormalization",
    "NoNormalization",
    "NoNormalization",
    "NoNormalization",
]


def build_plans(baseline_plans: dict,
                plans_name: str = "nnUNetPlans_interactive",
                configurations: List[str] = ("3d_fullres",),
                data_identifier: str = None,
                patch_size: List[int] = None,
                batch_size: int = None,
                dataset_name: str = None) -> dict:
    plans = deepcopy(baseline_plans)
    plans["plans_name"] = plans_name
    if dataset_name is not None:
        plans["dataset_name"] = dataset_name

    keep = {c: plans["configurations"][c] for c in configurations if c in plans["configurations"]}
    if not keep:
        raise KeyError(f"none of {configurations} present in baseline plans "
                       f"({list(plans['configurations'].keys())})")
    plans["configurations"] = keep

    for cname, cfg in plans["configurations"].items():
        cfg.pop("next_stage", None)
        cfg.pop("previous_stage", None)
        cfg["normalization_schemes"] = list(NORMALIZATION_SCHEMES)
        cfg["use_mask_for_norm"] = [False] * len(NORMALIZATION_SCHEMES)
        if data_identifier is not None:
            cfg["data_identifier"] = data_identifier
        if patch_size is not None:
            cfg["patch_size"] = [int(p) for p in patch_size]
        if batch_size is not None:
            cfg["batch_size"] = int(batch_size)

    fip = plans.get("foreground_intensity_properties_per_channel", {})
    for c in ("2", "3", "4"):
        fip.setdefault(c, {"max": 1.0, "mean": 0.0, "median": 0.0, "min": 0.0,
                           "percentile_00_5": 0.0, "percentile_99_5": 1.0, "std": 1.0})
    plans["foreground_intensity_properties_per_channel"] = fip
    return plans


def build_dataset_json(baseline_dataset_json: dict, num_training: int = None) -> dict:
    dj = deepcopy(baseline_dataset_json)
    dj["channel_names"] = dict(CHANNEL_NAMES)
    dj["channel_semantics"] = dict(CHANNEL_SEMANTICS)
    dj["labels"] = dj.get("labels", {"background": 0, "tumor": 1})
    dj["description"] = ("autoPET V interactive fine-tune: CT, PET, tumor guidance (clipped EDT), "
                         "background guidance (clipped EDT), previous prediction. Channels 2-4 are "
                         "generated on the fly by nnUNetTrainer_Interactive and are NOT on disk.")
    if num_training is not None:
        dj["numTraining"] = int(num_training)
    return dj


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--baseline-plans", required=True, help="path to the baseline plans.json")
    ap.add_argument("--baseline-dataset-json", required=True, help="path to the baseline dataset.json")
    ap.add_argument("--out-dir", required=True, help="nnUNet_preprocessed/<DatasetXXX> folder")
    ap.add_argument("--plans-name", default="nnUNetPlans_interactive")
    ap.add_argument("--configurations", nargs="+", default=["3d_fullres"])
    ap.add_argument("--data-identifier", default=None,
                    help="folder holding the preprocessed CT+PET data "
                         "(default: keep the baseline's, e.g. nnUNetPlans_3d_fullres)")
    ap.add_argument("--patch-size", nargs=3, type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--dataset-name", default=None)
    ap.add_argument("--num-training", type=int, default=None)
    args = ap.parse_args()

    with open(args.baseline_plans) as f:
        base_plans = json.load(f)
    with open(args.baseline_dataset_json) as f:
        base_dj = json.load(f)

    plans = build_plans(base_plans, args.plans_name, args.configurations,
                        args.data_identifier, args.patch_size, args.batch_size,
                        args.dataset_name)
    dj = build_dataset_json(base_dj, args.num_training)

    os.makedirs(args.out_dir, exist_ok=True)
    p_out = os.path.join(args.out_dir, f"{args.plans_name}.json")
    d_out = os.path.join(args.out_dir, "dataset.json")
    with open(p_out, "w") as f:
        json.dump(plans, f, indent=2, sort_keys=False)
    with open(d_out, "w") as f:
        json.dump(dj, f, indent=2, sort_keys=False)

    for c, cfg in plans["configurations"].items():
        print(f"[{c}] data_identifier={cfg['data_identifier']} patch={cfg['patch_size']} "
              f"batch={cfg['batch_size']} norm={cfg['normalization_schemes']}")
    print("wrote", p_out)
    print("wrote", d_out)


if __name__ == "__main__":
    main()
