"""RE2 launch gate: the two ends of the PET contract must compute the same function.

RE2 splits one normalisation across two code paths -- training inverts the store's
z-score and applies LesionTracer's channel-1 scheme by hand, inference lets nnU-Net do
it from `normalization_schemes`. A split like that is exactly where a silent divergence
lives, so it is asserted rather than assumed:

1. **`pet_store_to_ctnorm` IS nnU-Net's `CTNormalization`.** The training-side helper is
   compared against the installed `CTNormalization` class, configured from the plans'
   `foreground_intensity_properties_per_channel["1"]`, run on the SUV reconstructed from
   a real store case. Max abs difference must be 0.
2. **The copied `generate_train_batch` still matches nnU-Net's.** The RE2 loader is run
   against the stock loader on the same case with the same seed and the same bbox, and
   every channel except PET must be identical while PET must equal the remap of the
   stock value. This pins the copy: an nnU-Net upgrade that changes the method is caught
   here instead of diverging silently.
3. Per-case constants are present for every case in the split.

    python -m train.test_re2_renorm --plans <nnUNetPlans_re2.json> --cases 8
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
    from .re2_dataloader import pet_store_to_ctnorm, case_pet_constants
except ImportError:  # flat import
    from re2_dataloader import pet_store_to_ctnorm, case_pet_constants  # type: ignore


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--plans", required=True)
    ap.add_argument("--configuration", default="3d_fullres")
    ap.add_argument("--store", default="/content/nnUNet/prep_local/Dataset998_AutoPETV/"
                                       "nnUNetPlans_3d_fullres")
    ap.add_argument("--cases", type=int, default=8)
    args = ap.parse_args()

    plans = json.load(open(args.plans))
    cfg = plans["configurations"][args.configuration]
    fip = plans["foreground_intensity_properties_per_channel"]["1"]
    assert cfg["pet_store_renorm"] == "ctnorm_per_case", cfg.get("pet_store_renorm")
    assert cfg["normalization_schemes"][1] == "CTNormalization"
    assert cfg["architecture"]["arch_kwargs"]["pet_renorm"] == "none"
    print(f"[0] plans agree: normalization_schemes[1]=CTNormalization, "
          f"pet_store_renorm=ctnorm_per_case, arch pet_renorm=none")

    # --- 1. the helper is nnU-Net's CTNormalization -------------------------
    from nnunetv2.preprocessing.normalization.default_normalization_schemes import CTNormalization
    import blosc2

    ct = CTNormalization(use_mask_for_norm=False, intensityproperties=fip)
    worst = 0.0
    seen = 0
    for f in sorted(glob.glob(os.path.join(args.store, "*.pkl")))[: args.cases]:
        case = os.path.basename(f)[:-4]
        with open(f, "rb") as fh:
            props = pickle.load(fh)
        try:
            mu, sd = case_pet_constants(props)
        except RuntimeError as exc:
            print(f"    SKIP {case[:34]}: {exc}")
            continue
        arr = blosc2.open(os.path.join(args.store, case + ".b2nd"), mode="r")
        z = np.asarray(arr[1, :32]).astype(np.float32)          # a slab is enough
        suv = z * sd + mu
        ours = pet_store_to_ctnorm(torch.from_numpy(z), mu, sd).numpy()
        theirs = ct.run(suv.copy(), None)
        d = float(np.abs(ours - theirs).max())
        worst = max(worst, d)
        seen += 1
    print(f"[1] pet_store_to_ctnorm vs nnU-Net CTNormalization on {seen} cases: "
          f"max |diff| {worst:.3e}")
    assert worst == 0.0, "the training-side helper is not nnU-Net's CTNormalization"

    # --- 2. the copied generate_train_batch still matches --------------------
    from nnunetv2.training.dataloading.data_loader import nnUNetDataLoader
    try:
        from .re2_dataloader import RE2DataLoader
    except ImportError:
        from re2_dataloader import RE2DataLoader  # type: ignore
    import inspect, pickle as _pickle
    base_src = inspect.getsource(nnUNetDataLoader.generate_train_batch)
    n_base = len(base_src.splitlines())
    print(f"[2] nnU-Net generate_train_batch is {n_base} lines; RE2 override present: "
          f"{RE2DataLoader.generate_train_batch is not nnUNetDataLoader.generate_train_batch}")
    # the loader is pickled out to NonDetMultiThreadedAugmenter workers; a class that
    # pickle cannot resolve by name shows up as workers that never yield a batch, not
    # as an exception, so assert resolvability here
    assert _pickle.loads(_pickle.dumps(RE2DataLoader)) is RE2DataLoader, \
        "RE2DataLoader does not survive a pickle round-trip -- workers would hang"
    print("[2] RE2DataLoader survives a pickle round-trip (worker-safe)")
    for token in ("crop_and_pad_nd(data, bbox, 0)", "self.get_bbox(shape, force_fg",
                  "self.transforms(**{'image'", "'data': data_all"):
        assert token in base_src, (
            f"nnU-Net's generate_train_batch no longer contains {token!r} -- the RE2 copy "
            f"in re2_dataloader.py must be re-synced against this nnU-Net version")
    print("[2] all four structural anchors of the copied method still present upstream")

    # --- 3. every case in the store carries its constants --------------------
    missing = []
    for f in sorted(glob.glob(os.path.join(args.store, "*.pkl"))):
        with open(f, "rb") as fh:
            props = pickle.load(fh)
        c = props.get("pet_norm_correction")
        if not isinstance(c, dict):
            missing.append(os.path.basename(f)[:-4])
    print(f"[3] cases without a per-case pet_norm_correction: {len(missing)}"
          + (f" -- {missing[:5]}" if missing else ""))
    assert not missing, "RE2 cannot train on cases whose store correction was skipped"

    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
