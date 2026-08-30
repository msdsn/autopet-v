"""RE3 -- the ResEncL warm start fed our own PET channel, unconverted.

One knob against RE (`nnUNetTrainer_InteractiveRE_40epochs` + `nnUNetPlans_re`): the
network's PET remap is switched off. Same init (`re_init_5ch.pth`), same recipe, same
40-epoch schedule, so `RE3g9-s39` vs `REg9-s39` is a paired single-block ablation.

## Why, in one measurement

RE1 and RE2 both convert our store's per-case z-score into LesionTracer's own channel-1
scheme, `CTNormalization` with fip["1"] = clip to [1.0433, 51.211] then
`(x - 7.0638) / 7.9604`. Those constants were fitted on *lesion* SUV whose mean is 7.06,
and on a low-uptake lesion they are close to destructive. Measured on
`psma_41260c3678449a2f_2020-06-12` (44 voxels, 1.076 mL, SUV min 0.92 / mean 1.46 /
max 2.18), the contrast the PET channel carries between the lesion's brightest voxel and
the surrounding body:

| channel | background | lesion max | contrast |
|---|---|---|---|
| store z-score, per case (what C0 sees, and RE3) | -0.1992 | 3.0532 | **3.2524** |
| ctnorm, pooled inversion (RE1) | -0.7563 | -0.6340 | 0.1223 |
| ctnorm, exact per case (RE2) | -0.7563 | -0.6137 | 0.1426 |

**26.6x more contrast.** On that case RE1's foreground probability is identically 0.0000
over the whole volume, with or without a tumour scribble, while C0 reaches 0.6876 inside
the lesion from the first scribble. RE is not mis-calibrated there, it is blind, and the
blindness is a property of the normalisation its backbone was trained with. RE1 and RE2
optimise zero-shot agreement with the pretrained weights (`test_pet_renorm`: 0.769 and
0.797 mean Dice against 0.604 for the raw z-score) at the cost of the low-uptake
sensitivity the interactive protocol then has to recover through scribbles. RE3 takes the
other side of that trade and lets 40 epochs of fine-tuning move the stem instead.

## Why the learning rate is NOT changed

The obvious objection is that the stem's PET column now sees a differently-scaled input
and should be allowed to move faster. It already does: the gradient of an input column is
`x_c (x) delta`, so a channel with ~26x the dynamic range produces a proportionally larger
gradient on exactly the weights that need to change, with no schedule change at all.
Adding a stem learning-rate multiplier would also make this a two-knob row against RE and
cost the paired comparison. `RE_STEM_LR_MULT` remains available as a flag
(`nnUNetTrainer_InteractiveRE.configure_optimizers`, `GroupScaledPolyLRScheduler`) if a
follow-up wants to test it on its own.
"""

from __future__ import annotations

import torch

try:  # package import (src/train is a package)
    from .nnUNetTrainer_InteractiveRE import nnUNetTrainer_InteractiveRE
except ImportError:  # flat import (folder on sys.path, e.g. nnUNet_extTrainer)
    from nnUNetTrainer_InteractiveRE import nnUNetTrainer_InteractiveRE  # type: ignore


__all__ = [
    "nnUNetTrainer_InteractiveRE3",
    "nnUNetTrainer_InteractiveRE3_40epochs",
    "nnUNetTrainer_InteractiveRE3_100epochs",
    "nnUNetTrainer_InteractiveRE3_2epochs",
]


class nnUNetTrainer_InteractiveRE3(nnUNetTrainer_InteractiveRE):
    """RE with `pet_renorm="none"`: the store's per-case z-score goes straight in."""

    NUM_EPOCHS: int = 120

    def _check_re3_plans(self) -> None:
        """Running RE3's class against `nnUNetPlans_re` would silently be RE1.

        The trainer classes differ only in name -- everything that makes RE3 RE3 is the
        one plans field -- so without this the two rows would share a recipe, differ in
        nothing, and quietly produce a duplicate of RE1 under a new results folder.
        """
        cfg = self.configuration_manager.configuration
        arch = cfg["architecture"]["arch_kwargs"].get("pet_renorm", "none")
        norm = list(cfg["normalization_schemes"])
        store = cfg.get("pet_store_renorm", "none")
        if arch != "none":
            raise RuntimeError(
                f"[RE3] plans say arch_kwargs.pet_renorm={arch!r}; RE3 needs 'none' "
                f"(use nnUNetPlans_re3, not nnUNetPlans_re)")
        if store != "none":
            raise RuntimeError(
                f"[RE3] plans say pet_store_renorm={store!r}; RE3 needs 'none' "
                f"(use nnUNetPlans_re3, not nnUNetPlans_re2)")
        if norm[1] != "ZScoreNormalization":
            raise RuntimeError(
                f"[RE3] plans say normalization_schemes[1]={norm[1]!r}; RE3 feeds the "
                f"store's z-score, so inference must produce one too")
        self.print_to_log_file(
            "[RE3] pet_renorm=none: the store's per-case z-score is fed to the network "
            "unconverted, and inference normalization_schemes[1]=ZScoreNormalization "
            "produces the same thing -- 26.6x the low-uptake contrast of the ctnorm "
            "rows, at the cost of zero-shot agreement with the pretrained weights")

    def initialize(self):
        if not self.was_initialized:
            self._check_re3_plans()
        super().initialize()


class nnUNetTrainer_InteractiveRE3_40epochs(nnUNetTrainer_InteractiveRE3):
    """The screening schedule -- paired one-knob ablation against RE's re40."""
    NUM_EPOCHS: int = 40


class nnUNetTrainer_InteractiveRE3_100epochs(nnUNetTrainer_InteractiveRE3):
    NUM_EPOCHS: int = 100


class nnUNetTrainer_InteractiveRE3_2epochs(nnUNetTrainer_InteractiveRE3):
    """Smoke-test variant."""
    NUM_EPOCHS: int = 2
