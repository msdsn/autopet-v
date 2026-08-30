"""RE2 -- RE with the PET renormalisation made exact and moved off the network.

One block against RE: `nnUNetTrainer_InteractiveRE_40epochs` + `nnUNetPlans_re` becomes
`nnUNetTrainer_InteractiveRE2_40epochs` + `nnUNetPlans_re2`, same init
(`re_init_5ch.pth`), same 40-epoch schedule, same recipe, so `RE2g9-s39` against
`REg9-s39` is a paired single-knob ablation.

What changes is where the PET channel is put into LesionTracer's units:

| | RE | RE2 |
|---|---|---|
| plans `normalization_schemes[1]` | `ZScoreNormalization` | **`CTNormalization`** |
| network `arch_kwargs.pet_renorm` | `"ctnorm"` (pooled constants, inside `forward`) | **`"none"`** |
| plans `pet_store_renorm` | `"none"` | **`"ctnorm_per_case"`** |
| constants | 1 cohort pair for all 1611 cases | **each case's own `mu_full`/`sd_full`** |

Rationale for the shape, and the inference-risk argument that decided it, are in
`re2_dataloader.py`. In one line: the container path gains **no new code at all** --
`src/predictor.py` already builds the normalisation from `cm.normalization_schemes`, so
setting `CTNormalization` there makes inference apply exactly LesionTracer's own channel-1
scheme -- while training inverts the store's z-score per case with the constants already
sitting in each case's `.pkl`. Nothing is estimated on either end, and `network(x)` is
untouched, which is what keeps sliding-window inference, `torch.compile` and
`submission/process.py` out of the blast radius.

Measured justification (`train.test_pet_renorm`, unmodified 2-channel LesionTracer on
real store lesion patches, mean Dice): `none` 0.604, `pooled` (what RE ships) 0.769,
`tracer` 0.781, `case` **0.797**. The exact constants win on both tracers -- but the gain
is +0.0275 on FDG and +0.0273 on PSMA, i.e. **tracer-neutral**, so this does *not* explain
the FDG/PSMA split seen in the RE screen. RE2 is worth running because it removes an
approximation, not because it is expected to be large.
"""

from __future__ import annotations

import torch

from nnunetv2.training.dataloading.data_loader import nnUNetDataLoader
from nnunetv2.utilities.default_n_proc_DA import get_allowed_n_proc_DA
from batchgenerators.dataloading.single_threaded_augmenter import SingleThreadedAugmenter
from batchgenerators.dataloading.nondet_multi_threaded_augmenter import NonDetMultiThreadedAugmenter

try:  # package import (src/train is a package)
    from .nnUNetTrainer_InteractiveRE import nnUNetTrainer_InteractiveRE
    from .re2_dataloader import RE2DataLoader
except ImportError:  # flat import (folder on sys.path, e.g. nnUNet_extTrainer)
    from nnUNetTrainer_InteractiveRE import nnUNetTrainer_InteractiveRE  # type: ignore
    from re2_dataloader import RE2DataLoader  # type: ignore


__all__ = [
    "nnUNetTrainer_InteractiveRE2",
    "nnUNetTrainer_InteractiveRE2_40epochs",
    "nnUNetTrainer_InteractiveRE2_100epochs",
    "nnUNetTrainer_InteractiveRE2_2epochs",
]

class nnUNetTrainer_InteractiveRE2(nnUNetTrainer_InteractiveRE):
    """RE with the exact per-case PET renormalisation."""

    NUM_EPOCHS: int = 120

    def _check_re2_plans(self) -> None:
        """Fail loudly if the plans and the trainer disagree about who renormalises.

        Running RE2's trainer against `nnUNetPlans_re` would apply the per-case
        conversion *and* the network's pooled remap, i.e. renormalise twice, and the
        run would look merely mediocre rather than broken. Cheap to assert, expensive
        to debug from a loss curve.
        """
        cfg = self.configuration_manager.configuration
        mode = cfg.get("pet_store_renorm")
        arch = cfg["architecture"]["arch_kwargs"].get("pet_renorm", "none")
        norm = list(cfg["normalization_schemes"])
        if mode != "ctnorm_per_case":
            raise RuntimeError(
                f"[RE2] plans say pet_store_renorm={mode!r}; RE2 needs "
                f"'ctnorm_per_case' (build them with make_re_plans --pet-mode store_percase)")
        if arch != "none":
            raise RuntimeError(
                f"[RE2] plans say arch_kwargs.pet_renorm={arch!r}; with "
                f"pet_store_renorm='ctnorm_per_case' the network must not remap again")
        if norm[1] != "CTNormalization":
            raise RuntimeError(
                f"[RE2] plans say normalization_schemes[1]={norm[1]!r}; inference must "
                f"apply CTNormalization so it matches what training feeds the network")
        self.print_to_log_file(
            "[RE2] pet_store_renorm=ctnorm_per_case (exact per-case mu_full/sd_full from "
            "the store properties); network pet_renorm=none; inference "
            "normalization_schemes[1]=CTNormalization -- both ends compute "
            "CTNorm(clip(SUV)) on the same SUV")

    def initialize(self):
        if not self.was_initialized:
            self._check_re2_plans()
        super().initialize()

    def get_dataloaders(self):
        """nnU-Net's `get_dataloaders` with `RE2DataLoader` on **both** sides.

        Both, unlike S1's training-only substitution: the validation loss and pseudo-Dice
        have to be computed on the same input distribution the network is trained and
        evaluated on, or the curve measures the mismatch instead of the model.
        """
        from nnunetv2.training.dataloading.nnunet_dataset import infer_dataset_class

        if self.dataset_class is None:
            self.dataset_class = infer_dataset_class(self.preprocessed_dataset_folder)

        patch_size = self.configuration_manager.patch_size
        deep_supervision_scales = self._get_deep_supervision_scales()
        (rotation_for_DA, do_dummy_2d_data_aug, initial_patch_size,
         mirror_axes) = self.configure_rotation_dummyDA_mirroring_and_inital_patch_size()

        tr_transforms = self.get_training_transforms(
            patch_size, rotation_for_DA, deep_supervision_scales, mirror_axes,
            do_dummy_2d_data_aug,
            use_mask_for_norm=self.configuration_manager.use_mask_for_norm,
            is_cascaded=self.is_cascaded,
            foreground_labels=self.label_manager.foreground_labels,
            regions=self.label_manager.foreground_regions
            if self.label_manager.has_regions else None,
            ignore_label=self.label_manager.ignore_label)
        val_transforms = self.get_validation_transforms(
            deep_supervision_scales, is_cascaded=self.is_cascaded,
            foreground_labels=self.label_manager.foreground_labels,
            regions=self.label_manager.foreground_regions
            if self.label_manager.has_regions else None,
            ignore_label=self.label_manager.ignore_label)

        dataset_tr, dataset_val = self.get_tr_and_val_datasets()

        dl_tr = RE2DataLoader(dataset_tr, self.batch_size, initial_patch_size,
                              self.configuration_manager.patch_size, self.label_manager,
                              oversample_foreground_percent=self.oversample_foreground_percent,
                              sampling_probabilities=None, pad_sides=None,
                              transforms=tr_transforms,
                              probabilistic_oversampling=self.probabilistic_oversampling)
        dl_val = RE2DataLoader(dataset_val, self.batch_size,
                               self.configuration_manager.patch_size,
                               self.configuration_manager.patch_size, self.label_manager,
                               oversample_foreground_percent=self.oversample_foreground_percent,
                               sampling_probabilities=None, pad_sides=None,
                               transforms=val_transforms,
                               probabilistic_oversampling=self.probabilistic_oversampling)

        allowed = get_allowed_n_proc_DA()
        if allowed == 0:
            return SingleThreadedAugmenter(dl_tr, None), SingleThreadedAugmenter(dl_val, None)
        mt_tr = NonDetMultiThreadedAugmenter(
            data_loader=dl_tr, transform=None, num_processes=allowed,
            num_cached=max(6, allowed // 2), seeds=None,
            pin_memory=self.device.type == 'cuda', wait_time=0.002)
        mt_val = NonDetMultiThreadedAugmenter(
            data_loader=dl_val, transform=None, num_processes=max(1, allowed // 2),
            num_cached=max(3, allowed // 4), seeds=None,
            pin_memory=self.device.type == 'cuda', wait_time=0.002)
        _ = next(mt_tr)
        _ = next(mt_val)
        return mt_tr, mt_val


class nnUNetTrainer_InteractiveRE2_40epochs(nnUNetTrainer_InteractiveRE2):
    """The screening schedule -- paired one-block ablation against RE's re40."""
    NUM_EPOCHS: int = 40


class nnUNetTrainer_InteractiveRE2_100epochs(nnUNetTrainer_InteractiveRE2):
    NUM_EPOCHS: int = 100


class nnUNetTrainer_InteractiveRE2_2epochs(nnUNetTrainer_InteractiveRE2):
    """Smoke-test variant."""
    NUM_EPOCHS: int = 2
