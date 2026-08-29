"""S1 -- component-balanced foreground sampling on the B10 recipe.

Same network, same loss, same interaction distribution and same 120-epoch
continuation as ``nnUNetTrainer_InteractiveV2_negfp`` (B10). The only difference is
*which patches the network is shown*: a forced-foreground patch is centred on a
connected component drawn with probability proportional to ``|c| ** S1_GAMMA``
instead of on a foreground voxel drawn uniformly, which is volume-proportional.

``S1_GAMMA = 0`` (the default) makes a 0.2 mL lesion and a 100 mL lesion equally
likely to be the centre of a patch; ``S1_GAMMA = 1`` reproduces nnU-Net exactly. The
foreground/background patch ratio is untouched, and background patches are drawn by
the stock code path.

**Zero parameters.** The graft of the B10 checkpoint is a strict load and the
identity assertion inherited from ``nnUNetTrainer_InteractiveArch`` is exact, so the
run starts as B10 and every delta is the sampler.

Only the *training* loader is swapped; validation keeps nnU-Net's sampler so the
validation curve stays comparable with the other rows.
"""

from __future__ import annotations

import os

from batchgenerators.dataloading.nondet_multi_threaded_augmenter import NonDetMultiThreadedAugmenter
from batchgenerators.dataloading.single_threaded_augmenter import SingleThreadedAugmenter

from nnunetv2.training.dataloading.data_loader import nnUNetDataLoader
from nnunetv2.training.dataloading.nnunet_dataset import infer_dataset_class
from nnunetv2.utilities.default_n_proc_DA import get_allowed_n_proc_DA

try:  # package import (src/train is a package)
    from .identity_gate import SourceIdentityGateMixin
    from .nnUNetTrainer_Interactive import _env_float, _env_int
    from .nnUNetTrainer_InteractiveArch import nnUNetTrainer_InteractiveArch
    from .s1_sampler import (default_cache_dir, nnUNetDataLoaderS1,
                             recording_dataset_class)
except ImportError:  # flat import (folder on sys.path, e.g. nnUNet_extTrainer)
    from identity_gate import SourceIdentityGateMixin  # type: ignore
    from nnUNetTrainer_Interactive import _env_float, _env_int  # type: ignore
    from nnUNetTrainer_InteractiveArch import nnUNetTrainer_InteractiveArch  # type: ignore
    from s1_sampler import (default_cache_dir, nnUNetDataLoaderS1,  # type: ignore
                            recording_dataset_class)

__all__ = ["nnUNetTrainer_InteractiveS1", "nnUNetTrainer_InteractiveS1_2epochs"]


class nnUNetTrainer_InteractiveS1(SourceIdentityGateMixin, nnUNetTrainer_InteractiveArch):
    """B10 with a component-balanced foreground sampler. No new parameters."""

    NEW_PARAM_PREFIXES: tuple = ()
    S1_GAMMA: float = 0.0
    S1_CONNECTIVITY: int = 18
    S1_MAX_SAMPLES: int = 256

    def get_tr_and_val_datasets(self):
        """Use dataset classes that remember the case they just handed out.

        ``nnUNetDataLoader.generate_train_batch`` does not pass the identifier or the
        label to ``get_bbox``, so the sampler would otherwise have no way to know
        which case's components it should be balancing.
        """
        if self.dataset_class is None:
            self.dataset_class = infer_dataset_class(self.preprocessed_dataset_folder)
        self.dataset_class = recording_dataset_class(self.dataset_class)
        return super().get_tr_and_val_datasets()

    def get_dataloaders(self):
        """nnU-Net's ``get_dataloaders`` with ``nnUNetDataLoaderS1`` on the training side.

        Copied rather than monkey-patched because the training and the validation
        loader are built in the same call and only the training one is component
        balanced.
        """
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

        gamma = _env_float("S1_GAMMA", self.S1_GAMMA)
        connectivity = _env_int("S1_CONNECTIVITY", self.S1_CONNECTIVITY)
        max_samples = _env_int("S1_MAX_SAMPLES", self.S1_MAX_SAMPLES)
        cache_dir = default_cache_dir(dataset_tr.source_folder)
        self.print_to_log_file(
            f"[S1] component-balanced foreground sampling: gamma={gamma} "
            f"(0 = uniform over components, 1 = nnU-Net's volume-proportional draw), "
            f"connectivity={connectivity}, max_samples_per_component={max_samples}, "
            f"oversample_foreground_percent={self.oversample_foreground_percent} "
            f"(unchanged), cache={cache_dir}")

        dl_tr = nnUNetDataLoaderS1(dataset_tr, self.batch_size,
                                   initial_patch_size,
                                   self.configuration_manager.patch_size,
                                   self.label_manager,
                                   oversample_foreground_percent=self.oversample_foreground_percent,
                                   sampling_probabilities=None, pad_sides=None,
                                   transforms=tr_transforms,
                                   probabilistic_oversampling=self.probabilistic_oversampling,
                                   s1_gamma=gamma, s1_connectivity=connectivity,
                                   s1_max_samples=max_samples, s1_cache_dir=cache_dir)
        dl_val = nnUNetDataLoader(dataset_val, self.batch_size,
                                  self.configuration_manager.patch_size,
                                  self.configuration_manager.patch_size,
                                  self.label_manager,
                                  oversample_foreground_percent=self.oversample_foreground_percent,
                                  sampling_probabilities=None, pad_sides=None,
                                  transforms=val_transforms,
                                  probabilistic_oversampling=self.probabilistic_oversampling)

        allowed_num_processes = get_allowed_n_proc_DA()
        if allowed_num_processes == 0:
            mt_gen_train = SingleThreadedAugmenter(dl_tr, None)
            mt_gen_val = SingleThreadedAugmenter(dl_val, None)
        else:
            mt_gen_train = NonDetMultiThreadedAugmenter(
                data_loader=dl_tr, transform=None, num_processes=allowed_num_processes,
                num_cached=max(6, allowed_num_processes // 2), seeds=None,
                pin_memory=self.device.type == 'cuda', wait_time=0.002)
            mt_gen_val = NonDetMultiThreadedAugmenter(
                data_loader=dl_val, transform=None,
                num_processes=max(1, allowed_num_processes // 2),
                num_cached=max(3, allowed_num_processes // 4), seeds=None,
                pin_memory=self.device.type == 'cuda', wait_time=0.002)
        _ = next(mt_gen_train)
        _ = next(mt_gen_val)
        return mt_gen_train, mt_gen_val


class nnUNetTrainer_InteractiveS1_2epochs(nnUNetTrainer_InteractiveS1):
    """Smoke-test variant."""
    NUM_EPOCHS = 2
