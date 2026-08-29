"""nnU-Net trainer for the autoPET V interactive model.

Fine-tunes the organizers' Dataset998 baseline into a 5-channel model: ch0 CT, ch1
PET (SUV), ch2 tumor guidance, ch3 background guidance, ch4 previous prediction.
Channels 2-4 are NoNormalization and are generated per patch by
InteractionSimulationTransform. Register with nnUNet_extTrainer=<repo>/src/train.
"""

from __future__ import annotations

import os
from typing import List, Tuple, Union

import numpy as np
import torch

from batchgeneratorsv2.helpers.scalar_type import RandomScalar
from batchgeneratorsv2.transforms.base.basic_transform import BasicTransform
from batchgeneratorsv2.transforms.utils.compose import ComposeTransforms
from batchgeneratorsv2.transforms.utils.deep_supervision_downsampling import DownsampleSegForDSTransform

from nnunetv2.training.loss.compound_losses import DC_and_BCE_loss, DC_and_CE_loss
from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper
from nnunetv2.training.loss.dice import MemoryEfficientSoftDiceLoss
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer

try:  # package import (src/train is a package)
    from .interaction_transform import InteractionSimulationTransform, KeepFirstChannelsTransform
    from .corrupt import CorruptionConfig
except ImportError:  # flat import (folder placed on sys.path, e.g. nnUNet_extTrainer)
    from interaction_transform import (InteractionSimulationTransform,  # type: ignore
                                       KeepFirstChannelsTransform)
    from corrupt import CorruptionConfig  # type: ignore


def _env_float(name: str, default: float) -> float:
    v = os.environ.get(name)
    return float(v) if v is not None and v != "" else default


def _env_int(name: str, default: int) -> int:
    v = os.environ.get(name)
    return int(v) if v is not None and v != "" else default


def _env_bool(name: str, default: bool) -> bool:
    v = os.environ.get(name)
    if v is None or v == "":
        return default
    return v.strip().lower() in ("1", "true", "yes", "on")


class nnUNetTrainer_Interactive(nnUNetTrainer):
    # ---- interaction hyper-parameters (class attributes so variants can override) ----
    # P(k scribbles): 6 evaluation iterations -> at most 5 accumulated scribbles
    K_PROBS: Tuple[float, ...] = (0.28, 0.22, 0.18, 0.14, 0.10, 0.08)
    GUIDANCE_RADIUS: float = 10.0        # clipped-EDT radius, voxels
    SLICE_AXIS: int = 0                  # axial axis in the preprocessed array (spacing 3.0 mm)
    P_PERTURB: float = 0.2               # ScribblePrompt-style stroke perturbation
    P_INDEPENDENT_SCRIBBLES: float = 0.25  # Category-2: scribbles from another model's errors
    PET_CHANNEL: int = 1
    N_IMAGE_CHANNELS: int = 2            # CT, PET -- everything after is generated
    SAVE_EVERY: int = 5                  # ephemeral disk: sync checkpoints out often
    FORCE_EMPTY_PREV_IF_NO_SCRIBBLES: bool = True
    NO_SMOOTH_DICE: bool = True          # LesionTracer "noSmooth" trick (+0.6..1.1 Dice)
    NUM_EPOCHS: int = 200
    INITIAL_LR: float = 1e-3
    SKIP_FINAL_VALIDATION: bool = True   # see perform_actual_validation()

    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device("cuda")):
        # nnUNetTrainer.__init__ pops this, so read it first: on `--c` we must not
        # let the pretrained-weights env var clobber the resumed weights.
        self._continue_training = bool(plans.get("continue_training", False))
        super().__init__(plans, configuration, fold, dataset_json, device)

        self.save_every = _env_int("nnUNet_interactive_save_every", self.SAVE_EVERY)
        self.num_epochs = _env_int("nnUNet_interactive_epochs", self.NUM_EPOCHS)
        self.initial_lr = _env_float("nnUNet_interactive_lr", self.INITIAL_LR)
        self.num_iterations_per_epoch = _env_int("nnUNet_interactive_iters_per_epoch",
                                                 self.num_iterations_per_epoch)
        self.num_val_iterations_per_epoch = _env_int("nnUNet_interactive_val_iters_per_epoch",
                                                     self.num_val_iterations_per_epoch)
        self.no_smooth_dice = _env_bool("nnUNet_interactive_noSmooth", self.NO_SMOOTH_DICE)
        self.pretrained_weights_file = os.environ.get("nnUNet_interactive_pretrained") or None
        self.skip_final_validation = _env_bool("nnUNet_interactive_skip_final_val",
                                               self.SKIP_FINAL_VALIDATION)

        self.print_to_log_file(
            f"[interactive] epochs={self.num_epochs} lr={self.initial_lr} "
            f"k_probs={list(self.K_PROBS)} R={self.GUIDANCE_RADIUS} "
            f"slice_axis={self.SLICE_AXIS} p_perturb={self.P_PERTURB} "
            f"p_independent={self.P_INDEPENDENT_SCRIBBLES} noSmoothDice={self.no_smooth_dice} "
            f"save_every={self.save_every} continue={self._continue_training}")

    # ------------------------------------------------------------------
    # network init from the weight-surgery checkpoint
    # ------------------------------------------------------------------
    def initialize(self):
        already = self.was_initialized
        super().initialize()
        if self._continue_training and self.pretrained_weights_file:
            self.print_to_log_file(
                "[interactive] --c requested: ignoring nnUNet_interactive_pretrained, "
                "the checkpoint on disk wins")
            self.pretrained_weights_file = None
        if not already and self.pretrained_weights_file:
            from nnunetv2.run.load_pretrained_weights import load_pretrained_weights
            self.print_to_log_file(
                f"[interactive] loading pretrained weights from {self.pretrained_weights_file}")
            load_pretrained_weights(self.network, self.pretrained_weights_file, verbose=False)

    # ------------------------------------------------------------------
    # loss: DC + CE, optionally without the Dice smoothing term
    # ------------------------------------------------------------------
    def _set_batch_size_and_oversample(self):
        super()._set_batch_size_and_oversample()
        bs = _env_int("nnUNet_interactive_batch_size", 0)
        if bs > 0:
            self.batch_size = bs

    def _build_loss(self):
        if not getattr(self, "no_smooth_dice", True):
            return super()._build_loss()

        smooth = 0.0
        if self.label_manager.has_regions:
            loss = DC_and_BCE_loss({},
                                   {"batch_dice": self.configuration_manager.batch_dice,
                                    "do_bg": True, "smooth": smooth, "ddp": self.is_ddp},
                                   use_ignore_label=self.label_manager.ignore_label is not None,
                                   dice_class=MemoryEfficientSoftDiceLoss)
        else:
            loss = DC_and_CE_loss({"batch_dice": self.configuration_manager.batch_dice,
                                   "smooth": smooth, "do_bg": False, "ddp": self.is_ddp}, {},
                                  weight_ce=1, weight_dice=1,
                                  ignore_label=self.label_manager.ignore_label,
                                  dice_class=MemoryEfficientSoftDiceLoss)
        if self._do_i_compile():
            loss.dc = torch.compile(loss.dc)

        if self.enable_deep_supervision:
            deep_supervision_scales = self._get_deep_supervision_scales()
            weights = np.array([1 / (2 ** i) for i in range(len(deep_supervision_scales))])
            weights[-1] = 1e-6 if (self.is_ddp and not self._do_i_compile()) else 0
            weights = weights / weights.sum()
            loss = DeepSupervisionWrapper(loss, weights)
        return loss

    # ------------------------------------------------------------------
    # transform pipeline
    # ------------------------------------------------------------------
    @classmethod
    def build_interaction_transform(cls) -> InteractionSimulationTransform:
        k_probs = os.environ.get("nnUNet_interactive_k_probs")
        kp = tuple(float(x) for x in k_probs.split(",")) if k_probs else cls.K_PROBS
        return InteractionSimulationTransform(
            k_probs=kp,
            radius=_env_float("nnUNet_interactive_radius", cls.GUIDANCE_RADIUS),
            slice_axis=_env_int("nnUNet_interactive_slice_axis", cls.SLICE_AXIS),
            p_perturb=_env_float("nnUNet_interactive_p_perturb", cls.P_PERTURB),
            p_independent_scribbles=_env_float("nnUNet_interactive_p_independent",
                                               cls.P_INDEPENDENT_SCRIBBLES),
            force_empty_prev_if_no_scribbles=cls.FORCE_EMPTY_PREV_IF_NO_SCRIBBLES,
            pet_channel=cls.PET_CHANNEL,
            n_image_channels=_env_int("nnUNet_interactive_n_image_channels",
                                      cls.N_IMAGE_CHANNELS),
            corruption=CorruptionConfig(),
        )

    @classmethod
    def _wrap_pipeline(cls, compose: BasicTransform) -> BasicTransform:
        """Insert our two transforms into an nnU-Net pipeline.

        KeepFirstChannelsTransform goes at the front; the interaction simulation goes
        just before DownsampleSegForDSTransform, which turns `segmentation` into a
        list and would hide the full-resolution label we need.
        """
        assert isinstance(compose, ComposeTransforms), \
            f"expected ComposeTransforms, got {type(compose)}"
        idx = len(compose.transforms)
        for i, tr in enumerate(compose.transforms):
            if isinstance(tr, DownsampleSegForDSTransform):
                idx = i
                break
        compose.transforms.insert(idx, cls.build_interaction_transform())
        compose.transforms.insert(0, KeepFirstChannelsTransform(
            _env_int("nnUNet_interactive_n_image_channels", cls.N_IMAGE_CHANNELS)))
        return compose

    @classmethod
    def get_training_transforms(cls,
                                patch_size: Union[np.ndarray, Tuple[int]],
                                rotation_for_DA: RandomScalar,
                                deep_supervision_scales: Union[List, Tuple, None],
                                mirror_axes: Tuple[int, ...],
                                do_dummy_2d_data_aug: bool,
                                use_mask_for_norm: List[bool] = None,
                                is_cascaded: bool = False,
                                foreground_labels: Union[Tuple[int, ...], List[int]] = None,
                                regions: List[Union[List[int], Tuple[int, ...], int]] = None,
                                ignore_label: int = None) -> BasicTransform:
        # nnU-Net normalizes `use_mask_for_norm` against the stored channels; our
        # generated channels never need masking, so pass it through untouched.
        base = nnUNetTrainer.get_training_transforms(
            patch_size, rotation_for_DA, deep_supervision_scales, mirror_axes,
            do_dummy_2d_data_aug, use_mask_for_norm=use_mask_for_norm,
            is_cascaded=is_cascaded, foreground_labels=foreground_labels,
            regions=regions, ignore_label=ignore_label)
        return cls._wrap_pipeline(base)

    @classmethod
    def get_validation_transforms(cls,
                                  deep_supervision_scales: Union[List, Tuple, None],
                                  is_cascaded: bool = False,
                                  foreground_labels: Union[Tuple[int, ...], List[int]] = None,
                                  regions: List[Union[List[int], Tuple[int, ...], int]] = None,
                                  ignore_label: int = None) -> BasicTransform:
        base = nnUNetTrainer.get_validation_transforms(
            deep_supervision_scales, is_cascaded=is_cascaded,
            foreground_labels=foreground_labels, regions=regions, ignore_label=ignore_label)
        # validation patches need the same 5 channels or the network cannot run
        return cls._wrap_pipeline(base)

    # ------------------------------------------------------------------
    # final validation
    # ------------------------------------------------------------------
    def perform_actual_validation(self, save_probabilities: bool = False):
        """Skip nnU-Net's end-of-training validation by default.

        It runs sliding-window inference over the stored preprocessed cases, which
        have 2 channels; the 5-channel network cannot consume them. The evaluation
        that matters is the interactive loop in src/interactive_eval.py.
        """
        if self.skip_final_validation:
            self.print_to_log_file(
                "[interactive] skipping nnU-Net's final validation: the stored preprocessed data "
                "has 2 channels while the network expects 5, and the meaningful metric is the "
                "official interactive AUC (run src/interactive_eval.py instead). "
                "Set nnUNet_interactive_skip_final_val=0 to force it.")
            return
        return super().perform_actual_validation(save_probabilities)


# ---------------------------------------------------------------------------
# variants
# ---------------------------------------------------------------------------

class nnUNetTrainer_Interactive_2epochs(nnUNetTrainer_Interactive):
    """Smoke-test variant used by the synthetic end-to-end run."""
    NUM_EPOCHS = 2


class nnUNetTrainer_Interactive_50epochs(nnUNetTrainer_Interactive):
    NUM_EPOCHS = 50


class nnUNetTrainer_Interactive_150epochs(nnUNetTrainer_Interactive):
    NUM_EPOCHS = 150


class nnUNetTrainer_Interactive_100epochs(nnUNetTrainer_Interactive):
    NUM_EPOCHS = 100


class nnUNetTrainer_Interactive_250epochs(nnUNetTrainer_Interactive):
    NUM_EPOCHS = 250


# ---------------------------------------------------------------------------
# B6: k-reweighted continuation
# ---------------------------------------------------------------------------

class nnUNetTrainer_InteractiveB6(nnUNetTrainer_Interactive):
    """Continuation run that re-weights the interaction distribution.

    K_PROBS shifts mass to k = 0 and k = 1, the states that correspond to iterations
    0 and 1 of the evaluation loop, and P_INDEPENDENT_SCRIBBLES rises to 0.35.
    Started from the previous checkpoint via nnUNet_interactive_pretrained with a
    fresh optimizer and schedule, not with --c, so it gets its own results folder.
    """

    K_PROBS: Tuple[float, ...] = (0.36, 0.24, 0.16, 0.11, 0.08, 0.05)
    P_INDEPENDENT_SCRIBBLES: float = 0.35
    NUM_EPOCHS: int = 120
    INITIAL_LR: float = 5e-4

    # ------------------------------------------------------------------
    # continuation weight loading
    # ------------------------------------------------------------------
    def initialize(self):
        """Load every tensor of the source checkpoint, segmentation heads included.

        nnU-Net's load_pretrained_weights skips `.seg_layers.` keys, which is right
        for a foreign pretraining but wrong for a continuation of the same
        architecture on the same labels. Falls back to that loader if the strict
        load fails.
        """
        already = self.was_initialized
        wanted = self.pretrained_weights_file
        self.pretrained_weights_file = None      # keep the base class's loader out
        super().initialize()
        self.pretrained_weights_file = wanted

        if self._continue_training and wanted:
            self.print_to_log_file(
                "[B6] --c requested: ignoring nnUNet_interactive_pretrained, "
                "the checkpoint on disk wins")
            self.pretrained_weights_file = None
            return
        if already or not wanted:
            return

        from torch._dynamo import OptimizedModule
        from torch.nn.parallel import DistributedDataParallel as DDP

        ck = torch.load(wanted, map_location="cpu", weights_only=False)
        sd = ck["network_weights"]
        mod = self.network
        if isinstance(mod, DDP):
            mod = mod.module
        if isinstance(mod, OptimizedModule):
            mod = mod._orig_mod
        n_seg = sum(1 for k in sd if ".seg_layers." in k)
        try:
            mod.load_state_dict(sd, strict=True)
            self.print_to_log_file(
                f"[B6] continuation: loaded {len(sd)} tensors from {wanted} "
                f"(trainer={ck.get('trainer_name')}, epoch={ck.get('current_epoch')}), "
                f"{n_seg} of them segmentation-head tensors; optimizer state dropped")
        except Exception as exc:
            self.print_to_log_file(
                f"[B6] strict load failed ({exc}); falling back to nnU-Net's "
                "load_pretrained_weights, which skips the segmentation heads")
            from nnunetv2.run.load_pretrained_weights import load_pretrained_weights
            load_pretrained_weights(self.network, wanted, verbose=False)
