"""Training-time simulation of the three interactive input channels.

InteractionSimulationTransform replaces everything after CT/PET with ch2 (tumor
guidance, clipped EDT), ch3 (background guidance) and ch4 (previous prediction),
giving the 5-channel network input. The guidance is regenerated from the label
every time a patch is drawn, never cached.
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import numpy as np
import torch

from batchgeneratorsv2.transforms.base.basic_transform import BasicTransform, ImageOnlyTransform

try:  # package import
    from .corrupt import CorruptionConfig, make_fake_prev_prediction
    from .guidance import stamp_clipped_edt
    from .scribble_sim import STRATEGIES, simulate_interaction_sequence
except ImportError:  # flat import (folder on sys.path, e.g. nnUNet_extTrainer)
    from corrupt import CorruptionConfig, make_fake_prev_prediction  # type: ignore
    from guidance import stamp_clipped_edt  # type: ignore
    from scribble_sim import STRATEGIES, simulate_interaction_sequence  # type: ignore

__all__ = ["InteractionSimulationTransform", "KeepFirstChannelsTransform"]


class KeepFirstChannelsTransform(ImageOnlyTransform):
    """Drop the stored guidance placeholder channels before spatial augmentation.

    The store keeps 4 channels (CT, PET, all-zero fg, all-zero bg) to match the
    baseline preprocessing. Channels 2-3 get overwritten later anyway, so dropping
    them at position 0 saves SpatialTransform a cubic-spline pass over each.
    """

    def __init__(self, n_channels: int = 2):
        super().__init__()
        self.n_channels = int(n_channels)

    def _apply_to_image(self, img, **params):
        if img.shape[0] > self.n_channels:
            return img[:self.n_channels]
        return img

    def __repr__(self):
        return f"KeepFirstChannelsTransform(n_channels={self.n_channels})"


class InteractionSimulationTransform(BasicTransform):
    """Append (tumor-guidance, background-guidance, previous-prediction) channels.

    Must sit after SpatialTransform/RemoveLabelTansform and before
    DownsampleSegForDSTransform: the guidance has to land on exactly the label grid
    used in the loss, and ch4 must stay a crisp binary mask (a spline warp rings it).
    """

    def __init__(self,
                 # k in {0..5}: 6 evaluation iterations, so at most 5 accumulated scribbles
                 k_probs: Sequence[float] = (0.28, 0.22, 0.18, 0.14, 0.10, 0.08),
                 radius: float = 10.0,              # clipped-EDT radius, voxels
                 slice_axis: int = 0,               # axial axis at the plans spacing 3.0 x 2.036 x 2.036
                 strategies: Sequence[str] = STRATEGIES,
                 p_perturb: float = 0.2,
                 # k == 0 mirrors iteration 0, where there is no previous prediction either
                 force_empty_prev_if_no_scribbles: bool = True,
                 pet_channel: int = 1,
                 n_image_channels: int = 2,
                 p_independent_scribbles: float = 0.25,
                 corruption: Optional[CorruptionConfig] = None,
                 spacing: Optional[Tuple[float, ...]] = None,
                 seed: Optional[int] = None,
                 verbose_timing: bool = False):
        super().__init__()
        p = np.asarray(k_probs, dtype=np.float64)
        self.k_values = np.arange(len(p))
        self.k_probs = p / p.sum()
        self.radius = float(radius)
        self.slice_axis = int(slice_axis)
        self.strategies = tuple(strategies)
        self.p_perturb = float(p_perturb)
        self.force_empty_prev_if_no_scribbles = bool(force_empty_prev_if_no_scribbles)
        self.pet_channel = int(pet_channel)
        self.n_image_channels = int(n_image_channels)
        self.p_independent_scribbles = float(p_independent_scribbles)
        self.corruption = corruption or CorruptionConfig()
        self.spacing = tuple(spacing) if spacing is not None else None
        self._seed = seed
        self._rng: Optional[np.random.Generator] = None
        self.verbose_timing = verbose_timing
        self.last_info: dict = {}

    # -- rng is created lazily so every dataloader worker gets its own stream --
    @property
    def rng(self) -> np.random.Generator:
        if self._rng is None:
            if self._seed is not None:
                self._rng = np.random.default_rng(self._seed)
            else:
                import os
                self._rng = np.random.default_rng(
                    [int(np.random.randint(0, 2 ** 31 - 1)), os.getpid()])
        return self._rng

    # ------------------------------------------------------------------
    def apply(self, data_dict, **params):
        image = data_dict.get("image")
        seg = data_dict.get("segmentation")
        if image is None or seg is None:
            return data_dict

        seg_np = seg[0].numpy() if isinstance(seg, torch.Tensor) else np.asarray(seg[0])
        label = (seg_np > 0).astype(np.uint8)

        pet = None
        if image.shape[0] > self.pet_channel:
            pet = image[self.pet_channel]
            pet = pet.numpy() if isinstance(pet, torch.Tensor) else np.asarray(pet)

        rng = self.rng
        k = int(rng.choice(self.k_values, p=self.k_probs))
        independent = False

        if k == 0 and self.force_empty_prev_if_no_scribbles:
            prev = np.zeros(label.shape, dtype=np.uint8)
            fg_coords: list = []
            bg_coords: list = []
            n_iters = 0
        else:
            prev = make_fake_prev_prediction(label, rng, pet=pet, cfg=self.corruption)
            # The clinician scribbles were collected against a different model's errors,
            # so some arrive already satisfied by `prev`: a scribble is a constraint to
            # maintain, not only a change signal.
            independent = rng.random() < self.p_independent_scribbles
            src = (make_fake_prev_prediction(label, rng, pet=pet, cfg=self.corruption)
                   if independent else prev)
            inter = simulate_interaction_sequence(
                label, src, k, rng,
                slice_axis=self.slice_axis,
                strategies=self.strategies,
                p_perturb=self.p_perturb,
            )
            fg_coords, bg_coords, n_iters = inter.fg_coords, inter.bg_coords, inter.n_iters

        extra = np.zeros((3, *label.shape), dtype=np.float32)
        stamp_clipped_edt(extra[0], fg_coords, self.radius, self.spacing)
        stamp_clipped_edt(extra[1], bg_coords, self.radius, self.spacing)
        extra[2] = prev

        # keep only the real image channels; whatever the store put after them
        # (all-zero guidance placeholders) is replaced by what we just simulated
        n = min(self.n_image_channels, image.shape[0])
        if isinstance(image, torch.Tensor):
            data_dict["image"] = torch.cat((image[:n], torch.from_numpy(extra).to(image.dtype)), dim=0)
        else:
            data_dict["image"] = np.concatenate((np.asarray(image)[:n], extra), axis=0)

        self.last_info = {"k": k, "n_iters": n_iters,
                          "n_fg": len(fg_coords), "n_bg": len(bg_coords),
                          "independent": bool(independent),
                          "prev_vox": int(prev.sum()), "label_vox": int(label.sum())}
        return data_dict

    def __repr__(self):
        return (f"InteractionSimulationTransform(k_probs={list(np.round(self.k_probs, 3))}, "
                f"radius={self.radius}, slice_axis={self.slice_axis}, "
                f"strategies={self.strategies}, p_perturb={self.p_perturb}, "
                f"p_independent_scribbles={self.p_independent_scribbles})")
