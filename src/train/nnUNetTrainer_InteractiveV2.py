"""Interactive trainer v2: two optional loss terms on top of the B6 fine-tune.

A lesion-free patch penalty (soft false-positive volume where the label is empty)
and an instance-wise Dice that weights every connected component equally. Both are
off by default; the variants at the bottom switch them on, one per ablation.
Everything else is inherited from nnUNetTrainer_InteractiveB6 unchanged.
"""

from __future__ import annotations

import os
from typing import Optional

import numpy as np
import torch
from torch import nn

try:  # package import (src/train is a package)
    from .nnUNetTrainer_Interactive import (nnUNetTrainer_Interactive,  # noqa: F401
                                            nnUNetTrainer_InteractiveB6,
                                            _env_bool, _env_float, _env_int)
except ImportError:  # flat import (folder on sys.path, e.g. nnUNet_extTrainer)
    from nnUNetTrainer_Interactive import (nnUNetTrainer_Interactive,  # type: ignore # noqa: F401
                                           nnUNetTrainer_InteractiveB6,
                                           _env_bool, _env_float, _env_int)


__all__ = [
    "LesionFreeFPLoss",
    "InstanceDiceLoss",
    "InteractiveV2Loss",
    "nnUNetTrainer_InteractiveV2",
    "nnUNetTrainer_InteractiveV2_negfp",
    "nnUNetTrainer_InteractiveV2_blob",
    "nnUNetTrainer_InteractiveV2_both",
]


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _fg_prob(logits: torch.Tensor) -> torch.Tensor:
    """Foreground probability of a (b, C, ...) logit tensor, as float32 (b, ...).

    C == 1 is a sigmoid (region) head, C > 1 a softmax with class 0 = background.
    The two-class case is written as sigmoid(l1 - l0): same value, but it avoids
    materialising a (b, C, ...) softmax and its backward buffer.
    """
    c = logits.shape[1]
    if c == 1:
        return torch.sigmoid(logits[:, 0].float())
    if c == 2:
        return torch.sigmoid((logits[:, 1] - logits[:, 0]).float())
    return torch.softmax(logits.float(), 1)[:, 1:].sum(1)


def _binary_target(target: torch.Tensor) -> torch.Tensor:
    """(b, 1, ...) or (b, ...) label tensor -> boolean (b, ...) foreground mask.

    Negative values (nnU-Net's ignore label before RemoveLabelTansform) are not
    foreground.
    """
    t = target
    if t.ndim > 1 and t.shape[1] == 1:
        t = t[:, 0]
    return t > 0


# ---------------------------------------------------------------------------
# A. lesion-free patch penalty
# ---------------------------------------------------------------------------

class LesionFreeFPLoss(nn.Module):
    """Soft false-positive-volume penalty on patches whose label is empty.

    s / (s + smooth_voxels), with s the summed foreground probability over the
    patch; 50 voxels is ~0.6 mL at the plans spacing (3.0 x 2.036 x 2.036 mm).
    all_patches extends it to the empty part of patches that do contain lesions.
    """

    def __init__(self, smooth_voxels: float = 50.0, all_patches: bool = False):
        super().__init__()
        self.smooth_voxels = float(smooth_voxels)
        self.all_patches = bool(all_patches)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        p = _fg_prob(logits)                      # (b, ...)
        fg = _binary_target(target)               # (b, ...)
        b = p.shape[0]
        pf = p.reshape(b, -1)
        ff = fg.reshape(b, -1)
        if self.all_patches:
            s = (pf * (~ff)).sum(1)
            keep = torch.ones(b, dtype=torch.bool, device=p.device)
        else:
            keep = ~ff.any(1)
            s = pf.sum(1)
        if not bool(keep.any()):
            # keep the term in the graph so that `loss.backward()` never sees a
            # detached constant, but contribute exactly zero
            return p.sum() * 0.0
        s = s[keep]
        return (s / (s + self.smooth_voxels)).mean()


# ---------------------------------------------------------------------------
# B. instance-wise (blob) Dice
# ---------------------------------------------------------------------------

class InstanceDiceLoss(nn.Module):
    """Per-connected-component soft Dice, other components masked out.

    1 - mean_i dice_i over the batch samples that contain at least one component;
    empty samples contribute nothing (LesionFreeFPLoss covers that state). A small
    lesion and a large one count equally. connectivity 18 matches the challenge
    metric; smallest_k > 0 averages over the k smallest components only.
    """

    def __init__(self, smooth: float = 1.0, connectivity: int = 18,
                 smallest_k: int = 0, min_voxels: int = 1):
        super().__init__()
        self.smooth = float(smooth)
        self.connectivity = int(connectivity)
        self.smallest_k = int(smallest_k)
        self.min_voxels = int(min_voxels)
        self._cc3d = None

    def _label(self, mask_np: np.ndarray):
        if self._cc3d is None:
            import cc3d  # local import: keeps module import cheap and safe
            self._cc3d = cc3d
        return self._cc3d.connected_components(mask_np, connectivity=self.connectivity)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        p = _fg_prob(logits)                      # (b, ...)
        fg = _binary_target(target)               # (b, ...)
        b = p.shape[0]
        terms = []
        for i in range(b):
            m = fg[i]
            if not bool(m.any()):
                continue
            mask_np = np.ascontiguousarray(m.detach().to("cpu").numpy().astype(np.uint8))
            inst = self._label(mask_np)
            n = int(inst.max())
            if n == 0:
                continue
            # Only the foreground voxels carry an instance id, and they are a
            # small fraction of a whole-body patch.  Moving their linear indices
            # instead of the full id map keeps the host->device transfer at
            # O(lesion volume) rather than O(patch volume).
            flat_inst = inst.reshape(-1)
            sel = np.flatnonzero(flat_inst)
            lab = torch.as_tensor(flat_inst[sel].astype(np.int64), device=p.device)
            sel = torch.as_tensor(sel.astype(np.int64), device=p.device)
            pf = p[i].reshape(-1)
            pf_fg = pf[sel]
            counts = torch.bincount(lab, minlength=n + 1).to(pf.dtype)
            inter = torch.zeros(n + 1, dtype=pf.dtype,
                                device=pf.device).index_add(0, lab, pf_fg)
            counts = counts[1:]
            inter = inter[1:]
            total = pf.sum()
            # every voxel of every other component is masked out of the
            # prediction, which is the same as removing its probability mass
            pred_sum = total - (inter.sum() - inter)
            dice = (2.0 * inter + self.smooth) / (pred_sum + counts + self.smooth)
            keep = counts >= self.min_voxels
            if not bool(keep.any()):
                continue
            dice = dice[keep]
            if self.smallest_k > 0 and dice.numel() > self.smallest_k:
                sel = torch.topk(-counts[keep], self.smallest_k).indices
                dice = dice[sel]
            terms.append(1.0 - dice.mean())
        if not terms:
            return p.sum() * 0.0
        return torch.stack(terms).mean()


# ---------------------------------------------------------------------------
# the compound loss
# ---------------------------------------------------------------------------

class InteractiveV2Loss(nn.Module):
    """base(output, target) + w_lesion_free * lf + w_blob * blob.

    base is whatever _build_loss returned, normally a DeepSupervisionWrapper around
    DC+CE. The extra terms see the full-resolution head only, so the deep
    supervision pyramid keeps the unmodified compound loss.
    """

    def __init__(self, base: nn.Module,
                 lesion_free: Optional[nn.Module] = None, w_lesion_free: float = 0.0,
                 blob: Optional[nn.Module] = None, w_blob: float = 0.0):
        super().__init__()
        self.base = base
        self.lesion_free = lesion_free
        self.blob = blob
        self.w_lesion_free = float(w_lesion_free)
        self.w_blob = float(w_blob)
        self.last_terms: dict = {}

    @staticmethod
    def _head(x):
        return x[0] if isinstance(x, (list, tuple)) else x

    def forward(self, output, target):
        loss = self.base(output, target)
        o0, t0 = self._head(output), self._head(target)
        terms = {"base": float(loss.detach())}
        if self.lesion_free is not None and self.w_lesion_free != 0.0:
            lf = self.lesion_free(o0, t0)
            terms["lesion_free"] = float(lf.detach())
            loss = loss + self.w_lesion_free * lf
        if self.blob is not None and self.w_blob != 0.0:
            bl = self.blob(o0, t0)
            terms["blob"] = float(bl.detach())
            loss = loss + self.w_blob * bl
        self.last_terms = terms
        return loss


# ---------------------------------------------------------------------------
# trainer
# ---------------------------------------------------------------------------

class nnUNetTrainer_InteractiveV2(nnUNetTrainer_InteractiveB6):
    """B6 plus two switchable loss terms; with both weights 0 it is B6.

    The class attributes below are the defaults; each is overridden by the
    nnUNet_v2_* environment variable read in _build_loss.
    """

    W_LESION_FREE: float = 0.0
    LESION_FREE_SMOOTH_VOXELS: float = 50.0
    LESION_FREE_ALL_PATCHES: bool = False

    W_BLOB: float = 0.0
    BLOB_SMOOTH: float = 1.0
    BLOB_CONNECTIVITY: int = 18
    BLOB_SMALLEST_K: int = 0
    BLOB_MIN_VOXELS: int = 1

    def _build_loss(self):
        base = super()._build_loss()

        w_lf = _env_float("nnUNet_v2_lesionfree_weight", self.W_LESION_FREE)
        w_bl = _env_float("nnUNet_v2_blob_weight", self.W_BLOB)
        if w_lf == 0.0 and w_bl == 0.0:
            self.print_to_log_file("[interactiveV2] both extra loss terms are off "
                                   "-> loss is identical to nnUNetTrainer_InteractiveB6")
            return base

        lf = LesionFreeFPLoss(
            smooth_voxels=_env_float("nnUNet_v2_lesionfree_smooth_vox",
                                     self.LESION_FREE_SMOOTH_VOXELS),
            all_patches=_env_bool("nnUNet_v2_lesionfree_all_patches",
                                  self.LESION_FREE_ALL_PATCHES),
        ) if w_lf != 0.0 else None
        bl = InstanceDiceLoss(
            smooth=_env_float("nnUNet_v2_blob_smooth", self.BLOB_SMOOTH),
            connectivity=_env_int("nnUNet_v2_blob_connectivity", self.BLOB_CONNECTIVITY),
            smallest_k=_env_int("nnUNet_v2_blob_smallest_k", self.BLOB_SMALLEST_K),
            min_voxels=_env_int("nnUNet_v2_blob_min_voxels", self.BLOB_MIN_VOXELS),
        ) if w_bl != 0.0 else None

        self.print_to_log_file(
            f"[interactiveV2] w_lesion_free={w_lf} "
            f"(smooth_vox={getattr(lf, 'smooth_voxels', None)}, "
            f"all_patches={getattr(lf, 'all_patches', None)}) "
            f"w_blob={w_bl} (smooth={getattr(bl, 'smooth', None)}, "
            f"conn={getattr(bl, 'connectivity', None)}, "
            f"smallest_k={getattr(bl, 'smallest_k', None)}, "
            f"min_voxels={getattr(bl, 'min_voxels', None)})")
        return InteractiveV2Loss(base, lesion_free=lf, w_lesion_free=w_lf,
                                 blob=bl, w_blob=w_bl)


# ---------------------------------------------------------------------------
# ablation variants (each gets its own nnU-Net results folder)
# ---------------------------------------------------------------------------

class nnUNetTrainer_InteractiveV2_negfp(nnUNetTrainer_InteractiveV2):
    """B6 + lesion-free patch penalty (run b10)."""
    W_LESION_FREE: float = 1.0
    W_BLOB: float = 0.0


class nnUNetTrainer_InteractiveV2_blob(nnUNetTrainer_InteractiveV2):
    """B6 + instance-wise (blob) Dice (run b11)."""
    W_LESION_FREE: float = 0.0
    W_BLOB: float = 1.0


class nnUNetTrainer_InteractiveV2_both(nnUNetTrainer_InteractiveV2):
    """B6 + both terms (run b12, the shipped model)."""
    W_LESION_FREE: float = 1.0
    W_BLOB: float = 1.0
