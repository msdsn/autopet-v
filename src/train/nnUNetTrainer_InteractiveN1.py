"""N1 -- presence-prior gate on the B10 recipe.

Continues ``nnUNetTrainer_InteractiveV2_negfp`` (B10) onto the ``PresenceGateUNet``
of ``train.networks_n1``: the same interaction distribution, the same lesion-free
false-positive term, the same 120 epochs at lr 5e-4, plus

* a zero-initialised ``Conv3d(C, 1, 1)`` on encoder stage 3 whose coarse log-odds map
  is added to the foreground logit at full resolution and at every deep-supervision
  scale, and
* an auxiliary ``BCEWithLogits`` of that map against the max-pooled label -- lesion
  presence per cell -- with weight ``N1_AUX_W`` (0.5) on top of the ordinary loss.

The auxiliary term is the second reason for the block: on a label-empty patch the
configured Dice+CE has exactly zero Dice gradient and 52.6 % of training patches are
label-empty, so the BCE is the only *localised* signal that state produces.

``pos_weight`` is not a guess: it is the positive-cell fraction measured on the
training sampler by ``train.measure_n1_prior`` and passed in as
``N1_AUX_POS_WEIGHT``; the value actually used is logged at startup.
"""

from __future__ import annotations

import os

import torch
import torch.nn.functional as F
from torch import nn

try:  # package import (src/train is a package)
    from .identity_gate import SourceIdentityGateMixin
    from .nnUNetTrainer_Interactive import _env_float
    from .nnUNetTrainer_InteractiveArch import nnUNetTrainer_InteractiveArch
except ImportError:  # flat import (folder on sys.path, e.g. nnUNet_extTrainer)
    from identity_gate import SourceIdentityGateMixin  # type: ignore
    from nnUNetTrainer_Interactive import _env_float  # type: ignore
    from nnUNetTrainer_InteractiveArch import nnUNetTrainer_InteractiveArch  # type: ignore

__all__ = ["PresenceGateAuxLoss", "nnUNetTrainer_InteractiveN1",
           "nnUNetTrainer_InteractiveN1_noaux", "nnUNetTrainer_InteractiveN1_2epochs"]


class PresenceGateAuxLoss(nn.Module):
    """``inner(seg_outputs, target) + w * BCE(gate, maxpool(label))``.

    ``PresenceGateUNet`` appends its coarse map after the segmentation outputs, so a
    training output list is one longer than the target list. The map is stripped here
    and the rest is handed to the unmodified compound loss.
    """

    def __init__(self, inner: nn.Module, weight: float = 0.5, pos_weight: float = 1.0):
        super().__init__()
        self.inner = inner
        self.weight = float(weight)
        self.register_buffer("pos_weight", torch.tensor(float(pos_weight)))
        self.last_terms: dict = {}

    def forward(self, output, target):
        gate = None
        if isinstance(output, (list, tuple)) and isinstance(target, (list, tuple)) \
                and len(output) == len(target) + 1:
            gate = output[-1]
            output = list(output[:-1])
        loss = self.inner(output, target)
        if gate is None or self.weight == 0.0:
            return loss

        t = target[0] if isinstance(target, (list, tuple)) else target
        if t.ndim == gate.ndim - 1:
            t = t[:, None]
        fg = (t[:, :1] > 0).to(torch.float32)
        cells = F.adaptive_max_pool3d(fg, gate.shape[2:])
        aux = F.binary_cross_entropy_with_logits(
            gate.float(), cells, pos_weight=self.pos_weight.to(gate.device))
        self.last_terms = {"presence_bce": float(aux.detach()),
                           "positive_cells": float(cells.mean().detach())}
        return loss + self.weight * aux


class nnUNetTrainer_InteractiveN1(SourceIdentityGateMixin, nnUNetTrainer_InteractiveArch):
    """B10 + the presence-prior gate. Run with ``-p nnUNetPlans_n1``."""

    NEW_PARAM_PREFIXES = ("presence_gate.",)
    AUX_WEIGHT: float = 0.5
    #: measured on the training sampler by train.measure_n1_prior; override with
    #: N1_AUX_POS_WEIGHT rather than editing this
    AUX_POS_WEIGHT: float = 1.0

    def _build_loss(self):
        inner = super()._build_loss()
        w = _env_float("N1_AUX_W", self.AUX_WEIGHT)
        pw = _env_float("N1_AUX_POS_WEIGHT", self.AUX_POS_WEIGHT)
        if os.environ.get("N1_AUX_POS_WEIGHT") in (None, ""):
            self.print_to_log_file(
                "[N1] WARNING: N1_AUX_POS_WEIGHT is not set, using the class default "
                f"{pw} -- run train.measure_n1_prior and pass the measured value")
        self.print_to_log_file(
            f"[N1] presence-prior gate: aux BCE weight={w}, pos_weight={pw} "
            f"(from the measured positive-cell fraction)")
        if w == 0.0:
            self.print_to_log_file(
                "[N1] aux weight 0: the block is present but unsupervised -- this is "
                "the falsification control, not the N1 row")
        return PresenceGateAuxLoss(inner, weight=w, pos_weight=pw)


class nnUNetTrainer_InteractiveN1_noaux(nnUNetTrainer_InteractiveN1):
    """Falsification control: the same block, no auxiliary supervision."""
    AUX_WEIGHT: float = 0.0


class nnUNetTrainer_InteractiveN1_2epochs(nnUNetTrainer_InteractiveN1):
    """Smoke-test variant."""
    NUM_EPOCHS = 2
