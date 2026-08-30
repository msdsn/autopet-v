"""RE-N: the RE (ResEncL) backbone carrying N1's presence-gate head.

One block, ported one variable at a time. The backbone is
``networks_re.ResEncInteractiveUNet`` unchanged -- LesionTracer's ResEncL with five
input channels and the ``pet_renorm`` remap -- and the head is exactly the one
``networks_n1.PresenceGateUNet`` carries on the PlainConvUNet: a single
``Conv3d(C, 1, 1)`` whose coarse per-cell log-odds map is trilinearly upsampled and
added to the **foreground logit** of the final output and of every deep-supervision
output at its own scale.

## Why the tap is stage 3

N1's cell on the PlainConvUNet is 8x8x8 voxels = 6.37 mL, and its auxiliary
``pos_weight`` was measured at that density. The ResEncL's stage-3 skip is 24^3 for
the 192^3 patch, which at the same spacing is 8x8x8 voxels = **6.370 mL** -- the same
physical cell. Tapping stage 3 is therefore a one-variable port of a block whose
hyper-parameter is already measured; tapping stage 2 (4^3 = 0.80 mL) would change the
cell by 8x and require a second new number.

| stage | grid at 192^3 | C | cell | |
|---|---|---:|---|---|
| 2 | 48^3 | 128 | 4^3 = 0.796 mL | not this one |
| **3** | **24^3** | **256** | **8^3 = 6.370 mL** | **gate tap, matches N1** |
| 4 | 12^3 | 320 | 16^3 = 50.96 mL | too coarse |

## What is reused rather than re-derived

``_zero_and_freeze``, ``_FREEZE_FLAG`` and ``PresenceGateUNet._fuse`` are imported
from ``networks_n1`` and called directly, so the fusion contract is the same code
object that produced the N1 row: the map is added to ``gate_class`` (channel 1, the
tumour logit) only, ``forward`` returns ``fused + [gate]`` so
``DeepSupervisionWrapper`` zips against the target list and never sees the coarse
map, ``validation_step`` still reads ``output[0]``, and with deep supervision off the
return value is the single fused tensor -- the stock inference contract, so the
predictor, the post-processing and ``submission/process.py`` are untouched.

Weight and bias are exactly zero at construction and ``initialize`` skips the tagged
module, so nnU-Net's ``network.apply(network.initialize)`` cannot undo it: epoch 0 is
bit-exact RE. The gradient on the head is non-zero on the first step, because it has
its own auxiliary BCE -- it is not a starved path behind another zero.

Referenced from ``nnUNetPlans_ren.json`` as
``train.networks_ren.ResEncPresenceGateUNet``.
"""

from __future__ import annotations

import torch
from torch import nn

from dynamic_network_architectures.architectures.unet import ResidualEncoderUNet

try:  # package import (src/train is a package)
    from .networks_n1 import _FREEZE_FLAG, _zero_and_freeze, PresenceGateUNet
    from .networks_re import ResEncInteractiveUNet
except ImportError:  # flat import (folder on sys.path, e.g. nnUNet_extTrainer)
    from networks_n1 import _FREEZE_FLAG, _zero_and_freeze, PresenceGateUNet  # type: ignore
    from networks_re import ResEncInteractiveUNet  # type: ignore

__all__ = ["ResEncPresenceGateUNet", "GATE_PREFIX"]

#: the only tensors RE-N adds to the RE state dict
GATE_PREFIX = "gate_head."


class ResEncPresenceGateUNet(ResEncInteractiveUNet):
    """RE with a zero-initialised coarse presence-prior head on encoder stage 3."""

    def __init__(self, *args, gate_stage: int = 3, gate_class: int = 1, **kwargs):
        super().__init__(*args, **kwargs)
        self.gate_stage = int(gate_stage)
        self.gate_class = int(gate_class)
        conv_op = getattr(self.encoder, "conv_op", None) or kwargs.get("conv_op") or nn.Conv3d
        channels = int(self.encoder.output_channels[self.gate_stage])
        self.gate_head = _zero_and_freeze(conv_op(channels, 1, 1, 1, 0, bias=True))

    def _fuse(self, seg: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        # the N1 fusion, called as the same code object rather than re-derived
        return PresenceGateUNet._fuse(self, seg, gate)

    def forward(self, x):
        # the remap happens here exactly once; do not delegate to
        # ResEncInteractiveUNet.forward, which would apply it a second time
        x = self._remap_pet(x)
        skips = self.encoder(x)
        gate = self.gate_head(skips[self.gate_stage])
        out = self.decoder(skips)
        if not isinstance(out, (list, tuple)):
            return self._fuse(out, gate)
        return [self._fuse(o, gate) for o in out] + [gate]

    @staticmethod
    def initialize(module):
        if getattr(module, _FREEZE_FLAG, False):
            return
        ResidualEncoderUNet.initialize(module)
