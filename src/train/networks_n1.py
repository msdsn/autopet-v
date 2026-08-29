"""N1 -- presence-prior gate: a coarse learned log-odds map added to the foreground logit.

``PresenceGateUNet`` is the interactive 5-channel ``PlainConvUNet`` plus a single
``Conv3d(C, 1, 1)`` on one encoder stage. Its output is a coarse per-cell log-odds
map of "is there a lesion in this region", trilinearly upsampled and **added to the
foreground logit** of the final output and of every deep-supervision output at its
own scale.

The head sits on **stage 3** (14x20x16 for the 112x160x128 plans patch, one cell =
8x8x8 voxels = 6.4 mL at the plans spacing), not on the bottleneck: a bottleneck cell
covers 16x32x32 voxels = 204 mL, which cannot resolve the 0.5-3 mL false-positive
components the gate is meant to suppress.

Both the weight and the bias of the head are zero at initialisation, so the added
log-odds is exactly 0 and epoch 0 reproduces the source model bit for bit. The
gradient on that convolution is not zero, so it starts learning on the first step.

With deep supervision on, ``forward`` returns ``[seg_0 ... seg_n, gate]`` -- the
coarse map is appended **after** the segmentation outputs. ``DeepSupervisionWrapper``
zips outputs against targets and therefore ignores it, ``validation_step`` still
reads ``output[0]``, and the trainer's auxiliary BCE picks it off the end. With deep
supervision off (the inference path) the return value is the single fused logit
tensor, exactly the stock contract: nothing downstream of the network changes.

Referenced from ``plans.json`` as ``train.networks_n1.PresenceGateUNet`` (``src`` on
PYTHONPATH), which is how ``pydoc.locate`` in
``nnunetv2.utilities.get_network_from_plans`` resolves a network class.
"""

from __future__ import annotations

from typing import Type, Union

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.dropout import _DropoutNd

from dynamic_network_architectures.architectures.unet import PlainConvUNet
from dynamic_network_architectures.initialization.weight_init import InitWeights_He

__all__ = ["PresenceGateUNet"]

_FREEZE_FLAG = "_autopet_n1_freeze_init"


def _zero_and_freeze(module: nn.Module) -> nn.Module:
    """Zero every parameter of a module and protect it from re-initialisation."""
    for p in module.parameters():
        nn.init.zeros_(p)
    for m in module.modules():
        setattr(m, _FREEZE_FLAG, True)
    return module


class PresenceGateUNet(PlainConvUNet):
    """PlainConvUNet with a zero-initialised coarse presence-prior head.

    ``gate_stage`` indexes the encoder output the head reads (3 = the 8x-downsampled
    stage). ``gate_class`` is the logit channel the map is added to; with the
    two-class softmax head of this dataset that is channel 1, the tumor logit.
    """

    def __init__(self, input_channels: int, n_stages: int, features_per_stage,
                 conv_op: Type[_ConvNd], kernel_sizes, strides, n_conv_per_stage,
                 num_classes: int, n_conv_per_stage_decoder, conv_bias: bool = False,
                 norm_op: Union[None, Type[nn.Module]] = None, norm_op_kwargs: dict = None,
                 dropout_op: Union[None, Type[_DropoutNd]] = None, dropout_op_kwargs: dict = None,
                 nonlin: Union[None, Type[nn.Module]] = None, nonlin_kwargs: dict = None,
                 deep_supervision: bool = False, nonlin_first: bool = False,
                 gate_stage: int = 3, gate_class: int = 1):
        super().__init__(input_channels, n_stages, features_per_stage, conv_op, kernel_sizes,
                         strides, n_conv_per_stage, num_classes, n_conv_per_stage_decoder,
                         conv_bias, norm_op, norm_op_kwargs, dropout_op, dropout_op_kwargs,
                         nonlin, nonlin_kwargs, deep_supervision, nonlin_first)
        self.gate_stage = int(gate_stage)
        self.gate_class = int(gate_class)
        self.num_classes = int(num_classes)
        assert 0 <= self.gate_class < max(1, self.num_classes), \
            f"gate_class {gate_class} outside the {num_classes} output channels"
        channels = int(self.encoder.output_channels[self.gate_stage])
        self.presence_gate = _zero_and_freeze(conv_op(channels, 1, 1, 1, 0, bias=True))

    # ------------------------------------------------------------------
    def _fuse(self, seg: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        """seg + the upsampled gate map, on the foreground logit only."""
        g = F.interpolate(gate, size=seg.shape[2:], mode="trilinear", align_corners=False)
        g = g.to(seg.dtype)
        c = seg.shape[1]
        if c == 1:
            return seg + g
        zeros = torch.zeros_like(g)
        add = torch.cat([g if i == self.gate_class else zeros for i in range(c)], 1)
        return seg + add

    def forward(self, x):
        skips = self.encoder(x)
        gate = self.presence_gate(skips[self.gate_stage])
        out = self.decoder(skips)
        if not isinstance(out, (list, tuple)):
            return self._fuse(out, gate)
        fused = [self._fuse(o, gate) for o in out]
        # the coarse map rides along at the end for the auxiliary BCE; the deep
        # supervision wrapper zips against the target list and never sees it
        return fused + [gate]

    @staticmethod
    def initialize(module):
        if getattr(module, _FREEZE_FLAG, False):
            return
        InitWeights_He(1e-2)(module)
