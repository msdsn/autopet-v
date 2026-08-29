"""B17 -- a trainable 2.5D EVA-02-B encoder fused into the interactive 3D U-Net.

``EVAFusionUNet`` keeps the whole B10 network (5-channel ``PlainConvUNet``, encoder
and decoder) untouched and adds a second, *pretrained* view of the same patch:

1. every axial slice of the input patch is rendered on the GPU into the three
   channels ``eva02_features.py`` uses -- CT in a soft-tissue window, log-SUV PET,
   and a +/-4-slice log-SUV maximum-intensity slab -- so a 2D backbone can see out
   of plane at all;
2. the ``Z`` slices of the whole batch go through ImageNet-pretrained EVA-02-B as
   one ``(B*Z, 3, 224, 182)`` batch (timm, ``dynamic_img_size=True``, so RoPE and
   the absolute position table are resampled to the 16x13 token grid);
3. the resulting token grid is reshaped back into a **volume of tokens**
   ``(B, 768, Z, 16, 13)``, trilinearly resized to an encoder stage's grid and added
   to that stage's skip through a 1x1x1 projection whose weight *and* bias are
   **zero**.

The zero-initialised projections make the fused network numerically identical to
B10 at initialisation, which the trainer asserts (``nnUNetTrainer_InteractiveArch.
_assert_identity_at_init``) before the first optimizer step. Everything the variant
adds lives under the ``eva.`` and ``eva_fuse.`` prefixes.

Design notes that are easy to get wrong
---------------------------------------

**Axis order.** The preprocessed array is ``(axial, y, x)``, so the network input is
``(B, 5, 112, 160, 128)`` and an axial slice is the ``160x128`` in-plane grid at
2.04 mm -- 326 x 261 mm of body. ``224 x 182`` px is that aspect ratio to 1.4 % and
is a whole number of 14-px patches in both directions, giving **16 x 13 = 208**
tokens per slice.

**Inverting the store normalisation.** The network only ever sees *normalised*
channels, so the physical units have to be recovered inside ``forward``. CT is exact
-- ``CTNormalization`` uses the global fingerprint constants. PET is not: the store
z-scores it **per case**, and the per-case ``pet_norm_correction`` is not reachable
from a patch. The rendering therefore uses the cohort medians of that correction
(``mu`` 0.109, ``sd`` 0.625, measured over 120 store cases; the per-case ``sd``
spans 0.44-1.14). This makes the PET channel a fixed monotone function of the
z-score rather than of true SUV -- which is what actually matters, because the
*same* function is applied at training and at inference, and the backbone is
trained through it.

**Gradient budget.** ``patch_embed`` and the first ``eva_freeze_blocks`` blocks run
inside ``torch.no_grad()``. The rendering is a function of the network *input*, not
of any parameter, so nothing downstream needs a gradient through them; skipping the
graph there is what makes a 224-slice EVA batch fit. The remaining blocks are
gradient-checkpointed.

Referenced from ``plans.json`` as ``train.networks_eva.EVAFusionUNet``. ``timm`` is
imported lazily inside the constructor so this module stays importable on a box
without it (nnU-Net imports every module in ``nnUNet_extTrainer``).
"""

from __future__ import annotations

import math
import os
from typing import Sequence, Tuple, Type, Union

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.dropout import _DropoutNd
from torch.utils.checkpoint import checkpoint as _grad_checkpoint

from dynamic_network_architectures.architectures.unet import PlainConvUNet

try:  # package import (src/train is a package)
    from .networks import _freeze_init, _zero_, _init_skipping_frozen
except ImportError:  # flat import (folder on sys.path, e.g. nnUNet_extTrainer)
    from networks import _freeze_init, _zero_, _init_skipping_frozen  # type: ignore

__all__ = ["EVAFusionUNet", "EVA_MODEL_NAME", "render_eva_channels"]


# -- rendering constants, shared with src/train/eva02_features.py ------------
# plans.json foreground_intensity_properties_per_channel["0"] -- CTNormalization
CT_MEAN, CT_STD = 107.73438968591431, 286.34403119451997
CT_WINDOW = (-160.0, 240.0)          # soft tissue
PET_SUV_CLIP = 60.0                  # the log scale saturates well below this
MIP_HALF = 4                         # +/- 4 slices = +/- 12 mm at 3 mm spacing
# cohort medians of pet_norm_correction over 120 store cases; see the module docstring
PET_MU, PET_SD = 0.1088, 0.6249

EVA_MODEL_NAME = "eva02_base_patch14_448.mim_in22k_ft_in22k_in1k"
EVA_PATCH = 14
# OpenAI CLIP statistics -- timm.data.resolve_data_config for this checkpoint
EVA_MEAN = (0.48145466, 0.4578275, 0.40821073)
EVA_STD = (0.26862954, 0.26130258, 0.27577711)


def render_eva_channels(x: torch.Tensor) -> torch.Tensor:
    """Normalised network input ``(B, >=2, Z, Y, X)`` -> ``(B, 3, Z, Y, X)`` in [0, 1].

    Channel 0 CT in a soft-tissue window, 1 log-SUV PET, 2 the log-SUV of a
    +/-``MIP_HALF``-slice maximum-intensity slab. Identical in form to
    ``eva02_features.render_slices``; the PET inverse uses cohort constants because a
    patch carries no per-case correction.
    """
    lo, hi = CT_WINDOW
    ct = ((x[:, 0:1] * CT_STD + CT_MEAN) - lo) / (hi - lo)
    ct = ct.clamp(0.0, 1.0)

    suv = (x[:, 1:2] * PET_SD + PET_MU).clamp(0.0, PET_SUV_CLIP)
    pet = torch.log1p(suv) / math.log1p(PET_SUV_CLIP)
    # the slab MIP runs along the axial axis, which is spatial axis 0 = dim 2
    mip = F.max_pool3d(pet, kernel_size=(2 * MIP_HALF + 1, 1, 1), stride=1,
                       padding=(MIP_HALF, 0, 0))
    return torch.cat((ct, pet, mip), dim=1)


class EVAFusionUNet(PlainConvUNet):
    """B10's ``PlainConvUNet`` with a trainable 2.5D EVA-02-B branch fused into it.

    ``eva_fuse_stages`` are encoder stage indices; at the plans patch stage 4 is
    ``(320, 7, 10, 8)`` and stage 5 ``(320, 7, 5, 4)``. Each gets its own zero-init
    ``1x1x1`` projection from the 768-d token volume, added residually to the skip.
    """

    def __init__(self, input_channels: int, n_stages: int, features_per_stage, conv_op: Type[_ConvNd],
                 kernel_sizes, strides, n_conv_per_stage, num_classes: int, n_conv_per_stage_decoder,
                 conv_bias: bool = False, norm_op: Union[None, Type[nn.Module]] = None,
                 norm_op_kwargs: dict = None, dropout_op: Union[None, Type[_DropoutNd]] = None,
                 dropout_op_kwargs: dict = None, nonlin: Union[None, Type[nn.Module]] = None,
                 nonlin_kwargs: dict = None, deep_supervision: bool = False, nonlin_first: bool = False,
                 eva_model_name: str = EVA_MODEL_NAME, eva_pretrained: bool = True,
                 eva_img_size: Sequence[int] = (224, 182), eva_z_stride: int = 1,
                 eva_freeze_blocks: int = 4, eva_grad_checkpointing: bool = True,
                 eva_fuse_stages: Sequence[int] = (4, 5), eva_chunk: int = 0,
                 eva_drop_path_rate: float = 0.0):
        super().__init__(input_channels, n_stages, features_per_stage, conv_op, kernel_sizes, strides,
                         n_conv_per_stage, num_classes, n_conv_per_stage_decoder, conv_bias, norm_op,
                         norm_op_kwargs, dropout_op, dropout_op_kwargs, nonlin, nonlin_kwargs,
                         deep_supervision, nonlin_first)
        assert conv_op is nn.Conv3d, "the EVA branch renders axial slices, so it needs a 3D network"
        self.eva_img_size = tuple(int(v) for v in eva_img_size)
        assert all(v % EVA_PATCH == 0 for v in self.eva_img_size), \
            f"eva_img_size {self.eva_img_size} must be a multiple of the patch size {EVA_PATCH}"
        self.eva_token_grid = tuple(v // EVA_PATCH for v in self.eva_img_size)
        self.eva_z_stride = max(1, int(eva_z_stride))
        self.eva_freeze_blocks = int(eva_freeze_blocks)
        self.eva_grad_checkpointing = bool(eva_grad_checkpointing)
        self.eva_chunk = int(eva_chunk)
        self.eva_fuse_stages = [int(s) % int(n_stages) for s in eva_fuse_stages]

        self.eva = self._build_eva(eva_model_name, eva_pretrained, eva_drop_path_rate)
        self.eva_dim = int(self.eva.embed_dim)
        self.eva_num_prefix = int(getattr(self.eva, "num_prefix_tokens", 1))

        enc_ch = [int(c) for c in self.encoder.output_channels]
        self.eva_fuse = nn.ModuleList(
            [_zero_(conv_op(self.eva_dim, enc_ch[s], 1, 1, 0, bias=True))
             for s in self.eva_fuse_stages])

        # constants, not state: kept out of the checkpoint so the graft stays strict
        self.register_buffer("eva_img_mean", torch.tensor(EVA_MEAN).view(1, 3, 1, 1),
                             persistent=False)
        self.register_buffer("eva_img_std", torch.tensor(EVA_STD).view(1, 3, 1, 1),
                             persistent=False)

    # -- construction --------------------------------------------------
    def _build_eva(self, name: str, pretrained: bool, drop_path_rate: float) -> nn.Module:
        """Build the timm backbone and freeze its stem and first blocks.

        ``pretrained`` is what makes B17 a pretrained-encoder experiment, but the
        constructor also runs at *inference*, where the weights arrive from our own
        checkpoint and the box may be offline (the challenge container runs with
        ``--network=none``). A failed download is therefore a warning, not an error;
        ``AUTOPET_EVA_PRETRAINED=0`` skips the attempt outright.
        """
        import timm

        env = os.environ.get("AUTOPET_EVA_PRETRAINED")
        if env is not None:
            pretrained = env.lower() in ("1", "true", "t", "yes")
        kwargs = dict(num_classes=0, dynamic_img_size=True, drop_path_rate=float(drop_path_rate))
        try:
            model = timm.create_model(name, pretrained=pretrained, **kwargs)
            if pretrained:
                print(f"[eva] built {name} with pretrained ImageNet-22k/1k weights")
        except Exception as e:                       # offline / no cache
            print(f"[eva] WARNING: pretrained load failed ({type(e).__name__}: {e}); "
                  f"building {name} randomly -- the weights must come from a checkpoint")
            model = timm.create_model(name, pretrained=False, **kwargs)

        n_frozen = self.eva_freeze_blocks
        frozen = [model.patch_embed]
        for attr in ("cls_token", "reg_token", "pos_embed"):
            p = getattr(model, attr, None)
            if isinstance(p, nn.Parameter):
                p.requires_grad_(False)
        frozen += list(model.blocks[:n_frozen])
        for m in frozen:
            for p in m.parameters():
                p.requires_grad_(False)
        # nnU-Net runs network.apply(network.initialize) after construction; without
        # this the pretrained weights would be overwritten with Kaiming noise.
        _freeze_init(model)
        return model

    # -- the 2.5D branch -----------------------------------------------
    def _eva_blocks(self, img: torch.Tensor) -> torch.Tensor:
        """``(N, 3, H, W)`` normalised images -> ``(N, n_tokens, 768)`` patch tokens."""
        m = self.eva
        with torch.no_grad():
            t = m.patch_embed(img)
            t, rope = m._pos_embed(t)
            t = m.norm_pre(t)
            for blk in m.blocks[:self.eva_freeze_blocks]:
                t = blk(t, rope)
        t = t.detach()
        use_ckpt = self.eva_grad_checkpointing and self.training and torch.is_grad_enabled()
        for blk in m.blocks[self.eva_freeze_blocks:]:
            if use_ckpt:
                t = _grad_checkpoint(blk, t, rope, use_reentrant=False)
            else:
                t = blk(t, rope)
        t = m.norm(t)
        return t[:, self.eva_num_prefix:]

    def eva_tokens(self, x: torch.Tensor) -> torch.Tensor:
        """Network input ``(B, C, Z, Y, X)`` -> token volume ``(B, 768, Zt, gh, gw)``."""
        b = x.shape[0]
        img = render_eva_channels(x)                       # (B, 3, Z, Y, X)
        if self.eva_z_stride > 1:
            img = img[:, :, ::self.eva_z_stride]
        zt = img.shape[2]
        img = img.permute(0, 2, 1, 3, 4).reshape(b * zt, 3, *img.shape[3:])
        img = F.interpolate(img, size=self.eva_img_size, mode="bilinear", align_corners=False)
        img = (img - self.eva_img_mean) / self.eva_img_std

        step = self.eva_chunk if self.eva_chunk > 0 else img.shape[0]
        feats = torch.cat([self._eva_blocks(img[i:i + step])
                           for i in range(0, img.shape[0], step)], dim=0)
        gh, gw = self.eva_token_grid
        return feats.reshape(b, zt, gh, gw, self.eva_dim).permute(0, 4, 1, 2, 3)

    # -- fusion ---------------------------------------------------------
    def forward(self, x):
        skips = self.encoder(x)
        tok = self.eva_tokens(x)
        for i, s in enumerate(self.eva_fuse_stages):
            skip = skips[s]
            # resize the token volume *first*: projecting 768 -> C at 112x16x13 would
            # cost 40x the FLOPs of projecting at the stage's own 7x10x8 grid
            t = F.interpolate(tok.to(skip.dtype), size=tuple(skip.shape[2:]),
                              mode="trilinear", align_corners=False)
            skips[s] = skip + self.eva_fuse[i](t)
        return self.decoder(skips)

    @staticmethod
    def initialize(module):
        _init_skipping_frozen(module)

    def compute_conv_feature_map_size(self, input_size):
        # the EVA branch is not a conv feature map; nnU-Net only uses this for VRAM
        # heuristics during planning, and the plans are fixed here.
        return super().compute_conv_feature_map_size(input_size)

    # -- introspection used by the trainer and the tests ------------------
    def eva_param_groups(self, base_lr: float, layer_decay: float):
        """Layer-wise-lr-decay buckets over the *trainable* EVA parameters.

        ``patch_embed`` is layer 0, block ``i`` is layer ``i+1`` and the final norm is
        layer ``depth+1``; a layer's lr is ``base_lr * layer_decay ** (depth + 1 - id)``,
        the EVA-02 fine-tuning recipe (448 px, lr 1e-4, layer decay 0.7).
        """
        depth = len(self.eva.blocks)
        buckets: dict = {}
        for name, p in self.eva.named_parameters():
            if not p.requires_grad:
                continue
            if name.startswith("blocks."):
                layer = int(name.split(".")[1]) + 1
            elif name.startswith("patch_embed") or name in ("cls_token", "pos_embed", "reg_token"):
                layer = 0
            else:                                   # final norm and anything else on top
                layer = depth + 1
            buckets.setdefault(layer, []).append((name, p))
        out = []
        for layer in sorted(buckets):
            scale = layer_decay ** (depth + 1 - layer)
            params = [p for _, p in buckets[layer]]
            out.append({"layer": layer, "lr": base_lr * scale, "scale": scale, "params": params})
        return out
