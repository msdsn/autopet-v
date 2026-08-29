"""Architecture variants of the interactive 5-channel PlainConvUNet.

Both classes subclass ``dynamic_network_architectures`` ``PlainConvUNet`` and add
modules on top of it, so their state dict is a strict superset of the baseline's and
the pretrained encoder/decoder tensors load unchanged.

* ``GlobalContextUNet`` -- a residual global-context block (an EVA-02 style
  transformer with 3D rotary embeddings, or a bidirectional state-space block when
  ``mamba_ssm`` is available) on the deepest encoder feature map.
* ``EditBranchUNet`` -- a second, lightweight decoder branch that receives the
  guidance and previous-mask channels at every decoder scale and emits an additive
  edit logit.

Every added module is zero-initialised at its output, so a freshly grafted network
reproduces the baseline logits exactly (see ``test_networks.py``). nnU-Net calls
``network.apply(network.initialize)`` after construction; ``initialize`` therefore
skips modules tagged by ``_freeze_init`` so the zeros survive.

Referenced from ``plans.json`` as ``train.networks.<Class>`` (``src`` on PYTHONPATH),
which is how ``pydoc.locate`` in ``nnunetv2.utilities.get_network_from_plans``
resolves a network class.
"""

from __future__ import annotations

import math
from typing import List, Sequence, Tuple, Type, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.dropout import _DropoutNd

from dynamic_network_architectures.architectures.unet import PlainConvUNet
from dynamic_network_architectures.building_blocks.simple_conv_blocks import StackedConvBlocks
from dynamic_network_architectures.building_blocks.helper import get_matching_convtransp
from dynamic_network_architectures.initialization.weight_init import InitWeights_He

__all__ = [
    "GlobalContextUNet",
    "EditBranchUNet",
    "mamba_available",
    "graft_state_dict",
]

_FREEZE_FLAG = "_autopet_freeze_init"


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _freeze_init(module: nn.Module) -> nn.Module:
    """Mark a subtree so ``initialize`` leaves its weights alone."""
    for m in module.modules():
        setattr(m, _FREEZE_FLAG, True)
    return module


def _zero_(module: nn.Module) -> nn.Module:
    """Zero every parameter of a module and protect it from re-initialisation."""
    for p in module.parameters():
        nn.init.zeros_(p)
    return _freeze_init(module)


def _init_skipping_frozen(module: nn.Module):
    if getattr(module, _FREEZE_FLAG, False):
        return
    InitWeights_He(1e-2)(module)


def mamba_available() -> bool:
    """True if a CUDA ``mamba_ssm`` build can be imported."""
    try:
        import mamba_ssm  # noqa: F401
        from mamba_ssm import Mamba  # noqa: F401
    except Exception:
        return False
    return True


def graft_state_dict(network: nn.Module, state_dict: dict, verbose: bool = True) -> Tuple[List[str], List[str]]:
    """Load a baseline state dict into a variant network. Weight surgery.

    Every tensor of ``state_dict`` must be consumed by ``network`` with a matching
    shape; the only tolerated difference is keys the variant adds. Raises if a
    baseline tensor is left over or has the wrong shape, which is what makes
    "epoch 0 reproduces the baseline" a checkable statement rather than a hope.

    Returns ``(missing, unexpected)`` as ``load_state_dict`` does; ``unexpected`` is
    always empty because a non-empty one raises.
    """
    own = network.state_dict()
    bad = [k for k in state_dict if k in own and tuple(own[k].shape) != tuple(state_dict[k].shape)]
    if bad:
        raise RuntimeError(f"shape mismatch on {len(bad)} baseline tensors, e.g. {bad[:5]}")
    missing, unexpected = network.load_state_dict(state_dict, strict=False)
    if unexpected:
        raise RuntimeError(f"{len(unexpected)} baseline tensors not consumed by the variant: "
                           f"{list(unexpected)[:8]}")
    if verbose:
        print(f"[graft] loaded {len(state_dict)} baseline tensors, "
              f"{len(missing)} new tensors kept at their init")
    return list(missing), list(unexpected)


def _as_int_tuple(x, n: int) -> Tuple[int, ...]:
    if isinstance(x, int):
        return (x,) * n
    return tuple(int(v) for v in x)


# ---------------------------------------------------------------------------
# B13: global-context bottleneck
# ---------------------------------------------------------------------------

class _EVAStyleBottleneckBlock(nn.Module):
    """EVA-02 style transformer stack over the flattened bottleneck feature map.

    Follows EVA-02 (arXiv:2303.11331): pre-norm blocks with **sub-LN** (an extra
    LayerNorm on the attention and FFN outputs before their last projection), a
    **SwiGLU** feed-forward of hidden width 2/3 * 4d, and **3D rotary position
    embeddings** on the queries and keys instead of a learned positional table.

    The stack runs at a reduced width ``dim`` (default 128 against the 320 encoder
    features), so it stays under half a million parameters. The output 1x1
    convolution back to the encoder width is zero-initialised, which makes the whole
    block an exact identity at initialisation while leaving that convolution a
    non-zero gradient on the first step.
    """

    def __init__(self, channels: int, dim: int = 128, n_layers: int = 2, n_heads: int = 8,
                 mlp_ratio: float = 4.0, rope_theta=10000.0, conv_op=nn.Conv3d):
        super().__init__()
        assert dim % n_heads == 0, f"dim {dim} must be divisible by n_heads {n_heads}"
        self.dim, self.n_heads = int(dim), int(n_heads)
        self.head_dim = self.dim // self.n_heads
        assert self.head_dim % 2 == 0, "head_dim must be even for rotary embeddings"
        self.n_pairs = self.head_dim // 2
        # scalar, or one theta per axis. A single theta over a 7x5x4 grid with 2-3
        # bands per axis puts every band's wavelength either far below or far above the
        # axis extent, which is barely a position code at all; per-axis theta = extent
        # makes the slowest band span the axis exactly once.
        self.rope_theta = (float(rope_theta) if np.isscalar(rope_theta)
                           else tuple(float(t) for t in rope_theta))
        self._rope_cache: dict = {}

        hidden = int(round(mlp_ratio * dim * 2 / 3 / 8)) * 8      # SwiGLU: 2/3 * 4d
        self.in_proj = conv_op(channels, dim, 1, 1, 0, bias=True)
        self.out_proj = _zero_(conv_op(dim, channels, 1, 1, 0, bias=True))

        self.norm1, self.qkv, self.attn_subln, self.attn_out = (nn.ModuleList() for _ in range(4))
        self.norm2, self.w1, self.w3, self.ffn_subln, self.w2 = (nn.ModuleList() for _ in range(5))
        for _ in range(n_layers):
            self.norm1.append(nn.LayerNorm(dim))
            self.qkv.append(nn.Linear(dim, 3 * dim, bias=True))
            self.attn_subln.append(nn.LayerNorm(dim))
            self.attn_out.append(nn.Linear(dim, dim, bias=True))
            self.norm2.append(nn.LayerNorm(dim))
            self.w1.append(nn.Linear(dim, hidden, bias=True))
            self.w3.append(nn.Linear(dim, hidden, bias=True))
            self.ffn_subln.append(nn.LayerNorm(hidden))
            self.w2.append(nn.Linear(hidden, dim, bias=True))
        _freeze_init(self)

    # -- 3D rotary position embedding ------------------------------------
    def _rope(self, grid, device, dtype):
        key = (tuple(grid), device, dtype)
        if key in self._rope_cache:
            return self._rope_cache[key]
        base = self.n_pairs // 3
        sizes = [base + (1 if i < self.n_pairs - 3 * base else 0) for i in range(3)]
        coords = torch.meshgrid(*[torch.arange(int(g), device=device, dtype=torch.float32)
                                  for g in grid], indexing="ij")
        angles = []
        for axis, n in enumerate(sizes):
            if n == 0:
                continue
            theta = (self.rope_theta if isinstance(self.rope_theta, float)
                     else self.rope_theta[axis])
            inv = theta ** (-torch.arange(n, device=device, dtype=torch.float32) / n)
            angles.append(coords[axis].reshape(-1, 1) * inv[None, :])
        ang = torch.cat(angles, dim=1)                            # (N, n_pairs)
        out = (ang.cos().to(dtype), ang.sin().to(dtype))
        self._rope_cache[key] = out
        return out

    def _apply_rope(self, t: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        b, h, n, _ = t.shape
        t = t.view(b, h, n, self.n_pairs, 2)
        c, s = cos[None, None], sin[None, None]
        x, y = t[..., 0], t[..., 1]
        return torch.stack((x * c - y * s, x * s + y * c), dim=-1).reshape(b, h, n, self.head_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        spatial = x.shape[2:]
        h = self.in_proj(x)
        b, d = h.shape[:2]
        tok = h.flatten(2).transpose(1, 2)                        # (b, n, d)
        n = tok.shape[1]
        cos, sin = self._rope(spatial, tok.device, tok.dtype)
        for i in range(len(self.norm1)):
            y = self.norm1[i](tok)
            qkv = self.qkv[i](y).view(b, n, 3, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]
            q = self._apply_rope(q, cos, sin)
            k = self._apply_rope(k, cos, sin)
            a = F.scaled_dot_product_attention(q, k, v)
            a = a.transpose(1, 2).reshape(b, n, d)
            tok = tok + self.attn_out[i](self.attn_subln[i](a))
            y = self.norm2[i](tok)
            g = F.silu(self.w1[i](y)) * self.w3[i](y)
            tok = tok + self.w2[i](self.ffn_subln[i](g))
        h = tok.transpose(1, 2).reshape(b, d, *spatial)
        return x + self.out_proj(h)


class _MambaContext(nn.Module):
    """Bidirectional selective state-space block over the flattened feature map.

    Two ``mamba_ssm`` scans (forward and reversed sequence) share a zero-initialised
    output projection, so the block is the identity at initialisation.
    """

    def __init__(self, channels: int, n_layers: int = 2, d_state: int = 16,
                 d_conv: int = 4, expand: int = 2):
        super().__init__()
        from mamba_ssm import Mamba

        self.norm = nn.ModuleList()
        self.fwd = nn.ModuleList()
        self.bwd = nn.ModuleList()
        self.out = nn.ModuleList()
        for _ in range(n_layers):
            self.norm.append(nn.LayerNorm(channels))
            self.fwd.append(Mamba(d_model=channels, d_state=d_state, d_conv=d_conv, expand=expand))
            self.bwd.append(Mamba(d_model=channels, d_state=d_state, d_conv=d_conv, expand=expand))
            proj = nn.Linear(2 * channels, channels)
            _zero_(proj)
            self.out.append(proj)
        _freeze_init(self)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c = x.shape[:2]
        spatial = x.shape[2:]
        tok = x.flatten(2).transpose(1, 2)
        for norm, fwd, bwd, out in zip(self.norm, self.fwd, self.bwd, self.out):
            h = norm(tok)
            a = fwd(h)
            b_ = torch.flip(bwd(torch.flip(h, dims=(1,))), dims=(1,))
            tok = tok + out(torch.cat((a, b_), dim=-1))
        return tok.transpose(1, 2).reshape(b, c, *spatial)


class GlobalContextUNet(PlainConvUNet):
    """PlainConvUNet with a residual global-context block on the deepest features.

    The bottleneck of the autoPET V plans is 7x5x4 = 140 tokens of 320 features at
    the 112x160x128 patch, so whole-patch context costs a few hundred microseconds.
    ``context_stage`` selects which encoder output the block sees (-1 = deepest).
    """

    def __init__(self, input_channels: int, n_stages: int, features_per_stage, conv_op: Type[_ConvNd],
                 kernel_sizes, strides, n_conv_per_stage, num_classes: int, n_conv_per_stage_decoder,
                 conv_bias: bool = False, norm_op: Union[None, Type[nn.Module]] = None,
                 norm_op_kwargs: dict = None, dropout_op: Union[None, Type[_DropoutNd]] = None,
                 dropout_op_kwargs: dict = None, nonlin: Union[None, Type[nn.Module]] = None,
                 nonlin_kwargs: dict = None, deep_supervision: bool = False, nonlin_first: bool = False,
                 context_type: str = "attention", context_stage: int = -1, context_layers: int = 2,
                 context_heads: int = 8, context_dim: int = 128, context_mlp_ratio: float = 4.0,
                 context_rope_theta=10000.0, context_d_state: int = 16,
                 context_d_conv: int = 4, context_expand: int = 2):
        super().__init__(input_channels, n_stages, features_per_stage, conv_op, kernel_sizes, strides,
                         n_conv_per_stage, num_classes, n_conv_per_stage_decoder, conv_bias, norm_op,
                         norm_op_kwargs, dropout_op, dropout_op_kwargs, nonlin, nonlin_kwargs,
                         deep_supervision, nonlin_first)
        self.context_stage = int(context_stage)
        self.context_type = context_type
        channels = int(self.encoder.output_channels[self.context_stage])
        if context_type == "mamba":
            self.context = _MambaContext(channels, n_layers=context_layers, d_state=context_d_state,
                                         d_conv=context_d_conv, expand=context_expand)
        elif context_type == "attention":
            self.context = _EVAStyleBottleneckBlock(
                channels, dim=context_dim, n_layers=context_layers, n_heads=context_heads,
                mlp_ratio=context_mlp_ratio, rope_theta=context_rope_theta, conv_op=conv_op)
        else:
            raise ValueError(f"unknown context_type {context_type!r}")

    def forward(self, x):
        skips = self.encoder(x)
        skips[self.context_stage] = self.context(skips[self.context_stage])
        return self.decoder(skips)

    @staticmethod
    def initialize(module):
        _init_skipping_frozen(module)


# ---------------------------------------------------------------------------
# B14: edit branch
# ---------------------------------------------------------------------------

class EditBranchUNet(PlainConvUNet):
    """PlainConvUNet plus a lightweight decoder branch driven by the interaction.

    The pretrained encoder and decoder are the automatic branch and are untouched.
    The edit branch mirrors the decoder at 16-32 features per stage; at every scale
    it concatenates its own upsampled state, a 1x1 projection of the encoder skip and
    the guidance channels max-pooled to that scale, and emits an edit logit that is
    added to the automatic logit. Deep supervision sees only the sum.

    Max pooling rather than averaging carries the guidance down: a scribble is a thin
    structure whose *presence* must survive a 32x downsampling, and its clipped-EDT
    encoding is non-negative, so the maximum is the value at the stroke.
    """

    def __init__(self, input_channels: int, n_stages: int, features_per_stage, conv_op: Type[_ConvNd],
                 kernel_sizes, strides, n_conv_per_stage, num_classes: int, n_conv_per_stage_decoder,
                 conv_bias: bool = False, norm_op: Union[None, Type[nn.Module]] = None,
                 norm_op_kwargs: dict = None, dropout_op: Union[None, Type[_DropoutNd]] = None,
                 dropout_op_kwargs: dict = None, nonlin: Union[None, Type[nn.Module]] = None,
                 nonlin_kwargs: dict = None, deep_supervision: bool = False, nonlin_first: bool = False,
                 n_guidance_channels: int = 3,
                 edit_features_per_stage: Sequence[int] = (16, 16, 24, 24, 32, 32),
                 n_conv_per_edit_stage: int = 2):
        super().__init__(input_channels, n_stages, features_per_stage, conv_op, kernel_sizes, strides,
                         n_conv_per_stage, num_classes, n_conv_per_stage_decoder, conv_bias, norm_op,
                         norm_op_kwargs, dropout_op, dropout_op_kwargs, nonlin, nonlin_kwargs,
                         deep_supervision, nonlin_first)
        assert len(edit_features_per_stage) == n_stages, \
            f"edit_features_per_stage needs {n_stages} entries, got {len(edit_features_per_stage)}"
        self.n_guidance_channels = int(n_guidance_channels)
        assert 0 < self.n_guidance_channels < input_channels

        ef = [int(f) for f in edit_features_per_stage]
        enc_ch = [int(c) for c in self.encoder.output_channels]
        transpconv_op = get_matching_convtransp(conv_op=conv_op)
        ng = self.n_guidance_channels

        self.edit_stem = conv_op(enc_ch[-1] + ng, ef[-1], 1, 1, 0, bias=True)
        ups, skip_projs, blocks, segs = [], [], [], []
        for s in range(n_stages - 1):
            i_below, i_out = n_stages - 1 - s, n_stages - 2 - s
            stride = _as_int_tuple(strides[i_below], 3)
            ups.append(transpconv_op(ef[i_below], ef[i_out], stride, stride, bias=True))
            skip_projs.append(conv_op(enc_ch[i_out], ef[i_out], 1, 1, 0, bias=True))
            blocks.append(StackedConvBlocks(
                n_conv_per_edit_stage, conv_op, 2 * ef[i_out] + ng, ef[i_out],
                kernel_sizes[i_out], 1, conv_bias, norm_op, norm_op_kwargs, dropout_op,
                dropout_op_kwargs, nonlin, nonlin_kwargs, nonlin_first))
            segs.append(_zero_(conv_op(ef[i_out], num_classes, 1, 1, 0, bias=True)))
        self.edit_ups = nn.ModuleList(ups)
        self.edit_skip_projs = nn.ModuleList(skip_projs)
        self.edit_stages = nn.ModuleList(blocks)
        self.edit_seg_layers = nn.ModuleList(segs)

    def _guidance_at(self, guidance: torch.Tensor, size) -> torch.Tensor:
        if tuple(guidance.shape[2:]) == tuple(size):
            return guidance
        return F.adaptive_max_pool3d(guidance, tuple(int(v) for v in size))

    def forward(self, x):
        skips = self.encoder(x)
        guidance = x[:, -self.n_guidance_channels:]
        deep_supervision = self.decoder.deep_supervision

        bottleneck = skips[-1]
        e = self.edit_stem(torch.cat(
            (bottleneck, self._guidance_at(guidance, bottleneck.shape[2:])), 1))

        lres = bottleneck
        outputs = []
        n = len(self.decoder.stages)
        for s in range(n):
            skip = skips[-(s + 2)]
            y = self.decoder.transpconvs[s](lres)
            y = torch.cat((y, skip), 1)
            y = self.decoder.stages[s](y)

            e = self.edit_ups[s](e)
            g = self._guidance_at(guidance, e.shape[2:])
            e = self.edit_stages[s](torch.cat((e, self.edit_skip_projs[s](skip), g), 1))

            if deep_supervision or s == n - 1:
                outputs.append(self.decoder.seg_layers[s](y) + self.edit_seg_layers[s](e))
            lres = y

        outputs = outputs[::-1]
        return outputs if deep_supervision else outputs[0]

    @staticmethod
    def initialize(module):
        _init_skipping_frozen(module)

    def compute_conv_feature_map_size(self, input_size):
        base = super().compute_conv_feature_map_size(input_size)
        # rough: the edit branch mirrors the decoder at ef/enc_ch of its width
        return int(base * 1.4)
