"""RE -- the B10 recipe on the autoPET III LesionTracer ResEncL backbone.

Everything about the *method* is B10 (``nnUNetTrainer_InteractiveV2_negfp``): the same
online error-driven scribble simulation, the same k distribution and independent
scribble share, the same DC+CE at ``smooth = 0`` plus the lesion-free false-positive
term, the same store, the same 39-case screen. What changes is the **network and its
initialisation**:

| | C0 (control) | RE |
|---|---|---|
| network | ``PlainConvUNet``, 30.79 M | ``ResEncInteractiveUNet`` (ResEncL), 102.35 M |
| patch | 112 x 160 x 128 = 2.29 M voxels | 192^3 = 7.08 M voxels |
| init | organisers' 1000-epoch Dataset998 baseline | **autoPET III champion**, 1500 epochs, MultiTalent-pretrained |
| spacing | [3.0, 2.0364, 2.0364] | the same -- which is why no re-preprocessing is needed |

The 40-epoch screening variant is not an arbitrary budget: 40 x 250 x 2 x 7.08 M =
1.42e11 training voxels against the control's 120 x 250 x 2 x 2.29 M = 1.37e11, so
the screen sees **the same number of voxels** as the row it is compared with, in a
third of the epochs.

## Why there is no re-initialised head, and no head learning rate

Their ``decoder.seg_layers.*`` are ``(2, C, 1, 1, 1)`` and their ``dataset.json`` is
``{"background": 0, "tumor": 1}``: two classes, our two classes, at our spacing. They
are kept verbatim. The organ supervision lives in a *separate* module,
``decoder.organ_seg_layers.*`` (11 classes), which the surgery drops. So the network
at epoch 0 is a competent lesion segmenter reading a 5-channel tensor whose last three
channels its zero-initialised stem columns ignore -- and ``RE_STEM_LR_MULT`` exists
for the one part that genuinely starts from nothing, the stem, defaulting to 1.0 so
the shipped row runs one learning rate.

## The launch gate

"Epoch 0 is LesionTracer" is checked in-process, not assumed. ``_assert_identity_at_init``
rebuilds the **stock 2-channel** ``ResidualEncoderUNet`` from the same ``arch_kwargs``,
loads it with our own grafted weights sliced back to two stem columns, and asserts the
two networks agree to < 1e-5 on a fixed probe batch. Building both in the *same*
process is what makes the tolerance meaningful: ``run_training`` sets
``cudnn.benchmark = True``, so a cached reference from another process disagrees at
~1e-2 on logits of magnitude ~20 (see ``identity_gate.py``).
"""

from __future__ import annotations

import json
import os
from typing import Tuple

import torch

try:  # package import (src/train is a package)
    from .nnUNetTrainer_InteractiveV2 import nnUNetTrainer_InteractiveV2_negfp
    from .nnUNetTrainer_Interactive import _env_float
    from .nnUNetTrainer_InteractiveArch import GroupScaledPolyLRScheduler
    from .networks_re import STEM_CONV_KEYS, graft_lesiontracer_state_dict
except ImportError:  # flat import (folder on sys.path, e.g. nnUNet_extTrainer)
    from nnUNetTrainer_InteractiveV2 import nnUNetTrainer_InteractiveV2_negfp  # type: ignore
    from nnUNetTrainer_Interactive import _env_float  # type: ignore
    from nnUNetTrainer_InteractiveArch import GroupScaledPolyLRScheduler  # type: ignore
    from networks_re import STEM_CONV_KEYS, graft_lesiontracer_state_dict  # type: ignore


__all__ = [
    "nnUNetTrainer_InteractiveRE",
    "nnUNetTrainer_InteractiveRE_40epochs",
    "nnUNetTrainer_InteractiveRE_100epochs",
    "nnUNetTrainer_InteractiveRE_2epochs",
]

_STEM_PREFIX = "encoder.stem."


class nnUNetTrainer_InteractiveRE(nnUNetTrainer_InteractiveV2_negfp):
    """B10's recipe, warm-started from the LesionTracer ResEncL."""

    #: the 120-epoch variant; the screen runs nnUNetTrainer_InteractiveRE_40epochs
    NUM_EPOCHS: int = 120
    INITIAL_LR: float = 5e-4
    #: learning-rate multiplier for the stem, the only module with new (zero) columns
    STEM_LR_MULT: float = 1.0
    IDENTITY_TOL: float = 1e-5
    IDENTITY_BATCH: int = 1

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.stem_lr_mult = _env_float("RE_STEM_LR_MULT", self.STEM_LR_MULT)
        self.print_to_log_file(
            f"[RE] stem_lr_mult={self.stem_lr_mult} identity_tol={self.IDENTITY_TOL}")

    # ------------------------------------------------------------------
    # weight loading: a strict graft, organ heads excepted
    # ------------------------------------------------------------------
    def initialize(self):
        already = self.was_initialized
        wanted = self.pretrained_weights_file
        self.pretrained_weights_file = None       # keep the inherited loaders out
        from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
        nnUNetTrainer.initialize(self)
        self.pretrained_weights_file = wanted

        if self._continue_training and wanted:
            self.print_to_log_file(
                "[RE] --c requested: ignoring nnUNet_interactive_pretrained, "
                "the checkpoint on disk wins")
            self.pretrained_weights_file = None
            return
        if already or not wanted:
            return

        mod = self._unwrapped_network()
        ck = torch.load(wanted, map_location="cpu", weights_only=False)
        sd = ck["network_weights"]
        stem = sd.get(STEM_CONV_KEYS[0])
        if stem is None or int(stem.shape[1]) != int(self.num_input_channels):
            raise RuntimeError(
                f"[RE] {wanted} has stem {None if stem is None else tuple(stem.shape)}, "
                f"expected {self.num_input_channels} input channels -- run "
                f"train.init_from_lesiontracer first")
        _, dropped = graft_lesiontracer_state_dict(mod, sd, verbose=False)
        self.print_to_log_file(
            f"[RE] grafted {len(sd) - len(dropped)} tensors from {wanted} "
            f"(source trainer={ck.get('surgery', {}).get('source_trainer')}, "
            f"source epoch={ck.get('surgery', {}).get('source_epoch')}); "
            f"{len(dropped)} organ-head tensors dropped; 0 missing, 0 unexpected")
        self._assert_identity_at_init(mod)

    def _unwrapped_network(self):
        from torch._dynamo import OptimizedModule
        from torch.nn.parallel import DistributedDataParallel as DDP
        mod = self.network
        if isinstance(mod, DDP):
            mod = mod.module
        if isinstance(mod, OptimizedModule):
            mod = mod._orig_mod
        return mod

    # ------------------------------------------------------------------
    # launch gate
    # ------------------------------------------------------------------
    def _identity_probe_batch(self) -> torch.Tensor:
        patch = [int(p) for p in self.configuration_manager.patch_size]
        c, b = int(self.num_input_channels), int(self.IDENTITY_BATCH)
        g = torch.Generator().manual_seed(0)
        x = torch.randn(b, c, *patch, generator=g)
        if c >= 4:
            x[:, 2:4] = torch.rand(b, 2, *patch, generator=g)
        if c >= 5:
            x[:, 4] = (torch.rand(b, *patch, generator=g) > 0.98).float()
        return x

    def _assert_identity_at_init(self, mod) -> None:
        """The grafted 5-channel network must equal the stock 2-channel LesionTracer.

        The reference is built here, in this process, from the same ``arch_kwargs``
        with ``input_channels = 2`` and the *stock* ``ResidualEncoderUNet`` class, and
        loaded with this network's own weights, the stem sliced back to its first two
        columns. It therefore tests exactly the two claims the surgery makes: the
        interaction columns are zero (channels 2-4 cannot move the output) and nothing
        else was disturbed. ``pet_renorm`` is applied to the reference's input by hand,
        because the stock class does not carry it.
        """
        from dynamic_network_architectures.architectures.unet import ResidualEncoderUNet

        arch = self.configuration_manager.configuration["architecture"]
        kw = {k: v for k, v in arch["arch_kwargs"].items() if k not in ("pet_renorm",)}
        from nnunetv2.utilities.get_network_from_plans import get_network_from_plans
        ref = get_network_from_plans(
            "dynamic_network_architectures.architectures.unet.ResidualEncoderUNet",
            kw, arch["_kw_requires_import"], 2,
            int(self.label_manager.num_segmentation_heads),
            allow_init=True, deep_supervision=self.enable_deep_supervision)

        sd = {k: v.detach().clone() for k, v in mod.state_dict().items()}
        for k in STEM_CONV_KEYS:
            if k in sd:
                sd[k] = sd[k][:, :2].contiguous()
        ref.load_state_dict(sd, strict=True)

        x = self._identity_probe_batch().to(self.device)
        x_ref = mod._remap_pet(x)[:, :2] if hasattr(mod, "_remap_pet") else x[:, :2]
        was_training = mod.training
        ref = ref.to(self.device).eval()
        mod.eval()
        try:
            with torch.no_grad():
                out, r = mod(x), ref(x_ref)
        finally:
            if was_training:
                mod.train()
            ref.to("cpu")
            del ref
            torch.cuda.empty_cache()

        if not isinstance(out, (list, tuple)):
            out, r = [out], [r]
        d = max((a.float() - b.float()).abs().max().item() for a, b in zip(out, r))
        if d >= self.IDENTITY_TOL:
            raise RuntimeError(
                f"[RE] identity assertion FAILED: max |logit diff| {d:.3e} >= "
                f"{self.IDENTITY_TOL:g} against the stock 2-channel ResidualEncoderUNet")
        self.print_to_log_file(
            f"[RE] identity assertion PASS: max |logit diff| {d:.3e} < "
            f"{self.IDENTITY_TOL:g} on {tuple(x.shape)}, in-process against a stock "
            f"2-channel ResidualEncoderUNet carrying the same weights "
            f"(pet_renorm={getattr(mod, 'pet_renorm', 'n/a')})")

    # ------------------------------------------------------------------
    # optimizer
    # ------------------------------------------------------------------
    def configure_optimizers(self):
        if self.stem_lr_mult == 1.0:
            return super().configure_optimizers()
        mod = self._unwrapped_network()
        stem, rest = [], []
        for name, p in mod.named_parameters():
            (stem if name.startswith(_STEM_PREFIX) else rest).append(p)
        groups = [{"params": rest, "lr_scale": 1.0},
                  {"params": stem, "lr_scale": float(self.stem_lr_mult)}]
        optimizer = torch.optim.SGD(groups, self.initial_lr, weight_decay=self.weight_decay,
                                    momentum=0.99, nesterov=True)
        scheduler = GroupScaledPolyLRScheduler(optimizer, self.initial_lr, self.num_epochs)
        self.print_to_log_file(
            f"[RE] two parameter groups: {sum(p.numel() for p in rest) / 1e6:.2f} M at "
            f"lr {self.initial_lr:.2e}, {sum(p.numel() for p in stem) / 1e6:.4f} M (stem) at "
            f"lr {self.initial_lr * self.stem_lr_mult:.2e}, both on the same poly decay")
        return optimizer, scheduler


class nnUNetTrainer_InteractiveRE_40epochs(nnUNetTrainer_InteractiveRE):
    """The screening schedule -- same training voxels as the 120-epoch control."""
    NUM_EPOCHS: int = 40


class nnUNetTrainer_InteractiveRE_100epochs(nnUNetTrainer_InteractiveRE):
    """The full schedule, if the screen wins."""
    NUM_EPOCHS: int = 100


class nnUNetTrainer_InteractiveRE_2epochs(nnUNetTrainer_InteractiveRE):
    """Smoke-test variant."""
    NUM_EPOCHS: int = 2
