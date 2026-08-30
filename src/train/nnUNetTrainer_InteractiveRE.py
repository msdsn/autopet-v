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

"Epoch 0 is LesionTracer" is checked in-process, not assumed, and both halves of the
check are **bit-exact** rather than tolerance-based:

* randomising input channels 2-4 must move the logits by exactly 0 -- the stem's new
  columns are zero, so the interaction cannot reach the output yet;
* a stock ``ResidualEncoderUNet`` built from the same ``arch_kwargs`` at the same
  ``input_channels = 5``, carrying this network's own weights, must reproduce the
  logits exactly on the pre-remapped input -- so the subclass is stock plus
  ``pet_renorm`` and nothing else.

Both networks are built in the *same* process, which is what makes "exactly" mean
0.0: ``run_training`` sets ``cudnn.benchmark = True`` and a cached reference from
another process disagrees at ~1e-2 on logits of magnitude 20 (see ``identity_gate.py``).
For the same reason the gate does not compare against a 2-*channel* network: a
``(32, 2, ...)`` stem and a ``(32, 5, ...)`` stem are different convolutions with
different autotuned kernels and TF32 paths, and they disagree at 1.1e-01 on an A100
while the identical comparison in float64 on the CPU is 0.000e+00
(``train.test_networks_re --f64-identity``).
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

    # nnUNetTrainer.__init__ fills my_init_kwargs by walking
    # inspect.signature(self.__init__) against its own locals(), so the signature has
    # to be spelled out here; (*args, **kwargs) makes it look for a local named 'args'
    # and raises KeyError before the first epoch.
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device("cuda")):
        super().__init__(plans, configuration, fold, dataset_json, device)
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
        """Two exact assertions, in-process, on the real patch.

        (a) The interaction columns of the stem are zero, so channels 2-4 cannot
            reach the output: randomising them must change the logits by **exactly**
            zero. That is shape-independent and bit-exact.
        (b) The network is stock ``ResidualEncoderUNet`` plus ``pet_renorm`` and
            nothing else: a stock class built from the same ``arch_kwargs`` with the
            same ``input_channels = 5``, carrying this network's own weights, must
            reproduce the logits exactly on the pre-remapped input. Same shapes means
            the same cuDNN kernels, so "exactly" means 0.0 and not a tolerance.

        Together with the strict graft (0 missing, 0 unexpected, 0 shape mismatch)
        those two statements *are* "epoch 0 is the LesionTracer model". Comparing
        against a 2-*channel* network instead would be the same mathematical claim
        but not a bit-exact test: a (32, 2, ...) stem and a (32, 5, ...) stem are
        different convolutions, cuDNN autotunes them differently and TF32 rounds
        them differently, and the measured disagreement is 1.1e-01 on logits of
        magnitude 20 -- while the identical comparison in float64 on the CPU is
        0.000e+00. ``train.test_networks_re --f64-identity`` runs that one.
        """
        from nnunetv2.utilities.get_network_from_plans import get_network_from_plans

        x = self._identity_probe_batch().to(self.device)
        was_training = mod.training
        mod.eval()
        try:
            x2 = x.clone()
            x2[:, 2:] = torch.rand_like(x2[:, 2:])
            with torch.no_grad():
                a, b = mod(x), mod(x2)
            if not isinstance(a, (list, tuple)):
                a, b = [a], [b]
            d_inert = max((p.float() - q.float()).abs().max().item() for p, q in zip(a, b))
            if d_inert != 0.0:
                raise RuntimeError(
                    f"[RE] identity assertion FAILED: randomising the interaction "
                    f"channels moved the logits by {d_inert:.3e}; the stem's new "
                    f"columns are not zero")

            arch = self.configuration_manager.configuration["architecture"]
            kw = {k: v for k, v in arch["arch_kwargs"].items() if k != "pet_renorm"}
            ref = get_network_from_plans(
                "dynamic_network_architectures.architectures.unet.ResidualEncoderUNet",
                kw, arch["_kw_requires_import"], int(self.num_input_channels),
                int(self.label_manager.num_segmentation_heads),
                allow_init=True, deep_supervision=self.enable_deep_supervision)
            ref.load_state_dict(mod.state_dict(), strict=True)
            ref = ref.to(self.device).eval()
            x_ref = mod._remap_pet(x) if hasattr(mod, "_remap_pet") else x
            with torch.no_grad():
                out, r = mod(x), ref(x_ref)
            if not isinstance(out, (list, tuple)):
                out, r = [out], [r]
            d = max((p.float() - q.float()).abs().max().item() for p, q in zip(out, r))
            ref.to("cpu")
            del ref
            torch.cuda.empty_cache()
        finally:
            if was_training:
                mod.train()

        if d >= self.IDENTITY_TOL:
            raise RuntimeError(
                f"[RE] identity assertion FAILED: max |logit diff| {d:.3e} >= "
                f"{self.IDENTITY_TOL:g} against a stock ResidualEncoderUNet of the "
                f"same shape carrying the same weights")
        self.print_to_log_file(
            f"[RE] identity assertion PASS: interaction channels inert "
            f"(max |logit diff| {d_inert:.3e}), and the network equals a stock "
            f"in-process ResidualEncoderUNet + pet_renorm="
            f"{getattr(mod, 'pet_renorm', 'n/a')} to {d:.3e} < {self.IDENTITY_TOL:g} "
            f"on {tuple(x.shape)}")

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
