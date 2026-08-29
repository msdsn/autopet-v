"""B17 -- the trainable pretrained-encoder continuation of B10.

Same recipe as every other architecture row (``nnUNetTrainer_InteractiveV2_negfp``:
V2 lesion-free false-positive term, B6 interaction distribution, 120 epochs,
``save_every`` 5) and the same strict graft plus epoch-0 identity gate. Two things
differ, both forced by the fact that the added branch is an 86 M-parameter
*pretrained* backbone rather than a few zero-initialised convolutions:

* **Parameter groups.** The U-Net keeps B10's lr 5e-4 with SGD momentum 0.99. The
  EVA blocks get a much lower base lr (``EVA_LR`` 1e-4) with **layer-wise lr decay
  0.7** -- the published EVA-02 448-px fine-tuning recipe -- and momentum 0.9, since
  a 0.99 momentum multiplies the effective step by 100 and a pretrained ViT does not
  survive that. Frozen parameters (``patch_embed`` and the first four blocks) are
  never handed to the optimizer.
* **A per-group PolyLR.** nnU-Net's ``PolyLRScheduler`` writes one lr into *every*
  param group, which would erase the decay ladder on the first epoch.
  ``_PolyLRPerGroup`` decays each group from its own base instead.

The fusion projections (``eva_fuse.``) sit in the main group at 5e-4: they start at
zero, so nothing reaches the EVA blocks until they have grown, and they need to grow
quickly.
"""

from __future__ import annotations

import os

import torch

from nnunetv2.training.lr_scheduler.polylr import PolyLRScheduler

try:  # package import (src/train is a package)
    from .nnUNetTrainer_InteractiveArch import nnUNetTrainer_InteractiveArch
except ImportError:  # flat import (folder on sys.path, e.g. nnUNet_extTrainer)
    from nnUNetTrainer_InteractiveArch import nnUNetTrainer_InteractiveArch  # type: ignore


__all__ = [
    "nnUNetTrainer_InteractiveB17",
    "nnUNetTrainer_InteractiveB17_80epochs",
    "nnUNetTrainer_InteractiveB17_2epochs",
]


def _env_float(name: str, default: float) -> float:
    v = os.environ.get(name)
    return default if v is None or v == "" else float(v)


class _PolyLRPerGroup(PolyLRScheduler):
    """PolyLR that decays every param group from its **own** base lr.

    nnU-Net's version assigns a single ``new_lr`` to all groups, which is correct
    for a one-group optimizer and destroys a layer-wise decay ladder otherwise.
    """

    def __init__(self, optimizer, initial_lr: float, max_steps: int, exponent: float = 0.9,
                 current_step: int = None):
        # set before super().__init__, which calls step() once
        self._group_lrs = [float(g["lr"]) for g in optimizer.param_groups]
        super().__init__(optimizer, initial_lr, max_steps, exponent, current_step)

    def step(self, current_step=None):
        if current_step is None or current_step == -1:
            current_step = self.ctr
            self.ctr += 1
        factor = (1 - current_step / self.max_steps) ** self.exponent
        for group, base in zip(self.optimizer.param_groups, self._group_lrs):
            group["lr"] = base * factor
        self._last_lr = [group["lr"] for group in self.optimizer.param_groups]


class nnUNetTrainer_InteractiveB17(nnUNetTrainer_InteractiveArch):
    """B17 -- trainable 2.5D EVA-02-B branch fused into the B10 encoder."""

    NEW_PARAM_PREFIXES = ("eva.", "eva_fuse.")
    NUM_EPOCHS: int = 120
    #: base lr of the *last* EVA block; earlier blocks are scaled by the decay ladder
    EVA_LR: float = 1e-4
    EVA_LAYER_DECAY: float = 0.7
    EVA_MOMENTUM: float = 0.9

    def _do_i_compile(self) -> bool:
        """Never ``torch.compile`` this network.

        The EVA branch mixes a ``no_grad`` prefix, gradient checkpointing and a
        ``forward`` that dispatches on ``self.training``; dynamo either graph-breaks
        through all of it or recompiles per shape, and the compile itself costs
        minutes. The U-Net half is dataloader-bound anyway, so the loss is small and
        the failure mode it removes is large. This also makes the network the
        predictor rebuilds at evaluation time identical to the trained one.
        """
        if super()._do_i_compile():
            self.print_to_log_file(
                "[b17] torch.compile is disabled for the EVA branch (see _do_i_compile)")
        return False

    def configure_optimizers(self):
        eva_lr = _env_float("nnUNet_b17_eva_lr", self.EVA_LR)
        decay = _env_float("nnUNet_b17_eva_layer_decay", self.EVA_LAYER_DECAY)
        momentum = _env_float("nnUNet_b17_eva_momentum", self.EVA_MOMENTUM)

        net = self.network
        from torch.nn.parallel import DistributedDataParallel as DDP
        if isinstance(net, DDP):
            net = net.module
        from torch._dynamo import OptimizedModule
        if isinstance(net, OptimizedModule):
            net = net._orig_mod

        if not hasattr(net, "eva_param_groups"):
            self.print_to_log_file(
                "[b17] the plans did not build an EVAFusionUNet -- falling back to the "
                "stock single-group optimizer")
            return super().configure_optimizers()

        eva_ids = {id(p) for p in net.eva.parameters()}
        main = [p for p in net.parameters() if p.requires_grad and id(p) not in eva_ids]
        # group 0 is the U-Net (and the zero-init fusion projections): nnU-Net logs
        # param_groups[0]['lr'], so the logged number stays the one B10 reports
        groups = [{"params": main, "lr": self.initial_lr, "momentum": 0.99,
                   "nesterov": True, "weight_decay": self.weight_decay}]
        ladder = net.eva_param_groups(eva_lr, decay)
        for g in ladder:
            groups.append({"params": g["params"], "lr": g["lr"], "momentum": momentum,
                           "nesterov": True, "weight_decay": self.weight_decay})

        optimizer = torch.optim.SGD(groups, self.initial_lr, weight_decay=self.weight_decay,
                                    momentum=0.99, nesterov=True)
        lr_scheduler = _PolyLRPerGroup(optimizer, self.initial_lr, self.num_epochs)

        n_main = sum(p.numel() for p in main)
        n_eva = sum(p.numel() for g in ladder for p in g["params"])
        n_frozen = sum(p.numel() for p in net.eva.parameters() if not p.requires_grad)
        self.print_to_log_file(
            f"[b17] optimizer: {len(groups)} groups; main {n_main / 1e6:.2f} M @ lr "
            f"{self.initial_lr:g} (momentum 0.99); EVA trainable {n_eva / 1e6:.2f} M in "
            f"{len(ladder)} layers @ base lr {eva_lr:g}, layer decay {decay:g} "
            f"(momentum {momentum:g}); EVA frozen {n_frozen / 1e6:.2f} M")
        for g in ladder:
            self.print_to_log_file(
                f"[b17]   layer {g['layer']:>2}  lr {g['lr']:.3e}  (x{g['scale']:.4f})  "
                f"{sum(p.numel() for p in g['params']) / 1e6:.2f} M")
        return optimizer, lr_scheduler


class nnUNetTrainer_InteractiveB17_80epochs(nnUNetTrainer_InteractiveB17):
    """Short schedule, used when the measured s/epoch does not fit 120 in the slot."""
    NUM_EPOCHS = 80


class nnUNetTrainer_InteractiveB17_2epochs(nnUNetTrainer_InteractiveB17):
    """Smoke-test variant."""
    NUM_EPOCHS = 2
