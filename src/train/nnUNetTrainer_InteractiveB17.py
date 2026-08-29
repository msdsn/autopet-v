"""B17 -- the trainable pretrained-encoder continuation of B10.

Same recipe as every other architecture row (``nnUNetTrainer_InteractiveV2_negfp``:
V2 lesion-free false-positive term, B6 interaction distribution, 120 epochs,
``save_every`` 5) and the same strict graft plus epoch-0 identity gate. What differs
is forced by the fact that the added branch is an 86 M-parameter *pretrained* ViT
rather than a few zero-initialised convolutions.

**Two optimizers, one face.** nnU-Net's default is
``SGD(lr=5e-4, momentum=0.99, nesterov, wd=3e-5)`` over the whole network. Momentum
0.99 amplifies the steady-state step by ``1/(1-m) = 100x``, i.e. an effective ~5e-2
against ViT weights whose std is ~0.02, and it decays every LayerNorm gain and bias --
EVA-02's own recipe is AdamW with layer-wise lr decay and **no** weight decay on 1-D
parameters. Using AdamW for the *whole* network would fix the ViT and confound the row
against the SGD-trained C0 control ("was it the EVA branch or the optimizer?"). So
``_DualOptimizer`` presents one ``torch.optim.Optimizer`` face over an **AdamW for
``eva.*``** and the **stock SGD for everything else** -- the U-Net half trains exactly
as C0 does, and the row still measures the branch.

**A scheduler that preserves the ladder.** nnU-Net's ``PolyLRScheduler`` writes one
``new_lr`` into *every* param group, so layer-wise decay would be erased at epoch 0.
``LayerDecayPolyLR`` decays each group from its own ``base_lr`` instead.

**Pretrained weights enter at surgery time.** The network is constructed with
``eva_pretrained: false`` (so the predictor never reaches the network inside a
``--network=none`` container) and the trainer calls ``load_pretrained_eva()`` while the
network is being built *for training*, before ``nnUNetTrainer_InteractiveArch``
snapshots the added parameters -- which is what makes its ``|w-w0|/|w0|`` and
``|grad|`` diagnostic measure drift away from *pretraining*.
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
    "LayerDecayPolyLR",
    "nnUNetTrainer_InteractiveB17",
    "nnUNetTrainer_InteractiveB17_80epochs",
    "nnUNetTrainer_InteractiveB17_2epochs",
]

#: set only while this trainer is building its network, so that *training* loads the
#: timm weights and *inference* -- which constructs the same class from the shipped
#: plans, offline -- never does
_LOAD_PRETRAINED_EVA = False


def _env_float(name: str, default: float) -> float:
    v = os.environ.get(name)
    return default if v is None or v == "" else float(v)


def _env_flag(name: str, default: bool) -> bool:
    v = os.environ.get(name)
    return default if v is None or v == "" else v.lower() in ("1", "true", "t", "yes")


class LayerDecayPolyLR(PolyLRScheduler):
    """PolyLR that decays every param group from its **own** ``base_lr``.

    nnU-Net's version assigns a single ``new_lr`` to all groups, which is right for a
    one-group optimizer and silently erases a layer-wise decay ladder otherwise.
    """

    def step(self, current_step=None):
        if current_step is None or current_step == -1:
            current_step = self.ctr
            self.ctr += 1
        factor = (1 - current_step / self.max_steps) ** self.exponent
        for group in self.optimizer.param_groups:
            group["lr"] = group.get("base_lr", self.initial_lr) * factor
        self._last_lr = [group["lr"] for group in self.optimizer.param_groups]


class _DualOptimizer(torch.optim.Optimizer):
    """One ``Optimizer`` face over two real optimizers with disjoint parameters.

    ``param_groups`` is the concatenation of the children's *own* group dicts, so a
    scheduler writing ``group["lr"]`` reaches the child directly, and ``GradScaler``
    (which only walks ``param_groups`` to unscale) sees every parameter exactly once.
    ``torch.optim.lr_scheduler`` type-checks for a real ``Optimizer``, which is why
    this subclasses one rather than duck-typing.
    """

    def __init__(self, *optimizers: torch.optim.Optimizer):
        assert optimizers, "need at least one optimizer"
        self._opts = list(optimizers)
        params = [p for o in self._opts for g in o.param_groups for p in g["params"]]
        super().__init__(params, {"lr": self._opts[0].param_groups[0]["lr"]})
        # adopt the children's dicts; edits by a scheduler now land on the children
        self.param_groups = [g for o in self._opts for g in o.param_groups]

    @property
    def state(self):                      # nnU-Net never writes it; GradScaler reads
        merged = {}
        for o in self._opts:
            merged.update(o.state)
        return merged

    @state.setter
    def state(self, value):               # Optimizer.__init__ assigns an empty dict
        self._own_state = value

    def zero_grad(self, set_to_none: bool = True):
        for o in self._opts:
            o.zero_grad(set_to_none=set_to_none)

    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for o in self._opts:
            o.step()
        return loss

    def state_dict(self):
        return {"dual": [o.state_dict() for o in self._opts]}

    def load_state_dict(self, state_dict):
        for o, sd in zip(self._opts, state_dict["dual"]):
            o.load_state_dict(sd)
        self.param_groups = [g for o in self._opts for g in o.param_groups]

    def add_param_group(self, param_group):
        raise NotImplementedError("_DualOptimizer has fixed groups")


class nnUNetTrainer_InteractiveB17(nnUNetTrainer_InteractiveArch):
    """B17 -- trainable 2.5D EVA-02-B branch fused into the B10 encoder."""

    NEW_PARAM_PREFIXES = ("eva.", "eva_fuse.")
    NUM_EPOCHS: int = 120
    #: base lr of the *last* EVA block; earlier blocks are scaled by the decay ladder
    EVA_LR: float = 5e-5
    EVA_LAYER_DECAY: float = 0.7
    EVA_WEIGHT_DECAY: float = 0.05
    EVA_BETAS = (0.9, 0.98)
    EVA_EPS: float = 1e-6

    # ------------------------------------------------------------------
    # network construction: the one place the timm weights are fetched
    # ------------------------------------------------------------------
    @staticmethod
    def build_network_architecture(*args, **kwargs):
        from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer as _Base
        net = _Base.build_network_architecture(*args, **kwargs)
        if _LOAD_PRETRAINED_EVA and hasattr(net, "load_pretrained_eva"):
            net.load_pretrained_eva()
        return net

    def initialize(self):
        global _LOAD_PRETRAINED_EVA
        want = _env_flag("AUTOPET_EVA_INIT_PRETRAINED",
                         not getattr(self, "_continue_training", False))
        prev, _LOAD_PRETRAINED_EVA = _LOAD_PRETRAINED_EVA, want
        try:
            super().initialize()
        finally:
            _LOAD_PRETRAINED_EVA = prev

    def _do_i_compile(self) -> bool:
        """Never ``torch.compile`` this network.

        The EVA branch mixes a ``no_grad`` prefix, gradient checkpointing and a
        ``forward`` that dispatches on ``self.training``; dynamo either graph-breaks
        through all of it or recompiles per shape, and the compile itself costs
        minutes. The U-Net half is dataloader-bound anyway, so the loss is small and
        the failure mode it removes is large. It also makes the network the predictor
        rebuilds at evaluation time identical to the trained one.
        """
        if super()._do_i_compile():
            self.print_to_log_file(
                "[b17] torch.compile is disabled for the EVA branch (see _do_i_compile)")
        return False

    # ------------------------------------------------------------------
    # optimizer
    # ------------------------------------------------------------------
    def configure_optimizers(self):
        eva_lr = _env_float("nnUNet_b17_eva_lr", self.EVA_LR)
        decay = _env_float("nnUNet_b17_eva_layer_decay", self.EVA_LAYER_DECAY)
        eva_wd = _env_float("nnUNet_b17_eva_wd", self.EVA_WEIGHT_DECAY)

        net = self._unwrapped_network()
        if not hasattr(net, "eva_param_groups"):
            self.print_to_log_file(
                "[b17] the plans did not build an EVAFusionUNet -- falling back to the "
                "stock single-group optimizer")
            return super().configure_optimizers()

        # --- AdamW over the EVA blocks, layer-wise lr decay, no wd on 1-D params
        adam_groups = []
        ladder = net.eva_param_groups(eva_lr, decay)
        for g in ladder:
            decay_p = [p for p in g["params"] if p.ndim > 1]
            nodecay_p = [p for p in g["params"] if p.ndim <= 1]
            if decay_p:
                adam_groups.append({"params": decay_p, "lr": g["lr"],
                                    "base_lr": g["lr"], "weight_decay": eva_wd})
            if nodecay_p:
                adam_groups.append({"params": nodecay_p, "lr": g["lr"],
                                    "base_lr": g["lr"], "weight_decay": 0.0})
        adamw = torch.optim.AdamW(adam_groups, lr=eva_lr, betas=tuple(self.EVA_BETAS),
                                  eps=self.EVA_EPS)

        # --- the stock nnU-Net SGD over everything else: identical to C0's recipe
        eva_ids = {id(p) for p in net.eva.parameters()}
        rest = [p for p in net.parameters() if p.requires_grad and id(p) not in eva_ids]
        sgd = torch.optim.SGD([{"params": rest, "lr": self.initial_lr,
                                "base_lr": self.initial_lr}],
                              self.initial_lr, weight_decay=self.weight_decay,
                              momentum=0.99, nesterov=True)

        # SGD first: nnU-Net logs param_groups[0]['lr'], which should stay B10's number
        optimizer = _DualOptimizer(sgd, adamw)
        lr_scheduler = LayerDecayPolyLR(optimizer, self.initial_lr, self.num_epochs)

        n_rest = sum(p.numel() for p in rest)
        n_eva = sum(p.numel() for g in ladder for p in g["params"])
        n_frozen = sum(p.numel() for p in net.eva.parameters() if not p.requires_grad)
        self.print_to_log_file(
            f"[b17] SGD(momentum 0.99, wd {self.weight_decay:g}) over {n_rest / 1e6:.2f} M "
            f"at lr {self.initial_lr:g}  +  AdamW(betas {tuple(self.EVA_BETAS)}, "
            f"eps {self.EVA_EPS:g}, wd {eva_wd:g} on >1-D only) over {n_eva / 1e6:.2f} M "
            f"EVA parameters in {len(ladder)} layers at base lr {eva_lr:g}, layer decay "
            f"{decay:g}; {n_frozen / 1e6:.2f} M EVA parameters frozen")
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
