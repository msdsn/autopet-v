"""Trainers for the two architecture variants of the interactive model.

Both continue the ``nnUNetTrainer_InteractiveV2_negfp`` recipe -- same interaction
distribution, same lesion-free false-positive term, same 120 epochs at lr 5e-4 --
and differ from it only in the network the plans build:

* ``nnUNetTrainer_InteractiveB13`` with ``train.networks.GlobalContextUNet``
* ``nnUNetTrainer_InteractiveB14`` with ``train.networks.EditBranchUNet``

The source weights are the finished ``V2_negfp`` checkpoint. Every tensor of that
checkpoint must load into the variant; the tensors the variant adds are the only
ones allowed to keep their initialisation, and they are zero at their output, so
epoch 0 reproduces the source model exactly.
"""

from __future__ import annotations

import os

import torch

from nnunetv2.training.lr_scheduler.polylr import PolyLRScheduler

try:  # package import (src/train is a package)
    from .nnUNetTrainer_InteractiveV2 import nnUNetTrainer_InteractiveV2_negfp
    from .networks import graft_state_dict
except ImportError:  # flat import (folder on sys.path, e.g. nnUNet_extTrainer)
    from nnUNetTrainer_InteractiveV2 import nnUNetTrainer_InteractiveV2_negfp  # type: ignore
    from networks import graft_state_dict  # type: ignore


__all__ = [
    "GroupScaledPolyLRScheduler",
    "nnUNetTrainer_InteractiveArch",
    "nnUNetTrainer_InteractiveB13",
    "nnUNetTrainer_InteractiveB13b",
    "nnUNetTrainer_InteractiveB13c",
    "nnUNetTrainer_InteractiveB14",
    "nnUNetTrainer_InteractiveB13_2epochs",
    "nnUNetTrainer_InteractiveB14_2epochs",
]


class GroupScaledPolyLRScheduler(PolyLRScheduler):
    """PolyLR that keeps a per-group learning-rate ratio.

    nnU-Net's ``PolyLRScheduler.step`` writes one ``new_lr`` into *every* parameter
    group, so a second group created with a different learning rate is silently reset
    to the base rate on the first epoch and layer-wise rates are erased. This reads
    ``lr_scale`` off each group once and multiplies the poly-decayed rate by it, so
    the ratio survives the whole schedule.
    """

    def __init__(self, optimizer, initial_lr: float, max_steps: int, exponent: float = 0.9,
                 current_step: int = None):
        # must exist before the base __init__, which performs the first step()
        self.lr_scales = [float(g.get("lr_scale", 1.0)) for g in optimizer.param_groups]
        super().__init__(optimizer, initial_lr, max_steps, exponent, current_step)

    def step(self, current_step=None):
        if current_step is None or current_step == -1:
            current_step = self.ctr
            self.ctr += 1
        new_lr = self.initial_lr * (1 - current_step / self.max_steps) ** self.exponent
        for group, scale in zip(self.optimizer.param_groups, self.lr_scales):
            group["lr"] = new_lr * scale


class nnUNetTrainer_InteractiveArch(nnUNetTrainer_InteractiveV2_negfp):
    """Continuation onto a network that adds modules to the source architecture.

    ``nnUNetTrainer_InteractiveB6.initialize`` loads the source checkpoint with
    ``strict=True``, which cannot work once the network has tensors the checkpoint
    does not: it would fall back to nnU-Net's loader, which drops the segmentation
    heads. This override keeps the strictness where it matters -- no source tensor
    may be left over or mismatched -- while allowing the added modules through.
    """

    #: prefixes of the tensors the variant is allowed to add
    NEW_PARAM_PREFIXES: tuple = ()
    #: every architecture row runs the same schedule, so the deltas are comparable
    NUM_EPOCHS: int = 120
    #: tolerance of the identity assertion; the comparison is in-process, so it is exact
    IDENTITY_TOL: float = 1e-5
    #: learning-rate multiplier for the added modules (1.0 = one shared rate)
    NEW_PARAM_LR_MULT: float = 1.0
    #: how often the added modules' weight and gradient norms are logged
    LOG_NORMS_EVERY: int = 10

    def initialize(self):
        already = self.was_initialized
        wanted = self.pretrained_weights_file
        # Build network, loss and DDP wrapper with nnU-Net's own initialize; both
        # inherited overrides only add a weight load, and this class does its own.
        from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
        nnUNetTrainer.initialize(self)

        if self._continue_training and wanted:
            self.print_to_log_file(
                "[arch] --c requested: ignoring nnUNet_interactive_pretrained, "
                "the checkpoint on disk wins")
            self.pretrained_weights_file = None
            return
        if already or not wanted:
            return

        from torch._dynamo import OptimizedModule
        from torch.nn.parallel import DistributedDataParallel as DDP

        ck = torch.load(wanted, map_location="cpu", weights_only=False)
        sd = ck["network_weights"]
        mod = self.network
        if isinstance(mod, DDP):
            mod = mod.module
        if isinstance(mod, OptimizedModule):
            mod = mod._orig_mod

        missing, _ = graft_state_dict(mod, sd, verbose=False)
        if self.NEW_PARAM_PREFIXES and not missing:
            raise RuntimeError(
                "[arch] the plans built the baseline architecture: the source checkpoint "
                "covered every tensor, so none of "
                f"{list(self.NEW_PARAM_PREFIXES)} exists. Pass the variant plans "
                "(-p nnUNetPlans_b13 / nnUNetPlans_b14).")
        stray = [k for k in missing
                 if not any(k.startswith(p) for p in self.NEW_PARAM_PREFIXES)]
        if stray:
            raise RuntimeError(
                f"[arch] {len(stray)} tensors of the variant are outside the declared new "
                f"modules and were not in the source checkpoint: {stray[:8]}")
        own, seen, n_new = mod.state_dict(), set(), 0
        for k in missing:                      # nnU-Net aliases each conv/norm twice
            t = own[k]
            if t.data_ptr() not in seen:
                seen.add(t.data_ptr())
                n_new += t.numel()
        self.print_to_log_file(
            f"[arch] grafted {len(sd)} tensors from {wanted} "
            f"(trainer={ck.get('trainer_name')}, epoch={ck.get('current_epoch')}); "
            f"{len(missing)} new tensors ({n_new/1e6:.2f} M parameters) keep their "
            f"zero/identity initialisation")
        self._assert_identity_at_init(mod)
        self._snapshot_new_params(mod)

    # ------------------------------------------------------------------
    # did the added modules actually train?
    # ------------------------------------------------------------------
    def _unwrapped_network(self):
        from torch._dynamo import OptimizedModule
        from torch.nn.parallel import DistributedDataParallel as DDP
        mod = self.network
        if isinstance(mod, DDP):
            mod = mod.module
        if isinstance(mod, OptimizedModule):
            mod = mod._orig_mod
        return mod

    def _is_new(self, name: str) -> bool:
        return any(name.startswith(pre) for pre in self.NEW_PARAM_PREFIXES)

    def _snapshot_new_params(self, mod) -> None:
        """Keep a copy of the added parameters so their movement can be measured.

        A zero-initialised output projection makes the interior's gradient
        proportional to that projection's weight, so an added block can sit at its
        random initialisation for a whole run while only the output projection moves.
        Whether that happened is a measurement, not an argument -- hence the copy.
        """
        self._init_params = {n: p.detach().clone().cpu()
                             for n, p in mod.named_parameters() if self._is_new(n)}

    def _log_new_param_norms(self) -> None:
        init = getattr(self, "_init_params", None)
        if not init:
            return
        mod = self._unwrapped_network()
        tot_w = tot_w0 = tot_d = tot_g = 0.0
        rows = []
        for name, p in mod.named_parameters():
            if name not in init:
                continue
            w0 = init[name]
            d = p.detach().cpu() - w0
            nw, nw0, nd = float(p.detach().norm()), float(w0.norm()), float(d.norm())
            ng = float(p.grad.detach().norm()) if p.grad is not None else 0.0
            tot_w += nw ** 2; tot_w0 += nw0 ** 2; tot_d += nd ** 2; tot_g += ng ** 2
            rows.append((p.numel(), name, nw, nw0, nd, ng))
        if not rows:
            return
        import math
        ratio = math.sqrt(tot_w) / max(math.sqrt(tot_w0), 1e-12)
        rel = math.sqrt(tot_d) / max(math.sqrt(tot_w0), 1e-12)
        lrs = [g["lr"] for g in self.optimizer.param_groups]
        self.print_to_log_file(
            f"[arch] epoch {self.current_epoch}: new-module |w|/|w0|={ratio:.4f} "
            f"|w-w0|/|w0|={rel:.4f} |grad|={math.sqrt(tot_g):.3e} lr={['%.2e' % v for v in lrs]}")
        for _, name, nw, nw0, nd, ng in sorted(rows, reverse=True)[:5]:
            self.print_to_log_file(
                f"[arch]   {name}: |w|={nw:.4g} (x{nw / max(nw0, 1e-12):.4f} of init) "
                f"|w-w0|/|w0|={nd / max(nw0, 1e-12):.4g} |g|={ng:.3g}")

    def on_train_epoch_end(self, train_outputs):
        super().on_train_epoch_end(train_outputs)
        ep = self.current_epoch
        if ep < 3 or (ep + 1) % self.LOG_NORMS_EVERY == 0 or ep == self.num_epochs - 1:
            self._log_new_param_norms()

    def _assert_identity_at_init(self, mod) -> None:
        """Launch gate: the freshly grafted network must reproduce the source logits.

        ``nnUNet_arch_refbatch`` points at a file written by
        ``train.test_networks --emit-ref``: one fixed input batch plus the *baseline*
        architecture description. The baseline is rebuilt here, in this process and on
        this device, and loaded with the same source checkpoint, so the comparison is
        free of the cross-process cuDNN/TF32 kernel differences that make a cached
        output tensor agree only to ~1e-3. A run whose log does not carry this PASS
        line is not a valid ablation row.
        """
        ref_file = os.environ.get("nnUNet_arch_refbatch") or None
        if not ref_file:
            self.print_to_log_file(
                "[arch] WARNING: nnUNet_arch_refbatch is not set, the identity assertion "
                "was NOT run")
            return
        from nnunetv2.utilities.get_network_from_plans import get_network_from_plans

        blob = torch.load(ref_file, map_location="cpu", weights_only=False)
        x = blob["x"].to(self.device)
        arch = blob["arch"]
        ck = torch.load(self.pretrained_weights_file, map_location="cpu", weights_only=False)
        base = get_network_from_plans(
            arch["network_class_name"], arch["arch_kwargs"], arch["_kw_requires_import"],
            int(blob["in_channels"]), int(blob["num_classes"]),
            allow_init=True, deep_supervision=self.enable_deep_supervision)
        base.load_state_dict(ck["network_weights"], strict=True)
        base = base.to(self.device).eval()

        was_training = mod.training
        mod.eval()
        with torch.no_grad():
            out = mod(x)
            ref = base(x)
        if was_training:
            mod.train()
        base.to("cpu")
        del base
        torch.cuda.empty_cache()

        if not isinstance(out, (list, tuple)):
            out, ref = [out], [ref]
        d = max((o - r).abs().max().item() for o, r in zip(out, ref))
        if d >= self.IDENTITY_TOL:
            raise RuntimeError(f"[arch] identity assertion FAILED: max |logit diff| {d:.3e} "
                               f">= {self.IDENTITY_TOL:g} against a freshly built "
                               f"{arch['network_class_name']}")
        self.print_to_log_file(
            f"[arch] identity assertion PASS: max |logit diff| {d:.3e} < "
            f"{self.IDENTITY_TOL:g} on {tuple(x.shape)} against a freshly built "
            f"{arch['network_class_name']} carrying the same source weights")


    def configure_optimizers(self):
        """One parameter group per learning rate, and a scheduler that keeps the ratio.

        With ``NEW_PARAM_LR_MULT = 1`` this is nnU-Net's own optimizer with the groups
        split but both at the same rate, so the default path is unchanged.
        """
        if self.NEW_PARAM_LR_MULT == 1.0 or not self.NEW_PARAM_PREFIXES:
            return super().configure_optimizers()
        mod = self._unwrapped_network()
        new, old = [], []
        for name, p in mod.named_parameters():
            (new if self._is_new(name) else old).append(p)
        groups = [{"params": old, "lr_scale": 1.0},
                  {"params": new, "lr_scale": float(self.NEW_PARAM_LR_MULT)}]
        optimizer = torch.optim.SGD(groups, self.initial_lr, weight_decay=self.weight_decay,
                                    momentum=0.99, nesterov=True)
        scheduler = GroupScaledPolyLRScheduler(optimizer, self.initial_lr, self.num_epochs)
        self.print_to_log_file(
            f"[arch] two parameter groups: {sum(p.numel() for p in old) / 1e6:.2f} M at "
            f"lr {self.initial_lr:.2e}, {sum(p.numel() for p in new) / 1e6:.3f} M at "
            f"lr {self.initial_lr * self.NEW_PARAM_LR_MULT:.2e} "
            f"({self.NEW_PARAM_LR_MULT:g}x), both on the same poly decay")
        return optimizer, scheduler


class nnUNetTrainer_InteractiveC0(nnUNetTrainer_InteractiveArch):
    """C0 -- the no-change continuation control.

    The same network, the same recipe and the same schedule as every architecture
    row, so a variant's delta measures the block and not the extra epochs. It adds
    no tensors, which makes the graft a strict load.
    """

    NEW_PARAM_PREFIXES = ()


class nnUNetTrainer_InteractiveB13(nnUNetTrainer_InteractiveArch):
    """B13 -- residual global-context block on the deepest encoder stage."""

    NEW_PARAM_PREFIXES = ("context.",)


class nnUNetTrainer_InteractiveB13b(nnUNetTrainer_InteractiveB13):
    """B13b -- B13 with a rotary code that actually resolves position.

    Same block and the same parameter count; 4 heads instead of 8, so the head
    dimension is 32 and each axis gets 5-6 rotary bands instead of 2-3, and one theta
    per axis equal to that axis's extent on the 7x5x4 bottleneck grid. On the 7-voxel
    z axis the six bands then sweep 6.00, 4.34, 3.14, 2.27, 1.64 and 1.19 rad across
    the axis, against 6.00 / 0.28 / 0.013 rad for a single theta = 10000, where two of
    the three bands are constant over the whole grid.
    """


class nnUNetTrainer_InteractiveB13c(nnUNetTrainer_InteractiveB13b):
    """B13c -- B13b with the added block on its own, much larger learning rate.

    The zero-initialised output projection that buys the exact epoch-0 identity also
    attenuates every gradient inside the block by that projection's own magnitude: at
    epoch 50 of B13b it stood at 0.198 against a normal scale of ~6.5, so the interior
    saw roughly 1/33 of the gradient the rest of the network saw and was still within
    1 % of its random initialisation. The fix is a second parameter group at 30x the
    base rate -- 1.5e-2 against 5e-4 -- and a scheduler that does not overwrite it.
    The block still starts as an exact identity; only how fast it may leave it changes.
    """

    NEW_PARAM_LR_MULT: float = 30.0


class nnUNetTrainer_InteractiveB14(nnUNetTrainer_InteractiveArch):
    """B14 -- lightweight edit-branch decoder driven by the interaction channels."""

    NEW_PARAM_PREFIXES = ("edit_stem.", "edit_ups.", "edit_skip_projs.",
                          "edit_stages.", "edit_seg_layers.")


class nnUNetTrainer_InteractiveB13_2epochs(nnUNetTrainer_InteractiveB13):
    """Smoke-test variant."""
    NUM_EPOCHS = 2


class nnUNetTrainer_InteractiveB14_2epochs(nnUNetTrainer_InteractiveB14):
    """Smoke-test variant."""
    NUM_EPOCHS = 2
