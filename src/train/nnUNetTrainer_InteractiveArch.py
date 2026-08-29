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

try:  # package import (src/train is a package)
    from .nnUNetTrainer_InteractiveV2 import nnUNetTrainer_InteractiveV2_negfp
    from .networks import graft_state_dict
except ImportError:  # flat import (folder on sys.path, e.g. nnUNet_extTrainer)
    from nnUNetTrainer_InteractiveV2 import nnUNetTrainer_InteractiveV2_negfp  # type: ignore
    from networks import graft_state_dict  # type: ignore


__all__ = [
    "nnUNetTrainer_InteractiveArch",
    "nnUNetTrainer_InteractiveB13",
    "nnUNetTrainer_InteractiveB14",
    "nnUNetTrainer_InteractiveB13_2epochs",
    "nnUNetTrainer_InteractiveB14_2epochs",
]


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
