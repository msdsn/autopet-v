"""B18 -- B17's EVA-02 branch, made interaction-aware.

Everything about the recipe is B17's (``nnUNetTrainer_InteractiveB17``): the same
``nnUNetTrainer_InteractiveV2_negfp`` loss and interaction distribution, the same
strict graft, the same epoch-0 identity gate, the same dual optimizer -- stock SGD
(momentum 0.99, wd 3e-5) over the U-Net exactly as the C0 control, AdamW with
layer-wise lr decay over the ViT -- under ``GroupScaledPolyLRScheduler``.

**What the row changes.** One block: what the ViT is given. B17's branch reads only
channels 0 and 1 of the network input, so the token volume it adds to the stage-3
skip is identical at all six interaction iterations. The 39-case screen measured
exactly the deficit that predicts: paired Dice -0.002 at iteration 0, where the
interaction channels are zero and the two networks agree, then -0.012 to -0.017 from
iteration 2 on. ``EVAInteractiveFusionUNet`` patchifies the three interaction
channels through a zero-init embedding of its own and adds those tokens to the patch
tokens, so the branch becomes a conditional encoder whose global self-attention can
carry one click to the rest of the slab.

**The one thing the trainer has to add: a rate for the zero-init path.** Both new
modules start at zero, and ``dL/dtokens = eva_fuse^T dL/dskip``, so
``eva_interact_embed`` sees no gradient at all until ``eva_fuse`` has left zero -- the
B13 trap, which cost that row its whole run. So the rate on those two modules is a
real choice, and it was made from a measurement rather than from the schedule algebra.

The first B18 launch used ``FUSION_LR_MULT = 3.0``, argued from the poly integral: a
20-epoch schedule integrates ``int_0^20 (1-t/20)^0.9 dt = 10.5`` against the
``int_0^40 (1-t/80)^0.9 dt = 30.8`` that B17's epoch-40 snapshot received, a factor of
2.9. That argument assumed the B17 fusion was weak. It is not. Measured on five real
store patches (``/content/work/b3_real.py``), B17's ``checkpoint_final`` puts the fused
residual at **1.52x the rms of the stage-3 skip it is added to** -- the branch does not
perturb that skip, it dominates it -- with ``eva_fuse`` at norm 1.92 after 80 epochs.
The x3 run reached ``eva_fuse`` norm **3.84 by epoch 10** and a residual at **7.12x the
skip**, of which only **4.3 %** moved when a scribble was added. That is not a
measurement of "the ViT sees the interaction"; it is a measurement of a fusion driven
three times too hard, and the run was stopped at epoch 16 rather than screened.

``FUSION_LR_MULT = 1.0`` is B17's own rate, and the same numbers say it lands a
20-epoch run at ``eva_fuse`` ~1.8 against B17's 1.92 (measured: 1.53 at epoch 10). B18
then differs from B17 in exactly one thing -- the interaction tokens -- which is the row
this is supposed to be.

**But the conditional path is structurally starved, and that needs its own rate.**
``eva_interact_embed``'s gradient is proportional to ``eva_fuse``, which is itself
growing from zero, so the new path always lags the branch it rides on. Measured on the
x1 run's ``checkpoint_ep10``: ``eva_interact_embed`` norm 0.47, and the share of the
fused residual that moves when a scribble is added is **0.57 %** of a residual that is
3.49x the skip -- i.e. the interaction-driven part is 0.02x the skip, which is nothing.
A screen of that model would re-measure B17, not test the hypothesis. This is exactly
B13's failure (zero-init output projection starving the interior, `system.md` 2026-08-29
22:00 UTC), and it takes B13c's verified fix: a separate, much higher rate for the
starved module alone. ``EVA_INTERACT_LR_MULT = 30.0`` applies to
``eva_interact_embed`` only; ``eva_fuse`` stays at B17's rate, so the branch's overall
strength -- the thing that would confound the row -- is unchanged.
"""

from __future__ import annotations

import os

import torch

try:  # package import (src/train is a package)
    from .nnUNetTrainer_InteractiveB17 import nnUNetTrainer_InteractiveB17, _DualOptimizer
    from .nnUNetTrainer_InteractiveArch import GroupScaledPolyLRScheduler
except ImportError:  # flat import (folder on sys.path, e.g. nnUNet_extTrainer)
    from nnUNetTrainer_InteractiveB17 import (  # type: ignore
        nnUNetTrainer_InteractiveB17, _DualOptimizer)
    from nnUNetTrainer_InteractiveArch import GroupScaledPolyLRScheduler  # type: ignore


__all__ = [
    "nnUNetTrainer_InteractiveB18",
    "nnUNetTrainer_InteractiveB18_20epochs",
    "nnUNetTrainer_InteractiveB18_40epochs",
    "nnUNetTrainer_InteractiveB18_2epochs",
]


def _env_float(name: str, default: float) -> float:
    v = os.environ.get(name)
    return default if v is None or v == "" else float(v)


class nnUNetTrainer_InteractiveB18(nnUNetTrainer_InteractiveB17):
    """B18 -- interaction-conditioned 2.5D EVA-02-B branch fused into the B10 encoder."""

    NEW_PARAM_PREFIXES = ("eva.", "eva_fuse.", "eva_interact_embed.")
    NUM_EPOCHS: int = 120
    #: learning-rate multiplier of ``eva_fuse``; 1.0 = B17's own rate, which keeps the
    #: branch's overall strength identical to the row B18 is compared against
    FUSION_LR_MULT: float = 1.0
    #: learning-rate multiplier of ``eva_interact_embed`` alone. Its gradient is
    #: proportional to ``eva_fuse``, which starts at zero, so without this the new path
    #: never leaves the noise -- measured 0.57 % of the residual at epoch 10 at 1.0.
    #: 30x is B13c's verified remedy for the same zero-init starvation.
    EVA_INTERACT_LR_MULT: float = 30.0

    def configure_optimizers(self):
        """B17's dual optimizer, with the zero-init fusion path in its own SGD group.

        Three groups of parameters leave here:

        * the U-Net -- SGD, momentum 0.99, wd 3e-5, lr 5e-4, ``lr_scale`` 1.0. Bit for
          bit the C0 control's optimizer, which is what makes the row a measurement of
          the branch rather than of the recipe.
        * ``eva_fuse`` + ``eva_interact_embed`` -- the same SGD, same weight decay,
          ``lr_scale`` = ``FUSION_LR_MULT``. Same optimizer, one number different, so
          the correction is auditable.
        * the ViT -- AdamW with the EVA-02 fine-tuning ladder, untouched from B17.
        """
        eva_lr = _env_float("nnUNet_b17_eva_lr", self.EVA_LR)
        decay = _env_float("nnUNet_b17_eva_layer_decay", self.EVA_LAYER_DECAY)
        eva_wd = _env_float("nnUNet_b17_eva_wd", self.EVA_WEIGHT_DECAY)
        fuse_mult = _env_float("nnUNet_b18_fusion_lr_mult", self.FUSION_LR_MULT)
        ie_mult = _env_float("nnUNet_b18_interact_lr_mult", self.EVA_INTERACT_LR_MULT)

        net = self._unwrapped_network()
        if not hasattr(net, "eva_param_groups") or not hasattr(net, "eva_fusion_parameters"):
            self.print_to_log_file(
                "[b18] the plans did not build an EVAInteractiveFusionUNet -- falling "
                "back to the B17 optimizer")
            return super().configure_optimizers()

        # --- AdamW over the EVA blocks, layer-wise lr decay, no wd on 1-D params
        adam_groups = []
        ladder = net.eva_param_groups(eva_lr, decay)
        for g in ladder:
            decay_p = [p for p in g["params"] if p.ndim > 1]
            nodecay_p = [p for p in g["params"] if p.ndim <= 1]
            scale = g["lr"] / self.initial_lr      # GroupScaledPolyLR multiplies by this
            if decay_p:
                adam_groups.append({"params": decay_p, "lr": g["lr"],
                                    "lr_scale": scale, "weight_decay": eva_wd})
            if nodecay_p:
                adam_groups.append({"params": nodecay_p, "lr": g["lr"],
                                    "lr_scale": scale, "weight_decay": 0.0})
        adamw = torch.optim.AdamW(adam_groups, lr=eva_lr, betas=tuple(self.EVA_BETAS),
                                  eps=self.EVA_EPS)

        # --- the stock nnU-Net SGD, split in two so the zero-init path can be faster
        eva_ids = {id(p) for p in net.eva.parameters()}
        ie = [p for p in net.eva_interact_embed.parameters() if p.requires_grad]
        ie_ids = {id(p) for p in ie}
        fuse = [p for p in net.eva_fuse.parameters() if p.requires_grad]
        fuse_ids = {id(p) for p in fuse}
        rest = [p for p in net.parameters()
                if p.requires_grad and id(p) not in eva_ids and id(p) not in fuse_ids
                and id(p) not in ie_ids]
        sgd_groups = [{"params": rest, "lr": self.initial_lr, "lr_scale": 1.0}]
        if fuse:
            sgd_groups.append({"params": fuse, "lr": self.initial_lr * fuse_mult,
                               "lr_scale": float(fuse_mult)})
        if ie:
            sgd_groups.append({"params": ie, "lr": self.initial_lr * ie_mult,
                               "lr_scale": float(ie_mult)})
        sgd = torch.optim.SGD(sgd_groups, self.initial_lr, weight_decay=self.weight_decay,
                              momentum=0.99, nesterov=True)

        # SGD first: nnU-Net logs param_groups[0]['lr'], which should stay B10's number
        optimizer = _DualOptimizer(sgd, adamw)
        lr_scheduler = GroupScaledPolyLRScheduler(optimizer, self.initial_lr, self.num_epochs)

        n_rest = sum(p.numel() for p in rest)
        n_fuse = sum(p.numel() for p in fuse)
        n_ie = sum(p.numel() for p in ie)
        n_eva = sum(p.numel() for g in ladder for p in g["params"])
        n_frozen = sum(p.numel() for p in net.eva.parameters() if not p.requires_grad)
        self.print_to_log_file(
            f"[b18] SGD(momentum 0.99, wd {self.weight_decay:g}) over {n_rest / 1e6:.2f} M "
            f"U-Net parameters at lr {self.initial_lr:g}  +  {n_fuse / 1e6:.2f} M "
            f"zero-init eva_fuse parameters at lr {self.initial_lr * fuse_mult:g} "
            f"(x{fuse_mult:g})  +  {n_ie / 1e6:.2f} M zero-init eva_interact_embed "
            f"parameters at lr {self.initial_lr * ie_mult:g} (x{ie_mult:g})  +  AdamW(betas "
            f"{tuple(self.EVA_BETAS)}, eps {self.EVA_EPS:g}, wd {eva_wd:g} on >1-D only) "
            f"over {n_eva / 1e6:.2f} M EVA parameters in {len(ladder)} layers at base lr "
            f"{eva_lr:g}, layer decay {decay:g}; {n_frozen / 1e6:.2f} M EVA parameters "
            f"frozen (their activations are still in the graph, so the interaction "
            f"embedding gets a gradient)")
        for g in ladder:
            self.print_to_log_file(
                f"[b18]   layer {g['layer']:>2}  lr {g['lr']:.3e}  (x{g['scale']:.4f})  "
                f"{sum(p.numel() for p in g['params']) / 1e6:.2f} M")
        return optimizer, lr_scheduler


class nnUNetTrainer_InteractiveB18_20epochs(nnUNetTrainer_InteractiveB18):
    """The screening schedule: 20 epochs, snapshot at 10, screened at both."""
    NUM_EPOCHS = 20


class nnUNetTrainer_InteractiveB18_40epochs(nnUNetTrainer_InteractiveB18):
    """Continuation schedule if the screen passes."""
    NUM_EPOCHS = 40


class nnUNetTrainer_InteractiveB18_2epochs(nnUNetTrainer_InteractiveB18):
    """Smoke-test variant."""
    NUM_EPOCHS = 2
