"""Launch gate: a freshly grafted variant must reproduce the source model's logits.

The rule for every architecture row is "epoch 0 is the source model". The file-based
version of that check -- cache one forward pass of stock B10 to disk, assert
``< 1e-5`` against it in the trainer -- does not survive contact with cuDNN: with
``cudnn.benchmark = True`` (which ``nnunetv2.run.run_training`` sets) the convolution
algorithm is autotuned per process against the free VRAM at that moment, and two
processes therefore disagree by ~1e-2 on logits whose magnitude is ~23. Measured on
this box, the **zero-change control** failed the 1e-5 assertion against a cached
reference at 8.7e-3, i.e. the file-based gate is a float-noise detector, not an
architecture check.

This gate removes the noise instead of tolerating it: it rebuilds the *source*
network from the base plans **inside the training process**, loads the same
checkpoint into it, and compares the two forward passes there. Identical convolutions
of identical shape then get the identical autotuned algorithm, so a correct variant
scores exactly 0.0 and the 1e-5 tolerance means what it says.

Mix into a trainer ahead of ``nnUNetTrainer_InteractiveArch``; it overrides that
class's ``_assert_identity_at_init`` hook and needs no other change.
"""

from __future__ import annotations

import json
import os
from typing import Optional

import torch

__all__ = ["SourceIdentityGateMixin"]


class SourceIdentityGateMixin:
    """In-process identity assertion against the source checkpoint's own network."""

    #: tolerance of the assertion; 0.0 is the expected value
    IDENTITY_TOL: float = 1e-5
    #: batch size of the fixed probe batch
    IDENTITY_BATCH: int = 1

    def _identity_base_plans(self) -> Optional[str]:
        """Plans file that builds the *source* architecture.

        ``nnUNet_identity_base_plans`` wins; otherwise the interactive plans next to
        the variant's own plans, which is the file every row continues from.
        """
        env = os.environ.get("nnUNet_identity_base_plans")
        if env:
            return env
        folder = getattr(self, "preprocessed_dataset_folder_base", None) \
            or os.path.dirname(getattr(self, "preprocessed_dataset_folder", ""))
        cand = os.path.join(folder, "nnUNetPlans_interactive.json")
        return cand if os.path.isfile(cand) else None

    def _identity_probe_batch(self) -> torch.Tensor:
        """A fixed patch shaped like the real input: CT/PET z-scored, guidance in
        [0, 1], previous mask binary and sparse."""
        patch = [int(p) for p in self.configuration_manager.patch_size]
        c = int(self.num_input_channels)
        b = int(self.IDENTITY_BATCH)
        g = torch.Generator().manual_seed(0)
        x = torch.randn(b, c, *patch, generator=g)
        if c >= 4:
            x[:, 2:4] = torch.rand(b, 2, *patch, generator=g)
        if c >= 5:
            x[:, 4] = (torch.rand(b, *patch, generator=g) > 0.98).float()
        return x

    def _assert_identity_at_init(self, mod) -> None:
        plans_file = self._identity_base_plans()
        source = getattr(self, "pretrained_weights_file", None)
        if not plans_file or not source:
            self.print_to_log_file(
                "[gate] WARNING: no base plans or no source checkpoint -- the identity "
                "assertion was NOT run")
            return

        from nnunetv2.utilities.get_network_from_plans import get_network_from_plans

        with open(plans_file) as f:
            plans = json.load(f)
        arch = plans["configurations"][self.configuration_name]["architecture"]
        ref_net = get_network_from_plans(
            arch["network_class_name"], arch["arch_kwargs"], arch["_kw_requires_import"],
            int(self.num_input_channels), int(self.label_manager.num_segmentation_heads),
            allow_init=True, deep_supervision=self.enable_deep_supervision)
        ck = torch.load(source, map_location="cpu", weights_only=False)
        ref_net.load_state_dict(ck["network_weights"], strict=True)

        x = self._identity_probe_batch().to(self.device)
        was_training = mod.training
        ref_net = ref_net.to(self.device).eval()
        mod.eval()
        try:
            with torch.no_grad():
                ref = ref_net(x)
                out = mod(x)
        finally:
            if was_training:
                mod.train()
            ref_net.to("cpu")
            del ref_net
            torch.cuda.empty_cache()

        if not isinstance(ref, (list, tuple)):
            ref = [ref]
        if not isinstance(out, (list, tuple)):
            out = [out]
        # a variant may append extra tensors (N1's coarse map); zip stops at the
        # segmentation outputs, which are the ones that must be unchanged
        d = max((a.float() - b.float()).abs().max().item() for a, b in zip(out, ref))
        if d >= self.IDENTITY_TOL:
            raise RuntimeError(
                f"[gate] identity assertion FAILED: max |logit diff| {d:.3e} >= "
                f"{self.IDENTITY_TOL:g} against {source} built from {plans_file}")
        self.print_to_log_file(
            f"[gate] identity assertion PASS: max |logit diff| {d:.3e} < "
            f"{self.IDENTITY_TOL:g} on {tuple(x.shape)}, in-process against "
            f"{os.path.basename(source)} built from {os.path.basename(plans_file)}")
