"""RE-N -- the RE (ResEncL) backbone with N1's presence-prior gate.

Everything about the *method* is the RE row (which is B10's recipe): the same online
error-driven scribble simulation, the same k distribution, the same DC+CE at
``smooth = 0`` plus the lesion-free false-positive term, the same store, the same
192^3 patch, the same ``pet_renorm="ctnorm"``. What is added is exactly what N1 added
to the control backbone:

* a ``Conv3d(256, 1, 1)`` on encoder stage 3 (24^3 grid, cell 8^3 voxels = 6.370 mL,
  the same physical cell N1's ``pos_weight`` was measured at), zero weight and zero
  bias, whose coarse log-odds map is added to the foreground logit of the final
  output and of every deep-supervision output at its own scale; and
* ``0.5 * BCEWithLogits(gate, adaptive_max_pool3d(label, 24^3))`` with a measured
  ``pos_weight``, on top of the unchanged compound loss.

257 new parameters against RE's 102.35 M.

## The three launch assertions

The committee's finding was that inheriting these implicitly is how they get lost, so
all three are invoked explicitly from one overridden hook:

1. **Zero head after ``apply(initialize)``.** nnU-Net re-initialises the network after
   construction; ``networks_n1._zero_and_freeze`` tags the module and
   ``ResEncPresenceGateUNet.initialize`` skips tagged modules. Asserted, not assumed.
2. **Stock equivalence.** A stock ``ResidualEncoderUNet`` built from the same
   ``arch_kwargs`` at the same ``input_channels``, carrying this network's weights,
   must reproduce the logits **exactly** on the pre-remapped input. Built in the same
   process, so "exactly" means 0.0 rather than a tolerance -- a cached reference from
   another process disagrees at ~1e-2 because ``cudnn.benchmark`` autotunes per
   process. With the gate at zero this single check covers both "the subclass is
   stock + pet_renorm" and "the gate contributes nothing".
3. **Source equivalence.** ``SourceIdentityGateMixin`` rebuilds the *source* network
   from ``nnUNetPlans_re.json`` in-process, loads the same checkpoint into it and
   compares -- so epoch 0 is the RE checkpoint we grafted, not merely some stock
   network.

RE's own "randomise channels 2-4 => 0.000" assertion is deliberately **not** reused:
it holds only for a graft straight off LesionTracer, whose stem columns 2-4 are zero.
RE-N grafts from RE40/RE100, whose stem columns have trained, so that assertion would
be false. It is replaced by (2) + (3), not dropped in favour of (3) alone.

## Kill criteria, instrumented in the loop

B17 and B18 both died of a rider block that swamped the segmentation logits, and both
were diagnosed only after the fact. The ratio ``rms(gate contribution) / rms(seg
logit)`` at the final head is therefore sampled during training and is a **hard
abort** above ``REN_GATE_RATIO_MAX`` (0.25), not a number in a log; and the head's
weight norm is printed every epoch, because a gate that never engages is measuring RE
under another name.
"""

from __future__ import annotations

import os

import torch
import torch.nn.functional as F

try:  # package import (src/train is a package)
    from .identity_gate import SourceIdentityGateMixin
    from .networks_n1 import _FREEZE_FLAG
    from .networks_re import ORGAN_HEAD_PREFIX
    from .networks_ren import GATE_PREFIX
    from .nnUNetTrainer_Interactive import _env_float
    from .nnUNetTrainer_InteractiveN1 import PresenceGateAuxLoss
    from .nnUNetTrainer_InteractiveRE import nnUNetTrainer_InteractiveRE
except ImportError:  # flat import (folder on sys.path, e.g. nnUNet_extTrainer)
    from identity_gate import SourceIdentityGateMixin  # type: ignore
    from networks_n1 import _FREEZE_FLAG  # type: ignore
    from networks_re import ORGAN_HEAD_PREFIX  # type: ignore
    from networks_ren import GATE_PREFIX  # type: ignore
    from nnUNetTrainer_Interactive import _env_float  # type: ignore
    from nnUNetTrainer_InteractiveN1 import PresenceGateAuxLoss  # type: ignore
    from nnUNetTrainer_InteractiveRE import nnUNetTrainer_InteractiveRE  # type: ignore

__all__ = [
    "GateRatioAuxLoss",
    "nnUNetTrainer_InteractiveREN",
    "nnUNetTrainer_InteractiveREN_20epochs",
    "nnUNetTrainer_InteractiveREN_60epochs",
    "nnUNetTrainer_InteractiveREN_2epochs",
]


class GateRatioAuxLoss(PresenceGateAuxLoss):
    """N1's auxiliary BCE, plus a periodic measurement of the gate's share of the logit.

    The ratio is what distinguishes a prior from a takeover. Measuring it costs one
    upsample of a single-channel map, so it is sampled every ``probe_every`` steps
    rather than every step; the trainer reads ``last_ratio`` at the end of the epoch.
    """

    def __init__(self, inner, weight: float = 0.5, pos_weight: float = 1.0,
                 probe_every: int = 50):
        super().__init__(inner, weight=weight, pos_weight=pos_weight)
        self.probe_every = int(probe_every)
        self._step = 0
        self.last_ratio: float = 0.0

    def forward(self, output, target):
        gate = None
        if isinstance(output, (list, tuple)) and isinstance(target, (list, tuple)) \
                and len(output) == len(target) + 1:
            gate = output[-1]
        loss = super().forward(output, target)
        self._step += 1
        if gate is not None and self.probe_every > 0 and self._step % self.probe_every == 0:
            with torch.no_grad():
                seg = output[0]
                g = F.interpolate(gate.float(), size=seg.shape[2:], mode="trilinear",
                                  align_corners=False)
                fg = seg[:, 1:2].float()
                # seg already carries the gate; the pre-fusion logit is seg - g
                denom = (fg - g).pow(2).mean().sqrt().item()
                num = g.pow(2).mean().sqrt().item()
                self.last_ratio = num / max(denom, 1e-8)
        return loss


class nnUNetTrainer_InteractiveREN(SourceIdentityGateMixin, nnUNetTrainer_InteractiveRE):
    """RE + the presence-prior gate. Run with ``-p nnUNetPlans_ren``."""

    NUM_EPOCHS: int = 60
    #: the only tensors the variant adds to the RE checkpoint
    NEW_PARAM_PREFIXES = (GATE_PREFIX,)
    #: N1's loss weights; pos_weight is measured, never guessed
    AUX_WEIGHT: float = 0.5
    AUX_POS_WEIGHT: float = 91.96
    #: hard abort: the gate is a prior, not a second segmenter
    GATE_RATIO_MAX: float = 0.25
    GATE_PROBE_EVERY: int = 50

    # ------------------------------------------------------------------
    # loss
    # ------------------------------------------------------------------
    def _build_loss(self):
        inner = super()._build_loss()
        w = _env_float("N1_AUX_W", self.AUX_WEIGHT)
        pw = _env_float("N1_AUX_POS_WEIGHT", self.AUX_POS_WEIGHT)
        if os.environ.get("N1_AUX_POS_WEIGHT") in (None, ""):
            self.print_to_log_file(
                f"[REN] WARNING: N1_AUX_POS_WEIGHT is not set, using {pw} measured on the "
                f"112x160x128 sampler; run train.measure_n1_prior --plans nnUNetPlans_re "
                f"for the 192^3 value")
        self.gate_ratio_max = _env_float("REN_GATE_RATIO_MAX", self.GATE_RATIO_MAX)
        self.print_to_log_file(
            f"[REN] presence gate on stage {self._gate_stage()}: aux BCE weight={w}, "
            f"pos_weight={pw}, hard abort if rms(gate)/rms(seg logit) > {self.gate_ratio_max}")
        return GateRatioAuxLoss(inner, weight=w, pos_weight=pw,
                                probe_every=self.GATE_PROBE_EVERY)

    def _gate_stage(self):
        arch = self.configuration_manager.configuration["architecture"]
        return arch["arch_kwargs"].get("gate_stage", 3)

    # ------------------------------------------------------------------
    # weight loading: RE's strict graft, widened by exactly the new prefixes
    # ------------------------------------------------------------------
    def initialize(self):
        """RE's graft, but tolerating the tensors the gate adds.

        ``networks_re.graft_lesiontracer_state_dict`` raises on **any** missing
        tensor, which is right for RE and wrong here: the head is new by
        construction. The rest of the contract is kept exactly -- organ heads
        dropped, no shape mismatch, no unexpected source tensor, and nothing missing
        except the declared prefixes.
        """
        already = self.was_initialized
        wanted = self.pretrained_weights_file
        self.pretrained_weights_file = None       # keep the inherited loaders out
        from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
        nnUNetTrainer.initialize(self)
        self.pretrained_weights_file = wanted

        if self._continue_training and wanted:
            self.print_to_log_file(
                "[REN] --c requested: ignoring nnUNet_interactive_pretrained, "
                "the checkpoint on disk wins")
            self.pretrained_weights_file = None
            return
        if already or not wanted:
            return

        mod = self._unwrapped_network()
        ck = torch.load(wanted, map_location="cpu", weights_only=False)
        sd = {k: v for k, v in ck["network_weights"].items()
              if not k.startswith(ORGAN_HEAD_PREFIX)}
        dropped = len(ck["network_weights"]) - len(sd)

        own = mod.state_dict()
        bad = [k for k in sd if k in own and tuple(own[k].shape) != tuple(sd[k].shape)]
        if bad:
            raise RuntimeError(f"[REN] shape mismatch on {len(bad)} source tensors, e.g. "
                               f"{[(k, tuple(sd[k].shape), tuple(own[k].shape)) for k in bad[:4]]}")
        missing, unexpected = mod.load_state_dict(sd, strict=False)
        if unexpected:
            raise RuntimeError(f"[REN] {len(unexpected)} source tensors not consumed: "
                               f"{list(unexpected)[:8]}")
        stray = [k for k in missing
                 if not any(k.startswith(p) for p in self.NEW_PARAM_PREFIXES)]
        if stray:
            raise RuntimeError(f"[REN] {len(stray)} network tensors outside the declared "
                               f"new prefixes {self.NEW_PARAM_PREFIXES} were not in the "
                               f"source checkpoint: {stray[:8]}")
        self.print_to_log_file(
            f"[REN] grafted {len(sd)} tensors from {wanted} "
            f"(trainer={ck.get('trainer_name')}, epoch={ck.get('current_epoch')}); "
            f"{dropped} organ-head tensors dropped; new tensors kept at zero: "
            f"{sorted(missing)}")
        self._assert_identity_at_init(mod)

    # ------------------------------------------------------------------
    # launch gate: all three assertions, explicitly
    # ------------------------------------------------------------------
    def _identity_base_plans(self):
        """The source architecture is RE, not the PlainConvUNet default of the mixin."""
        env = os.environ.get("nnUNet_identity_base_plans")
        if env:
            return env
        folder = getattr(self, "preprocessed_dataset_folder_base", None) \
            or os.path.dirname(getattr(self, "preprocessed_dataset_folder", ""))
        cand = os.path.join(folder, "nnUNetPlans_re.json")
        return cand if os.path.isfile(cand) else None

    def _assert_gate_is_zero(self, mod) -> None:
        head = mod.gate_head
        w = head.weight.detach().abs().max().item()
        b = head.bias.detach().abs().max().item() if head.bias is not None else 0.0
        frozen = getattr(head, _FREEZE_FLAG, False)
        if w != 0.0 or b != 0.0:
            raise RuntimeError(
                f"[REN] assertion 1 FAILED: the gate head is not zero after "
                f"apply(initialize) (|w|max {w:.3e}, |b|max {b:.3e}); the "
                f"_FREEZE_FLAG guard did not hold (flag={frozen})")
        self.print_to_log_file(
            f"[REN] assertion 1 PASS: gate_head is exactly zero after apply(initialize) "
            f"(|w|max {w:.1f}, |b|max {b:.1f}, freeze flag {frozen})")

    def _assert_stock_equivalence(self, mod) -> None:
        from nnunetv2.utilities.get_network_from_plans import get_network_from_plans

        arch = self.configuration_manager.configuration["architecture"]
        kw = {k: v for k, v in arch["arch_kwargs"].items()
              if k not in ("pet_renorm", "pet_channel", "gate_stage", "gate_class")}
        ref = get_network_from_plans(
            "dynamic_network_architectures.architectures.unet.ResidualEncoderUNet",
            kw, arch["_kw_requires_import"], int(self.num_input_channels),
            int(self.label_manager.num_segmentation_heads),
            allow_init=True, deep_supervision=self.enable_deep_supervision)
        missing, unexpected = ref.load_state_dict(mod.state_dict(), strict=False)
        gate_only = sorted(k for k in unexpected if k.startswith(GATE_PREFIX))
        if missing or sorted(unexpected) != gate_only:
            raise RuntimeError(f"[REN] assertion 2 setup FAILED: missing={list(missing)[:5]} "
                               f"unexpected={list(unexpected)[:5]}")
        x = self._identity_probe_batch().to(self.device)
        was_training = mod.training
        mod.eval()
        ref = ref.to(self.device).eval()
        try:
            with torch.no_grad():
                out, r = mod(x), ref(mod._remap_pet(x))
        finally:
            if was_training:
                mod.train()
            ref.to("cpu")
            del ref
            torch.cuda.empty_cache()
        if not isinstance(out, (list, tuple)):
            out, r = [out], [r]
        d = max((p.float() - q.float()).abs().max().item() for p, q in zip(out, r))
        if d != 0.0:
            raise RuntimeError(
                f"[REN] assertion 2 FAILED: max |logit diff| {d:.3e} against a stock "
                f"in-process ResidualEncoderUNet of the same shape carrying the same "
                f"weights; expected exactly 0.0")
        self.print_to_log_file(
            f"[REN] assertion 2 PASS: equals a stock in-process ResidualEncoderUNet + "
            f"pet_renorm={getattr(mod, 'pet_renorm', 'n/a')} at {d:.1f} on "
            f"{tuple(x.shape)} ({len(gate_only)} gate tensors are the only addition)")

    def _assert_identity_at_init(self, mod) -> None:
        self._assert_gate_is_zero(mod)
        self._assert_stock_equivalence(mod)
        SourceIdentityGateMixin._assert_identity_at_init(self, mod)

    # ------------------------------------------------------------------
    # kill criteria
    # ------------------------------------------------------------------
    def on_epoch_end(self):
        loss = getattr(self, "loss", None)
        head = getattr(self._unwrapped_network(), "gate_head", None)
        if head is not None:
            w = head.weight.detach()
            ratio = float(getattr(loss, "last_ratio", 0.0))
            terms = getattr(loss, "last_terms", {}) or {}
            self.print_to_log_file(
                f"[REN] gate: |w|max {w.abs().max().item():.3e} |w|2 {w.norm().item():.3e} "
                f"bias {float(head.bias.detach()[0]):+.3e} | "
                f"rms(gate)/rms(seg logit) {ratio:.3f} | "
                f"presence BCE {terms.get('presence_bce', float('nan')):.4f} "
                f"positive cells {terms.get('positive_cells', float('nan')):.4f}")
            limit = getattr(self, "gate_ratio_max", self.GATE_RATIO_MAX)
            if ratio > limit:
                raise RuntimeError(
                    f"[REN] ABORT at epoch {self.current_epoch}: the gate contributes "
                    f"{ratio:.3f} of the segmentation logit's rms, above the {limit} "
                    f"kill threshold. B17 reached 1.52 and B18 7.12 on this measure; "
                    f"the row is on that trajectory and is not worth its slot")
        return super().on_epoch_end()


class nnUNetTrainer_InteractiveREN_20epochs(nnUNetTrainer_InteractiveREN):
    """The screening schedule, with a hard read at epoch 10."""
    NUM_EPOCHS: int = 20


class nnUNetTrainer_InteractiveREN_60epochs(nnUNetTrainer_InteractiveREN):
    """The full continuation, if the screen wins."""
    NUM_EPOCHS: int = 60


class nnUNetTrainer_InteractiveREN_2epochs(nnUNetTrainer_InteractiveREN):
    """Smoke-test variant."""
    NUM_EPOCHS: int = 2
