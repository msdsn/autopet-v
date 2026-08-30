"""Probability-level ensemble of two or more interactive checkpoints (row E2).

Why this exists
---------------
The earlier E1 row averaged two checkpoints by handing nnU-Net two *folds* of the same
architecture, so nnU-Net averaged the softmax of two networks that shared one
``plans.json``: one patch size, one target spacing, one normalization scheme, one crop.
That trick cannot combine two *different* architectures with different plans -- B10
(``PlainConvUNet``, ``nnUNetPlans_interactive``) and RE (``ResidualEncoderUNet``,
``nnUNetPlans_re``, 192^3 patches and the in-network ``pet_renorm``).  Their preprocessed
grids do not even have the same shape, so nothing can be averaged before the export.

The one geometry every member agrees on is the *original* image grid: nnU-Net's exporter
(`convert_predicted_logits_to_segmentation_with_correct_shape`) already resamples each
member's softmax back to the un-cropped, un-resampled volume.  So this class runs every
member to completion, takes each member's foreground probability in nibabel axis order,
and forms the weighted mean there.  The ensemble's mask is `argmax` over the averaged
two-class probability, which for two classes is ``p_fg > 0.5`` -- the same tie-breaking
nnU-Net's own ``argmax(0)`` uses, so with weights ``[1, 0]`` this class reproduces
member 0 bit for bit (``test_ensemble_predictor.py``).

Interaction contract
--------------------
Every member is given the *ensemble's* own previous final mask as channel 4, never its
own.  That is what the evaluation loop hands in as ``prev_pred``; this class simply
forwards the same array to all members, so no member ever sees a state the shipped
pipeline would not produce.

Cost
----
One iteration costs the sum of the members' iterations.  The CT/PET preprocessing is
cached per member per case exactly as it is for a single model, so the marginal cost of
a second member is one more sliding window plus one more resample-back, not a second
full preprocessing of the case.
"""

from __future__ import annotations

import time
from typing import Dict, List, Optional, Sequence

import numpy as np

from predictor import InteractiveNNUNetPredictor, Predictor, _empty_scribbles

__all__ = ["EnsembleInteractivePredictor", "parse_member_spec"]


def parse_member_spec(spec: str) -> Dict[str, object]:
    """``<model_folder>[:<checkpoint>[:<weight>]]`` -> dict.

    The separator is the last one or two colons, so a Windows-style drive letter or a
    folder containing a colon is not a concern here (paths on the boxes never have one);
    we split from the right and only accept a trailing float as the weight.
    """
    parts = spec.split(":")
    folder, ckpt, weight = parts[0], "checkpoint_final.pth", None
    if len(parts) >= 2 and parts[1]:
        ckpt = parts[1]
    if len(parts) >= 3 and parts[2]:
        weight = float(parts[2])
    if len(parts) > 3:
        raise ValueError(f"cannot parse ensemble member spec {spec!r}")
    return {"model_folder": folder, "checkpoint": ckpt, "weight": weight}


class EnsembleInteractivePredictor(Predictor):
    """Weighted mean of the members' foreground softmax, on the original image grid."""

    name = "ensemble"
    #: pure function of (ct, pet, scribbles, prev_pred), like every member
    stateless = True

    def __init__(self, members: Sequence[InteractiveNNUNetPredictor],
                 weights: Optional[Sequence[float]] = None,
                 member_labels: Optional[Sequence[str]] = None):
        if not len(members):
            raise ValueError("an ensemble needs at least one member")
        self.members: List[InteractiveNNUNetPredictor] = list(members)
        if weights is None:
            weights = [1.0 / len(self.members)] * len(self.members)
        w = [float(x) for x in weights]
        if len(w) != len(self.members):
            raise ValueError(f"{len(w)} weights for {len(self.members)} members")
        s = sum(w)
        if s <= 0:
            raise ValueError(f"ensemble weights must sum to > 0, got {w}")
        # normalised, so `--ensemble_weights 3 7` and `0.3 0.7` mean the same thing
        self.weights = [x / s for x in w]
        self.member_labels = (list(member_labels) if member_labels is not None
                              else [getattr(m, "model_folder", f"member{i}")
                                    for i, m in enumerate(self.members)])
        self.last_timings: Dict[str, object] = {}
        self.last_guidance_info: Dict[str, object] = {}
        #: set by the evaluation loop through `_set_iteration`
        self.current_iteration = 0

    # -- Predictor hooks ---------------------------------------------------
    def warmup(self) -> None:
        for m in self.members:
            if hasattr(m, "warmup"):
                m.warmup()

    def close(self) -> None:
        for m in self.members:
            if hasattr(m, "close"):
                m.close()

    def cache_state_key(self, prev_pred) -> Optional[str]:
        """Channel 4 is the same array for every member, so member 0's hash is the key."""
        fn = getattr(self.members[0], "cache_state_key", None)
        return fn(prev_pred) if callable(fn) else None

    # -- main --------------------------------------------------------------
    def predict(self, ct, pet, spacing, scribbles, prev_pred=None, case_cache_dir=None,
                *, affine=None, ct_path=None, pet_path=None, case_name="case",
                return_probabilities=False):
        scribbles = scribbles or _empty_scribbles()
        shape = np.asarray(pet).shape
        t_all = time.perf_counter()

        acc: Optional[np.ndarray] = None
        per_member: List[Dict[str, object]] = []
        for m, w, label in zip(self.members, self.weights, self.member_labels):
            # the loop tells the *stack* which iteration is running; members are leaves
            if hasattr(m, "current_iteration"):
                m.current_iteration = self.current_iteration
            t0 = time.perf_counter()
            mask_m, prob_m = m.predict(
                ct, pet, spacing, scribbles, prev_pred=prev_pred,
                case_cache_dir=case_cache_dir, affine=affine, ct_path=ct_path,
                pet_path=pet_path, case_name=case_name, return_probabilities=True,
            )
            fg = self._foreground(prob_m, shape)
            if acc is None:
                acc = np.zeros(shape, dtype=np.float32)
            if acc.shape != fg.shape:
                raise ValueError(f"member {label} returned probabilities of shape "
                                 f"{fg.shape}, expected {acc.shape}")
            if w:
                # `w == 0` must not touch the accumulator at all: the zero-weight
                # member has to be bit-exactly absent, not added as `0.0 * p`
                acc += np.float32(w) * fg
            per_member.append({
                "member": label, "weight": w,
                "seconds": round(time.perf_counter() - t0, 3),
                "volume_ml_argmax": float(np.asarray(mask_m).sum()),
                "timings": dict(getattr(m, "last_timings", {}) or {}),
            })
            del mask_m, prob_m, fg

        # two classes, so nnU-Net's `argmax(0)` over (1 - p, p) is exactly `p > 0.5`
        mask = (acc > 0.5).astype(np.uint8)
        self.last_timings = {
            "total_s": round(time.perf_counter() - t_all, 3),
            "network_s": round(sum(float(x["timings"].get("network_s", 0.0))
                                   for x in per_member), 3),
            "members": per_member,
        }
        self.last_guidance_info = dict(
            getattr(self.members[0], "last_guidance_info", {}) or {})
        self.last_guidance_info["ensemble_weights"] = list(self.weights)
        if return_probabilities:
            return mask, acc
        return mask

    # ----------------------------------------------------------------------
    @staticmethod
    def _foreground(prob, shape) -> np.ndarray:
        """Foreground probability as a float32 array of `shape`.

        Members return nnU-Net's class-first probability block in nibabel axis order,
        `(n_classes, *shape)`; a member that already reduced it is accepted as is.
        """
        p = np.asarray(prob)
        if p.ndim == len(shape) + 1:
            p = p[1] if p.shape[0] > 1 else p[0]
        if p.shape != tuple(shape):
            raise ValueError(f"probability of shape {p.shape} does not match {tuple(shape)}")
        return np.ascontiguousarray(p, dtype=np.float32)


def build_ensemble(specs: Sequence[str], weights: Optional[Sequence[float]] = None,
                   member_factory=None, **common) -> EnsembleInteractivePredictor:
    """Build the members from `<folder>[:<ckpt>[:<weight>]]` specs.

    `common` is forwarded to every member: device, tile_step_size, disable_tta,
    guidance_radius, the resampling knobs -- everything that must not differ between
    members, because a difference there would be a second uncontrolled variable.
    Per-member weights given inside a spec win over `weights`.
    """
    parsed = [parse_member_spec(s) for s in specs]
    factory = member_factory or InteractiveNNUNetPredictor
    members, labels = [], []
    for spec in parsed:
        kw = dict(common)
        kw["model_folder"] = spec["model_folder"]
        kw["checkpoint_name"] = spec["checkpoint"]
        members.append(factory(**kw))
        labels.append(f"{spec['model_folder']}#{spec['checkpoint']}")
    inline = [spec["weight"] for spec in parsed]
    if any(x is not None for x in inline):
        if any(x is None for x in inline):
            raise ValueError("give a weight for every member or for none of them")
        w = inline
    else:
        w = list(weights) if weights else None
    return EnsembleInteractivePredictor(members, w, member_labels=labels)
