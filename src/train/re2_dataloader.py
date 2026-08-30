"""RE2: the exact per-case PET renormalisation, applied at the store end.

## Why this shape, and not the other two

The RE row feeds the ResEncL a PET channel its weights were not trained on: LesionTracer
normalised PET with **`CTNormalization`** on SUV, our store holds a **per-case z-score**.
RE1 bridged that inside `forward` with one pooled pair of cohort constants
(``pet_renorm="ctnorm"``), which is an approximation worth +0.028 Dice to remove
(``test_pet_renorm.py``). Three ways to remove it, and only one of them is cheap on the
path that actually ships:

| | inference plumbing | exactness |
|---|---|---|
| per-tracer constants in `forward` | the tracer must reach `forward`, which takes only `x` | halves the error |
| per-case constants in `forward` | same problem, plus the constants must reach `forward` | exact |
| **this: `CTNormalization` in the plans** | **none** | **exact** |

`network(x)` is the contract that sliding-window inference, `torch.compile`, the
post-processing layer and `submission/process.py` are all written against. Passing a
per-case scalar into it means changing that signature or smuggling the value through a
side channel, in a code path that runs inside a `--network=none` container. So RE2 does
not touch the network at all (``pet_renorm="none"``); it changes the two **ends** of the
contract instead:

* **Inference** — `nnUNetPlans_re2.json` sets ``normalization_schemes[1] =
  "CTNormalization"``. `src/predictor.py` already resolves the scheme by name out of the
  configuration manager (`cm.normalization_schemes[c]`), so the container path applies
  *exactly* LesionTracer's own channel-1 normalisation,
  ``(clip(SUV, 1.0433, 51.211) - 7.0638) / 7.9604``, with **zero new code**. Note this is
  also strictly simpler than the ZScore path it replaces: `CTNormalization` uses global
  fingerprint constants, so unlike a z-score it is invariant to the crop box and needs
  none of the `pet_norm_correction` bookkeeping the store build had to do.
* **Training** — the store on disk still holds the z-score, so it is inverted here, per
  case, with that case's **own** ``mu_full``/``sd_full``. Those are the exact constants
  `build_store.py` normalised with (it restores the full-volume statistics after the body
  crop), they are already in every case's `.pkl`, and `properties` is in scope in
  `generate_train_batch` right where the patch is cropped. Nothing is estimated.

The two ends meet at ``SUV``: training computes ``CTNorm(clip(z*sd + mu))`` and inference
computes ``CTNorm(clip(SUV))``. They are the same function of the same quantity, and
`test_re2_renorm.py` asserts it numerically on real cases. The one residual difference is
that the store is **float16**, so ``z*sd + mu`` recovers SUV to ~1e-3 relative rather than
exactly; that quantisation is already in every row we have trained and is not specific to
RE2.

The remap is applied to the **cropped patch**, not the loaded case: the store is blosc2
and `load_case` hands back lazily-sliced arrays precisely so that one chunk is read per
patch. Materialising a case to remap it would undo that.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import torch

from nnunetv2.training.dataloading.data_loader import nnUNetDataLoader

try:
    from .networks_re import PET_LO, PET_HI, PET_MEAN, PET_STD
except ImportError:  # flat import
    from networks_re import PET_LO, PET_HI, PET_MEAN, PET_STD  # type: ignore

__all__ = ["pet_store_to_ctnorm", "case_pet_constants", "RE2DataLoader"]


def pet_store_to_ctnorm(z, mu: float, sd: float):
    """Store z-score -> LesionTracer's channel-1 value. Works on torch or numpy.

    The single definition both the trainer and the equivalence test call, so there is
    one place where this arithmetic lives.
    """
    suv = z * float(sd) + float(mu)
    if isinstance(suv, torch.Tensor):
        suv = suv.clamp(PET_LO, PET_HI)
    else:
        suv = np.clip(suv, PET_LO, PET_HI)
    return (suv - PET_MEAN) / PET_STD


def case_pet_constants(properties: dict) -> Tuple[float, float]:
    """(mu_full, sd_full) for a case, or raise -- never silently fall back.

    A silent fallback to cohort medians would make a handful of cases train on a
    different input distribution than the rest with nothing in the log to show it.
    """
    c = properties.get("pet_norm_correction")
    if not isinstance(c, dict) or "mu_full" not in c or "sd_full" not in c:
        raise RuntimeError(
            "RE2 needs the per-case pet_norm_correction from the store properties, got "
            f"{c!r}. That case was built with 'skipped: crop_to_nonzero changed shape'; "
            "exclude it from the split or fall back to nnUNetPlans_re.")
    sd = float(c["sd_full"])
    if not np.isfinite(sd) or sd <= 0:
        raise RuntimeError(f"RE2: non-positive per-case PET sd {sd!r}")
    return float(c["mu_full"]), sd


class RE2DataLoader(nnUNetDataLoader):
    """nnU-Net's dataloader with the per-case PET remap between crop and transforms.

    `generate_train_batch` is copied from nnU-Net rather than wrapped because the remap
    has to land between the crop and the transforms: after the crop so it stays lazy
    over blosc2 chunks, before the transforms so the interaction simulation and the
    network see the same channel. `test_re2_renorm.py` pins the copy against the
    installed nnU-Net, so a version bump that changes the method is caught rather than
    silently diverging.

    Defined at module level and **not** produced by a factory: `NonDetMultiThreadedAugmenter`
    pickles the loader out to its worker processes, and pickle resolves a class by
    `module.__name__` lookup. A class built inside a function -- or one whose `__name__`
    does not match the global it is bound to -- fails that lookup, and the failure
    surfaces as workers that never produce a batch rather than as an exception. This is
    the same constraint `s1_sampler.py` documents for `S1RecordingDatasetBlosc2`.
    """

    PET_CHANNEL = 1

    def generate_train_batch(self):
        import numpy as _np
        from threadpoolctl import threadpool_limits
        from nnunetv2.training.dataloading.data_loader import crop_and_pad_nd

        selected_keys = self.get_indices()
        data_all = seg_all = None
        with torch.no_grad():
            with threadpool_limits(limits=1, user_api=None):
                for j, i in enumerate(selected_keys):
                    force_fg = self.get_do_oversample(j)
                    data, seg, seg_prev, properties = self._data.load_case(i)
                    shape = data.shape[1:]
                    bbox_lbs, bbox_ubs = self.get_bbox(shape, force_fg,
                                                      properties['class_locations'])
                    bbox = [[a, b] for a, b in zip(bbox_lbs, bbox_ubs)]

                    data_cropped = torch.from_numpy(crop_and_pad_nd(data, bbox, 0)).float()
                    # ---- the one added step -------------------------------------
                    mu, sd = case_pet_constants(properties)
                    c = self.PET_CHANNEL
                    data_cropped[c] = pet_store_to_ctnorm(data_cropped[c], mu, sd)
                    # -------------------------------------------------------------
                    seg_cropped = torch.from_numpy(
                        crop_and_pad_nd(seg, bbox, -1, cast_cropped_to=_np.int16)
                    ).to(torch.int16)
                    if seg_prev is not None:
                        seg_prev_cropped = torch.from_numpy(
                            crop_and_pad_nd(seg_prev, bbox, -1, cast_cropped_to=_np.int16)
                        ).to(torch.int16)
                        seg_cropped = torch.cat((seg_cropped, seg_prev_cropped), 0)

                    if self.transforms is not None:
                        tr = self.transforms(**{'image': data_cropped,
                                                'segmentation': seg_cropped})
                        data_cropped, seg_cropped = tr['image'], tr['segmentation']
                    if data_all is None:
                        data_all = torch.zeros((len(selected_keys), *data_cropped.shape),
                                               dtype=torch.float32)
                        if isinstance(seg_cropped, list):
                            seg_all = [torch.zeros((len(selected_keys), *s.shape),
                                                   dtype=torch.int16) for s in seg_cropped]
                        else:
                            seg_all = torch.zeros((len(selected_keys), *seg_cropped.shape),
                                                  dtype=torch.int16)
                    data_all[j] = data_cropped
                    if isinstance(seg_cropped, list):
                        for k, s in enumerate(seg_cropped):
                            seg_all[k][j] = s
                    else:
                        seg_all[j] = seg_cropped
        return {'data': data_all, 'target': seg_all, 'keys': selected_keys}

