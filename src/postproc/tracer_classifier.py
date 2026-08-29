"""FDG vs PSMA detection from the PET, and from the CT when it is available.

The container names cases by UUID, so the tracer has to come from the images.  FDG has a
hot, large brain; PSMA a dark brain with very hot kidneys and bladder.  The score is
``log10(head_p99 / trunk_p999) + w * log10(head_hot_blob_ml / 100)`` and ``score >
decision_threshold`` means FDG; both terms are ratios, so scanner calibration drops out.
Slabs run along the superior axis of the affine, measured against the body extent.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import cc3d
import numpy as np

from .utils import voxel_volume_ml

__all__ = ["superior_axis", "tracer_features", "guess_tracer"]

DEFAULT_PARAMS: Dict[str, float] = {
    "head_fraction": 0.15,       # top slab of the body used as "the head"
    "trunk_lo": 0.15,            # trunk slab, as fractions of the body extent
    "trunk_hi": 0.75,
    "high_suv": 4.0,             # "high uptake" threshold for the head blob
    "body_hu": -500.0,           # CT threshold that finds the patient
    "body_suv": 0.2,             # PET fallback when no CT is available
    "blob_weight": 0.26,         # w in the score, fitted on the validation set
    "blob_ref_ml": 100.0,
    "decision_threshold": -0.63,  # midpoint of the measured gap [-0.737, -0.533]
    "margin": 0.20,              # measured class separation, in decades
    "downsample": 2,             # stride for the connected-component pass
    # legacy keys, kept so an existing sweep config does not break
    "brain_min_volume_ml": 150.0,
    "brain_min_suv": 5.0,
    "psma_body_suv": 25.0,
}


def superior_axis(affine: Optional[np.ndarray] = None) -> Tuple[int, int]:
    """Return ``(axis, sign)`` such that increasing ``axis * sign`` goes towards the head."""
    if affine is None:
        return 2, 1
    try:
        import nibabel as nib

        codes = nib.aff2axcodes(np.asarray(affine))
    except Exception:
        return 2, 1
    for ax, code in enumerate(codes):
        if code == "S":
            return ax, 1
        if code == "I":
            return ax, -1
    return 2, 1


def _body_extent(pet, ct, axis, params, stride: int = 4) -> Tuple[int, int]:
    """(lo, hi) index of the patient along ``axis``, from the CT if given, else the PET."""
    other = tuple(a for a in range(3) if a != axis)
    sl: List[Any] = [slice(None, None, stride)] * 3
    sl[axis] = slice(None)
    if ct is not None:
        occ = (np.asarray(ct[tuple(sl)], dtype=np.float32) > params["body_hu"]).sum(axis=other)
    else:
        occ = (np.asarray(pet[tuple(sl)], dtype=np.float32) > params["body_suv"]).sum(axis=other)
    nz = np.nonzero(occ > 0)[0]
    n = pet.shape[axis]
    if nz.size == 0:
        return 0, n - 1
    return int(nz[0]), int(nz[-1])


def tracer_features(
    pet: np.ndarray,
    spacing: Sequence[float] = (1.0, 1.0, 1.0),
    *,
    ct: Optional[np.ndarray] = None,
    affine: Optional[np.ndarray] = None,
    superior: Optional[Tuple[int, int]] = None,
    params: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """SUV statistics of the head slab and of the trunk, plus the decision score."""
    p = dict(DEFAULT_PARAMS)
    if params:
        p.update(params)
    pet = np.asarray(pet)
    axis, sign = superior if superior is not None else superior_axis(affine)
    lo, hi = _body_extent(pet, ct, axis, p)
    span = max(hi - lo, 1)
    step = max(1, int(p["downsample"]))

    def slab(a: float, b: float) -> np.ndarray:
        """The part of the body between fractions a..b of the caudal->cranial extent."""
        if sign > 0:
            i0, i1 = lo + int(a * span), lo + int(b * span)
        else:
            i0, i1 = hi - int(b * span), hi - int(a * span)
        idx: List[Any] = [slice(None, None, step)] * 3
        idx[axis] = slice(max(i0, 0), min(i1 + 1, pet.shape[axis]))
        return np.ascontiguousarray(np.asarray(pet[tuple(idx)], dtype=np.float32))

    head = slab(1.0 - p["head_fraction"], 1.0)
    trunk = slab(p["trunk_lo"], p["trunk_hi"])
    vox_ml = voxel_volume_ml(spacing) * step * step   # the slab keeps the superior axis

    blob_ml, blob_suv = 0.0, 0.0
    hot = np.ascontiguousarray(head >= p["high_suv"]).view(np.uint8)
    if hot.any():
        labels = cc3d.connected_components(hot, connectivity=26)
        counts = cc3d.statistics(labels)["voxel_counts"]
        if len(counts) > 1:
            biggest = int(np.argmax(counts[1:]) + 1)
            blob_ml = float(counts[biggest] * vox_ml)
            blob_suv = float(head[labels == biggest].max())

    def pct(a, q):
        return float(np.percentile(a, q)) if a.size else 0.0

    head_p99 = pct(head, 99)
    trunk_p999 = pct(trunk, 99.9)
    ratio = head_p99 / max(trunk_p999, 1e-3)
    score = (np.log10(max(ratio, 1e-6))
             + p["blob_weight"] * np.log10(max(blob_ml, 1.0) / p["blob_ref_ml"]))

    return {
        "superior_axis": int(axis),
        "superior_sign": int(sign),
        "body_lo": int(lo),
        "body_hi": int(hi),
        "body_span": int(span),
        "head_blob_ml": blob_ml,
        "head_blob_suv_max": blob_suv,
        "head_suv_max": float(head.max()) if head.size else 0.0,
        "head_suv_p99": head_p99,
        "head_suv_p999": pct(head, 99.9),
        "trunk_suv_p999": trunk_p999,
        "trunk_suv_max": float(trunk.max()) if trunk.size else 0.0,
        "head_over_trunk": float(ratio),
        "score": float(score),
        "global_suv_max": float(pet.max()),
        # legacy aliases, so anything that still reads the old names keeps working
        "body_suv_p999": trunk_p999,
        "body_suv_max": float(trunk.max()) if trunk.size else 0.0,
    }


def guess_tracer(
    pet: np.ndarray,
    spacing: Sequence[float] = (1.0, 1.0, 1.0),
    *,
    ct: Optional[np.ndarray] = None,
    affine: Optional[np.ndarray] = None,
    superior: Optional[Tuple[int, int]] = None,
    params: Optional[Dict[str, float]] = None,
    return_features: bool = False,
):
    """Return ``"fdg"`` / ``"psma"``, optionally with the feature dict.

    ``confidence`` is 0.5 on the decision boundary and 0.95 one margin away, so a caller
    can treat anything below ~0.6 as "unknown tracer".
    """
    p = dict(DEFAULT_PARAMS)
    if params:
        p.update(params)
    f = tracer_features(pet, spacing, ct=ct, affine=affine, superior=superior, params=p)

    d = f["score"] - p["decision_threshold"]
    tracer = "fdg" if d > 0 else "psma"
    conf = 0.5 + 0.45 * min(1.0, abs(d) / max(p["margin"], 1e-6))
    if f["head_blob_ml"] <= 0.0 and f["head_suv_p99"] <= 0.0:
        conf = min(conf, 0.4)          # nothing in the head slab: no evidence either way
        why = "empty head slab -- no evidence"
    elif tracer == "fdg":
        why = "hot, large head structure relative to the trunk (brain)"
    else:
        why = "cold or small head structure, very hot trunk (kidneys / bladder)"

    f.update({"tracer": tracer, "confidence": float(conf), "reason": why,
              "margin_decades": float(d)})
    if return_features:
        return tracer, f
    return tracer
