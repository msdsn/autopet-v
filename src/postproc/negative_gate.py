"""Lesion-free gate: decide whether to replace the prediction with an empty mask.

A lesion-absent case never receives a scribble, is excluded from DMM, and scores Dice
1.0 only if the prediction is empty -- so the gate has to fire at every iteration, not
just at iteration 0.  Hence the ``require_no_scribbles`` guard rather than an iteration
index: it is recoverable from the input alone when no state directory exists.  Defaults
are the leave-one-out fit of ``tools/gate_sweep.py`` on 100 validation cases.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import cc3d
import numpy as np

from .config import NegativeGateConfig
from .utils import foreground_prob, voxel_volume_ml

__all__ = ["component_stats", "negative_gate_features", "is_probably_negative"]


def _superior(affine):
    from .tracer_classifier import superior_axis
    return superior_axis(affine)


def _body_extent(ct: Optional[np.ndarray], axis: int, n: int, stride: int = 4,
                 body_hu: float = -500.0):
    """(lo, hi) index of the patient along ``axis``; the full axis when no CT is given."""
    if ct is None:
        return 0, n - 1
    sl: List[Any] = [slice(None, None, stride)] * 3
    sl[axis] = slice(None)
    sub = np.asarray(ct[tuple(sl)], dtype=np.float32)
    other = tuple(a for a in range(3) if a != axis)
    occupied = (sub > body_hu).sum(axis=other)
    nz = np.nonzero(occupied > 0)[0]
    if nz.size == 0:
        return 0, n - 1
    return int(nz[0]), int(nz[-1])


def component_stats(
    mask: np.ndarray,
    pet: np.ndarray,
    prob: Optional[np.ndarray] = None,
    spacing: Sequence[float] = (1.0, 1.0, 1.0),
    *,
    ct: Optional[np.ndarray] = None,
    affine: Optional[np.ndarray] = None,
    connectivity: int = 18,
    shell_radius_vox: int = 4,
    labels: Optional[np.ndarray] = None,
) -> List[Dict[str, Any]]:
    """Per-connected-component statistics of a predicted mask, largest component first.

    ``z_frac`` is the centroid along the superior axis normalised to the CT body extent
    (0 = caudal, 1 = cranial); without ``ct``/``affine`` it falls back to the raw axis,
    which is monotone but not comparable across fields of view.  ``shell_suv_max`` is
    the hottest PET voxel around -- but not inside -- the component, which separates the
    rim of a hot structure (bladder, kidney, bowel) from a lesion in cool tissue.
    """
    m = np.ascontiguousarray(np.asarray(mask) > 0)
    if not m.any():
        return []
    vox_ml = voxel_volume_ml(spacing)
    pet = np.asarray(pet)
    prob_fg = foreground_prob(prob, m.shape) if prob is not None else None

    if labels is None:
        labels = cc3d.connected_components(m.view(np.uint8), connectivity=connectivity)
    axis, sign = _superior(affine)
    lo_b, hi_b = _body_extent(ct, axis, m.shape[axis])
    span = max(hi_b - lo_b, 1)

    idx_all = np.argwhere(m).astype(np.int64)
    lab_all = labels[tuple(idx_all.T)]
    pet_all = pet[tuple(idx_all.T)].astype(np.float32)
    prob_all = (prob_fg[tuple(idx_all.T)].astype(np.float32) if prob_fg is not None
                else np.full(idx_all.shape[0], np.nan, dtype=np.float32))
    ct_all = (np.asarray(ct)[tuple(idx_all.T)].astype(np.float32) if ct is not None
              else np.full(idx_all.shape[0], np.nan, dtype=np.float32))

    out: List[Dict[str, Any]] = []
    for cid in np.unique(lab_all):
        if cid == 0:
            continue
        sel = lab_all == cid
        idx = idx_all[sel]
        s = pet_all[sel]
        p = prob_all[sel]
        cen = idx.mean(0)
        zf = (float(cen[axis]) - lo_b) / span
        if sign < 0:
            zf = 1.0 - zf
        lo = np.maximum(idx.min(0) - shell_radius_vox, 0)
        hi = np.minimum(idx.max(0) + shell_radius_vox + 1, np.asarray(m.shape))
        box = pet[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
        ring = np.ones(box.shape, dtype=bool)
        ring[tuple((idx - lo).T)] = False
        out.append({
            "id": int(cid),
            "n_voxels": int(sel.sum()),
            "volume_ml": float(sel.sum() * vox_ml),
            "suv_max": float(s.max()),
            "suv_mean": float(s.mean()),
            "prob_max": float(p.max()) if (prob_fg is not None and p.size) else float("nan"),
            "prob_mean": float(p.mean()) if (prob_fg is not None and p.size) else float("nan"),
            "ct_mean_hu": float(ct_all[sel].mean()) if ct is not None else float("nan"),
            "centroid": [float(c) for c in cen],
            "z_frac": float(zf),
            "shell_suv_max": float(np.asarray(box, dtype=np.float32)[ring].max()) if ring.any() else 0.0,
        })
    out.sort(key=lambda c: -c["volume_ml"])
    return out


def negative_gate_features(
    mask: np.ndarray,
    pet: np.ndarray,
    prob: Optional[np.ndarray] = None,
    spacing: Sequence[float] = (1.0, 1.0, 1.0),
    *,
    connectivity: int = 18,
    ct: Optional[np.ndarray] = None,
    affine: Optional[np.ndarray] = None,
    with_components: bool = False,
) -> Dict[str, Any]:
    """Case-level statistics used by the gate (cheap: one cc3d pass + bbox maxima)."""
    m = np.ascontiguousarray(np.asarray(mask) > 0)
    vox_ml = voxel_volume_ml(spacing)
    n = int(m.sum())
    feats: Dict[str, Any] = {
        "total_volume_ml": float(n * vox_ml),
        "n_components": 0,
        "largest_component_ml": 0.0,
        "suv_max_in_mask": 0.0,
        "prob_max_in_mask": None,
        "prob_mean_in_mask": None,
        "soft_volume_ml": 0.0,
        "prob_max_global": None,
    }
    prob_fg = foreground_prob(prob, m.shape) if prob is not None else None
    if prob_fg is not None:
        feats["prob_max_global"] = float(prob_fg.max())

    if n == 0:
        if prob_fg is not None:
            feats["prob_max_in_mask"] = 0.0
            feats["prob_mean_in_mask"] = 0.0
        if with_components:
            feats["components"] = []
        return feats

    pet = np.asarray(pet)
    feats["suv_max_in_mask"] = float(pet[m].max())
    if prob_fg is not None:
        pm = prob_fg[m]
        feats["prob_max_in_mask"] = float(pm.max())
        feats["prob_mean_in_mask"] = float(pm.mean())
        # "soft volume": the volume the network would claim if every voxel counted only
        # by its confidence.  A speck the network is unsure about has a tiny one.
        feats["soft_volume_ml"] = float(pm.sum() * vox_ml)

    labels = cc3d.connected_components(m.view(np.uint8), connectivity=connectivity)
    counts = cc3d.statistics(labels)["voxel_counts"]
    if len(counts) > 1:
        feats["n_components"] = int(np.count_nonzero(counts[1:]))
        feats["largest_component_ml"] = float(counts[1:].max() * vox_ml)
    if with_components:
        feats["components"] = component_stats(m, pet, prob_fg, spacing, ct=ct,
                                              affine=affine, connectivity=connectivity,
                                              labels=labels)
    return feats


def is_probably_negative(
    mask: np.ndarray,
    pet: np.ndarray,
    prob: Optional[np.ndarray] = None,
    params: Optional[NegativeGateConfig] = None,
    *,
    spacing: Sequence[float] = (1.0, 1.0, 1.0),
    iteration: int = 0,
    n_tumor_scribbles: int = 0,
    n_background_scribbles: int = 0,
    tracer: Optional[str] = None,
    ct: Optional[np.ndarray] = None,
    affine: Optional[np.ndarray] = None,
    return_features: bool = False,
):
    """Decide whether to replace the prediction with an empty mask.

    Fires only when every enabled criterion holds; a threshold of ``None`` disables its
    criterion.  Returns ``bool``, or ``(bool, features)`` with ``return_features``, where
    ``features["blocked_by"]`` lists the criteria that vetoed the decision.
    """
    p = params or NegativeGateConfig()
    feats = negative_gate_features(mask, pet, prob, spacing, connectivity=18,
                                   ct=ct, affine=affine,
                                   with_components=bool(getattr(p, "collect_components", False)))
    # A tumor scribble proves the case is positive, and the scribble set is cumulative,
    # so from here on the gate is permanently off for this case.
    if n_tumor_scribbles > 0:
        feats.update(fired=False, blocked_by=["tumor_scribble_proves_positive"],
                     iteration=int(iteration), n_tumor_scribbles=int(n_tumor_scribbles),
                     n_background_scribbles=int(n_background_scribbles))
        return (False, feats) if return_features else False
    feats["iteration"] = int(iteration)
    feats["n_tumor_scribbles"] = int(n_tumor_scribbles)
    feats["n_background_scribbles"] = int(n_background_scribbles)
    feats["tracer"] = tracer

    reasons: List[str] = []
    decision = True

    if not p.enabled:
        decision, reasons = False, ["disabled"]
    else:
        def veto(name: str) -> None:
            nonlocal decision
            decision = False
            reasons.append(name)

        if p.require_no_scribbles and (n_tumor_scribbles > 0 or n_background_scribbles > 0):
            veto("scribbles_present")
        if p.only_iteration_zero and iteration != 0:
            veto("not_iteration_zero")
        if p.max_total_volume_ml is not None and feats["total_volume_ml"] >= p.max_total_volume_ml:
            veto("volume")
        if (p.max_component_volume_ml is not None
                and feats["largest_component_ml"] >= p.max_component_volume_ml):
            veto("largest_component")
        if p.max_n_components is not None and feats["n_components"] > p.max_n_components:
            veto("n_components")

        suv_limit = p.max_suv
        if p.max_suv_by_tracer:
            suv_limit = p.max_suv_by_tracer.get(tracer or "unknown", suv_limit)
        if suv_limit is not None and feats["suv_max_in_mask"] >= suv_limit:
            veto("suv")

        prob_criteria = (p.max_prob, p.max_mean_prob, p.max_soft_volume_ml)
        if feats["prob_max_in_mask"] is None:
            if p.require_prob and any(c is not None for c in prob_criteria):
                veto("no_probability_map")
        else:
            if p.max_prob is not None and feats["prob_max_in_mask"] >= p.max_prob:
                veto("prob")
            if p.max_mean_prob is not None and feats["prob_mean_in_mask"] >= p.max_mean_prob:
                veto("mean_prob")
            if p.max_soft_volume_ml is not None and feats["soft_volume_ml"] >= p.max_soft_volume_ml:
                veto("soft_volume")

    feats["fired"] = bool(decision)
    feats["blocked_by"] = reasons
    if return_features:
        return bool(decision), feats
    return bool(decision)
