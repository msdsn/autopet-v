"""Hard scribble compliance: three guarantees, enforced and then re-checked.

G1: no background scribble voxel is inside the mask.  G2: every tumor scribble voxel is.
G3: a scribble that is already satisfied changes nothing, and a partially satisfied
tumor scribble grows only from the missing voxels.  Tumor growth treats background
scribble voxels as forbidden, so background-then-tumor satisfies both G1 and G2; on the
same voxel tumor wins.  A background scribble is drawn inside ``pred & ~gt``, a subset of
a predicted component that may still be a true positive that bled into cool tissue, so
the default is to split the component rather than delete it -- whole-component deletion
only when it has no confident core.  No RNG and no dict-order dependence anywhere:
components are visited in label order and every tie is broken by an explicit total order.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Tuple

import cc3d
import numpy as np
from scipy import ndimage

from .config import ComplianceConfig, TRACER_SUV_FLOOR
from .utils import (
    as_points_array,
    ball_offsets,
    bbox_of_points,
    bbox_slices,
    cluster_points,
    expand_bbox,
    foreground_prob,
    points_in_bounds,
    unique_points,
    voxel_volume_ml,
)

__all__ = [
    "apply_background_scribbles",
    "apply_tumor_scribbles",
    "apply_all_constraints",
    "check_constraints",
    "assert_constraints",
    "audit_removed_region",
]

_STRUCT_26 = np.ones((3, 3, 3), dtype=bool)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _as_bool(mask: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(mask).astype(bool, copy=True)


def _drop_conflicts(bg: np.ndarray, fg: np.ndarray) -> np.ndarray:
    """Remove background points that are also tumor points (tumor wins)."""
    if len(bg) == 0 or len(fg) == 0:
        return bg
    fset = {tuple(p) for p in fg.tolist()}
    keep = np.fromiter((tuple(p) not in fset for p in bg.tolist()), dtype=bool, count=len(bg))
    return bg[keep]


def _sample(mask: np.ndarray, pts: np.ndarray) -> np.ndarray:
    if len(pts) == 0:
        return np.zeros(0, dtype=mask.dtype)
    return mask[pts[:, 0], pts[:, 1], pts[:, 2]]


def _crop_points(pts: np.ndarray, lo: np.ndarray) -> np.ndarray:
    return pts - np.asarray(lo, dtype=np.int64)[None, :]


def _as_slices(box) -> Tuple[slice, slice, slice]:
    """cc3d.statistics bounding boxes are either slice tuples or (lo, hi) arrays."""
    if isinstance(box, tuple) and len(box) and isinstance(box[0], slice):
        return box  # type: ignore[return-value]
    b = np.asarray(box).reshape(-1)
    return (slice(int(b[0]), int(b[1])), slice(int(b[2]), int(b[3])), slice(int(b[4]), int(b[5])))


def _local_mask(shape, points: np.ndarray, lo) -> np.ndarray:
    """Boolean mask of ``points`` inside a crop (points outside the crop are dropped)."""
    out = np.zeros(shape, dtype=bool)
    if points is None or len(points) == 0:
        return out
    loc = _crop_points(points, lo)
    ok = np.ones(len(loc), dtype=bool)
    for ax in range(3):
        ok &= (loc[:, ax] >= 0) & (loc[:, ax] < shape[ax])
    loc = loc[ok]
    if len(loc):
        out[loc[:, 0], loc[:, 1], loc[:, 2]] = True
    return out


def _mm_box(radius_mm: float, spacing: Sequence[float]) -> list:
    return [max(1, int(2 * int(float(radius_mm) // float(s)) + 1)) for s in spacing]


# ---------------------------------------------------------------------------
# G1 -- background scribbles
# ---------------------------------------------------------------------------
def apply_background_scribbles(
    mask: np.ndarray,
    pet: np.ndarray,
    bg_points,
    connectivity: int = 18,
    *,
    prob: Optional[np.ndarray] = None,
    spacing: Sequence[float] = (1.0, 1.0, 1.0),
    fg_points=None,
    tracer: str = "fdg",
    cfg: Optional[ComplianceConfig] = None,
    gt: Optional[np.ndarray] = None,
    return_info: bool = False,
):
    """Remove what a background scribble points at, splitting the component when possible.

    A scribble voxel outside the mask is a no-op (G3).  Otherwise its 18-connected
    component gets a core -- the confident part, from the softmax when there is one and
    from the SUV ratios otherwise, eroded away from the scribble by ``bg_core_margin_mm``
    and unioned with any tumor scribble inside, so such a component is never deleted
    whole.  No core means delete; a core means a watershed on ``-prob`` (or ``-PET``)
    between scribble and core, deleting only the scribble's basin.  The split is
    rejected, and the component deleted, if it would remove more than
    ``bg_split_max_removed_fraction`` or leave less than ``bg_split_min_kept_volume_ml``.

    Returns ``mask``, or ``(mask, info)`` with ``info["removed"]`` the deleted volume.
    ``gt`` is validation-only and fills ``info["gt_audit"]``.
    """
    cfg = cfg or ComplianceConfig()
    out = _as_bool(mask)
    shape = out.shape
    pet = np.asarray(pet)
    vox_ml = voxel_volume_ml(spacing)
    suv_floor = TRACER_SUV_FLOOR.get(str(tracer).lower(), TRACER_SUV_FLOOR["unknown"])

    fg = points_in_bounds(unique_points(as_points_array(fg_points)), shape)
    bg = _drop_conflicts(points_in_bounds(unique_points(as_points_array(bg_points)), shape), fg)

    info: Dict[str, Any] = {
        "n_bg_points": int(len(bg)),
        "n_bg_points_outside_mask": 0,
        "n_components_hit": 0,
        "n_components_deleted": 0,
        "n_components_split": 0,
        "n_components_with_tumor_scribble": 0,
        "removed_voxels": 0,
        "removed": None,
        "split_details": [],
    }

    if len(bg) == 0 or not out.any():
        info["n_bg_points_outside_mask"] = int(len(bg))
        info["removed"] = np.zeros(shape, dtype=bool)
        return (out.astype(np.uint8), info) if return_info else out.astype(np.uint8)

    removed = np.zeros(shape, dtype=bool)
    labels = cc3d.connected_components(out.view(np.uint8), connectivity=connectivity)
    bg_labels = _sample(labels, bg)
    info["n_bg_points_outside_mask"] = int((bg_labels == 0).sum())  # G3: pure no-ops
    hit_labels = np.unique(bg_labels)
    hit_labels = hit_labels[hit_labels > 0]
    info["n_components_hit"] = int(len(hit_labels))

    if len(hit_labels):
        stats = cc3d.statistics(labels)
        boxes = stats["bounding_boxes"]
        counts = stats["voxel_counts"]
        prob_fg = foreground_prob(prob, shape) if prob is not None else None
        fg_labels = _sample(labels, fg) if len(fg) else np.zeros(0, dtype=labels.dtype)

        for lab in sorted(int(v) for v in hit_labels):  # deterministic visit order
            sl = _as_slices(boxes[lab])
            lo = np.array([s.start for s in sl], dtype=np.int64)
            comp = labels[sl] == lab
            comp_vox = int(counts[lab])

            bg_local = _crop_points(bg[bg_labels == lab], lo)
            fg_in = fg[fg_labels == lab] if len(fg) else fg
            has_tumor = len(fg_in) > 0
            if has_tumor:
                info["n_components_with_tumor_scribble"] += 1
            fg_local = _crop_points(fg_in, lo) if has_tumor else np.zeros((0, 3), np.int64)

            pet_crop = np.asarray(pet[sl], dtype=np.float32)
            prob_crop = prob_fg[sl] if prob_fg is not None else None

            core, core_strict = _component_core(
                comp=comp,
                pet_crop=pet_crop,
                prob_crop=prob_crop,
                bg_local=bg_local,
                fg_local=fg_local,
                spacing=spacing,
                vox_ml=vox_ml,
                suv_floor=suv_floor,
                cfg=cfg,
            )

            part = None
            if core is not None:
                use_prob = cfg.bg_split_use_prob and prob_crop is not None
                land = -prob_crop.astype(np.float32) if use_prob else -pet_crop
                part = _watershed_split(
                    comp, land, core, core_strict, bg_local, spacing, vox_ml, cfg
                )

            if part is None:
                out[sl] &= ~comp
                removed[sl] |= comp
                info["n_components_deleted"] += 1
            else:
                out[sl] &= ~part
                removed[sl] |= part
                info["n_components_split"] += 1
                info["split_details"].append(
                    {
                        "label": lab,
                        "component_ml": float(comp_vox * vox_ml),
                        "removed_ml": float(int(part.sum()) * vox_ml),
                        "had_tumor_scribble": bool(has_tumor),
                    }
                )

    # G1, belt and braces: nothing may leave a background voxel switched on.
    still = _sample(out, bg).astype(bool)
    if still.any():
        viol = bg[still]
        out[viol[:, 0], viol[:, 1], viol[:, 2]] = False
        removed[viol[:, 0], viol[:, 1], viol[:, 2]] = True

    info["removed"] = removed
    info["removed_voxels"] = int(removed.sum())
    if gt is not None:
        info["gt_audit"] = audit_removed_region(removed, gt, spacing, connectivity=connectivity)
    return (out.astype(np.uint8), info) if return_info else out.astype(np.uint8)


def _component_core(
    comp: np.ndarray,
    pet_crop: np.ndarray,
    prob_crop: Optional[np.ndarray],
    bg_local: np.ndarray,
    fg_local: np.ndarray,
    spacing: Sequence[float],
    vox_ml: float,
    suv_floor: float,
    cfg: ComplianceConfig,
) -> Tuple[Optional[np.ndarray], np.ndarray]:
    """``(core, core_strict)`` for one component.

    ``core`` is the watershed marker: the confident part eroded away from the scribble so
    the two markers cannot collide, unioned with any tumor scribble.  ``core_strict`` is
    the un-eroded confident part, never deleted even if the watershed claims it.
    """
    if prob_crop is not None:
        core = comp & (prob_crop > cfg.bg_core_prob) & (pet_crop > suv_floor)
    else:
        # No confidence map: the only usable signal is that the part we want to keep
        # must be markedly hotter than the part the scribble points at.  A homogeneous
        # false positive fails this test and is deleted whole.
        peak = float(pet_crop[comp].max()) if comp.any() else 0.0
        thr = max(cfg.bg_core_suv_ratio * peak, suv_floor)
        if len(bg_local):
            scribble_suv = float(pet_crop[bg_local[:, 0], bg_local[:, 1], bg_local[:, 2]].max())
            thr = max(thr, cfg.bg_core_min_suv_ratio * scribble_suv)
        core = comp & (pet_crop >= thr)

    core_strict = core.copy()

    if len(bg_local):
        seed = np.ones(comp.shape, dtype=bool)
        seed[bg_local[:, 0], bg_local[:, 1], bg_local[:, 2]] = False
        dist = ndimage.distance_transform_edt(seed, sampling=tuple(float(s) for s in spacing))
        core = core & (dist >= cfg.bg_core_margin_mm)

    if int(core.sum()) * vox_ml < cfg.bg_core_min_volume_ml:
        core = np.zeros_like(core)

    # An earlier tumor scribble is absolute evidence: it is always part of the core, so
    # a component holding one can never be deleted whole.
    if len(fg_local):
        core = core.copy()
        core[fg_local[:, 0], fg_local[:, 1], fg_local[:, 2]] = True
        core &= comp
        core_strict = core_strict | core

    return (core if core.any() else None), core_strict


def _watershed_split(
    comp: np.ndarray,
    landscape: np.ndarray,
    core: np.ndarray,
    core_strict: np.ndarray,
    bg_local: np.ndarray,
    spacing: Sequence[float],
    vox_ml: float,
    cfg: ComplianceConfig,
) -> Optional[np.ndarray]:
    """The sub-region of ``comp`` to delete, or ``None`` to delete the whole component.

    On a well-segmented lesion ``-prob`` is nearly flat, the watershed degenerates into a
    distance partition, and a scribble on the rim would be handed most of the lesion, so
    the raw basin is bounded three ways: it may hold no strictly-confident voxel, it must
    stay within ``bg_remove_max_radius_mm`` of the scribble measured geodesically inside
    the component, and it may not exceed ``bg_split_max_removed_fraction``.  If the last
    bound still fails the fallback is the scribble's own neighbourhood, not deletion.
    """
    from skimage.segmentation import watershed

    if len(bg_local) == 0:
        return None
    markers = np.zeros(comp.shape, dtype=np.int32)
    markers[core] = 2
    markers[bg_local[:, 0], bg_local[:, 1], bg_local[:, 2]] = 1
    if not (markers == 1).any() or not (markers == 2).any():
        return None

    ws = watershed(landscape, markers, mask=comp)
    part = comp & (ws == 1)

    bg_mask = np.zeros(comp.shape, dtype=bool)
    bg_mask[bg_local[:, 0], bg_local[:, 1], bg_local[:, 2]] = True

    if cfg.bg_protect_confident_core:
        part &= ~core_strict
        part |= bg_mask & comp          # G1 always outranks the core

    if cfg.bg_remove_max_radius_mm and cfg.bg_remove_max_radius_mm > 0:
        near = _geodesic_within(comp, bg_mask, spacing, cfg.bg_remove_max_radius_mm)
        if near is None:
            d = ndimage.distance_transform_edt(~bg_mask, sampling=tuple(float(s) for s in spacing))
            near = d <= cfg.bg_remove_max_radius_mm
        part &= near
        part |= bg_mask & comp

    n_part, n_comp = int(part.sum()), int(comp.sum())
    if n_part == 0:
        return None
    if n_part / max(n_comp, 1) > cfg.bg_split_max_removed_fraction:
        part = _local_removal(comp, core_strict, bg_mask, spacing, cfg)
        n_part = int(part.sum())
        if n_part == 0:
            return None
    if (n_comp - n_part) * vox_ml < cfg.bg_split_min_kept_volume_ml:
        return None
    return part


def _local_removal(comp, core_strict, bg_mask, spacing, cfg):
    """Least-damage fallback: the scribble's own neighbourhood inside the component."""
    ball = _ball_around(bg_mask, cfg.bg_local_fallback_radius_mm, spacing)
    part = comp & ball
    if cfg.bg_protect_confident_core:
        part &= ~core_strict
    return part | (bg_mask & comp)


def audit_removed_region(
    removed: np.ndarray,
    gt: np.ndarray,
    spacing: Sequence[float] = (1.0, 1.0, 1.0),
    *,
    connectivity: int = 18,
    fraction: float = 0.5,
) -> Dict[str, Any]:
    """Validation-only audit: how much ground truth a background scribble cost us.

    Reports the number of GT lesions of which more than ``fraction`` was deleted; it
    should be near zero, otherwise the removal rule is too aggressive.
    """
    vox_ml = voxel_volume_ml(spacing)
    removed = np.ascontiguousarray(np.asarray(removed) > 0)
    gt = np.ascontiguousarray(np.asarray(gt) > 0)
    out: Dict[str, Any] = {
        "removed_ml": float(removed.sum() * vox_ml),
        "removed_gt_ml": float((removed & gt).sum() * vox_ml),
        "n_gt_lesions": 0,
        "n_gt_lesions_over_half_removed": 0,
        "worst_fraction_removed": 0.0,
    }
    if not gt.any():
        return out
    labels = cc3d.connected_components(gt.view(np.uint8), connectivity=connectivity)
    counts = cc3d.statistics(labels)["voxel_counts"]
    out["n_gt_lesions"] = int(np.count_nonzero(counts[1:]))
    if not removed.any():
        return out
    hit = np.bincount(np.asarray(labels[removed]).ravel(), minlength=len(counts))[: len(counts)]
    frac = np.where(counts > 0, hit / np.maximum(counts, 1), 0.0)
    frac[0] = 0.0
    out["n_gt_lesions_over_half_removed"] = int((frac > fraction).sum())
    out["worst_fraction_removed"] = float(frac.max())
    return out


# ---------------------------------------------------------------------------
# G2 / G3 -- tumor scribbles
# ---------------------------------------------------------------------------
def apply_tumor_scribbles(
    mask: np.ndarray,
    pet: np.ndarray,
    ct: Optional[np.ndarray],
    fg_points,
    prob: Optional[np.ndarray] = None,
    tracer: str = "fdg",
    spacing: Sequence[float] = (1.0, 1.0, 1.0),
    *,
    cfg: Optional[ComplianceConfig] = None,
    forbidden: Optional[np.ndarray] = None,
    bg_points=None,
    method: Optional[str] = None,
    return_info: bool = False,
):
    """Grow a lesion from every tumor scribble that is not already covered.

    A cluster entirely inside the mask is skipped (G3); a partially covered one is seeded
    from its missing voxels only.  On a compact error component the ``centerline``
    strategy returns the whole 2-D cross-section rather than a line, and that footprint
    is then partial ground truth with known geometry: when it is thicker than
    ``fg_footprint_min_thickness_mm`` the SUV threshold is picked to maximise the IoU
    between the grown region on the scribble's slice and the footprint.  A thin footprint
    carries no area information, so the threshold falls back to
    ``max(fg_alpha * SUVpeak_local, SUV_floor[tracer])``.

    Growth is bounded geodesically inside the candidate set rather than euclidean, cut at
    competing SUV maxima by a watershed, kept from merging into any other predicted
    component (which would cost a detection), and capped by ``fg_max_volume_ml`` with the
    threshold escalated until it fits.  ``method="random_walker"`` solves the same problem
    with ``skimage.segmentation.random_walker`` on a ``rw_crop_size`` cube instead.
    """
    cfg = cfg or ComplianceConfig()
    method = method or cfg.fg_method
    out = _as_bool(mask)
    shape = out.shape
    pet = np.asarray(pet)
    spacing = tuple(float(s) for s in spacing)
    vox_ml = voxel_volume_ml(spacing)
    suv_floor = TRACER_SUV_FLOOR.get(str(tracer).lower(), TRACER_SUV_FLOOR["unknown"])

    fg = points_in_bounds(unique_points(as_points_array(fg_points)), shape)
    info: Dict[str, Any] = {
        "n_fg_points": int(len(fg)),
        "n_clusters": 0,
        "n_clusters_grown": 0,
        "n_clusters_satisfied": 0,
        "added_voxels": 0,
        "clusters": [],
        "method": method,
    }
    if len(fg) == 0:
        return (out.astype(np.uint8), info) if return_info else out.astype(np.uint8)

    forbid_full = np.asarray(forbidden).astype(bool) if forbidden is not None else None
    bg = _drop_conflicts(points_in_bounds(unique_points(as_points_array(bg_points)), shape), fg)

    clusters = cluster_points(fg, spacing, cfg.fg_cluster_radius_mm)
    info["n_clusters"] = len(clusters)

    labels = (
        cc3d.connected_components(out.view(np.uint8), connectivity=cfg.connectivity)
        if out.any()
        else None
    )
    prob_fg = foreground_prob(prob, shape) if prob is not None else None
    added_total = 0

    for cl in clusters:
        cluster_pts = fg[cl]
        inside = _sample(out, cluster_pts).astype(bool)
        if bool(inside.all()):
            # G3: already satisfied.  Do not re-grow, do not enlarge.
            info["n_clusters_satisfied"] += 1
            continue
        seeds = cluster_pts[~inside]  # grow only the missing part

        common = dict(
            seeds=seeds,
            own_points=cluster_pts,
            pet=pet,
            prob_fg=prob_fg,
            mask=out,
            labels=labels,
            spacing=spacing,
            suv_floor=suv_floor,
            cfg=cfg,
            forbid_full=forbid_full,
            bg=bg,
        )
        if method == "random_walker":
            sl, region, detail = _grow_random_walker(**common)
        else:
            sl, region, detail = _grow_threshold(ct=ct, vox_ml=vox_ml, **common)

        before = int(out[sl].sum())
        out[sl] |= region
        gained = int(out[sl].sum()) - before
        added_total += gained
        info["n_clusters_grown"] += 1
        detail.update(
            n_seeds=int(len(seeds)),
            n_points=int(len(cluster_pts)),
            seed=[int(v) for v in seeds[0]],
            added_ml=float(gained * vox_ml),
            region_ml=float(int(region.sum()) * vox_ml),
        )
        info["clusters"].append(detail)

    # G2, belt and braces
    missing = ~_sample(out, fg).astype(bool)
    if missing.any():
        m = fg[missing]
        out[m[:, 0], m[:, 1], m[:, 2]] = True
        added_total += int(missing.sum())

    info["added_voxels"] = added_total
    return (out.astype(np.uint8), info) if return_info else out.astype(np.uint8)


# --- growth helpers ---------------------------------------------------------
def _cluster_crop(seeds, shape, spacing, radius_mm):
    lo, hi = bbox_of_points(seeds)
    margin = np.ceil(radius_mm / np.asarray(spacing, dtype=np.float64)).astype(np.int64)
    lo, hi = expand_bbox(lo, hi, margin, shape)
    return bbox_slices(lo, hi), lo


def _forbidden_crop(sl, lo, seed_mask, labels, own_points, forbid_full, bg, cfg, spacing):
    """Voxels the grown region may not contain.

    Every existing component holding no point of this cluster, dilated by one voxel, plus
    the background scribble voxels and the accumulated background region -- minus the
    scribble voxels themselves, since G2 outranks everything.
    """
    forbid = np.zeros(seed_mask.shape, dtype=bool)

    if labels is not None:
        lab_c = labels[sl]
        own_mask = seed_mask | _local_mask(seed_mask.shape, own_points, lo)
        own = np.unique(lab_c[own_mask])
        own = own[own > 0]
        other = lab_c > 0
        if len(own):
            other &= ~np.isin(lab_c, own)
        if other.any():
            forbid |= ndimage.binary_dilation(other, structure=_STRUCT_26)

    if forbid_full is not None:
        forbid |= forbid_full[sl]

    if len(bg):
        if cfg.bg_forbid_radius_mm > 0:
            loc = _crop_points(bg, lo)
            offs = ball_offsets(cfg.bg_forbid_radius_mm, spacing)
            coords = (loc[:, None, :] + offs[None, :, :]).reshape(-1, 3)
            good = np.ones(len(coords), dtype=bool)
            for ax in range(3):
                good &= (coords[:, ax] >= 0) & (coords[:, ax] < seed_mask.shape[ax])
            coords = coords[good]
            if len(coords):
                forbid[coords[:, 0], coords[:, 1], coords[:, 2]] = True
        else:
            forbid |= _local_mask(seed_mask.shape, bg, lo)

    forbid &= ~seed_mask
    return forbid


def _ball_around(seed_mask, radius_mm, spacing):
    if radius_mm <= 0:
        return seed_mask.copy()
    d = ndimage.distance_transform_edt(~seed_mask, sampling=tuple(float(s) for s in spacing))
    return d <= radius_mm


def _geodesic_within(candidate, seed_mask, spacing, max_mm):
    """Geodesic distance from the seeds inside ``candidate``.  ``None`` if unavailable."""
    try:
        from skimage.graph import MCP_Geometric
    except Exception:  # pragma: no cover - skimage always ships it
        return None
    costs = np.where(candidate, 1.0, np.inf).astype(np.float64)
    try:
        mcp = MCP_Geometric(costs, sampling=tuple(float(s) for s in spacing))
        dist, _ = mcp.find_costs(np.argwhere(seed_mask).tolist())
    except Exception:  # pragma: no cover
        return None
    with np.errstate(invalid="ignore"):
        return np.isfinite(dist) & (dist <= max_mm)


def _stop_at_valleys(region, smooth, seed_mask, spacing, seed_peak, cfg):
    """Cut the region at watershed lines against competing, well-separated SUV maxima.

    A competing focus must stand ``fg_valley_depth_suv`` above the saddle joining it to
    the scribble; a homogeneous lesion has one plateau, so nothing competes.
    """
    if not cfg.fg_stop_at_valleys or not region.any():
        return region
    try:
        from skimage.morphology import h_maxima
        from skimage.segmentation import watershed
    except Exception:  # pragma: no cover
        return region

    field = np.where(region, smooth, 0.0).astype(np.float32)
    peaks = h_maxima(field, float(cfg.fg_valley_depth_suv)) > 0
    peaks &= region & (smooth >= cfg.fg_valley_peak_ratio * seed_peak)
    if not peaks.any():
        return region

    plateaus = cc3d.connected_components(np.ascontiguousarray(peaks).view(np.uint8), connectivity=26)
    own = np.unique(plateaus[seed_mask])
    own = own[own > 0]
    competing = plateaus > 0
    if len(own):
        competing &= ~np.isin(plateaus, own)
    d = ndimage.distance_transform_edt(~seed_mask, sampling=tuple(float(s) for s in spacing))
    competing &= d >= cfg.fg_valley_min_distance_mm
    if not competing.any():
        return region

    comp_lab = cc3d.connected_components(np.ascontiguousarray(competing).view(np.uint8), connectivity=26)
    markers = np.zeros(region.shape, dtype=np.int32)
    markers[comp_lab > 0] = comp_lab[comp_lab > 0] + 1
    markers[seed_mask] = 1
    ws = watershed(-smooth, markers, mask=region)
    kept = region & (ws == 1)
    return kept if kept.any() else region


def _candidate_region(
    thr, pet_c, prob_c, ct_ok, forbid, seed_mask, spacing, cfg, use_prob, smooth, seed_peak
):
    """Threshold -> candidate set -> geodesic bound -> valley cut -> seed component."""
    cand = pet_c >= thr
    if prob_c is not None and use_prob:
        cand |= prob_c >= cfg.fg_prob_include
    if ct_ok is not None:
        cand &= ct_ok
    cand &= ~forbid
    cand |= seed_mask

    within = _geodesic_within(cand, seed_mask, spacing, cfg.fg_max_radius_mm) if cfg.fg_geodesic else None
    if within is None:
        d = ndimage.distance_transform_edt(~seed_mask, sampling=tuple(spacing))
        within = d <= cfg.fg_max_radius_mm
    cand = (cand & within) | seed_mask

    lab = cc3d.connected_components(np.ascontiguousarray(cand).view(np.uint8), connectivity=cfg.connectivity)
    keep = np.unique(lab[seed_mask])
    keep = keep[keep > 0]
    region = np.isin(lab, keep) if len(keep) else seed_mask.copy()
    region = _stop_at_valleys(region, smooth, seed_mask, spacing, seed_peak, cfg)
    return region | seed_mask


def _footprint_target(seed_mask, own_mask, labels_crop):
    """The slice, the scribble footprint, and the in-plane target to calibrate against.

    The scribble is drawn on the error region, so on its slice the true lesion is the
    footprint plus whatever of that lesion we already predict there.
    """
    ks = np.argwhere(seed_mask)[:, 2]
    if len(ks) == 0:
        return None, None, None
    # One scribble is drawn on one axial slice, but accumulated scribbles from different
    # iterations cluster across slices, so calibrate on the slice holding most seeds.
    vals, counts = np.unique(ks, return_counts=True)
    k = int(vals[int(np.argmax(counts))])
    footprint = own_mask[:, :, k] | seed_mask[:, :, k]
    target = footprint.copy()
    if labels_crop is not None:
        own_lab = np.unique(labels_crop[own_mask])
        own_lab = own_lab[own_lab > 0]
        if len(own_lab):
            target |= np.isin(labels_crop[:, :, k], own_lab)
    return k, footprint, target


def _footprint_thickness_mm(footprint, spacing):
    """Inner radius of the footprint: > 1 voxel means it carries area information."""
    if footprint is None or not footprint.any():
        return 0.0
    d = ndimage.distance_transform_edt(footprint, sampling=(float(spacing[0]), float(spacing[1])))
    return float(d.max())


def _iou(a, b):
    union = int((a | b).sum())
    return int((a & b).sum()) / union if union else 0.0


def _grow_threshold(
    seeds, own_points, pet, prob_fg, mask, labels, spacing, suv_floor, cfg,
    forbid_full, bg, ct=None, vox_ml=1.0,
):
    shape = mask.shape
    sl, lo = _cluster_crop(seeds, shape, spacing, cfg.fg_max_radius_mm + cfg.fg_crop_margin_mm)
    pet_c = np.asarray(pet[sl], dtype=np.float32)
    seed_mask = _local_mask(pet_c.shape, seeds, lo)
    own_mask = _local_mask(pet_c.shape, own_points, lo)

    # Local SUVpeak: separable box approximation of the 1 mL sphere mean, over the whole
    # scribble -- on the missing voxels alone a cold tail collapses the threshold.
    smooth = ndimage.uniform_filter(pet_c, size=_mm_box(cfg.fg_peak_radius_mm, spacing), mode="nearest")
    peak_mask = own_mask if (cfg.fg_peak_over_whole_cluster and own_mask.any()) else seed_mask
    seed_peak = float(smooth[peak_mask].max())
    alpha_thr = max(cfg.fg_alpha * seed_peak, float(suv_floor))
    # The tracer floor is a lower bound on plausible uptake, not a delineation threshold,
    # so letting it threshold grows into normal anatomy.  Decided on the raw maximum: the
    # 1 mL box mean would dilute a genuine 6 mm node into "low contrast".
    raw_peak = float(pet_c[peak_mask].max())
    low_contrast = cfg.fg_low_contrast_fallback and (cfg.fg_alpha * raw_peak <= float(suv_floor))

    forbid = _forbidden_crop(sl, lo, seed_mask, labels, own_points, forbid_full, bg, cfg, spacing)
    ct_ok = None
    if ct is not None and cfg.fg_exclude_ct_below_hu is not None:
        ct_ok = np.asarray(ct[sl]) >= cfg.fg_exclude_ct_below_hu
    prob_c = prob_fg[sl] if prob_fg is not None else None

    # Volume cap, absolute and relative: the absolute one is a sane ceiling for a bulky
    # lesion and an absurd one for a 2 mL node, so the relative cap does the real work.
    labels_crop = labels[sl] if labels is not None else None
    own_vox = int(own_mask.sum())
    if labels_crop is not None:
        own_lab = np.unique(labels_crop[own_mask])
        own_lab = own_lab[own_lab > 0]
        if len(own_lab):
            own_vox += int(np.isin(labels_crop, own_lab).sum())
    reference_ml = own_vox * vox_ml
    relative_ml = max(cfg.fg_min_growth_ml, cfg.fg_max_relative_growth * reference_ml)
    max_vox = min(cfg.fg_max_volume_ml, relative_ml) / vox_ml

    def build(thr, use_prob=True):
        return _candidate_region(
            thr, pet_c, prob_c, ct_ok, forbid, seed_mask, spacing, cfg, use_prob, smooth, seed_peak
        )

    detail: Dict[str, Any] = {
        "alpha_threshold": alpha_thr,
        "seed_peak": seed_peak,
        "raw_peak": raw_peak,
        "low_contrast": bool(low_contrast),
        "calibrated": False,
        "reference_ml": reference_ml,
        "max_volume_ml": float(max_vox * vox_ml),
    }

    if low_contrast:
        # SUV cannot delineate anything here.  The softmax still might, so prefer the
        # model's own connected evidence when it exists and fits the cap; otherwise fall
        # back to a bounded ball, which is the least-damage way to satisfy G2.
        region = None
        if prob_c is not None:
            cand = (prob_c >= cfg.fg_prob_include) & ~forbid
            if ct_ok is not None:
                cand &= ct_ok
            cand |= seed_mask
            lab = cc3d.connected_components(
                np.ascontiguousarray(cand).view(np.uint8), connectivity=cfg.connectivity
            )
            keep = np.unique(lab[seed_mask])
            keep = keep[keep > 0]
            if len(keep):
                prob_region = np.isin(lab, keep)
                if int((prob_region & ~mask[sl]).sum()) <= max_vox:
                    region = prob_region
                    detail.update(threshold=None, fallback="prob_low_contrast")
        if region is None:
            region = _ball_around(seed_mask, cfg.fg_fallback_ball_radius_mm, spacing) & ~forbid
            if ct_ok is not None:
                region &= ct_ok
            detail.update(threshold=None, fallback="ball_low_contrast")
        region |= seed_mask
        return sl, region, detail

    # --- threshold selection ------------------------------------------------
    thr = alpha_thr
    k, footprint, target = _footprint_target(seed_mask, own_mask, labels_crop)
    thickness = _footprint_thickness_mm(footprint, spacing)
    detail["footprint_thickness_mm"] = thickness

    if (
        cfg.fg_calibrate_from_footprint
        and footprint is not None
        and thickness >= cfg.fg_footprint_min_thickness_mm
    ):
        vals = pet_c[:, :, k][footprint]
        cands = [float(np.percentile(vals, q)) for q in cfg.fg_calibration_percentiles]
        cands.append(alpha_thr)
        floor = 0.5 * float(suv_floor)
        cands = sorted({round(max(c, floor), 4) for c in cands})  # sorted -> deterministic
        best, best_iou = alpha_thr, -1.0
        for c in cands:
            region = build(c)
            if region.sum() > max_vox:
                continue
            score = _iou(region[:, :, k], target)
            if score > best_iou + 1e-9:
                best, best_iou = c, score
        if best_iou >= 0.0:
            thr, detail["calibrated"], detail["calibration_iou"] = best, True, best_iou

    # --- build, with a volume-cap escalation --------------------------------
    region = seed_mask.copy()
    already = mask[sl]
    for attempt, mult in enumerate(cfg.fg_threshold_escalation):
        region = build(thr * float(mult), use_prob=(attempt == 0))
        if int((region & ~already).sum()) <= max_vox:
            detail["threshold"] = float(thr * float(mult))
            break
    else:
        region = (_ball_around(seed_mask, cfg.fg_fallback_ball_radius_mm, spacing) & ~forbid) | seed_mask
        detail["threshold"] = None
        detail["fallback"] = "ball_volume_cap"

    # --- photopenic scribble: no threshold can help -------------------------
    if bool((pet_c[seed_mask] < alpha_thr).all()) and cfg.fg_fallback_ball_radius_mm > 0:
        ball = _ball_around(seed_mask, cfg.fg_fallback_ball_radius_mm, spacing) & ~forbid
        if ct_ok is not None:
            ball &= ct_ok
        region |= ball | seed_mask
        detail["fallback"] = "ball_cold_scribble"

    return sl, region, detail


def _grow_random_walker(
    seeds, own_points, pet, prob_fg, mask, labels, spacing, suv_floor, cfg, forbid_full, bg
):
    from skimage.segmentation import random_walker

    shape = mask.shape
    half = cfg.rw_crop_size // 2
    centre = np.rint(seeds.mean(axis=0)).astype(np.int64)
    lo = np.maximum(centre - half, 0)
    hi = np.minimum(lo + cfg.rw_crop_size - 1, np.asarray(shape, np.int64) - 1)
    lo = np.maximum(hi - cfg.rw_crop_size + 1, 0)
    slo, shi = bbox_of_points(seeds)
    lo, hi = np.minimum(lo, slo), np.maximum(hi, shi)
    sl = bbox_slices(lo, hi)

    pet_c = np.asarray(pet[sl], dtype=np.float32)
    seed_mask = _local_mask(pet_c.shape, seeds, lo)
    dist = ndimage.distance_transform_edt(~seed_mask, sampling=tuple(spacing))
    forbid = _forbidden_crop(sl, lo, seed_mask, labels, own_points, forbid_full, bg, cfg, spacing)

    seed_labels = np.zeros(pet_c.shape, dtype=np.int32)
    seed_labels[seed_mask] = 1
    bg_seed = (dist > cfg.rw_bg_distance_mm) | (pet_c < 0.5 * float(suv_floor)) | forbid
    bg_seed &= ~seed_mask
    seed_labels[bg_seed] = 2

    detail: Dict[str, Any] = {"method": "random_walker"}
    if not (seed_labels == 2).any():
        region = _ball_around(seed_mask, cfg.fg_fallback_ball_radius_mm, spacing) & ~forbid
        detail["fallback"] = "ball_no_bg_seed"
        return sl, region | seed_mask, detail

    scale = float(np.percentile(pet_c, 99.5))
    data = pet_c / max(scale, 1e-3)
    if prob_fg is not None:
        data = 0.5 * data + 0.5 * prob_fg[sl].astype(np.float32)

    kwargs = dict(beta=cfg.rw_beta, mode=cfg.rw_mode, spacing=tuple(spacing), tol=cfg.rw_tol)
    try:
        try:
            rw = random_walker(data, seed_labels, prob_tol=cfg.rw_prob_tol, **kwargs)
        except TypeError:  # older scikit-image has no prob_tol
            rw = random_walker(data, seed_labels, **kwargs)
    except Exception:
        region = _ball_around(seed_mask, cfg.fg_fallback_ball_radius_mm, spacing) & ~forbid
        detail["fallback"] = "ball_solver_failed"
        return sl, region | seed_mask, detail

    region = ((rw == 1) & ~forbid) | seed_mask
    lab = cc3d.connected_components(np.ascontiguousarray(region).view(np.uint8), connectivity=cfg.connectivity)
    keep = np.unique(lab[seed_mask])
    keep = keep[keep > 0]
    region = np.isin(lab, keep) if len(keep) else seed_mask.copy()
    return sl, region, detail


# ---------------------------------------------------------------------------
# combined + verification
# ---------------------------------------------------------------------------
def apply_all_constraints(
    mask: np.ndarray,
    pet: np.ndarray,
    ct: Optional[np.ndarray],
    fg_points,
    bg_points,
    *,
    prob: Optional[np.ndarray] = None,
    tracer: str = "fdg",
    spacing: Sequence[float] = (1.0, 1.0, 1.0),
    cfg: Optional[ComplianceConfig] = None,
    forbidden: Optional[np.ndarray] = None,
    gt: Optional[np.ndarray] = None,
    return_info: bool = False,
):
    """Background compliance, then tumor compliance.  Idempotent as a whole."""
    cfg = cfg or ComplianceConfig()
    mask, bg_info = apply_background_scribbles(
        mask, pet, bg_points,
        connectivity=cfg.connectivity, prob=prob, spacing=spacing,
        fg_points=fg_points, tracer=tracer, cfg=cfg, gt=gt, return_info=True,
    )
    removed = bg_info.get("removed")
    forbid = removed if forbidden is None else (removed | np.asarray(forbidden).astype(bool))
    mask, fg_info = apply_tumor_scribbles(
        mask, pet, ct, fg_points,
        prob=prob, tracer=tracer, spacing=spacing, cfg=cfg,
        forbidden=forbid, bg_points=bg_points, return_info=True,
    )
    if return_info:
        return mask, {"background": bg_info, "foreground": fg_info}
    return mask


def check_constraints(mask: np.ndarray, fg_points, bg_points) -> Dict[str, Any]:
    """Report constraint violations without raising."""
    m = np.asarray(mask) > 0
    fg = points_in_bounds(unique_points(as_points_array(fg_points)), m.shape)
    bg = _drop_conflicts(points_in_bounds(unique_points(as_points_array(bg_points)), m.shape), fg)
    fg_missing = fg[~_sample(m, fg).astype(bool)] if len(fg) else fg
    bg_inside = bg[_sample(m, bg).astype(bool)] if len(bg) else bg
    return {
        "ok": len(fg_missing) == 0 and len(bg_inside) == 0,
        "n_fg": int(len(fg)),
        "n_bg": int(len(bg)),
        "n_fg_missing": int(len(fg_missing)),
        "n_bg_inside": int(len(bg_inside)),
        "fg_missing": fg_missing.tolist(),
        "bg_inside": bg_inside.tolist(),
    }


def assert_constraints(mask: np.ndarray, fg_points, bg_points) -> Dict[str, Any]:
    res = check_constraints(mask, fg_points, bg_points)
    if not res["ok"]:
        raise AssertionError(
            f"scribble constraints violated: {res['n_fg_missing']}/{res['n_fg']} tumor points "
            f"outside the mask, {res['n_bg_inside']}/{res['n_bg']} background points inside "
            f"(first offenders fg={res['fg_missing'][:3]} bg={res['bg_inside'][:3]})"
        )
    return res
