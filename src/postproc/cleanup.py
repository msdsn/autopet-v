"""Component cleanup aimed at the detection metric.

An unmatched predicted component costs one false positive whatever its size, and
matching only needs IoU >= 0.1, so specks are pruned aggressively while anything that
could be a small lesion is kept -- hence the SUV gate, and hence the rule that a
component holding a tumor scribble is untouchable.  Connected components use
18-connectivity throughout, matching the scorer.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence

import cc3d
import numpy as np

from .config import CleanupConfig, TRACER_SUV_FLOOR
from .compliance import _as_slices
from .utils import as_points_array, points_in_bounds, unique_points, voxel_volume_ml

__all__ = [
    "split_large_components",
    "remove_small_components",
    "remove_components_v2",
    "recruit_components",
    "bridge_components",
    "tracer_suv_floor",
    "fill_small_holes",
    "cleanup_mask",
    "rank_components",
    "resolve_v2_rule",
    "resolve_v2_rules",
]


def rank_components(labels: np.ndarray, boxes, counts, pet: np.ndarray, keep: int) -> set:
    """The ``keep`` best components, by (SUVmax, voxel count, label) -- a total order.

    Used by the guard that stops cleanup from emptying a non-empty prediction; that is
    the negative gate's decision, not cleanup's.
    """
    scored = []
    for lab in range(1, len(counts)):
        if counts[lab] == 0:
            continue
        scored.append((_component_suv_max(pet, labels, boxes[lab], lab), int(counts[lab]), lab))
    scored.sort(reverse=True)
    return {lab for _, _, lab in scored[: max(int(keep), 0)]}


def _n_points(points) -> int:
    return len(unique_points(as_points_array(points)))


def _protected_labels(labels: np.ndarray, protect_points) -> np.ndarray:
    pts = points_in_bounds(unique_points(as_points_array(protect_points)), labels.shape)
    if len(pts) == 0:
        return np.zeros(0, dtype=np.int64)
    vals = labels[pts[:, 0], pts[:, 1], pts[:, 2]]
    vals = np.unique(vals)
    return vals[vals > 0].astype(np.int64)


def _component_suv_max(pet: np.ndarray, labels: np.ndarray, box, lab: int) -> float:
    """SUVmax of one component, over its bounding box only.

    Cheaper than a global ``ndimage.maximum`` when only a few components qualify.
    """
    sl = _as_slices(box)
    sub = labels[sl] == lab
    if not sub.any():
        return -np.inf
    return float(np.asarray(pet[sl])[sub].max())


def remove_small_components(
    mask: np.ndarray,
    pet: np.ndarray,
    spacing: Sequence[float],
    min_volume_ml: float = 0.3,
    suv_gate: float = 4.0,
    *,
    connectivity: int = 18,
    protect_points=None,
    min_components_kept: int = 1,
    return_info: bool = False,
):
    """Drop 18-connected components below ``min_volume_ml`` unless SUVmax >= ``suv_gate``.

    Never removes a component containing one of ``protect_points`` (G2), and never
    empties a non-empty prediction: the best ``min_components_kept`` survive.
    """
    m = np.ascontiguousarray(np.asarray(mask) > 0)
    info: Dict[str, Any] = {
        "n_components": 0,
        "n_removed": 0,
        "removed_ml": 0.0,
        "n_rescued_from_emptying": 0,
    }
    if not m.any():
        if return_info:
            return m.astype(np.uint8), info
        return m.astype(np.uint8)

    vox_ml = voxel_volume_ml(spacing)
    min_vox = min_volume_ml / vox_ml
    labels = cc3d.connected_components(m.view(np.uint8), connectivity=connectivity)
    stats = cc3d.statistics(labels)
    counts = stats["voxel_counts"]
    boxes = stats["bounding_boxes"]
    n_labels = len(counts) - 1
    info["n_components"] = int(n_labels)

    protected = set(_protected_labels(labels, protect_points).tolist())
    small = [
        lab
        for lab in range(1, n_labels + 1)
        if counts[lab] > 0 and counts[lab] < min_vox and lab not in protected
    ]
    if not small:
        if return_info:
            return m.astype(np.uint8), info
        return m.astype(np.uint8)

    doomed = []
    for lab in small:
        if suv_gate is not None and np.isfinite(suv_gate):
            if _component_suv_max(pet, labels, boxes[lab], lab) >= suv_gate:
                continue
        doomed.append(lab)

    survivors = n_labels - len(doomed)
    if survivors < min_components_kept and min_components_kept > 0:
        rescued = rank_components(labels, boxes, counts, pet, min_components_kept)
        doomed = [lab for lab in doomed if lab not in rescued]
        info["n_rescued_from_emptying"] = len(rescued & set(range(1, n_labels + 1)))

    for lab in doomed:
        sl = _as_slices(boxes[lab])
        sub = labels[sl] == lab
        m[sl] &= ~sub
        info["removed_ml"] += float(int(counts[lab]) * vox_ml)
    info["n_removed"] = len(doomed)

    if return_info:
        return m.astype(np.uint8), info
    return m.astype(np.uint8)


_V2_ALIAS = {
    "v2_min_volume_ml": "min_volume_ml",
    "v2_suv_gate": "suv_gate",
    "v2_prob_gate": "prob_gate",
    "v2_silence_decay": "silence_decay",
}


def _component_stat(arr: np.ndarray, labels: np.ndarray, box, lab: int, how: str) -> float:
    """One reduction of ``arr`` over one component, computed on its bounding box only."""
    sl = _as_slices(box)
    sub = labels[sl] == lab
    if not sub.any():
        return -np.inf
    vals = np.asarray(arr[sl])[sub]
    return float(vals.max() if how == "max" else vals.mean())


def resolve_v2_rule(cfg: CleanupConfig, tracer: str) -> Dict[str, Any]:
    """The single-conjunction v2 thresholds in force for ``tracer``.

    Separate from the pruning pass so sweeps, tests and the log all read the same
    resolved numbers instead of re-deriving the per-tracer override.
    """
    rule = {
        "min_volume_ml": cfg.v2_min_volume_ml,
        "suv_gate": cfg.v2_suv_gate,
        "prob_gate": cfg.v2_prob_gate,
        "silence_decay": cfg.v2_silence_decay,
    }
    by_tracer = cfg.v2_by_tracer or {}
    over = by_tracer.get(str(tracer).lower())
    if over:
        for key, value in over.items():
            if key in ("v2_rules", "rules"):
                continue          # handled by resolve_v2_rules, not a scalar threshold
            name = _V2_ALIAS.get(key, key)
            if name not in rule:
                raise KeyError(
                    f"unknown v2_by_tracer key {key!r}; valid: "
                    f"{sorted(rule) + sorted(_V2_ALIAS)}"
                )
            rule[name] = value
    return rule


def resolve_v2_rules(cfg: CleanupConfig, tracer: str) -> list:
    """The list of conjunctions in force for ``tracer``: prune if any of them holds.

    Cold components (low SUVmax) and unconfident ones (low mean softmax) are largely
    disjoint populations, so a single conjunction over both criteria keeps most of each.
    """
    base = resolve_v2_rule(cfg, tracer)
    rules = cfg.v2_rules
    by_tracer = cfg.v2_by_tracer or {}
    over = by_tracer.get(str(tracer).lower()) or {}
    if "v2_rules" in over or "rules" in over:
        rules = over.get("v2_rules", over.get("rules"))
    if not rules:
        return [base]
    out = []
    for r in rules:
        merged = {"min_volume_ml": None, "suv_gate": None, "prob_gate": None,
                  "silence_decay": base["silence_decay"]}
        for key, value in dict(r).items():
            name = _V2_ALIAS.get(key, key)
            if name not in merged:
                raise KeyError(f"unknown v2_rules key {key!r}; valid: {sorted(merged)}")
            merged[name] = value
        if all(merged[k] is None for k in ("min_volume_ml", "suv_gate", "prob_gate")):
            raise ValueError("a v2 rule with no enabled criterion would delete everything")
        out.append(merged)
    return out


def remove_components_v2(
    mask: np.ndarray,
    pet: np.ndarray,
    spacing: Sequence[float],
    *,
    prob: Optional[np.ndarray] = None,
    rules: Optional[Sequence[Dict[str, Any]]] = None,
    min_volume_ml: Optional[float] = 0.3,
    suv_gate: Optional[float] = 4.0,
    prob_gate: Optional[float] = None,
    silence_decay: float = 1.0,
    iteration: int = 0,
    connectivity: int = 18,
    protect_points=None,
    background_points=None,
    min_components_kept: int = 1,
    return_info: bool = False,
):
    """Prune a component that matches any of ``rules``.

    A rule is a conjunction over volume, SUVmax and mean softmax; ``None`` disables a
    criterion, and with ``prob_gate=None`` this reduces to ``remove_small_components``.
    The same guards apply: a component holding one of ``protect_points`` survives (G2),
    and the pass never empties a non-empty prediction.  ``silence_decay`` multiplies the
    size threshold by ``silence_decay ** iteration`` for a component that has survived
    that many rounds without attracting a background scribble.
    """
    m = np.ascontiguousarray(np.asarray(mask) > 0)
    info: Dict[str, Any] = {
        "n_components": 0,
        "n_removed": 0,
        "removed_ml": 0.0,
        "n_rescued_from_emptying": 0,
        "rules": list(rules) if rules else [
            {"min_volume_ml": min_volume_ml, "suv_gate": suv_gate,
             "prob_gate": prob_gate, "silence_decay": silence_decay}],
        "iteration": int(iteration),
    }
    if not m.any():
        if return_info:
            return m.astype(np.uint8), info
        return m.astype(np.uint8)

    vox_ml = voxel_volume_ml(spacing)
    labels = cc3d.connected_components(m.view(np.uint8), connectivity=connectivity)
    stats = cc3d.statistics(labels)
    counts = stats["voxel_counts"]
    boxes = stats["bounding_boxes"]
    n_labels = len(counts) - 1
    info["n_components"] = int(n_labels)

    protected = set(_protected_labels(labels, protect_points).tolist())
    scribbled = set(_protected_labels(labels, background_points).tolist())

    rule_list = list(info["rules"])
    cache: Dict[tuple, float] = {}

    def stat(arr, lab, how):
        key = (how, lab)
        if key not in cache:
            cache[key] = _component_stat(arr, labels, boxes[lab], lab, how)
        return cache[key]

    def matches(rule, lab) -> bool:
        """True when every enabled criterion of one conjunction holds."""
        v = rule.get("min_volume_ml")
        if v is not None and np.isfinite(v):
            thr = float(v)
            decay = float(rule.get("silence_decay", 1.0) or 1.0)
            if decay != 1.0 and lab not in scribbled:
                thr *= decay ** max(int(iteration), 0)
            if counts[lab] * vox_ml >= thr:
                return False
        g = rule.get("suv_gate")
        if g is not None and np.isfinite(g):
            if stat(pet, lab, "max") >= g:
                return False
        p = rule.get("prob_gate")
        if p is not None and np.isfinite(p):
            # No softmax: the criterion cannot be evaluated, so it cannot be used to
            # justify a deletion.  Failing closed here keeps the rule identical to v1
            # when the base predictor returns no probabilities.
            if prob is None or stat(prob, lab, "mean") >= p:
                return False
        return True

    doomed = []
    for lab in range(1, n_labels + 1):
        if counts[lab] == 0 or lab in protected:
            continue
        if any(matches(rule, lab) for rule in rule_list):
            doomed.append(lab)

    if not doomed:
        if return_info:
            return m.astype(np.uint8), info
        return m.astype(np.uint8)

    survivors = n_labels - len(doomed)
    if survivors < min_components_kept and min_components_kept > 0:
        rescued = rank_components(labels, boxes, counts, pet, min_components_kept)
        doomed = [lab for lab in doomed if lab not in rescued]
        info["n_rescued_from_emptying"] = len(rescued & set(range(1, n_labels + 1)))

    for lab in doomed:
        sl = _as_slices(boxes[lab])
        m[sl] &= ~(labels[sl] == lab)
        info["removed_ml"] += float(int(counts[lab]) * vox_ml)
    info["n_removed"] = len(doomed)

    if return_info:
        return m.astype(np.uint8), info
    return m.astype(np.uint8)


def recruit_components(
    mask: np.ndarray,
    prob: Optional[np.ndarray],
    pet: np.ndarray,
    spacing: Sequence[float],
    *,
    threshold: Optional[float] = None,
    min_suv_max: float = 4.0,
    min_volume_ml: float = 0.1,
    max_components: int = 5,
    connectivity: int = 18,
    return_info: bool = False,
):
    """Add components that only exist below the model's own argmax threshold.

    Matching needs only IoU >= 0.1, so covering a tenth of a lesion already scores a
    true positive.  Re-binarises the softmax at ``threshold`` and adds the components
    touching no voxel of ``mask`` that clear both gates; purely additive.
    """
    m = np.ascontiguousarray(np.asarray(mask) > 0)
    info: Dict[str, Any] = {"threshold": threshold, "n_added": 0, "added_ml": 0.0,
                            "n_candidates": 0}
    if threshold is None or prob is None:
        if return_info:
            return m.astype(np.uint8), info
        return m.astype(np.uint8)

    lower = np.ascontiguousarray(np.asarray(prob) >= float(threshold))
    if not lower.any():
        if return_info:
            return m.astype(np.uint8), info
        return m.astype(np.uint8)

    vox_ml = voxel_volume_ml(spacing)
    labels = cc3d.connected_components(lower.view(np.uint8), connectivity=connectivity)
    stats = cc3d.statistics(labels)
    counts = stats["voxel_counts"]
    boxes = stats["bounding_boxes"]

    # A component of the lower-threshold mask that contains an argmax voxel is the
    # grown version of an existing component, not a new finding: skip it, so this pass
    # can only add and never reshape.
    hit = set(np.unique(labels[m]).tolist())
    cand = []
    for lab in range(1, len(counts)):
        if counts[lab] == 0 or lab in hit:
            continue
        if counts[lab] * vox_ml < float(min_volume_ml):
            continue
        suv = _component_stat(pet, labels, boxes[lab], lab, "max")
        if suv < float(min_suv_max):
            continue
        cand.append((suv, int(counts[lab]), lab))
    info["n_candidates"] = len(cand)
    cand.sort(reverse=True)

    for _, n_vox, lab in cand[: max(int(max_components), 0)]:
        sl = _as_slices(boxes[lab])
        m[sl] |= labels[sl] == lab
        info["n_added"] += 1
        info["added_ml"] += float(n_vox * vox_ml)

    if return_info:
        return m.astype(np.uint8), info
    return m.astype(np.uint8)


def bridge_components(
    mask: np.ndarray,
    spacing: Sequence[float],
    *,
    closing_voxels: int = 0,
    forbidden=None,
    max_added_ml: Optional[float] = 5.0,
    return_info: bool = False,
):
    """Join components separated by a gap of at most ``2 * closing_voxels`` voxels.

    Multi-assignment is not punished, so a merge turns two unmatched components into one
    false positive; it only loses when the union's IoU with a small lesion falls below
    0.1, hence the one-voxel default.  ``max_added_ml`` refuses the whole closing if it
    would add more than that, and ``forbidden`` voxels are never filled.
    """
    m = np.ascontiguousarray(np.asarray(mask) > 0)
    info: Dict[str, Any] = {"closing_voxels": int(closing_voxels), "added_voxels": 0,
                            "added_ml": 0.0, "refused": False}
    n = int(closing_voxels)
    if n <= 0 or not m.any():
        if return_info:
            return m.astype(np.uint8), info
        return m.astype(np.uint8)

    from scipy import ndimage

    nz = np.argwhere(m)
    lo = np.maximum(nz.min(0) - (n + 1), 0)
    hi = np.minimum(nz.max(0) + (n + 2), np.asarray(m.shape))
    sl = tuple(slice(int(a), int(b)) for a, b in zip(lo, hi))
    sub = m[sl]
    st = ndimage.generate_binary_structure(3, 3)
    closed = ndimage.binary_erosion(
        ndimage.binary_dilation(sub, st, iterations=n), st, iterations=n, border_value=1)
    add = closed & ~sub
    if forbidden is not None:
        fb = np.asarray(forbidden)
        if fb.shape == m.shape:
            add &= ~fb[sl].astype(bool)
    n_add = int(add.sum())
    vox_ml = voxel_volume_ml(spacing)
    if max_added_ml is not None and n_add * vox_ml > float(max_added_ml):
        info["refused"] = True
        if return_info:
            return m.astype(np.uint8), info
        return m.astype(np.uint8)

    m[sl] |= add
    info["added_voxels"] = n_add
    info["added_ml"] = float(n_add * vox_ml)
    if return_info:
        return m.astype(np.uint8), info
    return m.astype(np.uint8)


def tracer_suv_floor(
    mask: np.ndarray,
    pet: np.ndarray,
    tracer: str = "fdg",
    *,
    floor: Optional[float] = None,
    mode: str = "component",
    connectivity: int = 18,
    protect_points=None,
    min_components_kept: int = 1,
    return_info: bool = False,
):
    """Remove mask content below the tracer-specific absolute SUV floor.

    FDG 1.5, PSMA 1.0 from ``config.TRACER_SUV_FLOOR``, overridable with ``floor``.
    ``mode="component"`` drops a component whose SUVmax is below the floor, ``"voxel"``
    also erases individual sub-floor voxels (sharper Dice, but it can fragment a lesion),
    ``"off"`` is a no-op.
    """
    m = np.ascontiguousarray(np.asarray(mask) > 0)
    info: Dict[str, Any] = {
        "floor": None,
        "n_removed": 0,
        "removed_voxels": 0,
        "n_rescued_from_emptying": 0,
    }
    if mode == "off" or not m.any():
        if return_info:
            return m.astype(np.uint8), info
        return m.astype(np.uint8)

    thr = float(floor) if floor is not None else float(
        TRACER_SUV_FLOOR.get(str(tracer).lower(), TRACER_SUV_FLOOR["unknown"])
    )
    info["floor"] = thr
    pet = np.asarray(pet)
    before = int(m.sum())

    protect = points_in_bounds(unique_points(as_points_array(protect_points)), m.shape)

    if mode == "voxel":
        keep_pts = np.zeros(m.shape, dtype=bool)
        if len(protect):
            keep_pts[protect[:, 0], protect[:, 1], protect[:, 2]] = True
        m &= (pet >= thr) | keep_pts

    labels = cc3d.connected_components(m.view(np.uint8), connectivity=connectivity)
    stats = cc3d.statistics(labels)
    counts = stats["voxel_counts"]
    boxes = stats["bounding_boxes"]
    protected = set(_protected_labels(labels, protect).tolist())

    doomed = [
        lab
        for lab in range(1, len(counts))
        if counts[lab] > 0
        and lab not in protected
        and _component_suv_max(pet, labels, boxes[lab], lab) < thr
    ]
    alive = int(np.count_nonzero(counts[1:]))
    if alive - len(doomed) < min_components_kept and min_components_kept > 0:
        rescued = rank_components(labels, boxes, counts, pet, min_components_kept)
        doomed = [lab for lab in doomed if lab not in rescued]
        info["n_rescued_from_emptying"] = len(rescued)

    n_removed = 0
    for lab in doomed:
        sl = _as_slices(boxes[lab])
        m[sl] &= ~(labels[sl] == lab)
        n_removed += 1

    info["n_removed"] = n_removed
    info["removed_voxels"] = before - int(m.sum())
    if return_info:
        return m.astype(np.uint8), info
    return m.astype(np.uint8)


def fill_small_holes(
    mask: np.ndarray,
    spacing: Sequence[float],
    max_hole_ml: float = 0.5,
    *,
    forbidden=None,
    return_info: bool = False,
):
    """Fill background cavities strictly inside a lesion and smaller than ``max_hole_ml``.

    Background components use 6-connectivity, the topological dual of an 18-connected
    foreground, inside the bounding box padded by one voxel, so a cavity touching the
    padding ring is outside the lesion and left alone.  ``forbidden`` is never filled.
    """
    m = np.ascontiguousarray(np.asarray(mask) > 0)
    info: Dict[str, Any] = {"n_filled": 0, "filled_voxels": 0}
    if not m.any() or max_hole_ml <= 0:
        if return_info:
            return m.astype(np.uint8), info
        return m.astype(np.uint8)

    vox_ml = voxel_volume_ml(spacing)
    max_vox = max_hole_ml / vox_ml

    nz = np.argwhere(m)
    lo = np.maximum(nz.min(axis=0) - 1, 0)
    hi = np.minimum(nz.max(axis=0) + 1, np.asarray(m.shape) - 1)
    sl = tuple(slice(int(a), int(b) + 1) for a, b in zip(lo, hi))
    sub = m[sl]

    holes = cc3d.connected_components((~sub).view(np.uint8), connectivity=6)
    border = set(
        np.unique(
            np.concatenate(
                [
                    holes[0, :, :].ravel(),
                    holes[-1, :, :].ravel(),
                    holes[:, 0, :].ravel(),
                    holes[:, -1, :].ravel(),
                    holes[:, :, 0].ravel(),
                    holes[:, :, -1].ravel(),
                ]
            )
        ).tolist()
    )
    counts = np.bincount(holes.ravel())
    fill_ids = [
        lab
        for lab in range(1, len(counts))
        if lab not in border and 0 < counts[lab] <= max_vox
    ]
    if fill_ids:
        add = np.isin(holes, fill_ids)
        if forbidden is not None:
            fb = np.asarray(forbidden)
            add &= ~(fb[sl].astype(bool) if fb.shape == m.shape else np.zeros_like(add))
        sub |= add
        m[sl] = sub
        info["n_filled"] = len(fill_ids)
        info["filled_voxels"] = int(add.sum())

    if return_info:
        return m.astype(np.uint8), info
    return m.astype(np.uint8)


def split_large_components(
    mask: np.ndarray,
    pet: np.ndarray,
    spacing: Sequence[float],
    *,
    min_volume_ml: float = 10.0,
    h_depth_suv: float = 1.0,
    peak_min_distance_mm: float = 10.0,
    min_fragment_ml: float = 0.5,
    max_fragments: int = 8,
    connectivity: int = 18,
    protect_points=None,
    return_info: bool = False,
):
    """Cut an oversized component along the watershed line between its PET maxima.

    A small lesion swallowed by a much larger predicted component is scored as a miss on
    *both* sides: the union's IoU against the small lesion falls below 0.1, so the lesion
    counts as a false negative and the component, matching nothing else, counts as a
    false positive.  The remedy needs no extra sensitivity -- the voxels are already
    predicted -- only a cut.

    A component qualifies when it is at least ``min_volume_ml`` and contains two or more
    maxima of the ~1 mL-smoothed SUV that stand at least ``h_depth_suv`` above the saddle
    joining them (``skimage.morphology.h_maxima``) and lie at least
    ``peak_min_distance_mm`` apart.  The cut is the watershed line between those maxima,
    removed from the mask so the fragments are genuinely disconnected under
    ``connectivity``; if removing the line does not separate them the line is thickened
    once, and if it still does not the split is abandoned and the component restored.

    Splitting a component that is already matched costs nothing (the matcher does not
    punish splits), so the only real risk is a fragment that matches nothing, which
    ``min_fragment_ml`` bounds.  A component holding a tumor scribble is never split --
    the scribble is a statement about one lesion, and a cut could put its voxels in a
    fragment we would then have to grow back.
    """
    from scipy import ndimage
    from skimage.morphology import h_maxima
    from skimage.segmentation import watershed

    m = np.ascontiguousarray(np.asarray(mask) > 0)
    info: Dict[str, Any] = {
        "n_candidates": 0, "n_split": 0, "n_fragments_added": 0, "removed_ml": 0.0,
    }
    if not m.any() or min_volume_ml <= 0:
        return (m.astype(np.uint8), info) if return_info else m.astype(np.uint8)

    vox_ml = voxel_volume_ml(spacing)
    labels = cc3d.connected_components(m.view(np.uint8), connectivity=connectivity)
    stats = cc3d.statistics(labels)
    counts, boxes = stats["voxel_counts"], stats["bounding_boxes"]
    protected = set(_protected_labels(labels, protect_points).tolist())
    box_mm = [max(1, int(2 * int(6.2 // float(sp)) + 1)) for sp in spacing]
    pet = np.asarray(pet)

    for lab in range(1, len(counts)):
        if counts[lab] == 0 or lab in protected:
            continue
        if counts[lab] * vox_ml < min_volume_ml:
            continue
        info["n_candidates"] += 1
        sl = _as_slices(boxes[lab])
        comp = labels[sl] == lab
        pet_c = np.asarray(pet[sl], dtype=np.float32)
        smooth = ndimage.uniform_filter(pet_c, size=box_mm, mode="nearest")

        field = np.where(comp, smooth, 0.0).astype(np.float32)
        peaks = (h_maxima(field, float(h_depth_suv)) > 0) & comp
        if not peaks.any():
            continue
        plateaus = cc3d.connected_components(
            np.ascontiguousarray(peaks).view(np.uint8), connectivity=26)
        ids = [i for i in range(1, int(plateaus.max()) + 1) if (plateaus == i).any()]
        if len(ids) < 2:
            continue
        # drop maxima that sit on top of each other
        cents = np.array([np.argwhere(plateaus == i).mean(axis=0) for i in ids])
        keep, sp = [], np.asarray(spacing, dtype=float)
        for idx, c in enumerate(cents):
            if all(np.linalg.norm((c - cents[j]) * sp) >= peak_min_distance_mm for j in keep):
                keep.append(idx)
        if len(keep) < 2:
            continue
        markers = np.zeros(comp.shape, dtype=np.int32)
        for new_id, idx in enumerate(keep[:max_fragments], start=1):
            markers[plateaus == ids[idx]] = new_id

        ws = watershed(-smooth, markers, mask=comp, watershed_line=True)
        cut = comp & (ws == 0)
        if not cut.any():
            continue
        kept = comp & ~cut
        if not kept.any():
            continue
        frag = cc3d.connected_components(
            np.ascontiguousarray(kept).view(np.uint8), connectivity=connectivity)
        n_frag = int(frag.max())
        if n_frag < 2:
            # the line was face-connected only; thicken it once and retry
            cut = ndimage.binary_dilation(cut, structure=np.ones((3, 3, 3), bool)) & comp
            kept = comp & ~cut
            if not kept.any():
                continue
            frag = cc3d.connected_components(
                np.ascontiguousarray(kept).view(np.uint8), connectivity=connectivity)
            n_frag = int(frag.max())
            if n_frag < 2:
                continue
        fc = np.bincount(frag.ravel())[1:]
        if len(fc) == 0 or (fc.min() * vox_ml) < min_fragment_ml:
            continue
        m[sl] &= ~cut
        info["n_split"] += 1
        info["n_fragments_added"] += n_frag - 1
        info["removed_ml"] += float(int(cut.sum()) * vox_ml)

    return (m.astype(np.uint8), info) if return_info else m.astype(np.uint8)


def cleanup_mask(
    mask: np.ndarray,
    pet: np.ndarray,
    spacing: Sequence[float],
    tracer: str = "fdg",
    cfg: Optional[CleanupConfig] = None,
    *,
    protect_points=None,
    background_points=None,
    prob=None,
    iteration: int = 0,
    forbidden=None,
    return_info: bool = False,
):
    """The full cleanup stage used by the pipeline.

    Runs floor -> component pruning -> recall recruitment -> hole filling.  Pruning uses
    the two-parameter rule unless ``cfg.rule_v2``; recruitment is off unless
    ``cfg.recruit_prob_threshold`` is set.
    """
    cfg = cfg or CleanupConfig()
    info: Dict[str, Any] = {}
    mask, info["suv_floor"] = tracer_suv_floor(
        mask,
        pet,
        tracer,
        floor=cfg.suv_floor,
        mode=cfg.suv_floor_mode,
        connectivity=cfg.connectivity,
        protect_points=protect_points,
        min_components_kept=cfg.min_components_kept,
        return_info=True,
    )
    if cfg.rule_v2:
        rules = resolve_v2_rules(cfg, tracer)
        if cfg.v2_silence_requires_bg and _n_points(background_points) == 0:
            rules = [dict(r, silence_decay=1.0) for r in rules]
        mask, info["small_components"] = remove_components_v2(
            mask,
            pet,
            spacing,
            prob=prob,
            rules=rules,
            iteration=iteration,
            connectivity=cfg.connectivity,
            protect_points=protect_points,
            background_points=background_points,
            min_components_kept=cfg.min_components_kept,
            return_info=True,
        )
    else:
        mask, info["small_components"] = remove_small_components(
            mask,
            pet,
            spacing,
            min_volume_ml=cfg.min_volume_ml,
            suv_gate=cfg.suv_gate,
            connectivity=cfg.connectivity,
            protect_points=protect_points,
            min_components_kept=cfg.min_components_kept,
            return_info=True,
        )
    if cfg.recruit_prob_threshold is not None:
        mask, info["recruit"] = recruit_components(
            mask,
            prob,
            pet,
            spacing,
            threshold=cfg.recruit_prob_threshold,
            min_suv_max=cfg.recruit_min_suv_max,
            min_volume_ml=cfg.recruit_min_volume_ml,
            max_components=cfg.recruit_max_components,
            connectivity=cfg.connectivity,
            return_info=True,
        )

    if cfg.split_large_components:
        mask, info["split"] = split_large_components(
            mask,
            pet,
            spacing,
            min_volume_ml=cfg.split_min_volume_ml,
            h_depth_suv=cfg.split_h_depth_suv,
            peak_min_distance_mm=cfg.split_peak_min_distance_mm,
            min_fragment_ml=cfg.split_min_fragment_ml,
            max_fragments=cfg.split_max_fragments,
            connectivity=cfg.connectivity,
            protect_points=protect_points,
            return_info=True,
        )
    if cfg.fill_holes:
        mask, info["holes"] = fill_small_holes(
            mask, spacing, cfg.fill_holes_max_ml, forbidden=forbidden, return_info=True
        )
    if return_info:
        return mask, info
    return mask
