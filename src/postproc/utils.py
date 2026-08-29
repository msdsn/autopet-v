"""Small shared helpers.  Everything here is O(crop) or O(n_points) unless noted."""

from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np

__all__ = [
    "as_points_array",
    "unique_points",
    "points_in_bounds",
    "foreground_prob",
    "voxel_volume_ml",
    "ml_to_voxels",
    "ball_offsets",
    "paint_points",
    "points_mask",
    "bbox_of_points",
    "expand_bbox",
    "bbox_slices",
    "cluster_points",
    "timer",
]


# ---------------------------------------------------------------------------
# points
# ---------------------------------------------------------------------------
def as_points_array(points: Optional[Iterable[Sequence[int]]]) -> np.ndarray:
    """Normalise any point container to an ``(N, 3)`` int64 array.

    Accepts ``None``, ``[]``, list-of-lists, list-of-tuples, an ``(N, 3)`` array.
    """
    if points is None:
        return np.zeros((0, 3), dtype=np.int64)
    arr = np.asarray(points)
    if arr.size == 0:
        return np.zeros((0, 3), dtype=np.int64)
    if arr.dtype.kind == "f":  # json may hand us floats
        arr = np.rint(arr)
    return arr.astype(np.int64, copy=False).reshape(-1, 3)


def unique_points(points: np.ndarray) -> np.ndarray:
    """De-duplicate while preserving first-seen order (order matters for debugging)."""
    points = as_points_array(points)
    if len(points) == 0:
        return points
    _, idx = np.unique(points, axis=0, return_index=True)
    return points[np.sort(idx)]


def points_in_bounds(points: np.ndarray, shape: Sequence[int]) -> np.ndarray:
    """Drop points that fall outside ``shape`` (defensive: GC json is external input)."""
    points = as_points_array(points)
    if len(points) == 0:
        return points
    ok = np.ones(len(points), dtype=bool)
    for ax in range(3):
        ok &= (points[:, ax] >= 0) & (points[:, ax] < int(shape[ax]))
    return points[ok]


def bbox_of_points(points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    points = as_points_array(points)
    return points.min(axis=0), points.max(axis=0)


def expand_bbox(
    lo: np.ndarray, hi: np.ndarray, margin_vox: Sequence[int], shape: Sequence[int]
) -> Tuple[np.ndarray, np.ndarray]:
    lo = np.maximum(np.asarray(lo, dtype=np.int64) - np.asarray(margin_vox, dtype=np.int64), 0)
    hi = np.minimum(
        np.asarray(hi, dtype=np.int64) + np.asarray(margin_vox, dtype=np.int64),
        np.asarray(shape, dtype=np.int64) - 1,
    )
    return lo, hi


def bbox_slices(lo: Sequence[int], hi: Sequence[int]) -> Tuple[slice, slice, slice]:
    """Inclusive ``hi`` -> python slices."""
    return tuple(slice(int(a), int(b) + 1) for a, b in zip(lo, hi))  # type: ignore[return-value]


def cluster_points(points: np.ndarray, spacing: Sequence[float], radius_mm: float) -> List[np.ndarray]:
    """Group points into spatial clusters (single-linkage at ``radius_mm``).

    One scribble is a short 2-D line, so it always collapses to a single cluster;
    accumulated scribbles from different iterations sit in different lesions and stay
    apart.  Returns a list of ``(n_i, 3)`` index arrays into ``points``.
    """
    points = as_points_array(points)
    if len(points) == 0:
        return []
    if len(points) == 1:
        return [np.array([0])]

    coords_mm = points * np.asarray(spacing, dtype=np.float64)[None, :]
    try:
        from scipy.spatial import cKDTree
        from scipy.sparse import csr_matrix
        from scipy.sparse.csgraph import connected_components

        tree = cKDTree(coords_mm)
        pairs = tree.query_pairs(radius_mm, output_type="ndarray")
        n = len(points)
        if len(pairs) == 0:
            return [np.array([i]) for i in range(n)]
        data = np.ones(len(pairs), dtype=np.uint8)
        graph = csr_matrix((data, (pairs[:, 0], pairs[:, 1])), shape=(n, n))
        n_comp, labels = connected_components(graph, directed=False)
        return [np.where(labels == c)[0] for c in range(n_comp)]
    except ImportError:  # pragma: no cover - scipy is a hard dependency anyway
        return [np.arange(len(points))]


# ---------------------------------------------------------------------------
# probabilities / volumes
# ---------------------------------------------------------------------------
def foreground_prob(prob: Optional[np.ndarray], shape: Optional[Sequence[int]] = None) -> Optional[np.ndarray]:
    """Return the lesion-class probability as a 3-D float32 array (or ``None``).

    Accepts ``(X, Y, Z)`` (already the foreground channel) or ``(C, X, Y, Z)``
    (nnU-Net softmax, channel 1 == lesion, per ``Predictor.predict``).
    """
    if prob is None:
        return None
    prob = np.asarray(prob)
    if prob.ndim == 3:
        out = prob
    elif prob.ndim == 4:
        if prob.shape[0] == 1:
            out = prob[0]
        elif prob.shape[0] == 2:
            out = prob[1]
        else:
            # multi-class: lesion is class 1 by nnU-Net convention for this dataset
            out = prob[1]
    else:
        raise ValueError(f"cannot interpret probability array of shape {prob.shape}")
    if shape is not None and tuple(out.shape) != tuple(shape):
        raise ValueError(f"probability shape {out.shape} != volume shape {tuple(shape)}")
    return np.ascontiguousarray(out, dtype=np.float32)


def voxel_volume_ml(spacing: Sequence[float]) -> float:
    """mm^3 -> mL (1 mL == 1000 mm^3)."""
    sx, sy, sz = (float(s) for s in spacing)
    return sx * sy * sz / 1000.0


def ml_to_voxels(volume_ml: float, spacing: Sequence[float]) -> float:
    return float(volume_ml) / voxel_volume_ml(spacing)


# ---------------------------------------------------------------------------
# ball painting (no full-volume EDT -- that would cost ~10 s on 400x400x330)
# ---------------------------------------------------------------------------
def ball_offsets(radius_mm: float, spacing: Sequence[float]) -> np.ndarray:
    """Integer voxel offsets of a mm-radius ball.  Always contains ``(0, 0, 0)``."""
    spacing = np.asarray(spacing, dtype=np.float64)
    if radius_mm <= 0:
        return np.zeros((1, 3), dtype=np.int64)
    rad_vox = np.maximum(np.floor(radius_mm / spacing).astype(int), 0)
    grids = np.meshgrid(
        *[np.arange(-r, r + 1) for r in rad_vox], indexing="ij"
    )
    offs = np.stack([g.ravel() for g in grids], axis=1)
    dist = np.sqrt(((offs * spacing[None, :]) ** 2).sum(axis=1))
    return offs[dist <= radius_mm + 1e-9].astype(np.int64)


def paint_points(
    mask: np.ndarray,
    points: np.ndarray,
    radius_mm: float = 0.0,
    spacing: Sequence[float] = (1.0, 1.0, 1.0),
    value: bool = True,
) -> np.ndarray:
    """Set a ball of ``radius_mm`` around every point.  In-place on ``mask``."""
    points = as_points_array(points)
    if len(points) == 0:
        return mask
    offs = ball_offsets(radius_mm, spacing)
    shape = np.asarray(mask.shape, dtype=np.int64)
    # (n_points * n_offsets, 3) -- with ~200 points and a 6 mm ball this is ~1e5 rows
    coords = (points[:, None, :] + offs[None, :, :]).reshape(-1, 3)
    ok = np.ones(len(coords), dtype=bool)
    for ax in range(3):
        ok &= (coords[:, ax] >= 0) & (coords[:, ax] < shape[ax])
    coords = coords[ok]
    mask[coords[:, 0], coords[:, 1], coords[:, 2]] = value
    return mask


def points_mask(
    shape: Sequence[int],
    points: np.ndarray,
    radius_mm: float = 0.0,
    spacing: Sequence[float] = (1.0, 1.0, 1.0),
) -> np.ndarray:
    out = np.zeros(tuple(int(s) for s in shape), dtype=bool)
    return paint_points(out, points, radius_mm, spacing)


# ---------------------------------------------------------------------------
@contextmanager
def timer(store: Optional[dict], key: str):
    """``with timer(info, "bg_compliance"): ...`` -> ``info["t_bg_compliance"]`` seconds."""
    t0 = time.perf_counter()
    try:
        yield
    finally:
        if store is not None:
            store[f"t_{key}"] = time.perf_counter() - t0
