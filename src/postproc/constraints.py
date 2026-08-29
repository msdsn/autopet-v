"""Accumulated constraint state, persisted in the per-case cache directory.

One container call is one iteration; nothing survives between calls except what we
write into the cache ourselves -- previous probability and mask, the region a background
scribble deleted, the tracer, the iteration counter.  Writes are atomic.  The directory
may be missing or read-only, so nothing here raises and every constraint stays derivable
from the evaluator's cumulative ``lesion-clicks.json`` via ``from_scribbles``.
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .utils import as_points_array, points_in_bounds, unique_points

__all__ = [
    "ConstraintState",
    "CaseCache",
    "save_packed_mask",
    "load_packed_mask",
    "save_prob_u8",
    "load_prob_u8",
]

CONSTRAINTS_FILE = "postproc_constraints.json"
PREV_PROB_FILE = "postproc_prev_prob.npy"
PREV_MASK_FILE = "postproc_prev_mask.npz"
BG_REGION_FILE = "postproc_bg_region.npz"


# ---------------------------------------------------------------------------
# atomic IO helpers
# ---------------------------------------------------------------------------
def _atomic_write(path: str, writer) -> bool:
    """Write atomically.  Returns False (never raises) if the location is unusable.

    Losing the cache costs us the monotone blend, not correctness: every constraint is
    re-derivable from the accumulated scribble list in the input.
    """
    directory = os.path.dirname(os.path.abspath(path)) or "."
    tmp = None
    try:
        os.makedirs(directory, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=directory, prefix=".pp_tmp_")
        os.close(fd)
        writer(tmp)
        os.replace(tmp, path)
        return True
    except OSError:
        return False
    finally:
        if tmp is not None and os.path.exists(tmp):
            try:
                os.remove(tmp)
            except OSError:
                pass


def save_packed_mask(path: str, mask: np.ndarray) -> None:
    """Bit-pack a boolean volume (52.8 M voxels -> 6.6 MB raw, ~50 kB compressed)."""
    mask = np.ascontiguousarray(mask.astype(bool))
    packed = np.packbits(mask.reshape(-1))
    def _write(tmp: str) -> None:
        # write through a file object: np.save/np.savez append an extension to a *path*
        with open(tmp, "wb") as fh:
            np.savez_compressed(fh, packed=packed, shape=np.asarray(mask.shape, np.int64))

    _atomic_write(path, _write)


def load_packed_mask(path: str) -> Optional[np.ndarray]:
    if not os.path.exists(path):
        return None
    try:
        with np.load(path) as z:
            shape = tuple(int(s) for s in z["shape"])
            n = int(np.prod(shape))
            flat = np.unpackbits(z["packed"], count=n).astype(bool)
        return flat.reshape(shape)
    except Exception:
        return None


def save_prob_u8(path: str, prob: np.ndarray) -> None:
    """Store a [0, 1] probability as uint8; 1/255 is far below any threshold we use.

    Uncompressed on purpose: 53 MB of ``np.save`` is ~0.1 s, compressing it is seconds.
    """
    q = np.clip(np.asarray(prob, dtype=np.float32), 0.0, 1.0)
    q = np.rint(q * 255.0).astype(np.uint8)
    def _write(tmp: str) -> None:
        with open(tmp, "wb") as fh:
            np.save(fh, q, allow_pickle=False)

    _atomic_write(path, _write)


def load_prob_u8(path: str) -> Optional[np.ndarray]:
    if not os.path.exists(path):
        return None
    try:
        q = np.load(path, allow_pickle=False)
    except Exception:
        return None
    return q.astype(np.float32) / 255.0


# ---------------------------------------------------------------------------
# state
# ---------------------------------------------------------------------------
@dataclass
class ConstraintState:
    """Everything the interaction layer accumulates over the 5 iterations of a case.

    ``tumor_points`` / ``background_points`` are ``[i, j, k]`` nibabel indices, the
    union over all iterations seen so far.  They are the authoritative constraint:
    every iteration re-applies all of them, never just the newest scribble.
    """

    tumor_points: List[List[int]] = field(default_factory=list)
    background_points: List[List[int]] = field(default_factory=list)
    #: index of the next iteration to be run (0 before the first call).
    iteration: int = 0
    #: number of completed ``predict`` calls.
    n_calls: int = 0
    tracer: Optional[str] = None
    shape: Optional[List[int]] = None
    spacing: Optional[List[float]] = None
    negative_gate_fired: bool = False
    history: List[Dict[str, Any]] = field(default_factory=list)
    extra: Dict[str, Any] = field(default_factory=dict)

    # -- accessors ---------------------------------------------------------
    def tumor_array(self, shape: Optional[Sequence[int]] = None) -> np.ndarray:
        pts = unique_points(as_points_array(self.tumor_points))
        shape = shape if shape is not None else self.shape
        if shape is not None:
            pts = points_in_bounds(pts, shape)
        return pts

    def background_array(self, shape: Optional[Sequence[int]] = None) -> np.ndarray:
        pts = unique_points(as_points_array(self.background_points))
        shape = shape if shape is not None else self.shape
        if shape is not None:
            pts = points_in_bounds(pts, shape)
        if len(pts) == 0:
            return pts
        # A voxel cannot be both.  Tumor wins (it is the guarantee we must satisfy
        # last in the pipeline).  The official simulator never produces a conflict,
        # but a human annotator in challenge Category 2 can.
        tumor = unique_points(as_points_array(self.tumor_points))
        if len(tumor) == 0:
            return pts
        tset = {tuple(p) for p in tumor.tolist()}
        keep = np.array([tuple(p) not in tset for p in pts.tolist()], dtype=bool)
        return pts[keep]

    @property
    def n_tumor(self) -> int:
        return len(self.tumor_array())

    @property
    def n_background(self) -> int:
        return len(self.background_array())

    # -- mutation ----------------------------------------------------------
    def add_scribbles(self, scribbles: Optional[Dict[str, Any]]) -> Tuple[int, int]:
        """Merge a ``{"tumor": [...], "background": [...]}`` dict into the state.

        Returns ``(n_new_tumor, n_new_background)``.  Idempotent: re-adding the same
        accumulated dict (which is what the evaluator hands us every iteration) is a
        no-op.
        """
        if not scribbles:
            return (0, 0)
        n_t0, n_b0 = len(self.tumor_points), len(self.background_points)

        def _merge(existing: List[List[int]], new: Any) -> List[List[int]]:
            new_pts = as_points_array(new)
            if len(new_pts) == 0:
                return existing
            seen = {tuple(p) for p in existing}
            for p in new_pts.tolist():
                t = tuple(int(v) for v in p)
                if t not in seen:
                    seen.add(t)
                    existing.append(list(t))
            return existing

        self.tumor_points = _merge(self.tumor_points, scribbles.get("tumor", []))
        self.background_points = _merge(self.background_points, scribbles.get("background", []))
        return (len(self.tumor_points) - n_t0, len(self.background_points) - n_b0)

    def infer_iteration(self, spacing: Optional[Sequence[float]] = None) -> int:
        """Estimate the iteration index from the accumulated scribbles alone.

        Used only when no state directory is available.  One scribble arrives per
        iteration, so the number of spatially separate clusters is the number of
        corrections so far; clamped to 0..5.
        """
        from .utils import cluster_points  # local import: avoids a cycle at module load

        sp = tuple(spacing or self.spacing or (1.0, 1.0, 1.0))
        n = 0
        for pts in (self.tumor_array(), self.background_array()):
            if len(pts):
                n += len(cluster_points(pts, sp, 10.0))
        return int(min(max(n, 0), 5))

    @classmethod
    def from_scribbles(
        cls,
        scribbles: Optional[Dict[str, Any]],
        shape: Optional[Sequence[int]] = None,
        spacing: Optional[Sequence[float]] = None,
    ) -> "ConstraintState":
        """Rebuild the whole constraint set from the input json, with no cache at all.

        ``lesion-clicks.json`` carries every scribble so far, so no constraint is lost;
        only the derived state (previous probability, mask, deleted regions) is.
        """
        st = cls(
            shape=[int(s) for s in shape] if shape is not None else None,
            spacing=[float(s) for s in spacing] if spacing is not None else None,
        )
        st.add_scribbles(scribbles)
        st.iteration = st.infer_iteration(spacing)
        return st

    # -- persistence -------------------------------------------------------
    def save(self, cache_dir: str) -> Optional[str]:
        if not cache_dir:
            return None
        path = os.path.join(cache_dir, CONSTRAINTS_FILE)
        payload = asdict(self)
        return path if _atomic_write(path, lambda p: _dump_json(p, payload)) else None

    @classmethod
    def load(cls, cache_dir: Optional[str]) -> "ConstraintState":
        """Load, or return a fresh empty state.  A corrupt cache degrades to
        "first iteration" rather than raising."""
        if not cache_dir:
            return cls()
        path = os.path.join(cache_dir, CONSTRAINTS_FILE)
        if not os.path.exists(path):
            return cls()
        try:
            with open(path, "r") as f:
                data = json.load(f)
        except Exception:
            return cls()
        known = {f_.name for f_ in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in data.items() if k in known})


def _dump_json(path: str, payload: Dict[str, Any]) -> None:
    with open(path, "w") as f:
        json.dump(payload, f, indent=1, default=_json_default)


def _json_default(o: Any) -> Any:
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.bool_,)):
        return bool(o)
    raise TypeError(f"not json serialisable: {type(o)}")


# ---------------------------------------------------------------------------
# cache directory wrapper
# ---------------------------------------------------------------------------
class CaseCache:
    """Failure-tolerant wrapper around the per-case cache directory.

    With ``cache_dir=None`` every ``load_*`` returns ``None`` and every ``save_*`` is a
    no-op, so the pipeline needs no special case for single-shot inference.
    """

    def __init__(self, cache_dir: Optional[str]):
        # No mkdir here: the directory may not exist and may not be creatable.
        # Directories are created lazily, on write.
        self.dir = cache_dir or None

    @property
    def exists(self) -> bool:
        return bool(self.dir) and os.path.isdir(self.dir)

    def has_state(self) -> bool:
        """True iff a constraint file from an earlier iteration is actually readable."""
        return bool(self.dir) and os.path.exists(os.path.join(self.dir, CONSTRAINTS_FILE))

    # -- state ------------------------------------------------------------
    def load_state(self) -> ConstraintState:
        return ConstraintState.load(self.dir)

    def save_state(self, state: ConstraintState) -> None:
        if self.dir:
            state.save(self.dir)

    # -- arrays -----------------------------------------------------------
    def _path(self, name: str) -> Optional[str]:
        return os.path.join(self.dir, name) if self.dir else None

    def load_prev_prob(self) -> Optional[np.ndarray]:
        p = self._path(PREV_PROB_FILE)
        return load_prob_u8(p) if p else None

    def save_prev_prob(self, prob: Optional[np.ndarray]) -> None:
        p = self._path(PREV_PROB_FILE)
        if p and prob is not None:
            save_prob_u8(p, prob)

    def load_prev_mask(self) -> Optional[np.ndarray]:
        p = self._path(PREV_MASK_FILE)
        return load_packed_mask(p) if p else None

    def save_prev_mask(self, mask: Optional[np.ndarray]) -> None:
        p = self._path(PREV_MASK_FILE)
        if p and mask is not None:
            save_packed_mask(p, mask)

    def load_bg_region(self) -> Optional[np.ndarray]:
        p = self._path(BG_REGION_FILE)
        return load_packed_mask(p) if p else None

    def save_bg_region(self, mask: Optional[np.ndarray]) -> None:
        p = self._path(BG_REGION_FILE)
        if p and mask is not None:
            save_packed_mask(p, mask)

    def accumulate_bg_region(self, removed: Optional[np.ndarray]) -> Optional[np.ndarray]:
        """Union the newly deleted region into the persisted background region.

        Monotone: a region deleted by a background scribble never comes back.
        """
        if removed is None:
            return self.load_bg_region()
        prev = self.load_bg_region()
        if prev is not None and prev.shape == removed.shape:
            removed = prev | removed
        self.save_bg_region(removed)
        return removed
