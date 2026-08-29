"""Turn a ground-truth patch label into a plausible "previous prediction".

Deriving it from the model's own error would cost a forward pass per sample, so we
sample an error pattern instead: dilate/erode a component, drop it, shift it, paste
a blob outside the label, or emit an empty / exact copy. The false-positive blobs
go on high-PET background, where physiological-uptake false positives actually sit.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence, Tuple

import numpy as np
from scipy import ndimage as ndi

__all__ = ["CorruptionConfig", "make_fake_prev_prediction"]


@dataclass
class CorruptionConfig:
    # P = 0 everywhere. Iteration 0 has no previous prediction and the cache can be
    # lost, so an all-zero ch4 has to be a well-trained state in its own right.
    p_empty: float = 0.20
    p_perfect: float = 0.10               # P = L exactly
    # per-component operation probabilities: keep / dilate / erode / drop / shift
    component_ops: Tuple[str, ...] = ("keep", "dilate", "erode", "drop", "shift")
    component_op_p: Tuple[float, ...] = (0.30, 0.20, 0.20, 0.15, 0.15)
    max_morph_iters: int = 2
    max_shift: int = 4
    p_add_fp: float = 0.5                 # probability of adding any FP blob at all
    # ... and when the patch has no lesion at all. One FP voxel zeroes the Dice of a
    # negative case for the whole run, so that state is worth over-generating.
    p_add_fp_empty_label: float = 0.8
    max_fp: int = 3
    fp_radius_range: Tuple[int, int] = (1, 5)
    pet_channel: int = 1
    pet_top_frac: float = 0.02            # FP blobs go on the top 2% PET voxels
    n_pet_samples: int = 20000            # subsample size for the PET percentile
    exclude_dilate: int = 2               # keep FP blobs this far from the label


def _component_slices(mask: np.ndarray, connectivity: int = 26):
    """Return (labelled array, list of slice tuples) for the 3-D components."""
    if not mask.any():
        return None, []
    try:
        import cc3d
        cc = cc3d.connected_components(mask.astype(np.uint8), connectivity=connectivity)
    except Exception:
        cc, _ = ndi.label(mask)
    objs = ndi.find_objects(cc)
    return cc, objs


def _pad_slices(objs: Sequence[slice], shape: Sequence[int], pad: int):
    out = []
    for sl, s in zip(objs, shape):
        out.append(slice(max(0, sl.start - pad), min(int(s), sl.stop + pad)))
    return tuple(out)


def _paste_ball(out: np.ndarray, centre: Sequence[int], radii: Sequence[int]) -> None:
    lo = [max(0, int(c) - int(r)) for c, r in zip(centre, radii)]
    hi = [min(int(s), int(c) + int(r) + 1) for c, r, s in zip(centre, radii, out.shape)]
    if any(h <= l for l, h in zip(lo, hi)):
        return
    grids = np.meshgrid(*[np.arange(l, h) - c for l, h, c in zip(lo, hi, centre)], indexing="ij")
    dist2 = sum((g.astype(np.float32) / max(1e-3, r)) ** 2 for g, r in zip(grids, radii))
    sl = tuple(slice(l, h) for l, h in zip(lo, hi))
    out[sl] |= (dist2 <= 1.0)


def _sample_high_pet_locations(pet: Optional[np.ndarray],
                               forbidden: np.ndarray,
                               n: int,
                               rng: np.random.Generator,
                               cfg: CorruptionConfig) -> list:
    """Pick `n` voxel indices with high PET intensity that lie outside `forbidden`."""
    shape = forbidden.shape
    total = int(np.prod(shape))
    m = min(cfg.n_pet_samples, total)
    flat_idx = rng.integers(0, total, size=m)
    idx = np.unravel_index(flat_idx, shape)
    ok = ~forbidden[idx]
    if not ok.any():
        return []
    idx = tuple(a[ok] for a in idx)
    if pet is not None:
        vals = pet[idx]
        n_top = max(n, int(np.ceil(cfg.pet_top_frac * len(vals))))
        n_top = min(n_top, len(vals))
        order = np.argpartition(-vals, n_top - 1)[:n_top]
        pick = rng.choice(order, size=min(n, len(order)), replace=False)
    else:
        pick = rng.choice(len(idx[0]), size=min(n, len(idx[0])), replace=False)
    return [tuple(int(a[p]) for a in idx) for p in np.atleast_1d(pick)]


def make_fake_prev_prediction(label: np.ndarray,
                              rng: np.random.Generator,
                              pet: Optional[np.ndarray] = None,
                              cfg: Optional[CorruptionConfig] = None) -> np.ndarray:
    """Return a uint8 (0/1) "previous prediction" derived from `label`.

    An all-background label is fine: the only corruption then available is a
    hallucinated blob, which is the negative-case state we want trained.
    """
    cfg = cfg or CorruptionConfig()
    L = np.asarray(label, dtype=bool)

    r = rng.random()
    if r < cfg.p_empty:
        return np.zeros(L.shape, dtype=np.uint8)
    if r < cfg.p_empty + cfg.p_perfect:
        return L.astype(np.uint8)

    P = np.zeros(L.shape, dtype=bool)
    cc, objs = _component_slices(L)

    ops = list(cfg.component_ops)
    op_p = np.asarray(cfg.component_op_p, dtype=np.float64)
    op_p = op_p / op_p.sum()

    if cc is not None:
        for cid, sl in enumerate(objs, start=1):
            if sl is None:
                continue
            op = str(rng.choice(ops, p=op_p))
            if op == "drop":
                continue
            pad = cfg.max_morph_iters + cfg.max_shift + 1
            psl = _pad_slices(sl, L.shape, pad)
            sub = (cc[psl] == cid)
            if op == "dilate":
                it = int(rng.integers(1, cfg.max_morph_iters + 1))
                sub = ndi.binary_dilation(sub, iterations=it)
            elif op == "erode":
                it = int(rng.integers(1, cfg.max_morph_iters + 1))
                eroded = ndi.binary_erosion(sub, iterations=it)
                sub = eroded if eroded.any() else sub
            elif op == "shift":
                shift = rng.integers(-cfg.max_shift, cfg.max_shift + 1, size=L.ndim)
                sub = ndi.shift(sub.astype(np.uint8), shift, order=0, mode="constant", cval=0).astype(bool)
            P[psl] |= sub

    # hallucinated false positives on high-PET background
    p_fp = cfg.p_add_fp_empty_label if not L.any() else cfg.p_add_fp
    if rng.random() < p_fp:
        n_fp = int(rng.integers(1, cfg.max_fp + 1))
        if cfg.exclude_dilate > 0 and L.any():
            forbidden = ndi.binary_dilation(L, iterations=int(cfg.exclude_dilate))
        else:
            forbidden = L.copy()
        forbidden |= P
        centres = _sample_high_pet_locations(pet, forbidden, n_fp, rng, cfg)
        lo, hi = cfg.fp_radius_range
        if centres:
            fp = np.zeros(L.shape, dtype=bool)
            for c in centres:
                radii = rng.integers(lo, hi + 1, size=L.ndim)
                _paste_ball(fp, c, radii)
            # a hallucination that overlaps the true label is not a false positive
            fp &= ~L
            P |= fp

    return P.astype(np.uint8)
