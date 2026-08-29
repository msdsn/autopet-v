"""Guidance-channel encoding: scribbles as a clipped Euclidean distance transform.

e(x) = max(0, 1 - d(x) / R), the EDT encoding from LesionLocator (arXiv:2508.21680),
bounded in [0, 1] so the channel passes nnU-Net's NoNormalization. The max over
per-point cones is the clipped EDT of the point set, so stamping a precomputed
kernel beats a full-patch EDT -- this runs in a dataloader worker per patch.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Iterable, Optional, Sequence, Tuple

import numpy as np

__all__ = ["clipped_edt_kernel", "stamp_clipped_edt", "guidance_map_from_coords"]


@lru_cache(maxsize=16)
def clipped_edt_kernel(radius: float, spacing: Optional[Tuple[float, ...]] = None,
                       ndim: int = 3) -> np.ndarray:
    """Return the (2r+1)^ndim kernel holding max(0, 1 - d/R) around its centre.

    radius is in voxels unless spacing is given, in which case it is in the same
    unit as spacing and the kernel becomes anisotropic.
    """
    if spacing is None:
        half = [int(np.ceil(radius))] * ndim
        sp = (1.0,) * ndim
    else:
        sp = tuple(float(s) for s in spacing)
        assert len(sp) == ndim
        half = [int(np.ceil(radius / s)) for s in sp]

    grids = np.meshgrid(*[np.arange(-h, h + 1) * s for h, s in zip(half, sp)], indexing="ij")
    dist = np.sqrt(sum(g.astype(np.float32) ** 2 for g in grids))
    kernel = np.clip(1.0 - dist / float(radius), 0.0, 1.0).astype(np.float32)
    return kernel


def stamp_clipped_edt(out: np.ndarray,
                      coords: Iterable[Sequence[int]],
                      radius: float = 10.0,
                      spacing: Optional[Tuple[float, ...]] = None) -> np.ndarray:
    """Accumulate max(0, 1 - d/R) cones for every point in coords into out.

    out is modified in place and returned. coords are integer index tuples in the
    frame of out; points outside it are ignored.
    """
    coords = list(coords)
    if len(coords) == 0:
        return out
    ndim = out.ndim
    kernel = clipped_edt_kernel(float(radius), spacing, ndim)
    half = [s // 2 for s in kernel.shape]

    for pt in coords:
        lo, hi, klo, khi = [], [], [], []
        ok = True
        for d in range(ndim):
            c = int(pt[d])
            l = c - half[d]
            h = c + half[d] + 1
            kl, kh = 0, kernel.shape[d]
            if l < 0:
                kl = -l
                l = 0
            if h > out.shape[d]:
                kh -= (h - out.shape[d])
                h = out.shape[d]
            if l >= h:
                ok = False
                break
            lo.append(l); hi.append(h); klo.append(kl); khi.append(kh)
        if not ok:
            continue
        sl = tuple(slice(l, h) for l, h in zip(lo, hi))
        ksl = tuple(slice(l, h) for l, h in zip(klo, khi))
        np.maximum(out[sl], kernel[ksl], out=out[sl])
    return out


def guidance_map_from_coords(shape: Sequence[int],
                             coords: Iterable[Sequence[int]],
                             radius: float = 10.0,
                             spacing: Optional[Tuple[float, ...]] = None,
                             dtype=np.float32) -> np.ndarray:
    """Allocate a fresh guidance map of `shape` and stamp `coords` into it."""
    out = np.zeros(tuple(int(s) for s in shape), dtype=dtype)
    return stamp_clipped_edt(out, coords, radius, spacing)
