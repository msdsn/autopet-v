"""Synthetic PET/CT phantoms for the post-processing tests.

Everything is in the challenge's coordinate convention: arrays are indexed [i, j, k]
with k the axial slice, and spacing is (sx, sy, sz) in mm in the same order. The default
2.04 x 2.04 x 3.0 mm is the baseline's nnU-Net spacing seen from nibabel (one voxel =
12.45 mm^3 = 0.01245 mL).
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np
import pytest

# make `import postproc` work when the tests are run from anywhere
_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.dirname(os.path.dirname(_HERE))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

SPACING = (2.04, 2.04, 3.0)
VOX_ML = SPACING[0] * SPACING[1] * SPACING[2] / 1000.0  # 0.012485 mL


@dataclass
class Blob:
    centre: Tuple[int, int, int]
    radius_mm: float
    suv: float
    name: str = "blob"


def sphere_mask(shape, centre, radius_mm, spacing=SPACING) -> np.ndarray:
    grids = np.ogrid[tuple(slice(0, s) for s in shape)]
    d2 = sum(((g - c) * s) ** 2 for g, c, s in zip(grids, centre, spacing))
    return d2 <= radius_mm ** 2


def make_phantom(
    shape: Sequence[int] = (80, 80, 64),
    blobs: Optional[List[Blob]] = None,
    spacing: Sequence[float] = SPACING,
    background_suv: float = 0.6,
    seed: int = 0,
):
    """Return (ct, pet, gt_mask, blob_masks).

    The body is soft tissue (0 HU) inside air (-1000 HU), so the CT gate in
    compliance.apply_tumor_scribbles has something to bite on. PET is a smooth low
    background plus the blobs; seed >= 0 adds mild deterministic noise.
    """
    shape = tuple(int(s) for s in shape)
    blobs = blobs if blobs is not None else default_blobs(shape)

    ct = np.full(shape, -1000.0, dtype=np.float32)
    body = sphere_body(shape, spacing)
    ct[body] = 0.0

    pet = np.zeros(shape, dtype=np.float32)
    pet[body] = background_suv

    gt = np.zeros(shape, dtype=np.uint8)
    blob_masks = {}
    for b in blobs:
        m = sphere_mask(shape, b.centre, b.radius_mm, spacing)
        pet[m] = b.suv
        gt[m] = 1
        blob_masks[b.name] = m

    if seed is not None and seed >= 0:
        rng = np.random.default_rng(seed)
        pet += rng.normal(0.0, 0.02, size=shape).astype(np.float32)
        pet = np.clip(pet, 0.0, None)
    return ct, pet, gt, blob_masks


def sphere_body(shape, spacing=SPACING) -> np.ndarray:
    """An elliptical "patient" occupying most of the volume."""
    grids = np.ogrid[tuple(slice(0, s) for s in shape)]
    centre = [(s - 1) / 2.0 for s in shape]
    rad = [(s * sp) / 2.2 for s, sp in zip(shape, spacing)]
    val = sum(((g - c) * sp / r) ** 2 for g, c, sp, r in zip(grids, centre, spacing, rad))
    return val <= 1.0


def default_blobs(shape=(80, 80, 64)) -> List[Blob]:
    """Three lesions of very different SUV."""
    return [
        Blob(centre=(24, 24, 20), radius_mm=9.0, suv=12.0, name="hot"),
        Blob(centre=(56, 30, 34), radius_mm=6.0, suv=4.5, name="warm"),
        Blob(centre=(30, 58, 46), radius_mm=4.0, suv=2.2, name="cool"),
    ]


def scribble_line(centre, axis=0, length=7, mask=None):
    """A short straight line of voxel indices, like the official 2-D scribble.

    With `mask` the line is clipped to voxels inside it, as the official simulator only
    emits voxels lying inside the error component.
    """
    pts = []
    half = length // 2
    for t in range(-half, half + 1):
        p = list(centre)
        p[axis] += t
        if mask is not None and not mask[tuple(p)]:
            continue
        pts.append([int(v) for v in p])
    return pts


@pytest.fixture
def phantom():
    ct, pet, gt, blobs = make_phantom()
    return {"ct": ct, "pet": pet, "gt": gt, "blobs": blobs, "spacing": SPACING}


@pytest.fixture
def cache_dir(tmp_path):
    d = tmp_path / "cache"
    d.mkdir()
    return str(d)
