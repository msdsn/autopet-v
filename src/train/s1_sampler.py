"""Component-balanced foreground sampling (S1) for the interactive fine-tune.

nnU-Net's ``nnUNetDataLoader.get_bbox(force_fg=True)`` draws a foreground *voxel*
uniformly from ``properties['class_locations']``, i.e. a lesion is chosen with
probability proportional to its **volume**. Sub-1-mL lesions are ~45 % of the FDG
lesions but 0.7 % of the FDG lesion volume, so with
``oversample_foreground_percent = 0.33`` roughly 0.2 % of training patches are
centred on one. This module replaces that draw with

    component c with probability  p(c) proportional to  |c| ** gamma
    then a voxel uniformly inside c

so ``gamma = 1`` reproduces the stock behaviour and ``gamma = 0`` (the default) makes
every connected component equally likely regardless of its size. Nothing else about
the sampler changes: the foreground/background patch ratio is still
``oversample_foreground_percent`` and background patches are drawn exactly as before.

Zero parameters, zero network change -- epoch 0 is the source model by construction.

The per-case component table is built lazily on first use with ``cc3d`` (18-connected,
matching the challenge metric) on the stored segmentation and cached in memory and,
if a cache directory is available, on disk so the other dataloader workers and later
runs do not repeat the labelling.
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from nnunetv2.training.dataloading.data_loader import nnUNetDataLoader
from nnunetv2.training.dataloading.nnunet_dataset import (nnUNetDatasetBlosc2,
                                                          nnUNetDatasetNumpy)

__all__ = [
    "ComponentTable",
    "build_component_table",
    "component_weights",
    "nnUNetDataLoaderS1",
    "S1RecordingDatasetBlosc2",
    "S1RecordingDatasetNumpy",
    "recording_dataset_class",
]

CACHE_SUBDIR = "s1_components"


class ComponentTable:
    """Connected components of one case's label, with sampled interior voxels.

    ``sizes[c]`` is the true voxel count of component ``c``; ``coords[offsets[c]:
    offsets[c + 1]]`` are up to ``max_samples`` voxels drawn uniformly from it. The
    sub-sample keeps the table small (a 100 mL lesion is 8000 voxels) while a uniform
    draw from it is still a uniform draw from the component up to that sub-sampling.
    """

    __slots__ = ("sizes", "offsets", "coords")

    def __init__(self, sizes: np.ndarray, offsets: np.ndarray, coords: np.ndarray):
        self.sizes = np.asarray(sizes, dtype=np.int64)
        self.offsets = np.asarray(offsets, dtype=np.int64)
        self.coords = np.asarray(coords, dtype=np.int32)

    def __len__(self) -> int:
        return int(self.sizes.shape[0])

    def voxels(self, c: int) -> np.ndarray:
        return self.coords[self.offsets[c]:self.offsets[c + 1]]

    def save(self, path: str) -> None:
        tmp = f"{path}.{os.getpid()}.tmp"
        np.savez_compressed(tmp, sizes=self.sizes, offsets=self.offsets, coords=self.coords)
        os.replace(tmp + ".npz" if not tmp.endswith(".npz") else tmp, path)

    @classmethod
    def load(cls, path: str) -> "ComponentTable":
        with np.load(path) as z:
            return cls(z["sizes"], z["offsets"], z["coords"])


def build_component_table(seg, connectivity: int = 18, max_samples: int = 256,
                          seed: int = 0) -> ComponentTable:
    """Label ``seg`` and return its component table. ``seg`` may be memory-mapped."""
    import cc3d

    arr = np.asarray(seg[:] if hasattr(seg, "__getitem__") and not isinstance(seg, np.ndarray)
                     else seg)
    while arr.ndim > 3:
        arr = arr[0]
    fg = np.ascontiguousarray(arr > 0).astype(np.uint8)
    del arr
    lab, n = cc3d.connected_components(fg, connectivity=connectivity, return_N=True)
    del fg
    if n == 0:
        return ComponentTable(np.zeros(0, np.int64), np.zeros(1, np.int64),
                              np.zeros((0, 3), np.int32))

    flat = lab.reshape(-1)
    idx = np.flatnonzero(flat)
    labs = flat[idx]
    order = np.argsort(labs, kind="stable")
    idx = idx[order]
    labs = labs[order]
    sizes = np.bincount(labs, minlength=n + 1)[1:].astype(np.int64)
    starts = np.concatenate(([0], np.cumsum(sizes)))

    rng = np.random.default_rng(seed)
    picks: List[np.ndarray] = []
    offsets = [0]
    for c in range(n):
        s, e = int(starts[c]), int(starts[c + 1])
        if e - s > max_samples:
            sel = rng.choice(e - s, size=max_samples, replace=False) + s
        else:
            sel = np.arange(s, e)
        picks.append(idx[sel])
        offsets.append(offsets[-1] + int(sel.shape[0]))
    lin = np.concatenate(picks) if picks else np.zeros(0, np.int64)
    coords = np.stack(np.unravel_index(lin, lab.shape), axis=1).astype(np.int32)
    return ComponentTable(sizes, np.asarray(offsets, np.int64), coords)


def component_weights(sizes: np.ndarray, gamma: float) -> np.ndarray:
    """Sampling probabilities proportional to ``size ** gamma`` (gamma 0 = uniform)."""
    if sizes.size == 0:
        return sizes.astype(np.float64)
    if gamma == 0.0:
        w = np.ones(sizes.shape[0], dtype=np.float64)
    else:
        w = np.power(sizes.astype(np.float64), float(gamma))
    total = w.sum()
    if not np.isfinite(total) or total <= 0:
        w = np.ones(sizes.shape[0], dtype=np.float64)
        total = float(w.size)
    return w / total


# ---------------------------------------------------------------------------
# datasets that remember which case they just handed out
# ---------------------------------------------------------------------------
# ``nnUNetDataLoader.generate_train_batch`` calls ``self._data.load_case(i)`` and then
# ``self.get_bbox(...)`` without passing the identifier or the label, so the sampler
# cannot see which case it is placing a patch in. These subclasses record both on the
# dataset instance the loader already holds. They are module-level (not built with
# ``type()``) so a dataloader that has to be pickled to a worker still works.

class _S1RecordingMixin:
    s1_last_identifier: Optional[str] = None
    s1_last_seg = None

    def load_case(self, identifier):
        out = super().load_case(identifier)  # type: ignore[misc]
        self.s1_last_identifier = identifier
        self.s1_last_seg = out[1]
        return out


class S1RecordingDatasetBlosc2(_S1RecordingMixin, nnUNetDatasetBlosc2):
    """blosc2 store variant."""


class S1RecordingDatasetNumpy(_S1RecordingMixin, nnUNetDatasetNumpy):
    """npy/npz store variant."""


def recording_dataset_class(base):
    """Map an nnU-Net dataset class to the recording subclass S1 needs."""
    if issubclass(base, _S1RecordingMixin):
        return base
    if issubclass(base, nnUNetDatasetBlosc2):
        return S1RecordingDatasetBlosc2
    if issubclass(base, nnUNetDatasetNumpy):
        return S1RecordingDatasetNumpy
    raise RuntimeError(f"no S1 recording subclass for dataset class {base}")


# ---------------------------------------------------------------------------
# the dataloader
# ---------------------------------------------------------------------------

class nnUNetDataLoaderS1(nnUNetDataLoader):
    """``nnUNetDataLoader`` whose foreground patches are component-balanced.

    Falls back to the stock draw whenever S1 cannot apply: a background patch, a case
    with no foreground, or a dataset that does not record what it loaded.
    """

    def __init__(self, *args, s1_gamma: float = 0.0, s1_connectivity: int = 18,
                 s1_max_samples: int = 256, s1_cache_dir: Optional[str] = None,
                 s1_verbose: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        self.s1_gamma = float(s1_gamma)
        self.s1_connectivity = int(s1_connectivity)
        self.s1_max_samples = int(s1_max_samples)
        self.s1_cache_dir = s1_cache_dir
        self.s1_verbose = bool(s1_verbose)
        self._s1_tables: Dict[str, ComponentTable] = {}
        self._s1_weights: Dict[str, np.ndarray] = {}
        self.s1_n_fg = 0          # forced-foreground patches seen
        self.s1_n_applied = 0     # of those, placed by the component sampler
        if self.s1_cache_dir:
            os.makedirs(self.s1_cache_dir, exist_ok=True)

    # ------------------------------------------------------------------
    def _table(self, identifier: str, seg) -> Optional[ComponentTable]:
        t = self._s1_tables.get(identifier)
        if t is not None:
            return t
        path = os.path.join(self.s1_cache_dir, identifier + ".npz") if self.s1_cache_dir else None
        if path and os.path.isfile(path):
            try:
                t = ComponentTable.load(path)
            except Exception:
                t = None
        if t is None:
            if seg is None:
                return None
            t = build_component_table(seg, connectivity=self.s1_connectivity,
                                      max_samples=self.s1_max_samples)
            if path:
                try:
                    tmp = f"{path}.{os.getpid()}.tmp.npz"
                    np.savez_compressed(tmp, sizes=t.sizes, offsets=t.offsets, coords=t.coords)
                    os.replace(tmp, path)
                except Exception:
                    pass
        self._s1_tables[identifier] = t
        self._s1_weights[identifier] = component_weights(t.sizes, self.s1_gamma)
        return t

    def s1_pick_voxel(self, identifier: str, seg) -> Optional[np.ndarray]:
        """Component-balanced foreground voxel of one case, or None."""
        t = self._table(identifier, seg)
        if t is None or len(t) == 0:
            return None
        w = self._s1_weights[identifier]
        c = int(np.random.choice(w.shape[0], p=w))
        pts = t.voxels(c)
        if pts.shape[0] == 0:
            return None
        return pts[np.random.randint(pts.shape[0])]

    # ------------------------------------------------------------------
    def get_bbox(self, data_shape: np.ndarray, force_fg: bool,
                 class_locations: Union[dict, None],
                 overwrite_class: Union[int, Tuple[int, ...]] = None,
                 verbose: bool = False):
        if not force_fg or overwrite_class is not None:
            return super().get_bbox(data_shape, force_fg, class_locations,
                                    overwrite_class, verbose)
        self.s1_n_fg += 1
        identifier = getattr(self._data, "s1_last_identifier", None)
        seg = getattr(self._data, "s1_last_seg", None)
        if identifier is None:
            return super().get_bbox(data_shape, force_fg, class_locations,
                                    overwrite_class, verbose)
        voxel = self.s1_pick_voxel(identifier, seg)
        if voxel is None:                       # lesion-free case: stock fallback
            return super().get_bbox(data_shape, force_fg, class_locations,
                                    overwrite_class, verbose)
        self.s1_n_applied += 1

        # identical bounds arithmetic to nnUNetDataLoader.get_bbox
        need_to_pad = self.need_to_pad.copy()
        dim = len(data_shape)
        for d in range(dim):
            if need_to_pad[d] + data_shape[d] < self.patch_size[d]:
                need_to_pad[d] = self.patch_size[d] - data_shape[d]
        lbs = [-need_to_pad[i] // 2 for i in range(dim)]
        bbox_lbs = [max(lbs[i], int(voxel[i]) - self.patch_size[i] // 2) for i in range(dim)]
        bbox_ubs = [bbox_lbs[i] + self.patch_size[i] for i in range(dim)]
        return bbox_lbs, bbox_ubs


def default_cache_dir(source_folder: str) -> str:
    """``<preprocessed>/<Dataset>/s1_components`` for a store folder.

    Deliberately a sibling of ``nnUNetPlans_3d_fullres``: the launcher verifies that
    folder byte-for-byte against Drive, so nothing may be written inside it.
    """
    env = os.environ.get("S1_CACHE_DIR")
    if env:
        return env
    return os.path.join(os.path.dirname(os.path.abspath(source_folder)), CACHE_SUBDIR)
