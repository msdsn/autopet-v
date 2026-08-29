"""Patch-level wrapper around the official autoPET V scribble simulator.

simulate_scribble_from_label iterates over the LAST array axis, while the nnU-Net
preprocessed array is (axial, y, x); calling it directly would draw the scribble on
a sagittal plane, so everything here moves the slice axis to the end and maps the
coordinates back. It also absorbs the simulator's 2-tuple return on an empty mask.
"""

from __future__ import annotations

import os
import sys
from typing import List, Optional, Sequence, Tuple

import numpy as np

__all__ = [
    "STRATEGIES",
    "get_official_simulator",
    "simulate_scribble",
    "perturb_scribble",
    "InteractionResult",
    "simulate_interaction_sequence",
]

STRATEGIES = ("centerline", "random", "boundary")

_SIM = None


def _candidate_repo_dirs() -> List[str]:
    here = os.path.dirname(os.path.abspath(__file__))
    cands = []
    env = os.environ.get("AUTOPETV_REPO")
    if env:
        cands.append(os.path.join(env, "interactive"))
        cands.append(env)
    # <repo>/src/train/  ->  <repo>/autoPETV/interactive
    cands.append(os.path.abspath(os.path.join(here, os.pardir, os.pardir, "autoPETV", "interactive")))
    cands.append(os.path.abspath(os.path.join(here, os.pardir, os.pardir, os.pardir, "autoPETV", "interactive")))
    cands += [
        "/content/autoPETV/interactive",
        "/content/work/autoPETV/interactive",
        "/opt/autoPETV/interactive",
    ]
    return cands


def get_official_simulator():
    """Import and return simulate_scribble_from_label from the official repo."""
    global _SIM
    if _SIM is not None:
        return _SIM
    try:
        from simulate_scribbles import simulate_scribble_from_label  # type: ignore
        _SIM = simulate_scribble_from_label
        return _SIM
    except Exception:
        pass
    for d in _candidate_repo_dirs():
        if os.path.isfile(os.path.join(d, "simulate_scribbles.py")):
            if d not in sys.path:
                sys.path.insert(0, d)
            from simulate_scribbles import simulate_scribble_from_label  # type: ignore
            _SIM = simulate_scribble_from_label
            return _SIM
    raise ImportError(
        "Could not locate the official autoPETV/interactive/simulate_scribbles.py. "
        "Set AUTOPETV_REPO=/path/to/autoPETV or put the repo next to this one. "
        f"Searched: {_candidate_repo_dirs()}"
    )


# ---------------------------------------------------------------------------
# axis handling
# ---------------------------------------------------------------------------

def _to_slice_last(arr: np.ndarray, slice_axis: int) -> np.ndarray:
    if slice_axis == arr.ndim - 1:
        return arr
    return np.ascontiguousarray(np.moveaxis(arr, slice_axis, -1))


def _coords_back(coords: Sequence[Sequence[int]], slice_axis: int, ndim: int = 3) -> List[List[int]]:
    """Map coordinates from the slice-last frame back to the original frame."""
    if slice_axis == ndim - 1:
        return [list(map(int, c)) for c in coords]
    out = []
    for c in coords:
        c = list(map(int, c))
        z = c.pop(-1)          # slice index (was moved to the end)
        c.insert(slice_axis, z)
        out.append(c)
    return out


# ---------------------------------------------------------------------------
# the official simulator, made safe
# ---------------------------------------------------------------------------

def simulate_scribble(error_mask: np.ndarray,
                      strategy: str = "centerline",
                      seed: int = 42,
                      slice_axis: int = 0) -> Tuple[List[List[int]], int]:
    """Run the official simulator on `error_mask`.

    Returns (coords in the original frame, n_scribble_voxels), or ([], 0) when the
    mask is empty or the simulator failed. Never raises.
    """
    if error_mask is None or not error_mask.any():
        return [], 0
    sim = get_official_simulator()
    arr = _to_slice_last(np.asarray(error_mask, dtype=np.uint8), slice_axis)
    try:
        res = sim(arr, strategy=strategy, seed=int(seed))
    except Exception:
        return [], 0
    if res is None:
        return [], 0
    # the 2-tuple quirk: ([], False) on empty input
    if len(res) < 3:
        return [], 0
    coords, _label_cls, size = res
    if coords is None or len(coords) == 0:
        return [], 0
    return _coords_back(coords, slice_axis), int(size)


# ---------------------------------------------------------------------------
# ScribblePrompt-style realism perturbation
# ---------------------------------------------------------------------------

def perturb_scribble(coords: Sequence[Sequence[int]],
                     rng: np.random.Generator,
                     shape: Optional[Sequence[int]] = None,
                     p_break: float = 0.6,
                     max_break_frac: float = 0.35,
                     p_warp: float = 0.7,
                     warp_amplitude: float = 1.5,
                     slice_axis: int = 0) -> List[List[int]]:
    """Make a simulated stroke look more like a human one.

    Deletes one or two contiguous runs (pen lifted) and adds a smooth low-frequency
    offset (tremor), the latter in-plane only so the stroke stays on its axial slice.
    Follows ScribblePrompt (arXiv:2312.07381).
    """
    pts = [list(map(int, c)) for c in coords]
    n = len(pts)
    if n == 0:
        return pts

    # 1) breaking
    if n >= 6 and rng.random() < p_break:
        n_breaks = int(rng.integers(1, 3))
        keep = np.ones(n, dtype=bool)
        for _ in range(n_breaks):
            ln = max(1, int(rng.integers(1, max(2, int(max_break_frac * n)) + 1)))
            st = int(rng.integers(0, max(1, n - ln)))
            keep[st:st + ln] = False
        if keep.sum() >= 2:
            pts = [p for p, k in zip(pts, keep) if k]
            n = len(pts)

    # 2) smooth warp, in-plane only
    if rng.random() < p_warp and n >= 2:
        arr = np.asarray(pts, dtype=np.float32)
        t = np.linspace(0.0, 1.0, n, dtype=np.float32)
        for ax in range(arr.shape[1]):
            if ax == slice_axis:
                continue  # keep the stroke on its slice
            amp = rng.normal(0.0, warp_amplitude)
            phase = rng.uniform(0.0, 2 * np.pi)
            freq = rng.uniform(0.5, 2.0)
            arr[:, ax] += amp * np.sin(2 * np.pi * freq * t + phase)
        arr = np.rint(arr).astype(np.int64)
        if shape is not None:
            for ax in range(arr.shape[1]):
                arr[:, ax] = np.clip(arr[:, ax], 0, int(shape[ax]) - 1)
        pts = [list(map(int, p)) for p in arr]

    return pts


# ---------------------------------------------------------------------------
# the error-driven protocol, at patch level
# ---------------------------------------------------------------------------

class InteractionResult:
    __slots__ = ("fg_coords", "bg_coords", "n_iters", "strategies")

    def __init__(self):
        self.fg_coords: List[List[int]] = []
        self.bg_coords: List[List[int]] = []
        self.n_iters: int = 0
        self.strategies: List[str] = []


def _erase_component(mask: np.ndarray, coords: Sequence[Sequence[int]], connectivity: int = 26) -> None:
    """Zero the 3-D connected components of `mask` that the scribble touches.

    Stands in for "the model fixed the error just pointed at", so the next simulated
    scribble lands on the next error, as it would across evaluation iterations.
    """
    if not len(coords):
        return
    try:
        import cc3d
        cc = cc3d.connected_components(mask.astype(np.uint8), connectivity=connectivity)
    except Exception:
        from scipy.ndimage import label as ndlabel
        cc, _ = ndlabel(mask)
    hit = set()
    for c in coords:
        try:
            v = int(cc[tuple(int(x) for x in c)])
        except IndexError:
            continue
        if v != 0:
            hit.add(v)
    if not hit:
        # scribble did not land on a labelled voxel (can happen after clipping);
        # fall back to erasing the largest component so we cannot loop forever
        vals, cnts = np.unique(cc[cc > 0], return_counts=True)
        if len(vals):
            hit.add(int(vals[int(np.argmax(cnts))]))
    if hit:
        mask[np.isin(cc, list(hit))] = 0


def simulate_interaction_sequence(label: np.ndarray,
                                  prev_pred: np.ndarray,
                                  k: int,
                                  rng: np.random.Generator,
                                  slice_axis: int = 0,
                                  strategies: Sequence[str] = STRATEGIES,
                                  p_perturb: float = 0.2,
                                  connectivity: int = 26) -> InteractionResult:
    """Reproduce the official iteration loop on a single training patch.

    Each of the k iterations simulates a scribble on the over- and under-segmentation
    masks, picks one by the official tie rule (tumor unless the background stroke is
    strictly longer), then erases the component it landed on. Never raises.
    """
    res = InteractionResult()
    if k <= 0:
        return res

    L = np.asarray(label, dtype=bool)
    P = np.asarray(prev_pred, dtype=bool)
    overseg = (P & ~L).astype(np.uint8)
    underseg = (~P & L).astype(np.uint8)

    cache_bg: Optional[Tuple[List[List[int]], int]] = None
    cache_fg: Optional[Tuple[List[List[int]], int]] = None
    strat_bg = strat_fg = None

    for _ in range(int(k)):
        if not overseg.any() and not underseg.any():
            break
        strategy = str(rng.choice(list(strategies)))
        seed = int(rng.integers(0, 2 ** 31 - 1))

        if cache_bg is None or strat_bg != strategy:
            cache_bg = simulate_scribble(overseg, strategy, seed, slice_axis)
            strat_bg = strategy
        if cache_fg is None or strat_fg != strategy:
            cache_fg = simulate_scribble(underseg, strategy, seed, slice_axis)
            strat_fg = strategy

        bg_coords, fp = cache_bg
        fg_coords, fn = cache_fg

        # official rule (interactive_loop.py): `if fp <= fn: tumor else background`
        if fp <= fn:
            chosen, is_fg, target = fg_coords, True, underseg
        else:
            chosen, is_fg, target = bg_coords, False, overseg

        if len(chosen) == 0:
            break

        kept = chosen
        if rng.random() < p_perturb:
            kept = perturb_scribble(chosen, rng, shape=L.shape, slice_axis=slice_axis)
            if len(kept) == 0:
                kept = chosen

        if is_fg:
            res.fg_coords.extend(kept)
        else:
            res.bg_coords.extend(kept)
        res.strategies.append(strategy)
        res.n_iters += 1

        # erase using the unperturbed coordinates (they are guaranteed on-component)
        _erase_component(target, chosen, connectivity)
        if is_fg:
            cache_fg = None
        else:
            cache_bg = None

    return res
