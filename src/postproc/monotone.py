"""Blend the lesion probability against the previous iteration to damp oscillation.

``max(p_new, p_prev)`` inside the foreground constraint region, ``min`` inside the
background one, an optional EMA elsewhere.  Where the two regions overlap nothing is
clamped, so a later scribble revokes an earlier one instead of freezing it forever.
Modes: ``none`` (default; anything else makes the output depend on the state cache),
``minmax``, ``ema``, ``ema_minmax`` (EMA first, so the clamps win).
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from .config import MonotoneConfig
from .utils import foreground_prob

__all__ = ["blend_with_previous", "blend_masks", "revoke_overlap"]


def blend_with_previous(
    prob_new: np.ndarray,
    prob_prev: Optional[np.ndarray],
    fg_mask_constraints: Optional[np.ndarray] = None,
    bg_mask_constraints: Optional[np.ndarray] = None,
    mode: str = "ema_minmax",
    *,
    ema_alpha: float = 0.6,
    cfg: Optional[MonotoneConfig] = None,
) -> np.ndarray:
    """Blend the new lesion probability with the cached previous one.

    ``prob_new`` / ``prob_prev`` may be ``(X, Y, Z)`` or ``(C, X, Y, Z)``; the result is
    always ``(X, Y, Z)`` float32.  ``prob_prev=None`` (iteration 0) returns ``prob_new``
    unchanged, so the caller needs no special case.
    """
    if cfg is not None:
        mode = cfg.mode
        ema_alpha = cfg.ema_alpha

    new = foreground_prob(prob_new)
    if new is None:
        raise ValueError("prob_new must not be None")
    if prob_prev is None or mode == "none":
        return new

    prev = foreground_prob(prob_prev)
    if prev is None or prev.shape != new.shape:
        return new

    if mode in ("ema", "ema_minmax"):
        a = float(np.clip(ema_alpha, 0.0, 1.0))
        out = a * new + (1.0 - a) * prev
    elif mode == "minmax":
        out = new.copy()
    else:
        raise ValueError(f"unknown blend mode {mode!r}")

    if mode in ("minmax", "ema_minmax"):
        fg_mask_constraints, bg_mask_constraints = revoke_overlap(
            fg_mask_constraints, bg_mask_constraints
        )
        if fg_mask_constraints is not None:
            fg = np.asarray(fg_mask_constraints).astype(bool)
            if fg.shape == out.shape and fg.any():
                np.maximum(out, np.maximum(new, prev), out=out, where=fg)
        if bg_mask_constraints is not None:
            bg = np.asarray(bg_mask_constraints).astype(bool)
            if bg.shape == out.shape and bg.any():
                np.minimum(out, np.minimum(new, prev), out=out, where=bg)

    return out.astype(np.float32, copy=False)


def revoke_overlap(fg_region, bg_region):
    """Drop the intersection from both constraint regions (see the module docstring)."""
    if fg_region is None or bg_region is None:
        return fg_region, bg_region
    fg = np.asarray(fg_region).astype(bool)
    bg = np.asarray(bg_region).astype(bool)
    if fg.shape != bg.shape:
        return fg, bg
    overlap = fg & bg
    if not overlap.any():
        return fg, bg
    return fg & ~overlap, bg & ~overlap


def blend_masks(
    mask_new: np.ndarray,
    mask_prev: Optional[np.ndarray],
    fg_mask_constraints: Optional[np.ndarray] = None,
    bg_mask_constraints: Optional[np.ndarray] = None,
    mode: str = "minmax",
) -> np.ndarray:
    """Binary fallback used when the base predictor exposes no probabilities.

    ``"minmax"``: inside the foreground constraint region take the union with the
    previous mask, inside the background constraint region take the intersection.
    Everything else is left to the new mask.  ``"none"`` passes through.
    """
    new = np.asarray(mask_new) > 0
    if mask_prev is None or mode == "none":
        return new.astype(np.uint8)
    prev = np.asarray(mask_prev) > 0
    if prev.shape != new.shape:
        return new.astype(np.uint8)

    fg_mask_constraints, bg_mask_constraints = revoke_overlap(
        fg_mask_constraints, bg_mask_constraints
    )
    out = new.copy()
    if fg_mask_constraints is not None:
        fg = np.asarray(fg_mask_constraints).astype(bool)
        if fg.shape == out.shape:
            out |= prev & fg
    if bg_mask_constraints is not None:
        bg = np.asarray(bg_mask_constraints).astype(bool)
        if bg.shape == out.shape:
            # inside the background-constraint region keep only what survived before
            out &= ~bg | prev
    return out.astype(np.uint8)
