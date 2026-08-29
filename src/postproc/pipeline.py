"""``PostProcPredictor``: the interaction layer wrapped around any base predictor.

One ``predict`` call is one challenge iteration -- rebuild the constraint set from the
accumulated scribble json, run the base predictor, then the negative gate, cleanup,
background compliance (split, or delete when there is no core), foreground compliance
(grow only what is missing), the optional threshold rescue and monotone blend, a
re-assert of the constraints, and the cache write.  With the default config the result
is a pure function of (CT, PET, accumulated scribbles): the state directory is only a
speed cache and the iteration index is derived from the scribbles, not from a counter.
The base predictor is duck-typed and its signature introspected once, so a predictor
taking only ``(ct, pet, spacing, scribbles)`` works.
"""

from __future__ import annotations

import inspect
import time
from typing import Any, Dict, Optional, Sequence, Tuple, Union

import numpy as np

from .cleanup import bridge_components, cleanup_mask
from .compliance import (
    apply_background_scribbles,
    apply_tumor_scribbles,
    assert_constraints,
    check_constraints,
)
from .config import PostProcConfig
from .constraints import CaseCache, ConstraintState
from .monotone import blend_masks, blend_with_previous
from .negative_gate import is_probably_negative
from .tracer_classifier import guess_tracer
from .utils import (
    as_points_array,
    foreground_prob,
    points_in_bounds,
    points_mask,
    timer,
    unique_points,
    voxel_volume_ml,
)

__all__ = ["PostProcPredictor"]


class PostProcPredictor:
    """Wrap a base predictor with hard scribble compliance and FP suppression.

    ``base`` is anything exposing ``predict(...)`` with the challenge signature;
    ``config`` is a dict or a ``PostProcConfig``.
    """

    def __init__(self, base: Any, config: Union[Dict[str, Any], PostProcConfig, None] = None):
        self.base = base
        self.cfg = PostProcConfig.from_dict(config)
        self.name = f"postproc({getattr(base, 'name', type(base).__name__)})"
        self.last_info: Dict[str, Any] = {}
        self._base_supports_prob: Optional[bool] = None
        self._sig_cache: Optional[Tuple[set, bool]] = None

    # -- optional Predictor hooks ------------------------------------------
    def warmup(self) -> None:
        if hasattr(self.base, "warmup"):
            self.base.warmup()

    def close(self) -> None:
        if hasattr(self.base, "close"):
            self.base.close()

    # ----------------------------------------------------------------------
    def predict(
        self,
        ct: np.ndarray,
        pet: np.ndarray,
        spacing: Sequence[float],
        scribbles: Optional[Dict[str, Any]] = None,
        prev_pred: Optional[np.ndarray] = None,
        case_cache_dir: Optional[str] = None,
        *,
        affine: Optional[np.ndarray] = None,
        ct_path: Optional[str] = None,
        pet_path: Optional[str] = None,
        case_name: str = "case",
        gt: Optional[np.ndarray] = None,
        return_probabilities: bool = False,
        **kwargs: Any,
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        t_start = time.perf_counter()
        info: Dict[str, Any] = {"case": case_name}
        cfg = self.cfg
        pet = np.asarray(pet)
        shape = pet.shape
        spacing = tuple(float(s) for s in spacing)
        vox_ml = voxel_volume_ml(spacing)
        scribbles = scribbles or {"tumor": [], "background": []}

        # ---- 1. constraint set, rebuilt from the input every time ---------
        cache = CaseCache(case_cache_dir)
        had_state = cache.has_state()
        cached = cache.load_state()
        state = ConstraintState.from_scribbles(scribbles, shape, spacing)
        if had_state and cached.shape is not None and tuple(cached.shape) == tuple(shape):
            # The cache may hold points from an input json we no longer see.  Merging is
            # monotone and cannot lose a constraint; with a well-behaved evaluator the
            # merge is a no-op, which is what keeps the output state-independent.
            state.add_scribbles(
                {"tumor": cached.tumor_points, "background": cached.background_points}
            )
        iteration = state.infer_iteration(spacing)
        fg_points = state.tumor_array(shape)
        bg_points = state.background_array(shape)
        info.update(
            iteration=iteration,
            iteration_source="scribbles",
            iteration_cached=int(cached.iteration) if had_state else None,
            state_dir=case_cache_dir,
            state_available=bool(had_state),
            n_tumor=int(len(fg_points)),
            n_background=int(len(bg_points)),
        )

        # ---- 1b. nothing to correct? then do not disturb anything ---------
        if cfg.skip_inference_if_satisfied and had_state:
            prev_mask = cache.load_prev_mask()
            if prev_mask is not None and prev_mask.shape == shape:
                new_fg, new_bg = _new_points(state, cached, shape)
                if _already_satisfied(prev_mask, new_fg, new_bg):
                    return self._return_unchanged(
                        prev_mask, state, cached, cache, info, spacing, vox_ml,
                        len(new_fg), len(new_bg), t_start, return_probabilities,
                    )

        # ---- 2. tracer (deterministic function of the PET) ----------------
        with timer(info, "tracer"):
            tracer = self._resolve_tracer(pet, spacing, affine, info, ct=ct)

        # ---- 3. base prediction -------------------------------------------
        with timer(info, "base_predict"):
            mask, prob = self._call_base(
                ct=ct,
                pet=pet,
                spacing=spacing,
                scribbles={"tumor": fg_points.tolist(), "background": bg_points.tolist()},
                prev_pred=_as_prev(prev_pred, cache, cfg.pass_cached_prev_pred),
                case_cache_dir=case_cache_dir,
                affine=affine,
                ct_path=ct_path,
                pet_path=pet_path,
                case_name=case_name,
                **kwargs,
            )
        mask = np.asarray(mask).astype(np.uint8)
        if mask.shape != shape:
            raise ValueError(f"base predictor returned shape {mask.shape}, expected {shape}")
        prob_fg = foreground_prob(prob, shape) if prob is not None else None
        info["base_volume_ml"] = float(mask.sum() * vox_ml)

        # ---- 4. negative gate ----------------------------------------------
        # Fires only while no scribble has ever arrived, which is true at every
        # iteration of a lesion-absent case and false as soon as we are corrected.
        gate_fired = False
        with timer(info, "negative_gate"):
            if cfg.negative_gate.enabled:
                gate_fired, gate_feats = is_probably_negative(
                    mask,
                    pet,
                    prob_fg,
                    cfg.negative_gate,
                    spacing=spacing,
                    iteration=iteration,
                    n_tumor_scribbles=len(fg_points),
                    n_background_scribbles=len(bg_points),
                    tracer=tracer,
                    ct=ct,
                    affine=affine,
                    return_features=True,
                )
                info["negative_gate"] = gate_feats
        info["negative_gate_fired"] = bool(gate_fired)
        removed = None
        base_mask = mask.copy()
        after_cleanup = mask
        if gate_fired:
            mask = np.zeros(shape, dtype=np.uint8)
            if prob_fg is not None:
                prob_fg = np.zeros(shape, dtype=np.float32)
            state.negative_gate_fired = True
        else:
            # ---- 5. cleanup -------------------------------------------------
            with timer(info, "cleanup"):
                if cfg.enable_cleanup:
                    mask, clean_info = cleanup_mask(
                        mask,
                        pet,
                        spacing,
                        tracer=tracer,
                        cfg=cfg.cleanup,
                        protect_points=fg_points,
                        background_points=bg_points,
                        prob=prob_fg,
                        iteration=iteration,
                        return_info=True,
                    )
                    info["cleanup"] = clean_info
                else:
                    info["cleanup"] = {"skipped": True}

            after_cleanup = mask.copy()

            # ---- 6. background compliance (G1) ------------------------------
            with timer(info, "bg_compliance"):
                if cfg.enable_bg_compliance:
                    mask, bg_info = apply_background_scribbles(
                        mask,
                        pet,
                        bg_points,
                        connectivity=cfg.compliance.connectivity,
                        prob=prob_fg,
                        spacing=spacing,
                        fg_points=fg_points,
                        tracer=tracer,
                        cfg=cfg.compliance,
                        gt=gt,
                        return_info=True,
                    )
                    removed = bg_info.pop("removed", None)
                    info["bg_compliance"] = bg_info
                    if cfg.compliance.bg_sticky_removals:
                        # A region a background scribble deleted stays deleted even if
                        # the model re-creates it; costs state-independence.
                        sticky = cache.load_bg_region()
                        if sticky is not None and sticky.shape == shape:
                            mask = (mask.astype(bool) & ~sticky).astype(np.uint8)
                            removed = sticky if removed is None else (removed | sticky)
                            info["bg_compliance"]["sticky_applied"] = True
                else:
                    info["bg_compliance"] = {"skipped": True}

            # ---- 7. foreground compliance (G2 / G3) -------------------------
            with timer(info, "fg_compliance"):
                if cfg.enable_fg_compliance:
                    mask, fg_info = apply_tumor_scribbles(
                        mask,
                        pet,
                        ct,
                        fg_points,
                        prob=prob_fg,
                        tracer=tracer,
                        spacing=spacing,
                        cfg=cfg.compliance,
                        forbidden=removed,
                        bg_points=bg_points,
                        return_info=True,
                    )
                    info["fg_compliance"] = fg_info
                else:
                    info["fg_compliance"] = {"skipped": True}

            # ---- 7b. merge components (ablation knob) -----------------------
            # Runs after compliance, so a background split is not undone, and it never
            # fills a forbidden voxel.
            with timer(info, "bridge"):
                if cfg.cleanup.bridge_closing_voxels:
                    forbid = points_mask(shape, bg_points) if len(bg_points) else None
                    if removed is not None:
                        rm = np.asarray(removed)
                        if rm.shape == tuple(shape):
                            forbid = rm.astype(bool) if forbid is None else (forbid | rm.astype(bool))
                    mask, bridge_info = bridge_components(
                        mask,
                        spacing,
                        closing_voxels=cfg.cleanup.bridge_closing_voxels,
                        forbidden=forbid,
                        max_added_ml=cfg.cleanup.bridge_max_added_ml,
                        return_info=True,
                    )
                    info["bridge"] = bridge_info

            # ---- 7c. second cleanup pass, after compliance (ablation knob) --
            # The first pass ran on the model's own output; split fragments and grown
            # regions have never been through the pruning rule.
            with timer(info, "cleanup2"):
                if cfg.enable_cleanup and cfg.cleanup_after_compliance:
                    mask, clean2 = cleanup_mask(
                        mask,
                        pet,
                        spacing,
                        tracer=tracer,
                        cfg=cfg.cleanup,
                        protect_points=fg_points,
                        background_points=bg_points,
                        prob=prob_fg,
                        iteration=iteration,
                        forbidden=removed,
                        return_info=True,
                    )
                    info["cleanup_after_compliance"] = clean2

            # ---- 8. scribble-calibrated global rescue (ablation knob) -------
            if cfg.rescue_threshold_from_scribble and prob_fg is not None and len(fg_points):
                with timer(info, "rescue"):
                    mask = self._rescue_from_scribble(
                        mask, pet, prob_fg, fg_points, bg_points, spacing, tracer, info
                    )

        # ---- 9. monotone blend, last (ablation knob; "none" by default) -----
        with timer(info, "monotone"):
            if not gate_fired and cfg.monotone.mode != "none":
                mask = self._blend(
                    mask, prob_fg, fg_points, bg_points, removed, cache, spacing, info
                )
                # blending re-binarises, so the guarantees must be re-imposed
                if cfg.enable_bg_compliance:
                    mask = apply_background_scribbles(
                        mask, pet, bg_points, connectivity=cfg.compliance.connectivity,
                        prob=prob_fg, spacing=spacing, fg_points=fg_points, tracer=tracer,
                        cfg=cfg.compliance,
                    )
                if cfg.enable_fg_compliance:
                    mask = apply_tumor_scribbles(
                        mask, pet, ct, fg_points, prob=prob_fg, tracer=tracer, spacing=spacing,
                        cfg=cfg.compliance, forbidden=removed, bg_points=bg_points,
                    )

        # ---- 9b. G4: bounded perturbation of what nobody corrected ----------
        # Compliance may only reshape the components a scribble points at; a base-mask
        # component holding no background scribble has to survive it.
        with timer(info, "guard"):
            if cfg.guard_enabled and not gate_fired:
                mask, guard_info = _apply_damage_guard(
                    mask, base_mask, after_cleanup, bg_points, spacing, cfg
                )
                info["guard"] = guard_info
                if guard_info["n_components_restored"]:
                    mask = apply_background_scribbles(
                        mask, pet, bg_points, connectivity=cfg.compliance.connectivity,
                        prob=prob_fg, spacing=spacing, fg_points=fg_points, tracer=tracer,
                        cfg=cfg.compliance,
                    )

        # ---- 10. re-assert --------------------------------------------------
        # Only guarantees whose stage is enabled are re-asserted: an ablation rung that
        # deliberately runs without tumor-scribble compliance must not fail `strict`.
        with timer(info, "assert"):
            chk_fg = fg_points if cfg.enable_fg_compliance else np.zeros((0, 3), dtype=np.int64)
            chk_bg = bg_points if cfg.enable_bg_compliance else np.zeros((0, 3), dtype=np.int64)
            res = check_constraints(mask, chk_fg, chk_bg)
            if not res["ok"]:
                if cfg.enable_bg_compliance:
                    mask = apply_background_scribbles(
                        mask, pet, bg_points, connectivity=cfg.compliance.connectivity,
                        spacing=spacing, fg_points=fg_points, tracer=tracer, cfg=cfg.compliance,
                    )
                if cfg.enable_fg_compliance:
                    mask = apply_tumor_scribbles(
                        mask, pet, ct, fg_points, tracer=tracer, spacing=spacing,
                        cfg=cfg.compliance, forbidden=removed, bg_points=bg_points,
                    )
                if cfg.strict:
                    assert_constraints(mask, chk_fg, chk_bg)
                res = check_constraints(mask, chk_fg, chk_bg)
            info["constraints"] = {
                k: res[k] for k in ("ok", "n_fg", "n_bg", "n_fg_missing", "n_bg_inside")
            }

        mask = np.ascontiguousarray(mask, dtype=np.uint8)
        n_final = int(mask.sum())
        # Emptying is the gate's decision, never a side effect of cleanup or compliance,
        # so an empty output without the gate is flagged.
        base_bool = base_mask.astype(bool)
        final_bool = mask.astype(bool)
        info["base_volume_after_gate_ml"] = float(int(base_bool.sum()) * vox_ml)
        info["base_removed_ml"] = float(int((base_bool & ~final_bool).sum()) * vox_ml)
        info["base_added_ml"] = float(int((~base_bool & final_bool).sum()) * vox_ml)
        info["empty_output"] = n_final == 0
        info["empty_without_gate"] = bool(n_final == 0 and not gate_fired)
        info["final_volume_ml"] = float(n_final * vox_ml)

        # ---- 11. persist the speed cache ------------------------------------
        with timer(info, "cache_save"):
            state.iteration = iteration          # the index we derived from the input
            state.n_calls = (int(cached.n_calls) + 1) if had_state else 1
            info["n_calls"] = state.n_calls
            state.tracer = tracer
            state.history = (cached.history if had_state else []) + [
                {
                    "iteration": iteration,
                    "volume_ml": info["final_volume_ml"],
                    "n_tumor": int(len(fg_points)),
                    "n_background": int(len(bg_points)),
                    "negative_gate": bool(gate_fired),
                }
            ]
            cache.save_state(state)
            if cfg.cache_mask:
                cache.save_prev_mask(mask.astype(bool))
            if cfg.cache_probabilities and prob_fg is not None:
                cache.save_prev_prob(prob_fg)
            if removed is not None and removed.any():
                cache.accumulate_bg_region(removed)

        info["t_total"] = time.perf_counter() - t_start
        info["tracer"] = tracer
        self.last_info = info
        if cfg.verbose:
            print(f"[postproc] {_fmt_info(info)}", flush=True)

        if return_probabilities:
            return mask, (prob_fg if prob_fg is not None else mask.astype(np.float32))
        return mask

    # ----------------------------------------------------------------------
    def _return_unchanged(self, prev_mask, state, cached, cache, info, spacing, vox_ml,
                          n_new_fg, n_new_bg, t_start, return_probabilities):
        """Hand back the previous mask without running the network.

        Reached only when every newly arrived scribble already holds for that mask, so
        all accumulated constraints still hold: the previous call asserted them for the
        points it knew about, and the new points are satisfied by construction.
        """
        mask = np.ascontiguousarray(prev_mask.astype(np.uint8))
        fg_points = state.tumor_array(mask.shape)
        bg_points = state.background_array(mask.shape)

        res = check_constraints(mask, fg_points, bg_points)
        if not res["ok"] and self.cfg.strict:
            # cannot happen by construction; never ship a non-compliant mask if it does
            assert_constraints(mask, fg_points, bg_points)
        info["constraints"] = {
            k: res[k] for k in ("ok", "n_fg", "n_bg", "n_fg_missing", "n_bg_inside")
        }

        n_final = int(mask.sum())
        info.update(
            skipped_inference=True,
            skip_reason="new scribble already satisfied",
            n_new_tumor=int(n_new_fg),
            n_new_background=int(n_new_bg),
            base_volume_ml=float(n_final * vox_ml),
            final_volume_ml=float(n_final * vox_ml),
            base_removed_ml=0.0,
            base_added_ml=0.0,
            negative_gate_fired=bool(cached.negative_gate_fired),
            empty_output=n_final == 0,
            empty_without_gate=bool(n_final == 0 and not cached.negative_gate_fired),
            tracer=cached.tracer,
        )

        state.iteration = int(info["iteration"])
        state.n_calls = int(cached.n_calls) + 1
        state.tracer = cached.tracer
        state.negative_gate_fired = bool(cached.negative_gate_fired)
        state.history = list(cached.history) + [{
            "iteration": int(info["iteration"]),
            "volume_ml": info["final_volume_ml"],
            "n_tumor": int(len(fg_points)),
            "n_background": int(len(bg_points)),
            "skipped_inference": True,
        }]
        cache.save_state(state)

        info["t_total"] = time.perf_counter() - t_start
        self.last_info = info
        if self.cfg.verbose:
            print(f"[postproc] {_fmt_info(info)}", flush=True)
        if return_probabilities:
            prev_prob = cache.load_prev_prob()
            if prev_prob is None or prev_prob.shape != mask.shape:
                prev_prob = mask.astype(np.float32)
            return mask, prev_prob
        return mask

    def _blend(self, mask, prob_fg, fg_points, bg_points, removed, cache, spacing, info):
        """Monotone blend against the cached previous iteration (opt-in)."""
        cfg = self.cfg
        shape = mask.shape
        fg_region = (
            points_mask(shape, fg_points, cfg.monotone.fg_constraint_radius_mm, spacing)
            if len(fg_points)
            else None
        )
        bg_region = cache.load_bg_region()
        if bg_region is not None and bg_region.shape != shape:
            bg_region = None
        if removed is not None:
            bg_region = removed if bg_region is None else (bg_region | removed)
        if len(bg_points):
            pts = points_mask(shape, bg_points, cfg.monotone.bg_constraint_radius_mm, spacing)
            bg_region = pts if bg_region is None else (bg_region | pts)

        if prob_fg is not None:
            prev = cache.load_prev_prob()
            if prev is not None and prev.shape == shape:
                blended = blend_with_previous(prob_fg, prev, fg_region, bg_region, cfg=cfg.monotone)
                info["monotone_applied"] = "probability"
                return (blended >= cfg.monotone.binarise_threshold).astype(np.uint8)
            return mask
        prev_mask = cache.load_prev_mask()
        if prev_mask is not None and prev_mask.shape == shape:
            info["monotone_applied"] = "mask"
            return blend_masks(mask, prev_mask, fg_region, bg_region, mode="minmax")
        return mask

    def _rescue_from_scribble(self, mask, pet, prob_fg, fg_points, bg_points, spacing, tracer, info):
        """Lower the global detection threshold to what the scribbled lesion needed.

        A tumor scribble lands on the largest missed component, so lesions of similar
        conspicuity elsewhere were probably missed too.
        """
        cfg = self.cfg
        vals = prob_fg[fg_points[:, 0], fg_points[:, 1], fg_points[:, 2]]
        level = max(float(np.percentile(vals, cfg.rescue_percentile)), cfg.rescue_min_threshold)
        if level >= cfg.monotone.binarise_threshold:
            info["rescue"] = {"applied": False, "level": level}
            return mask
        extra = cleanup_mask(
            (prob_fg >= level).astype(np.uint8),
            pet, spacing, tracer=tracer, cfg=cfg.cleanup, protect_points=fg_points,
        )
        merged = ((mask > 0) | (extra > 0)).astype(np.uint8)
        merged = apply_background_scribbles(
            merged, pet, bg_points, connectivity=cfg.compliance.connectivity,
            prob=prob_fg, spacing=spacing, fg_points=fg_points, tracer=tracer,
            cfg=cfg.compliance,
        )
        info["rescue"] = {
            "applied": True,
            "level": level,
            "added_ml": float((int(merged.sum()) - int(mask.sum())) * voxel_volume_ml(spacing)),
        }
        return merged

    def _resolve_tracer(self, pet, spacing, affine, info, ct=None) -> str:
        cfg_tracer = (self.cfg.tracer or "auto").lower()
        if cfg_tracer in ("fdg", "psma"):
            return cfg_tracer
        # The CT locates the body, so the head slab is measured against the patient and
        # not against the array extent.
        tracer, feats = guess_tracer(pet, spacing, ct=ct, affine=affine, return_features=True)
        info["tracer_features"] = feats
        return tracer

    def _introspect_base(self) -> Tuple[set, bool]:
        """Which keyword arguments the base predictor accepts, introspected once.

        A minimal predictor may take only ``(ct, pet, spacing, scribbles)``, and passing
        it ``affine=`` or ``ct_path=`` would be a TypeError.
        """
        if self._sig_cache is None:
            try:
                params = inspect.signature(self.base.predict).parameters
                names = {
                    n for n, p in params.items()
                    if p.kind in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY)
                }
                var_kw = any(p.kind == p.VAR_KEYWORD for p in params.values())
            except (TypeError, ValueError):  # builtins / C callables
                names, var_kw = set(), True
            self._sig_cache = (names, var_kw)
        return self._sig_cache

    def _call_base(self, **kw: Any) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Call the base predictor, asking for probabilities when it supports them."""
        names, var_kw = self._introspect_base()
        if not var_kw:
            kw = {k: v for k, v in kw.items() if k in names}
        supports_prob = var_kw or ("return_probabilities" in names)
        if self._base_supports_prob is False:
            supports_prob = False

        if self.cfg.use_probabilities and supports_prob:
            out = self.base.predict(**kw, return_probabilities=True)
            if isinstance(out, tuple) and len(out) == 2:
                self._base_supports_prob = True
                return out[0], out[1]
            self._base_supports_prob = False
            return out, None

        out = self.base.predict(**kw)
        if isinstance(out, tuple) and len(out) == 2:
            return out[0], out[1]
        return out, None


def _new_points(state, cached, shape):
    """The scribble points that arrived since the previous call.

    ``cached`` is the constraint state as it was saved at the end of the previous call,
    so anything in ``state`` but not in ``cached`` is new in this one.
    """
    def _fresh(now, before):
        seen = {tuple(int(v) for v in p) for p in as_points_array(before).tolist()}
        return np.array(
            [p for p in now.tolist() if tuple(p) not in seen], dtype=np.int64
        ).reshape(-1, 3)

    return (_fresh(state.tumor_array(shape), cached.tumor_points),
            _fresh(state.background_array(shape), cached.background_points))


def _already_satisfied(mask, new_fg, new_bg):
    """True when every new tumor point is inside ``mask`` and every new background point
    is outside it.

    Requires at least one new point: with none, there is no new scribble to be satisfied
    and the normal path runs, which keeps the rule strictly about *corrections we have
    already made* rather than a general "reuse the last answer" shortcut.
    """
    if len(new_fg) == 0 and len(new_bg) == 0:
        return False
    m = np.asarray(mask) > 0
    if not m.any():
        # An empty mask satisfies every background point vacuously, so the rule would
        # freeze an empty prediction for another iteration -- and an empty prediction on
        # a positive case is the worst outcome this evaluator scores.  Nothing can be
        # confirmed about an empty mask; always re-run.
        return False
    if len(new_fg) and not m[new_fg[:, 0], new_fg[:, 1], new_fg[:, 2]].all():
        return False
    if len(new_bg) and m[new_bg[:, 0], new_bg[:, 1], new_bg[:, 2]].any():
        return False
    return True


def _apply_damage_guard(mask, base_mask, after_cleanup, bg_points, spacing, cfg):
    """G4: restore compliance damage to base components that no background scribble hit.

    ``after_cleanup & ~mask`` is what compliance removed; cleanup's own removals are
    separately gated and exempt.  For a base component holding no background scribble,
    anything removed beyond ``guard_max_component_removed_fraction`` is put back.
    """
    import cc3d

    out = np.asarray(mask).astype(bool)
    base = np.asarray(base_mask).astype(bool)
    info = {"n_components_restored": 0, "restored_ml": 0.0}
    lost = np.asarray(after_cleanup).astype(bool) & ~out
    if not lost.any() or not base.any():
        return out.astype(np.uint8), info

    labels = cc3d.connected_components(
        np.ascontiguousarray(base).view(np.uint8), connectivity=cfg.compliance.connectivity
    )
    counts = np.bincount(labels.ravel())
    hit = np.bincount(np.asarray(labels[lost]).ravel(), minlength=len(counts))[: len(counts)]

    bg = points_in_bounds(unique_points(as_points_array(bg_points)), base.shape)
    scribbled = set()
    if len(bg):
        vals = labels[bg[:, 0], bg[:, 1], bg[:, 2]]
        scribbled = {int(v) for v in np.unique(vals) if v > 0}

    restore = []
    for lab in range(1, len(counts)):
        if counts[lab] == 0 or lab in scribbled:
            continue
        if hit[lab] / counts[lab] > cfg.guard_max_component_removed_fraction:
            restore.append(lab)
    if restore:
        put_back = np.isin(labels, restore) & lost
        out |= put_back
        info["n_components_restored"] = len(restore)
        info["restored_ml"] = float(int(put_back.sum()) * voxel_volume_ml(spacing))
    return out.astype(np.uint8), info


def _as_prev(
    prev_pred: Optional[np.ndarray], cache: CaseCache, use_cache: bool
) -> Optional[np.ndarray]:
    """The previous prediction handed to the base predictor, as a hint only.

    An explicit ``prev_pred`` is always forwarded; the cached one only when
    ``pass_cached_prev_pred`` is set, since a predictor that consumes it makes the
    end-to-end output depend on the state directory.
    """
    if prev_pred is not None:
        return np.asarray(prev_pred).astype(np.uint8)
    if not use_cache:
        return None
    cached = cache.load_prev_mask()
    return None if cached is None else cached.astype(np.uint8)


def _fmt_info(info: Dict[str, Any]) -> str:
    parts = [
        f"it={info.get('iteration')}",
        f"tracer={info.get('tracer')}",
        f"fg={info.get('n_tumor')}",
        f"bg={info.get('n_background')}",
        f"base={info.get('base_volume_ml', 0):.2f}mL",
        f"final={info.get('final_volume_ml', 0):.2f}mL",
        f"t={info.get('t_total', 0):.2f}s",
    ]
    for key in ("base_predict", "cleanup", "bg_compliance", "fg_compliance", "monotone", "cache_save"):
        k = f"t_{key}"
        if k in info:
            parts.append(f"{key}={info[k]:.2f}s")
    if info.get("skipped_inference"):
        parts.append("SKIPPED_INFERENCE(new scribble already satisfied)")
    if info.get("negative_gate_fired"):
        parts.append("NEGATIVE_GATE")
    if info.get("empty_without_gate"):
        parts.append("EMPTY_WITHOUT_GATE!")
    return " ".join(parts)
