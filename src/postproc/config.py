"""Configuration for the post-processing / interaction layer.

Every threshold lives here so a sweep is a one-line change.  18-connectivity is used
everywhere because the scorer uses it; ``fg_alpha = 0.41`` is the clinical
41 %-of-SUVpeak PET delineation threshold.  At the 3.0 x 2.04 x 2.04 mm working
spacing one voxel is 12.45 mm^3, so 10 voxels is 0.125 mL.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass, asdict
from typing import Any, Dict, Mapping, Tuple

__all__ = [
    "TRACER_SUV_FLOOR",
    "ComplianceConfig",
    "CleanupConfig",
    "NegativeGateConfig",
    "MonotoneConfig",
    "PostProcConfig",
]

#: Absolute SUV floor below which a voxel cannot plausibly be lesion.
TRACER_SUV_FLOOR: Dict[str, float] = {"fdg": 1.5, "psma": 1.0, "unknown": 1.0}


@dataclass
class ComplianceConfig:
    """Hard scribble-compliance parameters."""

    connectivity: int = 18  # must match metrics.MetricEvaluator(connectivity=18)

    # --- background scribbles -------------------------------------------------
    # Splitting a component is the primary path, not deleting it: the scribble sits in
    # the error mask, a subset of a component that may well be a true positive that bled.
    #: A voxel belongs to the component's "core" if the model is at least this sure.
    bg_core_prob: float = 0.9
    #: Fallback when no softmax is available: core = SUV >= ratio * SUVmax(component) ...
    bg_core_suv_ratio: float = 0.6
    #: ... and at least this much hotter than the hottest scribbled voxel.  Without a
    #: confidence map this ratio is what separates a true positive that bled into cool
    #: tissue (split) from a homogeneous false positive (no core -> delete).
    bg_core_min_suv_ratio: float = 1.5
    #: Core voxels must be at least this far from every background scribble voxel.
    bg_core_margin_mm: float = 5.0
    #: A core smaller than this does not count as a core (-> delete the component).
    bg_core_min_volume_ml: float = 0.05
    #: Geodesic radius, inside the component, that the removed region may reach.  Keeps
    #: a watershed on a flat probability landscape from claiming most of a good lesion.
    bg_remove_max_radius_mm: float = 20.0
    #: Never delete a voxel the model is confident about (``prob > bg_core_prob``), even
    #: when the watershed assigns it to the scribble's basin.
    bg_protect_confident_core: bool = True
    #: A split removing more than this fraction of a component with a confident core is
    #: destruction, not correction: fall back to the neighbourhood of the scribble.
    bg_split_max_removed_fraction: float = 0.5
    #: Radius of that fallback neighbourhood.
    bg_local_fallback_radius_mm: float = 8.0
    #: If the surviving part is smaller than this, delete the whole component instead.
    bg_split_min_kept_volume_ml: float = 0.05
    #: Keep whatever a background scribble has ever deleted deleted, from the cached
    #: background region.  Off: it makes the output depend on the state directory.
    bg_sticky_removals: bool = False
    #: Use the model softmax (if available) instead of -PET as the watershed landscape.
    bg_split_use_prob: bool = True
    #: Additionally forbid a ball of this radius around every background scribble voxel
    #: when growing from tumor scribbles.  0.0 = forbid exactly the scribble voxels.
    bg_forbid_radius_mm: float = 0.0

    # --- tumor scribbles ------------------------------------------------------
    #: Threshold = max(alpha * local SUVpeak, tracer floor).  0.41 = clinical 41 %-of-peak.
    fg_alpha: float = 0.41
    #: Radius of the sphere used for the local SUVpeak average (1 mL sphere ~ 6.2 mm).
    fg_peak_radius_mm: float = 6.2
    #: Take the local SUVpeak over the whole scribble cluster, not only the voxels being
    #: grown from: measured on the cold tail alone the 41 % threshold collapses onto the
    #: tracer floor.
    fg_peak_over_whole_cluster: bool = True
    #: If ``fg_alpha * SUVpeak_local`` does not clear the tracer floor, the 41 % rule has
    #: no information and the floor (a "cannot be lesion" bound, not a delineation
    #: criterion) would be doing the thresholding.  Fall back to the bounded ball.
    fg_low_contrast_fallback: bool = True
    #: Hard bound on how far the grown region may travel from the scribble.  Measured
    #: geodesically (inside the candidate set) when ``fg_geodesic``, else euclidean.
    fg_max_radius_mm: float = 30.0
    #: Bound the growth by geodesic rather than euclidean distance.
    fg_geodesic: bool = True
    #: Calibrate the SUV threshold against the scribble footprint (see compliance.py).
    fg_calibrate_from_footprint: bool = True
    #: A footprint whose inner radius is below this is a thin line, not a cross-section,
    #: and carries no area information -> fall back to the alpha * SUVpeak rule.
    fg_footprint_min_thickness_mm: float = 2.5
    #: SUV percentiles over the footprint used as candidate thresholds during
    #: calibration (plus the alpha rule and the tracer floor).
    fg_calibration_percentiles: Tuple[float, ...] = (2, 5, 10, 15, 20, 25, 30, 40, 50, 60)
    #: Stop the growth at local SUV minima (watershed against competing h-maxima).
    fg_stop_at_valleys: bool = True
    #: Depth (in SUV) a competing maximum must have to count as a separate focus.
    fg_valley_depth_suv: float = 1.0
    #: ... and how far it must be from the scribble ...
    fg_valley_min_distance_mm: float = 10.0
    #: ... and how hot, relative to the local SUVpeak at the scribble.
    fg_valley_peak_ratio: float = 0.5
    #: If every scribble voxel is below threshold, fall back to a ball of this radius.
    fg_fallback_ball_radius_mm: float = 4.0
    #: Voxels with model probability above this are eligible even if SUV is below thr.
    fg_prob_include: float = 0.3
    #: Absolute volume cap for one grown region; the threshold is escalated until it fits.
    fg_max_volume_ml: float = 100.0
    #: Relative cap: a scribble may not grow a region more than this many times the
    #: evidence it starts from (the components already holding it, plus its footprint).
    #: The absolute cap alone is far too loose for a 2 mL node.
    fg_max_relative_growth: float = 8.0
    #: Floor for that relative cap, so growing from a lesion the model missed entirely
    #: still works.  10 mL is a 26 mm sphere.
    fg_min_growth_ml: float = 10.0
    #: Multipliers applied to the base threshold when the volume cap is exceeded.
    fg_threshold_escalation: Tuple[float, ...] = (1.0, 1.3, 1.7, 2.2, 3.0)
    #: Exclude voxels whose CT value is below this (air / outside the patient).
    #: Set to None to ignore CT.
    fg_exclude_ct_below_hu: float = -400.0
    #: Scribble points closer than this (mm) are grown together in one crop.
    fg_cluster_radius_mm: float = 20.0
    #: Extra margin added to the crop around a scribble cluster.
    fg_crop_margin_mm: float = 6.0
    #: "grow" (SUV-adaptive region growing) or "random_walker".
    fg_method: str = "grow"

    # --- random-walker variant ------------------------------------------------
    rw_crop_size: int = 64  # cube side in voxels
    rw_beta: float = 130.0
    rw_bg_distance_mm: float = 25.0
    rw_prob_threshold: float = 0.5
    rw_mode: str = "cg_j"
    #: CG tolerance; tighter than skimage's 1e-3 default so the solver's probabilities
    #: stay inside [0, 1].
    rw_tol: float = 1e-4
    rw_prob_tol: float = 1e-2


@dataclass
class CleanupConfig:
    """Detection-metric-oriented component cleanup."""

    connectivity: int = 18
    #: Components below this volume are removed ...
    min_volume_ml: float = 0.3
    #: ... unless their SUVmax reaches this gate.  Gate on SUV, not on size alone.
    suv_gate: float = 4.0
    #: Apply the tracer SUV floor.  "component" removes a component whose SUVmax is
    #: below the floor; "voxel" additionally erases sub-floor voxels; "off" disables.
    suv_floor_mode: str = "component"
    #: Overrides TRACER_SUV_FLOOR when not None.
    suv_floor: float | None = None
    #: Fill background holes strictly smaller than this inside a lesion.
    fill_holes_max_ml: float = 0.5
    fill_holes: bool = True
    #: Cleanup must never empty a non-empty prediction: if every component would be
    #: pruned, the best this many are kept, ranked by SUVmax then volume.  Emptying is
    #: the negative gate's decision, not cleanup's.
    min_components_kept: int = 1

    # --- rule v2 (off by default) -------------------------------------------
    #: Use the fitted three-feature rule below instead of the volume/SUV rule above.
    rule_v2: bool = False
    #: v2 prunes a component iff every enabled criterion holds: ``volume_ml <
    #: v2_min_volume_ml`` and ``suv_max < v2_suv_gate`` and ``prob_mean < v2_prob_gate``.
    #: ``None`` disables a criterion, so ``(0.3, 4.0, None)`` is the v1 rule.
    v2_min_volume_ml: float | None = 0.3
    v2_suv_gate: float | None = 4.0
    #: Mean foreground softmax over the component.  The max saturates near 1.0 almost
    #: everywhere; the mean still separates a confident lesion from a marginal one.
    v2_prob_gate: float | None = None
    #: A union of conjunctions: prune a component matching any entry.  Each entry is a
    #: dict of the three keys above (missing = disabled).  ``None`` keeps the single
    #: conjunction defined by the scalars.
    v2_rules: tuple | None = None
    #: Per-tracer overrides, e.g. ``{"fdg": {"v2_min_volume_ml": 1.0}}``.
    v2_by_tracer: Dict[str, Dict[str, Any]] | None = None
    #: Multiply the size threshold by this factor per iteration for a component that has
    #: not attracted a background scribble: the evaluator points at the largest error
    #: every round, so a survivor is progressively more likely to be a true positive.
    #: ``1.0`` disables it; ``< 1`` protects survivors, ``> 1`` prunes them harder.
    v2_silence_decay: float = 1.0
    #: Apply the decay only once a background scribble has actually arrived, i.e. only
    #: when the scribbles demonstrably follow our errors.
    v2_silence_requires_bg: bool = True

    # --- component merging (off by default) ---------------------------------
    #: Close the mask by this many voxels, joining components separated by a gap of at
    #: most twice that.  Multi-assignment is not punished, so a merge is at worst neutral
    #: for detection; the failure mode is the union's IoU with a small lesion dropping
    #: below 0.1, which is why one voxel is the useful setting.  0 = off.
    bridge_closing_voxels: int = 0
    #: Refuse the whole closing if it would add more than this.
    bridge_max_added_ml: float | None = 5.0

    # --- recall recruitment (off by default) --------------------------------
    #: Matching needs only IoU >= 0.1, so a component that appears only below the argmax
    #: threshold still scores a true positive.  When set, the foreground softmax is
    #: re-binarised here and components touching no argmax voxel are added if they clear
    #: the two gates below.  Purely additive; the argmax mask is untouched.  ``None`` = off.
    recruit_prob_threshold: float | None = None
    recruit_min_suv_max: float = 4.0
    recruit_min_volume_ml: float = 0.1
    #: Hard cap on how many components one pass may add, so a badly chosen threshold
    #: cannot flood a case with false positives.
    recruit_max_components: int = 5


@dataclass
class NegativeGateConfig:
    """Lesion-free gate: when to replace the prediction with an empty mask.

    A lesion-absent case is excluded from DMM but scores Dice 1.0 only if the prediction
    is empty, and never receives a scribble.  Every threshold accepts ``None``, which
    disables that criterion; the gate fires when all enabled criteria hold.  Defaults are
    the leave-one-out fit of ``postproc/tools/gate_sweep.py``, with the threshold taken
    from the flat top of the curve rather than from the optimum on the data edge.
    """

    enabled: bool = True
    #: Total predicted volume must be below this ...
    max_total_volume_ml: float | None = 6.0
    #: ... and the largest component below this ...
    max_component_volume_ml: float | None = None
    #: ... and the number of components at most this ...
    max_n_components: int | None = None
    #: ... and the maximum model probability below this.  Off: a predicted voxel is the
    #: argmax of the softmax, so ``prob_max_in_mask`` is ~1.0 on every non-empty case.
    max_prob: float | None = None
    #: Mean foreground softmax over the predicted voxels must be below this.  It does
    #: discriminate, but adds nothing over the volume criterion; kept as a sweep knob.
    max_mean_prob: float | None = None
    #: Confidence-weighted volume (sum of the softmax over the mask, in mL).
    max_soft_volume_ml: float | None = None
    #: ... and the SUVmax inside the mask below this.  Off: false positives on
    #: lesion-free cases sit on physiologically hot tissue (bladder rim, bowel, brain),
    #: so their SUVmax is high exactly when they are false.
    max_suv: float | None = None
    #: Per-tracer override of ``max_suv``, e.g. ``{"fdg": 5.0, "psma": 8.0}``.
    max_suv_by_tracer: Dict[str, float] | None = None
    #: Fire only while no scribble of either kind has been seen.  True for all six
    #: iterations of a lesion-free case, false forever once one arrives.
    require_no_scribbles: bool = True
    #: Additionally restrict to iteration 0.  Off: a negative case whose state directory
    #: is missing would then stop being emptied after iteration 0.
    only_iteration_zero: bool = False
    #: If the model gives no probability map, ignore the probability criteria.
    require_prob: bool = False
    #: Also compute per-component statistics (``features["components"]``).  Costs one
    #: extra gather over the predicted voxels; only needed for logging and sweeps.
    collect_components: bool = False


@dataclass
class MonotoneConfig:
    """Anti-oscillation blending against the previous iteration."""

    #: "none" | "minmax" | "ema" | "ema_minmax".  "none" by default: any other mode
    #: reads the cached previous probability, so the pipeline stops being a pure
    #: function of (CT, PET, accumulated scribbles) and starts depending on a state
    #: directory that may not exist.  The other modes are ablation knobs.
    mode: str = "none"
    #: Weight of the new probability in the EMA.
    ema_alpha: float = 0.6
    #: Radius of the foreground-constraint region grown around tumor scribble voxels.
    fg_constraint_radius_mm: float = 6.0
    #: Radius of the background-constraint region grown around background scribbles.
    bg_constraint_radius_mm: float = 6.0
    #: Binarisation threshold applied to the blended probability.
    binarise_threshold: float = 0.5


@dataclass
class PostProcConfig:
    """Top-level configuration for ``postproc.pipeline.PostProcPredictor``."""

    compliance: ComplianceConfig = field(default_factory=ComplianceConfig)
    cleanup: CleanupConfig = field(default_factory=CleanupConfig)
    negative_gate: NegativeGateConfig = field(default_factory=NegativeGateConfig)
    monotone: MonotoneConfig = field(default_factory=MonotoneConfig)

    # --- stage switches (ablation knobs) ---------------------------------
    # All True == the shipped pipeline.  They let the ablation ladder be expressed as a
    # config instead of a code path per rung.  Turning a stage off also drops its
    # guarantee from the final re-assert, so `strict` does not then fail on it.
    #: Component cleanup (SUV floor, small components, hole filling).
    enable_cleanup: bool = True
    #: Run the cleanup stage a second time, after both compliance stages, so that the
    #: components the interaction layer itself creates (a fragment left by a split, a
    #: region grown from a tumor scribble) are pruned too.  Off by default.
    cleanup_after_compliance: bool = False
    #: Hard background-scribble compliance (G1).
    enable_bg_compliance: bool = True
    #: Hard tumor-scribble compliance (G2/G3).
    enable_fg_compliance: bool = True

    #: "fdg" | "psma" | "auto" | None.  "auto" runs the heuristic tracer classifier
    #: once at iteration 0 and caches the answer for the remaining iterations.
    tracer: str = "auto"
    #: Ask the base predictor for probabilities (falls back gracefully if unsupported).
    use_probabilities: bool = True
    #: Persist the probability map (quantised uint8) for the monotone blend.
    cache_probabilities: bool = True
    #: Persist the binary mask (bit-packed).
    cache_mask: bool = True
    #: Hand the cached previous mask to the base predictor as ``prev_pred``.  Off by
    #: default: it makes the end-to-end output depend on the state directory.  Turn it
    #: on only for a model with a previous-prediction input channel.  A ``prev_pred``
    #: passed explicitly by the caller is always forwarded regardless.
    pass_cached_prev_pred: bool = False
    #: Raise instead of warning if a constraint is still violated after everything.
    strict: bool = True
    #: Emit per-stage timings into the returned info dict / log.
    verbose: bool = False
    #: Lower the global binarisation threshold to the level at which the scribbled
    #: lesion would have been detected and re-run component analysis, recovering other
    #: lesions of similar conspicuity.  Off: it turns a local hint into a global change.
    rescue_threshold_from_scribble: bool = False
    #: Percentile of the model probability over the scribble voxels used as that level.
    rescue_percentile: float = 50.0
    #: Floor for the rescued threshold, so the knob can never flood the volume.
    rescue_min_threshold: float = 0.15
    #: G4: a base-mask component that no background scribble points at may not lose more
    #: than this fraction of its volume to compliance.  Cleanup's size/SUV rules are
    #: exempt; they are gated separately.
    guard_max_component_removed_fraction: float = 0.5
    #: Enable the G4 guard.
    guard_enabled: bool = True
    #: Skip inference when the newly arrived scribble is already satisfied.
    #:
    #: Under the challenge's Category-2 replay the scribbles were collected against the
    #: *baseline's* errors and are replayed to us unchanged, so many of them already hold
    #: for our mask.  Feeding such a scribble to the network anyway re-runs it with new
    #: guidance channels and a new previous-mask channel, and the model changes its output
    #: globally in response to a correction that asked for nothing.  When the new scribble
    #: is already satisfied there is nothing to correct, so the previous mask is returned
    #: unchanged and the network is not called.
    #:
    #: Inert under simulated interaction: a simulated tumor scribble is drawn on
    #: ``~pred & gt`` and a background scribble on ``pred & ~gt``, both computed from the
    #: mask we returned, so a new simulated scribble is never already satisfied.
    #:
    #: This is the one place where the layer deliberately depends on the state directory.
    #: With no previous mask it falls through to the normal path, so it degrades safely.
    skip_inference_if_satisfied: bool = True

    # ------------------------------------------------------------------
    @classmethod
    def from_dict(cls, cfg: Mapping[str, Any] | "PostProcConfig" | None) -> "PostProcConfig":
        """Build a config from a possibly nested, possibly partial dict.

        Unknown keys raise; a silently ignored typo in a sweep is worse than a crash.
        """
        if cfg is None:
            return cls()
        if isinstance(cfg, PostProcConfig):
            return cfg
        return _apply(cls(), cfg)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _apply(obj: Any, cfg: Mapping[str, Any]) -> Any:
    valid = {f.name: f for f in fields(obj)}
    for key, value in cfg.items():
        if key not in valid:
            raise KeyError(
                f"unknown config key {key!r} for {type(obj).__name__}; "
                f"valid keys: {sorted(valid)}"
            )
        current = getattr(obj, key)
        if is_dataclass(current) and isinstance(value, Mapping):
            _apply(current, value)
        else:
            setattr(obj, key, value)
    return obj
