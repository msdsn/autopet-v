# `src/postproc/` — inference-time interaction layer

Everything that happens *around* the network at inference time: hard scribble
compliance, false-positive cleanup aimed at the lesion-detection metric, the lesion-free
gate, and the state handling for the six container calls of one case.

It is model-agnostic. `PostProcPredictor` wraps any duck-typed object exposing
`predict(ct, pet, spacing, scribbles, prev_pred, case_cache_dir, ...) -> mask`, i.e. the
`Predictor` interface in `src/predictor.py`. It works unchanged on the stock baseline
weights and on a fine-tuned network.

---

## 1. Protocol facts this layer is built around

Confirmed by the organizers, and by reading `autoPETV/interactive/interactive_loop.py`,
`autoPETV/interactive/simulate_scribbles.py` and `autoPETV/metrics.py`:

* **6 iterations** per case (0..5): iteration 0 with no scribbles, then five
  corrections. The score is `np.trapz` over 0..5 — weights `.5, 1, 1, 1, 1, .5`,
  maximum **5.0** — of Dice and of the lesion-level F1.
* Each correction adds **one** scribble: a short 2-D line on the axial slice holding the
  largest connected error component. It is a *background* scribble if the FP scribble
  would be longer than the FN one, else a *tumor* scribble (ties go to tumor). The
  comparison is on **scribble voxel counts**, not error volume.
* **Lesion-absent cases** never receive a scribble, are **excluded from the detection
  metric** (`metrics.calc_f1` returns NaN, aggregated with a nanmean), and **count for
  AUC-Dice** under the loop's rule "1.0 if the prediction is empty, else 0.0",
  recomputed from a fresh prediction each iteration. One FP voxel therefore costs the
  full **5.0** of AUC-Dice on that case, unfixably. This makes the negative gate (§5.4)
  the single largest lever in the method, by roughly 5× over everything else.
* **Detection metric** uses `cc3d` with **18-connectivity** and matches at **IoU ≥ 0.1**;
  multi-assignment is not punished. Every unmatched predicted component is one false
  positive regardless of size.
* **The `centerline` scribble degenerates**: on a compact error component the skeleton
  collapses and `scribble_centerline` returns the **entire 2-D cross-section** (a sphere
  of radius 6 mm gives a 113-voxel "scribble", not a line). That is partial ground truth
  with known geometry, and §5.2 uses it to calibrate.
* **The simulator is deterministic** (`seed=42` hard-coded, never overridden by the
  loop), so the whole Category-1 scribble sequence is a deterministic function of our own
  predictions and can be replayed exactly offline.
* **Category-2 (clinician) scribbles** were collected in advance against the **baseline
  model's** predictions and are replayed unchanged to every algorithm. They will often
  land on voxels we never predicted, or on lesions we already segment — hence guarantee
  **G3** (§4). Do not assume one scribble per iteration, one slice per scribble, or
  contiguity.
* **Persistence**: the final offline test gives a directory that survives the six calls
  of a case; the preliminary phase has **no persistence** and runs iteration 0 only.
  So the persistence path would otherwise execute for the first time inside the single
  scored submission — which is why this layer is a **pure function** (§4).
* **The empty-error-region bug is absorbing.** `simulate_scribble_from_label` returns a
  2-tuple when its input mask is empty while the loop unpacks three, and the exception is
  raised *before* the scribble json is written and *before* the container is called — so
  the same stale state recurs and every remaining iteration scores 0. A prediction that
  is ever FP-free or ever FN-free can therefore score 0.0 for the whole case. The
  organizers say this will be fixed for `pred == GT`; the `pred ⊊ GT` branch (which is
  what aggressive pruning produces) has not been confirmed. Consequences here: cleanup
  never empties a prediction (§5.3), and the gate fires only on genuine lesion-free
  evidence.

## 2. Files

| File | Contents |
|---|---|
| `config.py` | `PostProcConfig` + four nested dataclasses; every threshold. `from_dict` raises on unknown keys. |
| `utils.py` | point normalisation, mm-aware ball painting, crop/bbox helpers, single-linkage point clustering, `timer`. |
| `constraints.py` | `ConstraintState` (accumulated scribbles, tracer, derived iteration) and `CaseCache` (atomic, failure-tolerant IO). |
| `compliance.py` | `apply_background_scribbles` (G1), `apply_tumor_scribbles` (G2/G3), `apply_all_constraints`, `check_constraints`, `assert_constraints`, `audit_removed_region`. |
| `cleanup.py` | `remove_small_components`, `remove_components_v2`, `recruit_components`, `tracer_suv_floor`, `fill_small_holes`, `cleanup_mask`, `rank_components`, `resolve_v2_rule`. |
| `negative_gate.py` | `negative_gate_features`, `component_stats`, `is_probably_negative`. |
| `tracer_classifier.py` | `guess_tracer`, `tracer_features`, `superior_axis`. |
| `monotone.py` | `blend_with_previous`, `blend_masks`, `revoke_overlap` (all opt-in). |
| `pipeline.py` | `PostProcPredictor` — the orchestration. |
| `tools/oscillation_scan.py` | reports cases whose Dice drops between consecutive iterations; run it against the post-processed and the bare-model runs to separate our instability from the model's. |
| `tools/real_case_check.py` | validation harness: replays the official loop on real evalset cases and scores against GT. |
| `tools/dmm_analysis.py` | builds the component / ground-truth-lesion dataset the detection rules are fitted on, from the cached softmax (CPU only, no network). |
| `tools/dmm_rule_fit.py` | grid-searches the pruning rule and reports its leave-one-out score. |
| `tools/negative_analysis.py` | runs iteration 0 over a whole labelled set with a chosen model and dumps one row per predicted component, plus a sparse replay bundle per case. |
| `tools/gate_sweep.py` | fits and leave-one-out cross-validates negative-gate rules against the expected change in mean AUC-Dice. |
| `tools/gate_replay.py` | replays the shipped gate over those cached iteration-0 outputs and asserts the fire / no-fire counts. |
| `tools/tracer_check.py` | measures the tracer heuristic against the case-name ground truth, with and without the CT. |
| `tests/` | 132 fast tests + 2 timing tests. |

## 3. Coordinate convention

Every array is in **nibabel index space**: `nib.load(f).get_fdata()[i, j, k]`, with `k`
the axial slice. Scribble points are `[i, j, k]` integer lists — exactly what
`simulate_scribble_from_label` emits and what `lesion-clicks.json` stores. `spacing` is
`nib_img.header.get_zooms()`, in the same axis order.

**Nothing in this package ever transposes.** `tests/test_official_scribbles.py` asserts
the round trip against the organizers' own code for all three scribble strategies:
`simulate_scribble_from_label → scribbles_to_gc_format → json → gc_to_swfastedit_format
→ our functions`, and checks `label[c[0], c[1], c[2]] == 1` for every emitted point.

## 4. Guarantees

Let `F` be the accumulated tumor scribble voxels and `B` the accumulated background
scribble voxels, over all iterations of the case.

* **G1** After `apply_background_scribbles`, `mask[b] == 0` for every `b ∈ B`.
* **G2** After `apply_tumor_scribbles`, `mask[f] == 1` for every `f ∈ F`.
* **G3 (safe no-op)** An already-satisfied scribble changes nothing. A tumor scribble
  entirely inside the mask triggers no growth; a background scribble on voxels we do not
  predict deletes nothing. A partially covered tumor scribble is grown **from its missing
  voxels only**. Mandatory for the replayed Category-2 scribbles.
* **G1 ∧ G2** hold together after background-then-tumor (the pipeline order) because the
  growth treats `B` as forbidden. On an exact-voxel conflict, **tumor wins**.
* **Purity** The output is a pure function of `(CT, PET, accumulated scribbles)`. The
  state directory is a *speed* cache (it holds the previous probability so a later
  iteration could skip re-inference), never a correctness requirement. Two tests assert
  this by running the same six-iteration sequence with a state directory and with none,
  and by wiping the directory between every call, comparing masks bytewise. The two
  places that could leak state are both **off by default**: the monotone blend (§5.6)
  and `pass_cached_prev_pred`, which decides whether the *cached* previous mask is handed
  to the base predictor at all — a base that consumes `prev_pred` would otherwise make
  the end-to-end output state-dependent. A `prev_pred` passed explicitly by the caller is
  always forwarded.
* **G4 (bounded perturbation)** A base-mask component that no background scribble points
  at may not lose more than `guard_max_component_removed_fraction` of its volume to
  *compliance*. Cleanup's size/SUV rules are exempt — they are separately gated and
  cannot touch a large confident lesion. `info["base_removed_ml"]` and
  `info["base_added_ml"]` report the perturbation relative to the base mask on every call.
* **Never empty by accident** Emptying is the negative gate's decision alone;
  `min_components_kept` stops cleanup from ever emptying a prediction, and
  `info["empty_without_gate"]` flags it if it somehow happens.
* **Idempotency** `f(f(x)) == f(x)` for both compliance passes, `apply_all_constraints`,
  `remove_small_components`, and the pipeline as a whole — a consequence of G3.
* **Determinism** No RNG anywhere; no dependence on dict or set iteration order
  (components are visited in ascending label order, candidate thresholds in a fixed
  sorted sequence, ties broken by an explicit total order); the tracer is a deterministic
  function of the PET. Necessary because a lesion-free case is re-predicted and re-scored
  six times, and a container that is empty only 60 % of the time scores 2.0 instead of
  5.0 on a case it "usually" gets right.
* **Failure tolerance** A missing, unwritable or corrupt state directory degrades to
  stateless operation rather than raising; `case_cache_dir=None` is fully supported.
  `CaseCache` creates nothing at construction and never raises on write.

The pipeline re-checks G1 and G2 at the end, runs one repair pass if violated, and (with
`strict=True`, the default) raises rather than shipping a non-compliant mask.

## 5. What each stage does

### 5.1 Background compliance — split, don't delete

A background scribble is drawn inside the *error* mask `pred & ~gt`, which is a **subset
of a predicted component**. Two very different situations produce one: a genuine
false-positive component (delete it), or a true positive that has bled into surrounding
tissue (deleting it costs a true positive — Dice falls and the detection metric gains a
false negative). The `boundary` strategy makes this acute: it walks the *inner rim* of
the error component, often within a voxel or two of the true lesion boundary.

So the rule is, per background scribble voxel:

1. **not inside our mask → no-op** (G3);
2. otherwise take its 18-connected component `C` and compute a **core** —
   `{prob > bg_core_prob} & {SUV > SUV_floor}`, or, with no softmax,
   `{SUV ≥ max(bg_core_suv_ratio · SUVmax(C), bg_core_min_suv_ratio · SUVmax(scribble))}`.
   Without a confidence map that second term is the only signal that separates "a true
   positive that bled into cool tissue" (scribble in the cool part, core far hotter →
   split) from "a homogeneous false positive" (scribble as hot as anything → no core →
   delete). The core is shrunk away from the scribble by `bg_core_margin_mm`;
3. **core empty → delete `C`**;
4. **core non-empty → split**: watershed on `-prob` (or `-PET`) over `C`, background
   voxels as one marker and the core as the other; delete only the background basin.

The raw basin is not safe to use as-is. When the model is confident and roughly uniform
over a well-segmented lesion, `-prob` is a *flat* landscape, the watershed degenerates
into a distance partition, and a scribble on the rim would be handed every voxel closer
to it than to the core — most of the lesion. So the basin is bounded three ways: it may
not contain a strictly-confident voxel (`bg_protect_confident_core`); it must lie within
`bg_remove_max_radius_mm` of the scribble, measured *geodesically inside the component*;
and it may not exceed `bg_split_max_removed_fraction` of `C`. If that last check fails
the fallback is the scribble's own neighbourhood (`bg_local_fallback_radius_mm`) —
**not** whole-component deletion, which would invert the intent. Deleting `C` whole
stays reserved for "no core at all", and a split that would leave less than
`bg_split_min_kept_volume_ml` also deletes it whole.

Removal is therefore local and convergent rather than one-shot: a long over-segmented
tail is trimmed near each scribble and the evaluator simply scribbles again.

A component containing an earlier **tumor** scribble can never be deleted whole: its
tumor points are unconditionally part of the core.

`audit_removed_region(removed, gt, ...)` (and `apply_background_scribbles(..., gt=...)`)
reports how many GT lesions lost more than half their volume to a deletion. On a
validation split that number must be near zero; if it is not, the rule is too aggressive.
`tools/real_case_check.py` prints it per run.

### 5.2 Foreground compliance — calibrate, then grow

G3 first: a cluster entirely inside the mask is skipped; a partially covered cluster is
seeded from its *missing* voxels only.

**Threshold calibration.** When the scribble footprint is thick enough to carry area
information (inner radius ≥ `fg_footprint_min_thickness_mm`), it is the exact in-plane
extent of the missed lesion on that slice. The SUV threshold `T` is then chosen to
**maximise the IoU** between the grown region restricted to that slice and the footprint,
over candidate thresholds drawn from the SUV distribution on the footprint
(`fg_calibration_percentiles`) plus the 41 %-of-peak value. This is a free,
self-calibrating, per-lesion threshold and it is strictly better than a fixed rule where
it applies. If the footprint is a thin line (`boundary` / `random`, or a small error) it
carries no area information and we fall back to
`T = max(fg_alpha · SUVpeak_local, SUV_floor[tracer])`, the clinical 41 %-of-peak rule.

`SUVpeak_local` is the maximum of a ~1 mL separable box mean taken over the **whole
scribble cluster**, not just the voxels being grown from (`fg_peak_over_whole_cluster`).
A stroke annotates one lesion, so its peak is a property of the whole stroke; measuring
it on a cold tail whose hot end is already segmented collapses `T` onto the tracer floor.

**Low-contrast fallback.** If `fg_alpha ·` (raw peak) does not even clear the tracer
floor, the 41 % rule has no information and the floor would be doing the thresholding —
but the floor is a "cannot possibly be lesion" bound, not a delineation criterion. In
FDG, SUV ≥ 1.5 selects most soft tissue, so thresholding there grows into normal anatomy.
Growth then prefers the model's own connected evidence (`{prob ≥ fg_prob_include}`
through the scribble, if it fits the volume cap) and otherwise falls back to a bounded
ball (`fg_low_contrast_fallback`). The decision uses the *raw* maximum rather than the
box mean, because the box mean dilutes a small lesion towards its surroundings and would
call a genuine 6 mm node low-contrast.

Calibration uses the slice holding most of the voxels being grown from. Accumulated
scribbles from different iterations land on different slices and cluster together, and
abandoning calibration whenever a cluster spans more than one slice silently drops the
threshold back onto the alpha rule on alternate iterations — which is how a small PSMA
lesion came to oscillate 0.914 → 0.414 → 0.901 → 0.225 across a run.

Every one of these decisions is reported per cluster in
`last_info["fg_compliance"]["clusters"]` — `calibrated`, `calibration_iou`,
`footprint_thickness_mm`, `threshold`, `alpha_threshold`, `raw_peak`, `low_contrast`,
`fallback`, `reference_ml`, `max_volume_ml`, `added_ml`. A rule that silently changes
policy is the hard kind of defect to find; these fields are what made both of the
oscillation causes obvious in one replay.

**The rule, stated once.** Growth from a scribble is a *local, evidence-bounded* edit:

1. the threshold comes from the **peak of the whole stroke**, never from the tracer
   floor and never from a cold sub-segment of the stroke;
2. if no threshold can be derived from intensity, we do not threshold at all — we take
   the model's connected evidence, or a bounded ball;
3. calibration is used whenever the footprint carries area information, on the slice
   that actually holds the voxels being grown from;
4. what a single scribble may add is capped in absolute *and* relative terms, and the
   cap is measured on the **added** volume;
5. a scribble that is already satisfied adds nothing at all (G3).

Together these bound the perturbation a single correction can make. That matters beyond
this layer: the interactive model conditions on its previous prediction, so an
over-grown region is fed straight back in and the case oscillates for the rest of the
run. Two defects violating rules 1 and 3 each produced exactly that, and the trajectories
are in §8.

**Bounding.** The candidate set `{SUV ≥ T}` (∪ `{prob ≥ fg_prob_include}`) is restricted
to `CT ≥ fg_exclude_ct_below_hu`, away from `forbidden` (background constraints), and
away from every predicted component holding no point of this cluster **dilated by one
voxel**, so the union cannot even become 18-connected to it — merging two lesions costs a
detection. Then:

* **geodesic** distance from the seed *inside the candidate set*
  (`skimage.graph.MCP_Geometric`, spacing-aware) bounds the growth at
  `fg_max_radius_mm`, not euclidean distance — a lesion 30 mm away around a corner is
  not 30 mm away;
* **stop at local SUV minima**: competing `h_maxima` (depth `fg_valley_depth_suv`, at
  least `fg_valley_min_distance_mm` away, at least `fg_valley_peak_ratio` as hot) become
  watershed markers and only the scribble's basin is kept. A homogeneous lesion has a
  single plateau containing the seed, so nothing competes and the region is unchanged;
* the volume cap is both absolute (`fg_max_volume_ml`) and relative
  (`fg_max_relative_growth` × the evidence the scribble starts from, floored at
  `fg_min_growth_ml`): 100 mL is a sane ceiling for a bulky lesion and an absurd one for
  a 2 mL node. It is applied to the **added** volume, not the total region, so it
  measures what the scribble actually contributes, and `T` is escalated through
  `fg_threshold_escalation` until it fits;
* a fully sub-threshold (photopenic) scribble additionally gets a
  `fg_fallback_ball_radius_mm` ball, so it still has a chance of reaching IoU ≥ 0.1.

`method="random_walker"` replaces the growth with `skimage.segmentation.random_walker`
on a `rw_crop_size` cube, scribble voxels as foreground Dirichlet seeds and far-away /
sub-floor / forbidden voxels as background seeds. Seeds are satisfied by construction
there too. It is ~3× slower and produces smoother boundaries; the region grow is the
default.

### 5.2b Skipping inference when there is nothing to correct

`skip_inference_if_satisfied` (default on). At any iteration after the first, the layer
compares the newly arrived scribble points against the mask it returned last time. If
every new tumor point is already inside that mask and every new background point is
already outside it, the scribble asks for no change: the previous mask is returned
unchanged and **the network is not called**. Three conditions guard it: at least one new
point must have arrived (so it is a statement about corrections we have already made,
not a general "reuse the last answer" shortcut); the previous mask must be **non-empty**,
because an empty mask satisfies every background point vacuously and skipping there would
freeze an empty prediction for another iteration; and the state directory and previous
mask must exist, otherwise it falls through to the normal path.

This exists for the challenge's Category-2 scribbles, which were collected against the
*baseline's* errors and are replayed to us unchanged, so a large fraction of them already
hold for our mask. Feeding one to the network anyway re-runs it with new guidance
channels and a new previous-mask channel, and the model revises its output *globally* in
response to a correction that asked for nothing — which is how the shipping pipeline lost
AUC on replayed scribbles while a lighter post-processing configuration did not.

Measured on the 30-case Category-2 replay (baseline scribbles replayed to the shipped
pipeline): the rule fires on **26 of 150 post-iteration steps** and moves the metric by
**+0.002 AUC-Dice / +0.001 AUC-DMM** — 3 cases better, 27 unchanged, none worse. It is
therefore safe and saves about a sixth of the inference calls, but it does **not** explain
or recover the Category-2 deficit; that gap has another cause. Without the empty-mask
guard the same measurement was −0.039 / −0.057, all of it from one case where an empty
prediction was frozen for an extra iteration.

Under simulated (Category-1) interaction the rule is **inert by construction**: the
evaluator draws a tumor scribble on `~pred & gt` and a background scribble on
`pred & ~gt`, both computed from the mask we just returned, so a newly arrived simulated
scribble can never be already satisfied. It is the one place where the layer deliberately
depends on the state directory; the purity property of §4 therefore holds with this flag
off, and the flag degrades safely to the normal path when no state is available.

### 5.3 Cleanup

Order: tracer SUV floor → small components → hole filling. Runs **before** compliance, so
it can never prune something a scribble just asked for.

* `tracer_suv_floor` — `"component"` (default) drops a component whose SUVmax is below
  the tracer floor (1.5 FDG / 1.0 PSMA); `"voxel"` also erases sub-floor voxels (sharper
  Dice, but fragmentation costs detections); `"off"`.
* `remove_small_components(min_volume_ml=0.3, suv_gate=4.0)` — drops components under
  0.3 mL unless SUVmax reaches 4.0. Every unmatched component is one detection false
  positive regardless of size, so pruning is cheap; but IoU ≥ 0.1 is loose, so
  small-lesion *recall* must never be traded away.
* `fill_small_holes` — background cavities (6-connectivity, the dual of an 18-connected
  foreground) inside the mask bbox; never fills a background scribble voxel.

**`min_components_kept` (default 1)**: if a pass would remove every component, the best
`min_components_kept` are kept, ranked by `(SUVmax, voxel count, label)` — a total order,
so the choice is deterministic. Aggressive pruning is exactly the operation that turns a
prediction into a strict subset of the GT, which is the trigger for the absorbing bug in
§1; emptying is the gate's decision alone.

#### 5.3.1 Rule v2 — a union of coarse conjunctions (`cleanup.rule_v2`, off by default)

`remove_components_v2` prunes a component that matches **any** rule in `v2_rules`; one
rule fires only when **every** enabled criterion of that rule holds:

```
rule fires  <=>  volume_ml < min_volume_ml  AND  suv_max < suv_gate  AND  prob_mean < prob_gate
prune       <=>  any rule fires
```

`None` disables a criterion, and `v2_rules = None` falls back to the single conjunction
`(v2_min_volume_ml, v2_suv_gate, v2_prob_gate)`, so `(0.3, 4.0, None)` *is*
`remove_small_components` — the two are the same function, and `tests/test_cleanup.py`
asserts it voxel for voxel. Each disjunct is still a conjunction of at most three
"unimpressive" criteria, so adding a criterion to a disjunct can only ever *save* a
component; the union exists because the false positives worth deleting are two disjoint
populations (§5.3.4).

**The measured setting is** `v2_rules = [{"suv_gate": 5.0}, {"prob_gate": 0.8}]` — delete
a component whose SUVmax is under 5, or whose mean foreground softmax is under 0.8 —
applied as a **second pass after compliance** (`cleanup_after_compliance`).

The third axis is the **mean** foreground softmax over the component, not the max. A
predicted voxel is the argmax of a two-class softmax, so `prob_max ≥ 0.5` on every
component and is ~1.0 on almost all of them; the mean is what separates a lesion the
network is sure about from a component it only just emitted. Without a softmax the
criterion cannot be evaluated and the pass **fails closed** — it refuses to delete on
account of it — so a base predictor that returns no probabilities silently gets v1 rather
than a rule it was never fitted for.

`v2_by_tracer` overrides any of the thresholds, or the whole `v2_rules` list, per
tracer (`{"fdg": {"v2_rules": [...]}}`). It ships unset: leave-one-out says one pooled
rule beats two fitted ones (§5.3.4).

**"Silence is evidence"** (`v2_silence_decay`, 1.0 = off). The evaluator points at our
*largest* error every iteration, deterministically. A component that has survived `k`
rounds without a background scribble landing in it is therefore progressively more likely
to be a true positive, so its size threshold is multiplied by `v2_silence_decay ** k`
(`< 1` protects survivors, `> 1` prunes them harder); a component that *has* been
scribbled keeps the undecayed threshold, because it is demonstrably wrong somewhere.
`v2_silence_requires_bg` (default True) switches the decay off until the evaluator has
actually produced a background scribble, so under challenge category 2 — where the
scribbles were recorded against a different algorithm and need not follow our errors —
the rule degrades to plain v2 instead of trusting a signal that is not there.

#### 5.3.2 Recall recruitment (`cleanup.recruit_prob_threshold`, off by default)

Matching needs only IoU ≥ 0.1, so a component that covers a tenth of a lesion scores a
full true positive: in principle recall is cheap and every pruning rule pushes the other
way. `recruit_components` re-binarises the softmax at `recruit_prob_threshold` and adds
the components that touch **no** voxel of the argmax mask and clear
`recruit_min_suv_max` / `recruit_min_volume_ml`, capped at `recruit_max_components`. It
is additive by construction — the argmax mask, and with it the lesion-free gate's
decision and every existing component, is untouched — because a lower global threshold on
a lesion-free case would cost that case its whole AUC-Dice.

Measured result: **this lever is empty on both models** (§5.3.4). It ships off.

#### 5.3.3 Component merging (`cleanup.bridge_closing_voxels`, off by default)

Multi-assignment is not punished: one predicted component may match several ground-truth
lesions and score them all. Merging two components is therefore at worst neutral and
usually a strict win — two unmatched components become **one** false positive instead of
two, and an unmatched component absorbed into a matched one stops being a false positive
at all. `bridge_components` closes the mask by `bridge_closing_voxels` voxels (an
26-connected structuring element, so it joins gaps of up to twice that), adding only the
bridge voxels. The failure mode is the union's IoU with a small lesion dropping below
0.1, which is why the useful setting is one voxel and the sweep stops at two.
`bridge_max_added_ml` refuses the whole closing if it would add more than that, and the
`forbidden` set (background scribble voxels and the region a background scribble just
deleted) is never filled, so G1 survives the pass.

Merging is a **pipeline stage of its own, after compliance**, not part of `cleanup_mask`:
a background scribble splitting a component again would undo it, and the pruning pass
that follows should see the merged components, not the fragments.

#### 5.3.4 What was measured

Fitted on the iteration-0..5 component tables that `tools/dmm_analysis.py` rebuilds from
the cached softmax of an A3 run (33 positive cases + 2 carried over, 18 FDG / 17 PSMA).
The objective is the per-case AUC of the lesion-level F1 over the six iterations — the
detection metric itself, not iteration 0, which carries only 0.5 of the 5.0 weight and
whose prediction is empty on a third of the positives. The offline scores reproduce
`metrics.MetricEvaluator` **exactly** on all 33 × 6 scored iterations (F1 and Dice), so
the only approximation is that the scribble sequence was the one A3 produced.

Control (A3): AUC-F1 **3.2665**, AUC-Dice (positives) 3.6737; FDG 2.6617 / PSMA 3.9069.

| rule (applied after compliance) | ΔAUC-F1 | ΔAUC-Dice | FDG | PSMA | better / same / worse | worst case |
|---|---|---|---|---|---|---|
| v1 `vol<0.3 & SUVmax<4` | +0.0000 | +0.0000 | +0.000 | +0.000 | 0 / 35 / 0 | — |
| `SUVmax < 4` | +0.0431 | +0.0128 | +0.005 | +0.084 | 9 / 24 / 2 | −0.30 |
| `SUVmax < 5` | +0.0948 | +0.0274 | +0.064 | +0.127 | 17 / 10 / 8 | −0.85 |
| `SUVmax < 6` | +0.0525 | +0.0372 | +0.043 | +0.063 | 14 / 9 / 12 | −1.30 |
| `prob_mean < 0.75` | +0.0476 | +0.0006 | +0.055 | +0.040 | 20 / 13 / 2 | −0.07 |
| `prob_mean < 0.8` | +0.0555 | +0.0006 | +0.074 | +0.037 | 19 / 11 / 5 | −0.11 |
| `prob_mean < 0.9` | +0.0289 | +0.0074 | −0.068 | +0.052 | — | — |
| `SUVmax<5 & prob_mean<0.9` | +0.0655 | +0.0118 | +0.021 | +0.112 | 19 / 11 / 5 | −0.39 |
| **`SUVmax<5` or `prob_mean<0.8`** | **+0.1292** | **+0.0269** | +0.121 | +0.138 | 19 / 8 / 8 | −0.83 |
| `SUVmax<4` or `prob_mean<0.8` | +0.0972 | +0.0130 | +0.076 | +0.120 | 20 / 9 / 6 | −0.28 |

Three things decided the shipped rule.

1. **v1 has nothing left to prune.** By the time compliance has run, no component is
   both under 0.3 mL and under SUVmax 4 — the rule fires zero times on all 210 scored
   iterations. Its +0.079 AUC-DMM was earned before this stage, on the raw model output.
2. **"Cold" and "unconfident" are disjoint populations.** A single conjunction keeps most
   of each; the union is almost exactly additive (+0.095 and +0.056 alone, +0.129
   together), which is why `v2_rules` is a union rather than a wider conjunction.
3. **One rule for both tracers beats two.** Leave-one-out over the 35 cases picks
   `SUVmax<5 or prob_mean<0.8` in **every** fold (one distinct rule in 35 folds), for a
   LOO ΔAUC-F1 of +0.1292; fitting per tracer halves the data and does worse on both
   (FDG +0.085, PSMA +0.083 against +0.121 / +0.138 for the pooled rule). So
   `v2_by_tracer` ships unset.

**Dice is not the constraint here** — the rule *raises* AUC-Dice by +0.027 on positives,
because most of what it deletes is false-positive volume. The risk it carries is a tail:
8 of 35 cases lose something and the worst loses 0.83 of AUC-F1 (a PSMA case whose small
lesions all sit under SUVmax 5). `SUVmax<4 or prob_mean<0.8` is the conservative
alternative — three quarters of the gain, a quarter of the tail.

**Measured end to end.** The rule was then run through the full six-iteration loop on the
same 100 cases, the same strategy assignment, the same seed and the same prediction cache
as the A3 control (`--postproc_set cleanup.rule_v2=true
'cleanup.v2_rules=[{"suv_gate":5.0},{"prob_gate":0.8}]' cleanup_after_compliance=true`):

| | A0 | A3 | **A9 (rule v2)** | A9 − A3 |
|---|---|---|---|---|
| AUC-Dice (100 cases) | 3.6351 | 3.6388 | **3.6554** | **+0.0166** |
| AUC-DMM (64 positives) | 3.2216 | 3.3007 | **3.4148** | **+0.1140** |
| FDG positives AUC-DMM | 2.6069 | 2.6445 | **2.7991** | **+0.1546** |
| PSMA positives AUC-DMM | 3.7991 | 3.9171 | **3.9931** | +0.0760 |
| lesion-free AUC-Dice | 3.6111 | 3.6111 | 3.6111 | 0.0000 |

33 of 64 positives improve, 15 are unchanged, 16 lose; worst −0.831, best +1.511. All 600
scored iterations satisfy G1 and G2. The cleanup stage costs 0.76 s mean / 1.72 s p95 per
iteration (A3: 0.58 / 1.33) against a 600–1200 s budget.

On the **fine-tuned** model the same rule is worth more than twice as much, and it buys
Dice as well, because that model's iteration-0 output is a real prediction rather than an
empty mask — there is far more false-positive volume for the rule to act on:

| | B3 (same post-processing, no rule v2) | **B9 (+ rule v2)** | B9 − B3 |
|---|---|---|---|
| AUC-Dice (100 cases) | 3.3242 | **3.4689** | **+0.1447** |
| AUC-DMM (64 positives) | 3.5635 | **3.8243** | **+0.2608** |
| FDG positives AUC-DMM | 3.2230 | **3.3645** | +0.1415 |
| PSMA positives AUC-DMM | 3.8833 | **4.2562** | +0.3729 |
| lesion-free AUC-Dice | 2.7778 | 2.7778 | 0.0000 |

38 of 64 positives improve, 6 are unchanged, 20 lose; worst −1.103, best +3.300. PSMA,
which barely moved on the baseline (+0.076), is the biggest winner here (+0.373).

The offline estimate was accurate on FDG (+0.121 predicted, +0.155 measured) and
optimistic on PSMA (+0.138 predicted, +0.076 measured): on PSMA the rule moves the mask
enough to move the scribble sequence, which a fixed-trajectory estimate cannot see.

**Recall recruitment is dead, and the mechanism is now known.** Of 191 ground-truth
lesions the baseline misses at iteration 0, **176 have a maximum foreground softmax of
exactly zero inside them** — there is no probability mass to recruit, at any threshold.
Sweeping 0.5 → 0.4 → 0.3 → 0.2 → 0.1 → 0.05 → 0.02 moves the mean iteration-0 F1 from
0.0994 to 0.1081 at best (threshold 0.1) and down to 0.0887 at 0.02, because every
threshold that adds a true positive adds more false positives. On the fine-tuned model
it is worse than useless: detections *fall* from 1007 to 1004 to 991 as the threshold
goes 0.5 → 0.3 → 0.1, because the lowered threshold merges components until their IoU
with the individual lesion drops under 0.1.

**Component merging was measured and rejected.** Closing the mask by one voxel removes
251 of 1613 false positives but also 206 of 3702 true positives: ΔAUC-F1 +0.004,
ΔAUC-Dice −0.014. Two voxels is clearly worse (−0.133). The mechanic is real — merges
are not punished — but at this operating point the components that merge are mostly
already matched. `bridge_closing_voxels` therefore ships at 0.

### 5.4 Negative gate — the dominant lever

Fires only when **all enabled criteria** hold: no scribble of either kind has ever
arrived (`require_no_scribbles`), and each threshold that is not `None` passes. Then the
mask is replaced with all-zeros. **`None` disables a criterion**, and the shipped
configuration enables exactly one:

```
total predicted volume < max_total_volume_ml (6.0 mL)
```

#### Why the first version never fired

Measured on the 100-case validation set (36 lesion-free), with `blocked_by` printed for
every lesion-free case whose iteration-0 prediction was non-empty:

| case | tracer | pred. volume | largest comp. | SUVmax | max softmax | v1 `blocked_by` |
|---|---|---:|---:|---:|---:|---|
| `psma_28f9ecc106…` | psma | 1.002 mL | 1.002 | 21.3 | 1.000 | `volume, largest_component, suv, prob` |
| `fdg_b1de3d4248…` | fdg | 0.734 | 0.734 | 96.8 | 1.000 | `largest_component, suv, prob` |
| `fdg_34aa521b46…` | fdg | 0.199 | 0.199 | 68.9 | 1.000 | `suv, prob` |
| `fdg_572fca6b44…` | fdg | 0.124 | 0.124 | 15.0 | 0.997 | `suv, prob` |
| `fdg_69ee62b035…` | fdg | 0.075 | 0.062 | 43.4 | 0.997 | `suv, prob` |
| `fdg_f0a1a38a3b…` | fdg | 0.062 | 0.062 | 18.9 | 0.975 | `suv, prob` |
| `fdg_a37b4bca43…` | fdg | 0.025 | 0.025 | 18.5 | 0.662 | `suv, prob` |
| `fdg_6e1dba94e8…` | fdg | 0.012 | 0.012 | 20.0 | 0.811 | `suv, prob` |
| `fdg_b3e923029c…` | fdg | 0.012 | 0.012 | 45.6 | 0.679 | `suv, prob` |
| `fdg_ca62984a81…` | fdg | 0.012 | 0.012 | 16.0 | 0.668 | `suv, prob` |

Two structural mistakes, both fatal on their own:

* **`max_prob = 0.60` can never pass.** A predicted voxel *is* the argmax of a two-class
  softmax, so `prob_max_in_mask ≥ 0.5` on every non-empty prediction and is ≥ 0.66 on all
  ten of these. Sweeping it over 0.45/0.60/0.75/0.90 could not have changed anything.
* **`max_suv = 3.0` is the wrong sign.** The false positives that survive on a lesion-free
  case sit on *physiologically hot* tissue — ureter, bowel, injection site, bladder rim —
  so their SUVmax (15–97, median 20.6) is high precisely when they are false. True lesion
  components on positive cases have a *lower* median SUVmax (10.9).

What is left after removing both is the one thing that does separate the classes: these
false positives are **1–41 voxels, 0.012–1.002 mL**, while the positives that carry any
Dice at iteration 0 are two orders of magnitude larger.

#### The measured trade-off

Emptying a lesion-free case whose prediction was non-empty is worth **+5.0** AUC-Dice.
Emptying a lesion-present case costs only iteration 0 — the whole error region is then
under-segmentation, so `fp ≤ fn` and iteration 1 delivers a *tumor* scribble that un-gates
the case permanently — i.e. **−0.5 × Dice@0**, and exactly 0 for a positive the model
already predicted empty. With the validation set's lesion-free fraction `f = 0.36`:

```
Δ mean AUC-Dice = ( 5.0 · #negatives_rescued − 0.5 · Σ Dice@0(emptied positives) ) / 100
```

Iteration-0 output of both networks, one row per candidate threshold
(`postproc/tools/negative_analysis.py` → `postproc/tools/gate_sweep.py`):

| `max_total_volume_ml` | A0 baseline: neg/36 · pos emptied · Δ | B0 fine-tuned: neg/36 · pos emptied · Δ |
|---:|---|---|
| 1.05 | 10 · 7 · **+0.4952** | 6 · 8 · +0.2896 |
| 2.0 | 10 · 8 · +0.4952 | 9 · 12 · +0.4300 |
| 3.0 | 10 · 9 · +0.4951 | 11 · 13 · +0.5278 |
| 5.0 | 10 · 11 · +0.4918 | 13 · 17 · +0.6153 |
| **6.0 (shipped)** | **10 · 11 · +0.4918** | **14 · 17 · +0.6653** |
| 8.0 | 10 · 11 · +0.4918 | 14 · 18 · +0.6620 |
| 12.0 | 10 · 13 · +0.4848 | 14 · 20 · +0.6551 |
| 25.0 | 10 · 17 · +0.4774 | 15 · 23 · **+0.6970** |
| 55.0 | 10 · 19 · +0.4726 | 16 · 30 · +0.7260 |

Headroom is 0.500 (A0: 10 non-empty negatives) and 0.800 (B0: 16). "pos emptied" counts
only positives whose prediction was **not already empty**; emptying an already-empty
prediction is a no-op.

Other rule shapes, leave-one-out cross-validated (threshold re-fitted on the other 99
cases each fold), on the baseline:

| rule | LOO neg/36 | LOO pos | LOO Δ |
|---|---:|---:|---:|
| total volume < V | 9 | 8 | **+0.4452** |
| confidence-weighted volume < Q | 9 | 8 | +0.4452 |
| per-tracer total volume < V[tracer] | 8 | 5 | +0.3984 |
| logistic regression on 15 case features | 8 | 22 | +0.3968 |
| volume < V **and** mean softmax < P | 8 | 25 | +0.3952 |
| mean softmax over the mask < P | 8 | 27 | +0.3939 |
| max softmax < P | 7 | 24 | +0.3483 |
| decision tree (depth 2 / 3) | 7 | 25 | +0.3453 |
| SUVmax of the largest component < S[tracer] | 8 | 48 | +0.3371 |
| SUVmax in mask < S | 9 | 63 | +0.3338 |

The volume rule wins on both networks; nothing learned beats it, and every criterion
added to it costs more lesion-free cases than the positives it spares. The **threshold**
is deliberately *not* the fitted optimum: that lands at 1.05 mL, 0.05 mL above the largest
lesion-free false positive, and leave-one-out drops a case whenever that edge case is held
out. 6.0 mL sits on the flat top of the curve — within 0.7 % of A0's optimum, 92 % of B0's
— with six times the margin over the largest false positive A0 leaves on a negative.

#### What it costs on positives

At 6.0 mL the baseline empties 11 positives at iteration 0, of which 8 had `Dice@0 = 0`
anyway; the entire cost is two cases (`fdg_91cfa804b0`, Dice@0 0.620; `psma_2ebe8e333…`,
0.316) plus rounding — **Σ Dice@0 = 1.64, i.e. 0.82 of 500 AUC points**. On the fine-tuned
network the same threshold empties 17 positives with Σ Dice@0 = 6.95 (3.47 AUC points),
because that network's iteration-0 predictions are much better; the gain still outweighs
it 20:1, but the exposure is real and is the reason 3.0 mL is documented as the
conservative alternative (B0 +0.5278, half as many positives touched).

#### Design points that did not change

* the guard is **"no scribbles seen"**, not "iteration 0". On a lesion-free case no
  scribble ever arrives, so the gate keeps firing at all six iterations — which is what it
  takes to hold Dice at 1.0 throughout. It is also recoverable from the input alone,
  unlike an iteration index. `only_iteration_zero` exists but is **off** by default.
  `tools/gate_replay.py` asserts iteration-stability on all 200 cached outputs.
* **a tumor scribble disables the gate permanently** and we then emit the **full model
  prediction**, not merely a region grown around the seed — recovering one lesion out of
  many would waste the correction. `blocked_by == ["tumor_scribble_proves_positive"]`
  records it.
* the gate **cannot hurt the detection metric at all** (lesion-absent cases are excluded
  from it), so the whole risk is Dice: 5.0 vs 0.0 per negative case, against the bounded
  cost above — bounded because the next iteration then delivers a tumor scribble. That
  bound depends on the §1 empty-error behaviour being the deployed one.
* the shipped rule is **tracer-agnostic on purpose**. A per-tracer threshold scored worse
  under cross-validation, and a tracer-agnostic gate cannot lose a lesion-free case to a
  tracer misclassification. `max_suv_by_tracer` exists for a sweep.

`is_probably_negative` returns `blocked_by` — which criteria vetoed the decision — and
`negative_gate_features(..., with_components=True)` adds `component_stats`: per component
volume, SUV max/mean, softmax max/mean, centroid, `z_frac` (position along the superior
axis normalised to the CT body extent) and `shell_suv_max` (hottest PET voxel in a box
around but not inside the component). Those are the features a component-level veto would
use; see §10.12 for why one is not shipped.

### 5.5 Tracer classifier

`guess_tracer(pet, spacing, ct=..., affine=...)` returns `"fdg"` / `"psma"` from a single
score:

```
s = log10( head_SUV_p99 / trunk_SUV_p999 )  +  0.26 · log10( head_hot_blob_mL / 100 )
FDG  ⇔  s > −0.63
```

`head` is the top 15 % of the **body** along the superior axis, `trunk` is 15–75 % of it,
and `head_hot_blob_mL` is the largest contiguous SUV ≥ 4 component in the head slab. The
physiology: FDG makes the brain the hottest *large* structure in the body; PSMA leaves it
dark and puts the extreme uptake in the kidneys and bladder, with only small parotid and
submandibular glands in the head slab. Both terms are ratios, so the answer does not
depend on the scanner's SUV calibration.

Measured on all 100 evaluation cases (63 FDG / 37 PSMA, ground truth = case-name prefix,
`postproc/tools/tracer_check.py`): **100/100**, minimum confidence 0.719, no case below
0.60. The classes are separated by a factor of **1.60** in the score. The head-to-trunk
ratio alone also reaches 100/100 but with a margin of 1.05 — one case's width — which is
why the blob-size term is carried: it is what tells a 60 mL pair of parotids from a 700 mL
brain.

The previous two-rule version scored **86/100**, with all 14 errors PSMA→FDG: its brain
test needed a ≥ 150 mL hot head blob (PSMA salivary glands reach 44–132 mL, so it never
fired), its PSMA test needed a body SUV p99.9 ≥ 25 (real PSMA trunks measure 6–28), and
the ratio fallback then defaulted to FDG.

Two implementation details that mattered:

* the slabs are measured against the **body extent**, not the array extent — from the CT
  (`> −500 HU`) when it is passed, from the PET (`SUV > 0.2`) otherwise. A scan with empty
  slices above the vertex has no "top 15 % of the array" that is a head. Both paths score
  100/100.
* the superior direction comes from `nibabel.aff2axcodes(affine)` when given, else `+k` is
  assumed superior; `superior=(axis, sign)` overrides.

`confidence` is 0.5 on the boundary and 0.95 one measured margin away, so a caller can
treat < 0.6 as "unknown tracer". Deliberately a scored heuristic rather than a second
network: the container runs with `--network=none`, so shipping a classifier means baking
in weights. Set `config.tracer = "fdg"` / `"psma"` to bypass it entirely. Known failure
modes: fields of view that exclude the head, PSMA with brain metastases, other tracers.
Nothing in the shipped negative gate is tracer-conditional, so a misclassification cannot
cost a lesion-free case.

### 5.6 Monotone blending — opt-in, and revocable

| mode | behaviour |
|---|---|
| `none` *(default)* | pass through |
| `minmax` | `max(new, prev)` in the fg-constraint region, `min(new, prev)` in the bg-constraint region, `new` elsewhere |
| `ema` | `α·new + (1−α)·prev` everywhere |
| `ema_minmax` | EMA, then the clamps |

**Default `none` on purpose.** Any other mode makes the output depend on the cached
probability of the previous call, which breaks the purity guarantee — and purity is worth
more than anti-oscillation damping, because the state directory may not exist and its
behaviour cannot be tested before the single final submission. Treat the modes as
ablation knobs.

**Constraints are revocable.** `revoke_overlap` drops the intersection from both
constraint regions, so a later background scribble next to an earlier tumor scribble
un-freezes that region and fresh model evidence decides there. This is order-free, which
matters because the input json does not preserve interleaved arrival order across the two
classes. The blend also runs **last**, after compliance, and the guarantees are re-imposed
afterwards, so a clamp can never override a scribble.

### 5.7 Optional: scribble-calibrated global rescue

`rescue_threshold_from_scribble` (default **off**). A tumor scribble is placed on the
*largest* missed component, so it proves a lesion of that conspicuity was missed —
lesions of similar conspicuity elsewhere probably were too. The knob lowers the global
binarisation threshold to the `rescue_percentile` of the model probability over the
scribble voxels (floored at `rescue_min_threshold`), re-runs component analysis, and
unions the result. Off by default because it turns a local hint into a global change.

### 5.8 Pipeline order

```
rebuild the constraint set from the accumulated scribble json
  -> base.predict(...)                       (mask [+ softmax])
  -> negative gate      (only while no scribble has ever arrived)
  -> cleanup            (SUV floor, pruning, recruitment, holes; never empties)
  -> background compliance (G1: split, or delete when there is no core)
  -> foreground compliance (G2/G3: grow only what is missing)
  -> optional component merging                     (ablation knob)
  -> optional second cleanup pass                   (ablation knob)
  -> optional scribble-calibrated global rescue     (ablation knob)
  -> optional monotone blend against the cached probability (ablation knob)
  -> re-assert G1 and G2
  -> save the speed cache
```

**`cleanup_after_compliance`** (off by default) runs the cleanup stage a second time,
after both compliance passes. Cleanup normally sees only the raw model output, so a
component the interaction layer itself creates — the fragment left behind when a
background scribble splits a component, a region grown from a tumor scribble — is never
checked against the pruning rule, and every one of those that misses its lesion is a full
detection false positive. The second pass can only delete; deletion cannot break G1,
components holding a tumor scribble are protected so G2/G3 hold, and
`min_components_kept` keeps the mask non-empty.

It is where the whole of rule v2's measured gain lives: applied to the raw model output
the v1 rule is worth +0.079 AUC-DMM (rung A3), but on the mask that reaches the scorer it
fires **zero times** in 210 scored iterations, because nothing that survives compliance is
both under 0.3 mL and under SUVmax 4 (§5.3.4).

The iteration index is derived from the scribbles (one scribble event ≈ one spatial
cluster), never from a stored counter, and **nothing keys on it** — under Category 2 it
degrades to "unknown" rather than mis-indexing. `last_info` reports `iteration`,
`iteration_source`, `iteration_cached`, `n_calls`, `state_available`, per-stage timings,
the gate features, compliance counters, cleanup counters, `empty_output` and
`empty_without_gate`.

`PostProcPredictor` introspects `base.predict` once and passes only the keyword arguments
it accepts, requesting `return_probabilities=True` only when the signature has it.

### 5.9 State directory (and its absence)

```
<state_dir>/postproc_constraints.json   points, tracer, derived index, history
<state_dir>/postproc_prev_prob.npy      lesion probability, uint8-quantised (~53 MB)
<state_dir>/postproc_prev_mask.npz      bit-packed previous mask (~50 kB compressed)
<state_dir>/postproc_bg_region.npz      bit-packed union of everything ever deleted
```

All writes are atomic (temp file + `os.replace`) and never raise. Directories are created
lazily, on first write. With no state directory the constraint set is still complete —
`lesion-clicks.json` carries every scribble collected so far and
`ConstraintState.from_scribbles` rebuilds from it. The probability is stored uncompressed
uint8 on purpose: 1/255 quantisation is far below any threshold we use, and `np.save` of
53 MB costs ~0.1 s where compression would cost seconds.

## 6. Usage

```python
from postproc import PostProcPredictor

pp = PostProcPredictor(base_predictor, {
    "tracer": "auto",                              # or "fdg" / "psma"
    "cleanup": {"min_volume_ml": 0.3, "suv_gate": 4.0},
    "compliance": {"fg_method": "grow"},           # or "random_walker"
    "negative_gate": {"enabled": True},
})
mask = pp.predict(ct, pet, spacing, scribbles, prev_pred, state_dir)
print(pp.last_info["t_total"], pp.last_info["constraints"])
```

`PostProcConfig.from_dict` raises `KeyError` on an unknown key — a silently ignored typo
in a sweep is worse than a crash.

Validation against ground truth on real cases — `tools/real_case_check.py` replays the
official evaluator's inner loop (including `simulate_scribble_from_label`) on real
volumes, scores Dice and the lesion-level F1 per iteration, and audits every
background-scribble deletion against the GT:

```
# realistic error profile: the GT, deliberately corrupted (default)
python src/postproc/tools/real_case_check.py --data <evalset> --n 8 --iters 6 \
       --strategy centerline --predictor corrupt --drop 1 --dilate 1 --n-fp 2

# anatomy / timing stress test: a global SUV threshold
python src/postproc/tools/real_case_check.py --data <evalset> --predictor suv
```

It also reports `n_zero_fp` / `n_zero_fn` — the number of iterations whose prediction had
no false positives (or no false negatives), i.e. the exposure of a given configuration to
the absorbing bug in §1. That number belongs in every ablation row, alongside the metric.

`--predictor corrupt` drops the largest lesions (false negatives for a tumor scribble to
recover), dilates the rest (boundary over-segmentation — the case where whole-component
deletion would destroy a true positive), and pastes false-positive blobs at the hottest
non-lesion locations, with a probability map that is confident on the surviving core and
unconfident on the rim and the pasted blobs. `--predictor suv` is a global SUV threshold:
on real whole-body FDG that segments brain, heart, kidneys, bladder and liver, so it says
nothing about accuracy — it exists to exercise the rules against real anatomy and to time
them on realistic volumes.

## 7. Parameters and defaults

`ComplianceConfig` — background

| name | default |
|---|---|
| `connectivity` | 18 (must match the scorer) |
| `bg_core_prob` | 0.9 |
| `bg_core_suv_ratio` | 0.6 |
| `bg_core_min_suv_ratio` | 1.5 |
| `bg_core_margin_mm` | 5.0 |
| `bg_protect_confident_core` | True |
| `bg_remove_max_radius_mm` | 20.0 |
| `bg_local_fallback_radius_mm` | 8.0 |
| `bg_sticky_removals` | False (opt-in; breaks purity) |
| `bg_core_min_volume_ml` | 0.05 |
| `bg_split_max_removed_fraction` | 0.5 |
| `bg_split_min_kept_volume_ml` | 0.05 |
| `bg_split_use_prob` | True |
| `bg_forbid_radius_mm` | 0.0 |

`ComplianceConfig` — foreground

| name | default |
|---|---|
| `fg_alpha` / `fg_peak_radius_mm` | 0.41 / 6.2 mm |
| `fg_max_radius_mm` / `fg_geodesic` | 30.0 mm / True |
| `fg_calibrate_from_footprint` | True |
| `fg_footprint_min_thickness_mm` | 2.5 |
| `fg_calibration_percentiles` | (2, 5, 10, 15, 20, 25, 30, 40, 50, 60) |
| `fg_stop_at_valleys` | True |
| `fg_valley_depth_suv` / `_min_distance_mm` / `_peak_ratio` | 1.0 / 10.0 / 0.5 |
| `fg_fallback_ball_radius_mm` | 4.0 |
| `fg_prob_include` | 0.3 |
| `fg_max_volume_ml` / `fg_threshold_escalation` | 100.0 / (1, 1.3, 1.7, 2.2, 3.0) |
| `fg_max_relative_growth` / `fg_min_growth_ml` | 8.0 / 10.0 mL |
| `fg_peak_over_whole_cluster` | True |
| `fg_low_contrast_fallback` | True |
| `fg_exclude_ct_below_hu` | −400.0 (`None` ignores CT) |
| `fg_cluster_radius_mm` / `fg_crop_margin_mm` | 20.0 / 6.0 |
| `fg_method` | `"grow"` |
| `rw_crop_size` / `rw_beta` / `rw_bg_distance_mm` / `rw_mode` / `rw_tol` / `rw_prob_tol` | 64 / 130 / 25 / `cg_j` / 1e-4 / 1e-2 |

`CleanupConfig`: `min_volume_ml` 0.3, `suv_gate` 4.0, `suv_floor_mode` `"component"`,
`suv_floor` `None`, `fill_holes` True, `fill_holes_max_ml` 0.5, **`min_components_kept` 1**;
rule v2 (§5.3.1, all off by default): **`rule_v2` False**, **`v2_rules` `None`**,
`v2_min_volume_ml` 0.3, `v2_suv_gate` 4.0, `v2_prob_gate` `None`, `v2_by_tracer` `None`,
`v2_silence_decay` 1.0, `v2_silence_requires_bg` True; recruitment (§5.3.2): **`recruit_prob_threshold` `None`**,
`recruit_min_suv_max` 4.0, `recruit_min_volume_ml` 0.1, `recruit_max_components` 5;
merging (§5.3.3): **`bridge_closing_voxels` 0**, `bridge_max_added_ml` 5.0.

`NegativeGateConfig` — every threshold accepts `None`, which disables that criterion, and
all but one ship disabled (§5.4): `enabled` True, **`max_total_volume_ml` 6.0**,
`max_component_volume_ml` None, `max_n_components` None, `max_prob` None,
`max_mean_prob` None, `max_soft_volume_ml` None, `max_suv` None, `max_suv_by_tracer` None,
**`require_no_scribbles` True**, `only_iteration_zero` False, `require_prob` False,
`collect_components` False.

Tracer heuristic (`tracer_classifier.DEFAULT_PARAMS`): `head_fraction` 0.15, `trunk_lo`
0.15, `trunk_hi` 0.75, `high_suv` 4.0, `body_hu` −500.0, `body_suv` 0.2, `blob_weight`
0.26, `blob_ref_ml` 100.0, `decision_threshold` −0.63, `margin` 0.20, `downsample` 2.

`MonotoneConfig`: **`mode` `"none"`**, `ema_alpha` 0.6, `fg_constraint_radius_mm` 6.0,
`bg_constraint_radius_mm` 6.0, `binarise_threshold` 0.5.

`PostProcConfig`: `tracer` `"auto"`, `use_probabilities` True, `cache_probabilities` True,
`cache_mask` True, **`pass_cached_prev_pred` False**, `strict` True, `verbose` False,
**`rescue_threshold_from_scribble` False**, `rescue_percentile` 50.0,
`rescue_min_threshold` 0.15, **`cleanup_after_compliance` False**.

`TRACER_SUV_FLOOR = {"fdg": 1.5, "psma": 1.0, "unknown": 1.0}`.

## 8. Timings

CPU only — nothing in this layer uses the GPU. Two measurements, because a synthetic
array and a real whole-body study stress different things.

**Synthetic, `400 × 400 × 330` = 52.8 M voxels** (`tests/test_timing.py`), float32 PET +
CT, 5 lesions + 300 small false-positive specks, 12 tumor and 12 background scribble
voxels, 4 cores:

| stage | time |
|---|---|
| background compliance (1 component hit, split path) | 0.61 s |
| foreground compliance, `grow` (with footprint calibration, geodesic bound, valley cut) | 0.30 s |
| foreground compliance, `random_walker` | 0.36 s |
| cleanup (SUV floor + 300-speck prune + holes) | 1.59 s |
| negative gate | 0.59 s |
| **sum, excluding random walker** | **3.10 s** |

Full pipeline over five consecutive iterations with the base predictor stubbed, i.e.
essentially pure post-processing overhead including ~0.5 s of state write per call:
2.60, 3.26, 3.42, 3.43, 3.45 s — **mean 3.25 s**.

Adding footprint calibration (≤ 11 candidate thresholds, each with its own geodesic pass
and watershed), the geodesic bound and the valley cut cost about 0.13 s in total, because
they all run on the scribble's crop rather than the volume.

**Real evalset cases** (`tools/real_case_check.py`, whole-body FDG from
`imagesTr`/`labelsTr`, 8 cores, `--strategy centerline --iters 6`).

`--predictor corrupt` — the GT with a realistic error profile (largest lesion dropped,
the rest dilated by one voxel, two false-positive blobs pasted at the hottest non-lesion
locations), 8 cases of which 2 are lesion-free. **No model is involved: every point of
improvement below comes from this layer alone.**

| | value |
|---|---|
| positives (6 cases), mean Dice | **0.125 → 0.502** over the 6 iterations |
| positives, mean lesion-level F1 (IoU ≥ 0.1, 18-conn) | **0.544 → 0.859** |
| positives, mean AUC-Dice / AUC-F1 | **1.875 / 3.841** (max 5.0) |
| background-scribble deletions audited | 8 |
| **GT lesions with > 50 % removed** | **0** |
| total GT volume removed by deletions | **0.00 mL** |
| post-processing per call | mean **3.45 s**, max **6.99 s** |

Per-case Dice, first → last iteration: 0.339→0.445, 0.169→0.332, 0.000→0.852,
0.048→0.461, 0.154→0.372, 0.040→0.548, plus two lesion-free cases at 0.000 throughout
(see §10.11).

`--predictor suv` — a global SUV threshold, which produces a pathological 1700–2700 mL
"prediction" covering brain, heart, kidneys and bladder, i.e. a deliberately worst-case
input for the component logic:

| | value |
|---|---|
| post-processing per call | mean **4.37 s**, max **6.18 s** |
| background-scribble deletions audited | 7 |
| **GT lesions with > 50 % removed** | **0** |

Against a 20 s target and a 20 min per-iteration container budget, with headroom even on
a shared, loaded host (measurements under load average ≈ 19 roughly double every number
and still land at 4–12 s).

**Zero ground-truth damage across 15 real background-scribble deletions** is the main
evidence that the split-not-delete rule (§5.1) does what it is supposed to. The runs also
exposed three genuine limitations, recorded in §10.9, §10.11 and §10.12.

### 8.1 Oscillation defects found on real cases, and their fix

Two independent defects in the growth rule made well-segmented cases swing across the
interaction, because the interactive model conditions on its previous prediction and
therefore amplifies any over-large edit we make. Both were found by replaying a single
case with the per-cluster diagnostics of §5.2, and both are pinned by regression tests
(`test_scribble_tail_in_normal_tissue_does_not_flood`,
`test_calibration_survives_a_cluster_spanning_several_slices`).

**A — the threshold collapsed onto the tracer floor.** A stroke ran from a hot lesion out
into normal tissue. Its hot end was already segmented, so growth was seeded from the cold
tail, the local SUVpeak was measured on that tail alone, `0.41 × peak` fell below the FDG
floor, and `T` became 1.5 — at which point 33 % of the crop qualifies and the region grew
by 46 mL of normal anatomy.

**B — calibration was silently skipped.** Accumulated scribbles from different iterations
land on different slices and cluster together; the footprint search abandoned calibration
whenever a cluster spanned more than one slice, dropping a small PSMA lesion back onto
the alpha rule on alternate iterations only.

Verified against the recorded run, same model checkpoint, same gate-v2 configuration,
6 iterations, per-iteration Dice:

| case | AUC-Dice before | after | before | after |
|---|---|---|---|---|
| `fdg_1a1712f7d0` (random) | 2.790 | **4.009** | 0.000 0.915 **0.316 0.483** 0.891 0.370 | 0.000 0.915 0.886 0.884 0.883 0.882 |
| `psma_37af1d5c2373d0c4` (random) | 2.883 | **3.752** | 0.000 0.777 0.824 0.844 **0.178** 0.519 | 0.000 0.777 0.824 0.844 0.856 0.901 |
| `psma_907f1345abedf4fa` (centerline) | 2.756 | **4.123** | 0.000 0.833 0.929 0.937 **0.030 0.054** | 0.000 0.833 0.929 0.937 0.963 0.923 |

Worst single-iteration Dice drop across the three: **0.906 → 0.040**. The trajectories are
identical up to the iteration that used to collapse, and continue upward instead.

Across a wider check of 18 cases (the 8 whose AUC-Dice regressed against the bare model,
4 further cases that oscillated without losing AUC, and 6 controls where the layer had
been neutral):

* all 8 regressions recovered, and all 8 now **match or beat** the bare model
  (Σ AUC-Dice 23.99 → 29.91, against the bare model's 29.04);
* all 4 extra oscillators improved, largest intra-case drop now 0.085 (was 0.879);
* the 6 controls moved by at most **0.003** AUC-Dice — the fix does not disturb cases it
  was not meant to touch.

`tools/oscillation_scan.py` is the standing check: run it on the post-processed and the
bare-model runs and compare. On the pre-fix run it reported 9 of 100 cases dropping more
than 0.15 Dice between consecutive iterations, 7 of which did not oscillate in the bare
model — those 7 were ours.

## 9. Tests

132 fast tests + 2 timing tests:

```
AUTOPET_SKIP_SLOW=1 python -m pytest src/postproc/tests -q
```

Coverage highlights:

* G1, G2, G3, all together, each under a second application (idempotency);
* an already-satisfied tumor scribble changing nothing, a background scribble on
  unpredicted voxels changing nothing, a partially covered scribble growing only its
  missing part;
* **split-not-delete**: a true positive that bled into cool tissue keeps > 90 % of the
  lesion while > 50 % of the bleed is removed; a homogeneous FP is deleted whole; a
  component with a tumor scribble is never deleted whole; the GT audit reports damage;
* **calibration**: a thick footprint calibrates the threshold (IoU > 0.9) and rejects a
  halo the 41 % rule would swallow; a thin footprint falls back to the alpha rule;
* **geodesic** bound tighter than euclidean on a U-shaped channel; growth stopping at a
  valley between two foci; the two-lesion non-merge guard; the boundary-scribble
  over-segmentation guard (< 2 mL added where a fixed 30 mm ball would add ~113 mL);
* **purity**: identical output with and without a state directory, and with the directory
  wiped between every call;
* **negative gate**: fires at all six iterations of a lesion-free case; each veto
  criterion; `None` disables a criterion (the mechanism the shipped defaults rely on) and
  a lesion-sized prediction is never emptied; a tumor scribble un-gates permanently *and*
  restores lesions no scribble mentioned; `component_stats` ranks and measures;
* **negative gate, on real data**: `tools/gate_replay.py --out_dir <model>` rebuilds every
  cached iteration-0 prediction from its sparse bundle, runs the real gate with the real
  config six times per case and asserts the decision is iteration-stable, then reports the
  fire / no-fire counts and the expected Δ. With the shipped defaults: baseline 10 of 10
  non-empty lesion-free cases emptied and 11 positives touched; fine-tuned 14 of 16 and
  17. `--expect_neg` / `--expect_pos` turn it into an assertion;
* **tracer heuristic**: score, confidence and margin are reported; confidence rises away
  from the boundary; the body extent — not the array extent — defines the head slab, with
  and without a CT; `tools/tracer_check.py` measures it against the case-name ground truth
  (100/100 on the evaluation set, both the CT and the PET-only path);
* cleanup never empties a prediction the gate did not veto;
* small cold specks removed, small hot specks kept, scribbled components never removed;
* **rule v2**: identical to v1 when the probability criterion is disabled; the criteria
  are a conjunction (a confident component survives being small and cold); no softmax ⇒
  the probability criterion cannot justify a deletion; per-tracer overrides resolve, and
  an unknown override key raises; the silence decay protects a component that survived
  four rounds unscribbled and does not protect one that was scribbled;
* **merging**: a two-voxel gap is bridged and a ten-voxel one is not; a background
  scribble in the gap is never bridged over (G1); an oversized bridge is refused;
  merging is a pipeline stage after compliance, so `cleanup_mask` is unaffected by it;
* **recruitment**: only new, hot, sub-threshold components are added, existing components
  are not reshaped, the count is capped, and no softmax or no threshold is a no-op;
* **second cleanup pass**: it prunes the fragment a background-scribble split left
  behind, still honours a tumor scribble, and is off by default;
* revocable constraints; every blend mode; channel-first softmax; shape mismatches;
* state round trips, corrupt-state tolerance, unwritable paths, `from_scribbles`,
  `infer_iteration`, and the cached previous mask not reaching the base predictor unless
  `pass_cached_prev_pred` is set;
* **the official generator**: all three strategies, the coordinate convention, one
  error-driven step, and a full multi-iteration loop with a base predictor that misses a
  lesion, invents a false positive and ignores every scribble.

`tests/test_official_scribbles.py` needs the official challenge repo at `autoPETV/`
(gitignored); it skips automatically when absent.

## 10. Known limitations and open issues

* **The detection rule is fitted off-policy.** `tools/dmm_rule_fit.py` recomputes the
  metric under a candidate rule on the *scribble sequence a different configuration
  produced*. A rule that changes the mask changes the next scribble, so the offline
  number is an estimate and the six-iteration run is the measurement. The two agreed to
  within the sampling noise of the 33-case fitting set, but that is one comparison, not a
  guarantee.
* **`cleanup_after_compliance` is redundant while `rule_v2` is on in both passes.** On
  real cases the first pass already removes everything the rule catches (a PSMA case goes
  146 → 89 components in pass 1) and the second pass then removes 0. It is kept on because
  it costs ~1 s and is the stage that would catch a compliance-created component if one
  ever appeared.
* **The chosen rule has a tail.** 8 of 35 fitting cases lose AUC-F1 and the worst loses
  0.83 — a PSMA case whose small lesions all sit below SUVmax 5. `SUVmax<4 or
  prob_mean<0.8` is the conservative alternative (+0.097 instead of +0.129, worst −0.28).
* **`interactive_eval.py` does not persist the new stage counters.** `case_info.json`
  stores a fixed key list from `last_info`, so `cleanup_after_compliance`, `bridge` and
  the resolved `rules` are not in the run record; they have to be re-derived by replaying
  the cache. Proposed fix (harness owner): add `"t_cleanup2"`, `"t_bridge"`, `"cleanup"`
  and `"cleanup_after_compliance"` to that tuple at `src/interactive_eval.py:852`.

1. **The empty-error-region bug (§1) is unverifiable before the final submission.** The
   preliminary phase runs iteration 0 only. Our mitigations — never empty except via the
   gate, `min_components_kept`, no pruning below a floor — bound the exposure but cannot
   remove it. The open question for the organizers: *after the fix, what happens when
   `overseg` is empty but `underseg` is not (prediction a strict subset of the GT), and
   vice versa?* `overseg` is unpacked first, so `pred ⊊ GT` crashes before `underseg` is
   reached.
2. **The negative-gate threshold is fitted on 36 lesion-free cases.** The rule *shape* is
   stable across both networks and across every leave-one-out fold, but the objective is
   flat in the threshold: on the baseline anything from 1.05 to 8 mL scores within 0.7 %
   of the optimum, and on the fine-tuned network the curve keeps rising to 25 mL. The
   lesion-absent fraction of the *test* cohort is unknown (0.36 here, ~0.5 in the FDG
   training cohort) and the size of the lever scales with it.
3. **The tracer heuristic is measured, not proven.** 100/100 on the evaluation set with a
   factor-1.60 class separation (§5.5) — but that is 100 studies from two centres. The
   shipped gate is tracer-agnostic so a misclassification cannot cost a lesion-free case,
   and `confidence` is exposed for the callers that do branch on the tracer.
4. **No organ veto.** A TotalSegmentator-based veto would slot between compliance and
   cleanup, but the container runs `--network=none` and TotalSegmentator downloads
   weights on first use, so it would have to be baked into the image with
   `TOTALSEG_HOME_DIR` pinned. Not attempted here.
5. **Calibration assumes the footprint is a cross-section of the *error*.** When our mask
   already covers part of that lesion on that slice, the target is footprint ∪ our own
   component on that slice, which is right; when the scribble spans two lesions on one
   slice it is not.
6. **`infer_iteration` is heuristic** — two scribbles within 10 mm count as one event.
   Nothing keys on it today; do not start without a stored counter.
7. **Growth cannot cross a background-constrained voxel**, so a tumor scribble next to an
   earlier background scribble grows a donut around it. Correct per the guarantees,
   possibly not what a clinician meant.
8. **The h-maxima valley rule can over-split a genuinely heterogeneous lesion** (necrotic
   core, rim uptake). `fg_valley_depth_suv` is the knob; untuned on real data.
9. **The split can stall on a pathologically large component.** Measured on real cases
   with a global-SUV stand-in predictor: a single 2700 mL physiological component absorbs
   every background scribble and each iteration removes only ~5 mL, so the corrections
   never converge. The core rule is behaving as designed — it refuses to delete something
   the evidence says is mostly real — but the outcome is useless. With a sane base model
   a component that large cannot occur, and adding a "component is enormous → delete
   more" escape would be tuning on a broken predictor, so it is deliberately not there.
   Re-check once the real network is wired in.
10. **Memory**: mask + probability + accumulated background region + one `cc3d` label
    volume ≈ 0.6–1.0 GB at 400×400×330, against a hard `--memory=30g` with no swap. Fine,
    but the network's own peak must be added and measured.
11. **The gate is validated on real negatives, but only at iteration 0.** Both networks'
    iteration-0 output on all 100 evaluation cases is cached and replayed
    (`tools/gate_replay.py`), which pins the fire / no-fire counts exactly and asserts
    that the decision is identical at all six iterations for a case that never receives a
    scribble. What is *not* measured end to end is the cost side: emptying a positive at
    iteration 0 is assumed to cost `0.5 × Dice@0` and nothing more, on the argument that
    iteration 1 then delivers a tumor scribble and un-gates the case permanently. On the
    fine-tuned network, whose fifth input channel is the previous prediction, an emptied
    iteration 0 also changes that channel at iteration 1, so the true cost there could
    exceed the estimate. A six-iteration run of just the emptied positives, gate on
    versus gate off, settles it for a few GPU-minutes per case.
12. **No component-level veto.** The lesion-free false positives are hot and sit at the
    ends of the body (5 of 10 at `z_frac` 0.00, 3 at 0.91–0.97), so a "hot and in an end
    zone" component veto is tempting. Measured on the baseline's components: `z_frac`
    outside [0.03, 0.95] with `shell_suv_max ≥ 8` deletes 7 of the 11 lesion-free
    false-positive components and 23 false positives on lesion-present cases — but also
    **4 true-positive components**, a detection-metric loss for no Dice gain, since the
    case-level gate already catches all 10 lesion-free cases on its own. Not shipped; the
    features (`component_stats`) are there if the detection rules want them.
13. **A long stroke into low-contrast tissue still costs a few mL.** When the intensity
    evidence cannot delineate anything, G2 is satisfied with a
    `fg_fallback_ball_radius_mm` ball around every scribble voxel. For a 100 mm stroke
    that is a ~5 mL tube, most of it false positive. It is bounded, and an order of
    magnitude better than thresholding at the tracer floor (~46 mL on the case that
    motivated the guard), but it is not free: the real fix is a model whose softmax
    covers the stroke, not a better rule.
14. **Background compliance can still empty a mask.** `min_components_kept` stops
    *cleanup* from emptying a prediction, but a background scribble that deletes the last
    surviving component legitimately empties it — that is the scribble's explicit
    instruction and G1 requires it. Observed once in the real run (case
    `fdg_1a1712f7d0`, iteration 1: 26.8 mL → 0.0 mL, then rebuilt to Dice 0.852 by the
    tumor scribble at iteration 2). It is flagged as `info["empty_without_gate"]` and
    counted by the harness, because an empty prediction is exactly the state that trips
    the absorbing bug in §1. If the organizers confirm that branch is unfixed, the
    remedy is to keep the single best component alive against a background scribble too
    — at the cost of violating G1, which is why it is not the default.
