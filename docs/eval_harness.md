# Evaluation harness

An in-process re-implementation of the official autoPET V evaluation loop
(`autoPETV/interactive/interactive_loop.py`): same scribbles, same metrics, same failure behaviour, but
the model is loaded once for the whole run instead of one `docker run` + `nnUNetv2_predict` per
iteration, and CT/PET preprocessing is not redone six times per case. A 100-case, 11-variant ablation is
therefore a single command rather than 6600 container calls.

| file | contents |
|---|---|
| `src/interactive_eval.py` | the loop, metrics, AUC, stratified aggregation, prediction cache, CLI |
| `src/predictor.py` | `Predictor` interface; `BaselineNNUNetPredictor` (reference, goes through NIfTI files like the baseline container); `FastBaselineNNUNetPredictor`; `InteractiveNNUNetPredictor` (our 5-channel model); `ThresholdPredictor` (for tests) |
| `src/smoke_test.py` | synthetic end-to-end tests (CPU), nnU-Net equivalence test (GPU), replay of the organizers' reference run |
| `src/profile_predictor.py` | per-stage profiler behind the numbers below |
| `src/ablate.py`, `configs/ablations.json` | the ablation ladder |
| `scripts/eval_variant.sh` | one trained model -> `summary.json` + `run.json`, the shipped pipeline |

Scribble simulation and metrics are **imported from the challenge repo** (`--repo`), never
re-implemented, so scribble generation matches the official code including the `seed=42` defaults and the
`fp <= fn` rule.

## Running it

```bash
python src/interactive_eval.py \
    --input_cases /data/evalset --image_dir /data/evalset/imagesTr --label_dir /data/evalset/labelsTr \
    --repo /path/to/autoPETV --out_dir runs/A0_baseline \
    --predictor fast_baseline_nnunet --cache_dir .cache/predictions
```

`--input_cases DIR` expects the layout of the official loop (`DIR/images/<case>_0000.nii.gz` CT, `_0001`
PET, `DIR/labels/<case>.nii.gz`); `--image_dir` / `--label_dir` override it, which is what the
`imagesTr` / `labelsTr` naming of our held-out set needs. `--predictor postproc --base_predictor
interactive_nnunet` runs our full method. The flags that change the protocol rather than the plumbing are
`--max_iters` (6), `--strategy {centerline,random,boundary,all}`, `--eval {fixed,buggy}` and
`--replay_scribbles_dir`; the rest are in `--help`.

`scripts/eval_variant.sh` wraps the two commands a results row needs — the loop under the shipped
post-processing configuration and `finalize_run.py` — behind `TAG` and `MODEL_FOLDER`, and copies the
model description files (`plans.json`, `dataset.json`, `dataset_fingerprint.json`) next to `fold_0/` if
the trainer did not. `EXTRA` passes flags that change the base model, e.g.
`EXTRA="--foveal_crop --foveal_fuse max"`.

`src/ablate.py` runs a named subset of `configs/ablations.json` sequentially on the same case list,
strategy assignment, seed and cache, and writes a markdown results table.

Outputs in `--out_dir`: `<tag>/iter_{0..5}.nii.gz` (uint8, PET geometry) and the matching
`_scribbles.json` / `lesion-clicks.json`, a `state/<tag>/` directory standing in for the mount the
container gets, `metric_scores.json` and `metric_scores_AUC.json` in the official schema, `case_info.json`
(geometry, per-iteration seconds and FP/FN counts) and `summary.json`. The score files are rewritten after
every case, so an interrupted run resumes with `--cache_dir` or is re-scored with `--prev_pred_dir`.
`<tag>` is the official tag, `basename(ct).replace(".nii.gz", "")`, keeping the `_0000` suffix. Pooled
numbers hide the two effects that decide this challenge, so `summary.json` always carries
`by_lesion_status` (a lesion-free case scores AUC-Dice 5.0 or 0.0 and nothing in between) and `by_tracer`
next to `mean_auc_dice` / `mean_auc_dmm` / `final_score_50_50`.

## Loop semantics

Six iterations per case. Iteration 0 gets no scribbles. For `k > 0` on a case with lesions, one scribble
is derived from the *previous* prediction:

```
overseg  = (prev_pred == 1) & (gt == 0)
underseg = (prev_pred == 0) & (gt == 1)
scribbles_bg, _, fp = simulate_scribble_from_label(overseg,  strategy)
scribbles_fg, _, fn = simulate_scribble_from_label(underseg, strategy)
data["tumor"] += scribbles_fg   if fp <= fn   else   data["background"] += scribbles_bg
```

`fp`/`fn` are the **scribble sizes** returned by the official function (voxel count of the simulated 2D
stroke), not the error component volumes; ties, including "both regions empty", go to tumor. On a
lesion-free case no scribble is ever added, so all six predictions are identical. Scribbles round-trip
through the Grand Challenge JSON exactly as in the official loop. `dice_score` is a verbatim copy of the
official function (empty prediction ∧ empty ground truth → 1.0; any false positive on a lesion-free case
→ 0.0) and `dmm` is `MetricEvaluator()(...)['f1']`, lesion-level F1 at IoU ≥ 0.1, connectivity 18. An
exception inside an iteration is caught and scores 0.0/0.0, as officially. Per case,
`auc = np.trapezoid(values, [0..5])`, maximum 5.0.

`--replay_scribbles_dir` feeds a fixed, pre-recorded scribble sequence instead of simulating from our own
errors. The challenge's category-2 scribbles are collected once against the *baseline's* predictions and
replayed unchanged to every algorithm, so a scribble may sit on a voxel we already got right; any
compliance logic must be a safe no-op in that case, and this flag is how we test it.

### Coordinate conventions

Everything the challenge exposes is in **nibabel index space**: `nib.load(x).get_fdata()[i,j,k]` with
`(i,j,k) = (x,y,z)`, `z` axial. The points in `lesion-clicks.json` are `np.argwhere` indices into that
array. nnU-Net reads NIfTI with SimpleITKIO, whose array axes are the **reverse** of nibabel's
(`SimpleITKIO.read_images(f)[0][c] == nib_array.transpose(2,1,0)`, `props['spacing'] == nib_zooms[::-1]`);
the harness does the transposes explicitly and only at that boundary, and transposes the returned softmax
back to `(C, x, y, z)`.

### The interactive 5-channel model

`InteractiveNNUNetPredictor` runs the model trained by `nnUNetTrainer_Interactive`: CT, PET, tumor
guidance, background guidance, previous mask, with `NoNormalization` on channels 2–4.

**The guidance lives on the preprocessed grid, not on the image grid.** In training,
`InteractionSimulationTransform` sits at the end of the pipeline, after `SpatialTransform`, so channels
2–4 are generated directly on the patch and the radius `R = 10` is in voxels of the plans grid. Inference
therefore preprocesses CT and PET through nnU-Net's normal path and then stamps channels 2–4 onto the
resampled grid, reusing `src/train/guidance.py` — the same function the training transform calls. Feeding
five NIfTIs to `predict_from_files` the way the baseline container does would be wrong for this model:
`NoNormalization` makes normalisation a no-op, but nnU-Net would still put the guidance through an
order-3 spline resampling that training never applied, ringing the cone-shaped channels and blurring the
binary previous-prediction channel.

Scribble coordinates are mapped to the preprocessed grid with the inverse of skimage's centre-aligned
resize map, `j = round((i + 0.5)·out/in − 0.5)`, after subtracting the crop-box origin. The crop box comes
from CT and PET only (`include_clicks_in_bbox=False`), matching how the training store was cropped;
scribbles outside it are counted and warned about. `src/test_interactive_predictor.py` checks the mapping
against nnU-Net's own segmentation resampler and the guidance channel against the one the training
transform produces from the same points.

Channel 4 is the previous iteration's **final** mask — after post-processing, not the raw network output.
In the container, `pass_cached_prev_pred=true` plus `--interactive_state_dir` make the caller persist it
with `save_prev_mask`; the predictor never writes that file itself, because under a post-processing layer
the mask it returns is not final. Since the output depends on `prev_pred`, `Predictor.cache_state_key`
returns a hash of the previous mask and both cache layers fold it into the key — otherwise iteration 1's
answer would be served at iteration 3.

### Foveal re-inference (`--foveal_crop`)

The sliding window sees a scribble only in whatever tiles happen to cover it, at whatever offset the
tiling grid gives; a scribble can therefore sit near a tile border, far from the receptive-field centre
of the voxels it is meant to correct. `--foveal_crop` adds one deliberate window: at every iteration
that carries a scribble, a patch-sized region (`patch_size` of the plans, 112×160×128 at the plans
spacing) is cut from the already-assembled 5-channel array, centred on the **newest scribble's
centroid** and clamped to the volume, and pushed through the network a second time. The two logit
fields are fused inside that window — `--foveal_fuse max` (default) or `mean` — and everything
downstream, post-processing included, is unchanged.

The window is fed back through `predict_logits_from_preprocessed_data`, so it is a single tile whose
Gaussian weighting divides out: one plain forward pass through the same code path as the full
prediction, with the same padding rules and the same mirroring setting. Cost is one forward pass
(~1 s) per iteration that fires.

*The newest scribble.* The loop appends one stroke per iteration to one of the two lists, so the newest
stroke is the tail of whichever list grew since the previous call on that case; the predictor remembers
the two counts per case. Where there is no memory — the first call of a case, which is the state after
iteration 0 was served from the prediction cache — exactly one list can be non-empty at iteration 1, so
its tail is the newest stroke and the fallback is exact where it is used. `predictor.last_foveal_info`
records the window, the stroke it came from, which of the two rules picked the centre, the wall time and
the mean absolute logit change; `last_timings["foveal_fired"]` is the flag.

At iteration 0 there is no scribble, the pass does not fire, and the output is **bitwise identical** to
the plain model — asserted on two cases before the row was run.

The option changes what the network computes, so it is folded into `base_predictor_identity` and a
foveal run gets its own cache namespace. It is added to the identity **only when the option is on**, so
every existing row keeps its namespace and its cached predictions; the price is that a foveal run
recomputes iteration 0 rather than reading the plain model's copy of it.

### Probability-level ensembling (`--base_predictor ensemble`)

`--base_predictor ensemble --ensemble_members <folder>[:<ckpt>[:<weight>]] ...` runs every member on the
same case, the same scribble set and the **same previous mask**, and averages their foreground softmax.

The earlier two-checkpoint row averaged weights by handing nnU-Net two *folds*, which only works for
members that share one `plans.json`. Members with different plans — a `PlainConvUNet` on
`nnUNetPlans_interactive` and a `ResidualEncoderUNet` on `nnUNetPlans_re` with 192³ patches and the
in-network PET renormalisation — do not even share a preprocessed grid, so nothing can be averaged before
the export. The one geometry they all agree on is the original image grid, which nnU-Net's exporter
already resamples each member's softmax back to, so that is where `src/ensemble_predictor.py` forms the
weighted mean. The ensemble mask is `p̄ > 0.5`, which for two classes is exactly nnU-Net's own `argmax(0)`
including its tie-break, and the averaged probability is what the post-processing layer then sees.

Weights are normalised (`--ensemble_weights 3 7` ≡ `0.3 0.7`, or per member as a third field of the spec);
the default is equal weights. A member with weight 0 is skipped in the sum rather than added as `0.0 · p`,
so `--ensemble_weights 1 0` reproduces member 0 **bit for bit**, mask and probability — the property that
makes the row interpretable, and the one `src/test_ensemble_predictor.py` asserts on real checkpoints
before any ensemble row is run. Channel 4 of every member is the ensemble's own previous final mask, never
the member's own, so no member ever sees a state the shipped pipeline would not produce.

The members and their weights are part of `base_predictor_identity`, so an ensemble gets a cache namespace
of its own and cannot read a single model's cached predictions. Cost per iteration is the sum of the
members' iterations; CT/PET preprocessing is cached per member per case as usual, so a second member costs
one more sliding window and one more resample-back, not a second preprocessing pass.

### Timing one configuration against another (`src/bench_inference.py`)

`run.json`'s `median_iteration_seconds` is measured over a whole evaluation, so two rows produced on
different boxes — or on the same box with a training job holding the GPU — are not comparable, and the
ratio that matters (what does TTA cost? what does a second member cost?) cannot be read off them.
`bench_inference.py` times several configurations **in one process, on the same cases, back to back**, at
iteration 0:

```bash
python3 src/bench_inference.py --cases 3 --case_dir /content/work/evalset \
    --config plain:/content/work/models/b10 \
    --config tta8:/content/work/models/b10 \
    --config ens:/content/work/models/b10,/content/work/models/re40
```

A config is `<name>:<spec>`, `<spec>` is comma-separated `<model_folder>[:<ckpt>]` (more than one = an
ensemble), and a name containing `tta` (or listed in `--tta_for`) runs with 8-way mirroring. The table
reports median total and network seconds per iteration and the ratio against the first config; `--out`
writes it as JSON.

## Deliberate discrepancies vs. the published loop

**Empty error region.** `simulate_scribble_from_label` returns `coords, label_cls, size` normally but
`[], False` when its input mask has no foreground, and `interactive_loop.py` unpacks three values from
both calls. In the published code an iteration with no false positives or no false negatives therefore
raises `ValueError` and scores 0.0/0.0 — a perfect prediction scores `1, 0, 0, 0, 0, 0`. The organizers
confirmed this will be fixed by propagating a perfect prediction to the remaining iterations, which is the
default here (`--eval fixed`); `--eval buggy` reproduces the published behaviour and the smoke test
asserts both. When only *one* region is empty the handling is unspecified; we assume the scribble comes
from the non-empty region, which is what `fp <= fn` does once the empty side reports size 0.
`summary.json → empty_error_region_exposure` reports how often we land in that corner.

**DMM on lesion-free cases.** `metrics.calc_f1` returns NaN when `tp + fn == 0`, always true on an empty
ground truth, and the organizers aggregate with nanmean — lesion-free cases are excluded from AUC-DMM.
`summary.json` reports both `mean_auc_dmm` (nanmean, the official number) and
`mean_auc_dmm_nan_propagating`. A false positive on a lesion-free case therefore costs that case's entire
AUC-Dice and costs nothing in DMM, which makes the negative gate the largest single Dice lever available.

**Case skipping.** The published loop does `if "fdg" in ct or '198' in ct: continue`, matching the full
path — that skips every FDG case and any PSMA case whose patient hash contains `198`. Off by default;
`--official_case_skip` turns it on.

**Smaller things.** `np.trapz` was removed in NumPy 2, so the AUC uses `np.trapezoid`. The published loop
`zip`s three independently sorted file lists, where one missing label silently shifts the pairing; we
check that the basenames belong to the same case and abort otherwise (`--no_strict_pairing` restores the
official behaviour). `prev_pred` is handed to `Predictor.predict` for convenience, but nothing survives
between container calls on Grand Challenge except what the algorithm writes itself, so a
submission-faithful method uses `case_cache_dir`.

`autoPETV/test/` ships a label, five predictions and five scribble JSONs from a reference run
(`--strategy random`, baseline nnU-Net). `smoke_test.py` replays it: `dice_score` reproduces all five
published values to < 1e-9, and feeding reference prediction *k* plus the ground truth into our scribble
step reproduces `iter_{k+1}_scribbles.json` exactly for k = 0..3 — and only for `strategy="random"`.

## Performance

On the reference path (NVIDIA L4, `src/profile_predictor.py`), a native-resolution PSMA case that nnU-Net
upsamples to 308×400×400 spends 115 s of a 151 s iteration in preprocessing, 108.5 s of it in a
single-threaded `skimage.resize(order=3)` over the four input channels. `FastBaselineNNUNetPredictor`
builds the channels in nnU-Net axis order without files, caches CT/PET preprocessing across the six
iterations, never builds the unused `seg` array, resamples channels in a thread pool (scipy releases the
GIL) and resamples the logits on the GPU — nnU-Net resamples probabilities with order 1, which *is*
trilinear interpolation, so that is the same operator.

| case | reference, 6 iterations | fast, 6 iterations |
|---|---:|---:|
| FDG 400×400×326 @ 2.04/2.04/3.0 mm (already at plans spacing) | 205 s | 84 s (14 s/iter) |
| PSMA 200×200×462 @ 4.07/4.07/2.0 mm (upsampled 2.7×) | 821 s | 245 s (41 s/iter) |

The rows differ because nnU-Net's resampler short-circuits when input and target shape match: the FDG half
of the data is already at the plans spacing, so an iteration is dominated by the network (~9 s), while the
PSMA half spends ~30 s per iteration on the spline resampling of the guidance channels. `smoke_test.py`
asserts that the fast and the reference predictor agree voxel for voxel on both cases.

`--cache_dir` is a content-addressed cache of predictions keyed by (base predictor identity, case,
scribble set). The identity hashes model folder, folds, checkpoint, TTA, tile step and resampling
settings, but *not* the post-processing config, so all variants of one model share a namespace: iteration
0 is computed once for the whole ladder, and a lesion-free case once instead of six times. Variants stop
sharing as soon as their predictions diverge, since the next scribble is simulated from their own error.
`--cache_probabilities` (implied by `--predictor postproc`, which needs the softmax) stores the foreground
probability as uint8. `--prev_pred_dir` re-scores saved predictions under new metric code with no model.

Two knobs are **not** equivalence-preserving and are off by default: `--resample_channels torch` (GPU
trilinear for the input channels, 108.5 s → 0.63 s, but max|difference| vs. the reference is 2.0 for CT,
21 for PET and 185 for the guidance in normalised units, so it changes the input distribution the network
was trained on), and `--enable_tta` (with `--mirror_axes 0 1 2`, full 8-way mirroring costs
**4.34× the whole iteration** and 6.97× the network alone -- 6.42 s -> 27.92 s median per iteration,
measured back to back with `bench_inference.py` on 5 screening cases with nothing else on an A100.
An earlier note here put it at 2.6×; that figure does not survive a same-process measurement of all
three axes). Stamping the guidance straight onto the resampled grid is likewise not a
valid shortcut in the *baseline* path: z-score normalisation of an almost-empty channel divides by a tiny
standard deviation, so each scribble voxel becomes a spike of ~472 normalised units that the order-3
spline then spreads over ~3000 voxels, and a 0/1 indicator has none of that support.

## Guarantee checking

With `--predictor postproc` the loop checks, on the mask it is about to score and at every iteration, that
every accumulated tumor scribble voxel is inside the mask (G2) and no accumulated background scribble
voxel is (G1), aggregating into `summary.json → guarantees`. Both are always measured; `enforced` records
which of the two the configuration actually claims, so a ladder rung that deliberately runs without
tumor-scribble compliance reports how often it misses a point without that counting as a failure.
`--strict_guarantees` turns a violation into an exception; `--check_guarantees always` measures them for
any predictor, with a bare baseline as the control.

## Known limitations

* **The strategy assignment for `--strategy all` is a guess.** The organizers assign a third of the cases
  to each strategy "consistently for all participants" but have not published the partition. We use a
  deterministic round-robin over the sorted case list, so per-strategy numbers are a 1/3 sample of each,
  not a reproduction of the official split.
* **The one-sided empty-error case is unspecified**; our assumption is documented above, not confirmed.
* **The reference-run check covers scribbles and metrics only**, not model inference — the organizers'
  test case ships CT and PET as git-lfs pointers.
* **`--tile_step_size` has not been swept.** It is the next knob if the network becomes the bottleneck.
