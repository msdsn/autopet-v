# Interactive fine-tuning pipeline

Turns the organizers' 4-channel `Dataset998` nnU-Net baseline into a 5-channel **interactive** model
trained with online, error-driven scribble simulation. Everything lives in `src/train/` and plugs into
stock `nnUNetv2_train` — no fork of nnU-Net, no changes to the preprocessed store.

| file | what it does |
|---|---|
| `guidance.py` | clipped-EDT encoding `max(0, 1 − d/R)` of a scribble point set |
| `scribble_sim.py` | safe wrapper around the **official** simulator + the error-driven iteration protocol at patch level |
| `corrupt.py` | ground-truth label → plausible "previous prediction" |
| `interaction_transform.py` | the batchgenerators-v2 transform that writes channels 2–4 |
| `nnUNetTrainer_Interactive.py` | the trainer (`-tr nnUNetTrainer_Interactive`) |
| `init_from_baseline.py` | 4 → 5 input-channel weight surgery |
| `make_plans.py` | writes `nnUNetPlans_interactive.json` + the 5-channel `dataset.json` |
| `make_synthetic_dataset.py` | tiny fake preprocessed dataset for end-to-end tests |
| `bench_transform.py` | per-sample cost and correctness checks of the simulation |
| `viz.py` | sanity-check figures straight out of the real dataloader |
| `networks.py` | the B13/B14 network classes and the weight surgery onto them |
| `nnUNetTrainer_InteractiveArch.py` | the B13/B14/C0 trainers and the epoch-0 identity gate |
| `make_arch_plans.py` | writes the variant plans (`nnUNetPlans_b13/b14.json`) |
| `test_networks.py` | shapes, epoch-0 equivalence, parameter counts, speed and VRAM |

## Model input

| ch | content | normalization | source |
|---|---|---|---|
| 0 | CT | `CTNormalization` | preprocessed store |
| 1 | PET (SUV) | `ZScoreNormalization` | preprocessed store |
| 2 | tumor guidance | `NoNormalization` | generated on the fly |
| 3 | background guidance | `NoNormalization` | generated on the fly |
| 4 | previous prediction (0/1) | `NoNormalization` | generated on the fly |

Two deliberate departures from the baseline:

* **Encoding.** The baseline writes σ = 0 "heatmaps" — a handful of isolated binary voxels — and then
  *z-scores* them, turning each scribble voxel into a ≈2500σ spike whose magnitude depends on the image
  size and on how many scribbles have accumulated. We use a clipped Euclidean distance transform,
  `e(x) = max(0, 1 − d(x)/R)` with `R = 10` voxels, and mark channels 2–4 `noNorm`, so the value the
  network sees is exactly the value we wrote, bounded in [0, 1] and spatially supported.
* **Previous-prediction channel.** The baseline has no memory of its own output, which is the structural
  reason it cannot refine iteratively. Channel 4 is RITM-style mask guidance.

The store keeps 4 channels (CT, PET and two all-zero placeholders, so it stays compatible with the
baseline preprocessing). The transform keeps only the first two and writes its own channels 2–4, so it
works unchanged against a 2-channel or a 4-channel store. `KeepFirstChannelsTransform` drops the
placeholders at the *front* of the pipeline so `SpatialTransform` never spends a cubic-spline resampling
pass on them.

`make_plans.py` derives `nnUNetPlans_interactive.json` from the baseline `plans.json`, changing only
`normalization_schemes`, `use_mask_for_norm` and the placeholder
`foreground_intensity_properties_per_channel` entries for channels 2–4. Spacing, patch size, batch size
and the whole PlainConvUNet architecture are untouched — that is what makes the weight surgery valid — and
`data_identifier` keeps pointing at the store, because the guidance never touches disk. Note that nnU-Net
resolves `normalization_schemes` entries as *class names*, so the plans file must say `NoNormalization`,
while `noNorm` is the `dataset.json` channel-name spelling that maps to the same class; `make_plans.py`
writes each spelling in its own file.

## How the guidance channels are produced

Per training patch, inside the dataloader worker:

1. **Sample the scribble count** `k ~ Categorical([.28, .22, .18, .14, .10, .08])` over `k ∈ {0..5}` — the
   evaluator runs 6 iterations, so 5 is the maximum a case can accumulate. The heavy mass on small `k` is
   deliberate: a model trained on dense guidance only can score 0 Dice at 0 clicks, and half of our
   AUC-Dice on lesion-free cases is the `k = 0` state.
2. **Invent a previous prediction `P`** by corrupting the label `L` (`corrupt.py`): per 3D component, one
   of keep / dilate / erode / drop / shift; then with p ≈ 0.5 paste 1–3 hallucinated blobs at high-PET
   voxels outside `L` (p = 0.8 on a lesion-free patch, the most valuable state to train, since a single
   false-positive voxel zeroes a lesion-free case's Dice for the whole run). Additionally `P = ∅` with
   p = 0.20 and `P = L` with p = 0.10. The `P = ∅` share is kept at ≥ 0.2 because the container has no
   previous prediction at iteration 0; with the `k = 0` coupling, channel 4 is all-zero for ≈40 % of
   patches.
3. **Run the official protocol** `k` times (`scribble_sim.py`): `overseg = P & ¬L`, `underseg = ¬P & L`;
   call the official `simulate_scribble_from_label` on each with a strategy drawn from
   {centerline, random, boundary}; apply the official selection rule; append the chosen stroke and
   **erase the 3D error component it landed on**, so the next iteration targets the next-largest error
   exactly as the evaluation loop does across container calls.
4. **Perturb** ≈20 % of strokes ScribblePrompt-style (random breaking into pieces plus a smooth in-plane
   warp), as insurance for the clinician-scribble category where real strokes are longer and sloppier.
5. **Stamp** the clipped EDT of the tumor list into channel 2 and of the background list into channel 3,
   and write `P` into channel 4.

**Independent ("already satisfied") scribbles, p = 0.25.** The category-2 clinician scribbles were
collected against the *baseline* model's errors and are replayed unchanged to ours, so many will already
be satisfied by our prediction. With p = 0.25 the scribbles are therefore derived from an independently
sampled corruption `P′` while channel 4 still carries `P`, which forces the network to treat a scribble as
a **constraint to maintain**, not only as a change signal.

### Two things that are easy to get wrong

**Axis convention.** `simulate_scribble_from_label` iterates `for z in range(label_array.shape[2])` — it
expects the axial slice axis to be the **last** array axis. In the nnU-Net preprocessed array the axes are
`(axial, y, x)`, so feeding a patch in directly draws the "axial" stroke on a *sagittal* plane.
`scribble_sim.py` moves the slice axis to the end before every official call and maps the coordinates
back; `slice_axis` is a parameter, default 0. With the fix every stroke occupies exactly one axis-0 slice
(0 of 145 stroke groups spanned more than one in the benchmark runs); without it, strokes spanned up to 13.

**The 2-tuple quirk.** `simulate_scribble_from_label` returns `(coords, label_cls, size)` normally but
`([], False)` — two values — on an empty mask, which is why the official `interactive_loop.py` raises
whenever an error mask is empty. `simulate_scribble()` handles both arities and returns `([], 0)`.

### Where the transform sits, and why

```
KeepFirstChannelsTransform(2)      <- position 0
SpatialTransform, Gaussian noise/blur, brightness, contrast,
low-res, gamma x2, Mirror, RemoveLabelTansform(-1, 0)
InteractionSimulationTransform     <- here
DownsampleSegForDSTransform
```

The guidance must live on **exactly the grid of the label used in the loss**, and generating it after all
spatial transforms makes that true by construction. Generating it earlier and letting the channels be
warped along as extra image channels was rejected because `SpatialTransform` resamples image channels with
a cubic spline, which puts ringing and fractional boundaries into what should be a crisp binary
previous-prediction mask; the pre-spatial patch is also 30–80 % larger, so the simulation would cost
proportionally more. Placing it before `DownsampleSegForDSTransform` is mandatory: that transform replaces
`segmentation` with a list of tensors. The guidance channels consequently receive no intensity
augmentation, which is correct — they are not images.

## Weight surgery

`init_from_baseline.py` expands the first convolution from `(32, 4, 3, 3, 3)` to `(32, 5, 3, 3, 3)`:
channels 0–3 are copied from the baseline (2 and 3 scaled by `guidance_scale`) and channel 4 is zero. The
tensor appears **four** times in the state dict — `encoder.stages.0.0.convs.0.conv.weight`,
`…convs.0.all_modules.0.weight`, and both again under `decoder.encoder.…` because the decoder holds a
reference to the encoder — and all four must be patched, or `load_pretrained_weights` aborts on a shape
mismatch. Optimizer and grad-scaler state are dropped and `current_epoch` is reset, so the output is a
valid `-pretrained_weights` source.

`guidance_scale` defaults to **1.0**. Rescaling to reproduce the baseline's *peak* activation would be
wrong: matching a ≈2500-magnitude single-voxel spike needs a ×2500 factor, which would make the first
layer's guidance response ~500× its CT/PET response and destroy the pretrained InstanceNorm statistics on
the first step. The right reference is the image channels, which have unit scale after normalisation: the
RMS response to PET is `‖w_{c,1}‖₂`, and for a guidance channel saturated at 1 over the kernel support it
is `‖w_{c,2}‖₂`. The baseline's per-channel weight RMS values are 0.230 / 0.384 / 0.207 / 0.235, already
comparable, so scale 1.0 is activation-matched. `--guidance-init kaiming` and `--guidance-scale` A/B it.

## Trainer

`nnUNetTrainer_Interactive` subclasses `nnUNetTrainer`. Every parameter below is overridable through the
corresponding environment variable, and the epoch count also through the `_50epochs` / `_100epochs` /
`_150epochs` / `_250epochs` trainer variants, which record the choice in the results folder name.

| | value | environment override |
|---|---|---|
| epochs | 200 | `nnUNet_interactive_epochs` |
| initial LR | 1e-3 (fine-tune; the baseline used 1e-2) | `nnUNet_interactive_lr` |
| optimizer | SGD, momentum 0.99 nesterov, wd 3e-5, PolyLR (nnU-Net default) | |
| loss | DC+CE with Dice smoothing term 0 | `nnUNet_interactive_noSmooth` |
| scribble count | `k ∈ {0..5}`, `[.28,.22,.18,.14,.10,.08]` | `nnUNet_interactive_k_probs` |
| EDT radius / slice axis | 10 voxels / 0 | `nnUNet_interactive_radius`, `..._slice_axis` |
| stroke perturbation / independent scribbles | 0.2 / 0.25 | `nnUNet_interactive_p_perturb`, `..._p_independent` |
| pretrained weights | — | `nnUNet_interactive_pretrained`, or `-pretrained_weights` |

`smooth=0` cannot produce a NaN here: with `batch_dice=True` the denominator of
`MemoryEfficientSoftDiceLoss` is a sum over the whole batch of softmax outputs and is never exactly 0.
`perform_actual_validation` is skipped by default (`nnUNet_interactive_skip_final_val=0` forces nnU-Net's
version): it runs sliding-window inference over the *stored* preprocessed cases, which do not carry
channels 2–4. The real number is the 6-iteration AUC from `src/interactive_eval.py`. Validation *during*
training uses the same interaction transform, so the reported pseudo-Dice is a noisy interactive proxy.
On resume (`--c`), `plans["continue_training"]` is read before `nnUNetTrainer.__init__` pops it, and
`nnUNet_interactive_pretrained` is then ignored so the checkpoint on disk wins.

## Metric-aligned loss terms (`nnUNetTrainer_InteractiveV2`)

`nnUNetTrainer_InteractiveV2` adds two optional terms to the compound loss. Both default to weight
0, so the class with no environment variables set is the same loss as `nnUNetTrainer_InteractiveB6`;
the named subclasses `…V2_negfp`, `…V2_blob` and `…V2_both` switch them on and give each variant its
own results folder. Both terms are evaluated on the **full-resolution** head only — the deep
supervision pyramid keeps the unmodified DC+CE.

### A — lesion-free patch penalty (`LesionFreeFPLoss`)

The challenge scores a lesion-free case 1.0 for an empty prediction and 0.0 for *any* false
positive, and excludes it from the lesion-level F1 altogether. A false positive on such a case is
therefore the most expensive single error available — and it is the error the configured loss is
blind to. With `batch_dice=True` and `smooth=0` (the baseline's setting and ours), a batch in which
no sample contains a label voxel has `sum_gt = 0`, so

```
dc = (2·0 + 0) / (0 + sum_pred + 0).clamp_min(1e-8)      = 0
```

and its gradient with respect to the logits is **identically zero**; the only remaining signal is
cross entropy averaged over ~2.3·10⁶ easy background voxels. `test_v2_loss.py` asserts this
(`|∇|₁ = 0.000e+00`). The term restores a gradient for that state as the soft false-positive volume
of an empty patch,

```
L_A = s / (s + c),        s = Σ p_fg over the patch,   c in voxels (default 50 ≈ 0.6 mL)
```

which is `1 − Dice(prediction, ∅)` with the smoothing constant reinstated. It is bounded in [0, 1),
its gradient is largest exactly where the metric's step is (`s → 0`) and it is self-limiting there
because `∂p/∂logit → 0` as `p → 0`. Measured on a near-empty patch carrying a 50-voxel false
positive — the realistic state of a fine-tuned model on a lesion-free case — it delivers **48×** the
gradient L1 norm of DC+CE; at 400 false-positive voxels, 1.4×. In the opposite, grossly-wrong regime
it saturates by design and DC+CE takes over. `all_patches=True` extends it to the label-free part of
patches that do contain lesions; off by default, because there the pooled Dice already sees them.

### B — instance-wise (blob) Dice (`InstanceDiceLoss`)

DMM is component-count F1 at IoU ≥ 0.1, and in the validation set 42 % of the ground-truth lesions
are smaller than 1 mL while carrying 1–2 % of the lesion volume. A volume-overlap loss is almost
indifferent to them; the metric counts each one. Following blob loss (Kofler et al., *blob loss:
instance imbalance aware loss functions for semantic segmentation*, IPMI 2023; reference
implementation MIT-licensed), the Dice is computed **once per connected component** with every
*other* component masked out of the prediction, and averaged over components, so a 0.2 mL lesion and
a 200 mL lesion contribute equally.

The reference implementation loops over instances with full-size tensors. Here the identical
quantity comes from two scatter reductions over the instance-id map: with `x` the foreground
probability, `Iᵢ` the i-th component and `T = Σ x` over the patch,

```
interᵢ    = Σ_{Iᵢ} x
pred_sumᵢ = T − (Σⱼ interⱼ − interᵢ)          ← the other blobs, masked out
diceᵢ     = (2·interᵢ + s) / (pred_sumᵢ + |Iᵢ| + s)
L_B       = 1 − mean_i diceᵢ
```

so the cost is independent of the number of lesions — which matters, since a single case in this
dataset carries 191. `test_v2_loss.py` checks the vectorised form against a literal masked loop to
1e-4. Components are 18-connected, matching the metric. `smooth` defaults to 1.0 voxel: unlike the
pooled Dice, which we deliberately run at `smooth=0`, a per-instance Dice over a 5-voxel lesion is
unusable without one. Samples with an empty label contribute nothing (that state is term A's job),
which is also the reference behaviour.

Illustration from the tests: a patch with one 1728-voxel lesion predicted perfectly and one 8-voxel
lesion missed entirely scores pooled Dice **0.9977** and instance-Dice loss **0.4444**.

### Knobs

| | class attribute | environment variable | default |
|---|---|---|---|
| lesion-free weight | `W_LESION_FREE` | `nnUNet_v2_lesionfree_weight` | 0 |
| its `c`, voxels | `LESION_FREE_SMOOTH_VOXELS` | `nnUNet_v2_lesionfree_smooth_vox` | 50 |
| apply to non-empty patches too | `LESION_FREE_ALL_PATCHES` | `nnUNet_v2_lesionfree_all_patches` | false |
| instance-Dice weight | `W_BLOB` | `nnUNet_v2_blob_weight` | 0 |
| its smoothing term | `BLOB_SMOOTH` | `nnUNet_v2_blob_smooth` | 1.0 |
| connectivity | `BLOB_CONNECTIVITY` | `nnUNet_v2_blob_connectivity` | 18 |
| average over the k smallest components only (0 = all) | `BLOB_SMALLEST_K` | `nnUNet_v2_blob_smallest_k` | 0 |
| ignore components below this size | `BLOB_MIN_VOXELS` | `nnUNet_v2_blob_min_voxels` | 1 |

`InteractiveV2Loss.last_terms` carries the three scalars (`base`, `lesion_free`, `blob`) of the last
forward pass, for logging.

### Cost

Measured at the plans patch size `(2, 2, 112, 160, 128)`, forward + backward of the term alone, on an
NVIDIA L4 that was not idle — so these are upper bounds: **term A 10.5 ms**, **term B 108 ms**
(on CPU at `nice 19` on the A100 host: 33 ms and 66 ms, which brackets the term-B overhead at roughly
2× term A rather than 10×). Term B breaks down as
one `cc3d` pass (4.3 ms per sample at 18-connectivity), one device→host copy of the label
(1.6 ms per sample) and the two scatter reductions plus their backward (~24 ms per sample). Only the
foreground voxels' linear indices cross the bus, not the full id map, and the two-class foreground
probability is written `sigmoid(l₁ − l₀)` rather than materialising a `(b, C, …)` float32 softmax —
37 MB instead of 150 MB per call. Against the measured 236 ms per optimizer step of the batch-2
configuration, term B is therefore worth budgeting for (~+25 s on a 59 s epoch) and term A is free.

`python -m train.test_v2_loss [--cuda]` runs the whole check list: finiteness and range on random,
empty and perfect patches; gradient flow; the zero-gradient pathology above; equality with the
reference blob-loss loop; size invariance; drop-in behaviour over a deep-supervision output list;
and the timings. `make_synthetic_dataset.py` plus a 2-epoch `nnUNetv2_train` run exercises all four
trainer variants end to end.

## Running it

nnU-Net ≥ 2.8 has an external trainer search path, so no site-packages surgery is needed:

```bash
export nnUNet_raw=... nnUNet_preprocessed=... nnUNet_results=...
export nnUNet_extTrainer=$PWD/src/train
export PYTHONPATH=$PWD/src:$PYTHONPATH
export AUTOPETV_REPO=/path/to/autoPETV      # for the official scribble simulator

# once: interactive plans + 5-channel dataset.json, and the 4 -> 5 channel surgery
python -m train.make_plans --out-dir $nnUNet_preprocessed/Dataset998_AutoPETV \
    --baseline-plans $nnUNet_preprocessed/Dataset998_AutoPETV/nnUNetPlans.json \
    --baseline-dataset-json $nnUNet_preprocessed/Dataset998_AutoPETV/dataset.json \
    --data-identifier nnUNetPlans_3d_fullres --dataset-name Dataset998_AutoPETV --num-training 1611
python -m train.init_from_baseline \
    --src <baseline>/fold_0/checkpoint_final.pth --dst weights/interactive_init_5ch.pth

export nnUNet_interactive_pretrained=$PWD/weights/interactive_init_5ch.pth
nnUNetv2_train Dataset998_AutoPETV 3d_fullres 0 \
    -tr nnUNetTrainer_Interactive -p nnUNetPlans_interactive
```

`recursive_find_trainer_class_by_name` imports every module in `nnUNet_extTrainer`, so all of
`src/train/*.py` must be import-safe — every script's work is behind `if __name__ == "__main__"`.

Write the interactive plans into a preprocessed root that is *separate* from the store: nothing may be
added inside `nnUNetPlans_3d_fullres/`, because `infer_dataset_class` asserts a single file extension in
there, and the store must keep its own 4-channel `dataset.json`. An evaluator pointed at a trained model
folder needs `plans.json`, `dataset.json` and `dataset_fingerprint.json` next to `fold_0/`, plus
`nnUNet_extTrainer` on the environment, because the checkpoint records
`trainer_name = nnUNetTrainer_Interactive` and nnU-Net has to import it.

## B6 continuation

`nnUNetTrainer_InteractiveB6` continues the fine-tune above with the interaction distribution re-fitted to
the measured deficit of the first run, which is **prompt-gated**: a large share of the lesion-bearing cases
scores Dice 0 at iteration 0 and only recovers once scribbles arrive. Two knobs of the interaction
distribution change, plus the schedule the continuation runs on.

| | `nnUNetTrainer_Interactive` | `nnUNetTrainer_InteractiveB6` |
|---|---|---|
| `K_PROBS` over `k ∈ {0..5}` | `[.28, .22, .18, .14, .10, .08]` | `[.36, .24, .16, .11, .08, .05]` |
| `P_INDEPENDENT_SCRIBBLES` | 0.25 | 0.35 |
| epochs / initial LR | 200 / 1e-3 | 120 / 5e-4 |

`k = 0` and `k = 1` *are* iterations 0 and 1 of the evaluation loop, which carry trapezoid weights 0.5 and
1.0, so they get more of the gradient budget; `k = 5` keeps a non-zero share so the many-scribble regime is
still trained. The independent ("already satisfied") share rises because the category-2 clinician scribbles
were collected against a different model, so a growing fraction of them is already satisfied by our own
prediction and has to be read as a constraint to maintain. Radius, slice axis, stroke perturbation,
corruption model, loss, batch size and `save_every = 5` are inherited unchanged, so the two runs differ in
exactly these knobs.

It is a **continuation, not a `--c` resume**: the weights are loaded from the first run's
`checkpoint_final.pth` through `nnUNet_interactive_pretrained`, while the optimizer, the grad scaler and the
PolyLR schedule start fresh at epoch 0 over 120 epochs. The class overrides `initialize()` to load that
checkpoint with `load_state_dict(..., strict=True)` instead of nnU-Net's `load_pretrained_weights`, which
skips every key containing `.seg_layers.` — correct when adapting a *foreign* pretraining whose output layer
predicts different classes, and wrong here, where re-initialising the segmentation heads would throw away
the calibration of the very model being continued. The architecture is identical, so the exact load is safe;
a mismatch falls back to nnU-Net's loader. The separate trainer class gives it its own results
folder, `Dataset998_AutoPETV/nnUNetTrainer_InteractiveB6__nnUNetPlans_interactive__3d_fullres`, so the first
run stays reproducible, the checkpoint mirror keeps the two models apart, and both can be evaluated side by
side. An evaluator needs `nnUNet_extTrainer` pointing at `src/train` as before — the checkpoint now records
`trainer_name = nnUNetTrainer_InteractiveB6`, which is defined in the same module as the base trainer.

`scripts/env/train_b6.sh` is the launcher. It refuses to start while another trainer or an inference job holds
the GPU (`--wait` polls every 5 min instead), verifies the staged store file-by-file count and the source
checkpoint's 5-channel first convolution, and bakes the environment into a generated launcher script,
because a new `tmux` session inherits the environment of the tmux *server* rather than of the caller. It
also unsets `nnUNet_interactive_k_probs`, `..._p_independent`, `..._epochs` and `..._lr`, so a stale export
cannot silently override the values the class defines.

```bash
JOBS=2 bash scripts/env/train_resume.sh --stage-only   # once per runtime: local copy of the store
bash scripts/env/train_b6.sh --wait                    # launch as soon as the GPU is free
```

`JOBS` is the parallelism of the staging copy and defaults to 4. Against a **cold** FUSE content cache, 4
concurrent readers of a 4833-file directory are enough to wedge the mount daemon: it pins a core, stops
answering, and every access to the mount — including unrelated jobs' — blocks in uninterruptible sleep.
Use `JOBS=2` for the first staging of a fresh runtime; 4 is only safe once the cache is warm.

Resume after a runtime loss — restage the store, then resume from the last checkpoint of the B6 folder
(`--resume` pulls `checkpoint_latest.pth` back from the mirror when the local folder is empty, never
`checkpoint_final.pth`, which `--c` would prefer over it):

```bash
bash scripts/env/train_resume.sh --stage-only
bash scripts/env/train_b6.sh --resume
```

### Running other trainers of the same family, and queueing them

`TRAINER=<class> TAG=<short-name>` selects any other trainer that subclasses
`nnUNetTrainer_InteractiveB6`. `TAG` names the log (`train_$TAG.log`), the progress file
(`progress_$TAG.txt`) and both tmux sessions (`train_$TAG`, `${TAG}progress`), so runs never collide; the
results folder and the checkpoint-mirror folder follow the trainer class name automatically, and the
progress watcher is handed `MODEL_NAME=<TRAINER>__nnUNetPlans_interactive__3d_fullres`, from which it
derives the mirror path. The source weights (`INIT`) are shared across the family — every one of these runs
continues from the same first-fine-tune `checkpoint_final.pth`.

`ALLOW_BUSY_GPU=1` starts while another job holds the GPU. That is a deliberate choice, not a default: this
training is dataloader-bound (~45 % GPU, 8.2 GiB of 80), so it coexists with an inference job, but both get
slower and the two contend for CPU rather than for VRAM. Pair it with `NICE=5` so the other job keeps CPU
priority. Measured cost of co-residency with a sliding-window inference job on the same box: **66.5 s/epoch
against 59.3 s solo, ≈ +12 %**.

```bash
ALLOW_BUSY_GPU=1 NICE=5 TAG=b10 TRAINER=nnUNetTrainer_InteractiveV2_negfp bash scripts/env/train_b6.sh
```

Only one training may run at a time on one GPU; the launcher enforces that (it refuses while an
`nnUNetv2_train 998` process exists). To chain runs, poll the previous `train_$TAG` session and launch the
next when it is gone — waiting for the *process* as well as the session, because the tmux session can
disappear a moment before the process group does.

Each run's progress watcher is started by the launcher; run it by hand like this:

```bash
MODEL_NAME=nnUNetTrainer_InteractiveB6__nnUNetPlans_interactive__3d_fullres \
PROGRESS_FILE=<work>/progress_b6.txt TOTAL_EPOCHS=120 python3 -u scripts/env/progress_watch.py
```

At the measured 59 s/epoch plus the one-off `torch.compile`, 120 epochs is ≈2 h alone, ≈2 h 15 min
alongside an inference job.

## Cost

A100-SXM4-80GB (12 vCPU), torch 2.11 + cu128, nnU-Net 2.8.1, `torch.compile` on, patch 112×160×128, 250
training + 50 validation iterations per epoch, fold 0 (1288 train / 323 val).

| batch | epoch 0 (compile) | steady epoch | per sample | peak VRAM | mean GPU util |
|---|---:|---:|---:|---:|---:|
| 2 | 104.3 s | **59.3 s** | 0.119 s | 8.2 GiB | 43 % |
| 4 | 166.9 s | 123.6 s | 0.124 s | 19.2 GiB | 45 % |

Doubling the batch doubles the epoch time exactly and leaves per-sample cost and GPU utilization
unchanged: the run is **dataloader-bound**, not GPU-bound. Use batch 2 — it matches the batch size the
baseline checkpoint was trained at and gives twice as many optimizer steps per hour. 200 epochs is ≈3.3 h.
The bottleneck is nnU-Net's own CPU augmentation, not our transform: the interaction simulation costs
99 ms mean / 53 ms median / 249 ms p90 per patch (`bench_transform.py`) against a per-sample budget of
~1.19 s. `KeepFirstChannelsTransform` is what keeps it there — without it `SpatialTransform` would
spline-resample the two all-zero placeholder channels.

The preprocessed store must be on **local disk**. Read randomly, one blosc2 chunk per patch, it is the
worst possible access pattern for a network filesystem; over a FUSE-mounted cloud drive epoch 0 did not
finish in 13.5 minutes at 0.5 % GPU utilization, against 104 s and 43 % from local disk.

## Checks

`bench_transform.py` measures containment on 40 real patches per configuration (fold 0). In the
error-driven setting every tumor stroke voxel lies inside the label and on a false-negative region and
every background stroke voxel outside it and on a false-positive region. With `--p-independent 1` the
strokes stay correct as absolute annotations but about half the tumor strokes and all the background
strokes are already satisfied by channel 4 — the category-2 regime. In the production setting stroke
perturbation moves 6 % of tumor stroke voxels off the label, which is the intended realism. No stroke in
any of the runs spanned more than one axial slice.

End to end, `nnUNetv2_train` runs the trainer on a synthetic blosc2 store (`make_synthetic_dataset.py`),
loads the surgered checkpoint, trains, resumes with `--c` and writes all three checkpoints. On the real
store every batch is `(2, 5, 112, 160, 128)` with channels 2–3 in [0, 1] and channel 4 in {0, 1}, and
validation pseudo-Dice is 0.73–0.76 from the first epochs — the weight surgery preserves the baseline's
competence rather than restarting from it.

## Figures

Real preprocessed patches straight out of the training dataloader (`viz.py`).

![tumor scribble on a missed lesion](img/train_interaction_fg_on_fn.png)
*Tumor scribble on a missed lesion: the previous prediction (magenta) misses the left lesion (blue = FN)
and over-segments the right one (red = FP).*

![background scribble on the largest false-positive component](img/train_interaction_bg_on_fp.png)
*More FP than FN voxels, so the official rule places a background scribble (yellow) on the largest
false-positive component.*

![an already-satisfied scribble](img/train_interaction_satisfied_scribble.png)
*Category-2 state: the previous prediction is already correct, yet a background scribble arrives.*

![lesion-free patch with a hallucinated component](img/train_interaction_negative_patch.png)
*Negative-case state: no lesion in the patch, one hallucinated component on physiological uptake.*

## Architecture variants

Two continuations of the B10 checkpoint (`nnUNetTrainer_InteractiveV2_negfp`) that change the
**network** and nothing else: same store, same interaction distribution, same loss, same 120-epoch
schedule at lr 5e-4. nnU-Net builds the network from `configurations.<cfg>.architecture` —
`network_class_name` resolved with `pydoc.locate`, `arch_kwargs` handed to the constructor — so a
variant is a plans file plus a class, not a fork of the trainer.

| | class | plans | trainer | added parameters |
|---|---|---|---|---|
| B13 | `train.networks.GlobalContextUNet` | `nnUNetPlans_b13` | `nnUNetTrainer_InteractiveB13` | **+0.48 M (+1.6 %)** |
| B14 | `train.networks.EditBranchUNet` | `nnUNetPlans_b14` | `nnUNetTrainer_InteractiveB14` | **+0.28 M (+0.9 %)** |
| C0 | `PlainConvUNet` (unchanged) | `nnUNetPlans_interactive` | `nnUNetTrainer_InteractiveC0` | 0 |

C0 is the control: the same continuation with no architectural change, so a variant's delta measures
the block rather than the extra epochs.

`train.networks` resolves because `src` is on `PYTHONPATH` both on the training box and inside the
container (`PYTHONPATH=/opt/algorithm:/opt/algorithm/src`, `src/` is copied into the image), and the
trainer writes its `plans.json` next to `fold_0/`, so the predictor rebuilds the same class from the
model folder with no extra wiring.

### B13 — global-context bottleneck

A residual transformer block on the **deepest** encoder feature map. At the plans patch the
bottleneck is `7×5×4 = 140` tokens of 320 features (the last stage's stride is `[1, 2, 2]`, not
`[2, 2, 2]`), so whole-patch context costs a few hundred microseconds. The rationale is the pair of
decisions a purely convolutional receptive field cannot make locally: physiological uptake versus
lesion, and "this patch contains nothing".

The block follows EVA-02 (arXiv:2303.11331): pre-norm layers with **sub-LN** — an extra `LayerNorm`
on the attention and feed-forward outputs before their last projection — a **SwiGLU** feed-forward of
hidden width `2/3 · 4d`, and **3D rotary position embeddings** on the queries and keys instead of a
learned positional table, so the block carries no size-dependent parameter and transfers to any patch
shape. The `n_head_dim/2` rotation pairs are split across the three axes, and each axis rotates by its
own voxel coordinate. The stack runs at a reduced width (`context_dim = 128` against 320 features)
behind a 1×1 convolution, which is what keeps it under half a million parameters; a second 1×1
convolution projects back and is **zero-initialised**, making the whole block an exact identity at
initialisation while still receiving a non-zero gradient on the first step.

`context_type = "mamba"` swaps in a bidirectional selective state-space block (two `mamba_ssm` scans
over the forward and reversed token sequence, shared zero-initialised output projection) where
`mamba_ssm` is installed. It is not part of the shipped configuration.

### B14 — edit branch

The interaction currently enters only at the input, five convolutions away from the deepest features
and 32 upsamplings away from the output. B14 keeps the pretrained encoder and decoder as the
*automatic* branch and adds a second, lightweight decoder branch (16–32 features per stage) that sees
the guidance at **every** scale. At each decoder stage it concatenates its own upsampled state, a 1×1
projection of the encoder skip, and the three interaction channels max-pooled to that stage's grid,
then emits an **edit logit** that is added to the automatic logit. Deep supervision is applied to the
sum only; the branch is never supervised on its own.

Max pooling rather than averaging carries the guidance down: a scribble is a thin structure whose
*presence* must survive a 32× downsampling, and the clipped-EDT encoding is non-negative, so the
maximum over a pooling window is the value at the stroke.

The five edit segmentation heads are zero-initialised, so the fused logits are the automatic logits at
initialisation.

### Weight surgery and the epoch-0 gate

`networks.graft_state_dict` loads the source checkpoint into the variant with `strict=False` and then
raises unless **every** source tensor was consumed with a matching shape; the trainer additionally
requires that the tensors left at their initialisation are exactly those under the variant's declared
prefixes (`context.` for B13, `edit_*` for B14, none for C0). nnU-Net calls
`network.apply(network.initialize)` after construction, which would re-initialise the zeroed
convolutions with Kaiming; both classes override `initialize` to skip modules tagged during
construction.

That makes "epoch 0 is B10" checkable rather than assumed. `nnUNet_arch_refbatch` points at a file
holding one fixed input batch and stock B10's deep-supervision outputs for it (written by
`test_networks.py --emit-ref`); the trainer runs the grafted network on it at startup and aborts
unless the maximum absolute logit difference is below 1e-5. Measured on the real checkpoint at the
plans patch and batch, both variants reproduce B10 **exactly** (max |diff| `0.000e+00` at every deep
supervision scale, deep supervision on and off).

```bash
python -m train.make_arch_plans --base-plans $PREP/$DS/nnUNetPlans_interactive.json \
    --variant b13 --out $PREP/$DS/nnUNetPlans_b13.json
python -m train.test_networks --plans $PREP/$DS/nnUNetPlans_interactive.json \
    --checkpoint <B10 checkpoint_final.pth> \
    --emit-ref <work>/b10_ref_batch.pt --cuda --bench

nnUNet_arch_refbatch=<work>/b10_ref_batch.pt INIT=<B10 checkpoint_final.pth> \
TAG=b13 TRAINER=nnUNetTrainer_InteractiveB13 PLANS=nnUNetPlans_b13 \
    bash scripts/env/train_b6.sh
```

`train_b6.sh` gained `PLANS=` (a variant plans file) and `ALLOW_CONCURRENT_TRAIN=1` (a second trainer
on the same GPU; pair it with `NICE` and a reduced `nnUNet_n_proc_DA`, because the runs contend for
CPU, not for VRAM).

### Cost of the two blocks

Forward + backward at the plans configuration (batch 2, patch 112×160×128, fp16 autocast, A100-80GB),
`test_networks.py --bench`:

| network | parameters | ms / optimizer step | peak VRAM | GPU share of a 250-step epoch |
|---|---:|---:|---:|---:|
| B10 (`PlainConvUNet`) | 30.79 M | 125.1 | 5.18 GiB | 31.3 s |
| B13 | 31.27 M | 130.8 | 5.18 GiB | 32.7 s |
| B14 | 31.07 M | 186.1 | 5.93 GiB | 46.5 s |

Against the measured 59.3 s steady-state epoch of the batch-2 configuration — which is
**dataloader-bound**, not GPU-bound — B13 is free and B14 costs at most the 15 s of GPU time it adds.

## Known limitations

* **The corruption model is a proxy for the model's own errors.** True RITM-style iterative training (a
  forward pass per sample) would roughly triple an already CPU-bound epoch. If the fine-tuned model's
  failure modes differ from the sampled ones, channel 4 could be miscalibrated; the cheap mitigation is to
  dump its errors on a few validation cases and re-fit the corruption probabilities.
* **About half of all patches carry no scribble at all** (`k = 0`, or `k ≥ 1` with no error to point at),
  so the effective number of scribble-carrying gradient steps is roughly half the nominal. At test time an
  individual sliding-window patch sees a scribble even less often, so the bias is in the right direction.
* **`R = 10` is isotropic in voxel units**, i.e. 30 mm through-slice against 20 mm in-plane. The transform
  accepts `spacing=` to switch to physical distance; untested and not exposed by the trainer.
* **`guidance_scale = 1.0` is an analytical choice, not a measured one.**
* **Validation pseudo-Dice is noisy** because the interaction is resampled every time a validation patch
  is drawn. It is a trend indicator only.
