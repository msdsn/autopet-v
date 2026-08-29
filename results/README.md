# `results/` — one folder per named ablation row

This folder holds the run records of every completed evaluation row: the metric files written by
the evaluation loop, copied verbatim from the run directories, plus the generated tables.

Each `<ROW>/` contains exactly the four decision-relevant files of the harness
(see `docs/eval_harness.md` in the repo):

| file | what it holds |
|---|---|
| `run.json` | the headline numbers (AUC-Dice, AUC-DMM, per-iteration means, `by_lesion_status`, `by_tracer`), the full argument set, git commit, seed, timing and cache statistics. **This is the file `../results_index.py` reads.** |
| `summary.json` | the same aggregation with the stratified breakdowns, guarantee counters and `empty_error_region_exposure` |
| `metric_scores.json` | per-case, per-iteration Dice and DMM in the official Grand Challenge schema |
| `case_info.json` | per-case geometry, per-iteration seconds, FP/FN component counts |

Predictions, softmax caches and logs are **not** copied here — they are large and reproducible.

## The rows

| row | in one line | model |
|---|---|---|
| `A0` | Official baseline as shipped: 4-channel nnU-Net, z-scored scribble spikes, no post-processing, no TTA. The control for the whole A ladder. | `../weights/nnUNet_results/Dataset998_AutoPETV/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_0/checkpoint_final.pth` |
| `A1` | A0 + background-scribble compliance (delete/split the component a background scribble lands in). | same as A0 |
| `A2` | A1 + tumor-scribble compliance (SUV-adaptive region growing from a tumor scribble). | same as A0 |
| `A3` | A2 + component cleanup (small components and hole filling, SUV floor off). **The post-processing setting that carries forward as "A3 post-processing".** | same as A0 |
| `A4` | A3 + tracer SUV floor (FDG 1.5 / PSMA 1.0). No measurable effect on this set. | same as A0 |
| `A5` | A4 + negative gate v1, `max_prob = 0.60`. | same as A0 |
| `A5a` | A5 with a stricter gate, `max_prob = 0.45`. | same as A0 |
| `A5b` | A5 with a looser gate, `max_prob = 0.75`. | same as A0 |
| `A5c` | A5 with a very loose gate, `max_prob = 0.90`. | same as A0 |
| `A9` | A3 + **DMM cleanup rule v2**: a second cleanup pass *after* compliance that drops a component if `SUVmax < 5` **or** `mean softmax < 0.8`. Best row of the A ladder on both metrics (+0.017 AUC-Dice, +0.114 AUC-DMM over A3). | same as A0 |
| `B0` | Fine-tuned interactive 5-channel nnU-Net (CT, PET, fg guidance, bg guidance, previous mask), `checkpoint_final`, no post-processing, no TTA. The control for the whole B family. | `../ckpt/Dataset998_AutoPETV/nnUNetTrainer_Interactive__nnUNetPlans_interactive__3d_fullres/fold_0/checkpoint_final.pth` |
| `B3` | B0 + A3 post-processing (bg + fg compliance, cleanup; SUV floor off, gate off, monotone blending off). | same as B0 |
| `B3g` | B3 + **negative gate v2**: empty the prediction when the total predicted volume is below 6 mL (`max_total_volume_ml = 6.0`, every other gate criterion off). The gate turns B0's worst weakness — 2.778 AUC-Dice on lesion-free cases — into 4.722. | same as B0 |
| `B9` | B3 + **DMM cleanup rule v2** (the A9 rule on the fine-tuned model; no gate). **Best AUC-DMM of every row, 3.824**, and the best positives-only AUC-Dice, 3.858; but with no gate the negatives stay at 2.778, so the 50/50 score is 3.647. | same as B0 |
| `B6g` | The **B6** checkpoint (k-reweighted continuation) under the shipping pipeline (A3 post-processing + gate v2 at 6 mL). 50/50 3.716 — *below* `B3g`, i.e. the k-reweighting does not pay once the gate is in place. | `../ckpt/…/nnUNetTrainer_InteractiveB6__…/fold_0/checkpoint_final.pth` |
| `B10g` | The **B10** checkpoint (lesion-free false-positive penalty) under the same shipping pipeline. | `../ckpt/…/nnUNetTrainer_InteractiveV2_negfp__…/fold_0/checkpoint_final.pth` |
| `B10g9_ship` | The B10 checkpoint under the **shipped** post-processing configuration (`submission/postproc_config.json`: gate v2 6 mL, cleanup rule v2 after compliance, compliance, monotone minmax) **with the compliance-collapse fix**. **Current shipping candidate — see below.** | same as `B10g` |
| `B10g9_gate25` | Exactly `B10g9_ship` with the gate threshold raised 6 mL → 25 mL, and nothing else. Best AUC-Dice of any row (4.224) and a **perfect 5.000 on lesion-free cases**, but −0.052 AUC-DMM. See the trade-off note below. | same as `B10g` |
| `B12g` | The **B12** checkpoint (lesion-free penalty *and* instance-wise blob Dice) under the A3 + gate pipeline. Adding the blob term on top of B10 **loses**: 50/50 3.650 against B10g's 3.835. | `../ckpt/…/nnUNetTrainer_InteractiveV2_both__…/fold_0/checkpoint_final.pth` |
| `B3g25` | `B3g` with the gate threshold raised 6 mL → 25 mL. Pre-fix code and the A3 configuration, so it is a one-variable comparison **against `B3g` only** — not against the shipping candidate. | same as B0 |

Three further rows are **subset rows** — evaluated on a stratified 30- or 39-case subset, so their
absolute numbers mean nothing on their own and they live in their own table in `../RESULTS.md`
against a control recomputed on the identical case list:

| row | n | question it answers | paired result |
|---|---:|---|---|
| `B0b_sub30` | 30 | is B's `checkpoint_best` (epoch 157) better than `checkpoint_final` (epoch 200)? | **no**, −0.133 AUC-Dice and −0.320 AUC-DMM against `B0` |
| `B10gbest_sub30` | 30 | is B10's `checkpoint_best` (epoch 99) better than `checkpoint_final` (epoch 120)? | **no**, −0.011 AUC-Dice and −0.067 AUC-DMM against `B10g` |
| `B3nostate_sub39` | 39 | how much is lost if the container cannot persist state between calls (`prev_pred` forced to zeros, `pass_cached_prev_pred=false`)? | **nothing is lost — it gains**: +0.183 AUC-Dice, −0.115 AUC-DMM, +0.034 on 50/50 against `B3` |
| `B9ens_sub30` | 30 | does averaging B's `checkpoint_final` and `checkpoint_best` beat `checkpoint_final` alone? | **no**, −0.035 AUC-Dice and −0.183 AUC-DMM against `B0` — consistent with `B0b_sub30`, since the ensemble is half made of the checkpoint that already lost |
| `X1_sub30` | 30 | what does the challenge's Category-2 scribble replay cost the **B3** pipeline? | −0.014 AUC-Dice, −0.100 AUC-DMM |
| `X1B10_sub30` | 30 | what does the same replay cost the **shipping** pipeline? | **−0.256 AUC-Dice, −0.337 AUC-DMM** — see the warning below |

Two readings follow, and both matter for the freeze:

* **`checkpoint_final` is the right checkpoint for both models.** The best-vs-final ambiguity flagged
  in `../models/README.md` is now measured, on paired case lists, and `checkpoint_final` wins in both
  runs. The EMA pseudo-Dice that picks `checkpoint_best` is a patch-level proxy; the 6-iteration AUC
  is the real objective and it disagrees.
* **The previous-prediction channel is not carrying its weight on `B3`.** Zeroing it *raises*
  AUC-Dice by 0.183 on the same 39 cases. That bounds the container's state-persistence risk at
  essentially zero for that configuration, and it is a hint worth chasing — but it is a `B3`
  measurement, not a shipping-candidate one. The equivalent probe on the shipping configuration is
  the run described below, which has to be redone.

**A5/A5a/A5b/A5c are all identical to A4** in every aggregate: the v1 gate never fired. The
measured reason is in the experiment log — `max_prob ≥ 0.5` holds for every predicted voxel by
construction (a predicted voxel *is* the softmax argmax), so sweeping that threshold could not
change anything, and `max_suv = 3.0` had the wrong sign. The redesigned gate is a pure volume
criterion, and it is row **`B3g`** — the row that finally converts the lesion-free headroom.

### Current shipping candidate: `B10g9_ship`

| | AUC-Dice | AUC-DMM | 50/50 | neg AUC-Dice | pos AUC-Dice | Dice@0 | Dice@5 |
|---|---:|---:|---:|---:|---:|---:|---:|
| A0 (baseline control) | 3.635 | 3.222 | 3.428 | 3.611 | 3.649 | 0.500 | 0.767 |
| B3g (B model + gate) | 4.003 | 3.508 | 3.755 | 4.722 | 3.598 | 0.678 | 0.809 |
| B6g (B6 model + gate) | 3.981 | 3.452 | 3.716 | 4.444 | 3.720 | 0.661 | 0.821 |
| B12g (B12 model + gate) | 3.834 | 3.466 | 3.650 | 4.583 | 3.413 | 0.645 | 0.788 |
| B10g (B10 model + gate) | 4.097 | 3.572 | 3.835 | 4.722 | 3.745 | 0.664 | 0.845 |
| **`B10g9_ship`** | **4.142** | **3.585** | **3.863** | 4.722 | **3.815** | 0.664 | **0.867** |

`B10g9_ship` is the best completed row on AUC-Dice, on AUC-DMM and on the 50/50 score: **+0.435 on
the 50/50 score over the A0 baseline**. Three one-variable readings, each holding everything else
fixed:

* **The shipped configuration and the compliance fix are worth +0.028 on 50/50.** `B10g9_ship` −
  `B10g` is the *same checkpoint*, differing only in the post-processing config (cleanup rule v2
  after compliance + monotone minmax blending) and the compliance-collapse fix: +0.045 AUC-Dice,
  +0.013 AUC-DMM, negatives identical at 4.722, all of the gain on positives (3.745 → 3.815).
* **It is the only `*g` row whose iteration curve never goes down** — see the per-iteration table in
  `../RESULTS.md`. Dice climbs 0.664 → 0.867 and DMM 0.439 → 0.798 across all six iterations, while
  `B3g` and `B10g` both dip at iteration 5. That is exactly what the monotone blending and the
  compliance fix are for, and it matters: the challenge scores the whole curve, not the endpoint.
* **B12 loses, so the loss-term ablation is settled.** `B12g` (both terms) is −0.263 AUC-Dice and
  −0.106 AUC-DMM against `B10g` (lesion-free penalty only), and its positives collapse from 3.745 to
  3.413. The plan's decision rule was to adopt B12 only if both terms won separately; the blob term
  does not, and stacking it costs more than it returns.

### Warning: the shipping pipeline is the most exposed to Category-2 replay

The challenge does **not** simulate scribbles from our own errors. It collects one sequence against
the *baseline's* predictions and replays it unchanged to every algorithm, so a scribble can land on
a voxel we already segment correctly and none of them ever targets our worst error. `X1_sub30` and
`X1B10_sub30` measure that, replaying the sequence recorded from `B0_interactive_final` into the B3
and the shipping pipelines on the same 30 cases:

| pipeline | Δ AUC-Dice | Δ AUC-DMM | Δ Dice@0 | Δ Dice@5 |
|---|---:|---:|---:|---:|
| B3 (`X1_sub30`) | −0.014 | −0.100 | +0.000 | −0.012 |
| **shipping (`X1B10_sub30`)** | **−0.256** | **−0.337** | +0.000 | **−0.069** |

`Δ Dice@0` is exactly zero in both — iteration 0 has no scribbles, so the models are untouched and
the whole effect is in iterations 1–5. **The shipping pipeline gives up an order of magnitude more
than B3 does**, i.e. a large part of its interactive gain depends on the scribbles being aimed at
its own errors, which is precisely the assumption the real evaluation breaks. This is measured on 30
cases and needs the full set before it drives a decision, but it is the largest open risk to the
freeze and it argues for repeating the replay against the final candidate on all 100 cases.

### The 25 mL gate: better Dice, worse DMM, and the ranking decides

`B10g9_gate25` is `B10g9_ship` with one number changed:

| | AUC-Dice | AUC-DMM | 50/50 | neg AUC-Dice | Dice@5 |
|---|---:|---:|---:|---:|---:|
| `B10g9_ship` (6 mL) | 4.142 | **3.585** | 3.863 | 4.722 | 0.867 |
| `B10g9_gate25` (25 mL) | **4.224** | 3.533 | **3.878** | **5.000** | **0.888** |
| Δ | **+0.083** | **−0.052** | +0.015 | +0.278 | +0.021 |

The wider gate empties **every one of the 36 lesion-free cases correctly** — a perfect 5.000, the
first row to reach it — and pays for it by discarding some true positives, which costs DMM.
**Do not read the 50/50 column as the decision.** The challenge ranks by the *mean of the two
metrics' ranks*, not by their average value, so a +0.083 / −0.052 split is not automatically a win:
it depends on how tightly the leaderboard is packed on each metric. The two candidates should be
ranked under the official aggregation before either is frozen.

**Open lever: the gate threshold.** `B3g25` − `B3g` was the first 6 mL → 25 mL comparison, on the B
checkpoint: +0.112 AUC-Dice and negatives 4.722 → 4.861. `B10g9_gate25` now repeats it on the
shipping configuration with the same sign and the same trade against DMM, so the effect is
reproducible across checkpoints and code revisions. A 12 mL run is queued to find the knee. `B9` remains the other one: it has the best
AUC-DMM of any row (3.824) from cleanup rule v2 without a gate, and rule v2 is now *in* the shipped
config, so part of that gain is already banked in `B10g9_ship`.

**`A9` and `B9` have no `run.json`.** They were launched directly (`run_a9.sh` and the matching B9
script) instead of through the run wrapper, and only the wrapper writes `run.json`; A9 itself
finished cleanly (`EXIT=0`, 100 cases, `GUARANTEES 600/600 scored iterations satisfy G1+G2`) and B9
logged the same 100-case summary. Their numbers therefore come from
`summary.json`, which the evaluation loop writes itself and which carries the same aggregation,
the same `args` and the same `postproc_config`. `../results_index.py` falls back to `summary.json`
automatically and marks each row's source in the generated table; the one-line description lives in
`A9/label.txt` / `B9/label.txt` because there is no `label` field to read.
`A9/a9_vs_a3_compare.log` and `B9/b9_vs_b3_compare.log` hold the paired A3→A9 and B3→B9
comparisons on the same 100 cases. Nothing about these rows is reconstructed.

## Protocol (identical for every row above)

* **Cases**: the frozen 100-case subset in `valset_v1.txt` — 63 FDG / 37 PSMA, 64 lesion-bearing /
  36 lesion-free (composition in `valset_v1_composition.json`). Fold-0 validation cases only.
* **Loop**: our in-process re-implementation of the official interactive loop
  (`src/interactive_eval.py`), 6 iterations per case. Iteration 0 gets no scribbles; each later
  iteration adds exactly one scribble simulated from the *previous* prediction's largest error,
  using the challenge repo's own `simulate_scribble_from_label`. On a lesion-free case no scribble
  is ever added, so all six predictions are identical.
* **Strategies**: all three (centerline / random / boundary), assigned by deterministic round-robin
  over the sorted case list, `seed = 42`. The subsets used by the partial rows are built from
  aligned blocks of three consecutive cases, so every case keeps the strategy it had in the full run.
* **Metrics**: the official `dice_score` and `MetricEvaluator()['f1']` (lesion-level F1 at
  IoU ≥ 0.1, 18-connectivity), imported from the challenge repo, never re-implemented.
  `AUC = trapezoid(values, [0..5])`, **maximum 5.0**.
* **Semantics**: `--eval fixed` — a perfect prediction propagates to the remaining iterations
  instead of raising, which is what the organizers confirmed they will ship. `--eval buggy`
  reproduces the published behaviour and is not used for any row here.
* **AUC-DMM is a nanmean**: DMM is NaN on lesion-free cases, so they are excluded from it.
  A lesion-free case scores AUC-Dice 5.0 or 0.0 and nothing in between — a single false-positive
  voxel costs the whole case. This is why `neg AUC-Dice` is reported separately in `../RESULTS.md`.

## Other files here

| file | what it is |
|---|---|
| `valset_v1.txt` | the frozen case list, one case per line — the definition of "same cases" for every row |
| `valset_v1_composition.json` | its FDG/PSMA and positive/negative counts |
| `B_per_case.csv` | per-case AUC-Dice / AUC-DMM for the B rows and their A controls, the source of that analysis |

## Regenerating the table

`RESULTS.md` is generated by `results_index.py` from the `run.json` (or `summary.json`) files in the
row folders; run it from the directory that holds the evaluation output tree.
