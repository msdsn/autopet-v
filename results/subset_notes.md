**Reading the two `X1` replay rows.** Their `Δ` is *not* a model regression — it is the price of
the challenge's Category-2 protocol. The organizers collect one scribble sequence against the
**baseline's** predictions and replay it unchanged to every algorithm, so a scribble may sit on a
voxel we already segment correctly and no scribble ever targets our own worst error. Both rows
replay the sequence recorded from `B0_interactive_final`. Their controls are the same pipelines on
the same 30 cases with scribbles simulated from their *own* errors, so `Δ` measures exactly how much
of each pipeline's interactive gain depends on the scribbles being adapted to it. Everything else —
model, checkpoint, post-processing configuration, case list — is identical.

**Excluded subset runs.**

* `runs/B10g_nostate_sub39` (2026-08-29T02:01 UTC) is **invalid and must not be quoted**. It was
  meant to be the no-state probe for the shipping configuration, but its `postproc_config` records
  `pass_cached_prev_pred: true` — identical to `B10g9_ship` — so the post-processing layer kept
  feeding the model its own cached mask and zeroing `prev_pred` changed nothing. Its AUC-Dice of
  4.133 on those 39 cases is simply the shipping configuration's number, not a no-state number.
  The corrected row (`prev_pred` zeros **and** `pass_cached_prev_pred=false`) will be added when it
  lands.
* `runs/B7B10_negfp_tta_sub30` is **cut** — the 8-axis TTA run was killed before it finished and
  will not be resumed.
* `runs/E1_ens_b10_b6_sub30` (B10 + B6 ensemble) is **partial** — the L4 box died mid-run, so it has
  `case_info.json` and `metric_scores.json` for the cases it reached but no `run.json` and no
  summary. It is left in place untouched and will be re-run on the A100; do not score the partial
  files.

`B3nostate_sub39` is *not* affected by the `pass_cached_prev_pred` bug: its `postproc_config`
records `pass_cached_prev_pred: false`, so it is a genuine no-state measurement and is tabled above.
