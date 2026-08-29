#!/bin/bash
# Full post-processing ablation sweep over the 100-case evaluation set.
#
# Run it on the GPU box after the A0 control run has finished, e.g.:
#
#   bash scripts/run_ablation_sweep.sh > runs/ablate_sweep.log 2>&1
#
# Every variant shares CACHE_DIR with the control run, so a rung only pays for a
# forward pass where its own scribbles differ from every set already computed. Point
# CACHE_DIR at the control run's cache and pass the control in with --include_run.
#
# Environment overrides: RUN_ID, VARIANTS, CACHE_DIR, OUT_ROOT, A0_RUN, CASES_FILE,
# EVALSET, REPO, REPO_CODE, DRIVE.
set -u

REPO_CODE=${REPO_CODE:-/content/autopet}
EVALSET=${EVALSET:-/content/drive/MyDrive/autoPET/evalset}
REPO=${REPO:-/content/autoPETV}
CACHE_DIR=${CACHE_DIR:-/content/work/cache}
OUT_ROOT=${OUT_ROOT:-/content/work/runs}
DRIVE=${DRIVE:-/content/drive/MyDrive/autoPET/runs}
# Empty on purpose: letting the harness discover all 100 cases reproduces the control
# run's case order and therefore its strategy assignment. Set CASES_FILE to a
# one-tag-per-line file to run a subset.
CASES_FILE=${CASES_FILE:-}
A0_RUN=${A0_RUN:-/content/work/runs/A0_baseline_20260826}
RUN_ID=${RUN_ID:-$(date -u +%Y%m%d-%H%M)_A0-A7_postproc_ladder}
VARIANTS=${VARIANTS:-"A1 A2 A3 A4 A5 A5a A5b A5c A6 A7"}

# The GPU boxes do not all ship the same env file; the one thing that must always be
# set is the CUDA driver path, or torch silently falls back to CPU.
if [ -f /content/env.sh ]; then
  # shellcheck disable=SC1091
  source /content/env.sh
fi
export LD_LIBRARY_PATH=/usr/lib64-nvidia:${LD_LIBRARY_PATH:-}
export AUTOPETV_REPO=${AUTOPETV_REPO:-$REPO}
python3 -c "import torch; assert torch.cuda.is_available(), 'no CUDA device'" || {
  echo "CUDA is not available -- refusing to start a sweep that would run on the CPU"; exit 1; }

echo "disk before the sweep:"; df -h /content | tail -1

CASES=()
[ -n "$CASES_FILE" ] && CASES=(--cases_file "$CASES_FILE")

INCLUDE=()
if [ -f "$A0_RUN/summary.json" ]; then
  INCLUDE=(--include_run "A0=$A0_RUN")
  echo "control A0 taken from $A0_RUN (not recomputed)"
else
  echo "no finished control at $A0_RUN -- add A0 to VARIANTS to produce one"
fi

cd "$REPO_CODE/src" || exit 1
exec python3 ablate.py \
  --config "$REPO_CODE/configs/ablations.json" \
  --run_id "$RUN_ID" \
  --variants $VARIANTS \
  --out_root "$OUT_ROOT" \
  --drive "$DRIVE" \
  --input_cases "$EVALSET" \
  --image_dir "$EVALSET/imagesTr" \
  --label_dir "$EVALSET/labelsTr" \
  --repo "$REPO" \
  --cache_dir "$CACHE_DIR" \
  --skip_existing \
  "${CASES[@]}" "${INCLUDE[@]}"
