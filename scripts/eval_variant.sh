#!/bin/bash
# Evaluate one trained model with the shipped pipeline on the 100-case validation set
# and write a run.json record, so every row in results/RESULTS.md is reproducible from
# one command.
#
#   TAG=B13g9 MODEL_FOLDER=<nnUNet_results>/<Dataset>/<trainer>__<plans>__3d_fullres \
#       bash scripts/eval_variant.sh
#
#   TAG=B15   MODEL_FOLDER=... EXTRA="--foveal_crop --foveal_fuse max" \
#       bash scripts/eval_variant.sh          # an option that changes the base model
#
# Environment: TAG (required), MODEL_FOLDER (required), CKPT, EVALSET, CACHE, OUT_ROOT,
# REPO_CODE, REPO, PREP, POSTPROC, EXTRA, LABEL, DRIVE, RESULTS_DIR, CASES_FILE.
#
# On success the four decision-relevant files land in three places: the run directory,
# `$DRIVE/$TAG` and `$RESULTS_DIR/$TAG` (the repo's results/ folder, which
# results_index.py reads). Set RESULTS_DIR=none to skip the repo copy.
set -uo pipefail

TAG=${TAG:?set TAG, the run id (e.g. B13g9)}
MODEL_FOLDER=${MODEL_FOLDER:?set MODEL_FOLDER, the trained model directory}
CKPT=${CKPT:-checkpoint_final.pth}
REPO_CODE=${REPO_CODE:-/content/autopet}
REPO=${REPO:-/content/autoPETV}
EVALSET=${EVALSET:-/content/work/evalset}
CACHE=${CACHE:-/content/work/cache}
OUT_ROOT=${OUT_ROOT:-/content/work/runs}
PREP=${PREP:-/content/nnUNet/prep_local/Dataset998_AutoPETV}
POSTPROC=${POSTPROC:-$REPO_CODE/submission/postproc_config.json}
DRIVE=${DRIVE:-/content/drive/MyDrive/autoPET/runs}
RESULTS_DIR=${RESULTS_DIR:-$REPO_CODE/results}
EXTRA=${EXTRA:-}
OUT=$OUT_ROOT/$TAG

[ -f /content/env.sh ] && . /content/env.sh
export LD_LIBRARY_PATH=/usr/lib64-nvidia:${LD_LIBRARY_PATH:-}
export PYTHONPATH=$REPO_CODE/src:${PYTHONPATH:-}
export AUTOPETV_REPO=$REPO

say() { echo "[$(date -u +%H:%M:%S)] $*"; }
die() { echo "[$(date -u +%H:%M:%S)] ERROR: $*" >&2; exit 1; }

[ -s "$MODEL_FOLDER/fold_0/$CKPT" ] || die "no $CKPT in $MODEL_FOLDER/fold_0"
[ -d "$EVALSET/imagesTr" ] || die "no evaluation set at $EVALSET"
[ -s "$POSTPROC" ] || die "no post-processing config at $POSTPROC"

# An evaluator needs plans.json, dataset.json and dataset_fingerprint.json next to
# fold_0/. The trainer writes the first two itself; the fingerprint comes from the
# preprocessed folder.
for f in plans.json dataset.json dataset_fingerprint.json; do
  [ -s "$MODEL_FOLDER/$f" ] || cp "$PREP/$f" "$MODEL_FOLDER/$f" \
    || die "missing $MODEL_FOLDER/$f and no copy in $PREP"
done
ARCH=$(python3 -c "
import json,sys
p=json.load(open('$MODEL_FOLDER/plans.json'))
c=p['configurations']['3d_fullres']
print(p['plans_name'], c['architecture']['network_class_name'])")
say "model $MODEL_FOLDER"
say "plans $ARCH"

CASES=()
[ -n "${CASES_FILE:-}" ] && CASES=(--cases_file "$CASES_FILE")

mkdir -p "$OUT"
cd "$REPO_CODE/src" || die "no $REPO_CODE/src"
python3 interactive_eval.py \
  --input_cases "$EVALSET" --image_dir "$EVALSET/imagesTr" --label_dir "$EVALSET/labelsTr" \
  --repo "$REPO" --out_dir "$OUT" \
  --predictor postproc --base_predictor interactive_nnunet \
  --postproc_config "$POSTPROC" \
  --model_folder "$MODEL_FOLDER" --folds 0 --checkpoint "$CKPT" \
  --strategy all --max_iters 6 --cache_dir "$CACHE" \
  --resample_logits torch --resample_threads 4 --save_predictions all \
  $EXTRA "${CASES[@]}" || die "the evaluation loop failed"

LABEL=${LABEL:-"$TAG: $(basename "$MODEL_FOLDER") $CKPT + the shipped post-processing configuration${EXTRA:+ ($EXTRA)}, 6 iters, strategy=all, 100 cases"}
python3 finalize_run.py --run_dir "$OUT" --run_id "$TAG" --label "$LABEL" --drive "$DRIVE" \
  || die "finalize_run failed"

if [ "$RESULTS_DIR" != "none" ]; then
  mkdir -p "$RESULTS_DIR/$TAG"
  for f in run.json summary.json metric_scores.json case_info.json; do
    [ -s "$OUT/$f" ] && cp "$OUT/$f" "$RESULTS_DIR/$TAG/$f"
  done
  say "copied the row to $RESULTS_DIR/$TAG"
fi
say "EVAL_DONE $TAG -> $OUT"
