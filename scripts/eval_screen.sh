#!/bin/bash
# Evaluate one trained model on a subset of the validation set (the screening list) or
# on all of it, and write a run.json record.
#
#   TAG=C0g9-s39 MODEL_FOLDER=<...> CASES_FILE=docs/valset_screen39.txt \
#       bash scripts/eval_screen.sh
#
# This exists next to eval_variant.sh because case names contain spaces: the list has
# to reach interactive_eval.py as a bash array of --cases arguments, and
# interactive_eval.py has no --cases_file option (eval_variant.sh's CASES_FILE branch
# passes a flag that does not exist and would abort the run).
#
# The screening list must be blocks of three aligned to a multiple of three in the
# sorted case list: `assign_strategies` round-robins centerline/random/boundary over
# the sorted cases, so only an aligned subset gives every case the same scribble
# strategy it has in the full run, which is what makes a paired delta meaningful.
set -uo pipefail

TAG=${TAG:?set TAG, the run id (e.g. C0g9-s39)}
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
CASES_FILE=${CASES_FILE:-}
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

for f in plans.json dataset.json dataset_fingerprint.json; do
  [ -s "$MODEL_FOLDER/$f" ] || cp "$PREP/$f" "$MODEL_FOLDER/$f" \
    || die "missing $MODEL_FOLDER/$f and no copy in $PREP"
done

CASES=()
if [ -n "$CASES_FILE" ]; then
  [ -s "$CASES_FILE" ] || die "no case list at $CASES_FILE"
  CASES=(--cases)
  while IFS= read -r line; do
    [ -n "$line" ] && CASES+=("$line")
  done < "$CASES_FILE"
  say "case list $CASES_FILE: $(( ${#CASES[@]} - 1 )) cases"
fi

say "model $MODEL_FOLDER ($CKPT)"
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

N=$([ -n "$CASES_FILE" ] && echo "$(( ${#CASES[@]} - 1 )) cases ($(basename "$CASES_FILE"))" || echo "100 cases")
LABEL=${LABEL:-"$TAG: $(basename "$MODEL_FOLDER") $CKPT + the shipped post-processing configuration${EXTRA:+ ($EXTRA)}, 6 iters, strategy=all, $N"}
python3 finalize_run.py --run_dir "$OUT" --run_id "$TAG" --label "$LABEL" --drive "$DRIVE" \
  || die "finalize_run failed"
say "EVAL_DONE $TAG -> $OUT"
