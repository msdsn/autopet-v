#!/bin/bash
# One-shot recovery of the interactive fold-0 fine-tune after a runtime loss.
#
#   bash scripts/env/train_resume.sh            # stage the store if needed, then resume
#   bash scripts/env/train_resume.sh --fresh    # same, but start training from epoch 0
#   bash scripts/env/train_resume.sh --stage-only
#
# Run it on the training box. A runtime loss wipes the ephemeral local disk, so this
# stages the preprocessed store, the surgery checkpoint and the results folder back
# from Drive. Idempotent: staging skips files already present at the right size, and
# it refuses to launch a second trainer if one is running.
set -uo pipefail

DRIVE=/content/drive/MyDrive/autoPET
STORE=$DRIVE/store/bodycrop
DATASET=Dataset998_AutoPETV
TRAINER=nnUNetTrainer_Interactive
PLANS=nnUNetPlans_interactive
PREP=/content/nnUNet/prep_local
DST=$PREP/$DATASET
RESULTS=/content/nnUNet/nnUNet_results
RESDIR=$RESULTS/$DATASET/${TRAINER}__${PLANS}__3d_fullres/fold_0
DRIVE_CKPT=$DRIVE/ckpt/$DATASET/${TRAINER}__${PLANS}__3d_fullres/fold_0
WEIGHTS=/content/work/train/weights/interactive_init_5ch.pth
BASELINE=$DRIVE/weights/nnUNet_results/$DATASET/nnUNetTrainer__nnUNetPlans__3d_fullres
REPO=${REPO:-/content/autopet}
JOBS=${JOBS:-4}            # NEVER raise this: >4 parallel readers wedge the Drive FUSE mount

MODE=${1:-resume}
say() { echo "[$(date -u +%H:%M:%S)] $*"; }
die() { echo "[$(date -u +%H:%M:%S)] ERROR: $*" >&2; exit 1; }

[ -d "$STORE" ] || die "Drive store not found at $STORE -- is /content/drive mounted?"

# A running trainer owns $RESDIR: copying checkpoints on top of a live run's results
# folder would hijack its next --c resume.
RUNNING=0
pgrep -f "nnUNetv2_train 998 " >/dev/null 2>&1 && RUNNING=1
tmux has-session -t train998 2>/dev/null && RUNNING=1
if [ "$RUNNING" = 1 ]; then
  say "a train998 session or an nnUNetv2_train 998 process is already running"
  say "this script will not touch a live run; kill it first if you really mean to restart:"
  say "  tmux kill-session -t train998"
  exit 1
fi

# ---------------------------------------------------------------- 1. the store
say "staging the preprocessed store (the slow part: ~6 min warm, longer cold)"
mkdir -p "$DST/nnUNetPlans_3d_fullres"
for f in nnUNetPlans.json dataset_fingerprint.json splits_final.json; do
  [ -s "$DST/$f" ] || cp "$STORE/$f" "$DST/$f"
done

# copy only what is missing or the wrong size -- a killed copy leaves 0-byte files
# behind and `cp -n` would keep them
SRCLIST=$(mktemp); DSTLIST=$(mktemp); TODO=$(mktemp)
trap 'rm -f "$SRCLIST" "$DSTLIST" "$TODO"' EXIT
(cd "$STORE/nnUNetPlans_3d_fullres" && find . -maxdepth 1 -type f -printf '%s %f\n' | sort) > "$SRCLIST"
(cd "$DST/nnUNetPlans_3d_fullres"   && find . -maxdepth 1 -type f -printf '%s %f\n' | sort) > "$DSTLIST"
comm -23 "$SRCLIST" "$DSTLIST" | cut -d' ' -f2- > "$TODO"
say "$(wc -l < "$TODO") of $(wc -l < "$SRCLIST") files need copying"
if [ -s "$TODO" ]; then
  (cd "$STORE/nnUNetPlans_3d_fullres" &&
   tr '\n' '\0' < "$TODO" | xargs -0 -r -P "$JOBS" -I{} cp -f "{}" "$DST/nnUNetPlans_3d_fullres/")
fi

# integrity gate: names AND byte counts must match, or training reads truncated data
if ! diff -q \
     <(cd "$STORE/nnUNetPlans_3d_fullres" && find . -maxdepth 1 -type f -printf '%s %f\n' | sort) \
     <(cd "$DST/nnUNetPlans_3d_fullres"   && find . -maxdepth 1 -type f -printf '%s %f\n' | sort) \
     >/dev/null; then
  die "local store does not match Drive byte-for-byte -- rerun this script, do not train"
fi
say "store verified: $(ls "$DST/nnUNetPlans_3d_fullres" | wc -l) files match Drive exactly"

# ------------------------------------------------- 2. plans + surgery checkpoint
export nnUNet_raw=/content/nnUNet/nnUNet_raw
export nnUNet_preprocessed=$PREP
export nnUNet_results=$RESULTS
export nnUNet_extTrainer=$REPO/src/train
export PYTHONPATH=$REPO/src:${PYTHONPATH:-}
export AUTOPETV_REPO=${AUTOPETV_REPO:-/content/autoPETV}
export LD_LIBRARY_PATH=/usr/lib64-nvidia:${LD_LIBRARY_PATH:-}
mkdir -p "$nnUNet_raw" "$RESULTS"

if [ ! -s "$DST/$PLANS.json" ] || [ ! -s "$DST/dataset.json" ]; then
  say "rebuilding $PLANS.json + the 5-channel dataset.json"
  (cd "$REPO/src" && python -m train.make_plans \
      --baseline-plans        "$DST/nnUNetPlans.json" \
      --baseline-dataset-json "$STORE/dataset.json" \
      --out-dir               "$DST" \
      --data-identifier       nnUNetPlans_3d_fullres \
      --dataset-name          "$DATASET" \
      --num-training          1611) || die "make_plans failed"
fi

if [ ! -s "$WEIGHTS" ]; then
  say "rebuilding the 4->5 channel surgery checkpoint"
  mkdir -p "$(dirname "$WEIGHTS")"
  (cd "$REPO/src" && python -m train.init_from_baseline \
      --src "$BASELINE/fold_0/checkpoint_final.pth" \
      --dst "$WEIGHTS") || die "init_from_baseline failed"
fi

if [ "$MODE" = "--stage-only" ]; then
  say "--stage-only: store staged and verified, results folder untouched"
  exit 0
fi

# --------------------------------------------------- 3. checkpoint to resume from
mkdir -p "$RESDIR"
if [ "$MODE" = "--fresh" ]; then
  say "--fresh: wiping the results folder, training will start at epoch 0"
  rm -rf "$RESULTS/$DATASET/${TRAINER}__${PLANS}__3d_fullres"
  mkdir -p "$RESDIR"
  CONT=""
else
  # Pull from Drive only when the local folder is empty, and only
  # checkpoint_latest.pth: nnU-Net's --c prefers checkpoint_final.pth, so importing a
  # `final` would resume from the wrong weights. Nothing local is overwritten.
  if [ ! -s "$RESDIR/checkpoint_latest.pth" ] && [ ! -s "$RESDIR/checkpoint_final.pth" ] \
     && [ -s "$DRIVE_CKPT/checkpoint_latest.pth" ]; then
    say "no local checkpoint: pulling checkpoint_latest.pth back from Drive"
    cp -n "$DRIVE_CKPT/checkpoint_latest.pth" "$RESDIR/checkpoint_latest.pth"
    [ -s "$DRIVE_CKPT/checkpoint_best.pth" ] && cp -n "$DRIVE_CKPT/checkpoint_best.pth" "$RESDIR/" || true
  fi
  if [ -s "$RESDIR/checkpoint_latest.pth" ] || [ -s "$RESDIR/checkpoint_final.pth" ]; then
    CONT="--c"
    python - "$RESDIR" <<'PY'
import glob, os, sys, torch
d = sys.argv[1]
p = os.path.join(d, "checkpoint_latest.pth")
if not os.path.isfile(p):
    p = os.path.join(d, "checkpoint_final.pth")
ck = torch.load(p, map_location="cpu", weights_only=False)
w = ck["network_weights"]["encoder.stages.0.0.convs.0.conv.weight"]
print(f"  resuming from {os.path.basename(p)}: epoch {ck['current_epoch']}, "
      f"trainer {ck['trainer_name']}, first conv {tuple(w.shape)}")
PY
  else
    say "no checkpoint anywhere -- starting from the surgery checkpoint at epoch 0"
    CONT=""
  fi
fi
# on --c the trainer ignores this on purpose; harmless to always export it
export nnUNet_interactive_pretrained=$WEIGHTS
export nnUNet_interactive_save_every=5

# ------------------------------------------------------------------- 4. launch
#
# A new tmux session inherits the tmux server's environment, not this script's, so
# exported variables do not reach the trainer (symptom: empty nnUNet_extTrainer and
# "Could not find requested nnunet trainer"). Bake them into a generated launcher.
mkdir -p /content/work/train
LAUNCH=/content/work/train/.train998_launch.sh
cat > "$LAUNCH" <<EOS
#!/bin/bash
export LD_LIBRARY_PATH=/usr/lib64-nvidia:\${LD_LIBRARY_PATH:-}
export nnUNet_raw=$nnUNet_raw
export nnUNet_preprocessed=$nnUNet_preprocessed
export nnUNet_results=$nnUNet_results
export nnUNet_extTrainer=$nnUNet_extTrainer
export PYTHONPATH=$PYTHONPATH
export AUTOPETV_REPO=$AUTOPETV_REPO
export nnUNet_interactive_pretrained=$nnUNet_interactive_pretrained
export nnUNet_interactive_save_every=$nnUNet_interactive_save_every
cd $REPO/src
exec nnUNetv2_train 998 3d_fullres 0 -tr $TRAINER -p $PLANS $CONT
EOS
chmod +x "$LAUNCH"

nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader || true
say "launching: nnUNetv2_train 998 3d_fullres 0 -tr $TRAINER -p $PLANS $CONT"
tmux new -d -s train998 "bash $LAUNCH >> /content/work/train/train998.log 2>&1"
tmux has-session -t progress 2>/dev/null || tmux new -d -s progress \
  "python3 -u $REPO/scripts/env/progress_watch.py > /content/work/train/progress_watch.log 2>&1"

# do not walk away until the trainer has actually got past trainer lookup
for i in $(seq 1 30); do
  sleep 5
  if grep -q "Traceback\|RuntimeError\|Error" /content/work/train/train998.log 2>/dev/null; then
    say "the trainer FAILED to start -- last lines:"; tail -15 /content/work/train/train998.log; exit 1
  fi
  grep -q "\[interactive\] epochs=" /content/work/train/train998.log 2>/dev/null && break
done
grep -q "\[interactive\] epochs=" /content/work/train/train998.log 2>/dev/null \
  || die "trainer did not report its config within 150 s -- check /content/work/train/train998.log"
grep -h "\[interactive\] epochs=" /content/work/train/train998.log | tail -1
tmux ls | grep -E 'train998|progress'
say "tail -f /content/work/train/train998.log  |  cat /content/work/train/progress.txt"
