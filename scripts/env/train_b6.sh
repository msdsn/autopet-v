#!/bin/bash
# B6 -- k-reweighted continuation of the interactive fold-0 fine-tune. Run it ON the
# GPU box. B6 is a continuation, not a `--c` resume: the weights come from the first
# fine-tune's checkpoint_final.pth, the optimizer and PolyLR schedule start fresh, and
# it has its own trainer class and results folder.
#
#   bash scripts/env/train_b6.sh              # launch now (refuses if the GPU is busy)
#   bash scripts/env/train_b6.sh --wait       # poll every 5 min, launch when the GPU frees
#   bash scripts/env/train_b6.sh --resume     # after a runtime loss: restage, then resume with --c
#   bash scripts/env/train_b6.sh --dry-run    # check everything, print the plan, launch nothing
#
#   TRAINER=<class> TAG=<name>            # another trainer of the same family; TAG names
#                                         # the log, progress file and tmux sessions
#   ALLOW_BUSY_GPU=1 NICE=5               # co-resident with another GPU job
#
# The B6 knobs live in the trainer class, so this script unsets the matching
# nnUNet_interactive_* overrides: a stale export must not change the configuration.
set -uo pipefail

DRIVE=/content/drive/MyDrive/autoPET
DATASET=Dataset998_AutoPETV
PLANS=nnUNetPlans_interactive
TRAINER=${TRAINER:-nnUNetTrainer_InteractiveB6}
SRC_TRAINER=nnUNetTrainer_Interactive          # the run B6 continues from
PREP=/content/nnUNet/prep_local
DST=$PREP/$DATASET
RESULTS=/content/nnUNet/nnUNet_results
MODEL=${TRAINER}__${PLANS}__3d_fullres
RESDIR=$RESULTS/$DATASET/$MODEL/fold_0
SRC_CKPT=$DRIVE/ckpt/$DATASET/${SRC_TRAINER}__${PLANS}__3d_fullres/fold_0/checkpoint_final.pth
DRIVE_CKPT=$DRIVE/ckpt/$DATASET/$MODEL/fold_0        # where the checkpoint sync mirrors this run
# the source weights are the same for every B6-family run, so they are shared
INIT=${INIT:-/content/work/train/weights/b6_init_from_final.pth}
TAG=${TAG:-b6}
LOG=/content/work/train/train_$TAG.log
PROGRESS=/content/work/train/progress_$TAG.txt
REPO=${REPO:-/content/autopet}
EPOCHS=${EPOCHS:-120}                           # must match the trainer class NUM_EPOCHS
# tmux sessions that own the GPU while an evaluation runs; B6 waits for all of them
BUSY_SESSIONS=${BUSY_SESSIONS:-"b0 bchain"}
# ALLOW_BUSY_GPU=1 runs alongside whatever else is on the GPU; this training is
# dataloader-bound, so NICE=<n> matters more than the GPU share -- CPU is what the
# two jobs actually contend for.
ALLOW_BUSY_GPU=${ALLOW_BUSY_GPU:-0}
NICE=${NICE:-}

MODE=${1:-now}
say() { echo "[$(date -u +%H:%M:%S)] $*"; }
die() { echo "[$(date -u +%H:%M:%S)] ERROR: $*" >&2; exit 1; }

# ------------------------------------------------------------------ 0. guards
tmux has-session -t train_$TAG 2>/dev/null && die "tmux session train_$TAG already exists"
pgrep -f "nnUNetv2_train 998 " >/dev/null 2>&1 && die "an nnUNetv2_train 998 process is already running"
CONT=""
if [ "$MODE" = "--resume" ]; then
  mkdir -p "$RESDIR"
  # Pull from the mirror only when the local folder is empty, and only
  # checkpoint_latest.pth: `--c` prefers checkpoint_final.pth, so importing a `final`
  # would resume from the wrong weights. Never overwrite anything local.
  if [ ! -s "$RESDIR/checkpoint_latest.pth" ] && [ ! -s "$RESDIR/checkpoint_final.pth" ] \
     && [ -s "$DRIVE_CKPT/checkpoint_latest.pth" ]; then
    say "no local checkpoint: pulling checkpoint_latest.pth back from the mirror"
    cp -n "$DRIVE_CKPT/checkpoint_latest.pth" "$RESDIR/checkpoint_latest.pth"
    [ -s "$DRIVE_CKPT/checkpoint_best.pth" ] && cp -n "$DRIVE_CKPT/checkpoint_best.pth" "$RESDIR/" || true
  fi
  [ -s "$RESDIR/checkpoint_latest.pth" ] || [ -s "$RESDIR/checkpoint_final.pth" ] \
    || die "--resume: no B6 checkpoint found locally or in the mirror"
  CONT="--c"
elif [ -s "$RESDIR/checkpoint_latest.pth" ] || [ -s "$RESDIR/checkpoint_final.pth" ]; then
  die "$RESDIR already holds a checkpoint -- use --resume, or move it aside to start B6 over"
fi

# ------------------------------------------------------------- 1. staged store
[ -s "$DST/$PLANS.json" ] || die "$DST/$PLANS.json missing -- run scripts/env/train_resume.sh --stage-only first"
[ -s "$DST/dataset.json" ] || die "$DST/dataset.json missing -- run scripts/env/train_resume.sh --stage-only first"
[ -s "$DST/splits_final.json" ] || die "$DST/splits_final.json missing"
NLOCAL=$(find "$DST/nnUNetPlans_3d_fullres" -maxdepth 1 -type f | wc -l)
[ "$NLOCAL" -ge 4833 ] || die "only $NLOCAL files staged in $DST/nnUNetPlans_3d_fullres (expected 4833) -- rerun scripts/env/train_resume.sh --stage-only"
say "store staged: $NLOCAL files, $(du -sh "$DST" | cut -f1)"

# ------------------------------------------- 2. the checkpoint B6 continues from
# On `--c` the trainer ignores nnUNet_interactive_pretrained on purpose (the
# checkpoint on disk wins), so the source checkpoint is only needed for a start.
if [ -z "$CONT" ] && [ ! -s "$INIT" ]; then
  [ -s "$SRC_CKPT" ] || die "source checkpoint not found: $SRC_CKPT"
  say "copying the source checkpoint off Drive (once)"
  mkdir -p "$(dirname "$INIT")"
  cp "$SRC_CKPT" "$INIT.part" || die "copy failed"
  mv "$INIT.part" "$INIT"
fi

export LD_LIBRARY_PATH=/usr/lib64-nvidia:${LD_LIBRARY_PATH:-}
export nnUNet_raw=/content/nnUNet/nnUNet_raw
export nnUNet_preprocessed=$PREP
export nnUNet_results=$RESULTS
export nnUNet_extTrainer=$REPO/src/train
export PYTHONPATH=$REPO/src:${PYTHONPATH:-}
export AUTOPETV_REPO=${AUTOPETV_REPO:-/content/autoPETV}
mkdir -p "$nnUNet_raw" "$RESULTS" /content/work/train

if [ -n "$CONT" ]; then
  CHECK=$RESDIR/checkpoint_latest.pth
  [ -s "$CHECK" ] || CHECK=$RESDIR/checkpoint_final.pth
else
  CHECK=$INIT
fi
python3 - "$CHECK" <<'PY' || die "$CHECK is not a usable 5-channel interactive checkpoint"
import os, sys, torch
ck = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
w = ck["network_weights"]["encoder.stages.0.0.convs.0.conv.weight"]
print(f"  {os.path.basename(sys.argv[1])}: epoch {ck['current_epoch']}, "
      f"trainer {ck['trainer_name']}, first conv {tuple(w.shape)}")
assert tuple(w.shape)[1] == 5, "expected a 5-channel first conv"
PY

# -------------------------------------------------- 3. the GPU must be free
# nvidia-smi failing must never read as "free" -- without LD_LIBRARY_PATH it exits
# non-zero with empty stdout, which would look exactly like an idle GPU.
nvidia-smi --query-gpu=name --format=csv,noheader >/dev/null 2>&1 \
  || die "nvidia-smi does not run here -- refusing to guess whether the GPU is free"
gpu_busy() {
  local apps
  apps=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | tr -d '[:space:]')
  [ -n "$apps" ] && return 0
  for s in $BUSY_SESSIONS; do tmux has-session -t "$s" 2>/dev/null && return 0; done
  return 1
}
if [ "$MODE" = "--dry-run" ]; then
  gpu_busy && say "note: the GPU is busy right now" || say "note: the GPU is free"
fi
if [ "$ALLOW_BUSY_GPU" = "1" ] && gpu_busy; then
  say "ALLOW_BUSY_GPU=1: starting while the GPU is in use -- deliberate co-residency"
  nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader
elif [ "$MODE" != "--dry-run" ] && gpu_busy; then
  if [ "$MODE" = "--wait" ]; then
    say "GPU busy (an evaluation owns it) -- polling every 5 min, it has priority"
    # Two consecutive free checks 5 min apart: an evaluation chain goes idle for a
    # moment between variants, and one check would race the next job.
    while true; do
      while gpu_busy; do sleep 300; done
      say "GPU looks free -- confirming in 5 min"
      sleep 300
      gpu_busy || break
      say "not free after all, still waiting"
    done
    say "GPU free"
    sleep 60                      # let the last process release its memory
  else
    nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader
    die "the GPU is busy; rerun with --wait, or wait for the evaluation to finish"
  fi
fi

# --------------------------------------------------------------- 4. the launcher
# A new tmux session inherits the tmux server's environment, not this script's, so
# the environment is baked into a generated launcher.
LAUNCH=/content/work/train/.train_${TAG}_launch.sh
cat > "$LAUNCH" <<EOS
#!/bin/bash
export LD_LIBRARY_PATH=/usr/lib64-nvidia:\${LD_LIBRARY_PATH:-}
export nnUNet_raw=$nnUNet_raw
export nnUNet_preprocessed=$nnUNet_preprocessed
export nnUNet_results=$nnUNet_results
export nnUNet_extTrainer=$nnUNet_extTrainer
export PYTHONPATH=$PYTHONPATH
export AUTOPETV_REPO=$AUTOPETV_REPO
# continuation: load the weights, start a fresh optimizer and a fresh PolyLR
export nnUNet_interactive_pretrained=$INIT
export nnUNet_interactive_save_every=5
# the B6 knobs live in the trainer class -- never let a stale export win
unset nnUNet_interactive_k_probs nnUNet_interactive_p_independent \\
      nnUNet_interactive_epochs nnUNet_interactive_lr \\
      nnUNet_interactive_p_perturb nnUNet_interactive_radius \\
      nnUNet_interactive_batch_size nnUNet_interactive_noSmooth
cd $REPO/src
exec ${NICE:+nice -n $NICE }nnUNetv2_train 998 3d_fullres 0 -tr $TRAINER -p $PLANS $CONT
EOS
chmod +x "$LAUNCH"

if [ "$MODE" = "--dry-run" ]; then
  say "--dry-run: everything checks out; would run:"; cat "$LAUNCH"; exit 0
fi

# ------------------------------------------------------------------- 5. launch
if [ -n "$CONT" ]; then
  say "resuming $TRAINER with --c from $(basename "$CHECK")"
else
  say "launching $TRAINER ($EPOCHS epochs, continuation from $(basename "$SRC_CKPT"))"
fi
nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader || true
# rotate the log: the start check below greps it, and a previous attempt's output
# would satisfy the grep before this launch has printed anything
[ -s "$LOG" ] && mv "$LOG" "$LOG.$(date -u +%Y%m%dT%H%M%SZ)"
tmux new -d -s train_$TAG "bash $LAUNCH >> $LOG 2>&1"
tmux has-session -t ${TAG}progress 2>/dev/null || tmux new -d -s ${TAG}progress \
  "bash -c 'export MODEL_NAME=$MODEL PROGRESS_FILE=$PROGRESS TOTAL_EPOCHS=$EPOCHS; \
            exec python3 -u $REPO/scripts/env/progress_watch.py' > /content/work/train/${TAG}progress.log 2>&1"

for _ in $(seq 1 40); do
  sleep 5
  if grep -qE "Traceback|RuntimeError|Could not find requested" "$LOG" 2>/dev/null; then
    say "the trainer FAILED to start -- last lines:"; tail -20 "$LOG"; exit 1
  fi
  grep -q "\[interactive\] epochs=" "$LOG" 2>/dev/null && break
done
grep -q "\[interactive\] epochs=" "$LOG" 2>/dev/null \
  || die "the trainer did not report its config within 200 s -- check $LOG"
grep -h "\[interactive\] epochs=" "$LOG" | tail -1
tmux ls | grep -E "train_$TAG|${TAG}progress"
say "tail -f $LOG   |   cat $PROGRESS"
