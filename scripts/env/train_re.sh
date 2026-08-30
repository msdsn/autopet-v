#!/bin/bash
# RE -- the B10 recipe warm-started from the autoPET III LesionTracer ResEncL.
# Same contract as scripts/env/train_b6.sh (TRAINER/TAG/PLANS/INIT/EPOCHS, tmux, the
# progress watcher, the GPU guard); it is a separate file for one reason: b6's
# source-checkpoint check reads `encoder.stages.0.0.convs.0.conv.weight`, which is the
# PlainConvUNet's first conv and does not exist in a ResEncL, where the stem is
# `encoder.stem.convs.0.conv.weight`. The check here finds the stem by structure
# instead, so it works for either architecture.
#
#   bash scripts/env/train_re.sh              # launch now (refuses if the GPU is busy)
#   bash scripts/env/train_re.sh --wait       # poll every 5 min, launch when the GPU frees
#   bash scripts/env/train_re.sh --resume     # after a runtime loss: restage, then --c
#   bash scripts/env/train_re.sh --dry-run    # check everything, launch nothing
set -uo pipefail

DRIVE=/content/drive/MyDrive/autoPET
DATASET=Dataset998_AutoPETV
PLANS=${PLANS:-nnUNetPlans_re}
TRAINER=${TRAINER:-nnUNetTrainer_InteractiveRE_40epochs}
PREP=/content/nnUNet/prep_local
DST=$PREP/$DATASET
RESULTS=/content/nnUNet/nnUNet_results
MODEL=${TRAINER}__${PLANS}__3d_fullres
RESDIR=$RESULTS/$DATASET/$MODEL/fold_0
DRIVE_CKPT=$DRIVE/ckpt/$DATASET/$MODEL/fold_0
INIT=${INIT:-/content/work/train/weights/re_init_5ch.pth}
TAG=${TAG:-re40}
LOG=/content/work/train/train_$TAG.log
PROGRESS=/content/work/train/progress_$TAG.txt
REPO=${REPO:-/content/autopet}
EPOCHS=${EPOCHS:-40}                            # must match the trainer class NUM_EPOCHS
BUSY_SESSIONS=${BUSY_SESSIONS:-"b0 bchain"}
ALLOW_BUSY_GPU=${ALLOW_BUSY_GPU:-0}
NICE=${NICE:-}

MODE=${1:-now}
say() { echo "[$(date -u +%H:%M:%S)] $*"; }
die() { echo "[$(date -u +%H:%M:%S)] ERROR: $*" >&2; exit 1; }

# ------------------------------------------------------------------ 0. guards
tmux has-session -t train_$TAG 2>/dev/null && die "tmux session train_$TAG already exists"
if pgrep -f "nnUNetv2_train 998 " >/dev/null 2>&1; then
  [ "${ALLOW_CONCURRENT_TRAIN:-0}" = "1" ] \
    || die "an nnUNetv2_train 998 process is already running"
  say "ALLOW_CONCURRENT_TRAIN=1: a second trainer will share the GPU"
fi
CONT=""
if [ "$MODE" = "--resume" ]; then
  mkdir -p "$RESDIR"
  if [ ! -s "$RESDIR/checkpoint_latest.pth" ] && [ ! -s "$RESDIR/checkpoint_final.pth" ] \
     && [ -s "$DRIVE_CKPT/checkpoint_latest.pth" ]; then
    say "no local checkpoint: pulling checkpoint_latest.pth back from the mirror"
    cp -n "$DRIVE_CKPT/checkpoint_latest.pth" "$RESDIR/checkpoint_latest.pth"
    [ -s "$DRIVE_CKPT/checkpoint_best.pth" ] && cp -n "$DRIVE_CKPT/checkpoint_best.pth" "$RESDIR/" || true
  fi
  [ -s "$RESDIR/checkpoint_latest.pth" ] || [ -s "$RESDIR/checkpoint_final.pth" ] \
    || die "--resume: no RE checkpoint found locally or in the mirror"
  CONT="--c"
elif [ -s "$RESDIR/checkpoint_latest.pth" ] || [ -s "$RESDIR/checkpoint_final.pth" ]; then
  die "$RESDIR already holds a checkpoint -- use --resume, or move it aside"
fi

# ------------------------------------------------------------- 1. staged store
[ -s "$DST/$PLANS.json" ] || die "$DST/$PLANS.json missing -- run train.make_re_plans first"
[ -s "$DST/dataset.json" ] || die "$DST/dataset.json missing"
[ -s "$DST/splits_final.json" ] || die "$DST/splits_final.json missing"
NLOCAL=$(find "$DST/nnUNetPlans_3d_fullres" -maxdepth 1 -type f | wc -l)
[ "$NLOCAL" -ge 4833 ] || die "only $NLOCAL files staged (expected 4833) -- rerun scripts/env/train_resume.sh --stage-only"
say "store staged: $NLOCAL files, $(du -sh "$DST" | cut -f1)"

[ -z "$CONT" ] && { [ -s "$INIT" ] || die "no RE init checkpoint at $INIT -- run train.init_from_lesiontracer"; }

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
# Architecture-agnostic: the stem is the only 5-D weight whose in-channel count is the
# network input, so pick the 5-D weight with the smallest second dimension.
python3 - "$CHECK" <<'PY' || die "$CHECK is not a usable 5-channel interactive checkpoint"
import os, sys, torch
ck = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
sd = ck["network_weights"]
cands = [(k, v) for k, v in sd.items()
         if hasattr(v, "ndim") and v.ndim == 5 and k.endswith("weight")]
assert cands, "no 5-D convolution weights in the checkpoint"
k, w = min(cands, key=lambda kv: kv[1].shape[1])
print(f"  {os.path.basename(sys.argv[1])}: epoch {ck['current_epoch']}, "
      f"trainer {ck['trainer_name']}, stem {k} {tuple(w.shape)}, "
      f"{len(sd)} tensors, organ heads {sum('organ' in x for x in sd)}")
assert tuple(w.shape)[1] == 5, f"expected a 5-channel stem, got {tuple(w.shape)}"
assert not any("organ_seg_layers" in x for x in sd), "organ heads were not dropped"
PY

# -------------------------------------------------- 2. the GPU must be free
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
    say "GPU busy -- polling every 5 min, the running job has priority"
    while true; do
      while gpu_busy; do sleep 300; done
      say "GPU looks free -- confirming in 5 min"
      sleep 300
      gpu_busy || break
      say "not free after all, still waiting"
    done
    say "GPU free"
    sleep 60
  else
    nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader
    die "the GPU is busy; rerun with --wait"
  fi
fi

# --------------------------------------------------------------- 3. launcher
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
export nnUNet_interactive_pretrained=$INIT
export nnUNet_interactive_save_every=5
${nnUNet_n_proc_DA:+export nnUNet_n_proc_DA=$nnUNet_n_proc_DA}
${RE_STEM_LR_MULT:+export RE_STEM_LR_MULT=$RE_STEM_LR_MULT}
# the recipe lives in the trainer class -- never let a stale export win
unset nnUNet_interactive_k_probs nnUNet_interactive_p_independent \\
      nnUNet_interactive_epochs nnUNet_interactive_lr \\
      nnUNet_interactive_p_perturb nnUNet_interactive_radius \\
      nnUNet_interactive_batch_size nnUNet_interactive_noSmooth \\
      nnUNet_arch_refbatch
cd $REPO/src
exec ${NICE:+nice -n $NICE }nnUNetv2_train 998 3d_fullres 0 -tr $TRAINER -p $PLANS $CONT
EOS
chmod +x "$LAUNCH"

if [ "$MODE" = "--dry-run" ]; then
  say "--dry-run: everything checks out; would run:"; cat "$LAUNCH"; exit 0
fi

# ------------------------------------------------------------------- 4. launch
say "launching $TRAINER ($EPOCHS epochs, plans $PLANS, init $(basename "$INIT"))"
nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader || true
[ -s "$LOG" ] && mv "$LOG" "$LOG.$(date -u +%Y%m%dT%H%M%SZ)"
tmux new -d -s train_$TAG "bash $LAUNCH >> $LOG 2>&1"
tmux has-session -t ${TAG}progress 2>/dev/null || tmux new -d -s ${TAG}progress \
  "bash -c 'export MODEL_NAME=$MODEL PROGRESS_FILE=$PROGRESS TOTAL_EPOCHS=$EPOCHS; \
            exec python3 -u $REPO/scripts/env/progress_watch.py' > /content/work/train/${TAG}progress.log 2>&1"

for _ in $(seq 1 60); do
  sleep 5
  if grep -qE "Traceback|RuntimeError|Could not find requested" "$LOG" 2>/dev/null; then
    say "the trainer FAILED to start -- last lines:"; tail -25 "$LOG"; exit 1
  fi
  grep -q "\[RE\] identity assertion PASS" "$LOG" 2>/dev/null && break
done
grep -q "\[RE\] identity assertion PASS" "$LOG" 2>/dev/null \
  || die "the RE identity gate did not pass within 300 s -- check $LOG"
grep -h "\[interactive\] epochs=\|\[RE\] " "$LOG" | tail -5
tmux ls | grep -E "train_$TAG|${TAG}progress"
say "tail -f $LOG   |   cat $PROGRESS"
