#!/bin/bash
# Rehydrate a fresh runtime. Idempotent. Run: bash scripts/env/colab_bootstrap.sh
set -euo pipefail
DRIVE=/content/drive/MyDrive/autoPET
REPO=/content/autopet
export LD_LIBRARY_PATH=/usr/lib64-nvidia:${LD_LIBRARY_PATH:-}
grep -q lib64-nvidia /root/.bashrc || echo 'export LD_LIBRARY_PATH=/usr/lib64-nvidia:$LD_LIBRARY_PATH' >> /root/.bashrc
[ -d /content/drive/MyDrive ] || { echo "Drive not mounted: mount it from the notebook first"; exit 1; }
# official repo, pinned, no LFS blobs
if [ ! -d /content/autoPETV/.git ]; then
  GIT_LFS_SKIP_SMUDGE=1 git clone -q https://github.com/lab-midas/autoPETV.git /content/autoPETV
fi
PIN=$(cut -d' ' -f1 "$REPO/docs/autoPETV_pin.txt" 2>/dev/null || true)
[ -n "$PIN" ] && GIT_LFS_SKIP_SMUDGE=1 git -C /content/autoPETV checkout -q "$PIN" || true
# python deps
pip install -q -r "$REPO/requirements-dev.txt"
# labels + weights off Drive
mkdir -p /content/work/runs /content/work/scratch /content/nnUNet/nnUNet_raw /content/nnUNet/nnUNet_preprocessed /content/nnUNet/nnUNet_results
[ -d /content/work/labelsTr ] || { [ -f "$DRIVE/labelsTr.tar" ] && tar -xf "$DRIVE/labelsTr.tar" -C /content/work || true; }
rsync -a --ignore-existing "$DRIVE/weights/nnUNet_results/" /content/nnUNet/nnUNet_results/
cat > /content/env.sh <<EOT
export LD_LIBRARY_PATH=/usr/lib64-nvidia:\${LD_LIBRARY_PATH:-}
export nnUNet_raw=/content/nnUNet/nnUNet_raw
export nnUNet_preprocessed=/content/nnUNet/nnUNet_preprocessed
export nnUNet_results=/content/nnUNet/nnUNet_results
export AUTOPETV_REPO=/content/autoPETV
export PYTHONPATH=$REPO/src:\${PYTHONPATH:-}
EOT
# background syncers: run records -> Drive
tmux has-session -t sync 2>/dev/null || tmux new -d -s sync "while true; do rsync -a --include '*/' --include '*.json' --include '*.log' --include '*.md' --include '*.txt' --include '*.png' --exclude '*' /content/work/ $DRIVE/runs/_worksync/ ; sleep 600; done > /content/worksync.log 2>&1"
tmux has-session -t ckptsync 2>/dev/null || tmux new -d -s ckptsync "while true; do rsync -a --include '*/' --include 'checkpoint_*.pth' --include '*.json' --include '*.txt' --include '*.png' --exclude '*' /content/nnUNet/nnUNet_results/ $DRIVE/ckpt/ ; sleep 900; done > /content/ckptsync.log 2>&1"
echo "bootstrap ok — every command: source /content/env.sh"
nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv
