#!/bin/bash
# Rehydrate a fresh runtime from the persistent store.
#
# /content dies with the runtime, Drive survives. This copies Tier A onto the
# ephemeral disk and exports the nnU-Net environment to train against it. Expect
# 10-25 min for a 45 GB store.
#
#   bash /content/src/data/session_setup.sh [variant]
#
# Do NOT let nnU-Net unpack an npz store: unpack_dataset materialises a full .npy per
# case and would need >200 GB. Either keep npz with self.unpack_dataset = False, or
# use the b2nd store, which the dataloader mmaps.
set -euo pipefail

VARIANT=${1:-bodycrop}
FMT=${FMT:-b2nd}
DRIVE=/content/drive/MyDrive/autoPET
LOCAL=/content/nnUNet
DS=Dataset998_AutoPETV

SRC="$DRIVE/store/$VARIANT"
DST="$LOCAL/nnUNet_preprocessed/$DS"

mkdir -p "$DST" "$LOCAL/nnUNet_results" "$LOCAL/nnUNet_raw"

echo "=== copying dataset-level json ==="
for f in dataset.json nnUNetPlans.json splits_final.json dataset_fingerprint.json; do
  [ -f "$SRC/$f" ] && cp -n "$SRC/$f" "$DST/$f"
done

echo "=== copying Tier A ($VARIANT) ==="
rsync -a --info=progress2 --ignore-existing \
      "$SRC/nnUNetPlans_3d_fullres/" "$DST/nnUNetPlans_3d_fullres/"

echo "=== weights ==="
rsync -a --ignore-existing "$DRIVE/weights/nnUNet_results/" "$LOCAL/nnUNet_results/"

export nnUNet_raw=$LOCAL/nnUNet_raw
export nnUNet_preprocessed=$LOCAL/nnUNet_preprocessed
export nnUNet_results=$LOCAL/nnUNet_results
export LD_LIBRARY_PATH=/usr/lib64-nvidia:${LD_LIBRARY_PATH:-}

cat <<EOF

Ready.  Dataset $DS, plans "nnUNetPlans", configuration "3d_fullres".
Add these to your shell / tmux command:

  export nnUNet_raw=$LOCAL/nnUNet_raw
  export nnUNet_preprocessed=$LOCAL/nnUNet_preprocessed
  export nnUNet_results=$LOCAL/nnUNet_results
  export LD_LIBRARY_PATH=/usr/lib64-nvidia:\$LD_LIBRARY_PATH

Then, e.g.:

  nnUNetv2_train $DS 3d_fullres 0 -tr <YourInteractiveTrainer> \\
      -pretrained_weights $LOCAL/nnUNet_results/$DS/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_0/checkpoint_final.pth

Do NOT unpack: the store is blosc2 and the dataloader mmaps it.

Cases in the store: $(ls "$DST/nnUNetPlans_3d_fullres"/*.pkl 2>/dev/null | wc -l)
Size: $(du -sh "$DST" 2>/dev/null | cut -f1)
EOF
