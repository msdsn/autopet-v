#!/bin/bash
# DEEP-PSMA (Zenodo record 15281784) -> nnU-Net format.
#
# 100 patients, each with a PSMA and an FDG PET/CT + TTB label, in five zips
# totalling 24.07 GB.  DO NOT RUN THIS WHILE THE 150 GB autoPET zip IS STILL
# DOWNLOADING -- it shares the same ~21 MB/s pipe.
#
# Run inside tmux:
#   tmux new -d -s dpsma "bash /content/src/data/fetch_deep_psma.sh > /content/data/dpsma.log 2>&1"
set -euo pipefail

DEST=${DEST:-/content/data/deep_psma}
NNUNET_OUT=${NNUNET_OUT:-/content/work/Dataset997_DeepPSMA}
REPO=${REPO:-/content/autoPETV}

mkdir -p "$DEST/zips" "$DEST/raw"
cd "$DEST/zips"

# md5 sums from the Zenodo API (https://zenodo.org/api/records/15281784)
FILES="0001-0020.zip:ae8198db5e8fc975b192fcb955ac6aa8
0021-0040.zip:66e985d54ca3b6d32c139c2e3ccb530a
0041-0060.zip:2668fe23ac06e12455c49009502a9496
0061-0080.zip:e95924233c1811d728983514aa4627f8
0081-0100.zip:f77e122734e77b90ab41d0570a1e9b3f"

for entry in $FILES; do
  name=${entry%%:*}
  want=${entry##*:}
  echo "=== $name ==="
  curl -L -C - --retry 10 --retry-delay 15 --fail \
       -o "$name" "https://zenodo.org/api/records/15281784/files/$name/content"
  got=$(md5sum "$name" | cut -d' ' -f1)
  [ "$got" = "$want" ] || { echo "CHECKSUM MISMATCH $name: $got != $want"; exit 1; }
  unzip -q -o "$name" -d "$DEST/raw"
done

echo "=== converting to nnU-Net format ==="
python "$REPO/DeepPSMA/convert_deep_psma_to_nnunet_format.py" \
  --root "$DEST/raw" --dest "$NNUNET_OUT"

echo "=== pre-simulated scribbles / heatmaps (already in the repo) ==="
unzip -q -o "$REPO/DeepPSMA/scribbles_deep_psma.zip" -d "$NNUNET_OUT/scribbles"
unzip -q -o "$REPO/DeepPSMA/heatmaps_deep_psma.zip"  -d "$NNUNET_OUT/heatmaps"

echo "DONE $(date -u)"
du -sh "$DEST" "$NNUNET_OUT"
