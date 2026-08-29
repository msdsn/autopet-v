#!/bin/bash
# Full Tier A + Tier B build, resumable, while the 150 GB zip is still downloading.
#
# Each pass processes whatever is fully downloaded. Both build and evalset are
# idempotent, so a pass that finds nothing new is a cheap no-op. Loops until all 1611
# Tier A and 100 Tier B cases are done.
#
#   tmux new -d -s store "nice -n 10 bash /content/src/data/run_full_build.sh \
#       > /content/work/store_build.log 2>&1"
set -uo pipefail

SRC=/content/src/data
DRIVE=/content/drive/MyDrive/autoPET
STORE=$DRIVE/store/bodycrop/nnUNetPlans_3d_fullres
EVAL=$DRIVE/evalset
WORKERS=${WORKERS:-6}
TARGET_A=1611
TARGET_B=100
SLEEP=${SLEEP:-120}

cd "$SRC"
mkdir -p "$EVAL"

pass=0
while true; do
  pass=$((pass + 1))
  echo "===================================================================="
  echo "PASS $pass  $(date -u '+%F %T UTC')"
  zip_bytes=$(stat -c%s /content/data/psma-fdg.zip 2>/dev/null || echo 0)
  echo "zip: $zip_bytes bytes ($(python3 -c "print(f'{$zip_bytes/150293961309*100:.2f}')")%)"

  echo "--- Tier A ---"
  nice -n 10 python build_store.py build \
      --variant bodycrop --fmt b2nd --dtype float16 \
      --workers "$WORKERS" --out "$STORE" 2>&1 | grep -v '^\[' | tail -20
  a=$(python build_store.py status --out "$STORE" --quiet)

  echo "--- Tier B ---"
  if [ "$(ls "$EVAL/labelsTr" 2>/dev/null | wc -l)" -lt "$TARGET_B" ]; then
    nice -n 10 python build_store.py evalset --n "$TARGET_B" --out "$EVAL" 2>&1 | tail -3
  fi
  b=$(ls "$EVAL/labelsTr" 2>/dev/null | wc -l)

  echo "PROGRESS  tierA=$a/$TARGET_A  tierB=$b/$TARGET_B"
  if [ "$a" -ge "$TARGET_A" ] && [ "$b" -ge "$TARGET_B" ]; then
    echo "ALL DONE $(date -u '+%F %T UTC')"
    break
  fi
  sleep "$SLEEP"
done

echo "===================================================================="
echo "=== final status ==="
python build_store.py status --out "$STORE"
du -sh "$STORE" "$EVAL"

echo "=== verify ==="
export LD_LIBRARY_PATH=/usr/lib64-nvidia:${LD_LIBRARY_PATH:-}
python verify_store.py --store "$STORE" --sample 12 --expect 1611 > /content/work/verify_report.txt 2>&1
tail -5 /content/work/verify_report.txt

echo "=== archiving manifest + verification to Drive ==="
OUT=$DRIVE/runs/_prehistory/store
mkdir -p "$OUT"
cp "$DRIVE/store/bodycrop/nnUNetPlans_3d_fullres.manifest.json" "$OUT/tierA_manifest.json"
cp /content/work/verify_report.txt "$OUT/tierA_verify.txt"
cp /content/work/store_build.log  "$OUT/tierA_build.log" 2>/dev/null
cp "$EVAL/cases.txt" "$OUT/tierB_cases.txt" 2>/dev/null
cp "$EVAL/composition.json" "$OUT/tierB_composition.json" 2>/dev/null
ls -la "$OUT"
echo "BUILD COMPLETE $(date -u '+%F %T UTC')"
