#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# autoPET V -- put the nnU-Net weights into a nnUNet_results-style model folder and
# check every file against a pinned sha256 before exiting 0.
#
# Sources, selected with WEIGHTS_SOURCE:
#
#   repo    (default)   the shipped checkpoint, split across three ~90 MB parts that
#                       live IN THIS REPOSITORY under model/ (GitHub rejects a single
#                       blob over 100 MB).  The parts are concatenated and the result
#                       is checked against model/checkpoint_final.pth.sha256, so the
#                       build needs no network at all for the weights and the repo may
#                       stay private.
#   gdrive_file         the same model as three Google Drive files shared as "anyone
#                       with the link", addressed by file id.  Kept as a fallback.
#   gdrive              the organizers' public weights.zip, sha256-checked whole;
#                       the three files we need are extracted and the 461 MB zip
#                       is deleted.
#   release             a GitHub Release of this repo, one asset per file. Needs a
#                       public repo: private release assets require an auth header.
#   local               copy from an existing model folder (WEIGHTS_LOCAL_DIR).
#
# Whatever the source, the result is one nnUNet_results-style model folder
#
#     <MODEL_DIR>/plans.json
#     <MODEL_DIR>/dataset.json
#     <MODEL_DIR>/fold_0/checkpoint_final.pth
#
# Called from the Dockerfile at build time, and by hand for Docker-free testing.
#
# Usage
#   bash scripts/fetch_weights.sh [MODEL_DIR]
#
# Environment
#   WEIGHTS_SOURCE       repo | gdrive_file | gdrive | release | local  (default: repo)
#   WEIGHTS_REPO_DIR     repo mode: the directory holding the parts.  Default: the
#                        first of /opt/algorithm/model_src (where the Dockerfile COPYs
#                        model/) and <this script>/../model (a repo checkout).
#   FT_CKPT_GDRIVE_ID / FT_PLANS_GDRIVE_ID / FT_DATASET_JSON_GDRIVE_ID
#                        gdrive_file mode: the three Drive file ids
#   WEIGHTS_GDRIVE_ID    gdrive mode: Drive file id of the organizers' weights.zip
#   WEIGHTS_ZIP_SHA256   sha256 of that zip
#   WEIGHTS_BASE_URL     release mode: base URL of the release assets
#   WEIGHTS_LOCAL_DIR    local mode: source model folder
#   CHECKPOINT_SHA256 / PLANS_SHA256 / DATASET_JSON_SHA256
#                        per-file hashes, checked in ALL modes
#   SKIP_SHA256=1        skip verification (debug only -- never in the Dockerfile)
# ---------------------------------------------------------------------------
set -euo pipefail

MODEL_DIR="${1:-${AUTOPETV_MODEL_FOLDER:-/opt/algorithm/model}}"
WEIGHTS_SOURCE="${WEIGHTS_SOURCE:-repo}"

# --- pinned hashes ---------------------------------------------------------
# The fine-tuned 5-channel model (nnUNetTrainer_Interactive__nnUNetPlans_interactive__
# 3d_fullres, fold 0, 200/200 epochs).
#
# checkpoint_final.pth  246476509 bytes
# plans.json                 7256 bytes
# dataset.json               1009 bytes
FT_CHECKPOINT_SHA256="ed015e29025b5634e1803f58fb0230894042b523b6f087cb3316d905926ca1b7"
FT_PLANS_SHA256="36b7c8b23dc00af23cf05f43f427650d796988f24fe8436efc3733187304a342"
FT_DATASET_JSON_SHA256="3274b61d39f1b4d73a805d07e10e07543d4febbb1c7f4a8605e63230798295f7"

# The organizers' 4-channel baseline files (WEIGHTS_SOURCE=gdrive), kept so that the
# baseline image still builds from this script.
BL_CHECKPOINT_SHA256="4f47a4bbbbddc86575dc815a363f816891222fc40a550f882539784838ef9948"
BL_PLANS_SHA256="8177372342b62ca9dbc3a4e20d07ecdcbc6ca59adc004d07641643023460073f"
BL_DATASET_JSON_SHA256="27334fa8c3401dbcf1c5f5d6ef7512916572de42c0e3ffe30fb5bb52f53cfb35"

# Per-file hashes.  Explicit env (what the Dockerfile passes) always wins; the default
# follows the source, so neither route can be verified against the other's artifact.
if [ "$WEIGHTS_SOURCE" = "gdrive" ]; then
  CHECKPOINT_SHA256="${CHECKPOINT_SHA256:-$BL_CHECKPOINT_SHA256}"
  PLANS_SHA256="${PLANS_SHA256:-$BL_PLANS_SHA256}"
  DATASET_JSON_SHA256="${DATASET_JSON_SHA256:-$BL_DATASET_JSON_SHA256}"
else
  CHECKPOINT_SHA256="${CHECKPOINT_SHA256:-$FT_CHECKPOINT_SHA256}"
  PLANS_SHA256="${PLANS_SHA256:-$FT_PLANS_SHA256}"
  DATASET_JSON_SHA256="${DATASET_JSON_SHA256:-$FT_DATASET_JSON_SHA256}"
fi

# gdrive_file mode: the three Drive file ids, set once the files are shared as
# "anyone with the link" (docs/submission.md).
FT_CKPT_GDRIVE_ID="${FT_CKPT_GDRIVE_ID:-PUT_CHECKPOINT_FILE_ID_HERE}"
FT_PLANS_GDRIVE_ID="${FT_PLANS_GDRIVE_ID:-PUT_PLANS_FILE_ID_HERE}"
FT_DATASET_JSON_GDRIVE_ID="${FT_DATASET_JSON_GDRIVE_ID:-PUT_DATASET_JSON_FILE_ID_HERE}"

# gdrive mode (the organizers' baseline zip). The zip hash also matches the git-LFS
# pointer oid in the organizers' repo (nnunet-baseline/weights.zip).
#
# weights.zip           461339877 bytes
# checkpoint_final.pth  246763997 bytes  4f47a4bb...ef9948
# plans.json                17439 bytes  81773723...60073f
# dataset.json                574 bytes  27334fa8...f53cfb35
WEIGHTS_GDRIVE_ID="${WEIGHTS_GDRIVE_ID:-1G0HGHzQMXzslGDxFSNs5fq3RCeAu7M6l}"
WEIGHTS_ZIP_SHA256="${WEIGHTS_ZIP_SHA256:-3e1a0dac9b78f60ddd5115687ab4f262f3cba264b252ba9643cd86823d231552}"

# path of each file inside the model folder == path inside the zip, after the
# leading "nnUNet_results/Dataset998_AutoPETV/nnUNetTrainer__nnUNetPlans__3d_fullres/"
REL_checkpoint_final_pth="fold_0/checkpoint_final.pth"
REL_plans_json="plans.json"
REL_dataset_json="dataset.json"

PY=$(command -v python3 || command -v python)

rel_for () {
  case "$1" in
    checkpoint_final.pth) echo "$REL_checkpoint_final_pth" ;;
    plans.json)           echo "$REL_plans_json" ;;
    dataset.json)         echo "$REL_dataset_json" ;;
    *) echo "unknown file $1" >&2; exit 1 ;;
  esac
}
want_for () {
  case "$1" in
    checkpoint_final.pth) echo "$CHECKPOINT_SHA256" ;;
    plans.json)           echo "$PLANS_SHA256" ;;
    dataset.json)         echo "$DATASET_JSON_SHA256" ;;
    *) echo "unknown file $1" >&2; exit 1 ;;
  esac
}
verify () {  # $1 = path, $2 = expected sha256, $3 = label
  if [ "${SKIP_SHA256:-0}" = "1" ]; then
    echo "[fetch_weights] WARNING: sha256 verification skipped for $3"
    return 0
  fi
  echo "$2  $1" | sha256sum -c - >/dev/null \
    || { echo "FATAL: sha256 mismatch for $3" >&2
         echo "       expected $2" >&2
         echo "       got      $(sha256sum "$1" | cut -d' ' -f1)" >&2
         exit 1; }
  echo "[fetch_weights]   sha256 ok: $3 ($(stat -c '%s' "$1") bytes)"
}

mkdir -p "$MODEL_DIR/fold_0"
echo "[fetch_weights] source=$WEIGHTS_SOURCE  model_dir=$MODEL_DIR"

case "$WEIGHTS_SOURCE" in

  # ---------------------------------------------------------------------
  # The checkpoint ships INSIDE this repository, split into ~90 MB parts because
  # GitHub refuses a single blob over 100 MB.  No network, no file ids, no release:
  # `git clone` is the whole delivery mechanism, and it works while the repo is private.
  repo)
    SRC="${WEIGHTS_REPO_DIR:-}"
    if [ -z "$SRC" ]; then
      HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
      for cand in /opt/algorithm/model_src "$HERE/../model"; do
        # never take the DESTINATION as the source: in the image the script sits at
        # /opt/algorithm/scripts/, so "$HERE/../model" is /opt/algorithm/model itself
        [ -d "$cand" ] || continue
        [ "$(cd "$cand" && pwd)" = "$(cd "$MODEL_DIR" 2>/dev/null && pwd || echo /nonexistent)" ] && continue
        SRC="$cand"; break
      done
    fi
    [ -n "$SRC" ] && [ -d "$SRC" ] || {
      echo "FATAL: WEIGHTS_SOURCE=repo but no parts directory was found." >&2
      echo "       Looked for WEIGHTS_REPO_DIR, /opt/algorithm/model_src and <repo>/model." >&2
      echo "       In the image the Dockerfile must COPY model/ /opt/algorithm/model_src/." >&2
      exit 1; }
    SRC="$(cd "$SRC" && pwd)"
    echo "[fetch_weights] repo parts dir: $SRC"

    SUMS="$SRC/SHA256SUMS"
    WHOLE="$SRC/checkpoint_final.pth.sha256"
    for f in "$SUMS" "$WHOLE"; do
      [ -f "$f" ] || { echo "FATAL: missing $f" >&2; exit 1; }
    done

    # 1. every part and json against SHA256SUMS (a missing part fails here, because
    #    sha256sum -c reports it rather than skipping it)
    n_listed=$(grep -c 'checkpoint_final\.pth\.part' "$SUMS" || true)
    n_present=$(ls -1 "$SRC"/checkpoint_final.pth.part* 2>/dev/null | wc -l | tr -d ' ')
    echo "[fetch_weights] $n_listed part(s) listed in SHA256SUMS, $n_present on disk"
    if [ "$n_listed" != "$n_present" ]; then
      echo "FATAL: SHA256SUMS lists $n_listed part(s) but $n_present are present." >&2
      echo "       An unlisted part would be concatenated unverified; refusing." >&2
      exit 1
    fi
    [ "$n_listed" -ge 1 ] || { echo "FATAL: SHA256SUMS lists no checkpoint parts" >&2; exit 1; }
    if [ "${SKIP_SHA256:-0}" = "1" ]; then
      echo "[fetch_weights] WARNING: sha256 verification skipped for the repo parts"
    else
      ( cd "$SRC" && sha256sum -c SHA256SUMS ) || {
        echo "FATAL: a file in $SRC does not match SHA256SUMS." >&2
        echo "       The repository copy is corrupt or truncated (git-lfs not pulled?)." >&2
        exit 1; }
    fi

    # 2. concatenate, in the order the names sort in
    mkdir -p "$MODEL_DIR/fold_0"
    OUT="$MODEL_DIR/$REL_checkpoint_final_pth"
    echo "[fetch_weights] concatenating $n_listed part(s) -> $OUT"
    : > "$OUT"
    for part in $(ls -1 "$SRC"/checkpoint_final.pth.part* | sort); do
      echo "[fetch_weights]   + $(basename "$part") ($(stat -c '%s' "$part") bytes)"
      cat "$part" >> "$OUT"
    done
    echo "[fetch_weights] assembled $(stat -c '%s' "$OUT") bytes"

    # 3. the assembled file against its own recorded hash.  This is the check that
    #    catches a wrong concatenation ORDER, which the per-part hashes cannot.
    # accepts a bare hex line or full `sha256sum` output ("<hex>  <name>")
    want_whole="$(awk 'NF{print $1; exit}' "$WHOLE")"
    verify "$OUT" "$want_whole" "checkpoint_final.pth (assembled, vs $(basename "$WHOLE"))"

    # 4. and cross-check that recorded hash against the pin in this script / the
    #    Dockerfile ARG, so parts updated without updating the pin fail the build
    if [ "${SKIP_SHA256:-0}" != "1" ] && [ "$want_whole" != "$CHECKPOINT_SHA256" ]; then
      echo "FATAL: $(basename "$WHOLE") says $want_whole" >&2
      echo "       but the pinned CHECKPOINT_SHA256 is $CHECKPOINT_SHA256." >&2
      echo "       The parts under model/ were replaced without updating the pin" >&2
      echo "       (ARG CHECKPOINT_SHA256 in the Dockerfile and FT_CHECKPOINT_SHA256 here)." >&2
      exit 1
    fi

    # 5. the two small json files
    for name in plans.json dataset.json; do
      [ -f "$SRC/$name" ] || { echo "FATAL: missing $SRC/$name" >&2; exit 1; }
      cp -f "$SRC/$name" "$MODEL_DIR/$(rel_for "$name")"
      echo "[fetch_weights]   copied $name"
    done
    ;;

  # ---------------------------------------------------------------------
  # The fine-tuned model: three individually shared Google Drive files.
  gdrive_file)
    command -v gdown >/dev/null 2>&1 || { echo "FATAL: gdown not installed (pip install gdown)" >&2; exit 1; }
    id_for () {
      case "$1" in
        checkpoint_final.pth) echo "$FT_CKPT_GDRIVE_ID" ;;
        plans.json)           echo "$FT_PLANS_GDRIVE_ID" ;;
        dataset.json)         echo "$FT_DATASET_JSON_GDRIVE_ID" ;;
      esac
    }
    # minimum plausible size per file -- a Drive quota / "can't scan for viruses"
    # interstitial comes back as a small HTML page with HTTP 200, not as an error
    min_for () {
      case "$1" in
        checkpoint_final.pth) echo 100000000 ;;
        plans.json)           echo 1000 ;;
        dataset.json)         echo 100 ;;
      esac
    }
    for name in checkpoint_final.pth plans.json dataset.json; do
      fid="$(id_for "$name")"
      case "$fid" in
        ""|PUT_*|*"<"*|*"FILE_ID"*)
          echo "FATAL: WEIGHTS_SOURCE=gdrive_file but the Drive file id for $name is" >&2
          echo "       still the placeholder: '$fid'" >&2
          echo "       Share the three files as 'Anyone with the link' and put their ids" >&2
          echo "       into the FT_*_GDRIVE_ID ARGs of the repo-root Dockerfile." >&2
          echo "       See docs/submission.md section 2.1." >&2
          exit 1 ;;
      esac
      out="$MODEL_DIR/$(rel_for "$name")"
      echo "[fetch_weights] gdown $name ($fid) -> $out"
      gdown --no-cookies -q "$fid" -O "$out" \
        || { echo "FATAL: gdown failed for $name (id $fid). Is the file shared as" >&2
             echo "       'Anyone with the link'?  Is the id the FILE id (the segment" >&2
             echo "       after /file/d/ in the share URL) and not the folder id?" >&2
             exit 1; }
      [ -s "$out" ] || { echo "FATAL: download produced no file for $name" >&2; exit 1; }
      fsize=$(stat -c '%s' "$out")
      minsize=$(min_for "$name")
      if [ "$fsize" -lt "$minsize" ]; then
        echo "FATAL: $name is only $fsize bytes (expected >= $minsize) -- Google Drive" >&2
        echo "       probably returned a quota or virus-scan interstitial, or the file" >&2
        echo "       is not shared publicly." >&2
        head -c 400 "$out" >&2; echo >&2
        exit 1
      fi
      echo "[fetch_weights]   downloaded $name ($fsize bytes)"
    done
    ;;

  # ---------------------------------------------------------------------
  gdrive)
    command -v gdown >/dev/null 2>&1 || { echo "FATAL: gdown not installed (pip install gdown)" >&2; exit 1; }
    ZIP="${WEIGHTS_ZIP_PATH:-${TMPDIR:-/tmp}/weights.zip}"
    ZIP="$(cd "$(dirname "$ZIP")" && pwd)/$(basename "$ZIP")"
    echo "[fetch_weights] gdown $WEIGHTS_GDRIVE_ID -> $ZIP"
    gdown --no-cookies -q "$WEIGHTS_GDRIVE_ID" -O "$ZIP"
    [ -s "$ZIP" ] || { echo "FATAL: download produced no file" >&2; exit 1; }
    # A Google Drive quota/interstitial page comes back as a small HTML file
    # instead of an error -- the size check catches that before the hash does.
    zsize=$(stat -c '%s' "$ZIP")
    if [ "$zsize" -lt 100000000 ]; then
      echo "FATAL: downloaded file is only $zsize bytes -- Google Drive probably" >&2
      echo "       returned a quota or virus-scan interstitial instead of the zip." >&2
      head -c 400 "$ZIP" >&2; echo >&2
      exit 1
    fi
    verify "$ZIP" "$WEIGHTS_ZIP_SHA256" "weights.zip"

    echo "[fetch_weights] extracting 3 of 12 members (skipping checkpoint_best.pth,"
    echo "[fetch_weights]   progress.png, the training log and dataset_fingerprint.json)"
    PREFIX="nnUNet_results/Dataset998_AutoPETV/nnUNetTrainer__nnUNetPlans__3d_fullres"
    for name in checkpoint_final.pth plans.json dataset.json; do
      rel="$(rel_for "$name")"
      "$PY" - "$ZIP" "$PREFIX/$rel" "$MODEL_DIR/$rel" <<'PYEOF'
import shutil, sys, zipfile
zip_path, member, dest = sys.argv[1:4]
with zipfile.ZipFile(zip_path) as z:
    names = set(z.namelist())
    if member not in names:
        sys.exit(f"FATAL: {member!r} not in the zip. Members:\n  " + "\n  ".join(sorted(names)))
    with z.open(member) as src, open(dest, "wb") as dst:
        shutil.copyfileobj(src, dst, 1 << 20)
PYEOF
      echo "[fetch_weights]   extracted $rel"
    done
    rm -f "$ZIP"
    echo "[fetch_weights] zip deleted (it is 461 MB and must not reach the image)"
    ;;

  # ---------------------------------------------------------------------
  release)
    BASE_URL="${WEIGHTS_BASE_URL:-}"
    [ -n "$BASE_URL" ] || { echo "FATAL: WEIGHTS_SOURCE=release but WEIGHTS_BASE_URL is empty." >&2; exit 1; }
    case "$BASE_URL" in
      *OWNER/REPO*|*"<user>"*|*"<owner>"*|*example.com*)
        echo "FATAL: WEIGHTS_BASE_URL is still the placeholder:" >&2
        echo "         $BASE_URL" >&2
        echo "       Create the GitHub Release (docs/submission.md), then edit the" >&2
        echo "       ARG WEIGHTS_BASE_URL line in the repo-root Dockerfile and commit." >&2
        exit 1 ;;
    esac
    command -v curl >/dev/null 2>&1 || { echo "FATAL: curl not installed" >&2; exit 1; }
    # WEIGHTS_BASE_URL must point at a release holding the same artifacts the
    # per-file hashes describe; pointing it at the 4-channel baseline release fails
    # the sha256 check rather than shipping the wrong model.
    for name in checkpoint_final.pth plans.json dataset.json; do
      out="$MODEL_DIR/$(rel_for "$name")"
      echo "[fetch_weights] curl  $BASE_URL/$name -> $out"
      curl -fsSL --retry 5 --retry-delay 3 --retry-connrefused -o "$out" "$BASE_URL/$name"
    done
    ;;

  # ---------------------------------------------------------------------
  local)
    [ -n "${WEIGHTS_LOCAL_DIR:-}" ] || { echo "FATAL: WEIGHTS_SOURCE=local but WEIGHTS_LOCAL_DIR is empty." >&2; exit 1; }
    for name in checkpoint_final.pth plans.json dataset.json; do
      rel="$(rel_for "$name")"
      src="$WEIGHTS_LOCAL_DIR/$rel"
      echo "[fetch_weights] copy  $src -> $MODEL_DIR/$rel"
      [ -f "$src" ] || { echo "FATAL: missing $src" >&2; exit 1; }
      cp -f "$src" "$MODEL_DIR/$rel"
    done
    ;;

  *)
    echo "FATAL: unknown WEIGHTS_SOURCE=$WEIGHTS_SOURCE (repo|gdrive_file|gdrive|release|local)" >&2
    exit 1 ;;
esac

# --- per-file verification, whatever the source ----------------------------
for name in checkpoint_final.pth plans.json dataset.json; do
  verify "$MODEL_DIR/$(rel_for "$name")" "$(want_for "$name")" "$name"
done

echo "[fetch_weights] model folder ready: $MODEL_DIR"
ls -l "$MODEL_DIR" "$MODEL_DIR/fold_0"
du -sh "$MODEL_DIR"
