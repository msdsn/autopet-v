# =============================================================================
# autoPET V -- submission image.  Build context = repo root.
#
# Grand Challenge builds this from https://github.com/msdsn/autopet-v (branch main).
# It passes NO --build-arg, so every ARG default below must be the real value.
#
# Evaluation runtime (organizers' test.sh, unchanged):
#   docker run --memory=30g --memory-swap=30g --network=none --cap-drop=ALL \
#              --security-opt=no-new-privileges --shm-size=2g --gpus=all \
#              -v .../input:/input -v .../output:/output -v .../cache:/cache
# A10G 24 GB (sm_86), 8 CPU, 30 GB RAM, 20 min per iteration, 6 iterations per case.
#
# `--network=none` => nothing may be downloaded at inference time.  The weights are
# baked in at build time (below) and nnU-Net is initialised from an absolute path.
#
# The image ships two fine-tuned 5-channel interactive models -- a PlainConvUNet and a
# ResEncL warm-started from LesionTracer -- whose foreground probabilities are averaged
# under the post-processing layer (AUTOPETV_PREDICTOR=ensemble_postproc, harness row E2).
# =============================================================================
FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime
# python 3.11 · torch 2.6.0+cu124 · cuDNN 9.  cu124 covers the evaluation A10G (sm_86)
# and the development L4 (sm_89).  Do not switch to a `-devel` tag: it triples the
# image size and nothing here is compiled.

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

# curl + ca-certificates are needed only for the weight download at BUILD time.
RUN apt-get update \
 && apt-get install -y --no-install-recommends curl ca-certificates \
 && rm -rf /var/lib/apt/lists/*

# --- non-root user + the directory contract ---------------------------------
# /output/images/tumor-lesion-segmentation must exist at build time (the organizers'
# Dockerfile does this too); submission/process.py also mkdir -p's it at runtime.
RUN groupadd -r algorithm \
 && useradd -m --no-log-init -r -g algorithm algorithm \
 && mkdir -p /opt/algorithm /opt/algorithm/tmp /opt/algorithm/scripts \
             /opt/algorithm/model /opt/algorithm/model/fold_0 \
             /opt/algorithm/nnUNet_raw /opt/algorithm/nnUNet_preprocessed \
             /opt/algorithm/nnUNet_results \
             /input /output /output/images/tumor-lesion-segmentation /output/state \
             /cache /cache/state \
 && chown -R algorithm:algorithm /opt/algorithm /input /output /cache

ENV nnUNet_raw="/opt/algorithm/nnUNet_raw" \
    nnUNet_preprocessed="/opt/algorithm/nnUNet_preprocessed" \
    nnUNet_results="/opt/algorithm/nnUNet_results" \
    nnUNet_extTrainer="/opt/algorithm/src/train" \
    AUTOPETV_MODEL_FOLDER="/opt/algorithm/model" \
    AUTOPETV_TMP_DIR="/opt/algorithm/tmp" \
    AUTOPETV_PREDICTOR="ensemble_postproc" \
    AUTOPETV_ENSEMBLE_MEMBERS="/opt/algorithm/model,/opt/algorithm/model_re" \
    AUTOPETV_CHECKPOINT="checkpoint_final.pth" \
    AUTOPETV_POSTPROC_CONFIG="/opt/algorithm/submission/postproc_config.json" \
    AUTOPETV_MIRROR_AXES="" \
    AUTOPETV_NPP="1" \
    AUTOPETV_NPS="1" \
    AUTOPETV_CACHE_DIR="/cache" \
    AUTOPETV_OUTPUT_ROOT="/output" \
    MPLCONFIGDIR="/opt/algorithm/tmp/mpl" \
    PYTHONPATH="/opt/algorithm:/opt/algorithm/src" \
    OMP_NUM_THREADS=8
# `nnUNet_extTrainer` is how nnU-Net ≥ 2.8 finds a trainer class that does not live in
# site-packages.  The checkpoint records trainer_name = nnUNetTrainer_Interactive, and
# `initialize_from_trained_model_folder` rebuilds the network through that class, so
# without this the model folder cannot be loaded at all.  `src/predictor.py` also sets
# it defensively at runtime; both point at the same directory.
# NOTE: `AUTOPETV_STATE_DIR` is deliberately NOT set.  Leaving it unset selects the
# default pair of state roots, `/cache/state` and `/output/state`: the organizers'
# `nnunet-baseline/test.sh` bind-mounts `/cache` for "inter-iteration storage" (and
# `interactive_loop.py` wipes it once per case, so it survives the 6 calls), while a
# forum answer says `/output` persists.  The container writes both and reads whichever
# actually carried this case's state.  Setting `AUTOPETV_STATE_DIR` pins it back to a
# single root -- do that only to move the state somewhere else entirely (e.g. /tmp).
# `AUTOPETV_MIRROR_AXES=""` == no mirroring TTA (the default, and what every harness
# number was measured with).  "0,1,2" together with AUTOPETV_ENABLE_TTA=1 turns on the
# full 8-way TTA -- 8x the sliding-window cost, re-measure the budget first.

# --- python deps (own layer: rebuilt only when requirements change) ----------
COPY requirements-submission.txt /opt/algorithm/requirements-submission.txt
RUN python -m pip install --no-cache-dir -U pip \
 && python -m pip install --no-cache-dir -r /opt/algorithm/requirements-submission.txt \
 && python -c "import nnunetv2, SimpleITK, nibabel, cc3d, scipy, skimage, networkx, numpy, torch; \
print('nnunetv2', nnunetv2.__version__ if hasattr(nnunetv2,'__version__') else 'n/a'); \
print('torch', torch.__version__); print('numpy', numpy.__version__)"

# --- weights, sha256-pinned -------------------------------------------------
# Default: the fine-tuned 5-channel checkpoint ships INSIDE this repository, split
# into three ~90 MB parts under model/ because GitHub refuses a single blob over
# 100 MB.  `git clone` is therefore the entire delivery mechanism:
#
#   * **no network is used for the weights at build time** (pip still needs one);
#   * nothing has to be shared, uploaded or made public first -- the repo may stay
#     private, and there are no file ids or release assets to keep in step;
#   * the parts are checked individually against model/SHA256SUMS and the assembled
#     file against model/checkpoint_final.pth.sha256, which is the check that catches
#     a wrong concatenation order.
#
# Other sources supported by scripts/fetch_weights.sh, all still working:
#   WEIGHTS_SOURCE=gdrive_file  the same model as three shared Google Drive files
#   WEIGHTS_SOURCE=gdrive       the organizers' 4-channel baseline zip
#   WEIGHTS_SOURCE=release      a GitHub Release of this repo (needs the repo public)
#   WEIGHTS_SOURCE=local        copy from WEIGHTS_LOCAL_DIR (offline re-verification)
ARG WEIGHTS_SOURCE="repo"
ARG FT_CKPT_GDRIVE_ID="PUT_CHECKPOINT_FILE_ID_HERE"
ARG FT_PLANS_GDRIVE_ID="PUT_PLANS_FILE_ID_HERE"
ARG FT_DATASET_JSON_GDRIVE_ID="PUT_DATASET_JSON_FILE_ID_HERE"
# sha256 / size of the fine-tuned model:
#   checkpoint_final.pth  246476509 B
#   plans.json                 7256 B   (nnUNetPlans_interactive, 5 channels)
#   dataset.json               1009 B   (5 channel_names, 2-4 = noNorm)
ARG CHECKPOINT_SHA256="ed015e29025b5634e1803f58fb0230894042b523b6f087cb3316d905926ca1b7"
ARG PLANS_SHA256="36b7c8b23dc00af23cf05f43f427650d796988f24fe8436efc3733187304a342"
ARG DATASET_JSON_SHA256="3274b61d39f1b4d73a805d07e10e07543d4febbb1c7f4a8605e63230798295f7"
# Baseline route (only read when WEIGHTS_SOURCE=gdrive / release)
ARG WEIGHTS_GDRIVE_ID="1G0HGHzQMXzslGDxFSNs5fq3RCeAu7M6l"
ARG WEIGHTS_ZIP_SHA256="3e1a0dac9b78f60ddd5115687ab4f262f3cba264b252ba9643cd86823d231552"
ARG WEIGHTS_BASE_URL="https://github.com/msdsn/autopet-v/releases/download/weights-v0"

# The parts land in their own directory, NOT in /opt/algorithm/model: the fetch step
# assembles into the latter, and a source that is also the destination would be a
# footgun.  Own layer, so a code edit does not re-copy 236 MB.
#
# Cost, stated plainly: the parts (236 MB) and the assembled checkpoint (236 MB) both
# end up in the image, because `rm -rf` in a later RUN cannot shrink an earlier COPY
# layer.  The `rm` below is still worth doing -- it keeps the running container clean
# -- but it saves no image size.  ~236 MB on a ~7 GB image is not worth the risk of a
# multi-stage restructure we have no way to build-test anywhere.
COPY model/ /opt/algorithm/model_src/

COPY scripts/fetch_weights.sh /opt/algorithm/scripts/fetch_weights.sh
# gdown is installed ONLY for the Drive routes.  With the default `repo` source the
# weights step touches no network at all.
RUN case "$WEIGHTS_SOURCE" in \
      gdrive|gdrive_file) python -m pip install --no-cache-dir "gdown==5.2.2" ;; \
      *) echo "[weights] source=$WEIGHTS_SOURCE -- no network needed, skipping gdown" ;; \
    esac \
 && WEIGHTS_SOURCE="$WEIGHTS_SOURCE" \
    WEIGHTS_REPO_DIR="/opt/algorithm/model_src" \
    FT_CKPT_GDRIVE_ID="$FT_CKPT_GDRIVE_ID" \
    FT_PLANS_GDRIVE_ID="$FT_PLANS_GDRIVE_ID" \
    FT_DATASET_JSON_GDRIVE_ID="$FT_DATASET_JSON_GDRIVE_ID" \
    WEIGHTS_GDRIVE_ID="$WEIGHTS_GDRIVE_ID" \
    WEIGHTS_ZIP_SHA256="$WEIGHTS_ZIP_SHA256" \
    WEIGHTS_BASE_URL="$WEIGHTS_BASE_URL" \
    WEIGHTS_ZIP_PATH="/tmp/weights.zip" \
    CHECKPOINT_SHA256="$CHECKPOINT_SHA256" \
    PLANS_SHA256="$PLANS_SHA256" \
    DATASET_JSON_SHA256="$DATASET_JSON_SHA256" \
    bash /opt/algorithm/scripts/fetch_weights.sh "$AUTOPETV_MODEL_FOLDER" \
 && (python -m pip uninstall -y gdown || true) \
 && rm -f /tmp/weights.zip \
 && rm -rf /opt/algorithm/model_src \
 && du -sh /opt/algorithm/model \
 && chown -R algorithm:algorithm /opt/algorithm/model

# --- second ensemble member: the ResEncL model (row E2) --------------------
# Its plans.json / dataset.json are small enough to live in the repo; the 410 MB
# checkpoint is not, so it is fetched once at build time from a read-only Drive link
# and pinned by sha256.  The file holds weights only -- the optimizer state that makes
# the training checkpoint 819 MB is stripped, and the remaining tensors were verified
# equal to the training checkpoint's before upload.  Set WITH_RE_MEMBER=0 to build the
# single-model image (then also set AUTOPETV_PREDICTOR=interactive_postproc).
ARG WITH_RE_MEMBER="1"
ARG RE_CKPT_GDRIVE_ID="1Vt-x2JZIbZRn5yL7lzTjZWzxmRIJxQNJ"
ARG RE_CHECKPOINT_SHA256="ec85432120826bee3ab841cf99469cb751603bd562e8af087271e50143dddeb1"
COPY model_re/ /opt/algorithm/model_re/
RUN if [ "$WITH_RE_MEMBER" = "1" ]; then \
      python -m pip install --no-cache-dir "gdown==5.2.2" \
   && mkdir -p /opt/algorithm/model_re/fold_0 \
   && gdown --no-cookies -q "$RE_CKPT_GDRIVE_ID" \
            -O /opt/algorithm/model_re/fold_0/checkpoint_final.pth \
   && echo "$RE_CHECKPOINT_SHA256  /opt/algorithm/model_re/fold_0/checkpoint_final.pth" \
      | sha256sum -c - \
   && (python -m pip uninstall -y gdown || true) \
   && du -sh /opt/algorithm/model_re \
   && chown -R algorithm:algorithm /opt/algorithm/model_re ; \
    else echo "[weights] WITH_RE_MEMBER=0 -- single-model image"; fi

# ---------------------------------------------------------------------------
# FALLBACK (git-LFS) -- if every network route fails, delete the block above and
# use this instead, after `git lfs track "weights/**"` and committing the files:
#
#   COPY weights/checkpoint_final.pth /opt/algorithm/model/fold_0/checkpoint_final.pth
#   COPY weights/plans.json           /opt/algorithm/model/plans.json
#   COPY weights/dataset.json         /opt/algorithm/model/dataset.json
#
# git-LFS gives a free account 1 GB/month of bandwidth and the build spends
# ~250 MB of it on every rebuild -- four rebuilds exhaust the quota.
# ---------------------------------------------------------------------------

# --- application code (last: code edits must not invalidate the weights layer) ---
# src/train must be inside the image: it holds `nnUNetTrainer_Interactive`, which
# nnU-Net imports (via nnUNet_extTrainer, above) to rebuild the network from the
# checkpoint, and `src/train/guidance.py`, whose clipped-EDT encoder the predictor
# calls so that inference stamps *exactly* the channels training saw.
COPY --chown=algorithm:algorithm src/ /opt/algorithm/src/
COPY --chown=algorithm:algorithm submission/ /opt/algorithm/submission/

USER algorithm
WORKDIR /opt/algorithm

# Build-time smoke test, so a broken image fails the BUILD and not the first
# 20-minute evaluation job.  It checks, without needing a GPU:
#   1. the entry point and the predictor factory import;
#   2. the shipped post-processing config parses into a PostProcConfig;
#   3. the model folder is a 5-channel interactive one (dataset.json + plans.json);
#   4. nnU-Net can actually FIND nnUNetTrainer_Interactive through nnUNet_extTrainer,
#      which is what `initialize_from_trained_model_folder` needs at runtime;
#   5. every module in src/train imports cleanly inside the image -- nnU-Net's trainer
#      search imports the whole folder, and it only stops early by luck of alphabet.
RUN python -m submission.tests.check_image

ENTRYPOINT ["python", "-m", "submission.process"]
