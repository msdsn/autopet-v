# Submitting to Grand Challenge

How the container is built, how to release the weights, how to create the algorithm on
Grand Challenge (GC), and what to look for in the logs. The tick-list version is
[`submission_checklist.md`](submission_checklist.md).

The repo `https://github.com/msdsn/autopet-v` (branch `main`) **is** the build context:
GC clones it and runs `docker build .` against the `Dockerfile` at the root. No
`--build-arg` is passed, so every `ARG` default in the Dockerfile must already be the
real value when you push.

**Versions.** `v0.1-skeleton` shipped the organizers' unmodified 4-channel baseline
through our own Dockerfile and entry point — a walking skeleton, to de-risk the build and
the interface. **`v0.2`** ships the real method: our fine-tuned **5-channel interactive
nnU-Net** (currently `nnUNetTrainer_InteractiveV2_negfp`) under the **post-processing
layer**, i.e.
variant `B3` of `configs/ablations.json`. Every v0.1 guarantee is unchanged — deterministic,
well inside 20 min/iteration and 30 GB, no network at runtime, exactly-one-input
assertions, empty-mask-on-error, and state that is entirely optional. §9 is the
step-by-step list of what the *user* has to do for v0.2.

---

## 1. What gets built

```
repo root                     -> image
  Dockerfile                     FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime
  requirements-submission.txt    pip install (pinned; torch comes from the base image)
  model/                         the checkpoint itself, in three ~90 MB parts + hashes
  scripts/fetch_weights.sh       assemble + sha256 the weights  -> /opt/algorithm/model
  src/                        -> /opt/algorithm/src         (model, predictors, postproc,
                                 and src/train: the trainer class + the guidance encoder)
  submission/                 -> /opt/algorithm/submission  (the /input -> /output glue,
                                 including the shipped postproc_config.json)
                                 ENTRYPOINT python -m submission.process
```

`submission/process.py` is the entry point. Per call it:

1. asserts **exactly one** file in `/input/images/ct/` and one in `/input/images/pet/`
   (the organizers' baseline uses `os.listdir(...)[0]`, whose order is undefined);
2. takes the **CT**'s uuid as the output filename — not the PET's;
3. reads `/input/lesion-clicks.json` (absent / empty / malformed → iteration 0);
4. converts both `.mha` to `.nii.gz` with SimpleITK (bit-exact; verified), derives the
   **case fingerprint** (geometry + a strided PET digest — the per-call uuid is random and
   cannot identify a case) and opens `/output/state/case_<fingerprint>/`;
5. runs the predictor named by `AUTOPETV_PREDICTOR` — objects imported from `src/`, never
   copied, so a harness number is a container number. For v0.2 that is
   `PostProcPredictor(InteractiveNNUNetPredictor(...))`;
6. writes `<CT-uuid>.mha`, uint8, with `CopyInformation` from the input CT, then re-reads
   it and asserts the geometry round-tripped;
7. writes the final mask back into the per-case state dir (channel 4 of the next
   iteration), **mirrors that directory into every other writable state root**, and drops
   a marker in each root, logging what was already there;
8. **on any exception in the model path: writes an empty mask of the right geometry and
   exits 0.** An empty mask costs one iteration; a crash can cost the whole case. This
   covers a missing/corrupt model folder, a CUDA OOM, and a malformed `/input` -- all
   verified (see §7).

### The model (v1.1: two members)

v1.1 ships an ensemble of two interactive models that share the five-channel contract
below but not their architecture:

| member | folder in the image | architecture | plans | weights |
|---|---|---|---|---|
| B10 | `/opt/algorithm/model` | `PlainConvUNet`, 30.8 M | `nnUNetPlans_interactive` | repo parts (§2.1) |
| RE | `/opt/algorithm/model_re` | `ResidualEncoderUNet`, 102.4 M, 192³ patches | `nnUNetPlans_re` | Drive file, sha256-pinned |

`src/ensemble_predictor.py::EnsembleInteractivePredictor` runs both members to
completion and averages their foreground probability **on the original image grid** —
the only geometry the two plans agree on, since their preprocessed grids differ in
spacing-independent shape. The mask is `p_fg > 0.5`, which is what nnU-Net's own
`argmax` reduces to for two classes, so weights `[1, 0]` reproduce member 0 bit for bit
(`src/test_ensemble_predictor.py`). Every member is handed the **ensemble's** previous
final mask as channel 4, never its own, so no member ever sees a state the shipped
pipeline would not produce.

The RE member is warm-started from the autoPET III winning model of team LesionTracer
(Rokuss et al., arXiv:2409.09478; Zenodo 10.5281/zenodo.14007247, CC BY 4.0) and
fine-tuned for 100 epochs on the interactive task; `docs/train_pipeline.md` has the
surgery, the PET renormalisation and the identity gate.

Cost: one iteration is the sum of the members' iterations. Measured on an idle A100 over
five cases at iteration 0 — one model 6.4 s median, the two-member ensemble 17.7 s
(×2.8). Scaled to the container's measured 34.8–57.6 s per iteration on the challenge
hardware that is 96–159 s, i.e. 8–13 % of the 20-minute per-iteration budget.

`AUTOPETV_PREDICTOR=ensemble_postproc` with `AUTOPETV_ENSEMBLE_MEMBERS` selects it;
setting `WITH_RE_MEMBER=0` at build time and `AUTOPETV_PREDICTOR=interactive_postproc`
builds the single-model v0.2 image unchanged.

### The model (v0.2)

`src/predictor.py::InteractiveNNUNetPredictor`, five input channels:

| ch | content |
|---|---|
| 0 | CT (`CTNormalization`) |
| 1 | PET SUV (`ZScoreNormalization`) |
| 2 | tumor guidance, clipped EDT `max(0, 1 − d/R)`, R = 10 voxels of the **preprocessed** grid |
| 3 | background guidance, same encoding |
| 4 | the **previous iteration's final mask**, binary |

The guidance is stamped on the plans-spacing grid with the very function the training
transform uses (`src/train/guidance.py`), not re-implemented — see `docs/eval_harness.md`
and `docs/train_pipeline.md`. Two consequences for the image:

* `src/train/` must be **inside** the image and on `nnUNet_extTrainer`: the checkpoint
  records the `trainer_name` it was trained with (currently
  `nnUNetTrainer_InteractiveV2_negfp`), and
  `initialize_from_trained_model_folder` rebuilds the network through that class. The
  Dockerfile sets `nnUNet_extTrainer=/opt/algorithm/src/train`; `predictor_gc.py` and
  `InteractiveNNUNetPredictor` also set it defensively at runtime, and the build-time
  self-check (`submission/tests/check_image.py`) proves nnU-Net can actually resolve the
  class before the image is ever run.
* channel 4 has to survive between two container calls — see §6.

The post-processing layer is `src/postproc/PostProcPredictor` configured from
`submission/postproc_config.json` — **the single source of truth for the shipped
thresholds**. It descends from the `B3` rung of `configs/ablations.json` (bg + fg scribble
compliance, component cleanup with the tracer SUV floor, negative gate, monotone min-max
blend, `pass_cached_prev_pred = true`) and the ablation sweep has since moved the cleanup
and gate thresholds on from it.

Drift between "what was measured" and "what is submitted" is designed out rather than
checked for: the offline harness is driven with **this same file**
(`interactive_eval.py --postproc_config submission/postproc_config.json`), and
`run_v02_suite.py` then compares the two sides' masks bitwise. **Editing that file is the
whole procedure for shipping a different rung** — nothing else changes, and no value is
duplicated anywhere.

### Configuration (env vars, all set in the Dockerfile)

| Variable | Default | Meaning |
|---|---|---|
| `AUTOPETV_PREDICTOR` | `interactive_postproc` | `interactive_postproc` (v0.2) \| `interactive` \| `postproc` \| `baseline` (v0.1) \| `threshold` |
| `AUTOPETV_MODEL_FOLDER` | `/opt/algorithm/model` | model folder: `plans.json`, `dataset.json`, `fold_0/checkpoint_final.pth` |
| `AUTOPETV_POSTPROC_CONFIG` | `/opt/algorithm/submission/postproc_config.json` | the `PostProcConfig`; keys starting with `_` are comments |
| `nnUNet_extTrainer` | `/opt/algorithm/src/train` | where nnU-Net finds the checkpoint's `nnUNetTrainer_*` class |
| `AUTOPETV_STATE_DIRS` / `AUTOPETV_STATE_DIR` | *(unset)* → `/cache/state` **and** `/output/state` | inter-iteration state roots, in read-preference order (see §6) |
| `AUTOPETV_CACHE_DIR` / `AUTOPETV_OUTPUT_ROOT` | `/cache` / `/output` | the two mounts the state roots are derived from |
| `AUTOPETV_STATE_ENABLED` | `1` | `0` disables all state reads/writes |
| `AUTOPETV_SAVE_PREV_MASK` | `1` | `0` skips `prev_final_mask.npz` (the redundant, bit-packed channel-4 copy, ~6 kB) |
| `AUTOPETV_INPUT_DIR` / `AUTOPETV_OUTPUT_DIR` / `AUTOPETV_TMP_DIR` | `/input` / `/output/images/tumor-lesion-segmentation` / `/opt/algorithm/tmp` | overridable for Docker-free testing |
| `AUTOPETV_GUIDANCE_RADIUS` | 10.0 (via `nnUNet_interactive_radius`) | clipped-EDT radius, **in voxels of the preprocessed grid**; must match the trainer |
| `AUTOPETV_MIRROR_AXES` | `""` = none | mirroring TTA axes, e.g. `0,1,2`; only used with `AUTOPETV_ENABLE_TTA=1` |
| `AUTOPETV_ENABLE_TTA`, `AUTOPETV_TILE_STEP`, `AUTOPETV_NPP`, `AUTOPETV_NPS`, `AUTOPETV_FOLDS`, `AUTOPETV_CHECKPOINT` | off, 0.5, 1, 1, `0`, `checkpoint_final.pth` | inference knobs |
| `AUTOPETV_CUDNN_DETERMINISTIC` | `1` | `cudnn.deterministic=True`, `cudnn.benchmark=False` |

`AUTOPETV_NPP=1 / AUTOPETV_NPS=1`: `nnUNetv2_predict` defaults to 3 preprocessing and 3
export workers, which pass volumes between processes through `/dev/shm`. The evaluator
gives us `--shm-size=2g`, and one whole-body PET/CT at native resolution is already
hundreds of MB per worker. Measurement showed no speed advantage from 3 workers on a
single case (223 s vs 188 s wall, within run-to-run noise), so we take the safe setting.

Inference is made **deterministic** — `cudnn.benchmark=False`, `cudnn.deterministic=True`,
and `random` / `numpy` / `torch` seeded — before the model is built, so the same input
gives a byte-identical mask on every call. This matters beyond tidiness: iterations 1-5
are scored against a mask the evaluator derived from our own earlier output, so
non-reproducible predictions make our offline measurements unfalsifiable.

---

## 2. Weights

Five build-time sources, selected by the `WEIGHTS_SOURCE` build ARG in the `Dockerfile`
and implemented in `scripts/fetch_weights.sh`. **Every one of them verifies sha256 and
exits 1 on a mismatch, so a bad or truncated artifact fails the build rather than
producing an image that silently segments nothing.**

| `WEIGHTS_SOURCE` | what it uses | network at build? | repo may be private? |
|---|---|---|---|
| `repo` *(default)* | the parts committed under `model/` in this repository | **no** | **yes** |
| `gdrive_file` | the same model as three shared Google Drive files, by file id | yes | yes |
| `gdrive` | the organizers' public `weights.zip` (v0.1 baseline) | yes | yes |
| `release` | one asset per file from a GitHub Release of this repo | yes | no — must be public |
| `local` | copy from an existing model folder | no | n/a |

Whatever the source, the result is one model folder at `/opt/algorithm/model`:

```
/opt/algorithm/model/plans.json
/opt/algorithm/model/dataset.json
/opt/algorithm/model/fold_0/checkpoint_final.pth
```

### Pinned hashes

**v0.2 — the fine-tuned 5-channel model** (`nnUNetTrainer_InteractiveV2_negfp__nnUNetPlans_interactive__3d_fullres`,
fold 0):

| file | bytes | sha256 |
|---|---|---|
| `checkpoint_final.pth` (assembled) | 246 476 509 | `ed015e29025b5634e1803f58fb0230894042b523b6f087cb3316d905926ca1b7` |
| `plans.json` | 7 256 | `36b7c8b23dc00af23cf05f43f427650d796988f24fe8436efc3733187304a342` |
| `dataset.json` | 1 009 | `3274b61d39f1b4d73a805d07e10e07543d4febbb1c7f4a8605e63230798295f7` |

The three per-part hashes live in `model/SHA256SUMS` and are checked there.

**v0.1 — the organizers' 4-channel baseline** (only with `WEIGHTS_SOURCE=gdrive`):

| file | bytes | sha256 |
|---|---|---|
| `weights.zip` (Drive id `1G0HGHzQMXzslGDxFSNs5fq3RCeAu7M6l`) | 461339877 | `3e1a0dac9b78f60ddd5115687ab4f262f3cba264b252ba9643cd86823d231552` |
| `checkpoint_final.pth` | 246763997 | `4f47a4bbbbddc86575dc815a363f816891222fc40a550f882539784838ef9948` |
| `plans.json` | 17439 | `8177372342b62ca9dbc3a4e20d07ecdcbc6ca59adc004d07641643023460073f` |
| `dataset.json` | 574 | `27334fa8c3401dbcf1c5f5d6ef7512916572de42c0e3ffe30fb5bb52f53cfb35` |

The zip hash independently matches the git-LFS pointer oid committed in the organizers'
own `nnunet-baseline/weights.zip`, which confirms it is their unmodified artifact.
`fetch_weights.sh` picks the baseline hashes automatically when `WEIGHTS_SOURCE=gdrive`
and ours otherwise, so the two routes can never be verified against each other's artifact.

### 2.1 Default: the checkpoint ships in the repository — **nothing to do**

`git clone` is the whole delivery mechanism. There is nothing to share, upload, or make
public, no file ids and no release assets to keep in step with the hashes, and **the
weights step touches no network at all** (pip still needs one for the Python
dependencies). This works while the repository is private.

GitHub refuses a single blob over 100 MB, so the 236 MB checkpoint is committed as three
parts:

```
model/
  checkpoint_final.pth.part00      94 371 840 B   (90 MiB)
  checkpoint_final.pth.part01      94 371 840 B   (90 MiB)
  checkpoint_final.pth.part02      57 732 829 B
  checkpoint_final.pth.sha256      sha256 of the ASSEMBLED file
  SHA256SUMS                       sha256 of each part and of the two json files
  plans.json                        7 256 B
  dataset.json                      1 009 B
```

The build then, in `fetch_weights.sh`:

1. locates the parts (`WEIGHTS_REPO_DIR`, else `/opt/algorithm/model_src` where the
   Dockerfile `COPY`s them, else `<repo>/model` — never the *destination* folder);
2. checks that the number of `.part*` files on disk equals the number listed in
   `SHA256SUMS`, so an unlisted extra part can never be concatenated unverified;
3. `sha256sum -c SHA256SUMS` over every part and both json files — a **missing** part is
   reported here rather than silently skipped;
4. concatenates the parts in sorted name order into
   `/opt/algorithm/model/fold_0/checkpoint_final.pth`;
5. verifies the assembled file against `checkpoint_final.pth.sha256`. **This is the check
   that catches a wrong concatenation order**, which per-part hashes cannot;
6. cross-checks that recorded hash against the pinned `ARG CHECKPOINT_SHA256`, so parts
   replaced without updating the pin fail the build instead of shipping a different model;
7. copies `plans.json` and `dataset.json`, and re-verifies all three files against the
   pinned hashes in the common tail.

Measured on the GPU box: **4 s**, 246 476 509 bytes assembled, sha256
`ed015e29…` — see §7.5.

**Updating the checkpoint** (regenerating the parts):

```bash
cd model/
split -b 90m -d -a 2 /path/to/checkpoint_final.pth checkpoint_final.pth.part
sha256sum /path/to/checkpoint_final.pth | cut -d" " -f1 > checkpoint_final.pth.sha256
sha256sum checkpoint_final.pth.part* plans.json dataset.json > SHA256SUMS
# then update ARG CHECKPOINT_SHA256 in the Dockerfile and FT_CHECKPOINT_SHA256 in
# scripts/fetch_weights.sh to the value in checkpoint_final.pth.sha256 -- step 6 above
# fails the build if you forget.
```

Keep each part under 100 MB: GitHub rejects the push otherwise, and `git` stores these as
ordinary blobs (no LFS, so no bandwidth quota — the LFS route in §2.5 exists precisely
because a free account gets only 1 GiB/month).

### 2.1b Fallback: the same model as three Google Drive files

`WEIGHTS_SOURCE=gdrive_file` downloads `checkpoint_final.pth`, `plans.json` and
`dataset.json` from three individually shared Drive files (`ARG FT_CKPT_GDRIVE_ID`,
`FT_PLANS_GDRIVE_ID`, `FT_DATASET_JSON_GDRIVE_ID`). It needs no auth header, so it also
works with a private repo — but it needs someone to share the files, keep the ids in the
Dockerfile, and it can be rate-limited by Drive. **The `repo` default removes all three
problems, so this route is now only a fallback** (for instance if a future checkpoint is
too large to be comfortable in git). Each file is size-checked before hashing, because a
Drive quota or virus-scan interstitial arrives as a small HTML page with HTTP 200 rather
than as an error, and a placeholder id exits 1 in the first seconds of the build.

### 2.2 v0.1 route: the organizers' Google Drive zip

`WEIGHTS_SOURCE=gdrive`. This is the same file their `nnunet-baseline/check_weights.sh`
downloads. We redistribute nothing, so **the repo can stay private.** The build:

1. `pip install gdown==5.2.2`
2. `gdown 1G0HGHzQMXzslGDxFSNs5fq3RCeAu7M6l -O /tmp/weights.zip`
3. size sanity check (a Drive quota or virus-scan interstitial comes back as a small HTML
   page, not an error — the size check catches that with a readable message before the
   hash does), then sha256 of the whole zip
4. extract **3 of the 12 members** with Python's `zipfile` — `checkpoint_final.pth`,
   `plans.json`, `dataset.json`; `checkpoint_best.pth`, `progress.png`, the 422 KB
   training log and `dataset_fingerprint.json` are left behind
5. sha256 each extracted file
6. delete the zip **in the same layer**, so the 461 MB never reaches the image
7. `pip uninstall gdown` — it is a build-time tool, not a runtime dependency

Measured end to end on the GPU box: **26 s**, final model folder **236 MB**.

The known risk of this route is that Google Drive rate-limits unattended downloads. If the
GC build ever fails with the interstitial message, switch to §2.3 (and make the repo
public) or to the git-LFS fallback in §2.5. That is exactly why the failure mode is a loud,
specific message rather than a hash mismatch.

### 2.3 `release`: a GitHub Release of our repo — only once the repo is public

> **The repo must be public before the build.** Release assets of a *private* repo need an
> `Authorization` header, which the build cannot supply. Everything else about linking a
> private repo to GC works fine (§3) — this is purely about the asset download.

Create a token with `Contents: write` on the repo (fine-grained) or the classic `repo`
scope, then, from the GPU box:

```bash
export GH_TOKEN=<token>
REPO=msdsn/autopet-v
TAG=weights-v1
W=/content/drive/MyDrive/autoPET/weights/nnUNet_results/Dataset998_AutoPETV/nnUNetTrainer__nnUNetPlans__3d_fullres

# 1. create the release (the tag is created from main if it does not exist)
RELEASE_ID=$(curl -sS -X POST \
  -H "Authorization: Bearer $GH_TOKEN" \
  -H "Accept: application/vnd.github+json" \
  https://api.github.com/repos/$REPO/releases \
  -d "{\"tag_name\":\"$TAG\",\"name\":\"$TAG\",\"target_commitish\":\"main\",
       \"body\":\"nnU-Net 3d_fullres fold 0, 4-channel (CT, PET, fg-heatmap, bg-heatmap).\",
       \"draft\":false,\"prerelease\":false}" \
  | python -c "import sys,json; print(json.load(sys.stdin)['id'])")
echo "release id = $RELEASE_ID"

# 2. upload the three assets
for f in "$W/fold_0/checkpoint_final.pth" "$W/plans.json" "$W/dataset.json"; do
  n=$(basename "$f")
  echo "uploading $n ..."
  curl -sS -X POST \
    -H "Authorization: Bearer $GH_TOKEN" \
    -H "Content-Type: application/octet-stream" \
    --data-binary @"$f" \
    "https://uploads.github.com/repos/$REPO/releases/$RELEASE_ID/assets?name=$n" \
    | python -c "import sys,json; d=json.load(sys.stdin); print(d.get('name'), d.get('size'), d.get('state'))"
done

# 3. verify the public, UNAUTHENTICATED URLs -- this is exactly what the build does
for n in checkpoint_final.pth plans.json dataset.json; do
  curl -fsIL -o /dev/null -w "%{http_code} $n\n" \
    https://github.com/$REPO/releases/download/$TAG/$n
done
```

Web-UI alternative: `https://github.com/msdsn/autopet-v/releases/new` → tag, target `main`,
drag the three files into "Attach binaries", **Publish release**. Then run step 3 above.

Re-uploading an asset under an existing name fails — delete the old one
(`DELETE /repos/$REPO/releases/assets/<asset_id>`) or bump the tag.

Then switch the Dockerfile over and recompute the hashes for the new checkpoint:

```bash
sed -i 's|ARG WEIGHTS_SOURCE=.*|ARG WEIGHTS_SOURCE="release"|' Dockerfile
sed -i 's|ARG WEIGHTS_BASE_URL=.*|ARG WEIGHTS_BASE_URL="https://github.com/msdsn/autopet-v/releases/download/weights-v1"|' Dockerfile
sha256sum "$W/fold_0/checkpoint_final.pth" "$W/plans.json" "$W/dataset.json"   # -> ARG *_SHA256
git commit -am "submission: switch to our own weights release" && git push
```

`fetch_weights.sh` refuses to run against a placeholder URL, so a forgotten edit fails the
build in seconds instead of producing a weightless image.

### 2.4 Re-verifying the weights without any network

Both offline modes run the *same* hash checks the build runs, so they answer "are these
the bytes the Dockerfile pins?" without a network:

```bash
# the shipped route, exactly as the build does it (assembles from model/ in a checkout)
WEIGHTS_SOURCE=repo bash scripts/fetch_weights.sh /tmp/weights_check

# an already-assembled model folder from somewhere else
WEIGHTS_SOURCE=local WEIGHTS_LOCAL_DIR=/path/to/model bash scripts/fetch_weights.sh /tmp/weights_check
```

`repo` mode is what to run after regenerating the parts: it fails if a part is missing,
if a part is corrupt, if the concatenation would be misordered, or if
`checkpoint_final.pth.sha256` and the pinned `ARG CHECKPOINT_SHA256` disagree. Measured
results and the failure-path table are in §7.5.

### 2.5 Fallback — git-LFS

If both network routes fail, track the weights with LFS and swap the download block in the
`Dockerfile` for the three `COPY` lines already written out in the comment there:

```bash
git lfs install
git lfs track "weights/**"
mkdir -p weights && cp "$W/fold_0/checkpoint_final.pth" "$W/plans.json" "$W/dataset.json" weights/
git add .gitattributes weights Dockerfile && git commit -m "weights via LFS" && git push
```

Cost: a free account gets 1 GiB/month of LFS bandwidth and **every GC rebuild spends
~250 MB of it**, so four rebuilds exhaust the quota and the fifth build fails. Grand
Challenge's own documentation warns about exactly this.

---

## 3. Create the algorithm on Grand Challenge

**Private repositories are supported.** Grand Challenge's own documentation on linking a
GitHub repository says to "select the Github account with the (private) repository you
want to link here" — you install their GitHub App on the repo and it clones with the App's
credentials. What the documentation *does* require, private or not:

* **a `Dockerfile` in the repository root** — ours is there, and the build context is the
  repo root;
* **an open-source licence on the repository** — we ship Apache-2.0 in `LICENSE`. Check
  this is present before linking; it is an explicit requirement, not a formality;
* **admin rights on the repo**, so you can authorise the App;
* builds are triggered **per tag** — "a build will be started for each tag".

The one thing a private repo does break is the `release` weights route (§2.3), because
release assets of a private repo need an auth header the build cannot supply. The default
`gdrive` route has no such problem, which is why the skeleton uses it.

1. Go to the challenge's submission page and use the **"create your algorithm"** link it
   provides. That link pre-fills the input/output sockets; creating an algorithm from
   scratch and guessing the sockets is the most common way to get a container that builds
   and then fails at runtime.
2. Confirm the sockets are exactly:

   | direction | socket | relative path in the container |
   |---|---|---|
   | input | CT image | `images/ct/<uuid>.mha` |
   | input | PET image | `images/pet/<uuid>.mha` |
   | input | lesion clicks (multiple points, JSON) | `lesion-clicks.json` |
   | output | tumor lesion segmentation | `images/tumor-lesion-segmentation/<uuid>.mha` |

3. Hardware / limits: **GPU enabled**, A10G-class, memory **30 GB**, and the per-job time
   limit at the challenge's maximum (the evaluation allows 20 min per iteration).
4. **Containers → Link GitHub repository.** Install the Grand-Challenge GitHub app on
   `msdsn/autopet-v` and grant it access to that repo.
5. Choose the tag/branch to build. Tagging is safer than tracking `main`:

   ```bash
   git tag -a v0.1-skeleton -m "walking skeleton: unmodified baseline through our container"
   git push origin v0.1-skeleton
   ```

6. Trigger the build and watch it. Expect ~10–20 min: the base image is ~2.5 GB, pip
   resolves nnunetv2 and its dependencies, and the checkpoint download adds ~250 MB.

### If the build fails

| symptom in the build log | cause | fix |
|---|---|---|
| `FATAL: downloaded file is only N bytes -- Google Drive probably returned a quota or virus-scan interstitial` | Drive rate-limited the build | retry the build; if it persists, switch to §2.3 (repo public) or §2.5 (git-LFS) |
| `FATAL: sha256 mismatch for weights.zip` | the Drive file changed, or a partial download | re-download on the GPU box, recompute, update `WEIGHTS_ZIP_SHA256` |
| `FATAL: gdown not installed` | the `pip install gdown` step was edited out | restore it in the weights `RUN` |
| `FATAL: WEIGHTS_BASE_URL is still the placeholder` | `release` mode selected without §2.3 | edit + push |
| `curl: (22) ... 404` on `checkpoint_final.pth` | `release` mode against a private repo, wrong tag, or asset name mismatch | make the repo public / fix the tag / re-upload |
| `FATAL: WEIGHTS_SOURCE=repo but no parts directory was found` | `COPY model/ /opt/algorithm/model_src/` missing, or `model/` was never pushed | check `git ls-files model/` and the `COPY` line |
| `FATAL: SHA256SUMS lists N part(s) but M are present` | a part did not reach the clone, or an unlisted extra part exists | re-push `model/`; regenerate `SHA256SUMS` if the split changed |
| `FATAL: a file in … does not match SHA256SUMS` | a corrupt or truncated blob in the clone | re-push; verify locally with `WEIGHTS_SOURCE=repo` (§2.4) |
| `FATAL: … says <hash> but the pinned CHECKPOINT_SHA256 is <other>` | the parts were replaced without updating the pin | update `ARG CHECKPOINT_SHA256` and `FT_CHECKPOINT_SHA256` (§2.1) |
| `FATAL: WEIGHTS_SOURCE=gdrive_file but the Drive file id for <file> is still the placeholder` | an `ARG FT_*_GDRIVE_ID` was not filled in (only when that fallback route is selected) | §2.1b, then commit + re-tag |
| `FATAL: gdown failed for <file> (id …)` | the file is not shared publicly, or the id is a *folder* id | re-share as "Anyone with the link", copy the id from `/file/d/<id>/view` |
| `FATAL: <file> is only N bytes` | Drive returned a quota / virus-scan interstitial | retry the build; if it persists use §2.3 or §2.5 |
| `FATAL: sha256 mismatch` on an extracted file | the artifact is not the one hashed here | recompute and update the `ARG`s **and** `scripts/fetch_weights.sh` |
| the self-check prints `[FAIL] nnU-Net resolves every trainer class in src/train` | `src/train` missing from the image or `nnUNet_extTrainer` unset | check the `COPY src/` line and the `ENV nnUNet_extTrainer` |
| the self-check prints `[FAIL] dataset.json declares 5 channels` | the 4-channel baseline weights were fetched instead of ours | `WEIGHTS_SOURCE` / the Drive ids point at the wrong files |
| pip resolver conflict | a pinned version has no cp311 wheel | relax that pin in `requirements-submission.txt` |
| `ModuleNotFoundError` at the final `RUN python -c "import submission.process"` | a `COPY` missed a file | check `.gitignore` — anything ignored is not in the build context |

That last `RUN` is deliberate: a broken import fails the **build**, not the first
20-minute evaluation job.

---

## 4. Submit to the preliminary phase

1. Algorithm page → **Try-out** on one case first if the challenge provides sample data.
2. Challenge → **Submit** → preliminary phase → pick the algorithm + the built container.
3. Check how many preliminary submissions the phase allows before spending one — the page
   states the limit; do not assume it is unlimited.

In the preliminary phase each case is run **once** (iteration 0, no scribbles) and only
DSC/F1 are reported. Interactive behaviour is *not* exercised there — it is validated by
our own offline loop. So a preliminary submission proves exactly four things: the image
builds, the interface is right, iteration 0 is sane, and the runtime fits.

---

## 5. What to check in the job log

The container prints one line per stage. In order:

```
[gpu] torch 2.6.0+cu124 cuda_available=True device_count=1
[gpu] NVIDIA A10G, 22.x GB, sm_86
```
`cuda_available=False` ⇒ GPU not attached to the algorithm ⇒ it will blow the time budget.

```
[input] uuid (from CT, used for the output filename) = <uuid>
[geom] size=[...] spacing=[...]
[clicks] format=grand-challenge tumor=N background=M unparsed=0
```
`unparsed` must be 0. In the preliminary phase expect `tumor=0 background=0`.

```
[state] roots in read-preference order: ['/cache/state', '/output/state']
[state] candidate /cache/state/case_<fp>: n_calls=2 files=6 writable=True
[state] candidate /output/state/case_<fp>: n_calls=2 files=6 writable=True
[state] reading case state from /cache/state/case_<fp>: 6 file(s) from previous iterations: [...]
[predictor] config={"predictor": "interactive_postproc", ...}
[predictor] postproc config from /opt/algorithm/submission/postproc_config.json: {...}
Using trainer 'nnUNetTrainer_InteractiveV2_negfp' from: /opt/algorithm/src/train
[predict] done in ~40 s
[predict] sub-timings: {"ct_pet_preproc_s":..,"guidance_preproc_s":..,"prev_pred_s":..,"network_s":..}
[predict] guidance: {"radius_voxels":10.0, "n_tumor":4, "n_tumor_mapped":4, "n_tumor_dropped":0,
                     "prev_pred_source":"argument", "prev_pred_voxels":...}
[postproc] {"iteration":1, "tracer":"...", "negative_gate_fired":false,
            "base_volume_ml":..., "final_volume_ml":..., "constraints":{"ok":true,...}}
[output] .../<uuid>.mha (... positive voxels ...)
[output] geometry verified identical to the input CT
[state] previous-mask file written: /cache/state/case_.../prev_final_mask.npz (6026 B, bit-packed)
[state] call 3 for this case; mirrored /cache/state/case_<fp> -> ['/output/state/case_<fp>']
[timing] {"convert_s":..,"load_s":..,"model_load_s":..,"inference_s":..,"total_s":..}
[mem] peak RSS self=..GB children=..GB
[done] total ... s  (budget 1200 s per iteration, 6 iterations/case)  status=ok
```

Red flags:

* `status=EMPTY-MASK-FALLBACK` — the model path threw; the traceback is directly above.
  The job still succeeds and scores 0 for that iteration.
* `[output] GEOMETRY MISMATCH!` — the written mask does not match the input CT.
* `total_s` above ~600 s — no headroom for the slowest cases; drop TTA or raise
  `AUTOPETV_TILE_STEP`.
* `[mem] peak RSS` approaching 30 GB.
* **`"prev_pred_source": "none"` at an iteration that has scribbles** — the state directory
  did not persist, so the model is running with an empty fifth channel. The container logs
  an explicit warning for exactly this case. It still produces a valid mask (§6.1), but the
  interactive gain is gone; that is the single most important line to read in the final
  offline run.
* `n_tumor_dropped` / `n_background_dropped` above 0 — scribble points fell outside the
  crop box or the resampled grid.
* `constraints: {"ok": false}` or `empty_without_gate: true` — the post-processing layer
  could not satisfy a scribble, or emptied a mask without the negative gate asking for it.
* `AUTOPETV_PREDICTOR` in the `[predictor] config=` line is not `interactive_postproc`.
* `[state] candidate <root>/case_...: n_calls=-` on **both** roots at an iteration that has scribbles — neither `/cache` nor `/output` persisted, see §6.
* `expected exactly ONE CT file ... found 2` — the input dir held a stray. The container
  cannot know which uuid the evaluator wants, so it falls back to the PET geometry *and
  the PET uuid* and writes an empty mask under that name. The job exits 0, but the
  evaluator will not find the file it expects; treat this as a platform bug report.

---

## 6. State between iterations: `/cache` **and** `/output`

Two pieces of evidence that do not agree, so we rely on neither alone:

* the organizers' own `nnunet-baseline/test.sh` **bind-mounts `/cache`** — the commit that
  added it calls it a "cache directory for inter-iteration storage" (PR #2) — and
  `interactive_loop.py` wipes `<interface>/cache` once **per case**, so it is meant to
  survive the 6 calls of a case and be empty at the start of the next one;
* a forum answer says **`/output` persists** across the 6 calls ("You can write to the
  /output directory and read from there").

Betting the interactive gain on either one would be a single point of failure we cannot
test before the one scored run. So the container uses **both**:

```
state roots, in read-preference order:   /cache/state     then   /output/state
per case:                                <root>/case_<fingerprint>/
```

* **Write** the per-case state (previous final mask, cached probabilities, constraint set,
  markers) into the chosen directory, then **mirror** it into every other writable root.
* **Read** from whichever root actually carries state for this case — ranked by the
  `n_calls` recorded in our own `autopetv_case.json`, then by how many state files are
  there, then by the preference order above. A root that silently stopped persisting
  scores zero and is simply ignored rather than serving a stale iteration.
* A root whose per-case directory holds a *different* case fingerprint is ignored too:
  a collision needs identical geometry **and** an identical strided PET digest, but
  serving the wrong previous mask would be much worse than starting from scratch.
* If state survived only on a root that has since become read-only, it is copied into a
  writable one first, so the predictor can still update it.
* With **neither** root persisting, every call behaves like iteration 0 — which is exactly
  right for the preliminary phase, where there *is* only iteration 0.

`case_<fingerprint>` comes from geometry plus a strided PET digest; the per-call uuid is
random and cannot identify a case. The relevant log lines:

```
[state] roots in read-preference order: ['/cache/state', '/output/state']
[state] candidate /cache/state/case_39be40e7c7b08c51: n_calls=2 files=6 writable=True
[state] candidate /output/state/case_39be40e7c7b08c51: n_calls=- files=0 writable=True
[state] reading case state from /cache/state/case_39be40e7c7b08c51: 6 file(s) ...
[state] call 3 for this case; mirrored /cache/state/... -> ['/output/state/...'] in 0.02 s
```

Read those in the first real job: `n_calls=-` on a root means it did **not** persist, and
one run therefore answers the question the forum answer and `test.sh` disagree about.
The older root-level `autopetv_calls.json` marker and its `iteration_guess` are still
written to both roots as a cross-check.

**Overrides.** `AUTOPETV_STATE_DIRS` (an `os.pathsep`-separated list) replaces the roots
outright; `AUTOPETV_STATE_DIR` / `STATE_DIR` pins a single root (this is what v0.1 used —
the Dockerfile now deliberately leaves it **unset** so both roots are active);
`AUTOPETV_STATE_ENABLED=0` disables state entirely, and then not even a directory is
created.

**Risk: extra files under `/output`.** The evaluator reads only
`/output/images/tumor-lesion-segmentation/<uuid>.mha`, and GC's own output validation
looks at the declared output sockets, so `/output/state/` should be ignored. This is not
something we can prove without a real run, so:

* `AUTOPETV_STATE_DIRS=/cache/state` drops `/output` from the picture in one line, and
  after the first real job we will know whether that is safe to do;
* `AUTOPETV_STATE_ENABLED=0` turns the whole thing off;
* **nothing is ever written directly into `/output/`, only into `/output/state/`**, and
  what goes there is now kilobytes rather than tens of megabytes (see below).

Read the preliminary-phase job log for a line like "unexpected file in output" before
relying on this in the final submission.

### 6.1 What v0.2 keeps there, and why it is still optional

v0.1 wrote the marker but *used* nothing (`prev_pred=None`, `case_cache_dir=None`), so
every iteration was independent. v0.2 does use it, because channel 4 of the model **is**
the previous iteration's final mask, and a container call is a fresh process with nothing
in memory. Per case, in `<root>/case_<fingerprint>/`:

| file | written by | size (45 M-voxel case) | used for |
|---|---|---|---|
| `autopetv_case.json` | `submission/process.py` | < 1 kB | the call count that ranks the roots |
| `postproc_constraints.json` | `PostProcPredictor` | ~1 kB | accumulated scribbles, tracer, history |
| `postproc_prev_mask.npz` | `PostProcPredictor` | ~6 kB (bit-packed) | **channel 4** via `pass_cached_prev_pred` |
| `postproc_prev_prob.npy` | `PostProcPredictor` | ~45 MB (uint8-quantised) | the monotone min-max blend |
| `postproc_bg_region.npz` | `PostProcPredictor` | ~6 kB | regions a background scribble deleted stay deleted |
| `prev_final_mask.npz` | `submission/process.py` | ~6 kB (bit-packed) | redundant fallback for channel 4 |

The mask reaches the model by **two independent routes** carrying the same array:

1. the post-processing layer's own cached mask (**primary**) — this is what the harness's
   `B3` config declares, and the harness passes the identical array in memory;
2. `prev_final_mask.npz`, which `submission/process.py` writes and reads back as an
   explicit `prev_pred` **only when route 1's file is missing**, so the primary route is
   never second-guessed and the container cannot diverge from the harness.

Both are written only when the model path succeeded, so an empty fallback mask can never
be fed back as "what we predicted last time". `prev_final_mask.npz` is bit-packed and
deflated — 45 M voxels become **~6 kB** rather than the 45 MB a raw `.npy` would cost at
every one of the 6 calls; `AUTOPETV_SAVE_PREV_MASK=0` drops it entirely if even that is
unwanted. The one genuinely large file left is `postproc_prev_prob.npy` (~45 MB), which
the monotone blend needs; setting `"cache_probabilities": false` in
`submission/postproc_config.json` removes it at the cost of that blend.

**Absence is a supported state, not a degradation to work around.** With no state
directory: channel 4 is all zeros, no cached probability exists so the monotone blend is a
no-op, and the constraint set is rebuilt from `lesion-clicks.json`, which always carries
*all* scribbles so far. That is exactly the preliminary phase, where there is only
iteration 0 anyway. Measured in §7: all four persistence regimes — `/cache` only,
`/output` only, both, neither — plus `AUTOPETV_STATE_ENABLED=0`.

The log line to read is `[predict] guidance: {... "prev_pred_source": ...}`:
`"argument"` = it came through the state directory (the post-processing cache, or
`prev_final_mask.npz` when that cache is missing), `"none"` = there was none. With scribbles present and
`prev_pred_source = none`, the container additionally logs a warning that the state
directory did not persist.

---

## 7. Testing without Docker

Where no Docker daemon is available, the container is exercised by reproducing the mount
layout on disk and pointing the `AUTOPETV_*` env vars at it.

### 7.1 The v0.2 acceptance suite (one command)

`submission/tests/run_v02_suite.py` runs the whole thing: the offline harness on N cases,
then the container on the *same* cases with the *same* scribbles, iteration by iteration,
with the state directory persisting between calls; then a determinism re-run against a
restored state snapshot; then the same iterations with the state directory wiped before
every call and once with `AUTOPETV_STATE_ENABLED=0`.

```bash
source /content/env.sh; cd /content/autopet
export PYTHONPATH=/content/autopet:/content/autopet/src
python3 -u -m submission.tests.run_v02_suite \
  --images_dir /content/work/v02_data/images \
  --labels_dir /content/work/v02_data/labels \
  --cases "psma_5203bac8a9bfd9e2_2020-06-06" \
          "fdg_74bbceaeeb_06-03-2005-NA-PET-CT Ganzkoerper  primaer mit KM-94697" \
          "fdg_402c061122_08-25-2003-NA-PET-CT Ganzkoerper nativ-22953" \
  --model_folder /content/work/ft_model \
  --repo /content/autoPETV --out /content/work/v02_suite --max_iters 3
```

Both sides read the shipped `submission/postproc_config.json`, so the only thing under
test is the container itself. It exits non-zero unless **every** container mask is bitwise
identical to the harness's **and** every persistence regime behaved as required, and it
writes `<out>/report.json` with the per-iteration wall time, peak RSS, Dice,
differing-voxel count, `prev_pred_source`, which state root was read, geometry check and
stray-file list.

`submission/tests/check_image.py` is the same self-check the Dockerfile runs at build
time (config parses, model folder is a 5-channel interactive one, nnU-Net can resolve
every `nnUNetTrainer_*` class in `src/train`); it needs no GPU and no data:

```bash
AUTOPETV_MODEL_FOLDER=/content/work/ft_model python3 -m submission.tests.check_image
```

### 7.2 One iteration by hand

```bash
# 1. build a fake /input from a real case (random uuids, CT uuid != PET uuid)
python -m submission.tests.build_gc_sim \
  --images_dir /content/work/testcase/images \
  --case psma_0198cdca94fbb95f_2020-05-09 \
  --out /content/work/gc_sim \
  --scribbles /content/work/eval_real/psma_0198cdca94fbb95f_2020-05-09_0000/iter_1_scribbles.json \
  --seed 12345
# omit --scribbles for iteration 0

# 2. run the entry point against it
export AUTOPETV_INPUT_DIR=/content/work/gc_sim/input
export AUTOPETV_OUTPUT_DIR=/content/work/gc_sim/output/images/tumor-lesion-segmentation
export AUTOPETV_STATE_DIRS=/content/work/gc_sim/cache/state:/content/work/gc_sim/output/state
export AUTOPETV_OUTPUT_ROOT=/content/work/gc_sim/output
export AUTOPETV_CACHE_DIR=/content/work/gc_sim/cache
export AUTOPETV_TMP_DIR=/content/work/gc_sim/tmp
export AUTOPETV_MODEL_FOLDER=/content/work/gc_weights_test
/usr/bin/time -v python -m submission.process

# 3. check filename, dtype, geometry, and equality with the offline harness
python -m submission.tests.check_gc_sim \
  --sim /content/work/gc_sim \
  --reference /content/work/eval_real/psma_0198cdca94fbb95f_2020-05-09_0000/iter_1.nii.gz
```

### 7.3 v0.2 measured, 2026-08-29, NVIDIA L4 (23 GB, 12 CPU, 52 GB RAM)

Three real cases — a PSMA lesion case (16.1 M voxels), an FDG lesion case (45.1 M voxels,
name with spaces) and an FDG **lesion-free** case (45.4 M voxels) — iterations 0 → 1 → 2,
run under every persistence regime. The harness and the container both read the shipped
`submission/postproc_config.json`, so only the container is under test:

| | |
|---|---|
| container mask vs harness mask | **bitwise identical in all 9 iterations** (0 differing voxels, Dice 1.0) |
| channel 4 | `prev_pred_source = "argument"` from iteration 1 on, i.e. it came through the state directory |
| wall time per iteration | 34.8 – 57.6 s → worst **4.8 % of the 1200 s budget** |
| peak RSS | 3.63 – 4.95 GB → worst **16.5 % of the 30 GB limit** |
| mirroring the state into the second root | 0.01 – 0.06 s per call |
| determinism | **both** roots snapshotted, iteration 1 re-run against the restored snapshot: **md5-identical** |
| only `/cache` persists | **all 3 iterations identical to the harness** |
| only `/output` persists | **all 3 iterations identical to the harness** (iteration 0 reads `/cache`, then it falls to `/output`) |
| neither persists | rc 0, valid mask, `prev_pred_source = "none"`; iteration 0 still identical, 1–2 differ as an empty channel 4 must |
| `AUTOPETV_STATE_ENABLED=0` | rc 0, byte-identical to the "neither" run, and no state root created |
| bogus model folder | 6.0 s, empty mask with the CT's geometry, exit 0 |
| two files in `/input/images/ct` | 1.8 s, loud assertion, empty mask, exit 0 |

The equality result is the load-bearing one: in the harness the previous final mask is
handed to the model in memory, in the container it has to survive a process boundary
through a state directory that may be `/cache`, may be `/output`, and may be neither. Zero
differing voxels at iterations 1 and 2 — in the default regime *and* with either root
wiped before every call — proves the two paths compute the same thing, which is what lets
an offline AUC number stand for the container.

Full per-iteration table, sub-timings and state-footprint numbers:
[`submission_checklist.md`](submission_checklist.md).

### 7.4 v0.1 measured, 2026-08-26, NVIDIA L4 (23 GB, 12 CPU, 52 GB RAM)

Case `psma_0198cdca94fbb95f_2020-05-09`, 200×200×462 = 18.5 M voxels, native PSMA
resolution, **the 4-channel baseline**:

| scenario | wall | inference | peak RSS | result |
|---|---|---|---|---|
| iteration 1, 8 tumor points | 188 s | 175 s | 4.53 GB | 5506 voxels |
| the same run again | 172 s | 158 s | 4.54 GB | byte-identical output |
| iteration 0, no clicks file | 174 s | 159 s | 4.57 GB | **bit-identical to the offline harness** |
| bogus model folder | 10 s | — | 1.14 GB | empty mask, exit 0 |
| two files in `/input/images/ct` | 3 s | — | 0.78 GB | empty mask, exit 0 |
| iteration 0, weights from the Drive zip (§2.2) | 144 s | 130 s | 4.55 GB | **bit-identical to the harness** |

Worst case: **16 % of the 1200 s budget, 15 % of the 30 GB memory limit.** The full table
and its caveats are in `submission_checklist.md`.

The equivalence result is the important one: at iteration 0 the container reproduces the
harness's prediction with **zero differing voxels out of 18.5 M**, so the `.mha` → `.nii.gz`
conversion, the click heatmaps and the nnU-Net call are provably the same computation. Two
runs of the container are byte-identical, because `submission/process.py` sets
`cudnn.benchmark=False`, `cudnn.deterministic=True` and seeds `random` / `numpy` / `torch`
before the model is built. Without those flags cuDNN's autotuner picks different
convolution algorithms depending on machine load, and the float summation order flips
voxels on the decision boundary — that is exactly the single-voxel wobble seen against the
older harness run.

What this does **not** cover, and only a real GC job can: the image actually building, the
CUDA driver on the A10G, `--network=none`, `--shm-size=2g`, `--cap-drop=ALL`, the non-root
user, and whether `/output` really persists between calls.

### 7.5 `WEIGHTS_SOURCE=repo` measured, 2026-08-29, NVIDIA A100-SXM4-80GB

Run from a **fresh clone-like copy** of the repository (`rsync` of the tree plus `model/`
into an empty directory, then `bash scripts/fetch_weights.sh` from inside it) — i.e. the
same thing the GC build server does with `git clone`.

| step | result |
|---|---|
| repo copy including `model/` | 238 MB |
| assemble 3 parts → `checkpoint_final.pth` | **4 s**, 246 476 509 bytes |
| sha256 of the assembled file | `ed015e29…` — matches `model/checkpoint_final.pth.sha256` **and** the pinned `ARG` |
| `sha256 ok` lines printed | **4** (assembled vs its recorded hash, then the three pinned per-file checks) |
| network used for the weights | **none** |
| `WEIGHTS_SOURCE` unset (the default) | resolves to `repo`, exit 0, same hash |
| build-time self-check `check_image` | **all PASS**, 5 channels, 11 trainer classes resolve |

Failure paths, each exiting **1** with a specific message:

| what was broken | message |
|---|---|
| one part deleted | `SHA256SUMS lists 3 part(s) but 2 are present` |
| one byte flipped in a part | `a file in <dir> does not match SHA256SUMS` |
| `checkpoint_final.pth.sha256` disagrees with the pin | `sha256 mismatch for checkpoint_final.pth (assembled, …)` |
| no parts directory at all | `no parts directory was found` |

So a truncated clone, a corrupt blob, a misordered concatenation and a stale pin all fail
`docker build` in seconds instead of shipping a broken model.

**End-to-end.** The model folder assembled this way was then run through the full
container simulation against the offline harness (case `psma_5203bac8a9bfd9e2_2020-06-06`,
iterations 0 → 1):

| | |
|---|---|
| container vs harness | **bitwise identical at both iterations** — 2322 vox at iteration 0, 2625 at iteration 1 |
| iteration 0 voxel count | **2322 — the same number the separately-staged checkpoint produced on the L4**, so the parts provably reassemble the same model |
| channel 4 at iteration 1 | `prev_pred_source=argument`, i.e. through the state directory |
| determinism | md5-identical re-run against the restored snapshot of both roots |
| the four persistence regimes | all as required (`cache_only` / `output_only` identical to the harness, `neither` degrades, `AUTOPETV_STATE_ENABLED=0` matches `neither` and creates no root) |
| wall / RSS on the A100 | 37–44 s, 3.68–3.77 GB |

The A100 is not the evaluation hardware, so the budget numbers to quote are the L4 ones in
§7.3 — those are the pessimistic ones.

---

## 8. v0.1 manual steps, in order (done)

Nothing here needed the repo to be public, and nothing needed weights uploaded
anywhere — the skeleton pulls the organizers' public zip.

1. `git remote add origin https://github.com/msdsn/autopet-v.git && git push -u origin main`
2. Confirm `LICENSE` (Apache-2.0) is committed at the repo root — GC requires an
   open-source licence on a linked repository, private or not (§3).
3. Create the algorithm on GC from the **challenge's own "create your algorithm" link**;
   check the four sockets, enable the GPU, set memory to 30 GB (§3).
4. Install the Grand-Challenge GitHub App on `msdsn/autopet-v` and link the repo (§3.4).
5. Tag and push, which triggers the build:
   `git tag -a v0.1-skeleton -m "walking skeleton" && git push origin v0.1-skeleton`
6. Watch the build. It must print four `sha256 ok` lines; anything else is §3 "If the
   build fails".
7. Submit to the preliminary phase (§4) and read the job log against §5.

---

## 9. v0.2 manual steps, in order

The algorithm on GC, the GitHub App and the sockets are already set up by §8 and do not
change. The repo stays **private**. Since the checkpoint now ships inside the repository
(§2.1), **there is nothing to share, upload or configure** — no Drive files, no file ids,
no release assets:

1. **Push `main`**, including `model/` (three parts + `SHA256SUMS` +
   `checkpoint_final.pth.sha256` + the two json files, 238 MB total). Ordinary git blobs,
   no LFS, each part under GitHub's 100 MB limit. Verify they really arrived —
   `git ls-files model/` must list all of them, and `cd model && sha256sum -c SHA256SUMS`
   must pass in a fresh clone.

2. **Tag and push** — this is what triggers the GC build:

   ```bash
   git tag -a v0.2 -m "fine-tuned 5-channel interactive nnU-Net + post-processing"
   git push origin v0.2
   ```

3. **Watch the build** (§3 "If the build fails"). A good build prints, in order:
   `[weights] source=repo -- no network needed, skipping gdown`,
   `[fetch_weights] repo parts dir: /opt/algorithm/model_src`,
   `3 part(s) listed in SHA256SUMS, 3 on disk`, three `part` lines,
   `assembled 246476509 bytes`, then **four** `sha256 ok` lines (the assembled file
   against its own recorded hash, then the three pinned per-file checks), and finally the
   self-check's `RESULT: IMAGE SELF-CHECK PASSED` with
   `nnU-Net resolves every trainer class in src/train`.

4. **Submit to the preliminary phase** (§4) as *preliminary submission #2*, and read the
   job log against §5. Check the phase page for how many preliminary submissions are left
   before spending one. The preliminary phase runs **iteration 0 only**, so it exercises
   the model and the interface but *not* the interactive path — that is what §7's suite
   is for. It is also the run that answers whether `/cache` or `/output` persists (§6).

If the checkpoint is ever replaced, regenerate the parts and update the two pinned hashes
with the snippet in §2.1; the build fails loudly if the pin and the parts disagree.
