# Submission checklist

Tick before every submission. Step-by-step instructions: [`submission.md`](submission.md).

Current version: **v0.2** — our fine-tuned 5-channel interactive nnU-Net under the
post-processing layer (`AUTOPETV_PREDICTOR=interactive_postproc`, == harness variant `B3`).
v0.1 (`v0.1-skeleton`) was the organizers' unmodified 4-channel baseline through the same
container; its measurements are kept at the bottom as the control.

## Repo (must exist and build)

- [ ] GitHub repo `msdsn/autopet-v` pushed; **`LICENSE` (Apache-2.0) present** — GC requires
      an open-source licence on a linked repo, private or public
- [ ] `Dockerfile` at the repo root (GC's build context), CUDA base matching the A10G (sm_86)
- [ ] `requirements-submission.txt`, every direct dep `==`-pinned; `cupy` / `cucim` **not** installed
- [ ] Weights: `WEIGHTS_SOURCE=repo` (the default) — the checkpoint ships **in the repository**
      under `model/`; **no network is used for the weights at build time**, nothing has to be
      shared or made public
- [ ] `git ls-files model/` lists the three `checkpoint_final.pth.part0{0,1,2}`
      (94 371 840 / 94 371 840 / 57 732 829 B, each under GitHub's 100 MB blob limit) plus
      `SHA256SUMS`, `checkpoint_final.pth.sha256`, `plans.json`, `dataset.json` — **tracked and
      pushed**, not just present locally
- [ ] `cd model && sha256sum -c SHA256SUMS` passes in a fresh clone (catches a truncated
      checkout before the build does)
- [ ] `ARG CHECKPOINT_SHA256 / PLANS_SHA256 / DATASET_JSON_SHA256` match the shipped model
      (`ed015e29…`, `36b7c8b2…`, `3274b61d…`); the build cross-checks `ARG CHECKPOINT_SHA256`
      against `model/checkpoint_final.pth.sha256` and fails if the parts were replaced without
      updating the pin
- [ ] The Dockerfile `COPY model/ /opt/algorithm/model_src/` — the parts are assembled into
      `/opt/algorithm/model`, never into their own source directory
- [ ] Model folder lands at `/opt/algorithm/model` as `plans.json`, `dataset.json`,
      `fold_0/checkpoint_final.pth`
- [ ] `src/train/` is copied into the image and `nnUNet_extTrainer=/opt/algorithm/src/train`
      is set — the checkpoint records a `trainer_name` (currently
      `nnUNetTrainer_InteractiveV2_negfp`) that nnU-Net must import to rebuild the network
- [ ] `submission/postproc_config.json` is in the image and holds the intended thresholds —
      the offline harness is driven with **that same file**, so the measured and the shipped
      configuration cannot drift apart
- [ ] Build-time self-check `python -m submission.tests.check_image` passes as the last `RUN`
- [ ] `submission/process.py`: `/input/images/ct/*.mha` + `/input/images/pet/*.mha` + `/input/lesion-clicks.json`
      → `/output/images/tumor-lesion-segmentation/<CT-uuid>.mha`
- [ ] exactly-one-file assertion on both input dirs; output dir `mkdir -p`'d at runtime *and* at build time
- [ ] per-case state written to **both** `/cache/state/case_<fp>/` and `/output/state/case_<fp>/`
      (the organizers' `test.sh` bind-mounts `/cache`; a forum answer promises `/output` — we
      rely on neither alone), read back from whichever root actually carried this case's state;
      **degrades to "iteration 0" when neither persists**, never fails
- [ ] `AUTOPETV_STATE_DIR` is **not** pinned in the Dockerfile (pinning it would disable the
      `/cache` root); `AUTOPETV_STATE_DIRS` / `AUTOPETV_STATE_ENABLED=0` still override
- [ ] `prev_final_mask.npz` is bit-packed (~6 kB, not 45 MB) — it is written into `/output`
      at every one of the 6 calls of every case
- [ ] Never writes outside `/output`, `/cache` and the scratch dir; no network at runtime (`--network=none`)
- [ ] Runs as the non-root `algorithm` user
- [ ] `nnUNet_raw` / `nnUNet_preprocessed` / `nnUNet_results` env vars set in the image
- [ ] `AUTOPETV_PREDICTOR` set to the intended predictor (`interactive_postproc` for v0.2)

## Behaviour — verified without Docker on an L4 (see `submission.md` §7)

- [ ] Iteration 0 (`lesion-clicks.json` **absent**) does not crash
- [ ] Iteration ≥ 1 parses the GC "Multiple points" JSON, `unparsed=0`
- [ ] Output filename is the **CT** uuid while the PET uuid differs
- [ ] Output is `uint8`, values ⊆ {0, 1}
- [ ] Output geometry (size, spacing, origin, direction) identical to the input CT
- [ ] **Container mask == harness mask, bitwise, at every iteration** (the load-bearing check:
      in the harness the previous final mask is passed in memory, in the container it travels
      through `/output/state`)
- [ ] Channel 4 really arrives: the log shows `prev_pred_source: "argument"` from iteration 1 on
- [ ] A lesion-free case stays empty at every iteration (the negative gate keys on
      "no scribble ever seen", not on the iteration index)
- [ ] `/output` wiped before every call (only `/cache` persists) → masks still identical to the harness
- [ ] `/cache` wiped before every call (only `/output` persists) → masks still identical to the harness
- [ ] both wiped before every call → still valid output, no crash, `prev_pred_source: "none"`
- [ ] `AUTOPETV_STATE_ENABLED=0` → identical output to the both-wiped run, and no state root created
- [ ] Empty mask still written as a valid `.mha` when the model path throws; exit code 0
- [ ] Two input files in one image dir → loud assertion, empty mask, exit 0 (no silent wrong-file pick)
- [ ] Scribble coordinates are `[i, j, k]` nibabel indices end-to-end
- [ ] Wall time per iteration well under 1200 s on the slowest case
- [ ] Peak RSS well under 30 GB; `npp = nps = 1` so `/dev/shm` (2 GB) is never a factor
- [ ] Deterministic: same input + same state → md5-identical output on two runs
- [ ] Case names with spaces (the FDG naming) survive the whole path

### v0.2 measured (2026-08-29, NVIDIA L4 23 GB / 12 CPU / 52 GB RAM, Docker-free simulation)

`python -m submission.tests.run_v02_suite`, 3 real cases including a lesion-free one,
iterations 0 → 1 → 2. Model = `nnUNetTrainer_InteractiveV2_negfp` fold 0
(`checkpoint_final.pth`, sha256 `ed015e29…`, 246 476 509 B, verified against the
Dockerfile ARG). **The harness and the container both read the shipped
`submission/postproc_config.json`** (`interactive_eval.py --postproc_config <that file>`),
so the configuration is not a variable in the comparison — only the container is.
`AUTOPETV_PREDICTOR=interactive_postproc`, `disable_tta`, `tile_step=0.5`, `npp=nps=1`,
cuDNN deterministic. Wall time is the whole process (import → model load → inference →
write); RSS from the container's own `[mem]` line. Another session held ~0.5 GB of the GPU
throughout, so timings are if anything pessimistic.

| case | tracer | GT | shape | voxels |
|---|---|---|---|---|
| `psma_5203bac8a9bfd9e2_2020-06-06` | PSMA | lesion | 200×200×403 @ 4.07×4.07×2.0 mm | 16.1 M |
| `fdg_74bbceaeeb_…KM-94697` | FDG | lesion | 400×400×282 @ 2.04×2.04×3.0 mm | 45.1 M |
| `fdg_402c061122_…nativ-22953` | FDG | **lesion-free** | 400×400×284 @ 2.04×2.04×3.0 mm | 45.4 M |

**Both state roots persisting (the default regime):**

| case | it | clicks | wall | inference | mirror | peak RSS self/children | read from | channel 4 | vs harness |
|---|---|---|---|---|---|---|---|---|---|
| psma | 0 | 0 | 42.3 s | 31.3 s | 0.01 s | 3.63 / 0.99 GB | `/cache/state` | — | **identical**, 2322 vox |
| psma | 1 | 4 | 57.4 s | 46.4 s | 0.01 s | 3.72 / 0.99 GB | `/cache/state` | state dir | **identical**, 2473 vox |
| psma | 2 | 8 | 56.6 s | 45.6 s | 0.01 s | 3.72 / 0.99 GB | `/cache/state` | state dir | **identical**, 2568 vox |
| fdg-pos | 0 | 0 | 37.9 s | 19.3 s | 0.02 s | 4.78 / 1.63 GB | `/cache/state` | — | **identical**, 0 vox (gate fired) |
| fdg-pos | 1 | 4 | 39.8 s | 20.9 s | 0.04 s | 4.91 / 1.63 GB | `/cache/state` | state dir | **identical**, 216 vox |
| fdg-pos | 2 | 8 | 37.9 s | 19.2 s | 0.04 s | 4.92 / 1.63 GB | `/cache/state` | state dir | **identical**, 284 vox |
| fdg-neg | 0 | 0 | 36.5 s | 16.8 s | 0.02 s | 4.80 / 1.64 GB | `/cache/state` | — | **identical**, 0 vox (gate fired) |
| fdg-neg | 1 | 0 | 35.5 s | 15.9 s | 0.05 s | 4.93 / 1.64 GB | `/cache/state` | state dir | **identical**, 0 vox |
| fdg-neg | 2 | 0 | 34.8 s | 15.3 s | 0.06 s | 4.95 / 1.64 GB | `/cache/state` | state dir | **identical**, 0 vox |

**All 9 container masks are bitwise identical to the harness (0 differing voxels, Dice 1.0.)**
Geometry verified identical to the input CT in all 9; the output filename is the CT uuid in
all 9 while the PET uuid differs; nothing written outside `output/`, `cache/` and the
scratch dir. The lesion-free case receives no scribble at any iteration and stays empty at
all three, as the protocol requires. Case names with (double) spaces survive end to end.

**Budget: worst wall time 57.6 s = 4.8 % of the 1200 s per-iteration budget; worst peak RSS
4.95 GB = 16.5 % of the 30 GB limit.** Mirroring the state into the second root costs
0.01–0.06 s. The A10G is comparable to the L4 for this workload. Re-measure before enabling
TTA (`AUTOPETV_MIRROR_AXES=0,1,2` + `AUTOPETV_ENABLE_TTA=1` is 8 forward passes).

**The four persistence regimes** (PSMA case, the named root(s) wiped before *every* call —
what a non-persistent mount looks like from inside the container):

| regime | what survives | result |
|---|---|---|
| both (above) | `/cache` + `/output` | reads `/cache/state`, **all 3 identical to the harness** |
| `cache_only` | `/cache` only | reads `/cache/state`, **all 3 identical** (2322 / 2473 / 2568 vox) |
| `output_only` | `/output` only | iteration 0 reads `/cache/state` (nothing anywhere yet), then falls to `/output/state` — **all 3 identical** |
| `neither` | nothing | rc 0, valid geometry, `prev_pred_source=none`; iteration 0 identical, 1–2 differ (2529 / 2620 vox) exactly as an empty channel 4 must |
| `AUTOPETV_STATE_ENABLED=0` | — | rc 0, **byte-identical to the `neither` run**, and **no state root is created at all** |

That is the point of the design: `/cache` is what the organizers' `test.sh` bind-mounts and
what `interactive_loop.py` wipes per case; `/output` is what the forum answer promised. The
container is correct under either, and degrades cleanly under neither.

| extra scenario | result |
|---|---|
| determinism: **both** roots snapshotted, iteration 1 re-run against the restored snapshot | **md5-identical** (`485f5abf61…`) |
| bogus `AUTOPETV_MODEL_FOLDER` | empty mask with the input CT's geometry, `status=EMPTY-MASK-FALLBACK`, **exit 0** |
| two files in `/input/images/ct` | loud `AssertionError: expected exactly ONE CT file … found 2`, PET-geometry fallback, empty mask, **exit 0** |

**Per-case state footprint**, per root, 45 M-voxel FDG case after 3 calls:
`postproc_prev_prob.npy` 45.1 MB, `postproc_prev_mask.npz` 6.2 kB,
`prev_final_mask.npz` **6.2 kB** (bit-packed — a raw `.npy` would have been 45 MB),
`postproc_constraints.json` 0.9 kB, `autopetv_case.json` 0.3 kB, plus ~6 kB of call
markers at the root. `AUTOPETV_SAVE_PREV_MASK=0` drops the redundant packed mask;
`"cache_probabilities": false` in the shipped config drops the one remaining large file
at the cost of the monotone blend.

### Weights from the repository (`WEIGHTS_SOURCE=repo`) — 2026-08-29, A100-SXM4-80GB

Run from a **fresh clone-like copy**: the tree plus `model/` rsynced into an empty
directory, then `bash scripts/fetch_weights.sh` from inside it — what the GC build server
does with `git clone`.

| | |
|---|---|
| repo copy including `model/` | 238 MB |
| assemble 3 parts → `checkpoint_final.pth` | **4 s**, 246 476 509 B |
| sha256 of the assembled file | `ed015e29…`, matching `model/checkpoint_final.pth.sha256` **and** the pinned `ARG` |
| `sha256 ok` lines | **4** (assembled vs its own recorded hash, then the 3 pinned per-file checks) |
| network used for the weights | **none** |
| `WEIGHTS_SOURCE` unset | resolves to `repo`, exit 0, same hash |
| `check_image` against the assembled folder | **all PASS** (5 channels, 11 trainer classes resolve) |
| container simulation vs the harness, assembled folder | **bitwise identical at both iterations** — 2322 vox at iteration 0 (the same count the separately-staged checkpoint produced on the L4, so the parts really do reassemble the same model) and 2625 vox at iteration 1 with channel 4 fed from the state dir |
| determinism on the assembled folder | md5-identical re-run |
| all four persistence regimes on the assembled folder | as required (`cache_only` and `output_only` identical to the harness; `neither` degrades) |

Failure paths, each exit **1** with a specific message: a part deleted
(`SHA256SUMS lists 3 part(s) but 2 are present`), one byte flipped in a part
(`does not match SHA256SUMS`), `checkpoint_final.pth.sha256` disagreeing with the pin
(`sha256 mismatch for checkpoint_final.pth (assembled, …)`), and no parts directory
(`no parts directory was found`). A truncated clone, a corrupt blob, a misordered
concatenation and a stale pin therefore all fail `docker build` in seconds.

The other modes still work: `local` exits **0** with three `sha256 ok` lines, and
placeholder Drive ids / unknown source / placeholder release URL / wrong checkpoint hash
each exit **1**.

`python -m submission.tests.check_image` (the last `RUN` of the Dockerfile) passes against
the V2_negfp model folder: 5 channels, channels 2-4 `NoNormalization`, `use_mask_for_norm`
all False, the shipped config parses with `pass_cached_prev_pred` on, **all 11
`nnUNetTrainer_*` classes in `src/train` resolve through `nnUNet_extTrainer`** (the check
enumerates them from the source rather than hard-coding a name, so swapping the shipped
checkpoint cannot silently break it), and every module in `src/train` imports cleanly
inside the image.

## Process

- [ ] `bash scripts/fetch_weights.sh` exercised in `repo` and `local` modes; every failure path exits 1
- [ ] Preliminary submission #1 (walking skeleton — unmodified baseline through our container)
- [ ] Preliminary submission #2 (v0.2: fine-tuned interactive model + post-processing)
- [ ] Confirm how many preliminary submissions the phase allows — check the phase page, do not assume
- [ ] Final submission submitted **a day early**, not on the deadline
- [ ] Write-up from the challenge's LNCS template
- [ ] Repo public + weights released; licences of every input re-audited

---

## v0.1 measured (2026-08-26, NVIDIA L4 23 GB / 12 CPU / 52 GB RAM, Docker-free simulation)

**The 4-channel baseline**, kept as the control. Case `psma_0198cdca94fbb95f_2020-05-09`,
200x200x462 = 18.5 M voxels, native PSMA resolution 4.07x4.07x2.0 mm. Timings from
`/usr/bin/time -v` around `python -m submission.process`; RSS from the container's own
`[mem]` line.

| run | scenario | wall | inference | peak RSS self / children | output |
|---|---|---|---|---|---|
| A | iteration 1, 8 tumor points | 188 s | 175.0 s | 4.53 / 3.11 GB | 5506 voxels |
| B | byte-for-byte rerun of A | 172 s | 157.6 s | 4.54 / 3.11 GB | **identical to A** (same md5) |
| C | iteration 0, `lesion-clicks.json` absent | 174 s | 159.1 s | 4.57 / 3.11 GB | 5481 voxels, **bit-identical to the offline harness** |
| D | bogus `AUTOPETV_MODEL_FOLDER` | 10 s | -- | 1.14 / 1.05 GB | empty mask, exit 0 |
| E | two files in `/input/images/ct` | 3 s | -- | 0.78 GB | empty mask, exit 0 |
| F | iteration 0, weights fetched from the organizers' Drive zip | 144 s | 130.0 s | 4.55 / 3.11 GB | 5481 voxels, **bit-identical to the offline harness** |

Weight fetch itself (`WEIGHTS_SOURCE=gdrive`, fresh dir, real download): **26 s**, 236 MB
model folder, zip deleted. All four sha256 checks pass. The three failure paths
(placeholder release URL, wrong zip hash, unknown source) each exit **1**, so a bad
artifact fails `docker build` instead of shipping.

`npp = nps = 3` was also measured (223 s / 173 s wall) with no advantage, so the defaults
are `1` to stay clear of the 2 GB `/dev/shm`.

**Known degenerate case (E).** If `/input/images/ct/` ever holds more than one file, the
CT uuid is unknowable; the container logs the assertion, falls back to the PET's geometry
*and the PET's uuid*, and writes an empty mask under that name. The evaluator would not
find the file it expects, but the job still exits 0 rather than crashing the case.
