# Data pipeline

How the 1611-case FDG + PSMA PET/CT dataset (`Dataset998_AutoPETV`) is turned into an nnU-Net
preprocessed store for training and into a raw held-out set for the interactive evaluation.

| script (`src/data/`) | what it does |
|---|---|
| `partial_zip_extract.py` | read/extract members from the (possibly still downloading) source zip: `layout`, `meta`, `avail`, `labels`, `cases`, `fetch-labels`, `fetch-cases` |
| `analyze_dataset.py` | cc3d lesion analysis + fingerprint/splits reconciliation → `label_stats.json` |
| `build_store.py` | `measure` (size and time per store variant), `project` (extrapolate to the cohort by voxels), `build` (the training store, resumable), `evalset` (held-out set selection and copy) |
| `verify_store.py` | checks that a built store loads through nnU-Net's own `infer_dataset_class` / `nnUNetDatasetNumpy` / `nnUNetDataLoader` |
| `run_full_build.sh` | build loop: process whatever is available, sleep, repeat until complete, then verify |

## Source data

The challenge distributes the FDG + PSMA cohort as one 140 GiB zip
([DOI 10.57754/FDAT.rdkqd-wdh87](https://doi.org/10.57754/FDAT.rdkqd-wdh87)), 4838 members. Two properties
of the member layout are worth knowing, because they let the analysis start hours before the download
finishes:

* the labels sit at the very end of the archive but are *tiny* — the label NIfTIs are written with a weak
  gzip setting, so the zip's deflate layer squeezes 0.26 GB down to **9.4 MB**. One HTTP range request for
  the last 9.4 MB yields all 1611 labels, i.e. the whole cohort's geometry and lesion statistics for
  0.006 % of the download;
* PSMA CT is stored after *all* PSMA PET, so no PSMA case is complete before 13.7 % of the archive; FDG
  members are interleaved per case and complete one by one from there.

`partial_zip_extract.py` supports this by handing Python's `zipfile` a file-like object that stitches
together byte ranges of the *virtual* full archive: the growing local file, a separately fetched copy of
the central directory, and any number of out-of-order patch ranges. A read that lands in a hole raises
`GapError`; member availability is decided by parsing the member's local file header rather than by a
pessimistic slack.

## Cohort

`analyze_dataset.py` runs cc3d with 26-connectivity over every label. The `dataset_fingerprint.json`
arrays carry no case ids — they are in `sorted(case_id)` order, which we confirmed by matching all 1611
shapes against the labels.

| tracer | cases | lesion-free | total lesions | median lesions / positive case | median tumour vol (cc) |
|---|---:|---:|---:|---:|---:|
| FDG | 1014 | 513 (50.6 %) | 8,620 | 6 | 99.2 |
| PSMA | 597 | 58 (9.7 %) | 18,922 | 8 | 44.1 |
| **all** | **1611** | **571 (35.4 %)** | **27,542** | 7 | 73.0 |

The two tracers are very different cohorts — half of FDG is lesion-free against a tenth of PSMA — so any
sampling has to be stratified on tracer *and* on lesion presence. Median lesion volume is ~1 cc, about 130
voxels at the plans spacing, and the tail is long (one FDG case has 1046 connected components), which
matters because DMM is a per-lesion metric.

Geometry: FDG is homogeneous at 3.0 × 2.0364 × 2.0364 mm, exactly the plans spacing, so it needs no
resampling. PSMA has three spacings and is mostly at 4.07 mm in-plane, i.e. half the plans resolution, so
preprocessing PSMA at the baseline spacing *upsamples* it 2× in-plane. That is the dominant cost in both
storage and preprocessing time.

Fold 0 of the official `splits_final.json` is 1288 train / 323 val and is tracer- and burden-proportional,
so it can be used as-is.

## The training store

The store is a real `nnUNet_preprocessed` folder, so the published baseline can be fine-tuned with a
stock trainer by pointing `nnUNet_preprocessed` at it — no bespoke dataset code. `build_store.py` drives
`DefaultPreprocessor.run_case_npy()` per case (rather than `nnUNetv2_preprocess`, which enumerates every
identifier in `dataset.json` and fails on a partial `imagesTr`), which is also what makes the build
resumable.

Four variants were built on 10 cases and measured, not estimated, before choosing one:

| variant / format | MB/case FDG | MB/case PSMA | Mvoxel FDG | s/case (1 worker) | projected total |
|---|---:|---:|---:|---:|---:|
| baseline / npz / fp32 | 56.7 | 86.1 | 48.9 | 38.3 | 111.6 GB |
| baseline / b2nd / fp16 | 27.8 | 25.5 | 48.9 | 32.9 | 45.8 GB |
| **bodycrop / b2nd / fp16** | **23.8** | **23.0** | **12.6** | **11.2** | **39.9 GB** |
| iso3 / b2nd / fp16 | 15.6 | 12.8 | 22.6 | 32.6 | 24.8 GB |
| iso3_bodycrop / b2nd / fp16 | 12.1 | 11.5 | 5.8 | 11.4 | 20.3 GB |

Body crop = largest 3D connected component of `CT > −500 HU`, computed on a 4× in-plane downsample,
expanded by 4 voxels and **unioned with the label bounding box so a lesion can never be cropped away**.
Projection to the full cohort scales by *voxels*, not by case count, using the true mean megavoxel count
from `label_stats.json`; it came out within ~1 % of the built store (predicted 14.4 / 11.7 Mvoxel for
FDG / PSMA, measured mean 14.1 over all 1611).

We use **`bodycrop` / `b2nd` / float16**, 1611 cases, 39.5 GB:

* **Same spacing as the baseline plans**, so the published weights fine-tune directly.
* **Same intensities as the baseline.** Body cropping naively would break this: nnU-Net z-scores PET over
  the *whole* volume, so deleting ~74 % of the voxels (almost all near-zero PET) shifts channel 1.
  `build_store.py` records the full-volume PET mean/std before cropping and applies the affine correction
  `z' = z·(σ_c/σ_f) + (μ_c−μ_f)/σ_f` afterwards. This is exact because both normalisations are affine in
  the voxel values and resampling is linear, so the correction commutes with the resampling; CT is
  unaffected because `CTNormalization` uses fixed fingerprint statistics.
* **blosc2 over npz**: 8–18 % smaller *and* mmap-able with patch-sized chunks, so the dataloader
  decompresses only the blocks it touches. An npz store has to be inflated whole for every sample.
* **float16 is safe**: `nnUNetDataLoader` does `torch.from_numpy(crop_and_pad_nd(...)).float()` and
  allocates the batch as float32, so the upcast happens before any transform sees the data.
* Body cropping removes ~74 % of the voxels but only ~15 % of the bytes — the air it deletes was nearly
  free to compress. Its real payoff is speed: 11 s vs. 33 s per case to preprocess, and a much smaller
  working set at training time.

`iso3` is not used because changing the spacing invalidates the baseline weights, which would mean
re-planning and training from scratch. The `baseline` variant (45.8 GB) is the fallback if zero geometric
deviation is ever preferred over speed.

### On-disk format

```
<case>.b2nd       float16, shape (4, z, y, x)   # blosc2, chunked to the patch size
<case>_seg.b2nd   uint8,   shape (1, z, y, x)   # values {0, 1}
<case>.pkl        properties dict
```

| thing | value |
|---|---|
| dataset | `Dataset998_AutoPETV`, labels `{"background": 0, "tumor": 1}` |
| plans / configuration | `nnUNetPlans.json`, `3d_fullres`, `data_identifier` `nnUNetPlans_3d_fullres` |
| patch / batch | `[112, 160, 128]` / 2 |
| spacing | `[3.0, 2.0364201068878174, 2.0364201068878174]` |
| splits | `splits_final.json`, official 5-fold, **fold 0** |

`.pkl` carries everything `DefaultPreprocessor` normally writes (`class_locations` for foreground
oversampling, `spacing`, the cropping shapes, `sitk_stuff`) plus two keys of ours: `body_bbox`, the crop
box in the raw voxel grid so a prediction can be placed back into the original volume, and
`pet_norm_correction`, the `{mu_full, sd_full, mu_crop, sd_crop}` used for the intensity correction above.

**Channels 2 and 3 are stored as all zeros.** `dataset.json` declares four channels (CT, PET, FG, BG)
because the baseline network has four input channels, but the scribble guidance is generated on the fly
during training, so storing it would be both wasteful and wrong. All-zero float16 planes cost essentially
nothing under blosc2/zstd, and keeping them means the array shape already matches the baseline network.
The consequence is that a *stock* `nnUNetTrainer` run on this store only ever sees the iteration-0,
no-interaction case; the interactive trainer fills channels 2–4 itself (`docs/train_pipeline.md`).
`verify_store.py` asserts that they are still zero on disk.

## Held-out evaluation set

100 cases drawn from the fold-0 validation split, stratified on tracer × lesion-count bucket in proportion
to the source split: FDG 31 positive / 32 lesion-free, PSMA 33 / 4. Raw NIfTI at native resolution (9.1
GiB), so the official interactive loop runs without any of our preprocessing in the way. Selection is
deterministic (`build_store.py evalset --n 100 --fold 0 --seed 0`) and the resulting list is pinned in the
repository as `docs/valset_v1.txt` with `docs/valset_v1_composition.json`.

## Running it

```bash
# early access while the archive is still downloading
python src/data/partial_zip_extract.py layout
python src/data/partial_zip_extract.py fetch-labels          # 9.4 MB range GET, all 1611 labels
python src/data/partial_zip_extract.py labels --out work/labelsTr

python src/data/analyze_dataset.py --labels work/labelsTr --out meta/label_stats.json

# measure store variants before committing to one
python src/data/build_store.py measure --n 10 --raw-dir work/testcase --workers 5 \
    --variants baseline iso3 bodycrop iso3_bodycrop --formats npz:float16 b2nd:float16

# the full build (training store + held-out set), resumable
bash src/data/run_full_build.sh
python src/data/verify_store.py --store <store>/nnUNetPlans_3d_fullres --sample 12 --expect 1611
```

`verify_store.py` does a light pass over every case (file triplet present, `.pkl` loads and carries
`class_locations`, shapes agree, 4 channels, float16) and a deep pass on a sample (finite values, `seg` in
{0,1}, channels 2–3 still zero), then assembles one real batch through `nnUNetDataLoader`.

## Known limitations

* **Body cropping changes the patch distribution.** Intensities are provably unchanged, but the sampling
  of background patches is not: there are no more pure-air patches. This is more likely to help than hurt
  — nnU-Net's InstanceNorm makes the network largely indifferent — but it is a real deviation from how the
  baseline was trained, and the `baseline` variant is the escape hatch.
* **Never let nnU-Net unpack the store.** `unpack_dataset` materialises a full uncompressed array per
  case, >200 GB here. blosc2 is mmap'd and chunked to the patch size, so unpacking buys nothing.
* **`infer_dataset_class` is brittle.** It asserts a single file extension across the whole store folder
  (`.pkl`/`.npy` excepted), so a stray `manifest.json` or a `.part.npz` from a killed run aborts training
  with a confusing assertion. The build manifest is therefore written as a *sibling* of the store
  directory, and npz output goes through a file object so the temporary name never ends in `.npz`.
* **The `dataset.json` inside the source zip is wrong** — it declares two channels, both named `CT`. The
  authoritative one is the baseline's (`CT, PET, FG, BG`), which is what `build_store.py` uses.
* **The held-out cases are in the training store.** They are fold-0 validation cases, so the split file
  excludes them from training — but only when training on fold 0. Training on `all` would leak them.
* **float16 has been checked for the data channels only.** A custom hook that reads the raw array *before*
  the loader's `.float()` would observe float16.
* **DEEP-PSMA is not used.** The extra 100 patients (Zenodo 15281784) are CC BY-NC, a licence risk for a
  challenge submission, and 200 extra volumes against 1611 is a small payoff.
  `src/data/fetch_deep_psma.sh` keeps the option costless to revive.
