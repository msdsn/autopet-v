# Interactive Whole-Body PET/CT Lesion Segmentation (autoPET V)

Code for our entry to the [autoPET V challenge](https://autopet-v.grand-challenge.org) (MICCAI 2026),
on interactive lesion segmentation in whole-body FDG and PSMA PET/CT.

## Method

The evaluator calls the algorithm six times per case. Iteration 0 sees only CT and PET; each of the next
five adds one simulated scribble — a short 2D stroke on the axial slice holding the largest connected
error component, labelled *tumor* or *background* — and the algorithm returns an updated lesion mask. The
score is the AUC over the six iterations of Dice and of DMM (a lesion-level F1 with IoU ≥ 0.1), 50/50.

We fine-tune the organizers' nnU-Net baseline into an interactive model with **five input channels**: CT,
PET (SUV), tumor guidance, background guidance and the previous iteration's mask. The guidance channels
encode the accumulated scribbles as a clipped Euclidean distance transform, `max(0, 1 − d/R)` with
`R = 10` voxels, instead of the baseline's z-scored point heatmaps. Training uses **online scribble
simulation**: for every patch a plausible previous prediction is sampled by corrupting the label and the
challenge's own simulator is run against that error, so the network meets the test-time interaction
protocol during training. A **scribble-consistent post-processing** layer then enforces the two
constraints the metric implicitly rewards — every tumor scribble ends up inside the mask, every
background scribble removes its component — together with small-component cleanup and a **negative-case
gate**, which matters because a single false-positive voxel zeroes the Dice of a lesion-free case for all
six iterations. Method description: arXiv (to be added).

## Results

Held-out set: 100 cases from the fold-0 validation split (`docs/valset_v1.txt`), 63 FDG / 37 PSMA, 36 of
them lesion-free. Six iterations, all three scribble strategies round-robin, `seed=42`, official metrics;
AUC is the trapezoid over iterations 0..5, max 5.0. AUC-DMM is a nanmean over lesion-bearing cases only,
so it equals the positive-only column by construction.

| row | configuration | AUC-Dice | AUC-DMM | neg AUC-Dice | pos AUC-DMM |
|---|---|---:|---:|---:|---:|
| `A0` | nnU-Net baseline (4 channels), no post-processing | 3.635 | 3.222 | 3.611 | 3.222 |
| `A3` | A0 + scribble compliance + component cleanup | 3.639 | 3.301 | 3.611 | 3.301 |
| `A9` | A3 + DMM cleanup rule v2 | 3.655 | 3.415 | 3.611 | 3.415 |
| `B0` | interactive 5-channel fine-tune, no post-processing | 3.344 | 3.497 | 2.778 | 3.497 |
| `B3` | B0 + A3 post-processing | 3.324 | 3.563 | 2.778 | 3.563 |
| `B9` | B3 + DMM cleanup rule v2 | 3.469 | 3.824 | 2.778 | 3.824 |
| `B3g` | B3 + negative-case gate (6 mL) | 4.003 | 3.508 | 4.722 | 3.508 |
| `B3g25` | B3g with the gate threshold raised to 25 mL | 4.115 | 3.509 | 4.861 | 3.509 |
| `B10g` | false-positive-aware fine-tune + B3g pipeline | 4.097 | 3.572 | 4.722 | 3.572 |
| **`B10g9_ship`** | **submitted system** (`submission/postproc_config.json`) | **4.142** | **3.585** | **4.722** | **3.585** |

Submitted system, per-iteration means over iterations 0..5 — Dice 0.664 / 0.818 / 0.840 / 0.855 / 0.863 /
0.867, DMM 0.439 / 0.695 / 0.730 / 0.761 / 0.781 / 0.798, both monotone. Full ladder: `results/RESULTS.md`.

## Installation

Python ≥ 3.11 with a CUDA-capable PyTorch install (developed against torch 2.6 / CUDA 12.4).

```bash
pip install -r requirements-dev.txt                  # development environment
git clone https://github.com/lab-midas/autoPETV.git  # the official challenge repo
```

`requirements-dev.txt` is the development environment; `requirements-submission.txt` pins the exact versions
in the container image. The official challenge repository is required at run time (path via `--repo`): its
scribble simulator and metrics are imported, not re-implemented; pinned commit in `docs/autoPETV_pin.txt`.

## Usage

### Evaluate a model with the interactive loop

`src/interactive_eval.py` reproduces the official six-iteration loop in-process — same scribbles, same
metrics — so a full ablation is one run rather than 600 container calls.

```bash
python src/interactive_eval.py \
    --input_cases /data/evalset --image_dir /data/evalset/imagesTr --label_dir /data/evalset/labelsTr \
    --repo /path/to/autoPETV --out_dir runs/interactive \
    --predictor postproc --base_predictor interactive_nnunet \
    --model_folder /models/nnUNetTrainer_Interactive__nnUNetPlans_interactive__3d_fullres \
    --postproc_config submission/postproc_config.json --cache_dir .cache/predictions
```

Results go to `<out_dir>/{metric_scores,metric_scores_AUC,summary}.json`, stratified by tracer and lesion
status (`docs/eval_harness.md`). `src/data/` converts the challenge archive into an nnU-Net preprocessed
store and picks the held-out set (`docs/data_pipeline.md`).

### Train

```bash
export nnUNet_raw=... nnUNet_preprocessed=... nnUNet_results=...
export nnUNet_extTrainer=$PWD/src/train          # external trainer search path (nnU-Net >= 2.8)
export PYTHONPATH=$PWD/src:$PYTHONPATH
export AUTOPETV_REPO=/path/to/autoPETV           # for the official scribble simulator
export nnUNet_interactive_pretrained=/models/interactive_init_5ch.pth

nnUNetv2_train Dataset998_AutoPETV 3d_fullres 0 \
    -tr nnUNetTrainer_Interactive -p nnUNetPlans_interactive
```

Plans and the 4→5 channel weight surgery: `train.make_plans`, `train.init_from_baseline` (`docs/train_pipeline.md`).

### Container

Grand Challenge builds `Dockerfile` from this repository; the weights are not in git but are downloaded and
sha256-checked at build time by `scripts/fetch_weights.sh`. Entry point `submission/process.py`, see
`docs/submission.md`. Locally:

```bash
docker build -t autopetv .
docker run --rm --gpus all --network=none -v $PWD/input:/input -v $PWD/output:/output autopetv
```

## Repository layout

```
src/interactive_eval.py   interactive evaluation loop, metrics, AUC, prediction cache
src/predictor.py          predictor interface: baseline nnU-Net, fast path, interactive 5-channel model
src/ablate.py             ablation driver over the ladder in configs/ablations.json
src/postproc/             scribble-consistent post-processing: compliance, cleanup, negative gate
src/train/                interactive trainer, scribble simulation transform, guidance encoding
src/data/                 dataset analysis, preprocessed-store builder, held-out set selection
submission/               Grand Challenge entry point, post-processing config, container tests
scripts/                  weight download, ablation sweep, tracked-file check
scripts/env/              environment glue for the GPU machines used during development
docs/                     eval_harness, postproc, train_pipeline, data_pipeline, submission, held-out set
results/                  evaluation runs behind the table above (RESULTS.md and per-row run.json)
```

## Citation

If you use this code, please cite the challenge and the underlying datasets:

```
Gatidis S, Kuestner T. A whole-body FDG-PET/CT dataset with manually annotated tumor lesions
(FDG-PET-CT-Lesions) [Dataset]. The Cancer Imaging Archive, 2022. DOI: 10.7937/gkr0-xv29

Gatidis S, Hepp T, Früh M, et al. A whole-body FDG-PET/CT dataset with manually annotated tumor
lesions. Sci Data 9, 601 (2022). https://doi.org/10.1038/s41597-022-01718-3

Jeblick K, et al. A whole-body PSMA-PET/CT dataset with manually annotated tumor lesions
(PSMA-PET-CT-Lesions) (Version 1) [Dataset]. The Cancer Imaging Archive, 2024. DOI: 10.7937/r7ep-3x37
```

## License

Apache License 2.0, see `LICENSE`.
