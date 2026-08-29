"""Frozen EVA-02-B features for the preprocessed autoPET V store ("B16").

Renders axial slices of a body-cropped case at plans spacing into the three input
channels of an ImageNet-pretrained EVA-02 base model, runs the frozen backbone, and
writes the patch-token features -- PCA-reduced to ``K`` channels -- as a small volume
aligned to the store grid.

Why a *token* grid on disk.  EVA-02-B at 448 px with patch 14 emits a 32x32 token
grid per slice.  A body-cropped case is about (300, 190, 215) voxels, so one token
covers roughly 6 voxels in-plane = 12 mm, and the feature volume is
``(K, Z, 32, 32)``: 4.9 MB per case at K=8 / float16 against 190 MB if it were
written at full in-plane resolution.  The consumer upsamples it with a trilinear
``interpolate`` inside the data pipeline, which is exactly the alignment the token
grid implies (a squash resize maps the 32x32 grid uniformly onto the (Y, X) extent).

Channels handed to EVA (the model was trained on natural RGB, so all three are
[0, 1] images before the CLIP mean/std normalisation):

* 0 -- CT in a soft-tissue window [-160, 240] HU;
* 1 -- PET, log SUV, ``log1p(SUV) / log1p(60)``;
* 2 -- PET, log SUV of a +/-4-slice (+/-12 mm) maximum-intensity slab, i.e. the
  thick-slab MIP a reader looks at.  A 2D backbone has no way to see out of the
  slice otherwise, and small lesions are the bucket that matters here.

The store keeps CT and PET *normalised*, so both are inverted back to physical
units first: CT with the fixed fingerprint statistics from ``plans.json``, PET with
the per-case ``pet_norm_correction`` written by ``src/data/build_store.py``.

Usage (on the GPU box, ``source /content/env.sh``)::

    python src/train/eva02_features.py smoke  --store <store> --out <dir> --n 3
    python src/train/eva02_features.py fit-pca --store <store> --out <dir> --n 50
    python src/train/eva02_features.py extract --store <store> --out <dir>

``smoke`` fits a PCA on the same handful of cases it extracts, so it is
self-contained; a real run does ``fit-pca`` once and then ``extract``.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import random
import time
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

# plans.json foreground_intensity_properties_per_channel["0"] -- CTNormalization
CT_MEAN, CT_STD = 107.73438968591431, 286.34403119451997
CT_WINDOW = (-160.0, 240.0)          # soft tissue
PET_SUV_CLIP = 60.0                  # log scale saturates well below this
MIP_HALF = 4                         # +/- 4 slices = +/- 12 mm at 3 mm spacing

MODEL_NAME = "eva02_base_patch14_448.mim_in22k_ft_in22k_in1k"
IMG_SIZE = 448
TOKEN_GRID = 32                      # 448 / 14


# ---------------------------------------------------------------------------
# store access
# ---------------------------------------------------------------------------

def list_cases(store: Path) -> List[str]:
    out = []
    for f in sorted(store.glob("*.b2nd")):
        if f.name.endswith("_seg.b2nd"):
            continue
        out.append(f.name[:-5])
    return out


def load_case(store: Path, case: str) -> Tuple[np.ndarray, np.ndarray, dict]:
    """Return (ct_hu, suv, properties) as float32 arrays of shape (Z, Y, X)."""
    import blosc2

    arr = blosc2.open(str(store / f"{case}.b2nd"), mode="r")
    data = np.asarray(arr[0:2], dtype=np.float32)          # (2, Z, Y, X)
    with open(store / f"{case}.pkl", "rb") as fh:
        props = pickle.load(fh)

    ct = data[0] * CT_STD + CT_MEAN
    corr = props["pet_norm_correction"]
    suv = data[1] * float(corr["sd_full"]) + float(corr["mu_full"])
    return ct, suv, props


# ---------------------------------------------------------------------------
# rendering
# ---------------------------------------------------------------------------

def render_slices(ct: np.ndarray, suv: np.ndarray) -> torch.Tensor:
    """(Z, Y, X) CT[HU] and PET[SUV] -> (Z, 3, Y, X) float32 in [0, 1]."""
    lo, hi = CT_WINDOW
    ct_w = np.clip((ct - lo) / (hi - lo), 0.0, 1.0)

    pet = np.clip(suv, 0.0, PET_SUV_CLIP)
    pet_log = np.log1p(pet) / np.log1p(PET_SUV_CLIP)

    t = torch.from_numpy(pet_log)[None, None]              # (1, 1, Z, Y, X)
    mip = F.max_pool3d(t, kernel_size=(2 * MIP_HALF + 1, 1, 1),
                       stride=1, padding=(MIP_HALF, 0, 0))[0, 0].numpy()

    out = np.stack([ct_w, pet_log, mip], axis=1)           # (Z, 3, Y, X)
    return torch.from_numpy(np.ascontiguousarray(out))


def to_model_input(batch: torch.Tensor, mean: torch.Tensor, std: torch.Tensor,
                   dtype: torch.dtype) -> torch.Tensor:
    """(B, 3, Y, X) in [0, 1] -> (B, 3, 448, 448) normalised, on the same device."""
    x = F.interpolate(batch, size=(IMG_SIZE, IMG_SIZE), mode="bilinear",
                      align_corners=False)
    return ((x - mean) / std).to(dtype)


# ---------------------------------------------------------------------------
# model
# ---------------------------------------------------------------------------

def build_model(device: str = "cuda", dtype: torch.dtype = torch.float16):
    import timm

    model = timm.create_model(MODEL_NAME, pretrained=True, num_classes=0)
    model.eval().to(device=device, dtype=dtype)
    for p in model.parameters():
        p.requires_grad_(False)
    cfg = timm.data.resolve_data_config({}, model=model)
    mean = torch.tensor(cfg["mean"], device=device).view(1, 3, 1, 1)
    std = torch.tensor(cfg["std"], device=device).view(1, 3, 1, 1)
    n_prefix = int(getattr(model, "num_prefix_tokens", 1))
    return model, mean, std, n_prefix


@torch.no_grad()
def tokens_for_case(model, mean, std, n_prefix, slices: torch.Tensor,
                    batch_size: int, device: str, dtype: torch.dtype) -> torch.Tensor:
    """(Z, 3, Y, X) -> (Z, 1024, 768) float32 patch tokens, on the GPU."""
    outs = []
    for i in range(0, slices.shape[0], batch_size):
        chunk = slices[i:i + batch_size].to(device, non_blocking=True)
        x = to_model_input(chunk, mean, std, dtype)
        feat = model.forward_features(x)[:, n_prefix:]
        outs.append(feat.float())
    return torch.cat(outs, dim=0)


# ---------------------------------------------------------------------------
# PCA
# ---------------------------------------------------------------------------

class PCAAccumulator:
    """Streaming mean / second-moment accumulator; PCA by eigh of the covariance.

    768 x 768 is small, so the exact covariance is cheaper and more accurate than a
    randomized SVD over a token subsample kept in RAM.
    """

    def __init__(self, dim: int, device: str = "cuda"):
        self.n = 0
        self.s1 = torch.zeros(dim, dtype=torch.float64, device=device)
        self.s2 = torch.zeros(dim, dim, dtype=torch.float64, device=device)

    def add(self, tokens: torch.Tensor) -> None:
        t = tokens.reshape(-1, tokens.shape[-1]).double()
        self.n += t.shape[0]
        self.s1 += t.sum(0)
        self.s2 += t.T @ t

    def fit(self, k: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        mu = self.s1 / self.n
        cov = self.s2 / self.n - torch.outer(mu, mu)
        cov = 0.5 * (cov + cov.T)
        evals, evecs = torch.linalg.eigh(cov)
        order = torch.argsort(evals, descending=True)
        evals, evecs = evals[order], evecs[:, order]
        total = float(evals.clamp_min(0).sum())
        comps = evecs[:, :k].T.contiguous()                # (k, dim)
        ratio = (evals[:k] / total).cpu().numpy()
        return mu.float().cpu().numpy(), comps.float().cpu().numpy(), ratio


def project(tokens: torch.Tensor, mu: torch.Tensor, comps: torch.Tensor,
            scale: torch.Tensor) -> torch.Tensor:
    """(Z, N, D) -> (K, Z, 32, 32) float16, unit-variance per component."""
    z = (tokens - mu) @ comps.T                            # (Z, N, K)
    z = z / scale
    zz, n, k = z.shape
    g = int(round(n ** 0.5))
    return z.reshape(zz, g, g, k).permute(3, 0, 1, 2).contiguous().half()


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------

def run_pca(store: Path, cases: Sequence[str], out: Path, k: int, batch_size: int,
            max_slices: int, device: str, dtype: torch.dtype) -> Path:
    model, mean, std, n_prefix = build_model(device, dtype)
    acc = PCAAccumulator(768, device)
    t0 = time.time()
    for i, case in enumerate(cases):
        ct, suv, _ = load_case(store, case)
        sl = render_slices(ct, suv)
        if sl.shape[0] > max_slices:                       # every n-th slice
            idx = np.linspace(0, sl.shape[0] - 1, max_slices).round().astype(int)
            sl = sl[idx]
        tok = tokens_for_case(model, mean, std, n_prefix, sl, batch_size, device, dtype)
        acc.add(tok[:, ::7])                               # 1 token in 7 is plenty
        print(f"[pca {i + 1}/{len(cases)}] {case[:40]} slices={sl.shape[0]} "
              f"n={acc.n} {time.time() - t0:.1f}s", flush=True)
    mu, comps, ratio = acc.fit(k)

    z = (torch.from_numpy(comps).to(device).double() @
         (acc.s2 / acc.n - torch.outer(acc.s1 / acc.n, acc.s1 / acc.n)) @
         torch.from_numpy(comps).to(device).double().T)
    scale = torch.sqrt(torch.diagonal(z).clamp_min(1e-12)).float().cpu().numpy()

    out.mkdir(parents=True, exist_ok=True)
    path = out / "eva02_pca.npz"
    np.savez(path, mean=mu, components=comps, scale=scale,
             explained_variance_ratio=ratio, k=k, model=MODEL_NAME,
             n_tokens=acc.n, cases=np.array(list(cases)))
    print(f"[pca] k={k} explained={ratio.sum():.3f} per-comp={np.round(ratio, 4)}")
    print(f"[pca] wrote {path}")
    return path


def run_extract(store: Path, cases: Sequence[str], out: Path, pca_path: Path,
                batch_size: int, device: str, dtype: torch.dtype) -> None:
    model, mean, std, n_prefix = build_model(device, dtype)
    p = np.load(pca_path, allow_pickle=True)
    mu = torch.from_numpy(p["mean"]).to(device)
    comps = torch.from_numpy(p["components"]).to(device)
    scale = torch.from_numpy(p["scale"]).to(device)

    feat_dir = out / "features"
    feat_dir.mkdir(parents=True, exist_ok=True)
    times, sizes = [], []
    for i, case in enumerate(cases):
        t0 = time.time()
        ct, suv, props = load_case(store, case)
        sl = render_slices(ct, suv)
        tok = tokens_for_case(model, mean, std, n_prefix, sl, batch_size, device, dtype)
        feat = project(tok, mu, comps, scale).cpu().numpy()
        dst = feat_dir / f"{case}.npz"
        np.savez_compressed(dst, feat=feat, shape=np.array(ct.shape),
                            token_grid=np.array([TOKEN_GRID, TOKEN_GRID]))
        dt, mb = time.time() - t0, dst.stat().st_size / 2 ** 20
        times.append(dt)
        sizes.append(mb)
        print(f"[extract {i + 1}/{len(cases)}] {case[:40]} vol={tuple(ct.shape)} "
              f"feat={feat.shape} {dt:.2f}s {mb:.2f}MB", flush=True)
    if times:
        print(f"[extract] mean {np.mean(times):.2f} s/case, {np.mean(sizes):.2f} MB/case; "
              f"projected 1611 cases: {np.mean(times) * 1611 / 3600:.2f} GPU-h, "
              f"{np.mean(sizes) * 1611 / 1024:.1f} GB")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["smoke", "fit-pca", "extract"])
    ap.add_argument("--store", required=True, type=Path,
                    help="<...>/Dataset998_AutoPETV/nnUNetPlans_3d_fullres")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--pca", type=Path, default=None)
    ap.add_argument("--cases", type=Path, default=None, help="file with one case id per line")
    ap.add_argument("--n", type=int, default=0, help="use only the first n cases (0 = all)")
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-pca-slices", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--fp32", action="store_true")
    args = ap.parse_args()

    dtype = torch.float32 if args.fp32 else torch.float16
    cases = ([c.strip() for c in args.cases.read_text().splitlines() if c.strip()]
             if args.cases else list_cases(args.store))
    if args.mode in ("smoke", "fit-pca") and not args.cases:
        random.Random(args.seed).shuffle(cases)
    if args.n:
        cases = cases[:args.n]
    print(f"[eva02] mode={args.mode} cases={len(cases)} dtype={dtype} store={args.store}")

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "run_args.json").write_text(json.dumps(
        {k: str(v) for k, v in vars(args).items()}, indent=2))

    if args.mode == "fit-pca":
        run_pca(args.store, cases, args.out, args.k, args.batch_size,
                args.max_pca_slices, args.device, dtype)
    elif args.mode == "extract":
        pca = args.pca or (args.out / "eva02_pca.npz")
        run_extract(args.store, cases, args.out, pca, args.batch_size, args.device, dtype)
    else:
        t0 = time.time()
        pca = run_pca(args.store, cases, args.out, args.k, args.batch_size,
                      args.max_pca_slices, args.device, dtype)
        run_extract(args.store, cases, args.out, pca, args.batch_size, args.device, dtype)
        print(f"[smoke] total {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
