#!/usr/bin/env python3
"""
Build the persistent autoPET V training store on Google Drive.

The raw dataset is 152 GB and nnU-Net's own preprocessed output would be ~1.2 TB, so
we write a compact store instead. Tier A: preprocessed cases as .npz + .pkl in the
layout nnUNetDatasetNumpy reads, float16 (the dataloader upcasts after cropping) with
the guidance channels zero-padded to the baseline's 4 inputs. Tier B: raw .nii.gz for
a stratified held-out subset. Cases go through DefaultPreprocessor one at a time
because nnUNetv2_preprocess refuses a half-downloaded imagesTr.

Variants:
    baseline          plans spacing, no body crop
    iso3              3 mm isotropic, no body crop
    bodycrop          plans spacing + crop to the CT body bounding box (HU>-500)
    iso3_bodycrop     both

Usage:
    # measure every variant on the cases already downloaded
    python build_store.py measure --n 10 --workdir /content/work/measure

    # build Tier A for a list of cases (resumable)
    python build_store.py build --variant bodycrop --cases-file cases.txt \
        --out /content/drive/MyDrive/autoPET/store/bodycrop

    # pick + copy the Tier B held-out raw set
    python build_store.py evalset --n 100 --out /content/drive/MyDrive/autoPET/evalset
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import shutil
import sys
import time
import traceback
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

DRIVE = Path(os.environ.get("AUTOPET_DRIVE", "/content/drive/MyDrive/autoPET"))
PLANS = DRIVE / "weights/nnUNet_results/Dataset998_AutoPETV/nnUNetTrainer__nnUNetPlans__3d_fullres/plans.json"
BASE_DATASET_JSON = DRIVE / "weights/nnUNet_results/Dataset998_AutoPETV/nnUNetTrainer__nnUNetPlans__3d_fullres/dataset.json"
META = DRIVE / "meta"
BODY_HU = -500.0

VARIANTS = {
    #  name            spacing override            body crop
    "baseline":        (None,             False),
    "iso3":            ([3.0, 3.0, 3.0],  False),
    "bodycrop":        (None,             True),
    "iso3_bodycrop":   ([3.0, 3.0, 3.0],  True),
}


# ------------------------------------------------------------- body cropping --

def body_bbox(ct: np.ndarray, seg: np.ndarray | None = None, margin: int = 4,
              ds: int = 4) -> tuple[slice, slice, slice]:
    """Bounding box of the patient in a CT volume (z, y, x), HU based.

    Thresholds at BODY_HU and keeps the largest connected component (drops the couch
    edge and gantry speckle), then expands by `margin` voxels. Any labelled voxel is
    always included, so cropping cannot remove a lesion. Runs on a `ds`-fold in-plane
    downsample.
    """
    import cc3d

    small = ct[:, ::ds, ::ds] > BODY_HU
    if not small.any():
        return tuple(slice(0, s) for s in ct.shape)
    lab, n = cc3d.connected_components(small, connectivity=26, return_N=True)
    if n > 1:
        counts = np.bincount(lab.reshape(-1))
        counts[0] = 0
        small = lab == int(counts.argmax())

    out = []
    for ax, full in enumerate(ct.shape):
        axes = tuple(a for a in range(3) if a != ax)
        idx = np.flatnonzero(small.any(axis=axes))
        scale = 1 if ax == 0 else ds
        lo = int(idx[0]) * scale - margin
        hi = (int(idx[-1]) + 1) * scale + margin
        out.append([max(lo, 0), min(hi, full)])

    if seg is not None and seg.any():
        for ax in range(3):
            axes = tuple(a for a in range(3) if a != ax)
            idx = np.flatnonzero(seg.any(axis=axes))
            out[ax][0] = min(out[ax][0], max(int(idx[0]) - margin, 0))
            out[ax][1] = max(out[ax][1], min(int(idx[-1]) + 1 + margin, ct.shape[ax]))
    return tuple(slice(a, b) for a, b in out)


def shift_origin(props: dict, sl: tuple[slice, slice, slice]) -> None:
    """Move the stored SimpleITK origin to the crop start so a prediction can
    still be written back into world coordinates."""
    st = props.get("sitk_stuff")
    if not st:
        return
    origin = np.asarray(st["origin"], dtype=float)          # (x, y, z)
    spacing = np.asarray(st["spacing"], dtype=float)        # (x, y, z)
    d = np.asarray(st["direction"], dtype=float).reshape(3, 3)
    idx = np.array([sl[2].start, sl[1].start, sl[0].start], dtype=float)  # -> x,y,z
    st["origin"] = tuple(origin + d @ (idx * spacing))


# ------------------------------------------------------------------ context --

class Ctx:
    """Per-process handles (plans, preprocessor, zip catalog).  Built lazily so
    each multiprocessing worker gets its own."""

    _inst: "Ctx | None" = None

    def __init__(self, variant: str, keep_zip: bool = True):
        from nnunetv2.preprocessing.preprocessors.default_preprocessor import DefaultPreprocessor
        from nnunetv2.utilities.plans_handling.plans_handler import PlansManager

        spacing_override, self.do_bodycrop = VARIANTS[variant]
        self.variant = variant
        self.plans_manager = PlansManager(str(PLANS))
        self.cfg = self.plans_manager.get_configuration("3d_fullres")
        if spacing_override is not None:
            self.cfg.configuration["spacing"] = list(spacing_override)
        # 2 real channels; the FG/BG guidance planes are appended as zeros after
        # preprocessing so the array matches the baseline network's 4 inputs.
        base = json.loads(BASE_DATASET_JSON.read_text())
        self.dataset_json = dict(base)
        self.dataset_json["channel_names"] = {"0": "CT", "1": "PET"}
        self.n_out_channels = len(base["channel_names"])
        self.pre = DefaultPreprocessor(verbose=False)
        self.cat = None
        if keep_zip:
            import partial_zip_extract as pz
            try:
                self.cat = pz.open_catalog()
            except Exception:
                self.cat = None

    @classmethod
    def get(cls, variant: str) -> "Ctx":
        if cls._inst is None or cls._inst.variant != variant:
            cls._inst = Ctx(variant)
        return cls._inst


# ------------------------------------------------------------------ one case --

def ensure_raw(case: str, raw_dir: Path, tmp: Path, ctx: Ctx) -> tuple[Path, Path, Path]:
    """Return (ct, pet, label) paths, extracting from the (partial) zip if needed."""
    ct = raw_dir / "images" / f"{case}_0000.nii.gz"
    pet = raw_dir / "images" / f"{case}_0001.nii.gz"
    lab = raw_dir / "labels" / f"{case}.nii.gz"
    if ct.exists() and pet.exists() and lab.exists():
        return ct, pet, lab
    if ctx.cat is None:
        raise FileNotFoundError(f"{case}: not in {raw_dir} and no zip catalog")
    ent = ctx.cat.case_entries(case)
    missing = [k for k, v in ent.items() if not ctx.cat.is_available(v)]
    if missing:
        raise FileNotFoundError(f"{case}: not downloaded yet ({','.join(missing)})")
    d = tmp / case
    (d / "images").mkdir(parents=True, exist_ok=True)
    (d / "labels").mkdir(parents=True, exist_ok=True)
    ctx.cat.extract(ent["ct"], d / "images")
    ctx.cat.extract(ent["pet"], d / "images")
    ctx.cat.extract(ent["label"], d / "labels")
    return (d / "images" / f"{case}_0000.nii.gz",
            d / "images" / f"{case}_0001.nii.gz",
            d / "labels" / f"{case}.nii.gz")


def preprocess_case(case: str, ctx: Ctx, ct: Path, pet: Path, lab: Path) -> tuple[np.ndarray, np.ndarray, dict, dict]:
    from nnunetv2.imageio.simpleitk_reader_writer import SimpleITKIO

    timing = {}
    t = time.time()
    rw = SimpleITKIO()
    data, props = rw.read_images([str(ct), str(pet)])
    seg, _ = rw.read_seg(str(lab))
    timing["read_s"] = time.time() - t
    props["raw_shape"] = list(data.shape[1:])

    pet_stats_full = None
    if ctx.do_bodycrop:
        t = time.time()
        # nnU-Net z-scores PET over the whole volume, so cropping away ~75% of the
        # voxels would shift channel 1 away from what the baseline was trained on.
        # Record the full-volume statistics and undo the shift after preprocessing.
        pet_stats_full = (float(data[1].mean()), float(data[1].std()))
        sl = body_bbox(data[0], seg[0])
        data = data[:, sl[0], sl[1], sl[2]]
        seg = seg[:, sl[0], sl[1], sl[2]]
        props["body_bbox"] = [[s.start, s.stop] for s in sl]
        shift_origin(props, sl)
        pet_stats_crop = (float(data[1].mean()), float(data[1].std()))
        shape_in = tuple(data.shape[1:])
        timing["bodycrop_s"] = time.time() - t
        timing["bodycrop_keep_frac"] = float(np.prod(data.shape[1:]) / np.prod(props["raw_shape"]))

    t = time.time()
    data, seg, props = ctx.pre.run_case_npy(data, seg, props, ctx.plans_manager,
                                            ctx.cfg, ctx.dataset_json)
    timing["preprocess_s"] = time.time() - t

    if pet_stats_full is not None:
        # run_case_npy z-scored the cropped volume; we want the full-volume mean and
        # std. Both are affine in x and resampling is linear, so the correction
        # commutes with the resampling: z' = z*(sd_c/sd_f) + (mu_c - mu_f)/sd_f.
        # Only valid if crop_to_nonzero did not shrink the volume further, hence the
        # shape check.
        mu_f, sd_f = pet_stats_full
        mu_c, sd_c = pet_stats_crop
        if tuple(props["shape_after_cropping_and_before_resampling"]) == shape_in and sd_f > 0:
            data[1] = data[1] * (sd_c / sd_f) + (mu_c - mu_f) / sd_f
            props["pet_norm_correction"] = {"mu_full": mu_f, "sd_full": sd_f,
                                            "mu_crop": mu_c, "sd_crop": sd_c}
            timing["pet_norm_corrected"] = True
        else:
            props["pet_norm_correction"] = "skipped: crop_to_nonzero changed shape"
            timing["pet_norm_corrected"] = False
    return data, seg, props, timing


def to_store_arrays(data: np.ndarray, seg: np.ndarray, ctx: Ctx,
                    dtype: str = "float16") -> tuple[np.ndarray, np.ndarray]:
    """Cast to the storage dtype and pad the guidance channels with zeros."""
    out = np.zeros((ctx.n_out_channels, *data.shape[1:]), dtype=np.dtype(dtype))
    out[: data.shape[0]] = data.astype(dtype, copy=False)
    return out, seg.astype(np.uint8, copy=False)


def write_case(out_dir: Path, case: str, data: np.ndarray, seg: np.ndarray,
               props: dict, fmt: str = "npz", patch_size=(112, 160, 128)) -> int:
    """Write one case in the layout nnUNetDatasetNumpy / Blosc2 reads."""
    from batchgenerators.utilities.file_and_folder_operations import write_pickle

    out_dir.mkdir(parents=True, exist_ok=True)
    stem = str(out_dir / case)
    if fmt == "npz":
        # Write through a file object: np.savez_compressed appends '.npz' only when
        # given a path, and a stale '<case>.part.npz' from a killed run would be read
        # back as a bogus identifier (get_identifiers strips the last 4 characters).
        tmp = stem + ".part"
        with open(tmp, "wb") as fh:
            np.savez_compressed(fh, data=data, seg=seg)
        os.replace(tmp, stem + ".npz")
        n = os.path.getsize(stem + ".npz")
    elif fmt == "b2nd":
        # nnU-Net's own chunk/block sizing: the chunks are tuned to the patch size,
        # which is what makes mmap'd patch reads cheap.
        from nnunetv2.training.dataloading.nnunet_dataset import nnUNetDatasetBlosc2
        from nnunetv2.utilities.default_n_proc_DA import get_allowed_n_proc_DA  # noqa: F401
        from nnunetv2.preprocessing.preprocessors.default_preprocessor import comp_blosc2_params

        for suffix in ("", "_seg"):
            f = stem + suffix + ".b2nd"
            if os.path.exists(f):
                os.remove(f)
        bs_d, cs_d = comp_blosc2_params(data.shape, tuple(patch_size), data.itemsize)
        bs_s, cs_s = comp_blosc2_params(seg.shape, tuple(patch_size), seg.itemsize)
        nnUNetDatasetBlosc2.save_case(data, seg, props, stem,
                                      chunks=cs_d, blocks=bs_d,
                                      chunks_seg=cs_s, blocks_seg=bs_s)
        n = sum(os.path.getsize(stem + s + ".b2nd") for s in ("", "_seg"))
        return n + os.path.getsize(stem + ".pkl")
    else:
        raise ValueError(fmt)
    write_pickle(props, stem + ".pkl")
    return n + os.path.getsize(stem + ".pkl")


# --------------------------------------------------------------- worker glue --

_JOB: dict = {}


def _init(job: dict) -> None:
    _JOB.update(job)


def _run_one(case: str) -> dict:
    rec = {"case": case, "variant": _JOB["variant"], "status": "ok"}
    t0 = time.time()
    tmp = Path(_JOB["tmp"])
    try:
        ctx = Ctx.get(_JOB["variant"])
        ct, pet, lab = ensure_raw(case, Path(_JOB["raw_dir"]), tmp, ctx)
        data, seg, props, timing = preprocess_case(case, ctx, ct, pet, lab)
        rec.update(timing)
        rec["shape"] = list(data.shape[1:])
        rec["megavoxels"] = round(float(np.prod(data.shape[1:])) / 1e6, 2)
        rec["n_lesion_voxels"] = int((seg > 0).sum())
        arr, segu = to_store_arrays(data, seg, ctx, _JOB["dtype"])
        del data, seg
        t = time.time()
        rec["bytes"] = write_case(Path(_JOB["out"]), case, arr, segu, props,
                                  _JOB["fmt"], tuple(ctx.cfg.patch_size))
        rec["write_s"] = time.time() - t
        rec["mb"] = round(rec["bytes"] / 1e6, 2)
    except Exception as e:  # noqa: BLE001
        rec["status"] = "error"
        rec["error"] = f"{type(e).__name__}: {e}"
        rec["traceback"] = traceback.format_exc()[-1500:]
    finally:
        if _JOB.get("cleanup_tmp", True):
            shutil.rmtree(tmp / case, ignore_errors=True)
    rec["total_s"] = round(time.time() - t0, 2)
    return rec


# ---------------------------------------------------------------- manifest --

def manifest_path(store: Path) -> Path:
    """Sibling of the store, never inside it.

    infer_dataset_class asserts that every file in the preprocessed folder shares one
    extension (ignoring .pkl/.npy), so a manifest.json in there breaks training.
    """
    store = Path(store)
    return store.parent / f"{store.name}.manifest.json"


def scaffold(store: Path) -> None:
    """Lay out the parent dir as a valid nnUNet_preprocessed/DatasetXXX.

    nnU-Net wants dataset.json, nnUNetPlans.json and splits_final.json in the dataset
    folder with the cases one level deeper, so point nnUNet_preprocessed at the
    store's grandparent.
    """
    store = Path(store)
    parent = store.parent
    parent.mkdir(parents=True, exist_ok=True)
    for src, dst in ((BASE_DATASET_JSON, parent / "dataset.json"),
                     (PLANS, parent / "nnUNetPlans.json"),
                     (META / "splits_final.json", parent / "splits_final.json"),
                     (META / "dataset_fingerprint.json", parent / "dataset_fingerprint.json")):
        if src.exists() and not dst.exists():
            shutil.copy2(src, dst)


def load_manifest(path: Path) -> dict:
    if path.exists():
        try:
            return json.loads(path.read_text())
        except json.JSONDecodeError:
            print(f"WARNING: unreadable manifest {path}, starting fresh")
    return {"cases": {}}


def save_manifest(path: Path, man: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.part")
    tmp.write_text(json.dumps(man, indent=1))
    os.replace(tmp, path)


def run_pool(cases: list[str], job: dict, workers: int, manifest: Path,
             flush_every: int = 5) -> list[dict]:
    man = load_manifest(manifest)
    recs = []
    todo = []
    for c in cases:
        prev = man["cases"].get(c)
        done = prev and prev.get("status") == "ok"
        if done and job["fmt"] == "npz" and not (Path(job["out"]) / f"{c}.npz").exists():
            done = False  # manifest says ok but the file is gone
        if done:
            recs.append(prev)
        else:
            todo.append(c)
    print(f"{len(cases)} requested, {len(cases) - len(todo)} already done, "
          f"{len(todo)} to do, {workers} workers")
    if not todo:
        return recs

    t0 = time.time()
    with mp.get_context("spawn").Pool(workers, initializer=_init, initargs=(job,)) as pool:
        for i, r in enumerate(pool.imap_unordered(_run_one, todo), 1):
            recs.append(r)
            man["cases"][r["case"]] = r
            flag = "ok " if r["status"] == "ok" else "ERR"
            print(f"[{i}/{len(todo)}] {flag} {r['case']}  "
                  f"{r.get('mb', 0):7.1f} MB  {r.get('total_s', 0):6.1f}s  "
                  f"{r.get('error', '')}", flush=True)
            if i % flush_every == 0:
                man["updated"] = time.time()
                save_manifest(manifest, man)
    man["updated"] = time.time()
    save_manifest(manifest, man)
    ok = [r for r in recs if r["status"] == "ok"]
    if ok:
        wall = time.time() - t0
        print(f"\n{len(ok)} ok, {len(recs) - len(ok)} errors, wall {wall:.0f}s "
              f"({wall / max(len(todo), 1):.1f}s/case at {workers} workers)")
    return recs


# ------------------------------------------------------------- case picking --

def all_cases() -> list[str]:
    return sorted(p.name[: -len(".nii.gz")]
                  for p in (DRIVE / "labelsTr").glob("*.nii.gz"))


def available_cases(raw_dir: Path | None = None) -> list[str]:
    if raw_dir and (raw_dir / "labels").exists():
        return sorted(p.name[: -len(".nii.gz")]
                      for p in (raw_dir / "labels").glob("*.nii.gz"))
    import partial_zip_extract as pz
    return pz.open_catalog().available_cases()["complete"]


def stratified_pick(n: int, cases: list[str], stats: dict, seed: int = 0) -> list[str]:
    """Pick n cases spread over tracer x lesion-count bucket, proportional to the
    source distribution."""
    import random
    rng = random.Random(seed)
    per = stats["per_case"]

    def bucket(c):
        k = per[c]["n_lesions"]
        for name, lo, hi in (("0", 0, 0), ("1", 1, 1), ("2-3", 2, 3),
                             ("4-10", 4, 10), ("11-30", 11, 30), ("31+", 31, 10 ** 9)):
            if lo <= k <= hi:
                return name
        return "31+"

    groups: dict[tuple[str, str], list[str]] = {}
    for c in cases:
        if c in per:
            groups.setdefault((("fdg" if c.startswith("fdg_") else "psma"), bucket(c)), []).append(c)
    total = sum(len(v) for v in groups.values())
    pick, remainder = [], []
    for k, v in sorted(groups.items()):
        v = sorted(v)
        rng.shuffle(v)
        want = n * len(v) / total
        take = int(want)
        pick += v[:take]
        remainder.append((want - take, v[take:]))
    remainder.sort(key=lambda x: -x[0])
    i = 0
    while len(pick) < n and any(v for _, v in remainder):
        _, v = remainder[i % len(remainder)]
        if v:
            pick.append(v.pop(0))
        i += 1
    return sorted(pick)


# --------------------------------------------------------------------- CLI --

def cmd_measure(args) -> None:
    raw = Path(args.raw_dir) if args.raw_dir else None
    cases = available_cases(raw)
    if args.cases_file:
        cases = [l.rstrip("\n") for l in Path(args.cases_file).read_text().splitlines() if l.strip()]
    elif args.cases:
        cases = args.cases
    cases = cases[: args.n]
    print(f"measuring on {len(cases)} cases: "
          f"{sum(c.startswith('fdg_') for c in cases)} FDG / "
          f"{sum(c.startswith('psma_') for c in cases)} PSMA")
    work = Path(args.workdir)
    results = {}
    for variant in (args.variants or list(VARIANTS)):
        for fmt, dtype in args.formats:
            key = f"{variant}/{fmt}/{dtype}"
            out = work / variant / f"{fmt}_{dtype}"
            job = {"variant": variant, "out": str(out), "tmp": str(work / "tmp"),
                   "raw_dir": str(raw) if raw else str(work / "raw"),
                   "dtype": dtype, "fmt": fmt, "cleanup_tmp": False}
            print(f"\n===== {key} =====")
            recs = run_pool(cases, job, args.workers, manifest_path(out))
            results[key] = recs
    summary = summarize_measure(results)
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(json.dumps(
        {"summary": summary, "raw": results}, indent=1))
    print(f"\nwrote {args.report}\n")
    print(measure_table(summary))


def summarize_measure(results: dict) -> dict:
    out = {}
    for key, recs in results.items():
        ok = [r for r in recs if r["status"] == "ok"]
        if not ok:
            out[key] = {"n": 0}
            continue
        by = {}
        for tr in ("fdg", "psma", "all"):
            sel = [r for r in ok if tr == "all" or r["case"].startswith(tr + "_")]
            if not sel:
                continue
            by[tr] = {
                "n": len(sel),
                "mb_mean": round(float(np.mean([r["mb"] for r in sel])), 2),
                "mb_median": round(float(np.median([r["mb"] for r in sel])), 2),
                "megavoxels_mean": round(float(np.mean([r["megavoxels"] for r in sel])), 2),
                "s_preprocess_mean": round(float(np.mean([r["preprocess_s"] for r in sel])), 1),
                "s_write_mean": round(float(np.mean([r["write_s"] for r in sel])), 1),
                "s_total_mean": round(float(np.mean([r["total_s"] for r in sel])), 1),
                "keep_frac_mean": round(float(np.mean(
                    [r["bodycrop_keep_frac"] for r in sel if "bodycrop_keep_frac" in r])), 3)
                if any("bodycrop_keep_frac" in r for r in sel) else None,
            }
        by["projected_gb_1611"] = round(
            (by.get("fdg", {}).get("mb_mean", 0) * 1014 +
             by.get("psma", {}).get("mb_mean", 0) * 597) / 1000, 1) \
            if "fdg" in by and "psma" in by else round(
            by["all"]["mb_mean"] * 1611 / 1000, 1)
        out[key] = by
    return out


def measure_table(summary: dict) -> str:
    L = ["| variant / format | MB/case FDG | MB/case PSMA | Mvox FDG | Mvox PSMA "
         "| s/case | proj. GB (1611) |", "|---|---:|---:|---:|---:|---:|---:|"]
    for key, s in summary.items():
        if not s or s.get("n") == 0:
            continue
        f, p = s.get("fdg", {}), s.get("psma", {})
        L.append(f"| {key} | {f.get('mb_mean', '-')} | {p.get('mb_mean', '-')} | "
                 f"{f.get('megavoxels_mean', '-')} | {p.get('megavoxels_mean', '-')} | "
                 f"{s.get('all', {}).get('s_total_mean', '-')} | "
                 f"{s.get('projected_gb_1611', '-')} |")
    return "\n".join(L)


def project_shape(shape, spacing, target) -> float:
    """Voxel count in millions after resampling shape@spacing to target."""
    return float(np.prod([max(int(round(sh * sp / tg)), 1)
                          for sh, sp, tg in zip(shape, spacing, target)])) / 1e6


def cmd_project(args) -> None:
    """Extrapolate the measured bytes/case to the whole cohort.

    Scales by voxels rather than by case count: the measured cases are not the cohort
    mean (FDG has a long volume tail), so take MB-per-megavoxel from the measurement
    and multiply by the mean megavoxel count from label_stats.json.
    """
    rep = json.loads(Path(args.report).read_text())
    stats = json.loads((META / "label_stats.json").read_text())
    per = stats["per_case"]

    # mean megavoxels per tracer, per variant, over all 1611 cases
    keep = {}
    for key, recs in rep["raw"].items():
        variant = key.split("/")[0]
        for r in recs:
            if r["status"] == "ok" and "bodycrop_keep_frac" in r:
                tr = "fdg" if r["case"].startswith("fdg_") else "psma"
                keep.setdefault((variant, tr), []).append(r["bodycrop_keep_frac"])
    keep = {k: float(np.mean(v)) for k, v in keep.items()}

    mvox = {}
    for variant, (spacing_override, do_crop) in VARIANTS.items():
        target = spacing_override or [3.0, 2.0364201068878174, 2.0364201068878174]
        for tr in ("fdg", "psma"):
            vals = [project_shape(v["shape"], v["spacing"], target)
                    for c, v in per.items()
                    if (c.startswith("fdg_") if tr == "fdg" else c.startswith("psma_"))]
            m = float(np.mean(vals))
            if do_crop:
                m *= keep.get((variant, tr), 1.0)
            mvox[(variant, tr)] = m

    n_cases = {"fdg": sum(1 for c in per if c.startswith("fdg_")),
               "psma": sum(1 for c in per if c.startswith("psma_"))}
    frac = args.fraction

    rows = []
    for key, s in rep["summary"].items():
        if not s or s.get("n") == 0:
            continue
        variant = key.split("/")[0]
        total_gb, detail = 0.0, {}
        for tr in ("fdg", "psma"):
            if tr not in s:
                continue
            mb_per_mvox = s[tr]["mb_mean"] / s[tr]["megavoxels_mean"]
            mb = mb_per_mvox * mvox[(variant, tr)]
            detail[tr] = round(mb, 2)
            total_gb += mb * n_cases[tr] * frac / 1000
        rows.append({"store": key, "mb_per_case_fdg": detail.get("fdg"),
                     "mb_per_case_psma": detail.get("psma"),
                     "mvox_fdg": round(mvox[(variant, "fdg")], 1),
                     "mvox_psma": round(mvox[(variant, "psma")], 1),
                     "gb_total": round(total_gb, 1),
                     "s_per_case_1w": s.get("all", {}).get("s_total_mean"),
                     "hours_at_%dw" % args.workers: round(
                         s.get("all", {}).get("s_total_mean", 0)
                         * (n_cases["fdg"] + n_cases["psma"]) * frac
                         / args.workers / 3600, 2)})
    rows.sort(key=lambda r: r["gb_total"])
    hkey = "hours_at_%dw" % args.workers
    print(f"cohort: {n_cases['fdg']} FDG + {n_cases['psma']} PSMA, "
          f"fraction={frac}\n")
    print(f"| store | Mvox FDG | Mvox PSMA | MB/case FDG | MB/case PSMA | "
          f"total GB | wall h @{args.workers}w |")
    print("|---|---:|---:|---:|---:|---:|---:|")
    for r in rows:
        print(f"| {r['store']} | {r['mvox_fdg']} | {r['mvox_psma']} | "
              f"{r['mb_per_case_fdg']} | {r['mb_per_case_psma']} | "
              f"**{r['gb_total']}** | {r[hkey]} |")
    if args.out:
        Path(args.out).write_text(json.dumps(
            {"fraction": frac, "n_cases": n_cases, "rows": rows}, indent=1))
        print(f"\nwrote {args.out}")


def cmd_status(args) -> None:
    """Print how far a Tier A build has got (used by run_full_build.sh)."""
    man = load_manifest(manifest_path(Path(args.out)))
    ok = [c for c, r in man["cases"].items() if r.get("status") == "ok"]
    err = {c: r.get("error") for c, r in man["cases"].items()
           if r.get("status") != "ok"}
    gb = sum(r.get("bytes", 0) for r in man["cases"].values()
             if r.get("status") == "ok") / 1e9
    if args.quiet:
        print(len(ok))
        return
    print(json.dumps({"ok": len(ok), "errors": len(err), "gb": round(gb, 2),
                      "error_cases": dict(list(err.items())[:10])}, indent=1))


def cmd_build(args) -> None:
    if args.cases_file:
        cases = [l.strip() for l in Path(args.cases_file).read_text().splitlines() if l.strip()]
    elif args.cases:
        cases = args.cases
    else:
        cases = available_cases(Path(args.raw_dir) if args.raw_dir else None)
    if args.limit:
        cases = cases[: args.limit]
    out = Path(args.out)
    job = {"variant": args.variant, "out": str(out), "tmp": args.tmp,
           "raw_dir": args.raw_dir or args.tmp, "dtype": args.dtype,
           "fmt": args.fmt, "cleanup_tmp": not args.keep_tmp}
    scaffold(out)
    run_pool(cases, job, args.workers, manifest_path(out))


def cmd_evalset(args) -> None:
    """Tier B: stratified held-out raw nii.gz from the fold-0 validation split."""
    stats = json.loads((META / "label_stats.json").read_text())
    splits = json.loads((META / "splits_final.json").read_text())
    val = splits[args.fold]["val"]
    pick = stratified_pick(args.n, val, stats, seed=args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "cases.txt").write_text("\n".join(pick) + "\n")
    comp = {}
    for c in pick:
        tr = "fdg" if c.startswith("fdg_") else "psma"
        k = f"{tr}_{'neg' if stats['per_case'][c]['n_lesions'] == 0 else 'pos'}"
        comp[k] = comp.get(k, 0) + 1
    print(f"picked {len(pick)} cases: {comp}")
    (out / "composition.json").write_text(json.dumps(comp, indent=1))
    if args.dry_run:
        print(f"wrote {out}/cases.txt (dry run, no image copy)")
        return

    import partial_zip_extract as pz
    cat = pz.open_catalog()
    (out / "imagesTr").mkdir(exist_ok=True)
    (out / "labelsTr").mkdir(exist_ok=True)
    total = 0
    for i, c in enumerate(pick, 1):
        ent = cat.case_entries(c)
        if not all(cat.is_available(v) for v in ent.values()):
            print(f"  [{i}] SKIP not downloaded: {c}")
            continue
        if (out / "imagesTr" / f"{c}_0001.nii.gz").exists():
            continue
        cat.extract(ent["ct"], out / "imagesTr")
        cat.extract(ent["pet"], out / "imagesTr")
        cat.extract(ent["label"], out / "labelsTr")
        total += sum(ent[k].file_size for k in ent)
        print(f"  [{i}/{len(pick)}] {c}  cumulative {total / 1e9:.1f} GB", flush=True)
    print(f"Tier B total {total / 1e9:.1f} GB in {out}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p):
        p.add_argument("--workers", type=int, default=6,
                       help="processes; peak ~3 GB each on FDG, 52 GB RAM total")
        p.add_argument("--raw-dir", default=None,
                       help="dir with images/ + labels/; else stream from the zip")
        p.add_argument("--cases", nargs="*", default=None)
        p.add_argument("--cases-file", default=None,
                       help="newline-separated case ids (FDG names contain spaces)")

    p = sub.add_parser("measure")
    common(p)
    p.add_argument("--n", type=int, default=10)
    p.add_argument("--workdir", default="/content/work/measure")
    p.add_argument("--report", default="/content/work/measure/report.json")
    p.add_argument("--variants", nargs="*", default=None)
    p.add_argument("--formats", nargs="*", default=[("npz", "float16")],
                   type=lambda s: tuple(s.split(":")),
                   help="fmt:dtype pairs, e.g. npz:float16 npz:float32 b2nd:float16")
    p.set_defaults(fn=cmd_measure)

    p = sub.add_parser("build")
    common(p)
    p.add_argument("--variant", default="bodycrop", choices=list(VARIANTS))
    p.add_argument("--out", required=True)
    p.add_argument("--tmp", default="/content/work/tmp")
    p.add_argument("--dtype", default="float16")
    p.add_argument("--fmt", default="npz", choices=["npz", "b2nd"])
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--keep-tmp", action="store_true")
    p.set_defaults(fn=cmd_build)

    p = sub.add_parser("status")
    p.add_argument("--out", required=True)
    p.add_argument("--quiet", action="store_true")
    p.set_defaults(fn=cmd_status)

    p = sub.add_parser("project")
    p.add_argument("--report", default="/content/work/measure/report.json")
    p.add_argument("--fraction", type=float, default=1.0,
                   help="fraction of the cohort to store (e.g. 0.5 for a subset)")
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--out", default=None)
    p.set_defaults(fn=cmd_project)

    p = sub.add_parser("evalset")
    p.add_argument("--n", type=int, default=100)
    p.add_argument("--fold", type=int, default=0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="/content/drive/MyDrive/autoPET/evalset")
    p.add_argument("--dry-run", action="store_true")
    p.set_defaults(fn=cmd_evalset)

    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
