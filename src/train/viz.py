"""Dump PNG sanity-check figures of what the training pipeline feeds the network.

Builds the real trainer, pulls a few batches, and draws the axial slice with the
most guidance: PET + label + previous prediction, the error map the scribbles were
drawn from (FP red / FN blue), then channels 2, 3 and 4.

    python -m train.viz --dataset Dataset999_InteractiveSynth --out <dir>
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np


def _load(dataset_name: str, plans_name: str):
    from batchgenerators.utilities.file_and_folder_operations import join, load_json
    from nnunetv2.paths import nnUNet_preprocessed
    base = join(nnUNet_preprocessed.get() if hasattr(nnUNet_preprocessed, "get")
                else str(nnUNet_preprocessed), dataset_name)
    plans = load_json(join(base, plans_name + ".json"))
    dataset_json = load_json(join(base, "dataset.json"))
    return plans, dataset_json


def build_trainer(dataset_name: str, plans_name: str, config: str, fold: int,
                  trainer_name: str = "nnUNetTrainer_Interactive", device: str = "cpu"):
    import torch
    from nnunetv2.utilities.find_objects import recursive_find_trainer_class_by_name
    plans, dataset_json = _load(dataset_name, plans_name)
    plans = dict(plans)
    plans["continue_training"] = False  # nnU-Net >= 2.8 pops this in __init__
    cls = recursive_find_trainer_class_by_name(trainer_name)
    return cls(plans, config, fold, dataset_json, device=torch.device(device))


def draw(batch, out_dir: str, tag: str, slice_axis: int = 0, max_samples: int = 2):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(out_dir, exist_ok=True)
    data = batch["data"].numpy() if hasattr(batch["data"], "numpy") else np.asarray(batch["data"])
    target = batch["target"]
    if isinstance(target, list):
        target = target[0]
    target = target.numpy() if hasattr(target, "numpy") else np.asarray(target)

    written = []
    for b in range(min(max_samples, data.shape[0])):
        ct, pet = data[b, 0], data[b, 1]
        fg, bg, prev = data[b, 2], data[b, 3], data[b, 4]
        lab = (target[b, 0] > 0)

        # pick the slice with the most guidance; fall back to the biggest label slice
        axes = tuple(i for i in range(3) if i != slice_axis)
        score = (fg + bg).sum(axis=axes)
        if score.max() <= 0:
            score = lab.sum(axis=axes)
        if score.max() <= 0:
            score = prev.sum(axis=axes)
        z = int(np.argmax(score))

        take = [slice(None)] * 3
        take[slice_axis] = z
        take = tuple(take)

        pet_s, lab_s = pet[take], lab[take]
        prev_s = prev[take] > 0.5
        fg_s, bg_s = fg[take], bg[take]
        fp = prev_s & ~lab_s
        fn = ~prev_s & lab_s

        fig, ax = plt.subplots(1, 5, figsize=(19, 4.2))
        fig.suptitle(f"{tag} sample {b} | axial slice {z} | "
                     f"label={int(lab.sum())} prev={int((prev > .5).sum())} "
                     f"fg_guid={int((fg > 0).sum())} bg_guid={int((bg > 0).sum())}",
                     fontsize=10)

        ax[0].imshow(pet_s, cmap="gray")
        ax[0].contour(lab_s.astype(float), levels=[0.5], colors="lime", linewidths=1.2)
        ax[0].contour(prev_s.astype(float), levels=[0.5], colors="orange", linewidths=1.0)
        ax[0].set_title("PET + GT(green) + prev-pred(orange)", fontsize=9)

        # the scribble voxels themselves: guidance == 1 exactly at a stroke voxel
        fg_seed = np.argwhere(fg_s >= 0.999)
        bg_seed = np.argwhere(bg_s >= 0.999)

        err = np.zeros((*pet_s.shape, 3), dtype=np.float32)
        err[..., 0] = fp          # false positive -> red
        err[..., 2] = fn          # false negative -> blue
        ax[1].imshow(pet_s, cmap="gray")
        ax[1].imshow(err, alpha=0.6)
        if len(fg_seed):
            ax[1].scatter(fg_seed[:, 1], fg_seed[:, 0], s=10, c="cyan", marker="s",
                          label="tumor scribble")
        if len(bg_seed):
            ax[1].scatter(bg_seed[:, 1], bg_seed[:, 0], s=10, c="yellow", marker="s",
                          label="bg scribble")
        if len(fg_seed) or len(bg_seed):
            ax[1].legend(fontsize=7, loc="lower right", framealpha=.6)
        ax[1].set_title("error map: FP red / FN blue + scribble voxels", fontsize=9)

        for a, m, t, cm, seed, sc in (
                (ax[2], fg_s, "ch2 tumor guidance (clipped EDT)", "hot", fg_seed, "cyan"),
                (ax[3], bg_s, "ch3 background guidance", "hot", bg_seed, "yellow"),
                (ax[4], prev[take].astype(float), "ch4 previous prediction", "cool", None, None)):
            a.imshow(pet_s, cmap="gray")
            a.imshow(np.ma.masked_where(m <= 0.01, m), cmap=cm, alpha=0.85, vmin=0, vmax=1)
            a.contour(lab_s.astype(float), levels=[0.5], colors="lime", linewidths=0.8)
            if seed is not None and len(seed):
                a.scatter(seed[:, 1], seed[:, 0], s=6, c=sc, marker="s")
            a.set_title(t, fontsize=9)

        for a in ax:
            a.set_xticks([]); a.set_yticks([])
        fig.tight_layout()
        p = os.path.join(out_dir, f"{tag}_s{b}.png")
        fig.savefig(p, dpi=110, bbox_inches="tight")
        plt.close(fig)
        written.append(p)
        print("wrote", p)
    return written


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="Dataset999_InteractiveSynth")
    ap.add_argument("--plans", default="nnUNetPlans_interactive")
    ap.add_argument("--config", default="3d_fullres")
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--trainer", default="nnUNetTrainer_Interactive")
    ap.add_argument("--out", default="/content/work/train/viz")
    ap.add_argument("--n-batches", type=int, default=4)
    ap.add_argument("--max-samples", type=int, default=2)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    tr = build_trainer(args.dataset, args.plans, args.config, args.fold, args.trainer, args.device)
    tr.initialize()
    dl_tr, _ = tr.get_dataloaders()
    stats = []
    for i in range(args.n_batches):
        batch = next(dl_tr)
        d = batch["data"]
        assert d.shape[1] == 5, f"expected 5 channels, got {d.shape}"
        stats.append({
            "batch": i,
            "shape": list(d.shape),
            "ch2_max": float(d[:, 2].max()), "ch2_nonzero": int((d[:, 2] > 0).sum()),
            "ch3_max": float(d[:, 3].max()), "ch3_nonzero": int((d[:, 3] > 0).sum()),
            "ch4_sum": float(d[:, 4].sum()),
            "label_sum": float((batch["target"][0] if isinstance(batch["target"], list)
                                else batch["target"]).sum()),
        })
        draw(batch, args.out, f"batch{i}", slice_axis=tr.SLICE_AXIS, max_samples=args.max_samples)
    with open(os.path.join(args.out, "stats.json"), "w") as f:
        json.dump(stats, f, indent=2)
    print(json.dumps(stats, indent=2))
    try:
        dl_tr._finish()
    except Exception:
        pass


if __name__ == "__main__":
    main()
