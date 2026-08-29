#!/usr/bin/env python3
"""Append one line per finished epoch to progress.txt and mirror it to Drive.

Parses the nnU-Net training logs and emits `epoch lr train_loss val_loss
pseudo_dice epoch_time_s finished_at` plus a rolling summary comment, then copies
progress.txt and the newest log into the Drive checkpoint folder. Loops every
INTERVAL seconds. Restart-safe: re-reads progress.txt on start and never raises
because of a slow or missing Drive mount.
"""
import glob
import os
import re
import shutil
import sys
import time

# Which run to watch. A concurrently trained variant sets MODEL_NAME plus its own
# PROGRESS_FILE, so two watchers never append to the same progress.txt.
MODEL_NAME = os.environ.get(
    "MODEL_NAME", "nnUNetTrainer_Interactive__nnUNetPlans_interactive__3d_fullres")
MODEL_DIR = os.environ.get(
    "MODEL_DIR", "/content/nnUNet/nnUNet_results/Dataset998_AutoPETV/" + MODEL_NAME)
DRIVE_MODEL = os.environ.get(
    "DRIVE_MODEL", "/content/drive/MyDrive/autoPET/ckpt/Dataset998_AutoPETV/" + MODEL_NAME)
OUT_DIR = os.path.join(MODEL_DIR, "fold_0")
DRIVE = os.path.join(DRIVE_MODEL, "fold_0")
# nnUNetPredictor.initialize_from_trained_model_folder() reads these from the
# model folder; without them the Drive copy is not an evaluation-ready folder
MODEL_JSONS = ("plans.json", "dataset.json", "dataset_fingerprint.json")
PROG = os.environ.get("PROGRESS_FILE", "/content/work/train/progress.txt")
TOTAL = int(os.environ.get("TOTAL_EPOCHS", "200"))
INTERVAL = int(os.environ.get("INTERVAL", "300"))

RE_EPOCH = re.compile(r"^(\S+ \S+?): Epoch (\d+)\s*$")
RE_LR = re.compile(r": Current learning rate: ([0-9.eE+-]+)")
RE_TR = re.compile(r": train_loss (-?[0-9.eE+-]+)")
RE_VL = re.compile(r": val_loss (-?[0-9.eE+-]+)")
RE_PD = re.compile(r": Pseudo dice \[(.*)\]")
RE_ET = re.compile(r"^(\S+ \S+?): Epoch time: ([0-9.]+) s")
RE_NUM = re.compile(r"([0-9.]+)\)")


def parse():
    """Return the finished epochs, oldest first, one dict each."""
    rows = {}
    cur = None
    for path in sorted(glob.glob(os.path.join(OUT_DIR, "training_log_*.txt"))):
        try:
            lines = open(path, errors="replace").read().splitlines()
        except OSError:
            continue
        for ln in lines:
            m = RE_EPOCH.search(ln)
            if m:
                cur = {"epoch": int(m.group(2)), "lr": "-", "tr": "-",
                       "vl": "-", "pd": "-", "et": "-", "at": "-"}
                continue
            if cur is None:
                continue
            m = RE_LR.search(ln)
            if m:
                cur["lr"] = m.group(1)
            m = RE_TR.search(ln)
            if m:
                cur["tr"] = m.group(1)
            m = RE_VL.search(ln)
            if m:
                cur["vl"] = m.group(1)
            m = RE_PD.search(ln)
            if m:
                cur["pd"] = ",".join(RE_NUM.findall(m.group(1))) or m.group(1)
            m = RE_ET.search(ln)
            if m:
                cur["at"] = m.group(1)
                cur["et"] = m.group(2)
                # a resumed run replays an epoch number: the newest wins
                rows[cur["epoch"]] = cur
                cur = None
    return [rows[k] for k in sorted(rows)]


def main():
    seen = set()
    if os.path.isfile(PROG):
        for ln in open(PROG, errors="replace"):
            head = ln.split()
            if head and head[0].isdigit():
                seen.add(int(head[0]))
    else:
        os.makedirs(os.path.dirname(PROG), exist_ok=True)
        with open(PROG, "w") as fh:
            fh.write("# epoch  lr        train_loss  val_loss   pseudo_dice  "
                     "epoch_time_s  finished_at(UTC)\n")

    while True:
        rows = parse()
        new = [r for r in rows if r["epoch"] not in seen]
        if new:
            with open(PROG, "a") as fh:
                for r in new:
                    fh.write("%-7d %-9s %-11s %-10s %-12s %-13s %s\n" % (
                        r["epoch"], r["lr"], r["tr"], r["vl"], r["pd"],
                        r["et"], r["at"]))
                    seen.add(r["epoch"])
                times = [float(x["et"]) for x in rows[-10:] if x["et"] != "-"]
                if times:
                    mean = sum(times) / len(times)
                    eta_h = (TOTAL - len(seen)) * mean / 3600.0
                    fh.write("# %s  done %d/%d  last10_mean_epoch %.1f s  ETA %.2f h\n" % (
                        time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                        len(seen), TOTAL, mean, eta_h))
        try:
            os.makedirs(DRIVE, exist_ok=True)
            # keep the Drive copy a complete nnU-Net model folder at all times:
            # plans.json + dataset.json next to fold_0/, or --model_folder fails
            for name in MODEL_JSONS:
                src = os.path.join(MODEL_DIR, name)
                dst = os.path.join(DRIVE_MODEL, name)
                if os.path.isfile(src) and (not os.path.isfile(dst)
                                            or os.path.getsize(src) != os.path.getsize(dst)):
                    shutil.copy2(src, dst)
                    print("[info] restored %s on Drive" % name, flush=True)
            shutil.copy2(PROG, os.path.join(DRIVE, "progress.txt"))
            for extra in ("debug.json", "progress.png"):
                p = os.path.join(OUT_DIR, extra)
                if os.path.isfile(p):
                    shutil.copy2(p, os.path.join(DRIVE, extra))
            logs = sorted(glob.glob(os.path.join(OUT_DIR, "training_log_*.txt")),
                          key=os.path.getmtime)
            if logs:
                shutil.copy2(logs[-1], os.path.join(DRIVE, os.path.basename(logs[-1])))
        except Exception as exc:  # Drive can be slow or wedged; never die for it
            print("[warn] drive mirror failed: %s" % exc, file=sys.stderr, flush=True)
        time.sleep(INTERVAL)


if __name__ == "__main__":
    main()
