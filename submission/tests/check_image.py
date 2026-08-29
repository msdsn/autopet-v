"""Build-time self-check for the submission image (no GPU, no network, no data).

Runs as the last `RUN` of the Dockerfile so that a broken image fails the build rather
than the first 20-minute evaluation job.  Also runnable from a checkout:

    AUTOPETV_MODEL_FOLDER=/content/work/ft_model python -m submission.tests.check_image
"""

from __future__ import annotations

import json
import os
import sys


def main() -> int:
    ok = True

    def check(label: str, condition: bool, detail: str = "", fail_detail: str = "") -> None:
        """`detail` is printed either way; `fail_detail` only when the check fails."""
        nonlocal ok
        extra = detail or (fail_detail if not condition else "")
        if not condition and detail and fail_detail:
            extra = f"{detail}; {fail_detail}"
        print(f"[{'PASS' if condition else 'FAIL'}] {label}{(' -- ' + extra) if extra else ''}")
        if not condition:
            ok = False

    # 1. imports -----------------------------------------------------------
    import submission.process as proc
    from submission.predictor_gc import (
        KNOWN_PREDICTORS,
        add_src_to_path,
        predictor_config,
        register_external_trainer,
    )

    print(f"version: {proc.VERSION}")
    cfg = predictor_config()
    print(f"config: {json.dumps(cfg, default=str, sort_keys=True)}")
    check("entry point imports", True)

    # 2. the selected predictor exists -------------------------------------
    check("AUTOPETV_PREDICTOR is known", cfg["predictor"] in KNOWN_PREDICTORS,
          repr(cfg["predictor"]), f"known: {KNOWN_PREDICTORS}")

    add_src_to_path()
    interactive = cfg["predictor"] in ("interactive", "interactive_postproc")
    postproc = cfg["predictor"] in ("postproc", "interactive_postproc")

    # 3. the shipped post-processing config --------------------------------
    if postproc:
        from postproc import PostProcConfig

        path = cfg["postproc_config"]
        check("postproc config file present", bool(path) and os.path.isfile(path), str(path))
        if path and os.path.isfile(path):
            raw = json.load(open(path))
            pp = PostProcConfig.from_dict({k: v for k, v in raw.items() if not k.startswith("_")})
            print(f"  gate={pp.negative_gate.enabled} max_prob={pp.negative_gate.max_prob} "
                  f"cleanup={pp.enable_cleanup}/{pp.cleanup.suv_floor_mode} "
                  f"monotone={pp.monotone.mode} pass_cached_prev_pred={pp.pass_cached_prev_pred}")
            check("postproc config parses", True)
            if interactive:
                # without this the previous FINAL mask never reaches channel 4 in the
                # container, where nothing can be passed in memory
                check("pass_cached_prev_pred is on for the interactive model",
                      pp.pass_cached_prev_pred)

    # 4. the model folder ---------------------------------------------------
    mf = cfg["model_folder"]
    check("AUTOPETV_MODEL_FOLDER is set", bool(mf), str(mf))
    if not mf:
        return 0 if ok else 1
    dj_path = os.path.join(mf, "dataset.json")
    pl_path = os.path.join(mf, "plans.json")
    ck_path = os.path.join(mf, "fold_0", cfg["checkpoint"])
    for p in (dj_path, pl_path, ck_path):
        check(f"exists: {os.path.relpath(p, mf)}", os.path.isfile(p), p)
    if not all(os.path.isfile(p) for p in (dj_path, pl_path, ck_path)):
        return 1

    dj = json.load(open(dj_path))
    pl = json.load(open(pl_path))
    ns = pl["configurations"]["3d_fullres"]["normalization_schemes"]
    n_ch = len(dj.get("channel_names", {}))
    size = os.path.getsize(ck_path)
    print(f"  plans_name={pl.get('plans_name')} channels={n_ch} norm={ns} "
          f"checkpoint={size} B")
    check("checkpoint is a real file, not an HTML error page", size > 100_000_000,
          f"{size} B")
    if interactive:
        check("dataset.json declares 5 channels", n_ch == 5, str(dj.get("channel_names")))
        check("plans mark channels 2-4 NoNormalization",
              list(ns[2:]) == ["NoNormalization"] * 3, str(ns))
        check("use_mask_for_norm is all False",
              not any(pl["configurations"]["3d_fullres"]["use_mask_for_norm"]))
    else:
        check("dataset.json declares 4 channels", n_ch == 4, str(dj.get("channel_names")))

    # 5. the trainer class the checkpoint names -----------------------------
    if interactive:
        ext = register_external_trainer()
        print(f"  nnUNet_extTrainer={ext}")
        check("nnUNet_extTrainer points at an existing dir",
              bool(ext) and any(os.path.isdir(p) for p in ext.split(os.pathsep)), str(ext))
        # The checkpoint names one trainer class and nnU-Net has to import it to rebuild
        # the network.  Which one it names depends on the model, so resolve every trainer
        # class the repo declares rather than hard-coding one.
        import re as _re

        train_dir0 = (ext or "").split(os.pathsep)[0]
        declared = []
        if train_dir0 and os.path.isdir(train_dir0):
            for fn in sorted(os.listdir(train_dir0)):
                if fn.endswith(".py"):
                    with open(os.path.join(train_dir0, fn)) as fh:
                        declared += _re.findall(r"^class\s+(nnUNetTrainer_\w+)", fh.read(), _re.M)
        check("src/train declares at least one trainer class", bool(declared),
              f"{len(declared)} found")
        try:
            from nnunetv2.utilities.find_objects import recursive_find_trainer_class_by_name

            unresolved = []
            for name in declared:
                try:
                    if recursive_find_trainer_class_by_name(name) is None:
                        unresolved.append(name)
                except Exception as exc:
                    unresolved.append(f"{name} ({exc!r})")
            check("nnU-Net resolves every trainer class in src/train", not unresolved,
                  f"{len(declared)} class(es)", "unresolved: " + ", ".join(unresolved))
        except Exception as exc:  # pragma: no cover
            check("nnU-Net resolves every trainer class in src/train", False, repr(exc))

        # `recursive_find_trainer_class_by_name` imports every module in that folder until
        # it finds the class, so all of them must be import-safe inside the image (no GPU,
        # no dataset, no official-challenge checkout).  Alphabetical order may stop the
        # search early, so check them all here.
        import importlib

        train_dir = (ext or "").split(os.pathsep)[0]
        if train_dir and train_dir not in sys.path:
            sys.path.insert(0, train_dir)
        bad = []
        for fn in sorted(os.listdir(train_dir)):
            if not fn.endswith(".py") or fn.startswith("_"):
                continue
            try:
                importlib.import_module(fn[:-3])
            except Exception as exc:
                bad.append(f"{fn}: {exc!r}")
        check("every module in src/train imports cleanly", not bad, "", "; ".join(bad))

    print("\nRESULT:", "IMAGE SELF-CHECK PASSED" if ok else "FAILURES ABOVE")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
