"""Predictor construction for the container.

Method code lives in `src/`; this module only decides which `src.predictor.Predictor`
to instantiate and how to wrap it.  Grand Challenge gives no CLI, so configuration is
by AUTOPETV_* environment variables -- `predictor_config` resolves and lists them all.
"""

from __future__ import annotations

import json
import os
import sys
from typing import Optional, Tuple

__all__ = [
    "build_predictor",
    "predictor_config",
    "add_src_to_path",
    "base_of",
    "DEFAULT_POSTPROC_CONFIG",
    "KNOWN_PREDICTORS",
]

KNOWN_PREDICTORS = (
    "ensemble_postproc",
    "ensemble",
    "interactive_postproc",
    "interactive",
    "postproc",
    "baseline",
    "baseline_nnunet",
    "threshold",
)

#: shipped in the repo so the image needs no external config file
DEFAULT_POSTPROC_CONFIG = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "postproc_config.json"
)


def add_src_to_path() -> None:
    """Make `src/` importable both inside the image and from a repo checkout."""
    here = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.dirname(here)
    for cand in (os.path.join(repo_root, "src"), "/opt/algorithm/src"):
        if os.path.isdir(cand) and cand not in sys.path:
            sys.path.insert(0, cand)


def register_external_trainer() -> Optional[str]:
    """Put `src/train` on `nnUNet_extTrainer` so `nnUNetTrainer_Interactive` is findable.

    nnU-Net rebuilds the network through the trainer class named in the checkpoint, and
    ours lives in this repo rather than inside nnunetv2.
    """
    add_src_to_path()
    for cand in (
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src", "train"),
        "/opt/algorithm/src/train",
    ):
        if os.path.isdir(cand):
            cur = os.environ.get("nnUNet_extTrainer", "")
            paths = [p for p in cur.split(os.pathsep) if p.strip()]
            if cand not in paths:
                paths.insert(0, cand)
            os.environ["nnUNet_extTrainer"] = os.pathsep.join(paths)
            return os.environ["nnUNet_extTrainer"]
    return None


def _env_bool(name: str, default: bool = False) -> bool:
    v = os.environ.get(name)
    if v is None:
        return default
    return v.strip().lower() in ("1", "true", "yes", "on")


def _env_axes(name: str) -> Optional[Tuple[int, ...]]:
    """"0,1,2" -> (0, 1, 2); "" / unset -> None (leave the checkpoint's axes alone)."""
    v = (os.environ.get(name) or "").replace(" ", "")
    if not v:
        return None
    return tuple(int(a) for a in v.split(",") if a != "")


def _guidance_radius() -> float:
    """Resolve the guidance radius the way the trainer does: this variable, else
    `nnUNet_interactive_radius`, else 10.0 voxels of the preprocessed grid."""
    v = os.environ.get("AUTOPETV_GUIDANCE_RADIUS") or os.environ.get(
        "nnUNet_interactive_radius"
    )
    return float(v) if v else 10.0


def _env_list(name: str) -> list:
    """Comma-separated environment list; empty entries dropped.

    A comma is the separator rather than `os.pathsep` because an ensemble member spec
    is `<folder>[:<checkpoint>[:<weight>]]` and already uses the colon.
    """
    raw = os.environ.get(name, "")
    return [item.strip() for item in raw.split(",") if item.strip()]


def predictor_config() -> dict:
    """The full resolved configuration; logged verbatim so a run can be replayed."""
    folds = os.environ.get("AUTOPETV_FOLDS", "0")
    pp = os.environ.get("AUTOPETV_POSTPROC_CONFIG")
    if not pp and os.path.isfile(DEFAULT_POSTPROC_CONFIG):
        pp = DEFAULT_POSTPROC_CONFIG
    return {
        "predictor": os.environ.get("AUTOPETV_PREDICTOR", "interactive_postproc").strip().lower(),
        "model_folder": os.environ.get("AUTOPETV_MODEL_FOLDER") or None,
        "folds": tuple(int(f) for f in folds.replace(" ", "").split(",") if f != ""),
        "checkpoint": os.environ.get("AUTOPETV_CHECKPOINT", "checkpoint_final.pth"),
        "device": os.environ.get("AUTOPETV_DEVICE", "cuda"),
        "enable_tta": _env_bool("AUTOPETV_ENABLE_TTA", False),
        "mirror_axes": _env_axes("AUTOPETV_MIRROR_AXES"),
        "tile_step_size": float(os.environ.get("AUTOPETV_TILE_STEP", "0.5")),
        "npp": int(os.environ.get("AUTOPETV_NPP", "1")),
        "nps": int(os.environ.get("AUTOPETV_NPS", "1")),
        "tmp_dir": os.environ.get("AUTOPETV_TMP_DIR") or None,
        "ensemble_members": _env_list("AUTOPETV_ENSEMBLE_MEMBERS"),
        "ensemble_weights": [float(x) for x in _env_list("AUTOPETV_ENSEMBLE_WEIGHTS")] or None,
        "postproc_config": pp or None,
        "guidance_radius": _guidance_radius(),
        "interactive_state_dir": _env_bool("AUTOPETV_INTERACTIVE_STATE_DIR", True),
        "cudnn_deterministic": _env_bool("AUTOPETV_CUDNN_DETERMINISTIC", True),
        "verbose_nnunet": _env_bool("AUTOPETV_VERBOSE_NNUNET", False),
    }


def load_postproc_config(path: Optional[str], logger=None) -> Optional[dict]:
    if not path:
        return None
    with open(path) as f:
        cfg = json.load(f)
    # a description/comment key is convenient in a shipped JSON file but is not a
    # PostProcConfig field, and PostProcConfig.from_dict raises on unknown keys
    cfg = {k: v for k, v in cfg.items() if not k.startswith("_")}
    if logger is not None:
        logger.info("[predictor] postproc config from %s: %s", path, json.dumps(cfg, sort_keys=True))
    return cfg


def base_of(predictor):
    """The model underneath any wrapper (`PostProcPredictor.base`)."""
    obj = predictor
    for _ in range(8):
        nxt = getattr(obj, "base", None)
        if nxt is None:
            return obj
        obj = nxt
    return obj


def _wrap_postproc(base, cfg: dict, logger=None):
    """Wrap `base` in `PostProcPredictor`.

    Fail-soft: if the package or its config is broken the container still produces the
    base prediction rather than crashing.
    """
    try:
        from postproc import PostProcPredictor

        return PostProcPredictor(base, load_postproc_config(cfg["postproc_config"], logger))
    except Exception:
        import traceback

        msg = (f"AUTOPETV_PREDICTOR={cfg['predictor']} requested but PostProcPredictor could "
               f"not be built -- falling back to the bare base model")
        if logger is not None:
            logger.error(msg)
            logger.error(traceback.format_exc())
        else:  # pragma: no cover
            print(msg, file=sys.stderr)
            traceback.print_exc()
        return base


def build_predictor(cfg: dict, logger=None):
    """Instantiate the predictor described by `cfg` (from `predictor_config`)."""
    add_src_to_path()
    kind = cfg["predictor"]
    if kind not in KNOWN_PREDICTORS:
        raise ValueError(f"unknown AUTOPETV_PREDICTOR={kind!r}; known: {KNOWN_PREDICTORS}")

    if kind == "threshold":  # smoke tests only, no GPU / no weights needed
        from predictor import ThresholdPredictor

        return ThresholdPredictor(threshold=float(os.environ.get("AUTOPETV_THRESHOLD", "2.5")))

    common = dict(
        model_folder=cfg["model_folder"],
        folds=cfg["folds"],
        checkpoint_name=cfg["checkpoint"],
        device=cfg["device"],
        disable_tta=not cfg["enable_tta"],
        tile_step_size=cfg["tile_step_size"],
        num_processes_preprocessing=cfg["npp"],
        num_processes_segmentation_export=cfg["nps"],
        verbose=cfg["verbose_nnunet"],
        tmp_root=cfg["tmp_dir"],
    )

    if kind in ("ensemble", "ensemble_postproc"):
        register_external_trainer()
        from ensemble_predictor import build_ensemble

        members = cfg["ensemble_members"]
        if not members:
            raise ValueError(f"AUTOPETV_PREDICTOR={kind} needs AUTOPETV_ENSEMBLE_MEMBERS")
        for spec in members:
            folder = spec.split(":")[0]
            if not os.path.isdir(folder):
                raise FileNotFoundError(f"ensemble member folder not found: {folder}")
        # every member gets the same inference settings; `model_folder`, `folds` and the
        # checkpoint name are per member, so they are not part of the shared block
        shared = {k: v for k, v in common.items()
                  if k not in ("model_folder", "checkpoint_name")}
        base = build_ensemble(
            members, cfg["ensemble_weights"],
            guidance_radius=cfg["guidance_radius"],
            use_state_dir=cfg["interactive_state_dir"],
            deterministic=cfg["cudnn_deterministic"],
            force_mirror_axes=cfg["mirror_axes"],
            **shared,
        )
        if logger is not None:
            logger.info("[predictor] ensemble of %d members: %s, weights %s",
                        len(base.members), base.member_labels,
                        [round(w, 4) for w in base.weights])
        if kind == "ensemble":
            return base
        return _wrap_postproc(base, cfg, logger)

    if kind in ("interactive", "interactive_postproc"):
        register_external_trainer()
        from predictor import InteractiveNNUNetPredictor

        if common["model_folder"] is None:
            # let the class pick its own (interactive) default rather than inheriting
            # BaselineNNUNetPredictor's, which points at the 4-channel baseline
            common.pop("model_folder")
        base = InteractiveNNUNetPredictor(
            guidance_radius=cfg["guidance_radius"],
            use_state_dir=cfg["interactive_state_dir"],
            deterministic=cfg["cudnn_deterministic"],
            force_mirror_axes=cfg["mirror_axes"],
            **common,
        )
        if kind == "interactive":
            return base
        return _wrap_postproc(base, cfg, logger)

    from predictor import BaselineNNUNetPredictor

    base = BaselineNNUNetPredictor(**common)
    if kind in ("baseline", "baseline_nnunet"):
        return base
    # kind == "postproc": the v0.1 seam, kept working
    return _wrap_postproc(base, cfg, logger)
