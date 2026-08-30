#!/usr/bin/env python3
"""Docker-free re-implementation of the official autoPET V evaluation loop.

Scribble simulation and metrics are imported from the challenge repo (`--repo`) rather
than reimplemented; the model is loaded once and stays in process instead of going
through docker per iteration.  Six iterations per case -- one without scribbles plus
five corrections -- and AUC is the trapezoid over them, maximum 5.0.  Departures from
the published loop are behind flags and described in docs/eval_harness.md.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import sys
import time
import traceback
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import nibabel as nib

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from predictor import (  # noqa: E402
    Predictor, ThresholdPredictor, BaselineNNUNetPredictor, FastBaselineNNUNetPredictor,
    InteractiveNNUNetPredictor,
)

STRATEGIES = ("centerline", "random", "boundary")
DEFAULT_REPO_CANDIDATES = (
    os.environ.get("AUTOPETV_REPO", ""),
    os.path.abspath(os.path.join(HERE, "..", "autoPETV")),
    "/content/autoPETV",
)


# =============================================================================
# official code import
# =============================================================================
def load_official(repo_root: str):
    """Import `simulate_scribbles` and `metrics` straight out of the challenge repo."""
    repo_root = os.path.abspath(repo_root)
    interactive_dir = os.path.join(repo_root, "interactive")
    for p in (interactive_dir, repo_root):
        if not os.path.isdir(p):
            raise FileNotFoundError(f"not a directory: {p}")
        if p not in sys.path:
            sys.path.insert(0, p)
    import simulate_scribbles  # noqa
    import metrics  # noqa

    for mod, expect in ((simulate_scribbles, interactive_dir), (metrics, repo_root)):
        got = os.path.dirname(os.path.abspath(mod.__file__))
        if got != expect:
            raise ImportError(f"{mod.__name__} imported from {got}, expected {expect}")
    return simulate_scribbles, metrics


def resolve_repo(explicit: Optional[str]) -> str:
    for cand in ([explicit] if explicit else []) + list(DEFAULT_REPO_CANDIDATES):
        if cand and os.path.isfile(os.path.join(cand, "interactive", "simulate_scribbles.py")):
            return os.path.abspath(cand)
    raise FileNotFoundError(
        "could not locate the autoPETV repo; pass --repo /path/to/autoPETV"
    )


# =============================================================================
# metrics
# =============================================================================
def dice_score(pred: np.ndarray, gt: np.ndarray) -> float:
    """Dice as `interactive_loop.py` computes it: empty prediction and empty GT -> 1.0."""
    pred = (pred > 0).astype(np.uint8)
    gt = (gt > 0).astype(np.uint8)

    intersection = np.sum(pred * gt)
    denom = np.sum(pred) + np.sum(gt)

    if denom == 0:
        return 1.0

    return 2.0 * intersection / denom


# =============================================================================
# logging
# =============================================================================
def setup_logger(log_file: str, quiet: bool = False) -> logging.Logger:
    logger = logging.getLogger("autopetv_eval")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s")
    fh = logging.FileHandler(log_file)
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    if not quiet:
        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(fmt)
        logger.addHandler(sh)
    return logger


# =============================================================================
# case discovery
# =============================================================================
class Case:
    __slots__ = ("tag", "stem", "ct", "pet", "label", "strategy")

    def __init__(self, tag, stem, ct, pet, label, strategy=None):
        self.tag = tag          # official tag == basename(ct) minus .nii.gz  (keeps `_0000`!)
        self.stem = stem        # case id without the channel suffix
        self.ct = ct
        self.pet = pet
        self.label = label
        self.strategy = strategy

    def __repr__(self):
        return f"Case({self.tag}, {self.strategy})"


def discover_cases(
    input_cases: str,
    image_dir: Optional[str] = None,
    label_dir: Optional[str] = None,
    official_case_skip: bool = False,
    strict_pairing: bool = True,
    logger: Optional[logging.Logger] = None,
) -> List[Case]:
    image_dir = image_dir or os.path.join(input_cases, "images")
    label_dir = label_dir or os.path.join(input_cases, "labels")

    # same listing/sorting as the official loop
    cts = sorted([os.path.join(image_dir, f) for f in os.listdir(image_dir) if "_0000" in f])
    pets = sorted([os.path.join(image_dir, f) for f in os.listdir(image_dir) if "_0001" in f])
    labels = sorted([os.path.join(label_dir, f) for f in os.listdir(label_dir)])

    cases: List[Case] = []
    for ct, pet, label in zip(cts, pets, labels):
        if official_case_skip and ("fdg" in ct or "198" in ct):
            if logger:
                logger.info(f"official_case_skip: skipping {os.path.basename(ct)}")
            continue
        tag = os.path.basename(ct).replace(".nii.gz", "")       # e.g. psma_xxx_2018-03-04_0000
        stem = tag[:-5] if tag.endswith("_0000") else tag
        pet_stem = os.path.basename(pet).replace(".nii.gz", "")
        lab_stem = os.path.basename(label).replace(".nii.gz", "")
        if strict_pairing and not (
            pet_stem == f"{stem}_0001" and lab_stem == stem
        ):
            raise ValueError(
                "positional CT/PET/label pairing looks wrong (the official loop zips three "
                f"independently sorted lists): ct={os.path.basename(ct)} "
                f"pet={os.path.basename(pet)} label={os.path.basename(label)}. "
                "Re-run with --no_strict_pairing to replicate it anyway."
            )
        cases.append(Case(tag, stem, ct, pet, label))
    return cases


def select_cases(cases: List[Case], wanted: Optional[Sequence[str]], limit: Optional[int]) -> List[Case]:
    if wanted:
        want = set()
        for w in wanted:
            want.add(w)
            want.add(w.replace(".nii.gz", ""))
        sel = [c for c in cases if c.tag in want or c.stem in want]
        missing = want - {c.tag for c in sel} - {c.stem for c in sel}
        # a name may legitimately match both forms; only complain about truly unmatched
        unmatched = [m for m in missing if not any(m in (c.tag, c.stem) for c in sel)]
        if len(sel) == 0:
            raise ValueError(f"none of --cases {sorted(want)} matched; unmatched={sorted(unmatched)}")
        cases = sel
    if limit is not None:
        cases = cases[:limit]
    return cases


def assign_strategies(cases: List[Case], strategy: str, order: Sequence[str]) -> None:
    """Give each case a scribble strategy; `all` round-robins the three over the sorted
    case list, so the mix is uniform and deterministic."""
    if strategy == "all":
        for i, c in enumerate(cases):
            c.strategy = order[i % len(order)]
    else:
        for c in cases:
            c.strategy = strategy


# =============================================================================
# helpers: I/O, replayed scribbles, prediction cache
# =============================================================================
def _write_scribbles(case_result_dir, it, data, logger, tag):
    try:
        with open(os.path.join(case_result_dir, f"iter_{it}_scribbles.json"), "w") as f:
            json.dump(data, f, indent=4)
    except Exception as e:  # pragma: no cover
        logger.warning(f"[{tag}] Failed to save scribbles at iter {it}: {e}")


def _save_pred(case_result_dir, it, mask, affine, logger, tag):
    try:
        out = nib.Nifti1Image(np.asarray(mask).astype(np.uint8), affine)
        out.set_data_dtype(np.uint8)
        nib.save(out, os.path.join(case_result_dir, f"iter_{it}.nii.gz"))
    except Exception as e:  # pragma: no cover
        logger.warning(f"[{tag}] Failed to save prediction at iter {it}: {e}")


def _load_replay_scribbles(replay_dir, tag, it, previous, logger):
    """Read iteration `it`'s scribbles from a recorded run instead of simulating them.

    This is the category-2 setting: clinician scribbles are collected once against the
    baseline and replayed to every algorithm, so they need not match our own errors.
    """
    path = os.path.join(replay_dir, tag, f"iter_{it}_scribbles.json")
    if not os.path.isfile(path):
        logger.warning(f"[{tag}] replay: {path} missing -> keeping the previous scribbles")
        return previous
    with open(path) as f:
        data = json.load(f)
    return {"tumor": list(data.get("tumor", [])), "background": list(data.get("background", []))}


def quantise_prob(prob_fg: np.ndarray, dtype: str) -> np.ndarray:
    """Storage representation of a foreground probability map.

    uint8 is the default: a whole-body softmax is 50-100 M voxels, so float16 does not
    fit the disk for a 100-case sweep, and 1/255 is two orders of magnitude below every
    threshold the post-processing layer uses.
    """
    p = np.clip(np.asarray(prob_fg, dtype=np.float32), 0.0, 1.0)
    if dtype == "uint8":
        return np.rint(p * 255.0).astype(np.uint8)
    if dtype == "float16":
        return p.astype(np.float16)
    raise ValueError(f"unknown probability storage dtype {dtype!r}")


def dequantise_prob(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr)
    if arr.dtype == np.uint8:
        return arr.astype(np.float32) / 255.0
    return arr.astype(np.float32)


class PredictionCache:
    """Content-addressed cache of predictions, keyed by predictor, case and scribble set.

    For `--predictor postproc` the identity is the base model's only, so every
    post-processing variant shares one namespace.  Entries are compressed npz holding the
    mask and, if the run stores them, the quantised foreground probability; asking for a
    probability that a mask-only entry does not have counts as a miss (a "prob upgrade")
    so the entry is rewritten with both rather than silently degrading the caller.
    """

    def __init__(self, root: str, predictor_key: str, logger=None, store_probabilities=False,
                 stateless: bool = True, prob_dtype: str = "uint8"):
        self.root = os.path.join(root, predictor_key)
        self.logger = logger
        self.store_probabilities = store_probabilities
        self.prob_dtype = prob_dtype
        # A stateless prediction depends only on (case, scribbles), so the iteration index
        # is left out of the key and a lesion-free case costs exactly one inference.
        self.stateless = bool(stateless)
        os.makedirs(self.root, exist_ok=True)
        self.hits = 0
        self.misses = 0
        self.prob_upgrades = 0

    @staticmethod
    def scribble_hash(data: Dict) -> str:
        import hashlib
        payload = json.dumps(
            {"tumor": [list(map(int, p)) for p in data.get("tumor", [])],
             "background": [list(map(int, p)) for p in data.get("background", [])]},
            sort_keys=True, separators=(",", ":"))
        return hashlib.sha1(payload.encode()).hexdigest()[:16]

    def _path(self, tag, it, data, state_key=None):
        stem = self.scribble_hash(data) if self.stateless else f"iter{it}_{self.scribble_hash(data)}"
        if state_key:
            # a predictor that consumes state the scribble set does not describe (the
            # interactive model reads the previous mask) folds it in here
            stem = f"{stem}_{state_key}"
        return os.path.join(self.root, tag, f"{stem}.npz")

    def load(self, tag, it, data, state_key=None):
        entry = self.load_entry(tag, it, data, state_key=state_key)
        return None if entry is None else entry[0]

    def load_entry(self, tag, it, data, want_probabilities: bool = False, state_key=None):
        """`(mask, prob_fg | None)`, or None on a miss.

        A mask-only entry counts as a miss when the caller wants the probability, so the
        entry is upgraded rather than pushing the caller onto its no-softmax fallbacks.
        """
        path = self._path(tag, it, data, state_key)
        if not os.path.isfile(path):
            self.misses += 1
            return None
        try:
            with np.load(path) as z:
                if want_probabilities and "prob_fg" not in z.files:
                    self.misses += 1
                    self.prob_upgrades += 1
                    return None
                mask = z["mask"].astype(np.uint8)
                prob = dequantise_prob(z["prob_fg"]) if want_probabilities else None
        except Exception as e:  # pragma: no cover
            if self.logger:
                self.logger.warning(f"cache read failed for {path}: {e}")
            self.misses += 1
            return None
        self.hits += 1
        return mask, prob

    def save(self, tag, it, data, mask, prob_fg=None, prequantised: bool = False, state_key=None):
        path = self._path(tag, it, data, state_key)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        try:
            arrays = {"mask": np.asarray(mask, dtype=np.uint8)}
            if prob_fg is not None and self.store_probabilities:
                arrays["prob_fg"] = (np.asarray(prob_fg) if prequantised
                                     else quantise_prob(prob_fg, self.prob_dtype))
            np.savez_compressed(path, **arrays)
        except Exception as e:  # pragma: no cover
            if self.logger:
                self.logger.warning(f"cache write failed for {path}: {e}")

    def stats(self) -> Dict:
        return {"root": self.root, "hits": self.hits, "misses": self.misses,
                "prob_upgrades": self.prob_upgrades,
                "store_probabilities": bool(self.store_probabilities),
                "prob_dtype": self.prob_dtype}


def base_predictor_name(args) -> str:
    """The model that actually runs; for `--predictor postproc` that is `--base_predictor`."""
    return args.base_predictor if args.predictor == "postproc" else args.predictor


def resolve_guidance_radius(args) -> float:
    """Resolve the guidance radius the way nnUNetTrainer_Interactive does.

    The cache key and the predictor have to agree about the guidance encoding.
    """
    r = getattr(args, "guidance_radius", None)
    if r is not None:
        return float(r)
    return float(os.environ.get("nnUNet_interactive_radius", 10.0))


def predictor_key(args) -> str:
    """Short, stable identity of everything that can change a base-model prediction.

    Blind to the post-processing configuration, so variants that differ only after the
    softmax share one cache namespace; anything touching the network's input or weights
    is in the hash and gets a namespace of its own.
    """
    import hashlib
    ident = {
        "predictor": base_predictor_name(args),
        "model_folder": args.model_folder,
        "folds": list(args.folds),
        "checkpoint": args.checkpoint,
        "tta": bool(args.enable_tta),
        "tile_step_size": args.tile_step_size,
        "threshold": args.threshold,
        "min_component_voxels": args.min_component_voxels,
        "resample_channels": getattr(args, "resample_channels", None),
        "resample_logits": getattr(args, "resample_logits", None),
    }
    if base_predictor_name(args) == "ensemble":
        # every member's weights and checkpoint, and the shared encoding, are in the key
        ident["members"] = list(getattr(args, "ensemble_members", None) or [])
        ident["weights"] = list(getattr(args, "ensemble_weights", None) or [])
        ident["guidance_radius"] = resolve_guidance_radius(args)
        ident["mirror_axes"] = (list(args.mirror_axes)
                                if getattr(args, "mirror_axes", None) else None)
    if base_predictor_name(args) == "interactive_nnunet":
        # guidance encoding and mirroring axes change what the network sees
        ident["guidance_radius"] = resolve_guidance_radius(args)
        ident["mirror_axes"] = (list(args.mirror_axes)
                                if getattr(args, "mirror_axes", None) else None)
        # Added only when the option is on, so every existing row keeps its namespace
        # and its cache; a foveal run gets one of its own, because its answer differs
        # from the plain model's at every iteration that carries a scribble.
        if getattr(args, "foveal_crop", False):
            ident["foveal"] = {"fuse": getattr(args, "foveal_fuse", "max")}
    h = hashlib.sha1(json.dumps(ident, sort_keys=True).encode()).hexdigest()[:10]
    return f"{base_predictor_name(args)}_{h}"


class CachedBasePredictor:
    """The prediction cache, sitting underneath the post-processing layer.

    Keyed by base model, case and accumulated scribble set, so post-processing variants
    share the network's answers wherever their inputs agree; once two variants disagree
    their scribble sets diverge and the network runs again.  The hit rate lands in
    summary.json.  The probability handed downstream goes through the storage
    quantisation on a hit and a miss alike, so a cached variant scores like a cold one.
    """

    def __init__(self, base, cache: Optional[PredictionCache], prob_dtype: str = "uint8",
                 logger=None):
        self.base = base
        self.cache = cache
        self.prob_dtype = prob_dtype
        self.logger = logger
        self.name = f"cached({getattr(base, 'name', type(base).__name__)})"
        self.stateless = getattr(base, "stateless", True)
        #: set by the loop before every call; only used when the base is not stateless
        self.current_iteration = 0
        self.last_timings: Dict[str, float] = {}
        self.n_model_calls = 0
        self.n_cache_calls = 0

    def warmup(self) -> None:
        if hasattr(self.base, "warmup"):
            self.base.warmup()

    def close(self) -> None:
        if hasattr(self.base, "close"):
            self.base.close()

    def predict(self, ct, pet, spacing, scribbles, prev_pred=None, case_cache_dir=None,
                *, affine=None, ct_path=None, pet_path=None, case_name="case",
                return_probabilities=False):
        data = scribbles or _empty_scribble_dict()
        it = self.current_iteration
        # a base model that consumes `prev_pred` (the interactive one) contributes an
        # extra key component, otherwise two iterations with the same scribbles collide
        state_key = None
        if hasattr(self.base, "cache_state_key"):
            state_key = self.base.cache_state_key(prev_pred)
        if self.cache is not None:
            entry = self.cache.load_entry(case_name, it, data,
                                          want_probabilities=return_probabilities,
                                          state_key=state_key)
            if entry is not None:
                self.n_cache_calls += 1
                self.last_timings = {"cache_hit": True, "total_s": 0.0}
                return (entry[0], entry[1]) if return_probabilities else entry[0]

        out = self.base.predict(
            ct, pet, spacing, data, prev_pred=prev_pred, case_cache_dir=case_cache_dir,
            affine=affine, ct_path=ct_path, pet_path=pet_path, case_name=case_name,
            return_probabilities=return_probabilities,
        )
        self.n_model_calls += 1
        self.last_timings = dict(getattr(self.base, "last_timings", {}) or {})
        self.last_timings["cache_hit"] = False

        if not return_probabilities:
            mask = np.asarray(out).astype(np.uint8)
            if self.cache is not None:
                self.cache.save(case_name, it, data, mask, state_key=state_key)
            return mask

        mask, probs = out
        mask = np.asarray(mask).astype(np.uint8)
        prob_fg = np.asarray(probs)
        if prob_fg.ndim == 4:
            prob_fg = prob_fg[1] if prob_fg.shape[0] > 1 else prob_fg[0]
        q = quantise_prob(prob_fg, self.prob_dtype)
        if self.cache is not None:
            self.cache.save(case_name, it, data, mask, prob_fg=q, prequantised=True,
                            state_key=state_key)
        return mask, dequantise_prob(q)


def _state_key(predictor, prev_pred) -> Optional[str]:
    """Extra cache-key component from `Predictor.cache_state_key`, or None."""
    fn = getattr(predictor, "cache_state_key", None)
    return fn(prev_pred) if callable(fn) else None


def _empty_scribble_dict() -> Dict[str, List]:
    return {"tumor": [], "background": []}


def _base_cache_layer(predictor):
    """The `CachedBasePredictor` inside a predictor stack, if there is one."""
    obj = predictor
    for _ in range(8):
        if obj is None or isinstance(obj, CachedBasePredictor):
            return obj
        obj = getattr(obj, "base", None)
    return None


def _set_iteration(predictor, it: int) -> None:
    """Tell every cache layer in the predictor chain which iteration is running."""
    obj = predictor
    for _ in range(8):
        if obj is None:
            return
        if hasattr(obj, "current_iteration"):
            obj.current_iteration = int(it)
        obj = getattr(obj, "base", None)


def check_scribble_guarantees(pred: np.ndarray, data: Dict,
                              scope: Sequence[str] = ("fg", "bg")) -> Dict:
    """Check G2 (every tumor point inside the mask) and G1 (no background point inside).

    Both are always measured; `scope` says which the configuration claims, so an ablation
    that runs with a compliance stage off reports the number without failing on it.
    Points outside the volume are ignored; the official simulator produces none.
    """
    m = np.asarray(pred) > 0
    shape = np.asarray(m.shape)
    out = {"n_tumor": 0, "n_background": 0, "n_tumor_outside": 0, "n_background_inside": 0}
    for key, n_field, bad_field, want_inside in (
        ("tumor", "n_tumor", "n_tumor_outside", True),
        ("background", "n_background", "n_background_inside", False),
    ):
        pts = list(data.get(key) or [])
        if not pts:
            continue
        a = np.asarray(pts, dtype=np.int64).reshape(-1, 3)
        a = a[np.all((a >= 0) & (a < shape), axis=1)]
        out[n_field] = int(len(a))
        if len(a) == 0:
            continue
        inside = m[a[:, 0], a[:, 1], a[:, 2]]
        out[bad_field] = int((~inside).sum()) if want_inside else int(inside.sum())
    out["scope"] = sorted(scope)
    out["ok"] = ((out["n_tumor_outside"] == 0 or "fg" not in scope)
                 and (out["n_background_inside"] == 0 or "bg" not in scope))
    return out


def load_postproc_config(config_path: Optional[str], overrides: Optional[Sequence[str]]) -> Dict:
    """`--postproc_config <json|yaml>` merged with dotted `--postproc_set k.v=value`.

    Override values are parsed as JSON and fall back to strings.  `PostProcConfig
    .from_dict` rejects unknown keys afterwards, so a typo in a sweep crashes rather than
    turning into a silently ignored knob.
    """
    cfg: Dict = {}
    if config_path:
        with open(config_path) as f:
            text = f.read()
        if config_path.lower().endswith((".yaml", ".yml")):
            import yaml  # optional dependency, only for yaml configs
            cfg = yaml.safe_load(text) or {}
        else:
            cfg = json.loads(text) if text.strip() else {}
        if not isinstance(cfg, dict):
            raise ValueError(f"{config_path} must contain a mapping, got {type(cfg).__name__}")
        # Strip the `_description` / `_variant` documentation keys as the container's
        # loader does, so the harness can read submission/postproc_config.json directly
        # instead of a copy that can drift from what ships.
        def _strip_underscored(d):
            return {k: (_strip_underscored(v) if isinstance(v, dict) else v)
                    for k, v in d.items() if not k.startswith("_")}
        cfg = _strip_underscored(cfg)
    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"--postproc_set expects key=value, got {item!r}")
        key, _, raw = item.partition("=")
        try:
            value = json.loads(raw)
        except json.JSONDecodeError:
            value = raw
        parts = [p for p in key.strip().split(".") if p]
        if not parts:
            raise ValueError(f"--postproc_set with an empty key: {item!r}")
        node = cfg
        for p in parts[:-1]:
            node = node.setdefault(p, {})
            if not isinstance(node, dict):
                raise ValueError(f"--postproc_set {key}: {p} is not a section")
        node[parts[-1]] = value
    return cfg


# =============================================================================
# one case
# =============================================================================
def run_case(
    case: Case,
    predictor: Predictor,
    sim,                       # official simulate_scribbles module
    evaluator,                 # official MetricEvaluator instance
    result_dir: str,
    max_iters: int,
    logger: logging.Logger,
    empty_error_crash: bool = False,
    prev_pred_dir: Optional[str] = None,
    save_predictions: str = "all",
    dmm_empty_gt: str = "official",
    replay_scribbles_dir: Optional[str] = None,
    cache: Optional["PredictionCache"] = None,
    determinism_check: bool = False,
    iter_budget_s: float = 600.0,
    check_guarantees: bool = False,
    strict_guarantees: bool = False,
    guarantee_scope: Sequence[str] = ("fg", "bg"),
    keep_state: bool = False,
) -> Tuple[List[Dict], Dict]:
    simulate_scribble_from_label = sim.simulate_scribble_from_label
    scribbles_to_gc_format = sim.scribbles_to_gc_format
    gc_to_swfastedit_format = sim.gc_to_swfastedit_format

    tag = case.tag
    case_result_dir = os.path.join(result_dir, tag)
    os.makedirs(case_result_dir, exist_ok=True)
    # The per-case state directory, standing in for the container's persistent mount.  It
    # sits beside the predictions rather than inside the case output dir so that a
    # persisted probability map is not mistaken for a run artifact.
    cache_dir = os.path.join(result_dir, "state", tag)
    shutil.rmtree(cache_dir, ignore_errors=True)
    os.makedirs(cache_dir, exist_ok=True)

    records: List[Dict] = []
    info: Dict = {"tag": tag, "strategy": case.strategy, "iter_seconds": [], "reused": []}
    if replay_scribbles_dir is not None:
        info["replay_scribbles_dir"] = replay_scribbles_dir

    # ---- load -----------------------------------------------------------
    t_load = time.time()
    ct_img = nib.load(case.ct)
    pet_img = nib.load(case.pet)
    label_img = nib.load(case.label)
    ct = np.asanyarray(ct_img.dataobj)
    pet = np.asanyarray(pet_img.dataobj)
    # the official loop uses get_fdata() (float64); we keep the stored dtype, since every
    # downstream use is `gt == 0/1`, `np.sum(gt)` or `gt.astype(np.uint8)`
    gt = np.asanyarray(label_img.dataobj)
    affine = pet_img.affine
    spacing = tuple(float(z) for z in pet_img.header.get_zooms()[:3])
    info["load_s"] = time.time() - t_load
    info["shape"] = list(gt.shape)
    info["spacing"] = list(spacing)

    if ct.shape != gt.shape or pet.shape != gt.shape:
        raise ValueError(f"[{tag}] shape mismatch ct={ct.shape} pet={pet.shape} gt={gt.shape}")

    empty_gt = np.sum(gt) == 0
    info["empty_gt"] = bool(empty_gt)
    logger.info(f"[{tag}] shape={gt.shape} spacing={spacing} empty_gt={empty_gt} strategy={case.strategy}")

    # The GC input-interface file, kept out of `cache_dir`: a Predictor may wipe its own
    # cache, and the loop's interaction state has to survive that.
    gc_json_path = os.path.join(case_result_dir, "lesion-clicks.json")
    if os.path.exists(gc_json_path):
        os.remove(gc_json_path)
    prev_dice: Optional[float] = None
    prev_dmm: Optional[float] = None
    last_pred: Optional[np.ndarray] = None   # what the interface `seg_dir` would hold
    scored: Optional[tuple] = None           # (mask, dice, dmm, n_fp, n_fn) actually measured
    data = {"tumor": [], "background": []}

    for it in range(max_iters):
        logger.info(f"[{tag}] Iteration {it}")
        t_iter = time.time()
        reused = None
        try:
            if it == 0:
                data = {"tumor": [], "background": []}
            else:
                if empty_gt:
                    # The official loop reuses the previous scores here.  Both are
                    # overwritten after inference below; kept for fidelity.
                    dice = prev_dice if prev_dice is not None else 0.0
                    dmm = prev_dmm if prev_dmm is not None else 0.0
                    logger.info(f"[{tag}] Empty GT -> reusing previous Dice")
                elif replay_scribbles_dir is not None:
                    data = _load_replay_scribbles(replay_scribbles_dir, tag, it, data, logger)
                else:
                    if last_pred is None:
                        raise FileNotFoundError("Missing segmentation")
                    pred = last_pred
                    with open(gc_json_path, "r") as f:
                        data = gc_to_swfastedit_format(json.load(f))

                    if pred.shape != gt.shape:
                        raise ValueError("Shape mismatch")

                    overseg = (pred == 1) & (gt == 0)
                    underseg = (pred == 0) & (gt == 1)

                    res_bg = simulate_scribble_from_label(overseg, case.strategy)
                    res_fg = simulate_scribble_from_label(underseg, case.strategy)
                    if not empty_error_crash:
                        # An empty error mask makes the official function return
                        # ([], False); a zero size lets the `fp <= fn` rule take the
                        # scribble from the non-empty region.
                        res_bg = res_bg if len(res_bg) == 3 else (res_bg[0], res_bg[1], 0)
                        res_fg = res_fg if len(res_fg) == 3 else (res_fg[0], res_fg[1], 0)
                    # under --official_empty_error_crash this unpacking of a 2-tuple is
                    # the published loop's ValueError
                    scribbles_bg, _, fp = res_bg
                    scribbles_fg, _, fn = res_fg
                    info.setdefault("fp_fn", []).append([int(fp), int(fn)])

                    if int(fp) == 0 and int(fn) == 0:
                        # perfect prediction: propagate it (and its scores) to all the
                        # remaining iterations, without corrections and without inference
                        logger.info(f"[{tag}] perfect prediction at iter {it} -> "
                                    f"propagating to iterations {it}..{max_iters - 1}")
                        info["propagated_from_iter"] = it
                        for k in range(it, max_iters):
                            _write_scribbles(case_result_dir, k, data, logger, tag)
                            if save_predictions == "all" or (save_predictions == "last"
                                                             and k == max_iters - 1):
                                _save_pred(case_result_dir, k, last_pred, affine, logger, tag)
                            records.append({"iteration": k, "dice": float(prev_dice),
                                            "dmm": float(prev_dmm)})
                            info["iter_seconds"].append(0.0)
                            info["reused"].append("propagated")
                            logger.info(f"[{tag}] Dice@{k}: {prev_dice:.4f}  "
                                        f"DMM@{k}: {prev_dmm:.4f}  (propagated)")
                        break

                    if fp <= fn:
                        data["tumor"] += scribbles_fg
                    else:
                        data["background"] += scribbles_bg

            # ---- persist the interaction state --------------------------
            with open(gc_json_path, "w") as f:
                json.dump(scribbles_to_gc_format(data), f)
            _write_scribbles(case_result_dir, it, data, logger, tag)

            # ---- inference ---------------------------------------------
            pred = None
            if prev_pred_dir:
                cand = os.path.join(prev_pred_dir, tag, f"iter_{it}.nii.gz")
                if os.path.isfile(cand):
                    pred = np.asanyarray(nib.load(cand).dataobj).astype(np.uint8)
                    reused = cand
                    logger.info(f"[{tag}] iter {it}: reusing prediction {cand}")
                    prev_scr = os.path.join(prev_pred_dir, tag, f"iter_{it}_scribbles.json")
                    if os.path.isfile(prev_scr):
                        with open(prev_scr) as f:
                            old = json.load(f)
                        if old != json.loads(json.dumps(data)):
                            logger.warning(
                                f"[{tag}] iter {it}: reused prediction was produced with "
                                "DIFFERENT scribbles than the ones recomputed now"
                            )
            if pred is None and cache is not None:
                pred = cache.load(tag, it, data, state_key=_state_key(predictor, last_pred))
                if pred is not None:
                    reused = "cache"
                    logger.info(f"[{tag}] iter {it}: prediction taken from --cache_dir")
            if pred is None:
                _set_iteration(predictor, it)
                # A run filling the shared cache asks for the softmax as well; nnU-Net
                # argmaxes the probabilities it would otherwise discard, so the only cost
                # is exporting one extra volume.
                want_prob = cache is not None and cache.store_probabilities
                out = predictor.predict(
                    ct, pet, spacing, data,
                    prev_pred=last_pred,
                    case_cache_dir=cache_dir,
                    affine=affine,
                    ct_path=case.ct,
                    pet_path=case.pet,
                    case_name=tag,
                    return_probabilities=want_prob,
                )
                prob_fg = None
                if want_prob and isinstance(out, tuple) and len(out) == 2:
                    pred, probs = out
                    probs = np.asarray(probs)
                    prob_fg = probs[1] if probs.ndim == 4 and probs.shape[0] > 1 else (
                        probs[0] if probs.ndim == 4 else probs)
                else:
                    pred = out[0] if isinstance(out, tuple) else out
                if getattr(predictor, "last_timings", None):
                    info.setdefault("predictor_timings", []).append(dict(predictor.last_timings))
                _bc = _base_cache_layer(predictor)
                if _bc is not None:
                    hit = bool(_bc.last_timings.get("cache_hit"))
                    info.setdefault("base_cache_hit", []).append(hit)
                    if hit:
                        reused = "base_cache"
                        logger.info(f"[{tag}] iter {it}: base inference served from "
                                    f"--cache_dir (no forward pass)")
                if getattr(predictor, "last_info", None):
                    li = predictor.last_info
                    info.setdefault("postproc", []).append({
                        k: li.get(k) for k in (
                            "iteration", "tracer", "n_tumor", "n_background",
                            "base_volume_ml", "final_volume_ml", "negative_gate_fired",
                            "empty_output", "empty_without_gate", "constraints",
                            "t_total", "t_base_predict", "t_cleanup", "t_cleanup2",
                            "t_bridge", "t_bg_compliance", "t_fg_compliance", "t_monotone",
                            "cleanup", "cleanup_after_compliance")
                        if k in li})
                if determinism_check and it == 0:
                    again = np.asarray(predictor.predict(
                        ct, pet, spacing, data, prev_pred=last_pred, case_cache_dir=cache_dir,
                        affine=affine, ct_path=case.ct, pet_path=case.pet, case_name=tag))
                    identical = bool(np.array_equal(np.asarray(pred), again))
                    info["determinism_iter0_identical"] = identical
                    (logger.info if identical else logger.error)(
                        f"[{tag}] determinism check at iter 0: "
                        f"{'bitwise identical' if identical else 'MISMATCH between two runs'}")
                    del again
                if cache is not None:
                    cache.save(tag, it, data, np.asarray(pred).astype(np.uint8),
                               state_key=_state_key(predictor, last_pred),
                               prob_fg=prob_fg)
                del prob_fg
                pred = np.asarray(pred)
                if pred.shape != gt.shape:
                    raise ValueError(
                        f"predictor returned shape {pred.shape}, expected {gt.shape}"
                    )
            pred = np.asarray(pred).astype(np.uint8)

            # ---- scribble guarantees on the mask that is about to be scored ----
            if check_guarantees:
                g = check_scribble_guarantees(pred, data, scope=guarantee_scope)
                info.setdefault("guarantees", []).append(g)
                if g["ok"]:
                    logger.info(f"[{tag}] iter {it}: guarantees OK "
                                f"({g['n_tumor'] - g['n_tumor_outside']}/{g['n_tumor']} "
                                f"tumor point(s) inside the mask, "
                                f"{g['n_background'] - g['n_background_inside']}/"
                                f"{g['n_background']} background point(s) outside; "
                                f"enforced={g['scope']})")
                else:
                    msg = (f"[{tag}] iter {it}: SCRIBBLE GUARANTEE VIOLATED -- "
                           f"{g['n_tumor_outside']}/{g['n_tumor']} tumor points outside "
                           f"the mask, {g['n_background_inside']}/{g['n_background']} "
                           f"background points inside it")
                    logger.error(msg)
                    if strict_guarantees:
                        raise AssertionError(msg)
            # Dice and DMM are pure functions of (prediction, GT), so an iteration that
            # predicts the same array as the last scored one reuses its metrics instead of
            # recomputing them over ~90 M voxels.  Lesion-free cases hit this every time.
            unchanged = scored is not None and np.array_equal(pred, scored[0])
            last_pred = pred

            if unchanged:
                # `scored` is only set by an iteration that computed the metrics, so this
                # cannot inherit the 0.0/0.0 of a failed one.
                _, dice, dmm, n_fp, n_fn = scored
                info.setdefault("metrics_reused_at", []).append(it)
            else:
                # exposure to the empty-error-region rule: an iteration with no FP (or no
                # FN) voxels is one where the published loop would have raised and scored 0
                n_fp = int(np.count_nonzero((pred == 1) & (gt == 0)))
                n_fn = int(np.count_nonzero((pred == 0) & (gt == 1)))
                dice = dice_score(pred, gt)
                dmm = evaluator(
                    prediction=pred,
                    ground_truth=gt.astype(np.uint8),
                    case_name="case1",
                )["f1"]
                scored = (pred, dice, dmm, n_fp, n_fn)
            info.setdefault("error_voxels", []).append([n_fp, n_fn])

            if dmm_empty_gt != "official" and empty_gt:
                if dmm_empty_gt == "nan_to_zero":
                    dmm = 0.0
                elif dmm_empty_gt == "empty_pred_one":
                    dmm = 1.0 if int(np.sum(pred)) == 0 else 0.0

            if save_predictions == "all" or (save_predictions == "last" and it == max_iters - 1):
                _save_pred(case_result_dir, it, last_pred, affine, logger, tag)

        except Exception as e:
            logger.warning(f"[{tag}] Iteration {it} failed: {e}")
            logger.debug(traceback.format_exc())
            dice, dmm = 0.0, 0.0
            info.setdefault("failed_iters", []).append({"iteration": it, "error": repr(e)})

        prev_dice = float(dice)
        prev_dmm = float(dmm)
        records.append({"iteration": it, "dice": float(dice), "dmm": float(dmm)})
        elapsed = time.time() - t_iter
        info["iter_seconds"].append(round(elapsed, 3))
        info["reused"].append(reused)
        if elapsed > iter_budget_s and reused is None:
            logger.warning(f"[{tag}] iteration {it} took {elapsed:.0f}s, over the "
                           f"{iter_budget_s:.0f}s per-iteration budget")
        logger.info(f"[{tag}] Dice@{it}: {dice:.4f}  DMM@{it}: {dmm:.4f}  "
                    f"({info['iter_seconds'][-1]:.1f}s)")

    if keep_state:
        info["state_dir"] = cache_dir
    else:
        shutil.rmtree(cache_dir, ignore_errors=True)
    if check_guarantees:
        bad = [i for i, g in enumerate(info.get("guarantees", [])) if not g["ok"]]
        info["guarantee_violations"] = bad
        if bad:
            logger.error(f"[{tag}] scribble guarantees violated at iteration(s) {bad}")
    return records, info


# =============================================================================
# AUC + aggregation
# =============================================================================
def compute_auc(case_dict: Dict[str, List[Dict]]) -> Dict[str, Dict[str, float]]:
    auc_results = {}
    for case_id, records in case_dict.items():
        records = sorted(records, key=lambda x: x["iteration"])
        iterations = np.array([r["iteration"] for r in records], dtype=float)
        dice = np.array([r["dice"] for r in records], dtype=float)
        dmm = np.array([r["dmm"] for r in records], dtype=float)
        auc_results[case_id] = {
            "auc_dice": float(np.trapezoid(dice, iterations)),
            "auc_dmm": float(np.trapezoid(dmm, iterations)),
        }
    return auc_results


def _tracer_of(tag: str) -> str:
    t = tag.lower()
    for known in ("fdg", "psma"):
        if t.startswith(known):
            return known
    return "other"


def _group_stats(tags, auc, case_info) -> Dict:
    """Mean AUC-Dice / AUC-DMM over a set of cases; DMM uses nanmean, so lesion-free
    cases (whose DMM is NaN) drop out of it."""
    if not tags:
        return {"n": 0, "auc_dice": float("nan"), "auc_dmm": float("nan")}
    d = np.array([auc[t]["auc_dice"] for t in tags], dtype=float)
    m = np.array([auc[t]["auc_dmm"] for t in tags], dtype=float)
    return {"n": len(tags), "auc_dice": float(np.mean(d)), "auc_dmm": float(_safe_nanmean(m))}


def summarise(case_dict, auc, cases, max_iters, case_info=None, extra=None) -> Dict:
    case_info = case_info or {}
    tags = list(case_dict.keys())
    dice = np.array([[r["dice"] for r in sorted(case_dict[t], key=lambda x: x["iteration"])]
                     for t in tags], dtype=float) if tags else np.zeros((0, max_iters))
    dmm = np.array([[r["dmm"] for r in sorted(case_dict[t], key=lambda x: x["iteration"])]
                    for t in tags], dtype=float) if tags else np.zeros((0, max_iters))
    auc_dice = np.array([auc[t]["auc_dice"] for t in tags], dtype=float)
    auc_dmm = np.array([auc[t]["auc_dmm"] for t in tags], dtype=float)
    nan_dmm = [t for t, row in zip(tags, dmm) if np.isnan(row).any()]

    # The negative gate moves a lesion-absent case between 0 and the maximum AUC-Dice, so
    # absent and present cases are always reported separately as well as pooled.
    absent = [t for t in tags if case_info.get(t, {}).get("empty_gt")]
    present = [t for t in tags if not case_info.get(t, {}).get("empty_gt")]
    by_tracer = {}
    for tr in sorted({_tracer_of(t) for t in tags}):
        sel = [t for t in tags if _tracer_of(t) == tr]
        by_tracer[tr] = {
            "all": _group_stats(sel, auc, case_info),
            "lesion_present": _group_stats([t for t in sel if t in present], auc, case_info),
            "lesion_absent": _group_stats([t for t in sel if t in absent], auc, case_info),
        }

    zero_fp = {t: sum(1 for fp, _ in case_info.get(t, {}).get("error_voxels", []) if fp == 0)
               for t in tags}
    zero_fn = {t: sum(1 for _, fn in case_info.get(t, {}).get("error_voxels", []) if fn == 0)
               for t in tags}

    strat = {c.tag: c.strategy for c in cases}
    out = {
        "n_cases": len(tags),
        "max_iters": max_iters,
        "strategies": strat,
        "mean_dice_per_iteration": [float(x) for x in np.nanmean(dice, axis=0)] if tags else [],
        "mean_dmm_per_iteration": [float(x) for x in _safe_nanmean(dmm, axis=0)] if tags else [],
        "mean_auc_dice": float(np.mean(auc_dice)) if tags else float("nan"),
        # official aggregation: DMM is NaN on lesion-free cases, which are excluded
        "mean_auc_dmm": float(_safe_nanmean(auc_dmm)) if tags else float("nan"),
        # what a naive np.mean over the AUC file would give (NaN as soon as one case is)
        "mean_auc_dmm_nan_propagating": float(np.mean(auc_dmm)) if tags else float("nan"),
        "final_score_50_50": (
            float(0.5 * np.mean(auc_dice) + 0.5 * _safe_nanmean(auc_dmm)) if tags else float("nan")
        ),
        "max_auc": float(max_iters - 1),
        "by_lesion_status": {
            "lesion_present": _group_stats(present, auc, case_info),
            "lesion_absent": _group_stats(absent, auc, case_info),
        },
        "by_tracer": by_tracer,
        "n_cases_with_nan_dmm": len(nan_dmm),
        "cases_with_nan_dmm": nan_dmm,
        "empty_error_region_exposure": {
            "n_iters_with_zero_fp": zero_fp,
            "n_iters_with_zero_fn": zero_fn,
            "total_iters_with_zero_fp": int(sum(zero_fp.values())),
            "total_iters_with_zero_fn": int(sum(zero_fn.values())),
            "cases_propagated": [t for t in tags
                                 if "propagated_from_iter" in case_info.get(t, {})],
        },
        "determinism_iter0_mismatches": [
            t for t in tags if case_info.get(t, {}).get("determinism_iter0_identical") is False],
        "seconds_per_iteration": {
            t: case_info.get(t, {}).get("iter_seconds") for t in tags},
    }
    if extra:
        out.update(extra)
    return out


def _safe_nanmean(a, axis=None):
    a = np.asarray(a, dtype=float)
    with np.errstate(invalid="ignore"):
        if np.all(np.isnan(a)):
            return np.full(a.shape[1:], np.nan) if axis is not None else float("nan")
        return np.nanmean(a, axis=axis)


# =============================================================================
# CLI
# =============================================================================
def build_base_predictor(args) -> Predictor:
    """The model itself, with no interaction layer around it."""
    name = base_predictor_name(args)
    if name == "threshold":
        return ThresholdPredictor(threshold=args.threshold,
                                  min_component_voxels=args.min_component_voxels)
    common = dict(
        model_folder=args.model_folder,
        folds=tuple(args.folds),
        checkpoint_name=args.checkpoint,
        device=args.device,
        disable_tta=not args.enable_tta,
        tile_step_size=args.tile_step_size,
        num_processes_preprocessing=args.npp,
        num_processes_segmentation_export=args.nps,
        verbose=args.verbose_nnunet,
    )
    if name == "baseline_nnunet":
        return BaselineNNUNetPredictor(**common)
    if name == "fast_baseline_nnunet":
        return FastBaselineNNUNetPredictor(
            resample_channels=args.resample_channels,
            resample_logits=args.resample_logits,
            num_resample_threads=args.resample_threads,
            **common,
        )
    if name == "ensemble":
        from ensemble_predictor import build_ensemble  # noqa: E402
        if not getattr(args, "ensemble_members", None):
            raise ValueError("--base_predictor ensemble needs --ensemble_members")
        common.pop("model_folder")
        common.pop("folds")
        common.pop("checkpoint_name")
        return build_ensemble(
            args.ensemble_members, getattr(args, "ensemble_weights", None),
            folds=tuple(args.folds),
            resample_channels=args.resample_channels,
            resample_logits=args.resample_logits,
            num_resample_threads=args.resample_threads,
            guidance_radius=resolve_guidance_radius(args),
            use_state_dir=args.interactive_state_dir,
            deterministic=not args.no_cudnn_deterministic,
            force_mirror_axes=(tuple(args.mirror_axes) if args.mirror_axes else None),
            **common,
        )
    if name == "interactive_nnunet":
        if args.model_folder is None:
            common.pop("model_folder")          # let the class pick its own default
        return InteractiveNNUNetPredictor(
            resample_channels=args.resample_channels,
            resample_logits=args.resample_logits,
            num_resample_threads=args.resample_threads,
            guidance_radius=resolve_guidance_radius(args),
            use_state_dir=args.interactive_state_dir,
            foveal_crop=getattr(args, "foveal_crop", False),
            foveal_fuse=getattr(args, "foveal_fuse", "max"),
            deterministic=not args.no_cudnn_deterministic,
            force_mirror_axes=(tuple(args.mirror_axes) if args.mirror_axes else None),
            **common,
        )
    raise ValueError(name)


def build_predictor(args, cache: Optional[PredictionCache] = None, logger=None):
    """The full predictor stack, `--predictor` included.

    Everything except `postproc` is the bare model, with the loop's own cache in front of
    it.  `postproc` builds PostProcPredictor(CachedBasePredictor(base)) and switches the
    loop cache off, since a key that does not mention the post-processing config would
    hand one variant another's mask.
    """
    base = build_base_predictor(args)
    if args.predictor != "postproc":
        return base
    from postproc import PostProcPredictor  # noqa: E402  (optional dependency)
    from postproc.config import PostProcConfig  # noqa: E402

    cfg = PostProcConfig.from_dict(load_postproc_config(args.postproc_config,
                                                        args.postproc_set))
    cached = CachedBasePredictor(base, cache, prob_dtype=args.cache_prob_dtype,
                                 logger=logger)
    predictor = PostProcPredictor(cached, cfg)
    predictor.resolved_config = cfg.to_dict()
    return predictor


def make_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input_cases", type=str, required=True,
                   help="dir containing images/ and labels/ (nnU-Net naming)")
    p.add_argument("--image_dir", type=str, default=None)
    p.add_argument("--label_dir", type=str, default=None)
    p.add_argument("--out_dir", type=str, required=True)
    p.add_argument("--repo", type=str, default=None,
                   help="path to the official autoPETV repo (for simulate_scribbles + metrics)")
    p.add_argument("--strategy", choices=list(STRATEGIES) + ["all"], default="all")
    p.add_argument("--strategy_order", type=str, default="centerline,random,boundary",
                   help="round-robin order used by --strategy all")
    p.add_argument("--max_iters", type=int, default=6,
                   help="iteration 0 (no scribbles) + 5 corrections, per the organizers")
    p.add_argument("--cases", type=str, nargs="*", default=None,
                   help="case tags (with or without the _0000 suffix) to evaluate")
    p.add_argument("--cases_file", type=str, default=None,
                   help="file with one case tag per line (blank lines and #-comments "
                        "ignored); merged into --cases. docs/valset_screen39.txt is the "
                        "stratified 39-case screening subset")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--prev_pred_dir", type=str, default=None,
                   help="reuse iter_k.nii.gz from a previous run (resume / re-score)")
    p.add_argument("--save_predictions", choices=["all", "last", "none"], default="all")
    p.add_argument("--official_case_skip", action="store_true",
                   help="replicate `if \"fdg\" in ct or '198' in ct: continue`")
    p.add_argument("--no_strict_pairing", dest="strict_pairing", action="store_false")
    p.add_argument("--eval", dest="eval_mode", choices=["fixed", "buggy"], default="fixed",
                   help="fixed (default) = the deployed evaluator's behaviour on an empty "
                        "error region (propagate the perfect prediction); buggy = the "
                        "published loop's ValueError, which scores those iterations 0")
    p.add_argument("--determinism_check", action="store_true",
                   help="predict iteration 0 twice per case and assert bitwise equality")
    p.add_argument("--iter_budget_s", type=float, default=600.0,
                   help="warn when one iteration exceeds this wall time")
    p.add_argument("--official_empty_error_crash", action="store_true",
                   help="reproduce the published loop's ValueError when an error region "
                        "is empty (scores that iteration 0.0/0.0); the deployed evaluator "
                        "propagates the perfect prediction instead, which is the default")
    p.add_argument("--replay_scribbles_dir", type=str, default=None,
                   help="feed the iter_k_scribbles.json of a previous run instead of "
                        "simulating scribbles from our own errors (category-2 setting)")
    p.add_argument("--cache_dir", type=str, default=None,
                   help="content-addressed prediction cache keyed by base predictor, "
                        "case, iteration and scribble set, shared by every "
                        "post-processing variant of the same model")
    p.add_argument("--cache_probabilities", action="store_true",
                   help="also store the foreground softmax in the cache (implied by "
                        "--predictor postproc)")
    p.add_argument("--cache_prob_dtype", choices=["uint8", "float16"], default="uint8",
                   help="storage precision of the cached probability; float16 does not "
                        "fit the disk for a 100-case sweep")
    p.add_argument("--dmm_empty_gt", choices=["official", "nan_to_zero", "empty_pred_one"],
                   default="official",
                   help="what to record as DMM for empty-GT cases (official = NaN)")
    p.add_argument("--stop_on_case_error", action="store_true",
                   help="abort instead of scoring a crashed case 0.0 x max_iters "
                        "(the official loop always continues)")
    p.add_argument("--quiet", action="store_true")

    p.add_argument("--predictor",
                   choices=["fast_baseline_nnunet", "baseline_nnunet", "interactive_nnunet",
                            "threshold", "postproc"],
                   default="fast_baseline_nnunet",
                   help="fast_baseline_nnunet is the in-process, cached version of "
                        "baseline_nnunet (which goes through NIfTI files); postproc wraps "
                        "--base_predictor in src/postproc's interaction layer")
    p.add_argument("--base_predictor",
                   choices=["fast_baseline_nnunet", "baseline_nnunet", "interactive_nnunet",
                            "threshold", "ensemble"],
                   default="fast_baseline_nnunet",
                   help="the model underneath --predictor postproc; it alone defines "
                        "the prediction-cache namespace. 'ensemble' averages the "
                        "foreground softmax of the --ensemble_members on the original "
                        "image grid, so members may have different plans")
    p.add_argument("--ensemble_members", type=str, nargs="*", default=None,
                   help="members of --base_predictor ensemble, each "
                        "'<model_folder>[:<checkpoint>[:<weight>]]'. Weights are "
                        "normalised; give them here or with --ensemble_weights, not both")
    p.add_argument("--ensemble_weights", type=float, nargs="*", default=None,
                   help="one weight per member (normalised); default is equal weights")
    p.add_argument("--postproc_config", type=str, default=None,
                   help="json or yaml file with a (possibly partial, possibly nested) "
                        "PostProcConfig; unknown keys are an error")
    p.add_argument("--postproc_set", type=str, nargs="*", default=None, metavar="KEY=VALUE",
                   help="dotted overrides applied on top of --postproc_config, e.g. "
                        "negative_gate.enabled=false cleanup.min_volume_ml=0.5 "
                        "enable_fg_compliance=false. Values are parsed as JSON")
    p.add_argument("--check_guarantees", choices=["auto", "always", "never"], default="auto",
                   help="verify per iteration that every tumor scribble is inside the "
                        "scored mask and no background scribble is; auto = on for "
                        "--predictor postproc, off otherwise")
    p.add_argument("--strict_guarantees", action="store_true",
                   help="raise (scoring the iteration 0.0) instead of only logging when "
                        "a scribble guarantee is violated")
    p.add_argument("--keep_state", action="store_true",
                   help="keep <out_dir>/state/<case>/ after the case instead of wiping it")
    p.add_argument("--model_folder", type=str, default=None)
    p.add_argument("--folds", type=int, nargs="+", default=[0])
    p.add_argument("--checkpoint", type=str, default="checkpoint_final.pth")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--enable_tta", action="store_true",
                   help="mirroring TTA over all trained axes (8 forward passes); the "
                        "baseline container runs with --disable_tta")
    p.add_argument("--tile_step_size", type=float, default=0.5)
    p.add_argument("--npp", type=int, default=3)
    p.add_argument("--nps", type=int, default=3)
    p.add_argument("--verbose_nnunet", action="store_true")
    p.add_argument("--resample_channels", choices=["scipy", "torch"], default="scipy",
                   help="scipy = nnU-Net's order-3 spline; torch = GPU trilinear, much "
                        "faster but not equivalent, so an ablation knob only")
    p.add_argument("--resample_logits", choices=["torch", "scipy"], default="torch",
                   help="torch resamples the logits on the GPU with the order-1 "
                        "interpolation nnU-Net uses for probabilities")
    p.add_argument("--resample_threads", type=int, default=4)
    p.add_argument("--guidance_radius", type=float, default=None,
                   help="radius of the clipped-EDT guidance encoding, in voxels of the "
                        "preprocessed grid; must match the trainer's GUIDANCE_RADIUS")
    p.add_argument("--foveal_crop", action="store_true",
                   help="interactive model only: at every iteration that carries a "
                        "scribble, run one extra forward pass on a patch-sized window "
                        "centred on the newest scribble and fuse it into the logits")
    p.add_argument("--foveal_fuse", choices=["max", "mean"], default="max",
                   help="how the foveal window's logits meet the sliding window's")
    p.add_argument("--interactive_state_dir", action="store_true",
                   help="let interactive_nnunet read the previous mask from the per-case "
                        "state dir when the caller passes no prev_pred (for the container)")
    p.add_argument("--no_cudnn_deterministic", action="store_true",
                   help="interactive_nnunet sets cudnn.deterministic and disables "
                        "cudnn.benchmark by default; this restores the nnU-Net defaults")
    p.add_argument("--mirror_axes", type=int, nargs="*", default=None,
                   help="override the allowed mirroring axes of the checkpoint, e.g. "
                        "--mirror_axes 0 1 2 for full 8-way TTA (needs --enable_tta)")
    p.add_argument("--threshold", type=float, default=2.5)
    p.add_argument("--min_component_voxels", type=int, default=0)
    return p


def main(argv: Optional[Sequence[str]] = None) -> Dict:
    args = make_parser().parse_args(argv)

    os.makedirs(args.out_dir, exist_ok=True)
    logger = setup_logger(os.path.join(args.out_dir, "run.log"), quiet=args.quiet)
    t_start = time.time()

    repo = resolve_repo(args.repo)
    sim, metrics_mod = load_official(repo)
    logger.info(f"official code imported from {repo}")

    cases = discover_cases(
        args.input_cases, args.image_dir, args.label_dir,
        official_case_skip=args.official_case_skip,
        strict_pairing=args.strict_pairing,
        logger=logger,
    )
    if args.cases_file:
        with open(args.cases_file) as f:
            from_file = [ln.strip() for ln in f
                         if ln.strip() and not ln.lstrip().startswith("#")]
        args.cases = list(args.cases or []) + from_file
    cases = select_cases(cases, args.cases, args.limit)
    assign_strategies(cases, args.strategy, args.strategy_order.split(","))
    if not cases:
        raise SystemExit("no cases selected")
    logger.info(f"{len(cases)} case(s): " + ", ".join(f"{c.tag}[{c.strategy}]" for c in cases[:10])
                + (" ..." if len(cases) > 10 else ""))

    # Keyed by the base model, so it is built before the stack and handed to it: under
    # --predictor postproc it sits below the interaction layer, otherwise the loop
    # consults it directly.
    cache = None
    if args.cache_dir:
        cache = PredictionCache(
            args.cache_dir, predictor_key(args), logger=logger,
            store_probabilities=bool(args.cache_probabilities
                                     or args.predictor == "postproc"),
            stateless=True,
            prob_dtype=args.cache_prob_dtype,
        )
        logger.info(f"prediction cache: {cache.root} "
                    f"(probabilities={'on' if cache.store_probabilities else 'off'}, "
                    f"dtype={cache.prob_dtype})")

    predictor = build_predictor(args, cache=cache, logger=logger)
    if cache is not None:
        cache.stateless = getattr(predictor, "stateless", True) if args.predictor != "postproc" \
            else getattr(getattr(predictor, "base", None), "stateless", True)
    loop_cache = cache if args.predictor != "postproc" else None
    t0 = time.time()
    predictor.warmup()
    logger.info(f"predictor '{predictor.name}' ready in {time.time() - t0:.1f}s")
    if args.predictor == "postproc":
        logger.info("postproc config: "
                    + json.dumps(load_postproc_config(args.postproc_config,
                                                      args.postproc_set), sort_keys=True))

    check_guarantees = (args.check_guarantees == "always"
                        or (args.check_guarantees == "auto" and args.predictor == "postproc"))
    # which guarantees this configuration claims; the rest are measured but not enforced
    _cfgd = getattr(predictor, "resolved_config", None) or {}
    guarantee_scope = tuple(
        n for n, k in (("fg", "enable_fg_compliance"), ("bg", "enable_bg_compliance"))
        if bool(_cfgd.get(k, True)))
    if check_guarantees:
        logger.info(f"guarantee check on, enforcing {list(guarantee_scope) or 'nothing'}")

    evaluator = metrics_mod.MetricEvaluator()

    output_metric_file = os.path.join(args.out_dir, "metric_scores.json")
    case_dict: Dict[str, List[Dict]] = {}
    case_info: Dict[str, Dict] = {}

    for case in cases:
        logger.info(f"Processing case: {case.tag}")
        t_case = time.time()
        try:
            records, info = run_case(
                case, predictor, sim, evaluator, args.out_dir, args.max_iters, logger,
                empty_error_crash=(args.official_empty_error_crash
                                   or args.eval_mode == "buggy"),
                determinism_check=args.determinism_check,
                iter_budget_s=args.iter_budget_s,
                prev_pred_dir=args.prev_pred_dir,
                save_predictions=args.save_predictions,
                dmm_empty_gt=args.dmm_empty_gt,
                replay_scribbles_dir=args.replay_scribbles_dir,
                cache=loop_cache,
                check_guarantees=check_guarantees,
                strict_guarantees=args.strict_guarantees,
                guarantee_scope=guarantee_scope,
                keep_state=args.keep_state,
            )
        except Exception as e:
            logger.error(f"[{case.tag}] Case failed completely: {e}")
            logger.debug(traceback.format_exc())
            if args.stop_on_case_error:
                raise
            records = [{"iteration": i, "dice": 0.0, "dmm": 0.0} for i in range(args.max_iters)]
            info = {"tag": case.tag, "strategy": case.strategy, "error": repr(e)}
        info["case_seconds"] = round(time.time() - t_case, 2)
        case_dict[case.tag] = records
        case_info[case.tag] = info
        with open(output_metric_file, "w") as f:
            json.dump(case_dict, f, indent=4)
        with open(os.path.join(args.out_dir, "case_info.json"), "w") as f:
            json.dump(case_info, f, indent=4)

    logger.info("All cases processed")

    with open(output_metric_file, "r") as f:
        data = json.load(f)
    auc_results = compute_auc(data)
    auc_output_file = output_metric_file.replace(".json", "_AUC.json")
    with open(auc_output_file, "w") as f:
        json.dump(auc_results, f, indent=4)
    logger.info(f"AUC results saved to: {auc_output_file}")

    summary = summarise(
        data, auc_results, cases, args.max_iters, case_info=case_info,
        extra={
            "repo": repo,
            "predictor": predictor.name,
            "args": {k: (list(v) if isinstance(v, tuple) else v) for k, v in vars(args).items()},
            "total_seconds": round(time.time() - t_start, 2),
            "cache": (dict(cache.stats(),
                           level=("base" if args.predictor == "postproc" else "loop"),
                           model_calls=getattr(_base_cache_layer(predictor),
                                               "n_model_calls", None),
                           cache_calls=getattr(_base_cache_layer(predictor),
                                               "n_cache_calls", None))
                      if cache is not None else None),
            "postproc_config": getattr(predictor, "resolved_config", None),
            "guarantees": ({
                "checked": True,
                "enforced": list(guarantee_scope),
                "cases_with_violations": [t for t in case_dict
                                          if case_info.get(t, {}).get("guarantee_violations")],
                "n_iterations_checked": int(sum(
                    len(case_info.get(t, {}).get("guarantees", []) or []) for t in case_dict)),
                "n_iterations_violating": int(sum(
                    len(case_info.get(t, {}).get("guarantee_violations", []) or [])
                    for t in case_dict)),
                # measured whatever is enforced: how often the mask misses a tumor
                # scribble or swallows a background one
                "n_iterations_fg_outside": int(sum(
                    1 for t in case_dict
                    for g in (case_info.get(t, {}).get("guarantees", []) or [])
                    if g.get("n_tumor_outside"))),
                "n_iterations_bg_inside": int(sum(
                    1 for t in case_dict
                    for g in (case_info.get(t, {}).get("guarantees", []) or [])
                    if g.get("n_background_inside"))),
                "n_iterations_with_fg_points": int(sum(
                    1 for t in case_dict
                    for g in (case_info.get(t, {}).get("guarantees", []) or [])
                    if g.get("n_tumor"))),
                "n_iterations_with_bg_points": int(sum(
                    1 for t in case_dict
                    for g in (case_info.get(t, {}).get("guarantees", []) or [])
                    if g.get("n_background"))),
            } if check_guarantees else {"checked": False}),
            "seconds_per_case": {t: case_info[t].get("case_seconds") for t in case_dict},
        },
    )
    with open(os.path.join(args.out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=4)

    logger.info(
        "SUMMARY  n=%d  mean AUC-Dice=%.4f  mean AUC-DMM=%.4f  (max %.1f)  "
        "lesion-free cases excluded from DMM: %d  total %.1fs"
        % (summary["n_cases"], summary["mean_auc_dice"], summary["mean_auc_dmm"],
           summary["max_auc"], summary["n_cases_with_nan_dmm"], summary["total_seconds"])
    )
    if cache is not None:
        c = summary["cache"]
        logger.info("CACHE    level=%s hits=%d misses=%d prob_upgrades=%d "
                    "model_calls=%s cached_calls=%s"
                    % (c["level"], c["hits"], c["misses"], c["prob_upgrades"],
                       c["model_calls"], c["cache_calls"]))
    if check_guarantees:
        g = summary["guarantees"]
        (logger.info if not g["n_iterations_violating"] else logger.error)(
            "GUARANTEES %d/%d scored iterations satisfy G1+G2%s"
            % (g["n_iterations_checked"] - g["n_iterations_violating"],
               g["n_iterations_checked"],
               "" if not g["cases_with_violations"]
               else f"; violated in {g['cases_with_violations']}"))
    predictor.close()
    return summary


if __name__ == "__main__":
    main()
