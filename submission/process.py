"""autoPET V Grand Challenge container entry point (`python -m submission.process`).

    /input/images/{ct,pet}/<uuid>.mha + /input/lesion-clicks.json
    -> /output/images/tumor-lesion-segmentation/<CT-uuid>.mha, uint8, input geometry

The final evaluation calls the container six times per case (iterations 0..5), the
preliminary phase once, so nothing here may depend on state existing.  Which model runs
is decided in `submission/predictor_gc.py`; every path is env-overridable (see `Paths`)
so the container can be exercised without Docker.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import sys
import time
import traceback
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

# --------------------------------------------------------------------------- #
# logging: single stream to stdout, which is what the GC job log captures
# --------------------------------------------------------------------------- #
logging.basicConfig(
    level=logging.INFO,
    stream=sys.stdout,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("autopetv")

VERSION = "v0.2 (fine-tuned 5-channel interactive nnU-Net + post-processing)"
MARKER_NAME = "autopetv_calls.json"
PREV_MASK_NAME = "prev_final_mask.npz"   # bit-packed; see save_packed_mask
MAX_ENTRIES_LOGGED = 50
N_ITERATIONS = 6          # 0..5 in the final offline evaluation (confirmed by the organizers)
BUDGET_S = 20 * 60
SEED = 42


# --------------------------------------------------------------------------- #
# paths
# --------------------------------------------------------------------------- #
def _env_bool(name: str, default: bool = False) -> bool:
    v = os.environ.get(name)
    return default if v is None else v.strip().lower() in ("1", "true", "yes", "on")


class Paths:
    def __init__(self) -> None:
        self.input = os.environ.get("AUTOPETV_INPUT_DIR", "/input")
        self.output = os.environ.get(
            "AUTOPETV_OUTPUT_DIR", "/output/images/tumor-lesion-segmentation"
        )
        # State roots, in read-preference order.  Both are used: /cache is the mount the
        # evaluation harness provides for inter-iteration storage and /output is also
        # documented as persisting, so we write to both and read from whichever one
        # actually carries this case's state.  It costs a few hundred kB.
        self.cache = os.environ.get("AUTOPETV_CACHE_DIR", "/cache")
        self.output_root = os.environ.get("AUTOPETV_OUTPUT_ROOT", "/output")
        explicit = os.environ.get("AUTOPETV_STATE_DIRS")
        single = os.environ.get("AUTOPETV_STATE_DIR") or os.environ.get("STATE_DIR")
        if explicit:
            roots = [r for r in explicit.split(os.pathsep) if r.strip()]
        elif single:
            roots = [single]
        else:
            roots = [os.path.join(self.cache, "state"),
                     os.path.join(self.output_root, "state")]
        # de-duplicate while keeping order (a user may point both envs at one path)
        seen, self.state_roots = set(), []
        for r in roots:
            r = os.path.normpath(r)
            if r not in seen:
                seen.add(r)
                self.state_roots.append(r)
        self.state_enabled = _env_bool("AUTOPETV_STATE_ENABLED", True)
        self.save_prev_mask = _env_bool("AUTOPETV_SAVE_PREV_MASK", True)
        tmp = os.environ.get("AUTOPETV_TMP_DIR")
        if not tmp:
            tmp = os.path.join(os.environ.get("TMPDIR", "/tmp"), "autopetv")
        self.tmp = tmp

    def describe(self) -> Dict[str, object]:
        return {
            "input": self.input,
            "output": self.output,
            "state_roots": self.state_roots,
            "cache": self.cache,
            "state_enabled": self.state_enabled,
            "save_prev_mask": self.save_prev_mask,
            "tmp": self.tmp,
        }


# --------------------------------------------------------------------------- #
# input discovery
# --------------------------------------------------------------------------- #
def _one_image(dir_path: str, what: str) -> str:
    """Path of the single image file in `dir_path`; raises if there is not exactly one.

    Deliberately not `os.listdir(...)[0]`: that order is undefined if a stray file
    ever appears next to the image.
    """
    if not os.path.isdir(dir_path):
        raise FileNotFoundError(f"{what} directory does not exist: {dir_path}")
    names = sorted(
        n
        for n in os.listdir(dir_path)
        if not n.startswith(".") and os.path.isfile(os.path.join(dir_path, n))
    )
    if len(names) != 1:
        raise AssertionError(
            f"expected exactly ONE {what} file in {dir_path}, found {len(names)}: {names}"
        )
    name = names[0]
    if not name.lower().endswith((".mha", ".mhd", ".nii", ".nii.gz")):
        logger.warning("[input] unexpected %s extension: %s", what, name)
    return os.path.join(dir_path, name)


def _stem(path: str) -> str:
    base = os.path.basename(path)
    return base[:-7] if base.lower().endswith(".nii.gz") else os.path.splitext(base)[0]


def find_inputs(input_dir: str) -> Tuple[str, str, str]:
    """-> (ct_path, pet_path, uuid).  `uuid` is the CT stem -- NOT the PET's."""
    ct = _one_image(os.path.join(input_dir, "images", "ct"), "CT")
    pet = _one_image(os.path.join(input_dir, "images", "pet"), "PET")
    uuid = _stem(ct)
    logger.info("[input] CT   %s", ct)
    logger.info("[input] PET  %s", pet)
    logger.info("[input] uuid (from CT, used for the output filename) = %s", uuid)
    if _stem(pet) != uuid:
        logger.info(
            "[input] note: PET stem %r differs from CT stem %r -- using the CT stem, "
            "as the organizers' baseline does",
            _stem(pet), uuid,
        )
    return ct, pet, uuid


# --------------------------------------------------------------------------- #
# scribbles
# --------------------------------------------------------------------------- #
def read_scribbles(input_dir: str) -> Tuple[Dict[str, List[List[int]]], Dict[str, object]]:
    """`lesion-clicks.json` -> {"tumor": [[i,j,k],...], "background": [...]}.

    Accepts the Grand Challenge "Multiple points" format and the flat swFastEdit
    format; a missing, empty or malformed file means "no scribbles yet", not an error.
    """
    path = os.path.join(input_dir, "lesion-clicks.json")
    scribbles: Dict[str, List[List[int]]] = {"tumor": [], "background": []}
    info: Dict[str, object] = {"clicks_file": path, "clicks_file_present": os.path.isfile(path)}

    if not info["clicks_file_present"]:
        logger.info("[clicks] %s absent -> iteration 0 (no scribbles)", path)
        return scribbles, info

    try:
        with open(path) as f:
            raw = json.load(f)
    except Exception as exc:
        logger.warning("[clicks] unreadable (%r) -> treating as no scribbles", exc)
        info["clicks_file_error"] = repr(exc)
        return scribbles, info

    unknown = 0
    if isinstance(raw, dict) and "points" in raw:
        info["clicks_format"] = "grand-challenge"
        info["clicks_version"] = raw.get("version")
        for p in raw.get("points") or []:
            try:
                name = str(p.get("name", "")).lower()
                pt = [int(round(float(c))) for c in p["point"][:3]]
            except Exception:
                unknown += 1
                continue
            if name in ("tumor", "tumour", "foreground", "fg"):
                scribbles["tumor"].append(pt)
            elif name in ("background", "bg", "non-tumor"):
                scribbles["background"].append(pt)
            else:
                unknown += 1
    elif isinstance(raw, dict) and ("tumor" in raw or "background" in raw):
        info["clicks_format"] = "swfastedit"
        for key in ("tumor", "background"):
            for p in raw.get(key) or []:
                try:
                    scribbles[key].append([int(round(float(c))) for c in list(p)[:3]])
                except Exception:
                    unknown += 1
    else:
        info["clicks_format"] = "unrecognised"
        logger.warning("[clicks] unrecognised JSON shape -> treating as no scribbles")

    info["n_tumor"] = len(scribbles["tumor"])
    info["n_background"] = len(scribbles["background"])
    info["n_unparsed"] = unknown
    logger.info(
        "[clicks] format=%s tumor=%d background=%d unparsed=%d",
        info.get("clicks_format"), info["n_tumor"], info["n_background"], unknown,
    )
    if unknown:
        logger.warning("[clicks] %d point(s) could not be parsed and were dropped", unknown)
    return scribbles, info


def clip_scribbles(
    scribbles: Dict[str, List[List[int]]], shape: Sequence[int]
) -> Dict[str, List[List[int]]]:
    """Drop points outside the volume and log them.

    `generate_gaussian_heatmap` bounds-checks silently, so an out-of-range point would
    otherwise disappear without trace.
    """
    out: Dict[str, List[List[int]]] = {}
    for key, pts in scribbles.items():
        keep, drop = [], 0
        for p in pts:
            if all(0 <= int(p[a]) < int(shape[a]) for a in range(3)):
                keep.append([int(p[0]), int(p[1]), int(p[2])])
            else:
                drop += 1
        if drop:
            logger.warning("[clicks] %d %s point(s) outside %s -- dropped", drop, key, tuple(shape))
        out[key] = keep
    return out


# --------------------------------------------------------------------------- #
# geometry / IO
# --------------------------------------------------------------------------- #
def geometry_of(img) -> Dict[str, object]:
    return {
        "size": list(img.GetSize()),
        "spacing": [round(float(s), 9) for s in img.GetSpacing()],
        "origin": [round(float(o), 9) for o in img.GetOrigin()],
        "direction": [round(float(d), 9) for d in img.GetDirection()],
    }


def write_mask_mha(mask: np.ndarray, ref_img, out_path: str) -> None:
    """Write a nibabel-space (i, j, k) uint8 mask as .mha with `ref_img`'s geometry.

    `mask[i, j, k]` is SimpleITK index (i, j, k) is `GetArrayFromImage()[k, j, i]`,
    hence the (2, 1, 0) transpose.  `CopyInformation` raises on a size mismatch.
    """
    import SimpleITK as sitk

    arr = np.ascontiguousarray(np.transpose(np.asarray(mask, dtype=np.uint8), (2, 1, 0)))
    img = sitk.GetImageFromArray(arr)
    img.CopyInformation(ref_img)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    sitk.WriteImage(img, out_path, True)  # useCompression=True, as the baseline does


def mha_to_nii(src_mha: str, dst_nii: str) -> None:
    """Exactly the organizers' `convert_mha_to_nii` (lossless, compression only)."""
    import SimpleITK as sitk

    if not src_mha or not os.path.isfile(src_mha):
        # SimpleITK is a SWIG binding: a None/garbage path can take the whole
        # interpreter down with SIGSEGV instead of raising.  Check first.
        raise FileNotFoundError(f"cannot convert, no such file: {src_mha!r}")
    sitk.WriteImage(sitk.ReadImage(src_mha), dst_nii, True)


# --------------------------------------------------------------------------- #
# state / persistence
# --------------------------------------------------------------------------- #
def inspect_state_dir(path: str, label: str) -> Dict[str, object]:
    """Snapshot a candidate state directory before we touch it."""
    rep: Dict[str, object] = {"label": label, "dir": path, "exists": os.path.isdir(path)}
    entries: List[dict] = []
    if rep["exists"]:
        for root, _dirs, files in os.walk(path):
            for fn in files:
                fp = os.path.join(root, fn)
                try:
                    st = os.stat(fp)
                    entries.append(
                        {
                            "path": os.path.relpath(fp, path),
                            "bytes": st.st_size,
                            "mtime": round(st.st_mtime, 3),
                        }
                    )
                except OSError:
                    pass
        entries.sort(key=lambda e: e["mtime"])
    rep["n_entries"] = len(entries)
    rep["entries"] = entries[:MAX_ENTRIES_LOGGED]

    # writability probe (creates the dir if needed -- harmless, and tells us early)
    try:
        os.makedirs(path, exist_ok=True)
        probe = os.path.join(path, ".autopetv_write_probe")
        with open(probe, "w") as f:
            f.write("ok")
        os.remove(probe)
        rep["writable"] = True
    except Exception as exc:
        rep["writable"] = False
        rep["write_error"] = repr(exc)

    logger.info(
        "[%s] %s exists=%s writable=%s pre-existing files=%d",
        label, path, rep["exists"], rep.get("writable"), len(entries),
    )
    for e in entries[:MAX_ENTRIES_LOGGED]:
        logger.info("[%s]   pre-existing: %s (%d B, mtime %.0f)", label, e["path"], e["bytes"], e["mtime"])
    return rep


def read_marker(state_dir: str) -> List[dict]:
    path = os.path.join(state_dir, MARKER_NAME)
    if not os.path.isfile(path):
        return []
    try:
        with open(path) as f:
            data = json.load(f)
        return data.get("calls", []) if isinstance(data, dict) else []
    except Exception as exc:
        logger.warning("[state] marker unreadable at %s: %r", path, exc)
        return []


def write_marker(state_dir: str, calls: List[dict], label: str) -> Optional[str]:
    try:
        os.makedirs(state_dir, exist_ok=True)
        path = os.path.join(state_dir, MARKER_NAME)
        tmp = path + ".tmp"
        with open(tmp, "w") as f:
            json.dump({"schema": 1, "calls": calls}, f, indent=1)
        os.replace(tmp, path)
        # one standalone file per call as well, so individual calls are still traceable
        # if the aggregate marker is clobbered
        with open(os.path.join(state_dir, f"call_{len(calls):02d}_{calls[-1]['uuid'][:8]}.json"), "w") as f:
            json.dump(calls[-1], f, indent=1)
        logger.info("[%s] marker written: %s (%d call(s) recorded)", label, path, len(calls))
        return path
    except Exception as exc:
        logger.warning("[%s] could NOT write marker in %s: %r", label, state_dir, exc)
        return None


# --------------------------------------------------------------------------- #
# per-case state: two roots, one truth
# --------------------------------------------------------------------------- #
CASE_MARKER = "autopetv_case.json"
POSTPROC_FILES = (
    "postproc_constraints.json",
    "postproc_prev_mask.npz",
    "postproc_prev_prob.npy",
    "postproc_bg_region.npz",
)


def _read_case_marker(case_dir: str) -> Optional[dict]:
    path = os.path.join(case_dir, CASE_MARKER)
    if not os.path.isfile(path):
        return None
    try:
        with open(path) as f:
            data = json.load(f)
        return data if isinstance(data, dict) else None
    except Exception as exc:
        logger.warning("[state] unreadable case marker %s: %r", path, exc)
        return None


def _case_evidence(case_dir: str, fp: str) -> Tuple[int, float, int]:
    """How much state this candidate carries -> (n_calls, mtime, n_files).

    `mtime` is for diagnostics only; it does not rank candidates, because mirroring
    rewrites the copy last and it would always win.  A directory whose marker names a
    different case fingerprint scores 0 rather than serving the wrong previous mask.
    """
    if not os.path.isdir(case_dir):
        return (-1, 0.0, 0)
    marker = _read_case_marker(case_dir)
    n_calls = 0
    mtime = 0.0
    if marker is not None:
        if marker.get("case_fingerprint") not in (None, fp):
            logger.warning("[state] %s holds fingerprint %s, expected %s -- ignoring it",
                           case_dir, marker.get("case_fingerprint"), fp)
            return (-1, 0.0, 0)
        n_calls = int(marker.get("n_calls", 0) or 0)
    try:
        names = [n for n in os.listdir(case_dir) if not n.startswith(".")]
        mtime = max([os.path.getmtime(os.path.join(case_dir, n)) for n in names] or [0.0])
    except OSError:
        names = []
    return (n_calls, mtime, len(names))


def resolve_case_dirs(roots: Sequence[str], fp: str) -> Tuple[Optional[str], List[str]]:
    """-> (read_dir, writable_dirs) for case fingerprint `fp`.

    `read_dir` is the candidate carrying the most previous calls, and is also the
    directory handed to the predictor.  With no state anywhere, the first writable
    candidate is used so iteration 0 writes to the preferred root.
    """
    writable: List[str] = []
    scored: List[Tuple[Tuple[int, int, int], str]] = []
    for idx, root in enumerate(roots):
        case_dir = os.path.join(root, "case_" + fp)
        evidence = _case_evidence(case_dir, fp)
        if evidence[0] >= 0:
            # rank by recorded call count, then by how much state is there, then by the
            # root's position in the preference list
            scored.append(((evidence[0], evidence[2], -idx), case_dir))
        # writability is decided by actually creating the directory: a read-only mount
        # and a missing mount are indistinguishable until you try.
        try:
            os.makedirs(case_dir, exist_ok=True)
            probe = os.path.join(case_dir, ".autopetv_write_probe")
            with open(probe, "w") as f:
                f.write("ok")
            os.remove(probe)
            writable.append(case_dir)
        except Exception as exc:
            logger.info("[state] %s is not writable (%r)", case_dir, exc)
        logger.info("[state] candidate %s: n_calls=%s files=%s writable=%s",
                    case_dir, evidence[0] if evidence[0] >= 0 else "-", evidence[2],
                    case_dir in writable)

    best = [c for c in scored if c[0][0] > 0 or c[0][1] > 0]
    if best:
        best.sort(key=lambda c: c[0], reverse=True)
        read_dir = best[0][1]
        if read_dir not in writable and writable:
            # the surviving state is on a read-only root: copy it somewhere writable so
            # the predictor can update it, and keep reading from the copy
            target = writable[0]
            _mirror_case_dir(read_dir, [target])
            logger.info("[state] state found in read-only %s -- mirrored into %s",
                        read_dir, target)
            read_dir = target
        return read_dir, writable
    return (writable[0] if writable else None), writable


def _mirror_case_dir(src: str, targets: Sequence[str]) -> List[str]:
    """Copy every state file from `src` into each target directory, atomically."""
    done = []
    if not src or not os.path.isdir(src):
        return done
    try:
        names = sorted(n for n in os.listdir(src)
                       if not n.startswith(".") and os.path.isfile(os.path.join(src, n)))
    except OSError:
        return done
    for target in targets:
        if os.path.normpath(target) == os.path.normpath(src):
            continue
        try:
            os.makedirs(target, exist_ok=True)
            for n in names:
                dst = os.path.join(target, n)
                tmp = dst + ".tmp"
                shutil.copyfile(os.path.join(src, n), tmp)
                os.replace(tmp, dst)
            done.append(target)
        except Exception as exc:
            logger.warning("[state] could not mirror %s -> %s: %r", src, target, exc)
    return done


def save_packed_mask(path: str, mask: np.ndarray) -> int:
    """Bit-pack a binary volume the way `postproc/constraints.py` does.

    A raw uint8 `.npy` of a 45 M-voxel case is 45 MB, bit-packed and deflated ~6 kB,
    and this is written into /output at all six iterations of every case.
    """
    a = np.ascontiguousarray(np.asarray(mask).astype(bool))
    tmp = path + ".tmp"
    with open(tmp, "wb") as fh:
        np.savez_compressed(fh, packed=np.packbits(a.reshape(-1)),
                            shape=np.asarray(a.shape, np.int64))
    os.replace(tmp, path)
    return os.path.getsize(path)


def load_packed_mask(path: str) -> Optional[np.ndarray]:
    if not os.path.isfile(path):
        return None
    try:
        with np.load(path) as z:
            shape = tuple(int(x) for x in z["shape"])
            n = int(np.prod(shape))
            return np.unpackbits(z["packed"], count=n).reshape(shape).astype(np.uint8)
    except Exception as exc:
        logger.warning("[state] unreadable packed mask %s: %r", path, exc)
        return None


def case_fingerprint(geom: Dict[str, object], pet: Optional[np.ndarray]) -> str:
    """Stable id for "the same case", independent of the random per-call uuid.

    Geometry alone almost always separates cases; the strided PET digest costs a few
    ms and makes a collision very unlikely.
    """
    h = hashlib.sha1()
    h.update(json.dumps(geom, sort_keys=True).encode())
    if pet is not None:
        try:
            sub = np.ascontiguousarray(np.asarray(pet)[::7, ::7, ::7], dtype=np.float32)
            h.update(sub.tobytes())
        except Exception:
            pass
    return h.hexdigest()[:16]


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main() -> int:
    t_start = time.time()
    timings: Dict[str, float] = {}
    paths = Paths()

    logger.info("=" * 78)
    logger.info("autoPET V submission container -- %s", VERSION)
    logger.info("paths: %s", json.dumps(paths.describe()))
    logger.info(
        "env: nnUNet_results=%s AUTOPETV_MODEL_FOLDER=%s nnUNet_extTrainer=%s",
        os.environ.get("nnUNet_results"), os.environ.get("AUTOPETV_MODEL_FOLDER"),
        os.environ.get("nnUNet_extTrainer"),
    )
    logger.info("python %s", sys.version.replace("\n", " "))

    # --- GPU banner + determinism -----------------------------------------
    try:
        import random

        import torch

        avail = torch.cuda.is_available()
        logger.info("[gpu] torch %s cuda_available=%s device_count=%d",
                    torch.__version__, avail, torch.cuda.device_count())
        if avail:
            props = torch.cuda.get_device_properties(0)
            logger.info("[gpu] %s, %.1f GB, sm_%d%d",
                        props.name, props.total_memory / 2**30, props.major, props.minor)
        else:
            logger.error("[gpu] NO CUDA DEVICE -- inference will be far slower than %d s", BUDGET_S)

        # cuDNN's autotuner picks a convolution algorithm based on what else the machine
        # is doing, and the algorithms differ in float summation order -- enough to flip
        # voxels on the decision boundary between two runs of the same input.
        random.seed(SEED)
        np.random.seed(SEED)
        torch.manual_seed(SEED)
        if avail:
            torch.cuda.manual_seed_all(SEED)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        logger.info("[det] seed=%d cudnn.benchmark=False cudnn.deterministic=True", SEED)
    except Exception:
        logger.warning("[gpu] torch probe failed:\n%s", traceback.format_exc())

    # --- state snapshot before we write anything ---------------------------
    state_reports: List[dict] = []
    prev_calls: List[dict] = []
    if paths.state_enabled:
        for path in paths.state_roots:
            label = "state:" + path
            rep = inspect_state_dir(path, label)
            state_reports.append(rep)
            found = read_marker(path)
            rep["marker_calls"] = len(found)
            logger.info("[%s] marker history: %d previous call(s)", label, len(found))
            for c in found[-6:]:
                logger.info(
                    "[%s]   prev call #%s uuid=%s case=%s n_points=%s mask_voxels=%s at %s",
                    label, c.get("call_index"), c.get("uuid"), c.get("case_fingerprint"),
                    c.get("n_points"), c.get("mask_voxels"), c.get("utc"),
                )
            if len(found) > len(prev_calls):
                prev_calls = found
        logger.info("[state] roots in read-preference order: %s", paths.state_roots)
    else:
        logger.info("[state] AUTOPETV_STATE_ENABLED=0 -- no state is read or written")

    # --- inputs ------------------------------------------------------------
    import SimpleITK as sitk

    ct_path = pet_path = None
    uuid = "unknown"
    ref_img = None
    try:
        ct_path, pet_path, uuid = find_inputs(paths.input)
        ref_img = sitk.ReadImage(ct_path)
    except Exception:
        logger.error("[input] FAILED to read the CT:\n%s", traceback.format_exc())
        # last resort: try the PET, so we can still emit a correctly-shaped empty mask
        try:
            pet_only = _one_image(os.path.join(paths.input, "images", "pet"), "PET")
            ref_img = sitk.ReadImage(pet_only)
            if uuid == "unknown":
                uuid = _stem(pet_only)
            logger.warning("[input] falling back to PET geometry; uuid=%s", uuid)
        except Exception:
            logger.error(
                "[input] no readable image at all -- cannot write an output of the right "
                "geometry.  Giving up (exit 0 so the job is not marked crashed).\n%s",
                traceback.format_exc(),
            )
            return 0

    geom = geometry_of(ref_img)
    logger.info("[geom] size=%s spacing=%s", geom["size"], geom["spacing"])
    logger.info("[geom] origin=%s", geom["origin"])
    logger.info("[geom] direction=%s", geom["direction"])

    os.makedirs(paths.output, exist_ok=True)  # also created at build time in the Dockerfile
    out_path = os.path.join(paths.output, uuid + ".mha")

    # shape in nibabel index order == SimpleITK size (both (i, j, k))
    shape = tuple(int(s) for s in geom["size"])

    scribbles, clicks_info = read_scribbles(paths.input)
    scribbles = clip_scribbles(scribbles, shape)
    n_points = len(scribbles["tumor"]) + len(scribbles["background"])

    # --- model path (everything below is allowed to fail) -------------------
    mask: Optional[np.ndarray] = None
    pet_arr: Optional[np.ndarray] = None
    error: Optional[str] = None
    predictor = None
    fp: Optional[str] = None          # case fingerprint; needs the PET, so set below
    case_dir: Optional[str] = None            # the per-case dir we READ and hand over
    case_dirs_writable: List[str] = []        # every per-case dir we can WRITE
    prev_hint: Optional[np.ndarray] = None    # redundant channel-4 fallback
    tmp_dir = os.path.join(paths.tmp, "case")
    try:
        if ct_path is None or pet_path is None:
            # We got here through the PET-geometry fallback: input discovery failed, so
            # there is nothing sane to feed the network.  Skip straight to the empty
            # mask.  (Handing a None path to SimpleITK segfaults the interpreter.)
            raise RuntimeError(
                "input discovery failed (see the traceback above) -- skipping the model "
                "path and writing an empty mask of the correct geometry"
            )
        os.makedirs(tmp_dir, exist_ok=True)
        conv_dir = os.path.join(tmp_dir, "conv")
        shutil.rmtree(conv_dir, ignore_errors=True)
        os.makedirs(conv_dir, exist_ok=True)

        # 1. .mha -> .nii.gz, exactly as the organizers' container does, so that
        #    nnU-Net sees identical input to the reference implementation.
        t0 = time.time()
        ct_nii = os.path.join(conv_dir, "case_0000.nii.gz")
        pet_nii = os.path.join(conv_dir, "case_0001.nii.gz")
        mha_to_nii(ct_path, ct_nii)
        mha_to_nii(pet_path, pet_nii)
        timings["convert_s"] = time.time() - t0
        logger.info("[convert] mha->nii.gz in %.1f s", timings["convert_s"])

        # 2. load in nibabel index space (convention A of src/predictor.py)
        t0 = time.time()
        import nibabel as nib

        ct_img_nib = nib.load(ct_nii)
        pet_img_nib = nib.load(pet_nii)
        ct_arr = np.asanyarray(ct_img_nib.dataobj)
        pet_arr = np.asanyarray(pet_img_nib.dataobj)
        affine = pet_img_nib.affine
        spacing = tuple(float(z) for z in pet_img_nib.header.get_zooms()[:3])
        timings["load_s"] = time.time() - t0
        logger.info(
            "[load] ct%s %s  pet%s %s  spacing=%s  in %.1f s",
            ct_arr.shape, ct_arr.dtype, pet_arr.shape, pet_arr.dtype, spacing, timings["load_s"],
        )
        if ct_arr.shape != pet_arr.shape:
            raise ValueError(f"CT shape {ct_arr.shape} != PET shape {pet_arr.shape}")
        if tuple(ct_arr.shape) != shape:
            raise ValueError(f"nibabel shape {ct_arr.shape} != SimpleITK size {shape}")
        logger.info(
            "[load] PET SUV min/mean/max = %.3f / %.3f / %.3f",
            float(pet_arr.min()), float(pet_arr.mean()), float(pet_arr.max()),
        )

        # 3. the per-case state directory, holding the previous iteration's final mask,
        #    probability and constraint set.  The per-call uuid is random, so the case is
        #    identified by its fingerprint instead.  Missing or unwritable is supported:
        #    every consumer degrades to iteration 0.
        fp = case_fingerprint(geom, pet_arr)
        if paths.state_enabled:
            case_dir, case_dirs_writable = resolve_case_dirs(paths.state_roots, fp)
        if case_dir:
            have = sorted(n for n in os.listdir(case_dir) if not n.startswith("."))
            logger.info("[state] reading case state from %s: %d file(s) from previous "
                        "iterations: %s", case_dir, len(have), have)
            logger.info("[state] will mirror the result into %d writable root(s): %s",
                        len(case_dirs_writable), case_dirs_writable)
        else:
            logger.info("[state] no writable per-case state directory in %s -- every "
                        "call behaves like iteration 0", paths.state_roots)

        # Redundant channel-4 fallback, consulted only when the post-processing layer's
        # own cached mask is missing.
        prev_hint = None
        if case_dir and not os.path.isfile(os.path.join(case_dir, "postproc_prev_mask.npz")):
            prev_hint = load_packed_mask(os.path.join(case_dir, PREV_MASK_NAME))
            if prev_hint is not None and prev_hint.shape != shape:
                logger.warning("[state] %s has shape %s, expected %s -- ignoring it",
                               PREV_MASK_NAME, prev_hint.shape, shape)
                prev_hint = None
            if prev_hint is not None:
                logger.info("[state] post-processing cache absent -- feeding channel 4 "
                            "from %s (%d voxels)", PREV_MASK_NAME, int(prev_hint.sum()))

        # 4. predictor (imported from src/, never copied)
        from submission.predictor_gc import base_of, build_predictor, predictor_config

        cfg = predictor_config()
        cfg["tmp_dir"] = cfg["tmp_dir"] or os.path.join(tmp_dir, "nnunet")
        logger.info("[predictor] config=%s", json.dumps(cfg, default=str))
        t0 = time.time()
        predictor = build_predictor(cfg, logger=logger)
        if hasattr(predictor, "warmup"):
            predictor.warmup()
        timings["model_load_s"] = time.time() - t0
        logger.info("[predictor] %s ready in %.1f s",
                    getattr(predictor, "name", "?"), timings["model_load_s"])

        # 5. inference.  A container call is a fresh process with nothing in memory to
        #    pass, so the previous iteration's final mask reaches the model's fifth
        #    channel through `case_cache_dir`: the post-processing layer's cached mask
        #    (`pass_cached_prev_pred`), or `prev_final_mask.npz` as a fallback.  Offline
        #    the harness hands the same array over as `prev_pred`.
        t0 = time.time()
        mask = predictor.predict(
            ct_arr, pet_arr, spacing, scribbles,
            prev_pred=prev_hint,
            case_cache_dir=case_dir,
            affine=affine,
            ct_path=ct_nii,
            pet_path=pet_nii,
            case_name=uuid,
        )
        if isinstance(mask, tuple):
            mask = mask[0]
        mask = np.asarray(mask, dtype=np.uint8)
        timings["inference_s"] = time.time() - t0
        logger.info("[predict] done in %.1f s", timings["inference_s"])
        base = base_of(predictor)
        sub = getattr(base, "last_timings", None) or getattr(predictor, "last_timings", None)
        if sub:
            logger.info("[predict] sub-timings: %s",
                        json.dumps({k: round(v, 2) if isinstance(v, float) else v
                                    for k, v in sub.items()}))
        guid = getattr(base, "last_guidance_info", None)
        if guid:
            logger.info("[predict] guidance: %s", json.dumps(guid, default=str))
            if guid.get("prev_pred_source") == "none" and n_points > 0:
                logger.warning(
                    "[predict] scribbles are present (iteration >= 1) but the "
                    "previous-prediction channel is empty -- the state directory did "
                    "not persist, so the model is running as if at iteration 0"
                )
        pinfo = getattr(predictor, "last_info", None)
        if pinfo:
            logger.info("[postproc] %s", json.dumps(
                {k: pinfo.get(k) for k in (
                    "iteration", "tracer", "n_tumor", "n_background", "state_available",
                    "base_volume_ml", "final_volume_ml", "negative_gate_fired",
                    "empty_output", "empty_without_gate", "constraints", "t_total")
                 if k in pinfo}, default=str))
        if mask.shape != shape:
            raise ValueError(f"prediction shape {mask.shape} != input shape {shape}")

    except Exception:
        error = traceback.format_exc()
        logger.error("[predict] MODEL PATH FAILED -- falling back to an empty mask")
        logger.error("%s", error)
        mask = None
    finally:
        if predictor is not None and hasattr(predictor, "close"):
            try:
                predictor.close()
            except Exception:
                pass

    if mask is None:
        mask = np.zeros(shape, dtype=np.uint8)

    # --- write the output --------------------------------------------------
    n_vox = int(mask.sum())
    try:
        t0 = time.time()
        write_mask_mha(mask, ref_img, out_path)
        timings["write_s"] = time.time() - t0
        logger.info(
            "[output] %s  (%d B, %d positive voxels, %.4f %% of volume) in %.1f s",
            out_path, os.path.getsize(out_path), n_vox,
            100.0 * n_vox / max(1, mask.size), timings["write_s"],
        )
        chk = geometry_of(sitk.ReadImage(out_path))
        if chk != geom:
            logger.error("[output] GEOMETRY MISMATCH!\n  in : %s\n  out: %s", geom, chk)
        else:
            logger.info("[output] geometry verified identical to the input CT")
    except Exception:
        logger.error("[output] FAILED to write %s:\n%s", out_path, traceback.format_exc())

    # --- the previous-prediction channel of the next iteration --------------
    # Only when the model path succeeded: an empty fallback mask must never be fed back
    # as "what we predicted last time".
    if case_dir and paths.save_prev_mask and error is None:
        try:
            t0 = time.time()
            n_bytes = save_packed_mask(os.path.join(case_dir, PREV_MASK_NAME), mask)
            timings["prev_mask_write_s"] = time.time() - t0
            logger.info("[state] previous-mask file written: %s (%d B, bit-packed) in %.2f s",
                        os.path.join(case_dir, PREV_MASK_NAME), n_bytes,
                        timings["prev_mask_write_s"])
        except Exception as exc:
            logger.warning("[state] could NOT write %s: %r (harmless: the "
                           "post-processing cache carries the same mask)", PREV_MASK_NAME, exc)

    # --- mirror this case's state into every other writable root ------------
    # whichever root survives to the next call then carries the same state
    if case_dir and paths.state_enabled:
        try:
            t0 = time.time()
            marker = {
                "schema": 1,
                "case_fingerprint": fp,
                "n_calls": int((_read_case_marker(case_dir) or {}).get("n_calls", 0)) + 1,
                "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "uuid": uuid,
                "shape": list(shape),
                "n_points": n_points,
                "mask_voxels": n_vox,
                "model_path_ok": error is None,
                "read_from": case_dir,
            }
            tmp_marker = os.path.join(case_dir, CASE_MARKER + ".tmp")
            with open(tmp_marker, "w") as f:
                json.dump(marker, f, indent=1)
            os.replace(tmp_marker, os.path.join(case_dir, CASE_MARKER))
            mirrored = _mirror_case_dir(case_dir, case_dirs_writable)
            timings["state_mirror_s"] = time.time() - t0
            logger.info("[state] call %d for this case; mirrored %s -> %s in %.2f s",
                        marker["n_calls"], case_dir, mirrored or "(nothing to mirror)",
                        timings["state_mirror_s"])
        except Exception:
            logger.warning("[state] mirroring failed (state is still usable in %s):\n%s",
                           case_dir, traceback.format_exc())

    # --- state marker ------------------------------------------------------
    if paths.state_enabled:
        if fp is None:
            fp = case_fingerprint(geom, pet_arr)
        same_case = [c for c in prev_calls if c.get("case_fingerprint") == fp]
        record = {
            "call_index": len(prev_calls),
            "iteration_guess": len(same_case),  # == the real iteration iff state persists
            "uuid": uuid,
            "case_fingerprint": fp,
            "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "shape": list(shape),
            "spacing": geom["spacing"],
            "n_tumor": len(scribbles["tumor"]),
            "n_background": len(scribbles["background"]),
            "n_points": n_points,
            "mask_voxels": n_vox,
            "predictor": os.environ.get("AUTOPETV_PREDICTOR", "interactive_postproc"),
            "case_dir": case_dir,
            "case_dirs_writable": case_dirs_writable,
            "case_dir_files": (sorted(os.listdir(case_dir)) if case_dir and os.path.isdir(case_dir)
                               else None),
            "model_path_ok": error is None,
            "runtime_s": round(time.time() - t_start, 2),
            "dirs_seen_before": {r["label"]: r.get("n_entries", 0) for r in state_reports},
        }
        logger.info(
            "[state] this call: n_points=%d -> iteration_guess=%d "
            "(stays 0 on every call == no persistence)",
            n_points, record["iteration_guess"],
        )
        if n_points > 0 and not same_case:
            logger.info(
                "[state] VERDICT SO FAR: scribbles are present (iteration >= 1) but no "
                "marker records this case -> the state dir did NOT persist."
            )
        if n_points == 0 and same_case:
            logger.warning(
                "[state] state persisted but this call has no scribbles -- new case with "
                "identical geometry, or the evaluator restarted the case."
            )
        for path in paths.state_roots:
            hist = read_marker(path)
            write_marker(path, hist + [record], "state:" + path)

    # --- scratch cleanup (never leave anything outside /output and /cache) --
    shutil.rmtree(tmp_dir, ignore_errors=True)

    timings["total_s"] = time.time() - t_start
    logger.info("[timing] %s", json.dumps({k: round(v, 2) for k, v in timings.items()}))
    try:
        import resource

        logger.info(
            "[mem] peak RSS self=%.2f GB children=%.2f GB",
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20,
            resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / 2**20,
        )
    except Exception:
        pass
    logger.info("[done] total %.1f s  (budget %d s per iteration, %d iterations/case)  status=%s",
                timings["total_s"], BUDGET_S, N_ITERATIONS,
                "ok" if error is None else "EMPTY-MASK-FALLBACK")
    logger.info("=" * 78)
    return 0


if __name__ == "__main__":
    # Always exit 0: an empty mask scores 0 for one iteration, a non-zero exit can fail
    # the whole job.
    try:
        code = main()
    except Exception:
        logger.error("[fatal] unhandled exception:\n%s", traceback.format_exc())
        code = 0
    sys.exit(code)
