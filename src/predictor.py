"""Predictor interface and implementations for the autoPET V harness.

One Predictor is built per run and kept alive across all cases and iterations.  Arrays
crossing the `predict` boundary are in nibabel index space (x, y, z, z = axial), the
shape of `nib.load(pet).get_fdata()`, and `spacing` is `header.get_zooms()` in that same
order -- not the reversed (z, y, x) order nnU-Net and `plans.json` use internally.
"""

from __future__ import annotations

import os
import shutil
import time
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import nibabel as nib
from scipy.ndimage import gaussian_filter

__all__ = [
    "ScribbleDict",
    "Predictor",
    "ThresholdPredictor",
    "BaselineNNUNetPredictor",
    "FastBaselineNNUNetPredictor",
    "InteractiveNNUNetPredictor",
    "generate_gaussian_heatmap",
    "affine_from_spacing",
]

# {"tumor": [[x, y, z], ...], "background": [[x, y, z], ...]}  -- nibabel indices
ScribbleDict = Dict[str, List[Sequence[int]]]


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def generate_gaussian_heatmap(coords, shape, sigma: float = 0.0) -> np.ndarray:
    """Re-implementation of the baseline's `utils.py::generate_gaussian_heatmap`.

    Not imported from the baseline, which does `import cupy` at module level.  `coords`
    are nibabel indices; with the baseline's sigma=0 the filter is a no-op and the result
    is a sparse binary float32 mask.
    """
    heatmap = np.zeros(shape, dtype=np.float32)
    for coord in coords:
        if 0 <= coord[0] < shape[0] and 0 <= coord[1] < shape[1] and 0 <= coord[2] < shape[2]:
            heatmap[tuple(int(c) for c in coord)] = 1.0
    heatmap = gaussian_filter(heatmap, sigma=sigma)
    return heatmap


def affine_from_spacing(spacing: Sequence[float], origin: Sequence[float] = (0.0, 0.0, 0.0)) -> np.ndarray:
    """Fallback RAS affine for synthetic data: LPS identity direction -> RAS."""
    sx, sy, sz = [float(s) for s in spacing]
    aff = np.eye(4, dtype=np.float64)
    aff[0, 0] = -sx
    aff[1, 1] = -sy
    aff[2, 2] = sz
    aff[:3, 3] = [-origin[0], -origin[1], origin[2]]
    return aff


def _empty_scribbles() -> ScribbleDict:
    return {"tumor": [], "background": []}


# ---------------------------------------------------------------------------
# interface
# ---------------------------------------------------------------------------
class Predictor(ABC):
    """One iteration of the challenge = one call to `predict`.

    `ct`, `pet` (SUV) and `prev_pred` are same-shaped 3D arrays in nibabel index space;
    `spacing` is in mm in that same (x, y, z) order.  `scribbles` are the accumulated
    scribble voxels, {"tumor": [[x, y, z], ...], "background": [...]}.  `case_cache_dir`
    survives the iterations of one case, like the container's /cache mount.  Returns a
    uint8 mask shaped like `pet`, or `(mask, probabilities)` of shape
    (n_classes, *pet.shape) if `return_probabilities`.
    """

    name = "predictor"
    # A stateless predictor is a pure function of (ct, pet, spacing, scribbles).  The
    # harness keys its prediction cache on the scribble set alone for these, so a
    # lesion-free case costs one inference instead of six.  Set False if state is carried
    # across iterations.
    stateless = True

    @abstractmethod
    def predict(
        self,
        ct: np.ndarray,
        pet: np.ndarray,
        spacing: Sequence[float],
        scribbles: ScribbleDict,
        prev_pred: Optional[np.ndarray] = None,
        case_cache_dir: Optional[str] = None,
        *,
        affine: Optional[np.ndarray] = None,
        ct_path: Optional[str] = None,
        pet_path: Optional[str] = None,
        case_name: str = "case",
        return_probabilities: bool = False,
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        ...

    def cache_state_key(self, prev_pred) -> Optional[str]:
        """Extra cache-key component for state this predictor consumes.

        The harness caches by (predictor, case, scribble set).  A predictor that also
        reads `prev_pred` must fold it in, or two iterations with the same scribbles but
        different previous masks collide.  None means there is nothing to add.
        """
        return None

    # optional hooks -------------------------------------------------------
    def warmup(self) -> None:
        """Load weights etc. so that the first `predict` is not penalised."""

    def close(self) -> None:
        """Release resources."""


# ---------------------------------------------------------------------------
# trivial predictor for smoke tests
# ---------------------------------------------------------------------------
class ThresholdPredictor(Predictor):
    """PET > `threshold` (SUV).  Ignores scribbles entirely -- smoke tests only."""

    name = "threshold"

    def __init__(self, threshold: float = 2.5, min_component_voxels: int = 0):
        self.threshold = float(threshold)
        self.min_component_voxels = int(min_component_voxels)

    def predict(
        self,
        ct,
        pet,
        spacing,
        scribbles,
        prev_pred=None,
        case_cache_dir=None,
        *,
        affine=None,
        ct_path=None,
        pet_path=None,
        case_name="case",
        return_probabilities=False,
    ):
        mask = (np.asarray(pet) > self.threshold).astype(np.uint8)
        if self.min_component_voxels > 0:
            import cc3d

            lab, n = cc3d.connected_components(mask, connectivity=26, return_N=True)
            if n:
                keep = np.zeros(n + 1, dtype=bool)
                counts = np.bincount(lab.reshape(-1), minlength=n + 1)
                keep[1:] = counts[1:] >= self.min_component_voxels
                mask = keep[lab].astype(np.uint8)
        if return_probabilities:
            p = mask.astype(np.float32)
            return mask, np.stack([1.0 - p, p], axis=0)
        return mask


# ---------------------------------------------------------------------------
# the official baseline, in-process
# ---------------------------------------------------------------------------
class BaselineNNUNetPredictor(Predictor):
    """The official `nnunet-baseline/process.py` pipeline, run in-process.

    Four input channels in nnU-Net order: _0000 CT, _0001 PET (SUV), _0002 tumor-click
    heatmap, _0003 background heatmap.  The channels go to disk as NIfTIs and through
    `predict_from_files`, so nnU-Net reads and writes them with one reader as it does in
    the baseline container.
    """

    name = "baseline_nnunet"

    DEFAULT_MODEL_FOLDER = (
        "/content/drive/MyDrive/autoPET/weights/nnUNet_results/"
        "Dataset998_AutoPETV/nnUNetTrainer__nnUNetPlans__3d_fullres"
    )

    def __init__(
        self,
        model_folder: Optional[str] = None,
        folds: Sequence[Union[int, str]] = (0,),
        checkpoint_name: str = "checkpoint_final.pth",
        device: str = "cuda",
        disable_tta: bool = True,          # == the baseline's `--disable_tta`
        use_mirroring: Optional[bool] = None,   # explicit override; None -> not disable_tta
        tile_step_size: float = 0.5,
        use_gaussian: bool = True,
        perform_everything_on_device: bool = True,
        num_processes_preprocessing: int = 3,   # nnUNetv2_predict default -npp
        num_processes_segmentation_export: int = 3,  # ... -nps
        sigma: float = 0.0,                # baseline save_click_heatmaps sigma
        verbose: bool = False,
        keep_tmp: bool = False,
        tmp_root: Optional[str] = None,
    ):
        self.model_folder = model_folder or os.environ.get(
            "AUTOPETV_MODEL_FOLDER", self.DEFAULT_MODEL_FOLDER
        )
        self.folds = tuple(folds)
        self.checkpoint_name = checkpoint_name
        self.device = device
        self.disable_tta = bool(disable_tta)
        # Mirroring TTA costs one forward pass per combination of the mirrored axes
        # (normally 8x the sliding window); the baseline runs `--disable_tta`.
        self.use_mirroring = (not self.disable_tta) if use_mirroring is None else bool(use_mirroring)
        self.tile_step_size = tile_step_size
        self.use_gaussian = use_gaussian
        self.perform_everything_on_device = perform_everything_on_device
        self.npp = int(num_processes_preprocessing)
        self.nps = int(num_processes_segmentation_export)
        self.sigma = float(sigma)
        self.verbose = verbose
        self.keep_tmp = keep_tmp
        self.tmp_root = tmp_root
        self._predictor = None
        self._reader_is_reversed: Optional[bool] = None
        self.last_timings: Dict[str, float] = {}

    # -- lazy init ---------------------------------------------------------
    def warmup(self) -> None:
        self._ensure_predictor()

    def _ensure_predictor(self):
        if self._predictor is not None:
            return self._predictor
        # nnunetv2 whines at import time if these are unset; the trained-model folder is
        # given by absolute path so the values themselves are irrelevant.
        for var in ("nnUNet_raw", "nnUNet_preprocessed", "nnUNet_results"):
            os.environ.setdefault(var, os.path.join(os.sep, "tmp", var))
        import torch
        from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

        t0 = time.time()
        pred = nnUNetPredictor(
            tile_step_size=self.tile_step_size,
            use_gaussian=self.use_gaussian,
            use_mirroring=self.use_mirroring,
            perform_everything_on_device=self.perform_everything_on_device,
            device=torch.device(self.device),
            verbose=self.verbose,
            verbose_preprocessing=self.verbose,
            allow_tqdm=self.verbose,
        )
        pred.initialize_from_trained_model_folder(
            self.model_folder, use_folds=self.folds, checkpoint_name=self.checkpoint_name
        )
        self._predictor = pred
        self.last_timings["model_load_s"] = time.time() - t0

        # which array order will nnU-Net use for the .npz probabilities?
        try:
            from nnunetv2.imageio.simpleitk_reader_writer import SimpleITKIO

            self._reader_is_reversed = issubclass(
                pred.plans_manager.image_reader_writer_class, SimpleITKIO
            )
        except Exception:
            self._reader_is_reversed = None
        return pred

    # -- main --------------------------------------------------------------
    def predict(
        self,
        ct,
        pet,
        spacing,
        scribbles,
        prev_pred=None,
        case_cache_dir=None,
        *,
        affine=None,
        ct_path=None,
        pet_path=None,
        case_name="case",
        return_probabilities=False,
    ):
        predictor = self._ensure_predictor()
        scribbles = scribbles or _empty_scribbles()
        pet = np.asarray(pet)
        shape = pet.shape
        if affine is None:
            affine = affine_from_spacing(spacing)

        root = self.tmp_root or case_cache_dir or os.path.join(os.sep, "tmp", "autopetv_pred")
        in_dir = os.path.join(root, "_nnunet_in")
        out_dir = os.path.join(root, "_nnunet_out")
        for d in (in_dir, out_dir):
            shutil.rmtree(d, ignore_errors=True)
            os.makedirs(d, exist_ok=True)

        stem = "case"  # keep nnU-Net's file names short and predictable
        f0 = os.path.join(in_dir, f"{stem}_0000.nii.gz")
        f1 = os.path.join(in_dir, f"{stem}_0001.nii.gz")
        f2 = os.path.join(in_dir, f"{stem}_0002.nii.gz")
        f3 = os.path.join(in_dir, f"{stem}_0003.nii.gz")

        t0 = time.time()
        self._materialise(ct, ct_path, affine, f0)
        self._materialise(pet, pet_path, affine, f1)

        # --- click heatmaps, as in utils.save_click_heatmaps --------------
        tumor_heatmap = generate_gaussian_heatmap(scribbles.get("tumor", []), shape, self.sigma)
        bg_heatmap = generate_gaussian_heatmap(scribbles.get("background", []), shape, self.sigma)
        nib.save(nib.Nifti1Image(tumor_heatmap, affine), f2)
        nib.save(nib.Nifti1Image(bg_heatmap, affine), f3)
        t_write = time.time() - t0

        # --- inference ----------------------------------------------------
        t0 = time.time()
        predictor.predict_from_files(
            [[f0, f1, f2, f3]],
            [os.path.join(out_dir, stem)],
            save_probabilities=bool(return_probabilities),
            overwrite=True,
            num_processes_preprocessing=self.npp,
            num_processes_segmentation_export=self.nps,
        )
        t_infer = time.time() - t0

        # --- read the result back (nibabel -> convention A) ---------------
        t0 = time.time()
        seg_file = os.path.join(out_dir, f"{stem}.nii.gz")
        if not os.path.isfile(seg_file):
            raise FileNotFoundError(f"nnUNet produced no segmentation at {seg_file}")
        mask = np.asanyarray(nib.load(seg_file).dataobj).astype(np.uint8)
        if mask.shape != shape:
            raise ValueError(f"nnUNet output shape {mask.shape} != input shape {shape}")

        probs = None
        if return_probabilities:
            probs = self._load_probabilities(os.path.join(out_dir, f"{stem}.npz"), shape)
        t_read = time.time() - t0

        self.last_timings.update(
            {"write_inputs_s": t_write, "inference_s": t_infer, "read_output_s": t_read}
        )
        if not self.keep_tmp:
            shutil.rmtree(in_dir, ignore_errors=True)
            shutil.rmtree(out_dir, ignore_errors=True)

        if return_probabilities:
            return mask, probs
        return mask

    # -- internals ---------------------------------------------------------
    @staticmethod
    def _materialise(array, src_path, affine, dst_path):
        """Put one channel on disk, linking the original file when we have one."""
        if src_path is not None and os.path.isfile(src_path) and src_path.endswith(".nii.gz"):
            try:
                os.symlink(os.path.abspath(src_path), dst_path)
            except OSError:
                shutil.copyfile(src_path, dst_path)
            return
        arr = np.asarray(array)
        if arr.dtype == np.float64:
            arr = arr.astype(np.float32)
        nib.save(nib.Nifti1Image(arr, affine), dst_path)

    def _load_probabilities(self, npz_path, shape):
        if not os.path.isfile(npz_path):
            raise FileNotFoundError(f"no softmax at {npz_path}")
        with np.load(npz_path) as z:
            probs = z["probabilities"]
        # nnU-Net saves probabilities in reader order (SimpleITKIO -> reversed axes).
        if tuple(probs.shape[1:]) == tuple(shape):
            pass
        elif tuple(probs.shape[1:]) == tuple(shape[::-1]):
            probs = probs.transpose(0, 3, 2, 1)
        else:
            raise ValueError(
                f"cannot map softmax shape {probs.shape} onto volume shape {shape}"
            )
        return np.ascontiguousarray(probs)


# ---------------------------------------------------------------------------
# the same model without the file round trip
# ---------------------------------------------------------------------------
class FastBaselineNNUNetPredictor(BaselineNNUNetPredictor):
    """`BaselineNNUNetPredictor` without the file round trip and the redundant work.

    Most of the baseline's time per iteration is scipy spline resampling of the input
    channels, so this class builds the channels in nnU-Net's axis order in memory, caches
    the crop box and the resampled CT/PET across the iterations of a case, skips channels
    that are still all zero, resamples the rest in a thread pool and resamples the logits
    back on the GPU.  `resample_logits="scipy"` restores nnU-Net's own export path.

    `resample_channels="torch"` moves the input resampling to the GPU as well.  Torch has
    no order-3 spline, so that is an ablation knob rather than a default.
    """

    name = "fast_baseline_nnunet"

    def __init__(self, *args, resample_channels: str = "scipy", resample_logits: str = "torch",
                 num_resample_threads: int = 4, cache_ct_pet: bool = True, **kwargs):
        super().__init__(*args, **kwargs)
        assert resample_channels in ("scipy", "torch")
        assert resample_logits in ("scipy", "torch")
        self.resample_channels = resample_channels
        self.resample_logits = resample_logits
        self.num_resample_threads = int(num_resample_threads)
        self.cache_ct_pet = bool(cache_ct_pet)
        self._case_cache: Dict[str, object] = {}

    # -- helpers -----------------------------------------------------------
    def _check_supported(self, cm):
        if any(cm.use_mask_for_norm):
            raise RuntimeError(
                "use_mask_for_norm is True for some channel; the fast path assumes the "
                "nonzero mask is irrelevant. Use BaselineNNUNetPredictor instead."
            )

    @staticmethod
    def _click_bbox(coords, shape_nnunet):
        """bbox (nnU-Net axis order) covering the click voxels, or None."""
        if not len(coords):
            return None
        a = np.asarray(coords, dtype=np.int64)          # (n, 3) in nibabel order x,y,z
        a = a[:, ::-1]                                  # -> nnU-Net order z,y,x
        lo = a.min(0)
        hi = a.max(0) + 1
        return [[int(lo[i]), int(hi[i])] for i in range(3)]

    def _resample_channels(self, chans, new_shape, orig_spacing, target_spacing, cm):
        """Resample a list of single-channel float32 arrays; returns a list."""
        import torch
        if self.resample_channels == "torch":
            from nnunetv2.preprocessing.resampling.resample_torch import resample_torch_fornnunet
            out = []
            for c in chans:
                r = resample_torch_fornnunet(torch.from_numpy(c[None]), new_shape, orig_spacing,
                                             target_spacing, is_seg=False,
                                             device=torch.device(self.device))
                out.append(np.asarray(r.cpu() if hasattr(r, "cpu") else r, dtype=np.float32)[0])
            return out

        fn = cm.resampling_fn_data
        if tuple(int(x) for x in new_shape) == tuple(chans[0].shape):
            return list(chans)                       # nnU-Net short-circuits this too
        if len(chans) == 1 or self.num_resample_threads <= 1:
            return [fn(c[None], new_shape, orig_spacing, target_spacing)[0] for c in chans]
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=min(self.num_resample_threads, len(chans))) as ex:
            return [r[0] for r in ex.map(
                lambda c: fn(c[None], new_shape, orig_spacing, target_spacing), chans)]

    def _normalize_channel(self, arr, c, pm, cm):
        """Exactly `DefaultPreprocessor._normalize` for one channel index `c`."""
        from nnunetv2.utilities.find_class_by_name import recursive_find_python_class
        from batchgenerators.utilities.file_and_folder_operations import join
        import nnunetv2

        scheme = cm.normalization_schemes[c]
        cls = recursive_find_python_class(join(nnunetv2.__path__[0], "preprocessing", "normalization"),
                                          scheme, "nnunetv2.preprocessing.normalization")
        if cls is None:
            raise RuntimeError(f"Unable to locate class '{scheme}' for normalization")
        normalizer = cls(use_mask_for_norm=cm.use_mask_for_norm[c],
                         intensityproperties=pm.foreground_intensity_properties_per_channel[str(c)])
        return normalizer.run(arr, None)

    # ------------------------------------------------------------------
    # stages, shared with InteractiveNNUNetPredictor
    # ------------------------------------------------------------------
    def _geometry(self, ct, pet, spacing, scribbles, case_name, include_clicks_in_bbox=True):
        """Crop box, resampled shape and the `props` dict nnU-Net's exporter needs.

        `include_clicks_in_bbox` follows how the model was trained.  The baseline gets its
        click heatmaps as input files, so `crop_to_nonzero` sees them and a click outside
        the CT/PET box would enlarge the crop.  The interactive model's guidance is
        stamped after preprocessing, and its store was cropped from CT/PET alone, so
        there the crop box must not depend on the clicks.
        """
        from nnunetv2.preprocessing.cropping.cropping import create_nonzero_mask, get_bbox_from_mask
        from acvl_utils.cropping_and_padding.bounding_boxes import bounding_box_to_slice
        from nnunetv2.preprocessing.resampling.default_resampling import compute_new_shape

        cm = self._ensure_predictor().configuration_manager
        pet = np.asarray(pet)
        orig_spacing = [float(s) for s in list(spacing)[::-1]]   # nibabel (x,y,z) -> nnU-Net (z,y,x)
        shape_before_cropping = tuple(pet.shape[::-1])
        key = (str(case_name), shape_before_cropping, tuple(np.round(orig_spacing, 6)))
        cache = self._case_cache if self._case_cache.get("key") == key else {}

        t0 = time.perf_counter()
        if cache:
            bbox = cache["bbox"]
        else:
            nz = create_nonzero_mask(np.stack([np.asarray(ct).transpose(2, 1, 0),
                                               pet.transpose(2, 1, 0)]))
            bbox = get_bbox_from_mask(nz)
            del nz

        n_clicks_outside = 0
        cb = self._click_bbox(list(scribbles.get("tumor", [])) + list(scribbles.get("background", [])),
                              shape_before_cropping)
        if cb is not None and any(cb[i][0] < bbox[i][0] or cb[i][1] > bbox[i][1] for i in range(3)):
            if include_clicks_in_bbox:
                bbox = [[min(bbox[i][0], cb[i][0]), max(bbox[i][1], cb[i][1])] for i in range(3)]
                cache = {}                      # geometry changed -> cached CT/PET invalid
            else:
                n_clicks_outside = -1           # counted properly by the caller if it cares

        slicer = bounding_box_to_slice(bbox)
        shape_after_crop = tuple(int(bbox[i][1] - bbox[i][0]) for i in range(3))
        new_shape = compute_new_shape(shape_after_crop, orig_spacing, cm.spacing)
        return {
            "key": key, "cache": cache, "bbox": bbox, "slicer": slicer,
            "orig_spacing": orig_spacing, "target_spacing": cm.spacing,
            "shape_before_cropping": shape_before_cropping,
            "shape_after_crop": shape_after_crop,
            "new_shape": tuple(int(x) for x in new_shape),
            "clicks_outside_crop": n_clicks_outside,
            "t_crop": time.perf_counter() - t0,
            "props": {
                "spacing": orig_spacing,
                "shape_before_cropping": shape_before_cropping,
                "bbox_used_for_cropping": bbox,
                "shape_after_cropping_and_before_resampling": shape_after_crop,
            },
        }

    def _ct_pet_channels(self, ct, pet, geom, pm, cm):
        """Normalised + resampled CT and PET, cached for the 5-6 iterations of a case."""
        cache = geom["cache"]
        if cache:
            return cache["ct_r"], cache["pet_r"]
        slicer = geom["slicer"]
        ct_c = np.asarray(ct).transpose(2, 1, 0)[slicer].astype(np.float32)
        pet_c = np.asarray(pet).transpose(2, 1, 0)[slicer].astype(np.float32)
        ct_c = self._normalize_channel(ct_c, 0, pm, cm)
        pet_c = self._normalize_channel(pet_c, 1, pm, cm)
        ct_r, pet_r = self._resample_channels([ct_c, pet_c], geom["new_shape"],
                                              geom["orig_spacing"], geom["target_spacing"], cm)
        del ct_c, pet_c
        if self.cache_ct_pet:
            self._case_cache = {"key": geom["key"], "bbox": geom["bbox"],
                                "ct_r": ct_r, "pet_r": pet_r}
        return ct_r, pet_r

    def _refine_logits(self, logits, data, p, pm, cm):
        """Hook for a second, targeted forward pass. The base predictor does nothing."""
        return logits

    def _infer_and_export(self, data, geom, return_probabilities, p, pm, cm):
        """Sliding window + resample back + argmax; returns arrays in nibabel axis order."""
        import torch
        from nnunetv2.inference.export_prediction import (
            convert_predicted_logits_to_segmentation_with_correct_shape,
        )
        t0 = time.perf_counter()
        logits = p.predict_logits_from_preprocessed_data(torch.from_numpy(data)).cpu()
        if self.device.startswith("cuda"):
            torch.cuda.synchronize()
        logits = self._refine_logits(logits, data, p, pm, cm)
        t_net = time.perf_counter() - t0

        t0 = time.perf_counter()
        shape_after_crop = geom["shape_after_crop"]
        if self.resample_logits == "torch" and tuple(logits.shape[1:]) != tuple(shape_after_crop):
            # Resample the logits on the GPU; the exporter then short-circuits its own
            # resampling because the shapes already match.  nnU-Net resamples
            # probabilities with order 1, i.e. torch's trilinear interpolation.
            from nnunetv2.preprocessing.resampling.resample_torch import resample_torch_fornnunet
            current_spacing = (cm.spacing if len(cm.spacing) == len(shape_after_crop)
                               else [geom["orig_spacing"][0], *cm.spacing])
            logits = resample_torch_fornnunet(logits, list(shape_after_crop), current_spacing,
                                              geom["orig_spacing"], is_seg=False,
                                              device=torch.device(self.device))
            if hasattr(logits, "cpu"):
                logits = logits.cpu()
        out = convert_predicted_logits_to_segmentation_with_correct_shape(
            logits, pm, cm, p.label_manager, geom["props"],
            return_probabilities=return_probabilities)
        del logits
        seg = out[0] if return_probabilities else out
        probs = out[1] if return_probabilities else None
        mask = np.asarray(seg).transpose(2, 1, 0).astype(np.uint8)   # nnU-Net order -> nibabel
        if probs is not None:
            probs = np.ascontiguousarray(np.asarray(probs).transpose(0, 3, 2, 1))
        return mask, probs, t_net, time.perf_counter() - t0

    # ------------------------------------------------------------------
    def predict(
        self,
        ct,
        pet,
        spacing,
        scribbles,
        prev_pred=None,
        case_cache_dir=None,
        *,
        affine=None,
        ct_path=None,
        pet_path=None,
        case_name="case",
        return_probabilities=False,
    ):
        p = self._ensure_predictor()
        pm, cm = p.plans_manager, p.configuration_manager
        self._check_supported(cm)
        if list(pm.transpose_forward) != [0, 1, 2]:
            raise RuntimeError("fast path assumes transpose_forward == [0,1,2]")

        scribbles = scribbles or _empty_scribbles()
        shape_nib = np.asarray(pet).shape
        t_all = time.perf_counter()

        geom = self._geometry(ct, pet, spacing, scribbles, case_name, include_clicks_in_bbox=True)
        cached = bool(geom["cache"])
        t0 = time.perf_counter()
        ct_r, pet_r = self._ct_pet_channels(ct, pet, geom, pm, cm)
        t_ctpet = time.perf_counter() - t0

        # ---- guidance channels (baseline: sparse heatmaps preprocessed like images) --
        t0 = time.perf_counter()
        new_shape, slicer = geom["new_shape"], geom["slicer"]
        guid_r, todo, todo_idx = {}, [], []
        for idx, name in ((2, "tumor"), (3, "background")):
            pts = scribbles.get(name, [])
            if len(pts) == 0:
                # all-zero -> ZScore keeps it zero -> resampling keeps it zero
                guid_r[idx] = np.zeros(new_shape, dtype=np.float32)
                continue
            hm = generate_gaussian_heatmap(pts, shape_nib, self.sigma).transpose(2, 1, 0)[slicer]
            todo.append(self._normalize_channel(np.ascontiguousarray(hm, dtype=np.float32), idx, pm, cm))
            todo_idx.append(idx)
        if todo:
            for idx, r in zip(todo_idx, self._resample_channels(todo, new_shape,
                                                                geom["orig_spacing"],
                                                                geom["target_spacing"], cm)):
                guid_r[idx] = r
        del todo
        t_guid = time.perf_counter() - t0

        t0 = time.perf_counter()
        data = np.empty((4, *new_shape), dtype=np.float32)
        data[0] = ct_r
        data[1] = pet_r
        data[2] = guid_r[2]
        data[3] = guid_r[3]
        del guid_r
        t_assemble = time.perf_counter() - t0

        mask, probs, t_net, t_export = self._infer_and_export(data, geom, return_probabilities,
                                                              p, pm, cm)
        del data
        self.last_timings = {
            "crop_bbox_s": round(geom["t_crop"], 3), "ct_pet_preproc_s": round(t_ctpet, 3),
            "guidance_preproc_s": round(t_guid, 3), "assemble_s": round(t_assemble, 3),
            "network_s": round(t_net, 3), "export_s": round(t_export, 3),
            "total_s": round(time.perf_counter() - t_all, 3),
            "ct_pet_cached": cached,
        }
        return (mask, probs) if return_probabilities else mask


# ---------------------------------------------------------------------------
# the 5-channel interactive model
# ---------------------------------------------------------------------------
def _import_train_guidance():
    """Import the guidance encoder from the training transform, adding src/ to the path."""
    try:
        from train.guidance import guidance_map_from_coords, stamp_clipped_edt
    except ImportError:
        import sys
        here = os.path.dirname(os.path.abspath(__file__))
        if here not in sys.path:
            sys.path.insert(0, here)
        from train.guidance import guidance_map_from_coords, stamp_clipped_edt
    return guidance_map_from_coords, stamp_clipped_edt


class InteractiveNNUNetPredictor(FastBaselineNNUNetPredictor):
    """The 5-channel interactive model (`nnUNetTrainer_Interactive`).

    Input channels in nnU-Net order:

        0 CT                  CTNormalization, resampled to the plans spacing
        1 PET (SUV)           ZScoreNormalization, resampled
        2 tumor guidance      clipped EDT max(0, 1 - d/R) of the tumor scribbles
        3 background guidance clipped EDT of the background scribbles
        4 previous prediction binary, the final mask of the previous iteration

    Channels 2-4 are stamped on the preprocessed, plans-spacing grid rather than fed to
    nnU-Net as files.  The training transform runs after `SpatialTransform`, so it builds
    them on that grid and its radius is in voxels of it; resampling them with the images
    would put an order-3 spline through a cone-shaped channel and a binary mask.
    Scribble coordinates are mapped there with the inverse of skimage's centre-aligned
    resize mapping, `j = round((i + 0.5) * out/in - 0.5)`, minus the crop-box origin.

    The answer depends on `prev_pred`, so `cache_state_key` hashes the previous mask into
    the harness cache key.  `use_state_dir=True` reads it from disk instead, for the
    container, where nothing can be passed in memory between calls.
    """

    name = "interactive_nnunet"
    n_input_channels = 5
    #: pure function of (ct, pet, scribbles, prev_pred); see `cache_state_key`
    stateless = True

    DEFAULT_MODEL_FOLDER = (
        "/content/drive/MyDrive/autoPET/ckpt/Dataset998_AutoPETV/"
        "nnUNetTrainer_Interactive__nnUNetPlans_interactive__3d_fullres"
    )

    def __init__(self, *args, guidance_radius: Optional[float] = None,
                 use_state_dir: bool = False, prev_mask_filename: str = "prev_final_mask.npy",
                 deterministic: bool = True, force_mirror_axes: Optional[Sequence[int]] = None,
                 foveal_crop: bool = False, foveal_fuse: str = "max",
                 **kwargs):
        kwargs.setdefault("model_folder", os.environ.get("AUTOPETV_INTERACTIVE_MODEL_FOLDER")
                          or self.DEFAULT_MODEL_FOLDER)
        super().__init__(*args, **kwargs)
        # The trainer resolves the radius the same way: class default, overridable by
        # `nnUNet_interactive_radius`.  Keep the two in step or the encoding drifts.
        self.guidance_radius = float(
            guidance_radius if guidance_radius is not None
            else os.environ.get("nnUNet_interactive_radius", 10.0))
        self.use_state_dir = bool(use_state_dir)
        self.prev_mask_filename = prev_mask_filename
        self.deterministic = bool(deterministic)
        self.force_mirror_axes = tuple(force_mirror_axes) if force_mirror_axes else None
        self.last_guidance_info: Dict[str, object] = {}
        # foveal re-inference (B15): a second forward pass on one patch-sized window
        # centred on the newest scribble, fused into the sliding-window logits
        self.foveal_crop = bool(foveal_crop)
        if foveal_fuse not in ("max", "mean"):
            raise ValueError(f"foveal_fuse must be 'max' or 'mean', got {foveal_fuse!r}")
        self.foveal_fuse = foveal_fuse
        #: set by the evaluation loop through `_set_iteration`; only used for logging
        self.current_iteration = 0
        self._foveal_center: Optional[Sequence[int]] = None
        self._foveal_seen: Dict[str, tuple] = {}
        self.last_foveal_info: Dict[str, object] = {}

    # -- validation --------------------------------------------------------
    def _ensure_predictor(self):
        already = self._predictor is not None
        if not already:
            self._register_external_trainer()
        p = super()._ensure_predictor()
        if already:
            return p
        import torch

        cm, dj = p.configuration_manager, p.dataset_json
        schemes = list(cm.normalization_schemes)
        n_ch = len(dj.get("channel_names", {}))
        if n_ch != self.n_input_channels or len(schemes) != self.n_input_channels:
            raise RuntimeError(
                f"{self.name} expects a {self.n_input_channels}-channel model but "
                f"{self.model_folder} declares {n_ch} channels / {len(schemes)} "
                f"normalization schemes. Wrong plans (nnUNetPlans_interactive.json) or "
                f"wrong dataset.json?")
        bad = [(i, s) for i, s in enumerate(schemes[2:], start=2) if s != "NoNormalization"]
        if bad:
            raise RuntimeError(
                f"channels 2-4 must be NoNormalization (the guidance is already in [0,1] "
                f"and the previous-prediction channel is binary); got {bad}. The token in "
                f"dataset.json is 'noNorm', which nnU-Net maps to NoNormalization.")
        if any(cm.use_mask_for_norm):
            raise RuntimeError("use_mask_for_norm must be all False for this model")

        if self.force_mirror_axes is not None:
            p.allowed_mirroring_axes = self.force_mirror_axes
        if self.deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        return p

    @staticmethod
    def _register_external_trainer():
        """Put `src/train` on `nnUNet_extTrainer` so nnU-Net can find our trainer class.

        `initialize_from_trained_model_folder` rebuilds the network through the trainer
        named in the checkpoint, and ours lives in this repo rather than inside nnunetv2.
        """
        train_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "train")
        if not os.path.isdir(train_dir):
            return
        cur = os.environ.get("nnUNet_extTrainer", "")
        paths = [p for p in cur.split(os.pathsep) if p.strip()]
        if train_dir not in paths:
            paths.insert(0, train_dir)
        os.environ["nnUNet_extTrainer"] = os.pathsep.join(paths)

    # -- cache key ---------------------------------------------------------
    def cache_state_key(self, prev_pred) -> Optional[str]:
        """Hash of channel 4 -- the part of the input the scribble set does not describe."""
        import hashlib
        if prev_pred is None:
            return "prev0"
        a = np.ascontiguousarray(np.asarray(prev_pred) > 0)
        if not a.any():
            return "prev0"
        h = hashlib.sha1()
        h.update(str(a.shape).encode())
        h.update(np.packbits(a.reshape(-1)).tobytes())
        return "prev" + h.hexdigest()[:12]

    # -- previous mask -----------------------------------------------------
    def _resolve_prev_pred(self, prev_pred, case_cache_dir, shape_nib):
        if prev_pred is not None:
            a = np.asarray(prev_pred)
            if a.shape != tuple(shape_nib):
                raise ValueError(f"prev_pred shape {a.shape} != image shape {tuple(shape_nib)}")
            return (a > 0).astype(np.uint8), "argument"
        if self.use_state_dir and case_cache_dir:
            path = os.path.join(case_cache_dir, self.prev_mask_filename)
            if os.path.isfile(path):
                a = np.load(path)
                if a.shape == tuple(shape_nib):
                    return (a > 0).astype(np.uint8), "state_dir"
        return None, "none"

    @staticmethod
    def save_prev_mask(case_cache_dir: str, mask: np.ndarray,
                       filename: str = "prev_final_mask.npy") -> str:
        """Persist this iteration's final mask so the next container call can read it back.

        Not called from `predict`: under a post-processing layer the mask this predictor
        returns is not the final one.
        """
        os.makedirs(case_cache_dir, exist_ok=True)
        path = os.path.join(case_cache_dir, filename)
        np.save(path, np.asarray(mask, dtype=np.uint8))
        return path

    # -- coordinate mapping ------------------------------------------------
    @staticmethod
    def map_coords_to_grid(coords_xyz, bbox, shape_after_crop, new_shape):
        """nibabel (x, y, z) indices -> indices on the preprocessed (plans-spacing) grid.

        skimage's resize maps output index j to input coordinate (j + 0.5) * in/out - 0.5;
        this is that map inverted and rounded.  Returns (mapped, n_dropped); points
        outside the crop box or the grid are dropped.
        """
        if not len(coords_xyz):
            return np.zeros((0, 3), dtype=np.int64), 0
        a = np.asarray(coords_xyz, dtype=np.int64)[:, ::-1]        # -> nnU-Net (z, y, x)
        lo = np.asarray([b[0] for b in bbox], dtype=np.int64)
        hi = np.asarray([b[1] for b in bbox], dtype=np.int64)
        inside = np.all((a >= lo) & (a < hi), axis=1)
        dropped = int((~inside).sum())
        a = a[inside] - lo
        scale = np.asarray(new_shape, dtype=np.float64) / np.asarray(shape_after_crop, dtype=np.float64)
        m = np.floor((a + 0.5) * scale - 0.5 + 0.5).astype(np.int64)   # round-half-up
        ns = np.asarray(new_shape, dtype=np.int64)
        ok = np.all((m >= 0) & (m < ns), axis=1)
        dropped += int((~ok).sum())
        return m[ok], dropped

    # -- main --------------------------------------------------------------
    def predict(
        self,
        ct,
        pet,
        spacing,
        scribbles,
        prev_pred=None,
        case_cache_dir=None,
        *,
        affine=None,
        ct_path=None,
        pet_path=None,
        case_name="case",
        return_probabilities=False,
    ):
        p = self._ensure_predictor()
        pm, cm = p.plans_manager, p.configuration_manager
        if list(pm.transpose_forward) != [0, 1, 2]:
            raise RuntimeError("fast path assumes transpose_forward == [0,1,2]")
        guidance_map_from_coords, _ = _import_train_guidance()

        scribbles = scribbles or _empty_scribbles()
        shape_nib = np.asarray(pet).shape
        t_all = time.perf_counter()

        # the interactive store was cropped from CT/PET only -> clicks must not move the box
        geom = self._geometry(ct, pet, spacing, scribbles, case_name, include_clicks_in_bbox=False)
        cached = bool(geom["cache"])
        t0 = time.perf_counter()
        ct_r, pet_r = self._ct_pet_channels(ct, pet, geom, pm, cm)
        t_ctpet = time.perf_counter() - t0

        new_shape = geom["new_shape"]

        # ---- channels 2 and 3: clipped EDT on the preprocessed grid -------
        t0 = time.perf_counter()
        info = {"radius_voxels": self.guidance_radius}
        chans, mapped_by_name = {}, {}
        for idx, name in ((2, "tumor"), (3, "background")):
            pts = scribbles.get(name, [])
            mapped, dropped = self.map_coords_to_grid(pts, geom["bbox"],
                                                      geom["shape_after_crop"], new_shape)
            mapped_by_name[name] = mapped
            info[f"n_{name}"] = len(pts)
            info[f"n_{name}_mapped"] = int(len(mapped))
            info[f"n_{name}_dropped"] = dropped
            if dropped:
                self._warn(f"{dropped} {name} scribble voxel(s) fell outside the crop box "
                           f"or the resampled grid and were dropped")
            chans[idx] = (guidance_map_from_coords(new_shape, mapped, self.guidance_radius)
                          if len(mapped) else np.zeros(new_shape, dtype=np.float32))
        self._foveal_center = (self._newest_scribble_center(case_name, scribbles, mapped_by_name)
                               if self.foveal_crop else None)
        t_guid = time.perf_counter() - t0

        # ---- channel 4: the previous FINAL mask, on the same grid ---------
        t0 = time.perf_counter()
        prev, prev_src = self._resolve_prev_pred(prev_pred, case_cache_dir, shape_nib)
        if prev is None:
            chans[4] = np.zeros(new_shape, dtype=np.float32)
        else:
            prev_c = prev.transpose(2, 1, 0)[geom["slicer"]].astype(np.uint8)
            if tuple(prev_c.shape) == tuple(new_shape):
                prev_r = prev_c
            else:
                # nnU-Net's segmentation resampler: one-hot, order-1, argmax -> stays binary
                prev_r = cm.resampling_fn_seg(prev_c[None], new_shape, geom["orig_spacing"],
                                              geom["target_spacing"])[0]
            chans[4] = np.asarray(prev_r, dtype=np.float32)
            del prev_c
        info["prev_pred_source"] = prev_src
        info["prev_pred_voxels"] = int(chans[4].sum())
        t_prev = time.perf_counter() - t0

        t0 = time.perf_counter()
        data = np.empty((5, *new_shape), dtype=np.float32)
        data[0] = ct_r
        data[1] = pet_r
        data[2] = chans[2]
        data[3] = chans[3]
        data[4] = chans[4]
        del chans
        t_assemble = time.perf_counter() - t0

        mask, probs, t_net, t_export = self._infer_and_export(data, geom, return_probabilities,
                                                              p, pm, cm)
        del data
        self.last_guidance_info = info
        self.last_timings = {
            "crop_bbox_s": round(geom["t_crop"], 3), "ct_pet_preproc_s": round(t_ctpet, 3),
            "guidance_preproc_s": round(t_guid, 3), "prev_pred_s": round(t_prev, 3),
            "assemble_s": round(t_assemble, 3), "network_s": round(t_net, 3),
            "export_s": round(t_export, 3), "total_s": round(time.perf_counter() - t_all, 3),
            "ct_pet_cached": cached,
        }
        if self.foveal_crop:
            self.last_timings["foveal_fired"] = bool(self.last_foveal_info.get("fired"))
        return (mask, probs) if return_probabilities else mask

    # ------------------------------------------------------------------
    # foveal re-inference (B15)
    # ------------------------------------------------------------------
    def _newest_scribble_center(self, case_name, scribbles, mapped_by_name):
        """Centroid, on the preprocessed grid, of the stroke that arrived last.

        The evaluation loop appends one stroke per iteration to one of the two lists,
        so the newest stroke is the tail of whichever list grew since the previous call
        on this case. The counts are remembered per case; when there is no memory --
        the first call of a case, which is the state after iteration 0 was served from
        the prediction cache -- exactly one list can be non-empty at iteration 1, so its
        tail is the newest stroke and the fallback is exact where it is used.
        """
        n_t, n_b = len(scribbles.get("tumor", [])), len(scribbles.get("background", []))
        if n_t + n_b == 0:
            return None
        prev = self._foveal_seen.get(case_name)
        self._foveal_seen = {case_name: (n_t, n_b)}      # one case at a time
        if prev is not None and (n_t > prev[0] or n_b > prev[1]):
            name = "tumor" if n_t > prev[0] else "background"
            start = prev[0] if name == "tumor" else prev[1]
            source = "grown"
        else:
            name = "tumor" if n_t else "background"
            start = max(0, (n_t if name == "tumor" else n_b) - 1)
            source = "tail"
        pts = np.asarray(mapped_by_name.get(name, []), dtype=np.int64)
        if pts.size == 0:
            other = "background" if name == "tumor" else "tumor"
            pts = np.asarray(mapped_by_name.get(other, []), dtype=np.int64)
            start, name = 0, other
        if pts.size == 0:
            return None                                   # every voxel fell outside the box
        stroke = pts[min(start, len(pts) - 1):]
        center = np.rint(stroke.mean(axis=0)).astype(int)
        self.last_foveal_info = {"center": [int(c) for c in center], "stroke_name": name,
                                 "stroke_voxels": int(len(stroke)), "center_source": source}
        return center

    def _refine_logits(self, logits, data, p, pm, cm):
        """Fuse a patch-sized forward pass centred on the newest scribble into `logits`.

        The sliding window sees a scribble only in whatever tiles happen to cover it, at
        whatever offset the tiling gives; this puts one window on it deliberately, at the
        patch size the network was trained at, and fuses the two logit fields inside that
        window. Feeding a volume of exactly the patch size back through
        `predict_logits_from_preprocessed_data` makes the window a single tile whose
        Gaussian weighting divides out, so it is one plain forward pass through the same
        code path as the full prediction -- same padding rules, same mirroring setting.
        """
        if not self.foveal_crop:
            return logits
        center = self._foveal_center
        info = dict(self.last_foveal_info) if center is not None else {}
        info["iteration"] = int(self.current_iteration)
        info["fuse"] = self.foveal_fuse
        if center is None:
            info["fired"] = False
            info["reason"] = "no scribbles"
            self.last_foveal_info = info
            return logits

        import torch

        shape = tuple(int(v) for v in data.shape[1:])
        patch = [int(v) for v in cm.patch_size]
        lo, hi = [], []
        for d in range(3):
            size = min(patch[d], shape[d])
            start = int(center[d]) - size // 2
            start = max(0, min(start, shape[d] - size))
            lo.append(start)
            hi.append(start + size)
        window = tuple(slice(a, b) for a, b in zip(lo, hi))
        info.update({"fired": True, "window_lo": lo, "window_hi": hi,
                     "window_shape": [b - a for a, b in zip(lo, hi)],
                     "volume_shape": list(shape)})

        t0 = time.perf_counter()
        sub = np.ascontiguousarray(data[(slice(None), *window)])
        sub_logits = p.predict_logits_from_preprocessed_data(torch.from_numpy(sub)).cpu()
        if self.device.startswith("cuda"):
            torch.cuda.synchronize()
        info["foveal_s"] = round(time.perf_counter() - t0, 3)

        cur = logits[(slice(None), *window)]
        sub_logits = sub_logits.to(cur.dtype)
        fused = torch.maximum(cur, sub_logits) if self.foveal_fuse == "max" \
            else 0.5 * (cur + sub_logits)
        info["mean_abs_change"] = round(float((fused - cur).abs().mean()), 5)
        logits[(slice(None), *window)] = fused
        del sub, sub_logits, cur, fused
        self.last_foveal_info = info
        self._warn_foveal(info)
        return logits

    def _warn_foveal(self, info):
        if not self.verbose:
            return
        print(f"[{self.name}] foveal pass at iteration {info.get('iteration')}: "
              f"window {info.get('window_lo')}..{info.get('window_hi')} of "
              f"{info.get('volume_shape')}, {info.get('stroke_voxels')} "
              f"{info.get('stroke_name')} voxels, fuse={info.get('fuse')}, "
              f"{info.get('foveal_s')} s, mean |Δlogit| {info.get('mean_abs_change')}")

    def _warn(self, msg):
        import warnings
        warnings.warn(f"[{self.name}] {msg}", RuntimeWarning, stacklevel=2)
