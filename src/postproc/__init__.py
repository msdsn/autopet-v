"""Inference-time interaction logic wrapped around the network.

Model-agnostic: it only needs a duck-typed ``predict(...)`` callable, see
``src/predictor.py``.  Arrays are in nibabel index space, ``get_fdata()[i, j, k]``
with ``k`` the axial slice; scribble points are ``[i, j, k]`` int lists and
``spacing`` is mm in the same axis order.  Nothing here transposes.
"""

from .config import (
    PostProcConfig,
    ComplianceConfig,
    CleanupConfig,
    NegativeGateConfig,
    MonotoneConfig,
    TRACER_SUV_FLOOR,
)
from .constraints import ConstraintState, CaseCache
from .compliance import (
    apply_background_scribbles,
    apply_tumor_scribbles,
    apply_all_constraints,
    assert_constraints,
    check_constraints,
)
from .cleanup import (
    remove_small_components,
    tracer_suv_floor,
    fill_small_holes,
)
from .negative_gate import is_probably_negative, negative_gate_features
from .tracer_classifier import guess_tracer, tracer_features
from .monotone import blend_with_previous, blend_masks
from .pipeline import PostProcPredictor

__all__ = [
    "PostProcConfig",
    "ComplianceConfig",
    "CleanupConfig",
    "NegativeGateConfig",
    "MonotoneConfig",
    "TRACER_SUV_FLOOR",
    "ConstraintState",
    "CaseCache",
    "apply_background_scribbles",
    "apply_tumor_scribbles",
    "apply_all_constraints",
    "assert_constraints",
    "check_constraints",
    "remove_small_components",
    "tracer_suv_floor",
    "fill_small_holes",
    "is_probably_negative",
    "negative_gate_features",
    "guess_tracer",
    "tracer_features",
    "blend_with_previous",
    "blend_masks",
    "PostProcPredictor",
]
