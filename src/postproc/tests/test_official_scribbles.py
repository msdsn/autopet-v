"""Contract test against the official scribble generator.

Checks that our functions take the coordinate lists the organizers' loop produces, in
their order and after their json round trip (simulate_scribble_from_label ->
scribbles_to_gc_format -> gc_to_swfastedit_format), and that the [i, j, k] indices
really index our arrays: a silent transpose here looks just like "the model ignores
scribbles". Skipped if autoPETV/interactive is not next to src/.
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import pytest

from postproc.compliance import apply_all_constraints, check_constraints
from postproc.pipeline import PostProcPredictor

from conftest import SPACING, Blob, make_phantom, sphere_mask

# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(os.path.dirname(_HERE)))  # .../autoPET

# The official challenge repo is gitignored, so it may sit next to ours, inside it, or
# wherever AUTOPETV_REPO points.  Try all three before giving up.
_CANDIDATES = [
    os.path.join(os.environ.get("AUTOPETV_REPO", ""), "interactive"),
    os.path.join(_REPO, "autoPETV", "interactive"),
    os.path.join(os.path.dirname(_REPO), "autoPETV", "interactive"),
]

sim = None
for _INTERACTIVE in _CANDIDATES:
    if not _INTERACTIVE or not os.path.isdir(_INTERACTIVE):
        continue
    if _INTERACTIVE not in sys.path:
        sys.path.insert(0, _INTERACTIVE)
    try:
        import simulate_scribbles as sim  # type: ignore
        break
    except Exception:  # pragma: no cover
        sim = None

pytestmark = pytest.mark.skipif(sim is None, reason="autoPETV/interactive not available")

STRATEGIES = ["centerline", "boundary", "random"]


def _official_scribble(error_mask, strategy):
    """What interactive_loop.py does for one error mask."""
    out = sim.simulate_scribble_from_label(error_mask.astype(np.uint8), strategy)
    if len(out) == 2:  # the empty-mask 2-tuple that crashes the reference loop
        return [], 0
    coords, _cls, size = out
    return coords, int(size)


def _json_roundtrip(data):
    """scribbles_to_gc_format -> json -> gc_to_swfastedit_format."""
    gc = sim.scribbles_to_gc_format(data)
    gc = json.loads(json.dumps(gc))
    return sim.gc_to_swfastedit_format(gc)


# ---------------------------------------------------------------------------
@pytest.mark.parametrize("strategy", STRATEGIES)
def test_official_coordinates_index_our_arrays(strategy):
    """label[c[0], c[1], c[2]] == 1 for every emitted point, i.e. no transpose."""
    shape = (64, 64, 48)
    ct, pet, gt, bm = make_phantom(
        shape=shape, blobs=[Blob(centre=(30, 30, 24), radius_mm=9.0, suv=8.0, name="l")]
    )
    coords, size = _official_scribble(gt, strategy)
    assert size > 0 and len(coords) == size

    arr = np.asarray(coords)
    assert arr.shape[1] == 3
    assert gt[arr[:, 0], arr[:, 1], arr[:, 2]].all(), "coordinates are transposed"
    # all on one axial slice: the simulator draws on a single k
    assert len(np.unique(arr[:, 2])) == 1


@pytest.mark.parametrize("strategy", STRATEGIES)
def test_error_driven_scribbles_flow_into_our_compliance(strategy):
    """Reproduce one full evaluator step on a synthetic error and enforce it."""
    shape = (72, 72, 56)
    blobs = [
        Blob(centre=(24, 24, 20), radius_mm=9.0, suv=9.0, name="hit"),
        Blob(centre=(50, 46, 34), radius_mm=8.0, suv=5.0, name="missed"),
    ]
    ct, pet, gt, bm = make_phantom(shape=shape, blobs=blobs)

    fp = sphere_mask(shape, (24, 52, 18), 7.0, SPACING)
    pred = gt.copy()
    pred[bm["missed"]] = 0     # false negative
    pred[fp] = 1               # false positive

    overseg = (pred == 1) & (gt == 0)
    underseg = (pred == 0) & (gt == 1)
    scribbles_bg, n_fp = _official_scribble(overseg, strategy)
    scribbles_fg, n_fn = _official_scribble(underseg, strategy)

    data = {"tumor": [], "background": []}
    if n_fp <= n_fn:       # the evaluator's exact tie-break: ties go to tumor
        data["tumor"] += scribbles_fg
    else:
        data["background"] += scribbles_bg
    data = _json_roundtrip(data)
    assert data["tumor"] or data["background"]

    out = apply_all_constraints(
        pred, pet, ct, data["tumor"], data["background"], tracer="fdg", spacing=SPACING
    )
    assert check_constraints(out, data["tumor"], data["background"])["ok"]

    twice = apply_all_constraints(
        out, pet, ct, data["tumor"], data["background"], tracer="fdg", spacing=SPACING
    )
    assert np.array_equal(out, twice)


@pytest.mark.parametrize("strategy", STRATEGIES)
def test_full_five_iteration_loop_with_official_scribbles(strategy, tmp_path):
    """The whole evaluator inner loop, with our pipeline as the container."""
    shape = (72, 72, 56)
    blobs = [
        Blob(centre=(24, 24, 20), radius_mm=9.0, suv=9.0, name="a"),
        Blob(centre=(50, 46, 34), radius_mm=8.0, suv=4.0, name="b"),
    ]
    ct, pet, gt, bm = make_phantom(shape=shape, blobs=blobs)
    fp = sphere_mask(shape, (24, 54, 18), 7.0, SPACING)

    class Base:
        """Misses lesion b, invents a false positive, and ignores every scribble."""

        name = "stubborn"

        def predict(self, ct, pet, spacing, scribbles=None, prev_pred=None,
                    case_cache_dir=None, *, return_probabilities=False, **kw):
            m = gt.copy()
            m[bm["b"]] = 0
            m[fp] = 1
            if return_probabilities:
                p = np.where(m > 0, 0.9, 0.05).astype(np.float32)
                return m, p
            return m

    cache = str(tmp_path / "cache")
    os.makedirs(cache, exist_ok=True)
    pp = PostProcPredictor(Base(), {"tracer": "fdg"})

    data = {"tumor": [], "background": []}
    dices = []
    for it in range(5):
        if it > 0:
            overseg = (pred == 1) & (gt == 0)
            underseg = (pred == 0) & (gt == 1)
            s_bg, n_fp = _official_scribble(overseg, strategy)
            s_fg, n_fn = _official_scribble(underseg, strategy)
            if n_fp <= n_fn:
                data["tumor"] += s_fg
            else:
                data["background"] += s_bg
            data = _json_roundtrip(data)

        pred = pp.predict(ct, pet, SPACING, data, None, cache)
        assert check_constraints(pred, data["tumor"], data["background"])["ok"], (
            f"[{strategy}] iteration {it}: constraint violated"
        )
        inter = int((pred.astype(bool) & gt.astype(bool)).sum())
        dices.append(2 * inter / (int(pred.sum()) + int(gt.sum())))

    print(f"\n[{strategy}] dice over iterations: " + ", ".join(f"{d:.3f}" for d in dices))
    assert dices[-1] >= dices[0], f"interaction made things worse: {dices}"


# ---------------------------------------------------------------------------
# regression: no oscillation when the base model consumes our previous mask
# ---------------------------------------------------------------------------
class _FeedbackInteractiveBase:
    """A model that trusts its own previous prediction, like the interactive fine-tune.

    Modelled as "detect, then keep only what was predicted last time (dilated)", so
    anything the interaction layer deletes the model cannot recover. Its probability map
    is high and flat, the landscape on which a marker watershed degenerates into a
    distance partition.
    """

    name = "feedback_interactive"

    def __init__(self, pet, thr=2.5):
        self.pet = np.asarray(pet)
        self.thr = thr

    def predict(self, ct, pet, spacing, scribbles=None, prev_pred=None,
                case_cache_dir=None, *, return_probabilities=False, **kw):
        from scipy import ndimage

        m = self.pet >= self.thr
        if prev_pred is not None:
            m &= ndimage.binary_dilation(np.asarray(prev_pred) > 0, iterations=2)
        m = m.astype(np.uint8)
        if return_probabilities:
            return m, np.where(m > 0, 0.95, 0.02).astype(np.float32)
        return m


@pytest.mark.parametrize("strategy", STRATEGIES)
def test_interaction_never_collapses_a_well_segmented_case(strategy, tmp_path):
    """End-to-end stability with a base model that consumes our previous mask.

    The background scribble lands on a halo sharing a connected component with the
    lesion, and the model's next prediction is conditioned on what we hand back, so this
    is where a feedback loop would show up as a swinging Dice.
    """
    shape = (72, 72, 56)
    ct, pet, gt, bm = make_phantom(
        shape=shape,
        blobs=[Blob(centre=(36, 36, 28), radius_mm=12.0, suv=8.0, name="lesion")],
        background_suv=0.6,
        seed=-1,
    )
    halo = sphere_mask(shape, (36, 36, 28), 17.0, SPACING) & ~bm["lesion"]
    pet[halo] = 3.0                       # over-segmented by the model, not in the GT

    cache = str(tmp_path / "state")
    os.makedirs(cache, exist_ok=True)
    pp = PostProcPredictor(_FeedbackInteractiveBase(pet), {"tracer": "fdg"})

    data = {"tumor": [], "background": []}
    pred, dices = None, []
    for it in range(6):
        if it > 0:
            overseg = (pred == 1) & (gt == 0)
            underseg = (pred == 0) & (gt == 1)
            s_bg, n_fp = _official_scribble(overseg, strategy)
            s_fg, n_fn = _official_scribble(underseg, strategy)
            if n_fp <= n_fn:
                data["tumor"] += s_fg
            else:
                data["background"] += s_bg
            data = _json_roundtrip(data)

        pred = pp.predict(ct, pet, SPACING, data, pred, cache)
        assert check_constraints(pred, data["tumor"], data["background"])["ok"]
        inter = int((pred.astype(bool) & gt.astype(bool)).sum())
        dices.append(2 * inter / (int(pred.sum()) + int(gt.sum())))

    print(f"\n[{strategy}] dice: " + ", ".join(f"{d:.3f}" for d in dices))
    worst_drop = max(
        (dices[i] - dices[i + 1] for i in range(len(dices) - 1)), default=0.0
    )
    assert worst_drop <= 0.15, f"[{strategy}] oscillation: {dices}"
    assert min(dices) >= dices[0] - 0.15, f"[{strategy}] collapsed below the start: {dices}"
    assert dices[-1] >= dices[0] - 0.05, f"[{strategy}] ended below the start: {dices}"
