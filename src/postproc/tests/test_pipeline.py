"""End-to-end behaviour of PostProcPredictor across the 5 challenge iterations."""

from __future__ import annotations

import os

import numpy as np
import pytest

from postproc.compliance import check_constraints
from postproc.config import PostProcConfig
from postproc.pipeline import PostProcPredictor

from conftest import SPACING, make_phantom, scribble_line, sphere_mask


# ---------------------------------------------------------------------------
# stand-in base predictors (duck typed, no nnU-Net involved)
# ---------------------------------------------------------------------------
class ThresholdBase:
    """SUV threshold, deterministic, completely ignores the scribbles.

    The pessimistic case: the network answers the same thing every iteration, so all
    the interaction has to come from the post-processing layer.
    """

    name = "threshold"

    def __init__(self, thr=3.0, fp_blob=None, prob=True, conf=0.8):
        self.thr = thr
        self.fp_blob = fp_blob
        self.prob = prob
        self.conf = conf  # confidence the model reports inside its own mask
        self.calls = 0

    def predict(
        self,
        ct,
        pet,
        spacing,
        scribbles=None,
        prev_pred=None,
        case_cache_dir=None,
        *,
        return_probabilities=False,
        **kw,
    ):
        self.calls += 1
        mask = (np.asarray(pet) >= self.thr).astype(np.uint8)
        if self.fp_blob is not None:
            mask[self.fp_blob] = 1
        if return_probabilities and self.prob:
            p = np.clip(np.asarray(pet) / (2 * self.thr), 0, 1).astype(np.float32)
            p[mask > 0] = np.maximum(p[mask > 0], self.conf)
            return mask, p
        return mask


class NoProbBase(ThresholdBase):
    """Same, but its predict() has no return_probabilities parameter at all."""

    name = "noprob"

    def predict(self, ct, pet, spacing, scribbles=None, prev_pred=None, case_cache_dir=None):
        self.calls += 1
        mask = (np.asarray(pet) >= self.thr).astype(np.uint8)
        if self.fp_blob is not None:
            mask[self.fp_blob] = 1
        return mask


# ---------------------------------------------------------------------------
def test_iteration_zero_without_scribbles_runs_and_caches(phantom, cache_dir):
    ct, pet = phantom["ct"], phantom["pet"]
    pp = PostProcPredictor(ThresholdBase(), {"tracer": "fdg"})
    out = pp.predict(ct, pet, SPACING, {"tumor": [], "background": []}, None, cache_dir)
    assert out.dtype == np.uint8 and out.shape == pet.shape
    assert os.path.exists(os.path.join(cache_dir, "postproc_constraints.json"))
    assert pp.last_info["iteration"] == 0
    assert pp.last_info["tracer"] == "fdg"


def test_five_iterations_honour_every_scribble(phantom, cache_dir):
    """The base predictor never sees the scribbles; all compliance is ours."""
    ct, pet, gt, blobs = phantom["ct"], phantom["pet"], phantom["gt"], phantom["blobs"]
    fp = sphere_mask(pet.shape, (60, 60, 20), 7.0, SPACING)
    base = ThresholdBase(thr=3.0, fp_blob=fp)
    pp = PostProcPredictor(base, {"tracer": "fdg"})

    scribbles = {"tumor": [], "background": []}
    fg_pts = scribble_line((30, 58, 46), axis=0, length=5, mask=blobs["cool"])
    bg_pts = scribble_line((60, 60, 20), axis=0, length=5, mask=fp)

    masks = []
    for it in range(5):
        if it == 1:
            scribbles["background"] += bg_pts
        if it == 2:
            scribbles["tumor"] += fg_pts
        out = pp.predict(ct, pet, SPACING, scribbles, None, cache_dir)
        masks.append(out)
        assert check_constraints(out, scribbles["tumor"], scribbles["background"])["ok"], (
            f"iteration {it} violates a scribble constraint"
        )
        # The index counts scribble events, not calls: an iteration that adds no
        # scribble does not advance it.  Nothing keys on the index -- it degrades to
        # "unknown" rather than mis-indexing, which is what Category 2 needs.
        expected = (1 if it >= 1 else 0) + (1 if it >= 2 else 0)
        assert pp.last_info["iteration"] == expected

    # the FP blob was deleted at iteration 1 and must never come back, even though the
    # base predictor keeps producing it every single call
    for it in (1, 2, 3, 4):
        overlap = (masks[it].astype(bool) & fp).sum()
        assert overlap == 0, f"the deleted false positive came back at iteration {it}"
    # Every iteration reaches the network: the hand-placed background scribble at
    # iteration 1 would be vacuously satisfied by this phantom's empty mask, and the
    # skip rule refuses to fire on an empty mask for exactly that reason.
    assert base.calls == 5


def test_negative_case_stays_empty_for_all_iterations(cache_dir):
    """A lesion-free case: iteration 0 decides everything."""
    shape = (60, 60, 48)
    ct, pet, gt, _ = make_phantom(shape=shape, blobs=[], background_suv=0.5)
    speck = np.zeros(shape, dtype=bool)
    speck[30:33, 30, 24] = True
    pet[speck] = 2.0
    base = ThresholdBase(thr=1.8, conf=0.5)  # small, cold and unconfident
    pp = PostProcPredictor(base, {"tracer": "fdg"})

    outs, infos = [], []
    for _ in range(5):
        outs.append(pp.predict(ct, pet, SPACING, {"tumor": [], "background": []}, None, cache_dir))
        infos.append(pp.last_info)
    assert infos[0].get("negative_gate_fired") is True, infos[0].get("negative_gate")
    for it, o in enumerate(outs):
        assert o.sum() == 0, f"iteration {it} is not empty ({o.sum()} voxels)"


def test_negative_gate_can_be_disabled(cache_dir):
    shape = (60, 60, 48)
    ct, pet, gt, _ = make_phantom(shape=shape, blobs=[], background_suv=0.5)
    pet[30:33, 30, 24] = 2.0
    pp = PostProcPredictor(
        ThresholdBase(thr=1.8),
        {"tracer": "fdg", "negative_gate": {"enabled": False}, "cleanup": {"min_volume_ml": 0.0}},
    )
    out = pp.predict(ct, pet, SPACING, None, None, cache_dir)
    assert out.sum() > 0


def test_base_without_probability_support(phantom, cache_dir):
    ct, pet = phantom["ct"], phantom["pet"]
    base = NoProbBase(thr=3.0)
    # the gate is off here: this test is about the no-softmax path, and the phantom's
    # few-mL prediction would otherwise be emptied by the volume criterion
    pp = PostProcPredictor(base, {"tracer": "fdg", "negative_gate": {"enabled": False}})
    out = pp.predict(ct, pet, SPACING, None, None, cache_dir)
    assert out.sum() > 0
    assert pp._base_supports_prob is not True


def test_pipeline_is_deterministic(phantom, tmp_path):
    """Same inputs, fresh cache -> the same output."""
    ct, pet = phantom["ct"], phantom["pet"]
    outs = []
    for i in range(2):
        d = tmp_path / f"c{i}"
        d.mkdir()
        pp = PostProcPredictor(ThresholdBase(), {"tracer": "fdg"})
        outs.append(pp.predict(ct, pet, SPACING, None, None, str(d)))
    assert np.array_equal(outs[0], outs[1])


def test_repeating_the_same_iteration_is_stable(phantom, cache_dir):
    """Re-running an iteration with the same scribbles must not drift."""
    ct, pet, gt, blobs = phantom["ct"], phantom["pet"], phantom["gt"], phantom["blobs"]
    fg = scribble_line((30, 58, 46), axis=0, length=5, mask=blobs["cool"])
    pp = PostProcPredictor(ThresholdBase(), {"tracer": "fdg"})
    s = {"tumor": fg, "background": []}
    a = pp.predict(ct, pet, SPACING, s, None, cache_dir)
    b = pp.predict(ct, pet, SPACING, s, None, cache_dir)
    c = pp.predict(ct, pet, SPACING, s, None, cache_dir)
    assert np.array_equal(a, b) and np.array_equal(b, c)


def test_config_from_dict_rejects_typos():
    with pytest.raises(KeyError):
        PostProcConfig.from_dict({"cleanupp": {}})
    with pytest.raises(KeyError):
        PostProcConfig.from_dict({"cleanup": {"min_volume_mL": 1.0}})


def test_config_nested_override():
    cfg = PostProcConfig.from_dict({"cleanup": {"min_volume_ml": 1.5}, "tracer": "psma"})
    assert cfg.cleanup.min_volume_ml == 1.5
    assert cfg.cleanup.suv_gate == 4.0  # untouched default
    assert cfg.tracer == "psma"


def test_cache_dir_is_optional(phantom):
    ct, pet = phantom["ct"], phantom["pet"]
    pp = PostProcPredictor(ThresholdBase(), {"tracer": "fdg"})
    out = pp.predict(ct, pet, SPACING, None, None, None)
    assert out.shape == pet.shape


def test_stale_cache_from_another_case_is_discarded(phantom, cache_dir):
    ct, pet = phantom["ct"], phantom["pet"]
    pp = PostProcPredictor(ThresholdBase(), {"tracer": "fdg"})
    pp.predict(ct, pet, SPACING, {"tumor": [[1, 1, 1]], "background": []}, None, cache_dir)
    small_ct, small_pet, _, _ = make_phantom(shape=(30, 30, 24), blobs=[])
    out = pp.predict(small_ct, small_pet, SPACING, None, None, cache_dir)
    assert out.shape == (30, 30, 24)
    assert pp.last_info["iteration"] == 0, "the stale state was not reset"


# ---------------------------------------------------------------------------
# no persistence (preliminary phase) and the 6-iteration protocol
# ---------------------------------------------------------------------------
def test_pipeline_without_any_state_directory(phantom, tmp_path):
    """The final test gives us a persistent state dir; the preliminary phase does not."""
    ct, pet, gt, blobs = phantom["ct"], phantom["pet"], phantom["gt"], phantom["blobs"]
    missing = str(tmp_path / "no" / "such" / "dir")
    pp = PostProcPredictor(ThresholdBase(), {"tracer": "fdg"})
    fg = scribble_line((30, 58, 46), axis=0, length=5, mask=blobs["cool"])

    out = pp.predict(ct, pet, SPACING, {"tumor": fg, "background": []}, None, None)
    assert check_constraints(out, fg, [])["ok"]
    assert pp.last_info["iteration_source"] == "scribbles"
    assert pp.last_info["iteration"] == 1, "one scribble event -> index 1"

    out2 = pp.predict(ct, pet, SPACING, {"tumor": fg, "background": []}, None, missing)
    assert np.array_equal(out, out2), "with no usable state the result must be stable"


def test_negative_case_stays_empty_for_six_iterations_without_state(tmp_path):
    """Lesion-absent cases never get a scribble and are re-scored every iteration."""
    shape = (60, 60, 48)
    ct, pet, gt, _ = make_phantom(shape=shape, blobs=[], background_suv=0.5)
    pet[30:33, 30, 24] = 2.0
    pp = PostProcPredictor(ThresholdBase(thr=1.8, conf=0.5), {"tracer": "fdg"})
    for it in range(6):
        out = pp.predict(ct, pet, SPACING, {"tumor": [], "background": []}, None, None)
        assert out.sum() == 0, f"iteration {it} is not empty"
        assert pp.last_info["negative_gate_fired"] is True


def test_six_iteration_protocol_with_a_persistent_state_dir(phantom, cache_dir):
    ct, pet, gt, blobs = phantom["ct"], phantom["pet"], phantom["gt"], phantom["blobs"]
    pp = PostProcPredictor(ThresholdBase(), {"tracer": "fdg"})
    s = {"tumor": [], "background": []}
    fg = scribble_line((30, 58, 46), axis=0, length=5, mask=blobs["cool"])
    for it in range(6):
        if it == 3:
            s["tumor"] += fg
        out = pp.predict(ct, pet, SPACING, s, None, cache_dir)
        # index == number of scribble events seen so far, derived from the input alone
        assert pp.last_info["iteration"] == (1 if it >= 3 else 0)
        assert pp.last_info["iteration_source"] == "scribbles"
        assert pp.last_info["n_calls"] == it + 1, "the state dir counts calls"
        assert check_constraints(out, s["tumor"], s["background"])["ok"]


def test_already_satisfied_scribble_does_not_change_the_output(phantom, cache_dir):
    """Replayed Category-2 scribbles are often already satisfied -- that must be inert."""
    ct, pet, gt, blobs = phantom["ct"], phantom["pet"], phantom["gt"], phantom["blobs"]
    pp = PostProcPredictor(ThresholdBase(thr=3.0), {"tracer": "fdg"})
    base_out = pp.predict(ct, pet, SPACING, None, None, cache_dir)

    inside = np.argwhere(base_out > 0)[:5]
    fg = [[int(v) for v in p] for p in inside]
    with_scribble = pp.predict(ct, pet, SPACING, {"tumor": fg, "background": []}, None, cache_dir)
    assert np.array_equal(base_out, with_scribble), "a satisfied scribble changed the mask"


# ---------------------------------------------------------------------------
# purity: the state directory is a speed cache, never a correctness requirement
# ---------------------------------------------------------------------------
def _scribble_sequence(blobs, fp):
    """The scribble json as the evaluator would grow it over 6 iterations."""
    fg = scribble_line((56, 30, 34), axis=0, length=5, mask=blobs["warm"])
    bg = scribble_line((60, 60, 20), axis=0, length=5, mask=fp)
    seq = []
    acc = {"tumor": [], "background": []}
    for it in range(6):
        if it == 1:
            acc["background"] = acc["background"] + bg
        if it == 3:
            acc["tumor"] = acc["tumor"] + fg
        seq.append({"tumor": list(acc["tumor"]), "background": list(acc["background"])})
    return seq


def test_output_is_identical_with_and_without_a_state_directory(phantom, tmp_path):
    """The output is a pure function of (CT, PET, accumulated scribbles).

    The preliminary phase has no persistence at all, so deleting the state dir between
    calls must change nothing; the cache is a speed optimisation only.
    """
    ct, pet, gt, blobs = phantom["ct"], phantom["pet"], phantom["gt"], phantom["blobs"]
    fp = sphere_mask(pet.shape, (60, 60, 20), 7.0, SPACING)
    seq = _scribble_sequence(blobs, fp)

    cache = str(tmp_path / "state")
    os.makedirs(cache, exist_ok=True)
    pure = {"tracer": "fdg", "skip_inference_if_satisfied": False}
    pp_cached = PostProcPredictor(ThresholdBase(fp_blob=fp), pure)
    with_state = [pp_cached.predict(ct, pet, SPACING, s, None, cache) for s in seq]

    pp_stateless = PostProcPredictor(ThresholdBase(fp_blob=fp), pure)
    without_state = [pp_stateless.predict(ct, pet, SPACING, s, None, None) for s in seq]

    for it, (a, b) in enumerate(zip(with_state, without_state)):
        assert np.array_equal(a, b), f"iteration {it} differs when the state dir is missing"


def test_state_deleted_between_iterations_changes_nothing(phantom, tmp_path):
    import shutil

    ct, pet, gt, blobs = phantom["ct"], phantom["pet"], phantom["gt"], phantom["blobs"]
    fp = sphere_mask(pet.shape, (60, 60, 20), 7.0, SPACING)
    seq = _scribble_sequence(blobs, fp)
    cache = str(tmp_path / "state")

    pure = {"tracer": "fdg", "skip_inference_if_satisfied": False}
    keep = PostProcPredictor(ThresholdBase(fp_blob=fp), pure)
    os.makedirs(cache, exist_ok=True)
    a = [keep.predict(ct, pet, SPACING, s, None, cache) for s in seq]

    wipe = PostProcPredictor(ThresholdBase(fp_blob=fp), pure)
    b = []
    for s in seq:
        shutil.rmtree(cache, ignore_errors=True)
        os.makedirs(cache, exist_ok=True)
        b.append(wipe.predict(ct, pet, SPACING, s, None, cache))

    for it, (x, y) in enumerate(zip(a, b)):
        assert np.array_equal(x, y), f"iteration {it} depends on persisted state"


def test_blending_is_opt_in_and_breaks_purity_when_enabled(phantom, tmp_path):
    """The blend modes are ablation knobs; "none" is the default."""
    from postproc.config import PostProcConfig

    assert PostProcConfig().monotone.mode == "none"
    ct, pet, gt, blobs = phantom["ct"], phantom["pet"], phantom["gt"], phantom["blobs"]
    cache = str(tmp_path / "state")
    os.makedirs(cache, exist_ok=True)
    # the gate short-circuits the blend by design, so it is off for this test
    pp = PostProcPredictor(ThresholdBase(), {"tracer": "fdg", "monotone": {"mode": "ema_minmax"},
                                             "negative_gate": {"enabled": False}})
    pp.predict(ct, pet, SPACING, None, None, cache)
    out = pp.predict(ct, pet, SPACING, None, None, cache)
    assert out.shape == pet.shape
    assert pp.last_info.get("monotone_applied") == "probability"


# ---------------------------------------------------------------------------
# the negative gate as the dominant lever
# ---------------------------------------------------------------------------
def test_tumor_scribble_permanently_ungates_and_restores_the_full_prediction(cache_dir):
    """A tumor scribble proves the case is positive: emit the whole model prediction,
    not merely a region grown around the seed."""
    shape = (64, 64, 48)
    ct, pet, gt, _ = make_phantom(shape=shape, blobs=[], background_suv=0.5)
    lesions = []
    for c in ((20, 20, 16), (44, 40, 32)):
        m = sphere_mask(shape, c, 4.0, SPACING)
        pet[m] = 2.2
        lesions.append(m)
    base = ThresholdBase(thr=1.9, conf=0.5)   # small, cold and unconfident -> gate fires
    pp = PostProcPredictor(base, {"tracer": "fdg", "cleanup": {"min_volume_ml": 0.1}})

    empty = pp.predict(ct, pet, SPACING, {"tumor": [], "background": []}, None, cache_dir)
    assert empty.sum() == 0 and pp.last_info["negative_gate_fired"] is True

    fg = [[20, 20, 16]]
    out = pp.predict(ct, pet, SPACING, {"tumor": fg, "background": []}, None, cache_dir)
    assert pp.last_info["negative_gate_fired"] is False
    assert pp.last_info["negative_gate"]["blocked_by"] == ["tumor_scribble_proves_positive"]
    assert check_constraints(out, fg, [])["ok"]
    # the other lesion, which no scribble ever mentioned, must be back too
    assert (out.astype(bool) & lesions[1]).sum() > 0, "only the scribbled lesion was restored"


def test_cleanup_never_empties_a_positive_prediction(cache_dir):
    """Emptying a prediction is the negative gate's decision, never cleanup's."""
    shape = (60, 60, 48)
    ct, pet, gt, _ = make_phantom(shape=shape, blobs=[], background_suv=0.5)
    pet[30:33, 30, 24] = 2.6                       # one small, coldish speck
    pp = PostProcPredictor(
        ThresholdBase(thr=2.0, conf=0.95),
        # the gate is held off by a volume threshold below the speck's 0.037 mL, so that
        # what this test measures is cleanup's behaviour and not the gate's
        {"tracer": "fdg", "cleanup": {"min_volume_ml": 50.0, "suv_gate": 99.0},
         "negative_gate": {"max_total_volume_ml": 0.02}},
    )
    out = pp.predict(ct, pet, SPACING, None, None, cache_dir)
    assert pp.last_info["negative_gate_fired"] is False
    assert out.sum() > 0, "cleanup emptied a prediction the gate had not vetoed"
    assert pp.last_info["empty_without_gate"] is False
    assert pp.last_info["cleanup"]["small_components"]["n_rescued_from_emptying"] >= 1


def test_rescue_knob_recovers_similar_lesions(cache_dir):
    """Ablation knob: a tumor scribble lowers the global threshold to what that lesion
    needed, so lesions of similar conspicuity elsewhere come back too."""
    shape = (64, 64, 48)
    ct, pet, gt, _ = make_phantom(shape=shape, blobs=[], background_suv=0.5)
    faint = []
    for c in ((20, 20, 16), (44, 40, 32)):
        m = sphere_mask(shape, c, 6.0, SPACING)
        pet[m] = 3.0
        faint.append(m)

    class Timid:
        """Confident about nothing: 0.35 on both lesions, below the 0.5 threshold."""

        name = "timid"

        def predict(self, ct, pet, spacing, scribbles=None, prev_pred=None,
                    case_cache_dir=None, *, return_probabilities=False, **kw):
            p = np.zeros(pet.shape, dtype=np.float32)
            for m in faint:
                p[m] = 0.35
            m0 = (p >= 0.5).astype(np.uint8)
            return (m0, p) if return_probabilities else m0

    fg = [[20, 20, 16]]
    off = PostProcPredictor(Timid(), {"tracer": "fdg", "negative_gate": {"enabled": False}})
    on = PostProcPredictor(
        Timid(),
        {"tracer": "fdg", "negative_gate": {"enabled": False},
         "rescue_threshold_from_scribble": True},
    )
    a = off.predict(ct, pet, SPACING, {"tumor": fg, "background": []}, None, None)
    b = on.predict(ct, pet, SPACING, {"tumor": fg, "background": []}, None, None)

    assert on.last_info["rescue"]["applied"] is True
    assert (b.astype(bool) & faint[1]).sum() > (a.astype(bool) & faint[1]).sum()
    assert check_constraints(b, fg, [])["ok"]


def test_cached_previous_prediction_is_not_handed_to_the_base_by_default(phantom, cache_dir):
    """A base predictor that consumes prev_pred would make the output stateful."""
    seen = []

    class Spy(ThresholdBase):
        def predict(self, ct, pet, spacing, scribbles=None, prev_pred=None,
                    case_cache_dir=None, *, return_probabilities=False, **kw):
            seen.append(prev_pred)
            return super().predict(ct, pet, spacing, scribbles, prev_pred,
                                   case_cache_dir, return_probabilities=return_probabilities)

    ct, pet = phantom["ct"], phantom["pet"]
    pp = PostProcPredictor(Spy(), {"tracer": "fdg"})
    pp.predict(ct, pet, SPACING, None, None, cache_dir)
    pp.predict(ct, pet, SPACING, None, None, cache_dir)
    assert seen == [None, None], "the cached mask leaked into the base predictor"

    seen.clear()
    pp2 = PostProcPredictor(Spy(), {"tracer": "fdg", "pass_cached_prev_pred": True})
    pp2.predict(ct, pet, SPACING, None, None, cache_dir)
    assert seen[0] is not None, "the knob did not forward the cached mask"

    # an explicit prev_pred from the caller is always forwarded
    seen.clear()
    explicit = np.zeros(pet.shape, dtype=np.uint8)
    pp3 = PostProcPredictor(Spy(), {"tracer": "fdg"})
    pp3.predict(ct, pet, SPACING, None, explicit, cache_dir)
    assert seen[0] is not None


def test_second_cleanup_pass_prunes_what_compliance_created(cache_dir):
    """A background scribble splits a component and leaves a small cold fragment behind,
    which the first cleanup pass ran too early to see."""
    shape = (60, 60, 48)
    ct, pet, gt, _ = make_phantom(shape=shape, blobs=[], background_suv=0.5)
    # one bar: a hot core plus a cool tail, joined -- one component for the model
    pet[20:26, 30, 24] = 9.0          # core
    pet[26:40, 30, 24] = 2.4          # tail, above the 2.0 threshold, below the SUV gate
    bg = scribble_line((32, 30, 24), axis=0, length=3)   # scribble across the tail

    cfg = {"tracer": "fdg", "negative_gate": {"enabled": False},
           "cleanup": {"min_volume_ml": 0.3, "suv_gate": 4.0, "fill_holes": False}}
    base = ThresholdBase(thr=2.0, conf=0.95)
    one = PostProcPredictor(base, cfg).predict(
        ct, pet, SPACING, {"tumor": [], "background": bg}, None, cache_dir)
    cfg2 = dict(cfg, cleanup_after_compliance=True)
    two = PostProcPredictor(ThresholdBase(thr=2.0, conf=0.95), cfg2).predict(
        ct, pet, SPACING, {"tumor": [], "background": bg}, None, os.path.join(cache_dir, "b"))

    import cc3d
    n1 = cc3d.connected_components(np.ascontiguousarray(one > 0).view(np.uint8),
                                   connectivity=18, return_N=True)[1]
    n2 = cc3d.connected_components(np.ascontiguousarray(two > 0).view(np.uint8),
                                   connectivity=18, return_N=True)[1]
    assert n2 < n1, f"second pass pruned nothing ({n1} -> {n2} components)"
    assert two[20:26, 30, 24].sum() == 6, "the hot core must survive"
    assert not (two[np.array(bg)[:, 0], np.array(bg)[:, 1], np.array(bg)[:, 2]]).any()


def test_second_cleanup_pass_still_honours_a_tumor_scribble(cache_dir):
    """G2 survives the extra pass: a scribbled component is protected even when the rule
    would otherwise delete it."""
    shape = (60, 60, 48)
    ct, pet, gt, _ = make_phantom(shape=shape, blobs=[], background_suv=0.5)
    pet[20:26, 30, 24] = 9.0
    pet[40:44, 20, 24] = 2.4          # small, cold, and about to be scribbled as tumor
    fg = scribble_line((41, 20, 24), axis=0, length=3)
    pp = PostProcPredictor(
        ThresholdBase(thr=2.0, conf=0.95),
        {"tracer": "fdg", "negative_gate": {"enabled": False},
         "cleanup_after_compliance": True,
         "cleanup": {"min_volume_ml": 5.0, "suv_gate": 99.0, "fill_holes": False}},
    )
    out = pp.predict(ct, pet, SPACING, {"tumor": fg, "background": []}, None, cache_dir)
    pts = np.array(fg)
    assert out[pts[:, 0], pts[:, 1], pts[:, 2]].all()
    assert pp.last_info["constraints"]["ok"]


def test_second_cleanup_pass_is_off_by_default(phantom, cache_dir):
    cfg = PostProcConfig()
    assert cfg.cleanup_after_compliance is False
    pet, ct = phantom["pet"], phantom["ct"]
    pp = PostProcPredictor(ThresholdBase(thr=3.0), {"tracer": "fdg"})
    pp.predict(ct, pet, SPACING, None, None, cache_dir)
    assert "cleanup_after_compliance" not in pp.last_info


def test_bridging_merges_components_after_compliance_and_keeps_g1(cache_dir):
    """Two hot blobs two voxels apart are one component after the merge stage, and the
    background scribble between them is still outside the mask."""
    import cc3d
    shape = (60, 60, 48)
    ct, pet, gt, _ = make_phantom(shape=shape, blobs=[], background_suv=0.5)
    pet[20:26, 30, 24] = 9.0
    pet[28:34, 30, 24] = 9.0          # 2-voxel gap
    n = lambda m: cc3d.connected_components(
        np.ascontiguousarray(m > 0).view(np.uint8), connectivity=18, return_N=True)[1]

    cfg = {"tracer": "fdg", "negative_gate": {"enabled": False},
           "cleanup": {"fill_holes": False}}
    off = PostProcPredictor(ThresholdBase(thr=2.0, conf=0.95), cfg).predict(
        ct, pet, SPACING, None, None, cache_dir)
    on = PostProcPredictor(
        ThresholdBase(thr=2.0, conf=0.95),
        {**cfg, "cleanup": {"fill_holes": False, "bridge_closing_voxels": 1}},
    ).predict(ct, pet, SPACING, None, None, os.path.join(cache_dir, "b"))
    assert n(off) == 2 and n(on) == 1

    # ... and a background scribble in the gap is never bridged over
    bg = [[26, 30, 24], [27, 30, 24]]
    pp = PostProcPredictor(
        ThresholdBase(thr=2.0, conf=0.95),
        {**cfg, "cleanup": {"fill_holes": False, "bridge_closing_voxels": 1}},
    )
    out = pp.predict(ct, pet, SPACING, {"tumor": [], "background": bg}, None,
                     os.path.join(cache_dir, "c"))
    assert out[26, 30, 24] == 0 and out[27, 30, 24] == 0
    assert pp.last_info["constraints"]["ok"]


# ---------------------------------------------------------------------------
# regression: the pipeline must not damage a base mask nobody corrected
# ---------------------------------------------------------------------------
def test_already_satisfied_scribbles_leave_the_base_mask_intact(phantom, cache_dir):
    """With both constraints already satisfied the output must be the base mask.

    Any reshaping here is amplified on the next iteration through the interactive
    model's previous-prediction channel.
    """
    ct, pet, gt, blobs = phantom["ct"], phantom["pet"], phantom["gt"], phantom["blobs"]
    base = ThresholdBase(thr=3.0, conf=0.95)
    raw = base.predict(ct, pet, SPACING, None, None, None)

    fg = [[int(a), int(b), int(c)] for a, b, c in np.argwhere(raw > 0)[:6]]   # inside
    bg = [[2, 2, 2], [2, 3, 2]]                                              # outside
    assert raw[2, 2, 2] == 0

    pp = PostProcPredictor(base, {"tracer": "fdg"})
    out = pp.predict(ct, pet, SPACING, {"tumor": fg, "background": bg}, None, cache_dir)

    assert check_constraints(out, fg, bg)["ok"]
    inter = int((out.astype(bool) & (raw > 0)).sum())
    d = 2 * inter / (int(out.sum()) + int((raw > 0).sum()))
    assert d > 0.99, f"a satisfied scribble set reshaped the mask (Dice vs base {d:.3f})"
    assert pp.last_info["base_removed_ml"] < 0.1
    assert pp.last_info["fg_compliance"]["n_clusters_satisfied"] >= 1
    assert pp.last_info["bg_compliance"]["n_components_hit"] == 0


def test_damage_guard_restores_an_unscribbled_component():
    """Unit test of G4: compliance may not gut a component no scribble points at."""
    from postproc.config import PostProcConfig
    from postproc.pipeline import _apply_damage_guard

    shape = (40, 40, 30)
    base = np.zeros(shape, dtype=np.uint8)
    base[5:15, 5:15, 5:15] = 1        # component A -- no scribble anywhere near it
    base[25:35, 25:35, 15:25] = 1     # component B -- holds the background scribble
    after_cleanup = base.copy()

    damaged = base.copy()
    damaged[5:14, 5:15, 5:15] = 0     # 90 % of A destroyed
    damaged[25:34, 25:35, 15:25] = 0  # 90 % of B destroyed, but B was scribbled

    cfg = PostProcConfig()
    out, info = _apply_damage_guard(
        damaged, base, after_cleanup, [[30, 30, 20]], SPACING, cfg
    )
    assert info["n_components_restored"] == 1
    assert out[5:15, 5:15, 5:15].all(), "component A was not restored"
    assert out[25:34, 25:35, 15:25].sum() == 0, "the scribbled component was restored"


def test_guard_leaves_cleanup_removals_alone():
    """Cleanup's size/SUV rules are deliberately exempt from the guard."""
    from postproc.config import PostProcConfig
    from postproc.pipeline import _apply_damage_guard

    shape = (30, 30, 20)
    base = np.zeros(shape, dtype=np.uint8)
    base[5:8, 5, 5] = 1                       # a speck cleanup will prune
    after_cleanup = np.zeros(shape, dtype=np.uint8)   # cleanup already removed it
    out, info = _apply_damage_guard(
        after_cleanup, base, after_cleanup, [], SPACING, PostProcConfig()
    )
    assert info["n_components_restored"] == 0
    assert out.sum() == 0


# ---------------------------------------------------------------------------
# skip inference when the new scribble is already satisfied
# ---------------------------------------------------------------------------
class CountingBase(ThresholdBase):
    """Counts how many times the network was actually asked for a prediction."""

    name = "counting"


def test_skip_when_the_new_scribble_is_already_satisfied(phantom, cache_dir):
    """Category-2 scribbles are replayed from the baseline's errors, so many of them
    already hold for our mask.  Re-running the network on such a scribble changes the
    output globally in response to a correction that asked for nothing."""
    ct, pet, gt, blobs = phantom["ct"], phantom["pet"], phantom["gt"], phantom["blobs"]
    base = CountingBase(thr=3.0)
    pp = PostProcPredictor(base, {"tracer": "fdg", "negative_gate": {"enabled": False}})

    first = pp.predict(ct, pet, SPACING, {"tumor": [], "background": []}, None, cache_dir)
    assert base.calls == 1

    inside = [[int(a), int(b), int(c)] for a, b, c in np.argwhere(first > 0)[:4]]
    outside = [[2, 2, 2], [2, 3, 2]]
    assert first[2, 2, 2] == 0

    second = pp.predict(ct, pet, SPACING,
                        {"tumor": inside, "background": outside}, None, cache_dir)
    assert base.calls == 1, "the network was called for a scribble that asked for nothing"
    assert pp.last_info["skipped_inference"] is True
    assert pp.last_info["n_new_tumor"] == 4 and pp.last_info["n_new_background"] == 2
    assert np.array_equal(first, second), "the previous mask was not returned unchanged"
    assert check_constraints(second, inside, outside)["ok"]


def test_no_skip_when_the_new_scribble_asks_for_something(phantom, cache_dir):
    """Under simulated interaction a new scribble always lies on our own error, so the
    rule must be inert there: a tumor point outside the mask, or a background point
    inside it, both force the normal path."""
    ct, pet, gt, blobs = phantom["ct"], phantom["pet"], phantom["gt"], phantom["blobs"]
    base = CountingBase(thr=3.0)
    pp = PostProcPredictor(base, {"tracer": "fdg", "negative_gate": {"enabled": False}})
    first = pp.predict(ct, pet, SPACING, {"tumor": [], "background": []}, None, cache_dir)

    missed = [[int(a), int(b), int(c)] for a, b, c in np.argwhere((first == 0) & (gt > 0))[:3]]
    assert missed, "fixture needs a false negative"
    pp.predict(ct, pet, SPACING, {"tumor": missed, "background": []}, None, cache_dir)
    assert base.calls == 2, "an unsatisfied tumor scribble must reach the network"
    assert not pp.last_info.get("skipped_inference")

    base2 = CountingBase(thr=3.0)
    pp2 = PostProcPredictor(base2, {"tracer": "fdg", "negative_gate": {"enabled": False}})
    d2 = os.path.join(cache_dir, "b")
    os.makedirs(d2, exist_ok=True)
    f2 = pp2.predict(ct, pet, SPACING, {"tumor": [], "background": []}, None, d2)
    hit = [[int(a), int(b), int(c)] for a, b, c in np.argwhere(f2 > 0)[:3]]
    pp2.predict(ct, pet, SPACING, {"tumor": [], "background": hit}, None, d2)
    assert base2.calls == 2, "an unsatisfied background scribble must reach the network"


def test_no_skip_without_state_or_without_new_points(phantom, cache_dir):
    ct, pet, gt = phantom["ct"], phantom["pet"], phantom["gt"]

    # no state directory -> nothing to reuse, normal path
    base = CountingBase(thr=3.0)
    pp = PostProcPredictor(base, {"tracer": "fdg", "negative_gate": {"enabled": False}})
    pp.predict(ct, pet, SPACING, {"tumor": [], "background": []}, None, None)
    pp.predict(ct, pet, SPACING, {"tumor": [], "background": []}, None, None)
    assert base.calls == 2 and not pp.last_info.get("skipped_inference")

    # a state directory but no *new* points -> normal path (the rule is about
    # corrections we have already made, not a general "reuse the last answer")
    base2 = CountingBase(thr=3.0)
    pp2 = PostProcPredictor(base2, {"tracer": "fdg", "negative_gate": {"enabled": False}})
    first = pp2.predict(ct, pet, SPACING, {"tumor": [], "background": []}, None, cache_dir)
    inside = [[int(a), int(b), int(c)] for a, b, c in np.argwhere(first > 0)[:2]]
    pp2.predict(ct, pet, SPACING, {"tumor": inside, "background": []}, None, cache_dir)
    assert pp2.last_info["skipped_inference"] is True
    pp2.predict(ct, pet, SPACING, {"tumor": inside, "background": []}, None, cache_dir)
    assert not pp2.last_info.get("skipped_inference"), "no new points must not skip"


def test_skip_can_be_disabled(phantom, cache_dir):
    ct, pet, gt = phantom["ct"], phantom["pet"], phantom["gt"]
    base = CountingBase(thr=3.0)
    pp = PostProcPredictor(base, {"tracer": "fdg", "negative_gate": {"enabled": False}, "skip_inference_if_satisfied": False})
    first = pp.predict(ct, pet, SPACING, {"tumor": [], "background": []}, None, cache_dir)
    inside = [[int(a), int(b), int(c)] for a, b, c in np.argwhere(first > 0)[:3]]
    pp.predict(ct, pet, SPACING, {"tumor": inside, "background": []}, None, cache_dir)
    assert base.calls == 2 and not pp.last_info.get("skipped_inference")


def test_no_skip_when_the_previous_mask_is_empty(phantom, cache_dir):
    """An empty mask satisfies every background point vacuously.

    Skipping there freezes an empty prediction for another iteration, which on a
    positive case is the worst outcome the evaluator can score -- measured as a
    -1.21 AUC-Dice regression on one Category-2 replay case before this guard.
    """
    ct, pet, gt, blobs = phantom["ct"], phantom["pet"], phantom["gt"], phantom["blobs"]
    base = CountingBase(thr=3.0)
    pp = PostProcPredictor(base, {"tracer": "fdg"})   # gate on -> phantom is emptied
    first = pp.predict(ct, pet, SPACING, {"tumor": [], "background": []}, None, cache_dir)
    assert first.sum() == 0, "fixture needs an empty first prediction"

    pp.predict(ct, pet, SPACING, {"tumor": [], "background": [[2, 2, 2]]}, None, cache_dir)
    assert base.calls == 2, "skipped inference on an empty mask"
    assert not pp.last_info.get("skipped_inference")
