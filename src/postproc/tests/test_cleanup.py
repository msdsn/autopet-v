"""Small-component pruning, SUV floors, hole filling."""

from __future__ import annotations

import numpy as np
import pytest

from postproc.cleanup import (
    bridge_components,
    cleanup_mask,
    fill_small_holes,
    recruit_components,
    remove_components_v2,
    remove_small_components,
    resolve_v2_rule,
    resolve_v2_rules,
    tracer_suv_floor,
)
from postproc.config import CleanupConfig

from conftest import SPACING, make_phantom


def _speck(mask, pet, centre, n=3, suv=1.0):
    i, j, k = centre
    mask[i : i + n, j, k] = 1
    pet[i : i + n, j, k] = suv
    return mask, pet


def test_small_cold_speck_removed_hot_speck_kept():
    shape = (60, 60, 48)
    ct, pet, gt, _ = make_phantom(shape=shape, blobs=[], background_suv=0.5)
    mask = np.zeros(shape, dtype=np.uint8)
    mask, pet = _speck(mask, pet, (10, 10, 10), n=3, suv=1.8)   # cold speck -> gone
    mask, pet = _speck(mask, pet, (40, 40, 30), n=3, suv=9.0)   # hot speck  -> kept

    out, info = remove_small_components(
        mask, pet, SPACING, min_volume_ml=0.3, suv_gate=4.0, return_info=True
    )
    assert info["n_components"] == 2
    assert info["n_removed"] == 1
    assert out[10:13, 10, 10].sum() == 0
    assert out[40:43, 40, 30].sum() == 3


def test_component_with_a_tumor_scribble_is_never_removed():
    shape = (60, 60, 48)
    ct, pet, gt, _ = make_phantom(shape=shape, blobs=[], background_suv=0.5)
    mask = np.zeros(shape, dtype=np.uint8)
    mask, pet = _speck(mask, pet, (10, 10, 10), n=3, suv=1.0)
    protect = [[11, 10, 10]]
    out = remove_small_components(
        mask, pet, SPACING, min_volume_ml=10.0, suv_gate=99.0, protect_points=protect
    )
    assert out[10:13, 10, 10].sum() == 3


def test_large_components_untouched_small_cold_one_pruned(phantom):
    """The "cool" blob is 0.24 mL at SUV 2.2, below both gates; the other two stay."""
    pet, gt, blobs = phantom["pet"], phantom["gt"], phantom["blobs"]
    out, info = remove_small_components(
        gt, pet, SPACING, min_volume_ml=0.3, suv_gate=4.0, return_info=True
    )
    assert info["n_components"] == 3 and info["n_removed"] == 1
    assert not (out.astype(bool) & blobs["cool"]).any()
    for name in ("hot", "warm"):
        assert (out.astype(bool) & blobs[name]).sum() == blobs[name].sum()


def test_small_cold_component_survives_when_the_gates_are_loosened(phantom):
    pet, gt = phantom["pet"], phantom["gt"]
    out = remove_small_components(gt, pet, SPACING, min_volume_ml=0.1, suv_gate=4.0)
    assert out.sum() == gt.sum()


def test_remove_small_components_is_idempotent():
    shape = (60, 60, 48)
    ct, pet, gt, _ = make_phantom(shape=shape, blobs=[], background_suv=0.5)
    mask = np.zeros(shape, dtype=np.uint8)
    mask, pet = _speck(mask, pet, (10, 10, 10), n=3, suv=1.0)
    mask, pet = _speck(mask, pet, (40, 40, 30), n=3, suv=9.0)
    once = remove_small_components(mask, pet, SPACING)
    twice = remove_small_components(once, pet, SPACING)
    assert np.array_equal(once, twice)


@pytest.mark.parametrize("tracer,floor", [("fdg", 1.5), ("psma", 1.0)])
def test_tracer_suv_floor_component_mode(tracer, floor):
    shape = (60, 60, 48)
    ct, pet, gt, _ = make_phantom(shape=shape, blobs=[], background_suv=0.3)
    mask = np.zeros(shape, dtype=np.uint8)
    # a big but sub-floor component and a big supra-floor one
    mask[10:20, 10:20, 10:20] = 1
    pet[10:20, 10:20, 10:20] = floor - 0.2
    mask[35:45, 35:45, 25:35] = 1
    pet[35:45, 35:45, 25:35] = floor + 2.0

    out, info = tracer_suv_floor(mask, pet, tracer, return_info=True)
    assert info["floor"] == pytest.approx(floor)
    assert out[10:20, 10:20, 10:20].sum() == 0
    assert out[35:45, 35:45, 25:35].sum() == 10 * 10 * 10


def test_tracer_suv_floor_voxel_mode_trims_edges():
    shape = (40, 40, 40)
    ct, pet, gt, _ = make_phantom(shape=shape, blobs=[], background_suv=0.3)
    mask = np.zeros(shape, dtype=np.uint8)
    mask[10:20, 10:20, 10:20] = 1
    pet[10:20, 10:20, 10:20] = 5.0
    pet[10, 10:20, 10:20] = 0.5  # one cold face
    out = tracer_suv_floor(mask, pet, "fdg", mode="voxel")
    assert out[10, 15, 15] == 0
    assert out[15, 15, 15] == 1


def test_suv_floor_off_is_a_noop(phantom):
    pet, gt = phantom["pet"], phantom["gt"]
    out = tracer_suv_floor(gt, pet, "fdg", mode="off")
    assert np.array_equal(out.astype(bool), gt.astype(bool))


def test_fill_small_holes():
    shape = (40, 40, 40)
    mask = np.zeros(shape, dtype=np.uint8)
    mask[10:25, 10:25, 10:25] = 1
    mask[16:18, 16:18, 16:18] = 0  # 8-voxel cavity = 0.1 mL
    out, info = fill_small_holes(mask, SPACING, max_hole_ml=0.5, return_info=True)
    assert info["n_filled"] == 1
    assert out[16:18, 16:18, 16:18].all()

    # too big a cavity is left alone
    mask2 = np.zeros(shape, dtype=np.uint8)
    mask2[5:35, 5:35, 5:35] = 1
    mask2[12:28, 12:28, 12:28] = 0  # 4096 voxels = 51 mL
    out2, info2 = fill_small_holes(mask2, SPACING, max_hole_ml=0.5, return_info=True)
    assert info2["n_filled"] == 0


def test_fill_small_holes_never_fills_a_background_scribble():
    shape = (40, 40, 40)
    mask = np.zeros(shape, dtype=np.uint8)
    mask[10:25, 10:25, 10:25] = 1
    mask[16:18, 16:18, 16:18] = 0
    forbidden = np.zeros(shape, dtype=bool)
    forbidden[16, 16, 16] = True
    out = fill_small_holes(mask, SPACING, max_hole_ml=0.5, forbidden=forbidden)
    assert out[16, 16, 16] == 0
    assert out[17, 17, 17] == 1


def test_cleanup_mask_end_to_end(phantom):
    pet, gt, blobs = phantom["pet"], phantom["gt"], phantom["blobs"]
    mask = gt.copy()
    mask[5:8, 5, 5] = 1  # cold speck in the air
    out, info = cleanup_mask(mask, pet, SPACING, tracer="fdg", cfg=CleanupConfig(), return_info=True)
    assert out[5:8, 5, 5].sum() == 0
    for name in ("hot", "warm"):
        assert (out.astype(bool) & blobs[name]).sum() > 0.9 * blobs[name].sum()


# ---------------------------------------------------------------------------
# rule v2: the same guards, one more axis, and the iteration-aware relaxation
# ---------------------------------------------------------------------------
def test_v2_without_a_prob_gate_is_exactly_v1(phantom):
    """With the third criterion disabled the two rules are the same function, so v2 is
    v1 plus an extra reason not to delete."""
    pet, gt = phantom["pet"], phantom["gt"]
    mask = gt.copy()
    mask[5:8, 5, 5] = 1
    v1 = remove_small_components(mask, pet, SPACING, min_volume_ml=0.3, suv_gate=4.0)
    v2 = remove_components_v2(mask, pet, SPACING, min_volume_ml=0.3, suv_gate=4.0,
                              prob_gate=None)
    assert np.array_equal(v1, v2)


def test_v2_prob_gate_is_a_conjunction_not_a_new_reason_to_delete():
    """A confident component survives even though it is small and cold."""
    shape = (60, 60, 48)
    _, pet, _, _ = make_phantom(shape=shape, blobs=[], background_suv=0.5)
    mask = np.zeros(shape, dtype=np.uint8)
    mask, pet = _speck(mask, pet, (10, 10, 10), n=3, suv=1.8)   # small, cold
    mask, pet = _speck(mask, pet, (40, 40, 30), n=3, suv=1.8)   # small, cold
    prob = np.zeros(shape, dtype=np.float32)
    prob[10:13, 10, 10] = 0.55                                  # ... and unconfident
    prob[40:43, 40, 30] = 0.99                                  # ... but confident

    out, info = remove_components_v2(
        mask, pet, SPACING, prob=prob, min_volume_ml=0.3, suv_gate=4.0,
        prob_gate=0.90, min_components_kept=0, return_info=True
    )
    assert info["n_removed"] == 1
    assert out[10:13, 10, 10].sum() == 0
    assert out[40:43, 40, 30].sum() == 3


def test_v2_prob_gate_without_a_softmax_fails_closed():
    """With no probability map the criterion cannot be checked, so nothing is deleted on
    account of it; a base predictor without a softmax must not get a harsher rule."""
    shape = (60, 60, 48)
    _, pet, _, _ = make_phantom(shape=shape, blobs=[], background_suv=0.5)
    mask = np.zeros(shape, dtype=np.uint8)
    mask, pet = _speck(mask, pet, (10, 10, 10), n=3, suv=1.8)
    out = remove_components_v2(mask, pet, SPACING, prob=None, min_volume_ml=0.3,
                               suv_gate=4.0, prob_gate=0.9, min_components_kept=0)
    assert out[10:13, 10, 10].sum() == 3


def test_v2_never_prunes_a_tumor_scribble_and_never_empties():
    shape = (60, 60, 48)
    _, pet, _, _ = make_phantom(shape=shape, blobs=[], background_suv=0.5)
    mask = np.zeros(shape, dtype=np.uint8)
    mask, pet = _speck(mask, pet, (10, 10, 10), n=3, suv=1.0)
    kept = remove_components_v2(mask, pet, SPACING, min_volume_ml=99.0, suv_gate=99.0,
                                protect_points=[[11, 10, 10]])
    assert kept[10:13, 10, 10].sum() == 3
    # ... and with no scribble at all the "never empty" guard keeps the best component
    kept = remove_components_v2(mask, pet, SPACING, min_volume_ml=99.0, suv_gate=99.0)
    assert kept.sum() == 3


def test_v2_silence_decay_protects_a_component_that_survived_iterations():
    """Same component and thresholds: pruned at iteration 0, kept at iteration 4,
    because no background scribble landed on it in between."""
    shape = (60, 60, 48)
    _, pet, _, _ = make_phantom(shape=shape, blobs=[], background_suv=0.5)
    mask = np.zeros(shape, dtype=np.uint8)
    mask, pet = _speck(mask, pet, (10, 10, 10), n=8, suv=1.8)
    mask, pet = _speck(mask, pet, (40, 40, 30), n=40, suv=9.0)   # keeps it non-empty

    early = remove_components_v2(mask, pet, SPACING, min_volume_ml=0.3, suv_gate=4.0,
                                 silence_decay=0.5, iteration=0)
    late = remove_components_v2(mask, pet, SPACING, min_volume_ml=0.3, suv_gate=4.0,
                                silence_decay=0.5, iteration=4)
    assert early[10:18, 10, 10].sum() == 0
    assert late[10:18, 10, 10].sum() == 8


def test_v2_silence_decay_does_not_protect_a_scribbled_component():
    shape = (60, 60, 48)
    _, pet, _, _ = make_phantom(shape=shape, blobs=[], background_suv=0.5)
    mask = np.zeros(shape, dtype=np.uint8)
    mask, pet = _speck(mask, pet, (10, 10, 10), n=8, suv=1.8)
    mask, pet = _speck(mask, pet, (40, 40, 30), n=40, suv=9.0)
    out = remove_components_v2(mask, pet, SPACING, min_volume_ml=0.3, suv_gate=4.0,
                               silence_decay=0.5, iteration=4,
                               background_points=[[12, 10, 10]])
    assert out[10:18, 10, 10].sum() == 0


def test_resolve_v2_rule_per_tracer_override():
    cfg = CleanupConfig(rule_v2=True, v2_min_volume_ml=0.3, v2_suv_gate=4.0,
                        v2_by_tracer={"fdg": {"v2_min_volume_ml": 1.0,
                                              "v2_prob_gate": 0.9}})
    assert resolve_v2_rule(cfg, "psma")["min_volume_ml"] == 0.3
    fdg = resolve_v2_rule(cfg, "FDG")
    assert fdg["min_volume_ml"] == 1.0 and fdg["prob_gate"] == 0.9
    with pytest.raises(KeyError):
        resolve_v2_rule(CleanupConfig(v2_by_tracer={"fdg": {"nonsense": 1}}), "fdg")


# ---------------------------------------------------------------------------
# recall recruitment
# ---------------------------------------------------------------------------
def test_recruit_adds_only_new_hot_components():
    shape = (60, 60, 48)
    _, pet, _, _ = make_phantom(shape=shape, blobs=[], background_suv=0.5)
    mask = np.zeros(shape, dtype=np.uint8)
    prob = np.zeros(shape, dtype=np.float32)

    mask[20:26, 20:26, 20:26] = 1                 # an existing component ...
    prob[19:27, 19:27, 19:27] = 0.6               # ... which grows below 0.5
    pet[20:26, 20:26, 20:26] = 8.0

    prob[40:46, 40:46, 30:36] = 0.35              # a new, hot sub-threshold finding
    pet[40:46, 40:46, 30:36] = 9.0
    prob[5:11, 5:11, 5:11] = 0.35                 # a new, cold one -> rejected
    pet[5:11, 5:11, 5:11] = 1.0

    out, info = recruit_components(mask, prob, pet, SPACING, threshold=0.3,
                                   min_suv_max=4.0, min_volume_ml=0.1, return_info=True)
    assert info["n_added"] == 1
    assert out[40:46, 40:46, 30:36].sum() == 6 * 6 * 6
    assert out[5:11, 5:11, 5:11].sum() == 0
    # additive only: the existing component is not reshaped by the lower threshold
    assert out[19, 19, 19] == 0
    assert np.array_equal(out.astype(bool) & (mask > 0), mask > 0)


def test_recruit_is_a_noop_without_a_threshold_or_a_softmax(phantom):
    pet, gt = phantom["pet"], phantom["gt"]
    assert np.array_equal(recruit_components(gt, None, pet, SPACING, threshold=0.3), gt > 0)
    assert np.array_equal(recruit_components(gt, gt.astype(np.float32), pet, SPACING,
                                             threshold=None), gt > 0)


def test_recruit_caps_the_number_it_adds():
    shape = (60, 60, 48)
    _, pet, _, _ = make_phantom(shape=shape, blobs=[], background_suv=0.5)
    mask = np.zeros(shape, dtype=np.uint8)
    mask[30:34, 30:34, 24:28] = 1
    pet[30:34, 30:34, 24:28] = 8.0
    prob = np.zeros(shape, dtype=np.float32)
    prob[30:34, 30:34, 24:28] = 0.9
    for n, i in enumerate(range(5, 45, 8)):
        prob[i:i + 4, 5:9, 5:9] = 0.35
        pet[i:i + 4, 5:9, 5:9] = 6.0 + n
    out, info = recruit_components(mask, prob, pet, SPACING, threshold=0.3,
                                   min_suv_max=4.0, min_volume_ml=0.05,
                                   max_components=2, return_info=True)
    assert info["n_candidates"] == 5 and info["n_added"] == 2


# ---------------------------------------------------------------------------
# cleanup_mask wiring
# ---------------------------------------------------------------------------
def test_cleanup_mask_defaults_are_unchanged_by_the_new_knobs(phantom):
    """The default configuration ignores the v2 knobs entirely."""
    pet, gt = phantom["pet"], phantom["gt"]
    mask = gt.copy()
    mask[5:8, 5, 5] = 1
    prob = np.where(mask > 0, 0.55, 0.0).astype(np.float32)
    a = cleanup_mask(mask, pet, SPACING, tracer="fdg", cfg=CleanupConfig())
    b = cleanup_mask(mask, pet, SPACING, tracer="fdg", cfg=CleanupConfig(),
                     prob=prob, background_points=[[5, 5, 5]], iteration=3)
    assert np.array_equal(a, b)


def test_cleanup_mask_v2_uses_the_prob_gate_and_the_tracer_override():
    shape = (60, 60, 48)
    _, pet, _, _ = make_phantom(shape=shape, blobs=[], background_suv=0.5)
    mask = np.zeros(shape, dtype=np.uint8)
    mask, pet = _speck(mask, pet, (10, 10, 10), n=3, suv=1.8)
    mask[40:46, 40:46, 30:36] = 1
    pet[40:46, 40:46, 30:36] = 9.0
    prob = np.zeros(shape, dtype=np.float32)
    prob[10:13, 10, 10] = 0.55
    prob[40:46, 40:46, 30:36] = 0.99

    cfg = CleanupConfig(rule_v2=True, v2_min_volume_ml=0.3, v2_suv_gate=4.0,
                        v2_prob_gate=None,
                        v2_by_tracer={"fdg": {"v2_prob_gate": 0.9}})
    out, info = cleanup_mask(mask, pet, SPACING, tracer="fdg", cfg=cfg, prob=prob,
                             return_info=True)
    assert info["small_components"]["n_removed"] == 1
    assert out[10:13, 10, 10].sum() == 0
    # the same case under PSMA thresholds: no prob gate -> the speck is cold and small,
    # so v1 and v2 agree and it goes anyway; assert the resolved rule, not the mask
    assert info["small_components"]["rules"][0]["prob_gate"] == 0.9


def test_cleanup_mask_silence_decay_needs_a_background_scribble():
    shape = (60, 60, 48)
    _, pet, _, _ = make_phantom(shape=shape, blobs=[], background_suv=0.5)
    mask = np.zeros(shape, dtype=np.uint8)
    mask, pet = _speck(mask, pet, (10, 10, 10), n=8, suv=1.8)
    mask, pet = _speck(mask, pet, (40, 40, 30), n=40, suv=9.0)
    cfg = CleanupConfig(rule_v2=True, v2_min_volume_ml=0.3, v2_suv_gate=4.0,
                        v2_silence_decay=0.5, v2_silence_requires_bg=True)
    # no background scribble yet -> the decay is disabled -> the speck is pruned
    out = cleanup_mask(mask, pet, SPACING, tracer="fdg", cfg=cfg, iteration=4)
    assert out[10:18, 10, 10].sum() == 0
    # a background scribble elsewhere proves the scribbles follow our errors
    out = cleanup_mask(mask, pet, SPACING, tracer="fdg", cfg=cfg, iteration=4,
                       background_points=[[41, 40, 30]])
    assert out[10:18, 10, 10].sum() == 8


# ---------------------------------------------------------------------------
# component merging (the metric does not punish multi-assignment)
# ---------------------------------------------------------------------------
def test_bridge_joins_a_two_voxel_gap_and_leaves_a_wide_one():
    import cc3d
    shape = (60, 60, 48)
    _, pet, _, _ = make_phantom(shape=shape, blobs=[], background_suv=0.5)
    mask = np.zeros(shape, dtype=np.uint8)
    mask[10:14, 30, 24] = 1
    mask[16:20, 30, 24] = 1          # 2-voxel gap  -> joined by a 1-voxel closing
    mask[30:34, 30, 24] = 1          # 10 voxels away -> untouched
    n = lambda m: cc3d.connected_components(
        np.ascontiguousarray(m > 0).view(np.uint8), connectivity=18, return_N=True)[1]
    assert n(mask) == 3
    out, info = bridge_components(mask, SPACING, closing_voxels=1, return_info=True)
    assert n(out) == 2
    assert info["added_voxels"] == 2 and not info["refused"]


def test_bridge_is_a_noop_at_zero_and_never_fills_a_forbidden_voxel():
    shape = (60, 60, 48)
    mask = np.zeros(shape, dtype=np.uint8)
    mask[10:14, 30, 24] = 1
    mask[16:20, 30, 24] = 1
    assert np.array_equal(bridge_components(mask, SPACING, closing_voxels=0), mask > 0)
    forbidden = np.zeros(shape, dtype=bool)
    forbidden[14, 30, 24] = True     # a background scribble sits in the gap
    out = bridge_components(mask, SPACING, closing_voxels=1, forbidden=forbidden)
    assert out[14, 30, 24] == 0


def test_bridge_refuses_when_it_would_add_too_much():
    shape = (60, 60, 48)
    mask = np.zeros(shape, dtype=np.uint8)
    mask[10:50, 10:50, 20] = 1
    mask[10:50, 10:50, 24] = 1       # two slabs 3 voxels apart: a huge bridge
    out, info = bridge_components(mask, SPACING, closing_voxels=2,
                                  max_added_ml=0.5, return_info=True)
    assert info["refused"] and np.array_equal(out, mask > 0)


def test_bridge_is_not_part_of_the_cleanup_stage():
    """Merging runs as its own stage after compliance, so cleanup_mask ignores the
    knob; a background scribble splitting a component again would otherwise undo it."""
    shape = (60, 60, 48)
    _, pet, _, _ = make_phantom(shape=shape, blobs=[], background_suv=0.5)
    mask = np.zeros(shape, dtype=np.uint8)
    mask[10:14, 30, 24] = 1
    mask[16:20, 30, 24] = 1
    pet[10:20, 30, 24] = 9.0
    a = cleanup_mask(mask, pet, SPACING, tracer="fdg", cfg=CleanupConfig())
    b = cleanup_mask(mask, pet, SPACING, tracer="fdg",
                     cfg=CleanupConfig(bridge_closing_voxels=1))
    assert CleanupConfig().bridge_closing_voxels == 0
    assert np.array_equal(a, b)


def test_v2_rules_is_a_union_of_conjunctions():
    """Cold and unconfident are different populations, so it takes two conjunctions."""
    shape = (60, 60, 48)
    _, pet, _, _ = make_phantom(shape=shape, blobs=[], background_suv=0.5)
    mask = np.zeros(shape, dtype=np.uint8)
    prob = np.zeros(shape, dtype=np.float32)

    mask[10:14, 10, 10] = 1                  # cold but confident
    pet[10:14, 10, 10] = 3.0
    prob[10:14, 10, 10] = 0.99
    mask[20:24, 20, 20] = 1                  # hot but unconfident
    pet[20:24, 20, 20] = 12.0
    prob[20:24, 20, 20] = 0.6
    mask[40:46, 40:46, 30:36] = 1            # hot and confident -> always kept
    pet[40:46, 40:46, 30:36] = 12.0
    prob[40:46, 40:46, 30:36] = 0.99

    def run(rules):
        out, info = remove_components_v2(mask, pet, SPACING, prob=prob, rules=rules,
                                         min_components_kept=0, return_info=True)
        return out, info["n_removed"]

    _, n = run([{"suv_gate": 5.0}])
    assert n == 1
    _, n = run([{"prob_gate": 0.8}])
    assert n == 1
    out, n = run([{"suv_gate": 5.0}, {"prob_gate": 0.8}])
    assert n == 2
    assert out[10:14, 10, 10].sum() == 0 and out[20:24, 20, 20].sum() == 0
    assert out[40:46, 40:46, 30:36].sum() == 6 * 6 * 6


def test_resolve_v2_rules_defaults_to_the_scalar_conjunction_and_validates():
    cfg = CleanupConfig(rule_v2=True, v2_min_volume_ml=0.3, v2_suv_gate=4.0)
    assert resolve_v2_rules(cfg, "fdg") == [
        {"min_volume_ml": 0.3, "suv_gate": 4.0, "prob_gate": None, "silence_decay": 1.0}]

    cfg = CleanupConfig(rule_v2=True, v2_rules=({"suv_gate": 5.0}, {"v2_prob_gate": 0.8}))
    got = resolve_v2_rules(cfg, "psma")
    assert [r["suv_gate"] for r in got] == [5.0, None]
    assert [r["prob_gate"] for r in got] == [None, 0.8]

    with pytest.raises(KeyError):
        resolve_v2_rules(CleanupConfig(v2_rules=({"nonsense": 1},)), "fdg")
    with pytest.raises(ValueError):
        resolve_v2_rules(CleanupConfig(v2_rules=({},)), "fdg")


def test_v2_rules_per_tracer_override():
    cfg = CleanupConfig(rule_v2=True, v2_rules=({"suv_gate": 5.0},),
                        v2_by_tracer={"psma": {"v2_rules": [{"suv_gate": 3.0},
                                                            {"prob_gate": 0.7}]}})
    assert [r["suv_gate"] for r in resolve_v2_rules(cfg, "fdg")] == [5.0]
    psma = resolve_v2_rules(cfg, "psma")
    assert [r["suv_gate"] for r in psma] == [3.0, None]
    assert [r["prob_gate"] for r in psma] == [None, 0.7]


def test_v2_union_still_protects_a_tumor_scribble_and_never_empties():
    shape = (60, 60, 48)
    _, pet, _, _ = make_phantom(shape=shape, blobs=[], background_suv=0.5)
    mask = np.zeros(shape, dtype=np.uint8)
    prob = np.zeros(shape, dtype=np.float32)
    mask[10:14, 10, 10] = 1
    pet[10:14, 10, 10] = 1.0
    prob[10:14, 10, 10] = 0.2
    rules = [{"suv_gate": 5.0}, {"prob_gate": 0.8}]
    kept = remove_components_v2(mask, pet, SPACING, prob=prob, rules=rules,
                                protect_points=[[11, 10, 10]])
    assert kept[10:14, 10, 10].sum() == 4
    kept = remove_components_v2(mask, pet, SPACING, prob=prob, rules=rules)
    assert kept.sum() == 4        # the "never empty" guard
