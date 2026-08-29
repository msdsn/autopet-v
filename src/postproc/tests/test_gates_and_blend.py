"""Negative gate, tracer heuristic, monotone blending."""

from __future__ import annotations

import numpy as np
import pytest

from postproc.config import NegativeGateConfig
from postproc.monotone import blend_masks, blend_with_previous
from postproc.negative_gate import is_probably_negative, negative_gate_features
from postproc.tracer_classifier import guess_tracer, superior_axis, tracer_features

from conftest import SPACING, make_phantom, sphere_mask


# ---------------------------------------------------------------------------
# negative gate
# ---------------------------------------------------------------------------
def test_gate_fires_on_a_tiny_cold_low_confidence_prediction():
    shape = (60, 60, 48)
    ct, pet, gt, _ = make_phantom(shape=shape, blobs=[], background_suv=0.5)
    mask = np.zeros(shape, dtype=np.uint8)
    mask[30:33, 30, 24] = 1
    pet[30:33, 30, 24] = 2.0
    prob = np.zeros(shape, dtype=np.float32)
    prob[30:33, 30, 24] = 0.51

    fired, feats = is_probably_negative(
        mask, pet, prob, NegativeGateConfig(), spacing=SPACING, return_features=True
    )
    assert fired, feats


# Every optional criterion switched on, so each veto can be exercised. The defaults
# leave max_suv / max_prob at None; see negative_gate.py.
ALL_CRITERIA = NegativeGateConfig(
    max_total_volume_ml=1.0, max_component_volume_ml=0.5, max_prob=0.60, max_suv=3.0,
)


@pytest.mark.parametrize(
    "suv,prob_val,extra_voxels,expected_reason",
    [
        (9.0, 0.51, 0, "suv"),         # hot -> real lesion
        (2.0, 0.95, 0, "prob"),        # confident -> real lesion
        (2.0, 0.51, 200, "volume"),    # large -> real lesion
    ],
)
def test_gate_does_not_fire_when_any_criterion_fails(suv, prob_val, extra_voxels, expected_reason):
    shape = (60, 60, 48)
    ct, pet, gt, _ = make_phantom(shape=shape, blobs=[], background_suv=0.5)
    mask = np.zeros(shape, dtype=np.uint8)
    mask[30:33, 30, 24] = 1
    pet[30:33, 30, 24] = suv
    if extra_voxels:
        mask[10:20, 10:20, 10:14] = 1
        pet[10:20, 10:20, 10:14] = suv
    prob = np.full(shape, 0.01, dtype=np.float32)
    prob[mask > 0] = prob_val

    fired, feats = is_probably_negative(
        mask, pet, prob, ALL_CRITERIA, spacing=SPACING, return_features=True
    )
    assert not fired
    assert expected_reason in feats["blocked_by"], feats


def test_none_threshold_disables_a_criterion():
    """None means "criterion off" -- the mechanism the defaults rely on."""
    shape = (60, 60, 48)
    ct, pet, gt, _ = make_phantom(shape=shape, blobs=[], background_suv=0.5)
    mask = np.zeros(shape, dtype=np.uint8)
    mask[30:33, 30, 24] = 1
    pet[30:33, 30, 24] = 40.0            # physiologically hot, as the real FPs are
    prob = np.full(shape, 0.01, dtype=np.float32)
    prob[mask > 0] = 1.0                 # argmax voxel, as every real predicted voxel is

    assert not is_probably_negative(mask, pet, prob, ALL_CRITERIA, spacing=SPACING)
    # ... and with the shipped defaults (max_suv / max_prob = None) it fires.
    fired, feats = is_probably_negative(mask, pet, prob, NegativeGateConfig(),
                                        spacing=SPACING, return_features=True)
    assert fired, feats
    assert feats["blocked_by"] == []


def test_gate_blocks_on_a_realistic_positive_volume():
    """A lesion-sized prediction must never be emptied, however confident or cold."""
    shape = (60, 60, 48)
    ct, pet, gt, _ = make_phantom(shape=shape, blobs=[], background_suv=0.5)
    mask = np.zeros(shape, dtype=np.uint8)
    mask[20:40, 20:40, 14:34] = 1        # ~ 8000 voxels
    pet[mask > 0] = 6.0
    fired, feats = is_probably_negative(mask, pet, None, NegativeGateConfig(),
                                        spacing=SPACING, return_features=True)
    assert not fired and "volume" in feats["blocked_by"], feats


def test_component_stats_ranks_and_measures():
    from postproc.negative_gate import component_stats

    shape = (60, 60, 48)
    ct, pet, gt, _ = make_phantom(shape=shape, blobs=[], background_suv=0.5)
    mask = np.zeros(shape, dtype=np.uint8)
    mask[10:14, 10:14, 10:14] = 1        # big
    mask[40, 40, 40] = 1                 # small
    pet[10:14, 10:14, 10:14] = 3.0
    pet[40, 40, 40] = 20.0
    pet[41, 40, 40] = 30.0               # hot neighbour of the small component
    stats = component_stats(mask, pet, None, SPACING, ct=ct)
    assert len(stats) == 2
    assert stats[0]["volume_ml"] > stats[1]["volume_ml"]      # largest first
    assert stats[0]["n_voxels"] == 64 and stats[1]["n_voxels"] == 1
    assert stats[1]["suv_max"] == 20.0
    assert stats[1]["shell_suv_max"] >= 30.0                 # sees the hot neighbour
    assert 0.0 <= stats[0]["z_frac"] <= 1.0


def test_gate_never_fires_once_any_scribble_has_arrived():
    """A scribble means the evaluator found an error, so the case is not lesion-free."""
    shape = (40, 40, 32)
    ct, pet, gt, _ = make_phantom(shape=shape, blobs=[], background_suv=0.5)
    mask = np.zeros(shape, dtype=np.uint8)
    fired, feats = is_probably_negative(
        mask, pet, None, NegativeGateConfig(), spacing=SPACING,
        n_tumor_scribbles=3, return_features=True,
    )
    assert not fired
    assert feats["blocked_by"] == ["tumor_scribble_proves_positive"], feats

    fired, feats = is_probably_negative(
        mask, pet, None, NegativeGateConfig(), spacing=SPACING,
        n_background_scribbles=3, return_features=True,
    )
    assert not fired and "scribbles_present" in feats["blocked_by"]


def test_gate_keeps_firing_at_every_iteration_of_a_lesion_free_case():
    """A lesion-free case gets no scribble and Dice is recomputed at each of the 6
    iterations, so the gate has to fire every time."""
    shape = (40, 40, 32)
    ct, pet, gt, _ = make_phantom(shape=shape, blobs=[], background_suv=0.5)
    mask = np.zeros(shape, dtype=np.uint8)
    for it in range(6):
        assert is_probably_negative(
            mask, pet, None, NegativeGateConfig(), spacing=SPACING, iteration=it
        ), f"gate stopped firing at iteration {it}"


def test_only_iteration_zero_can_be_re_enabled():
    shape = (40, 40, 32)
    ct, pet, gt, _ = make_phantom(shape=shape, blobs=[], background_suv=0.5)
    mask = np.zeros(shape, dtype=np.uint8)
    p = NegativeGateConfig(only_iteration_zero=True)
    assert is_probably_negative(mask, pet, None, p, spacing=SPACING, iteration=0)
    assert not is_probably_negative(mask, pet, None, p, spacing=SPACING, iteration=2)


def test_gate_features_on_an_empty_mask():
    shape = (30, 30, 24)
    ct, pet, gt, _ = make_phantom(shape=shape, blobs=[], background_suv=0.5)
    f = negative_gate_features(np.zeros(shape, np.uint8), pet, None, SPACING)
    assert f["total_volume_ml"] == 0.0 and f["n_components"] == 0


# ---------------------------------------------------------------------------
# tracer heuristic
# ---------------------------------------------------------------------------
def _body_phantom(shape=(60, 60, 120), brain_suv=0.3, kidney_suv=8.0):
    """A crude whole-body PET: a "brain" ball at the top (high k) and two "kidneys"."""
    ct, pet, gt, _ = make_phantom(shape=shape, blobs=[], background_suv=0.8, seed=-1)
    brain = sphere_mask(shape, (30, 30, shape[2] - 10), 22.0, SPACING)
    pet[brain] = brain_suv
    for cx in (22, 38):
        k = sphere_mask(shape, (cx, 30, int(shape[2] * 0.45)), 10.0, SPACING)
        pet[k] = kidney_suv
    return pet


def test_tracer_heuristic_fdg_vs_psma():
    fdg = _body_phantom(brain_suv=10.0, kidney_suv=6.0)
    psma = _body_phantom(brain_suv=0.4, kidney_suv=60.0)
    assert guess_tracer(fdg, SPACING) == "fdg"
    assert guess_tracer(psma, SPACING) == "psma"


def test_tracer_heuristic_respects_orientation():
    """Flipping the volume must flip the answer only if we do not tell it the affine."""
    fdg = _body_phantom(brain_suv=10.0, kidney_suv=6.0)
    flipped = fdg[:, :, ::-1].copy()
    assert guess_tracer(flipped, SPACING, superior=(2, -1)) == "fdg"


def test_superior_axis_from_affine():
    aff = np.diag([-2.0, -2.0, 3.0, 1.0])  # RAS: +k -> superior
    assert superior_axis(aff) == (2, 1)
    aff2 = np.diag([-2.0, -2.0, -3.0, 1.0])
    assert superior_axis(aff2) == (2, -1)
    assert superior_axis(None) == (2, 1)


def test_tracer_features_report_a_score_and_a_confidence():
    """The heuristic reports its decision score, its distance from the boundary and a
    confidence a caller can fall back on."""
    pet = _body_phantom()
    _, f = guess_tracer(pet, SPACING, return_features=True)
    assert 0.0 < f["confidence"] <= 1.0
    assert "score" in f and "margin_decades" in f
    assert f["head_over_trunk"] > 0.0


def test_tracer_confidence_is_higher_further_from_the_boundary():
    clear_fdg = _body_phantom(brain_suv=12.0, kidney_suv=4.0)
    clear_psma = _body_phantom(brain_suv=0.3, kidney_suv=90.0)
    _, a = guess_tracer(clear_fdg, SPACING, return_features=True)
    _, b = guess_tracer(clear_psma, SPACING, return_features=True)
    assert a["confidence"] > 0.8 and b["confidence"] > 0.8
    assert a["score"] > b["score"]


def test_tracer_uses_the_body_extent_not_the_array_extent():
    """Empty slices above the vertex must not become "the head slab"."""
    pet = _body_phantom(brain_suv=10.0, kidney_suv=6.0)
    pad = np.zeros((pet.shape[0], pet.shape[1], 40), dtype=pet.dtype)
    padded = np.concatenate([pet, pad], axis=2)      # 40 empty slices above the head
    ct = np.full(padded.shape, -1000.0, dtype=np.float32)
    ct[..., : pet.shape[2]] = 0.0                    # the CT knows where the patient is
    assert guess_tracer(padded, SPACING, ct=ct) == "fdg"
    assert guess_tracer(padded, SPACING) == "fdg"    # PET-only fallback finds it too


# ---------------------------------------------------------------------------
# monotone blending
# ---------------------------------------------------------------------------
def test_blend_passthrough_at_iteration_zero():
    new = np.full((4, 4, 4), 0.7, dtype=np.float32)
    assert np.array_equal(blend_with_previous(new, None), new)
    assert np.array_equal(blend_with_previous(new, new, mode="none"), new)


def test_blend_fg_region_takes_the_max_bg_region_the_min():
    shape = (4, 4, 4)
    new = np.full(shape, 0.2, dtype=np.float32)
    prev = np.full(shape, 0.9, dtype=np.float32)
    fg = np.zeros(shape, dtype=bool)
    fg[0, 0, 0] = True
    bg = np.zeros(shape, dtype=bool)
    bg[1, 1, 1] = True

    out = blend_with_previous(new, prev, fg, bg, mode="minmax")
    assert out[0, 0, 0] == pytest.approx(0.9)   # fg constraint: max
    assert out[1, 1, 1] == pytest.approx(0.2)   # bg constraint: min
    assert out[2, 2, 2] == pytest.approx(0.2)   # elsewhere: the new value


def test_ema_damps_oscillation():
    shape = (4, 4, 4)
    new = np.full(shape, 0.0, dtype=np.float32)
    prev = np.full(shape, 1.0, dtype=np.float32)
    out = blend_with_previous(new, prev, None, None, mode="ema", ema_alpha=0.6)
    assert out[0, 0, 0] == pytest.approx(0.4)


def test_blend_accepts_channel_first_softmax():
    prob = np.zeros((2, 4, 4, 4), dtype=np.float32)
    prob[1] = 0.8
    out = blend_with_previous(prob, None)
    assert out.shape == (4, 4, 4) and out[0, 0, 0] == pytest.approx(0.8)


def test_blend_masks_fallback():
    shape = (4, 4, 4)
    new = np.zeros(shape, dtype=np.uint8)
    prev = np.zeros(shape, dtype=np.uint8)
    prev[0, 0, 0] = 1
    new[1, 1, 1] = 1
    fg = np.zeros(shape, dtype=bool)
    fg[0, 0, 0] = True
    bg = np.zeros(shape, dtype=bool)
    bg[1, 1, 1] = True

    out = blend_masks(new, prev, fg, bg)
    assert out[0, 0, 0] == 1, "foreground constraint should keep the previous voxel"
    assert out[1, 1, 1] == 0, "background constraint should suppress the new voxel"


def test_blend_shape_mismatch_falls_back_to_new():
    new = np.full((4, 4, 4), 0.3, dtype=np.float32)
    prev = np.full((5, 5, 5), 0.9, dtype=np.float32)
    assert np.array_equal(blend_with_previous(new, prev), new)


def test_constraints_are_revocable():
    """Where the two constraint regions overlap neither clamp applies, so a later
    scribble of the opposite class can undo an earlier one."""
    from postproc.monotone import revoke_overlap

    shape = (4, 4, 4)
    fg = np.zeros(shape, dtype=bool)
    fg[0:2, 0, 0] = True
    bg = np.zeros(shape, dtype=bool)
    bg[1:3, 0, 0] = True
    f2, b2 = revoke_overlap(fg, bg)
    assert f2[0, 0, 0] and not f2[1, 0, 0]
    assert b2[2, 0, 0] and not b2[1, 0, 0]

    new = np.full(shape, 0.2, dtype=np.float32)
    prev = np.full(shape, 0.9, dtype=np.float32)
    out = blend_with_previous(new, prev, fg, bg, mode="minmax")
    assert out[0, 0, 0] == pytest.approx(0.9)   # fg only
    assert out[2, 0, 0] == pytest.approx(0.2)   # bg only
    assert out[1, 0, 0] == pytest.approx(0.2)   # contested -> fresh evidence decides
