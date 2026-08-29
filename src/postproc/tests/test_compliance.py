"""Guarantees G1 / G2, splitting, non-merging, idempotency, and the two fg methods."""

from __future__ import annotations

import time

import numpy as np
import pytest

from postproc.compliance import (
    apply_all_constraints,
    apply_background_scribbles,
    apply_tumor_scribbles,
    check_constraints,
)
from postproc.config import ComplianceConfig

from conftest import SPACING, VOX_ML, Blob, make_phantom, scribble_line, sphere_mask


# ---------------------------------------------------------------------------
# G1 -- background
# ---------------------------------------------------------------------------
def test_background_scribble_removes_its_component(phantom):
    pet, gt, blobs = phantom["pet"], phantom["gt"], phantom["blobs"]
    mask = gt.copy()
    bg = scribble_line((56, 30, 34), axis=0, length=5, mask=blobs["warm"])
    assert len(bg) >= 3

    out = apply_background_scribbles(mask, pet, bg, spacing=SPACING)

    assert check_constraints(out, [], bg)["ok"]
    assert not (out.astype(bool) & blobs["warm"]).any(), "the pointed-at component survived"
    for name in ("hot", "cool"):
        kept = (out.astype(bool) & blobs[name]).sum()
        assert kept == blobs[name].sum(), f"unrelated component {name!r} was damaged"


def test_background_scribble_outside_any_component_is_a_noop(phantom):
    pet, gt = phantom["pet"], phantom["gt"]
    bg = [[5, 5, 5], [5, 6, 5]]
    out = apply_background_scribbles(gt.copy(), pet, bg, spacing=SPACING)
    assert np.array_equal(out.astype(bool), gt.astype(bool))
    assert check_constraints(out, [], bg)["ok"]


def test_background_scribble_is_idempotent(phantom):
    pet, gt, blobs = phantom["pet"], phantom["gt"], phantom["blobs"]
    bg = scribble_line((56, 30, 34), axis=1, length=5, mask=blobs["warm"])
    once = apply_background_scribbles(gt.copy(), pet, bg, spacing=SPACING)
    twice = apply_background_scribbles(once, pet, bg, spacing=SPACING)
    assert np.array_equal(once, twice)


def test_background_scribble_splits_a_bridged_component():
    """Two hot foci joined by a thin bridge: a background scribble on one deletes only
    that focus while the other carries a tumor scribble."""
    shape = (80, 60, 48)
    blobs = [
        Blob(centre=(28, 30, 24), radius_mm=9.0, suv=10.0, name="lesion"),
        Blob(centre=(52, 30, 24), radius_mm=9.0, suv=9.0, name="organ"),
    ]
    ct, pet, gt, bm = make_phantom(shape=shape, blobs=blobs)
    bridge = np.zeros(shape, dtype=bool)
    bridge[31:49, 29:32, 23:26] = True  # spans the gap between the two spheres
    pet[bridge] = 3.0
    mask = (gt.astype(bool) | bridge).astype(np.uint8)

    fg = scribble_line((28, 30, 24), axis=2, length=5, mask=bm["lesion"])
    bg = scribble_line((52, 30, 24), axis=2, length=5, mask=bm["organ"])

    cfg = ComplianceConfig()
    out, info = apply_background_scribbles(
        mask, pet, bg, spacing=SPACING, fg_points=fg, cfg=cfg, return_info=True
    )

    assert info["n_components_split"] == 1, info
    assert check_constraints(out, fg, bg)["ok"]
    kept = out.astype(bool)
    assert kept[tuple(np.array(fg[0]))], "the tumor-scribbled focus was deleted"
    assert (kept & bm["lesion"]).sum() > 0.5 * bm["lesion"].sum()
    assert (kept & bm["organ"]).sum() < 0.2 * bm["organ"].sum()


def test_background_split_rejected_falls_back_to_full_deletion(phantom):
    """A homogeneous component has no core hotter than the scribble -> delete it whole."""
    pet, gt, blobs = phantom["pet"], phantom["gt"], phantom["blobs"]
    bg = scribble_line((24, 24, 20), axis=0, length=5, mask=blobs["hot"])
    out, info = apply_background_scribbles(gt.copy(), pet, bg, spacing=SPACING, return_info=True)
    assert info["n_components_deleted"] == 1
    assert not (out.astype(bool) & blobs["hot"]).any()


# ---------------------------------------------------------------------------
# G2 -- tumor
# ---------------------------------------------------------------------------
def test_tumor_scribble_grows_the_missed_lesion(phantom):
    ct, pet, gt, blobs = phantom["ct"], phantom["pet"], phantom["gt"], phantom["blobs"]
    mask = gt.copy()
    mask[blobs["warm"]] = 0  # the model missed it
    fg = scribble_line((56, 30, 34), axis=0, length=5, mask=blobs["warm"])

    out, info = apply_tumor_scribbles(
        mask, pet, ct, fg, tracer="fdg", spacing=SPACING, return_info=True
    )

    assert check_constraints(out, fg, [])["ok"]
    recovered = (out.astype(bool) & blobs["warm"]).sum() / blobs["warm"].sum()
    assert recovered > 0.6, f"only recovered {recovered:.2%} of the lesion"
    # and it must not have flooded the background
    outside = out.astype(bool) & ~gt.astype(bool)
    assert outside.sum() * VOX_ML < 1.0, "growth leaked into the background"


def test_tumor_scribble_on_the_lesion_boundary_does_not_oversegment(phantom):
    """The `boundary` scribble strategy puts the seed on the lesion rim."""
    from scipy import ndimage

    ct, pet, gt, blobs = phantom["ct"], phantom["pet"], phantom["gt"], phantom["blobs"]
    warm = blobs["warm"]
    rim = warm & ~ndimage.binary_erosion(warm)
    rim_pts = np.argwhere(rim)
    fg = [list(map(int, p)) for p in rim_pts[:: max(1, len(rim_pts) // 6)][:6]]

    mask = gt.copy()
    mask[warm] = 0
    out = apply_tumor_scribbles(mask, pet, ct, fg, tracer="fdg", spacing=SPACING)

    assert check_constraints(out, fg, [])["ok"]
    grown = out.astype(bool) & ~gt.astype(bool)
    # a fixed 30 mm ball would add ~113 mL; intensity-constrained growth must not
    assert grown.sum() * VOX_ML < 2.0, f"over-segmented by {grown.sum() * VOX_ML:.2f} mL"


def test_tumor_scribble_does_not_merge_two_lesions():
    """Growth must stop at a neighbouring component: merging costs DMM."""
    import cc3d

    shape = (70, 60, 48)
    blobs = [
        Blob(centre=(26, 30, 24), radius_mm=7.0, suv=8.0, name="a"),
        Blob(centre=(40, 30, 24), radius_mm=7.0, suv=8.0, name="b"),
    ]
    ct, pet, gt, bm = make_phantom(shape=shape, blobs=blobs, background_suv=2.0)
    # a warm corridor between them: without the anti-merge rule the grow bridges it
    pet[30:38, 27:34, 21:28] = 5.0

    mask = np.zeros(shape, dtype=np.uint8)
    mask[bm["b"]] = 1  # only lesion b is predicted
    fg = scribble_line((26, 30, 24), axis=2, length=5, mask=bm["a"])

    out = apply_tumor_scribbles(mask, pet, ct, fg, tracer="fdg", spacing=SPACING)

    assert check_constraints(out, fg, [])["ok"]
    n = cc3d.connected_components(np.ascontiguousarray(out), connectivity=18).max()
    assert n >= 2, "the two lesions were merged into one component"


def test_tumor_scribble_cold_lesion_falls_back_to_a_ball():
    """A photopenic scribble (SUV below the tracer floor) still yields a small blob."""
    shape = (60, 60, 40)
    ct, pet, gt, _ = make_phantom(shape=shape, blobs=[], background_suv=0.4)
    fg = scribble_line((30, 30, 20), axis=0, length=5)
    mask = np.zeros(shape, dtype=np.uint8)

    out = apply_tumor_scribbles(mask, pet, ct, fg, tracer="fdg", spacing=SPACING)

    assert check_constraints(out, fg, [])["ok"]
    vol = out.sum() * VOX_ML
    assert vol > len(fg) * VOX_ML, "no ball fallback was applied"
    assert vol < 1.5, f"the fallback ball is far too big ({vol:.2f} mL)"


def test_tumor_scribble_is_idempotent(phantom):
    ct, pet, gt, blobs = phantom["ct"], phantom["pet"], phantom["gt"], phantom["blobs"]
    mask = gt.copy()
    mask[blobs["warm"]] = 0
    fg = scribble_line((56, 30, 34), axis=0, length=5, mask=blobs["warm"])
    once = apply_tumor_scribbles(mask, pet, ct, fg, tracer="fdg", spacing=SPACING)
    twice = apply_tumor_scribbles(once, pet, ct, fg, tracer="fdg", spacing=SPACING)
    assert np.array_equal(once, twice)


def test_tumor_growth_respects_the_probability_map(phantom):
    ct, pet, gt, blobs = phantom["ct"], phantom["pet"], phantom["gt"], phantom["blobs"]
    mask = gt.copy()
    mask[blobs["warm"]] = 0
    fg = scribble_line((56, 30, 34), axis=0, length=5, mask=blobs["warm"])
    prob = np.zeros(pet.shape, dtype=np.float32)
    prob[blobs["warm"]] = 0.9
    out = apply_tumor_scribbles(
        mask, pet, ct, fg, prob=prob, tracer="fdg", spacing=SPACING
    )
    assert (out.astype(bool) & blobs["warm"]).sum() / blobs["warm"].sum() > 0.9


# ---------------------------------------------------------------------------
# both together
# ---------------------------------------------------------------------------
def test_fg_and_bg_constraints_coexist(phantom):
    ct, pet, gt, blobs = phantom["ct"], phantom["pet"], phantom["gt"], phantom["blobs"]
    mask = gt.copy()
    mask[blobs["warm"]] = 0
    fg = scribble_line((56, 30, 34), axis=0, length=5, mask=blobs["warm"])
    bg = scribble_line((24, 24, 20), axis=1, length=5, mask=blobs["hot"])

    out = apply_all_constraints(mask, pet, ct, fg, bg, tracer="fdg", spacing=SPACING)
    assert check_constraints(out, fg, bg)["ok"]
    assert not (out.astype(bool) & blobs["hot"]).any()
    assert (out.astype(bool) & blobs["warm"]).any()

    twice = apply_all_constraints(out, pet, ct, fg, bg, tracer="fdg", spacing=SPACING)
    assert np.array_equal(out, twice), "apply_all_constraints is not idempotent"


def test_conflicting_scribbles_tumor_wins(phantom):
    ct, pet, gt = phantom["ct"], phantom["pet"], phantom["gt"]
    pt = [[24, 24, 20]]
    out = apply_all_constraints(gt.copy(), pet, ct, pt, pt, tracer="fdg", spacing=SPACING)
    assert out[24, 24, 20] == 1


# ---------------------------------------------------------------------------
# random walker variant
# ---------------------------------------------------------------------------
def test_random_walker_variant_satisfies_g2_and_is_reported(phantom):
    ct, pet, gt, blobs = phantom["ct"], phantom["pet"], phantom["gt"], phantom["blobs"]
    mask = gt.copy()
    mask[blobs["warm"]] = 0
    fg = scribble_line((56, 30, 34), axis=0, length=5, mask=blobs["warm"])

    t0 = time.perf_counter()
    grow = apply_tumor_scribbles(mask, pet, ct, fg, tracer="fdg", spacing=SPACING, method="grow")
    t_grow = time.perf_counter() - t0

    t0 = time.perf_counter()
    rw = apply_tumor_scribbles(
        mask, pet, ct, fg, tracer="fdg", spacing=SPACING, method="random_walker"
    )
    t_rw = time.perf_counter() - t0

    print(f"\n[timing] fg growth: grow={t_grow * 1000:.0f} ms  random_walker={t_rw * 1000:.0f} ms")
    assert check_constraints(rw, fg, [])["ok"]
    assert check_constraints(grow, fg, [])["ok"]
    assert (rw.astype(bool) & blobs["warm"]).sum() > 0


# ---------------------------------------------------------------------------
# G3 -- safe no-op on an already-satisfied scribble
# ---------------------------------------------------------------------------
# Category-2 scribbles were collected against the baseline model's errors and replayed
# unchanged to every algorithm, so many arrive already satisfied by our prediction.
def test_tumor_scribble_already_inside_changes_nothing(phantom):
    ct, pet, gt, blobs = phantom["ct"], phantom["pet"], phantom["gt"], phantom["blobs"]
    fg = scribble_line((56, 30, 34), axis=0, length=5, mask=blobs["warm"])
    assert gt[tuple(np.asarray(fg).T)].all()  # the scribble is inside the prediction

    out, info = apply_tumor_scribbles(
        gt, pet, ct, fg, tracer="fdg", spacing=SPACING, return_info=True
    )
    assert np.array_equal(out.astype(bool), gt.astype(bool)), "an satisfied scribble grew the mask"
    assert info["n_clusters_satisfied"] == 1 and info["n_clusters_grown"] == 0
    assert info["added_voxels"] == 0


def test_background_scribble_on_unpredicted_voxels_changes_nothing(phantom):
    ct, pet, gt, blobs = phantom["ct"], phantom["pet"], phantom["gt"], phantom["blobs"]
    mask = gt.copy()
    mask[blobs["warm"]] = 0                      # we do not predict this lesion ...
    bg = scribble_line((56, 30, 34), axis=0, length=5, mask=blobs["warm"])  # ... it is scribbled

    out, info = apply_background_scribbles(
        mask, pet, bg, spacing=SPACING, return_info=True
    )
    assert np.array_equal(out.astype(bool), mask.astype(bool))
    assert info["n_components_hit"] == 0 and info["removed_voxels"] == 0


def test_partially_covered_tumor_scribble_grows_only_the_missing_part(phantom):
    """Half the scribble is already covered: add the missing part, do not re-grow the
    part the mask already had."""
    ct, pet, gt, blobs = phantom["ct"], phantom["pet"], phantom["gt"], phantom["blobs"]
    warm = blobs["warm"]
    mask = gt.copy()
    # erase the half of the lesion with the larger i index
    cut = np.zeros_like(warm)
    cut[57:, :, :] = True
    mask[warm & cut] = 0

    fg = scribble_line((56, 30, 34), axis=0, length=9, mask=warm)
    inside_before = mask[tuple(np.asarray(fg).T)].astype(bool)
    assert inside_before.any() and not inside_before.all(), "fixture must be partial"

    out, info = apply_tumor_scribbles(
        mask, pet, ct, fg, tracer="fdg", spacing=SPACING, return_info=True
    )
    assert check_constraints(out, fg, [])["ok"]
    assert info["n_clusters_grown"] == 1
    # nothing outside the lesion was added
    added = out.astype(bool) & ~mask.astype(bool)
    assert (added & ~warm).sum() * VOX_ML < 0.2, "growth spilled outside the lesion"
    assert (added & warm).sum() > 0, "the missing half was not recovered"


# ---------------------------------------------------------------------------
# split-not-delete: the core rule
# ---------------------------------------------------------------------------
def test_background_scribble_splits_off_only_the_bled_part():
    """A true positive that bled into cool tissue: keep the core, delete the bleed."""
    shape = (72, 60, 48)
    ct, pet, gt, bm = make_phantom(
        shape=shape, blobs=[Blob(centre=(28, 30, 24), radius_mm=8.0, suv=11.0, name="lesion")]
    )
    bleed = np.zeros(shape, dtype=bool)
    bleed[31:52, 27:34, 21:28] = True   # touches the lesion
    pet[bleed] = 2.4                       # cool over-segmented tail
    mask = (gt.astype(bool) | bleed).astype(np.uint8)
    prob = np.where(gt > 0, 0.97, 0.0).astype(np.float32)
    prob[bleed] = 0.55                     # the model is not confident about the tail

    bg = [[int(i), 30, 24] for i in range(44, 50)]
    out, info = apply_background_scribbles(
        mask, pet, bg, prob=prob, spacing=SPACING, tracer="fdg", return_info=True
    )

    assert info["n_components_split"] == 1 and info["n_components_deleted"] == 0, info
    assert check_constraints(out, [], bg)["ok"]
    kept = out.astype(bool)
    # The lesion must be untouched, and the part of the bleed near the scribble gone.
    # The removal is deliberately local, so one scribble does not have to clear a 43 mm
    # tail in one go -- the evaluator simply scribbles again, and it converges.
    assert (kept & bm["lesion"]).sum() == bm["lesion"].sum(), "the lesion was damaged"
    removed_frac = 1 - (kept & bleed).sum() / bleed.sum()
    assert removed_frac > 0.25, f"only {removed_frac:.0%} of the bleed was removed"

    bg2 = bg + [[int(i), 30, 24] for i in range(32, 38)]
    out2 = apply_background_scribbles(
        mask, pet, bg2, prob=prob, spacing=SPACING, tracer="fdg"
    )
    k2 = out2.astype(bool)
    assert 1 - (k2 & bleed).sum() / bleed.sum() > removed_frac, "a second scribble did not converge"
    assert (k2 & bm["lesion"]).sum() > 0.9 * bm["lesion"].sum()


def test_homogeneous_false_positive_is_deleted_whole():
    """No part of the component is markedly hotter than the scribble -> no core."""
    shape = (60, 60, 48)
    ct, pet, gt, bm = make_phantom(
        shape=shape, blobs=[Blob(centre=(30, 30, 24), radius_mm=8.0, suv=5.0, name="fp")]
    )
    prob = np.where(gt > 0, 0.55, 0.0).astype(np.float32)   # uniformly unconfident
    bg = scribble_line((30, 30, 24), axis=0, length=5, mask=bm["fp"])

    out, info = apply_background_scribbles(
        gt.copy(), pet, bg, prob=prob, spacing=SPACING, tracer="fdg", return_info=True
    )
    assert info["n_components_deleted"] == 1 and info["n_components_split"] == 0
    assert not (out.astype(bool) & bm["fp"]).any()


def test_component_with_a_tumor_scribble_is_never_deleted_whole(phantom):
    """G2 outranks a background scribble: the tumor point is always part of the core."""
    pet, gt, blobs = phantom["pet"], phantom["gt"], phantom["blobs"]
    hot = blobs["hot"]
    fg = [[24, 24, 20]]
    bg = [[int(i), 24, 20] for i in np.argwhere(hot)[:, 0][-3:]]
    out, info = apply_background_scribbles(
        gt.copy(), pet, bg, fg_points=fg, spacing=SPACING, return_info=True
    )
    assert info["n_components_with_tumor_scribble"] == 1
    assert info["n_components_deleted"] == 0, "a scribbled component was deleted whole"
    assert out[24, 24, 20] == 1


def test_audit_reports_ground_truth_damage(phantom):
    pet, gt, blobs = phantom["pet"], phantom["gt"], phantom["blobs"]
    bg = scribble_line((24, 24, 20), axis=0, length=5, mask=blobs["hot"])
    _, info = apply_background_scribbles(
        gt.copy(), pet, bg, spacing=SPACING, gt=gt, return_info=True
    )
    audit = info["gt_audit"]
    assert audit["n_gt_lesions"] == 3
    assert audit["n_gt_lesions_over_half_removed"] == 1     # we deleted a real lesion
    assert audit["worst_fraction_removed"] > 0.9


# ---------------------------------------------------------------------------
# threshold calibration from the scribble footprint
# ---------------------------------------------------------------------------
def test_thick_footprint_calibrates_the_threshold():
    """A centerline scribble on a compact error is the whole 2-D cross-section, so it
    picks the threshold instead of the 41 % rule."""
    shape = (64, 64, 48)
    ct, pet, gt, bm = make_phantom(
        shape=shape,
        blobs=[Blob(centre=(32, 32, 24), radius_mm=10.0, suv=6.0, name="lesion")],
        background_suv=1.6,
        seed=-1,
    )
    # a halo the 41 %-of-peak rule would swallow (0.41 * 6.0 = 2.46 > 2.2)
    halo = sphere_mask(shape, (32, 32, 24), 16.0, SPACING) & ~bm["lesion"]
    pet[halo] = 2.9

    footprint = bm["lesion"][:, :, 24]
    fg = [[int(i), int(j), 24] for i, j in np.argwhere(footprint)]
    mask = np.zeros(shape, dtype=np.uint8)

    out, info = apply_tumor_scribbles(
        mask, pet, ct, fg, tracer="fdg", spacing=SPACING, return_info=True
    )
    detail = info["clusters"][0]
    assert detail["footprint_thickness_mm"] > 2.5
    assert detail["calibrated"] is True, detail
    assert detail["calibration_iou"] > 0.9, detail
    assert check_constraints(out, fg, [])["ok"]
    grown = out.astype(bool)
    assert (grown & halo).sum() < 0.2 * halo.sum(), "calibration failed to reject the halo"


def test_thin_footprint_falls_back_to_the_alpha_rule(phantom):
    ct, pet, gt, blobs = phantom["ct"], phantom["pet"], phantom["gt"], phantom["blobs"]
    mask = gt.copy()
    mask[blobs["warm"]] = 0
    fg = scribble_line((56, 30, 34), axis=0, length=5, mask=blobs["warm"])
    _, info = apply_tumor_scribbles(
        mask, pet, ct, fg, tracer="fdg", spacing=SPACING, return_info=True
    )
    detail = info["clusters"][0]
    assert detail["footprint_thickness_mm"] < 2.5
    assert detail["calibrated"] is False
    assert detail["threshold"] == pytest.approx(detail["alpha_threshold"], rel=1e-6)


def test_growth_stops_at_a_valley_between_two_foci():
    """Two hot foci joined by a cool isthmus above the threshold: do not swallow both."""
    shape = (80, 60, 48)
    ct, pet, gt, bm = make_phantom(
        shape=shape,
        blobs=[
            Blob(centre=(24, 30, 24), radius_mm=7.0, suv=9.0, name="a"),
            Blob(centre=(52, 30, 24), radius_mm=7.0, suv=9.0, name="b"),
        ],
        background_suv=0.6,
        seed=-1,
    )
    isthmus = np.zeros(shape, dtype=bool)
    isthmus[28:50, 27:34, 21:28] = True
    isthmus &= ~bm["a"] & ~bm["b"]
    pet[isthmus] = 4.2                       # above 0.41 * 9.0 = 3.7

    fg = scribble_line((24, 30, 24), axis=2, length=3, mask=bm["a"])
    out = apply_tumor_scribbles(
        np.zeros(shape, np.uint8), pet, ct, fg, tracer="fdg", spacing=SPACING
    )
    grown = out.astype(bool)
    assert check_constraints(out, fg, [])["ok"]
    assert (grown & bm["a"]).sum() > 0.5 * bm["a"].sum()
    assert (grown & bm["b"]).sum() < 0.25 * bm["b"].sum(), "growth crossed the valley"


def test_geodesic_bound_is_tighter_than_euclidean():
    """A lesion 30 mm away in a straight line but far around an obstacle must not join."""
    shape = (64, 64, 40)
    ct, pet, gt, _ = make_phantom(shape=shape, blobs=[], background_suv=0.5, seed=-1)
    # a U-shaped hot channel: the two ends are close in euclidean terms, far geodesically
    pet[20:44, 18:26, 16:24] = 12.0
    pet[20:28, 20:44, 16:24] = 12.0
    pet[38:44, 20:44, 16:24] = 12.0
    fg = [[21, 42, 20]]

    from postproc.config import ComplianceConfig

    # generous volume caps: this test is about the distance bound, nothing else
    caps = dict(fg_max_volume_ml=500.0, fg_min_growth_ml=500.0, fg_stop_at_valleys=False)
    tight = ComplianceConfig(fg_max_radius_mm=30.0, fg_geodesic=True, **caps)
    loose = ComplianceConfig(fg_max_radius_mm=30.0, fg_geodesic=False, **caps)
    geo = apply_tumor_scribbles(
        np.zeros(shape, np.uint8), pet, ct, fg, tracer="fdg", spacing=SPACING, cfg=tight
    )
    euc = apply_tumor_scribbles(
        np.zeros(shape, np.uint8), pet, ct, fg, tracer="fdg", spacing=SPACING, cfg=loose
    )
    assert geo.sum() < euc.sum(), "the geodesic bound was not tighter"
    assert geo[41, 42, 20] == 0, "the far arm was reached despite the geodesic bound"


# ---------------------------------------------------------------------------
# a scribble is a local correction, never a demolition
# ---------------------------------------------------------------------------
# A well-segmented lesion has a high and roughly flat probability over the whole
# component, and `watershed(-prob, ...)` on a flat landscape degenerates into a
# distance-based partition: a background scribble on the rim would take every voxel
# closer to it than to the core, i.e. most of the lesion.
def test_rim_scribble_on_a_confident_lesion_removes_almost_nothing():
    shape = (64, 64, 48)
    ct, pet, gt, bm = make_phantom(
        shape=shape,
        blobs=[Blob(centre=(32, 32, 24), radius_mm=12.0, suv=8.0, name="lesion")],
        background_suv=0.6,
        seed=-1,
    )
    from scipy import ndimage

    lesion = bm["lesion"]
    rim = ndimage.binary_dilation(lesion) & ~lesion
    pet[rim] = 3.0
    mask = (lesion | rim).astype(np.uint8)
    prob = np.where(mask > 0, 0.95, 0.0).astype(np.float32)   # flat and confident

    rim_pts = np.argwhere(rim)
    bg = [[int(a), int(b), int(c)] for a, b, c in rim_pts[:: max(1, len(rim_pts) // 8)][:8]]

    out, info = apply_background_scribbles(
        mask, pet, bg, prob=prob, spacing=SPACING, tracer="fdg", return_info=True
    )

    assert check_constraints(out, [], bg)["ok"]
    kept = (out.astype(bool) & lesion).sum() / lesion.sum()
    assert kept > 0.98, f"the split ate the lesion: only {kept:.1%} of it survived"
    removed_ml = info["removed_voxels"] * VOX_ML
    assert removed_ml < 0.5, f"removed {removed_ml:.2f} mL for an 8-voxel scribble"


def test_split_removal_is_bounded_to_the_scribble_neighbourhood():
    """Even with a confidence gradient, the deletion stays near the scribble."""
    shape = (110, 48, 40)
    ct, pet, gt, _ = make_phantom(shape=shape, blobs=[], background_suv=0.6, seed=-1)
    # a long confident bar; only the far tip is unconfident and scribbled
    bar = np.zeros(shape, dtype=bool)
    bar[10:100, 20:28, 16:24] = True
    pet[bar] = 6.0
    prob = np.where(bar, 0.95, 0.0).astype(np.float32)
    tip = np.zeros(shape, dtype=bool)
    tip[88:100, 20:28, 16:24] = True
    prob[tip] = 0.4                       # the model is unsure about the tip only
    pet[tip] = 2.5

    bg = [[95, 24, 20]]
    out, info = apply_background_scribbles(
        bar.astype(np.uint8), pet, bg, prob=prob, spacing=SPACING, tracer="fdg",
        return_info=True,
    )
    assert check_constraints(out, [], bg)["ok"]
    kept = out.astype(bool)
    # the confident far end (i < 60) is > 40 mm away and must be untouched
    assert kept[10:60, 20:28, 16:24].all(), "the removal reached far past the scribble"
    assert info["n_components_deleted"] == 0, "a component with a core was deleted whole"


def test_rejected_split_falls_back_to_local_removal_not_demolition():
    """When the basin is too large the fallback must be less damage, not more."""
    shape = (64, 64, 48)
    ct, pet, gt, bm = make_phantom(
        shape=shape,
        blobs=[Blob(centre=(32, 32, 24), radius_mm=12.0, suv=6.0, name="blob")],
        background_suv=0.6,
        seed=-1,
    )
    blob = bm["blob"]
    prob = np.where(blob, 0.93, 0.0).astype(np.float32)
    # scribble through the middle: the basin would claim a huge share
    bg = [[32, int(j), 24] for j in range(24, 40)]

    from postproc.config import ComplianceConfig

    cfg = ComplianceConfig(bg_protect_confident_core=False)   # isolate the size rule
    out, info = apply_background_scribbles(
        blob.astype(np.uint8), pet, bg, prob=prob, spacing=SPACING, tracer="fdg",
        cfg=cfg, return_info=True,
    )
    assert check_constraints(out, [], bg)["ok"]
    assert info["n_components_deleted"] == 0, "rejecting the split deleted everything"
    kept = (out.astype(bool) & blob).sum() / blob.sum()
    assert kept > 0.5, f"the fallback removed too much ({1 - kept:.1%})"


def test_growth_is_capped_relative_to_the_evidence_it_starts_from():
    """A scribble on a small node inside a large warm region must not grow to 100 mL."""
    shape = (80, 80, 60)
    ct, pet, gt, _ = make_phantom(shape=shape, blobs=[], background_suv=0.6, seed=-1)
    warm = sphere_mask(shape, (40, 40, 30), 40.0, SPACING)
    pet[warm] = 4.0
    node = sphere_mask(shape, (40, 40, 30), 5.0, SPACING)
    pet[node] = 8.0

    fg = [[40, 40, 30]]
    out, info = apply_tumor_scribbles(
        np.zeros(shape, np.uint8), pet, ct, fg, tracer="fdg", spacing=SPACING,
        return_info=True,
    )
    assert check_constraints(out, fg, [])["ok"]
    grown_ml = out.sum() * VOX_ML
    detail = info["clusters"][0]
    assert detail["max_volume_ml"] <= 12.0, detail
    assert grown_ml < 5.0, f"grew {grown_ml:.1f} mL from one scribble on a 0.5 mL node"
    assert (out.astype(bool) & node).sum() > 0.5 * node.sum(), "the node was not recovered"


# ---------------------------------------------------------------------------
# regression: the tracer floor is not a delineation threshold
# ---------------------------------------------------------------------------
def _tail_flood_case():
    """A lesion, a tract of elevated uptake leading away from it, and a scribble running
    down the tract.

    The model found the lesion and the near part of the tract, so growth is seeded from
    the distal tail alone -- far enough that its 1 mL box mean sees only the 2.0 SUV
    tract.
    """
    shape = (90, 130, 60)
    ct, pet, gt, bm = make_phantom(
        shape=shape,
        blobs=[Blob(centre=(45, 40, 30), radius_mm=13.0, suv=12.0, name="lesion")],
        background_suv=0.8,
        seed=-1,
    )
    lesion = bm["lesion"]
    tract = np.zeros(shape, dtype=bool)
    tract[40:50, 45:115, 27:34] = True
    tract &= ~lesion
    pet[tract] = 2.0                       # above the FDG floor, far below the lesion

    near = np.zeros(shape, dtype=bool)
    near[:, :60, :] = True
    mask = (lesion | (tract & near)).astype(np.uint8)   # what the model found
    fg = [[45, int(j), 30] for j in range(40, 110)]     # one stroke, hot end to far tail
    return ct, pet, mask, lesion, tract, fg


def test_scribble_tail_in_normal_tissue_does_not_flood():
    """Growth seeded from the cold tail of a stroke must not flood normal tissue.

    Measuring the local SUVpeak on the tail alone drops `0.41 * peak` below the tracer
    floor, and the floor is a "cannot possibly be lesion" bound rather than a
    delineation criterion, so the region grows by tens of mL -- still under the volume
    cap. Run with the guards off and on.
    """
    from postproc.config import ComplianceConfig

    ct, pet, mask, lesion, tract, fg = _tail_flood_case()
    inside = mask[tuple(np.asarray(fg).T)].astype(bool)
    assert inside.any() and not inside.all(), "fixture must straddle what the model found"

    broken = ComplianceConfig(fg_peak_over_whole_cluster=False, fg_low_contrast_fallback=False)
    out_broken, info_broken = apply_tumor_scribbles(
        mask, pet, ct, fg, tracer="fdg", spacing=SPACING, cfg=broken, return_info=True
    )
    flooded = (int(out_broken.sum()) - int(mask.sum())) * VOX_ML

    out, info = apply_tumor_scribbles(
        mask, pet, ct, fg, tracer="fdg", spacing=SPACING, return_info=True
    )
    added = (int(out.sum()) - int(mask.sum())) * VOX_ML

    assert check_constraints(out, fg, [])["ok"]
    assert info_broken["clusters"][0]["threshold"] == pytest.approx(1.5), (
        "the fixture no longer collapses onto the tracer floor: "
        f"{info_broken['clusters'][0]}"
    )
    assert flooded > 10.0, (
        f"fixture no longer reproduces the flood ({flooded:.1f} mL) -- the guards are "
        "being tested against nothing"
    )
    # What survives is the bounded ball G2 requires around a 100 mm stroke, not a flood:
    # a 4 mm tube along the scribble is ~5 mL and is the honest answer to "these voxels
    # are tumour" when the intensity evidence cannot delineate anything.
    assert added < 8.0, f"growth still floods: added {added:.1f} mL (was {flooded:.1f})"
    assert added < 0.25 * flooded, f"added {added:.1f} mL vs {flooded:.1f} mL unguarded"
    detail = info["clusters"][0]
    assert detail["threshold"] is None or detail["threshold"] > 1.5 + 1e-6, detail


def test_peak_is_measured_over_the_whole_scribble_not_the_missing_part():
    """A stroke annotates one lesion; its cold tail says nothing about that lesion's peak."""
    ct, pet, mask, lesion, tract, fg = _tail_flood_case()
    _, info = apply_tumor_scribbles(
        mask, pet, ct, fg, tracer="fdg", spacing=SPACING, return_info=True
    )
    detail = info["clusters"][0]
    assert detail["raw_peak"] > 8.0, (
        f"peak {detail['raw_peak']:.1f} was taken from the cold tail, not the stroke"
    )
    assert detail["low_contrast"] is False


def test_low_contrast_scribble_falls_back_to_a_ball():
    """No usable intensity evidence -> a small bounded ball, never a flood."""
    shape = (70, 70, 50)
    ct, pet, gt, _ = make_phantom(shape=shape, blobs=[], background_suv=1.9, seed=-1)
    fg = [[35, 35, 25], [35, 36, 25], [35, 37, 25]]
    out, info = apply_tumor_scribbles(
        np.zeros(shape, np.uint8), pet, ct, fg, tracer="fdg", spacing=SPACING,
        return_info=True,
    )
    detail = info["clusters"][0]
    assert detail["low_contrast"] is True, detail
    assert detail["fallback"] == "ball_low_contrast", detail
    assert check_constraints(out, fg, [])["ok"]
    assert out.sum() * VOX_ML < 1.5, f"the fallback is not bounded ({out.sum() * VOX_ML:.2f} mL)"


def test_calibration_survives_a_cluster_spanning_several_slices():
    """Accumulated scribbles land on different slices and cluster together.

    Calibration must then use the slice holding most of the voxels being grown from;
    abandoning it altogether drops the cluster back onto the alpha rule.
    """
    shape = (64, 64, 48)
    ct, pet, gt, bm = make_phantom(
        shape=shape,
        blobs=[Blob(centre=(32, 32, 24), radius_mm=9.0, suv=7.0, name="lesion")],
        background_suv=1.2,
        seed=-1,
    )
    halo = sphere_mask(shape, (32, 32, 24), 15.0, SPACING) & ~bm["lesion"]
    pet[halo] = 2.4                       # the alpha rule would swallow this

    # a footprint on slice 24 plus a couple of stray points on slice 25, as the
    # accumulated scribble set looks after a few iterations
    fp = bm["lesion"][:, :, 24]
    fg = [[int(i), int(j), 24] for i, j in np.argwhere(fp)]
    fg += [[32, 32, 25], [33, 32, 25]]

    out, info = apply_tumor_scribbles(
        np.zeros(shape, np.uint8), pet, ct, fg, tracer="fdg", spacing=SPACING,
        return_info=True,
    )
    detail = info["clusters"][0]
    assert detail["calibrated"] is True, detail
    assert detail["footprint_thickness_mm"] > 2.5, detail
    assert check_constraints(out, fg, [])["ok"]
    grown = out.astype(bool)
    assert (grown & halo).sum() < 0.2 * halo.sum(), "calibration failed to reject the halo"


def test_footprint_target_picks_the_dominant_slice():
    """Unit test of the slice choice, independent of the growth."""
    from postproc.compliance import _footprint_target

    seed_mask = np.zeros((10, 10, 5), dtype=bool)
    seed_mask[3:8, 3:8, 2] = True      # 25 voxels on slice 2
    seed_mask[4, 4, 4] = True          # one stray voxel on slice 4
    own_mask = seed_mask.copy()
    k, footprint, target = _footprint_target(seed_mask, own_mask, None)
    assert k == 2, f"picked slice {k} instead of the dominant one"
    assert footprint.sum() == 25
