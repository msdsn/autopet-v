"""Point / geometry helpers -- mostly a guard on the coordinate convention."""

from __future__ import annotations

import numpy as np
import pytest

from postproc.utils import (
    as_points_array,
    ball_offsets,
    cluster_points,
    foreground_prob,
    ml_to_voxels,
    points_in_bounds,
    points_mask,
    unique_points,
    voxel_volume_ml,
)

from conftest import SPACING


def test_as_points_array_accepts_every_shape_the_challenge_uses():
    assert as_points_array(None).shape == (0, 3)
    assert as_points_array([]).shape == (0, 3)
    assert as_points_array([[1, 2, 3]]).tolist() == [[1, 2, 3]]
    assert as_points_array(((1, 2, 3), (4, 5, 6))).tolist() == [[1, 2, 3], [4, 5, 6]]
    assert as_points_array(np.array([[1.0, 2.0, 3.0]])).tolist() == [[1, 2, 3]]


def test_unique_points_preserves_first_seen_order():
    pts = as_points_array([[3, 3, 3], [1, 1, 1], [3, 3, 3], [2, 2, 2]])
    assert unique_points(pts).tolist() == [[3, 3, 3], [1, 1, 1], [2, 2, 2]]


def test_points_in_bounds():
    pts = as_points_array([[0, 0, 0], [9, 9, 9], [10, 0, 0], [0, -1, 0]])
    assert points_in_bounds(pts, (10, 10, 10)).tolist() == [[0, 0, 0], [9, 9, 9]]


def test_points_index_arrays_in_ijk_order():
    """point[0] indexes axis 0, point[2] the axial slice -- never transposed."""
    vol = np.zeros((7, 11, 13), dtype=bool)
    m = points_mask(vol.shape, [[1, 2, 3]], radius_mm=0.0, spacing=SPACING)
    assert m.sum() == 1
    assert m[1, 2, 3]


def test_ball_offsets_are_anisotropy_aware():
    offs = ball_offsets(6.0, SPACING)  # 6 mm with 2.04/2.04/3.0 mm voxels
    assert (offs == 0).all(axis=1).sum() == 1  # contains the centre
    reach = np.abs(offs).max(axis=0)
    assert reach.tolist() == [2, 2, 2]  # floor(6/2.04)=2, floor(6/3.0)=2
    d = np.sqrt(((offs * np.asarray(SPACING)) ** 2).sum(axis=1))
    assert d.max() <= 6.0 + 1e-9


def test_ball_offsets_zero_radius():
    assert ball_offsets(0.0, SPACING).tolist() == [[0, 0, 0]]


def test_points_mask_paints_a_ball():
    m = points_mask((20, 20, 20), [[10, 10, 10]], radius_mm=4.0, spacing=SPACING)
    assert m[10, 10, 10] and m[11, 10, 10] and not m[15, 10, 10]


def test_points_mask_clips_at_the_border():
    m = points_mask((6, 6, 6), [[0, 0, 0], [5, 5, 5]], radius_mm=5.0, spacing=SPACING)
    assert m[0, 0, 0] and m[5, 5, 5]  # no IndexError, no wraparound
    assert not m[0, 5, 0]


def test_cluster_points_groups_one_scribble_and_separates_lesions():
    a = [[10, 10, 10], [11, 10, 10], [12, 10, 10]]
    b = [[60, 60, 40], [61, 60, 40]]
    clusters = cluster_points(as_points_array(a + b), SPACING, radius_mm=20.0)
    assert len(clusters) == 2
    assert sorted(len(c) for c in clusters) == [2, 3]


def test_voxel_volume():
    assert voxel_volume_ml(SPACING) == pytest.approx(0.0124849, rel=1e-4)
    assert ml_to_voxels(1.0, SPACING) == pytest.approx(1 / 0.0124849, rel=1e-4)


def test_foreground_prob_channel_handling():
    p3 = np.zeros((4, 4, 4), dtype=np.float32)
    assert foreground_prob(p3).shape == (4, 4, 4)
    p4 = np.zeros((2, 4, 4, 4), dtype=np.float32)
    p4[1] = 0.7
    assert foreground_prob(p4)[0, 0, 0] == pytest.approx(0.7)
    assert foreground_prob(None) is None
    with pytest.raises(ValueError):
        foreground_prob(p3, shape=(5, 5, 5))
