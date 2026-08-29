"""Accumulated constraint state and the per-case cache."""

from __future__ import annotations

import json
import os

import numpy as np

from postproc.constraints import CaseCache, ConstraintState, load_packed_mask, save_packed_mask


def test_add_scribbles_accumulates_and_dedups():
    s = ConstraintState()
    assert s.add_scribbles({"tumor": [[1, 2, 3]], "background": []}) == (1, 0)
    # the evaluator re-sends the whole accumulated list every iteration
    assert s.add_scribbles({"tumor": [[1, 2, 3]], "background": [[9, 9, 9]]}) == (0, 1)
    assert s.n_tumor == 1 and s.n_background == 1
    assert s.add_scribbles(None) == (0, 0)
    assert s.add_scribbles({}) == (0, 0)


def test_tumor_wins_over_background_on_a_shared_voxel():
    s = ConstraintState(tumor_points=[[4, 4, 4]], background_points=[[4, 4, 4], [5, 5, 5]])
    assert s.tumor_array().tolist() == [[4, 4, 4]]
    assert s.background_array().tolist() == [[5, 5, 5]]


def test_out_of_bounds_points_are_dropped():
    s = ConstraintState(tumor_points=[[1, 1, 1], [999, 0, 0], [-1, 0, 0]], shape=[10, 10, 10])
    assert s.tumor_array().tolist() == [[1, 1, 1]]


def test_state_roundtrip(cache_dir):
    s = ConstraintState(
        tumor_points=[[1, 2, 3]],
        background_points=[[4, 5, 6]],
        iteration=2,
        tracer="psma",
        shape=[10, 10, 10],
    )
    s.save(cache_dir)
    t = ConstraintState.load(cache_dir)
    assert t.tumor_points == s.tumor_points
    assert t.background_points == s.background_points
    assert t.iteration == 2 and t.tracer == "psma"


def test_load_of_a_missing_or_corrupt_cache_degrades_gracefully(cache_dir):
    assert ConstraintState.load(None).iteration == 0
    assert ConstraintState.load(cache_dir).iteration == 0
    with open(os.path.join(cache_dir, "postproc_constraints.json"), "w") as f:
        f.write("{not json")
    assert ConstraintState.load(cache_dir).iteration == 0


def test_packed_mask_roundtrip(tmp_path):
    rng = np.random.default_rng(0)
    m = rng.random((37, 41, 29)) > 0.97
    p = str(tmp_path / "m.npz")
    save_packed_mask(p, m)
    back = load_packed_mask(p)
    assert back is not None and back.shape == m.shape
    assert np.array_equal(back, m)
    assert load_packed_mask(str(tmp_path / "nope.npz")) is None


def test_case_cache_prob_roundtrip_and_bg_accumulation(cache_dir):
    c = CaseCache(cache_dir)
    assert c.load_prev_prob() is None
    prob = np.linspace(0, 1, 8 * 8 * 8, dtype=np.float32).reshape(8, 8, 8)
    c.save_prev_prob(prob)
    back = c.load_prev_prob()
    assert back is not None and np.abs(back - prob).max() < 1.0 / 255 + 1e-6

    a = np.zeros((8, 8, 8), dtype=bool)
    a[0, 0, 0] = True
    b = np.zeros((8, 8, 8), dtype=bool)
    b[1, 1, 1] = True
    c.accumulate_bg_region(a)
    acc = c.accumulate_bg_region(b)
    assert acc[0, 0, 0] and acc[1, 1, 1], "background regions must accumulate monotonically"


def test_no_cache_dir_is_a_silent_noop():
    c = CaseCache(None)
    c.save_prev_mask(np.ones((4, 4, 4), dtype=bool))
    assert c.load_prev_mask() is None
    assert c.load_state().iteration == 0


# ---------------------------------------------------------------------------
# no state directory at all (preliminary phase)
# ---------------------------------------------------------------------------
def test_cache_never_creates_or_raises_on_a_missing_directory(tmp_path):
    missing = str(tmp_path / "does" / "not" / "exist")
    c = CaseCache(missing)
    assert not c.exists and not c.has_state()
    assert c.load_state().iteration == 0
    assert c.load_prev_prob() is None and c.load_prev_mask() is None
    assert c.load_bg_region() is None
    # saving creates it lazily; if that is impossible it must still not raise
    c.save_state(ConstraintState(iteration=1))
    assert CaseCache(missing).load_state().iteration == 1

    unwritable = CaseCache("/proc/definitely/not/writable")
    unwritable.save_state(ConstraintState(iteration=3))   # must not raise
    unwritable.save_prev_mask(np.zeros((4, 4, 4), dtype=bool))
    assert unwritable.load_state().iteration == 0


def test_constraints_are_fully_recoverable_from_the_scribble_json():
    """The json carries every scribble so far, so the state survives with no cache."""
    scribbles = {"tumor": [[10, 10, 10], [11, 10, 10]], "background": [[40, 40, 30]]}
    st = ConstraintState.from_scribbles(scribbles, shape=(64, 64, 48), spacing=(2.04, 2.04, 3.0))
    assert st.tumor_array().tolist() == scribbles["tumor"]
    assert st.background_array().tolist() == scribbles["background"]
    assert st.shape == [64, 64, 48]


def test_infer_iteration_counts_scribble_events():
    sp = (2.04, 2.04, 3.0)
    empty = ConstraintState(shape=[64, 64, 48], spacing=list(sp))
    assert empty.infer_iteration() == 0                       # iteration 0: no scribbles

    one = ConstraintState.from_scribbles(
        {"tumor": [[10, 10, 10], [11, 10, 10], [12, 10, 10]], "background": []},
        (64, 64, 48), sp,
    )
    assert one.infer_iteration() == 1                          # one line == one event

    two = ConstraintState.from_scribbles(
        {"tumor": [[10, 10, 10], [11, 10, 10]], "background": [[50, 50, 40]]},
        (64, 64, 48), sp,
    )
    assert two.infer_iteration() == 2

    many = ConstraintState.from_scribbles(
        {"tumor": [[3 + 20 * i, 3, 3] for i in range(3)],
         "background": [[3, 3 + 20 * i, 40] for i in range(5)]},
        (128, 128, 64), sp,
    )
    assert many.infer_iteration() == 5                          # clamped to 6 iterations
