"""Timing on a realistic whole-body array (400 x 400 x 330).

A container call has 20 minutes on an A10G and the network needs most of it, so the
post-processing layer is budgeted at 20 s on 8 CPU cores. AUTOPET_SKIP_SLOW=1 skips it.

    python -m pytest src/postproc/tests/test_timing.py -s
"""

from __future__ import annotations

import os
import time

import numpy as np
import pytest

from postproc.cleanup import cleanup_mask
from postproc.compliance import apply_background_scribbles, apply_tumor_scribbles, check_constraints
from postproc.negative_gate import is_probably_negative
from postproc.pipeline import PostProcPredictor

from conftest import SPACING, sphere_mask

pytestmark = pytest.mark.skipif(
    os.environ.get("AUTOPET_SKIP_SLOW") == "1", reason="AUTOPET_SKIP_SLOW=1"
)

SHAPE = (400, 400, 330)
BUDGET_S = 20.0


def _wholebody():
    """A whole-body-sized phantom built cheaply (no meshgrid over 53 M voxels twice)."""
    rng = np.random.default_rng(0)
    pet = rng.random(SHAPE, dtype=np.float32) * 0.8          # background 0-0.8 SUV
    ct = np.full(SHAPE, -200.0, dtype=np.float32)
    ct[40:360, 40:360, :] = 0.0

    lesions = []
    centres = [(100, 120, 60), (250, 180, 140), (300, 300, 250), (150, 320, 200), (200, 200, 300)]
    for n, c in enumerate(centres):
        m = sphere_mask(SHAPE, c, 12.0 if n % 2 else 8.0, SPACING)
        pet[m] = 6.0 + n
        lesions.append(m)

    # 300 small false-positive specks, the DMM killers
    idx = rng.integers([50, 50, 20], [350, 350, 310], size=(300, 3))
    mask = np.zeros(SHAPE, dtype=np.uint8)
    for i, j, k in idx:
        mask[i : i + 2, j : j + 2, k : k + 1] = 1
        pet[i : i + 2, j : j + 2, k : k + 1] = 1.9
    for m in lesions:
        mask[m] = 1
    return ct, pet, mask, lesions


@pytest.fixture(scope="module")
def wholebody():
    return _wholebody()


class _Base:
    name = "cached"

    def __init__(self, mask, prob):
        self.mask, self.prob = mask, prob

    def predict(self, ct, pet, spacing, scribbles=None, prev_pred=None, case_cache_dir=None,
                *, return_probabilities=False, **kw):
        if return_probabilities:
            return self.mask.copy(), self.prob
        return self.mask.copy()


def test_stage_timings(wholebody):
    ct, pet, mask, lesions = wholebody
    fg = [[int(c) for c in p] for p in np.argwhere(lesions[1])[::200][:12]]
    bg = [[int(c) for c in p] for p in np.argwhere(lesions[0])[::200][:12]]

    timings = {}
    t0 = time.perf_counter()
    m1 = apply_background_scribbles(mask, pet, bg, spacing=SPACING)
    timings["bg_compliance"] = time.perf_counter() - t0

    m2 = m1.copy()
    m2[lesions[1]] = 0
    t0 = time.perf_counter()
    m3 = apply_tumor_scribbles(m2, pet, ct, fg, tracer="fdg", spacing=SPACING)
    timings["fg_compliance(grow)"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    apply_tumor_scribbles(m2, pet, ct, fg, tracer="fdg", spacing=SPACING, method="random_walker")
    timings["fg_compliance(random_walker)"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    m4 = cleanup_mask(m3, pet, SPACING, tracer="fdg", protect_points=fg)
    timings["cleanup"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    is_probably_negative(m4, pet, None, spacing=SPACING)
    timings["negative_gate"] = time.perf_counter() - t0

    print(f"\n[timing] volume {SHAPE} = {np.prod(SHAPE) / 1e6:.1f} M voxels")
    for k, v in timings.items():
        print(f"[timing]   {k:32s} {v:7.2f} s")
    print(f"[timing]   {'SUM (excl. random_walker)':32s} "
          f"{sum(v for k, v in timings.items() if 'random_walker' not in k):7.2f} s")

    assert check_constraints(m4, fg, bg)["ok"]
    assert sum(v for k, v in timings.items() if "random_walker" not in k) < BUDGET_S


def test_full_pipeline_timing(wholebody, tmp_path):
    ct, pet, mask, lesions = wholebody
    prob = np.where(mask > 0, 0.9, 0.05).astype(np.float32)
    pp = PostProcPredictor(_Base(mask, prob), {"tracer": "fdg", "verbose": True})
    cache = str(tmp_path / "cache")
    os.makedirs(cache, exist_ok=True)

    fg = [[int(c) for c in p] for p in np.argwhere(lesions[1])[::200][:12]]
    bg = [[int(c) for c in p] for p in np.argwhere(lesions[0])[::200][:12]]
    scribbles = {"tumor": [], "background": []}

    totals = []
    for it in range(5):
        if it == 1:
            scribbles["background"] += bg
        if it == 2:
            scribbles["tumor"] += fg
        t0 = time.perf_counter()
        out = pp.predict(ct, pet, SPACING, scribbles, None, cache)
        dt = time.perf_counter() - t0
        totals.append(dt)
        info = pp.last_info
        overhead = dt - info.get("t_base_predict", 0.0)
        print(f"[timing] iteration {it}: total {dt:.2f} s, postproc overhead {overhead:.2f} s")
        assert check_constraints(out, scribbles["tumor"], scribbles["background"])["ok"]
        assert overhead < BUDGET_S, f"iteration {it} took {overhead:.1f} s of post-processing"

    print(f"[timing] mean per-iteration total {np.mean(totals):.2f} s")
