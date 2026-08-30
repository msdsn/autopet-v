"""Checks for `ensemble_predictor.EnsembleInteractivePredictor` (row E2).

Two parts:

* arithmetic, on stub members, with no GPU -- weight normalisation, the `w == 0` member
  being absent rather than added as `0.0 * p`, the `p > 0.5` tie-break;
* the real thing, on one case and two real checkpoints: with weights `[1, 0]` the
  ensemble must reproduce member 0's mask **and** its foreground probability bit for bit,
  and equally for `[0, 1]` and member 1.  That is the property that makes an ensemble row
  interpretable: any difference from a member is the other member's doing, not the
  wrapper's.

    python3 src/test_ensemble_predictor.py                          # stubs only
    python3 src/test_ensemble_predictor.py --members A B --case_dir /content/work/evalset

Run it from `src/` (or with `src` on PYTHONPATH), like the rest of the harness.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ensemble_predictor import (  # noqa: E402
    EnsembleInteractivePredictor,
    parse_member_spec,
)


# ---------------------------------------------------------------------------
# part 1: arithmetic, no GPU
# ---------------------------------------------------------------------------
class _StubMember:
    """A member that returns a fixed probability field, ignoring its inputs."""

    def __init__(self, prob):
        self.prob = np.asarray(prob, dtype=np.float32)
        self.last_timings = {"network_s": 0.0}
        self.calls = 0

    def predict(self, ct, pet, spacing, scribbles, prev_pred=None, case_cache_dir=None,
                **kw):
        self.calls += 1
        self.seen_prev = None if prev_pred is None else np.asarray(prev_pred).copy()
        mask = (self.prob > 0.5).astype(np.uint8)
        return mask, np.stack([1.0 - self.prob, self.prob])

    def cache_state_key(self, prev_pred):
        return "k" if prev_pred is None else str(int(np.asarray(prev_pred).sum()))


def test_arithmetic():
    shape = (4, 5, 6)
    rng = np.random.default_rng(0)
    pa = rng.random(shape, dtype=np.float32)
    pb = rng.random(shape, dtype=np.float32)
    a, b = _StubMember(pa), _StubMember(pb)
    ct = pet = np.zeros(shape, dtype=np.float32)
    sp = (2.0, 2.0, 3.0)

    # weights are normalised
    e = EnsembleInteractivePredictor([a, b], [3.0, 7.0])
    assert np.allclose(e.weights, [0.3, 0.7]), e.weights
    m, p = e.predict(ct, pet, sp, None, return_probabilities=True)
    exp = np.float32(0.3) * pa + np.float32(0.7) * pb
    assert np.array_equal(p, exp), "0.3/0.7 mean is not bit-exact"
    assert np.array_equal(m, (exp > 0.5).astype(np.uint8))

    # a zero-weight member is absent from the sum, bit for bit
    e = EnsembleInteractivePredictor([a, b], [1.0, 0.0])
    m, p = e.predict(ct, pet, sp, None, return_probabilities=True)
    assert np.array_equal(p, pa), "w=[1,0] did not reproduce member 0 exactly"
    assert np.array_equal(m, (pa > 0.5).astype(np.uint8))
    e = EnsembleInteractivePredictor([a, b], [0.0, 1.0])
    m, p = e.predict(ct, pet, sp, None, return_probabilities=True)
    assert np.array_equal(p, pb), "w=[0,1] did not reproduce member 1 exactly"

    # equal weights is the default
    e = EnsembleInteractivePredictor([a, b])
    _, p = e.predict(ct, pet, sp, None, return_probabilities=True)
    assert np.array_equal(p, np.float32(0.5) * pa + np.float32(0.5) * pb)

    # every member is handed the ensemble's own previous mask
    prev = (rng.random(shape) > 0.5).astype(np.uint8)
    e.predict(ct, pet, sp, None, prev_pred=prev)
    assert np.array_equal(a.seen_prev, prev) and np.array_equal(b.seen_prev, prev)

    # exactly p > 0.5, so a 0.5 tie is background (nnU-Net's argmax(0) tie-break)
    half = _StubMember(np.full(shape, 0.5, dtype=np.float32))
    _, p = EnsembleInteractivePredictor([half]).predict(
        ct, pet, sp, None, return_probabilities=True)
    assert (p == 0.5).all()
    assert EnsembleInteractivePredictor([half]).predict(ct, pet, sp, None).sum() == 0

    # spec parsing
    assert parse_member_spec("/m/a") == {"model_folder": "/m/a",
                                         "checkpoint": "checkpoint_final.pth",
                                         "weight": None}
    assert parse_member_spec("/m/a:checkpoint_best.pth:0.3") == {
        "model_folder": "/m/a", "checkpoint": "checkpoint_best.pth", "weight": 0.3}
    for bad in (["x"], [1.0, 2.0, 3.0]):
        try:
            EnsembleInteractivePredictor([a, b], bad)
        except (ValueError, TypeError):
            pass
        else:
            raise AssertionError(f"bad weights {bad} accepted")
    try:
        EnsembleInteractivePredictor([a, b], [0.0, 0.0])
    except ValueError:
        pass
    else:
        raise AssertionError("all-zero weights accepted")
    print("ok  arithmetic: normalisation, zero-weight exactness, 0.5 tie-break, prev_pred")


# ---------------------------------------------------------------------------
# part 2: the real members, one case, on the GPU
# ---------------------------------------------------------------------------
def _load_case(case_dir, case=None):
    import nibabel as nib
    img_dir = os.path.join(case_dir, "imagesTr")
    names = sorted(f[:-12] for f in os.listdir(img_dir) if f.endswith("_0000.nii.gz"))
    if not names:
        raise SystemExit(f"no _0000.nii.gz in {img_dir}")
    tag = case or names[0]
    ct = nib.load(os.path.join(img_dir, f"{tag}_0000.nii.gz"))
    pet = nib.load(os.path.join(img_dir, f"{tag}_0001.nii.gz"))
    sp = [float(x) for x in ct.header.get_zooms()[:3]]
    return tag, np.asarray(ct.dataobj, dtype=np.float32), \
        np.asarray(pet.dataobj, dtype=np.float32), sp, ct.affine


def test_real(members, case_dir, case=None, device="cuda", tile_step_size=0.5,
              enable_tta=False, mirror_axes=None):
    from predictor import InteractiveNNUNetPredictor

    tag, ct, pet, sp, affine = _load_case(case_dir, case)
    print(f"case {tag}  shape={pet.shape}  spacing={tuple(round(x, 4) for x in sp)}")
    common = dict(device=device, folds=(0,), tile_step_size=tile_step_size,
                  disable_tta=not enable_tta,
                  force_mirror_axes=tuple(mirror_axes) if mirror_axes else None,
                  resample_channels="scipy", resample_logits="torch",
                  num_resample_threads=4, deterministic=True)
    specs = [parse_member_spec(s) for s in members]

    def build():
        return [InteractiveNNUNetPredictor(model_folder=s["model_folder"],
                                           checkpoint_name=s["checkpoint"], **common)
                for s in specs]

    ms = build()
    scribbles = {"tumor": [], "background": []}
    solo_mask, solo_prob = [], []
    for m, s in zip(ms, specs):
        mask, prob = m.predict(ct, pet, sp, scribbles, prev_pred=None, affine=affine,
                               case_name=tag, return_probabilities=True)
        solo_mask.append(np.asarray(mask).copy())
        solo_prob.append(np.asarray(prob)[1].astype(np.float32).copy())
        print(f"  member {os.path.basename(s['model_folder'])}#{s['checkpoint']}: "
              f"{int(mask.sum())} fg voxels, "
              f"network_s={m.last_timings.get('network_s')}")
        del mask, prob

    for i in range(len(ms)):
        w = [0.0] * len(ms)
        w[i] = 1.0
        e = EnsembleInteractivePredictor(ms, w)
        mask, prob = e.predict(ct, pet, sp, scribbles, prev_pred=None, affine=affine,
                               case_name=tag, return_probabilities=True)
        assert np.array_equal(np.asarray(prob), solo_prob[i]), \
            f"w={w}: probabilities differ from member {i} standalone"
        assert np.array_equal(np.asarray(mask), solo_mask[i]), \
            f"w={w}: mask differs from member {i} standalone"
        print(f"ok  w={w} reproduces member {i} bit-exactly "
              f"({int(mask.sum())} fg voxels)")
        del mask, prob

    e = EnsembleInteractivePredictor(ms)
    mask, prob = e.predict(ct, pet, sp, scribbles, prev_pred=None, affine=affine,
                           case_name=tag, return_probabilities=True)
    exp = np.zeros_like(solo_prob[0])
    for p, w in zip(solo_prob, e.weights):
        exp += np.float32(w) * p
    assert np.array_equal(np.asarray(prob), exp), "equal-weight mean is not bit-exact"
    assert np.array_equal(np.asarray(mask), (exp > 0.5).astype(np.uint8))
    agree = [float((np.asarray(mask) == sm).mean()) for sm in solo_mask]
    print(f"ok  equal weights: {int(mask.sum())} fg voxels, "
          f"voxel agreement with the members {['%.6f' % a for a in agree]}, "
          f"total_s={e.last_timings['total_s']}, "
          f"network_s={e.last_timings['network_s']}")
    for m in ms:
        m.close()


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--members", nargs="*", default=None,
                    help="'<model_folder>[:<checkpoint>]' per member; without this only "
                         "the arithmetic part runs")
    ap.add_argument("--case_dir", default="/content/work/evalset")
    ap.add_argument("--case", default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--tile_step_size", type=float, default=0.5)
    ap.add_argument("--enable_tta", action="store_true")
    ap.add_argument("--mirror_axes", type=int, nargs="*", default=None)
    a = ap.parse_args()

    test_arithmetic()
    if a.members:
        test_real(a.members, a.case_dir, a.case, a.device, a.tile_step_size,
                  a.enable_tta, a.mirror_axes)
    else:
        print("(skipped the GPU part; pass --members to run it)")
    print("ALL OK")


if __name__ == "__main__":
    main()
