"""Synthetic-patch checks for the two nnUNetTrainer_InteractiveV2 loss terms.

Covers ranges and gradients on random, empty and perfect patches; the all-empty
batch, where DC+CE at smooth=0 has an exactly zero Dice gradient; the vectorised
instance Dice against a literal per-instance loop; size invariance; the compound
loss over a deep-supervision list; and cost per call. Run with --cuda for the GPU.
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import torch

try:
    from .nnUNetTrainer_InteractiveV2 import (InstanceDiceLoss, InteractiveV2Loss,
                                              LesionFreeFPLoss)
except ImportError:  # flat import
    from nnUNetTrainer_InteractiveV2 import (InstanceDiceLoss,  # type: ignore
                                             InteractiveV2Loss, LesionFreeFPLoss)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def make_patch(shape=(2, 24, 32, 32), lesions=((3, 3), (8, 1)), device="cpu", seed=0):
    """(logits, target).  `lesions` is a list of (radius, count) per batch element."""
    g = torch.Generator().manual_seed(seed)
    b = shape[0]
    tgt = torch.zeros((b, 1, *shape[1:]), dtype=torch.long)
    rng = np.random.default_rng(seed)
    for i in range(b):
        r, n = lesions[i % len(lesions)]
        for _ in range(n):
            c = [int(rng.integers(r + 1, s - r - 1)) for s in shape[1:]]
            sl = tuple(slice(ci - r, ci + r + 1) for ci in c)
            tgt[(i, 0) + sl] = 1
    logits = torch.randn((b, 2, *shape[1:]), generator=g)
    return logits.to(device).requires_grad_(True), tgt.to(device)


def assert_close(a, b, tol=1e-5, msg=""):
    assert abs(float(a) - float(b)) <= tol, f"{msg}: {float(a)} vs {float(b)}"


def check(name, cond, extra=""):
    print(f"  [{'ok ' if cond else 'FAIL'}] {name} {extra}")
    assert cond, name


# ---------------------------------------------------------------------------
# reference implementation of blob loss, straight from the paper's formulation
# ---------------------------------------------------------------------------

def reference_instance_dice(logits, target, smooth=1.0, connectivity=18):
    import cc3d
    p = torch.softmax(logits.float(), 1)[:, 1]
    fg = target[:, 0] > 0
    per_sample = []
    for i in range(p.shape[0]):
        m = fg[i].cpu().numpy().astype(np.uint8)
        if m.sum() == 0:
            continue
        inst = cc3d.connected_components(m, connectivity=connectivity)
        n = int(inst.max())
        dl = []
        for k in range(1, n + 1):
            the = torch.as_tensor(inst == k, device=p.device)
            others = torch.as_tensor((inst > 0) & (inst != k), device=p.device)
            masked = p[i] * (~others)                      # blob-loss masking
            inter = (masked * the).sum()
            dice = (2 * inter + smooth) / (masked.sum() + the.sum() + smooth)
            dl.append(1.0 - dice)
        if dl:
            per_sample.append(torch.stack(dl).mean())
    if not per_sample:
        return p.sum() * 0.0
    return torch.stack(per_sample).mean()


# ---------------------------------------------------------------------------
# tests
# ---------------------------------------------------------------------------

def test_finite_and_range(device):
    print("1. finiteness / range")
    lf, bl = LesionFreeFPLoss(), InstanceDiceLoss()
    for name, (logits, tgt) in {
        "random": make_patch(device=device),
        "empty": (torch.randn(2, 2, 16, 16, 16, device=device, requires_grad=True),
                  torch.zeros(2, 1, 16, 16, 16, dtype=torch.long, device=device)),
    }.items():
        a, b = lf(logits, tgt).detach(), bl(logits, tgt).detach()
        check(f"{name}: lesion-free finite in [0,1)", torch.isfinite(a) and 0 <= a < 1,
              f"= {float(a):.4f}")
        check(f"{name}: blob finite in [0,1]", torch.isfinite(b) and -1e-6 <= b <= 1 + 1e-6,
              f"= {float(b):.4f}")

    # perfect prediction -> instance Dice ~ 0, lesion-free term ~ 0 on an empty patch
    logits, tgt = make_patch(device=device)
    perfect = torch.stack([(tgt[:, 0] == 0).float(), (tgt[:, 0] > 0).float()], 1) * 30.0 - 15.0
    perfect = perfect.detach().requires_grad_(True)
    b = bl(perfect, tgt)
    b = float(b.detach())
    check("perfect prediction: blob loss ~ 0", b < 0.02, f"= {b:.5f}")
    empty_t = torch.zeros_like(tgt)
    a = LesionFreeFPLoss()(torch.stack([torch.full_like(tgt[:, 0].float(), 15.0),
                                        torch.full_like(tgt[:, 0].float(), -15.0)], 1),
                           empty_t)
    a = float(a.detach())
    check("empty prediction on empty patch: lesion-free ~ 0", a < 1e-4, f"= {a:.3e}")


def test_fg_prob_identity(device):
    print("1b. two-class shortcut == softmax")
    try:
        from .nnUNetTrainer_InteractiveV2 import _fg_prob
    except ImportError:
        from nnUNetTrainer_InteractiveV2 import _fg_prob  # type: ignore
    lg = torch.randn(2, 2, 8, 8, 8, device=device)
    a = _fg_prob(lg)
    b = torch.softmax(lg.float(), 1)[:, 1]
    check("sigmoid(l1-l0) == softmax[...,1]", float((a - b).abs().max()) < 1e-6,
          f"max|diff| = {float((a - b).abs().max()):.2e}")


def test_gradients(device):
    print("2. gradients")
    logits, tgt = make_patch(device=device)
    for name, fn in (("lesion-free(all_patches)", LesionFreeFPLoss(all_patches=True)),
                     ("blob", InstanceDiceLoss())):
        lg = logits.detach().clone().requires_grad_(True)
        v = fn(lg, tgt)
        v.backward()
        g = lg.grad
        check(f"{name}: grad finite", bool(torch.isfinite(g).all()))
        check(f"{name}: grad non-zero", float(g.abs().sum()) > 0,
              f"|g|1 = {float(g.abs().sum()):.4f}")

    # zero-label patch: blob loss must be exactly 0 with a valid (zero) gradient
    lg = torch.randn(2, 2, 16, 16, 16, device=device, requires_grad=True)
    zt = torch.zeros(2, 1, 16, 16, 16, dtype=torch.long, device=device)
    v = InstanceDiceLoss()(lg, zt)
    v.backward()
    check("zero-label patch: blob loss == 0", float(v) == 0.0)
    check("zero-label patch: blob grad is all-zero and finite",
          bool(torch.isfinite(lg.grad).all()) and float(lg.grad.abs().sum()) == 0.0)

    # ... while the lesion-free term does produce a gradient there
    lg = torch.randn(2, 2, 16, 16, 16, device=device, requires_grad=True)
    v = LesionFreeFPLoss()(lg, zt)
    v.backward()
    check("zero-label patch: lesion-free grad non-zero",
          float(lg.grad.abs().sum()) > 0, f"|g|1 = {float(lg.grad.abs().sum()):.4f}")


def test_dice_blind_spot(device):
    print("3. the blind spot the lesion-free term fills")
    from nnunetv2.training.loss.compound_losses import DC_and_CE_loss
    from nnunetv2.training.loss.dice import MemoryEfficientSoftDiceLoss

    dc_ce = DC_and_CE_loss({"batch_dice": True, "smooth": 0.0, "do_bg": False, "ddp": False},
                           {}, weight_ce=1, weight_dice=1,
                           dice_class=MemoryEfficientSoftDiceLoss)
    dice_only = DC_and_CE_loss({"batch_dice": True, "smooth": 0.0, "do_bg": False,
                                "ddp": False}, {}, weight_ce=0, weight_dice=1,
                               dice_class=MemoryEfficientSoftDiceLoss)
    zt = torch.zeros(2, 1, 16, 16, 16, dtype=torch.long, device=device)

    lg = torch.randn(2, 2, 16, 16, 16, device=device, requires_grad=True)
    v = dice_only(lg, zt)
    v.backward()
    gd = float(lg.grad.abs().sum())
    check("all-empty batch: Dice term gradient is exactly zero", gd == 0.0,
          f"|g|1 = {gd:.3e}, loss = {float(v):.3e}")

    # The regime that matters is not a random network but a fine-tuned one on a
    # lesion-free case: background almost everywhere, one small confident blob.
    # That is where the term has to beat cross entropy, and it does; in the
    # opposite (saturated) regime it deliberately does not, because s/(s+c)
    # flattens out once the prediction is already grossly wrong and the pooled
    # Dice / CE have taken over.
    def near_empty(n_fp_voxels, p_fp=0.9, shape=(2, 32, 32, 32)):
        lg = torch.full((shape[0], 2, *shape[1:]), 0.0, device=device)
        lg[:, 0] = 7.0                              # p_fg = 9e-4 everywhere
        lg[:, 1] = 0.0
        flat = lg.reshape(shape[0], 2, -1)
        logit_fp = float(np.log(p_fp / (1 - p_fp)))
        flat[:, 1, :n_fp_voxels] = logit_fp
        flat[:, 0, :n_fp_voxels] = 0.0
        return lg.detach().requires_grad_(True)

    zt2 = torch.zeros(2, 1, 32, 32, 32, dtype=torch.long, device=device)
    for n_fp in (50, 400):
        lg = near_empty(n_fp)
        dc_ce(lg, zt2).backward()
        g_ce = float(lg.grad.abs().sum())
        lg2 = near_empty(n_fp)
        LesionFreeFPLoss()(lg2, zt2).backward()
        g_lf = float(lg2.grad.abs().sum())
        check(f"near-empty patch, {n_fp} FP voxels: lesion-free signal > DC+CE",
              g_lf > g_ce, f"|g|1 DC+CE={g_ce:.3e} vs lesion-free={g_lf:.3e} "
                           f"(x{g_lf / max(g_ce, 1e-12):.1f})")


def test_matches_reference(device):
    print("4. vectorised instance Dice == literal masked loop")
    for seed in (0, 1, 2):
        logits, tgt = make_patch(lesions=((3, 4), (2, 6)), device=device, seed=seed)
        a = InstanceDiceLoss(smooth=1.0)(logits, tgt)
        b = reference_instance_dice(logits, tgt, smooth=1.0)
        assert_close(a, b, 1e-4, f"seed {seed}")
    check("matches the reference implementation to 1e-4", True)


def test_size_invariance(device):
    print("5. size invariance")
    shape = (1, 24, 32, 32)
    tgt = torch.zeros((1, 1, *shape[1:]), dtype=torch.long, device=device)
    tgt[0, 0, 4:16, 4:16, 4:16] = 1        # big lesion, 1728 voxels
    tgt[0, 0, 20:22, 26:28, 26:28] = 1     # small lesion, 8 voxels
    # prediction: the big lesion perfectly, the small one missed entirely
    logits = torch.full((1, 2, *shape[1:]), 0.0, device=device)
    logits[0, 1] = -15.0
    logits[0, 0] = 15.0
    logits[0, 1, 4:16, 4:16, 4:16] = 15.0
    logits[0, 0, 4:16, 4:16, 4:16] = -15.0
    logits = logits.requires_grad_(True)

    from nnunetv2.training.loss.dice import MemoryEfficientSoftDiceLoss
    from nnunetv2.utilities.helpers import softmax_helper_dim1
    pooled = MemoryEfficientSoftDiceLoss(apply_nonlin=softmax_helper_dim1, batch_dice=True,
                                         do_bg=False, smooth=0.0, ddp=False)
    pooled_loss = float(-pooled(logits, tgt))          # = pooled Dice
    blob = float(InstanceDiceLoss(smooth=1.0)(logits, tgt))
    check("pooled Dice barely notices the missed 8-voxel lesion",
          pooled_loss > 0.99, f"pooled Dice = {pooled_loss:.4f}")
    # the missed 8-voxel lesion scores dice = (0 + smooth)/(0 + 8 + smooth) = 0.111,
    # the big one ~1.0, so the mean loss is ~(1 - 0.111)/2
    check("instance Dice loses ~half its score to it",
          0.40 < blob < 0.55, f"blob loss = {blob:.4f} "
                              f"(pooled Dice loss = {1 - pooled_loss:.4f})")


def test_compound_and_ds(device):
    print("6. compound loss over a deep-supervision list")

    class Base(torch.nn.Module):
        def forward(self, out, tgt):
            return sum(o.float().mean() * 0.0 for o in out) + torch.tensor(
                0.5, device=out[0].device)

    logits, tgt = make_patch(device=device)
    small = torch.nn.functional.avg_pool3d(logits, 2)
    small_t = tgt[:, :, ::2, ::2, ::2]
    loss = InteractiveV2Loss(Base(), lesion_free=LesionFreeFPLoss(), w_lesion_free=1.0,
                             blob=InstanceDiceLoss(), w_blob=1.0)
    v = loss([logits, small], [tgt, small_t])
    v.backward()
    check("compound loss finite", bool(torch.isfinite(v)), f"= {float(v):.4f}")
    check("compound loss reports its terms", set(loss.last_terms) ==
          {"base", "lesion_free", "blob"}, str({k: round(x, 4)
                                                for k, x in loss.last_terms.items()}))
    check("compound loss gradient reaches the full-res head",
          float(logits.grad.abs().sum()) > 0)

    # both weights zero -> exactly the base loss, no extra work
    loss0 = InteractiveV2Loss(Base())
    v0 = loss0([logits, small], [tgt, small_t])
    assert_close(v0, 0.5, 1e-6, "w=0 passthrough")
    check("both weights zero == base loss", True)


def test_cost(device):
    print("7. cost on a realistic patch (2, 2, 112, 160, 128)")
    logits = torch.randn(2, 2, 112, 160, 128, device=device)
    tgt = torch.zeros(2, 1, 112, 160, 128, dtype=torch.long, device=device)
    rng = np.random.default_rng(0)
    for i in range(2):
        for _ in range(8):
            c = [int(rng.integers(6, s - 7)) for s in (112, 160, 128)]
            tgt[(i, 0) + tuple(slice(x - 4, x + 5) for x in c)] = 1
    for name, fn in (("lesion-free", LesionFreeFPLoss()), ("blob", InstanceDiceLoss())):
        lg = logits.clone().requires_grad_(True)
        fn(lg, tgt)  # warm up
        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(5):
            v = fn(lg, tgt)
            v.backward()
        if device == "cuda":
            torch.cuda.synchronize()
        print(f"  {name}: {(time.time() - t0) / 5 * 1000:.1f} ms per fwd+bwd")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cuda", action="store_true")
    a = ap.parse_args()
    device = "cuda" if a.cuda and torch.cuda.is_available() else "cpu"
    print(f"device = {device}\n")
    torch.manual_seed(0)
    test_finite_and_range(device)
    test_fg_prob_identity(device)
    test_gradients(device)
    test_dice_blind_spot(device)
    test_matches_reference(device)
    test_size_invariance(device)
    test_compound_and_ds(device)
    test_cost(device)
    print("\nall checks passed")


if __name__ == "__main__":
    main()
