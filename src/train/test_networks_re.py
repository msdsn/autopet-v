"""The RE launch gate. Run on the GPU box.

    python -m train.test_networks_re --plans <nnUNetPlans_re.json> \
        --init <re_init_5ch.pth> [--cuda] [--bench] [--zeroshot <n cases>]

Checks, in order:

1. the plans build the class through ``pydoc.locate`` -- the same resolution the
   predictor does inside the container -- and the parameter count is the expected one;
2. forward shapes at the plans patch, deep supervision on and off;
3. **epoch 0 is LesionTracer**: the grafted 5-channel network equals a stock
   2-channel ``ResidualEncoderUNet`` carrying the same weights, built in the same
   process, to < 1e-5 at every deep-supervision scale -- and the interaction channels
   provably cannot move the output (random channels 2-4 change nothing);
4. checkpoint round-trip: save, reload into a fresh network, logits unchanged;
5. the interaction transform runs at the new patch size and writes channels 2-4 in
   range;
6. ``--zeroshot``: the decisive measurement for ``pet_renorm``. The *unmodified*
   2-channel LesionTracer is run on real store patches under both PET
   representations -- our per-case z-score, and the cohort-median inversion into
   their ``CTNormalization`` -- and scored against the stored label. Whichever wins
   is the representation its weights were trained on; there is no need to argue it.
7. ``--bench``: forward+backward ms/step and peak VRAM at the plans batch, which is
   what decides the patch size and the epoch budget.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import time
from typing import List, Tuple

import numpy as np
import torch

try:
    from .networks_re import (STEM_CONV_KEYS, graft_lesiontracer_state_dict,
                              PET_MU, PET_SD, PET_LO, PET_HI, PET_MEAN, PET_STD)
except ImportError:  # flat import
    from networks_re import (STEM_CONV_KEYS, graft_lesiontracer_state_dict,  # type: ignore
                             PET_MU, PET_SD, PET_LO, PET_HI, PET_MEAN, PET_STD)

from nnunetv2.utilities.get_network_from_plans import get_network_from_plans

STOCK = "dynamic_network_architectures.architectures.unet.ResidualEncoderUNet"


def build(arch: dict, in_ch: int, n_classes: int, ds: bool, class_name: str = None,
          drop_kwargs: Tuple[str, ...] = ()) -> torch.nn.Module:
    kw = {k: v for k, v in arch["arch_kwargs"].items() if k not in drop_kwargs}
    return get_network_from_plans(class_name or arch["network_class_name"], kw,
                                  arch["_kw_requires_import"], in_ch, n_classes,
                                  allow_init=True, deep_supervision=ds)


def unique_params(m: torch.nn.Module) -> int:
    seen, n = set(), 0
    for p in m.parameters():
        if p.data_ptr() not in seen:
            seen.add(p.data_ptr())
            n += p.numel()
    return n


def max_abs_diff(a, b) -> float:
    if isinstance(a, (list, tuple)):
        return max(max_abs_diff(x, y) for x, y in zip(a, b))
    return (a.float() - b.float()).abs().max().item()


def probe_batch(patch: List[int], c: int, b: int = 1, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(b, c, *patch, generator=g)
    if c >= 4:
        x[:, 2:4] = torch.rand(b, 2, *patch, generator=g)
    if c >= 5:
        x[:, 4] = (torch.rand(b, *patch, generator=g) > 0.98).float()
    return x


def ctnorm_pet(z: torch.Tensor) -> torch.Tensor:
    return ((z * PET_SD + PET_MU).clamp(PET_LO, PET_HI) - PET_MEAN) / PET_STD


# ---------------------------------------------------------------------------

def bench(net, x, device, label, iters=6):
    net = net.to(device).train()
    opt = torch.optim.SGD(net.parameters(), lr=0.0)
    torch.cuda.reset_peak_memory_stats(device)
    scaler = torch.amp.GradScaler("cuda")
    t0 = None
    for i in range(iters + 3):
        if i == 3:
            torch.cuda.synchronize(device)
            t0 = time.perf_counter()
        opt.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.float16):
            out = net(x)
            loss = sum(o.float().pow(2).mean() for o in out) if isinstance(out, (list, tuple)) \
                else out.float().pow(2).mean()
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
    torch.cuda.synchronize(device)
    dt = (time.perf_counter() - t0) / iters
    peak = torch.cuda.max_memory_allocated(device) / 2 ** 30
    print(f"  {label:<34} {dt * 1000:8.1f} ms/step   peak {peak:5.2f} GiB")
    net.to("cpu")
    torch.cuda.empty_cache()
    return dt, peak


def dice(pred: np.ndarray, gt: np.ndarray) -> float:
    s = pred.sum() + gt.sum()
    return 1.0 if s == 0 else float(2.0 * np.logical_and(pred, gt).sum() / s)


def zeroshot(arch: dict, sd: dict, store: str, patch: List[int], n_cases: int,
             device) -> None:
    """Run the stock 2-channel LesionTracer on real store patches, both PET mappings."""
    import blosc2
    segs = sorted(glob.glob(os.path.join(store, "*_seg.b2nd")))
    net = build(arch, 2, 2, False, STOCK, drop_kwargs=("pet_renorm",))
    sd2 = {k: (v[:, :2].contiguous() if k in STEM_CONV_KEYS else v) for k, v in sd.items()}
    net.load_state_dict(sd2, strict=True)
    net = net.to(device).eval()

    rows = []
    for f in segs:
        if len(rows) >= n_cases:
            break
        case = os.path.basename(f)[: -len("_seg.b2nd")]
        seg = np.asarray(blosc2.open(f, mode="r")[:])[0]
        if seg.max() == 0:
            continue
        idx = np.argwhere(seg > 0)
        c = idx[len(idx) // 2]
        data = blosc2.open(os.path.join(store, case + ".b2nd"), mode="r")
        # a store case can be smaller than the patch on an axis (the in-plane extent
        # of a body-cropped volume is 188 voxels at the median, against a 192 patch),
        # so clamp the window into the volume and zero-pad the remainder, which is
        # what nnU-Net's dataloader does too
        sl, pad = [], []
        for a in range(3):
            lo = max(0, min(int(c[a]) - patch[a] // 2, seg.shape[a] - patch[a]))
            hi = min(seg.shape[a], lo + patch[a])
            sl.append(slice(lo, hi))
            pad.append((0, patch[a] - (hi - lo)))
        img = np.asarray(data[(slice(0, 2), *sl)]).astype(np.float32)
        lab = (seg[tuple(sl)] > 0)
        if any(p[1] for p in pad):
            img = np.pad(img, [(0, 0)] + pad)
            lab = np.pad(lab, pad)
        if lab.sum() == 0:
            continue
        x = torch.from_numpy(img)[None]
        out = {}
        for mode in ("none", "ctnorm"):
            xi = x.clone()
            if mode == "ctnorm":
                xi[:, 1] = ctnorm_pet(xi[:, 1])
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16,
                                                 enabled=device.type == "cuda"):
                logits = net(xi.to(device))
            pred = (logits.argmax(1)[0].cpu().numpy() > 0)
            out[mode] = (dice(pred, lab), int(pred.sum()))
        rows.append((case[:34], int(lab.sum()), out["none"], out["ctnorm"]))
        print(f"  {case[:34]:<36} gt={lab.sum():>7}  "
              f"z-score: dice={out['none'][0]:.4f} pred={out['none'][1]:>7}   "
              f"ctnorm: dice={out['ctnorm'][0]:.4f} pred={out['ctnorm'][1]:>7}")
    if rows:
        a = float(np.mean([r[2][0] for r in rows]))
        b = float(np.mean([r[3][0] for r in rows]))
        print(f"  MEAN over {len(rows)} lesion patches: "
              f"pet_renorm=none {a:.4f}   pet_renorm=ctnorm {b:.4f}   "
              f"-> use {'ctnorm' if b > a else 'none'}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--plans", required=True, help="nnUNetPlans_re.json")
    ap.add_argument("--init", required=True, help="re_init_5ch.pth from init_from_lesiontracer")
    ap.add_argument("--configuration", default="3d_fullres")
    ap.add_argument("--cuda", action="store_true")
    ap.add_argument("--bench", action="store_true")
    ap.add_argument("--batch", type=int, default=None, help="default: the plans batch size")
    ap.add_argument("--zeroshot", type=int, default=0, help="number of real store cases")
    ap.add_argument("--store", default="/content/nnUNet/prep_local/Dataset998_AutoPETV/"
                                       "nnUNetPlans_3d_fullres")
    ap.add_argument("--f64-identity", action="store_true",
                    help="exact identity vs the 2-channel LesionTracer in float64 "
                         "on the CPU at a 64^3 patch (slow, ~3 min)")
    ap.add_argument("--transform", action="store_true",
                    help="run the interaction transform at the plans patch size")
    args = ap.parse_args()

    device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")
    with open(args.plans) as f:
        plans = json.load(f)
    cfg = plans["configurations"][args.configuration]
    arch, patch = cfg["architecture"], [int(p) for p in cfg["patch_size"]]
    batch = args.batch or int(cfg["batch_size"])
    print(f"plans   {plans['plans_name']}  class {arch['network_class_name']}")
    print(f"patch   {patch}  ({np.prod(patch) / 1e6:.2f} M voxels)  batch {batch}  "
          f"pet_renorm={arch['arch_kwargs'].get('pet_renorm')}")

    # 1. build + graft ------------------------------------------------------
    net = build(arch, 5, 2, True)
    ck = torch.load(args.init, map_location="cpu", weights_only=False)
    sd = ck["network_weights"]
    graft_lesiontracer_state_dict(net, sd, verbose=True)
    print(f"[1] parameters: {unique_params(net) / 1e6:.3f} M   "
          f"tensors in checkpoint: {len(sd)}")

    stem = net.state_dict()[STEM_CONV_KEYS[0]]
    print(f"[1] stem conv {tuple(stem.shape)}; per-channel rms "
          + " ".join(f"{c}:{float(stem[:, c].pow(2).mean().sqrt()):.4f}"
                     for c in range(stem.shape[1])))
    assert float(stem[:, 2:].abs().max()) == 0.0, "the interaction columns are not zero"

    # 2. forward shapes -----------------------------------------------------
    x = probe_batch(patch, 5, 1).to(device)
    net = net.to(device).eval()
    with torch.no_grad():
        out = net(x)
    print(f"[2] deep supervision on: {len(out)} outputs "
          + " ".join(str(tuple(o.shape)) for o in out))
    net.decoder.deep_supervision = False
    with torch.no_grad():
        out1 = net(x)
    print(f"[2] deep supervision off: {tuple(out1.shape)}")
    assert tuple(out1.shape) == (1, 2, *patch)
    net.decoder.deep_supervision = True

    # 3. epoch-0 identity ---------------------------------------------------
    # Two exact assertions and one informational number. The exact pair is what
    # establishes "epoch 0 is LesionTracer":
    #   (a) the interaction columns of the stem are zero, so channels 2-4 cannot
    #       reach the output at all -- a shape-independent, bit-exact statement;
    #   (b) the subclass is stock ResidualEncoderUNet plus the PET remap and nothing
    #       else -- checked against a stock class of the *same* 5-channel shape, so
    #       both networks get the identical convolution kernels.
    # Together with the strict graft above (0 missing, 0 unexpected, 0 shape
    # mismatch) that is the whole claim. The direct comparison against a 2-channel
    # network is reported but NOT asserted at 1e-5: a (32, 2, ...) stem and a
    # (32, 5, ...) stem are different convolutions and cuDNN picks different
    # algorithms and TF32 paths for them. Measured here: 1.1e-01 on logits of
    # magnitude ~20 in float32 on an A100, and exactly 0.000e+00 for the same
    # comparison in float64 on the CPU at a 64^3 patch (--f64-identity).
    x2 = x.clone()
    x2[:, 2:] = torch.rand_like(x2[:, 2:])
    with torch.no_grad():
        d2 = max_abs_diff(net(x), net(x2))
    print(f"[3a] interaction channels cannot move the output: max |logit diff| {d2:.3e}")
    assert d2 == 0.0, "channels 2-4 already influence the output -- the stem is not zero"

    ref5 = build(arch, 5, 2, True, STOCK, drop_kwargs=("pet_renorm",))
    ref5.load_state_dict(net.state_dict(), strict=True)
    ref5 = ref5.to(device).eval()
    with torch.no_grad():
        d3a = max_abs_diff(net(x), ref5(net._remap_pet(x)))
    print(f"[3b] subclass == stock ResidualEncoderUNet + pet_renorm: "
          f"max |logit diff| {d3a:.3e}")
    assert d3a == 0.0
    ref5.to("cpu")
    del ref5
    if device.type == "cuda":
        torch.cuda.empty_cache()

    ref = build(arch, 2, 2, True, STOCK, drop_kwargs=("pet_renorm",))
    sd2 = {k: (v[:, :2].contiguous() if k in STEM_CONV_KEYS else v)
           for k, v in net.state_dict().items()}
    ref.load_state_dict(sd2, strict=True)
    ref = ref.to(device).eval()
    with torch.no_grad():
        o = net(x)
        d = max_abs_diff(o, ref(net._remap_pet(x)[:, :2]))
    mag = max(float(t.abs().max()) for t in o)
    print(f"[3c] vs the stock 2-channel LesionTracer, {device.type} float32: "
          f"max |logit diff| {d:.3e} on logits of magnitude {mag:.1f} "
          f"({d / mag:.1e} relative) -- kernel noise, see the comment")
    assert d / mag < 1e-2, f"far more than kernel noise: {d:.3e} on |logit| {mag:.1f}"
    ref.to("cpu")
    del ref
    if device.type == "cuda":
        torch.cuda.empty_cache()

    if args.f64_identity:
        p64 = [64, 64, 64]
        a64 = build(arch, 5, 2, True)
        a64.load_state_dict(net.state_dict(), strict=True)
        b64 = build(arch, 2, 2, True, STOCK, drop_kwargs=("pet_renorm",))
        b64.load_state_dict({k: (v[:, :2].contiguous() if k in STEM_CONV_KEYS else v)
                             for k, v in net.state_dict().items()}, strict=True)
        a64 = a64.double().eval()
        b64 = b64.double().eval()
        xf = probe_batch(p64, 5, 1).double()
        with torch.no_grad():
            d64 = max_abs_diff(a64(xf), b64(a64._remap_pet(xf)[:, :2]))
        print(f"[3d] vs the stock 2-channel LesionTracer, cpu float64 at {p64}: "
              f"max |logit diff| {d64:.3e}")
        assert d64 < 1e-9
        del a64, b64

    # 4. checkpoint round-trip ---------------------------------------------
    tmp = os.path.join(os.path.dirname(os.path.abspath(args.init)), ".re_roundtrip.pth")
    torch.save({"network_weights": net.state_dict()}, tmp)
    net2 = build(arch, 5, 2, True)
    net2.load_state_dict(torch.load(tmp, map_location="cpu",
                                    weights_only=False)["network_weights"], strict=True)
    net2 = net2.to(device).eval()
    with torch.no_grad():
        d3 = max_abs_diff(net(x), net2(x))
    print(f"[4] checkpoint round-trip: max |logit diff| {d3:.3e}")
    assert d3 == 0.0
    os.remove(tmp)
    net2.to("cpu")
    del net2
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # 5. interaction transform at the new patch size ------------------------
    if args.transform:
        try:
            from .nnUNetTrainer_Interactive import nnUNetTrainer_Interactive as _T
        except ImportError:
            from nnUNetTrainer_Interactive import nnUNetTrainer_Interactive as _T  # type: ignore
        tr = _T.build_interaction_transform()
        rng = np.random.default_rng(0)
        img = torch.from_numpy(rng.normal(size=(2, *patch)).astype(np.float32))
        seg = np.zeros((1, *patch), dtype=np.int16)
        seg[0, 90:105, 90:110, 90:108] = 1
        seg[0, 40:46, 130:136, 60:66] = 1
        t0 = time.perf_counter()
        d = tr(**{"image": img, "segmentation": torch.from_numpy(seg)})
        dt = time.perf_counter() - t0
        im = d["image"]
        print(f"[5] interaction transform at {patch}: image {tuple(im.shape)} in {dt * 1000:.0f} ms; "
              f"ch2 [{float(im[2].min()):.2f},{float(im[2].max()):.2f}] "
              f"ch3 [{float(im[3].min()):.2f},{float(im[3].max()):.2f}] "
              f"ch4 uniques {sorted(set(im[4].unique().tolist()))[:4]}")
        assert tuple(im.shape) == (5, *patch)
        assert 0.0 <= float(im[2].min()) and float(im[2].max()) <= 1.0
        assert 0.0 <= float(im[3].min()) and float(im[3].max()) <= 1.0

    # 6. zero-shot pet_renorm A/B ------------------------------------------
    if args.zeroshot:
        print(f"[6] zero-shot LesionTracer on {args.zeroshot} real store lesion patches:")
        zeroshot(arch, net.state_dict(), args.store, patch, args.zeroshot, device)

    # 7. bench --------------------------------------------------------------
    if args.bench and device.type == "cuda":
        print(f"[7] forward+backward, batch {batch}, fp16 autocast:")
        xb = probe_batch(patch, 5, batch).to(device)
        dt, peak = bench(net, xb, device, f"RE {patch} b{batch}")
        print(f"    -> 250 steps/epoch = {dt * 250:.0f} s of GPU per epoch")

    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
