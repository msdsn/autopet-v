"""Launch gate for B17 (``train.networks_eva.EVAFusionUNet``). Run on the GPU box.

    python -m train.test_networks_eva --plans <nnUNetPlans_interactive.json> \
        --checkpoint <B10 checkpoint_final.pth> --cuda --bench

Asserts, in order:

* the b17 plans build the class through ``pydoc.locate``, the way the container will;
* the EVA token volume has the shape the fusion assumes, ``(B, 768, Z, 16, 13)``;
* every tensor of the B10 checkpoint is consumed, and the tensors B17 adds are all
  under ``eva.`` / ``eva_fuse.``;
* the grafted network's logits equal B10's at every deep-supervision scale, with deep
  supervision on and off -- i.e. epoch 0 is B10;
* the pretrained weights survived ``network.apply(network.initialize)``.

``--bench`` adds forward+backward time and peak VRAM at the plans batch size, which is
what decides between the 120- and the 80-epoch schedule.
"""

from __future__ import annotations

import argparse
import json
import time

import torch

try:
    from .make_arch_plans import build_plans
    from .networks import graft_state_dict
except ImportError:  # flat import
    from make_arch_plans import build_plans  # type: ignore
    from networks import graft_state_dict  # type: ignore

from nnunetv2.utilities.get_network_from_plans import get_network_from_plans

NEW_PREFIXES = ("eva.", "eva_fuse.")


def build(arch: dict, in_ch: int, n_classes: int, deep_supervision: bool) -> torch.nn.Module:
    return get_network_from_plans(arch["network_class_name"], arch["arch_kwargs"],
                                  arch["_kw_requires_import"], in_ch, n_classes,
                                  allow_init=True, deep_supervision=deep_supervision)


def n_params(m: torch.nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())


def max_abs_diff(a, b) -> float:
    if isinstance(a, (list, tuple)):
        return max(max_abs_diff(x, y) for x, y in zip(a, b))
    return (a - b).abs().max().item()


def bench(net: torch.nn.Module, x: torch.Tensor, device, label: str, iters: int = 6) -> float:
    net = net.to(device).train()
    opt = torch.optim.SGD([p for p in net.parameters() if p.requires_grad], lr=0.0)
    torch.cuda.reset_peak_memory_stats(device)
    scaler = torch.amp.GradScaler("cuda")
    t0 = time.perf_counter()
    for i in range(iters + 2):
        if i == 2:
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
    print(f"  {label:<34} {dt * 1000:8.1f} ms/step   peak {peak:5.2f} GiB   "
          f"-> {dt * 250:6.1f} s / 250-step epoch")
    return dt


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--plans", required=True, help="nnUNetPlans_interactive.json")
    ap.add_argument("--configuration", default="3d_fullres")
    ap.add_argument("--checkpoint", default=None, help="B10 checkpoint_final.pth")
    ap.add_argument("--cuda", action="store_true")
    ap.add_argument("--bench", action="store_true")
    ap.add_argument("--bench-baseline", action="store_true", help="also time stock B10")
    ap.add_argument("--patch", nargs=3, type=int, default=None)
    ap.add_argument("--batch", type=int, default=None)
    ap.add_argument("--tol", type=float, default=1e-4)
    ap.add_argument("--z-stride", type=int, default=1)
    ap.add_argument("--chunk", type=int, default=0)
    ap.add_argument("--freeze-blocks", type=int, default=4)
    ap.add_argument("--fuse-stages", nargs="+", type=int, default=None)
    args = ap.parse_args()

    torch.manual_seed(0)
    device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")
    with open(args.plans) as f:
        base_plans = json.load(f)
    cfg = base_plans["configurations"][args.configuration]
    base_arch = cfg["architecture"]
    in_ch = len(cfg["normalization_schemes"])
    patch = list(args.patch or cfg["patch_size"])
    batch = int(args.batch or cfg["batch_size"])
    n_classes = 2

    print(f"plans      {args.plans}")
    print(f"patch      {patch}   batch {batch}   input channels {in_ch}   device {device}")

    base = build(base_arch, in_ch, n_classes, True)
    if args.checkpoint:
        ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        sd = ck["network_weights"]
        base.load_state_dict(sd, strict=True)
        print(f"source     {args.checkpoint} (trainer {ck.get('trainer_name')}, "
              f"epoch {ck.get('current_epoch')})")
    else:
        sd = base.state_dict()
        print("source     random weights (no --checkpoint given)")
    n_base = n_params(base)
    print(f"baseline   {n_base / 1e6:.2f} M parameters")

    x = torch.randn(batch, in_ch, *patch)
    x[:, 2:4] = torch.rand(batch, 2, *patch).clamp_(0, 1)
    x[:, 4] = (torch.rand(batch, *patch) > 0.98).float()

    base = base.to(device).eval()
    with torch.no_grad():
        ref = base(x.to(device))
    print(f"deep supervision outputs: {[tuple(o.shape) for o in ref]}")
    base.to("cpu")

    print("\n=== B17 ===")
    plans = build_plans(base_plans, "b17", "nnUNetPlans_b17", args.configuration,
                        eva_z_stride=args.z_stride, eva_chunk=args.chunk,
                        eva_freeze_blocks=args.freeze_blocks,
                        eva_fuse_stages=args.fuse_stages)
    arch = plans["configurations"][args.configuration]["architecture"]
    print(f"  network_class_name  {arch['network_class_name']}")
    print(f"  arch_kwargs (eva)   "
          f"{ {k: v for k, v in arch['arch_kwargs'].items() if k.startswith('eva_')} }")
    net = build(arch, in_ch, n_classes, True)

    # the pretrained weights must have survived network.apply(network.initialize)
    import timm
    fresh = timm.create_model(arch["arch_kwargs"].get("eva_model_name",
                              "eva02_base_patch14_448.mim_in22k_ft_in22k_in1k"),
                              pretrained=True, num_classes=0, dynamic_img_size=True)
    fsd, nsd = fresh.state_dict(), net.eva.state_dict()
    dmax = max((nsd[k].float() - v.float()).abs().max().item() for k, v in fsd.items()
               if k in nsd and v.dtype.is_floating_point)
    print(f"  pretrained preserved  max |diff| vs a fresh timm model {dmax:.3e} "
          f"({'OK' if dmax == 0.0 else 'FAIL -- initialize() overwrote EVA'})")
    assert dmax == 0.0, "network.apply(initialize) clobbered the pretrained EVA weights"
    del fresh, fsd

    n_eva = n_params(net.eva)
    n_eva_train = sum(p.numel() for p in net.eva.parameters() if p.requires_grad)
    n_fuse = sum(p.numel() for p in net.eva_fuse.parameters())
    print(f"  parameters          {n_params(net) / 1e6:.2f} M total "
          f"(+{(n_params(net) - n_base) / 1e6:.2f} M over B10)")
    print(f"    EVA-02-B          {n_eva / 1e6:.2f} M "
          f"({n_eva_train / 1e6:.2f} M trainable, {(n_eva - n_eva_train) / 1e6:.2f} M frozen)")
    print(f"    fusion 1x1x1      {n_fuse / 1e6:.3f} M "
          f"(stages {net.eva_fuse_stages}, zero-init)")

    missing, unexpected = graft_state_dict(net, sd, verbose=False)
    assert not unexpected, f"source tensors not consumed: {unexpected[:5]}"
    stray = [k for k in missing if not any(k.startswith(p) for p in NEW_PREFIXES)]
    assert not stray, f"undeclared new tensors {stray[:5]}"
    print(f"  grafted             {len(sd)} source tensors, {len(missing)} new")

    net = net.to(device).eval()
    with torch.no_grad():
        tok = net.eva_tokens(x.to(device))
    zt = -(-patch[0] // args.z_stride)
    want = (batch, net.eva_dim, zt, *net.eva_token_grid)
    print(f"  token volume        {tuple(tok.shape)}  expected {want}  "
          f"({'OK' if tuple(tok.shape) == want else 'FAIL'})")
    assert tuple(tok.shape) == want
    del tok
    torch.cuda.empty_cache() if device.type == "cuda" else None

    with torch.no_grad():
        out = net(x.to(device))
    assert len(out) == len(ref), f"{len(out)} outputs vs {len(ref)}"
    for o, r in zip(out, ref):
        assert o.shape == r.shape, f"shape {o.shape} vs {r.shape}"
    d = max_abs_diff(out, ref)
    ok = d < args.tol
    print(f"  max |logit diff|    {d:.3e}  ({'OK' if ok else 'FAIL'}, tol {args.tol:g})")
    failures = [] if ok else [f"epoch-0 equivalence failed, max diff {d:.3e}"]
    del out
    torch.cuda.empty_cache() if device.type == "cuda" else None

    net.decoder.deep_supervision = False
    base_ds = build(base_arch, in_ch, n_classes, False)
    base_ds.load_state_dict(sd, strict=True)
    base_ds = base_ds.to(device).eval()
    with torch.no_grad():
        d2 = (net(x.to(device)) - base_ds(x.to(device))).abs().max().item()
    print(f"  max |diff| (ds off) {d2:.3e}  ({'OK' if d2 < args.tol else 'FAIL'})")
    if d2 >= args.tol:
        failures.append(f"inference-path equivalence failed, max diff {d2:.3e}")
    net.decoder.deep_supervision = True
    base_ds.to("cpu")
    del base_ds
    torch.cuda.empty_cache() if device.type == "cuda" else None

    if args.bench and device.type == "cuda":
        print("\n=== forward + backward, plans batch ===")
        if args.bench_baseline:
            bench(base, x.to(device), device, "B10 PlainConvUNet")
        dt = bench(net, x.to(device), device, "B17 EVAFusionUNet")
        print(f"  -> 120 epochs at 250 steps: {dt * 250 * 120 / 3600:.2f} GPU-h "
              f"(GPU time only, the run is also dataloader-bound)")

    print()
    if failures:
        for f in failures:
            print(f"FAILED: {f}")
        raise SystemExit(1)
    print("all B17 gate checks passed")


if __name__ == "__main__":
    main()
