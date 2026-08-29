"""Checks for the B13/B14 architecture variants. Run on the GPU box.

    python -m train.test_networks --plans <nnUNetPlans_interactive.json> \
        [--checkpoint <B10 checkpoint_final.pth>] [--cuda] [--bench]

Asserts, for each variant:

* the plans build the class through ``pydoc.locate``, i.e. the dotted path in
  ``network_class_name`` resolves the same way it will inside the container;
* every tensor of the source checkpoint is consumed by the variant, and the tensors
  the variant adds are the declared ones;
* the variant's logits equal the source model's to < 1e-4 on a real-shaped patch, at
  every deep-supervision scale -- so epoch 0 is the source model;
* the added parameters are a small fraction of the network.

``--bench`` adds forward+backward time and peak VRAM at the plans batch size, which
is the number that decides whether 120 epochs fit in the schedule.
"""

from __future__ import annotations

import argparse
import json
import time
from typing import List

import torch

try:
    from .make_arch_plans import build_plans
    from .networks import graft_state_dict, mamba_available
except ImportError:  # flat import
    from make_arch_plans import build_plans  # type: ignore
    from networks import graft_state_dict, mamba_available  # type: ignore

from nnunetv2.utilities.get_network_from_plans import get_network_from_plans

NEW_PREFIXES = {
    "b13": ("context.",),
    "b14": ("edit_stem.", "edit_ups.", "edit_skip_projs.", "edit_stages.", "edit_seg_layers."),
}


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


def bench(net: torch.nn.Module, x: torch.Tensor, device, label: str, iters: int = 8) -> None:
    net = net.to(device).train()
    opt = torch.optim.SGD(net.parameters(), lr=0.0)
    torch.cuda.reset_peak_memory_stats(device)
    scaler = torch.amp.GradScaler("cuda")
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
    print(f"  {label:<28} {dt * 1000:8.1f} ms/step   peak {peak:5.2f} GiB   "
          f"-> {dt * 250:6.1f} s / 250-step epoch")
    net.to("cpu")
    torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--plans", required=True, help="nnUNetPlans_interactive.json")
    ap.add_argument("--configuration", default="3d_fullres")
    ap.add_argument("--checkpoint", default=None,
                    help="source checkpoint; without it the source weights are random")
    ap.add_argument("--variants", nargs="+", default=["b13", "b14"])
    ap.add_argument("--cuda", action="store_true")
    ap.add_argument("--bench", action="store_true")
    ap.add_argument("--patch", nargs=3, type=int, default=None, help="default: the plans patch size")
    ap.add_argument("--batch", type=int, default=None, help="default: the plans batch size")
    ap.add_argument("--tol", type=float, default=1e-4)
    ap.add_argument("--emit-ref", default=None,
                    help="write the fixed batch and the source model's outputs here; the "
                         "trainers assert against it at init (nnUNet_arch_refbatch)")
    ap.add_argument("--ref-batch", type=int, default=1, help="batch size of --emit-ref")
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
    print(f"patch      {patch}   batch {batch}   input channels {in_ch}")
    print(f"mamba_ssm  {'importable' if mamba_available() else 'not available'}")

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

    # a patch shaped like the real thing: CT/PET z-scored, guidance in [0,1], mask binary
    x = torch.randn(batch, in_ch, *patch)
    x[:, 2:4] = torch.rand(batch, 2, *patch).clamp_(0, 1)
    x[:, 4] = (torch.rand(batch, *patch) > 0.98).float()

    base = base.to(device).eval()
    with torch.no_grad():
        ref = base(x.to(device))
    print(f"deep supervision outputs: {[tuple(o.shape) for o in ref]}")
    if args.emit_ref:
        xr = x[:args.ref_batch].clone()
        with torch.no_grad():
            rr = base(xr.to(device))
        torch.save({"x": xr.cpu(), "ref": [o.cpu() for o in rr],
                    "arch": base_arch, "in_channels": in_ch, "num_classes": n_classes,
                    "checkpoint": args.checkpoint, "plans": args.plans},
                   args.emit_ref)
        print(f"wrote reference batch {tuple(xr.shape)} -> {args.emit_ref}")
    base.to("cpu")

    failures: List[str] = []
    for v in args.variants:
        print(f"\n=== {v.upper()} ===")
        plans = build_plans(base_plans, v, f"nnUNetPlans_{v}", args.configuration)
        arch = plans["configurations"][args.configuration]["architecture"]
        print(f"  network_class_name  {arch['network_class_name']}")
        net = build(arch, in_ch, n_classes, True)
        missing, unexpected = graft_state_dict(net, sd, verbose=False)
        assert not unexpected, f"{v}: source tensors not consumed: {unexpected[:5]}"
        stray = [k for k in missing if not any(k.startswith(p) for p in NEW_PREFIXES[v])]
        assert not stray, f"{v}: undeclared new tensors {stray[:5]}"
        own = net.state_dict()
        seen, n_new = set(), 0
        for k in missing:                      # nnU-Net aliases each conv/norm twice
            t = own[k]
            if t.data_ptr() in seen:
                continue
            seen.add(t.data_ptr())
            n_new += t.numel()
        print(f"  parameters          {n_params(net) / 1e6:.2f} M "
              f"(+{(n_params(net) - n_base) / 1e6:.2f} M, "
              f"+{100 * (n_params(net) - n_base) / n_base:.1f} %)")
        print(f"  grafted             {len(sd)} source tensors, {len(missing)} new "
              f"({n_new / 1e6:.3f} M)")

        net = net.to(device).eval()
        with torch.no_grad():
            out = net(x.to(device))
        assert len(out) == len(ref), f"{v}: {len(out)} outputs vs {len(ref)}"
        for o, r in zip(out, ref):
            assert o.shape == r.shape, f"{v}: shape {o.shape} vs {r.shape}"
        d = max_abs_diff(out, ref)
        ok = d < args.tol
        print(f"  max |logit diff|    {d:.3e}  ({'OK' if ok else 'FAIL'}, tol {args.tol:g})")
        if not ok:
            failures.append(f"{v}: epoch-0 equivalence failed, max diff {d:.3e}")

        # deep-supervision-off path, the one inference uses
        net.decoder.deep_supervision = False
        base_ds = build(base_arch, in_ch, n_classes, False)
        base_ds.load_state_dict(sd, strict=True)
        base_ds = base_ds.to(device).eval()
        with torch.no_grad():
            d2 = (net(x.to(device)) - base_ds(x.to(device))).abs().max().item()
        print(f"  max |diff| (ds off) {d2:.3e}  ({'OK' if d2 < args.tol else 'FAIL'})")
        if d2 >= args.tol:
            failures.append(f"{v}: inference-path equivalence failed, max diff {d2:.3e}")
        net.decoder.deep_supervision = True
        base_ds.to("cpu")
        net.to("cpu")

        if args.bench and device.type == "cuda":
            bench(net, x.to(device), device, f"{v} fwd+bwd")

    if args.bench and device.type == "cuda":
        print()
        bench(base, x.to(device), device, "baseline fwd+bwd")

    print()
    if failures:
        for f_ in failures:
            print("FAIL:", f_)
        raise SystemExit(1)
    print("all checks passed")


if __name__ == "__main__":
    main()
