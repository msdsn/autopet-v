"""Checks for RE-N. Run on the GPU box.

    python -m train.test_networks_ren --plans <prep>/nnUNetPlans_ren.json \
        [--checkpoint <RE40 checkpoint_final.pth>] [--cuda] [--patch 64 64 64]

Asserts:

* the plans build the class through ``pydoc.locate``, i.e. the dotted path resolves
  the way it will inside the container;
* the gate head is **exactly zero after** ``network.apply(network.initialize)`` --
  the regression nnU-Net's post-construction re-initialisation would cause;
* every source tensor is consumed and the only new ones are ``gate_head.*``;
* the logits equal a stock ``ResidualEncoderUNet`` of the same shape carrying the same
  weights, on the pre-remapped input, at **exactly 0.0** -- deep supervision on and
  off, so the inference path is covered too;
* deep supervision on returns ``[seg_0 ... seg_n, gate]`` with the coarse map last and
  at the stage-3 grid.
"""

from __future__ import annotations

import argparse
import json

import torch

from nnunetv2.utilities.get_network_from_plans import get_network_from_plans

try:
    from .networks_n1 import _FREEZE_FLAG
    from .networks_ren import GATE_PREFIX
except ImportError:  # flat import
    from networks_n1 import _FREEZE_FLAG  # type: ignore
    from networks_ren import GATE_PREFIX  # type: ignore

STOCK = "dynamic_network_architectures.architectures.unet.ResidualEncoderUNet"


def build(arch: dict, in_ch: int, n_classes: int, ds: bool, class_name: str = None):
    kw = dict(arch["arch_kwargs"])
    if class_name is not None:
        for k in ("pet_renorm", "pet_channel", "gate_stage", "gate_class"):
            kw.pop(k, None)
    return get_network_from_plans(class_name or arch["network_class_name"], kw,
                                  arch["_kw_requires_import"], in_ch, n_classes,
                                  allow_init=True, deep_supervision=ds)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--plans", required=True)
    ap.add_argument("--base-plans", default=None, help="nnUNetPlans_re.json (for the graft)")
    ap.add_argument("--configuration", default="3d_fullres")
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--cuda", action="store_true")
    ap.add_argument("--patch", nargs=3, type=int, default=[64, 64, 64])
    args = ap.parse_args()

    torch.manual_seed(0)
    device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")
    with open(args.plans) as f:
        plans = json.load(f)
    cfg = plans["configurations"][args.configuration]
    arch = cfg["architecture"]
    in_ch = len(cfg["normalization_schemes"])
    patch = list(args.patch)
    print(f"plans   {args.plans}")
    print(f"class   {arch['network_class_name']}")
    print(f"patch   {patch}  in_ch {in_ch}  gate_stage {arch['arch_kwargs'].get('gate_stage')}")

    net = build(arch, in_ch, 2, True)
    failures = []

    # 1. zero after apply(initialize) -- get_network_from_plans already applied it
    head = net.gate_head
    w = head.weight.detach().abs().max().item()
    b = head.bias.detach().abs().max().item()
    print(f"  gate_head zero after apply(initialize): |w|max {w:.1e} |b|max {b:.1e} "
          f"(freeze flag {getattr(head, _FREEZE_FLAG, False)})")
    if w != 0.0 or b != 0.0:
        failures.append("gate head is not zero after apply(initialize)")

    # 2. graft
    if args.checkpoint:
        ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        sd = {k: v for k, v in ck["network_weights"].items()
              if not k.startswith("decoder.organ_seg_layers.")}
        missing, unexpected = net.load_state_dict(sd, strict=False)
        stray = [k for k in missing if not k.startswith(GATE_PREFIX)]
        print(f"  graft: {len(sd)} tensors, new={sorted(missing)}, unexpected={list(unexpected)[:4]}")
        if stray or unexpected:
            failures.append(f"graft not clean: stray={stray[:4]} unexpected={list(unexpected)[:4]}")
    else:
        print("  graft: skipped (no --checkpoint), weights are random")

    n_params = sum(p.numel() for p in net.parameters())
    n_gate = sum(p.numel() for n, p in net.named_parameters() if n.startswith(GATE_PREFIX))
    print(f"  parameters {n_params/1e6:.3f} M, of which gate {n_gate}")

    x = torch.randn(1, in_ch, *patch)
    x[:, 2:4] = torch.rand(1, 2, *patch)
    x[:, 4] = (torch.rand(1, *patch) > 0.98).float()
    net = net.to(device).eval()
    xd = x.to(device)

    for ds in (True, False):
        net.decoder.deep_supervision = ds
        ref = build(arch, in_ch, 2, ds, class_name=STOCK)
        miss, unexp = ref.load_state_dict(net.state_dict(), strict=False)
        gate_only = sorted(k for k in unexp if k.startswith(GATE_PREFIX))
        if miss or sorted(unexp) != gate_only:
            failures.append(f"ds={ds}: stock ref did not accept the weights")
        ref = ref.to(device).eval()
        with torch.no_grad():
            out = net(xd)
            r = ref(net._remap_pet(xd))
        if ds:
            if not isinstance(out, (list, tuple)) or len(out) != len(r) + 1:
                failures.append(f"ds=True: expected {len(r)}+1 outputs, got "
                                f"{len(out) if isinstance(out,(list,tuple)) else 1}")
            gate = out[-1]
            print(f"  ds on : {len(out)} outputs, seg {tuple(out[0].shape)}, "
                  f"gate {tuple(gate.shape)} (last)")
            d = max((p.float() - q.float()).abs().max().item() for p, q in zip(out, r))
        else:
            if isinstance(out, (list, tuple)):
                failures.append("ds=False: expected a single tensor (the inference contract)")
            print(f"  ds off: single tensor {tuple(out.shape)}")
            d = (out.float() - r.float()).abs().max().item()
        print(f"          max |logit diff| vs stock {d:.3e}  ({'OK' if d == 0.0 else 'FAIL'})")
        if d != 0.0:
            failures.append(f"ds={ds}: not bit-exact against stock ({d:.3e})")
        ref.to("cpu")
        del ref
    net.decoder.deep_supervision = True

    print()
    if failures:
        for f_ in failures:
            print("FAIL:", f_)
        raise SystemExit(1)
    print("all checks passed")


if __name__ == "__main__":
    main()
