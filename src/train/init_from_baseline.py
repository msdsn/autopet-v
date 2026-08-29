"""Weight surgery: 4-channel baseline checkpoint -> 5-channel interactive checkpoint.

ch0/ch1 (CT, PET) and ch2/ch3 (the baseline's scribble heatmaps, scaled by
--guidance-scale) are copied over; ch4, the previous prediction, is zero-initialised.
Every alias of the first conv is patched: the tensor appears four times in the state
dict and nnU-Net's loader asserts on all of them.
"""

from __future__ import annotations

import argparse
import os
from typing import Dict

import numpy as np
import torch

FIRST_CONV_MARKER = "stages.0.0.convs.0."
FIRST_CONV_SUFFIXES = (".conv.weight", ".all_modules.0.weight")


def find_first_conv_keys(sd: Dict[str, torch.Tensor]):
    """Return every alias of the first conv's weight.

    dynamic_network_architectures registers each conv both as `.conv` and inside an
    `.all_modules` Sequential, and the decoder holds a reference to the encoder, so
    a PlainConvUNet checkpoint reaches the same tensor under four keys.
    """
    keys = [k for k, v in sd.items()
            if isinstance(v, torch.Tensor) and v.ndim in (4, 5)
            and FIRST_CONV_MARKER in k and k.endswith(FIRST_CONV_SUFFIXES)]
    if keys:
        return sorted(keys)
    # fall back: any 4-/5-D weight tensor whose shape matches the smallest in_channels
    cands = [(k, v) for k, v in sd.items()
             if isinstance(v, torch.Tensor) and v.ndim in (4, 5) and k.endswith("weight")]
    if not cands:
        raise KeyError("could not locate the first convolution in the checkpoint")
    ref_shape = min(c[1].shape[1] for c in cands)
    return sorted([k for k, v in cands if v.shape[1] == ref_shape])


def expand_first_conv(w: torch.Tensor,
                      new_in: int = 5,
                      guidance_scale: float = 1.0,
                      guidance_init: str = "copy",
                      prev_init: str = "zero",
                      seed: int = 0) -> torch.Tensor:
    out_c, in_c = w.shape[0], w.shape[1]
    assert in_c == 4, f"expected 4 input channels in the baseline conv, got {in_c}"
    g = torch.Generator().manual_seed(seed)
    new_w = torch.zeros((out_c, new_in, *w.shape[2:]), dtype=w.dtype)

    new_w[:, 0] = w[:, 0]                       # CT
    new_w[:, 1] = w[:, 1]                       # PET

    if guidance_init == "copy":
        new_w[:, 2] = w[:, 2] * guidance_scale  # tumor guidance
        new_w[:, 3] = w[:, 3] * guidance_scale  # background guidance
    elif guidance_init == "kaiming":
        fan_in = int(np.prod(w.shape[2:])) * new_in
        std = float(np.sqrt(2.0 / fan_in))
        new_w[:, 2] = torch.randn(new_w[:, 2].shape, generator=g) * std * guidance_scale
        new_w[:, 3] = torch.randn(new_w[:, 3].shape, generator=g) * std * guidance_scale
    elif guidance_init == "zero":
        pass
    else:
        raise ValueError(guidance_init)

    if prev_init == "zero":
        pass
    elif prev_init == "kaiming":
        fan_in = int(np.prod(w.shape[2:])) * new_in
        std = float(np.sqrt(2.0 / fan_in))
        new_w[:, 4] = torch.randn(new_w[:, 4].shape, generator=g) * std
    elif prev_init == "copy_fg":
        new_w[:, 4] = w[:, 2] * guidance_scale
    else:
        raise ValueError(prev_init)
    return new_w


def surgery(checkpoint: dict,
            new_in: int = 5,
            guidance_scale: float = 1.0,
            guidance_init: str = "copy",
            prev_init: str = "zero",
            seed: int = 0,
            trainer_name: str = "nnUNetTrainer_Interactive") -> dict:
    sd = checkpoint["network_weights"]
    keys = find_first_conv_keys(sd)
    ref = sd[keys[0]]
    new_w = expand_first_conv(ref, new_in, guidance_scale, guidance_init, prev_init, seed)
    for k in keys:
        assert sd[k].shape == ref.shape, f"aliased first-conv keys disagree: {k}"
        sd[k] = new_w.clone()

    out = {
        "network_weights": sd,
        # optimizer/grad-scaler state belongs to a different parameter shape -- drop it
        "optimizer_state": None,
        "grad_scaler_state": None,
        "logging": checkpoint.get("logging"),
        "_best_ema": None,
        "current_epoch": 0,
        "init_args": checkpoint.get("init_args"),
        "trainer_name": trainer_name,
        "inference_allowed_mirroring_axes": checkpoint.get("inference_allowed_mirroring_axes", (0, 1, 2)),
        "surgery": {
            "source_in_channels": int(ref.shape[1]),
            "target_in_channels": int(new_in),
            "guidance_scale": float(guidance_scale),
            "guidance_init": guidance_init,
            "prev_init": prev_init,
            "patched_keys": keys,
        },
    }
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True, help="baseline checkpoint_final.pth")
    ap.add_argument("--dst", required=True, help="output checkpoint (5 input channels)")
    ap.add_argument("--new-in-channels", type=int, default=5)
    # 1.0 already matches the guidance response to the CT/PET stem: the baseline's
    # per-channel weight norms are comparable, and our EDT encoding saturates at 1
    # over the kernel support where the baseline's z-scored map spiked on one voxel.
    ap.add_argument("--guidance-scale", type=float, default=1.0)
    ap.add_argument("--guidance-init", choices=["copy", "kaiming", "zero"], default="copy")
    ap.add_argument("--prev-init", choices=["zero", "kaiming", "copy_fg"], default="zero")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--trainer-name", default="nnUNetTrainer_Interactive")
    args = ap.parse_args()

    ck = torch.load(args.src, map_location="cpu", weights_only=False)
    keys = find_first_conv_keys(ck["network_weights"])
    w = ck["network_weights"][keys[0]]
    print(f"source first conv {keys} shape {tuple(w.shape)}")
    for c in range(w.shape[1]):
        print(f"  in-ch {c}: rms={float(w[:, c].pow(2).mean().sqrt()):.4f} "
              f"absmean={float(w[:, c].abs().mean()):.4f}")

    out = surgery(ck, args.new_in_channels, args.guidance_scale, args.guidance_init,
                  args.prev_init, args.seed, args.trainer_name)
    nw = out["network_weights"][keys[0]]
    print(f"target first conv shape {tuple(nw.shape)}")
    for c in range(nw.shape[1]):
        print(f"  in-ch {c}: rms={float(nw[:, c].pow(2).mean().sqrt()):.4f} "
              f"absmean={float(nw[:, c].abs().mean()):.4f}")

    os.makedirs(os.path.dirname(os.path.abspath(args.dst)), exist_ok=True)
    torch.save(out, args.dst)
    print("wrote", args.dst, f"({os.path.getsize(args.dst) / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
