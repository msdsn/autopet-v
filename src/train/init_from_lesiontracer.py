"""Weight surgery: LesionTracer ResEncL (2 channels, + organ heads) -> our 5-channel RE init.

Source: ``fold_<k>/checkpoint_final.pth`` of the autoPET III LesionTracer model,
Zenodo record 14007247 (CC BY 4.0, Rokuss et al. 2024, arXiv:2409.09478); 5 folds,
``autoPET3_Trainer__nnUNetResEncUNetLPlansMultiTalent__3d_fullres_bs3``.

Three edits, and nothing else:

1. **Stem convolution 2 -> 5 input channels.** CT and PET columns are copied, the
   three interaction columns start at zero. The tensor appears **four** times in the
   state dict -- ``encoder.stem.convs.0.conv.weight``,
   ``encoder.stem.convs.0.all_modules.0.weight`` and both again under
   ``decoder.encoder.…`` because the decoder holds a reference to the encoder -- and
   all four are patched, exactly as ``init_from_baseline.py`` patches the four
   aliases of the PlainConvUNet's first conv.
2. **Drop ``decoder.organ_seg_layers.*``** (10 tensors, 11 classes each). These are
   the MultiTalent organ supervision. They are an auxiliary *training* head of their
   trainer; stock ``ResidualEncoderUNet`` has no such module, and our task has no
   organ labels to supervise them with.
3. **Keep ``decoder.seg_layers.*`` verbatim.** This is the decision that makes the
   row worth running, so it is worth stating why it is safe: their shipped
   ``dataset.json`` is ``{"background": 0, "tumor": 1}`` and the heads are
   ``(2, C, 1, 1, 1)`` -- *two* classes, our two classes, at our spacing. There is
   nothing to slice and nothing to re-initialise; re-initialising them would throw
   away the calibration of a model that scores Dice 0.687 on its own autoPET III
   validation and would make "warm start" mean "warm encoder, cold output".

The result is checked, not asserted: the surgered state dict is grafted into the
network the RE plans actually build, and every tensor must match on both sides
(``graft_lesiontracer_state_dict``). Optimizer and grad-scaler state are dropped and
``current_epoch`` is reset, so the output is a valid continuation source for
``nnUNetTrainer_InteractiveRE``.
"""

from __future__ import annotations

import argparse
import json
import os

import torch

try:  # package import (src/train is a package)
    from .networks_re import (STEM_CONV_KEYS, ORGAN_HEAD_PREFIX, expand_stem_conv,
                              graft_lesiontracer_state_dict)
except ImportError:  # flat import (folder on sys.path, e.g. nnUNet_extTrainer)
    from networks_re import (STEM_CONV_KEYS, ORGAN_HEAD_PREFIX, expand_stem_conv,  # type: ignore
                             graft_lesiontracer_state_dict)


def surgery(checkpoint: dict, new_in: int = 5, guidance_init: str = "zero", seed: int = 0,
            trainer_name: str = "nnUNetTrainer_InteractiveRE") -> dict:
    sd = dict(checkpoint["network_weights"])

    keys = [k for k in STEM_CONV_KEYS if k in sd]
    if len(keys) != len(STEM_CONV_KEYS):
        raise KeyError(f"expected the four stem aliases {STEM_CONV_KEYS}, found {keys}")
    ref = sd[keys[0]]
    for k in keys:
        if tuple(sd[k].shape) != tuple(ref.shape):
            raise AssertionError(f"aliased stem keys disagree: {k}")
    new_w = expand_stem_conv(ref, new_in, guidance_init, seed)
    for k in keys:
        sd[k] = new_w.clone()

    dropped = sorted(k for k in sd if k.startswith(ORGAN_HEAD_PREFIX))
    for k in dropped:
        sd.pop(k)

    return {
        "network_weights": sd,
        "optimizer_state": None,          # a different parameter shape -- drop it
        "grad_scaler_state": None,
        "logging": None,                  # 1500 epochs of their curves, not ours
        "_best_ema": None,
        "current_epoch": 0,
        "init_args": None,
        "trainer_name": trainer_name,
        "inference_allowed_mirroring_axes": checkpoint.get(
            "inference_allowed_mirroring_axes", (0, 1, 2)),
        "surgery": {
            "source": "autoPET III LesionTracer, Zenodo 14007247 (CC BY 4.0)",
            "source_trainer": checkpoint.get("trainer_name"),
            "source_epoch": checkpoint.get("current_epoch"),
            "source_in_channels": int(ref.shape[1]),
            "target_in_channels": int(new_in),
            "guidance_init": guidance_init,
            "patched_keys": keys,
            "dropped_keys": dropped,
        },
    }


def _stats(w: torch.Tensor, tag: str) -> None:
    print(f"{tag} shape {tuple(w.shape)}")
    for c in range(w.shape[1]):
        print(f"  in-ch {c}: rms={float(w[:, c].pow(2).mean().sqrt()):.4f} "
              f"absmean={float(w[:, c].abs().mean()):.4f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True, help="LesionTracer fold_<k>/checkpoint_final.pth")
    ap.add_argument("--dst", required=True, help="output checkpoint (5 input channels)")
    ap.add_argument("--plans", default=None,
                    help="nnUNetPlans_re.json; when given, the result is grafted into the "
                         "network these plans build and every tensor must match")
    ap.add_argument("--configuration", default="3d_fullres")
    ap.add_argument("--new-in-channels", type=int, default=5)
    ap.add_argument("--guidance-init", choices=["zero", "kaiming"], default="zero")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--trainer-name", default="nnUNetTrainer_InteractiveRE")
    args = ap.parse_args()

    ck = torch.load(args.src, map_location="cpu", weights_only=False)
    print(f"source: trainer={ck.get('trainer_name')} epoch={ck.get('current_epoch')} "
          f"best_ema={ck.get('_best_ema')} tensors={len(ck['network_weights'])}")
    _stats(ck["network_weights"][STEM_CONV_KEYS[0]], "source stem conv")

    out = surgery(ck, args.new_in_channels, args.guidance_init, args.seed, args.trainer_name)
    _stats(out["network_weights"][STEM_CONV_KEYS[0]], "target stem conv")
    print(f"dropped {len(out['surgery']['dropped_keys'])} organ-head tensors")

    if args.plans:
        from nnunetv2.utilities.get_network_from_plans import get_network_from_plans
        with open(args.plans) as f:
            plans = json.load(f)
        arch = plans["configurations"][args.configuration]["architecture"]
        net = get_network_from_plans(arch["network_class_name"], arch["arch_kwargs"],
                                     arch["_kw_requires_import"], args.new_in_channels, 2,
                                     allow_init=True, deep_supervision=True)
        graft_lesiontracer_state_dict(net, out["network_weights"], verbose=True)
        seen, n = set(), 0
        for p in net.parameters():
            if p.data_ptr() not in seen:
                seen.add(p.data_ptr())
                n += p.numel()
        print(f"grafted into {arch['network_class_name']}: {n / 1e6:.3f} M parameters, "
              f"pet_renorm={arch['arch_kwargs'].get('pet_renorm')}")

    os.makedirs(os.path.dirname(os.path.abspath(args.dst)), exist_ok=True)
    torch.save(out, args.dst)
    print("wrote", args.dst, f"({os.path.getsize(args.dst) / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
