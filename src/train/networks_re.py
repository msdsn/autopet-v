"""The RE network: the autoPET III LesionTracer ResEncL, made interactive.

The row's whole point is the *backbone*. `dynamic_network_architectures`'
``ResidualEncoderUNet`` built from the LesionTracer plans is bit-for-bit the network
whose weights the Zenodo record ships (record 14007247, CC BY 4.0, Rokuss et al.
2024): building it with ``input_channels=2, num_classes=2`` and diffing against
``fold_0/checkpoint_final.pth`` leaves **0 missing tensors, 0 shape mismatches** and
exactly 10 unexpected ones -- the ``decoder.organ_seg_layers.*`` heads of the
MultiTalent organ supervision, which stock ``ResidualEncoderUNet`` does not have.
So the class here subclasses it and changes exactly two things:

1. ``input_channels = 5``. Channels 2-4 are our guidance and previous-mask channels;
   the surgery in ``init_from_lesiontracer.py`` zero-initialises their columns of the
   stem convolution, so the network at epoch 0 is the LesionTracer model reading a
   5-channel tensor whose last three channels it ignores.
2. an optional **fixed, parameter-free remap of the PET channel** (``pet_renorm``),
   because our store and theirs do not normalise PET the same way.

## The PET normalisation mismatch, and what ``pet_renorm`` does about it

The CT channel needs nothing: their ``foreground_intensity_properties_per_channel``
["0"] is *byte-identical* to ours (mean 107.73438968591431, std 286.34403119451997)
-- both fingerprints come from the same autoPET cohort -- and both sides run
``CTNormalization``. PET is where they differ:

| | channel 1 of the input |
|---|---|
| LesionTracer | ``CTNormalization`` with fip["1"]: ``(clip(SUV, 1.0433, 51.211) - 7.0638) / 7.9604`` |
| our store | ``ZScoreNormalization``, **per case** |

Feeding our z-scored PET to their stem is feeding it a channel it has never seen:
theirs is floored at SUV 1.04 -- about 93 % of body voxels sit on the floor at
-0.755 -- while ours is a per-case z-score that preserves the low-uptake range and
reaches ~+30 on a hot lesion.

``pet_renorm="ctnorm"`` undoes ours and applies theirs inside ``forward``:

```
SUV ~= z * PET_SD + PET_MU                       # cohort medians of pet_norm_correction
x1   = (clip(SUV, 1.0433, 51.211) - 7.0638) / 7.9604
```

``PET_MU``/``PET_SD`` are the same cohort medians ``networks_eva.py`` uses (0.1088 /
0.6249 over 120 store cases; the per-case ``sd`` spans 0.44-1.14), because a training
patch carries no per-case correction. That is an approximation -- a case at sd 1.14
has its reconstructed SUV under-estimated by 1.8x -- and it is acceptable for the
same reason it is in B17: the *same* function is applied at training and at
inference, the network is fine-tuned through it, and the alternative is a channel
that is wrong by construction rather than by a factor.

The mode is a plans field, not an environment variable, so the predictor rebuilds the
identical function from ``plans.json`` with no extra wiring. ``pet_renorm="none"``
keeps our z-score and is the ablation control.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import torch
from torch import nn

from dynamic_network_architectures.architectures.unet import ResidualEncoderUNet

__all__ = [
    "ResEncInteractiveUNet",
    "graft_lesiontracer_state_dict",
    "PET_MU", "PET_SD", "PET_LO", "PET_HI", "PET_MEAN", "PET_STD",
    "STEM_CONV_KEYS", "ORGAN_HEAD_PREFIX",
]

# cohort medians of the store's per-case pet_norm_correction; identical to the
# constants networks_eva.py renders with, measured over 120 store cases
PET_MU, PET_SD = 0.1088, 0.6249
# LesionTracer plans, foreground_intensity_properties_per_channel["1"] -- the
# constants their CTNormalization on the PET channel used
PET_LO, PET_HI = 1.0433332920074463, 51.211158752441406
PET_MEAN, PET_STD = 7.063827929027176, 7.960414805306728

#: every alias of the stem convolution's weight in a ResidualEncoderUNet state dict.
#: dynamic_network_architectures registers each conv both as `.conv` and inside an
#: `.all_modules` Sequential, and the decoder holds a reference to the encoder.
STEM_CONV_KEYS: Tuple[str, ...] = (
    "encoder.stem.convs.0.conv.weight",
    "encoder.stem.convs.0.all_modules.0.weight",
    "decoder.encoder.stem.convs.0.conv.weight",
    "decoder.encoder.stem.convs.0.all_modules.0.weight",
)

#: the MultiTalent organ-supervision heads, which stock ResidualEncoderUNet has not
ORGAN_HEAD_PREFIX = "decoder.organ_seg_layers."


class ResEncInteractiveUNet(ResidualEncoderUNet):
    """LesionTracer's ResEncL with 5 input channels and an optional PET remap.

    Referenced from ``nnUNetPlans_re.json`` as
    ``train.networks_re.ResEncInteractiveUNet``; ``src`` is on ``PYTHONPATH`` both on
    the training box and inside the container, so the predictor rebuilds it from the
    ``plans.json`` the trainer writes next to ``fold_0/`` with no extra wiring.
    """

    def __init__(self, *args, pet_renorm: str = "none", pet_channel: int = 1, **kwargs):
        super().__init__(*args, **kwargs)
        if pet_renorm not in ("none", "ctnorm"):
            raise ValueError(f"pet_renorm must be 'none' or 'ctnorm', got {pet_renorm!r}")
        self.pet_renorm = str(pet_renorm)
        self.pet_channel = int(pet_channel)

    def _remap_pet(self, x: torch.Tensor) -> torch.Tensor:
        if self.pet_renorm == "none":
            return x
        c = self.pet_channel
        z = x[:, c]
        suv = z * PET_SD + PET_MU
        pet = (suv.clamp(PET_LO, PET_HI) - PET_MEAN) / PET_STD
        # out-of-place: x may be a leaf the caller still needs (the identity probe
        # batch of the gate reuses it) and autograd forbids the in-place write anyway
        return torch.cat((x[:, :c], pet.unsqueeze(1), x[:, c + 1:]), dim=1)

    def forward(self, x):
        return super().forward(self._remap_pet(x))


# ---------------------------------------------------------------------------
# weight surgery
# ---------------------------------------------------------------------------

def expand_stem_conv(w: torch.Tensor, new_in: int = 5,
                     guidance_init: str = "zero", seed: int = 0) -> torch.Tensor:
    """(32, 2, 3, 3, 3) -> (32, new_in, 3, 3, 3); the new columns start at zero.

    Zero, not Kaiming: a zero column makes epoch 0 the LesionTracer model exactly --
    the extra channels are ignored -- while still receiving a non-zero gradient on the
    first step, because the gradient of a *input* column is ``x_c (x) delta``, not a
    multiple of the column itself. (That is the difference from a zero-initialised
    *output* projection, which does attenuate its own interior's gradient.)
    """
    out_c, in_c = int(w.shape[0]), int(w.shape[1])
    if in_c != 2:
        raise AssertionError(f"expected 2 input channels in the LesionTracer stem, got {in_c}")
    new_w = torch.zeros((out_c, new_in, *w.shape[2:]), dtype=w.dtype)
    new_w[:, 0] = w[:, 0]                       # CT
    new_w[:, 1] = w[:, 1]                       # PET
    if guidance_init == "zero":
        pass
    elif guidance_init == "kaiming":
        import numpy as np
        g = torch.Generator().manual_seed(seed)
        std = float(np.sqrt(2.0 / (int(np.prod(w.shape[2:])) * new_in)))
        new_w[:, 2:] = torch.randn(new_w[:, 2:].shape, generator=g) * std
    else:
        raise ValueError(guidance_init)
    return new_w


def graft_lesiontracer_state_dict(network: nn.Module, state_dict: Dict[str, torch.Tensor],
                                  verbose: bool = True) -> Tuple[List[str], List[str]]:
    """Strict graft of a LesionTracer checkpoint onto a ``ResEncInteractiveUNet``.

    The contract is the one ``networks.graft_state_dict`` enforces for the B13/B14
    rows, with one documented exception: the ten ``decoder.organ_seg_layers.*``
    tensors of the organ supervision have no home in a 2-class network and are
    dropped. Every *other* source tensor must be consumed with a matching shape, and
    the target must have no tensor the source did not fill -- so "epoch 0 is the
    LesionTracer model" is checkable rather than hoped for.
    """
    own = network.state_dict()
    sd = dict(state_dict)

    dropped = sorted(k for k in sd if k.startswith(ORGAN_HEAD_PREFIX))
    for k in dropped:
        sd.pop(k)

    bad = [k for k in sd if k in own and tuple(own[k].shape) != tuple(sd[k].shape)]
    if bad:
        raise RuntimeError(f"shape mismatch on {len(bad)} source tensors, e.g. "
                           f"{[(k, tuple(sd[k].shape), tuple(own[k].shape)) for k in bad[:4]]}")
    missing, unexpected = network.load_state_dict(sd, strict=False)
    if unexpected:
        raise RuntimeError(f"{len(unexpected)} source tensors not consumed by the network: "
                           f"{list(unexpected)[:8]}")
    if missing:
        raise RuntimeError(f"{len(missing)} network tensors were not in the source "
                           f"checkpoint: {list(missing)[:8]}")
    if verbose:
        print(f"[graft-re] loaded {len(sd)} tensors, dropped {len(dropped)} organ-head "
              f"tensors ({ORGAN_HEAD_PREFIX}*), 0 missing, 0 unexpected")
    return list(missing), dropped
