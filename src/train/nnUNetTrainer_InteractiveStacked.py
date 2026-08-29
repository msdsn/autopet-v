"""Second-round stacked variants: the S1 sampler combined with an architecture block.

Round one asks *exposure or capacity* one mechanism at a time. These two rows ask
whether the answers add, which the B12g precedent says must never be assumed: a
lesion-free penalty and a blob loss were individually motivated and together scored
-0.263 AUC-Dice / -0.106 AUC-DMM against the better of the two.

* ``nnUNetTrainer_InteractiveS1N1``  -- S1's component-balanced sampler **and** N1's
  presence-prior head (plans ``nnUNetPlans_n1``). One mechanism adds components, the
  other removes them; this is the orthogonality row.
* ``nnUNetTrainer_InteractiveS1B14`` -- S1's sampler on Engineer 1's
  ``EditBranchUNet`` (plans ``nnUNetPlans_b14``): does feeding the edit branch far
  more small lesions change what it learns?

Nothing is re-implemented. Each class is the multiple-inheritance combination of the
single-mechanism trainers, so the sampler, the auxiliary loss, the weight surgery and
the in-process identity gate are the same code that produced the round-one rows. The
only per-class declaration is ``NEW_PARAM_PREFIXES``, because ``nnUNetTrainer_Interactive
S1`` sets it to ``()`` and would otherwise shadow the architecture row's prefixes
through the MRO.

``EpochOverrideMixin`` reads ``NUM_EPOCHS`` from the environment. The poly LR schedule
is computed from ``self.num_epochs``, so a 40-epoch screen decays the learning rate
over 40 epochs rather than truncating a 120-epoch schedule. ``train_b6.sh`` unsets the
``nnUNet_interactive_*`` overrides on purpose, which is why this is a separate name
that the launcher passes through.
"""

from __future__ import annotations

import os

try:  # package import (src/train is a package)
    from .nnUNetTrainer_InteractiveArch import nnUNetTrainer_InteractiveB14
    from .nnUNetTrainer_InteractiveN1 import nnUNetTrainer_InteractiveN1
    from .nnUNetTrainer_InteractiveS1 import nnUNetTrainer_InteractiveS1
except ImportError:  # flat import (folder on sys.path, e.g. nnUNet_extTrainer)
    from nnUNetTrainer_InteractiveArch import nnUNetTrainer_InteractiveB14  # type: ignore
    from nnUNetTrainer_InteractiveN1 import nnUNetTrainer_InteractiveN1  # type: ignore
    from nnUNetTrainer_InteractiveS1 import nnUNetTrainer_InteractiveS1  # type: ignore

__all__ = [
    "EpochOverrideMixin",
    "nnUNetTrainer_InteractiveS1N1",
    "nnUNetTrainer_InteractiveS1B14",
    "nnUNetTrainer_InteractiveS1N1_screen40",
    "nnUNetTrainer_InteractiveS1B14_screen40",
]


class EpochOverrideMixin:
    """``NUM_EPOCHS=<n>`` in the environment sets the schedule length.

    Applied after the base ``__init__`` has read ``nnUNet_interactive_epochs``, so it
    wins over both that variable and the class attribute. Everything downstream --
    the poly LR decay and the progress watcher's ETA -- reads ``self.num_epochs``.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        n = os.environ.get("NUM_EPOCHS")
        if n:
            self.num_epochs = int(n)
            self.print_to_log_file(
                f"[epochs] NUM_EPOCHS={n} overrides the class schedule "
                f"({type(self).__name__}.NUM_EPOCHS={getattr(type(self), 'NUM_EPOCHS', None)}); "
                f"the poly LR decays over {self.num_epochs} epochs")


class nnUNetTrainer_InteractiveS1N1(EpochOverrideMixin,
                                    nnUNetTrainer_InteractiveN1,
                                    nnUNetTrainer_InteractiveS1):
    """S1 sampler + N1 presence-prior head. Run with ``-p nnUNetPlans_n1``.

    MRO: the N1 auxiliary loss, then the S1 dataloaders, then the shared identity gate
    and the weight surgery of ``nnUNetTrainer_InteractiveArch``.
    """

    NEW_PARAM_PREFIXES = ("presence_gate.",)


class nnUNetTrainer_InteractiveS1B14(EpochOverrideMixin,
                                     nnUNetTrainer_InteractiveS1,
                                     nnUNetTrainer_InteractiveB14):
    """S1 sampler on the edit-branch decoder. Run with ``-p nnUNetPlans_b14``."""

    NEW_PARAM_PREFIXES = ("edit_stem.", "edit_ups.", "edit_skip_projs.",
                          "edit_stages.", "edit_seg_layers.")


class nnUNetTrainer_InteractiveS1N1_screen40(nnUNetTrainer_InteractiveS1N1):
    """40-epoch screening schedule, in its own results folder."""
    NUM_EPOCHS = 40


class nnUNetTrainer_InteractiveS1B14_screen40(nnUNetTrainer_InteractiveS1B14):
    """40-epoch screening schedule, in its own results folder."""
    NUM_EPOCHS = 40
