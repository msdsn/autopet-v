# Second ensemble member: the ResEncL model

`plans.json` (`nnUNetPlans_re`) and `dataset.json` describe the second member of the shipped
ensemble — a `ResidualEncoderUNet` with 102.4 M parameters and 192³ patches, adapted to the
five-channel interactive contract and fine-tuned for 100 epochs
(trainer `nnUNetTrainer_InteractiveRE_100epochs`, fold 0).

Its `checkpoint_final.pth` is not in this repository: at 410 MB it is over GitHub's file limit
even split, so the container build fetches it once from a read-only link and verifies it against
the sha256 pinned in the `Dockerfile` (`RE_CHECKPOINT_SHA256`). The file holds network weights
only; the optimizer state that makes the training checkpoint 819 MB was stripped after checking
that the remaining 956 tensors are equal to the training checkpoint's.

## Provenance and licence

The weights are warm-started from the autoPET III challenge-winning model of team LesionTracer
and then fine-tuned on the interactive task by us:

> M. Rokuss, Y. Kirchhoff, S. Roy, B. Wittmann, N. Navab, K. Maier-Hein.
> *From FDG to PSMA: A Hitchhiker's Guide to Multitracer, Multicenter Lesion Segmentation in
> PET/CT Imaging*, arXiv:2409.09478, 2024.
> Weights: Zenodo record 14007247, https://doi.org/10.5281/zenodo.14007247, **CC BY 4.0**.

The surgery (2 → 5 input channels, zero-initialised interaction columns, organ heads dropped),
the PET renormalisation and the epoch-0 identity gate that proves the adapted network starts as
the source network are described in `docs/train_pipeline.md`.
