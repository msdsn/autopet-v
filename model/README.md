# Trained weights

`checkpoint_final.pth` of the submitted model (nnU-Net v2, trainer `nnUNetTrainer_InteractiveV2_negfp`,
plans `nnUNetPlans_interactive`, 3d_fullres, fold 0), split into three parts to stay under GitHub's
100 MB file limit. Reassemble and verify:

```bash
cat model/checkpoint_final.pth.part0? > checkpoint_final.pth
sha256sum -c <(echo "$(cat model/checkpoint_final.pth.sha256)  checkpoint_final.pth")
```

`plans.json` and `dataset.json` are the nnU-Net model description files that go next to `fold_0/`.
The container build (`scripts/fetch_weights.sh`, `WEIGHTS_SOURCE=repo`) does this automatically.
