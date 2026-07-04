---
type: method
title: Masked Autoencoding for Brain Data
tags: [method, representation-learning, foundation-models]
description: "Self-supervised pretraining that masks part of a brain scan or recording and trains an encoder-decoder to reconstruct it, yielding transferable foundation-model representations."
timestamp: 2026-07-03T00:00:00-07:00
category: representation-learning
implementations: ["CortexMAE:src/cortex_mae/main_pretrain.py", "CortexMAE:src/cortex_mae/masking.py", "reve_eeg:src/train.py", "smri-fm-fomo26:src/smri_mae", "fomo25:src/pretrain.py"]
related: [foundation-models.md, fmri.md, structural-mri.md, eeg.md, benchmark-datasets.md, fm-diagnostics.md]
---

# Masked Autoencoding for Brain Data

Masked autoencoding (MAE) is the dominant self-supervised recipe behind neuroimaging foundation models: mask a large fraction of the input (image patches, cortical parcels, or EEG segments), encode the visible remainder, and train a decoder to reconstruct the masked content. The learned encoder is then frozen or fine-tuned for downstream probes. It is data-hungry but label-free, which suits the large unlabelled scan archives used here.

## Design axes

- **Tokenization** — how the brain is patched: fMRI parcels / flatmaps / volumes (`CortexMAE:masking`), 3D structural-MRI patches (`smri-fm`, `fomo25`), EEG time–channel segments (`reve_eeg`).
- **Masking ratio & strategy** — high random masking (ViT-MAE style) vs structured/temporal masking.
- **Reconstruction target** — raw signal vs latent features.

## MAE vs JEPA

Reconstruction-space MAE predicts the *input*; **JEPA** (joint-embedding predictive architecture) predicts *latent representations* of the masked region, avoiding pixel-level reconstruction and often yielding more semantic features. `smri-fm-neurojepa` is the JEPA-style structural-MRI variant of the same family.

## Implementations

- **CortexMAE** — fMRI MAE on 2.1K hours of HCP; parcel/flatmap/volume variants; Brainmarks eval.
- **reve_eeg** — EEG MAE pretrained on 25K subjects + linear probing.
- **smri-fm / -fomo26 / -neurojepa** — structural-MRI MAE (and JEPA) with morphometry baselines.
- **fomo25** — MRI pretrain+finetune for the FOMO25 challenge.

## Citations

[1] He et al. (2022). Masked autoencoders are scalable vision learners. CVPR.
[2] Assran et al. (2023). Self-supervised learning from images with a joint-embedding predictive architecture (I-JEPA). CVPR.

## See Also

- [foundation-models.md](foundation-models.md) - foundation-model overview
- [fm-diagnostics.md](fm-diagnostics.md) - auditing the resulting encoders
- [fmri.md](fmri.md) / [structural-mri.md](structural-mri.md) / [eeg.md](eeg.md) - input modalities
