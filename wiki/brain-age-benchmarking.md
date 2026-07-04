---
type: concept
title: Brain-Age Benchmarking
tags: [foundation-models, benchmarks]
description: "Predicting chronological age from brain data and using the residual (brain-age delta) as a downstream benchmark task shared across MRI and M/EEG foundation models."
timestamp: 2026-07-03T00:00:00-07:00
implementations: ["meeg-brain-age-benchmark-paper:compute_benchmark_age_prediction.py", "BrainIAC:src/generate_brainage_vit_saliency.py", "fomo25:src/finetune.py", "neoba:src/neoba"]
related: [foundation-models.md, benchmark-datasets.md, structural-mri.md, eeg.md, meg.md, method-spectral-analysis.md]
---

# Brain-Age Benchmarking

Brain age is the age predicted from a brain scan or recording; the **brain-age delta** (predicted − chronological) is a widely used marker of accelerated or resilient aging. Because ground-truth age is cheap and universal, brain-age prediction has become a **canonical downstream benchmark** for representation quality — it recurs across structural-MRI, fMRI, and M/EEG models in this corpus, which makes it a useful common yardstick for foundation-model transfer.

## Modelling axes

- **Features** — morphometry / voxel patterns (MRI), resting-state power & connectivity (M/EEG), or learned FM embeddings.
- **Interpretability** — from black-box ViT saliency (`BrainIAC`) to explicitly interpretable oscillatory features: `neoba` predicts EEG brain age from specparam-derived oscillatory settling functions + sparse-group-lasso dependency coefficients (aperiodic/periodic aging).
- **Regressor & bias correction** — age-bias regression (predictions shrink toward the mean) is a standard confounder to correct.

## As a benchmark protocol

Reusable pipelines fix datasets, splits, and metrics (MAE, R²) so models are comparable. `meeg-brain-age-benchmark-paper` provides a resting-state M/EEG benchmark (filterbank-Riemannian and deep baselines); MRI challenges (`fomo25`) and FMs (`BrainIAC`) report the same task, letting cross-modal representations be compared on a shared axis.

## Implementations

- **meeg-brain-age-benchmark-paper** — reusable M/EEG resting-state brain-age benchmark + deep-learning baselines.
- **BrainIAC** — structural-MRI ViT with brain-age head + saliency.
- **fomo25** — MRI brain-age among the FOMO25 challenge tasks.
- **neoba** — interpretable EEG oscillatory brain-age predictor.

## Citations

[1] Cole & Franke (2017). Predicting age using neuroimaging: innovative brain ageing biomarkers. Trends Neurosci.
[2] Engemann et al. (2022). A reusable benchmark of brain-age prediction from M/EEG resting-state signals. NeuroImage.

## See Also

- [foundation-models.md](foundation-models.md) - models evaluated on this task
- [benchmark-datasets.md](benchmark-datasets.md) - datasets and cohorts
- [method-spectral-analysis.md](method-spectral-analysis.md) - oscillatory feature extraction
