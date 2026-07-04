---
type: method
title: Foundation-Model Diagnostics
tags: [method, interpretability, foundation-models]
description: "Data-light audits of trained brain foundation models — weight-spectrum quality, subject-identity leakage, and sparse-autoencoder interpretability — that probe a frozen encoder without retraining it."
timestamp: 2026-07-03T00:00:00-07:00
category: interpretability
implementations: ["fmscope:fmscope/diagnostics", "emeg-fm:emeg_fm/sae.py", "neural-decoding-transfer:alpha_report.py", "wwj:src/wwj/core.py"]
related: [foundation-models.md, htsr-weight-analysis.md, method-masked-autoencoding.md, eeg.md, meg.md]
---

# Foundation-Model Diagnostics

As brain foundation models proliferate ([masked-autoencoding](method-masked-autoencoding.md), JEPA, spiking decoders), the bottleneck shifts from *training* to *trusting* them. FM diagnostics are cheap, often data-free audits applied to a **frozen** model to answer: is it well-trained, what has it actually learned, and is it exploiting shortcuts?

## Three probes used here

1. **Weight-spectrum quality (HT-SR α)** — power-law fit of each layer's weight-eigenvalue spectrum; α near 2 indicates a well-trained, heavy-tailed layer. Data-free. See [htsr-weight-analysis.md](htsr-weight-analysis.md). (`wwj`, `weightwatcher`, `neural-decoding-transfer:alpha_report`).
2. **Identity / shortcut leakage** — does a "cognition" encoder actually encode *subject identity*? `fmscope` runs a battery of diagnostics on frozen EEG FMs and issues a verdict (subject-identity leakage among them).
3. **Sparse-autoencoder interpretability** — train a TopK SAE on frozen activations to extract monosemantic features (mechanistic probing). `emeg-fm:sae` (JAX), plus activation adapters for E/MEG encoders.

## Why data-light matters

These audits need no (or little) labelled downstream data, so they scale across the many checkpoints a foundation-model project produces and catch failure modes — collapse, memorization, identity leakage — that a single downstream accuracy number hides.

## Implementations

- **fmscope** — 5-diagnostic audit of frozen EEG FMs with a pass/fail verdict.
- **emeg-fm** — WeightWatcher-α + TopK SAEs + FMScope audit + realtime decoding for E/MEG FMs.
- **wwj / weightwatcher** — HT-SR α scoring (JAX port and reference).
- **neural-decoding-transfer** — α-report scoring of POYO spiking-FM checkpoints.

## Citations

[1] Martin & Mahoney (2021). Implicit self-regularization in deep neural networks (HT-SR / heavy-tailed). JMLR.
[2] Cunningham et al. (2023). Sparse autoencoders find highly interpretable features in language models.

## See Also

- [htsr-weight-analysis.md](htsr-weight-analysis.md) - weight-spectrum α theory
- [foundation-models.md](foundation-models.md) - models being audited
- [method-masked-autoencoding.md](method-masked-autoencoding.md) - how the encoders are trained
