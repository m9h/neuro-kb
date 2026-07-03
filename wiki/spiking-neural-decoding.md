---
type: method
title: Spiking Neural Decoding
description: "Decoding behavior from populations of spiking neurons with transformer foundation models (POYO/torch_brain) that tokenize individual spikes across sessions and animals."
timestamp: 2026-07-03T00:00:00-07:00
category: representation-learning
implementations: ["neural-decoding-transfer:poyo_harness", "neural-decoding-transfer:alpha_report.py"]
related: [foundation-models.md, htsr-weight-analysis.md, fm-diagnostics.md, benchmark-datasets.md]
---

# Spiking Neural Decoding

Spiking neural decoding maps the activity of many simultaneously recorded neurons to a behavioral variable (e.g. cursor velocity). The modern foundation-model approach — **POYO** and the `torch_brain` ecosystem — tokenizes *individual spikes* (unit identity + time) rather than binned rates, and trains a transformer with learned unit embeddings so a single model transfers across sessions, arrays, and animals.

## Key ideas

- **Spike tokenization** — each spike becomes a token; rotary/temporal encodings carry timing; per-unit latent embeddings absorb session-specific wiring.
- **Cross-session transfer** — new sessions are handled by fitting only unit embeddings ("unit identification"), keeping the backbone frozen.
- **Evaluation** — decoding R²/R²(tp) is the accuracy axis; weight-spectrum α ([fm-diagnostics](fm-diagnostics.md), [htsr-weight-analysis](htsr-weight-analysis.md)) is the data-free quality axis.

## Reproduction status

`neural-decoding-transfer` reproduces the neuro-galaxy POYO spiking-FM and studies transfer. Note (project state): full reproduction is **not yet confirmed** — a weight-collapse issue is under investigation, with the decisive test being an unmodified upstream POYO run at ≥50 epochs on a pinned Torch. The `poyo_harness` drives training; `alpha_report.py` scores checkpoints with `wwj`.

## Implementations

- **neural-decoding-transfer** — POYO reproduction + transfer study; `poyo_harness` training, HTSR-α checkpoint scoring.

## Citations

[1] Azabou et al. (2023). A unified, scalable framework for neural population decoding (POYO). NeurIPS.
[2] Azabou et al. (2024). POYO+ / torch_brain — multi-session neural decoding.

## See Also

- [foundation-models.md](foundation-models.md) - foundation-model context
- [fm-diagnostics.md](fm-diagnostics.md) - α-based checkpoint scoring
- [benchmark-datasets.md](benchmark-datasets.md) - neural decoding datasets
