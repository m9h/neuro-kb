---
type: concept
timestamp: 2026-06-12T16:08:30-07:00
title: Neuroimaging Foundation Models
tags: [foundation-models]
description: "A foundation model (FM) here is a large model self-supervised on brain data (masked autoencoding, contrastive, or diffusion objectives) that is then *frozen* and evaluated by training light linear…"
implementations: ["CortexMAE:src/cortex_mae", "BrainIAC:src", "smri-fm-fomo26:src/smri_mae", "reve_eeg:src/train.py", "emeg-fm:emeg_fm", "neural-decoding-transfer:poyo_harness"]
related: [method-masked-autoencoding.md, fm-diagnostics.md, brain-age-benchmarking.md, spiking-neural-decoding.md, htsr-weight-analysis.md, benchmark-datasets.md, structural-mri.md, fmri.md, eeg.md, meg.md]
---

# Neuroimaging Foundation Models

A **foundation model (FM)** here is a large model self-supervised on brain data
(masked autoencoding, contrastive, or diffusion objectives) that is then *frozen*
and evaluated by training light linear probes on its representations. The project
family studies these FMs through a consistent, partly **weights-first** lens:
score a model before — and independently of — any downstream task.

## Why a shared page

These projects ask the same question across modalities: *does a pretrained brain FM
actually encode generalizable structure, or is it riding on shortcuts?* The shared
machinery is (1) data-free weight diagnostics, (2) frozen linear-probe decoding, and
(3) identity / shortcut audits. Each modality-specific project reuses the same axes.

## The two-axis (plus audit) benchmark

| Axis | What it measures | Where it lives |
|---|---|---|
| **Spectral (weights-first)** | HT-SR power-law exponent α per layer; participation count; correlation traps. Data-free pretraining-quality score. | [htsr-weight-analysis.md](htsr-weight-analysis.md) (`wwj`, `emeg-fm`) |
| **Decoding** | Frozen linear/logistic probe accuracy or R² on state & trait targets | training harnesses; [benchmark-datasets.md](benchmark-datasets.md) |
| **Identity audit** | Whether accuracy rides on subject-identity leakage rather than task signal | FMScope, *Identity Trap* [@lin2026identity] |

A key methodological finding across the family: a strong morphometry / functional-
connectivity baseline is hard to dethrone on **trait** tasks, so FMs are only
credited when they beat that floor — not merely when they decode above chance.

## The MedARC FM family

| Project | Modality | Role |
|---|---|---|
| `fmri-fm` (CortexMAE) | Functional MRI | Masked-autoencoder cortical FM; the eval target of Brainmarks [@lane2025scaling] |
| **`smri-fm`** | Structural MRI | MedARC structural-MRI FM; benchmarks FMs against morphometry baselines |
| **`emeg-fm`** | EEG / MEG | Diagnostics, interpretability, identity audit & realtime decoding for E/MEG FMs |

**Sibling, non-brain medical FMs** (same recipe, different organ — listed for
cross-pollination, not brain-imaging):

- `nanopath` — tile-level computational **pathology** FM (nanochat-style lean harness;
  ~1 M tiles in ~1 h on one GPU, broad probe suite).
- `mars-fm` — MarS-FM, generative **molecular-dynamics** modeling via Markov State
  Models [@kapusniak2025mars] (ICLR 2026); not an imaging model.

## Structural MRI — `smri-fm`

The MedARC structural-MRI foundation model. The benchmark scopes ~14 model arms and
pits FM representations against **morphometry baselines** (FreeSurfer / T1Prep
volumetric and surface features — see [structural-mri.md](structural-mri.md)). The
recurring result in this family is that minimally-preprocessed pipelines and
morphometry floors are strong, so the FM has to clear a real bar. The
[benchmark-datasets.md](benchmark-datasets.md) cohorts supply the probe targets.

## EEG / MEG — `emeg-fm`

The E/MEG counterpart to `fmri-fm`/`smri-fm`, auditing the current generation of
E/MEG foundation models — **REVE, LaBraM, LUNA, BENDR, BIOT, CBraMod, ZUNA** — on four
fronts:

1. **HT-SR / WeightWatcher α** per weight matrix, ranking pretraining quality with no
   data ([htsr-weight-analysis.md](htsr-weight-analysis.md)). E.g. REVE α≈3.61 (5.6%
   of layers α<2, well-trained) vs BENDR α≈2.10 (68% α<2, severely under-trained).
2. **TopK sparse autoencoders** on block activations — mechanistic decomposition into
   a sparse dictionary, then probing each feature against HBN clinical concepts
   (p-factor, internalizing, externalizing, attention, age, sex) [@gao2024scaling].
   Activation participation ratio predicts SAE yield.
3. **FMScope identity audit** — the five frozen-representation diagnostics of *The
   Identity Trap in EEG Foundation Models* [@lin2026identity], testing whether a
   model's accuracy is subject-identity leakage rather than genuine task signal.
4. **Realtime-EMEG decoding** — frozen EEG-FM → image retrieval on Alljoined-1.6M
   (32-ch consumer EEG), the EMEG analogue of the NSD/MindEye fMRI demo.

Why the weight and activation analyses belong together: Martin's RG theory predicts a
well-trained dense layer sits near α≈2; `emeg-fm` connects that spectral picture to
interpretability, finding the activation participation ratio is the activation-space
analogue of the weight-space dominant-tail regime. See
[htsr-weight-analysis.md](htsr-weight-analysis.md).

## Tokenization & inputs

- `ephys-tokenizer-jax` — JAX/Equinox port of EphysTokenizer for sample-level MEG/EEG
  tokenization, the front-end that turns continuous traces into FM tokens.
- `hippy-feat` — preprocessing-comparison framework (MindEye-RT lineage) feeding
  fMRI FM probes.

## Relevant Projects

- **smri-fm**: structural-MRI FM vs morphometry baselines (this family's sMRI arm).
- **emeg-fm**: E/MEG FM diagnostics — HT-SR α, SAEs, FMScope identity audit, realtime decode.
- **ephys-tokenizer-jax**: sample-level E/MEG tokenization front-end.
- **hippy-feat**: preprocessing-comparison harness for fMRI FM probes.
- **brainmarks-hbn / -wand / -dlbs / -syn**: the fMRI-FM evaluation cohorts ([benchmark-datasets.md](benchmark-datasets.md)).
- **nanopath**, **mars-fm**: sibling non-brain medical/scientific FMs.

## Citations
- **lane2025scaling**: CortexMAE — scaling cortical masked-autoencoder fMRI foundation
  models (arXiv 2510.13768); the model Brainmarks evaluates.
- **lin2026identity**: Lin, Wu & Jung (2026). The Identity Trap in EEG Foundation
  Models: A Diagnostic Audit (arXiv 2606.06647).
- **martin2026rg**: Martin (2026). A Renormalization Group Theory of Learning.
- **gao2024scaling**: Gao et al. (2024). Scaling and evaluating sparse autoencoders.
- **kapusniak2025mars**: MarS-FM (arXiv 2509.24779, ICLR 2026) — sibling molecular-dynamics FM.

## See Also

- [htsr-weight-analysis.md](htsr-weight-analysis.md) — the weights-first scoring axis
- [benchmark-datasets.md](benchmark-datasets.md) — fMRI-FM evaluation cohorts (Brainmarks)
- [structural-mri.md](structural-mri.md) — morphometry baselines for smri-fm
- [eeg.md](eeg.md) / [meg.md](meg.md) — modalities behind emeg-fm
- [fmri.md](fmri.md) — modality behind fmri-fm / CortexMAE
