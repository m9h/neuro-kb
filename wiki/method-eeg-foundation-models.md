---
type: method
title: EEG Foundation Models and Representation Learning
category: inference
implementations:
  - nanoeeg:nanoeeg/frozen/probes.py
  - nanoeeg:nanoeeg/frozen/scoring.py
  - nanoeeg:nanoeeg/frozen/accounting.py
  - emeg-fm:emeg_fm/eeg_fm.py
  - emeg-fm:emeg_fm/sae.py
  - emeg-fm:emeg_fm/lora.py
  - emeg-fm:emeg_fm/adapters.py
  - zuna-eeg-pipeline:zuna_batch_eval.py
related:
  - foundation-models.md
  - method-masked-autoencoding.md
  - htsr-weight-analysis.md
  - fm-diagnostics.md
  - eeg.md
  - meg.md
  - fmri.md
  - method-spectral-analysis.md
  - method-pharmaco-eeg.md
  - method-hmm-dynamics.md
  - method-sbi.md
  - data-formats.md
  - coordinate-systems.md
  - compute-infrastructure.md
  - jax-ecosystem.md
description: "This page collects three kinds of shared knowledge from the project family:"
timestamp: 2026-11-05T15:59:53-08:00
tags: [method, foundation-model, eeg, representation-learning]
---

# EEG Foundation Models and Representation Learning

**EEG foundation models (EEG-FMs)** are transformer or conv-transformer encoders pretrained with self-supervised objectives on thousands of hours of unlabeled [EEG](eeg.md). They are evaluated by freezing the encoder and fitting a linear probe or LoRA adapter on downstream tasks. The current public generation includes REVE, LaBraM, LUNA, BENDR, BIOT, CBraMod, and ZUNA.

This page collects three kinds of shared knowledge from the project family:

1. **How to benchmark EEG-FMs honestly.** Subject-identity leakage, metric mixing, compute accounting, and montage harmonisation.
2. **How to diagnose a checkpoint without data.** Weight-spectrum (HTSR) analysis, activation participation ratio, and sparse autoencoders.
3. **How to adapt and apply them.** LoRA targeting, optimizer choice, and channel imputation.

No bibliography keys are available for this topic yet, so facts are cited to project source files.

## Model landscape

The table below is from the WeightWatcher analysis of the NeuralBench-EEG v1.0 checkpoints, run 2026-05-25 (`emeg-fm:docs/WEIGHTWATCHER_EEGFM_ANALYSIS.md`). The ZUNA row comes from `emeg-fm:README.md` and `zuna-eeg-pipeline:README.md`.

| Model | HF checkpoint | Params | Pretraining | α-mean | α-median | %α<2 | %α∈[2,6] | %α>6 |
|---|---|---:|---|---:|---:|---:|---:|---:|
| LUNA | `PulpBio/LUNA` | 40.5 M | ~60k+ h, multi-source | **3.93** | 3.64 | 10.4% | 79.1% | 10.4% |
| LaBraM | `braindecode/labram-pretrained` | 5.8 M | ~2.5k h, ~20 datasets; VQ tokens → masked-token prediction | 3.76 | 3.63 | 4.2% | **93.8%** | ~1% |
| REVE | `brain-bzh/reve-base` (gated) | 69.2 M | 60k h, 25k subjects; masked autoencoding | 3.61 | 3.20 | 5.6% | 87.6% | ~7% |
| CBraMod | `braindecode/cbramod-pretrained` | 4.9 M | varied | 3.32 | 2.96 | 26.7% | 68.0% | 5.3% |
| BIOT | `braindecode/biot-pretrained-six-datasets-18chs` | 3.2 M | 6 datasets, few k h | 2.48 | 2.13 | 44.0% | 52.0% | 4.0% |
| BENDR | `braindecode/braindecode-bendr` | **157 M** | ~3k h, Conv1D + transformer | **2.10** | 1.79 | **68.0%** | 24.0% | 8.0% |
| ZUNA | `mhough/zuna-base` | 380 M | masked-diffusion autoencoder, 32-d latent | — | — | — | — | — |

**Key regularities:**

- **Parameter count does not predict training quality.** BENDR is the largest model and the worst-trained. LaBraM is among the smallest yet has the highest health fraction.
- **α-mean roughly tracks pretraining hours × quality.** LaBraM is the unexplained outlier: it reaches REVE-level α with about 10× less data.
- **Access differs by checkpoint.** REVE requires accepting a HuggingFace Responsible Use Agreement. LaBraM and ZUNA load without a token.

## Benchmarking: the nanoeeg evaluation contract

`nanoeeg` pretrains a small EEG-FM under a fixed compute cap. It scores the model on a **frozen** probe suite whose content hash (`nanoeeg/frozen/seal.py`) invalidates any run that edits it.

### Probe suite (11 slots / 14 sub-probes)

The suite has **8 probes inherited from OpenEEGBench (OEB)**, with splits transcribed verbatim from OEB commit `3e4d034a…` (BSD-3): `bcic2a`, `bcic2020_3`, `physionet`, `chbmit`, `isruc_sleep`, `faced`, `mdd_mumtaz2016`, `arithmetic_zyma2019`.

OEB ships 12 datasets. An unauthenticated HF check on 2026-07-26 found 8 public (HTTP 200) and 4 gated (HTTP 401: `tuab`, `tuev`, `seed-v`, `seed-vig`) (`nanoeeg:docs/DECISIONS.md` D1). The gated sets are replaced by **3 OpenNeuro/NEMAR probes**:

| Replaces | Accession (version) | N | Licence | Axis |
|---|---|---:|---|---|
| TUAB | `ds004504` (1.0.9) | 88 | CC0 | 3-way AD / FTD / healthy control |
| TUEV | `ds006018` (1.2.2) | 127 | CC0 | ERP decoding, 4 paradigms → 4 sub-probes, **1 slot** |
| SEED-V | `ds005505` (1.0.1) | 136 | CC-BY-SA 4.0 | pediatric (HBN Release 1) |

Design notes on `ds006018` and the suite (`nanoeeg:docs/DECISIONS.md` D2–D3):

- **Why one slot, not four.** As 4 separate slots, `ds006018` would carry 4/14 ≈ 29% of the score. The four paradigms therefore share one subject partition and are averaged into a single slot.
- **No regression probe.** `seed-vig` (continuous PERCLOS vigilance) was the only regression target in OEB. The suite is now all-classification, and this is flagged as an open problem.

### Scoring rules (`nanoeeg/frozen/scoring.py`)

| Quantity | Definition |
|---|---|
| `mean_probe_score` | Mean Cohen's κ over 11 slots. This is the primary rank. |
| `identity_free_score` | The same mean after subject-identity erasure (LEACE). Published as a co-equal column. |
| Metrics per probe | κ **and** balanced accuracy are both required. OEB's leaderboard mixes the two under identical column names. |
| Leader threshold | `2 × pooled seed SD`, pooled as RMS across sub-probes, measured by baseline B1. This replaces nanopath's fixed `+0.006`. |
| Seeds | ≥3 per run; ≥5 on `ds004504`. |

- **Trap-climbing gate (D5).** A run whose raw score rises while its identity-free score falls is ranked but **cannot take the lead**.
- **Identity-sensitive probes.** On `ds005505`, and on `bcic2020_3` (split by run, so subjects are shared between train and test), a missing identity-free score makes the run ineligible.

### Compute accounting (`nanoeeg/frozen/accounting.py`, D6)

| Parameter | Value |
|---|---|
| Budget | 1e18 FLOPs (~1 H100-hour) |
| FLOPs/token | `6N + 12·n_layers·d_model·seq_len` (attention included) |
| Reference shape | 10M params, 12 layers, d_model 256; 4 s × 64-ch window, 64-sample patches → ~512 tokens |
| Attention overhead | ~30% beyond `6ND` |
| Tokens affordable at 10M params | 1.27e10 (not 1.7e10) → ~2.8 epochs over a 10k-hour corpus |

The conclusion is that **at 10–20M parameters the corpus binds before the FLOPs do.** Data curation is therefore the highest-leverage editable surface.

Baseline **B2** (frozen public checkpoint + linear probe) tests the project's premise. If a frozen checkpoint beats anything trainable within the cap, the project re-scopes toward adaptation (D7).

### Montage harmonisation (D8, not yet implemented)

The default is the **19-channel standard 10-20 subset** (see [coordinate-systems.md](coordinate-systems.md)). This is deliberately lossy. Channel-agnostic architectures that exploit the full montage count as leaderboard headroom.

## The identity trap

Subject identity is strongly encoded in EEG. A probe can therefore score well by recognising *who* is recording rather than decoding *what* construct is present.

- **FMScope audit.** `emeg-fm` vendors FMScope (`emeg-fm:fmscope/`), which implements the five frozen-representation diagnostics of *"The Identity Trap in EEG Foundation Models"* (arXiv 2606.06647).
- **Commensurability blocker.** nanoeeg's identity-free numbers must match FMScope's LEACE implementation to be comparable with existing results. Reimplementing LEACE from its closed form would break that comparability (`nanoeeg:docs/STATE.md`, Blocker 1).

## Data-free and activation-side diagnostics

### HTSR / WeightWatcher

For each weight matrix W, WeightWatcher fits a power-law tail exponent α to the eigenvalue spectrum of WᵀW, using `analyze(min_evals=50)` (`emeg-fm:docs/WEIGHTWATCHER_EEGFM_ANALYSIS.md`). The α ranges are read as:

| α range | Reading |
|---|---|
| α < 2 | Under-trained, close to random init; a good LoRA target |
| 2 ≤ α ≤ 6 | Healthy heavy-tailed regime; α ≈ 2 is Martin's self-averaging / RG balance point |
| α > 6 | Over-parameterised; possibly safe for low-rank factorisation (open question) |

The **participation count** is `M_tr = (Σλ)²/Σλ²`. The same statistic computed on *activations* (the participation ratio, PR) predicts sparse autoencoder (SAE) yield. This is the activation-side analogue of the eigen-spectrum measures in [method-spectral-analysis.md](method-spectral-analysis.md).

### REVE per-block findings

- **All 5 α<2 matrices are attention, none feed-forward:**
  - `layers.2.to_qkv` α = 1.56
  - `layers.1.to_qkv` α = 1.58
  - `layers.0.to_out` α = 1.72
  - `layers.3.to_qkv` α = 1.81
  - `layers.10.to_out` α = 1.92
- **This independently recovers REVE's published recipe** of linear probe + LoRA on QKVO, and localises it to blocks 0–3 and block 10.
- **Block 5 `to_out` has α = 9.63**, the over-parameterised outlier.
- **Block 6** (α-mean 3.63, 100% healthy) is the SAE extraction target. The guideline is to use the *earliest* block that WeightWatcher calls healthy.

### Sparse autoencoders (`emeg-fm:emeg_fm/sae.py`)

The SAE is a pure-JAX TopK autoencoder with Gao-2024 `aux_k` dead-feature revival (see [jax-ecosystem.md](jax-ecosystem.md)).

**Typical configuration:**
- K = 32, `d_dict` = 4096, `aux_k` = 32.
- Input: HBN RestingState windows at 200 Hz.

**Workflow:**
1. Train the SAE on block activations.
2. Probe each SAE feature against HBN clinical concepts: p-factor, internalizing, externalizing, attention, age, and sex.

**Results by model:**

| Model / layer | Activation PR | SAE outcome |
|---|---:|---|
| REVE (wide, contrastive/MAE) | ≈ 4.9 | Viable; structured single-feature clinical signal |
| ZUNA (32-d latent bottleneck) | ≈ 1.7 | ~99% dead features. This is correct behaviour, because activations are ≈ rank-2. |
| REVE block 11 (7.4 M tokens, d_dict 2048, k 32) | — | 81% dead, EV = 1.000, ~370 effective dims of 512 |

**The "block-11 paradox."** Block 11 weights are healthy (α-mean 3.46), yet its activations are rank-collapsed. The two diagnostics answer different questions: weight α is a static property of the trained matrices, while activation rank reflects the data flowing through them. Rule: **measure activation PR before sizing a dictionary.**

## Adaptation: LoRA and optimizer choice

Data-free α-ranking and SAE-suitability rank the models in **inverse** order:

| Use case | Ranking (best → worst) |
|---|---|
| SAE interpretability | LaBraM, LUNA, REVE, CBraMod, BIOT, BENDR |
| LoRA yield | BENDR, BIOT, CBraMod, REVE, LUNA, LaBraM |

`emeg-fm:docs/MUON_EXPERIMENT.md` tests Martin's RG prediction that a gradient-orthogonalising optimizer (Muon, which uses a Newton–Schulz step) pushes layers toward α ≈ 2. Each hypothesis compares Muon against AdamW:

- **H1:** a tighter α distribution near 2.
- **H2:** a higher activation PR.
- **H3:** a higher SAE yield.

The test is run at three scopes:

| Scope | Setup | Status |
|---|---|---|
| A | Muon on the SAE itself (`train_sae.py --optimizer muon`); MUON_LR 2e-2 vs ADAM_LR 1e-3 | Scaffolded. Adam baseline: EV 0.974, 75% dead |
| C | LoRA on REVE's 5 α<2 attention matrices, fine-tuned on the HBN bifactor probe | Scaffolded |
| B | From-scratch LaBraM-style encoder | Parked (multi-GPU-day) |

**Controls:**
- Use a Muon/AdamW hybrid: Muon only on 2-D matrices, AdamW on biases, norms, and the head.
- Sweep learning rates per optimizer.
- Use the bit-identical `weight_spectral_summary` α fit (Hill-estimator proxy, relative A/B only).

## Downstream application: channel imputation

`zuna-eeg-pipeline` benchmarks ZUNA's diffusion-based reconstruction against MNE spherical-spline interpolation:

- **Model:** ZUNA, 380 M parameters.
- **Data:** the Wakeman–Henson simultaneous [MEG](meg.md)/EEG dataset `ds000117`.
- **Protocol:** **25% of EEG channels are dropped** before reconstruction.

`emeg-fm` also targets frozen-EEG-FM → image retrieval on Alljoined-1.6M (32-channel consumer EEG). This is the EEG analogue of the MindEye [fMRI](fmri.md) demo.

## Practical gotchas

**Data access** (`emeg-fm:docs/EEGDASH_ISSUES.md`; eegdash 0.7.2, braindecode 1.5.1, mne 1.12.1):
- `EEGChallengeDataset` routes to non-existent `EEG2025r*` repositories. Use `RELEASE_TO_OPENNEURO_DATASET_MAP` instead (R1 = `ds005505`).
- HBN R7 (`ds005511`) and R11 (`ds005516`) return 0 subjects from the index, even though S3 holds 741 and 567 RestingState files respectively. Use anonymous boto3 against `s3://openneuro.org/`.
- A single `DataIntegrityError` kills the whole joblib preprocessing pass. Pre-validate recordings before preprocessing.
- `_ensure_raw()` caches every Raw object, causing OOM (~150 MB/subject). Set `rec._raw = None` after each load.
- HBN GSN-128 `.set` files load with channel type `misc`, so `pick_types(eeg=True)` returns nothing. See [data-formats.md](data-formats.md).

**Checkpoint loading:**
- CBraMod and LUNA checkpoints do not match the default constructor (CBraMod uses `d_model` 576, not 256). Use a state-dict wrapper.
- BENDR needs its `weight_norm` parametrisations removed before WeightWatcher can read the weights.

**Hardware** (see [compute-infrastructure.md](compute-infrastructure.md)):
- On DGX Spark (Blackwell sm_121), set `TORCHDYNAMO_DISABLE=1`.
- Patch ZUNA's vendored DataLoader `in_order=False` kwarg.
- Cap `num_workers` at 4 to avoid OOM (`zuna-eeg-pipeline:CLAUDE.md`).
- On NSG V100s, pin `torch==2.4.1` (cu121) before installing peft/transformers (`nanoeeg:docs/STATE.md`).

## Relevant Projects

- **`nanoeeg`**: compute-capped from-scratch EEG-FM pretraining. Provides the frozen 11-slot probe suite, the κ + balanced-accuracy scorer, the identity-free gate, and FLOP accounting. Status: M1 scaffold, 35 tests passing.
- **`emeg-fm`**: E/MEG-FM diagnostics. Covers WeightWatcher/HTSR, TopK SAEs in JAX, the FMScope identity audit, LoRA + Muon experiments, and HBN extraction scripts.
- **`zuna-eeg-pipeline`**: ZUNA 380M channel-imputation benchmark on `ds000117` against spherical splines.
- **`hippy-feat`**: hosts the original WeightWatcher analyzer scripts and the MindEye real-time decoding counterpart.

## See Also

- [Neuroimaging Foundation Models](foundation-models.md)
- [Masked Autoencoding for Brain Data](method-masked-autoencoding.md)
- [HT-SR Weight-Spectrum Analysis](htsr-weight-analysis.md)
- [Foundation-Model Diagnostics](fm-diagnostics.md)
**Modalities:**
- [eeg.md](eeg.md)
- [meg.md](meg.md)
- [fmri.md](fmri.md)

**Related methods:**
- [method-spectral-analysis.md](method-spectral-analysis.md)
- [method-pharmaco-eeg.md](method-pharmaco-eeg.md)
- [method-hmm-dynamics.md](method-hmm-dynamics.md)
- [method-sbi.md](method-sbi.md)

**Infrastructure:**
- [data-formats.md](data-formats.md)
- [coordinate-systems.md](coordinate-systems.md)
- [compute-infrastructure.md](compute-infrastructure.md)
- [jax-ecosystem.md](jax-ecosystem.md)
