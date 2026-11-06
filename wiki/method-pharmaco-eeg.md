---
type: method
title: "Pharmaco-EEG: Spectral Parameterization and Drug Effects"
category: spectral
implementations:
  - neurojax:analysis (spectral, specparam)
  - neurojax:models (HMM state spectra)
  - neurojax:examples/dmt_observations.py
  - vpjax:examples/dmt_shared_drive.py
related:
  - eeg.md
  - meg.md
  - method-spectral-analysis.md
  - method-hmm-dynamics.md
  - neural-mass-models.md
  - method-neural-fields.md
  - mrs.md
  - tms.md
  - method-sbi.md
  - method-variational-inference.md
  - method-eeg-foundation-models.md
  - physics-cerebrovascular.md
  - physics-hemodynamic.md
  - fmri.md
  - jax-ecosystem.md
description: "Drug effects are then expressed as shifts in these physically interpretable parameters."
timestamp: 2026-11-05T15:59:53-08:00
tags: [method, spectral, eeg, pharmacology]
---

# Pharmaco-EEG: Spectral Parameterization and Drug Effects

**Pharmaco-EEG** quantifies how centrally acting drugs change the electrical activity of the brain. Its main tool is the power spectral density (PSD) of resting or task [EEG](eeg.md) and [MEG](meg.md). The modern approach does not use fixed canonical bands (delta/theta/alpha/beta/gamma). Instead it **parameterizes** each spectrum into two parts:

- an **aperiodic** (1/f-like) component;
- a set of **periodic** oscillatory peaks.

Drug effects are then expressed as shifts in these physically interpretable parameters. In this knowledge base, pharmaco-EEG connects two layers:

- **Spectral modelling of electrophysiology** ([neurojax](#relevant-projects)).
- **State-space models of physiology and autonomic signals** ([vpjax](#relevant-projects)). These share a latent "drive" with EEG.

## Why parameterize instead of band-power?

Canonical band power mixes two things: changes in oscillation amplitude and changes in the aperiodic background. Suppose a drug steepens the 1/f slope. Low-frequency band power then appears to rise and high-frequency band power appears to fall, even with no change in any oscillation. Parameterization separates these effects so that:

1. **Peak shifts** (e.g. alpha centre frequency slowing) are measured directly.
2. **Broadband changes** (aperiodic exponent and offset) are reported separately. These are often interpreted as excitation/inhibition balance, although that interpretation is model-dependent.
3. **Spurious band effects** from background tilt are avoided.

## Spectral model (specparam / FOOOF form)

neurojax lists `specparam` among its dynamics tools (neurojax:README.md, Capabilities table). The standard model fits the log-power spectrum as:

$$
\log_{10} P(f) = L(f) + \sum_{n=1}^{N} G_n(f)
$$

**Aperiodic component.** The knee is optional:

$$
L(f) = b - \log_{10}\!\left(k + f^{\chi}\right)
$$

**Gaussian peaks.**

$$
G_n(f) = a_n \exp\!\left(-\frac{(f - c_n)^2}{2 w_n^2}\right)
$$

### Parameters

| Symbol | Name | Units | Typical adult resting EEG range | Pharmacological interpretation |
|---|---|---|---|---|
| $b$ | Aperiodic offset | log10(µV²/Hz) | Montage/reference dependent | Broadband power shift (e.g. overall population firing) |
| $\chi$ | Aperiodic exponent | dimensionless | ~1–2 over 1–40 Hz | Steeper (larger χ) is often linked to increased inhibition; flatter to increased excitation or noise |
| $k$ | Knee | Hz^χ | 0 in "fixed" mode; fitted in "knee" mode for wide ranges | Knee frequency $f_k = k^{1/\chi}$ relates to a dominant time constant |
| $c_n$ | Peak centre frequency | Hz | Alpha ~8–12 Hz; beta ~13–30 Hz | Peak slowing/acceleration (e.g. alpha slowing under sedation) |
| $a_n$ | Peak power above aperiodic | log10 power | 0.1–1.5 | Oscillation strength independent of the background |
| $w_n$ | Peak SD (bandwidth ≈ 2w) | Hz | Bandwidth limits typically 1–8 Hz | Rhythm regularity |

*Note:* the ranges above are conventional specparam settings and commonly reported literature magnitudes. They are not values extracted from a `raw/` source in this KB. Add citations when a source paper is ingested.

### Fitting settings that must be reported

The neurojax convention is "report all regularisation choices explicitly — never hide thresholds" (neurojax:CLAUDE.md). For spectral parameterization, that means always logging:

- **Frequency range.** Typically 1–40 Hz for EEG. A wider range, e.g. 1–100 Hz for MEG, requires knee mode.
- **Aperiodic mode.** Fixed vs knee.
- **Peak width limits, max number of peaks, minimum peak height and peak threshold.**
- **PSD estimator.** Welch window length, overlap and taper; see [method-spectral-analysis.md](method-spectral-analysis.md).
- **Goodness of fit.** R² and error for each fit. Exclude fits with poor R² rather than silently keeping them.

Because the model is differentiable, a JAX implementation can fit (b, χ, k, {a, c, w}) by gradient descent. It can also embed the fit inside larger graphs, for example as a loss on simulated spectra from [neural-mass-models.md](neural-mass-models.md). See [jax-ecosystem.md](jax-ecosystem.md).

## Typical drug-class spectral signatures (qualitative)

| Drug class | Example | Commonly reported spectral effect |
|---|---|---|
| GABA-A positive modulators | Benzodiazepines | Increased beta (~13–30 Hz) power; reduced alpha |
| GABAergic anaesthetics | Propofol | Steeper aperiodic exponent; frontal alpha emergence at loss of consciousness; increased slow/delta power |
| NMDA antagonists | Ketamine | Increased gamma power; reduced alpha; flatter aperiodic component |
| Serotonergic psychedelics | DMT, psilocybin, LSD | Reduced alpha power and broadband power decrease |

These directions are general literature knowledge and carry no bibliography key in this KB. Treat them as hypotheses to test with parameterized spectra, not as calibrated effect sizes.

## State-space linkage: shared drive across EEG and autonomic signals (vpjax)

vpjax contains a first inference example for a DMT dataset [@damelio2026multimodal]. The dataset (Zenodo DOI 10.5281/zenodo.19893951) has **no fMRI**. The workflow (vpjax:docs/tutorials/dmt_shared_drive.md) has three steps:

1. **Export.** neurojax `examples/dmt_observations.py` exports baseline-standardized observations to `observations.npz`.
2. **Decouple.** vpjax consumes that file without a neurojax dependency.
3. **Fit.** A deliberately reduced linear-Gaussian latent model is fitted:

$$
y_k(t) = \lambda_k\, x(t) + \varepsilon_k, \qquad x(t+\Delta t) = x(t) + \eta, \quad \eta \sim \mathcal{N}(0, \sigma_p^2 \Delta t)
$$

Inference uses scalar **Kalman filtering + Rauch–Tung–Striebel smoothing** on the union of sensor timestamps and the output grid:

- Missing observations are skipped, which increases conditional uncertainty.
- No interpolation is used to fabricate samples.
- The prior mean is 0 with variance `prior_sd²`.

### Illustrative assumptions (from the vpjax tutorial)

| Parameter | Value | Note |
|---|---|---|
| `loadings.heart_rate` | 1.0 | Fixed, signed |
| `loadings.smna` | 1.0 | Sympathetic nerve activity proxy |
| `loadings.rvt` | 1.0 | Positive RVT loading is a *hypothesis*, not a requirement |
| `noise_sd` (each sensor) | 1.0 | Must be explicit and positive |
| `process_sd` | 0.1 | Random-walk diffusion per √s |
| `prior_sd` | 1.0 | Prior on x(0) |

**Where EEG fits.** EEG can enter this model as another baseline-standardized scalar observation. Natural candidates are a parameterized feature such as alpha peak power $a_\alpha$ or the aperiodic exponent χ, each with an explicitly chosen signed loading and noise.

The tutorial is explicit about what this does and does not mean:

- Including EEG *asserts* a shared-drive relationship. It does not establish a causal EEG–autonomic mechanism.
- The latent $x(t)$ is an **uncalibrated shared drive**. It is not an identified sympathetic, vagal or receptor-occupancy state.
- Posterior uncertainty is conditional on the assumed loadings and noise. It excludes parameter, preprocessing and model uncertainty.

### Validation guidance

- Compare alternate loading signs and sensor subsets on **held-out sessions**.
- Do not count overlapping samples as independent participants.
- Benchmark against the dataset authors' PCA arousal index.

### Explicitly not yet implemented

The vpjax tutorial lists the following as subsequent model extensions:

- sensor response delays;
- respiratory gas exchange;
- a coupled pressure-generating circulation (Diamond/SimCVR);
- **drug PK/PD**;
- neural-to-BOLD coupling.

A natural PK/PD extension would replace the random walk with plasma/effect-site concentration $C_e(t)$ driving the spectral parameters through an Emax link:

$$
\theta(t) = \theta_0 + \frac{E_{\max} C_e^n}{EC_{50}^n + C_e^n}
$$

This equation is a standard pharmacodynamic form and is not yet in either codebase. Identifiability of $(E_{\max}, EC_{50}, n)$ from spectral data alone is a key open question. [method-sbi.md](method-sbi.md) and [method-variational-inference.md](method-variational-inference.md) cover posterior diagnostics for this.

## Mechanistic interpretation via generative models

Spectral parameters become mechanistically interpretable when fitted with a forward model:

- **Neural mass models** ([neural-mass-models.md](neural-mass-models.md)). Drugs map onto synaptic gains and time constants. For example, prolonged GABA-A IPSP decay should lower the alpha centre frequency and steepen χ. neurojax inherits Wendling-family cortical column models (neurojax:README.md, research lineage).
- **Unit discipline matters.** The neurojax RWW model had `gamma_E = 0.641/1000`, which drove the single-node fixed point to S_E ≈ 0.0002 instead of ~0.164. The fix was `gamma_E = 0.641`, `gamma_I = 1.0` with τ in seconds (neurojax:docs/BUG_REPORTS.md, item 3, commit `49acac2`). A drug-effect fit on such a model would be meaningless until this kind of error is fixed.
- **Neural fields** ([method-neural-fields.md](method-neural-fields.md)). These offer spatially extended spectra on connectome harmonics.
- **Brain-state models** ([method-hmm-dynamics.md](method-hmm-dynamics.md)). neurojax writes per-state spectra `state_psd.npy` (K, F, C) and coherence (K, F, C, C) to `derivatives/neurojax-meg/` (neurojax:docs/BIDS_DERIVATIVES_STRUCTURE.md). Each state PSD can be parameterized, so drug effects decompose into **changes in state occupancy** vs **changes within-state spectra**.

## Multimodal constraints (WAND)

The WAND dataset [@mcnabb2025wand] provides independent measurements that constrain pharmaco-EEG interpretation in the same subjects:

| Constraint | Session / sequence | Relevance |
|---|---|---|
| GABA/Glu concentrations | ses-05 MEGA-PRESS, TE = 68 ms, 3T; ses-04 sLASER, TE = 78 ms, 7T | Baseline inhibitory tone vs beta power / aperiodic χ; see [mrs.md](mrs.md) |
| TMS-EEG SICI | ses-08 | GABA-A-mediated cortical inhibition; see [tms.md](tms.md) |
| Resting CTF MEG (275 ch) | ses-01 | Source-level spectra and HMM states |
| pCASL / TRUST / qBOLD | Various | Vascular confounds of drug effects; see [physics-cerebrovascular.md](physics-cerebrovascular.md) |

Sources: neurojax:docs/CUBRIC_QUESTIONS.md and neurojax:docs/BIDS_DERIVATIVES_STRUCTURE.md.

## Practical pitfalls

- **Vascular and autonomic confounds.** Many drugs change heart rate, blood pressure and respiration. Cardiac pulsatility produces ~1 Hz CBV fluctuations (vpjax cardiac tutorial). This motivates the joint autonomic model above. See [physics-hemodynamic.md](physics-hemodynamic.md).
- **Muscle artefact.** EMG contaminates beta/gamma power and flattens χ above ~20 Hz. This matters especially for ketamine and psychedelics. Clean the data first (ASR/ICA in neurojax `preprocessing/`).
- **Baseline standardization.** Effects should be expressed relative to a per-session pre-drug baseline, as in the DMT export.
- **Foundation-model features.** Learned embeddings ([method-eeg-foundation-models.md](method-eeg-foundation-models.md)) can classify drug states but lack the interpretability of (b, χ, a, c, w). Probe embeddings against parameterized features.

## Relevant Projects

- **neurojax**
  - Spectral analysis and `specparam` (`analysis/`).
  - HMM/DyNeMo state spectra (`models/`).
  - Preprocessing (ASR, ICA).
  - Export of baseline-standardized DMT observations (`examples/dmt_observations.py`).
  - Description in the KB project table: "pharmaco-EEG spectral models".
- **vpjax**
  - Kalman/RTS shared-drive model over autonomic and EEG observations (`examples/dmt_shared_drive.py`).
  - Cardiac (vagal, baroreceptor, pulsatility) models.
  - Planned PK/PD and state-space identifiability.
- **vbjax / hnn-jax / gnf-jax**
  - Generative neural models whose simulated spectra can be fitted to drug-induced parameter shifts.
- **inferjax**
  - Amortized SBI for posterior calibration of drug-effect parameters.
- **nanoeeg, emeg-fm, zuna-eeg-pipeline**
  - EEG foundation models for drug-state decoding baselines.

## See Also

- [method-spectral-analysis.md](method-spectral-analysis.md): PSD estimation, multitaper, coherence
- [eeg.md](eeg.md), [meg.md](meg.md): sensor modalities
- [method-hmm-dynamics.md](method-hmm-dynamics.md): state-resolved spectra
- [neural-mass-models.md](neural-mass-models.md), [method-neural-fields.md](method-neural-fields.md): mechanistic spectral generators
- [method-sbi.md](method-sbi.md), [method-variational-inference.md](method-variational-inference.md): inference over drug-effect parameters
- [mrs.md](mrs.md): GABA/Glu constraints
- [tms.md](tms.md): TMS-EEG inhibition markers
- [physics-cerebrovascular.md](physics-cerebrovascular.md), [physics-hemodynamic.md](physics-hemodynamic.md), [fmri.md](fmri.md): vascular confounds and coupling
- [method-eeg-foundation-models.md](method-eeg-foundation-models.md): learned representations
- [jax-ecosystem.md](jax-ecosystem.md): differentiable fitting stack
