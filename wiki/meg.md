---
type: modality
title: Magnetoencephalography
physics: electromagnetic
measurement: extracranial magnetic flux density (fT–pT) produced by intracellular primary currents of synchronously active cortical pyramidal neurons
spatial_resolution: 2-10 mm (source space, after inverse modeling); ~2-3 cm sensor spacing
temporal_resolution: <1 ms (sampling 600-12000 Hz)
related: [eeg.md, method-source-imaging.md, method-bem.md, method-fem.md, physics-electromagnetic.md, tissue-electrical-properties.md, tissue-skull.md, neural-mass-models.md, method-hmm-dynamics.md, method-spectral-analysis.md, coordinate-systems.md, data-formats.md, structural-mri.md, sci-head-model.md, head-model-mida.md, method-eeg-foundation-models.md, method-pharmaco-eeg.md, connectomics.md, fmri.md, neuronal-current-mri.md, jax-ecosystem.md]
description: "Magnetoencephalography (MEG) measures the magnetic fields produced by primary (intracellular) currents in the apical dendrites of synchronously active pyramidal neurons."
timestamp: 2026-11-05T15:59:53-08:00
tags: [modality, electromagnetic, electrophysiology]
---

# Magnetoencephalography

Magnetoencephalography (MEG) measures the magnetic fields produced by **primary (intracellular) currents** in the apical dendrites of synchronously active pyramidal neurons. A detectable field requires roughly 10⁴–10⁵ co-aligned neurons. This corresponds to a net current dipole moment of about 10–100 nAm. The resulting fields at the scalp are 10–1000 fT. For comparison, the Earth's static field is about 50 µT, roughly 10⁸–10⁹ times larger. MEG shares its generators and inverse problem with [eeg.md](eeg.md). It differs from EEG in sensitivity profile, in its dependence on head-model conductivities, and in its hardware.

## Physics

Under the quasi-static approximation of Maxwell's equations ([physics-electromagnetic.md](physics-electromagnetic.md)), the field outside the head follows from the Biot–Savart law applied to the total current, J = Jᵖ − σ∇V:

```
B(r) = (μ₀/4π) ∫ J(r') × (r − r') / |r − r'|³ d³r'
```

Two consequences follow for a **spherically symmetric conductor**:

1. **Radial dipoles produce no external field.** The volume currents exactly cancel the primary-current contribution. MEG is therefore dominated by tangential sources in sulcal walls, whereas EEG sees both orientations, with a bias toward gyral crowns.
2. **The external field does not depend on the conductivity profile.** The Sarvas (1987) closed form gives B directly from the dipole moment q at position r_q:
   ```
   B(r) = μ₀/(4πF²) · [F (q × r_q) − ((q × r_q)·r) ∇F],
   F = a(ra + r² − r_q·r),   a = r − r_q,  a = |a|
   ```
   Skull conductivity (see [tissue-skull.md](tissue-skull.md)) therefore matters far less for MEG than for EEG. In realistic geometries a **single-compartment (inner-skull) BEM** is usually sufficient for MEG. EEG needs a 3-layer scalp/skull/brain model ([method-bem.md](method-bem.md)).

Field magnitude falls off steeply with distance from the source, at roughly 1/r² to 1/r³ for a dipole. This produces a strong **depth bias**: MEG is most sensitive to superficial cortex and only weakly sensitive to deep structures such as the hippocampus and thalamus.

## Sensor Technology

| Parameter | Typical value | Notes |
|---|---|---|
| Sensor | SQUID (low-T_c, ~4 K liquid helium) | Optically pumped magnetometers (OPMs) are the wearable, cryogen-free alternative |
| Channel count | 275 (CTF), 306 (MEGIN/Neuromag: 102 magnetometers + 204 planar gradiometers) | WAND uses CTF 275-ch [@mcnabb2025wand] |
| CTF sensor geometry | First-order axial gradiometers, 5 cm baseline | Third-order synthetic gradiometry uses reference channels |
| SQUID white-noise floor | ~3-5 fT/√Hz above ~1 Hz | Rises below ~1 Hz (1/f) |
| OPM noise floor | ~7-15 fT/√Hz | Sensors sit a few mm from the scalp, which raises signal amplitude |
| Sampling rate | 600-2400 Hz typical (CTF up to 12 kHz) | |
| Sensor-to-cortex distance | ~3-4 cm (SQUID helmet) | |
| Shielding | Magnetically shielded room (mu-metal + Al) | Shielding factor 10³–10⁵ at low frequency |

## Typical Signal Amplitudes

| Signal | Amplitude at sensor | Notes |
|---|---|---|
| Spontaneous posterior alpha (8-12 Hz) | ~1 pT | Largest spontaneous rhythm |
| Sensory evoked fields (e.g., N100m/M100) | ~100-500 fT | |
| Cortical gamma (30-100 Hz) | ~10-50 fT | Near the noise floor of single trials |
| Cardiac field (MCG) | ~50-100 pT | Artifact source |
| Urban environmental noise | nT–µT | Removed by shielding and gradiometry |

## Forward Model

The measurement model is linear:

```
B = L · J + n,     B ∈ ℝ^(C×T),  L ∈ ℝ^(C×S),  J ∈ ℝ^(S×T)
```

Each column of the leadfield L is the sensor field produced by a unit dipole at one source location. In neurojax this appears as `Y = L·J + n` across the head-modeling → source-imaging pipeline (neurojax:README.md). Building L requires four ingredients:

1. **Anatomy.** A T1w MRI is segmented, either by FreeSurfer surfaces or by CHARM 60-tissue segmentation in neurojax ([structural-mri.md](structural-mri.md)).
2. **Conductor model.** Options are a single-shell or 3-layer BEM (MNE or OpenMEEG; [method-bem.md](method-bem.md)) or differentiable FEM (JAX-FEM + PETSc; [method-fem.md](method-fem.md)). FEM allows anisotropic conductivity from DTI and qMRI-informed σ(T1, BPF) (neurojax:README.md). Detailed head models include [sci-head-model.md](sci-head-model.md) and [head-model-mida.md](head-model-mida.md).
3. **Source space.** A cortical mesh with roughly 4-8 mm vertex spacing, typically 5,000–10,000 vertices per hemisphere. Dipoles may be orientation-free or constrained to the surface normal.
4. **Coregistration.** MEG head coordinates are defined by the nasion and left/right pre-auricular fiducials (CTF: +x toward nasion, +y toward left). They are aligned to MRI space using the digitized head shape ([coordinate-systems.md](coordinate-systems.md)). In WAND, MEG (ses-01) is registered to the ses-02 T1w anchor via MNE coregistration (neurojax:docs/BIDS_DERIVATIVES_STRUCTURE.md).

Because neurojax's forward chain is fully differentiable, `jax.grad` can propagate a source-space loss back through the leadfield and FEM assembly into qMRI tissue parameters (neurojax:README.md; [jax-ecosystem.md](jax-ecosystem.md)).

## Source Reconstruction

The inverse problem is severely underdetermined: about 275 sensors must explain roughly 8,000–20,000 sources. neurojax implements 15 solvers in five families (neurojax:README.md; see [method-source-imaging.md](method-source-imaging.md)):

| Family | Methods | Key idea |
|---|---|---|
| L2 minimum norm | MNE, dSPM, sLORETA, eLORETA | Tikhonov-regularized distributed estimate. dSPM and sLORETA noise-normalize the estimate. eLORETA gives zero localization error for single dipoles in noise-free data |
| Biophysical prior | LAURA, VARETA | LAURA uses a 1/d³ spatial decay prior. VARETA uses adaptive spatial resolution |
| Beamformers | LCMV, SAM, DICS | Data-adaptive spatial filters. SAM is a scalar time-domain variant. DICS is a frequency-domain (coherence) variant |
| Bayesian sparse | CHAMPAGNE, HIGGS | Type-II maximum likelihood / ARD. HIGGS jointly estimates sources and their connectivity |
| Deep learning | PI-GNN | Physics-informed graph convolution on the cortical mesh, with vertex features (normals, curvature, myelin) |

neurojax enforces explicit regularization as a project convention. It uses Tikhonov regularization rather than hard SVD rank cutoffs, and λ is always reported, for example via `estimate_tikhonov_reg(leadfield)` (neurojax:CLAUDE.md). The project's autoresearch loop benchmarks differentiable against classical inverses using the metric `source_advantage = (DLE_classical − DLE_neurojax)/DLE_classical`. DLE is dipole localization error, evaluated on simulated dipoles, WAND, and Wakeman–Henson data (neurojax:autoresearch/program.md).

## Preprocessing

| Step | Typical setting | Notes |
|---|---|---|
| Interference suppression | Synthetic 3rd-order gradiometry (CTF); SSS/tSSS (MEGIN) | SSS reduces data rank to ~64-80, which downstream covariance methods must account for |
| Filtering | High-pass 0.1-1 Hz; low-pass 40-150 Hz; notch at 50/60 Hz and harmonics | neurojax uses a `lax.scan` DF-II-T IIR filter (neurojax:docs/BUG_REPORTS.md #5) |
| Artifact removal | ICA (blinks, cardiac); ASR | neurojax:preprocessing |
| Head-motion check | Continuous head-localization coils | Displacements of a few mm or more degrade localization |
| Noise covariance | Empty-room recording or pre-stimulus baseline | Needed for dSPM, sLORETA, and beamformer whitening |

## Downstream Analysis

- **Covariance / Riemannian pipelines.** Band-limited sensor covariances, for example over IPEG-defined bands such as delta, theta, and alpha1, are projected to handle rank deficiency and field spread. They are then vectorized, either by tangent-space Riemannian embedding or log-variance, and passed to a regressor. Sabbagh et al. showed that Riemannian embedding of MEG covariances approximates source-power models without requiring a forward model [@sabbagh2020predictive]. Implemented in coffeine (`compute_coffeine`, `make_filter_bank_regressor`).
- **Brain-state dynamics.** Source time courses are parcellated to Desikan (T×68) or Destrieux (T×148) regions. They are then modeled with an HMM (outputs: γ, state means and covariances, fractional occupancy, lifetimes, switching rate) or with DyNeMo, followed by state-wise PSD and coherence. These outputs make up the neurojax-meg WAND derivatives (neurojax:docs/BIDS_DERIVATIVES_STRUCTURE.md; [method-hmm-dynamics.md](method-hmm-dynamics.md)). Time-delay embedding before the HMM follows osl-dynamics (neurojax:docs/DESIGN_PROPOSAL_BIDS_DYNAMICS.md).
- **Data-driven dynamical systems.** neurojax provides SINDy [@brunton2016discovering], Koopman/DMD operators [@schmid2010dynamic; @brunton2022modern], and specparam aperiodic/periodic fitting ([method-spectral-analysis.md](method-spectral-analysis.md), [method-pharmaco-eeg.md](method-pharmaco-eeg.md)).
- **Differentiable connectivity.** hippy-feat's `jaxoccoli` modules are modality-agnostic. They include `matrix.py` (SPD tangent projection, Fréchet mean), `fourier.py` (analytic signal, amplitude envelope, instantaneous phase), and `connectivity.py` (sliding-window FC). These apply directly to MEG amplitude-envelope connectivity [@ciric2022differentiable; @hough2026hippyfeat]; see [connectomics.md](connectomics.md).

## Generative Forward Models of the MEG Signal

Biophysical simulators produce dipole moment time series, Q(t) in nAm, which are mapped to sensors through L. The main examples are:

- **hnn-jax.** A JAX port of the Human Neocortical Neurosolver. It models a laminar neocortical column, with L2/3 and L5 pyramidal cells plus interneurons, driven by proximal and distal thalamocortical inputs. The output is the net intracellular current dipole along the apical dendrite axis, which can be fit to evoked MEG/EEG responses.
- **vbjax / neurojax `bench/`.** Neural-mass models such as reduced Wong–Wang and Wendling are coupled through a connectome and projected by `bench/` leadfield adapters ([neural-mass-models.md](neural-mass-models.md)). For the reduced Wong–Wang model, the correct gains are γ_E = 0.641 and γ_I = 1.0 (in 1/s with τ in seconds). These recover the Deco 2013 fixed point S_E ≈ 0.164 (neurojax:docs/BUG_REPORTS.md #3).

Differentiable simulators combined with amortized inference ([method-sbi.md](method-sbi.md)) allow posterior estimation over circuit parameters from MEG waveforms.

## Properties

| Property | Value | Notes |
|---|---|---|
| Temporal resolution | <1 ms | Direct measurement of neural currents; no hemodynamic lag (compare [fmri.md](fmri.md)) |
| Spatial resolution (sensor) | ~2-3 cm | Set by sensor spacing and distance |
| Spatial resolution (source) | ~2-5 mm superficial single dipole; 5-10+ mm distributed | Depends on SNR, inverse method, and head-model accuracy |
| Orientation sensitivity | Tangential ≫ radial | Radial sources are silent in a sphere |
| Depth sensitivity | Strongly superficial-weighted | Poor for deep sources |
| Useful band | ~0.1-150 Hz (DC-1 kHz recordable) | |
| Conductivity dependence | Weak (single-shell BEM adequate) | Contrast: EEG depends strongly on skull σ ([tissue-electrical-properties.md](tissue-electrical-properties.md)) |
| Data formats | CTF `.ds`, MEGIN `.fif`, BIDS-MEG | [data-formats.md](data-formats.md) |

## Advantages and Limitations Relative to EEG

**Advantages**
- Reference-free measurement.
- Much smaller skull-conductivity uncertainty.
- Less spatial smearing from volume conduction.
- Rapid whole-head coverage without electrode preparation.

**Limitations**
- Blind to radial sources.
- Poorer than EEG for deep sources.
- SQUID systems need cryogenics, a shielded room, and a fixed helmet. Head movement relative to the sensors degrades localization. OPMs address the helmet-fit and portability constraints.
- High cost limits availability. EEG is far more common, which is why foundation models are mostly EEG-trained ([method-eeg-foundation-models.md](method-eeg-foundation-models.md)).

## Datasets in Use

- **WAND** (CUBRIC): 170 subjects. CTF 275-ch MEG in ses-01, alongside 7T MRI, Connectom 300 mT/m dMRI, MRS, and TMS-EEG [@mcnabb2025wand]. Loaded via neurojax `io/` (`WANDMEGLoader`).
- **Wakeman–Henson**: multimodal M/EEG face-processing dataset. neurojax has a loader for it in `io/`.

## Relevant Projects

- **neurojax**:
  - CTF/WAND MEG loaders and preprocessing (ASR, ICA, IIR filters).
  - Differentiable BEM/FEM leadfields.
  - 15 inverse solvers.
  - HMM/DyNeMo dynamics, SINDy/Koopman, specparam.
  - MEG pipeline tutorial covering raw CTF data through source-level HMM (`docs/tutorials/meg_pipeline.md`).
- **coffeine**: filter-bank covariance DataFrames and Riemannian regression for M/EEG prediction [@sabbagh2020predictive].
- **hippy-feat**:
  - `jaxoccoli` SPD-manifold, analytic-signal, and differentiable connectivity tools, shared with MEG envelope/covariance analysis.
  - `hf_encoder.py` adapter pattern for M/EEG foundation models.
  - Primarily an fMRI project.
- **hnn-jax**: laminar neocortical-column simulator producing MEG/EEG current dipoles for model-based inference.

## See Also

- [eeg.md](eeg.md) — complementary electric-potential modality with the same generators
- [method-source-imaging.md](method-source-imaging.md) — inverse solvers in detail
- [method-bem.md](method-bem.md), [method-fem.md](method-fem.md) — forward modeling
- [physics-electromagnetic.md](physics-electromagnetic.md) — quasi-static Maxwell equations
- [tissue-electrical-properties.md](tissue-electrical-properties.md), [tissue-skull.md](tissue-skull.md) — conductivities (critical for EEG, secondary for MEG)
- [neural-mass-models.md](neural-mass-models.md), [method-neural-fields.md](method-neural-fields.md) — generative source models
- [method-hmm-dynamics.md](method-hmm-dynamics.md), [method-spectral-analysis.md](method-spectral-analysis.md) — downstream dynamics
- [coordinate-systems.md](coordinate-systems.md) — CTF/Neuromag head frames and coregistration
- [neuronal-current-mri.md](neuronal-current-mri.md) — MRI-based detection of the same neural currents
- [sci-head-model.md](sci-head-model.md), [head-model-mida.md](head-model-mida.md) — high-resolution head models
