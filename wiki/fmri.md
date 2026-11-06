---
type: modality
title: Functional MRI
physics: hemodynamic
measurement: Blood oxygen level-dependent (BOLD) T2*-weighted signal changes driven by neurovascular coupling (CBF, CBV, CMRO2); optionally CBF via ASL
spatial_resolution: 1-3 mm isotropic at 3T (typically 2-2.5 mm); 0.7-1 mm at 7T for laminar fMRI
temporal_resolution: 0.5-3.0 s TR (1.5 s in hippy-feat RT pipeline); hemodynamic response limits effective resolution to ~seconds
related: [physics-hemodynamic.md, physics-cerebrovascular.md, physics-bloch.md, quantitative-mri.md, structural-mri.md, diffusion-mri.md, mrs.md, eeg.md, fnirs.md, fus.md, neural-mass-models.md, connectomics.md, method-hmm-dynamics.md, method-monte-carlo.md, method-variational-inference.md, method-spectral-analysis.md, coordinate-systems.md, data-formats.md, jax-ecosystem.md]
description: "Functional MRI (fMRI) measures brain activity indirectly through the blood oxygen level-dependent (BOLD) contrast."
timestamp: 2026-11-05T15:59:53-08:00
tags: [modality, hemodynamic, mri]
---

# Functional MRI

Functional MRI (fMRI) measures brain activity indirectly through the **blood oxygen level-dependent (BOLD)** contrast. Deoxyhemoglobin is paramagnetic; when neural activity triggers a cerebral blood flow (CBF) increase that exceeds the rise in oxygen consumption (CMRO2), venous deoxyhemoglobin falls, local field inhomogeneity drops, and T2*-weighted signal rises by a few percent. fMRI is therefore a **hemodynamic** modality. It shares its vascular physics with [fnirs.md](fnirs.md) and [fus.md](fus.md) and its spin physics with the rest of the MRI family ([physics-bloch.md](physics-bloch.md)).

In this project family, fMRI work covers four areas:

- real-time and Bayesian first-level analysis in `hippy-feat`
- quantitative BOLD (qBOLD) and multi-echo T2* fitting in `neurojax`
- physiological HRF and vascular models in `vpjax`
- forward simulation of BOLD from neural mass dynamics in `vbjax`

## Signal Chain (Neurovascular Coupling)

1. **Neural activation** raises local metabolic demand.
2. **Vasodilation** raises CBF and CBV. See [physics-cerebrovascular.md](physics-cerebrovascular.md).
3. **Oxygen delivery exceeds consumption**, so the deoxyhemoglobin concentration falls. The Balloon/Windkessel dynamics are described in [physics-hemodynamic.md](physics-hemodynamic.md).
4. **T2* lengthens**, and the gradient-echo EPI signal rises.

The canonical hemodynamic response function (HRF) peaks about 5-6 s after stimulus onset. It is followed by an undershoot at roughly 10-15 s, and the full response lasts about 15-20 s. The double-gamma prior used in `hippy-feat` Variant G encodes this as `peak_delay ~ N(6 s, 1²)` and `peak_width ~ N(1, 0.3²)` (`hippy-feat:docs/design/DESIGN_bayesian_first_level.md`).

## Acquisition Parameters

| Parameter | Typical value | Notes |
|---|---|---|
| TR | 0.5-3.0 s | The hippy-feat RT budget assumes a 1.5 s TR [@hough2026hippyfeat] |
| Voxel size | 2-3 mm (3T); 0.7-1.8 mm (7T) | The NSD dataset is at 7T [@scotti2024mindeye2] |
| Matrix (example) | 76 × 90 × 74 | Benchmark volume for the hippy-feat timings below |
| Readout | Single-shot gradient-echo EPI | Multiband 2-8× is common |
| Optimal TE | ≈ tissue T2* (~30 ms at 3T, ~20-25 ms at 7T) | Maximises BOLD CNR |
| Gray-matter T2* | ~45-65 ms (3T); ~25-35 ms (7T) | See note below |
| Task BOLD change | ~1-5% (3T) | Higher in primary sensory/motor cortex |
| Physiological noise | Cardiac ~1 Hz, respiratory ~0.3 Hz, slow drift | Aliased at TR ≥ 1 s |

**Correction note on T2* values.** The previous version of this page listed gray-matter T2* at 3T as 25-35 ms. Those values are typical of 7T, not 3T. The values are now general literature ranges. No bibliography key in the current source set supports them, so they should be verified against a datasheet in `raw/datasheets/` when one is ingested.

### Related MRI sequences used alongside BOLD

**pCASL (CBF).**
- The WAND ses-02 protocol has 110 volumes. This is consistent with 55 control-label pairs at a single post-labelling delay (PLD) of 2.0 s.
- Whether the acquisition is multi-PLD is still an open question to CUBRIC (`neurojax:docs/CUBRIC_QUESTIONS.md`).

**TRUST (venous SvO2).**
- The WAND protocol has 24 volumes.
- Standard effective echo times are [0, 40, 80, 160] ms. The asymmetric short-TE variant used in WAND may differ.
- These measurements are required for the T2→SvO2 calibration that anchors calibrated BOLD and qBOLD.

**qBOLD / multi-echo GRE (OEF, T2*, R2*).**
- These are fitted in `neurojax:qmri`.
- WAND ses-06 provides the `T2star_map` and `R2star_map` derivatives (`neurojax:docs/BIDS_DERIVATIVES_STRUCTURE.md`).
- See [quantitative-mri.md](quantitative-mri.md).

## Preprocessing

Standard offline preprocessing proceeds as follows:

1. Slice-timing correction.
2. Six-DOF rigid motion correction.
3. EPI susceptibility distortion correction.
4. Coregistration to T1w.
5. Normalisation to standard space.
6. Smoothing.

In the WAND derivatives layout, fMRI stays in **native space first**:
- ses-03 func is registered to T1w with a FLIRT affine (`ses-03_func_to_T1w.mat`).
- All sessions are then anchored to the ses-02 T1w.
- MNI152/OM-1 warps are applied only when needed (`neurojax:docs/BIDS_DERIVATIVES_STRUCTURE.md`).
- Frame conventions are described in [coordinate-systems.md](coordinate-systems.md), and NIfTI/BIDS conventions in [data-formats.md](data-formats.md).

### Real-time preprocessing (hippy-feat / jaxoccoli)

The full JAX/XLA-compiled pipeline runs in **~54 ms per volume on GPU**. That is **28× headroom** inside a 1.5 s TR [@hough2026hippyfeat].

| Module | Function | Time (76×90×74 vol) |
|---|---|---|
| `glm.py` | JIT OLS GLM | ~0.3 ms |
| `spatial.py` | 3D bilateral filter + spherical convolution | 3.7 ms |
| `motion.py` | Gauss-Newton rigid registration (6 DOF) | 16.4 ms |
| `stats.py` | t/F statistics, p-values | <1 ms |
| `signatures.py` | Log-signature streaming features (signax) | ~5 ms |

The motivation is the **train/test mismatch** in real-time MindEye decoding. MindEye2 is trained on fMRIPrep + GLMsingle betas [@scotti2024mindeye2; @scotti2024mindeye]. Real-time deployment, by contrast, typically uses a bare-bones GLM.

`hippy-feat` therefore benchmarks eight interchangeable variants. Each outputs a `(8627,)` z-scored beta vector (`hippy-feat:README.md`).

| Variant | Approach | Per-TR time |
|---|---|---|
| A | Glover HRF baseline (current RT-MindEye) | 163 ms |
| A+N | + CSF/WM nuisance regression | ~170 ms |
| B | FLOBS 3-basis HRF | 155 ms |
| C | Per-voxel HRF from a 20-HRF library (GLMsingle-style) | 33 ms |
| D | Conjugate Bayesian shrinkage | 237 ms |
| E | Spatial Laplacian regularisation | 103 ms |
| F | Log-signature artifact/drift monitoring | 187 ms |
| C+D | Per-voxel HRF + Bayesian | 39 ms |
| G | Full Bayesian NUTS, AR(p) noise (planned) | ~30-60 s |

## Statistical Analysis

### General Linear Model

The per-voxel generative model used across `hippy-feat` is:

```
y(t) = Σ_k β_k · [x_k(t) ⊛ h(t; θ_hrf)] + ε(t)
ε(t) = Σ_j a_j ε(t−j) + η(t),   η ~ N(0, σ²)
```

`hippy-feat:bayesian_beta.py` implements this model at three levels of cost and completeness.

**Closed-form conjugate GLM (real-time path).**
- `make_conjugate_glm` uses a normal-inverse-gamma posterior and runs at ~0.5 ms/voxel.
- It is equivalent to BROCCOLI's Gibbs Block 1 [@eklund2014broccoli].

**AR(1)-prewhitened conjugate GLM.**
- `make_ar1_conjugate_glm` adds prewhitening with precomputed S matrices.

**Full NUTS GLM (offline, Variant G).**
- `make_bayesian_glm` uses blackjax NUTS, vmapped over voxels.
- Priors: β ~ N(0, 10²); AR coefficients ~ N(0, 0.5²) with typically p = 2; σ² ~ InvGamma(1, 1).
- Sampling: 1000 samples after 500 warmup.
- Estimated runtime is ~30-60 s whole-brain on an A100. BROCCOLI reports ~1-2 min per subject using a VB approximation.
- See [method-monte-carlo.md](method-monte-carlo.md) and [method-variational-inference.md](method-variational-inference.md).

**HRF levels in Variant G.**

| Level | Parameterisation | Priors |
|---|---|---|
| 1 | FLOBS weights | w1 ~ N(1, 0.5²); w2, w3 ~ N(0, 0.3²) |
| 2 | Parameterised double-gamma | As in the signal-chain section |
| 3 | Riera neurovascular coupling via `vpjax.hemodynamics.riera_hrf` | Posterior over physiological parameters, informed by MRS (GABA/Glu), qMRI and angiography |

At Level 3 the posterior is over physiological parameters rather than over basis weights.

| Tool | Method | HRF | Noise | Spatial | Posterior |
|---|---|---|---|---|---|
| FSL FEAT | OLS/FLAME | FLOBS/Glover | Prewhitened AR | Gaussian smoothing | No |
| SPM | ReML | Canonical + derivatives | AR(1) | Gaussian smoothing | No |
| BROCCOLI | Bayesian GLM | Canonical | AR(4) | MRF prior | VB approximation |
| FSL FABBER | VB | Flexible | AR | Optional spatial | VB approximation |
| hippy-feat G | NUTS | FLOBS / parametric / vpjax | AR(p) | Optional MRF | Full samples |

### Functional Connectivity and the Variance-Propagation Gap

In beta-series correlation [@rissman2004measuring; @mumford2012deconvolving], OLS point-estimate betas are correlated directly and their standard errors are discarded. This attenuates the measured connectivity.

`jaxoccoli` provides three fixes of increasing rigour:
- `weighted_corr`, with weights = 1/β_std.
- `attenuated_corr`, the Spearman correction `r_corr = r_obs / sqrt(rel_i · rel_j)`.
- `posterior_corr`, which marginalises over `(beta_mean, beta_var)`.

Variance is also carried through parcellation. `make_atlas_linear_uncertain` computes `W @ mean` and `W² @ var`.

The whole chain from GLM to parcellation to FC to loss (modularity, eigenmaps, decoder) is differentiable. This reimplements the core of hypercoil [@ciric2022differentiable] using vbjax-style `make_*() → (params, forward_fn)` factories.

Additional connectivity tools in `jaxoccoli`:
- Sliding-window and dynamic FC. HMM-style state models are covered in [method-hmm-dynamics.md](method-hmm-dynamics.md).
- SPD tangent-space operations.
- Chebyshev spectral graph filters, ported from `hgx`.
- Sinkhorn and Gromov-Wasserstein distances for cross-subject FC comparison.

See [connectomics.md](connectomics.md) and [method-spectral-analysis.md](method-spectral-analysis.md).

### Dimensionality

On the 7T Natural Scenes Dataset (8 subjects), BIC estimates **11-18 components** consistently in 7 of 8 subjects. 7T data shows about **5× higher intrinsic dimensionality than 3T** (`hippy-feat:README.md`). These estimates come from broken-stick, parallel-analysis and MELODIC-consensus methods.

## Forward Models: Simulating BOLD

**Neural mass → BOLD (vbjax, neurojax).**
- Whole-brain simulators drive a Balloon-Windkessel hemodynamic stage with neural mass activity on a structural connectome [@woodman2024insamuvbjax]. See [neural-mass-models.md](neural-mass-models.md) and [connectomics.md](connectomics.md).
- `neurojax:bench/models/rww.py` implements the reduced Wong-Wang model with `gamma_E = 0.641` and `gamma_I = 1.0`, in s⁻¹ consistent with τ in seconds.
- A previous `/1000` unit bug drove the single-node fixed point to S_E ≈ 0.0002 instead of the expected ≈ 0.164 (Deco 2013). It was fixed in commit `49acac2` (`neurojax:docs/BUG_REPORTS.md`).

**Physiological HRF (vpjax).**
- The Riera electrovascular coupling model serves as a learnable HRF.
- The planned validation compares vpjax-derived CBF against WAND pCASL CBF (`hippy-feat:docs/design/DESIGN_bayesian_first_level.md`).
- `hippy-feat:angiography.py` builds a TOF-MRA vessel tree (Frangi filter → skeleton → radii). It emits `{points, radii, branch_ids}` for `vpjax.vascular.angiography.VesselTree`.

**EEG-fMRI fusion.**
- `hippy-feat:fusion.py` provides experimental balloon-model fusion. See [eeg.md](eeg.md).

**Encoding models.**
- `TribeV2Adapter` (`facebook/tribev2`) predicts fsaverage5 BOLD from video, audio and text.
- It is validated against NSD with RSA and noise ceilings in `hippy-feat:nsd.py`.

## Datasets

| Dataset | Field | Content | Used by |
|---|---|---|---|
| WAND [@mcnabb2025wand] | 3T Prisma (func, ses-03); 7T and Connectom sessions for other modalities | 170 subjects; fMRI including category localiser, pCASL, TRUST, qMRI, MEG, MRS | neurojax, hippy-feat (Variant G Phase 1 validation) |
| Natural Scenes Dataset | 7T | 8 subjects; betas at `/data/3t/nsd_multisubject/` on DGX Spark | hippy-feat (MindEye, dimensionality, TRIBE validation) |

## Limitations

- **Hemodynamic lag.** The 5-6 s peak and ~15-20 s duration blur sub-second cognitive dynamics. Complementary modalities are [eeg.md](eeg.md) and [meg.md](meg.md).
- **Venous weighting.** Draining veins spread the GE-BOLD point-spread function to several mm.
- **Susceptibility dropout.** Signal is lost in orbitofrontal and inferior temporal cortex near air-tissue interfaces.
- **HRF variability across voxels and subjects.** This motivates Variants B, C and G.
- **Two-stage estimation bias.** It affects beta-series connectivity unless variance is propagated [@mumford2012deconvolving].
- **Indirect measure.** For neuronal-current alternatives, see [neuronal-current-mri.md](neuronal-current-mri.md).

## Relevant Projects

| Project | What it implements | Location |
|---|---|---|
| `hippy-feat` | RT preprocessing (54 ms/vol), RT-MindEye variants A-G, Bayesian GLM, differentiable variance-aware FC | `jaxoccoli/`, `scripts/rt_glm_variants.py` |
| `neurojax` | qBOLD OEF, multi-echo T2*, WAND fMRI registration, RWW BOLD bench, permutation GLM/cluster stats | `src/neurojax/qmri/`, `bench/` |
| `vbjax` | Neural mass simulation with BOLD monitors | — |
| `vpjax` | Riera HRF, cerebrovascular/metabolic models, VesselTree | `hemodynamics/`, `vascular/angiography` |
| `LAYNII` | Layer-fMRI (0.7-1 mm) laminar analysis | — |
| `libspm` | Registration and segmentation used for coregistration/normalisation | — |
| `gnf-jax` | Wilson-Cowan on connectome harmonics, a candidate source of BOLD-scale dynamics | — |

## See Also

- [physics-hemodynamic.md](physics-hemodynamic.md) — Balloon/Windkessel model, BOLD signal equation
- [physics-cerebrovascular.md](physics-cerebrovascular.md) — CBF, CBV, autoregulation, vessel trees
- [physics-bloch.md](physics-bloch.md) — T2* contrast and EPI signal formation
- [quantitative-mri.md](quantitative-mri.md) — qBOLD, T2*/R2* mapping
- [structural-mri.md](structural-mri.md) — T1w anchors for coregistration
- [mrs.md](mrs.md) — GABA/Glu priors for physiological HRFs
- [fnirs.md](fnirs.md), [fus.md](fus.md) — other hemodynamic modalities
- [neural-mass-models.md](neural-mass-models.md), [method-neural-fields.md](method-neural-fields.md) — neural drivers of simulated BOLD
- [connectomics.md](connectomics.md) — FC matrices and structural connectomes
- [method-hmm-dynamics.md](method-hmm-dynamics.md) — dynamic connectivity states
- [method-monte-carlo.md](method-monte-carlo.md), [method-variational-inference.md](method-variational-inference.md), [method-sbi.md](method-sbi.md) — Bayesian GLM and HRF inference
- [jax-ecosystem.md](jax-ecosystem.md), [compute-infrastructure.md](compute-infrastructure.md) — JAX stack, DGX Spark

*Maintenance note:* This update removed the previous `related:` links (`bold-signal.md`, `hemodynamic-response.md`, `glm.md`, `connectivity.md`, `preprocessing.md`) because none of those pages exist. It also replaced reference keys that were absent from the bibliography with project-file citations.
