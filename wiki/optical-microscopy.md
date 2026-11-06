---
type: modality
title: Computational Optical Microscopy
physics: optical
measurement: Transmitted or scattered intensity under coded LED-array illumination, inverted to quantitative phase (DPC/FPM). Also epi-fluorescence through a pinhole (confocal/ISM).
spatial_resolution: ~0.1–0.5 µm lateral, diffraction-limited at λ/(2·NA). 108 nm/px Nyquist sampling at 60× with 6.5 µm sCMOS pixels. FPM extends the effective NA beyond the objective's NA.
temporal_resolution: Single-shot to seconds per frame for widefield/DPC. Galvo confocal scans at kHz line rates. Fixed-sample stacks take minutes to hours.
related: [tissue-optical-properties.md, fnirs.md, physics-diffusion-equation.md, physics-electromagnetic.md, method-monte-carlo.md, method-fem.md, method-sbi.md, method-variational-inference.md, method-neural-fields.md, method-source-imaging.md, jax-ecosystem.md, data-formats.md, compute-infrastructure.md, diffuse-optical-tomography.md]
description: "Computational optical microscopy treats the microscope as a differentiable information pipeline with four stages:"
timestamp: 2026-11-05T15:59:53-08:00
tags: [modality, optical, microscopy, computational-imaging]
---

# Computational Optical Microscopy

Computational optical microscopy treats the microscope as a **differentiable information pipeline** with four stages:

- **Design** what to measure.
- **Simulate** the optics with a differentiable forward model.
- **Reconstruct** the object by inverting that forward model.
- Optionally **translate** label-free contrast into predicted fluorescence.

Source: `bsccm-jax:docs/ECOSYSTEM.md`.

In this knowledge base it is the **ballistic-photon** end of the optical spectrum. Microscopy works at depths where unscattered light still carries image information. [fnirs.md](fnirs.md) and diffuse optical tomography (DOT) work at depths where it does not, and photon transport becomes diffusive [@Arridge1999inverse; @Ishimaru1978wave].

Both regimes share the same JAX inverse-problem machinery (Equinox forward models, Lineax linear solves, Optimistix optimizers; [jax-ecosystem.md](jax-ecosystem.md)). This shared stack is why `bsccm-jax`, `openconfocal-cm`, and `dot-jax` are documented together.

## Regimes of optical imaging in the project family

| Regime | Photon transport | Forward model | Project | Typical depth |
|---|---|---|---|---|
| Coded-illumination QPI (DPC/FPM) | Ballistic, weak-object (first Born) | 2D weak-object transfer function (WOTF) / coherent pupil | `bsccm-jax` | single cells, ~µm-thin |
| Confocal / ISM fluorescence | Ballistic, scalar wave optics | `h_ill · (h_det ⊛ P)` | `openconfocal-cm` | 10 µm – ~3 mm (cleared) |
| Diffuse optical tomography / fNIRS | Multiply scattered (diffusion approximation) | FEM diffusion equation [@Arridge1993fem; @Schweiger1993fem] | `dot-jax` | cm (scalp → cortex) |

The diffuse regime is covered in [physics-diffusion-equation.md](physics-diffusion-equation.md) and [method-monte-carlo.md](method-monte-carlo.md) [@Fang2010mcx]. Tissue μa/μs′ values are in [tissue-optical-properties.md](tissue-optical-properties.md). This page covers the ballistic regime.

## Physics: differential phase contrast (DPC)

### Measurement and transfer functions

An LED-array microscope illuminates the sample from asymmetric half-annuli (Top/Bottom, Left/Right). The normalized DPC image for one axis is

```
I_DPC = (I_1 − I_2) / (I_1 + I_2)
```

For a weak object, this image is linear in phase φ (and absorption μ) through the WOTF:

```
Ĩ_DPC,j(u) = H_φ,j(u) · φ̃(u) + H_μ,j(u) · μ̃(u)
```

### Tikhonov reconstruction

Combining all illumination axes j gives the closed-form solution

```
φ = F⁻¹[ Σ_j H*_φ,j Ĩ_j / (Σ_j |H_φ,j|² + α) ]
```

Source: Tian & Waller 2015, *Opt. Express* 23(9):11394, cited via `bsccm-jax:docs/RESEARCH.md`.

> ⚠️ A 2023 erratum corrects sign and typo errors in the original transfer functions. `generate_wotf` should be audited against it (`bsccm-jax:docs/RESEARCH.md`).

### FPM, the coherent sibling

Fourier ptychographic microscopy (FPM) uses the same LED-array platform. It applies single-LED coherent illumination and stitches pupil-shifted spectra together, synthesizing an effective NA beyond the objective's own NA. FPM is nonlinear and much more ill-posed than DPC. That makes it the regime where learned or diffusion priors earn their cost [@li2023poissongaussian].

## Reconstruction methods and measured performance

`bsccm-jax` re-JAX-ifies BSCCM's `functional_dpc.py` and adds several reconstructors.

| Method | Implementation | Result | Source |
|---|---|---|---|
| Analytic Tikhonov | `dpc.py` | baseline; corr ~0.55 on static real cells | `bsccm-jax:docs/ECOSYSTEM.md` |
| Matrix-free CG, `Normal(CG())` | `dpc.py` (Lineax) [@rader2023lineax] | passes phantom spec | `bsccm-jax:tests/test_dpc.py` |
| Nonlinear CG | `dpc.py` (Optimistix) | passes phantom spec | `bsccm-jax:tests/test_dpc.py` |
| TV-ADMM (SCICO, `L21Norm` on gradient) | `dpc.py` | **corr 0.9996** (phantom, best) | `bsccm-jax:docs/RESEARCH.md` |
| Per-cell neural field (NSTM-style coordinate MLP) | `neural_field.py` | ≈ Tikhonov on static cells; no gain | `bsccm-jax:docs/ECOSYSTEM.md` |
| Amortized self-supervised (PtychoPINN paradigm) | `amortized.py` | single-pass inference | `bsccm-jax:docs/ECOSYSTEM.md` |
| NeuPh conditional neural field | `neuph.py` | **0.97 held-out** | `bsccm-jax:docs/ECOSYSTEM.md` |
| FPM super-resolution | `fpm.py` | **phase corr 0.99** | `bsccm-jax:docs/ECOSYSTEM.md` |
| Differentiable coded-illumination design | `design.py` | **46% better than DPC under noise** | `bsccm-jax:docs/ECOSYSTEM.md` |

### Lineage of untrained and neural-representation methods

These reconstructors follow a line of work on untrained and neural representations:

- Deep Phase Decoder [@bostan2020deep]
- Untrained physics-informed nets for structured illumination microscopy (SIM) [@burns2022untrained]
- Physics-based optical neural networks for refractive-index tomography [@yang2023refractive]

The pedagogical framing is Kellman et al.'s "autodiff through the forward model twice" [@kellman2020how]. Autodiff is used once to build the reconstruction network and again to train it, so only the physics has to be implemented.

> **Naming collision.** "Neural field" here means an implicit neural representation (a coordinate MLP with annealed positional encoding). It is unrelated to the Wilson–Cowan neural fields of [method-neural-fields.md](method-neural-fields.md).

### Design decisions

- **Diffusion priors are skipped for DPC.** DPC is near-linear and well-conditioned thanks to multi-axis diversity, so classical solvers and conditional neural fields win on quality per unit of compute.
- **NSTM motion estimation is not ported.** BSCCM cells are static, so NSTM's motion machinery adds nothing.
- **Amortized reconstruction is preferred at scale.** BSCCM has ~400k cells, so training once and inferring in a single pass beats per-cell optimization.

### Uncertainty quantification

Because the WOTF is fully differentiable, it plugs directly into amortized posterior estimation. This yields calibrated phase uncertainty rather than a point estimate (see [method-sbi.md](method-sbi.md) and [method-variational-inference.md](method-variational-inference.md)).

## BSCCM dataset and benchmarks

The Berkeley Single-Cell Computational Microscopy dataset (Pinkard et al. 2024, arXiv:2402.06191) contains:

- **12M+ images** of **~400k white blood cells**.
- Label-free channels: `DPC_Top/Bottom/Left/Right` and a `DF_*` darkfield ring.
- **6 fluorescence bands** of surface markers.
- A coherent variant for FPM.

`bsccm-jax` has fetched 197 GB via Dryad OAuth2 and converts it to HCS OME-Zarr (see [data-formats.md](data-formats.md)). Virtual-staining fine-tunes (Cytoland/VisCy UNeXt2) run on a DGX Spark GB10 (see [compute-infrastructure.md](compute-infrastructure.md)).

| Task | Metric | Result | Baseline |
|---|---|---|---|
| WBC classification (BSCCM) | accuracy | 0.919 | Nazir 0.913 |
| CD16 regression (BSCCM) | Pearson | 0.774 | ~0.72 |
| FPM, BSCCM-coherent | spectral extension | 0.99 | brightfield-limited |
| Virtual staining Phase → 6 fluorophores | PCC vs floor | 0.799 / 0.800 | at task floor |
| VSCyto2D oracle | PCC | 0.717 | paper ~0.71 |
| FPM, UConn USAF target | spectral extension | 0.230 → 0.52 (5× aperture) | 1-LED brightfield |
| Raabin Test-B (domain shift) | accuracy | 0.472 | collapses |

Source: `bsccm-jax:docs/RESULTS.md`.

## Physics: point-scanning confocal for cleared tissue

`openconfocal-cm` models a galvo point-scanning confocal. Its scalar forward model is `h_ill · (h_det ⊛ P)` (`confocal.py`). That model **understates axial extent above NA ~0.7**, so absolute numbers at high NA must be validated against the Chromatix wave-optics oracle (`openconfocal-cm:docs/DECISIONS.md`).

### Index-mismatch spherical aberration decides the objective

Spherical aberration from an index mismatch grows **linearly with depth**. Once refocusing absorbs the piston and defocus terms, it scales as **NA⁴**: doubling NA costs ~16× in usable depth.

The table gives the depth at which Strehl ratio falls to 0.8 (Maréchal criterion) at 520 nm (`openconfocal-cm:docs/CLEARING.md`).

| Objective | into CUBIC (n=1.48) | into iDISCO (n=1.56) |
|---|---|---|
| 60×/1.4 oil | 6.6 µm | 7.2 µm |
| 40×/1.3 oil | 15.5 µm | 15.3 µm |
| 20×/0.75 air | 9.8 µm | 9.3 µm |
| 10×/0.45 air | 119.8 µm | 113.1 µm |
| 10×/0.30 air | 681.5 µm | 641.3 µm |
| 4×/0.16 air | 8.98 mm | 8.42 mm |
| 20×/1.0 glycerol (n=1.47) | 319.0 µm | 41.1 µm |
| 25×/1.0 multi-immersion, collar at 1.48 | unlimited | 47.2 µm |

Working distance is a second, independent wall. A 60×/1.4 oil objective has ~130 µm WD, while reaching the centre of a 300 µm organoid needs 150 µm. This is why cleared-tissue objectives are low-NA and long-WD, and why they cost $28,090–$31,500.

### Clearing protocols are an optics decision

Each clearing protocol fixes the refractive index, which in turn fixes the objective (`openconfocal-cm:docs/CLEARING.md`):

| Regime | RI | Example protocols |
|---|---|---|
| Aqueous / hydrogel | 1.38–1.48 | ScaleA2 1.38, CLARITY/PACT 1.45–1.46, CUBIC ~1.48 |
| High-aqueous | 1.49–1.52 | SeeDB ~1.49, SeeDB2G 1.52 |
| Solvent | 1.55–1.56 | iDISCO, uDISCO, BABB, ECi |

- iDISCO shrinks tissue by 10–30% and quenches GFP.
- Solvent protocols dissolve the adhesives used in most objective lenses.

### Other quantitative design constraints

- **Spinning-disk crosstalk.** Contrast is completely lost at ~30 µm, and background exceeds 60% of signal by ~80 µm (Shimozawa 2013, *PNAS* 110:3399). Clearing does not help, because the crosstalk is driven by out-of-focus fluorescence rather than scattering.
- **Pinhole chosen by information content.** The pinhole is set with IDEAL's decoder-free estimator `I(Y;X) = H(Y) − H(Y|X)`, using a Gaussian entropy upper bound, rather than a fixed 1 Airy unit (`design.py`). A 2D in-focus simulation understates the benefit of closing the aperture, so a 3D phantom is required.
- **ISM / Sheppard reassignment.** Gives √2 resolution over widefield with no photon loss, or ~2× after deconvolution (`ism.py`).
- **Optical section formula.** The Zeiss quadrature expression gives 14.4 µm at NA 0.25. The older `4nλ/NA²` depth-of-field formula gives 33 µm. These are different quantities and should not be mixed.
- **Thermal drift.** Aluminium expands at 23 ppm/K, so a 100 mm structural loop moves ~2.3 µm/K. PLA is ~70 ppm/K.
- **Drift identifiability.** A single raster cannot separate drift from object stretch: residual error is 1.805 px at a 1.8 px drift amplitude. Three passes recover the trajectory to **0.010 px**, with object correlation 0.988 (`openconfocal-cm:docs/DECISIONS.md`, D10).
- **Lateral sampling.** 6.5 µm sCMOS pixels at 60× give 108 nm/px, exactly Nyquist (`openconfocal-cm:docs/BOM.md`).

## Shared methodological threads

- **Linear inverse problems with regularization.** DPC Tikhonov and TV-ADMM have the same structure as DOT Jacobian inversion [@Arridge1999inverse; @Dehghani2009nirfast] and M/EEG minimum-norm source imaging ([method-source-imaging.md](method-source-imaging.md)).
- **Autodiff Jacobians.** `dot-jax` differentiates through its FEM forward solve with respect to optical properties ([method-fem.md](method-fem.md)). `bsccm-jax` differentiates through the WOTF with respect to phase and LED patterns.
- **Information-driven design.** IDEAL ranks LED patterns (`bsccm-jax:design.py`) and pinhole diameters (`openconfocal-cm:design.py`). The same principle could rank fNIRS optode layouts, for example short-separation choices [@Brigadoi2015short].
- **Electromagnetic basis.** Both regimes are limits of Maxwell's equations ([physics-electromagnetic.md](physics-electromagnetic.md)): scalar wave optics for microscopy, radiative transfer and its diffusion limit for DOT [@Ishimaru1978wave; @Haskell1994boundary].

## Relevant Projects

- **`bsccm-jax`** — DPC WOTF forward model plus four solvers, neural-field, amortized, NeuPh and FPM reconstructors, differentiable coded-illumination design, virtual staining, and phenotyping, all on BSCCM.
- **`openconfocal-cm`** — galvo confocal design tools. Modules: `optics.py`, `confocal.py`, `aberration.py`, `scanhead.py`, `design.py`, `ism.py`, `stages.py`. Phase 0 (simulation) only; nothing has been built.
- **`dot-jax`** — the diffuse-regime counterpart: a JAX/Equinox reimplementation of redbirdpy for CW DOT/fNIRS, sharing the Lineax/Optimistix stack.
- **`inferjax`** — amortized SBI that can wrap the differentiable WOTF for posterior phase estimation.

## See Also

- [Diffuse Optical Tomography](diffuse-optical-tomography.md)
- [tissue-optical-properties.md](tissue-optical-properties.md) — μa, μs′, and refractive indices
- [fnirs.md](fnirs.md) — the diffuse optical modality
- [physics-diffusion-equation.md](physics-diffusion-equation.md) — the photon diffusion approximation
- [method-monte-carlo.md](method-monte-carlo.md) — photon Monte Carlo
- [method-fem.md](method-fem.md) — FEM forward solvers
- [method-sbi.md](method-sbi.md) — uncertainty-aware inversion
- [method-variational-inference.md](method-variational-inference.md)
- [method-source-imaging.md](method-source-imaging.md) — the analogous regularized inverse problem
- [method-neural-fields.md](method-neural-fields.md) — note the naming collision
- [jax-ecosystem.md](jax-ecosystem.md) — Equinox, Lineax, Optimistix, SCICO
- [data-formats.md](data-formats.md) — OME-Zarr
- [compute-infrastructure.md](compute-infrastructure.md)
- [physics-electromagnetic.md](physics-electromagnetic.md)
