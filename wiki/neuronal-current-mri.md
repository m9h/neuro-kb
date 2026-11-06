---
type: modality
title: Neuronal-Current MRI, MREIT and MRCDI
physics: electromagnetic
measurement: "MR phase (and sometimes magnitude) perturbations from the current-induced magnetic-field component parallel to B0 (ΔBz). The source is either endogenous neuronal currents (ncMRI/msMRI, spin-lock) or externally injected currents (MRCDI). Conductivity is reconstructed from injected-current fields (MREIT, DT-MREIT, fMREIT)."
spatial_resolution: "~1–3 mm voxels in human MRCDI/MREIT; 3 mm claimed for msMRI (Xiong 2003); neuronal-current claims remain unreplicated"
temporal_resolution: "Phase accrues over encoding windows of ~10–100 ms (field must overlap TE); 100 ms sampling claimed for msMRI; conductivity maps take seconds–minutes"
related: [physics-electromagnetic.md, physics-bloch.md, tissue-electrical-properties.md, method-fem.md, method-bem.md, method-source-imaging.md, method-sbi.md, meg.md, eeg.md, fmri.md, mrs.md, quantitative-mri.md, diffusion-mri.md, structural-mri.md, temporal-interference.md, acoustoelectric-imaging.md, tms.md, tissue-csf.md, tissue-gray-matter.md, tissue-white-matter.md, tissue-skull.md, sci-head-model.md, head-model-mida.md, coordinate-systems.md]
description: "1."
timestamp: 2026-11-05T15:59:53-08:00
tags: [modality, electromagnetic, mri, mreit]
---

# Neuronal-Current MRI, MREIT and MRCDI

**Neuroelectric MRI** is an umbrella for three related programs. Each uses the MR phase to read out magnetic fields produced by electrical currents in tissue:

1. **Neuronal-current MRI (ncMRI)** tries to detect *endogenous* neural currents directly, without hemodynamic mediation. Variants include msMRI, spin-lock/NEMO, SIRS, NAMO and the retracted DIANA work.
2. **MR current-density imaging (MRCDI)** maps the fields and current densities of *externally applied* currents. Examples include tDCS/tACS currents and calibrated phantom injections.
3. **MR electrical impedance tomography (MREIT)** inverts injected-current fields to recover low-frequency **conductivity**. Variants are DT-MREIT for anisotropic tensors and fMREIT for activity-dependent conductivity changes.

> **Evidence status.** Endogenous ncMRI is still an experimental research problem. The reviewed literature does not establish a reproducible human method that combines useful spatial localization with millisecond neural specificity. Measuring *imposed* current fields (MRCDI/MREIT) is substantially further along. Source: `neuroelectric-mri:README.md`.

## Physical principle

Current density **J** in a volume conductor produces a magnetic field by Biot–Savart. Only the component parallel to the main field, ΔB_z (**B0** along +z), shifts the Larmor frequency to first order. The accumulated phase is

```text
φ(r) = γ ∫ ΔB_z(r, t) · s(t) dt        γ = 2.675×10⁸ rad s⁻¹ T⁻¹
```

Here s(t) is the signed encoding sensitivity of the sequence. In a spin echo, s flips sign at the refocusing pulse. A static field is therefore cancelled, but a field present in only one half of the echo survives. This timing trick underlies both the transient-field detection of [@bodurka2002toward] and the polarity-switched current pulses of MRCDI. See [physics-bloch.md](physics-bloch.md) for Bloch dynamics and [physics-electromagnetic.md](physics-electromagnetic.md) for the quasi-static field equations.

In the quasi-static, passive, ohmic regime (see [tissue-electrical-properties.md](tissue-electrical-properties.md)):

```text
∇·(σ∇u) = 0,   J = −σ∇u,   ∇×B = μ0 J (closed circuit, incl. leads)
```

The physical mechanism differs by readout:

| Readout | Mechanism | Relevant frequency |
|---|---|---|
| Phase (msMRI, MRCDI, MREIT) | ΔB_z ∥ B0 shifts precession | DC to tens of Hz, timed to the sequence |
| Magnitude (msMRI) | Intravoxel phase dispersion leads to dephasing | Same as above |
| Spin-lock / rotary saturation (NEMO, SIRS) | Resonant tipping when the field oscillates at f_SL = γB_SL/2π | Matched to neural rhythm, e.g. α ≈ 10 Hz |
| Lorentz effect (NAMO) | Force on current-carrying tissue in B0 leads to displacement contrast | Motion encoding |
| EPT (contrast case, *not* MREIT) | B1 phase/magnitude at the Larmor frequency | ~128 MHz at 3 T |

Worked number (derived here): a 10 Hz spin-lock resonance requires B_SL = 10 Hz / 42.58 MHz T⁻¹ ≈ 0.23 µT.

Electrical properties tomography (EPT) reconstructs σ and ε at the RF frequency from B1 maps. Its values are not interchangeable with the low-frequency conductivities that MREIT targets. `mri-rf-stack` computes the B1 components B⁺ = μ0(Hx + iHy)/2 and B⁻ = μ0(Hx − iHy)/2 at 127.732435554 MHz (3 T protons) (`mri-rf-stack:README.md`).

## Phase-budget parameters

| Quantity | Value | Source |
|---|---|---|
| Phantom transient field detected | 200 pT lasting 40 ms (wire phantom, spin-echo timing) | [@bodurka2002toward] |
| Implied ideal phase for that field | γ·2×10⁻¹⁰ T·0.04 s ≈ **2.1×10⁻³ rad (0.12°)** | derived from the line above |
| Same duration at 1 nT | ≈ 1.1×10⁻² rad (0.61°) | derived, illustrative |
| msMRI claim, human visuomotor | ~1% magnitude contrast, 100 ms sampling, 3 mm voxels | [@xiong2003directly] |
| 3 T proton Larmor frequency (RF/EPT regime) | 127.732435554 MHz | `mri-rf-stack:README.md` |
| Synthetic RF benchmark phantom | εr = 60, σ = 0.5 S/m, ρ = 1000 kg/m³ (declared, not tissue data) | `mri-rf-stack:README.md` |
| Synthetic tank conductivity (MREIT workbench) | 0.5 S/m homogeneous; 0.2/0.8 S/m layered | `neuroelectric-mri:docs/fem-complete-electrode.md` |
| CEM contact impedance (synthetic) | z = 0.002, 0.005 Ω·m² | `neuroelectric-mri:docs/fem-complete-electrode.md` |
| Injected test current (synthetic) | 1 mA | `neuroelectric-mri:docs/fem-complete-electrode.md` |

The executable sanity check is `neuroelectric-mri:examples/phase_budget.py`. It computes ideal field-to-phase budgets, closed-loop wire fields, cancellation cases and timing-dependent phase errors.

## Evidence chronology (endogenous currents)

| Period | Development | Takeaway |
|---|---|---|
| 1999–2002 | Kamei et al. (current-distribution proposal); Bodurka & Bandettini wire phantom | Phantom sensitivity is not the same as human attribution [@bodurka2002toward] |
| 2003 | Xiong, Fox & Gao (UTHSCSA), msMRI | Influential human claim [@xiong2003directly] |
| 2004 | Chu et al., visual-response study | No convincing rapid neuronal-current signal [@chu2004hunting] |
| 2005 | "Fantasy, possibility, or reality?" review | Snapshot of the debate [@bandettini2005direct] |
| 2006 | Petridou et al. organotypic cultures (TTX-blocked, avascular); Truong–Song NAMO | Biological evidence without vasculature (`neuroelectric-mri:docs/history.md`) |
| 2008 | Realistic neuronal simulations | Morphology, synchrony, orientation and voxel size dominate predictions [@cassar2008realistic] |
| 2010 | POSSUM time-varying-field extension | Acquisition timing changes the apparent spatial pattern [@drobnjak2010simulating] |
| 2016 | Isolated turtle cerebellum with concurrent LFP | Positive phase/current reconstruction, in vitro [@sundaram2016direct] |
| 2019 | NEMO spin-lock human α-band report | Spin-lock route with phantom and EEG work [@truong2019toward] |
| 2024–2026 | Spin-lock stimulus-induced and MRI–MEG studies | 2026 study is a quantitative **negative** human benchmark [@capiglioni2024stimulusinduced; @capiglioni2026exploring] |
| 2025 | DIANA (Science 2022) **retracted 25 Sept 2025** after a 2024 nonreplication | `neuroelectric-mri:README.md` |
| 2026 | qsMRI preprint (Qian et al.) with new human claims | Independent validation still open (`neuroelectric-mri:README.md`) |

Neural fields at the voxel scale are small, and partial cancellation within a voxel is substantial. This makes spatial averaging and source geometry first-order concerns [@cassar2008realistic]. Compare the ~fT scalp fields that [meg.md](meg.md) measures.

## MRCDI and MREIT: imposed currents

- **Human brain MRCDI.** Fields from transcranial currents have been measured in humans, giving direct validation of stimulation-field models [@gksu2018human]. Follow-up work characterized sensitivity, volumetric acquisition and SNR gains [@gksu2021sensitivity; @gksu2023volumetric; @gregersen2025increasing], along with the safety of the injected currents [@gregersen2021safety]. A 2024 model-calibration study and a 2026 SMS acquisition study continue the line (`neuroelectric-mri:README.md`).
- **MREIT conductivity.** Since only B_z is measured, the inverse problem recovers σ from one field component per injection. Human-leg MREIT was reported in 2009 (`neuroelectric-mri:docs/history.md`). Conductivity-model validation work includes [@sajib2024validation].
- **Functional MREIT.** Activity-dependent conductivity is an alternative mechanism to field detection. It is supported by in-vitro and modeling studies [@sadleir2017direct; @sadleir2018functional] and later work [@kim2024functional].
- **DT-MREIT** combines diffusion-tensor orientation with MREIT to estimate anisotropic conductivity tensors. DTI-derived orientation is a *candidate prior*, not an automatic calibration (`neuroelectric-mri:docs/fem-roadmap.md`). See [diffusion-mri.md](diffusion-mri.md).

### Identifiability null

In the ideal-contact conductive-tank model at fixed total current, multiplying σ by 3 everywhere leaves **J unchanged** and divides potential and **E** by 3. Absolute conductivity therefore cannot be recovered from J alone. Voltage measurements or other justified constraints are required (`neuroelectric-mri:docs/conductive-tank.md`). This null control is a precondition for any amortized inverse; see [method-sbi.md](method-sbi.md).

## Simulation workbench (neuroelectric-mri)

All checked-in datasets are synthetic. No conductivity inversion or acquired-scanner reproduction exists yet.

| Component | Key result |
|---|---|
| Finite-volume tank (harmonic interface conductance R = h/(2σ_pA) + h/(2σ_qA)) | Potential error 0.748% → 0.204% → 0.0526% over 8³–32³ grids; observed order 1.87, 1.96 |
| DOLFINx/Gmsh complete-electrode model (CEM), 120×80×60 mm tank | 1 mA, full-face: **51.458333 mV** (0.5 S/m), **79.583333 mV** (0.2/0.8 S/m layered) |
| Finite 40×30 mm electrode patches, CG2 | 73.4761 / 73.7505 / 74.0074 mV across mesh levels; last change 0.347% |
| Raviart–Thomas equilibrated current + adaptive refinement | Relative energy gap **2.79% → 0.817%**; resistance interval **74.000–74.604 Ω**; max cell current imbalance 4.05×10⁻¹⁷ A |
| Independent GetDP CEM, nine matched cases | Relative voltage difference ≤ **1.20×10⁻¹²**; finest global mesh 74.0073682 Ω, inside the interval |
| Test suite | 71 distinct tests (37 spectroscopy/FV, 25 DOLFINx, 9 GetDP) |

The CEM weak form is ∫∇v·σ∇u dV + Σ_e z_e⁻¹∫(u−U_e)(v−V_e) dS = Σ_e I_e V_e. The energy interval bounds the **resistance** problem only. It does not bound pointwise J or MR phase. See [method-fem.md](method-fem.md).

The next gate is the complete-circuit magnetic field, with explicit outgoing and return leads. After that comes projection onto B0 and the signed encoding integral, then PETSc/MPI scaling and adjoint derivatives. The target experimental platform is a Siemens Prisma 3 T phantom program with synchronized current monitoring (`neuroelectric-mri:docs/fem-roadmap.md`).

## Not to be confused with: MR inverse imaging (InI)

Lin's InI uses multichannel receive-coil sensitivities and EEG/MEG-style regularized inverses to accelerate BOLD/magnetization imaging [@lin2006dynamic; @lin2008eventrelated]. RF reception in InI is **not** a direct neural-current measurement (`mri-rf-stack:docs/HANDOFF.md`).

`mri-rf-stack` implements the noise-whitened minimum-norm building block with λ = α·σ_max. It reports point-spread functions and effective degrees of freedom on a synthetic 32-channel example. The method link to [method-source-imaging.md](method-source-imaging.md) is mathematical, not physical.

## Cross-modality connections

- **Head-model validation.** MRCDI field maps test the volume-conductor models used for [tms.md](tms.md), tDCS and [temporal-interference.md](temporal-interference.md). The same σ values feed EEG/MEG leadfields ([eeg.md](eeg.md), [method-bem.md](method-bem.md)).
- **Conductivity priors.** `neurojax` maps qMRI parameters (T1, BPF) to σ and takes anisotropy from DTI inside a differentiable FEM/leadfield graph (`neurojax:README.md`). MREIT/DT-MREIT measurements are the natural calibration targets. Relevant pages: [quantitative-mri.md](quantitative-mri.md), [tissue-csf.md](tissue-csf.md), [tissue-skull.md](tissue-skull.md).
- **Spectroscopy confound.** Injected currents shift spectral phase. A six-control current/MRSI benchmark separates injected phase from metabolite concentration (`neuroelectric-mri:data/synthetic_mrs_current/README.md`); see [mrs.md](mrs.md).
- **Coupled-physics relatives.** [acoustoelectric-imaging.md](acoustoelectric-imaging.md) also images current density, but uses ultrasound modulation of conductivity.

## Relevant Projects

| Project | Role |
|---|---|
| `neuroelectric-mri` | Literature atlas (54 annotated sources); phase budgets; FV tank, DOLFINx CEM, RT adaptive and GetDP verification; current/MRSI benchmark; Prisma open-data plan |
| `mri-rf-stack` | 3 T birdcage B1/SAR benchmark (Hertzon/EMerge); B⁺/B⁻ conventions for EPT-type work; InI minimum-norm inverse; JEMRIS coil export |
| `neurojax` | Differentiable head modeling (BEM/FEM, CHARM 60-tissue), qMRI→σ mapping and DTI anisotropy; consumer of MREIT-calibrated conductivities |
| `ti-sim` | Gmsh/OpenCASCADE sphere builder; shared mesh, tag and current conventions for stimulation fields |
| `mrs-jax` | MRSI/EPSI code used in the current-phase spectroscopy benchmark |
| `inferjax` | Amortized SBI, applicable once forward models and null controls are credible |

## See Also

- [physics-electromagnetic.md](physics-electromagnetic.md) and [physics-bloch.md](physics-bloch.md) for field and phase physics
- [tissue-electrical-properties.md](tissue-electrical-properties.md), [tissue-gray-matter.md](tissue-gray-matter.md) and [tissue-white-matter.md](tissue-white-matter.md) for conductivity targets
- [method-fem.md](method-fem.md) and [method-bem.md](method-bem.md) for forward solvers
- [method-source-imaging.md](method-source-imaging.md) and [method-sbi.md](method-sbi.md) for inverse and inference methods
- [meg.md](meg.md), [eeg.md](eeg.md) and [fmri.md](fmri.md) for complementary readouts of neural activity
- [temporal-interference.md](temporal-interference.md), [tms.md](tms.md) and [acoustoelectric-imaging.md](acoustoelectric-imaging.md)
- [structural-mri.md](structural-mri.md), [diffusion-mri.md](diffusion-mri.md), [quantitative-mri.md](quantitative-mri.md) and [mrs.md](mrs.md)
- [sci-head-model.md](sci-head-model.md) and [head-model-mida.md](head-model-mida.md) for anatomical conductor geometries
- [coordinate-systems.md](coordinate-systems.md) for B0-aligned scanner frames versus head frames
