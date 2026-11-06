---
type: modality
title: Temporal Interference Stimulation
physics: electromagnetic
measurement: "None (stimulation modality). Delivers two kHz quasi-static electric fields whose sum is amplitude-modulated at Δf. The quantity of interest is the envelope amplitude E_AM (V/m) in tissue."
spatial_resolution: "Envelope hotspot about 3-4 cm in humans (HWHM 0.34-0.5 R in the Grossman phantom). Steerable by current ratio and montage."
temporal_resolution: "Beat Δf of 1-100 Hz (typically 5-20 Hz) on a 1-10 kHz carrier (typically 1-2 kHz)."
related: [physics-electromagnetic.md, method-fem.md, method-bem.md, head-model-mida.md, sci-head-model.md, tissue-electrical-properties.md, tissue-skull.md, tissue-csf.md, tissue-scalp.md, tissue-gray-matter.md, tissue-white-matter.md, diffusion-mri.md, structural-mri.md, coordinate-systems.md, tms.md, tus.md, eeg.md, fmri.md, acoustoelectric-imaging.md, compute-infrastructure.md, neural-mass-models.md, tes-efield-modeling.md]
description: "The field physics is settled and reproducible with open tools."
timestamp: 2026-11-05T15:59:53-08:00
tags: [modality, electromagnetic, stimulation, tes]
---

# Temporal Interference Stimulation

**Temporal interference (TI, tTIS)** is a transcranial electrical stimulation scheme. Two electrode pairs are driven at kHz frequencies f₁ and f₂ = f₁ + Δf. Neurons are assumed not to follow the carrier. Where the two fields overlap, their sum is a carrier whose amplitude beats at Δf. The beat is strongest where **E₁** and **E₂** are equal and parallel, and that region can lie deep in the brain rather than under the electrodes [@grossman2017noninvasive].

The field physics is settled and reproducible with open tools. The human field strengths are small. The cellular mechanism remains open (ti-sim:docs/research/01-literature.md).

## Physics: one Laplace solve per pair plus geometry

At 1-10 kHz, brain tissue is in the quasi-static regime (see [physics-electromagnetic.md](physics-electromagnetic.md)). Each pair's field is therefore a scaled solution of ∇·(σ∇φ) = 0 with electrode boundary conditions. Along a unit direction **n**, the modulation (envelope) amplitude is [@grossman2017noninvasive]:

$$E_{AM}(\mathbf n) = \big|\,|(\mathbf E_1+\mathbf E_2)\cdot\mathbf n| - |(\mathbf E_1-\mathbf E_2)\cdot\mathbf n|\,\big|$$

The maximum over **n** (E_AM,max, "E_TIfree") has a closed form. Where one field is much weaker, it reduces to E_AM = 2|E_weak|.

This has a practical consequence: **in the target, the envelope is set by the weaker pair**. Placement tolerance is therefore asymmetric between the two pairs (ti-sim:docs/research/02-replication-targets.md §3).

The closed form is implemented in:
- `ti_sim.envelope`
- SimNIBS `simnibs.utils.TI_utils.get_maxTI` / `get_dirTI`

The two agree to 1e-12 (ti-sim:README.md).

Because the fields are linear in current, a **lead field** (one solve per electrode) lets every two-pair montage be scored without further FEM solves. This is the basis of exhaustive montage search [@lee2020individually; @rampersad2019prospects].

**Definition caveat.** Some tools report peak-to-trough E_AM (Grossman / SimNIBS / ti-sim), while others report "modulation amplitude" (half of it). This is an exact factor of 2. It is the leading explanation for ti-sim's residual 1.9× discrepancy against Sim4Life numbers (see below).

## Typical parameters

| Parameter | Value | Source |
|---|---|---|
| Carrier f₁ / f₂, phantom | 1 kHz / 1.02 kHz, 1 mA per pair | [@grossman2017noninvasive] |
| Carrier, mouse | 2 kHz / 2.01 kHz, 125 µA per channel (hippocampus) | [@grossman2017noninvasive] |
| Carrier, human hippocampus | 2 kHz / 2.005 kHz, 1 mA per pair (model), 2 mA (fMRI) | [@violante2023noninvasive] |
| Carrier, human striatum | F3-F4 + TP7-TP8, 2 mA each, 9.8 mm electrodes; theta-burst 2.00 / 2.10 kHz | [@wessel2023noninvasive] |
| Human envelope, deep targets | 0.2-0.4 V/m at 2 mA | [@rampersad2019prospects; @violante2023noninvasive; @wessel2023noninvasive] |
| Hippocampus, MIDA / Sim4Life | 0.26 ± 0.04 V/m; overlying cortex 0.10-0.18 V/m | [@violante2023noninvasive] |
| Striatum, MIDA / Sim4Life | 0.26 ± 0.04 V/m (median 0.27, p99 0.35); hand knob 0.20 V/m | [@wessel2023noninvasive] |
| Standard montage, 1 mA per pair | hippocampus 0.064, pallidum 0.11, M1 0.25 V/m | [@rampersad2019prospects] |
| Optimised four-electrode montage, 2 mA total | hippocampus 0.24, pallidum 0.37, M1 0.56 V/m | [@rampersad2019prospects] |
| Current for suprathreshold human stimulation | ~38 mA per pair | [@rampersad2019prospects] |
| Mouse brain field (Grossman montage, 0.776 mA) | 383 V/m (suprathreshold) | [@rampersad2019prospects] |
| Slice threshold to modulate gamma | ~5 V/m (100 Hz carrier), ~60 V/m (1 kHz), ~80 V/m (2 kHz) | [@esmaeilpour2021temporal] |
| Safety | 24 studies, 853 participants, no serious adverse events at ≤ 2 mA per channel | [@demchenko2026safety] |

The TI-vs-tACS trade-off is focality, not intensity. tACS with the same electrodes is slightly stronger in the target, but covers 10-30× more brain above 0.25 V/m [@rampersad2019prospects]. On the SimNIBS `ernie` head, brain volume above 0.2 V/m is 20 cm³ for TI vs 330 cm³ for tACS, a 17× difference (ti-sim:docs/research/02-replication-targets.md §3).

## Conductivities used in TI models

Conductivity tables matter at the 10 % level. See [tissue-electrical-properties.md](tissue-electrical-properties.md) for full tables.

| Tissue | Grossman 4-layer sphere (S/m) | Thickness (fraction of R) | Notes |
|---|---|---|---|
| Scalp ([tissue-scalp.md](tissue-scalp.md)) | 0.333 | 0.05 | |
| Skull ([tissue-skull.md](tissue-skull.md)) | 0.0083 | 0.085 | SimNIBS default 0.010; Rampersad table 0.007 |
| CSF ([tissue-csf.md](tissue-csf.md)) | 1.79 | 0.023 | |
| Brain ([tissue-gray-matter.md](tissue-gray-matter.md), [tissue-white-matter.md](tissue-white-matter.md)) | 0.333 | 0.83 | |

Source for the sphere: [@grossman2017noninvasive], fig. S2.

Moving from the SimNIBS skull value (0.010 S/m) to Rampersad's (0.007 S/m) on the MIDA 1 mm mesh lowers the deep medians by **8 %** (ti-sim:docs/research/02-replication-targets.md §4). White-matter anisotropy from [diffusion-mri.md](diffusion-mri.md) changes the envelope by up to 18 % in WM and < 12 % in deep targets [@yatsuda2026impact].

## Replication results (ti-sim)

### Grossman phantom

The phantom is a 2-D scikit-fem disc: 50 mm diameter, 10 mm saline layer, σ = 0.333 S/m.

- **Geometry reproduced.** Every peak position and HWHM in layouts A-E matches to ≤ 0.03 R. For example, layout C peaks at (0, 0) with HWHM (0.42, 0.50) R, vs the paper's (0.42, 0.49). Current-ratio steering moves the peak to x = −0.23 / −0.34 R at 1:2.5 / 1:4, vs the paper's −0.2 / −0.35 R.
- **Amplitudes differ.** ti-sim's amplitudes are a constant **~7×** above the paper's simulated values (4.7-5.6× next to the wires).
- **Likely cause.** A uniform factor with identical geometry points to an unstated definition or saline conductivity. Two candidates are the factor-2 conventions, or physical 0.9 % NaCl at ~1.5 S/m.

Source: ti-sim:docs/research/02-replication-targets.md §1.

### 4-layer sphere, R = 85 mm, 1 mA per pair, 1:1

| Quantity | Analytic (point electrodes) | DOLFINx | SimNIBS |
|---|---|---|---|
| E_AM at centre | 0.0565 V/m | 0.0518 V/m | 0.049-0.051 V/m |
| Pair resistance | — | 280 Ω | 279 Ω |

Finite 10 mm pads lower the centre field by 8 % relative to point electrodes. The envelope is a **plateau** over the central ~30 mm, so the element-wise argmax is mesh noise. Report the top-1 % centroid instead.

### Wessel montage on MIDA ([head-model-mida.md](head-model-mida.md)), DGX Spark

| Voxel size | Tets | Wall time | RSS | Putamen median E_AM | GM median |
|---|---|---|---|---|---|
| 2 mm | 1.47 M | 28.7 min | 5.3 GB | 0.549 V/m | 0.275 V/m |
| 1 mm | 3.52 M | 43.7 min | 5.5 GB | 0.493 V/m | 0.257 V/m |
| 0.5 mm (native) | 6.95 M | 111 min | 9.9 GB | 0.493 V/m | 0.253 V/m |

- **Mesh convergence.** The model is mesh-converged at 1 mm. The deep-to-GM ratio is 1.9, matching Wessel's figure.
- **Conductivity and definition.** With the Rampersad table, putamen falls to 0.453 V/m. Halving that (the modulation-amplitude definition) gives **0.23 V/m**, inside Wessel's 0.22-0.30.
- **Fiducial sensitivity.** Corrected fiducials (`fiducials_v2.json`) move the electrodes 3-6 mm but change the medians by < 2 %.

Source: ti-sim:docs/research/02-replication-targets.md §4.

## Numerical pitfalls (a methods checklist)

1. **SimNIBS current normalisation is biased high.**
   - `fem.tdcs` rescales the field by a flux estimate of the electrode current, which can be off by several percent.
   - The exact discrete current follows from the energy identity ∫σ|∇v|² dV = ΔV·I_h.
   - Measured bias: +6.1 % on the sphere, +4 to +13.5 % on ernie, and 64 % for TP7-TP8 on the 2 mm MIDA pads.
   - ti-sim's `head.discrete_current` corrects this. Lead fields (`tdcs_leadfield`, Neumann right-hand side) are unaffected (SimNIBS/exact = 0.9985-1.022).
   - Source: ti-sim:docs/research/02-replication-targets.md §2.
2. **Template heads mislead.** MNI152 montages underestimate deep-target fields, falling at the 16th-21st percentile of 132 CamCAN heads. Personalise montages [@lee2020individually; @conta2021interindividual] (ti-sim:docs/research/01-literature.md).
3. **Average over meshes.** Independent remeshing gives 2-4 % CoV in the hippocampal envelope; average 5-10 meshes (Ivanov 2026; ti-sim:docs/research/01-literature.md).
4. **Report placement sensitivity.**
   - Yamada 2026 (±10 mm displacement): ROI intensity changes 0.4-15 % and the hotspot moves 0.6-6.9 mm [@yamada2026electrodespecific].
   - ti-sim on ernie (±5 mm): F3 up to 7.0 %, TP7/TP8 0.0 % (the weaker-pair effect).
5. **Optimising for intensity is not optimising for focality.**
   - The best ROI median (0.293 V/m) came with 198 cm³ of spill above 0.25 V/m.
   - A focal ranking gives 0.245 V/m with 57 cm³, which matches Rampersad's 0.24.
   - SimNIBS free placement (`TesFlexOptimization`, `e_postproc="max_TI"`) reached a 0.453 V/m ROI mean at 1 mA per pair, but with 511 cm³ of spill.
6. **Anatomy predicts depth.** Skull thickness and CSF fraction predict hippocampal modulation depth without a simulation [@meng2026anatomically].

## Mechanism: the open question

The founding experiments demonstrated effects in vivo but offered no biophysical model of demodulation [@grossman2017noninvasive]. Later work challenges the simple envelope-extraction picture:

- **Passive membranes cannot demodulate.** Passive filtering of two summed sinusoids still yields two sinusoids. Rectification requires ion channels, and that same mechanism causes kHz conduction block off target [@mirzakhalili2020biophysics]. `ti_sim.membrane` and `ti_sim.hh` reproduce this arithmetic.
- **The carrier itself matters.** Thresholds rise steeply with carrier frequency [@esmaeilpour2021temporal]. Partial-demodulation models make the same point [@cerpa2022partially].
- **Peripheral nerves ignore modulation depth.** Budde 2023 and Jabban 2025 found thresholds independent of modulation depth (ti-sim:docs/research/01-literature.md).
- **Human effects are real but modest.** The meta-analysis effect size is Hedges' g ≈ 0.30 (ti-sim:docs/research/01-literature.md). Reported outcomes include hippocampal memory [@violante2023noninvasive], post-task connectivity [@kurtin2025taskrelated], motor learning [@wessel2023noninvasive] and depression [@wang2026temporal]. A null α-aftereffect has also been reported (von Conta 2022).

Variants include:
- Multipair TI [@lee2022multipair]
- A third phase-shifted pair to cancel off-target modulation [@savvateev2026multipair]
- Pulse-width-modulated TI [@luff2024pulsewidth]
- Optimisation studies [@karimi2019computational]

Physical validation comes from a paediatric head phantom with HD-EEG reconstruction, where 7-9 kHz carriers gave the most confined envelopes [@damirchi2026multimodal].

## Software landscape

| Tool | Role |
|---|---|
| SimNIBS 4.6 | charm heads, FEM, `TI_utils`, lead fields, `TesFlexOptimization` |
| TI-Toolbox (GPL-3) | End-user Docker/Electron pipeline for individual MRIs [@haber2026titoolbox] |
| Sim4Life (commercial) | Used by Grossman, Violante and Wessel with MIDA |
| ti-sim | Open phantom → sphere → ernie → MIDA path; montage search; membrane arithmetic |

## Relevant Projects

- **`ti-sim`**: the primary implementation.
  - Modules: `src/ti_sim/{envelope, phantom, head, electrodes, labelvol, mida, montage, membrane, hh, tissues}`.
  - Tests: 30 pytest tests and a 13-stage BATS pipeline asserting published numbers.
  - MIDA mapping: all 115 MIDA structures to SimNIBS tags, with deep nuclei kept as ROIs (`ti_sim.mida.EXTRA_COND`).
  - Electrode placement: a geodesic 10-10 cap from Nz/Iz/LPA/RPA (`ti_sim.electrodes.ten_ten`), with a mean error of 6.4 mm vs ernie's cap.
- **`fem-trials`**: solver trials for ti-sim and ae-sim.
  - dolfinx 0.11 complex path, plus python-mumps (METIS ordering) for large indefinite systems.
  - Benchmark: 1.03 M dofs factored in 1481 s using 42 GB on the DGX Spark.
  - TI itself needs only real-valued Laplace solves. The complex path matters for the coupled acoustic-electric problems of [acoustoelectric-imaging.md](acoustoelectric-imaging.md).
- **`neurojax`**: differentiable FEM assembly (`neurojax.geometry.fem_forward.assemble_stiffness`), qMRI-informed conductivity (`sigma_from_qmri`) and CHARM segmentation.
  - These are candidate building blocks for gradient-based TI montage or conductivity optimisation. They are not yet wired to the TI envelope.
- **`ae-sim`**: shares the open-FEM toolchain and the quasi-static conductivity tables.
- **`neurofedora`**: packages `python3-simnibs 4.6.0` and dolfinx/MUMPS for Fedora (x86_64 and aarch64). It also hosts the independent DOLFINx sphere cross-check (`ci/conductivity-admm/sphere_ti_forward.py`).

## See Also

- [Transcranial Electric Stimulation E-Field Modeling](tes-efield-modeling.md)
- [physics-electromagnetic.md](physics-electromagnetic.md): quasi-static approximation, Laplace equation
- [method-fem.md](method-fem.md), [method-bem.md](method-bem.md): forward solvers
- [head-model-mida.md](head-model-mida.md): the shared reference head for TI (licence forbids redistribution)
- [sci-head-model.md](sci-head-model.md), [structural-mri.md](structural-mri.md): alternative and individual heads
- [tissue-electrical-properties.md](tissue-electrical-properties.md), [tissue-skull.md](tissue-skull.md), [tissue-csf.md](tissue-csf.md), [tissue-scalp.md](tissue-scalp.md), [tissue-gray-matter.md](tissue-gray-matter.md), [tissue-white-matter.md](tissue-white-matter.md)
- [diffusion-mri.md](diffusion-mri.md): conductivity anisotropy
- [coordinate-systems.md](coordinate-systems.md): MNI targets, 10-10 cap, fiducials
- [tms.md](tms.md), [tus.md](tus.md): other deep / non-invasive stimulation modalities
- [eeg.md](eeg.md), [fmri.md](fmri.md): readouts used to validate TI
- [neural-mass-models.md](neural-mass-models.md): population-level response models
- [acoustoelectric-imaging.md](acoustoelectric-imaging.md): sibling coupled-field simulation
- [compute-infrastructure.md](compute-infrastructure.md): DGX Spark memory and timing budgets
