---
type: modality
title: Structural MRI
description: "Structural MRI uses T1-, T2-, and proton-density-weighted relaxation contrast to image brain anatomy, and is the geometric substrate for segmentation, registration, head models, morphometry, and structural foundation models across the project family."
timestamp: 2026-11-05T16:02:17-08:00
tags: [mri, anatomy, segmentation, morphometry, head-modeling]
physics: electromagnetic
measurement: Proton-density- and relaxation-weighted (T1, T2, T2*) tissue contrast; derived tissue labels, surfaces, and morphometry
spatial_resolution: 0.5-2.0 mm isotropic (1 mm standard; 1 mm grids for SCI/MNI152 head models)
temporal_resolution: 3-8 min per 3D anatomical volume (not time-resolved)
related: [quantitative-mri.md, physics-bloch.md, diffusion-mri.md, fmri.md, meg.md, eeg.md, mrs.md, foundation-models.md, brain-age-benchmarking.md, benchmark-datasets.md, coordinate-systems.md, data-formats.md, head-model-mida.md, sci-head-model.md, method-bem.md, method-fem.md, method-source-imaging.md, tes-efield-modeling.md, tus.md, tissue-gray-matter.md, tissue-white-matter.md, tissue-csf.md, tissue-skull.md, tissue-scalp.md, tissue-electrical-properties.md]
---

# Structural MRI

Structural MRI (sMRI) acquires anatomical images whose contrast comes from differences in tissue proton density and relaxation times (T1, T2, T2*), governed by the Bloch equations ([physics-bloch.md](physics-bloch.md)). Unlike [quantitative-mri.md](quantitative-mri.md), which fits physical parameters per voxel, conventional sMRI produces *weighted* images in arbitrary units. In this project family it plays three roles:

1. **Geometry** — tissue labels and surfaces for BEM/FEM/acoustic head models ([method-bem.md](method-bem.md), [method-fem.md](method-fem.md)).
2. **Reference frame** — the native-space anchor that every other modality (DWI, fMRI, MRS, MEG) is registered to ([coordinate-systems.md](coordinate-systems.md)).
3. **Features** — morphometry (volumes, thickness, area) and raw T1w volumes for foundation models and brain-age ([foundation-models.md](foundation-models.md), [brain-age-benchmarking.md](brain-age-benchmarking.md)).

## Core Sequences

| Contrast | Sequence | Typical 3T parameters | Tissue ordering |
|---|---|---|---|
| T1w | MPRAGE | TR ≈ 2000–2300 ms, TE ≈ 3 ms, TI ≈ 900 ms, 1 mm iso | WM > GM > CSF |
| T1w (bias-robust) | MP2RAGE | Two inversions; ratio image cancels B1−/PD | WM > GM > CSF |
| T1w / VFA | SPGR (spoiled GRE) | Multiple flip angles (WAND: 8 volumes) | Flip-angle dependent |
| T2w | FSE/TSE | TR ≈ 5000 ms, TE ≈ 100 ms | CSF > GM > WM |
| T2-FLAIR | IR-TSE | TR ≈ 9000 ms, TE ≈ 125 ms, TI ≈ 2500 ms | GM > WM, CSF nulled |
| T2*w | Multi-echo GRE | Several echoes; also feeds QSM | Susceptibility-weighted |
| bSSFP | Balanced SSFP | WAND: 16 volumes (mcDESPOT) | T2/T1 ratio |

Sequence parameters above are typical vendor-protocol values carried forward from the prior page revision; the WAND-specific counts come from `neurojax:docs/CUBRIC_QUESTIONS.md`.

### Tissue Relaxation Times (3T)

| Tissue | T1 (ms) | T2 (ms) | Source / note |
|---|---|---|---|
| Grey matter | 1331 ± 13 | 110 ± 2 | Wansapura et al. 1999 (JMRI 9:531) — no bibliography key yet |
| White matter | 832 ± 10 | 79.6 ± 0.6 | Wansapura et al. 1999 — no bibliography key yet |
| CSF | 4163 ± 1155 | 2569 ± 1183 | Carried forward from prior revision; **primary source unverified** |
| Blood | 1932 ± 85 | 275 ± 50 | Prior revision attributed this row to Lu et al. 2004. These numbers match Stanisz et al. 2005 (MRM) instead. Lu 2004 reports a blood T1 near 1.65 s. **Verify before use.** |

The large GM/WM T1 gap (~500 ms at 3T) is what makes MPRAGE the default segmentation contrast. For per-tissue property pages, see [tissue-gray-matter.md](tissue-gray-matter.md), [tissue-white-matter.md](tissue-white-matter.md), and [tissue-csf.md](tissue-csf.md).

## Reference Protocol: WAND (CUBRIC)

The Welsh Advanced Neuroimaging Database [@mcnabb2025wand] is the main real-data testbed for `neurojax` and `sbi4dwi`. It contains 170 subjects with multi-session data acquired on several scanners: Connectom 300 mT/m, Prisma, and 7T.

| Session | Structural / qMRI content | Notes |
|---|---|---|
| ses-02 | T1w, VFA (mcDESPOT SPGR ×8, SSFP ×16, SPGR-IR), QMT | Connectom 300 mT/m. QMT saturation angles listed as 332/628/333°. This is the **native-space anchor**. |
| ses-03 | T1w, T2w | Prisma. The T2w feeds FreeSurfer pial refinement and T1w/T2w myelin maps. |
| ses-06 | MP2RAGE, multi-echo GRE | Used for T1, T2*, and R2* maps. |
| — | No UTE/ZTE acquisition identified | Skull must come from pseudo-CT derived from T1w (open question to CUBRIC). |

Sources: `neurojax:docs/BIDS_DERIVATIVES_STRUCTURE.md` and `neurojax:docs/CUBRIC_QUESTIONS.md`.

Several protocol details are still unresolved and have been raised with CUBRIC:

- The exact SPGR flip angles. The assumed set is [2, 4, 6, 8, 10, 12, 14, 18]°, but the BIDS sidecar reports only 18°.
- The SSFP phase-cycling scheme.
- The QMT RF pulse duration, estimated at ~15 ms.

These matter for the qMRI fits in [quantitative-mri.md](quantitative-mri.md).

## Processing Pipeline (neurojax WAND derivatives)

Processing stays in **native space first** and follows the BIDS-derivatives layout (BEP003). Tool versions are FSL 6.0.7.22 and FreeSurfer 8.2.0 (`neurojax:docs/BIDS_DERIVATIVES_STRUCTURE.md`).

| Stage | Tool | Key outputs |
|---|---|---|
| A. Structural preprocessing | `fsl_anat` | `T1_biascorr`, brain mask, FAST PVE (0 = CSF, 1 = GM, 2 = WM), FIRST subcortical, FNIRT warp to MNI |
| B. Surface reconstruction | FreeSurfer `recon-all` (T1w + T2w) | `lh/rh.white`, `pial`, `sphere.reg`; `aparc+aseg` (Desikan, 68 cortical parcels); `aparc.a2009s+aseg` (Destrieux, 148 parcels) |
| G. Registration | FLIRT / FNIRT | Cross-session 6-DOF rigid to ses-02 T1w; T1w→MNI152 1 mm (affine + nonlinear) |
| L. Advanced FreeSurfer | SAMSEG, SynthSeg, subfield tools | T1w/T2w ratio myelin maps (volume + surface); cross-session SynthSeg; thalamic nuclei (v13); hippocampal subfields (v22); hypothalamic subunits |
| I. MRS support | `svs_segment` | SVS voxel mask in T1w space → GM/WM/CSF fractions for water-scaled quantification ([mrs.md](mrs.md)) |

**Registration strategy.** Every session is registered *to* the ses-02 T1w, which keeps the structural connectome as the spatial reference. MEG is coregistered to the same image through MNE fiducials. The planned upgrade from FNIRT to FSL's deep-learning registration with the OM-1 template is pending; FNIRT remains the fallback.

Whether rigid alignment is sufficient across scanners (Connectom vs Prisma) has not yet been validated. See [coordinate-systems.md](coordinate-systems.md).

**Known tool bugs** (`neurojax:docs/BUG_REPORTS.md`):

- FreeSurfer 8.2.0 `qatools.py --screenshots/--fornix` crashes because of a scipy sparse 0-d indexing change.
- The same tool also fails on a pandas `warn_bad_lines` removal (line 114).
- QC metrics still work without screenshots.

### Deep-learning preprocessing

- **T1Prep / PyCAT** handles T1w preprocessing.
  - It combines deepmriprep (DL bias correction, lesion detection, and an AMAP-style segmentation that mimics CAT12) with CAT-Surface for cortical thickness.
  - Output naming is BIDS-derivatives compatible.
  - It is a fast, FreeSurfer-alternative source of morphometry features.
- **SynthSeg / SAMSEG** provide contrast-agnostic segmentation for heterogeneous clinical data. In WAND, SynthSeg is used for cross-session consistency.

## Structural MRI → Head Models

The `neurojax` head-modeling chain (`neurojax:README.md`) is:

```
T1w ─► CHARM 60-tissue segmentation (SAMSEG + extended atlas)
    ─► tissue surfaces (marching cubes)
    ─► conductivity assignment (literature or qMRI-informed σ(T1, BPF))
    ─► BEM (MNE / OpenMEEG, 3-layer) or FEM (JAX-FEM + PETSc)
    ─► leadfield L  (Y = L·J + n)  ─► 15 inverse solvers
```

Because `sigma_from_qmri(t1, bpf, labels, params)` is differentiable, `jax.grad` can flow from source-localisation error back to tissue parameters. This links sMRI segmentation to [tissue-electrical-properties.md](tissue-electrical-properties.md), [method-source-imaging.md](method-source-imaging.md), [meg.md](meg.md), and [eeg.md](eeg.md).

The PI-GNN source solver uses cortical-surface vertex features derived from sMRI: normals, curvature, and the T1w/T2w myelin proxy.

Labelled anatomical volumes are also rasterized for acoustic simulation in `sbi4dwi` (`sbi4dwi:docs/openlifu/README.md`):

| Head model | Grid | Voxel | Tissue classes | Use |
|---|---|---|---|---|
| SCI Institute (Warner 2019) | 208×256×256 | 1 mm | 8 | j-Wave TUS; 93.0% skull attenuation at 400 kHz |
| SimNIBS "Ernie" (MNI152) | 182×238×282 | 1 mm | 10 → 6 remapped | 256-element FWI at 555 kHz |
| CHARM (per subject, from T1w) | native | ~1 mm | 60 | neurojax BEM/FEM |

Tissue labels are mapped to acoustic properties using ITRUSST benchmark values [@aubry2022itrusst], or estimated continuously from HU when a CT or pseudo-CT is available.

Skull is the weak point of T1w-only pipelines because cortical bone has very short T2*. In sbi4dwi's dp/dc sensitivity analysis, focal pressure was **232× more sensitive** to sound-speed changes in the skull region than in water. Missing UTE/ZTE data therefore propagates directly into [tus.md](tus.md) and [tes-efield-modeling.md](tes-efield-modeling.md) error budgets. See [sci-head-model.md](sci-head-model.md), [head-model-mida.md](head-model-mida.md), and [tissue-skull.md](tissue-skull.md).

## Integration With Other Modalities

- **[diffusion-mri.md](diffusion-mri.md)**
  - DWI is registered to the T1w anchor (`ses-02_dwi_to_T1w.mat`).
  - Desikan/Destrieux parcellations are warped to diffusion space for SIFT2 connectomes.
  - FAST/aseg labels provide priors for microstructure fitting [@fick2019dmipy].
- **[fmri.md](fmri.md)**
  - EPI→T1w registration (`ses-03_func_to_T1w.mat`), anatomical ROIs, and WM/CSF nuisance regressors.
  - hippy-feat uses the T1w as the anatomical reference for its GPU preprocessing and parcellation-based connectivity [@hough2026hippyfeat; @ciric2022differentiable].
- **[mrs.md](mrs.md)**: voxel tissue fractions computed from the T1w segmentation.
- **MR-ARFI**: the sbi4dwi `mr_arfi`/`multimodal_tus` modules predict MRI phase from ultrasound-induced displacement.
  - Displacement sensitivity is ~0.1 µm at ~100 ms per image [@kaye2011mrarfi; @kaye2013mrarfi].
  - The modules require structural tissue labels to assign region-specific shear moduli [@linka2023cann].

## Foundation Models & Morphometry Baselines

`smri-fm` benchmarks frozen structural-MRI foundation-model embeddings against **morphometry baselines**. These baselines are ASEG/voxel volumes and cortical thickness and area from FreeSurfer and T1Prep.

The recurring finding is that morphometry floors and minimal preprocessing are strong. Minimal preprocessing is near-optimal for segmentation and brain-age in cost-utility terms, so an FM must clear a real bar.

The DLBS cohort (ds004856) is shared as the structural sibling of the fMRI evaluation cohorts in [benchmark-datasets.md](benchmark-datasets.md). See [foundation-models.md](foundation-models.md) and [brain-age-benchmarking.md](brain-age-benchmarking.md).

## Quality Control

| Check | Criterion |
|---|---|
| SNR | Background ROI 5×5×5 voxels; tissue ROIs from FreeSurfer; target > 20 (GM), > 15 (WM) |
| Motion | Visual inspection and automated metrics (FreeSurfer qatools) |
| B0 / B1 | Field-map analysis; prefer MP2RAGE when bias fields are severe |
| Cross-session | Rigid residuals to the ses-02 anchor |

## Relevant Projects

- **neurojax**
  - WAND structural pipeline (fsl_anat, FreeSurfer, SAMSEG/SynthSeg).
  - CHARM 60-tissue segmentation feeding differentiable BEM/FEM.
  - qMRI-informed conductivity.
  - PI-GNN cortical-mesh features.
- **sbi4dwi**
  - Rasterizes label maps (SCI, MNI152/Ernie) for acoustic TUS simulation and FWI.
  - mcDESPOT/QMT/EPG signal models sharing WAND ses-02 data.
  - Pulseq Bloch simulation.
- **hippy-feat**: T1w anchor for fMRI registration and parcellation-based connectivity.
- **smri-fm**: structural-MRI foundation model evaluated against morphometry baselines.
- **T1Prep**: deepmriprep + CAT-Surface segmentation, VBM, and cortical thickness.
- **insula**: listed as a source project. No structural-MRI-specific module was documented in the ingested material, so details are pending.

## See Also

- [quantitative-mri.md](quantitative-mri.md) — DESPOT1/mcDESPOT, QMT BPF, T2* mapping
- [physics-bloch.md](physics-bloch.md) — relaxation physics behind contrast
- [quantitative-susceptibility-mapping.md](quantitative-susceptibility-mapping.md) — multi-echo GRE phase
- [diffusion-mri.md](diffusion-mri.md), [fmri.md](fmri.md), [mrs.md](mrs.md) — modalities anchored to T1w
- [method-bem.md](method-bem.md), [method-fem.md](method-fem.md) — forward models built on sMRI geometry
- [sci-head-model.md](sci-head-model.md), [head-model-mida.md](head-model-mida.md) — reference head models
- [connectomics.md](connectomics.md) — parcellation-based structural connectomes
- [coordinate-systems.md](coordinate-systems.md) — native, MNI152, and FreeSurfer surface spaces
- [data-formats.md](data-formats.md) — NIfTI, MGZ, and BIDS derivatives
- [foundation-models.md](foundation-models.md), [brain-age-benchmarking.md](brain-age-benchmarking.md)

## Citations

Bibliography keys:

- [@mcnabb2025wand] — WAND multi-modal dataset (CUBRIC)
- [@fick2019dmipy] — Dmipy multi-compartment modeling (structural priors for microstructure)
- [@aubry2022itrusst] — ITRUSST transcranial ultrasound benchmarks (label→acoustic property mapping)
- [@kaye2011mrarfi], [@kaye2013mrarfi] — MR-ARFI sensitivity and timing
- [@linka2023cann] — region-specific brain shear moduli
- [@hough2026hippyfeat], [@ciric2022differentiable] — fMRI preprocessing and connectivity using a T1w anchor

Project sources:

- `neurojax:README.md` — CHARM → BEM/FEM pipeline, `sigma_from_qmri`
- `neurojax:docs/BIDS_DERIVATIVES_STRUCTURE.md` — derivatives layout, registration strategy, tool versions
- `neurojax:docs/CUBRIC_QUESTIONS.md` — WAND VFA/QMT protocol details and missing UTE/ZTE data
- `neurojax:docs/BUG_REPORTS.md` — FreeSurfer 8.2.0 qatools bugs
- `sbi4dwi:docs/openlifu/README.md` — SCI and Ernie grids, attenuation, dp/dc sensitivity

References without a bibliography key yet (to be added):

- Wansapura et al. 1999, JMRI 9:531–538
- Stanisz et al. 2005, MRM 54:507–512
- Fischl 2012, NeuroImage 62:774–781
- Dale et al. 1999, NeuroImage 9:179–194
- Jenkinson et al. 2012, NeuroImage 62:782–790
- Puonti et al. 2016 (SAMSEG), NeuroImage 143:235–249
- Billot et al. 2023 (SynthSeg), PNAS 120(9)
- Hoffmann et al. 2022 (SynthMorph), IEEE TMI 41:543–558
- Fisch et al. 2024 (deepmriprep)
