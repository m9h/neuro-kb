---
type: modality
timestamp: 2026-06-12T16:08:30-07:00
title: Structural MRI
description: "Structural MRI encompasses T1-weighted, T2-weighted, and proton density-weighted imaging sequences that provide anatomical contrast based on tissue-specific magnetic relaxation properties."
physics: electromagnetic
measurement: T1, T2, proton density, tissue contrast
spatial_resolution: 0.5-2.0 mm
temporal_resolution: 1-20 minutes per sequence
related: [quantitative-mri.md, foundation-models.md, coordinate-systems.md, head-model-mida.md, method-sbi.md]
---

# Structural MRI

Structural MRI encompasses T1-weighted, T2-weighted, and proton density-weighted imaging sequences that provide anatomical contrast based on tissue-specific magnetic relaxation properties. These sequences form the foundation for brain segmentation, registration, and anatomical modeling in neuroimaging.

## Core Sequences

### T1-weighted Imaging
- **MPRAGE/MP2RAGE**: Magnetization-prepared rapid gradient echo
- **SPGR**: Spoiled gradient recalled echo
- **Inversion Recovery**: Variable inversion time sequences

### T2-weighted Imaging
- **FSE/TSE**: Fast/turbo spin echo
- **FLAIR**: Fluid-attenuated inversion recovery (CSF suppression)
- **T2*-weighted GRE**: Gradient echo with susceptibility contrast

### Proton Density
- **Dual-echo sequences**: Simultaneous T2 and PD contrast
- **Balanced SSFP**: Steady-state free precession

## Quantitative Parameters

### Tissue Relaxation Times (3T)

| Tissue | T1 (ms) | T2 (ms) | Source |
|--------|---------|---------|--------|
| Grey matter | 1331 ± 13 | 110 ± 2 | Wansapura et al. 1999 |
| White matter | 832 ± 10 | 79.6 ± 0.6 | Wansapura et al. 1999 |
| CSF | 4163 ± 1155 | 2569 ± 1183 | Wansapura et al. 1999 |
| Blood | 1932 ± 85 | 275 ± 50 | Lu et al. 2004 |

### Signal Intensities (Normalized)
- **T1w**: WM > GM > CSF
- **T2w**: CSF > GM > WM
- **FLAIR**: GM > WM, CSF suppressed

## Preprocessing Pipeline

### Standard Workflow
1. **Bias field correction** (N4ITK, ANTs)
2. **Brain extraction** (BET, 3dSkullStrip)
3. **Tissue segmentation** (FAST, FreeSurfer, SAMSEG)
4. **Registration** (FLIRT, ANTs, FreeSurfer)

### Advanced Processing
- **Surface reconstruction** (FreeSurfer recon-all)
- **Cortical parcellation** (Desikan-Killiany, Destrieux)
- **Subcortical segmentation** (FIRST, ASEG)

### Deep-learning preprocessing
- **T1Prep / PyCAT** — T1-weighted preprocessing for segmentation and cortical
  surface reconstruction, integrating **deepmriprep** [@fisch2024deepmriprep] (DL
  bias-field correction, lesion detection, and an AMAP-segmentation initializer that
  mimics CAT12) with CAT-Surface for thickness. BIDS-derivatives–compatible naming;
  single-subject or batch. A fast, FreeSurfer-alternative source of the **morphometry
  features** used as baselines in structural-MRI foundation-model benchmarks.
- **DeepMriPrep / SynthSeg / SAMSEG** — contrast-robust DL segmentation for
  large-scale heterogeneous clinical data.

## Applications in Connected Projects

### Head Modeling
Structural MRI provides the anatomical foundation for electromagnetic forward modeling:
- **Tissue segmentation** → conductivity assignment
- **Surface meshes** → BEM/FEM geometry
- **Registration** → sensor-to-anatomy alignment

### Quantitative Property Mapping
Multi-parameter protocols enable tissue property estimation:
- **VFA sequences** → T1 mapping via DESPOT1
- **Multi-echo GRE** → T2* mapping, susceptibility
- **QMT protocols** → macromolecular content estimation

## BIDS Structure

```text
sub-XX/
├── ses-YY/
│   └── anat/
│       ├── sub-XX_ses-YY_T1w.nii.gz
│       ├── sub-XX_ses-YY_T2w.nii.gz
│       ├── sub-XX_ses-YY_FLAIR.nii.gz
│       └── sub-XX_ses-YY_acq-VFA_flip-NN_T1w.nii.gz
```

## Quality Control

### SNR Assessment
- **Background ROI sampling**: 5×5×5 voxel regions
- **Tissue ROI definition**: FreeSurfer parcellation
- **SNR threshold**: >20 for GM, >15 for WM

### Artifact Detection
- **Motion artifacts**: Visual inspection, automated metrics
- **B0 inhomogeneity**: Field map analysis
- **RF coil uniformity**: Intensity profiles across FOV

## Integration Points

### With Diffusion MRI
- **Registration target**: T1w as anatomical reference
- **Tissue priors**: Structural segmentation for microstructure fitting
- **Tractography constraints**: WM/GM boundaries

### With Functional MRI
- **Registration**: EPI → T1w alignment
- **Parcellation**: Anatomical ROIs for connectivity
- **Nuisance modeling**: CSF/WM signal regression

### With MEG/EEG
- **Head model construction**: Multi-tissue BEM/FEM
- **Source space definition**: Cortical surface constraints
- **Coregistration**: Fiducials to MRI space

## Technical Specifications

### Acquisition Parameters
- **Resolution**: 1mm isotropic (standard), 0.5mm (high-res)
- **Matrix size**: 256×256×176 typical
- **Acceleration**: GRAPPA R=2, partial Fourier 6/8
- **Scan time**: 3-8 minutes per sequence

### Contrast Optimization
- **T1w**: TR ~2000ms, TE ~3ms, TI ~900ms
- **T2w**: TR ~5000ms, TE ~100ms
- **FLAIR**: TR ~9000ms, TE ~125ms, TI ~2500ms

## Citations
- **Fischl2012freesurfer**: Fischl (2012). FreeSurfer. NeuroImage 62:774-781.
- **Billot2023synthseg**: Billot et al. (2023). Robust machine learning segmentation for large-scale analysis of heterogeneous clinical brain MRI datasets. PNAS 120(9).
- **Hoffmann2022synthmorph**: Hoffmann et al. (2022). SynthMorph: learning contrast-invariant registration without acquired images. IEEE TMI 41:543-558.
- **Puonti2016samseg**: Puonti et al. (2016). Fast and sequence-adaptive whole-brain segmentation using parametric Bayesian modeling. NeuroImage 143:235-249.
- **Dale1999cortical**: Dale et al. (1999). Cortical surface-based analysis I: segmentation and surface reconstruction. NeuroImage 9:179-194.
- **Jenkinson2012fsl**: Jenkinson et al. (2012). FSL. NeuroImage 62:782-790.
- **wansapura1999nmr**: Wansapura et al. (1999). NMR relaxation times in the human brain at 3.0 Tesla. JMRI 9:531-538.

## Foundation Models & Morphometry Baselines

Structural MRI is the substrate for the **`smri-fm`** structural-MRI foundation model,
which benchmarks frozen FM representations against **morphometry baselines** — the
volumetric (ASEG/voxel-based) and surface (cortical thickness/area) features produced
by FreeSurfer and T1Prep above. The recurring finding in this benchmark family is that
morphometry floors and minimal preprocessing are strong, so the FM must clear a real
bar (cf. the cost-utility result that minimal preprocessing is near-optimal for
segmentation and brain-age). The DLBS (ds004856) cohort is shared with the
[benchmark-datasets.md](benchmark-datasets.md) fMRI plugin as its structural sibling.
See [foundation-models.md](foundation-models.md).

## Relevant Projects

- **smri-fm**: MedARC structural-MRI foundation model; FM features vs FreeSurfer/T1Prep morphometry baselines
- **T1Prep**: DL-based (deepmriprep + CAT-Surface) T1 preprocessing, segmentation, and cortical thickness — a morphometry-feature source
- **neurojax**: Structural MRI preprocessing and head model construction via FreeSurfer integration
- **sbi4dwi**: Anatomical priors for diffusion microstructure estimation
- **hippy-feat**: T1w reference for fMRI registration and parcellation-based connectivity

## See Also

- [foundation-models.md](foundation-models.md) — smri-fm and the two-axis FM benchmark
- [benchmark-datasets.md](benchmark-datasets.md) — evaluation cohorts (DLBS shared with smri-fm)
- [quantitative-mri.md](quantitative-mri.md) — Relaxometry and tissue property mapping
- [head-model-mida.md](head-model-mida.md) — Forward modeling geometry
- [coordinate-systems.md](coordinate-systems.md) — MNI / FreeSurfer / surface spaces