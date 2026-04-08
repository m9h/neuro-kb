```yaml
---
type: modality
title: Structural MRI
physics: electromagnetic
measurement: T1, T2, proton density, tissue contrast
spatial_resolution: 0.5-2.0 mm
temporal_resolution: 1-20 minutes per sequence
related: [quantitative-mri.md, tissue-properties.md, head-models.md, t1-mapping.md, t2-mapping.md]
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

```
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

## Relevant Projects

- **neurojax**: Structural MRI preprocessing and head model construction via FreeSurfer integration
- **sbi4dwi**: Anatomical priors for diffusion microstructure estimation
- **hippy-feat**: T1w reference for fMRI registration and parcellation-based connectivity

## See Also

- [quantitative-mri.md](quantitative-mri.md) — Relaxometry and tissue property mapping
- [tissue-properties.md](tissue-properties.md) — Electromagnetic and mechanical properties
- [head-models.md](head-models.md) — Forward modeling geometry
- [registration.md](registration.md) — Cross-modal alignment methods
- [freesurfer.md](freesurfer.md) — Surface-based analysis pipeline
```