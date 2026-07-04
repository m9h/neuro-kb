---
type: method
title: Quantitative Susceptibility Mapping
tags: [method, optimization, quantitative-mri]
description: "Reconstructs a voxelwise map of magnetic susceptibility from gradient-echo MRI phase by unwrapping, removing background fields, and solving an ill-posed dipole deconvolution."
timestamp: 2026-07-03T00:00:00-07:00
category: optimization
implementations: ["QSM.jl:src/inversion", "QSM.jl:src/bgremove", "QUIT:Source/Susceptibility", "qMRLab:src/Models/QSM"]
related: [quantitative-mri.md, physics-bloch.md, structural-mri.md, method-variational-inference.md]
---

# Quantitative Susceptibility Mapping

Quantitative susceptibility mapping (QSM) turns the **phase** of a gradient-echo MRI acquisition into a voxelwise map of tissue magnetic susceptibility χ (in ppm). Susceptibility reflects tissue iron, myelin, calcium, and deoxyhemoglobin, making QSM a biomarker for iron deposition, microbleeds, venous oxygenation, and demyelination.

## Pipeline

QSM is a three-stage inverse problem:

1. **Phase unwrapping** — remove 2π ambiguities (Laplacian, region-growing, or path-based). `QSM.jl:src/unwrap/laplacian.jl`.
2. **Background field removal** — strip fields from sources outside the ROI (PDF, SHARP/V-SHARP, LBV). `QSM.jl:src/bgremove`.
3. **Dipole inversion** — deconvolve the unit dipole kernel `D(k) = 1/3 − k_z²/|k|²`, which is **zero on a conical surface** in k-space, so the inverse is ill-posed. Regularized solvers: TV, RTS, NLTV, direct/TKD. `QSM.jl:src/inversion`.

The zero-cone of the dipole kernel is why dipole inversion needs regularization or multi-orientation data (COSMOS); single-orientation methods trade streaking artifacts against smoothing.

## Implementations

- **QSM.jl** — Julia toolbox: Laplacian unwrap, PDF/SHARP/LBV background removal, TV/RTS/NLTV/direct inversion.
- **QUIT** — C++/ITK `Source/Susceptibility` alongside relaxometry and MT.
- **qMRLab** — MATLAB `Models/QSM` within a multi-contrast quantitative-MR suite.

## Citations

[1] Wang & Liu (2015). Quantitative susceptibility mapping (QSM): Decoding MRI data for a tissue magnetic biomarker. Magn Reson Med.
[2] Langkammer et al. (2018). QSM reconstruction challenge.

## See Also

- [quantitative-mri.md](quantitative-mri.md) - parent quantitative-MR family
- [physics-bloch.md](physics-bloch.md) - phase accrual physics
- [structural-mri.md](structural-mri.md) - anatomical substrate
