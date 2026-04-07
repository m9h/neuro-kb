---
type: head-model
title: SCI Head Model
source: Neurophotonics Lab (Fang group, Northeastern University)
tissues: 8
resolution: tetrahedral mesh (9.9M tet4 elements)
formats: [nrrd, vtk, mat, h5, node/elem, json]
local_path: /data/datasets/sci_head_model/
related: [tissue-optical-properties.md]
---

# SCI Head Model

Open-source 8-layer head model designed for photon transport simulation (Monte Carlo, MMC). Distributed with mesh-based Monte Carlo (MMC) software by Qianqian Fang's group.

## Contents

### Segmentation (`segmentation/`)
- `HeadSegmentation.nrrd` — 8-layer volumetric segmentation (Seg3D format, each label = separate mask)

### Mesh (`mesh/`)
- `sci_head.node` / `sci_head.elem` — tetrahedral mesh (tet4, ~9.9M elements)
- `HeadMesh.vtk` — VTK format of the same mesh
- `HeadMesh.mat` — MATLAB format
- `sci_head_dynamic.h5` — HDF5 format
- `sci_head_mmc.json` — MMC simulation config with default optical properties

## Tissue layers and optical properties

From `sci_head_mmc.json` Domain.Media (labels 1–7, label 0 = exterior/void):

| Label | Tissue (probable) | μa (1/mm) | μs (1/mm) | g | n |
|-------|-------------------|-----------|-----------|-----|------|
| 0 | Exterior (void) | 0.0 | 0.0 | 1.0 | 1.0 |
| 1 | Scalp | 0.015 | 10.0 | 0.9 | 1.4 |
| 2 | Skull | 0.010 | 15.0 | 0.9 | 1.4 |
| 3 | CSF | 0.002 | 0.1 | 0.9 | 1.33 |
| 4 | Gray matter | 0.020 | 20.0 | 0.9 | 1.4 |
| 5 | White matter | 0.018 | 20.0 | 0.9 | 1.4 |
| 6 | Layer 6 | 0.018 | 20.0 | 0.9 | 1.4 |
| 7 | Layer 7 | 0.018 | 20.0 | 0.9 | 1.4 |

**Note:** Tissue labels 6–7 need verification — may be cerebellum/brainstem or further WM/GM subdivisions. The optical properties are identical to WM.

## Cross-modality relevance

This model is natively set up for **fNIRS / diffuse optical tomography** (photon transport), but the 8-layer segmentation is reusable for:
- **EEG/MEG** forward models — needs conductivity values assigned to each tissue label (see [tissue-electrical-properties.md](tissue-electrical-properties.md))
- **Transcranial ultrasound** — needs acoustic impedance and attenuation per tissue
- **TMS/tDCS** — needs conductivity tensors, especially for anisotropic WM

The tetrahedral mesh is directly usable in FEM solvers (FEniCS, MFEM, custom JAX).

## Usage with MMC

The `sci_head_mmc.json` configures a pencil-beam source at position `[11.06, 87.48, 9.31]` pointing in `-y`, with 10M photons and a 5 ns time gate. This is a starting point for fNIRS simulations.
