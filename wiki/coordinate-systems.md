---
type: coordinate-system
title: Neuroimaging Coordinate Systems
definition: Spatial reference frames for brain imaging, stimulation, and simulation
used_by: [neurojax, vbjax, libspm, LAYNII, sbi4dwi, brain-fwi]
related: [data-formats.md, structural-mri.md, eeg.md, meg.md, sci-head-model.md, method-fem.md, method-source-imaging.md]
---

# Neuroimaging Coordinate Systems

Neuroimaging involves multiple coordinate systems: scanner hardware coordinates, standardized anatomical templates, device-specific sensor frames, and electrode placement systems. Correct transforms between these spaces are essential for source imaging, coregistration, and group analysis.

## Standard Anatomical Spaces

### MNI152

The Montreal Neurological Institute template is the de facto standard for volumetric group analysis.

| Variant | Subjects | Method | Origin | Voxel size |
|---------|----------|--------|--------|------------|
| MNI305 | 305 | Linear average to Talairach | AC | 1 mm |
| ICBM152 Linear (6th gen) | 152 | 9-parameter affine | AC | 1 mm |
| ICBM152 Nonlinear 2009c | 152 | Nonlinear symmetric average | AC | 0.5/1/2 mm |

**MNI305** (Evans et al. 1993) was the original target, created by averaging 305 T1w scans linearly registered to a single Talairach-aligned brain. FreeSurfer still uses MNI305 as its default template (the `talairach.xfm` in recon-all maps to MNI305, not Talairach).

**ICBM152** replaced MNI305 for most purposes. The 2009c nonlinear symmetric version is the current standard used by FSL (`MNI152_T1_1mm.nii.gz`) and SPM (`TPM.nii`). Coordinates between MNI305 and ICBM152 differ by up to ~5 mm, particularly in frontal and temporal regions.

**Origin**: The anterior commissure (AC) is at approximately (0, 0, 0) in all MNI variants. The Y-axis passes through the posterior commissure (PC), defining the AC-PC line.

### Talairach Space

The Talairach atlas (Talairach & Tournoux 1988) was based on a single post-mortem brain of a 60-year-old woman. It defined a piecewise-linear coordinate system using AC, PC, and cortical extent landmarks.

**Talairach-to-MNI conversion** (Lancaster et al. 2007):
```
MNI = T_tal2mni * Talairach
```
where `T_tal2mni` is a piecewise-linear transform (different scaling above/below AC-PC plane). The `icbm2tal` transform is approximately:
```
x_mni ≈ 0.9900 * x_tal
y_mni ≈ 0.9688 * y_tal + 0.0460 * z_tal
z_mni ≈ -0.0485 * y_tal + 0.9189 * z_tal
```

Talairach coordinates should not be used for new analyses. Legacy Talairach coordinates can be converted using the `tal2mni` function in SPM or `mni2tal` from BrainMap.

## Scanner and Voxel Coordinates

### Voxel Indices (ijk)

Raw array indices into the 3D volume: integer values (i, j, k) indexing into the data matrix. No physical meaning without the affine transform.

### The Affine Transform

The 4x4 affine matrix maps voxel indices to physical (typically scanner) coordinates:

```
[x]   [m11 m12 m13 t1] [i]
[y] = [m21 m22 m23 t2] [j]
[z]   [m31 m32 m33 t3] [k]
[1]   [0   0   0   1 ] [1]
```

The 3x3 submatrix encodes voxel size, rotation, and shear. The translation column `[t1, t2, t3]` gives the physical coordinate of voxel (0,0,0).

NIfTI stores two affine representations:
- **qform** (quaternion-based): Rigid body only (6 DOF). Set via `quatern_b/c/d` and `qoffset_x/y/z`.
- **sform** (full affine): 12 DOF. Set via `srow_x`, `srow_y`, `srow_z`.

The `qform_code` and `sform_code` indicate the target space: 1 = scanner, 2 = aligned (AC-PC), 3 = Talairach, 4 = MNI.

### RAS vs LPS Orientation Conventions

| Convention | +X | +Y | +Z | Used by |
|------------|----|----|-----|---------|
| **RAS** | Right | Anterior | Superior | NIfTI, FreeSurfer, MNE-Python |
| **LPS** | Left | Posterior | Superior | DICOM, ITK, 3D Slicer |
| **LAS** | Left | Anterior | Superior | Some legacy SPM, FSL internal |

**RAS-to-LPS conversion**: Negate the first two coordinates:
```
x_LPS = -x_RAS
y_LPS = -y_RAS
z_LPS =  z_RAS
```

NIfTI files always store the affine in RAS convention regardless of the data array layout. The `dim_info` and `xyzt_units` fields describe the mapping between array axes and physical axes. Tools like `nibabel.as_closest_canonical()` reorient data to RAS+ layout.

## FreeSurfer Coordinate Spaces

FreeSurfer defines three coordinate systems, all RAS-oriented:

| Space | Definition | Use |
|-------|-----------|-----|
| **Voxel CRS** | Column-row-slice array indices | Volume indexing |
| **Scanner RAS** | From the NIfTI/MGH affine (Vox2RAS) | Native scanner geometry |
| **tkRAS** | Scanner RAS shifted so volume center = (0,0,0) | Surface display in tkmedit/Freeview |
| **MNI305 RAS** | After `talairach.xfm` (12 DOF affine to MNI305) | Group comparison |

**Key transform chain**:
```
Voxel CRS → Scanner RAS → tkRAS → MNI305 RAS
         Vox2RAS       Vox2RAS-tkr   talairach.xfm
```

The `tkRAS` offset is: `tkRAS = Scanner_RAS - center_of_volume`. Surface vertex coordinates (e.g., `lh.white`, `lh.pial`) are stored in tkRAS. To map surface vertices to scanner coordinates:
```
scanner_RAS = tkRAS + [c_r, c_a, c_s]
```
where `c_r`, `c_a`, `c_s` are read from the volume header.

## MEG Device Coordinate Systems

### CTF Device Coordinates

Used by CTF MEG systems (e.g., CTF-275 at CUBRIC):
- **+X**: Nasion direction (anterior)
- **+Y**: Left ear (left preauricular point)
- **+Z**: Superior (completing right-handed system)
- **Origin**: Midpoint between LPA and RPA, projected onto the nasion-to-inion line

Units: centimeters. Sensor positions in `.ds/res4` file are in this frame.

### Neuromag/Elekta/MEGIN Device Coordinates

- **+X**: Right (toward right preauricular point)
- **+Y**: Anterior (toward nasion)
- **+Z**: Superior
- **Origin**: Midpoint between LPA and RPA

Units: meters. Stored in `.fif` files. MNE-Python uses this convention internally and converts CTF data on import.

### BESA Head Coordinates

- **+X**: Right (RPA direction)
- **+Y**: Anterior (nasion direction)
- **+Z**: Superior
- **Origin**: Midpoint between LPA and RPA

Similar to Neuromag but historically associated with BESA Research software for EEG source analysis.

## EEG Electrode Placement Systems

### International 10-20 System

Electrodes placed at 10% and 20% intervals along standardized skull measurements:
- **Nasion to inion** (sagittal): Fpz, Fz, Cz, Pz, Oz
- **LPA to RPA** (coronal): T7, C3, Cz, C4, T8
- **Standard montage**: 21 electrodes

### 10-10 and 10-5 Extensions

| System | Electrodes | Spacing | Use |
|--------|-----------|---------|-----|
| 10-20 | 21 | ~6 cm | Clinical EEG |
| 10-10 | 81 | ~3 cm | Research EEG |
| 10-5 | 345 | ~1.5 cm | Dense-array EEG, source imaging |

Electrode names follow a systematic convention: letter = region (F=frontal, C=central, P=parietal, T=temporal, O=occipital), odd numbers = left, even numbers = right, z = midline.

### Digitized Electrode Coordinates

For source imaging, electrode positions must be digitized in 3D (e.g., Polhemus Fastrak, photogrammetry) and expressed in a common head coordinate system. The digitized positions are stored as (x, y, z) triplets in the head coordinate frame defined by fiducials.

## Fiducials and Coregistration

### Anatomical Fiducials

| Fiducial | Abbreviation | Location |
|----------|-------------|----------|
| Nasion | NAS / Nz | Bridge of the nose, between eyes |
| Left preauricular point | LPA | Anterior to left tragus |
| Right preauricular point | RPA | Anterior to right tragus |

These three points define the head coordinate system for MEG/EEG. Precise definitions vary across vendors: CTF defines preauricular points at the tragus; Neuromag uses the ear canal entrance.

### MRI-to-MEG Coregistration

The rigid-body transform (6 DOF: 3 rotation, 3 translation) aligning MRI and MEG/EEG coordinate systems is estimated from:

1. **Fiducial matching**: Identify NAS, LPA, RPA in both MRI and MEG digitization (minimum 3 points)
2. **Head shape refinement**: ICP (Iterative Closest Point) between digitized head points and the MRI scalp surface (typically 50-200 extra points)
3. **Verification**: Visual overlay of digitized points on the MRI scalp mesh; mean distance should be <3 mm

Coregistration error propagates directly into source localization accuracy. A 5 mm coregistration error can cause 5-10 mm source localization error.

## Coordinate Transforms

### Rigid Body (6 DOF)

Preserves distances and angles. Parameterized by 3 translations + 3 rotations:
```
T_rigid = [R  t]    R ∈ SO(3), t ∈ ℝ³
          [0  1]
```
Used for: MEG-MRI coregistration, within-session head movement correction.

### Affine (12 DOF)

Adds scaling (3 DOF) and shear (3 DOF) to the rigid transform:
```
T_affine = [A  t]    A ∈ GL(3), t ∈ ℝ³
           [0  1]
```
Used for: Linear MNI registration (FLIRT), FreeSurfer `talairach.xfm`.

### Nonlinear Warping

Voxel-wise displacement fields that capture local anatomical differences:
- **SPM**: Diffeomorphic (DARTEL, Geodesic Shooting); deformation stored as 3-volume NIfTI (`y_*.nii`)
- **FSL**: FNIRT; warp stored as coefficient or displacement field (`*_warp.nii.gz`)
- **ANTs**: SyN diffeomorphic registration; stored as displacement field + affine (`.mat` + `Warp.nii.gz`)
- **FreeSurfer**: Spherical registration for cortical surfaces (`?h.sphere.reg`)

libspm provides the core C routines (`spm/diffeo.h`) for diffeomorphic composition, Jacobian computation, and push/pull warping used in SPM's nonlinear registration.

## SPM vs FSL Conventions

| Aspect | SPM | FSL |
|--------|-----|-----|
| **Internal coordinates** | mm, world (RAS-like) | mm, voxel-scaled |
| **Affine convention** | Uses sform (NIfTI header) | Stores separate `.mat` for FLIRT |
| **MNI template** | `TPM.nii` (ICBM152 tissue priors) | `MNI152_T1_1mm.nii.gz` |
| **Nonlinear registration** | DARTEL / Geodesic Shooting | FNIRT |
| **Deformation format** | 3-volume NIfTI (`y_*.nii`) | Spline coefficient or warp field |
| **Voxel indexing** | 1-based (MATLAB) | 0-based (C/Python) |
| **Linear registration** | Stored in NIfTI header (sform update) | Separate `.mat` file (FLIRT) |

**Common pitfall**: FSL's FLIRT `.mat` files encode a transform between **voxel** spaces (scaled by voxel dimensions), not between world coordinates. Converting FLIRT matrices to world-space affines requires pre/post-multiplication by the source and target sform matrices:
```
T_world = sform_ref * T_flirt * inv(sform_src)
```

## Key References

- **Fischl2012freesurfer**: Fischl (2012). FreeSurfer. NeuroImage 62:774-781. Defines surface-based coordinate spaces (tkRAS, scanner RAS, MNI305 transformations).
- **Fischl1999cortical**: Fischl et al. (1999). Cortical surface-based analysis II: Inflation, flattening, and a surface-based coordinate system. NeuroImage 9:195-207.
- **Dale1999cortical**: Dale et al. (1999). Cortical surface-based analysis I: Segmentation and surface reconstruction. NeuroImage 9:179-194.
- **Jenkinson2012fsl**: Jenkinson et al. (2012). FSL. NeuroImage 62:782-790. FLIRT/FNIRT registration tools defining FSL coordinate conventions.

## Relevant Projects

- **neurojax**: Manages CTF device-to-head and head-to-MNI transforms for source imaging; FreeSurfer surface coordinate handling; electrode digitization I/O via MNE-Python
- **libspm**: Core C routines for diffeomorphic registration (`spm/diffeo.h`), B-spline interpolation (`spm/bsplines.h`), and affine registration via joint histograms (`spm/histogram.h`)
- **vbjax**: Whole-brain simulation on cortical surface meshes registered to MNI via FreeSurfer; connectome parcellations in MNI space
- **LAYNII**: Layer-fMRI analysis in native voxel space with equi-volume layering; transforms between columnar and flat-map coordinates
- **sbi4dwi**: Diffusion MRI processing with gradient direction transforms between scanner and voxel frames; BIDS-compliant coordinate metadata
- **brain-fwi**: Acoustic simulations on head meshes derived from MRI segmentations; source/receiver positions in MNI or scanner coordinates

## See Also

- [data-formats.md](data-formats.md) — NIfTI affine storage, DICOM orientation, FreeSurfer surface formats
- [structural-mri.md](structural-mri.md) — Registration and preprocessing pipelines
- [eeg.md](eeg.md) — Electrode placement and EEG forward modeling
- [meg.md](meg.md) — MEG sensor coordinates and coregistration
- [sci-head-model.md](sci-head-model.md) — Tetrahedral head model geometry
- [method-fem.md](method-fem.md) — FEM solvers requiring mesh coordinate systems
- [method-source-imaging.md](method-source-imaging.md) — Inverse methods requiring coregistered source spaces
