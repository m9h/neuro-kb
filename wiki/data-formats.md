---
type: concept
title: "Neuroimaging Data Formats"
related: [coordinate-systems.md, structural-mri.md, diffusion-mri.md, fmri.md, meg.md, eeg.md, mrs.md, fnirs.md, quantitative-mri.md, connectomics.md, method-sbi.md, method-hmm-dynamics.md, method-fem.md, method-bem.md, sci-head-model.md, head-model-mida.md, jax-ecosystem.md, compute-infrastructure.md]
description: "This page covers the file formats and directory conventions used to store and exchange neuroimaging data across the project family."
timestamp: 2026-11-05T15:59:53-08:00
tags: [concept, data, bids, formats]
---

# Neuroimaging Data Formats

This page covers the file formats and directory conventions used to store and exchange neuroimaging data across the project family. Each format has fixed header layouts, axis conventions, and byte orders. Mismatches between these silently corrupt results; the classic examples are flipped b-vectors and swapped endianness. The reference dataset throughout is **WAND**, the 170-subject multi-modal CUBRIC dataset described in `neurojax:README.md`, which combines CTF MEG, 300 mT/m Connectom DWI, 7T and 3T MRS, qMRI, and TMS-EEG.

## Format Summary

| Format | Extension | Header | Byte order | Key limits / conventions | Used for |
|---|---|---|---|---|---|
| NIfTI-1 | `.nii`, `.nii.gz` | 348 B (+4 B extension flag; `vox_offset` ≥ 352) | either; detected from `sizeof_hdr` | `dim[]` is `int16`, so at most 32,767 voxels per axis | volumes, 4D DWI/BOLD, FSL outputs |
| NIfTI-2 | `.nii`, `.nii.gz` | 540 B (`vox_offset` ≥ 544) | either | `dim[]` is `int64` | large volumes, CIFTI containers |
| NIfTI-MRS | `.nii.gz` | NIfTI-2 + JSON header extension | either | dim 4 = spectral (complex FID) | single-voxel and MRSI spectroscopy |
| FreeSurfer MGH/MGZ | `.mgh`, `.mgz` | 284 B fixed | big-endian | `.mgz` = gzipped MGH | `aparc+aseg.mgz`, surface overlays |
| FreeSurfer surface | `lh.white`, `lh.pial` | 3-byte magic `0xFFFFFE` + "created by" line | big-endian | 0-indexed triangles, tkr-RAS coordinates | cortical meshes |
| FreeSurfer annotation | `.annot` | vertex/label pairs + colour table | big-endian | Desikan (68 cortical ROIs), Destrieux (148) | parcellations |
| CTF MEG | `.ds/` directory | `.res4` resource file | big-endian | 275 axial gradiometers (CTF-275) | WAND ses-01 MEG |
| bval / bvec | `.bval`, `.bvec` | plain text | — | bval: 1×N (s/mm²); bvec: 3×N in image-axis (FSL) convention | DWI gradient tables |
| FLIRT affine | `.mat` | plain text 4×4 | — | maps scaled-mm voxel space, not world RAS | FSL registration |
| HDF5 | `.h5` | self-describing | either | chunked and compressed groups | SBI libraries, SNIRF |
| NumPy | `.npy` | ~128 B ASCII-padded header | stored in dtype | single array with shape and dtype | neurojax dynamics derivatives |
| Equinox checkpoint | `.eqx` | none (serialised pytree leaves) | — | needs a matching skeleton module to load | trained SBI posteriors |

## BIDS (Brain Imaging Data Structure)

BIDS fixes the hierarchy `dataset/sub-XX/ses-YY/<datatype>/`. Each data file has a JSON sidecar, and events are stored in TSV files. WAND distributes one session per modality:

| Session | Datatype | Content (WAND, sub-08033) |
|---|---|---|
| ses-01 | `meg/` | CTF `.ds` MEG |
| ses-02 | `dwi/`, `anat/` | AxCaliber (4 shells), CHARMED; T1w, VFA, QMT |
| ses-03 | `anat/`, `func/` | T1w, T2w; fMRI |
| ses-04 / ses-05 | `mrs/` | sLASER (4 VOIs, TE = 78 ms, 7T); MEGA-PRESS (TE = 68 ms, 3T) |
| ses-06 | `anat/` | MP2RAGE, multi-echo GRE |
| ses-08 | `tms/` | TMS-EEG SICI, `.mat` (non-BIDS) |

Sources: `neurojax:docs/BIDS_DERIVATIVES_STRUCTURE.md` and `neurojax:docs/CUBRIC_QUESTIONS.md`.

### Sidecar gaps

BIDS conversion can drop acquisition parameters that downstream models need. In WAND, the 8-volume SPGR sidecar lists only `FlipAngle=18`, which is presumably the last volume of the suspected [2–18]° series. The QMT sidecars report MT saturation angles of 332/628/333° but omit the MT pulse duration needed by `qi qmt`. The TRUST eTEs are also missing (`neurojax:docs/CUBRIC_QUESTIONS.md`).

These gaps matter most for [quantitative-mri.md](quantitative-mri.md) fitting. Always check 4D sidecars against the protocol PDF.

### Derivatives (BEP003)

Each pipeline writes its own `derivatives/<pipeline>/` tree containing a `dataset_description.json`. That file records `BIDSVersion` (1.9.0 for WAND), `PipelineDescription`, `GeneratedBy` (for example FSL 6.0.7.22), and `SourceDatasets`. The neurojax WAND layout defines 14 stages, A–N:

| Stage | Directory | Representative outputs | Format |
|---|---|---|---|
| A | `fsl-anat/` | `T1_fast_pve_{0,1,2}.nii.gz` (CSF/GM/WM) | NIfTI |
| B | `freesurfer/` | `aparc+aseg.mgz`, `lh.pial`, `lh.T1wT2w_ratio.mgz` | MGZ / surf |
| C | `fsl-dwi/` | `*_eddy.nii.gz`, `.eddy_rotated_bvecs`, `dti_V1.nii.gz` | NIfTI + text |
| D | `fsl-bedpostx/` | `merged_{f,th,ph}{1,2,3}samples.nii.gz`, `dyads1.nii.gz` | NIfTI (4D samples) |
| G | `fsl-reg/` | `*_to_T1w.mat`, `T1w_to_MNI_warp.nii.gz` | FLIRT text / FNIRT warp |
| H | `fsl-connectome/` | `sub-08033_atlas-desikan_desc-sift2_connectivity.csv` | CSV |
| I | `fsl-mrs/` | `concentrations.csv`, `tissue_fractions.json`, `spectra/data.nii.gz` | CSV / JSON / NIfTI-MRS |
| J | `microstructure/sbi4dwi/` | `axcaliber_diameter.nii.gz`, `sandi_fsoma.nii.gz` | NIfTI |
| K | `qmri/` | `T1map.nii.gz`, `QMT_bpf.nii.gz`, `R2star_map.nii.gz` | NIfTI |
| M | `neurojax-meg/` | `parcellated_desikan.npy` (T, 68), `state_covariances.npy` (K, C, C), `state_coherence.npy` (K, F, C, C) | NumPy |
| N | `neurojax-tms/` | `empirical_tep.npy` (n_sensors, n_samples), `fitted_params.json` | NumPy / JSON |

Native space is kept as long as possible, with ses-02 T1w as the anchor. Other sessions are aligned to it with FLIRT 6-DOF rigid transforms. See [coordinate-systems.md](coordinate-systems.md).

The `.npy` HMM/DyNeMo outputs ([method-hmm-dynamics.md](method-hmm-dynamics.md)) fill a gap: osl-dynamics cannot ingest BIDS directly and requires manual conversion to `.npy`/`.mat` (`neurojax:docs/DESIGN_PROPOSAL_BIDS_DYNAMICS.md`).

## NIfTI-1 / NIfTI-2

```c
short dim[8];      // [ndim, nx, ny, nz, nt, ...]  (int64 in NIfTI-2)
float pixdim[8];   // [qfac, dx, dy, dz, dt, ...]
short qform_code;  // 0 unknown, 1 scanner, 2 aligned, 3 Talairach, 4 MNI-152
short sform_code;  // same codes; sform takes precedence in most tools
float srow_x[4], srow_y[4], srow_z[4];  // sform affine (voxel -> world RAS mm)
```

The voxel-to-world mapping is a 4×4 affine expressed in RAS+ millimetres.

FSL tools additionally use a **scaled-voxel** convention. When the affine determinant is positive, the x-axis is flipped. This is why FLIRT `.mat` files cannot be applied as world-space affines without conversion (for example with `nibabel` or `wb_command -convert-affine`).

## Diffusion Gradient Tables

- **`.bval`** holds one row of N b-values in s/mm².
- **`.bvec`** holds 3 rows × N unit vectors in **image-axis** coordinates (the FSL convention, adopted by BIDS).
  - If the image is resliced or reoriented, the bvecs must be rotated.
  - eddy writes `*.eddy_rotated_bvecs` with motion-corrected directions; downstream fits must use these rather than the raw `.bvec`.

In WAND, the four AxCaliber acquisitions are merged into `axcaliber_combined.{nii.gz,bval,bvec}` before microstructure fitting. sbi4dwi's `SBIPredictor.predict_volume(dwi_img, bvals, bvecs, mask)` consumes exactly this triple (`sbi4dwi:README.md`). See [diffusion-mri.md](diffusion-mri.md).

## NIfTI-MRS

NIfTI-MRS stores complex time-domain FIDs in a NIfTI-2 container. Its dimensions are laid out as follows:

- **Dims 1–3** are spatial (1×1×1 for single-voxel spectroscopy).
- **Dim 4** is the spectral/time axis.
- **Dims 5–7** are tagged in the JSON extension, e.g. `DIM_COIL`, `DIM_DYN`, `DIM_EDIT`. MEGA-PRESS ON/OFF sub-spectra go on a `DIM_EDIT` axis.

Required header-extension keys include `SpectrometerFrequency` (MHz) and `ResonantNucleus`. The example below is illustrative, for a 3T MEGA-PRESS acquisition:

```json
{"SpectrometerFrequency": [123.2], "ResonantNucleus": ["1H"],
 "EchoTime": 0.068, "ConversionMethod": "spec2nii v0.8.7"}
```

**Known conversion bug.** WAND Siemens TWIX `.dat` files store `PatientName` as a float (e.g. `31408033.0`). The `nifti_mrs` validator in spec2nii 0.8.7 / FSL-MRS 2.4.12 rejects this before `--anon` is applied. The workaround is to monkey-patch `validate_hdr_ext` so it coerces the value to `str` (`neurojax:docs/FSL_MRS_UPSTREAM_ISSUES.md`).

Basis-set formats are not interchangeable: FSL-MRS `.BASIS`/JSON, LCModel `.basis`, and VESPA output each need conversion. See [mrs.md](mrs.md).

## CTF MEG (`.ds`)

A CTF dataset is a directory, not a single file:

```
sub-XX_task-YY_meg.ds/
├── *.res4        # resource: channel names, gains, sensor geometry, sampling rate
├── *.meg4        # raw samples, big-endian int32, trial-blocked
├── *.hc          # head-coil (fiducial) positions
├── *.infods      # acquisition info
├── MarkerFile.mrk
├── BadChannels
└── ClassFile.cls
```

The CTF-275 system has 275 axial gradiometers plus reference sensors used for synthetic 3rd-order gradiometry.

CTF head coordinates are defined by the fiducials:
- **+X** points toward the nasion.
- **+Y** points toward the left pre-auricular point.
- **+Z** points up.

neurojax reads `.ds` data through MNE-Python and registers it to ses-02 T1w via MNE coregistration. See [meg.md](meg.md) and [coordinate-systems.md](coordinate-systems.md).

## FreeSurfer Formats

- **Surfaces** (`lh.white`, `lh.pial`, `lh.sphere.reg`) start with a 3-byte magic number and a text "created by" line, followed by vertex and face counts and then big-endian float32 vertices and int32 triangles.
- **Surface coordinates** are in tkr-RAS. The `c_ras` offset from `orig.mgz` is needed to reach scanner RAS.
- **MGH/MGZ** files have a 284-byte header. They carry volumes such as `aparc+aseg.mgz` and `ThalamicNuclei.v13.T1.mgz`, and per-vertex overlays such as the T1w/T2w myelin map.
- **Use in neurojax:** these surfaces feed PI-GNN source imaging as cortical-mesh graphs (`mesh_to_graph`) and BEM surface extraction ([method-bem.md](method-bem.md)).

## Meshes and Head Models

Tetrahedral and voxel head models travel through `meshio`, which reads Gmsh `.msh`, VTK, and others. Rasterisation between the two representations is common. sbi4dwi's `biophysics/mesh_rasterizer.py` converts the [sci-head-model.md](sci-head-model.md) tetrahedral mesh to a 208×256×256 grid at 1 mm with 8 tissue labels. These labels are then mapped to acoustic properties via ITRUSST reference values [@aubry2022itrusst] for j-Wave simulation [@stanziola2023jwave].

The SimNIBS "Ernie" MNI152 head (182×238×282 at 1 mm, 10 tissues) is remapped to 6 acoustic classes (`sbi4dwi:docs/openlifu/README.md`). Optical forward models use mesh formats in the NIRFAST lineage [@Dehghani2009nirfast] and mesh-based Monte Carlo [@Fang2010mcx]. See [method-fem.md](method-fem.md) and [head-model-mida.md](head-model-mida.md).

## HDF5 Simulation Libraries and SNIRF

sbi4dwi follows the rule "build where differentiable, wrap where not". External oracles (DIPY, ReMiDi, MCMRSimulator.jl) meet the JAX pipeline at an **HDF5 boundary** (`sbi4dwi:CLAUDE.md`). A `SimulationLibrary` stores `(theta, signal)` pairs:

```
/library.h5
├── /metadata      parameter_names, acquisition (bvals, bvecs, δ, Δ), oracle id
├── /parameters    [N × P]   prior draws θ
└── /signals       [N × G]   noisy signals per gradient
```

`HybridLibraryGenerator` mixes analytical simulations (70%) with oracle libraries (30%). Trained posteriors are saved as Equinox `.eqx` checkpoints and then applied voxel-wise to NIfTI DWI. This is how the Ball+2Stick flow reaches a 2.8° median fibre-1 error [@manzanopatron2025sbi; @fick2019dmipy]. See [method-sbi.md](method-sbi.md).

**SNIRF**, the fNIRS exchange format, is also HDF5. It stores `/nirs/data1/dataTimeSeries`, `/nirs/probe` optode geometry, and `measurementList` source–detector–wavelength indices. It is the natural input for high-density DOT [@Eggebrecht2014mapping]. See [fnirs.md](fnirs.md).

## Common Pitfalls

1. **Reorientation without bvec rotation.** Reorienting a DWI series without rotating its bvecs produces systematic fibre-orientation errors.
2. **FLIRT `.mat` applied as an RAS affine.** These matrices use FSL's scaled-voxel convention and must be converted first.
3. **4D sidecars with a single scalar.** A sidecar such as `FlipAngle=18` for an 8-volume SPGR describes only one volume.
4. **NIfTI-1 dimension limit.** NIfTI-1's 32,767 per-dimension limit is exceeded by long time series and dense surfaces; use NIfTI-2.
5. **Missing dimension tags in NIfTI-MRS.** Without them, coil combination and edit subtraction are ambiguous.

## Relevant Projects

- **neurojax**
  - `io/` contains the loaders: `WANDMEGLoader`, `BIDSConnectomeLoader`, Wakeman-Henson, and UCI.
  - I/O runs through MNE-Python (CTF, FIF, BrainVision), nibabel (NIfTI, MGZ, FreeSurfer surfaces), and meshio.
  - It owns the WAND BIDS-derivatives spec and the `.npy` dynamics outputs.
- **sbi4dwi**
  - `io/` contains loaders for BIDS, HCP, IXI, BigMac, WAND, multi-TE, mesh, and SWC data.
  - It provides the HDF5 `SimulationLibrary`, `.eqx` checkpoints, and NIfTI volume inference.
  - `mesh_rasterizer` handles mesh-to-grid conversion, and `tus_solution_export` exports OpenLIFU solutions.
- **dot-jax**
  - Works with optical meshes and the HDF5-based SNIRF probe/measurement format for DOT/fNIRS.
- **hippy-feat**
  - Consumes BIDS `func/` NIfTI BOLD series with `_events.tsv` for GLMsingle, AR(1) GLM, and real-time MindEye decoding. See [fmri.md](fmri.md).
- **mrs-jax**, **vbjax**
  - Downstream consumers of NIfTI-MRS spectra and connectome CSVs ([connectomics.md](connectomics.md)).

## See Also

- [coordinate-systems.md](coordinate-systems.md) — RAS, tkr-RAS, CTF head, MNI frames
- [structural-mri.md](structural-mri.md), [quantitative-mri.md](quantitative-mri.md) — anatomical and qMRI NIfTI outputs
- [diffusion-mri.md](diffusion-mri.md) — bval/bvec conventions, microstructure maps
- [meg.md](meg.md), [eeg.md](eeg.md) — CTF `.ds` and EEG formats
- [mrs.md](mrs.md) — NIfTI-MRS and basis sets
- [fnirs.md](fnirs.md) — SNIRF
- [fmri.md](fmri.md) — BOLD series and events
- [method-sbi.md](method-sbi.md) — HDF5 simulation libraries
- [method-hmm-dynamics.md](method-hmm-dynamics.md) — `.npy` state outputs
- [sci-head-model.md](sci-head-model.md), [head-model-mida.md](head-model-mida.md) — mesh and voxel head models
- [compute-infrastructure.md](compute-infrastructure.md), [jax-ecosystem.md](jax-ecosystem.md)
