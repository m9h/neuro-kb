---
type: concept
title: "Neuroimaging Data Formats"
related: [sci-head-model.md, eeg.md, meg.md, fmri.md, diffusion-mri.md, structural-mri.md]
---

# Neuroimaging Data Formats

Standard file formats for storing and exchanging neuroimaging data across different modalities and analysis pipelines.

## BIDS (Brain Imaging Data Structure)

**Purpose:** Standardized directory structure and metadata organization for neuroimaging datasets.

### Core Principles
- **Hierarchical organization:** `dataset/sub-XX/ses-YY/modality/`
- **JSON sidecars:** Metadata stored alongside data files
- **TSV event files:** Experimental paradigms and timing
- **Derivative tracking:** Processed data maintains provenance

### Modality-Specific Extensions
| Modality | Extension | Key Files |
|----------|-----------|-----------|
| MEG | `meg/` | `.ds` (CTF), `.fif` (Elekta), `_meg.json` |
| EEG | `eeg/` | `.eeg/.vhdr/.vmrk` (BrainVision), `_eeg.json` |
| DWI | `dwi/` | `.nii.gz`, `.bval`, `.bvec`, `_dwi.json` |
| fMRI | `func/` | `_bold.nii.gz`, `_events.tsv` |
| MRS | `mrs/` | `.nii.gz` (NIfTI-MRS), `_mrs.json` |
| Anatomical | `anat/` | `_T1w.nii.gz`, `_T2w.nii.gz` |

### Derivatives Structure
```
derivatives/
├── pipeline-name/
│   ├── dataset_description.json
│   └── sub-XX/
│       └── processed_data.nii.gz
```

## NIfTI (Neuroimaging Informatics Technology Initiative)

**Format:** `.nii` or `.nii.gz` (compressed)
**Header size:** 348 bytes (fixed)
**Data types:** int8, int16, int32, float32, float64
**Orientation:** Stored as 4×4 affine transformation matrix

### Key Header Fields
```c
short dim[8];           // [0, nx, ny, nz, nt, 1, 1, 1]
float pixdim[8];        // [0.0, dx, dy, dz, dt, 1.0, 1.0, 1.0]  
float srow_x[4];        // First row of affine matrix
float srow_y[4];        // Second row of affine matrix
float srow_z[4];        // Third row of affine matrix
short qform_code;       // Coordinate system (1=scanner, 2=aligned, 3=Talairach, 4=MNI)
char descrip[80];       // Free-form description
```

### Coordinate Systems
- **Scanner coordinates** (qform_code=1): Raw scanner space
- **Aligned coordinates** (qform_code=2): AC-PC aligned
- **MNI space** (qform_code=4): Montreal Neurological Institute template

## NIfTI-MRS Extension

**Purpose:** Store MR spectroscopy data with acquisition parameters
**Spec:** [RFC 32](https://github.com/wexeee/mrs_nifti_standard)

### Header Extension
```json
{
  "SpectrometerFrequency": 123.25,
  "ResonantNucleus": ["1H"],
  "EchoTime": 0.144,
  "RepetitionTime": 5.0,
  "ConversionMethod": "spec2nii v0.8.7",
  "OriginalFile": ["meas_MID00123_sLASER_TE144.dat"]
}
```

### Data Layout
- **Dimensions:** [frequency, averages, coils, measurements, other...]
- **Complex data:** Real and imaginary components interleaved
- **Units:** Arbitrary (typically µV or relative to water)

## DICOM Integration

### Key DICOM Tags for Neuroimaging
| Tag | Description | BIDS Mapping |
|-----|-------------|--------------|
| (0018,0087) | MagneticFieldStrength | MagneticFieldStrength |
| (0018,0024) | SequenceName | PulseSequenceDetails |
| (0018,0081) | EchoTime | EchoTime |
| (0018,0080) | RepetitionTime | RepetitionTime |
| (0018,1314) | FlipAngle | FlipAngle |
| (0008,103E) | SeriesDescription | SeriesDescription |

### DICOM to BIDS Conversion
**Tools:** dcm2niix, dcm2bids, spec2nii (for MRS)
**Process:**
1. Extract DICOM headers
2. Map to BIDS-compliant filenames
3. Generate JSON sidecars
4. Organize directory structure

## CTF MEG Format (.ds)

**Structure:** Directory containing acquisition files
```
dataset.ds/
├── res4         # Resource file (sensor positions, system info)
├── meg4         # Raw MEG data (time series)
├── MarkerFile   # Event markers and timing
├── params       # Acquisition parameters
└── processing/  # Head shape, fiducials
```

### Key Parameters
- **Sampling rate:** Typically 600-2400 Hz
- **Channels:** 275 axial gradiometers (CTF-275)
- **Units:** Tesla (T) for magnetic field measurements
- **Coordinate system:** CTF device coordinates (nose at +X)

## FreeSurfer Surface Format

### Surface Files (.surf)
- **Vertices:** 3D coordinates in surface RAS space
- **Faces:** Triangular connectivity (0-indexed)
- **Binary format:** Big-endian, no header

### Annotation Files (.annot)
- **Parcellation:** Vertex-wise region labels
- **Color table:** RGB values for each region
- **Standard atlases:** Desikan-Killiany, Destrieux, DKT

### Statistics Files (.mgh/.mgz)
- **Values:** Scalar data per vertex (thickness, curvature, activation)
- **Compressed:** .mgz uses gzip compression
- **Header:** 284 bytes containing dimensions and voxel size

## HDF5 Scientific Data

**Purpose:** Hierarchical storage for simulation libraries and processed datasets
**Advantages:** Cross-platform, self-describing, chunked/compressed storage

### Simulation Library Structure
```
/simulations.h5
├── /metadata
│   ├── n_simulations (scalar)
│   ├── parameter_names (string array)
│   └── acquisition_scheme (structured)
├── /parameters [N × P]
└── /signals [N × G]
```

### Chunking Strategy
- **Parameters:** Chunk by simulation (row-wise)
- **Signals:** Chunk by gradient direction (column-wise)
- **Compression:** gzip level 6 (balance speed/size)

## Cross-Platform Considerations

### Endianness
- **NIfTI:** Little-endian (Intel byte order)
- **FreeSurfer:** Big-endian (historical SGI format)
- **CTF MEG:** Mixed (res4: big-endian, meg4: native)

### File Size Limits
- **NIfTI:** 2³¹-1 bytes (2 GB) for classic format
- **NIfTI-2:** 2⁶³-1 bytes (8 exabytes) for large datasets
- **HDF5:** Platform-dependent (typically 2⁶³-1 bytes)

## Relevant Projects

### neurojax
- **BIDS I/O:** Native BIDS dataset loading via mne-bids
- **NIfTI processing:** Volume-based source imaging and head modeling
- **Format conversion:** MEG (.ds) to MNE-Python Raw objects
- **Surface mapping:** FreeSurfer integration for cortical visualization

### sbi4dwi
- **DWI data:** BIDS-compliant diffusion MRI loading with gradient table validation
- **HDF5 libraries:** Simulation-based inference training data storage
- **Multi-format support:** DICOM, PAR/REC, Bruker format readers
- **Oracle integration:** External simulator results in standardized HDF5 format

### dot-jax
- **NIfTI volumes:** Optical property mapping for diffuse optical tomography
- **Mesh formats:** Triangle soup (.obj), tetrahedral (.msh) for FEM simulation
- **SNIRF compliance:** Shared Near Infrared Spectroscopy Format for fNIRS data

## Key References

- **Esteban2019fmriprep**: Esteban et al. (2019). fMRIPrep: a robust preprocessing pipeline for functional MRI. Nature Methods 16:111-116. BIDS-native fMRI preprocessing.
- **Luke2021bids**: Luke et al. (2021). fNIRS-BIDS: the brain imaging data structure extended to functional near-infrared spectroscopy.
- **SNIRF2023spec**: fNIRS community (2023). Shared Near Infrared Spectroscopy Format (SNIRF). Standard format for fNIRS data exchange.
- **Fischl2012freesurfer**: Fischl (2012). FreeSurfer. NeuroImage 62:774-781. Defines surface and annotation file formats.
- **Clarke2021fslmrs**: Clarke et al. (2021). FSL-MRS: An end-to-end spectroscopy analysis package. MRM 85:2950-2964. NIfTI-MRS format support.

## See Also

- [sci-head-model.md](sci-head-model.md) — SCI anatomical head model
- [eeg.md](eeg.md) — Electroencephalography
- [meg.md](meg.md) — Magnetoencephalography
- [fmri.md](fmri.md) — Functional MRI
- [diffusion-mri.md](diffusion-mri.md) — Diffusion MRI
- [structural-mri.md](structural-mri.md) — Structural MRI