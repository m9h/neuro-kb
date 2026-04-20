# homebrew-neuro

Homebrew tap for neuroimaging, electrophysiology, and medical imaging tools not available in homebrew-core.

## Installation

```sh
brew tap m9h/neuro
brew install <formula>
```

Or install directly:

```sh
brew install m9h/neuro/<formula>
```

## Formulas

### Medical Image Registration & Segmentation

| Formula | Version | Description |
|---------|---------|-------------|
| [ants](https://github.com/ANTsX/ANTs) | 2.6.5 | Advanced Normalization Tools — image registration and segmentation |
| [elastix](https://elastix.dev) | 5.3.1 | Medical image registration toolbox |
| [niftyreg](https://github.com/KCL-BMEIS/niftyreg) | 2.0.0 | Medical image registration tools |
| [charm-gems](https://github.com/simnibs/charm-gems) | 1.3.3 | C++ segmentation library with Python bindings for SimNIBS CHARM |

### Medical Image Processing & Visualization

| Formula | Version | Description |
|---------|---------|-------------|
| [connectome-workbench](https://www.humanconnectome.org/software/connectome-workbench) | 2.1.0 | HCP visualization and analysis (wb_command, wb_view) |
| [c3d](http://www.itksnap.org/c3d) | 1.4.6 | Convert3D — medical image processing tool |
| [laynii](https://github.com/layerfMRI/LAYNII) | 2.10.0 | Layer fMRI analysis tools |
| [vmtk](http://www.vmtk.org) | 1.5.1 | Vascular Modeling Toolkit — vessel segmentation, centerlines, CFD preprocessing |

### MRI Reconstruction & Raw Data

| Formula | Version | Description |
|---------|---------|-------------|
| [bart](https://mrirecon.github.io/bart/) | 1.0.00 | Berkeley Advanced Reconstruction Toolbox for MRI |
| [ismrmrd](https://ismrmrd.github.io) | 1.15.0 | ISMRM Raw Data format — library and tools |
| [mrtrix3](https://www.mrtrix.org) | 3.0.8 | Diffusion MRI processing tools |

### Electrophysiology (EEG/MEG)

| Formula | Version | Description |
|---------|---------|-------------|
| [mne-cpp](https://mne-cpp.github.io) | 2.1.0 | Cross-platform C++ library for MEG and EEG data processing |
| [openmeeg](https://openmeeg.github.io) | 2.5.16 | BEM solver for forward problems in EEG and MEG |
| [sigviewer](https://github.com/cbrnr/sigviewer) | 0.6.5 | Biosignal viewer (EDF, GDF, BDF, XDF) |
| [brainflow](https://brainflow.org) | 5.21.0 | Library for obtaining EEG, EMG, ECG, and other biosignal data |

### Lab Streaming Layer (LSL)

| Formula | Version | Description |
|---------|---------|-------------|
| [liblsl](https://labstreaminglayer.org) | 1.17.5 | C/C++ library for multi-modal time-synced data streaming |
| [libxdf](https://github.com/xdf-modules/libxdf) | 0.99.10 | C++ library for reading XDF files |
| [labrecorder](https://github.com/labstreaminglayer/App-LabRecorder) | 1.17.1 | Record and write LSL streams to XDF files |

### Foundation Library

| Formula | Version | Description |
|---------|---------|-------------|
| [itk-neuro](https://itk.org) | 5.4.5 | ITK with neuroimaging remote modules (GenericLabelInterpolator, AdaptiveDenoising, MGHIO, DCMTK) |

## Notes

### itk-neuro vs homebrew-core itk

`itk-neuro` is a customized build of ITK 5.4.5 that includes remote modules required by neuroimaging tools (ANTs, FreeSurfer MGH I/O) and builds with `ITK_LEGACY_REMOVE=OFF` for downstream compatibility. It conflicts with the `itk` formula in homebrew-core — install one or the other.

Packages that depend on `itk-neuro`: ants, c3d, elastix.

### Brewfile

To install everything:

```ruby
tap "m9h/neuro"

# Foundation
brew "itk-neuro"

# Registration & segmentation
brew "ants"
brew "elastix"
brew "niftyreg"
brew "charm-gems"

# Image processing & visualization
brew "connectome-workbench"
brew "c3d"
brew "laynii"
brew "vmtk"

# MRI reconstruction
brew "bart"
brew "ismrmrd"
brew "mrtrix3"

# Electrophysiology
brew "mne-cpp"
brew "openmeeg"
brew "sigviewer"
brew "brainflow"

# Lab Streaming Layer
brew "liblsl"
brew "libxdf"
brew "labrecorder"
```

## License

Each formula installs software under its own license. See the individual project pages for details.
