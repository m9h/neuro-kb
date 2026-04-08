```yaml
type: modality
title: Functional MRI
physics: hemodynamic
measurement: Blood oxygen level-dependent (BOLD) contrast from hemodynamic changes
spatial_resolution: 1-3mm isotropic (typically 2-3mm)
temporal_resolution: 0.5-3.0s (TR dependent)
related: [bold-signal.md, hemodynamic-response.md, glm.md, connectivity.md, preprocessing.md]
```

# Functional MRI

Functional magnetic resonance imaging (fMRI) measures brain activity indirectly through blood oxygen level-dependent (BOLD) contrast, which reflects hemodynamic changes following neural activity. The BOLD signal arises from the paramagnetic properties of deoxyhemoglobin, creating T2* contrast changes that can be detected with gradient-echo sequences.

## Physical Principles

The BOLD signal relies on the neurovascular coupling between neural activity and local blood flow changes:

1. **Neural activation** increases local metabolic demand
2. **Vasodilation** increases cerebral blood flow (CBF) and blood volume (CBV)
3. **Oxygen delivery** exceeds consumption, reducing deoxyhemoglobin concentration
4. **T2* signal increase** due to reduced magnetic susceptibility effects

The hemodynamic response function (HRF) describes the temporal dynamics of this coupling, typically peaking 4-6 seconds post-stimulus with an undershoot at 10-15 seconds.

## Signal Properties

### Temporal Characteristics
- **Repetition time (TR)**: 0.5-3.0s, with 1.5-2.0s common for cognitive studies
- **HRF peak delay**: 4-6s after neural onset
- **HRF duration**: ~15-20s total response
- **Temporal SNR**: Typically 50-200 in gray matter at 3T

### Spatial Characteristics  
- **Voxel resolution**: 2-4mm isotropic typical, sub-millimeter possible at 7T
- **Spatial extent**: Point spread function ~3-5mm due to venous drainage
- **Gray matter CNR**: ~2-5% signal change for strong activation at 3T

### Signal-to-Noise Properties
- **Thermal noise**: Scales with √(bandwidth × voxel volume)
- **Physiological noise**: Cardiac (~1Hz), respiratory (~0.3Hz), scanner drift
- **CNR scaling**: ~Linear with field strength (3T → 7T gives ~2× CNR improvement)

## Acquisition Sequences

### Echo-Planar Imaging (EPI)
Most common fMRI sequence for rapid whole-brain coverage:
- **Single-shot EPI**: Entire slice in one excitation (~20-50ms)
- **Multiband EPI**: Simultaneous multi-slice with 2-8× acceleration
- **Spin-echo EPI**: Reduced susceptibility artifacts, used at high field

### Specialized Sequences
- **Arterial spin labeling (ASL)**: Direct CBF measurement via magnetic labeling
- **Calibrated BOLD**: Combined BOLD + CBF for quantitative CMRO₂
- **Layer fMRI**: High-resolution (0.7-1mm) for cortical depth analysis

## Preprocessing Pipeline

Standard fMRI preprocessing addresses systematic artifacts and noise:

1. **Slice timing correction**: Correct for within-TR acquisition delays
2. **Motion correction**: 6-parameter rigid-body realignment  
3. **Distortion correction**: EPI susceptibility artifact correction via fieldmaps
4. **Coregistration**: Functional → structural alignment
5. **Normalization**: Transform to standard space (MNI152)
6. **Spatial smoothing**: 4-8mm FWHM Gaussian kernel (traditional)

### Real-time Preprocessing
Modern pipelines achieve sub-TR processing for neurofeedback applications:
- **hippy-feat**: 54ms per volume (76×90×74) on GPU via JAX compilation
- **AFNI real-time**: ~100-200ms depending on spatial resolution
- **Turbo-BrainVoyager**: Sub-second preprocessing with GPU acceleration

## Statistical Analysis

### General Linear Model (GLM)
Standard approach models BOLD timeseries as:
```
Y(t) = Σᵢ βᵢ Xᵢ(t) + ε(t)
```
where βᵢ are regression coefficients, Xᵢ(t) are predicted regressors (task convolved with HRF), and ε(t) is noise.

### Connectivity Analysis
- **Functional connectivity**: Temporal correlation between regions
- **Effective connectivity**: Directed causal relationships (DCM, Granger)
- **Dynamic connectivity**: Time-varying connectivity via sliding windows or hidden Markov models

### Advanced Methods
- **Multivariate pattern analysis (MVPA)**: Decode mental states from spatial patterns
- **Independent component analysis (ICA)**: Blind source separation for artifact removal
- **Graph theory**: Network analysis of brain connectivity matrices

## Quantitative Parameters

### Typical BOLD Signal Changes
| Region | Task Type | Signal Change (%) | Baseline T2* (ms) |
|--------|-----------|-------------------|-------------------|
| Visual cortex | Checkerboard | 2-5% | 25-35 (3T) |
| Motor cortex | Finger tapping | 3-6% | 30-40 (3T) |
| Auditory cortex | Tones | 2-4% | 25-30 (3T) |
| Default network | Rest vs task | 1-3% | 25-35 (3T) |

### Field Strength Dependencies
| Field | CNR gain | Spatial resolution | Susceptibility artifacts |
|-------|----------|-------------------|-------------------------|
| 1.5T | 1× | 3-4mm typical | Minimal |
| 3T | ~2× | 2-3mm typical | Moderate |
| 7T | ~3-4× | 1-2mm achievable | Significant |

## Relevant Projects

- **neurojax**: 15 source imaging methods, differentiable head modeling, quantitative MRI integration
- **hippy-feat**: Real-time fMRI preprocessing (54ms/volume), differentiable connectivity analysis in JAX  
- **vbjax**: Whole-brain simulation with BOLD forward models from neural dynamics
- **neuro-nav**: Reinforcement learning models of spatial navigation using fMRI-derived place representations

## Limitations

- **Hemodynamic lag**: 4-6s delay limits temporal resolution for fast cognitive processes
- **Susceptibility artifacts**: Signal dropout near air-tissue boundaries (orbitofrontal, temporal)
- **Venous weighting**: BOLD reflects downstream venous changes, not local neural activity
- **Nonlinear neurovascular coupling**: HRF varies with stimulus duration, attention, aging
- **Global signal fluctuations**: Respiratory/cardiac artifacts require careful preprocessing

## See Also

- [bold-signal.md](bold-signal.md) - Detailed biophysics of BOLD contrast mechanism
- [hemodynamic-response.md](hemodynamic-response.md) - HRF modeling and neurovascular coupling
- [glm.md](glm.md) - Statistical analysis methods for activation detection
- [connectivity.md](connectivity.md) - Functional and effective connectivity analysis
- [preprocessing.md](preprocessing.md) - Artifact correction and data preparation methods