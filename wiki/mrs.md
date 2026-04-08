---
type: modality
title: MR Spectroscopy
physics: electromagnetic
measurement: Chemical shifts and metabolite concentrations via nuclear magnetic resonance
spatial_resolution: 8 cm³ (single voxel) to 1 mL (whole-brain MRSI)
temporal_resolution: 2-20 minutes (static) to 100 ms (dynamic fMRS)
related: [electromagnetic-fields.md, tissue-properties.md, head-models.md, quantitative-mri.md]
---

# MR Spectroscopy

Magnetic Resonance Spectroscopy (MRS) is the only non-invasive technique that can measure neurochemical concentrations *in vivo* in the human brain. Unlike structural and functional MRI, which provide anatomical detail and hemodynamic responses, MRS directly quantifies the molecular substrates of brain function—neurotransmitters, energy metabolites, and markers of cellular integrity.

## Physical Principles

MRS exploits the same nuclear magnetic resonance phenomenon as MRI but focuses on chemical shift differences rather than spatial localization. Different molecular environments cause proton resonances to appear at distinct frequencies (chemical shifts measured in parts per million, ppm), creating a spectrum where each peak corresponds to a specific metabolite.

The fundamental equation governing signal intensity is:

$$S = N \cdot f(T_1, T_2, TE, TR) \cdot \sin(\alpha) \cdot e^{-TE/T_2} \cdot (1 - e^{-TR/T_1})$$

where N is the number of contributing nuclei, and the relaxation terms depend on tissue-specific T₁ and T₂ values.

## Key Metabolites and Clinical Significance

| Metabolite | Chemical Shift | Concentration | Clinical Significance |
|------------|----------------|---------------|---------------------|
| **NAA** (N-acetylaspartate) | 2.01 ppm | 10-12 mM | Neuronal integrity marker |
| **Creatine** | 3.03 ppm | 8-10 mM | Energy metabolism (ATP/PCr) |
| **Choline** | 3.22 ppm | 1-2 mM | Cell membrane turnover |
| **GABA** | 3.01 ppm | 1.0-1.5 mM | Primary inhibitory neurotransmitter |
| **Glutamate** | 2.35 ppm | 8-12 mM | Primary excitatory neurotransmitter |
| **Glutathione** | 2.95 ppm | 0.5-1.5 mM | Antioxidant, oxidative stress |
| **Myo-inositol** | 3.56 ppm | 5-8 mM | Glial marker, osmolyte |
| **Lactate** | 1.33 ppm | <0.5 mM | Anaerobic metabolism |

## MRS Techniques

### Single Voxel Spectroscopy (SVS)
- **PRESS** (Point REolved SpectroScopy): Standard clinical sequence, TE 30-144 ms
- **STEAM** (STimulated Echo Acquisition Mode): Shorter minimum TE, useful for coupled spins
- **sLASER** (semi-Localized by Adiabatic SElective Refocusing): Improved localization, reduced chemical shift displacement

### Spectral Editing
Required to resolve overlapping peaks and measure low-concentration metabolites:

- **MEGA-PRESS**: J-difference editing for GABA at 3.0 ppm, removing creatine contamination
- **HERMES**: Hadamard encoding for simultaneous GABA + glutathione in a single acquisition
- **HERCULES**: Extends HERMES to include NAA, aspartate, and other metabolites

### Whole-Brain Spectroscopic Imaging
- **EPSI** (Echo-Planar Spectroscopic Imaging): Fast spatial-spectral encoding
- **FID-MRSI**: Short TE for improved metabolite visibility
- **MIDAS**: Maudsley's 118,000+ voxel whole-brain system

## Quantification Methods

### Water-Referenced Quantification
The gold standard approach uses tissue water as an internal concentration reference:

$$[Metabolite] = \frac{A_{met}}{A_{water}} \cdot \frac{55,556 \text{ mM}}{n_H} \cdot f_{corr}$$

where:
- A_met, A_water are metabolite and water peak areas
- n_H is the number of contributing protons
- f_corr includes tissue fraction corrections (GM/WM/CSF), T₁/T₂ relaxation corrections

Tissue correction factors (Gasparovic et al. 2006):
- **GM**: 0.78 water fraction, T₁=1832ms, T₂=99ms (3T)
- **WM**: 0.65 water fraction, T₁=1220ms, T₂=69ms (3T)  
- **CSF**: 0.97 water fraction, T₁=4163ms, T₂=503ms (3T)

### Creatine-Referenced Ratios
Common in clinical practice but problematic in pathology where creatine may be altered:
- **GABA/NAA**: Typical values 0.059 ± 0.006 (3T, anterior cingulate)
- **Cho/NAA**: Elevated in tumors (>0.6 suggests high-grade glioma)
- **NAA/Cr**: Reduced in neurodegeneration

## Processing Pipeline

### 1. Preprocessing
- **Coil combination**: SVD-based optimal weighting
- **Eddy current correction**: Klose method using water reference
- **Frequency referencing**: Peak-based alignment to NAA (2.01 ppm) or creatine (3.03 ppm)
- **Apodization**: Exponential (Lorentzian) or Gaussian line broadening

### 2. Edited Sequence Processing
- **Spectral alignment**: Near et al. (2015) frequency/phase correction
- **Outlier rejection**: MAD-based removal of corrupted dynamics
- **Subtraction**: ON - OFF editing conditions
- **Phase correction**: Zero-order and first-order optimization

### 3. Quantitative Fitting
- **Linear combination modeling**: LCModel, TARQUIN, Osprey
- **Gaussian fitting**: Simple peaks (GABA in edited spectra)
- **Basis set simulation**: GAMMA, VeSPA for accurate lineshapes
- **Quality metrics**: Cramér-Rao Lower Bounds (CRLB < 20% acceptable)

## Clinical Applications

### Oncology (Ross & Nelson lineage)
- **Choline elevation**: Membrane turnover in proliferating tumors
- **NAA reduction**: Loss of neuronal integrity
- **Lactate appearance**: Anaerobic glycolysis (Warburg effect)
- **Combined choline/NAA ratio**: Tumor grading and treatment monitoring

### Neurodegeneration
- **NAA decline**: Early marker in Alzheimer's, multiple sclerosis, TBI
- **Glutamate dysregulation**: ALS, Huntington's disease
- **Oxidative stress**: Glutathione depletion in aging and disease

### Psychiatry and Neurotransmission
- **GABA deficits**: Depression, anxiety, autism spectrum disorders
- **Glutamate excess**: Bipolar disorder, schizophrenia
- **Metabolic alterations**: Energy metabolism in major depression

## Technical Challenges and Solutions

### Spectral Overlap
Many brain metabolites have overlapping resonances requiring advanced separation:
- **J-coupling**: Scalar coupling creates multiplet patterns
- **T₂ decay**: Shorter T₂ metabolites disappear at long TE
- **Editing pulses**: Frequency-selective manipulation of coupled spins

### Motion and Drift
Long acquisition times (5-20 minutes) are susceptible to:
- **B₀ field drift**: Frequency shifts over time
- **Patient movement**: Voxel displacement, shimming degradation
- **Scanner heating**: Gradual frequency/phase changes

### Quantification Variability
Sources of measurement uncertainty:
- **Tissue segmentation errors**: GM/WM/CSF fraction estimation
- **Relaxation parameter assumptions**: Literature vs. individual values
- **Water suppression artifacts**: Incomplete nulling affects baseline

## Validation Datasets

### Big GABA (Mikkelsen et al. 2017)
Multi-site MEGA-PRESS validation across 24 research centers:
- **Sample**: 272 healthy adults
- **Protocol**: 3T Siemens/Philips/GE, TE=68ms, 14×14×30mm³ voxel
- **Key result**: GABA/NAA = 0.12 ± 0.03, 8.8% coefficient of variation

### ISMRM MRS Fitting Challenge (Marjańska et al. 2022)
Standardized synthetic spectra for algorithm validation:
- **28 synthetic datasets**: Known ground truth concentrations
- **Multiple pathologies**: Tumor, stroke, normal aging
- **Consensus analysis**: 19 research groups, various fitting algorithms

## Recent Advances

### Hyperpolarized ¹³C MRS (Vigneron group, UCSF)
Dynamic nuclear polarization increases ¹³C signal >10,000-fold:
- **Real-time metabolism**: Pyruvate → lactate conversion in tumors
- **Clinical translation**: FDA-approved trials in prostate cancer
- **Metabolic flux**: Direct measurement of glycolysis vs. oxidative metabolism

### Ultra-High Field (7T and beyond)
Increased spectral resolution and sensitivity:
- **Enhanced separation**: Better resolved multiplets, more metabolites
- **Reduced voxel sizes**: Sub-cubic centimeter localization
- **New biomarkers**: Glutathione, NAAG, 2-HG in brain tumors

### AI-Enhanced Processing
Machine learning for improved quantification:
- **BrainSpec**: FDA-cleared automated analysis platform (Lin group)
- **Deep learning fitting**: Neural networks replace traditional optimization
- **Artifact detection**: Automated quality control and outlier identification

## Implementation in mrs-jax

The mrs-jax project provides complete MEGA-PRESS and HERMES processing pipelines:

- **JAX backend**: GPU-accelerated batch processing with `jit`, `vmap`, `grad`
- **Validated algorithms**: Near et al. spectral registration, Gasparovic quantification
- **Quality control**: Automated HTML reports with inline plots
- **Benchmark compliance**: Tested against Big GABA and ISMRM datasets

Key functions:
```python
# MEGA-PRESS pipeline
result = quantify_mega_press(data, dwell_time, centre_freq, 
                           water_ref, tissue_fracs, te, tr)

# HERMES processing  
hermes_result = process_hermes(data, dwell_time, centre_freq)
```

## Relevant Projects

| Project | Domain | MRS Connection |
|---------|--------|---------------|
| `mrs-jax` | MR Spectroscopy | Complete MEGA-PRESS/HERMES pipeline |
| `neurojax` | Multi-modal neuroimaging | MRS integration with MEG/EEG/qMRI |
| `vbjax` | Whole-brain simulation | Metabolite-informed neural mass models |
| `qcccm` | Quantum cognition | NMR relaxometry for neural dynamics |

## See Also

- [electromagnetic-fields.md](electromagnetic-fields.md) - Physical basis of NMR
- [tissue-properties.md](tissue-properties.md) - T₁, T₂ relaxation parameters
- [head-models.md](head-models.md) - Anatomical localization for SVS
- [quantitative-mri.md](quantitative-mri.md) - Complementary tissue characterization

## References

- Gasparovic C et al. (2006) Use of tissue water as a concentration reference for proton spectroscopic imaging. *MRM* 55:1219-1226
- Mikkelsen M et al. (2017) Big GABA: Edited MR spectroscopy at 24 research sites. *NeuroImage* 159:32-45
- Near J et al. (2015) Frequency and phase drift correction by spectral registration in the time domain. *MRM* 73:44-50
- Chan KL et al. (2016) HERMES: Hadamard encoding and reconstruction of MEGA-edited spectroscopy. *MRM* 76:11-19
- Ross BD, Lin AP (2005) Efficacy of proton MRS in neurological diagnosis. *Neurotherapeutics* 2:197-214
- Marjańska M et al. (2022) Results of a fitting challenge for MR spectroscopy. *MRM* 87:2198-2211