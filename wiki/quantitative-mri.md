---
type: modality
title: Quantitative MRI
physics: electromagnetic
measurement: tissue-specific relaxation times, proton density, magnetization transfer
spatial_resolution: 1-2 mm isotropic
temporal_resolution: 5-30 min per map
related: [structural-mri.md, physics-bloch.md, tissue-gray-matter.md, tissue-white-matter.md, tissue-csf.md, diffusion-mri.md, method-sbi.md]
---

# Quantitative MRI

Quantitative MRI (qMRI) produces maps of physical tissue properties---relaxation times (T1, T2, T2*), proton density (PD), and magnetization transfer (MT) parameters---rather than the contrast-weighted images of conventional structural MRI. Each voxel value has defined physical units, enabling direct cross-subject and cross-site comparison and providing inputs to biophysical models such as electromagnetic forward solvers and microstructure estimation pipelines.

## T1 Mapping

T1 (longitudinal relaxation time) reflects the rate at which magnetization recovers along the main field axis after RF excitation. It is sensitive to macromolecular content, water content, and paramagnetic ion concentration.

### Acquisition Methods

- **Inversion Recovery (IR)**: Gold standard. Multiple TI values sampled after a 180-degree inversion pulse. Slow but accurate. Fit: S(TI) = S0 (1 - 2 exp(-TI/T1)).
- **MP2RAGE**: Two inversion-prepared GRE readouts at different TI values yield a bias-field-robust T1 map in a single scan (~8 min at 1 mm isotropic). Widely used at 7T (Marques et al. 2010).
- **Variable Flip Angle (VFA) / DESPOT1**: Two or more SPGR acquisitions at different flip angles. Fast (~2 min total) but requires accurate B1 mapping for correction. Fit: S/sin(a) = E1 S/tan(a) + M0(1 - E1), where E1 = exp(-TR/T1).
- **Look-Locker / MOLLI**: Continuous sampling during magnetization recovery. Standard for cardiac T1 mapping; adapted for brain at 7T.

### T1 as a Myelin Proxy

R1 = 1/T1 correlates with myelin content in cortex and white matter (Stueber et al. 2014). Higher myelin density shortens T1 by increasing macromolecular relaxation sinks. This relationship underpins the use of R1 maps for myeloarchitectonic parcellation of cortical areas (Glasser & Van Essen 2011).

## T2 and T2* Mapping

T2 (transverse relaxation) and T2* (effective transverse relaxation including B0 inhomogeneity) are sensitive to tissue water environment, iron content, and blood oxygenation.

### Acquisition Methods

- **Multi-Echo Spin Echo (MESE)**: Multiple echoes after a single 90-degree excitation. Provides pure T2 (refocuses B0 inhomogeneity). Typical: 32 echoes, TE spacing 10 ms.
- **GraSE (Gradient and Spin Echo)**: Hybrid combining spin echo refocusing with EPI readout for faster T2 mapping. Used in multi-component T2 relaxometry for myelin water imaging.
- **Multi-Echo GRE**: Multiple gradient echoes at increasing TE. Maps T2* and can extract susceptibility via QSM. Typical: 8 echoes, TE1 = 2 ms, dTE = 2 ms.

### Multi-Component T2 Relaxometry

Brain tissue contains water in distinct microenvironments with different T2 values:
- **Myelin water** (T2 ~ 10-40 ms): trapped between myelin bilayers
- **Intra/extracellular water** (T2 ~ 60-150 ms): main tissue water pool
- **CSF** (T2 > 1000 ms): free water

The myelin water fraction (MWF = myelin water signal / total signal) is estimated via multi-exponential fitting or regularized non-negative least squares (NNLS) of MESE data. Typical MWF values: WM 5-15%, GM 2-5% (MacKay et al. 2006).

## Proton Density Mapping

Proton density (PD or M0) maps reflect the concentration of MR-visible hydrogen nuclei per unit volume. PD is approximately proportional to water content, making it useful for tissue segmentation and as a normalisation factor for other quantitative maps.

- Estimated jointly with T1 from VFA data (intercept of the DESPOT1 linearisation)
- Alternatively from long-TR, short-TE spin echo (minimal T1 and T2 weighting)
- Requires B1 receive-field correction for accuracy

## Magnetization Transfer

Magnetization transfer (MT) imaging probes the exchange between free water protons and protons bound to macromolecules (primarily myelin lipids and proteins).

### MT Ratio (MTR)

Simple semi-quantitative metric:
```
MTR = (S0 - Smt) / S0 x 100%
```
where S0 is the signal without MT saturation and Smt is with off-resonance saturation. Typical values: WM 40-50%, GM 30-40%. Sensitive to demyelination but also affected by T1, B1, and pulse parameters.

### Quantitative MT (qMT)

Two-pool model separating free and bound water:
- **Parameters**: bound pool fraction (BPF or F), forward exchange rate (kf), free pool T2f, bound pool T2b
- **Acquisition**: multiple off-resonance saturation frequencies and powers
- **Fitting**: Henkelman model (Henkelman et al. 1993) or Ramani model
- **BPF** correlates strongly with histological myelin content (Schmierer et al. 2007)

BPF is used in NeuroJAX to derive tissue conductivity: sigma = f(T1, BPF) via empirical models (see [Tissue Property Estimation](#connection-to-tissue-property-estimation) below).

## Myelin-Sensitive Composite Metrics

### T1w/T2w Ratio

Ratio of T1-weighted to T2-weighted images enhances myelin contrast while cancelling receive-bias fields (Glasser & Van Essen 2011). No quantitative units, but robust proxy for myeloarchitecture. Used extensively in HCP cortical parcellation.

### Macromolecular Tissue Volume (MTV)

MTV = 1 - (PD_tissue / PD_csf), representing the fraction of voxel volume occupied by non-water macromolecules (Mezer et al. 2013). Derived from PD maps normalised by CSF signal. Correlates with myelin and membrane content.

## Relaxation Time Values for Major Brain Tissues

### At 3T

| Tissue | T1 (ms) | T2 (ms) | T2* (ms) | PD (a.u.) | Source |
|--------|---------|---------|----------|-----------|--------|
| Gray matter | 1331 +/- 13 | 110 +/- 2 | 66 +/- 2 | 0.82 | Stanisz 2005, Wansapura 1999 |
| White matter | 832 +/- 10 | 79.6 +/- 0.6 | 53 +/- 3 | 0.70 | Stanisz 2005, Wansapura 1999 |
| CSF | 4163 +/- 1155 | 2569 +/- 1183 | ~500 | 1.00 | Wansapura 1999 |
| Thalamus | 1126 +/- 63 | 76 +/- 5 | 42 +/- 3 | 0.77 | Stanisz 2005 |
| Putamen | 1261 +/- 68 | 80 +/- 6 | 31 +/- 4 | 0.79 | Stanisz 2005 |
| Caudate | 1291 +/- 55 | 85 +/- 5 | 45 +/- 3 | 0.80 | Stanisz 2005 |
| Cerebellum (GM) | 1168 +/- 40 | 103 +/- 4 | 59 +/- 4 | 0.82 | Stanisz 2005 |

### At 7T

| Tissue | T1 (ms) | T2 (ms) | T2* (ms) | Source |
|--------|---------|---------|----------|--------|
| Gray matter | 1939 +/- 149 | 55 +/- 4 | 33 +/- 3 | Rooney 2007, Peters 2007 |
| White matter | 1220 +/- 36 | 46 +/- 2 | 27 +/- 2 | Rooney 2007, Peters 2007 |
| CSF | ~4500 | ~1500 | ~300 | Wright 2008 |

**Field-strength dependence**: T1 increases with B0 (approximately T1 proportional to B0^0.3-0.4 for tissue). T2 decreases modestly. T2* decreases substantially due to increased susceptibility effects.

## Connection to Tissue Property Estimation

qMRI maps provide the bridge between anatomical imaging and biophysical forward models used in EEG/MEG source imaging.

### Conductivity from T1

Empirical models relate T1 to electrical conductivity (Michel et al. 2004, Fernandez-Corazza et al. 2018):

```
sigma = a / T1^b + c
```

where parameters (a, b, c) are calibrated against ex-vivo conductivity measurements. In NeuroJAX, the `sigma_from_qmri()` function implements a differentiable version:

```python
sigma = sigma_from_qmri(t1_values, bpf_values, labels, params)
```

This allows `jax.grad` to flow from the EEG/MEG data-fit loss back through the forward model, the FEM assembly, and the conductivity mapping into the qMRI-derived tissue parameters. See [method-fem.md](method-fem.md) and [tissue-electrical-properties.md](tissue-electrical-properties.md) for the downstream computation.

### Anisotropy from DTI

White matter conductivity is anisotropic. The conductivity tensor can be estimated from the diffusion tensor (Tuch et al. 2001):

```
sigma_tensor = (sigma_e / d_e) * D_tensor
```

where sigma_e and d_e are effective scalar conductivity and diffusivity. This connects qMRI T1 (scalar conductivity magnitude) with diffusion MRI (conductivity tensor orientation). See [diffusion-mri.md](diffusion-mri.md).

## Connection to Simulation-Based Inference

Multi-compartment relaxometry models---such as those used in myelin water imaging or qMT---have complex, often non-differentiable likelihood functions. SBI provides a natural framework for fitting these models:

1. **Forward model**: Bloch simulation with multi-pool exchange (see [physics-bloch.md](physics-bloch.md))
2. **Prior**: Physiologically plausible parameter ranges for T1, T2, BPF, exchange rates
3. **Neural posterior estimation**: Amortised inference over the parameter space

This parallels the SBI approach used in sbi4dwi for diffusion microstructure (see [method-sbi.md](method-sbi.md)), extending it to relaxometry and MT parameters.

### PINN/Neural ODE Relaxometry

NeuroJAX implements physics-informed neural networks (PINNs) and neural ODEs for relaxometry fitting, embedding the Bloch equations as soft constraints. This enables fitting multi-pool models (e.g., two-pool qMT, three-pool mcDESPOT) without closed-form signal equations, using Diffrax for ODE integration.

## BIDS Structure

```
sub-XX/
  ses-YY/
    anat/
      sub-XX_ses-YY_T1map.nii.gz          # T1 map (ms)
      sub-XX_ses-YY_T2map.nii.gz          # T2 map (ms)
      sub-XX_ses-YY_T2starmap.nii.gz      # T2* map (ms)
      sub-XX_ses-YY_PDmap.nii.gz          # Proton density map
      sub-XX_ses-YY_MTRmap.nii.gz         # MT ratio map (%)
      sub-XX_ses-YY_acq-VFA_flip-05_T1w.nii.gz   # VFA source images
      sub-XX_ses-YY_acq-VFA_flip-20_T1w.nii.gz
      sub-XX_ses-YY_acq-MTon_MTw.nii.gz   # MT-weighted source
      sub-XX_ses-YY_acq-MToff_MTw.nii.gz
```

See [data-formats.md](data-formats.md) for general BIDS conventions.

## Quality Control

- **B1 mapping**: Mandatory for VFA T1 and qMT. Bloch-Siegert, AFI, or double-angle methods. Residual B1 error of 5% produces ~10% T1 error in VFA.
- **B0 mapping**: Required for T2* accuracy. Dual-echo GRE field maps.
- **Repeatability**: Test-retest CoV for T1 at 3T is typically 2-5% in WM, 3-7% in GM.
- **Phantom validation**: NIST/ISMRM system phantom with known T1/T2 values (14 spheres, T1: 20-2000 ms, T2: 5-600 ms).

## Key References

- **Deoni2003despot**: Deoni et al. (2003). Rapid combined T1 and T2 mapping using gradient recalled acquisition in the steady state (DESPOT). MRM 49:515-526.
- **stanisz2005t1**: Stanisz et al. (2005). T1, T2 relaxation and magnetization transfer in tissue at 3T. MRM 54:507-512. doi:10.1002/mrm.20605
- **Sled2001qmt**: Sled & Pike (2001). Quantitative imaging of magnetization transfer exchange and relaxation properties in vivo using MRI. MRM 46:923-931.
- **Ramani2002qmt**: Ramani et al. (2002). Precise estimate of fundamental in-vivo MT parameters in human brain. MRI 20:721-731.
- **Stikov2015gratio**: Stikov et al. (2015). In vivo histology of the myelin g-ratio with magnetic resonance imaging. NeuroImage 118:397-405.
- **Wood2018quit**: Wood (2018). QUIT: QUantitative Imaging Tools. JOSS 3:656.
- **Glasser2011myelin**: Glasser & Van Essen (2011). Mapping human cortical areas in vivo based on myelin content. J Neurosci 31:11597-11616.

## Relevant Projects

- **neurojax**: qMRI module (`src/neurojax/qmri/`) implements DESPOT1/mcDESPOT, QMT BPF, multi-echo T2*, qBOLD OEF, B1 correction, and PINN/NODE relaxometry. The `sigma_from_qmri()` function maps T1 and BPF to tissue conductivity for differentiable head modeling.
- **sbi4dwi**: Simulation-based inference for microstructure; the SBI framework generalises to multi-compartment relaxometry models.
- **mrs-jax**: MR spectroscopy quantification shares the Bloch equation foundation and metabolite T1/T2 values for spectral fitting.

## See Also

- [structural-mri.md](structural-mri.md) -- Contrast-weighted anatomical imaging
- [physics-bloch.md](physics-bloch.md) -- Bloch equations governing MR signal formation
- [diffusion-mri.md](diffusion-mri.md) -- Diffusion tensor and microstructure imaging
- [method-sbi.md](method-sbi.md) -- Simulation-based inference for parameter estimation
- [method-fem.md](method-fem.md) -- Finite element forward modeling using qMRI-derived conductivity
- [tissue-electrical-properties.md](tissue-electrical-properties.md) -- Conductivity values across tissues
- [tissue-gray-matter.md](tissue-gray-matter.md) -- Gray matter composition and properties
- [tissue-white-matter.md](tissue-white-matter.md) -- White matter myelin and anisotropy
- [tissue-csf.md](tissue-csf.md) -- CSF properties and relaxation times
- [method-neural-ode.md](method-neural-ode.md) -- Neural ODEs for biophysical model fitting

## References

- Stanisz GJ, et al. T1, T2 relaxation and magnetization transfer in tissue at 3T. *Magn Reson Med* 2005;54:507-512.
- Rooney WD, et al. Magnetic field and tissue dependencies of human brain longitudinal 1H2O relaxation in vivo. *Magn Reson Med* 2007;57:308-318.
- Marques JP, et al. MP2RAGE, a self bias-field corrected sequence for improved segmentation and T1-mapping at high field. *NeuroImage* 2010;49:1271-1281.
- Stueber C, et al. Myelin and iron concentration in the human brain: a quantitative study of MRI contrast. *NeuroImage* 2014;93:95-106.
- Glasser MF, Van Essen DC. Mapping human cortical areas in vivo based on myelin content as revealed by T1- and T2-weighted MRI. *J Neurosci* 2011;31:11597-11616.
- Mezer A, et al. Quantifying the local tissue volume and composition in individual brains with MRI. *Nat Med* 2013;19:1667-1672.
- Henkelman RM, et al. Quantitative interpretation of magnetization transfer. *Magn Reson Med* 1993;29:759-766.
- Schmierer K, et al. Quantitative magnetization transfer imaging in postmortem multiple sclerosis brain. *J Magn Reson Imaging* 2007;26:41-51.
- MacKay AL, et al. Insights into brain microstructure from the T2 distribution. *Magn Reson Imaging* 2006;24:515-525.
- Tuch DS, et al. Conductivity tensor mapping of the human brain using diffusion tensor MRI. *Proc Natl Acad Sci* 2001;98:11697-11701.
- Fernandez-Corazza M, et al. Skull modeling effects in conductivity estimates using parametric electrical impedance tomography. *IEEE Trans Biomed Eng* 2018;65:1785-1797.
