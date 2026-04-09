```yaml
type: modality
title: Functional Near-Infrared Spectroscopy
physics: optical
measurement: hemoglobin concentration changes via near-infrared light attenuation
spatial_resolution: 10-30 mm (standard fNIRS), 10-15 mm (HD-DOT)
temporal_resolution: 0.1-10 Hz
related: [diffuse-optical-tomography.md, hemodynamics.md, photon-transport.md, beer-lambert-law.md]
```

# Functional Near-Infrared Spectroscopy

**Functional near-infrared spectroscopy (fNIRS)** is a non-invasive optical neuroimaging modality that measures brain activity by detecting changes in hemoglobin concentration through near-infrared light attenuation. fNIRS exploits the relative transparency of biological tissue to light in the 650-950 nm spectral window, where scattering dominates over absorption.

## Physical Principles

### Near-Infrared Optical Window

The "biological window" spans approximately 650-950 nm wavelengths, where:
- **Water absorption** is minimal (< 0.01 /cm)
- **Hemoglobin absorption** varies significantly between oxygenated (HbO₂) and deoxygenated (Hb) states
- **Scattering** by cellular structures enables photon penetration of 1-3 cm into tissue

Light attenuation follows the **modified Beer-Lambert law**:

```
A = log(I₀/I) = ε·c·L·DPF + G
```

Where A is optical density, ε is extinction coefficient, c is chromophore concentration, L is source-detector separation, DPF is differential pathlength factor (~6-7 for adult head), and G accounts for scattering losses.

### Chromophore Spectroscopy

Key absorbers in brain tissue at fNIRS wavelengths:

| Chromophore | 690 nm (mM⁻¹cm⁻¹) | 830 nm (mM⁻¹cm⁻¹) | Notes |
|-------------|-------------------|-------------------|--------|
| HbO₂ | 0.64 | 2.24 | Decreases with wavelength |
| Hb | 5.38 | 1.60 | Higher at 690 nm |
| Water | 0.544 | 3.38 | Minimal in NIR window |
| **Isosbestic point** | **~800 nm** | **1.75** | **HbO₂ = Hb extinction** |

Source: Oregon Medical Laser Center (Prahl, 1999)

The **isosbestic point** near 800 nm, where oxy- and deoxyhemoglobin have equal extinction coefficients, serves as an important calibration reference for fNIRS systems.

## Instrumentation

### Source-Detector Configurations

| Configuration | Separation | Sensitivity | Spatial Resolution |
|---------------|------------|-------------|-------------------|
| **Short-separation** | 8-15 mm | Scalp/skull | ~5 mm |
| **Standard fNIRS** | 20-40 mm | Brain cortex | 20-30 mm |
| **High-density DOT** | Dense arrays | Tomographic | 10-15 mm |

Short-separation channels measure superficial hemodynamic changes, enabling systemic artifact removal through regression (Brigadoi & Cooper, 2015).

### Multi-Wavelength Systems

Most fNIRS systems use **dual-wavelength** measurements (typically 690 nm and 830 nm) to separate HbO₂ and Hb concentration changes. Advanced systems may employ:
- **Multi-spectral**: 3-8 wavelengths for improved chromophore separation
- **Broadband**: White light sources with spectrometers
- **Time-domain**: Pulsed sources measuring photon time-of-flight

## Signal Processing

### Hemodynamic Response Function

The canonical fNIRS response to neural activation:
- **Onset delay**: 2-3 seconds after stimulus
- **Peak latency**: 6-8 seconds  
- **Duration**: 15-20 seconds total
- **HbO₂ increase**: 0.3-0.8 μM typical
- **Hb decrease**: 0.1-0.3 μM (neurovascular coupling)

### Motion Artifact Correction

fNIRS is sensitive to head movement, requiring robust preprocessing:
- **Wavelet filtering**: Remove high-frequency motion spikes
- **Spline interpolation**: Correct discrete movement artifacts  
- **Principal component analysis**: Remove systematic artifacts
- **Short-separation regression**: Remove superficial contamination

## Diffuse Optical Tomography Integration

High-density fNIRS arrays enable **diffuse optical tomography (DOT)** reconstruction, providing improved spatial resolution and depth sensitivity. The forward model relates tissue absorption changes to surface measurements:

```
Δy = J·Δμₐ
```

Where Δy are detector measurements, J is the Jacobian (sensitivity matrix), and Δμₐ are absorption coefficient changes.

Systems like **Kernel Flow 2** (3000+ channels) and laboratory HD-DOT systems achieve ~10-15 mm spatial resolution through tomographic inversion.

## Clinical Applications

### Neurocritical Care
- **Cerebral oxygenation monitoring**: Continuous bedside assessment
- **Stroke detection**: Asymmetric hemodynamic responses
- **Traumatic brain injury**: Non-invasive intracranial pressure correlation

### Developmental Neuroscience
- **Infant brain imaging**: Motion-tolerant alternative to fMRI
- **Language development**: Cortical activation in pre-verbal children
- **Autism spectrum disorders**: Social cognition studies

### Brain-Computer Interfaces
- **Motor imagery**: Classification for prosthetic control
- **Cognitive load assessment**: Real-time mental state monitoring
- **Neurofeedback**: Closed-loop brain training protocols

### Comparison with fMRI

| Metric | fNIRS | fMRI |
|--------|-------|------|
| **Spatial resolution** | 10-30 mm | 2-4 mm |
| **Temporal resolution** | 0.1-10 Hz | 0.5-2 Hz |
| **Depth penetration** | 15-30 mm | Whole brain |
| **Motion tolerance** | High | Low |
| **Portability** | High | Fixed |
| **Cost** | $50K-500K | $1M-3M |

## Properties/Parameters

### Typical Tissue Optical Properties (800 nm)

| Tissue | μₐ (mm⁻¹) | μₛ' (mm⁻¹) | Notes |
|--------|-----------|-----------|--------|
| Scalp | 0.019 | 0.86 | Similar to muscle |
| Skull | 0.019 | 0.86 | Highly scattering |
| CSF | 0.0004 | 0.001 | Nearly transparent |
| Gray matter | 0.020 | 0.99 | Primary fNIRS target |
| White matter | 0.080 | 4.50 | High scattering |

Source: MCX Colin27 benchmark values

### Hemodynamic Parameters

| Parameter | Baseline | Activation Change | Units |
|-----------|----------|------------------|-------|
| **Total hemoglobin** | 50-80 | +5-15 | μM |
| **Oxygen saturation** | 60-70% | +5-15% | % |
| **Cerebral blood flow** | 50 | +20-50% | ml/100g/min |
| **Blood volume** | 3-4 | +10-30% | ml/100g |

## Key References

- **Jobsis1977noninvasive**: Jobsis (1977). Noninvasive, infrared monitoring of cerebral and myocardial oxygen sufficiency and circulatory parameters. Science 198:1264-1267.
- **Scholkmann2014review**: Scholkmann et al. (2014). A review on continuous wave functional near-infrared spectroscopy and imaging instrumentation and methodology. NeuroImage 85:6-27.
- **Brigadoi2015short**: Brigadoi & Cooper (2015). How short is short? Optimum source-detector distance for short-separation channels in fNIRS. Neurophotonics 2(2):025005.
- **Eggebrecht2014mapping**: Eggebrecht et al. (2014). Mapping distributed brain function and networks with diffuse optical tomography. Nature Photonics 8:448-454.
- **Cope1988system**: Cope & Delpy (1988). System for long-term measurement of cerebral blood and tissue oxygenation on newborn infants by near infra-red transillumination.
- **Prahl1999omlc**: Prahl (1999). Optical absorption of hemoglobin. Oregon Medical Laser Center spectra database.
- **Luke2021bids**: Luke et al. (2021). fNIRS-BIDS: the brain imaging data structure extended to functional near-infrared spectroscopy.
- **Sherafati2025hddot**: Sherafati et al. (2025). A high-density diffuse optical tomography dataset of naturalistic viewing. Scientific Data 12:1762.

## Relevant Projects

- **dot-jax**: Differentiable DOT forward modeling and image reconstruction pipeline
- **sbi4dwi**: Tissue property estimation through simulation-based inference
- **vbjax**: Neural mass model integration for neurovascular coupling
- **neuro-nav**: RL-based spatial cognition with fNIRS validation
- **neurotech-primer-book**: Educational content on fNIRS physics and safety
- **RatInABox**: Synthetic neural data generation for fNIRS algorithm validation

## See Also

- [diffuse-optical-tomography.md](diffuse-optical-tomography.md) — Tomographic reconstruction methods
- [hemodynamics.md](hemodynamics.md) — Neurovascular coupling mechanisms  
- [photon-transport.md](photon-transport.md) — Light propagation in tissue
- [beer-lambert-law.md](beer-lambert-law.md) — Spectroscopic foundations
- [head-models.md](head-models.md) — Anatomical models for forward simulation
- [eeg.md](eeg.md) — Complementary electrophysiological measurements
- [skull-tissue.md](skull-tissue.md) — Optical properties of skull tissue