---
type: modality
title: Transcranial Ultrasound
physics: acoustic
measurement: pressure waves, radiation force, tissue displacement
spatial_resolution: 0.5-2 mm
temporal_resolution: microseconds (pressure), milliseconds (ARFI displacement)
related: [acoustic-simulation.md, tissue-properties.md, j-wave.md, radiation-force.md, mr-arfi.md]
---

# Transcranial Ultrasound

Transcranial ultrasound (TUS) is an acoustic modality that uses focused ultrasound beams to non-invasively target brain tissue through the skull. The modality encompasses both therapeutic applications (thermal ablation, neuromodulation) and imaging (transcranial ultrasound computed tomography).

## Physics

TUS operates on the principles of acoustic wave propagation through heterogeneous media. The acoustic wave equation describes pressure field evolution:

$$
\frac{1}{c(\mathbf{x})^2} \frac{\partial^2 p}{\partial t^2} = \nabla \cdot \left( \frac{1}{\rho(\mathbf{x})} \nabla p \right) + s(\mathbf{x}, t)
$$

where $p$ is acoustic pressure, $c(\mathbf{x})$ is spatially-varying sound speed, $\rho(\mathbf{x})$ is density, and $s(\mathbf{x}, t)$ is the source term.

### Radiation Force

When ultrasound propagates through tissue, it generates a time-averaged radiation force:

$$
F = \frac{2\alpha I}{c}
$$

where $\alpha$ is absorption coefficient, $I$ is acoustic intensity, and $c$ is sound speed. This force displaces tissue by 1-10 μm, detectable via MR-ARFI imaging.

## Frequency Ranges and Applications

| Frequency | Application | Penetration | Focus Quality |
|-----------|-------------|-------------|---------------|
| 180-250 kHz | Deep brain neuromodulation | Excellent (32% skull loss) | Good |
| 400-500 kHz | Cortical targeting | Good (55% skull loss) | Better |
| 650-800 kHz | Superficial targets | Limited (>70% skull loss) | Best |

Lower frequencies penetrate skull more effectively but with reduced focal precision due to longer wavelengths.

## Device Categories

### Entry-level (1-4 elements)
- **NeuroFUS CTX-500**: Single-element, manual targeting
- **Sonic Concepts**: Research transducers

### Mid-range (16-32 elements)
- **OpenLIFU** (Openwater Health): 32-element array with heterogeneous skull correction
- **InSightec ExAblate**: Clinical thermal ablation system

### High-end (128-256 elements)
- **Oxford-UCL 256-element helmet**: MRI-compatible, Fibonacci spiral arrangement
- **Research arrays**: Custom geometries for specific applications

## Skull Challenge

The skull presents the primary obstacle for transcranial ultrasound:

### Acoustic Properties (ITRUSST benchmark values)
- **Cortical bone**: 2800 m/s sound speed, 1850 kg/m³ density
- **Trabecular bone**: 2300 m/s sound speed, 1700 kg/m³ density  
- **Brain tissue**: 1560 m/s sound speed, 1040 kg/m³ density
- **CSF**: 1500 m/s sound speed, 1000 kg/m³ density

### Correction Strategies
1. **CT-based planning**: Estimate skull properties from Hounsfield units
2. **Multi-element phase correction**: Optimize delays to compensate for skull aberration
3. **Adaptive focusing**: Real-time feedback using MR-ARFI or cavitation detection

## Simulation Methods

### Pseudospectral Time-Domain (j-Wave)
- Spectral accuracy with 2-4 points per wavelength
- GPU-accelerated via JAX
- Automatic differentiation for optimization

### Finite Difference Methods
- **k-Wave**: MATLAB-based, widely used
- **Stride**: Full waveform inversion framework
- **PRESTUS**: SimNIBS integration

### Hybrid Approaches  
- **BabelBrain**: Multi-solver comparison platform
- **OpenLIFU**: Clinical workflow integration

## Multi-Modal Integration

### MRI-TFUS Systems
Interleaved MRI monitoring during ultrasound therapy:
- **PRF thermometry**: Temperature monitoring via phase changes
- **MR-ARFI**: Motion-sensitized gradients detect μm-scale displacements
- **Real-time targeting**: Update focus based on anatomical shifts

### Simulation Chain
```
Acoustic simulation → Radiation force → Tissue mechanics → MRI signal
```

This enables end-to-end modeling from transducer parameters to MRI-visible effects.

## Key References

- **aubry2022itrusst**: Aubry et al. (2022). Benchmark problems for transcranial ultrasound simulation: intercomparison of compressional wave models. JASA.
- **stanziola2023jwave**: Stanziola et al. (2023). j-Wave: an open-source differentiable wave simulator. SoftwareX.
- **martin2025tfus**: Martin et al. (2025). MRI-guided transcranial focused ultrasound neuromodulation with a 256-element helmet array. Nature Communications.
- **kaye2011mrarfi**: Kaye et al. (2011). Rapid MR-ARFI method for focal spot localization during focused ultrasound therapy. doi:10.1002/mrm.22662
- **kaye2013mrarfi**: Kaye & Pauly (2013). Adapting MRI acoustic radiation force imaging for in vivo human brain focused ultrasound. doi:10.1002/mrm.24316
- **rieke2008mr**: Rieke & Butts Pauly (2008). MR Thermometry. JMRI 27:376-390.

## Relevant Projects

- **sbi4dwi**: j-Wave adapter, TUS optimization, radiation force modeling, MR-ARFI simulation
- **jwave**: Differentiable acoustic solver, pseudospectral methods
- **brain-fwi**: Full waveform inversion for skull property estimation
- **openlifu-python**: Clinical workflow, skull segmentation, phase correction

## Key Challenges

1. **Skull heterogeneity**: Individual skull properties vary significantly
2. **Standing waves**: Reflections create complex interference patterns  
3. **Nonlinear propagation**: High intensities cause harmonic generation
4. **Real-time constraints**: Clinical systems require <100ms computation
5. **Safety margins**: Thermal dose and mechanical index monitoring

## See Also

- [acoustic-simulation.md](acoustic-simulation.md) — Wave equation solvers and methods
- [tissue-properties.md](tissue-properties.md) — ITRUSST benchmark acoustic values
- [j-wave.md](j-wave.md) — Differentiable pseudospectral solver
- [radiation-force.md](radiation-force.md) — F = 2αI/c force generation
- [mr-arfi.md](mr-arfi.md) — MRI detection of ultrasound-induced displacement