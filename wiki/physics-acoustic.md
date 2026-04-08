---
type: physics
title: Acoustic Wave Propagation
governing_equations: Linear acoustic wave equation, Helmholtz equation (frequency domain)
related: [modality-fus.md, modality-us.md, tissue-skull.md, tissue-brain.md, method-pseudospectral.md, method-finite-difference.md]
---

# Acoustic Wave Propagation

The physics of acoustic wave propagation forms the foundation of ultrasound imaging, focused ultrasound therapy, and transcranial ultrasound applications. Understanding these principles is essential for modeling wave behavior in heterogeneous brain tissue.

## Governing Equations

### Time-Domain Wave Equation

The linear acoustic wave equation in heterogeneous media describes pressure field evolution:

$$
\frac{1}{c(\mathbf{x})^2} \frac{\partial^2 p}{\partial t^2} = \nabla \cdot \left( \frac{1}{\rho(\mathbf{x})} \nabla p \right) + s(\mathbf{x}, t)
$$

where:
- $p(\mathbf{x}, t)$ is acoustic pressure (Pa)
- $c(\mathbf{x})$ is spatially varying sound speed (m/s) 
- $\rho(\mathbf{x})$ is density (kg/m³)
- $s(\mathbf{x}, t)$ is the source term

### Frequency-Domain Helmholtz Equation

For time-harmonic fields at angular frequency $\omega$:

$$
\nabla^2 P + k(\mathbf{x})^2 P = -S_M
$$

where $k(\mathbf{x}) = \omega/c(\mathbf{x})$ is the spatially varying wavenumber and $P(\mathbf{x})$ is the complex pressure amplitude.

With absorption, the wavenumber becomes complex:

$$
k = \frac{\omega}{c} \left(1 + i\frac{\alpha c}{\omega}\right)
$$

## Key Physical Parameters

### Sound Speed (c)

Sound speed determines wave propagation velocity and wavelength. Typical values for brain tissues (ITRUSST benchmark):

| Tissue | Sound Speed (m/s) |
|--------|------------------|
| Water/CSF | 1500 |
| Grey matter | 1560 |
| White matter | 1560 |
| Skull (cortical) | 2800 |
| Skull (trabecular) | 2300 |

### Density (ρ)

Acoustic density affects impedance and reflection coefficients:

| Tissue | Density (kg/m³) |
|--------|-----------------|
| Water/CSF | 1000 |
| Grey matter | 1040 |
| White matter | 1040 |
| Skull (cortical) | 1850 |
| Skull (trabecular) | 1700 |

### Acoustic Impedance (Z)

Impedance $Z = \rho c$ governs reflection at interfaces. The large impedance mismatch between brain tissue (~1.62 MRayl) and skull (~5.18 MRayl) causes strong reflections.

### Attenuation (α)

Frequency-dependent absorption, typically specified in dB/(MHz cm):

| Tissue | Attenuation (dB/MHz/cm) |
|--------|------------------------|
| Brain tissue | 0.6 |
| Skull (cortical) | 4.0 |
| Skull (trabecular) | 8.0 |

## Wave Phenomena

### Reflection and Transmission

At interfaces, reflection coefficient:

$$
R = \left(\frac{Z_2 - Z_1}{Z_2 + Z_1}\right)^2
$$

### Scattering

Rayleigh scattering occurs when wavelength >> scatterer size. For transcranial ultrasound at 500 kHz (λ ≈ 3mm in brain), cellular structures (~10 μm) produce Rayleigh scattering.

### Mode Conversion

At skull interfaces, compressional waves partially convert to shear waves, complicating transcranial propagation modeling.

### Absorption Mechanisms

- **Viscous losses**: Dominant in soft tissue
- **Thermal conduction**: Minor contribution
- **Scattering losses**: Significant in heterogeneous media like skull

## Numerical Methods

### Pseudospectral Time-Domain (PSTD)

Used in j-Wave for high-accuracy simulation:
- FFT-based spatial derivatives
- 2-4 points per wavelength sufficient
- Excellent dispersion properties

### Finite Difference Time Domain (FDTD)

Traditional approach:
- Explicit time stepping  
- 10-20 points per wavelength required
- Simple parallelization

### Boundary Element Method (BEM)

For complex geometries:
- Surface mesh only
- Exact far-field conditions
- Computationally expensive for large problems

## Stability Conditions

### CFL Condition

For explicit time-domain methods:

$$
\Delta t \leq \text{CFL} \cdot \frac{\Delta x}{c_{\max}}
$$

Typical CFL values: 0.3-0.5 for stability.

### Grid Sampling

Spatial sampling requirements:
- FDTD: $\Delta x \leq \lambda_{\min}/10$
- PSTD: $\Delta x \leq \lambda_{\min}/4$

## Perfectly Matched Layers (PML)

Absorbing boundary conditions prevent artificial reflections:
- Complex coordinate stretching
- Matches impedance of interior medium
- 10-20 grid points thickness typical

## Key References

- **aubry2022itrusst**: Aubry et al. (2022). Benchmark problems for transcranial ultrasound simulation: intercomparison of compressional wave models. JASA.
- **stanziola2023jwave**: Stanziola et al. (2023). j-Wave: an open-source differentiable wave simulator. SoftwareX.
- **martin2025tfus**: Martin et al. (2025). MRI-guided transcranial focused ultrasound neuromodulation with a 256-element helmet array. Nature Communications.
- **Ishimaru1978wave**: Ishimaru (1978). Wave Propagation and Scattering in Random Media. Academic Press. Foundational text on wave propagation in heterogeneous media.
- **Fang2010mcx**: Fang (2010). Mesh-based Monte Carlo method using fast ray-tracing in Plucker coordinates. Biomed Optics Express 1:165-175.

## Relevant Projects

- **jwave**: JAX-based pseudospectral acoustic simulator with automatic differentiation
- **brain-fwi**: Full waveform inversion for transcranial ultrasound computed tomography
- **sbi4dwi**: Acoustic property mapping and j-Wave adapter for focused ultrasound simulation
- **libspm**: Statistical parametric mapping with acoustic modeling capabilities
- **LAYNII**: Layer-fMRI analysis including ultrasound applications

## See Also

- [modality-fus.md](modality-fus.md) - Focused ultrasound applications
- [modality-us.md](modality-us.md) - Ultrasound imaging principles  
- [tissue-skull.md](tissue-skull.md) - Skull acoustic properties and modeling
- [tissue-brain.md](tissue-brain.md) - Brain tissue acoustic characteristics
- [method-pseudospectral.md](method-pseudospectral.md) - Spectral methods for wave simulation
- [method-finite-difference.md](method-finite-difference.md) - Finite difference acoustic solvers