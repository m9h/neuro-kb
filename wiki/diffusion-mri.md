---
type: modality
title: Diffusion MRI
physics: electromagnetic
measurement: water molecule diffusion via signal attenuation
spatial_resolution: 1-3mm isotropic voxels
temporal_resolution: seconds to minutes per image
related: [monte-carlo-simulation.md, finite-element-methods.md, bloch-torrey-equation.md, tissue-microstructure.md, apparent-diffusion-coefficient.md]
---

# Diffusion MRI

Diffusion MRI (dMRI) is a non-invasive magnetic resonance imaging technique that measures the random motion of water molecules in biological tissues. By applying diffusion-sensitizing magnetic field gradients, dMRI can probe tissue microstructure at length scales much smaller than the imaging voxel size, making it a powerful tool for studying white matter architecture and tissue organization in the brain.

## Physics Principles

Diffusion MRI exploits the Brownian motion of water molecules to infer structural properties of tissues. The technique applies pairs of strong magnetic field gradients that encode molecular displacement into the MRI signal phase. Water molecules that diffuse between the two gradient pulses accumulate phase differences that cause signal attenuation proportional to the diffusion properties.

The fundamental relationship is governed by the **Bloch-Torrey equation**, which extends the classical Bloch equations to include diffusion effects:

```
∂M/∂t = γ(M × B) - R·M + ∇·(D∇M)
```

where M is the magnetization, γ is the gyromagnetic ratio, B is the magnetic field, R is the relaxation tensor, and D is the diffusion tensor.

## Acquisition Protocols

### b-value
The diffusion weighting is characterized by the b-value:

```
b = γ²g²δ²(Δ - δ/3)
```

where:
- γ = 267.5 × 10⁶ rad/(s·T) (gyromagnetic ratio for protons)
- g = gradient amplitude (T/m)
- δ = gradient duration (s)
- Δ = separation between gradient pairs (s)

Common b-values range from 0 s/mm² (no diffusion weighting) to 3000+ s/mm² for advanced microstructure imaging.

### Gradient Sequences

| Sequence | Description | Applications |
|----------|-------------|--------------|
| **PGSE** | Pulsed Gradient Spin Echo - standard sequence with two gradient pulses | DTI, basic diffusion mapping |
| **Double-PGSE** | Two PGSE blocks in series | Microscopic anisotropy, compartment size |
| **OGSE** | Oscillating gradients with sinusoidal/cosine waveforms | Short diffusion times, surface-to-volume ratio |

## Signal Models

### Analytical Models
Biophysical compartment models represent tissue as combinations of geometric shapes with different diffusion properties:

- **Ball**: Isotropic Gaussian diffusion (CSF, gray matter)
- **Stick**: Perfectly restricted diffusion along cylinder axis (intra-axonal)
- **Zeppelin**: Hindered Gaussian diffusion (extra-axonal)
- **Sphere**: Restricted diffusion in spherical geometry (cell bodies)

### Multi-Compartment Models

| Model | Compartments | Key Parameters |
|-------|-------------|----------------|
| **NODDI** [@zhang2012noddi] | Intra-axonal (Stick) + Extra-axonal (Zeppelin) + CSF (Ball) | Neurite density (νᵢc), orientation dispersion (ODI) |
| **SANDI** [@palombo2020sandi] | Soma (Sphere) + Neurite (Stick) + Extra-cellular (Ball) | Soma radius (Rsoma), neurite density (νn) |
| **Multi-TE NODDI** [@gong2020mte] | Standard NODDI with T₂ relaxation | T₂ times for each compartment |

### Advanced Models
- **Bingham-NODDI** [@tariq2016bingham]: Uses Bingham distribution for more complex fiber orientations
- **C-NODDI** [@alsameen2023cnoddi]: Constrained version addressing parameter degeneracies
- **Karger Exchange** [@karger1985nmr]: Models water exchange between compartments

## Measurement Properties

### Spatial Resolution
- **Typical voxel size**: 1.5-3.0 mm isotropic
- **High-resolution**: 0.8-1.2 mm (specialized sequences)
- **Trade-offs**: Smaller voxels reduce signal-to-noise ratio

### Temporal Resolution  
- **Single image**: 3-10 seconds
- **Complete protocol**: 5-45 minutes depending on:
  - Number of gradient directions (6-256+)
  - b-value shells (1-4 shells typical)
  - Spatial coverage and resolution

### Diffusion Time Scales
The diffusion time (Δ - δ/3) determines the length scale probed:
- **Short times** (<10 ms): Surface-to-volume ratio, restriction
- **Long times** (>40 ms): Tortuosity, connectivity

## Tissue Properties

Different tissue types exhibit characteristic diffusion signatures:

| Tissue Type | Mean Diffusivity (μm²/ms) | Fractional Anisotropy | Notes |
|-------------|---------------------------|----------------------|-------|
| **CSF** | 3.0-3.2 | <0.1 | Free water diffusion |
| **Gray matter** | 0.7-0.9 | 0.1-0.3 | Isotropic with some structure |
| **White matter** | 0.6-0.8 | 0.3-0.8 | Highly anisotropic along fibers |
| **Corpus callosum** | 0.7-1.1 | 0.6-0.9 | Highly coherent fiber bundle |

## Simulation Methods

### Monte Carlo Simulation
The gold standard for modeling complex geometries uses random walk simulations:
- **MCMRSimulator.jl** [@cottaarMultimodalMonteCarlo2026]: Multi-modal Monte Carlo with realistic tissue phantoms
- **MEDUSA** [@Ginsburger_2019]: GPU-accelerated sphere packing for realistic microstructure

### Finite Element Methods  
**SpinDoctor.jl** solves the Bloch-Torrey equation on tetrahedral meshes:
- Matrix formalism using Laplace eigenfunctions
- Support for complex geometries (neurons, cylinders, spheres)
- Permeable membrane modeling

### Analytical Solutions
Closed-form expressions for simple geometries:
- Gaussian compartments (Ball, Zeppelin, Tensor)
- Restricted diffusion (Sphere, Cylinder, Plane)
- Van Gelderen cylinder model for axons

## Analysis Methods

### Simulation-Based Inference (SBI)
Modern approach using neural networks to estimate posterior distributions [@manzanopatron2025sbi]:
- **Normalizing flows**: Map between parameter and observation spaces
- **Mixture Density Networks**: Direct posterior approximation  
- **Score-based models**: Diffusion-based posterior sampling

### Traditional Fitting
- **Least squares**: Minimize residual between model and data
- **Bayesian inference**: Full uncertainty quantification with MCMC
- **Dictionary matching**: Pre-computed lookup table approach

## Derived Metrics

From the diffusion tensor **D**, several scalar measures are computed:

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| **Mean Diffusivity (MD)** | (λ₁ + λ₂ + λ₃)/3 | Overall diffusion magnitude |
| **Fractional Anisotropy (FA)** | √(3/2)·‖D̃‖/‖D‖ | Degree of anisotropy (0-1) |
| **Axial Diffusivity (AD)** | λ₁ | Diffusion along principal axis |
| **Radial Diffusivity (RD)** | (λ₂ + λ₃)/2 | Diffusion perpendicular to principal axis |

where λ₁ ≥ λ₂ ≥ λ₃ are the eigenvalues and D̃ is the deviatoric tensor.

## Clinical Applications

### Neuroimaging
- **Tractography**: Reconstruction of white matter pathways
- **Stroke assessment**: Acute ischemia detection via restricted diffusion
- **Tumor characterization**: Differentiate tumor types by diffusion properties
- **Neurodevelopment**: Track myelination and white matter maturation

### Research Applications
- **Microstructure mapping**: Axon diameter, density, myelination
- **Tissue modeling**: Validate biophysical models of brain tissue
- **Method development**: New acquisition and analysis techniques

## Quality Control

### Signal-to-Noise Ratio
Typical SNR requirements:
- **b=0 images**: SNR > 30
- **High b-value**: SNR > 15-20
- **Multi-shell**: Balance between shells

### Motion Correction
- **Eddy current correction**: Compensate for gradient-induced distortions
- **Subject motion**: Retrospective realignment
- **Cardiac pulsation**: Gating or retrospective correction

### Artifacts
- **Susceptibility**: Signal dropout near air-tissue interfaces
- **Partial voluming**: Mixed tissue types within voxels
- **Rician noise**: Non-Gaussian noise distribution at low SNR

## Key References

- **zhang2012noddi**: Zhang et al. (2012). NODDI: practical in vivo neurite orientation dispersion and density imaging. NeuroImage 61:1000-1016.
- **palombo2020sandi**: Palombo et al. (2020). SANDI: a compartment-based model for non-invasive apparent soma and neurite imaging by diffusion MRI. NeuroImage 215:116835.
- **fick2019dmipy**: Fick et al. (2019). The Dmipy Toolbox: Diffusion MRI Multi-Compartment Modeling and Microstructure Recovery Made Easy. Frontiers in Neuroinformatics 13:64.
- **cottaarMultimodalMonteCarlo2026**: Cottaar et al. (2026). Multi-Modal Monte Carlo MRI Simulator of Tissue Microstructure. Imaging Neuroscience. doi:10.1162/IMAG.a.1177
- **manzanopatron2025sbi**: Manzano-Patron et al. (2025). Uncertainty mapping and probabilistic tractography using SBI in diffusion MRI. Medical Image Analysis 103:103580.
- **Jones2022connectom**: Jones et al. (2022). Mapping the human connectome using diffusion MRI at 300 mT/m gradient strength. NeuroImage 254:119146.
- **Stejskal1965**: Stejskal & Tanner (1965). Spin Diffusion Measurements: Spin Echoes in the Presence of a Time-Dependent Field Gradient. J Chem Phys 42:288-292.

## Relevant Projects

| Project | Implementation | Focus |
|---------|---------------|-------|
| **sbi4dwi** | JAX/Equinox | SBI pipeline for microstructure estimation |
| **SpinDoctor.jl** | Julia | Finite element Bloch-Torrey solver |
| **MCMRSimulator.jl** | Julia | Monte Carlo diffusion simulation |

## See Also

- [monte-carlo-simulation.md](monte-carlo-simulation.md) - Random walk methods for diffusion
- [finite-element-methods.md](finite-element-methods.md) - FEM solutions to Bloch-Torrey
- [bloch-torrey-equation.md](bloch-torrey-equation.md) - Governing physics equations
- [tissue-microstructure.md](tissue-microstructure.md) - Biological basis for diffusion
- [apparent-diffusion-coefficient.md](apparent-diffusion-coefficient.md) - Key quantitative measure