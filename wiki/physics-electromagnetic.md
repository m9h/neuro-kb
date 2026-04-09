```yaml
type: physics
title: Electromagnetic Forward Problem
physics: electromagnetic
governing_equations: Maxwell equations in quasistatic approximation
related: [tissue-conductivity.md, head-model-bem.md, head-model-fem.md, leadfield-matrix.md]
```

# Electromagnetic Forward Problem

The electromagnetic forward problem in neuroimaging computes the electric potential and magnetic field at sensor locations given a current source distribution in the brain. This forms the foundation for EEG and MEG source imaging.

## Governing Physics

### Quasistatic Approximation

For EEG/MEG frequencies (0.1-100 Hz), the wavelength (λ ≈ 3000 km at 100 Hz) is much larger than head dimensions (~20 cm). This enables the **quasistatic approximation**:

```
∇ × E = 0          (no magnetic induction)
∇ · J = 0          (current conservation)  
J = σE = -σ∇φ      (Ohm's law)
```

The electric field derives from a scalar potential φ, and magnetic fields are computed via Biot-Savart law from current density J.

### Poisson Equation

Combining the above yields the **forward equation**:
```
∇ · (σ∇φ) = ∇ · Jp
```

Where:
- φ: electric potential (V)
- σ: tissue conductivity tensor (S/m)
- Jp: primary current density from neural activity (A/m²)

## Tissue Properties

| Tissue | Conductivity (S/m) | Temperature | Frequency | Source |
|--------|-------------------|-------------|-----------|---------|
| Scalp | 0.465 | 37°C | 10 Hz | Gabriel et al. 1996 |
| Skull | 0.010 | 37°C | 10 Hz | Gabriel et al. 1996 |
| CSF | 1.79 | 37°C | 10 Hz | Gabriel et al. 1996 |
| Gray matter | 0.275 | 37°C | 10 Hz | Gabriel et al. 1996 |
| White matter | 0.126 | 37°C | 10 Hz | Gabriel et al. 1996 |

### Anisotropy

White matter exhibits **electrical anisotropy** with conductivity parallel to fiber tracts ~10× higher than perpendicular:

```
σ_parallel / σ_perpendicular ≈ 10:1
```

The conductivity tensor is:
```
σ = σ_⊥I + (σ_∥ - σ_⊥)vv^T
```

Where v is the principal diffusion eigenvector from DTI.

## Numerical Methods

### Boundary Element Method (BEM)

For piecewise-constant conductivity regions, the forward problem reduces to surface integral equations. **Symmetric BEM** (Kybic et al. 2005) handles the dipole singularity:

```
φ(r) = Σᵢ σᵢ ∫_{∂Ωᵢ} G(r,r') φ(r') ds'
```

Where G(r,r') = 1/(4π|r-r'|) is the free-space Green's function.

### Finite Element Method (FEM)

For continuously varying conductivity (e.g., from qMRI), FEM discretizes the Poisson equation:

```
Kφ = b
```

Where:
- K: stiffness matrix from ∫∇Nᵢ·(σ∇Nⱼ)dV
- b: source vector from current dipoles
- Nᵢ: FEM basis functions

## Forward Models

### EEG Forward Model

Electric potential at electrode i:
```
φᵢ = Σⱼ Lᵢⱼ Jⱼ
```

Where L is the **leadfield matrix** linking source j to electrode i.

### MEG Forward Model

Magnetic field at magnetometer/gradiometer i:
```
Bᵢ = (μ₀/4π) ∫ J(r') × (r-r')/|r-r'|³ dV'
```

MEG is **reference-free** (independent of volume conductor) but sensitive to tangential currents only.

## Implementation Details

### Dipole Source Model

Neural current sources are modeled as current dipoles:
```
Jp(r) = Σₖ qₖ δ(r - rₖ) ηₖ
```

Where:
- qₖ: dipole moment magnitude (nAm)
- rₖ: dipole location
- ηₖ: dipole orientation (unit vector)

### Source Space

Typical source spaces:
- **Cortical surface**: ~4000-8000 vertices from FreeSurfer
- **Volume grid**: ~5mm isotropic, ~6000 voxels
- **Mixed**: Surface + subcortical structures

## Validation

### Spherical Head Models

Analytical solutions exist for concentric spheres (Sarvas 1987 for MEG, de Munck 1993 for EEG). Used for:
- Algorithm validation
- Leadfield computation in source localization software
- Benchmarking realistic head models

### Reciprocity

**EEG reciprocity**: φ(electrode|dipole) = φ(dipole|electrode)
**MEG reciprocity**: B(sensor|dipole) = B(dipole|sensor)

Used for efficient leadfield computation and validation.

## Error Sources

### Geometric Errors

- Head model segmentation: ±2mm affects leadfield by ~20%
- Electrode/sensor registration: ±5mm affects source localization by ~1cm
- Skull conductivity uncertainty: factor of 2-5 variation in literature

### Modeling Assumptions

- Quasistatic approximation breaks down >1 kHz
- Piecewise constant conductivity ignores tissue microstructure  
- Neglect of skull anisotropy introduces ~15% error in EEG forward fields

## Key References

- **Wolters2004fem**: Wolters et al. (2006). Influence of tissue conductivity anisotropy on EEG/MEG field and return current computation using high-resolution FEM. NeuroImage 30:813-826.
- **Michel2004eegsi**: Michel et al. (2004). EEG source imaging. Clinical Neurophysiology 115:2195-2222.
- **Hamalainen1994mne**: Hamalainen & Ilmoniemi (1994). Interpreting magnetic fields of the brain: minimum norm estimates. Med & Biol Eng & Comp 32:35-42.
- **Lin2006loose**: Lin et al. (2006). Distributed current estimates using cortical orientation constraints. Human Brain Mapping 27:1-13.
- **GraveDePeralta2004laura**: Grave de Peralta Menendez et al. (2004). Electrical neuroimaging based on biophysical constraints. NeuroImage 21:527-539.

## Relevant Projects

**neurojax**: Differentiable BEM/FEM forward modeling with JAX
- `geometry/bem_forward.py`: Symmetric BEM with OpenMEEG integration
- `geometry/fem_forward.py`: JAX-FEM assembly with PETSc solvers
- `geometry/sigma_from_qmri.py`: qMRI → conductivity mapping

**vbjax**: Neural mass model forward operators for EEG/MEG simulation
- Source-level dynamics → sensor-level observations via leadfield multiplication
- Validation against analytical spherical head model solutions

## See Also

- [tissue-conductivity.md](tissue-conductivity.md)
- [head-model-bem.md](head-model-bem.md) 
- [head-model-fem.md](head-model-fem.md)
- [leadfield-matrix.md](leadfield-matrix.md)
- [source-imaging-methods.md](source-imaging-methods.md)