---
type: physics
title: Diffusion Equation (water, photon)
governing_equations: ∇·(D∇u) = ∂u/∂t (parabolic), ∇²u = k²u (Helmholtz)
related: [physics-bloch-torrey-equation.md, physics-acoustic-wave-equation.md, method-fem-spectral.md, tissue-brain-white-matter.md, tissue-csf.md]
---

# Diffusion Equation (water, photon)

The diffusion equation governs the transport of water molecules, photons, and other diffusing particles through biological tissues. It serves as the mathematical foundation for diffusion MRI microstructure modeling and optical imaging applications.

## Governing Equations

### Time-Dependent Diffusion (Parabolic)
The fundamental diffusion equation in isotropic media:
```
∂u/∂t = D∇²u
```

For anisotropic tissues with diffusion tensor **D**:
```
∂u/∂t = ∇·(D∇u)
```

### Steady-State Diffusion (Helmholtz)
For frequency-domain analysis (OGSE sequences, optical imaging):
```
∇²u + k²u = 0
```
where k² = iω/D for diffusion coefficient D and angular frequency ω.

## Parameters

| Parameter | Symbol | Typical Values | Units |
|-----------|--------|---------------|-------|
| **Water diffusion (free)** | D₀ | 3.0 × 10⁻⁹ | m²/s |
| **Brain white matter (parallel)** | D∥ | 1.7 × 10⁻⁹ | m²/s |
| **Brain white matter (perpendicular)** | D⊥ | 0.2 × 10⁻⁹ | m²/s |
| **CSF** | D_CSF | 3.2 × 10⁻⁹ | m²/s |
| **Photon diffusion (brain)** | D_ph | 0.35-0.85 | mm |
| **Optical absorption (brain)** | μₐ | 0.1-0.2 | cm⁻¹ |
| **Optical scattering (brain)** | μₛ' | 8-15 | cm⁻¹ |

## Boundary Conditions

### Water Diffusion
- **Impermeable membranes**: ∇u·**n** = 0 (zero flux)
- **Permeable membranes**: κ[u] = D∇u·**n** (Robin condition)
  - κ: permeability coefficient (10⁻⁶ to 10⁻⁴ m/s)
- **Periodic boundaries**: for modeling packed geometries

### Optical Diffusion
- **Tissue-air interface**: Robin boundary condition
- **Extrapolated boundary**: u = 0 at distance z₀ = 2D outside tissue

## Solution Methods

### Analytical Solutions
- **Infinite medium**: u(r,t) = (4πDt)⁻³/² exp(-r²/4Dt)
- **Spherical geometry**: eigenfunction series with Bessel functions
- **Cylindrical geometry**: modified Bessel functions I₀, K₀

### Numerical Methods (from SpinDoctor.jl, sbi4dwi)

#### Matrix Formalism (Spectral)
1. **Eigendecomposition**: Solve (S - λM)v = 0
   - S: stiffness matrix (∇·(D∇) discretization)
   - M: mass matrix (finite element)
2. **Projection**: Express solution in eigenspace
3. **Time evolution**: Matrix exponential exp(-λt)

#### Finite Element Method
- **P1 elements**: Linear basis functions on tetrahedra/triangles
- **Mesh generation**: TetGen (3D), Triangle (2D)
- **Time stepping**: Backward Euler, Crank-Nicolson

#### Monte Carlo Random Walk
- **Step size**: Δr = √(6DΔt) for 3D
- **Boundary interactions**: reflection, transmission
- **Statistical convergence**: N ~ 10⁶ walkers typical

## Applications

### Diffusion MRI (SpinDoctor.jl, sbi4dwi)
- **PGSE sequences**: b-value = γ²g²δ²(Δ-δ/3)
- **OGSE sequences**: frequency-dependent diffusion
- **Multi-shell protocols**: multiple b-values for microstructure
- **ADC estimation**: D_ADC = -∂log(S)/∂b|_{b=0}

### Optical Imaging
- **Diffuse optical tomography (DOT)**
- **Time-resolved spectroscopy**
- **Fluorescence lifetime imaging**

### Matrix Formalism Signal Prediction
For PGSE sequence with gradient pulses at times [0,δ] and [Δ,Δ+δ]:
```
S/S₀ = Tr(ρ exp(-δK) exp(-(Δ-δ)Λ) exp(-δK))
```
where:
- K = Λ + iγgq·**R** (encoding matrix)
- Λ = diag(λᵢ) (eigenvalues)
- **R** = V†**M_x**V (position matrices in eigenspace)

## Physical Interpretation

### Diffusion Length Scale
Characteristic distance: L_d = √(2Dt)
- **Clinical dMRI** (Δ ~ 40ms): L_d ~ 17 μm
- **Microstructure sensitivity**: L_d comparable to axon diameter (1-20 μm)

### Frequency Dependence (OGSE)
Penetration depth: δ = √(2D/ω)
- **Low frequency**: sensitive to restrictions
- **High frequency**: approaches free diffusion

## Implementation Notes

### Computational Considerations
- **Matrix formalism**: ~100-1000 eigenvalues sufficient
- **Time stepping**: Δt ≤ h²/(6D) stability condition
- **Memory scaling**: O(N²) for dense matrices, O(N) for sparse

### Validation Benchmarks
- **Analytical comparisons**: spheres, cylinders, parallel plates
- **Monte Carlo ground truth**: 10⁶ walkers
- **Experimental validation**: phantoms with known geometry

## Related Physics

The diffusion equation is closely related to:
- [physics-bloch-torrey-equation.md](physics-bloch-torrey-equation.md) - adds T₂ relaxation and phase evolution
- [physics-acoustic-wave-equation.md](physics-acoustic-wave-equation.md) - wave vs diffusive transport
- Heat equation (same mathematical form)
- Schrödinger equation (imaginary time)

## Relevant Projects

### sbi4dwi
- **Analytical models**: Ball, Stick, Zeppelin compartments
- **FEM simulation**: `MatrixFormalismSimulator` class
- **Monte Carlo**: ground truth validation
- **SBI training**: synthetic signal generation

### SpinDoctor.jl
- **Geometry recipes**: neurons, spheres, cylinders
- **Matrix formalism**: eigenfunction-based solutions  
- **Multi-compartment**: permeable membrane modeling
- **Validation suite**: analytical vs numerical comparisons

### dot-jax
- **Optical diffusion**: photon transport modeling
- **Time-domain**: pulse-response simulation
- **Frequency-domain**: modulated light sources

## Key References

- **Stejskal1965**: Stejskal & Tanner (1965). Spin Diffusion Measurements: Spin Echoes in the Presence of a Time-Dependent Field Gradient. J Chem Phys 42:288-292.
- **Grebenkov2007**: Grebenkov (2007). NMR survey of reflected Brownian motion. Reviews of Modern Physics 79:1077-1137.
- **Li2019**: Li et al. (2019). SpinDoctor: A MATLAB toolbox for diffusion MRI simulation. NeuroImage 202:116120.
- **Agdestein2021**: Agdestein et al. (2022). Practical computation of the diffusion MRI signal based on Laplace eigenfunctions: permeable interfaces. NMR in Biomedicine, e4646.
- **Farrell1992diffusion**: Farrell et al. (1992). A diffusion theory model of spatially resolved, steady-state diffuse reflectance. Medical Physics 19:879-888.
- **Haskell1994boundary**: Haskell et al. (1994). Boundary conditions for the diffusion equation in radiative transfer. JOSA A 11:2727-2741.

## See Also

- [method-fem-spectral.md](method-fem-spectral.md) - Finite element eigendecomposition
- [tissue-brain-white-matter.md](tissue-brain-white-matter.md) - Anisotropic diffusion properties
- [coordinate-system-gradient-table.md](coordinate-system-gradient-table.md) - Experimental encoding
- [concept-microstructure-modeling.md](concept-microstructure-modeling.md) - Biophysical applications