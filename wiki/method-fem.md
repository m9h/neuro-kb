---
type: method
title: Finite Element Method
category: FEM
implementations: ["neurojax:geometry/fem_forward.py", "sbi4dwi:simulation/mesh_sim.py", "SpinDoctor.jl:src/solve.jl"]
related: [method-bem.md, physics-bioelectromagnetism.md, coordinate-system-native.md, tissue-conductivity.md]
---

# Finite Element Method

The Finite Element Method (FEM) is a numerical technique for solving partial differential equations (PDEs) over complex geometries by discretizing the domain into simple elements (tetrahedra, triangles, hexahedra) and approximating the solution using polynomial basis functions.

In neuroimaging, FEM is primarily used for:
- **EEG/MEG forward modeling** — solving Poisson's equation ∇·(σ∇φ) = -∇·J for electric potential φ given source current density J and tissue conductivity σ
- **Diffusion MRI simulation** — solving the diffusion equation ∂u/∂t = D∇²u in complex tissue geometries
- **Transcranial ultrasound** — solving the wave equation for acoustic pressure through heterogeneous skull

## Mathematical Foundation

### Weak Formulation
Given a PDE in strong form, FEM converts it to a weak (integral) form by multiplying by test functions ψᵢ and integrating over the domain Ω:

For Poisson's equation ∇·(σ∇φ) = -∇·J:
```
∫_Ω σ∇φ·∇ψᵢ dΩ = ∫_Ω J·∇ψᵢ dΩ + ∫_∂Ω σ(∇φ·n)ψᵢ ds
```

This yields a linear system **Kφ = b** where:
- **K** is the stiffness matrix: K_ij = ∫_Ω σ∇ψⱼ·∇ψᵢ dΩ
- **b** is the load vector: b_i = ∫_Ω J·∇ψᵢ dΩ + boundary terms

### Element Assembly
The global matrices are assembled from element-wise contributions:
```
K = Σ_e K^(e)    where K^(e)_ij = ∫_Ωₑ σ∇ψⱼ·∇ψᵢ dΩₑ
```

## Properties and Parameters

| Parameter | Typical Values | Units | Notes |
|-----------|---------------|-------|-------|
| **Element size (brain)** | 1-2 mm | mm | Compromise between accuracy and computation |
| **Element size (skull)** | 0.5-1 mm | mm | Finer mesh needed for thin skull layers |
| **Tetrahedral elements** | 10⁴-10⁷ | count | Linear (4 nodes) or quadratic (10 nodes) |
| **Condition number** | 10³-10⁶ | - | Higher for heterogeneous conductivity |
| **Solver tolerance** | 10⁻⁶-10⁻⁸ | - | CG/GMRES convergence criterion |
| **Memory scaling** | O(N^1.5) | - | For sparse direct solvers (N = nodes) |

### Tissue Conductivity Ranges (EEG/MEG Forward Modeling)

| Tissue | Conductivity | Reference |
|--------|-------------|-----------|
| **Scalp** | 0.465 S/m | IT'IS Foundation v4.1 |
| **Skull (compact)** | 0.01 S/m | IT'IS Foundation v4.1 |
| **Skull (spongy)** | 0.025 S/m | IT'IS Foundation v4.1 |
| **CSF** | 1.79 S/m | Gabriel et al. 1996 |
| **Gray matter** | 0.126 S/m | IT'IS Foundation v4.1 |
| **White matter** | 0.062 S/m | IT'IS Foundation v4.1 |

## Implementation Details

### neurojax Implementation
```python
# neurojax/geometry/fem_forward.py
def assemble_stiffness(vertices, elements, conductivity):
    """Assemble FEM stiffness matrix for Poisson equation"""
    # Compute element matrices using barycentric coordinates
    # Assembly via scatter-add operations
    # Returns sparse matrix K
```

Key features:
- **Differentiable assembly** — `jax.grad` flows through conductivity parameters
- **qMRI conductivity mapping** — σ(T1, BPF) from quantitative MRI
- **GPU acceleration** — JAX-FEM + PETSc backend

### sbi4dwi Implementation
```python
# sbi4dwi/simulation/mesh_sim.py
class MatrixFormalismSimulator:
    """FEM diffusion simulation via matrix exponentials"""
    def simulate_acquisition(self, acquisition):
        # Solve generalized eigenvalue problem: K·φ = λ·M·φ
        # Compute signal via S = exp(-b·D_eff)
```

Features:
- **Spectral ROM** — reduced-order model via eigendecomposition
- **PGSE sequences** — arbitrary gradient waveforms
- **Surface meshes** — triangular meshes for cell/axon geometries

### SpinDoctor.jl Implementation
- **Laplace eigenfunction basis** — analytical eigenfunctions for simple geometries
- **Adaptive refinement** — mesh refinement based on error estimates
- **Parallel assembly** — distributed-memory parallelization

## Advantages vs Boundary Element Method

| Aspect | FEM | BEM |
|--------|-----|-----|
| **Domain discretization** | Volume (3D) | Surface (2D) |
| **Heterogeneous conductivity** | Natural | Requires multiple surfaces |
| **Anisotropic conductivity** | Fully supported | Limited |
| **Matrix properties** | Sparse, symmetric | Dense, non-symmetric |
| **Memory scaling** | O(N) | O(N²) |
| **Boundary conditions** | Natural/essential | Natural only |
| **Commercial tools** | COMSOL, Ansys | OpenMEEG |

FEM excels for highly heterogeneous head models (>10 tissue types) while BEM is preferred for 3-4 layer models with isotropic conductivities.

## Validation and Benchmarks

### Spherical Head Models
For concentric sphere models with analytical solutions:
- **3-layer sphere**: FEM achieves <1% relative error with 50k tetrahedra
- **4-layer sphere**: FEM matches BEM to <0.5% (Vorwerk et al. 2014)

### CHARM Atlas
60-tissue FEM head models achieve:
- **Leadfield correlation** vs 3-layer BEM: r > 0.95 for superficial sources
- **Deep source improvement**: 10-20% better localization than BEM (Huang et al. 2016)

## Key References

- **Wolters2004fem**: Wolters et al. (2006). Influence of tissue conductivity anisotropy on EEG/MEG field and return current computation using high-resolution FEM. NeuroImage 30:813-826.
- **Arridge1993fem**: Arridge et al. (1993). A finite element approach for modeling photon transport in tissue. Medical Physics 20:299-309.
- **Schweiger1993fem**: Schweiger et al. (1995). The finite element method for the propagation of light in scattering media: boundary and source conditions. Medical Physics 22:1779-1792.
- **Li2019**: Li et al. (2019). SpinDoctor: A MATLAB toolbox for diffusion MRI simulation. NeuroImage 202:116120.
- **Agdestein2021**: Agdestein et al. (2022). Practical computation of the diffusion MRI signal based on Laplace eigenfunctions: permeable interfaces. NMR in Biomedicine, e4646.
- **Raissi2019pinns**: Raissi et al. (2019). Physics-informed neural networks for solving forward and inverse problems involving nonlinear PDEs. J Comp Physics 378:686-707.

## Relevant Projects

- **neurojax**: Differentiable FEM for EEG/MEG source imaging with qMRI conductivity
- **sbi4dwi**: FEM diffusion simulation for microstructure inference
- **SpinDoctor.jl**: Mature FEM toolbox for diffusion MRI simulation in complex geometries

## See Also

- [method-bem.md](method-bem.md) — Boundary Element Method comparison
- [physics-bioelectromagnetism.md](physics-bioelectromagnetism.md) — Maxwell equations in biological tissue
- [tissue-conductivity.md](tissue-conductivity.md) — Electrical properties of brain tissues
- [head-model-charm.md](head-model-charm.md) — High-resolution FEM head models