---
type: method
title: Finite Element Method
category: FEM
implementations: ["neurojax:geometry/fem_forward.py", "sbi4dwi:dmipy_jax/simulation/mesh_sim.py", "SpinDoctor.jl:src/solve.jl", "ti-sim", "ae-sim", "fem-trials"]
related: [method-bem.md, physics-electromagnetic.md, physics-diffusion-equation.md, physics-acoustic.md, tissue-electrical-properties.md, tissue-skull.md, tissue-csf.md, temporal-interference.md, acoustoelectric-imaging.md, diffusion-mri.md, method-source-imaging.md, eeg.md, meg.md, sci-head-model.md, head-model-mida.md, coordinate-systems.md, method-monte-carlo.md, method-sbi.md, jax-ecosystem.md]
description: "The Finite Element Method (FEM) solves partial differential equations (PDEs) on complex geometries by discretizing the domain into simple elements (tetrahedra in 3D, triangles on surfaces) and approximating the solution with piecewise-polyn"
timestamp: 2026-11-05T15:59:53-08:00
tags: [method, fem, forward-model]
---

# Finite Element Method

The Finite Element Method (FEM) solves partial differential equations (PDEs) on complex geometries by discretizing the domain into simple elements (tetrahedra in 3D, triangles on surfaces) and approximating the solution with piecewise-polynomial basis functions ψᵢ. Unlike the [Boundary Element Method](method-bem.md), FEM discretizes the whole volume, so it handles arbitrary heterogeneity and anisotropy element by element.

Across the neuro-kb project family, FEM appears in three distinct physical roles:

| Role | PDE | Unknown | Projects |
|------|-----|---------|----------|
| EEG/MEG forward model | ∇·(σ∇φ) = ∇·J_p | Electric potential φ | neurojax |
| Stimulation field solves (TI, tDCS-like) | ∇·(σ∇φ) = 0 + electrode BCs | φ, E = −∇φ | ti-sim, ae-sim |
| Diffusion MRI signal simulation | Bloch–Torrey: ∂M/∂t = D∇²M − iγ(g(t)·x)M | Transverse magnetization M | sbi4dwi, SpinDoctor.jl |
| Acoustoelectric coupling | Acoustic pressure → Δσ → Poisson | P, φ | ae-sim |

Transcranial ultrasound in this family is mostly **not** FEM: sbi4dwi and brain-fwi use j-Wave pseudospectral time-domain (PSTD) and Helmholtz/GMRES solvers on regular grids [@stanziola2023jwave] (see [physics-acoustic.md](physics-acoustic.md)). FEM enters acoustics only where acoustic and electric fields are coupled on a shared mesh (ae-sim).

## Mathematical Foundation

### Weak formulation (bioelectric forward problem)

With quasi-static Maxwell equations (see [physics-electromagnetic.md](physics-electromagnetic.md)), total current J = −σ∇φ + J_p is divergence-free, giving

```
∇·(σ∇φ) = ∇·J_p        in Ω (head)
σ ∂φ/∂n = 0            on ∂Ω (scalp/air boundary)
```

Multiplying by a test function ψᵢ and integrating by parts (the Neumann boundary term vanishes):

```
∫_Ω σ∇φ·∇ψᵢ dΩ = ∫_Ω J_p·∇ψᵢ dΩ
```

Expanding φ = Σⱼ φⱼψⱼ yields the sparse linear system **Kφ = b**:

- **Stiffness matrix**: K_ij = ∫_Ω σ∇ψⱼ·∇ψᵢ dΩ (σ may be a 3×3 tensor per element)
- **Load vector** (point dipole q at r₀, "partial integration" source model): b_i = q·∇ψᵢ(r₀)

The pure-Neumann problem is singular (φ defined up to a constant); one node is fixed as reference or a zero-mean constraint is imposed. Dipole singularities are alternatively handled with subtraction or Saint-Venant source models — the same singularity issue OpenMEEG addresses on the BEM side (neurojax:README.md, "Research lineage").

### Element assembly

```
K = Σ_e K^(e),   K^(e)_ij = |V_e| · (∇ψⱼ)ᵀ σ_e (∇ψᵢ)     (linear P1 tetrahedra)
```

For linear tetrahedra the gradients are constant per element, so each K^(e) is a 4×4 dense block computed from barycentric coordinates and scattered into the global sparse matrix. In JAX this scatter-add is differentiable with respect to σ_e and to vertex positions.

### Stimulation field solves

For [temporal interference](temporal-interference.md) (ti-sim), two independent Laplace solves ∇·(σ∇φ_k) = 0 are performed, one per electrode pair (fixed current injection on electrode surfaces, Neumann elsewhere). The envelope amplitude along direction n̂ is |E_AM(n̂)| = | |(E₁+E₂)·n̂| − |(E₁−E₂)·n̂| |, which for collinear fields reduces to 2·min(|E₁|, |E₂|). Because FEM is linear in injected current, one solve per pair suffices and current ratios are rescaled post hoc.

### Diffusion MRI (Bloch–Torrey) on meshes

Discretizing the Bloch–Torrey equation with P1 elements gives

```
M dξ/dt = −(D·K + iγ (g(t)·J)) ξ
```

with mass matrix M_ij = ∫ψᵢψⱼ, stiffness K, and first-moment matrices J_ij = ∫ x ψᵢψⱼ. The **matrix formalism** [@Callaghan1997] projects onto the lowest Laplace eigenfunctions from the generalized eigenproblem K·φ = λ·M·φ, turning time stepping into small matrix exponentials per gradient interval [@Li2020; @Grebenkov2007]. Permeable interfaces between compartments are treated in [@agdestein2021practical]. The signal is S = ∫_Ω M(x, T_E) dx.

## Properties and Parameters

| Parameter | Typical value | Source / notes |
|-----------|---------------|----------------|
| Element type (head) | Linear tetrahedra (4 nodes); quadratic (10 nodes) for higher accuracy | Standard practice |
| Element edge length, brain | 1–2 mm | Accuracy/cost compromise |
| Element edge length, skull | 0.5–1 mm | Skull is 3–7 mm thick; needs ≥2–3 layers across ([tissue-skull.md](tissue-skull.md)) |
| Mesh size, full head | 10⁵–10⁷ tetrahedra | Depends on tissue count and refinement |
| Tissue classes, CHARM | 60 | neurojax:README.md (CHARM 60-tissue segmentation) |
| Tissue classes, SCI head model | 8 (208×256×256 at 1 mm) | sbi4dwi:docs/openlifu/README.md; [sci-head-model.md](sci-head-model.md) |
| Iterative solver tolerance | 10⁻⁶–10⁻⁸ relative residual | CG with AMG/ILU preconditioning (PETSc backend in neurojax) |
| K condition number | 10³–10⁶ | Grows with conductivity contrast (CSF:skull ≈ 100:1 or more) |
| Sparse direct solver memory | ~O(N^1.5) in 3D (N = nodes) | Nested-dissection fill-in |
| Free diffusivity (dMRI sims) | D = 2×10⁻⁹ m²/s | sbi4dwi:README.md (`MatrixFormalismSimulator.from_mesh(..., D=2e-9)`) |
| Laplace eigenmodes retained | Truncated by eigenvalue cutoff | Controls ROM accuracy vs speed [@Li2020] |

### Conductivity inputs (EEG/MEG/TI field solves)

FEM accuracy is dominated by conductivity choice rather than discretization for most realistic meshes. Detailed values live on [tissue-electrical-properties.md](tissue-electrical-properties.md); per-tissue pages are [tissue-scalp.md](tissue-scalp.md), [tissue-skull.md](tissue-skull.md), [tissue-csf.md](tissue-csf.md), [tissue-gray-matter.md](tissue-gray-matter.md), [tissue-white-matter.md](tissue-white-matter.md).

| Tissue | Commonly used isotropic value | Notes |
|--------|------------------------------|-------|
| Scalp | 0.465 S/m | Common default in FEM head pipelines |
| Skull (compact) | ~0.01 S/m | Most uncertain parameter; large inter-subject variation |
| Skull (spongy) | ~0.025 S/m | Only resolvable in multi-compartment skull meshes |
| CSF | 1.79 S/m at 10 Hz (Gabriel 1996) | Highest-conductivity compartment; shunts current |
| Gray matter | ~0.275 S/m | See note below |
| White matter | ~0.126 S/m (isotropic) | Anisotropic tensors from DTI in neurojax |

> **Correction note:** A previous revision of this page listed GM = 0.126 S/m and WM = 0.062 S/m. These appear shifted relative to the defaults used by mainstream FEM head-modeling pipelines (GM ≈ 0.275, WM ≈ 0.126 S/m). Verify against [tissue-electrical-properties.md](tissue-electrical-properties.md) before use.

neurojax replaces fixed lookup values with a differentiable map σ(T1, BPF) from [quantitative MRI](quantitative-mri.md) (VFA T1 and qMT bound-pool fraction) plus DTI-derived anisotropy (neurojax:README.md, "Head modeling pipeline").

## Implementation Details

### neurojax — differentiable bioelectric FEM

```python
from neurojax.geometry.fem_forward import assemble_stiffness, sigma_from_qmri

sigma = sigma_from_qmri(t1_values, bpf_values, labels, params)  # differentiable
K = assemble_stiffness(vertices, elements, sigma)
grad_params = jax.grad(loss)(params)  # end-to-end through FEM
```

- Pipeline: T1w → CHARM 60-tissue segmentation (SAMSEG + extended atlas) → surfaces/meshes → conductivity assignment → BEM (MNE/OpenMEEG) **or** FEM (JAX-FEM + PETSc) → leadfield L → any of 15 inverse solvers ([method-source-imaging.md](method-source-imaging.md)).
- `jax.grad` flows from source-imaging loss through L, FEM assembly, conductivity mapping, into qMRI tissue parameters.
- Lineage: CAUCHY/SimBio (1993+, Wolters/Buchner/Anwander; NeuroFEM kernel in BESA MRI and CURRY) → Fijee (2014, Cobigo, C++/FEniCS coupling FEM forward with Wendling neural mass models) → neurojax (neurojax:CLAUDE.md, "Research heritage").
- The autoresearch loop explicitly varies "BEM vs FEM, mesh resolution, tissue conductivity" to measure source-localization advantage (DLE) (neurojax:autoresearch/program.md).

### sbi4dwi — FEM spectral ROM for dMRI

```python
from dmipy_jax.simulation.mesh_sim import MatrixFormalismSimulator
fem = MatrixFormalismSimulator.from_mesh(vertices, triangles, D=2e-9)
signal = fem.simulate_acquisition(acquisition)
```

- Builds stiffness/mass matrices from triangular surface meshes, solves the generalized eigendecomposition, and simulates PGSE sequences by matrix exponentials; supports arbitrary gradient directions and full `JaxAcquisition` protocols (sbi4dwi:README.md).
- Differentiable, unlike the `MonteCarloSimulator` ground-truth walker ([method-monte-carlo.md](method-monte-carlo.md)); used inside `SimulationComparisonRunner` (analytical vs FEM vs MC vs oracle) and as a training-data source for [simulation-based inference](method-sbi.md) alongside analytical compartments [@fick2019dmipy].
- ReMiDi (Docker) is wrapped as an external FEM-class oracle producing HDF5 `SimulationLibrary` datasets.

### SpinDoctor.jl — reference dMRI FEM

- Julia port of the SpinDoctor MATLAB toolbox [@Li2019]: tetrahedral meshes of cells/neurons (e.g. NeuroMorpho morphologies [@Ascoli2007]), Bloch–Torrey solved either by time stepping or by the Laplace-eigenfunction matrix formalism [@Li2020].
- Supports permeable membranes between compartments [@agdestein2021practical] and generalized gradient waveforms [@Callaghan1997; @Callaghan1995].
- Serves as the high-fidelity reference against which the faster JAX ROM in sbi4dwi is checked.

### ti-sim and ae-sim — stimulation and acoustoelectric field solves

- **ti-sim**: FEM field solves for temporal interference, validated on the Grossman phantom geometry; computes per-pair E-fields and TI envelopes ([temporal-interference.md](temporal-interference.md)).
- **ae-sim**: coupled acoustic + electric FEM with open-source solvers. The acoustoelectric effect modulates conductivity with pressure (Δσ/σ ∝ −K·ΔP), so the electric Poisson problem is re-solved on the same mesh with a pressure-dependent σ ([acoustoelectric-imaging.md](acoustoelectric-imaging.md)).
- **fem-trials**: sandbox for comparing FEM solver stacks; details not yet ingested into the wiki.

## FEM vs BEM

| Aspect | FEM | BEM ([method-bem.md](method-bem.md)) |
|--------|-----|-----|
| Discretization | Volume (tetrahedra) | Interfaces between nested surfaces |
| Heterogeneity | Arbitrary, per element | One constant σ per closed compartment |
| Anisotropy (WM, skull layers) | Native (tensor σ_e) | Not supported in standard formulations |
| System matrix | Sparse, symmetric positive (semi)definite | Dense; symmetric in OpenMEEG's symmetric formulation |
| Memory | ~O(N) storage (sparse) | O(N²) |
| Unknowns | 10⁵–10⁷ nodes | 10³–10⁴ surface vertices |
| Typical tools | SimBio/NeuroFEM, FEniCS, JAX-FEM, COMSOL | OpenMEEG, MNE-Python |

FEM is preferred for many-tissue models (CHARM 60 classes, [head-model-mida.md](head-model-mida.md), [sci-head-model.md](sci-head-model.md)) and anisotropic conductivity; 3-layer BEM remains the efficient default for isotropic scalp/skull/brain MEG models.

## Validation Strategy

- **Concentric spheres**: multilayer sphere models have analytical EEG solutions; FEM leadfields are checked by relative difference measure (RDM) and magnitude error versus the series solution.
- **FEM vs BEM cross-check**: on identical 3-layer geometries the two methods should agree closely; disagreement localizes meshing errors near thin skull.
- **dMRI**: FEM signals are compared against closed-form restricted diffusion in spheres/cylinders/planes [@Callaghan1997; @Barzykin1999] and against Monte Carlo random walks; short-time surface-to-volume behaviour follows [@Mitra1992].
- **Coordinates**: mesh, sensor, and MRI frames must agree before leadfield computation ([coordinate-systems.md](coordinate-systems.md)).

## Relevant Projects

- **neurojax** — differentiable FEM EEG/MEG forward model (JAX-FEM + PETSc), qMRI-informed conductivity, CHARM 60-tissue meshes.
- **sbi4dwi** — `MatrixFormalismSimulator` FEM spectral ROM for dMRI; ReMiDi oracle; feeds SBI training.
- **SpinDoctor.jl** — reference Bloch–Torrey FEM toolbox for neurons and cell geometries.
- **ti-sim** — FEM field solves for temporal interference stimulation (Grossman phantom).
- **ae-sim** — coupled acoustic + electric FEM for acoustoelectric neuroimaging.
- **fem-trials** — FEM solver comparison sandbox.
- **setae** — FEM for contact mechanics and tissue properties (mechanical, not electromagnetic).
- **brain-fwi** — contrast case: acoustic FWI uses pseudospectral grids rather than FEM.

## See Also

- [method-bem.md](method-bem.md) — surface-based alternative for nested isotropic compartments
- [physics-electromagnetic.md](physics-electromagnetic.md) — quasi-static Maxwell equations behind the Poisson problem
- [physics-diffusion-equation.md](physics-diffusion-equation.md) — diffusion / Bloch–Torrey PDEs
- [physics-acoustic.md](physics-acoustic.md) — wave equation and pseudospectral solvers
- [tissue-electrical-properties.md](tissue-electrical-properties.md) — conductivity tables
- [temporal-interference.md](temporal-interference.md) — TI envelope fields from FEM solves
- [acoustoelectric-imaging.md](acoustoelectric-imaging.md) — coupled acoustic–electric modelling
- [diffusion-mri.md](diffusion-mri.md) — dMRI microstructure context
- [method-source-imaging.md](method-source-imaging.md) — inverse solvers consuming FEM leadfields
- [method-monte-carlo.md](method-monte-carlo.md) — random-walk alternative for dMRI simulation
- [sci-head-model.md](sci-head-model.md), [head-model-mida.md](head-model-mida.md) — meshable head models
- [jax-ecosystem.md](jax-ecosystem.md) — JAX-FEM, Lineax, and differentiable assembly
