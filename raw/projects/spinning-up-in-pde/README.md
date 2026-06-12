---
license: apache-2.0
pretty_name: Spinning Up in PDE Solvers
language:
  - en
tags:
  - education
  - curriculum
  - scientific-machine-learning
  - sciml
  - pde
  - partial-differential-equations
  - jax
  - finite-differences
  - finite-elements
  - spectral-methods
  - neural-operators
  - pinns
  - differentiable-physics
size_categories:
  - n<1K
---

# Spinning Up in PDE Solvers

**From Lewis Fry Richardson's forecast factory to differentiable physics — a hands-on curriculum tracing the lineage of computational PDE solvers**

<p align="center">
  <img src="https://img.shields.io/badge/modules-12-blue" alt="12 modules"/>
  <img src="https://img.shields.io/badge/JAX-first-green" alt="JAX-first"/>
  <img src="https://img.shields.io/badge/license-Apache_2.0-orange" alt="License"/>
  <img src="https://img.shields.io/badge/community-HF_Science_%23pde-yellow" alt="HF Science #pde"/>
</p>

---

In 1922, Lewis Fry Richardson published *Weather Prediction by Numerical Process* — a 236-page proposal to forecast the atmosphere by hand, using a "forecast factory" of 64,000 human computers seated in a vast amphitheatre, each solving a finite-difference cell. His own six-hour test forecast diverged into nonsense. Six years later, Courant, Friedrichs, and Lewy explained why: he had violated a stability bound on the timestep. By 1950, Charney, Fjørtoft, and von Neumann ran the same idea on ENIAC and got a forecast that worked. The scheme they used — forward-time centred-space, FTCS — is the same baseline you'll find in every PDEBench plot today, three quarters of a century later.

A century after Richardson, the same problem has a new numerical machine. In 2023, Google DeepMind's [GraphCast](https://deepmind.google/discover/blog/graphcast-ai-model-for-faster-and-more-accurate-global-weather-forecasting/) — a graph neural network operating on a refined icosahedral mesh of the Earth, trained on 39 years of ERA5 reanalysis — produced 10-day global weather forecasts in under sixty seconds on a single TPU and beat ECMWF's HRES, the gold-standard physics-based deterministic forecast, on roughly 90% of measured variables ([Lam et al., *Science* 382, 1416–1421](https://www.science.org/doi/10.1126/science.adi2336)). In December 2024, [GenCast](https://deepmind.google/discover/blog/gencast-predicts-weather-and-the-risks-of-extreme-conditions-with-state-of-the-art-accuracy/) extended the approach with a diffusion-based ensemble of graph networks, again outperforming ECMWF — this time the *ensemble* system, ENS — on probabilistic skill ([Price et al., *Nature* 637, 84–90 (2025)](https://www.nature.com/articles/s41586-024-08252-9)). Both replace Richardson's hand-cranked primitive equations with a learned operator that takes past atmospheric states to future ones.

The numerical primitive Richardson actually wrote down at each grid cell — once per cell, once per six-hour timestep, six times across his test forecast — was a *finite-difference stencil*: a small fixed pattern of neighbouring grid values, multiplied by rational coefficients and summed to approximate a partial derivative. The simplest example, the one this curriculum builds in [its first notebook](notebooks/01_finite_differences_cfl.py), is the three-point second-derivative

```
u[i-1] − 2·u[i] + u[i+1]    ≈    Δx² · ∂²u/∂x²    + O(Δx⁴),
```

which reads three neighbours and approximates the second spatial derivative to second order. Stitch it together with a forward Euler step in time and you have **FTCS**, the workhorse Richardson rolled across central Europe and the same primitive that compresses time-stepping in roughly every classical PDE solver since. Richardson's actual stencils were 2D versions of the same idea, applied to the primitive equations of meteorology on a 200-km horizontal grid; the structure (read a small fixed neighbourhood, combine with hand-derived weights, write the result to the next timestep) is identical.

GraphCast does the same kind of operation — read from neighbours, combine with coefficients, write to self — except the neighbour set is a *learned graph* on an icosahedral mesh of the sphere, and the coefficients are *trained* from forty years of ERA5 reanalysis instead of derived from a Taylor expansion. The stencil generalises into a graph; the fixed integer coefficients into learned weights; the explicit time-step into a message-passing layer. The thread runs straight from a 1922 amphitheatre through ENIAC's vacuum tubes to a graph-neural-network layer on a sphere: same problem, same data origin, very different numerical machine — but at the level of the per-cell update, the two computations are kin.

This curriculum traces that intellectual lineage from Euler's 1768 forward step through finite elements, spectral methods, multigrid, and adjoints, into modern differentiable physics and neural operators — and on to the graph-based neural-operator architectures behind models like GraphCast. You will implement every method on a small problem, see why each one was invented, and understand what the modern PDE benchmarks (PDEBench, PDEArena, SciMLBenchmarks) are actually measuring — and where they intersect the adjacent benchmarks for **control theory and dynamical-systems identification** (e.g., [DynaDojo, NeurIPS 2023 D&B](https://proceedings.neurips.cc/paper_files/paper/2023/hash/32093649cbbcff773d9a991d8c30a7fe-Abstract-Datasets_and_Benchmarks.html)), which share most of the same numerical primitives.

> Modeled on OpenAI's [Spinning Up in RL](https://spinningup.openai.com/) and the author's [Spinning Up in Active Inference](https://github.com/m9h/spinning-up-alf). Companion to the [Hugging Face Science](https://huggingface.co/HuggingFaceScience) Discord, channel **#pde**, founded around the December 2025 blog post [_Why You Should Care About Partial Differential Equations (PDEs)_](https://huggingface.co/blog/hugging-science/pde) by Balaji, Chen, Nápoles, Sinha, and Ben Chaim. The channel meets weekly; this curriculum is the long-form educational companion to that founding post's case for PDE-as-SciML.

## How this fits with other resources

There are excellent neural-operator benchmarks already. This curriculum exists to give them a backstory.

|  | [PDEBench](https://github.com/pdebench/PDEBench) | [PDEArena](https://github.com/microsoft/pdearena) | [SciMLBenchmarks](https://github.com/SciML/SciMLBenchmarks.jl) | [DeepXDE](https://github.com/lululxvi/deepxde) | [DeepInverse](https://deepinv.github.io/) | **This curriculum** |
|---|---|---|---|---|---|---|
| **Classical PDE numerics** | -- | -- | Some | -- | -- | Modules 1-5 (FD, FV, FEM, spectral, multigrid) |
| **Adjoint & AD** | -- | -- | Implicit (Julia AD) | -- | Yes (gradients through forward op) | Module 6, with jaxctrl crosswalk |
| **Differentiable solvers** | -- | -- | -- | -- | -- | Module 7 (JAX-CFD, PhiFlow, jwave) |
| **PINNs** | -- | -- | -- | Yes | -- | Module 8 |
| **Neural operators** | FNO, U-Net | FNO, U-Net, GNO | -- | -- | -- | Modules 9-10 |
| **Standardised metrics** | Conservation, spectral RMSE | Trajectory rollout | Work-precision diagrams | -- | PSNR / SSIM / LPIPS for imaging IPs | Module 11 (uses all of the above) |
| **Inverse problems / FWI** | -- | -- | -- | -- | Yes — learned priors, plug-and-play, diffusion-based | Module 12 (brain-fwi, jwave, fijee leadfield) |
| **Historical narrative** | -- | -- | -- | -- | -- | Every module |
| **Primary language** | PyTorch | PyTorch | Julia | TF/PyTorch/JAX | PyTorch | JAX |
| **Format** | Datasets + baselines | Datasets + baselines | Benchmarks + WPDs | Solver framework | Inverse-problem framework | Curriculum + weekly sessions |

**The gap we fill:** PDEBench tells you which architecture wins on Burgers. It does not tell you what FTCS is, why CFL exists, or why the FEM community spent thirty years arguing about test functions. This curriculum is the "spinning-up" prequel: read it, run the notebooks, and the benchmark papers will read like the next chapter instead of an alien language.

## Where the work happens

The history of computational PDE solvers is also a history of specific institutions, and the modern SciML community is concentrated in a recognisable set of them. The curriculum cites them where their work is load-bearing; this section is the orienting map.

- **[Courant Institute (NYU)](https://cims.nyu.edu/)** — Founded by Richard Courant, who put the **C** in CFL (Module 01). The lineage runs from Friedrichs and Lewy through Peter Lax, Cathleen Morawetz, Leslie Greengard (FMM), and a generation of mathematical-PDE-numerics PhDs who staffed every other institution on this list. Still the gravitational centre for theoretical PDE numerics.
- **US Department of Energy national laboratories.** Each lab carries a distinct numerical tradition:
  - **[Los Alamos (LANL)](https://www.lanl.gov/)** — von Neumann's nuclear simulations; the institutional context in which his stability analysis (Module 01) matured. Modern: ALE methods, kinetic codes, FLAG, xRAGE.
  - **[Argonne (ANL)](https://www.anl.gov/)** — home of [PETSc](https://petsc.org/), the dominant scalable PDE-solver toolkit; also the [MPICH](https://www.mpich.org/) reference implementation.
  - **[Lawrence Berkeley (LBNL)](https://www.lbl.gov/)** — Phil Colella's adaptive-mesh-refinement lineage; [BoxLib / AMReX](https://amrex-codes.github.io/amrex/), [Chombo](https://commons.lbl.gov/display/chombo/Chombo+-+Software+for+Adaptive+Solutions+of+Partial+Differential+Equations).
  - **[Lawrence Livermore (LLNL)](https://www.llnl.gov/)** — [hypre](https://hypre.readthedocs.io/) (algebraic multigrid; the production analogue of Module 05), [MFEM](https://mfem.org/), [SUNDIALS](https://computing.llnl.gov/projects/sundials), and [libROM](https://www.librom.net/) (reduced-order modelling). LLNL's libROM team also hosts the [**DDPS** webinar series](https://www.librom.net/ddps.html) — see _Recurring seminars_ below.
  - **[Sandia](https://www.sandia.gov/)** — [Trilinos](https://trilinos.github.io/), [Kokkos](https://kokkos.org/) — the performance-portability stack underneath much of US HPC.
  - **[Oak Ridge (ORNL)](https://www.ornl.gov/)** — Jack Dongarra's group; [LAPACK / ScaLAPACK / MAGMA](https://www.icl.utk.edu/research/magma); the [Frontier exascale system](https://www.olcf.ornl.gov/frontier/), now the world's flagship machine for climate, fusion, and materials PDE workloads.
- **[Brown University, Division of Applied Mathematics](https://appliedmath.brown.edu/)** — Lax-Wendroff lineage; the modern home of [George Em Karniadakis's CRUNCH Group](https://www.brown.edu/research/projects/crunch/) (the [CRUNCH lecture archive](https://youtu.be/Md2-qh2OEkY) is in _Recurring seminars_). PINNs (Raissi-Perdikaris-Karniadakis 2019) and DeepONet (Lu et al. 2021) were both invented here, anchoring Modules 08-09.
- **[Caltech Computing + Mathematical Sciences](https://www.cms.caltech.edu/)** — One of the densest current concentrations of SciML. [Andrew Stuart](http://stuart.caltech.edu/) (operator learning, Bayesian inverse problems), [Houman Owhadi](http://users.cms.caltech.edu/~owhadi/index.html) (operator-valued kernels), [Anima Anandkumar](https://www.cms.caltech.edu/people/anima) (co-creator of FNO).
- **[University of Washington — AI Institute in Dynamic Systems](https://dynamicsai.org/)** — NSF-funded institute led by Steve Brunton, Nathan Kutz, and Bing Brunton. The home of the data-driven dynamical-systems lineage — [Dynamic Mode Decomposition](https://www.cambridge.org/core/books/dynamic-mode-decomposition/), [SINDy](https://www.pnas.org/doi/10.1073/pnas.1517384113) (Sparse Identification of Nonlinear Dynamics), and modern Koopman-operator approximations. The conceptual bridge from PDE numerics through dynamical-systems identification into control theory runs through this group's work.
- **[UT Austin Oden Institute](https://www.oden.utexas.edu/)** — Tinsley Oden's institutional legacy; PDE-constrained optimization, uncertainty quantification, scientific machine learning. Directed by Karen Willcox.
- **[ETH Zurich SAM (Seminar for Applied Mathematics)](https://math.ethz.ch/sam)** — [Christoph Schwab](https://www.sam.math.ethz.ch/people/personal-data.html?u=schwab) and [Ralf Hiptmair](https://www.sam.math.ethz.ch/people/personal-data.html?u=hiptmair) lead Europe's strongest concentration of FEM theory and UQ.
- **[Oxford Mathematical Institute, Numerical Analysis Group](https://www.maths.ox.ac.uk/groups/numerical-analysis)** — Lloyd N. Trefethen (now Harvard since 2023; Oxford archive [here](https://people.maths.ox.ac.uk/trefethen/)) and Endre Süli. The reading list for Modules 04+ leans heavily on the Oxford NA textbooks.
- **[MPI for Mathematics in the Sciences, Leipzig](https://www.mis.mpg.de/)** — Wolfgang Hackbusch's institute. Hackbusch invented multigrid (1976); the institutional lineage anchors Module 05.
- **[Heidelberg IWR (Interdisciplinary Center for Scientific Computing)](https://www.iwr.uni-heidelberg.de/)** — Peter Bastian's group; home of [DUNE](https://dune-project.org/).
- **[INRIA (France)](https://www.inria.fr/)** — National applied-math + HPC institute, multi-site. Olivier Pironneau (mesh adaptation, optimal control); birthplace of [FreeFEM](https://freefem.org/).
- **[TU Munich (TUM) — Munich Center for Computational Sciences](https://www.mcsc.tum.de/)** — Hans-Joachim Bungartz's sparse-grid lineage; current home of [JAX-Fluids](https://github.com/tumaer/JAXFLUIDS) and [PhiFlow](https://github.com/tum-pbs/PhiFlow).
- **[SIMULA Research Laboratory](https://www.simula.no/) (Oslo)** — Hans Petter Langtangen's legacy; long-time home of [FEniCS](https://fenicsproject.org/) and the broader Python finite-element ecosystem. The UFL variational form referenced in [Module 03](crosswalk.md#module-03--finite-elements--variational-forms) — `a = inner(a_sigma * grad(u), grad(v)) * dx` — is FEniCS syntax, written at SIMULA.
- **[Flatiron Institute — Center for Computational Mathematics (CCM)](https://www.simonsfoundation.org/flatiron/center-for-computational-mathematics/)** — Simons Foundation. Leslie Greengard (Fast Multipole Method); pure applied math at scale.
- **[Santa Fe Institute](https://www.santafe.edu/)** — Complex-systems and dynamical-systems framings of the SciML problem. The Crutchfield / West / Kauffman lineage informs the data-driven dynamics community.
- **[IPAM at UCLA](https://www.ipam.ucla.edu/)** — Long-program semesters where the SciML field's working consensus actually gets worked out: *Machine Learning for Physical Sciences* (2019), *Tensor Methods and Emerging Applications* (2021), ongoing PDE-and-learning workshops. Recordings on the [IPAM YouTube channel](https://www.youtube.com/c/IPAMUCLA).

For the longer survey — including Asian institutes (RIKEN, Tsinghua, NUS, ANU), additional European groups (Cambridge DAMTP, Imperial, EPFL, KTH, BCAM, ICTP), more US national-lab software, and the institutions behind Gmsh/CGAL/visualization tooling — see [`RESOURCES.md`](RESOURCES.md). Corrections and additions welcome via [Discussion on Hugging Face](https://huggingface.co/datasets/mhough/spinning-up-in-pde/discussions) or PR.

## Open-source PDE software

The curriculum's notebooks are JAX-native, but the broader open-source PDE ecosystem is most of what students will encounter in industry and at national labs. Modules 03–05 will look at production-grade C++ and Python codes alongside the teaching code, and a working scientific computing practitioner needs at least passing fluency with the following stacks. The longer annotated list lives in [`RESOURCES.md`](RESOURCES.md).

### FEM and multi-physics frameworks

- **[FEniCS](https://fenicsproject.org/)** — Python FEM with automatic weak-form compilation via UFL (the `tCS_model.ufl` example in Module 03 is FEniCS syntax). Originated at SIMULA / KTH.
- **[Firedrake](https://www.firedrakeproject.org/)** — Imperial's fork of FEniCS; tighter PETSc + adjoint integration via [pyadjoint](https://www.dolfin-adjoint.org/).
- **[deal.II](https://www.dealii.org/)** — C++ FEM. The reference for adaptive mesh refinement.
- **[NGSolve](https://ngsolve.org/)** — Joachim Schöberl's Vienna project. Strong in HDG, mixed methods.
- **[MOOSE](https://mooseframework.inl.gov/)** — Idaho National Lab's multi-physics framework. Built on libMesh; nuclear-reactor-scale problems.
- **[DUNE](https://dune-project.org/)** — Heidelberg IWR. The most flexible C++ PDE toolkit (modular FEM/FV/DG).
- **[FreeFEM](https://freefem.org/)** — INRIA/Sorbonne. DSL-driven FEM widely used in the French community.

### Finite-volume CFD and shock capturing

- **[OpenFOAM](https://www.openfoam.com/)** — the dominant open-source CFD package; industrial standard.
- **[SU2](https://su2code.github.io/)** — Stanford-origin; gradient-based shape and topology optimization via discrete adjoint.
- **[Clawpack](https://www.clawpack.org/)** — Randall LeVeque's lineage; the canonical reference codebase for hyperbolic PDEs and shock capturing. Companion to Module 02.
- **[Trixi.jl](https://trixi-framework.github.io/)** — Julia-native discontinuous Galerkin solver.

### Mesh generation and computational geometry

Mesh quality is half of every nontrivial PDE simulation. The curriculum's tetrahedral-mesh example in Module 03 ([Fijee tCS](https://github.com/m9h/Fijee-Project/blob/master/Fijee/Finite_element_method_models/tCS_model.ufl)) was generated with these tools.

- **[Gmsh](https://gmsh.info/)** — The standard open-source mesh generator. Built-in geometry kernel, OpenCASCADE backend, scriptable in Python and a custom DSL. Pairs natively with FEniCS, Firedrake, deal.II, MFEM.
- **[CGAL](https://www.cgal.org/)** — Computational Geometry Algorithms Library. C++ reference implementation for triangulations, mesh generation, surface reconstruction, and the algorithms beneath most modern meshers.
- **[Cubit / Coreform](https://coreform.com/)** — Sandia-origin commercial mesher; open-source releases via [Coreform Cubit](https://coreform.com/products/coreform-cubit/learn/).
- **[MeshPy](https://documen.tician.de/meshpy/)** / **[TetGen](https://www.tetgen.org/)** / **[Triangle](https://www.cs.cmu.edu/~quake/triangle.html)** — Python wrappers and underlying tetrahedral / 2D mesh generators.

### Scalable solver toolkits (already named under _Where the work happens_)

[PETSc](https://petsc.org/), [Trilinos](https://trilinos.github.io/), [Kokkos](https://kokkos.org/), [hypre](https://hypre.readthedocs.io/), [MFEM](https://mfem.org/), [SUNDIALS](https://computing.llnl.gov/projects/sundials), [AMReX](https://amrex-codes.github.io/amrex/), [MAGMA](https://www.icl.utk.edu/research/magma).

### Differentiable simulators (JAX-first)

- **[JAX-CFD](https://github.com/google/jax-cfd)** (Google) — pseudospectral and finite-volume Navier-Stokes.
- **[JAX-Fluids](https://github.com/tumaer/JAXFLUIDS)** (TUM) — differentiable compressible Navier-Stokes; shock capturing, multi-phase.
- **[PhiFlow](https://github.com/tum-pbs/PhiFlow)** (TUM) — differentiable physics with JAX/PyTorch/TF backends; tight ML integration.
- **[Diffrax](https://github.com/patrick-kidger/diffrax)** + **[Equinox](https://github.com/patrick-kidger/equinox)** (Patrick Kidger) — ODE/SDE/CDE solvers and JAX neural-network library; the substrate for time-evolved PDEs in JAX.
- **[jwave](https://github.com/m9h/jwave)** — differentiable acoustic FWI (Modules 04, 07, 12).

### Visualization

[VTK](https://vtk.org/), [ParaView](https://www.paraview.org/), and [VisIt](https://visit-dav.github.io/visit-website/) cover almost every PDE-data visualization need. ParaView is the typical first move for FEniCS / MFEM output.

## Recurring seminars and talks

If you want to keep the field at peripheral vision while you work through the curriculum, the following recurring seminars, webinar series, and self-paced courses are the easiest way in. Most have full archives going back several years.

- **[DDPS — Data-Driven Physical Simulations](https://www.librom.net/ddps.html)** (LLNL libROM team). Weekly webinar on machine learning + AI methods for computational science and physical simulation: deep learning for simulation, generative models, data assimilation, fluid dynamics, plasma physics. Recorded archive from 2020 onwards (e.g. [_The Nexus of Machine Learning, Physics-based Modeling, and Uncertainty Quantification_](https://youtu.be/0oWFK8Hlom8) is a representative DDPS talk on the ML / physics-modelling / UQ axis). Subscribe on the page. Organised by Youngsoo Choi and Siu Wun Cheung.
- **[CRUNCH Group lecture archive](https://youtu.be/Md2-qh2OEkY?si=HlP2f9Xk6ScGZT2q)** ([George Em Karniadakis](https://www.brown.edu/research/projects/crunch/), Division of Applied Mathematics, Brown). The CRUNCH Group invented PINNs (Raissi–Perdikaris–Karniadakis, 2019) and DeepONet (Lu et al., 2021); their YouTube channel is the primary lecture archive for those methods, the natural companion to Modules 08 and 09.
- **[12 steps to Navier-Stokes](https://github.com/barbagroup/CFDPython)** (Lorena Barba, GW Engineering). Twelve self-paced Python notebooks taking the reader from 1D linear convection to 2D incompressible Navier-Stokes. Genuinely the best free CFD onramp; pairs perfectly with Module 02. The associated [12 steps blog series](https://lorenabarba.com/blog/cfd-python-12-steps-to-navier-stokes/) gives the pedagogical backstory.
- **[Trefethen's Oxford NA video lectures](https://people.maths.ox.ac.uk/trefethen/lectures.html)** — Lloyd N. Trefethen's recorded lectures, principally from his Oxford NA group years (he moved to Harvard in 2023). The companion to his textbook *Spectral Methods in MATLAB* and the Chebfun tooling. Module 04 reading.

_(For the broader list — additional textbooks, MIT OCW, Stanford CME, IPAM workshop archives, and SIAM Visualization recordings — see [`RESOURCES.md`](RESOURCES.md). Suggestions welcome via Discord or PR.)_

## Who is this for?

- ML researchers entering SciML who want to understand what they're benchmarking against
- Computational scientists whose first move is FEniCS or PETSc, curious about JAX-native differentiable solvers
- Graduate students preparing to read [Karniadakis et al.](https://www.nature.com/articles/s42254-021-00314-5), [Li et al. (FNO)](https://arxiv.org/abs/2010.08895), or [Lu et al. (DeepONet)](https://arxiv.org/abs/1910.03193) and wanting context
- Anyone in the [Hugging Face Science](https://huggingface.co/HuggingFaceScience) Discord #pde channel who wants to follow along week by week

**Prerequisites:** Python, NumPy, ODE basics. Vector calculus helpful but reviewed as we go. No prior FEM or neural-operator experience required.

---

## The Curriculum

### Part 1 — Classical PDE Numerics

The 250-year backstory. What every modern surrogate model is compared against, and why those comparisons are the right ones.

| | Module | What you'll build | Anchored to |
|---|---|---|---|
| [01](articles/pde-series-part1-finite-differences.md) | **Finite Differences & CFL** | 1D heat equation in JAX, explicit FTCS vs Crank-Nicolson, watch the CFL boundary in real time. | PDEBench FTCS baseline |
| 02 | **Finite Volume & Conservation Laws** | 1D Burgers with Godunov / Roe / minmod limiters. Why shocks need conservative discretisation. | PDEBench compressible flow |
| 03 | **Finite Elements & Variational Forms** | Anisotropic Poisson on a tetrahedral mesh in FEniCSx, reading from a real `.ufl` file. | [`fijee/Finite_element_method_models/tCS_model.ufl`](https://github.com/m9h/Fijee-Project/blob/master/Fijee/Finite_element_method_models/tCS_model.ufl) |
| 04 | **Spectral & Pseudospectral Methods** | 2D Navier-Stokes with FFT-based pseudospectral, Orszag's 2/3 dealiasing rule. | [jwave](https://github.com/m9h/jwave) k-Wave-style propagation |
| 05 | **Multigrid & Preconditioners** | Full Multigrid V-cycles on a Poisson problem, comparison vs CG / PCG. | [`libspm/field.h`](https://github.com/m9h/libspm) — production FMG/CG in C |

### Part 2 — Differentiable Physics

Where adjoints meet automatic differentiation, and the simulator becomes a layer.

| | Module | What you'll build | Anchored to |
|---|---|---|---|
| 06 | **Adjoints & Automatic Differentiation** | Hand-derived adjoint of a 1D advection solver, then `jax.grad` through the same solver. The two answers had better match. | [jaxctrl](https://github.com/m9h/jaxctrl) (Lyapunov / Riccati adjoints) |
| 07 | **Differentiable Solvers in JAX** | Tour of JAX-CFD, PhiFlow, jwave: how a forward solver becomes a gradient operator. Train a learned closure on a coarse grid. | [jwave](https://github.com/m9h/jwave), [vpjax](https://github.com/m9h/vpjax), [vbjax](https://github.com/m9h/vbjax), [dot-jax](https://github.com/m9h/dot-jax) |

### Part 3 — Neural Operators

Surrogate models that learn maps between function spaces, not point values.

| | Module | What you'll build | Anchored to |
|---|---|---|---|
| 08 | **Physics-Informed Neural Networks** | A PINN for 1D Burgers from scratch in Equinox, then read [Raissi et al. 2019](https://www.sciencedirect.com/science/article/pii/S0021999118307125). Discuss: when do PINNs beat classical solvers, and when don't they? | [DeepXDE](https://github.com/lululxvi/deepxde) baseline comparison |
| 09 | **DeepONet & Fourier Neural Operators** | Reproduce a small FNO on Darcy flow. Spectral bias, resolution invariance, the operator-learning premise. | [neuraloperator](https://github.com/neuraloperator/neuraloperator) |
| 10 | **Geometric & Mesh-Aware Operators** | GraphNet-based operators (MeshGraphNets style) on irregular meshes. Why FNO's regular-grid assumption breaks for engineering problems. | [hgx](https://github.com/m9h/hgx) hypergraph operator overlap |

### Part 4 — Benchmarks, Inverse Problems, Discovery

What "good" looks like, and what real problems look like.

| | Module | What you'll build | Anchored to |
|---|---|---|---|
| 11 | **The Benchmark Landscape** | Run the same FNO on PDEBench Burgers, then on PDEArena Navier-Stokes, then on a SciMLBenchmarks WPD. Read the metric definitions: spectral RMSE, conservation RMSE, work-precision. Then look across at the adjacent dynamical-systems-identification benchmarks (DynaDojo) and the control-theory benchmarks ([jaxctrl](https://github.com/m9h/jaxctrl), Module 06) — the numerical primitives are shared even though the communities have grown apart. | [PDEBench](https://github.com/pdebench/PDEBench), [PDEArena](https://github.com/microsoft/pdearena), [SciMLBenchmarks](https://github.com/SciML/SciMLBenchmarks.jl), [DynaDojo](https://proceedings.neurips.cc/paper_files/paper/2023/hash/32093649cbbcff773d9a991d8c30a7fe-Abstract-Datasets_and_Benchmarks.html), [jaxctrl](https://github.com/m9h/jaxctrl) |
| 12 | **Inverse Problems & FWI** | Full-waveform inversion on a small acoustic test case in jwave; brief tour of brain-fwi (transcranial FWI) and the EEG forward-inverse problem (fijee leadfield → source localization). Compare classical regularised inversion against learned-prior approaches via [DeepInverse](https://deepinv.github.io/). | [brain-fwi](https://github.com/m9h/brain-fwi), [jwave](https://github.com/m9h/jwave), [Fijee-Project](https://github.com/m9h/Fijee-Project), [DeepInverse](https://deepinv.github.io/) |

A 13th module — **Agent-driven PDE discovery** — is planned but not scheduled, pending [agentsciml](https://github.com/m9h/agentsciml) maturity. The published reference is Jiang & Karniadakis (2025), [_AgenticSciML: Collaborative Multi-Agent Systems for Emergent Discovery in Scientific Machine Learning_](https://arxiv.org/abs/2511.07262), out of the [CRUNCH Group](https://www.brown.edu/research/projects/crunch/) at Brown — the same group whose lecture archive is cited in _Recurring seminars_.

---

## The Voice

Each module ships as both a notebook and a companion article. The articles follow the [Fedora Magazine](https://fedoramagazine.org/) cadence the author has used in the [linear-algebra series](https://github.com/m9h/fedora-linear-algebra-mag-series): a hook tied to a real-world artifact, the historical motivation (people, places, dates), the math, the code you can run today, and a forward link to the next idea in the sequence.

The companion notebooks are written in [jupytext](https://jupytext.readthedocs.io/) `.py` percent format for clean diffs, and convert to `.ipynb` with one command. See the [notebooks README](notebooks/) for the conversion recipe.

## Companion repositories

This curriculum is the front door to a constellation of JAX-native scientific-computing repositories by the same author. Each module names the relevant ones above; here is the full crosswalk:

| Repo | Role in the curriculum |
|---|---|
| [jaxctrl](https://github.com/m9h/jaxctrl) | Differentiable control (Lyapunov, Riccati, Gramians). Adjoint duality with PDE-constrained optimization. |
| [agentsciml](https://github.com/m9h/agentsciml) | Multi-agent evolutionary framework for SciML discovery. PDE-as-tool-use. |
| [jwave](https://github.com/m9h/jwave) | Differentiable acoustics in JAX, pseudospectral. Module 4, Module 7, Module 12. |
| [vpjax](https://github.com/m9h/vpjax) | Differentiable cerebrovascular models. Coupled hyperbolic / parabolic systems. |
| [vbjax](https://github.com/m9h/vbjax) | Virtual brain modelling — neural mass + integration. |
| [dot-jax](https://github.com/m9h/dot-jax) | Diffuse Optical Tomography (a parabolic forward, ill-posed inverse). |
| [brain-fwi](https://github.com/m9h/brain-fwi) | Full waveform inversion through the skull. Capstone-grade inverse problem. |
| [libspm](https://github.com/m9h/libspm) | Standalone C library for SPM's PDE solvers — Full Multigrid, B-splines, regularizers. Module 5. |
| [Fijee-Project](https://github.com/m9h/Fijee-Project) | FEM forward EEG (anisotropic Poisson) + Jansen-Rit/Wendling biophysics. Module 3, Module 12. |

See [crosswalk.md](crosswalk.md) for module-by-module pointers.

## Installation

This repository is mirrored on both Hugging Face and GitHub:

```bash
# From Hugging Face (primary; uses the `hf` CLI):
git clone https://huggingface.co/datasets/mhough/spinning-up-in-pde
cd spinning-up-in-pde

# Or from GitHub (mirror):
# git clone https://github.com/m9h/spinning-up-in-pde.git

# Create environment (requires uv: https://docs.astral.sh/uv/)
uv venv .venv --python 3.13
source .venv/bin/activate

# Core dependencies
uv pip install -r requirements.txt

# Convert jupytext .py notebooks to .ipynb
jupytext --to notebook notebooks/*.py

# Launch
jupyter lab
```

JAX install: on Linux/CUDA add `jax[cuda12]`; on Apple Silicon use the standard `jax` wheel (CPU) or `jax-mps` for MLX-backed acceleration on M-series hardware. See [JAX install docs](https://docs.jax.dev/en/latest/installation.html).

## Community

- **Discord:** [Hugging Face Science](https://huggingface.co/HuggingFaceScience), channel `#pde`
- **Weekly session:** one module per week, walkthrough + open Q&A. Time TBA in the channel.
- **Issues / PRs:** corrections, extra references, alternative implementations all welcome — especially historical citations we missed.

## License

Apache 2.0. Content is reusable; please cite the curriculum if you adapt it.
