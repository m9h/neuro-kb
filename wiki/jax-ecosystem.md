---
type: concept
title: JAX Ecosystem for Neuroimaging
related: [method-sbi.md, method-fem.md, method-bem.md, method-source-imaging.md, method-neural-ode.md, method-variational-inference.md, method-monte-carlo.md, diffusion-mri.md, quantitative-mri.md, mrs.md, physics-acoustic.md, tus.md, neural-mass-models.md, method-neural-fields.md, method-hypergraph.md, method-active-inference.md, method-hmm-dynamics.md, compute-infrastructure.md, sci-head-model.md]
description: "The neuro-kb project family shares one numerical substrate, JAX, plus the 'Kidger stack' (Equinox, Diffrax, Optimistix, Lineax) and a small set of probabilistic and physics libraries."
timestamp: 2026-11-05T15:59:53-08:00
tags: [concept, jax, tooling]
---

# JAX Ecosystem for Neuroimaging

The neuro-kb project family shares one numerical substrate, **JAX**, plus the "Kidger stack" (Equinox, Diffrax, Optimistix, Lineax) and a small set of probabilistic and physics libraries. The common goal is a single computational graph in which `jax.grad` runs from a loss on sensor data back to biophysical parameters. Those parameters include tissue conductivity, skull sound speed, axon radius, and neural-mass coupling. This page lists which libraries each project uses, the conventions they share, and the measured performance reported in project READMEs.

> Design principle (sbi4dwi): **"build where differentiable, wrap where not."** JAX models handle gradient-based optimisation. Non-differentiable external simulators produce training data at an HDF5 boundary (`sbi4dwi:CLAUDE.md`).

## Core Stack

| Layer | Library | Role | Used by (documented) |
|---|---|---|---|
| Arrays & autodiff | JAX | XLA compilation, `grad`/`vmap`/`jit` | all projects below |
| Modules | Equinox (`eqx.Module`) | Immutable pytree models and networks | sbi4dwi, neurojax, hgx |
| Optimisation | Optax / Optimistix | Stochastic training / deterministic voxel fitting | sbi4dwi (both), neurojax (Optax) |
| ODE/SDE | Diffrax | Bloch, diffusion SDE, neural ODE/SDE | sbi4dwi, neurojax |
| Linear algebra | Lineax | GLM, beamforming solves | neurojax |
| Graphs | jraph | PI-GNN on cortical meshes | neurojax |
| Normalizing flows | FlowJAX | Neural posterior estimation | sbi4dwi |
| MCMC | BlackJAX | NUTS posteriors | sbi4dwi |
| Probabilistic programming | NumPyro | Variational inference, model discovery | sbi4dwi (`bayesian/`) |
| Equivariance | e3nn-jax | E(3)-equivariant score networks | sbi4dwi |
| Tensor decomposition | ttax | Tensor-train surrogates | sbi4dwi |
| Acoustics | j-Wave [@stanziola2023jwave] | Pseudospectral (PSTD) + Helmholtz wave solver | sbi4dwi, brain-fwi |
| FEM | JAX-FEM + PETSc | Differentiable bioelectric FEM | neurojax |
| BEM (external) | OpenMEEG, MNE-Python | 3-layer BEM leadfields | neurojax |
| I/O | nibabel, h5py, meshio, DIPY | NIfTI, HDF5 libraries, meshes, gradient tables | sbi4dwi, neurojax |

Sources: `sbi4dwi:README.md` (Tech Stack), `neurojax:README.md` (The Stack), `neurojax:CLAUDE.md`.

## Shared Conventions

Most of these conventions are written in the per-project `CLAUDE.md` files. They apply across repos.

- **uv only.** Use `uv sync`, `uv add`, and `uv run pytest`. Never use pip, conda, or bare `python` (sbi4dwi, neurojax).
- **Immutable pytrees.** Never mutate `eqx.Module` fields; use `eqx.tree_at`. Plain Python classes are reserved for orchestration objects such as `ModelSimulator`, `ComparisonRunner`, and `LibraryGenerator` (sbi4dwi).
- **SI units internally.**
  - b-values are in s/m². DIPY and FSL use s/mm², so divide by 1e6.
  - Free-water diffusivity is 2.0e-9 m²/s.
  - Lengths are in metres; a 5 µm radius is 5e-6.
- **Train/deploy symmetry.** b0 normalisation must be identical in `ModelSimulator.sample_and_simulate()` and `SBIPredictor.predict_volume()`.
- **Explicit regularisation.** Use Tikhonov regularisation, not truncated SVD with a hidden rank cutoff, and always report the λ used. Check memory with `psutil` before large matrix operations (neurojax).
- **Red-green TDD.** neurojax has 120+ tests, for example 26 for PI-GNN, 22 for qMRI, and 9 for LAURA.
- Use `np.trapezoid`, not `np.trapz`, which was removed in NumPy 2.x.

## Pattern 1 — Differentiable Forward Models as `eqx.Module`

sbi4dwi reimplements the dmipy compartments as Equinox modules [@fick2019dmipy]:

- **Ball/Stick/Zeppelin/Tensor:** Gaussian compartments.
- **Sphere/Cylinder/Plane:** restricted diffusion, using the GPD, Callaghan, Söderman and Stejskal–Tanner models.
- **Extended models:**
  - NODDI [@zhang2012noddi] and Multi-TE NODDI [@gong2020mte]
  - SANDI [@palombo2020sandi]
  - IVIM, free-water DTI, mcDESPOT, EPG, QMT
  - stimulated-echo Kärger exchange [@karger1985nmr]

Models compose via `compose_models()` with Watson/Bingham dispersion and tortuosity constraints. See [diffusion-mri.md](diffusion-mri.md) and [quantitative-mri.md](quantitative-mri.md).

```python
model = compose_models([C1Stick(), G1Ball()])
params = fit_voxel(model, acquisition, signal)   # Optimistix, vmapped over voxels
```

Differentiable simulators extend the analytical models:

- **`MatrixFormalismSimulator`:** FEM spectral reduced-order model on triangular meshes. It solves the generalised eigenproblem S v = λ M v and builds PGSE from 3 matrix exponentials. It switches to BCOO sparse matrices above about 2K vertices.
- **`DifferentiableWalker`:** confined Brownian motion.
- **`SDESimulator`:** SDE integration with Diffrax.

A non-differentiable `MonteCarloSimulator` with SDF geometry serves as ground truth. See [method-fem.md](method-fem.md), [method-monte-carlo.md](method-monte-carlo.md) and [physics-diffusion-equation.md](physics-diffusion-equation.md).

## Pattern 2 — Amortized Simulation-Based Inference

The pipeline runs `ModelSimulator → train_sbi() → checkpoint (.eqx + .config.json) → SBIPredictor.predict_volume()`. It generates `(θ, signal)` pairs from prior → forward → noise. Supported estimators are MDN, FlowJAX flows, score-based diffusion, BlackJAX NUTS, and amortized VI. See [method-sbi.md](method-sbi.md) and [method-variational-inference.md](method-variational-inference.md).

### Ball+2Stick NPE configuration (sbi4dwi `autoresearch/program.md`)

| Parameter | Value |
|---|---|
| Flow | Masked autoregressive, `RationalQuadraticSpline(knots=8, interval=3.0)` |
| Layers / hidden | 10 layers / 128 units (2-layer MLPs) |
| Optimiser | Adam, lr 3e-4, cosine decay, batch 512 |
| Noise | Rician inside `sim_fn`; SNR 10–50 for training, 30 for evaluation |
| Acquisition | HCP-like 90 directions, b = 0/1000/2000/3000 s/mm² |
| Aggregation | 500 posterior samples, then clip, unit-normalise, and take the median |
| Symmetry breaking | f1 ≥ f2 ordering; hemisphere z ≥ 0 |

### Measured accuracy

| Estimator | Fiber 1 median error | d_stick r | f1 r | Steps |
|---|---|---|---|---|
| Affine flow baseline | 10.7° | 0.95 | 0.85 | 30k |
| Spline + noise fix + label switching | 6.4° | 0.978 | 0.902 | 30k |
| Spline flow | 3.2° | 0.986 | 0.935 | 200k |
| Spline flow | **2.8°** | **0.987** | **0.943** | 300k |
| Score-based (DDPM, spherical coords) | 12.8° | — | — | 100k |
| MDN (10 components) | ~5–8° | — | — | 30k |

The orientation targets follow Manzano-Patron et al. [@manzanopatron2025sbi]. For the score-based estimator, the SDE sampler diverged (29°) and DDPM sampling was required (15.5°).

### Supporting modules

- **UQ:** SBC, PPC, conformal prediction, OOD detection, and ensembles, all in `pipeline/`.
- **Multi-fidelity training:** `HybridLibraryGenerator` mixes 70% analytical and 30% oracle data. The oracles are DIPY, ReMiDi (Docker), and MCMRSimulator.jl (Julia subprocess). `OracleModelSimulator` maps these into the training pipeline via k-NN inverse-distance interpolation or an MLP emulator.

## Pattern 3 — End-to-End Differentiable Physics

### Electromagnetic chain (neurojax)

The chain runs: qMRI (T1, QMT bound-pool fraction) → `sigma_from_qmri` → `assemble_stiffness` (FEM) → leadfield L → Y = L·J + n → inverse solver. Gradients flow through the entire chain:

```python
sigma = sigma_from_qmri(t1_values, bpf_values, labels, params)
K = assemble_stiffness(vertices, elements, sigma)
grad_params = jax.grad(loss)(params)
```

- **Head models:** CHARM 60-tissue segmentation, 3-layer BEM (MNE/OpenMEEG), and differentiable FEM.
- **15 inverse solvers in five families:**
  - minimum norm: MNE, dSPM, sLORETA, eLORETA
  - biophysical prior: LAURA, VARETA
  - beamformers: LCMV, SAM, DICS
  - Bayesian sparse: CHAMPAGNE, HIGGS
  - deep learning: PI-GNN on jraph
- **Validation data:** the WAND dataset (170 subjects, CTF 275-channel MEG, 300 mT/m Connectom).

See [method-source-imaging.md](method-source-imaging.md), [method-bem.md](method-bem.md), [meg.md](meg.md), [eeg.md](eeg.md) and [tissue-electrical-properties.md](tissue-electrical-properties.md).

### Acoustic chain (sbi4dwi → brain-fwi)

j-Wave simulations ran on the [sci-head-model.md](sci-head-model.md) (208×256×256, 1 mm, 8 tissues) on a Modal A100. Tissue properties follow ITRUSST benchmark values [@aubry2022itrusst]. See [tus.md](tus.md), [physics-acoustic.md](physics-acoustic.md) and [tissue-skull.md](tissue-skull.md).

| Experiment | Result |
|---|---|
| 2D skull attenuation, 400 kHz | 93.0% |
| Frequency sweep (slab), 180 kHz / 400 kHz / 1 MHz | 32.0% / 55.1% / 34.3% loss |
| 32-element delay optimisation, 2D, 20 iterations | 1.4× focal gain, 1.8 s/iter |
| 16-element delay optimisation, 3D 4 mm grid | 4.7× focal gain, 2.6 s/iter |
| Sensitivity \|dp/dc\|, skull vs water | 232× |
| Helmholtz (GMRES) vs PSTD | r = 0.82 (p_max 0.287 vs 0.311) |
| 3-band FWI, 50→600 kHz, 256 elements | skull c recovered 4312 m/s (true 4080), 222 s |
| 3D USCT, 96³ at 2 mm, 128 elements (DGX Spark) | brain RMSE 52.7 m/s, skull RMSE 1255.6 m/s, 13.3 min |

Caveats from the source (`sbi4dwi:docs/openlifu/README.md`):

- The multi-target runs (53–55%) used a slab geometry, so they are not directly comparable with the 93% full-head figure.
- The 3D 2 mm run gave an anomalous −7.6% attenuation, flagged as "needs investigation."
- The README calls the suite "7-experiment" while the OpenLIFU doc reports "9/9 experiments."

The planned extension chains j-Wave → radiation force F = 2αI/c → elastodynamics µ∇²u + F = ρü → Bloch-simulated MR-ARFI phase. It would use brain shear moduli from CANNs [@linka2023cann] and MR-ARFI encoding [@kaye2011mrarfi]. See [physics-bloch.md](physics-bloch.md).

The 256-element helmet reference [@martin2025tfus] has a title mismatch between sources. The bib entry reads "MRI-guided transcranial focused ultrasound neuromodulation with a 256-element helmet array". The sbi4dwi docs cite "Ultrasound system for precise neuromodulation of human deep brain circuits", *Nat. Commun.* 16:8024. Verify before citing.

## Pattern 4 — Dynamics, Control and Inference Models

These projects have no library-level dependency list in the current source material. Entries are taken from the schema project table.

- **vbjax:** whole-brain neural mass simulation on connectomes. neurojax ships a `bench/` vbjax adapter. See [neural-mass-models.md](neural-mass-models.md) and [connectomics.md](connectomics.md).
- **hnn-jax:** neocortical-column model producing MEG/EEG dipoles.
- **gnf-jax:** stochastic Wilson–Cowan dynamics on connectome harmonics. See [method-neural-fields.md](method-neural-fields.md).
- **hgx:** hypergraph neural networks in Equinox. See [method-hypergraph.md](method-hypergraph.md).
- **jaxctrl:** differentiable controllability and Lyapunov analysis.
- **alf:** active-inference agents [@friston2022active; @smith2022step]. See [method-active-inference.md](method-active-inference.md).
- **inferjax:** amortized SBI and posterior calibration for any simulator. It generalises the sbi4dwi pipeline above.
- **vpjax:** cerebrovascular and metabolic models, pharmacokinetics, and identifiability. See [physics-cerebrovascular.md](physics-cerebrovascular.md) and [method-pharmaco-eeg.md](method-pharmaco-eeg.md).
- **mrs-jax:** MEGA-PRESS and HERMES [@chan2016hermes] quantification. See [mrs.md](mrs.md).
- **setae:** contact mechanics and FEM for soft tissue.
- **neurojax `models/`:** HMM and DyNeMo brain-state models. See [method-hmm-dynamics.md](method-hmm-dynamics.md) and [method-neural-ode.md](method-neural-ode.md).

## Compute Targets

- **Modal A100:** used for TUS, FWI and SBI experiments, launched with `modal run scripts/...`.
- **DGX Spark:** local GPU, used for 3D USCT.
- **CUDA 13+:** required for sbi4dwi GPU support.

See [compute-infrastructure.md](compute-infrastructure.md).

## Relevant Projects

| Project | JAX role |
|---|---|
| `sbi4dwi` | Equinox signal models, FlowJAX/BlackJAX SBI, Diffrax SDEs, j-Wave TUS |
| `neurojax` | Differentiable BEM/FEM, 15 inverse solvers, qMRI, MRS, HMM/DyNeMo |
| `brain-fwi` | 3D multi-frequency ultrasound FWI (sigmoid-bounded velocity) |
| `vbjax`, `hnn-jax`, `gnf-jax` | Neural mass / neural field forward models |
| `hgx`, `jaxctrl` | Hypergraph networks, network control |
| `alf`, `inferjax`, `PGMax` | Active inference, amortized SBI, factor graphs |
| `vpjax`, `mrs-jax`, `setae` | Physiology, spectroscopy, mechanics |

## See Also

- [method-sbi.md](method-sbi.md)
- [method-fem.md](method-fem.md)
- [method-bem.md](method-bem.md)
- [method-source-imaging.md](method-source-imaging.md)
- [method-neural-ode.md](method-neural-ode.md)
- [method-variational-inference.md](method-variational-inference.md)
- [diffusion-mri.md](diffusion-mri.md)
- [quantitative-mri.md](quantitative-mri.md)
- [tus.md](tus.md)
- [physics-acoustic.md](physics-acoustic.md)
- [tissue-acoustic-properties.md](tissue-acoustic-properties.md)
- [neural-mass-models.md](neural-mass-models.md)
- [data-formats.md](data-formats.md)
- [compute-infrastructure.md](compute-infrastructure.md)
