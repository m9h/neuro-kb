---
type: method
title: Neural Field Models on Connectome Harmonics
category: spectral
implementations:
  - gnf-jax:gnfjax.laplacian_eigensystem
  - gnf-jax:gnfjax.graph_kernel
  - gnf-jax:gnfjax.harmonic_spectrum
  - gnf-jax:gnfjax.temporal_spectrum
  - gnf-jax:gnfjax.functional_connectivity
  - gnf-jax:gnfjax.spectral_loss
  - gnf-jax:gnfjax.simulate_nonlinear
  - vbjax:vbjax.make_shtdiff
  - vbjax:vbjax.make_sde
  - vbjax:vbjax.mpr_dfun
related:
  - connectome-harmonics.md
  - connectomics.md
  - neural-mass-models.md
  - method-spectral-analysis.md
  - physics-diffusion-equation.md
  - diffusion-mri.md
  - structural-mri.md
  - fmri.md
  - eeg.md
  - meg.md
  - method-sbi.md
  - method-monte-carlo.md
  - method-variational-inference.md
  - method-neural-ode.md
  - method-hmm-dynamics.md
  - method-hypergraph.md
  - method-pharmaco-eeg.md
  - jax-ecosystem.md
  - compute-infrastructure.md
description: "A neural field describes population activity as a continuous (or densely sampled) function over cortex, coupled through a spatial kernel rather than a sparse region-to-region matrix."
timestamp: 2026-11-05T15:59:53-08:00
tags: [method, dynamics, connectome, neural-field]
---

# Neural Field Models on Connectome Harmonics

A **neural field** describes population activity as a continuous (or densely sampled) function over cortex, coupled through a spatial kernel rather than a sparse region-to-region matrix. A **graph neural field** (GNF) replaces the continuous cortical sheet with a graph — typically a mesh-plus-tractography connectome — and applies the coupling kernel *spectrally*, on the eigenmodes ("connectome harmonics") of the graph Laplacian [@aqil2021graph]. Because the kernel is diagonal in the harmonic basis, the linearised field decouples into one small stochastic system per mode, which yields closed-form spatial and temporal power spectra, stability criteria, and functional-connectivity (FC) predictions [@aqil2021graph; @aqil2026brain].

This page covers the formulation shared by `gnf-jax` (stochastic Wilson–Cowan on graph harmonics) and `vbjax` (Montbrió–Pazó–Roxin fields with spherical-harmonic coupling) [@woodman2024insamuvbjax]. For the local population dynamics themselves see [neural-mass-models.md](neural-mass-models.md); for how the graph is built see [connectomics.md](connectomics.md).

## Connectome harmonics

Given a weighted adjacency `W` (e.g. cortical mesh edges plus streamline counts from [diffusion-mri.md](diffusion-mri.md)) with degree matrix `D`, `gnf-jax` uses the **negative combinatorial Laplacian**

```
L = -(D - W),   L U = U diag(λ),   λ_k ≤ 0
```

The eigenvectors `U` are the connectome harmonics; `|λ_k|` plays the role of squared spatial frequency. Small `|λ|` modes are smooth, large-scale patterns; large `|λ|` modes are spatially fine (gnf-jax:README.md). This sign convention means kernels written as `exp(t λ)` decay for high-frequency modes.

On a sphere, the analogous eigenfunctions of the Laplace–Beltrami operator are spherical harmonics `Y_l^m` with eigenvalue `−l(l+1)`. `vbjax.make_shtdiff` exploits this: local cortical coupling is applied by forward spherical-harmonic transform, spectral filtering up to degree `lmax`, and inverse transform [@woodman2024insamuvbjax]. Connectome harmonics generalise this to arbitrary (non-spherical, tractography-augmented) geometry.

## Graph kernels

A kernel `K(λ, t)` acts on a field `f` as `U diag(K(λ)) Uᵀ f`. The default in `gnf-jax` is the **heat kernel**

```
K(λ) = exp(t λ),   t = σ² / 2D
```

i.e. the Green's function of the graph diffusion equation evaluated at time `t` — a graph analogue of a Gaussian spatial kernel of width `σ` (see [physics-diffusion-equation.md](physics-diffusion-equation.md)). A `damped_wave_kernel` is also provided (gnf-jax:README.md). Each of the four Wilson–Cowan couplings (EE, IE, EI, II) has its own width `σ_xy`.

## Stochastic Wilson–Cowan field

The nonlinear field (gnf-jax:README.md, after [@aqil2021graph]):

```
τ_e dE/dt = −d_e E + S(α_ee K_ee E − α_ie K_ie I + P) + noise_e dW
τ_i dI/dt = −d_i I + S(α_ei K_ei E − α_ii K_ii I + Q) + noise_i dW
```

with logistic `S`. Linearising about a homogeneous steady state `(E*, I*)`, each harmonic `k` becomes an independent 2-state Ornstein–Uhlenbeck process `dβ_k = −A_k β_k dt + √D dW`:

```
A_k = [[ (d_e − a α_ee K_ee(λ_k)) / τ_e,    a α_ie K_ie(λ_k) / τ_e ],
       [ −b α_ei K_ei(λ_k) / τ_i,           (d_i + b α_ii K_ii(λ_k)) / τ_i ]]

a = d_e E*(1 − d_e E*),   b = d_i I*(1 − d_i I*)
D = diag((noise_e/τ_e)², (noise_i/τ_i)²)
```

`a` and `b` are logistic slopes at the operating point, so gain — and therefore stability — depends on where the steady state sits.

### Closed-form observables

| Quantity | Expression | `gnf-jax` function |
|---|---|---|
| Harmonic (spatial) power spectrum | Per-mode 2×2 Lyapunov solution `A S + S Aᵀ = D` | `harmonic_spectrum`, `mode_covariance` |
| Temporal power spectrum | `2 Σ_k S_k(ω)`, `S_k(ω) = (D_ee(A₁₁² + ω²) + D_ii A₀₁²) / |det(A_k + iω)|²` | `temporal_spectrum`, `model_temporal_spectrum` |
| Functional connectivity | Correlation matrix of `U diag(P) Uᵀ` (CHAOSS method) | `functional_connectivity` |
| Stability | Per-mode Jacobian eigenvalues; class code `2` = stable, oscillatory | `jacobian_eigenvalues`, `classify_steady_state` |
| Distance to instability | `−max Re(λ_J)` (> 0 means stable) | `distance_to_instability` |
| Long-range temporal correlations | Detrended fluctuation analysis exponent per mode | `dfa_exponent` |

The temporal spectrum is what [eeg.md](eeg.md)/[meg.md](meg.md) sensor spectra are compared against; the harmonic spectrum and FC are compared with [fmri.md](fmri.md) data projected onto harmonics [@aqil2026brain]. Background on spectral estimators is in [method-spectral-analysis.md](method-spectral-analysis.md).

**Deliberate deviation from the original code:** the original explicit E-spectrum formula assumes `noise_e = noise_i` (all original fits used 1). `gnf-jax` solves the general 2×2 Lyapunov equation, so it remains exact for unequal noise and agrees with the original when the amplitudes match (gnf-jax:README.md).

## Parameters

### gnf-jax reference parameter set (README usage example)

| Parameter | Symbol | Value | Meaning |
|---|---|---|---|
| `alpha_ee` | α_ee | 20 | E→E coupling strength |
| `alpha_ie` | α_ie | 15 | I→E coupling strength |
| `alpha_ei` | α_ei | 15 | E→I coupling strength |
| `alpha_ii` | α_ii | 5 | I→I coupling strength |
| `p` | P | −5 | Excitatory external drive |
| `q` | Q | −8 | Inhibitory external drive |
| `sigma_ee/ie/ei/ii` | σ | 2 (each) | Heat-kernel spatial widths |
| `tau_e` | τ_e | 1.0 | Excitatory time constant |
| `tau_i` | τ_i | 0.5 | Inhibitory time constant |
| Demo graph | — | 64-node chain | `grid_laplacian_eigensystem(64)` |
| Frequency grid | ω | 0–40, 801 points | Temporal spectrum evaluation |
| Linearised sim | dt, steps | 1e-3, 100 000 (↓10) | Euler–Maruyama |
| Nonlinear sim | dt, steps | 1e-3, 10 000 | Full field in vertex space |
| DFA windows | — | 16, 32, …, 512 | Per-mode scaling exponent |

Steady states are found by multi-start Newton with deduplication (`steady_states`). Enable `jax_enable_x64` — harmonic spectra span many decades and float32 loses the high-`|λ|` tail.

### vbjax spherical neural-field example

| Parameter | Value | Meaning |
|---|---|---|
| `lmax` | 16 | Max spherical-harmonic degree in coupling |
| `nlat × nlon` | 32 × 64 | Spatial grid |
| `dt` | 0.1 | SDE step |
| Additive noise | 0.2 | `gfun` in `make_sde` |
| MPR `eta` | −3.9 | Mean excitability |
| MPR `cr` | 5.0 | Coupling gain on rate variable |
| Steps | 500 | Simulation length |

Source: vbjax:README.md ("Simplest neural field"). Patterns form gradually despite noise because of the local spectral coupling [@woodman2024insamuvbjax].

## Fitting to data

`gnf-jax.spectral_loss` reproduces the original `Full_Analysis` objective: an affine map from model E and I spectra onto the empirical spectrum, a log-L2 distance plus a log-slope distance, the best *stable* steady state wins, and unstable parameter sets receive a large penalty. The README example discards the first two harmonics (`emp_hps[2:]`, `first_k=2`). The loss is a plain function of a `GNFParams` NamedTuple, so it works with `jax.grad` or with `scipy.optimize.differential_evolution` (the optimiser of the original fits) (gnf-jax:README.md).

`vbjax` supports inference over field states. In `tests/test_field_inference.py`, NumPyro NUTS (500 warmup, 500 samples) recovers the initial field `x0`. Pass criteria are ESS > 100, posterior shrinkage > 0.7, and |z-score| < 1.5. Runtime is about 5 min on GPU, 7 min on an M1 core, and 12 min on an x86-64 core (vbjax:README.md). See [method-monte-carlo.md](method-monte-carlo.md) and, for amortised alternatives, [method-sbi.md](method-sbi.md).

**Practical caveat:** gradient-based spectral fitting of mass/field models frequently stalls, with parameters not moving and identical fits across subjects. The vbjax autoresearch loop exists to evolve optimiser settings, free-parameter subsets, and initialisation for Liley (14 states), CMC, RRW and CBEI (8 states each) under a < 5 min budget on an RTX 2080 (vbjax:autoresearch/program.md). Analytic linearised spectra, as in GNF, give smoother losses than simulated ones and are the recommended starting point. They also support analytic fixed-point initialisation, which corresponds to that program's Q3.

## Performance

`vbjax` throughput for network SDEs (vbjax:README.md). The reference point is single-threaded C/C++ at 50–200 k iter/s for a 164-node (Destrieux) network.

| Hardware | Power | Throughput | Efficiency |
|---|---|---|---|
| Xeon W-2133 (Skylake, 14 nm) | 88 W | 5.7 Miter/s | 65 Kiter/W |
| Quadro RTX 5000 (Turing, 12 nm) | 200 W | 26 Miter/s | 130 Kiter/W |
| Apple M1 Air (5 nm) | 18 W | 3.7 Miter/s | 205 Kiter/W |

GNF analytic spectra cost one eigendecomposition (`O(N³)`, done once per connectome) plus `O(N)` 2×2 solves per parameter set, so they are vmappable across thousands of parameter draws. On Apple GPUs via `jax-metal`, roughly a third of vbjax tests fail because n-dim scatter/gather and FFTs are missing, which affects spectral-transform coupling. See [compute-infrastructure.md](compute-infrastructure.md).

## Validation (gnf-jax)

- Every analytic quantity is regression-tested against the original NumPy implementation (`tests/data/gnf_reference.npz`).
- The spectra are cross-checked three ways:
  - the Lyapunov identity;
  - Wiener–Khinchin integration of the temporal spectrum;
  - Euler–Maruyama simulation of both the linearised and the full nonlinear field.
- Not ported: connectome construction from raw mesh and tractography (`construct_fibers_from_data`, `construct_adjacency_matrix_from_data`). Build `W` externally and pass it to `laplacian_eigensystem`.

## Relationship to other methods

- **Region-based neural mass networks** ([neural-mass-models.md](neural-mass-models.md)) use a sparse parcellated connectome. GNF uses a dense vertex-level graph with spectral kernels. Both are implemented on vbjax's `make_sde`.
- **Hypergraph dynamics** ([method-hypergraph.md](method-hypergraph.md)) generalise the Laplacian to higher-order interactions. Hypergraph Laplacian eigenmodes are a natural extension of connectome harmonics.
- **Neural ODEs / differentiable simulators** ([method-neural-ode.md](method-neural-ode.md)): the GNF linearisation is a fixed, interpretable linear SDE per mode.
- **HMM state models** ([method-hmm-dynamics.md](method-hmm-dynamics.md)) describe discrete brain states. GNF describes stationary fluctuations around a single steady state.
- **Pharmaco-EEG spectral models** ([method-pharmaco-eeg.md](method-pharmaco-eeg.md)) can use GNF temporal spectra as a mechanistic forward model.

## Relevant Projects

- **`gnf-jax`**: JAX port of Aqil's Graph-Stochastic-Wilson-Cowan-Model (MIT). It covers harmonic and temporal spectra, CHAOSS FC, stability, simulation, DFA and `spectral_loss` [@aqil2021graph; @aqil2026brain].
- **`vbjax`**: virtual-brain SDE toolkit. It provides MPR mass model, spherical-harmonic local coupling (`make_shtdiff`), NumPyro field inference, and parameter sweeps via `vmap`/`pmap` [@woodman2024insamuvbjax].
- **`hgx`**, **`jaxctrl`**: hypergraph Laplacians and controllability of graph dynamics. Natural consumers of the per-mode `A_k` systems.
- **`inferjax`**: amortised SBI over `GNFParams` as a faster alternative to differential evolution.
- **`neurojax`**: EEG/MEG forward models that map vertex-level field activity to sensors ([method-source-imaging.md](method-source-imaging.md)).

## See Also

- [Connectome Harmonics](connectome-harmonics.md)
- [connectomics.md](connectomics.md): building `W` from tractography
- [neural-mass-models.md](neural-mass-models.md): local E–I and MPR dynamics
- [physics-diffusion-equation.md](physics-diffusion-equation.md): heat kernel origin
- [method-spectral-analysis.md](method-spectral-analysis.md): empirical spectra and DFA
- [diffusion-mri.md](diffusion-mri.md), [structural-mri.md](structural-mri.md): connectome and mesh inputs
- [eeg.md](eeg.md), [meg.md](meg.md), [fmri.md](fmri.md): target data modalities
- [method-sbi.md](method-sbi.md), [method-monte-carlo.md](method-monte-carlo.md), [method-variational-inference.md](method-variational-inference.md): parameter inference
- [jax-ecosystem.md](jax-ecosystem.md): jit/vmap/grad patterns used throughout
