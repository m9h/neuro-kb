---
type: method
title: Neural Mass Models
category: differentiable
implementations: [vbjax:mpr_dfun, vbjax:make_sde, vbjax:autoresearch/program.md, neurojax:bench/models/rww.py, neurojax:bench, gnf-jax:gnfjax, hnn-jax:src/hnnjax/population.py, vc2008-netpyne:jaxmodel]
related: [method-neural-fields.md, method-neural-ode.md, connectomics.md, eeg.md, meg.md, fmri.md, physics-electromagnetic.md, physics-hemodynamic.md, method-spectral-analysis.md, method-source-imaging.md, method-sbi.md, method-hmm-dynamics.md, method-pharmaco-eeg.md, method-active-inference.md, method-hypergraph.md, jax-ecosystem.md, tms.md, diffusion-mri.md, connectome-harmonics.md, in-vitro-mea-cultures.md]
description: "Neural mass models (NMMs) describe the collective activity of large neuronal populations (~10^4–10^7 neurons) with low-dimensional ODEs/SDEs."
timestamp: 2026-11-05T15:59:53-08:00
tags: [method, dynamics, neural-mass, simulation]
---

# Neural Mass Models

Neural mass models (NMMs) describe the collective activity of large neuronal populations (~10^4–10^7 neurons) with low-dimensional ODEs/SDEs. Each node is a cortical column or parcellated region with 2–14 state variables. Networks of these nodes are coupled through a structural connectome ([connectomics.md](connectomics.md)). They are then mapped to EEG/MEG through a leadfield ([physics-electromagnetic.md](physics-electromagnetic.md)) or to BOLD through a hemodynamic model ([physics-hemodynamic.md](physics-hemodynamic.md)).

The workspace covers several levels of biological detail:

- **Phenomenological/mean-field NMMs**: vbjax, neurojax.
- **Spatially continuous fields on graph eigenmodes**: gnf-jax.
- **Biophysical column models that NMMs approximate**: hnn-jax, vc2008-netpyne.

## Model Families

| Model | States/node | Variables | Key feature | Workspace implementation |
|-------|-------------|-----------|-------------|--------------------------|
| Wilson-Cowan | 2 | E, I rates | Sigmoid E–I rate model | gnf-jax (graph field form) |
| Jansen-Rit (JR) | 6 | 3 PSPs + derivatives | ~10 Hz alpha; EEG = y1 − y2 | vbjax `jr_dfun` |
| Wendling | 8–10 | JR + fast somatic inhibition | Epileptiform/gamma activity | neurojax heritage (LTSI Rennes, Fijee) |
| Montbrio-Pazo-Roxin (MPR) | 2 | r, V | Exact mean field of QIF/theta neurons | vbjax `mpr_dfun` |
| Canonical Microcircuit (CMC) | 8 | ss, sp, ii, dp (×2) | Laminar; predictive coding | vbjax autoresearch |
| Robinson corticothalamic (RRW) | 8 | cortex–thalamus loop | Alpha/beta via CT delays | vbjax autoresearch |
| CBEI (next-gen E-I) | 8 | exact mean-field E–I | QIF-derived E–I pair | vbjax autoresearch |
| Liley | 14 | conductance-based, long-range axonal | Used as ground-truth generator | vbjax autoresearch |
| Reduced Wong-Wang (RWW) | 1–2 | S_E (, S_I) gating | BOLD/resting-state FC | neurojax `bench/models/rww.py` |
| Epileptor (BVEP) | 2 | x, z | Seizure onset/offset | vbjax `bvep_dfun` |

State counts for Liley, CMC, RRW and CBEI follow the vbjax spectral-fitting table (`vbjax:autoresearch/program.md`).

## Jansen-Rit

The JR model contains three populations: pyramidal cells, excitatory interneurons and inhibitory interneurons. Each population uses a second-order PSP kernel, and a sigmoid converts membrane potential to firing rate.

```
S(v) = 2 e0 / (1 + exp(r (v0 − v)))
dy0/dt = y3;  dy3/dt = A a S(y1 − y2)              − 2a y3 − a² y0
dy1/dt = y4;  dy4/dt = A a (p(t) + C2 S(C1 y0))    − 2a y4 − a² y1
dy2/dt = y5;  dy5/dt = B b C4 S(C3 y0)             − 2b y5 − b² y2
```

| Parameter | Symbol | Value | Units |
|-----------|--------|-------|-------|
| Max firing rate / 2 | e0 | 2.5 | s⁻¹ |
| Sigmoid midpoint | v0 | 6 | mV |
| Sigmoid slope | r | 0.56 | mV⁻¹ |
| Excitatory gain | A | 3.25 | mV |
| Inhibitory gain | B | 22.0 | mV |
| Excitatory rate constant | a | 100 (τ = 10 ms) | s⁻¹ |
| Inhibitory rate constant | b | 50 (τ = 20 ms) | s⁻¹ |
| Connectivity | C1, C2, C3, C4 | 135, 0.8·C1, 0.25·C1, 0.25·C1 | – |
| Input rate | p | ~120–320 | s⁻¹ |

The external input p is the main bifurcation parameter:

- Within roughly 120–320 s⁻¹, a Hopf-born limit cycle produces ~10 Hz alpha.
- At the edges of this range, bistable regions produce spike-wave-like transients. This is why JR and its Wendling extension are standard seizure models.
- The EEG observable is `y1 − y2`.

## Montbrio-Pazo-Roxin (MPR)

MPR is the exact Ott–Antonsen reduction of an infinite all-to-all network of quadratic integrate-and-fire (QIF) neurons. The neurons have Lorentzian-distributed excitabilities, so no phenomenological sigmoid is needed.

```
τ dr/dt = Δ/(π τ) + 2 r V
τ dV/dt = V² + η + J τ r − (π τ r)² + I + coupling
```

| Parameter | Symbol | vbjax default (`mpr_default_theta`) |
|-----------|--------|--------------------------------------|
| Time constant | tau | 1.0 |
| External current | I | 0.0 |
| Excitability spread | Delta | 1.0 |
| Recurrent weight | J | 15.0 |
| Mean excitability | eta | −5.0 |
| Rate-coupling gain | cr | 1.0 |
| Voltage-coupling gain | cv | 0.0 |

The vbjax README neural-field example uses `eta = −3.9, cr = 5.0`. Its parameter sweep explores η ∈ [−6, −4] and log k ∈ [−9, −2] with noise σ ∈ {0, 0.2, 0.3, 0.4} [@woodman2024insamuvbjax].

**Correction to an earlier version of this page.** The 2-D MPR system with instantaneous synapses has no stable limit cycle. As η varies, two saddle-node bifurcations bound a bistable region. In that region, a low-activity node coexists with a high-activity focus that shows damped oscillations. Sustained oscillations require synaptic kinetics, delays, or noise-driven excitation of the focus. An earlier version described a SNIC bifurcation into a limit cycle; that statement has been removed.

Because QIF neurons are equivalent to theta neurons, MPR is also the mean-field limit of the theta-neuron network in `vc2008-netpyne/jaxmodel` (see "Spiking-network ground truth" below).

## Graph Neural Fields (gnf-jax)

gnf-jax places a stochastic Wilson-Cowan model on the eigenmodes of a negative combinatorial graph Laplacian `L = −(D − W)`, with eigenvalues λ_k ≤ 0 [@aqil2021graph; @aqil2026brain]. The field equations are:

```
τ_e dE/dt = −d_e E + S(α_ee K_ee E − α_ie K_ie I + P) + noise_e dW
τ_i dI/dt = −d_i I + S(α_ei K_ei E − α_ii K_ii I + Q) + noise_i dW
```

- **Kernel.** The default heat kernel is `K(λ, t) = exp(t λ)` with `t = σ²/2D`.
- **Linearised dynamics.** Around a steady state, each mode is an independent 2-state Ornstein–Uhlenbeck process.
- **Harmonic spectrum.** Solved as the 2×2 Lyapunov equation `A S + S Aᵀ = D`. This stays exact for unequal E/I noise; the original code assumed equal noise.
- **Temporal spectrum.** Given by `2 Σ_k S_k(ω)`.
- **Functional connectivity.** Predicted by the CHAOSS method, `corr(U diag(P) Uᵀ)`.
- **Fitting.** The loss is a log-L2 distance plus a log-slope distance. It is compatible with `jax.grad` and `scipy.optimize.differential_evolution`, and unstable parameter sets are penalised.

Example parameters from the gnf-jax README:

| Parameter | Value |
|-----------|-------|
| α_ee, α_ie, α_ei, α_ii | 20, 15, 15, 5 |
| P, Q | −5, −8 |
| σ (all four kernels) | 2 |
| τ_e, τ_i | 1.0, 0.5 |

Use 64-bit mode for spectra that span many decades. See [method-neural-fields.md](method-neural-fields.md) for the broader field-model context. vbjax offers a related approach: MPR on the sphere with spherical-harmonic local coupling (`make_shtdiff`, lmax = 16, 32×64 grid).

## Biophysical Column Models (hnn-jax)

hnn-jax is a differentiable Diffrax reimplementation of the Human Neocortical Neurosolver. It maps drive parameters θ to a current dipole(t) that underlies ERP components such as auditory P1/N1/P2.

- **v0 (mean-field):** correlation with the `hnn_core` oracle was +0.22.
- **v1 (population):** N cells, each with a reparameterized drive time `t_i = μ + σ ε_i`, vmapped and then averaged. Correlation rose to +0.43, and the CI floor is set at `FORWARD_CORR_MIN = 0.40`.

The gain from v0 to v1 shows the core limitation of a mean-field closure: averaging cell *responses* is not the same as the response to the *average* drive. Only the population version carries a gradient through `tc_sync`, the thalamocortical synchrony parameter.

Oracle reproducibility caveat: `hnn_core` 0.6.1 and 0.3 produce dipoles of the same scale but different waveforms. The RMS ratio is 0.76–1.13, while waveform correlation is only 0.62 (range 0.45–0.81). The project therefore pins `hnn_core ≥ 0.6.0`.

## Spiking-Network Ground Truth (vc2008-netpyne)

vc2008-netpyne is a NetPyNE re-implementation of the Vierling-Claassen et al. auditory-entrainment model [@vierlingclaassen2008modeling]. It is a reference for which oscillatory phenomena a mean-field model must reproduce.

Network structure:

- 240 two-compartment HH cells: 160 pyramidal, 40 basket, 40 chandelier.
- 4 Hz Poisson background input.
- 20/30/40 Hz click-train drive.
- MEG proxy = mean EPSC onto pyramidal cells.

The only manipulation is the chandelier IPSC decay, τ2,ch = 8 ms (control) → 25 ms (schizophrenia). Results from 600 simulations:

| Measure | Control | SZ |
|---------|---------|----|
| P40 under 40 Hz drive (evoked) | 83.9 | 8.7 (~9.7× lower) |
| P20 under 20 Hz drive | 59.6 | 93.6 |
| P20 under 40 Hz drive (total power) | 0.03 | 0.40 (13×) |

The 20 Hz subharmonic arises from pulse skipping with random phase. It therefore cancels in evoked averages, so total and evoked spectra must be distinguished when fitting NMMs to ASSR data.

The JAX theta-neuron version (20 E + 10 I) is fully differentiable. Its network phase response inherits Type I excitability, whereas the HH version is Type II and gives the opposite alignment. This choice of single-neuron model is invisible to rate models that use a sigmoid.

## Coupling on Connectomes

The coupling input to node i is:

```
I_i(t) = k Σ_j W_ij G(x_j(t − d_ij)),    d_ij = L_ij / v
```

| Parameter | Typical value | Notes |
|-----------|---------------|-------|
| Global coupling k | e^−9 – e^−2 (vbjax sweep); model-specific | Primary network bifurcation parameter |
| Conduction velocity v | 1–10 m/s (3–6 common) | Myelin-dependent |
| Parcellation | Desikan 68; Destrieux 148 cortical (164 with subcortical in vbjax benchmark); Schaefer 100–1000 | neurojax WAND derivatives: `(T, 68)`, `(T, 148)` |
| W_ij source | SIFT2 tractography weights + mean lengths | neurojax `fsl-connectome` derivatives |

In vbjax, coupling is applied to r in the simplest case, for example `c = 0.03*x.sum(axis=1)`. The network is then integrated with `make_sde(dt, dfun, gfun)`, an Euler–Maruyama scan.

W_ij comes from diffusion tractography ([diffusion-mri.md](diffusion-mri.md)). hgx extends pairwise coupling W_ij to hyperedges ([method-hypergraph.md](method-hypergraph.md)).

## Observables

- **EEG/MEG**: `Y = L x_obs + n`.
  - x_obs is `y1 − y2` for JR, superficial pyramidal current for CMC, or the HNN dipole.
  - L comes from BEM/FEM ([method-bem.md](method-bem.md), [method-fem.md](method-fem.md)).
  - neurojax can invert the same L with 15 solvers ([method-source-imaging.md](method-source-imaging.md)).
- **fMRI**: NMM activity drives the Balloon–Windkessel model, which adds a ~5 s lag and low-pass filtering to ~0.5 Hz ([fmri.md](fmri.md)). RWW gating S_E is the conventional driver.
- **Spectra**: linearised transfer functions predict M/EEG spectral peaks. This is the basis of DCM for cross-spectral densities ([method-spectral-analysis.md](method-spectral-analysis.md), [method-pharmaco-eeg.md](method-pharmaco-eeg.md)).
- **TMS-evoked potentials**: neurojax WAND stage N fits a global G and per-region A to empirical TEPs. It outputs region × time contribution matrices and virtual-lesion effects ([tms.md](tms.md)).

## Differentiable Fitting

All implementations listed here are JAX-native, so `jax.grad`, `vmap`, `pmap` and `jit` compose with them ([jax-ecosystem.md](jax-ecosystem.md), [method-neural-ode.md](method-neural-ode.md)).

- **Jacobian / bifurcation analysis.** `jax.jacobian(vb.mpr_dfun)(y0, (0.4, 0.), theta)` gives the local Jacobian, and vmapping over τ ∈ [1, 2] tracks eigenvalues.
- **Gradient descent.** In the vbjax MVAR demo, log loss falls from 5.80 to −3.78 in 50 steps, with a final parameter MSE of 8.4×10⁻⁵.
- **MCMC.** NumPyro NUTS with 500 warmup and 500 samples on a neural field requires ESS > 100, shrinkage > 0.7 and z < 1.5. Runtime is ~5 min on GPU and ~12 min on an x86 core.
- **SBI.** Amortized posteriors are covered in [method-sbi.md](method-sbi.md). hnn-jax enables flow- and score-based SBI over thousands of vmapped θ.
- **Spectral model comparison** (`vbjax:autoresearch/program.md`):
  - CMC, RRW and CBEI are fitted to synthetic Liley EEG.
  - Models are compared by BIC free energy (Stephan et al. 2009 framework), since the Hessian is too slow on an RTX 2080 with 8 GB.
  - The metric is `spectral_fit = −mean log-spectral distance`, with a <5 min budget per experiment.
  - Known failure mode: gradient descent gets stuck, giving identical fits across subjects.

Throughput reported in the vbjax README, from the `bench_vmap.py` benchmark; single-threaded C typically reaches 50–200 Kiter/s per core:

| Hardware | Throughput | Efficiency |
|----------|-----------|------------|
| Xeon W-2133 (88 W) | 5.7 Miter/s | 65 Kiter/W |
| Quadro RTX 5000 (200 W) | 26 Miter/s | 130 Kiter/W |
| M1 Air (18 W) | 3.7 Miter/s | 205 Kiter/W |

## Pitfalls

- **Units.** neurojax `bench/models/rww.py` originally had `gamma_E = 0.641/1000` and `gamma_I = 1.0/1000`. With τ in seconds, γ must be in s⁻¹, so the correct values are `gamma_E = 0.641` and `gamma_I = 1.0`. With the bug, the fixed point was S_E ≈ 0.0002 instead of ≈ 0.164 (Deco 2013). The fix is commit `49acac2` (`neurojax:docs/BUG_REPORTS.md`).
- **Oracle versions.** Biophysical reference simulators differ across releases (hnn_core 0.3 vs 0.6.1), so pin versions before validating an NMM against them.
- **Evoked vs total power.** Non-phase-locked subharmonics vanish in evoked spectra (vc2008-netpyne).
- **Mean-field closure.** Jitter and synchrony effects require population averaging (hnn-jax v0 → v1).
- **Published equations.** The vc2008 paper's printed α_n prefactor (0.1) prevents spiking; the canonical HH value is 0.01. Check printed parameters before porting.

## Citations
- vbjax: Woodman et al., ins-amu/vbjax [@woodman2024insamuvbjax]
- Graph neural fields [@aqil2021graph; @aqil2026brain]
- GABA/chandelier entrainment model [@vierlingclaassen2008modeling]
- WAND dataset used for neurojax NMM/TMS fitting [@mcnabb2025wand]
- Literature without local bibliography keys:
  - Jansen & Rit 1995
  - Montbrio, Pazo & Roxin 2015
  - Wendling et al. 2002
  - Deco et al. 2013
  - Bastos et al. 2012
  - Sanz Leon et al. 2013

## Relevant Projects

- **vbjax**: MPR, JR, Epileptor and dopamine models; `make_sde`; spherical-harmonic neural fields; NumPyro inference; spectral-fitting autoresearch across Liley, CMC, RRW and CBEI.
- **neurojax**: RWW (`bench/models/rww.py`), vbjax adapter and monitors in `bench/`, and leadfield forward models. Fits TEPs on WAND and applies HMM/DyNeMo to simulated output ([method-hmm-dynamics.md](method-hmm-dynamics.md)). Heritage: Wendling NMM → Fijee.
- **gnf-jax**: stochastic Wilson-Cowan graph neural field with closed-form spectra, stability analysis and FC.
- **hnn-jax**: differentiable HNN neocortical column; dipole forward model with a gradient oracle.
- **vc2008-netpyne**: HH and theta-neuron E/I networks as spiking ground truth for 40 Hz ASSR mean-field models.
- **alf**: NMMs as generative models for active inference ([method-active-inference.md](method-active-inference.md)).
- **hgx**: higher-order coupling beyond pairwise W_ij.
- **jaxctrl**: controllability and optimal perturbation of NMM networks.
- **inferjax**: amortized SBI with posterior calibration for NMM simulators.

## See Also

- [Connectome Harmonics](connectome-harmonics.md)
- [In-Vitro MEA Cortical Cultures](in-vitro-mea-cultures.md)
- [method-neural-fields.md](method-neural-fields.md): spatially continuous and graph field models
- [method-neural-ode.md](method-neural-ode.md): differentiable ODE/SDE integration
- [connectomics.md](connectomics.md): structural coupling matrices
- [physics-electromagnetic.md](physics-electromagnetic.md), [eeg.md](eeg.md), [meg.md](meg.md): electrophysiological observables
- [physics-hemodynamic.md](physics-hemodynamic.md), [fmri.md](fmri.md): BOLD observables
- [method-spectral-analysis.md](method-spectral-analysis.md), [method-pharmaco-eeg.md](method-pharmaco-eeg.md): spectral fitting
- [method-sbi.md](method-sbi.md), [method-variational-inference.md](method-variational-inference.md): parameter inference
- [method-source-imaging.md](method-source-imaging.md): inverse problem on the same leadfield
- [method-hypergraph.md](method-hypergraph.md): higher-order coupling
- [tms.md](tms.md): TMS-evoked potential modelling
