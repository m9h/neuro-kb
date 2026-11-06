---
type: method
title: Simulation-Based Inference
category: inference
implementations: [sbi4dwi:dmipy_jax/pipeline, sbi4dwi:dmipy_jax/inference, sbi4dwi:dmipy_jax/library, inferjax:inferjax, hnn-jax:src/hnnjax, vpjax:vpjax/hemodynamics]
related: [method-variational-inference.md, method-monte-carlo.md, method-fem.md, method-neural-ode.md, diffusion-mri.md, neural-mass-models.md, eeg.md, meg.md, fmri.md, physics-hemodynamic.md, physics-cerebrovascular.md, physics-diffusion-equation.md, physics-acoustic.md, tus.md, jax-ecosystem.md, compute-infrastructure.md, method-source-imaging.md, tissue-skull.md]
description: "Simulation-Based Inference (SBI) estimates a Bayesian posterior p(θ|x) when the likelihood p(x|θ) is intractable but the forward simulator x ~ p(x|θ) can be sampled."
timestamp: 2026-11-05T15:59:53-08:00
tags: [method, inference, simulation-based]
---

# Simulation-Based Inference

Simulation-Based Inference (SBI) estimates a Bayesian posterior p(θ|x) when the likelihood p(x|θ) is intractable but the forward simulator x ~ p(x|θ) can be sampled. In this project family it is used to recover tissue microstructure from diffusion MRI ([diffusion-mri.md](diffusion-mri.md)). It is also planned for circuit parameters from M/EEG dipoles ([neural-mass-models.md](neural-mass-models.md)) and for hemodynamic parameters from BOLD/ASL/VASO ([physics-hemodynamic.md](physics-hemodynamic.md)).

## Principle

1. **Prior and simulate.** Draw θᵢ ~ p(θ), simulate xᵢ = f(θᵢ) + noise, and collect N pairs (θᵢ, xᵢ). Typical values are N = 10⁵–2·10⁵ in sbi4dwi.
2. **Train a conditional density estimator** q_φ(θ|x) by maximising Σ log q_φ(θᵢ|xᵢ). Neural Posterior Estimation (NPE) uses normalizing flows, MDNs or score-based diffusion models.
3. **Amortize.** After training, posterior samples for a new observation cost one network pass and no new simulations. This makes per-voxel whole-brain inference practical.
4. **Validate calibration.** Check that posteriors are neither over- nor under-confident. Simulation-based calibration (SBC) is the standard check [@talts2018validating].

Most of the practical engineering is in step 1. The noise model, the normalisation and the prior symmetries must be identical at train time and deploy time.

## Train/Deploy Invariants

These are the lessons that recur across sbi4dwi and inferjax:

| Invariant | Why it matters | Where enforced |
|---|---|---|
| Same normalisation at train and inference | The flow otherwise sees a shifted input distribution | `sbi4dwi:dmipy_jax/pipeline/simulator.py` (`sample_and_simulate`) and `deploy.py` (`predict_volume`); `inferjax` `sim.normalize()` |
| Noise applied inside the simulator | Rician noise is not additive Gaussian, so `noise_std=0.0` in `train_loop` | `sbi4dwi:autoresearch/program.md` |
| Label-switching broken in the prior | Exchangeable compartments give multimodal posteriors; enforce f₁ ≥ f₂ and swap orientations accordingly | Ball+2Stick prior |
| Hemisphere canonicalization | Fibre orientation ±μ is antipodally symmetric; constrain z ≥ 0 | Ball+2Stick prior |
| SI units internally | b-values in s/m² (FSL/DIPY s/mm² values must be multiplied by 10⁶) | `JaxAcquisition` |
| Pinned simulator version | The reference simulator must be reproducible across machines | `hnn-jax:oracle/aggregate.py` fails on mixed versions |

## Reference Configuration: Ball+2Stick Spline NPE (sbi4dwi)

**Data.** Synthetic HCP-like data with 90 directions and multi-shell b = 0/1000/2000/3000 s/mm². Training uses Rician noise at SNR 10–50; evaluation uses SNR 30 (`sbi4dwi:autoresearch/program.md`).

| Parameter | Value |
|---|---|
| Density estimator | FlowJAX masked autoregressive flow |
| Transformer | `RationalQuadraticSpline(knots=8, interval=3.0)` |
| Flow layers | 10 |
| Hidden dim | 128 (2-layer MLP per flow layer) |
| Optimizer | Adam, lr 3·10⁻⁴, cosine decay |
| Batch size | 512 |
| Training steps | 200k (baseline) → 300k (best) |
| Noise | Rician, per-sample SNR ∈ [10, 50] |
| Normalisation | Divide by mean b0 |
| Posterior aggregation | 500 samples, clipped to prior bounds, orientations renormalised to S², median |

### Results (Fiber 1 median angular error)

| Configuration | Fiber 1 median | d_stick r | f₁ r | Steps |
|---|---|---|---|---|
| Baseline affine flow | 10.7° | 0.95 | 0.85 | 30k |
| + spline + noise fix + label-switching | 6.4° | 0.978 | 0.902 | 30k |
| + 200k steps | 3.2° | 0.986 | 0.935 | 200k |
| + 300k steps | **2.8°** | **0.987** | **0.943** | 300k |

These targets follow the dMRI SBI orientation benchmarks of [@manzanopatron2025sbi]. The signal models are JAX ports of dmipy [@fick2019dmipy]. The same pipeline accepts NODDI [@zhang2012noddi] and SANDI [@palombo2020sandi] forward functions.

### Alternative Posterior Estimators

| Estimator | Best Fiber 1 median | Notes |
|---|---|---|
| Spline NPE (flow) | 2.8° | Current best |
| MDN (10 components) | ~5–8° | 30k steps; fast but limited expressiveness |
| Score-based, MLP + SDE sampler | 29° | SDE sampler diverges |
| Score-based, MLP + DDPM sampler | 15.5° | DDPM sampler is critical |
| + v-prediction, 1024-wide, 100k steps | 14.9° | |
| + spherical-coordinate parameterization | 12.8° | Best score-based result; FiLM conditioning |

BlackJAX NUTS gives non-amortized reference posteriors for spot checks. Amortized variational inference is also available ([method-variational-inference.md](method-variational-inference.md)).

The open research targets are:
- under 3.0° (ideally under 2.5°) median error;
- or ~3.2° reached in under 50k steps.

## Multi-Fidelity Training Data

The guiding principle is "build where differentiable, wrap where not" (`sbi4dwi:CLAUDE.md`). Non-JAX simulators are treated as **oracles** behind an HDF5 boundary.

**Analytical models** are closed-form compartment models. They are fast and differentiable.

**Differentiable simulators** include:
- `MatrixFormalismSimulator`: FEM spectral ROM on triangle meshes, solving S v = λ M v with PGSE via three matrix exponentials ([method-fem.md](method-fem.md));
- Diffrax SDE walkers ([physics-diffusion-equation.md](physics-diffusion-equation.md)).

A non-differentiable `MonteCarloSimulator` serves as ground truth ([method-monte-carlo.md](method-monte-carlo.md)).

**Oracles** are `DIPYMultiTensorOracle`, `ReMiDiOracle` (Docker/Python) and `MCMROracle` (MCMRSimulator.jl via Julia subprocess). They write a `SimulationLibrary` with these HDF5 datasets:
- `/params` (N, P)
- `/signals` (N, M)
- optional `/contrast_signals/{t1,t2}`
- `/source` provenance

**Adapter.** `OracleModelSimulator` turns a library into a sampler using k-NN inverse-distance interpolation. It warns when queries fall far from library coverage, and can optionally fit an MLP emulator for smoother interpolation.

**Mixing.** `HybridLibraryGenerator` combines **70% analytical / 30% oracle** samples. `train_multi_fidelity_sbi()` then reports metrics stratified by fidelity level.

## Calibration and Uncertainty Quantification

| Diagnostic | Module (sbi4dwi) | Purpose |
|---|---|---|
| Simulation-Based Calibration | `pipeline/sbc.py` | Rank-statistic uniformity of true θ within posterior samples [@talts2018validating] |
| Posterior Predictive Checks | `pipeline/ppc.py` | Re-simulate from the posterior and compare to observed x |
| Conformal prediction | `pipeline/conformal.py` | Distribution-free intervals |
| OOD detection | `pipeline/ood.py` | Flag voxels outside training support |
| Ensembles | `pipeline/ensemble.py` | Model-disagreement uncertainty |

In inferjax, SBC, PPC, conformal prediction and OOD detection are still being migrated from sbi4dwi (`inferjax:README.md`).

## inferjax: Modality-Agnostic SBI Core

inferjax extracts the SBI machinery after it had been written twice: once for dMRI and once needed for M/EEG. It treats any signal (a dMRI vector, an evoked waveform, a power spectrum) as an `(m,)` array.

**Simulator.** A `Simulator` composes four pieces:
- `forward_fn(theta, context)` — vmap-traceable;
- a uniform prior from `parameter_ranges`;
- a noise model;
- a normalizer.

**Normalizers.**

| Normalizer | Operation | Use |
|---|---|---|
| `Identity` | none | default |
| `ReferenceScale(mask)` | divide by mean of masked channels | dMRI b0 normalisation |
| `BaselineSubtract(mask)` | subtract mean of masked channels | ERP pre-stimulus baseline ([eeg.md](eeg.md), [meg.md](meg.md)) |

**Noise models.** `Gaussian`, `Rician` and `NoNoise` share a `SigmaSchedule`. It supports:
- fixed SNR;
- per-sample SNR augmentation, e.g. `snr_range=(10, 50)`;
- curriculum ramping via `sim.curriculum_step = (step, total_steps)`, which starts at high SNR.

**Extras.** `inferjax[flows]` adds FlowJAX and `inferjax[mcmc]` adds BlackJAX NUTS.

**Status.** Early. The simulator core, noise models and normalizers are tested. Flow, MDN and score-based estimators and sequential NPE are being migrated.

## hnn-jax: Differentiable Simulators for SBI on M/EEG

The reference HNN implementation (`hnn_core` on NEURON) is CPU-bound and non-differentiable. Fitting therefore relied on CMA-ES and SBI over expensive simulations. hnn-jax reimplements the forward model in JAX/Diffrax so that thousands of θ can be evaluated in one `vmap` call. This enables flow- and score-based SBI and gradient-based fitting of ERP generators ([neural-mass-models.md](neural-mass-models.md), [method-source-imaging.md](method-source-imaging.md)).

**Validation oracle.** The forward model is checked against a three-role oracle:
- **Forward:** `hnn_core(θ)` dipoles on Sobol-dense θ.
- **Gradient:** finite-difference Jacobians of the reference model.
- **Property:** invariants such as zero-drive quiescence, superposition, gain monotonicity and finite autodiff.

| Quantity | Value |
|---|---|
| Forward correlation vs oracle, v1 population model | +0.43 |
| Forward correlation vs oracle, v0 mean-field | +0.22 |
| CI threshold `FORWARD_CORR_MIN` | 0.40 (ratcheted upward) |
| hnn_core 0.6.1 vs 0.3 waveform correlation, identical θ | 0.62 (range 0.45–0.81) |
| RMS amplitude ratio between those versions | 0.76–1.13 |
| 8 serial sims: NSG/Expanse vs local | ~858 s vs ~247 s (NSG value is 128-core parallel width) |

Source: `hnn-jax:README.md`. NSG/Expanse execution is described in [compute-infrastructure.md](compute-infrastructure.md).

**Lesson for any SBI training set.** The two backends produced the same amplitude scale but different waveforms, so the training simulator itself must be version-pinned (`hnn_core >= 0.6.0`). Otherwise the amortized posterior is conditioned on an irreproducible likelihood.

**Population model detail.** The v1 model reparameterizes per-cell drive times as tᵢ = μ + σεᵢ. This lets gradients flow through `tc_sync` (σ, thalamocortical synchrony), a parameter the mean-field v0 could not represent.

## vpjax: Hemodynamic Simulators as SBI Targets

vpjax provides a differentiable chain from neural activity to multimodal MRI signals ([physics-cerebrovascular.md](physics-cerebrovascular.md), [fmri.md](fmri.md)):

neural activity → CMRO₂ → CBF → Balloon-Windkessel (s, f, v, q) → BOLD / ASL / VASO / qBOLD

**Integration.** `solve_balloon` uses the Diffrax Tsit5 solver. Gradients (e.g. ∂ peak BOLD/∂κ at κ = 0.65) are finite and non-zero (`vpjax:docs/tutorials/balloon_windkessel.md`).

**Fit for SBI.** This suits amortized inference over Balloon parameters (κ, γ, τ, α, E₀) and Riera neurovascular-coupling parameters.

**Identifiability.** vpjax's design mitigates the identifiability problem that SBI posteriors expose. Each state variable has an independent WAND measurement:
- f from pCASL;
- q from qBOLD;
- v from VASO (planned).

With VASO, the README notes there would be "zero free hemodynamic parameters".

## SBI vs. Gradient-Based Inversion

Differentiable simulators support both SBI and direct optimization, and the two should not be conflated:

| | SBI (amortized NPE) | Gradient-based inversion |
|---|---|---|
| Output | Full posterior q(θ\|x) | Point estimate θ* |
| Cost per new observation | One network pass | Full optimization loop |
| Example | sbi4dwi Ball+2Stick flow | brain-fwi / sbi4dwi FWI: skull c recovered to 4312 m/s (true 4080) over 3 bands in 222 s on A100 (`sbi4dwi:docs/openlifu/README.md`) |

j-Wave acoustic sensitivity analysis [@stanziola2023jwave] found mean |∂p/∂c| in skull **232×** that in water. Skull property uncertainty therefore dominates TUS focal accuracy ([tus.md](tus.md), [tissue-skull.md](tissue-skull.md), [physics-acoustic.md](physics-acoustic.md)). This makes skull acoustic properties a natural candidate for posterior estimation rather than point FWI. That application is prospective; it is not implemented.

## Limitations

- **Coverage.** The posterior is only valid inside the prior and simulator support. Use OOD flags and k-NN coverage warnings.
- **Model misspecification.** This is not detected by SBC; use PPC and multi-fidelity oracle validation.
- **Training cost.** The best dMRI result needs 300k steps at batch 512, and each autoresearch experiment has a 10-minute budget.
- **Orientation outputs.** These need manifold-aware aggregation (S² renormalisation; geometric or Fréchet means are an open direction).

## Relevant Projects

- **sbi4dwi**: reference implementation (dMRI NPE, multi-fidelity oracles, UQ suite, NIfTI deployment via `SBIPredictor`). The Julia companion DMI.jl tracks the leaderboard.
- **inferjax**: simulator-agnostic SBI core (Simulator, noise schedules, normalizers); estimators and calibration are being migrated.
- **hnn-jax**: differentiable HNN dipole simulator plus a version-pinned NSG oracle for SBI/gradient fitting of ERPs.
- **vpjax**: differentiable cerebrovascular/hemodynamic simulators; candidate SBI forward models with independent measurements per state variable.
- **brain-fwi**: gradient-based (non-SBI) acoustic inversion; a contrast case.

## See Also

- [diffusion-mri.md](diffusion-mri.md), [quantitative-mri.md](quantitative-mri.md), [tissue-white-matter.md](tissue-white-matter.md)
- [method-variational-inference.md](method-variational-inference.md), [method-monte-carlo.md](method-monte-carlo.md), [method-fem.md](method-fem.md), [method-neural-ode.md](method-neural-ode.md)
- [neural-mass-models.md](neural-mass-models.md), [eeg.md](eeg.md), [meg.md](meg.md), [method-source-imaging.md](method-source-imaging.md)
- [physics-hemodynamic.md](physics-hemodynamic.md), [physics-cerebrovascular.md](physics-cerebrovascular.md), [fmri.md](fmri.md)
- [tus.md](tus.md), [physics-acoustic.md](physics-acoustic.md), [tissue-skull.md](tissue-skull.md)
- [jax-ecosystem.md](jax-ecosystem.md), [compute-infrastructure.md](compute-infrastructure.md)

## Revision Notes

This revision corrects the prior version of the page:

- **Dead links removed.** Links to non-existent pages (method-npe, method-mcmc, method-abc, modality-dwi, modality-tfus, concept-microstructure) were dropped.
- **Unsupported figures removed:**
  - 20-component MDN results;
  - score-model d_stick/f₁ correlations;
  - training-time, throughput and memory benchmarks.
- **Misattribution fixed.** "256 elements 4.7×" was misattributed; 4.7× is the 3D **16-element** delay optimization on a 4 mm grid.
- **Scope clarified.** FWI is now framed as optimization, not SBI.
