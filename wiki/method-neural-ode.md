---
type: method
title: Neural ODEs / Differentiable Simulation
category: differentiable
implementations: [sbi4dwi:dmipy_jax/simulation/simulator.py, sbi4dwi:dmipy_jax/simulation/differentiable_walker.py, sbi4dwi:dmipy_jax/simulation/mesh_sim.py, sbi4dwi:dmipy_jax/biophysics/jwave_adapter.py, sbi4dwi:dmipy_jax/biophysics/tus_optimizer.py, sbi4dwi:dmipy_jax/core/pinns.py, sbi4dwi:dmipy_jax/pulseq, jaxctrl:simulate_lti, jaxctrl:simulate_closed_loop, jaxctrl:SINDyOptimizer, jaxctrl:KoopmanEstimator, vbjax:make_sde, vbjax:mpr_dfun, vbjax:make_shtdiff, vpjax:hemodynamics, vpjax:integrators/local_linearization.py]
related: [jax-ecosystem.md, neural-mass-models.md, method-neural-fields.md, method-sbi.md, method-monte-carlo.md, method-fem.md, physics-diffusion-equation.md, physics-bloch.md, physics-hemodynamic.md, physics-cerebrovascular.md, physics-acoustic.md, diffusion-mri.md, tus.md, method-hypergraph.md, method-variational-inference.md, method-spectral-analysis.md]
description: "Correction to the earlier version of this page: it claimed that neural ODE gradients are 'machine precision'."
timestamp: 2026-11-05T15:59:53-08:00
tags: [method, differentiable, ode]
---

# Neural ODEs / Differentiable Simulation

**Differentiable simulation** writes a physical or biological forward model (ODE, SDE, PDE, or matrix-exponential propagator) so that `jax.grad` can pass through the numerical integrator. Gradients of a loss with respect to parameters, initial conditions, or control inputs then come from autodiff, not finite differences. A **neural ODE** is the special case where some or all of the vector field `f_θ` is a neural network. Most uses in this project family are *mechanistic*: `f` is a known biophysical model (neural mass, Balloon-Windkessel, Bloch, wave equation) and only its parameters are learned. Neural networks appear mainly as posteriors, surrogates, or PINNs around the simulator.

## Core formulation

| Form | Equation | Typical use here |
|---|---|---|
| ODE | `dx/dt = f_θ(x, t, u)` | Neurovascular coupling (vpjax), closed-loop LTI control (jaxctrl) |
| SDE | `dx = f_θ(x) dt + g_θ(x) dW` | Neural mass networks (vbjax), confined Brownian spins (sbi4dwi) |
| Linear propagator | `x(t) = exp(A t) x0` | PGSE matrix formalism (sbi4dwi), LTI fallback (jaxctrl) |
| PDE (discretised) | `∇²p − (1/c²) ∂²p/∂t² = s` | Transcranial acoustic propagation via j-Wave [@stanziola2023jwave] |

## Gradient strategies

| Strategy | How gradients are obtained | Memory in # steps N | Notes |
|---|---|---|---|
| Discretise-then-optimise (backprop through solver) | Reverse-mode AD through every solver step | O(N), or ~O(log N) with recursive checkpointing | Exact gradient of the *discrete* solution. Diffrax default (`RecursiveCheckpointAdjoint`) |
| Optimise-then-discretise (continuous adjoint) | Solve `da/dt = −aᵀ ∂f/∂x` backward, then `dL/dθ = ∫ aᵀ ∂f/∂θ dt` | O(1) | Gradient matches the true gradient only up to solver tolerance. Fragile for stiff or chaotic systems (Chen et al. 2018, NeurIPS; no bibliography key) |
| Implicit differentiation | Differentiate the fixed-point / matrix equation directly | O(1) | Used for Lyapunov/Riccati solves [@kao2020automatic] |
| Fixed noise (reparameterisation) | Pre-sample Wiener increments `zs`, so the SDE becomes deterministic in θ | O(N) | vbjax `make_sde(dt, dfun, gfun)` takes `zs` as input, so `jax.grad` of a trajectory loss is well defined |

Correction to the earlier version of this page: it claimed that neural ODE gradients are "machine precision". That is wrong in general. Continuous-adjoint gradients carry integration error. Only discretise-then-optimise gives exact gradients, and those are gradients of the discrete scheme.

## Implementations and quantitative results

### vbjax — whole-brain SDE networks

vbjax builds network simulations as `make_sde(dt, dfun, gfun)` loops over neural mass models such as Montbrio-Pazo-Roxin (`mpr_dfun`). It uses `jax.vmap`/`jax.pmap` for parameter sweeps and `jax.grad` for fitting [@woodman2024insamuvbjax].

| Quantity | Value | Source |
|---|---|---|
| Example integration step | `dt = 0.01` (MPR network), `dt = 0.1` (neural field) | vbjax:README.md |
| Neural field grid | `lmax = 16`, 32 × 64 lat/lon, spherical-harmonic diffusion (`make_shtdiff`) | vbjax:README.md |
| Throughput, Xeon W-2133 | 5.7 Miter/s at 88 W = 65 Kiter/W | vbjax:README.md |
| Throughput, Quadro RTX 5000 | 26 Miter/s at 200 W = 130 Kiter/W | vbjax:README.md |
| Throughput, M1 Air | 3.7 Miter/s at 18 W = 205 Kiter/W | vbjax:README.md |
| Reference C/C++ single core | 50–200 Kiter/s for 164-node (Destrieux) networks | vbjax:README.md |
| Gradient-descent MVAR fit | 8 nodes, 1000 steps, 51 GD iterations (lr 0.01): log loss 5.80 → −3.78, MSE(A) = 8.4e−5 | vbjax:README.md |
| NUTS on neural field initial state | 500 warmup + 500 samples. Pass criteria: ESS > 100, shrinkage > 0.7, z-score < 1.5. Runtime ~5 min GPU, 7 min M1, 12 min x86 core | vbjax:vbjax/tests/test_field_inference.py |

The vbjax autoresearch loop shows a known failure mode: gradient descent through simulated EEG spectra often stalls ("parameters don't move, identical fits across subjects"). Four model families are benchmarked: Liley (14 states, used as ground truth), CMC, RRW, and CBEI (8 states each). Model comparison uses a BIC free-energy approximation because Hessians are too slow on an 8 GB RTX 2080 (vbjax:autoresearch/program.md). The loop's open questions are learning-rate schedules, gradient clipping, informed initialisation from fixed points, and how `dt` and the noise level affect loss-landscape smoothness. See [neural-mass-models.md](neural-mass-models.md) and [method-spectral-analysis.md](method-spectral-analysis.md).

### vpjax — stiff neurovascular ODEs

vpjax extends vbjax's simplified `make_bold()` Balloon-Windkessel into a differentiable chain (vpjax:README.md):

activity → CMRO₂ → vasodilation → CBF → CBV/oxygenation (Balloon) → BOLD / ASL / qBOLD

Key components:
- **Riera (2006/2007) neurovascular coupling** (`hemodynamics/riera.py`).
- **Fick's principle**: `CMRO₂ = CBF × OEF × CaO₂` (`metabolism/fick.py`).
- **Local Linearization integrator** (`integrators/local_linearization.py`). This is the Ozaki/Riera LL scheme for stiff neurovascular ODEs. It uses the update `x_{n+1} = x_n + J⁻¹(e^{J·Δt} − I) f(x_n)`, where `J = ∂f/∂x` is obtained by `jax.jacobian`. The scheme is exact for linear dynamics and stable at step sizes where explicit Euler diverges.

See [physics-hemodynamic.md](physics-hemodynamic.md) and [physics-cerebrovascular.md](physics-cerebrovascular.md).

### jaxctrl — differentiable control and system identification

Built on Equinox, Lineax, Optimistix, and Diffrax (jaxctrl:README.md):
- **Simulation.** `simulate_lti` and `simulate_closed_loop` use a Diffrax adaptive ODE solver when available and fall back to the matrix exponential otherwise.
- **Implicit differentiation of control equations.** `lqr`, `solve_continuous_are`, and `solve_continuous_lyapunov` are implicitly differentiable [@kao2020automatic]. For example, `jax.grad` of LQR cost with respect to the weight matrix Q.
- **Data-driven vector-field discovery.**
  - `SINDyOptimizer` fits sparse regression over `polynomial_library` / `fourier_library` terms. It is the sparse, interpretable counterpart of a neural ODE.
  - `KoopmanEstimator` (exact DMD) gives linear surrogates of nonlinear dynamics.
- **Tensor and hypergraph dynamics.** These include `tensor_lyapunov` and `multilinear_lqr` [@wang2024algebraic] and temporal-hypergraph controllability [@dong2024controllability]. See [method-hypergraph.md](method-hypergraph.md).

The autoresearch benchmark scores `control_advantage = (cost_classical − cost_jaxctrl)/cost_classical` at matched wall time. It covers Van der Pol, Lorenz, pendulum, cartpole, and coupled oscillators, with state dimension 2–20 and input dimension 1–5 (jaxctrl:autoresearch/program.md).

### sbi4dwi — differentiable physics for MRI and ultrasound

The design principle is "build where differentiable, wrap where not" (sbi4dwi:CLAUDE.md).

| Simulator | Scheme | Differentiable |
|---|---|:-:|
| `SDESimulator` | Diffrax SDE integration of spin random walks | Yes |
| `DifferentiableWalker` | Confined Brownian motion | Yes |
| `MatrixFormalismSimulator` | FEM stiffness/mass matrices → generalised eigenproblem `S v = λ M v` → PGSE as 3 matrix exponentials (pulse1 → gap → pulse2). Sparse BCOO used for meshes > 2K vertices | Yes |
| `MonteCarloSimulator` | SDF-geometry random walk (ground truth) | No |
| `pulseq/` | Bloch simulation from PyPulseq sequences (Diffrax) | Yes |
| `core/pinns.py` | Physics-informed networks for diffusion PDEs | Yes |
| `jwave_adapter`, `tus_optimizer`, `fwi_us` | Pseudospectral acoustic simulation [@stanziola2023jwave] | Yes |

Default free-water diffusivity is `D = 2.0e−9 m²/s`. Internally, b-values are stored in SI units (s/m²) (sbi4dwi:CLAUDE.md). The analytical compartments (Ball, Stick, Zeppelin, NODDI, SANDI) are **closed-form** signal equations, not ODEs [@fick2019dmipy; @zhang2012noddi; @palombo2020sandi]. The previous version of this page mislabelled them as neural ODEs. They become differentiable simulators only when replaced by the SDE/FEM paths above. See [diffusion-mri.md](diffusion-mri.md), [physics-diffusion-equation.md](physics-diffusion-equation.md), [method-fem.md](method-fem.md), and [method-monte-carlo.md](method-monte-carlo.md).

Gradient-through-simulator results on the SCI head model (208 × 256 × 256, 1 mm, 8 tissues; Modal A100) are listed below, all from sbi4dwi:docs/openlifu/README.md:

| Experiment | Result |
|---|---|
| 32-element delay optimisation (2D, 20 iter) | loss −0.299 → −0.414 (1.4×), 1.8 s/iter |
| 16-element delay optimisation (3D, 4 mm, 52 × 64 × 64) | 4.7× focal pressure, 2.6 s/iter |
| Sensitivity `∂p/∂c` | skull region 232× water |
| 3-band FWI (50→200→500 kHz), brain-fwi | skull velocity recovered 4312 m/s (true 4080), 222 s total |

The sensitivity result shows that skull property uncertainty dominates focal accuracy (see [tissue-skull.md](tissue-skull.md)). For FWI context see [@lin2024survey], [physics-acoustic.md](physics-acoustic.md), and [tus.md](tus.md).

**Continuous-time generative posteriors.** In sbi4dwi's score-based NPE, integrating the reverse-time SDE sampler *diverged* (29° median fibre error), whereas DDPM discrete sampling reached 15.5°, improving to 12.8° with spherical coordinates. Spline flows remain best at 2.8° (sbi4dwi:README.md; target from [@manzanopatron2025sbi]). Continuous normalizing flows (neural-ODE flows) are listed as an untested direction (sbi4dwi:autoresearch/program.md). See [method-sbi.md](method-sbi.md).

The companion Julia package DMI.jl uses DifferentialEquations.jl ODE/SDE samplers. Its AxCaliber PINN recovers axon radius R = 3.15 µm from WAND Connectom data (sbi4dwi:README.md).

## Practical guidance

1. **Stiff systems.** Use implicit solvers (e.g. Diffrax Kvaerno) or LL integration (vpjax). Explicit Dopri5 step-size collapse inflates both runtime and gradient memory.
2. **SDE gradients.** Fix the noise realisation (vbjax `zs`) or use Diffrax `VirtualBrownianTree` so gradients are reproducible.
3. **Chaotic or long rollouts.** Gradients explode. Fit spectra or summary statistics instead of raw trajectories, or switch to amortised inference ([method-sbi.md](method-sbi.md)) or MCMC/VI ([method-variational-inference.md](method-variational-inference.md)).
4. **Non-differentiable oracles.** Simulators such as MCMRSimulator.jl and ReMiDi stay outside the autodiff graph. They feed HDF5 training libraries instead.
5. **Validate solvers against each other.** For example, sbi4dwi's Helmholtz (GMRES) and time-domain PSTD solvers agree with r = 0.82 correlation on homogeneous water.

## Minimal Diffrax pattern

```python
import diffrax, jax

def vf(t, y, theta):
    return -theta["k"] * y + theta["u"]

def loss(theta, y0, y_obs, ts):
    sol = diffrax.diffeqsolve(
        diffrax.ODETerm(vf), diffrax.Kvaerno5(), t0=ts[0], t1=ts[-1], dt0=0.01, y0=y0,
        args=theta, saveat=diffrax.SaveAt(ts=ts),
        stepsize_controller=diffrax.PIDController(rtol=1e-6, atol=1e-8),
        adjoint=diffrax.RecursiveCheckpointAdjoint())   # or BacksolveAdjoint() for O(1) memory
    return ((sol.ys - y_obs) ** 2).mean()

grads = jax.grad(loss)(theta, y0, y_obs, ts)
```

## Background references (no bibliography keys available)

- Chen et al. (2018), *Neural Ordinary Differential Equations*, NeurIPS — continuous-adjoint neural ODEs.
- Kidger (2022), *On Neural Differential Equations*, Oxford thesis — Diffrax and the discretise- vs optimise-first trade-offs.
- Raissi et al. (2019), PINNs, *J. Comput. Phys.* 378:686–707.
- Sanz Leon et al. (2013), The Virtual Brain, *Front. Neuroinform.* 7:10.

## Relevant Projects

- **vbjax** — SDE neural mass and neural field networks, gradient fitting, NumPyro MCMC.
- **vpjax** — stiff neurovascular/metabolic ODEs (Riera, Balloon), Local Linearization integrator, BOLD/ASL/qBOLD forward models.
- **jaxctrl** — differentiable LTI/closed-loop simulation, implicit-diff Lyapunov/Riccati, SINDy and Koopman system ID.
- **sbi4dwi** — Diffrax SDE spin walkers, FEM matrix formalism, Bloch/Pulseq, PINNs, j-Wave TUS optimisation and FWI.
- Related downstream projects: `brain-fwi` (3D FWI), `gnf-jax` (stochastic Wilson-Cowan fields), `hnn-jax`, `neurojax`, `inferjax` (SBI on these simulators).

## See Also

- [jax-ecosystem.md](jax-ecosystem.md) — Diffrax, Equinox, Optimistix, Lineax
- [neural-mass-models.md](neural-mass-models.md), [method-neural-fields.md](method-neural-fields.md) — vbjax dynamics
- [physics-hemodynamic.md](physics-hemodynamic.md), [physics-cerebrovascular.md](physics-cerebrovascular.md) — vpjax models
- [physics-bloch.md](physics-bloch.md), [physics-diffusion-equation.md](physics-diffusion-equation.md), [diffusion-mri.md](diffusion-mri.md) — MRI simulation physics
- [physics-acoustic.md](physics-acoustic.md), [tus.md](tus.md), [tissue-skull.md](tissue-skull.md) — differentiable acoustics
- [method-sbi.md](method-sbi.md), [method-variational-inference.md](method-variational-inference.md) — inference on top of simulators
- [method-fem.md](method-fem.md), [method-monte-carlo.md](method-monte-carlo.md) — non-ODE forward solvers
- [method-hypergraph.md](method-hypergraph.md) — hypergraph dynamics and control
