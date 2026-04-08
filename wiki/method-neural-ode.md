```yaml
type: method
title: Neural ODEs / Differentiable Simulation
category: differentiable
implementations: [sbi4dwi:biophysics, jaxctrl:core, vbjax:integration]
related: [method-matrix-formalism.md, method-sde-simulation.md, physics-diffusion-equation.md]
```

# Neural ODEs / Differentiable Simulation

Neural ODEs and differentiable simulation enable end-to-end gradient-based optimization of dynamical systems by making the numerical integration process differentiable with respect to parameters, initial conditions, and system dynamics.

## Core Principle

Neural ODEs parameterize the time derivative of a system as a neural network or differentiable function:

```
dx/dt = f_θ(x, t)
```

where `f_θ` is a differentiable function (neural network, analytical model, or hybrid) with parameters `θ`. The solution `x(t)` is computed via numerical integration, with gradients flowing through the solver using the adjoint method.

## Adjoint Method

The adjoint method enables memory-efficient backpropagation through ODE solutions by solving a backward ODE:

```
da/dt = -a^T ∂f/∂x
dL/dθ = ∫[0,T] a^T ∂f/∂θ dt
```

This approach has O(1) memory complexity independent of the number of integration steps, unlike naive backpropagation which requires O(N) memory.

## Properties

| Property | Value/Range | Notes |
|----------|-------------|-------|
| Memory complexity | O(1) | Via adjoint method |
| Gradient accuracy | Machine precision | Continuous-time gradients |
| Integration tolerance | 1e-6 to 1e-9 | Adaptive stepping |
| Time complexity | O(N log N) | Depends on stiffness |
| Parameter efficiency | High | Shared dynamics across time |

## Implementation Frameworks

### Diffrax (JAX)
```python
import diffrax

def vector_field(t, y, args):
    return -y + args["forcing"]

term = diffrax.ODETerm(vector_field)
solver = diffrax.Dopri5()
solution = diffrax.diffeqsolve(term, solver, t0=0, t1=10, dt0=0.1, y0=y0)
```

### Neural ODE Applications

**sbi4dwi** implements several neural ODE variants:

1. **Biophysical Signal Models** — Analytical compartment models (Ball, Stick, Zeppelin) with exact derivatives
2. **SDE Simulation** — `SDESimulator` uses Diffrax for Brownian motion with drift terms
3. **Matrix Formalism** — Spectral methods via eigendecomposition for PGSE sequences
4. **TUS Optimization** — Gradient-based delay optimization through heterogeneous skull models

**jaxctrl** focuses on control-theoretic neural ODEs:

1. **System Identification** — SINDy optimizer for discovering governing equations from data
2. **LQR with Neural Dynamics** — Differentiable Linear-Quadratic Regulators
3. **Tensor Control** — Multilinear system dynamics with tensor eigenvalue methods
4. **Hypergraph Dynamics** — Control of higher-order network interactions

**vbjax** implements whole-brain simulation neural ODEs:

1. **Neural Mass Models** — Montbrio-Pazo-Roxin, Jansen-Rit, CMC with coupling
2. **Neural Field Simulation** — Spherical harmonic spatial dynamics on cortical surfaces  
3. **MCMC Parameter Estimation** — Bayesian inference of model parameters from EEG/MEG
4. **Forward Models** — EEG/MEG observation models linking activity to sensor data

## Neuroimaging Applications

### Diffusion MRI Microstructure (sbi4dwi)
- **Multi-compartment models**: NODDI, SANDI parameterized as neural ODEs
- **Simulation-based inference**: Neural posterior estimation from synthetic data
- **Acoustic simulation**: j-Wave adapter for transcranial focused ultrasound
- **Material property estimation**: Gradient-based optimization of tissue parameters

### Brain Dynamics (vbjax)
- **Connectome simulation**: Large-scale brain network dynamics
- **Neural field modeling**: Spatially-extended cortical activity patterns
- **Parameter inference**: MCMC estimation of neural mass model parameters
- **Forward modeling**: EEG/MEG prediction from simulated neural activity

### Control Theory (jaxctrl)
- **Neural control**: Differentiable feedback control of dynamical systems
- **System identification**: Data-driven discovery of governing equations
- **Optimal control**: Gradient-based optimization of control policies
- **Network controllability**: Analysis of hypergraph control landscapes

## Advantages

1. **Memory efficiency**: O(1) memory via adjoint method vs O(N) for discrete methods
2. **Continuous dynamics**: Natural representation of physical processes
3. **Adaptive integration**: Error-controlled timestep selection
4. **Parameter sharing**: Efficient representation across time
5. **Gradient quality**: Continuous-time gradients avoid discretization errors

## Limitations

1. **Stiff systems**: May require implicit solvers (computational overhead)
2. **Discontinuities**: Adjoint method requires smooth dynamics
3. **Numerical precision**: Integration errors can accumulate
4. **Solver selection**: Performance depends on problem-specific solver choice

## Relevant Projects

- **sbi4dwi**: Diffusion MRI microstructure modeling, TUS simulation, SBI pipeline
- **jaxctrl**: Control theory primitives, system identification, tensor methods
- **vbjax**: Whole-brain simulation, neural mass models, parameter inference

## See Also

- [method-matrix-formalism.md](method-matrix-formalism.md) — Spectral methods for diffusion simulation
- [method-sde-simulation.md](method-sde-simulation.md) — Stochastic differential equation approaches
- [physics-diffusion-equation.md](physics-diffusion-equation.md) — Underlying physics of diffusion processes