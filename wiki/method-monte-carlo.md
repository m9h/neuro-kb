```yaml
type: method
title: Monte Carlo Simulation
category: stochastic
implementations: 
  - dot-jax:analytical
  - MCMRSimulator.jl:Spins
  - sbi4dwi:forward_models
related: [method-finite-element.md, physics-diffusion.md, physics-photon-transport.md]
```

# Monte Carlo Simulation

Monte Carlo methods use repeated random sampling to solve computational problems that might be deterministic in principle. In neuroimaging, Monte Carlo simulation is particularly powerful for modeling complex transport phenomena where analytical solutions are intractable.

## Physics Basis

Monte Carlo methods model physical processes by tracking individual particles (photons, spins, molecules) through random walks governed by probabilistic rules. The key insight is that macroscopic observables emerge from the statistical behavior of many microscopic events.

### Random Walk Framework

The fundamental equation governing diffusion-based Monte Carlo is:

```
r(t + Δt) = r(t) + √(2D·Δt)·ξ
```

where:
- `r(t)` is particle position at time t
- `D` is the diffusion coefficient
- `ξ` is a random vector from normal distribution N(0,1)
- `Δt` is the simulation timestep

For photon transport in tissue, the Beer-Lambert law governs absorption:

```
P(survival) = exp(-μₐ · s)
```

where `μₐ` is absorption coefficient and `s` is path length.

## Implementation Properties

### Convergence and Accuracy

Monte Carlo estimators converge as **1/√N** where N is the number of samples. This universal convergence rate is independent of problem dimensionality, making Monte Carlo particularly attractive for high-dimensional problems.

| Parameter | Typical Value | Impact |
|-----------|---------------|---------|
| Number of particles | 10⁴-10⁸ | Standard error ∝ 1/√N |
| Timestep (diffusion) | 0.1-1.0 ms | Accuracy vs computational cost |
| Step size (photon) | Mean free path/10 | Balance between accuracy and efficiency |
| Boundary tolerance | 10⁻⁶ μm | Geometric precision |

### Variance Reduction

Advanced Monte Carlo implementations use variance reduction techniques:

- **Importance sampling**: Sample more frequently from high-contribution regions
- **Control variates**: Use analytical approximations to reduce variance  
- **Russian roulette**: Probabilistically terminate low-weight particles
- **Stratified sampling**: Divide parameter space into strata

## Neuroimaging Applications

### Diffusion MRI (MCMRSimulator.jl)

Models spin diffusion through complex tissue geometries:

```julia
# Cylinder-restricted diffusion
geometry = Cylinders(radius=1.0, position=[[0,0]], repeats=[2.5,2.5])
simulation = Simulation(sequences, geometry, diffusivity=2.0)
signal = readout(evolve(simulation, 80.0))  # 80ms evolution
```

**Key features:**
- Handles arbitrary tissue geometries (cylinders, spheres, walls, meshes)
- Models membrane permeability and surface relaxation
- Accounts for susceptibility-induced off-resonance fields
- Supports multi-compartment exchange dynamics

**Tissue parameters modeled:**
- Axon diameter: 0.5-5.0 μm
- Diffusivity: 0.5-3.0 μm²/ms  
- Membrane permeability: 0-∞ μm/ms
- T₁/T₂ relaxation rates by compartment

### Diffuse Optical Tomography (dot-jax)

Uses analytical Green's functions for validation, Monte Carlo for complex geometries:

```python
# Infinite medium analytical solution
phi_analytical = analytical_cw_infinite(mua=0.01, musp=1.0, src=[0,0,0], det=[30,0,0])

# Monte Carlo validation for complex head models
mesh = FEMMesh.create(node, elem)
phi_mc = monte_carlo_photons(mesh, mua=0.01, musp=1.0, n_photons=1e6)
```

**Validation against analytical solutions:**
- Infinite medium: Agreement within 1% for >10⁶ photons
- Semi-infinite medium: 2-3% agreement with image source method
- Sphere models: <5% deviation from Mie scattering theory

### Simulation-Based Inference (sbi4dwi)

Monte Carlo forward models enable Bayesian parameter estimation:

```python
# NODDI model with Monte Carlo simulation
def forward_model(theta):
    ficvf, odi, fiso = theta
    return mcmr_simulate(ficvf=ficvf, odi=odi, fiso=fiso)

# Train neural posterior estimator
posterior = train_sbi(simulator=forward_model, 
                     observations=dwi_data,
                     num_simulations=10000)
```

## Computational Considerations

### Parallelization

Monte Carlo methods are "embarrassingly parallel" - each particle trajectory is independent:

- **CPU**: OpenMP threading across particles
- **GPU**: CUDA/OpenCL with thousands of threads  
- **Distributed**: MPI across compute nodes
- **JAX**: Automatic vectorization with `jax.vmap`

### Memory Requirements

| Simulation Type | Memory per Particle | Scaling |
|-----------------|-------------------|---------|
| Diffusion MRI | ~100 bytes | Linear with N_spins |
| Photon transport | ~50 bytes | Linear with N_photons |
| Combined multimodal | ~200 bytes | Linear with N_particles |

### Computational Complexity

- **Time complexity**: O(N·T) where N = particles, T = timesteps
- **Space complexity**: O(N) for particle states
- **Communication**: Minimal (only final statistics collection)

## Validation Benchmarks

### Cross-Platform Validation

Monte Carlo implementations are validated against:

1. **Analytical solutions**: Infinite/semi-infinite media
2. **Other simulators**: MCX, CAMINO, MEDUSA cross-comparison  
3. **Experimental data**: Phantom measurements
4. **Finite element**: FEM solutions for simple geometries

### Statistical Validation

Monte Carlo estimators must satisfy:
- **Unbiased**: E[estimate] = true_value
- **Consistent**: estimate → true_value as N → ∞
- **Minimum variance**: Optimal sampling strategy

## Advantages and Limitations

### Advantages
- **Geometric flexibility**: Handle arbitrary complex boundaries
- **Physical realism**: Natural representation of transport processes
- **Scalable parallelism**: Linear speedup with more processors
- **Statistical uncertainty**: Built-in error estimates

### Limitations  
- **Computational cost**: Slow convergence (1/√N)
- **Random noise**: Statistical fluctuations in output
- **Parameter sensitivity**: Small changes can cause large variance
- **Memory requirements**: Scale with number of particles

## Key References

- **cottaarMultimodalMonteCarlo2026**: Cottaar et al. (2026). Multi-Modal Monte Carlo MRI Simulator of Tissue Microstructure. Imaging Neuroscience. doi:10.1162/IMAG.a.1177
- **Fang2010mcx**: Fang (2010). Mesh-based Monte Carlo method using fast ray-tracing in Plucker coordinates. Biomed Optics Express 1:165-175.
- **Fang2020**: Fang et al. (2020). Diffusion MRI simulation of realistic neurons with SpinDoctor and the Neuron Module. NeuroImage 222:117198.
- **Ginsburger_2019**: Ginsburger et al. (2019). MEDUSA: A GPU-based tool to create realistic phantoms of the brain microstructure. NeuroImage 193:10-24.
- **Lee2021**: Lee et al. (2021). Realistic Microstructure Simulator (RMS): Monte Carlo simulations of diffusion in three-dimensional cell segmentations. J Neurosci Methods 350:109018.

## Relevant Projects

- **dot-jax**: Analytical validation, phantom studies, chromophore spectroscopy
- **MCMRSimulator.jl**: Multi-modal MRI simulation, tissue microstructure  
- **sbi4dwi**: Bayesian inference, NODDI/SANDI parameter estimation

## See Also

- [method-finite-element.md](method-finite-element.md) - Complementary numerical approach
- [physics-diffusion.md](physics-diffusion.md) - Underlying transport physics
- [physics-photon-transport.md](physics-photon-transport.md) - Optical Monte Carlo physics
- [tissue-white-matter.md](tissue-white-matter.md) - Microstructural modeling target
- [method-bayesian-inference.md](method-bayesian-inference.md) - Parameter estimation framework