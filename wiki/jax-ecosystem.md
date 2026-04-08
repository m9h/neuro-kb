```yaml
type: concept
title: JAX Ecosystem for Neuroimaging
related: [differentiable-programming.md, neural-posterior-estimation.md, equinox-modules.md, scientific-ml.md]
```

# JAX Ecosystem for Neuroimaging

The **JAX ecosystem** has emerged as the dominant framework for differentiable neuroimaging research, enabling end-to-end gradient flow from raw sensor data to biophysical parameters. This ecosystem combines automatic differentiation, GPU acceleration, and functional programming to solve previously intractable inverse problems in brain imaging.

## Core JAX Stack

| Layer | Library | Purpose |
|-------|---------|---------|
| **Arrays & autodiff** | JAX | GPU acceleration, `jax.grad`, XLA compilation |
| **Neural networks** | Equinox | Pytree-compatible `eqx.Module` classes |
| **Optimization** | Optax/Optimistix | Stochastic training / deterministic fitting |
| **Dynamics** | Diffrax | Neural ODEs/SDEs for Bloch equations |
| **Linear algebra** | Lineax | GLM, beamforming, sparse solvers |
| **Graphs** | jraph | GNNs on cortical meshes |
| **Flows** | FlowJAX | Normalizing flows for posterior estimation |
| **MCMC** | BlackJAX | Bayesian inference (NUTS) |
| **Probabilistic** | NumPyro | Variational inference, hierarchical models |

## Neuroimaging-Specific Extensions

### Multi-Modal Signal Models

JAX enables differentiable biophysical models across imaging modalities:

```python
# Diffusion MRI microstructure
class NODDIModel(eqx.Module):
    def __call__(self, params, acquisition):
        return watson_stick + tortuosity_zeppelin + csf_ball

# Quantitative MRI relaxometry  
class McDesspot(eqx.Module):
    def simulate_sequence(self, tissue_params, flip_angles, TRs):
        return spoiled_gre_signal + balanced_ssfp_signal

# MEG/EEG source imaging
class BeamformerLCMV(eqx.Module):
    def __call__(self, Y, leadfield, regularization):
        return spatial_filter @ Y
```

All models are `eqx.Module` pytrees, making them compatible with `jax.vmap`, `jax.grad`, and `jax.jit`.

## Simulation-Based Inference Pipeline

The neuroimaging JAX ecosystem excels at **simulation-based inference** (SBI) — training neural networks to invert forward models:

### 1. Forward Simulation
```python
# Generate training data
def simulate_batch(key, n_samples):
    params = prior.sample(key, n_samples)  # Sample biophysical parameters
    signals = vmap(forward_model)(params, acquisition)
    noisy_signals = add_rician_noise(signals, snr=30)
    return params, noisy_signals
```

### 2. Neural Posterior Estimation
```python
# Train normalizing flow
flow = NeuralSplineFlow(input_dim=signal_dim, context_dim=param_dim)
flow, losses = train_flow(flow, training_data, n_steps=200_000)

# Infer parameters from new data
posterior_samples = flow.sample(observed_signal, n_samples=1000)
param_estimate = jnp.median(posterior_samples, axis=0)
```

### 3. Uncertainty Quantification
- **Simulation-Based Calibration** — validate posterior coverage
- **Conformal Prediction** — distribution-free confidence intervals  
- **Out-of-Distribution Detection** — flag unreliable estimates

## Multi-Fidelity Integration

JAX projects seamlessly integrate multiple simulation fidelities:

**Analytical models** (fast, differentiable) for bulk training data:
```python
# Closed-form Gaussian compartments
signal = volume_fraction * stick_signal + (1 - volume_fraction) * ball_signal
```

**Physics simulation** (slow, accurate) for validation:
```python
# FEM on tetrahedral meshes
def fem_simulator(mesh, diffusivity, acquisition):
    K, M = assemble_fem_matrices(mesh.vertices, mesh.tetrahedra, diffusivity)
    return matrix_formalism_pgse(K, M, acquisition)
```

**External oracles** (non-differentiable) wrapped at HDF5 boundary:
```python
# Monte Carlo, DIPY, commercial simulators
oracle_data = h5py.File("monte_carlo_library.h5")
emulator = train_neural_emulator(oracle_data)  # Fast approximation
```

## Performance Characteristics

### Speed Benchmarks

| Task | Traditional | JAX | Speedup |
|------|-------------|-----|---------|
| DTI fitting (100k voxels) | 45 minutes | 2.3 seconds | **1170x** |
| NODDI whole-brain | 6 hours | 8 seconds | **2700x** |
| Flow posterior training | N/A | 3.2° median error in 200k steps | Novel capability |
| TUS optimization (32 elements) | Minutes | 1.8s/iteration | **>100x** |

### Memory Efficiency
- **Automatic batching** via `jax.vmap` eliminates explicit loops
- **JIT compilation** fuses operations, reduces memory allocations
- **Gradient accumulation** for large datasets exceeding GPU memory

## Differentiable Physics Integration

JAX enables end-to-end differentiation through complex physics:

### Acoustic Simulation (sbi4dwi)
```python
# Transcranial focused ultrasound through heterogeneous skull
def optimize_delays(delays, skull_properties, target_location):
    pressure_field = jwave_simulation(delays, skull_properties)
    focal_intensity = pressure_field[target_location]
    return -focal_intensity  # Maximize at target

optimal_delays = jax.scipy.optimize.minimize(optimize_delays, initial_delays)
```

### Electromagnetic Forward Models (neurojax)
```python
# MEG/EEG leadfield computation
def differentiable_leadfield(conductivity_map, source_locations):
    K = assemble_fem_stiffness(mesh, conductivity_map)
    phi = solve_forward_potential(K, source_locations)
    return sensor_projection(phi)

# Optimize tissue conductivity to match MEG data
grad_conductivity = jax.grad(lambda sigma: mse(Y, differentiable_leadfield(sigma, J)))
```

## Cross-Project Integration

### VBJax Integration (neurojax)
Neural mass models for MEG simulation:
```python
# Wendling-David-Friston cortical column
def neural_mass_ode(state, conductivity, connectivity):
    return wendling_dynamics(state) + connectivity @ state

# Couple to MEG forward model  
meg_signal = leadfield @ source_activity
```

### HGX Integration (hgx)
Hypergraph neural networks on brain connectivity:
```python
# Higher-order brain dynamics
hypergraph = construct_functional_hypergraph(meg_data)
dynamics = hypergraph_neural_network(hypergraph, node_features)
```

### JaxCtrl Integration (jaxctrl)  
Controllability analysis of brain networks:
```python
# Gramian-based controllability on structural connectivity
def controllability_gramian(adjacency, control_nodes):
    return jax.scipy.linalg.expm(adjacency @ adjacency.T)

control_energy = trace(inv(controllability_gramian(connectome, stimulation_sites)))
```

## Relevant Projects

| Project | Domain | JAX Integration |
|---------|--------|-----------------|
| **sbi4dwi** | Diffusion MRI microstructure | Full SBI pipeline, 15+ biophysical models |
| **neurojax** | Multi-modal M/EEG analysis | 15 source imaging methods, differentiable BEM/FEM |
| **vbjax** | Whole-brain simulation | Neural mass models, connectome dynamics |
| **hgx** | Hypergraph neural networks | Higher-order brain connectivity |
| **alf** | Active inference | Generative models, Bayesian inference |
| **jaxctrl** | Control theory | Network controllability, Lyapunov analysis |
| **setae** | Bio-inspired mechanics | Contact mechanics, tissue deformation |

## Key Advantages

**Composability** — Forward models, inference networks, and physics simulators compose naturally as JAX functions.

**Scalability** — Single GPU handles datasets that required HPC clusters (100k+ voxels in seconds).

**Reproducibility** — Deterministic compilation and functional programming eliminate state-dependent bugs.

**Extensibility** — New physics or inference methods integrate with existing pipeline infrastructure.

**Clinical Translation** — Fast inference enables real-time applications (TUS optimization, MRI reconstruction).

## See Also

- [neural-posterior-estimation.md](neural-posterior-estimation.md) — SBI methodology and benchmarks
- [equinox-modules.md](equinox-modules.md) — PyTree neural network patterns  
- [differentiable-programming.md](differentiable-programming.md) — Automatic differentiation principles
- [scientific-ml.md](scientific-ml.md) — Physics-informed machine learning