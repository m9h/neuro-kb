---
type: method
title: Simulation-Based Inference
category: inference
implementations: [sbi4dwi:pipeline, sbi4dwi:inference]
related: [diffusion-mri.md, tus.md, method-neural-ode.md, method-monte-carlo.md, method-fem.md]
---

# Simulation-Based Inference

Simulation-Based Inference (SBI) is a family of methods for Bayesian parameter estimation when the likelihood function is intractable but forward simulation is feasible. In neuroimaging, SBI enables recovery of tissue microstructure parameters from diffusion MRI by training neural networks to approximate the posterior distribution p(θ|x) using synthetic data generated from biophysical forward models.

## Principle

Traditional parameter fitting requires optimizing a likelihood function L(θ|x), which is often intractable for complex biophysical models. SBI circumvents this by:

1. **Forward simulation**: Generate synthetic data pairs (θ, x) from prior p(θ) and simulator p(x|θ)
2. **Neural posterior estimation**: Train a neural network to approximate p(θ|x) 
3. **Amortized inference**: Once trained, predict parameters for new observations instantly

The key insight is that simulators are typically easier to implement than their inverse likelihood functions, especially for multi-compartment diffusion models with complex geometry.

## Implementation in SBI4DWI

The SBI pipeline in `sbi4dwi` follows a three-stage architecture:

### Stage 1: Forward Model Definition
```python
# Ball + 2-Stick multi-compartment model
model = compose_models([
    G1Ball(),           # CSF compartment
    C1Stick(),          # Primary fiber
    C1Stick()           # Secondary fiber
])

def forward_fn(params, acquisition):
    return model(params, acquisition.gradient_directions, 
                 acquisition.b_values)
```

### Stage 2: Simulation and Training
```python
simulator = ModelSimulator(
    forward_fn=forward_fn,
    prior_sampler=sample_ball_2stick_prior,
    noise_model=RicianNoise(snr_range=(10, 50))
)

posterior = train_sbi(
    simulator, 
    n_simulations=200_000,
    method="flow",  # Neural spline flow
    n_steps=200_000
)
```

### Stage 3: Clinical Deployment
```python
predictor = SBIPredictor(posterior, acquisition)
results = predictor.predict_volume(dwi_nifti, brain_mask)
```

## Neural Posterior Architectures

### Normalizing Flows
Rational-quadratic spline flows achieve **2.8° median fiber orientation error** on Ball+2Stick:

| Component | Configuration |
|-----------|---------------|
| Transform | Rational-quadratic spline (8 knots) |
| Layers | 10 masked autoregressive |
| Hidden dim | 128 per layer |
| Training | 300k steps, cosine LR decay |
| SNR augmentation | Variable (10-50) during training |

Key design: hemisphere canonicalization (z ≥ 0) and label-switching symmetry breaking (f₁ ≥ f₂).

### Score-Based Diffusion
Denoising score matching with DDPM sampling:

| Architecture | Performance |
|-------------|-------------|
| MLP + FiLM conditioning | 12.8° fiber error |
| Spherical coordinates | Eliminates unit sphere constraints |
| v-prediction parameterization | Improved numerical stability |

### Mixture Density Networks
Fast baseline method with Gaussian mixture posteriors:

| Components | Fiber Error | Training Time |
|-----------|-------------|---------------|
| 10 Gaussians | ~5-8° | 30k steps |
| 20 Gaussians | ~4-6° | 30k steps |

## Multi-Fidelity Training

SBI4DWI supports hybrid training combining analytical and high-fidelity simulators:

```python
hybrid = HybridLibraryGenerator(
    analytical_sim,     # 70% - fast, differentiable
    oracle_library,     # 30% - slow, high-fidelity
)
```

Oracles include:
- **DIPY**: Multi-tensor analytical models
- **ReMiDi**: Monte Carlo random walk simulation
- **MCMRSimulator.jl**: Julia-based finite element methods

## Transcranial Focused Ultrasound Extension

SBI principles extend to acoustic parameter estimation for transcranial focused ultrasound (TFUS). The differentiable j-Wave acoustic simulator enables gradient-based optimization:

### Skull Property Estimation
Full waveform inversion recovers acoustic properties from multi-frequency ultrasound measurements:

| Method | Skull Velocity Recovery | Training |
|--------|------------------------|----------|
| 3-band FWI (50→200→500 kHz) | 4312 m/s (true: 4080) | 222s on A100 |
| Single frequency (500 kHz) | 2166 m/s | 50s |

### Multi-Element Array Optimization
Gradient-based delay pattern optimization through heterogeneous skull:

| Array Size | Focal Improvement | Time per Iteration |
|------------|-------------------|-------------------|
| 16 elements | 1.3× pressure gain | 1.8s |
| 32 elements | 1.4× pressure gain | 1.8s |
| 256 elements | 4.7× pressure gain | 2.6s |

## Uncertainty Quantification

SBI naturally provides uncertainty estimates through posterior sampling:

### Simulation-Based Calibration
Validates posterior coverage by checking if true parameters fall within predicted confidence intervals across synthetic test cases.

### Conformal Prediction
Distribution-free confidence intervals with guaranteed coverage:
- **Marginal coverage**: 95% intervals contain true value 95% of the time
- **Conditional coverage**: Adapts interval width based on input difficulty

### Out-of-Distribution Detection
Flags inputs that deviate significantly from training distribution:
- **Likelihood ratio**: p(x|θ_posterior) vs p(x|θ_prior)
- **Embedding distance**: Neural network feature space metrics
- **Ensemble disagreement**: Variation across multiple trained models

## Performance Benchmarks

### Accuracy (Ball+2Stick on HCP-like acquisition)
| Method | Median Fiber Error | d_stick r | f1 r | Training Steps |
|--------|-------------------|-----------|------|----------------|
| **Spline Flow** | **2.8°** | **0.987** | **0.943** | 300k |
| Score Diffusion | 12.8° | 0.92 | 0.85 | 100k |
| 20-component MDN | 4.6° | 0.96 | 0.88 | 30k |

### Speed (Clinical deployment)
- **Training**: 200k simulations in ~45 minutes (A100 GPU)
- **Inference**: 100k voxels/second (whole-brain in ~30 seconds)
- **Memory**: <2GB GPU memory for typical clinical volumes

## Advantages over Traditional Fitting

1. **Amortized cost**: Training expensive, inference instant
2. **Full posteriors**: Uncertainty quantification included
3. **Robustness**: Neural networks handle noise and model violations gracefully  
4. **Parallelization**: Trivially scales to population studies
5. **Complex priors**: Can incorporate anatomical constraints and multi-modal information

## Limitations

1. **Training data requirements**: 10⁴-10⁶ simulations needed
2. **Generalization**: Performance depends on training distribution coverage
3. **Black box**: Less interpretable than analytical fitting
4. **Computational setup**: Requires GPU infrastructure and ML expertise

## Key References

- **Cranmer2020sbi**: Cranmer et al. (2020). The frontier of simulation-based inference. PNAS 117:30055-30062. Comprehensive review of SBI methodology.
- **manzanopatron2025sbi**: Manzano-Patron et al. (2025). Uncertainty mapping and probabilistic tractography using SBI in diffusion MRI. Medical Image Analysis 103:103580.
- **zhang2012noddi**: Zhang et al. (2012). NODDI: practical in vivo neurite orientation dispersion and density imaging. NeuroImage 61:1000-1016.
- **palombo2020sandi**: Palombo et al. (2020). SANDI: a compartment-based model for non-invasive apparent soma and neurite imaging. NeuroImage 215:116835.
- **fick2019dmipy**: Fick et al. (2019). The Dmipy Toolbox: Diffusion MRI Multi-Compartment Modeling and Microstructure Recovery Made Easy. Frontiers in Neuroinformatics 13:64.

## Relevant Projects

- **sbi4dwi**: Primary implementation with JAX/Equinox
- **brain-fwi**: Full waveform inversion for ultrasound parameter estimation  
- **DMI.jl**: Julia implementation using SciML stack
- **organoid-hgx-benchmark**: Gene regulatory network inference via SBI

## See Also

- [diffusion-mri.md](diffusion-mri.md) — Diffusion-weighted imaging physics
- [tus.md](tus.md) — Transcranial focused ultrasound
- [method-neural-ode.md](method-neural-ode.md) — Neural ODEs / differentiable simulation
- [method-monte-carlo.md](method-monte-carlo.md) — Monte Carlo simulation methods
- [method-fem.md](method-fem.md) — Finite element methods for forward models