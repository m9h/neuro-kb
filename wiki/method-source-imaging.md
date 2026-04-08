---
type: method
title: Source Imaging / Inverse Problems
category: inference
implementations: [neurojax:source, brain-fwi:inversion]
related: [method-bem-forward-modeling.md, method-fem-forward-modeling.md, physics-electromagnetic-fields.md, head-model-mida.md, coordinate-system-mni.md]
---

# Source Imaging / Inverse Problems

Source imaging (also called source localization or inverse modeling) reconstructs the spatial distribution of neural current sources from sensor measurements recorded outside the head. This fundamental inverse problem underlies M/EEG analysis, transcranial ultrasound imaging, and other neuroimaging modalities.

## The Forward-Inverse Relationship

The inverse problem is defined through the forward model:

```
Y = L·J + n
```

where:
- `Y` ∈ ℝ^(M×T) is the sensor data (M sensors, T timepoints)
- `L` ∈ ℝ^(M×N) is the leadfield matrix (M sensors, N sources)
- `J` ∈ ℝ^(N×T) is the neural current density to be estimated
- `n` is measurement noise

The leadfield `L` encapsulates the physics of signal propagation from brain sources to sensors, computed via [method-bem-forward-modeling.md](method-bem-forward-modeling.md) or [method-fem-forward-modeling.md](method-fem-forward-modeling.md).

## Problem Characteristics

### Ill-posed nature
- **Underdetermined**: Typically M << N (275 MEG sensors vs ~10,000 cortical sources)
- **Non-unique**: Multiple source configurations can produce identical sensor data
- **Unstable**: Small measurement noise can cause large reconstruction errors

### Regularization necessity
All practical inverse methods impose priors or constraints:
- **Minimum norm**: Prefer spatially smooth solutions (MNE, sLORETA)
- **Sparsity**: Assume few active sources (CHAMPAGNE, HIGGS)
- **Biophysical**: 1/r³ decay with distance (LAURA, VARETA)
- **Beamforming**: Adaptive spatial filtering (LCMV, SAM, DICS)

## Method Families

### L2 Minimum Norm Family
Based on Tikhonov regularization with quadratic penalty:

| Method | Innovation | Noise normalization |
|--------|-----------|-------------------|
| MNE | Basic minimum norm | Raw leadfield |
| dSPM | Dynamic statistical parametric mapping | Noise covariance weighted |
| sLORETA | Standardized low-resolution brain electromagnetic tomography | Depth-weighted normalization |
| eLORETA | Exact low-resolution brain electromagnetic tomography | Zero localization error (point sources) |

**Implementation (neurojax)**:
```python
from neurojax.source import estimate_mne, estimate_dspm

# Basic MNE with explicit Tikhonov regularization
J_mne = estimate_mne(Y, L, alpha=0.1)

# dSPM with noise covariance normalization
J_dspm = estimate_dspm(Y, L, noise_cov=C_n, alpha=0.1)
```

### Biophysical Prior Methods

| Method | Prior assumption | Regularization |
|--------|----------------|---------------|
| LAURA | 1/r³ distance decay | Local auto-regressive average |
| VARETA | Variable resolution | Adaptive spatial smoothing |

LAURA applies the biophysical constraint that neural current strength decays as 1/r³ from measurement sites, matching the physics of volume conduction.

### Beamformer Family
Construct adaptive spatial filters that pass signals from target locations while suppressing interference:

| Method | Domain | Optimization |
|--------|--------|-------------|
| LCMV | Time domain | Minimum variance with unit gain constraint |
| SAM | Time domain | Pseudo-Z statistic on source power |
| DICS | Frequency domain | Dynamic imaging of coherent sources |

**LCMV weight vector**:
```
w = (L^T C^(-1) L)^(-1) L^T C^(-1)
```

where `C` is the data covariance matrix.

### Bayesian Sparse Methods
Use hierarchical priors to promote sparsity and estimate connectivity:

| Method | Innovation | Joint estimation |
|--------|-----------|------------------|
| CHAMPAGNE | Type-II maximum likelihood | Source covariance structure |
| HIGGS | Hidden Gaussian graphical spectral models | Source + connectivity |

**CHAMPAGNE** iteratively estimates hyperparameters γ_i controlling source strength:
```
p(J_i | γ_i) = N(0, γ_i I)
```

### Physics-Informed Graph Neural Networks
**PI-GNN** (neurojax implementation) combines graph convolution with the physical forward model:

```python
from neurojax.source import SourceGNN, mesh_to_graph

graph = mesh_to_graph(vertices, faces)
model = SourceGNN(n_features=6, hidden_dim=64, tikhonov_reg=alpha)
J_gnn = model(Y, L, graph, features)
```

The graph convolution operates on cortical mesh topology, incorporating geometric features (normals, curvature) and the physics-based leadfield constraint.

## Transcranial Ultrasound FWI

Full Waveform Inversion (FWI) reconstructs acoustic properties (sound speed, density) from transcranial ultrasound measurements. Unlike M/EEG, the observed data are acoustic waveforms rather than electromagnetic fields.

**Forward model** (brain-fwi):
```
u(x,t) = F[c(x), ρ(x), s(t)]
```

where `c(x)` is sound speed, `ρ(x)` is density, and `F` is the acoustic wave equation solver.

**FWI objective**:
```python
def fwi_loss(c_params):
    c = sigmoid_transform(c_params, c_min=1400, c_max=3200)
    u_pred = acoustic_forward(c, rho, sources, sensors)
    return jnp.mean((u_pred - u_obs)**2)

grad_c = jax.grad(fwi_loss)(c_params)
```

Multi-frequency banding (50-100 kHz → 100-200 kHz → 200-300 kHz) prevents cycle-skipping artifacts in the optimization.

## Regularization Parameter Selection

**Tikhonov parameter α**:
- L-curve method: Plot ||L·J - Y||² vs ||J||² for varying α
- Generalized cross-validation (GCV)
- Discrepancy principle: Choose α such that ||L·J - Y|| ≈ noise level

**neurojax convention**: Always report regularization parameters explicitly:
```python
alpha = estimate_tikhonov_reg(L, snr=3.0)  # SNR-based estimate
print(f"Using Tikhonov regularization: α = {alpha:.2e}")
J = estimate_mne(Y, L, alpha=alpha)
```

## Validation Metrics

### Simulation studies
- **Dipole Localization Error (DLE)**: Euclidean distance between true and estimated dipole
- **Area Under Curve (AUC)**: ROC analysis for source detection
- **Spatial dispersion**: Spread of reconstructed activity around true source

### Real data
- **Cross-validation**: Split trials, train on subset, test on remainder
- **Anatomical plausibility**: Consistency with known functional anatomy
- **Temporal dynamics**: Physiologically reasonable time courses

## Resolution Analysis

The resolution matrix `R = G L` where `G` is the inverse operator quantifies spatial blurring:
- Point Spread Function: R(:,i) shows how source i spreads in reconstruction
- Crosstalk Function: R(i,:) shows contamination of source i from other sources

For MNE: `R = L (L^T L + α I)^(-1) L^T`

## Relevant Projects

- **neurojax**: 15 inverse solvers (MNE, dSPM, sLORETA, eLORETA, LCMV, SAM, DICS, CHAMPAGNE, VARETA, LAURA, HIGGS, PI-GNN)
- **brain-fwi**: Full waveform inversion for transcranial ultrasound using j-Wave and JAX autodiff
- **vbjax**: Forward models for whole-brain simulation validation
- **hgx**: Hypergraph structure for multi-scale source connectivity

## See Also

- [method-bem-forward-modeling.md](method-bem-forward-modeling.md) — Boundary Element Method for leadfield computation
- [method-fem-forward-modeling.md](method-fem-forward-modeling.md) — Finite Element Method for detailed head modeling  
- [head-model-mida.md](head-model-mida.md) — High-resolution anatomical reference
- [physics-electromagnetic-fields.md](physics-electromagnetic-fields.md) — Maxwell equations underlying M/EEG forward models