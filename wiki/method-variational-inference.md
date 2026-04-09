---
type: method
title: Variational Inference
category: inference
implementations: [vbjax:inference, alf:variational, sbi4dwi:posteriors]
related: [method-sbi.md, method-active-inference.md, method-neural-ode.md, physics-hemodynamic.md, method-source-imaging.md]
---

# Variational Inference

Variational Inference (VI) recasts Bayesian posterior computation as an optimization problem: instead of sampling from the intractable posterior p(theta|y), find the member q*(theta) of a tractable family Q that minimizes KL divergence to the true posterior. This makes VI orders of magnitude faster than MCMC for the large-scale generative models common in neuroimaging, at the cost of approximation bias.

## Core Formulation

### Evidence Lower Bound (ELBO)

The marginal log-evidence decomposes as:

```
ln p(y) = L(q) + KL[q(theta) || p(theta|y)]
```

where the ELBO (Evidence Lower Bound) is:

```
L(q) = E_q[ln p(y,theta)] - E_q[ln q(theta)]
     = E_q[ln p(y|theta)] - KL[q(theta) || p(theta)]
```

Since KL >= 0, maximizing L(q) is equivalent to minimizing KL[q || p(theta|y)]. The second form decomposes into an expected log-likelihood (data fit) and a KL penalty (regularization toward the prior).

### KL Divergence

The KL divergence between the variational approximation and posterior:

```
KL[q(theta) || p(theta|y)] = E_q[ln q(theta) - ln p(theta|y)]
```

This is the "exclusive" or "mode-seeking" KL: q tends to concentrate on a single mode of p, which is appropriate when the posterior is unimodal (as in many neuroimaging models) but problematic for multimodal posteriors.

### Variational Families

| Family | Form | Parameters | Captures correlations |
|--------|------|------------|-----------------------|
| Mean-field | q(theta) = prod_i q_i(theta_i) | Per-dimension mean + variance | No |
| Full-rank Gaussian | q = N(mu, Sigma) | Mean + full covariance | Yes, quadratic cost |
| Low-rank Gaussian | q = N(mu, D + VV^T) | Mean + diagonal + low-rank factors | Partial |
| Normalizing flow | q = T_K ... T_1(z), z ~ N(0,I) | Flow parameters | Yes, flexible |

Mean-field is fastest but underestimates posterior variance. Full-rank Gaussian (the Laplace/VL approach) captures correlations but scales as O(d^2) in parameter dimension d.

## Variational Laplace (VL)

Variational Laplace, used extensively in SPM/DCM, approximates the posterior with a Gaussian centered at the MAP estimate, using the curvature of the log-joint to form the covariance:

```
q(theta) = N(mu, Sigma)
mu = argmax_theta ln p(y,theta)
Sigma = (-nabla^2 ln p(y,theta)|_{theta=mu})^{-1}
```

### Gauss-Newton Optimization

SPM implements VL via a Gauss-Newton scheme. For a generative model y = g(theta) + e with e ~ N(0, Ce):

```
Delta mu = (J^T Ce^{-1} J + Sigma_prior^{-1})^{-1}
           (J^T Ce^{-1} (y - g(mu)) - Sigma_prior^{-1} (mu - mu_prior))
```

where J = dg/dtheta|_mu is the Jacobian of the forward model. This converges in 4-16 iterations for typical DCM models, compared to 10^4-10^6 samples for MCMC.

### Hyperparameter Estimation

VL jointly optimizes observation noise precision lambda via an EM-like scheme:

```
F(q, lambda) = E_q[ln p(y|theta,lambda)] - KL[q(theta) || p(theta)] + ln p(lambda)
```

The free energy F serves as both the objective and the model evidence approximation for Bayesian model comparison.

## Dynamic Causal Modeling (DCM)

DCM uses variational inference to invert generative models of effective connectivity from neuroimaging time series. The generative model has two layers:

1. **Neural dynamics**: dx/dt = (A + sum_j u_j B_j) x + Cu, where A is intrinsic connectivity, B_j are modulatory inputs, C is driving input
2. **Observation model**: y = g(x) + e, where g is the hemodynamic forward model (Balloon-Windkessel for fMRI) or lead field (for EEG/MEG)

### Model Inversion

VI estimates the full parameter set theta = {A, B, C, hemodynamic params, noise precision}:

- **fMRI-DCM**: ~50-200 parameters for a typical 5-region model
- **EEG-DCM**: ~100-500 parameters including neural mass model and lead field
- **Convergence**: 4-16 Gauss-Newton steps, each requiring forward integration + Jacobian

### Bayesian Model Comparison

The variational free energy F approximates the log-model evidence:

```
ln p(y|m) ~ F(m) = accuracy(m) - complexity(m)
```

This enables model selection over competing network architectures (which connections exist, which are modulated) without cross-validation, via Bayes factors: BF = exp(F_1 - F_2).

## Variational Autoencoders in Neuroimaging

Variational autoencoders (VAEs) use neural networks to parameterize both the generative model p_theta(x|z) and the inference model q_phi(z|x):

```
L(theta, phi) = E_{q_phi(z|x)}[ln p_theta(x|z)] - KL[q_phi(z|x) || p(z)]
```

Applications in neuroimaging:

| Application | Encoder input | Latent space | Decoder output |
|-------------|---------------|--------------|----------------|
| fMRI denoising | Noisy volumes | Low-D manifold | Clean volumes |
| Structural parcellation | T1w patches | Anatomical features | Tissue labels |
| Connectome generation | Demographics | Graph structure | Connectivity matrices |
| dMRI microstructure | DWI signals | Tissue params | Predicted signals |

### Reparameterization Trick

To backpropagate through the stochastic sampling z ~ q_phi(z|x) = N(mu_phi, sigma_phi^2):

```
z = mu_phi(x) + sigma_phi(x) * epsilon,  epsilon ~ N(0, I)
```

This moves the stochasticity into the input epsilon, making the ELBO gradient:

```
nabla_{phi} L ~ nabla_{phi} [ln p_theta(x | mu_phi + sigma_phi * epsilon) - KL[q_phi || p(z)]]
```

computable via standard automatic differentiation (e.g., `jax.grad`).

## Amortized Variational Inference

Standard VI optimizes q per observation. Amortized VI trains an inference network f_phi that maps any observation y directly to approximate posterior parameters:

```
q_phi(theta|y) = N(mu_phi(y), Sigma_phi(y))
```

This is the bridge between VI and simulation-based inference (SBI): neural posterior estimation in sbi4dwi is amortized VI where the inference network is trained on simulated data from the forward model. Key advantages:

- **O(1) inference** per new observation after training
- **Generalization** across the prior predictive distribution
- **No per-observation optimization** required at test time

See [method-sbi.md](method-sbi.md) for implementation details in the diffusion MRI context.

## Active Inference Connection

The Free Energy Principle posits that biological systems minimize variational free energy as a unified account of perception, action, and learning. In the active inference framework (implemented in `alf`):

- **Perceptual inference**: Minimize F = E_q[ln q(s) - ln p(o,s)] to update beliefs about hidden states
- **Active inference**: Select actions minimizing expected free energy G(pi) to achieve goals while reducing uncertainty
- **Learning**: Update generative model parameters (A, B, C, D matrices) via gradient descent on F

The variational free energy in `alf` is computed in JAX:

```python
def jax_variational_free_energy(q_s, A, B, o, prior):
    """VFE: KL[q(s)||p(s)] - E_q[ln p(o|s)]"""
    kl_term = jnp.sum(q_s * (jnp.log(q_s + 1e-16) - jnp.log(prior + 1e-16)))
    likelihood = jnp.sum(q_s * jnp.log(A[o, :] + 1e-16))
    return kl_term - likelihood
```

See [method-active-inference.md](method-active-inference.md) for the full framework.

## JAX Implementation Patterns

### Differentiable ELBO

JAX's automatic differentiation makes VI natural to implement:

```python
import jax
import jax.numpy as jnp

def elbo(params, key, data, n_samples=10):
    """Monte Carlo ELBO estimate with reparameterization."""
    mu, log_sigma = params['mu'], params['log_sigma']
    sigma = jnp.exp(log_sigma)
    eps = jax.random.normal(key, (n_samples, mu.shape[0]))
    z = mu + sigma * eps  # reparameterization trick
    log_lik = jnp.mean(log_likelihood(data, z))
    kl = 0.5 * jnp.sum(sigma**2 + mu**2 - 1 - 2*log_sigma)
    return log_lik - kl

grad_elbo = jax.grad(elbo)  # exact gradients via autodiff
```

### Batched Inference with vmap

```python
# Parallel VI across subjects/voxels
batched_elbo = jax.vmap(elbo, in_axes=(None, 0, 0))
batched_grad = jax.vmap(grad_elbo, in_axes=(None, 0, 0))
```

### Integration with Neural ODEs

For DCM-like models in vbjax, VI requires differentiating through ODE solvers:

```python
def dcm_elbo(params, y_obs):
    x = integrate_neural_ode(params['A'], params['C'], params['u'])
    y_pred = hemodynamic_forward(x, params['hemo'])
    log_lik = -0.5 * jnp.sum((y_obs - y_pred)**2 / params['noise_var'])
    log_prior = gaussian_log_prob(params, prior_mean, prior_cov)
    log_q = gaussian_log_prob(params, params['q_mu'], params['q_cov'])
    return log_lik + log_prior - log_q
```

The key advantage of JAX here is that `jax.grad` differentiates through the entire computational graph including the ODE solver (via adjoint methods), eliminating the need for hand-derived Jacobians.

## Comparison: VI vs MCMC vs SBI

| Property | Variational Inference | MCMC (NUTS/HMC) | Simulation-Based Inference |
|----------|----------------------|------------------|---------------------------|
| **Objective** | Optimize ELBO | Sample from posterior | Train inference network |
| **Asymptotic exactness** | No (approximation bias) | Yes (with mixing) | No (bounded by network capacity) |
| **Convergence criterion** | ELBO plateau | R-hat, ESS | Validation loss |
| **Speed (5-region DCM)** | ~1-10 s | ~10-60 min | ~30 s training, <1 ms inference |
| **Speed (whole-brain)** | ~1-30 min | Intractable | ~45 min training, ~30 s inference |
| **Posterior form** | Parametric (Gaussian) | Samples | Flexible (flows, MDNs) |
| **Multimodality** | Poor (mean-field/Laplace) | Good (if mixing) | Good (with expressive networks) |
| **Uncertainty calibration** | Often underestimates | Gold standard | Requires SBC validation |
| **Amortization** | No (standard) / Yes (VAE) | No | Yes |
| **Hyperparameter estimation** | Jointly optimized | Requires nesting | Implicit in training |
| **Model comparison** | Free energy F | Thermodynamic integration | Not directly available |
| **Scalability (parameters)** | O(d^2) Laplace, O(d) mean-field | O(d) per sample | O(1) at inference |
| **GPU acceleration** | Natural (JAX grad) | Good (NumPyro/JAX) | Natural (neural networks) |

### When to Use What

- **VI (Laplace/VL)**: DCM model inversion, Bayesian model comparison, moderate parameter counts (<500), when F is needed for model selection
- **MCMC**: Ground truth validation, multimodal posteriors, small models where exact posteriors matter, diagnostics via R-hat and ESS
- **SBI**: High-dimensional observations (voxelwise inference), intractable likelihoods, population-level studies where amortization pays off

## Key References

- **friston2022active**: Friston et al. (2022). Active inference and the free energy principle. Nature Reviews Neuroscience. Free energy principle underpinning VI in neuroscience.
- **Cranmer2020sbi**: Cranmer et al. (2020). The frontier of simulation-based inference. PNAS 117:30055-30062. Amortized VI via normalizing flows.
- **Hess2025bayesian**: Hess et al. (2025). Bayesian Workflow for Generative Modeling in Computational Psychiatry. Comp Psychiatry 9:76-99. Variational Laplace for DCM.
- **zhou2022pgmax**: Zhou et al. (2022). PGMax: Factor Graphs for Discrete Probabilistic Graphical Models and Loopy Belief Propagation in JAX. arXiv:2202.04110.
- **smith2022step**: Smith et al. (2022). A Step-by-Step Tutorial on Active Inference. J Math Psychology 107:102632.

## Relevant Projects

| Project | Role | VI flavor |
|---------|------|-----------|
| **vbjax** | Whole-brain simulation + Bayesian estimation | NumPyro MCMC + gradient-based optimization; differentiable forward models enable VI |
| **alf** | Active inference agents | Variational free energy minimization over discrete state spaces (A, B, C, D matrices) |
| **sbi4dwi** | Diffusion MRI microstructure | Amortized VI via normalizing flows and mixture density networks |
| **PGMax** | Probabilistic graphical models | Loopy belief propagation on factor graphs; differentiable message passing in JAX |
| **libspm** | Statistical Parametric Mapping | Variational Laplace for DCM, Gauss-Newton EM for model inversion |

## See Also

- [method-sbi.md](method-sbi.md) -- Simulation-based inference (amortized VI variant)
- [method-active-inference.md](method-active-inference.md) -- Active inference via free energy minimization
- [method-neural-ode.md](method-neural-ode.md) -- Differentiable ODE solvers for DCM-like models
- [physics-hemodynamic.md](physics-hemodynamic.md) -- Hemodynamic forward models inverted by VI in fMRI-DCM
- [method-source-imaging.md](method-source-imaging.md) -- EEG/MEG source reconstruction as Bayesian inverse problem
