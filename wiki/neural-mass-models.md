---
type: method
title: Neural Mass Models
category: simulation
implementations: [vbjax:models, vbjax:coupling, neurojax:neural_models]
related: [method-neural-ode.md, physics-electromagnetic.md, eeg.md, meg.md, method-hmm-dynamics.md, method-spectral-analysis.md, connectomics.md]
---

# Neural Mass Models

Neural mass models (NMMs) describe the collective dynamics of large neuronal populations (10^4--10^7 neurons) as low-dimensional systems of ordinary or stochastic differential equations. Each "node" represents a cortical column or parcellated brain region, reducing the problem from millions of spiking neurons to 2--8 state variables per region. NMMs are the backbone of whole-brain simulation, linking structural connectivity to emergent oscillatory dynamics and, via forward models, to measurable EEG, MEG, and fMRI signals.

## Major Model Families

| Model | States/node | Variables | Key feature | Primary reference |
|-------|-------------|-----------|-------------|-------------------|
| Wilson-Cowan | 2 | E, I firing rates | First rate model; excitatory-inhibitory balance | Wilson & Cowan 1972 |
| Jansen-Rit (JR) | 6 | y0-y5 (PSPs + derivatives) | Alpha rhythm genesis; EEG forward model | Jansen & Rit 1995 |
| Lopes da Silva | 4 | thalamo-cortical loop | Alpha rhythm via thalamic pacemaker | Lopes da Silva 1974 |
| Montbrio-Pazo-Roxin (MPR) | 2 | r, V (rate, mean voltage) | Exact mean-field of QIF neurons | Montbrio et al. 2015 |
| Canonical Microcircuit (CMC) | 8 | ss, sp, ii, dp populations | Laminar decomposition; predictive coding | Bastos et al. 2012 |
| Reduced Wong-Wang | 2 | S_E, S_I (synaptic gating) | BOLD-fMRI coupling; DMF studies | Deco et al. 2014 |

## Jansen-Rit Model

The Jansen-Rit model describes a cortical column as three interacting populations: pyramidal cells, excitatory interneurons, and inhibitory interneurons. The six state variables are membrane potentials and their derivatives for each population.

### Equations

The model uses a sigmoid activation function to convert average membrane potential to firing rate:

```
S(v) = 2e0 / (1 + exp(r(v0 - v)))
```

with `e0 = 2.5 s^-1`, `v0 = 6 mV`, `r = 0.56 mV^-1`.

The dynamics of each population follow second-order equations (here in first-order form):

```
dy0/dt = y3
dy3/dt = Aa * S(y1 - y2) - 2a*y3 - a^2*y0          # pyramidal PSP
dy1/dt = y4
dy4/dt = Aa * (p(t) + C2*S(C1*y0)) - 2a*y4 - a^2*y1  # excitatory interneuron PSP
dy2/dt = y5
dy5/dt = Bb * C4*S(C3*y0) - 2b*y5 - b^2*y2          # inhibitory interneuron PSP
```

### Standard Parameters

| Parameter | Symbol | Value | Units |
|-----------|--------|-------|-------|
| Excitatory gain | A | 3.25 | mV |
| Inhibitory gain | B | 22.0 | mV |
| Excitatory time constant | 1/a | 10 | ms |
| Inhibitory time constant | 1/b | 20 | ms |
| Connectivity C1 | C1 | 135 | - |
| Connectivity C2 | C2 | 0.8 * C1 | - |
| Connectivity C3 | C3 | 0.25 * C1 | - |
| Connectivity C4 | C4 | 0.25 * C1 | - |
| External input mean | p | 120-320 | s^-1 |

### Bifurcation and Alpha Rhythm

The JR model exhibits a Hopf bifurcation as external input p increases. For p in the range 120--320 s^-1, the model produces ~10 Hz alpha oscillations. Below p ~ 90 s^-1 the system is quiescent; above p ~ 350 s^-1 it saturates to a fixed point. The transition between regimes passes through a bistable region generating epileptiform spikes, making JR a standard model for seizure dynamics.

### EEG Observable

The model EEG signal is the difference of excitatory and inhibitory postsynaptic potentials at the pyramidal population: `V_EEG(t) = y1(t) - y2(t)`.

## Montbrio-Pazo-Roxin (MPR) Model

The MPR model is the exact mean-field reduction of an infinite population of all-to-all coupled quadratic integrate-and-fire (QIF) neurons. It is the analytically most principled NMM: no phenomenological sigmoid is needed, since the firing rate emerges from the Lorentzian ansatz on the Ott-Antonsen manifold.

### Equations

```
tau * dr/dt = Delta / (pi * tau) + 2*r*V
tau * dV/dt = V^2 + eta + J*tau*r - (pi*tau*r)^2 + I_ext(t)
```

where:
- `r`: population firing rate (Hz)
- `V`: mean membrane potential (mV)
- `tau`: membrane time constant (ms)
- `Delta`: heterogeneity width of the Lorentzian distribution of excitabilities (mV)
- `eta`: center of the excitability distribution (mV, bifurcation parameter)
- `J`: recurrent synaptic weight (mV*s)
- `I_ext`: external input (coupling from other regions)

### Default Parameters (vbjax)

| Parameter | Symbol | vbjax default | Units |
|-----------|--------|---------------|-------|
| Time constant | tau | 1.0 | ms |
| Heterogeneity | Delta | 1.0 | mV |
| Excitability | eta | -5.0 | mV |
| Coupling strength | J | 15.0 | mV*s |
| Coupling rate | cr | 1.0 | - |
| Coupling voltage | cv | 0.0 | - |

### Bifurcation Structure

The MPR model undergoes a saddle-node on invariant circle (SNIC) bifurcation as eta increases through a critical value eta_c ~ -((pi*Delta)^2 + J^2)/(4*J). Below eta_c the system is at a stable fixed point (low firing rate). Above eta_c the system enters a limit cycle with oscillation frequency that increases continuously from zero (Type I excitability). The parameter J controls the bistability region: higher J widens the hysteresis loop.

## Canonical Microcircuit (CMC)

The CMC model implements a 4-population laminar cortical circuit with 8 state variables, designed for hierarchical predictive coding:

| Population | Abbreviation | Layer | Role |
|-----------|--------------|-------|------|
| Spiny stellate | ss | L4 | Granular input |
| Superficial pyramidal | sp | L2/3 | Prediction errors (forward) |
| Inhibitory interneurons | ii | L2/3 | Intrinsic inhibition |
| Deep pyramidal | dp | L5/6 | Predictions (backward) |

Hierarchical coupling is asymmetric: forward connections (lower to higher cortical areas) drive spiny stellate cells; backward connections (higher to lower) modulate superficial pyramidal and inhibitory populations. This architecture implements Bayesian message passing: `sp` encodes prediction errors, `dp` encodes predictions.

```python
# vbjax CMC example: 2-node hierarchy
import vbjax as vb
G_fwd, G_bwd = 80.0, 80.0
p = (G_fwd, G_bwd, vb.cmc_default_theta._replace(I=220.0))
_, loop = vb.make_sde(dt=0.5, dfun=vb.cmc_hier_2node_dfun, gfun=1e-3)
ys = loop(vb.np.zeros((8, 2)), vb.randn(4000, 8, 2), p)

# Separate observables by cortical layer
sp_signal = vb.cmc_observe_sp(ys.T)  # prediction errors
dp_signal = vb.cmc_observe_dp(ys.T)  # predictions
```

## Coupling on Structural Connectomes

Individual NMM nodes are coupled into whole-brain networks via the structural connectome. The coupling term for node i takes the general form:

```
I_i(t) = k * sum_j [ W_ij * G(x_j(t - d_ij)) ]
```

where:
- `W_ij`: connection weight from the structural connectome (typically from diffusion MRI tractography)
- `d_ij = L_ij / v`: conduction delay, with tract length L_ij (mm) and conduction velocity v (typically 1--10 m/s)
- `G(x)`: coupling function (may act on firing rate, membrane potential, or both)
- `k`: global coupling scaling parameter

### Coupling Parameters

| Parameter | Typical range | Notes |
|-----------|--------------|-------|
| Global coupling k | 0.01--100 | Model-specific; often the primary bifurcation parameter |
| Conduction velocity v | 1--10 m/s | Myelination-dependent; 3--6 m/s most common |
| Mean delay | 5--20 ms | For cortical parcellations (~80 mm mean tract length) |
| Parcellation size | 68--1000 regions | Desikan-Killiany (68), Destrieux (164), Schaefer (100--1000) |

In vbjax, coupling is typically applied to the firing rate variable (for MPR, the `r` component):

```python
import vbjax as vb
import jax.numpy as np

def network(x, p):
    c = 0.03 * x.sum(axis=1)   # global coupling on firing rate
    return vb.mpr_dfun(x, c, p)

_, loop = vb.make_sde(dt=0.01, dfun=network, gfun=0.1)
```

For distance-dependent delays, vbjax supports delay differential equation integration or discretized delay buffers indexed by the tract-length matrix.

## Connection to Observables

### EEG/MEG Forward Model

NMM output couples to sensor-level EEG/MEG via the leadfield matrix L (see [physics-electromagnetic.md](physics-electromagnetic.md)):

```
Y_EEG(t) = L * x_obs(t) + noise
```

where `x_obs` depends on the model: `y1 - y2` for JR, superficial pyramidal current for CMC, or a projection of `V` for MPR. The leadfield L encodes volume conduction through skull and scalp tissues, computed via BEM or FEM on an anatomical head model (see [method-fem.md](method-fem.md), [method-bem.md](method-bem.md)).

### fMRI via Balloon-Windkessel

For fMRI observables, neural activity from the NMM drives the Balloon-Windkessel hemodynamic model (see [physics-hemodynamic.md](physics-hemodynamic.md)):

```
Neural activity z(t) → blood flow f(t) → blood volume v(t) → BOLD y(t)
```

The neural-to-hemodynamic coupling introduces a ~5 s lag and acts as a severe low-pass filter, limiting effective temporal resolution to ~0.5 Hz. The reduced Wong-Wang model was specifically designed for this use case, with synaptic gating variables S_E and S_I that directly drive the hemodynamic cascade.

### Spectral Observables

NMM simulations can also be compared to data via their power spectral density. The linearized transfer function of the JR model around its operating point predicts spectral peaks that match empirical M/EEG spectra (see [method-spectral-analysis.md](method-spectral-analysis.md)). This is the basis of Dynamic Causal Modelling (DCM) for M/EEG, which fits NMM parameters to cross-spectral densities.

## Bifurcation Analysis

Bifurcation analysis maps the qualitative behavior of NMMs as parameters vary, identifying transitions between:

- **Fixed points** (asynchronous / resting state)
- **Limit cycles** (sustained oscillations, e.g., alpha rhythm at ~10 Hz)
- **Bistable regimes** (coexistence of rest and oscillation, relevant to epilepsy)
- **Chaotic attractors** (complex dynamics in coupled networks)

Key bifurcation types in NMMs:

| Bifurcation | Model | Control parameter | Dynamical signature |
|-------------|-------|-------------------|---------------------|
| Hopf | Jansen-Rit | external input p | Onset of alpha oscillation |
| SNIC | MPR | excitability eta | Type I oscillation onset |
| Pitchfork | Wilson-Cowan | coupling strength | Symmetry breaking |
| Period-doubling | Coupled JR | inter-regional coupling k | Route to chaos |

In vbjax, Jacobian-based bifurcation analysis is straightforward via `jax.jacobian`:

```python
import jax
import vbjax as vb
import jax.numpy as jp

y0 = jp.r_[0.1, -2.0]

def eig1_tau(tau):
    theta = vb.mpr_default_theta._replace(tau=tau)
    J = jax.jacobian(vb.mpr_dfun)(y0, (0.4, 0.), theta)
    return jp.linalg.eigvals(J)[0]

# Sweep tau to find bifurcation
eigs = jax.vmap(eig1_tau)(jp.r_[1.0:2.0:32j])
```

## JAX / Differentiable Implementation

vbjax provides NMMs as pure JAX functions, making them composable with `jax.grad`, `jax.vmap`, `jax.pmap`, and `jax.jit`. The core pattern is:

1. **Define `dfun`**: a function `(state, coupling, params) -> d(state)/dt`
2. **Wrap with `make_sde`**: creates an Euler-Maruyama integrator with configurable dt and noise
3. **Run via `loop`**: the returned scan-based loop function integrates the SDE

Available model functions in vbjax:

| Function | Model | States | Notes |
|----------|-------|--------|-------|
| `vb.mpr_dfun` | Montbrio-Pazo-Roxin | 2 (r, V) | Default for most applications |
| `vb.jr_dfun` | Jansen-Rit | 6 (y0-y5) | Well-characterized alpha generator |
| `vb.cmc_dfun` | Canonical Microcircuit | 8 (ss, sp, ii, dp) | Laminar + predictive coding |
| `vb.bvep_dfun` | Epileptor (BVEP) | 2 | Seizure dynamics |
| `vb.dopa_dfun` | Dopamine-modulated | 6 (r, V, u, Sa, Sg, Dp) | Neuromodulation, 3-channel coupling |

Default parameter sets are available as named tuples (e.g., `vb.mpr_default_theta`, `vb.cmc_default_theta`) and can be modified via `._replace()`.

### Performance

vbjax achieves high throughput for network simulations (164-node Destrieux parcellation):

| Hardware | Throughput | Efficiency |
|----------|-----------|------------|
| Xeon W-2133 (Skylake, 88W) | 5.7 Miter/s | 65 Kiter/W |
| Quadro RTX 5000 (Turing, 200W) | 26 Miter/s | 130 Kiter/W |
| M1 Air (5nm, 18W) | 3.7 Miter/s | 205 Kiter/W |

### Parameter Estimation

Because vbjax models are fully differentiable, they support:

- **Gradient descent**: Direct optimization of NMM parameters to fit empirical time series or spectral features
- **MCMC with NumPyro**: Bayesian posterior estimation via NUTS (see [method-neural-ode.md](method-neural-ode.md))
- **Parallel parameter sweeps**: `jax.vmap` over parameter grids, `jax.pmap` over CPU cores or GPUs

## Key References

- **Deco2013rww**: Deco et al. (2013). Resting-state functional connectivity emerges from structurally and dynamically shaped slow linear fluctuations. J Neurosci 33:11239-11252. Reduced Wong-Wang model for resting-state simulation.
- **SanzLeon2013tvb**: Sanz Leon et al. (2013). The Virtual Brain: a simulator of primate brain network dynamics. Frontiers in Neuroinformatics 7:10. TVB framework for connectome-coupled NMMs.
- **Griffiths2024whobpyt**: Griffiths et al. (2024). WhoBPyT: Whole-Brain Modelling in PyTorch. Differentiable NMM fitting.
- **ValdesSosa2026xialphanet**: Valdes-Sosa et al. (2026). xi-alphaNet: conduction delays shape alpha oscillations across the lifespan. National Science Review.
- **Vidaurre2018hmm**: Vidaurre et al. (2018). Spontaneous cortical activity transiently organises into frequency specific phase-coupling networks. Nature Communications 9:2987. HMM state inference on NMM outputs.

## Relevant Projects

- **vbjax** (primary): Full NMM library with MPR, JR, CMC, BVEP, Dopa models; SDE integration; connectome coupling; EEG/MEG forward models; neural field simulation on spherical harmonics; NumPyro-based Bayesian inference. Source: `github.com/ins-amu/vbjax`
- **neurojax**: Provides `bench/` module for vbjax adapter, monitors, and leadfield-based forward modeling. HMM/DyNeMo models for state inference on NMM-generated time series. Uses Diffrax for neural ODE/SDE integration.
- **alf**: Active inference agents can use NMMs as their generative model of brain dynamics, enabling agents that infer neural parameters from observations while selecting actions that minimize expected free energy. The B matrix (transition model) can be parameterized by NMM dynamics.
- **hgx**: Higher-order coupling beyond pairwise connectomes. Standard NMMs use linear pairwise coupling `W_ij`; hgx extends this to hyperedge interactions where triplets or k-tuples of regions interact simultaneously, capturing structure not present in the structural connectome alone.
- **jaxctrl**: Control-theoretic analysis of NMM networks, including controllability metrics and optimal perturbation design for brain stimulation protocols.

## See Also

- [method-neural-ode.md](method-neural-ode.md) -- Differentiable simulation and adjoint methods underlying NMM integration
- [physics-electromagnetic.md](physics-electromagnetic.md) -- Forward problem linking NMM output to EEG/MEG sensors
- [physics-hemodynamic.md](physics-hemodynamic.md) -- Balloon-Windkessel model for NMM-to-BOLD coupling
- [eeg.md](eeg.md) -- EEG modality: the primary observable for NMM validation
- [meg.md](meg.md) -- MEG modality: complementary observable with different lead field structure
- [method-hmm-dynamics.md](method-hmm-dynamics.md) -- HMM state inference applied to NMM-generated time series
- [method-spectral-analysis.md](method-spectral-analysis.md) -- Spectral comparison between NMM output and empirical data
- [method-active-inference.md](method-active-inference.md) -- Active inference framework using NMMs as generative models
