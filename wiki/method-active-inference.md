---
type: method
title: Active Inference
category: inference
implementations: [alf:agent, spinning-up-alf:notebooks/08-12]
related: [method-variational-inference.md, method-free-energy-principle.md, method-belief-propagation.md]
---

# Active Inference

Active Inference (AIF) is a unified framework for perception, action selection, and learning based on the minimization of variational free energy. Unlike traditional reinforcement learning which maximizes reward signals, Active Inference agents build generative models of their environment and act to minimize expected free energy — thereby satisfying both goal-directed (pragmatic) and information-seeking (epistemic) imperatives.

## Core Principles

Active Inference is grounded in the Free Energy Principle, which posits that biological systems maintain their organization by minimizing surprise (or equivalently, maximizing evidence for their generative model). This leads to three fundamental drives:

1. **Perceptual inference**: Update beliefs about hidden states given observations
2. **Active inference**: Select actions that minimize expected free energy  
3. **Model learning**: Update generative model parameters to better predict observations

## Mathematical Framework

### Generative Model Structure

Active Inference agents maintain a generative model specified by four key matrices:

- **A matrix**: `P(o|s)` — observation model mapping states to observations `(n_obs, n_states)`
- **B tensor**: `P(s'|s,a)` — transition model for state dynamics `(n_states, n_states, n_actions)`
- **C vector**: `ln P(o)` — log-preferences over observations `(n_obs,)`
- **D vector**: `P(s)` — prior beliefs over initial states `(n_states,)`

### Expected Free Energy Decomposition

The expected free energy G for policy π decomposes into pragmatic and epistemic components:

```
G(π) = E_π[F] = E_π[KL[q(o,s)|p(o,s,C)]]
     = -pragmatic_value - epistemic_value
```

Where:
- **Pragmatic value**: Expected utility under preferred observations (exploitation)
- **Epistemic value**: Expected information gain about hidden states (exploration)

This decomposition provides a principled solution to the exploration-exploitation dilemma — agents naturally seek information when uncertainty is high and exploit when confident.

## Properties/Parameters

| Parameter | Typical Range | Description |
|-----------|---------------|-------------|
| Planning horizon | 1-8 steps | Depth of policy tree evaluation |
| Precision γ | 0.1-16.0 | Inverse temperature for action selection |
| Learning rate α | 0.01-0.5 | Rate of generative model updates |
| Prior concentration | 0.25-1.0 | Dirichlet concentration for A/B learning |
| Observation precision | 1.0-32.0 | Confidence in sensory observations |

## Implementation Details

### Action Selection

Actions are selected via softmax over negative expected free energy:

```python
def select_action(G_pi):
    """Select action minimizing expected free energy."""
    return softmax(-gamma * G_pi)  # Lower G = higher probability
```

### Belief Updating

Posterior beliefs are updated via Bayesian inference:

```python
def update_beliefs(prior, observation, A_matrix):
    """Update beliefs given observation."""
    likelihood = A_matrix[observation, :]
    posterior = prior * likelihood
    return posterior / posterior.sum()
```

### Parameter Learning

Generative model parameters are learned through gradient-based updates on variational free energy:

```python
def learn_parameters(A, B, observations, actions, learning_rate):
    """Learn A/B matrices from experience."""
    vfe = variational_free_energy(A, B, observations, actions)
    grad_A, grad_B = jax.grad(vfe, argnums=(0, 1))(A, B, observations, actions)
    return A - learning_rate * grad_A, B - learning_rate * grad_B
```

## Relationship to Reinforcement Learning

Active Inference exhibits formal equivalences with RL under specific conditions:

| RL Quantity | AIF Equivalent | Equivalence Condition |
|-------------|----------------|----------------------|
| Value V(s) | -G(π) | Epistemic value = 0 |
| Q-function Q(s,a) | -G(a) | Fully observable (A = I) |
| Reward R(s) | ln C(o) | Log-preferences = log-rewards |
| Policy π(a\|s) | softmax(-γG) | Temperature equivalence |
| Exploration ε-greedy | Epistemic value | AIF principled, RL ad-hoc |

The key insight: **Q(s,a) = -G(a)** when the observation model A is identity and epistemic value is zero [@smith2022step].

## Advantages Over Traditional RL

1. **Principled exploration**: Information-seeking emerges naturally from epistemic value
2. **Model-based**: Explicit world model enables planning and counterfactual reasoning  
3. **Unified framework**: Single principle governs perception, action, and learning
4. **Biological plausibility**: Grounded in neuroscience and predictive processing
5. **Robust to sparse rewards**: Preferences over observations rather than scalar rewards

## Relevant Projects

| Project | Implementation | Purpose |
|---------|----------------|---------|
| `alf` | Core library with AnalyticAgent, BatchAgent | JAX-native AIF with GPU scaling |
| `spinning-up-alf` | Educational notebooks 08-12 | Tutorial curriculum bridging RL↔AIF |
| `vbjax` | Whole-brain simulation | Neural mass models with AIF control |
| `alf` | Active inference agents | Hierarchical AIF, multi-agent systems |
| `neuro-nav` | Grid world environments | Spatial navigation with successor representations |

## Benchmarks and Applications

### Classic Environments

- **T-maze**: 8 states, 5 observations, 4 actions — tests memory and planning
- **Grid worlds**: Spatial navigation with partial observability
- **Multi-armed bandits**: Pure epistemic vs pragmatic value trade-offs

### Performance Metrics

```python
# Typical benchmark results (T-maze, 1000 episodes)
episodic_return = 0.85 ± 0.12      # Reward per episode
belief_accuracy = 0.92 ± 0.08      # P(correct state belief)
exploration_efficiency = 0.78       # Information gain per action
```

## Implementation Considerations

### Computational Complexity

- **Policy enumeration**: O(|A|^T) for horizon T — requires pruning for T > 5
- **Belief updating**: O(|S|²) matrix operations per timestep
- **Batch processing**: JAX vmap enables 1000+ parallel agents with sub-linear scaling

### Numerical Stability

- Use log-space computations for small probabilities
- Clip gradients during parameter learning
- Initialize A/B matrices with small random perturbations to break symmetry

### Memory Requirements

- Generative model storage: O(|S|² × |A| + |O| × |S|) for B and A matrices  
- Policy trees: Exponential in planning horizon — require approximation

## Current Research Directions

1. **Deep Active Inference**: Neural network generative models [@fountas2020deep]
2. **Hierarchical AIF**: Multi-level models with temporal abstraction
3. **Continuous control**: Extending discrete POMDP framework to continuous state-action spaces
4. **Social Active Inference**: Multi-agent coordination and communication

## Limitations

- Policy enumeration intractable for large action spaces
- Requires manual specification of preferences C
- Assumes Markovian dynamics (though hierarchical extensions address this)
- Limited theoretical guarantees compared to RL convergence proofs

## Key References

- **friston2022active**: Friston et al. (2022). Active inference and the free energy principle. Nature Reviews Neuroscience. Comprehensive overview of AIF from its originator.
- **smith2022step**: Smith et al. (2022). A Step-by-Step Tutorial on Active Inference. J Math Psychology 107:102632. Practical implementation guide.
- **george2024ratinabox**: George et al. (2024). RatInABox: A toolkit for modelling locomotion and neuronal activity in continuous environments. eLife. Spatial navigation benchmark for AIF agents.
- **Hess2025bayesian**: Hess et al. (2025). Bayesian Workflow for Generative Modeling in Computational Psychiatry. Computational Psychiatry 9:76-99.
- **weber2024pyhgf**: Weber et al. (2024). The generalized Hierarchical Gaussian Filter. pyhgf package for Bayesian filtering in JAX.

## See Also

- [method-variational-inference.md](method-variational-inference.md) — Mathematical foundations
- [method-free-energy-principle.md](method-free-energy-principle.md) — Broader theoretical framework  
- [method-belief-propagation.md](method-belief-propagation.md) — Message passing implementation
- [concept-exploration-exploitation.md](concept-exploration-exploitation.md) — Principled exploration in AIF
- [concept-generative-models.md](concept-generative-models.md) — World model learning