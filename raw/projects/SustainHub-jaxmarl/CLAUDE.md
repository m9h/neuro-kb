# SustainHub-jaxmarl

OREL extension project. Replicates Vidhi Rohira's SustainHub (GSoC 2025) MARL baseline and extends it using JaxMARL patterns for vectorized multi-agent simulation.

## Quick start

```bash
pip install -e ".[test]"
pytest tests/ -v
```

## Module map

| Module | Purpose |
|--------|---------|
| `environment.py` | JaxMARL-compatible SustainHubEnv (health, urgency, trust, burnout, stress) |
| `agents.py` | Thompson Sampling + SARSA agents (Rohira GSoC '25 baseline) |
| `metrics.py` | Community health metrics: HI, RQ, BRS, SUE, CHS |
| `train.py` | purejaxrl-style `make_train` factory with `jax.lax.scan` |

## Key patterns

- `make_train(config)` returns a pure `train(rng)` function
- `jax.jit(train)(key)` — single compiled run
- `jax.vmap(train)(keys)` — parallel seeds in one kernel launch
- `run_experiment(config, num_seeds=N)` — convenience wrapper
- `sweep_epsilon(config, epsilons)` — hyperparameter sweep

## Three-way OREL comparison

| System | Approach | This repo covers |
|--------|----------|-----------------|
| Rohira SustainHub (GSoC '25) | MARL (Thompson + SARSA) | Yes — `agents.py` |
| Bastawala LLAMOSC (GSoC '24) | MA-LLM | No — see concordia/ |
| Concordia SustainHub | LLM + Active Inference | No — see concordia/ |

## Metrics

- **HI** (Harmony Index): productivity × fairness (Gini)
- **RQ** (Resilience Quotient): post-stress / pre-stress HI
- **BRS** (Burnout Risk Score): max non-preferred task streaks
- **SUE** (Skill Utilization Efficiency): skill-task match quality
- **CHS** (Community Health Score): weighted composite

## Testing

```bash
pytest tests/ -v
```
