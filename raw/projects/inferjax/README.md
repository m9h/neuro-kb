# inferjax

**Amortized simulation-based inference and posterior calibration for any JAX simulator.**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![JAX](https://img.shields.io/badge/JAX-Accelerated-9cf)](https://github.com/google/jax)

`inferjax` is the simulator-agnostic half of a simulation-based inference
workflow: wrap a forward model, train a posterior estimator, and check that
the result is actually calibrated. It knows nothing about any particular
measurement modality — a diffusion-MRI signal vector, an evoked-response
waveform and a power spectrum are all just `(m,)` arrays.

It exists because that machinery was written twice: once inside a diffusion
MRI codebase and once needed by an M/EEG one. Shared code in a shared package
beats two copies drifting apart.

## Install

```bash
uv add inferjax                    # core
uv add "inferjax[flows]"           # + flowjax normalizing flows
uv add "inferjax[mcmc]"            # + blackjax NUTS reference posteriors
```

## Core idea

A `Simulator` composes four injected pieces. Only the first is domain-specific.

```python
import jax, jax.numpy as jnp
from inferjax import Simulator, ReferenceScale

def forward(theta, context):
    """(theta, context) -> signal. Called under vmap, so keep it traceable."""
    return theta[0] * jnp.exp(-context * theta[1])

sim = Simulator(
    forward_fn=forward,
    parameter_names=["amplitude", "rate"],
    parameter_ranges={"amplitude": (0.5, 1.5), "rate": (0.0, 3.0)},
    signal_dim=64,
    context=jnp.linspace(0.0, 1.0, 64),
    noise="gaussian",              # or "rician", "none", or a NoiseModel
    snr_range=(10.0, 50.0),        # per-sample SNR augmentation
    normalizer=ReferenceScale(mask),
)

theta, x = sim.sample_and_simulate(jax.random.PRNGKey(0), n=10_000)
```

### Normalisation is a first-class object

Train-time and inference-time preprocessing **must match**, or the density
estimator sees a different distribution than it was trained on. That invariant
is easy to state and easy to break when the arithmetic is repeated at every
call site. Here it is one object, applied by `sample_and_simulate` and exposed
as `sim.normalize()` for observed data:

```python
posterior_samples = trained.sample(key, condition=sim.normalize(observed))
```

| Normalizer | Operation | Typical use |
|---|---|---|
| `Identity` | none | default |
| `ReferenceScale(mask)` | divide by mean of masked channels | diffusion MRI b0 normalisation |
| `BaselineSubtract(mask)` | subtract mean of masked channels | evoked responses, pre-stimulus baseline |

### Noise models

`Gaussian`, `Rician`, and `NoNoise`, all sharing a `SigmaSchedule` that
supports fixed SNR, per-sample SNR augmentation, and curriculum ramping
(train on high SNR first, introduce low SNR progressively):

```python
sim.curriculum_step = (step, total_steps)   # during training
```

## Status

Early. The simulator core, noise models and normalisation strategies are
implemented and tested. Posterior estimators (normalizing flows, MDN,
score-based), calibration diagnostics (SBC, PPC, conformal, OOD) and
sequential NPE are being migrated in.

## Development

```bash
uv sync --extra flows --group dev
uv run pytest tests/ -q
```

## License

MIT
