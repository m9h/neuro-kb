# JAX Refactoring in BETSE

This document outlines the strategy and progress of refactoring the BETSE simulation engine to support JAX acceleration and differentiable bioelectricity.

## Architecture

### 1. Unified Backend
The module `betse.lib.math.jnp` provides a proxy for either `numpy` or `jax.numpy`.
- To enable JAX, set the environment variable `BETSE_JAX=1`. This variable is read once at `src/betse/lib/math/jnp.py` import time and selects the active backend for the lifetime of the process.
- The `get_backend()` function returns the active module.
- A `set_at(arr, idx, val)` helper is provided to handle JAX's immutable array updates (`arr.at[idx].set(val)`) vs Numpy's in-place updates.
- Legacy code paths that import `numpy` directly continue to use stock NumPy regardless of `BETSE_JAX`; only modules that route through `betse.lib.math.jnp` participate in backend switching.

### 2. Functional State Management
Traditional `betse` stores simulation state in mutable class attributes (e.g., `Simulator.vm`). For JAX compatibility (JIT and Grad), the state is being refactored into immutable Pytrees.
- `SimState`: Contains dynamic state variables like `vm`, `rho_cells`.
- `CellData`: Contains static grid/geometry data like `diviterm`, `map_mem2ecm`.
- `SimParams`: Contains simulation parameters like `dt`, `cm`.

### 3. Pure Update Functions
Core physics solvers are being refactored into pure functions in `betse.science.jax.*`.
- `update_v_pure`: A JIT-compatible version of the membrane voltage update.

### 4. Inverse Design & Optimization
A new optimization engine has been introduced in `betse.science.jax.inverse` that allows for **Top-Down Anatomical Control**.
- `optimize_pattern`: Uses `jax.grad` to automatically discover biological parameters (like ion currents) that minimize the difference between the current simulation state and a target $V_{mem}$ pattern.
- Demonstrated in `tests/betse_test/a00_unit/science/test_jax_inverse.py`.

### 5. Inverse Design YAML schema
The simulation config (`src/betse/data/yaml/sim_config.yaml`) carries an `inverse:` block parsed by `SimConfInverse` in `src/betse/science/config/model/confinverse.py`. Fields:

- `enabled` (bool) — toggle gradient-based optimization for this run.
- `epochs` (int) — number of optimization epochs.
- `learning rate` (float) — gradient-descent step size.
- `loss function` (str) — loss to minimize (e.g. `mse`).
- `target pattern` (str) — name of the target bioelectric pattern (e.g. `ring`).

The connector lives in `src/betse/science/jax/bridge.py`. `optimize_from_parameters(p, cells, sim=None)` reads the parsed `p.inverse` block, packages a legacy `Cells` mesh and `Parameters` into `CellData`/`SimState`/`SimParams`, and drives `optimize_pattern`. The named `target pattern` string is mapped to a per-membrane Vmem array by `make_target_pattern(name, ...)` — currently `ring`, `gradient-x`, `gradient-y`, `flat-depolarized`, `flat-hyperpolarized`. Callers with a bespoke target field can pass `target_vm=...` to override the named factory.

## Progress
- [x] Unified Project Structure
- [x] PySide6 (Qt 6) Migration
- [x] JAX Backend Proxy
- [x] Functional State (SimState, CellData, SimParams)
- [x] JIT-compatible `update_v_pure`
- [x] JIT-compatible `update_gj_pure` (Gap Junction dynamics)
- [x] JIT-compatible `update_conc_pure` (Ion concentrations)
- [x] `jax.grad` support for parameter optimization
- [x] Inverse Design Prototype (Optimizing $V_{mem}$ patterns)
- [x] Xenobot Motility Prototype (Bioelectric Force → Movement)
- [x] Bioelectric-GRN Triggering (Vm → Transcriptional Bias)
- [x] Morphoceutical Intervention Optimization (Temporal schedules)
- [x] Large-Scale Pattern Integration (100 cells, 1000 steps < 0.1s)
- [x] Whole-Step JIT Fusion (`step_pure` consolidates all physics)
- [x] High-Performance Solver (2.2µs per step for 50 cells via `lax.scan`)
- [x] GPU-accelerated Poisson Solver (`jax.numpy.linalg.solve`)
- [x] Bio-Anatomical Bridge (Import 3D VTK/STL biomeshes)
- [x] Regulome-Bioelectric Mapping (Project AnnData/H5AD onto meshes)
- [x] Standardized Markup Support (NeuroML 2 / LEMS placeholders)
- [x] Unit-test coverage for all JAX physics kernels (run with `BETSE_JAX=1`)
- [x] Legacy `Cells`/`Parameters` → JAX `CellData`/`SimState`/`SimParams` bridge (`betse.science.jax.bridge`)
- [x] YAML `inverse:` block wired to `optimize_pattern` via `optimize_from_parameters`
- [x] GPU-friendly Laplacian inversion in motility (`jnp.linalg.solve` on the forward `lapGJ` when available; dense `lapGJinv @ b` fallback preserved)

## Testing
Run the full JAX test suite with:
```bash
BETSE_JAX=1 .venv/bin/python -m pytest tests/betse_test/a00_unit/science/
```

The suite spans the following files under `tests/betse_test/a00_unit/science/`:

- `test_jax_pure.py` — `update_v_pure` membrane-voltage kernel.
- `test_jax_conc.py` — `update_conc_pure` ion-concentration kernel.
- `test_jax_gj.py` — `update_gj_pure` gap-junction gating.
- `test_jax_grn.py` — Vm → transcriptional bias coupling.
- `test_jax_inverse.py` — `optimize_pattern` / `jax.grad` inverse design.
- `test_jax_xenobot.py` — bioelectric-force → motility prototype.
- `test_jax_morphoceutical.py` — temporal-schedule intervention optimization.
- `test_jax_integration.py` — whole-step fused `step_pure` integration.
- `test_jax_performance.py` — `lax.scan` throughput regression.
- `test_jax_biobridge.py` — VTK/STL mesh import and AnnData projection.
- `test_jax_bridge.py` — legacy `Cells`/`Parameters` → JAX `CellData`/`SimState`/`SimParams` bridge, `make_target_pattern` factory, end-to-end `optimize_from_parameters`, GPU-vs-dense motility solver parity.

## Future Work
- Implement the Poisson-driven pressure step inside `step_pure` (the `solve_poisson_pure` helper is in place but the pressure/fluid-flow body is still a placeholder — see `solver/stepper.py:46-48`).
- NeuroML 2 / LEMS support is currently placeholders only; round-trip a real LEMS file through the SBML parser path.
- Add an online integration test for `test_jax_demo` cloud-fetch (gated today by `BETSE_JAX_DEMO_NETWORK=1`).
- Bridge SimState from a fully-initialised legacy `Simulator` snapshot end-to-end (the current bridge accepts `sim=None` and starts from a flat resting state; pulling live `vm`/`cc_cells` is wired but unexercised in tests).
