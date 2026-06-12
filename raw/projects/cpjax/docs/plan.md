# cpjax — phase plan

Authoritative plan lives in [`anatomical-compiler/docs/differentiable-cp.md`](https://github.com/biopunk-lab/anatomical-compiler/blob/master/docs/differentiable-cp.md).
This file is the short local index so a future agent opening just `cpjax`
sees where everything sits.

## Phase 0 — Oracle harness *(in progress)*

- `src/cpjax/oracles/cc3d.py`, `src/cpjax/oracles/morpheus.py` — run CC3D / Morpheus, dump trajectories.
- `src/cpjax/benchmarks/` — the 5 canonical cases (cell-sorting, vasculogenesis, chemotaxis, gastrulation, tumour-stroma).
- **Deliverable independent of differentiability:** the cached oracle trajectories are reusable as a JAX-callable testbed for the rest of the project.

### Phase 0 status (TDD-driven, 118 tests green + 1 skipped)

| component | status |
|---|---|
| `oracles/_trajectory.py` — Trajectory dataclass + validation | **green** (6 tests) |
| `oracles/_manifest.py` — Manifest + SHA-256 hashing + checksums | **green** (7 tests) |
| `oracles/_loader.py` — shared cache loader with checksum-verify | **green** |
| `oracles/_dispatch.py` — binary lookup, version probe, pre-dispatch manifest | **green** |
| `oracles/_vtk_parse.py` — ASCII STRUCTURED_POINTS parser (FIELD + SCALARS forms) | **green** (7 tests) |
| `oracles/cc3d.py` — load_trajectory + run + VTK→trajectory.npy post-processing | **green** (5 unit + 4 integration) |
| `oracles/morpheus.py` — load_trajectory + run + VtkPlotter post-processing | **green** (5 unit + 3 integration) |
| `benchmarks/cell_sorting.py` — Graner & Glazier 1992, aligned with vendored CC3D demo | **green** (8 tests) |
| `benchmarks/_specs/cell_sorting/` — vendored CC3D `cellsort_2D` demo + authored Morpheus spec (MIT) | **green** (real-oracle parity tests pass for both oracles) |
| `benchmarks/vasculogenesis.py` + Morpheus spec — Merks & Glazier 2005, VEGF chemotaxis | **green** (8 unit + 2 oracle parity) |
| `benchmarks/chemotaxis.py` + Morpheus spec — Savill & Hogeweg 1997, static-gradient migration | **green** (9 unit + 2 oracle parity) |
| `benchmarks/gastrulation_toy.py` + Morpheus spec — Drasdo & Forgacs 2000, two-layer DAH folding | **green** (9 unit + 2 oracle parity) |
| `benchmarks/tumour_stroma.py` + Morpheus spec — Szabó & Merks 2013, heterotypic adhesion (non-invasive) | **green** (8 unit + 2 oracle parity) |
| Benchmark `provenance.toml` for all 5 | **green** (15 tests, CURE-compliance) |
| `tests/integration/` with skipif-gated real-oracle tests | **green** (gated on binary availability) |
| Real CC3D end-to-end cell-sorting parity (the canonical Phase-0 deliverable) | **green** |
| Real Morpheus end-to-end cell-sorting parity (authored MorpheusML spec) | **green** |

### Phase 0 acceptance milestone reached for **both** oracles

`test_cell_sorting_oracle_run_actually_sorts` (CC3D) and
`test_cell_sorting_morpheus_oracle_run_actually_sorts` (Morpheus) both run a
real cell-sorting model end-to-end through the cpjax wrappers, parse the
resulting VTK frame series, and verify the cluster-count acceptance metric
decreases by ≥30% from initial to final state — i.e., sorting actually
happens under real oracle dynamics for both oracles.

### Phase 0 complete

All five benchmarks have:
- Python module exporting `oracle_config`, `init_lattice`, `ground_truth`, `acceptance_metric`
- Unit tests (TDD-driven)
- `provenance.toml` with `status = "implemented"`, populated `[original_params]`, paper citation
- A working Morpheus model spec under `_specs/{name}/{name}.morpheus.xml`
- Two integration tests gated by `requires_morpheus` + `requires_*_spec`:
  one round-trip + one behaviour test. All pass against real Morpheus 2.4.

`cell_sorting` has a vendored CC3D `cellsort_2D` demo and matching CC3D
parity tests. The other four benchmarks now also have authored CC3D
specs (landed 2026-05-15) plus 8 additional parity tests
(round-trip + behaviour per benchmark, gated on the CC3D binary and
spec presence). All five benchmarks now have **dual-oracle parity**:

| benchmark | CC3D | Morpheus | parity tests |
|---|---|---|---|
| cell_sorting | vendored from CC3D 4.8 demo | authored | 4 |
| vasculogenesis | authored | authored | 4 |
| chemotaxis | authored (Lambda=10 — see NOTICE for normalisation) | authored | 4 |
| gastrulation_toy | authored | authored | 4 |
| tumour_stroma | authored | authored | 4 |

Phase 0 done. Phase 1 (forward-only JAX CP) is unblocked.

## Phase 1 — Forward-only JAX CP *(spine complete, 154 tests green)*

| component | status |
|---|---|
| `lattice.py` — `Lattice` PyTree, `init`, `neighbours`, `area`, `perimeter`, jittable | **green** (11 tests) |
| `hamiltonian.py` — `Hamiltonian` PyTree, `adhesion_energy`, `area_energy`, `energy`, `delta_h`; differentiable in J | **green** (9 tests) |
| `update.py::metropolis_step` — `lax.scan` over single-site attempts, jit-compileable, closed-form local ΔH | **green** (7 tests) |
| `simulate.py::run` + `run_batched` — `lax.scan` over MCS, static kernel dispatch, record_every stride | **green** (7 tests) |
| Forward-parity vs Morpheus cell_sorting oracle (qualitative) | **green** (2 tests) |

Phase-1 sorting demo: cpjax JAX-native CP on 32×32 lattice with cell_sorting
ground-truth parameters drives the cluster count from 5 → 2 in 30 MCS — same
qualitative DAH-sorting behaviour the Morpheus oracle produces.

**Phase 1 limitation (Phase 1.5 target):** the state field encodes cell-*type*
labels, not unique per-cell ids. Real CC3D/Morpheus track each cell as a
distinct entity (CellId) with a separate CellType lookup. The Steinberg
sorting dynamics emerge correctly (J-matrix term is symmetric) but
cell-granularity-dependent metrics (per-cell area constraint, perimeter)
look qualitatively right but quantitatively different from the oracle —
Morpheus has ~64 individual cells each at target_A=25; cpjax has effectively
2 "cells" (one per type). Phase 1.5 separates cell-id from cell-type and
gets quantitative parity.

## Phase 1.5 — Separate cell-id from cell-type *(landed)*

- `Lattice` gained an optional ``cell_type`` field — index `i` maps cell-id
  `i` to its type. None = Phase-1 behaviour (state is type directly).
- New `init_kind="unique_voronoi"` — Voronoi-fill central disc, each cell
  gets a unique id, types round-robined.
- `hamiltonian.adhesion_energy_typed` / `energy_typed` / `delta_h_typed`
  variants accept `cell_type` and index `J[type[state], type[neighbor]]`.
- `update.metropolis_step` and `_local_delta_h` take an optional `cell_type`.
- `simulate.run` carries `lattice0.cell_type` through to the kernel.

Phase-1.5 demonstration: with `unique_voronoi` init and 8 cells at
target_A=25, the cells reach ~25 sites each (200 total) rather than the
~50 we saw in Phase 1 where the whole type was being pulled to target 25.

## Phase 2 — REINFORCE gradient *(spine landed)*

- `update.reinforce_step` returns `(next_state, next_key, log_prob)`.
  Per-attempt log-prob is closed-form (`min(0, -ΔH/T)` for accept,
  `log(-expm1(min(0, -ΔH/T)))` for reject). Differentiable in `params.J`,
  `lam_A`, `target_A`.
- `inverse.reinforce_grad(loss_fn, theta, n_seeds, baseline)` returns the
  variance-reduced score-function gradient. Baselines: `"none"`,
  `"running_mean"`.
- `inverse.recover_adhesion(trajectory, theta_init, n_steps, kernel)` —
  gradient-descent on shape distance using REINFORCE.

**Phase-2 spine vs Phase-2 acceptance criterion**: the spine (all three
functions implemented + tested) is done.

**Phase-2 convergence directionality**: also done. ``reinforce_grad`` has
a ``rollout_fn`` mode that runs fresh stochastic rollouts per seed via
``reinforce_step``, so different seeds produce different trajectories →
real loss variance → real REINFORCE gradient. Demonstrated on a single-J-
entry recovery: with truth J[1,2]=11 and guess J[1,2]=9.0 (just above
the DAH phase boundary at J=8), 10 gradient-descent steps move J[1,2]
from 9.000 → 9.101, *consistently in the right direction*. Below the
phase boundary (J<8) the loss landscape is flat (cells never sort, loss
is constant in J) and REINFORCE has no signal — that's a property of the
problem, not a bug in the gradient.

**Phase-2 formal acceptance** *(achieved at 9.4% rel_err)*.

Path from 19.3% to 9.4%, across five v1→v5 benchmark configurations:

| run | strategy | result |
|---|---|---|
| v1 | vanilla SGD, joint J update | 19.3% — froze in flat-gradient basin |
| v2 | Adam joint, high LR (0.3) | diverged (J[1,1] overshot to 5.0) |
| v3 | vanilla SGD per-entry | 18.5% — J[1,2] flat-gradient unchanged |
| v4 | Adam per-entry, lr=0.2 | 12.1% — J[1,1]=0%, J[2,2]=8%, J[1,2]=15% |
| **v5** | **Adam-per-entry warm-start on J[1,2], lr=0.1, n_seeds=48, NO bias-correction** | **9.4% best at step 6** ✓ |

Key insights surfaced during tuning:

* **Per-entry recovery is necessary** — joint Adam diverges on the weakly-
  identifiable homotypic entries when momentum amplifies REINFORCE noise.
* **J[1,2] (the DAH-threshold entry) is the load-bearing one.** Cluster-
  count metric is dominantly sensitive to J[1,2] near the phase boundary;
  homotypic entries have weaker gradient but recover cleanly with Adam.
* **Disable Adam's bias correction** when warm-starting from a non-zero
  parameter — the bias correction makes the first step always ≈ ±lr
  regardless of gradient magnitude, which wrecks the warm-start advantage.

Recipe captured in [[project-phase2-10pct-recipe]] memory.

## Phase 2 — REINFORCE gradient

- `src/cpjax/update.py::reinforce_step`, `src/cpjax/inverse.py::reinforce_grad`, `src/cpjax/inverse.py::recover_adhesion`.
- Inverse benchmark: recover known adhesion energies on the cell-sorting case.

## Phase 3 — Relaxed gradient kernels *(both landed; bake-off done)*

| kernel | landed | tests | wall-clock per MCS (24×24) | gradient path | physics |
|---|---|---|---|---|---|
| Phase 1 metropolis | yes | 7 | 120 ms (baseline) | none | exact CPM |
| Phase 2 reinforce | yes | 8 | 120 ms (1.0×) | score-function (high variance) | exact CPM |
| Phase 3b gumbel-softmax | yes | 7 | 137 ms (1.1×) | none (int state severs grad) | Metropolis with Gumbel-noise accepts |
| Phase 3c soft-lattice | yes | 9 | **2.9 ms (0.024× = 40× faster)** | direct ``jax.grad`` | structurally biased relaxation |

### Bake-off verdict (per design doc §4)

**Winner: soft-lattice (Phase 3c).** Two-orders-of-magnitude faster per step
than the discrete kernels (vectorized gradient-descent on the whole lattice
vs. ``lax.scan`` over H×W sequential single-site attempts), and the
gradient flows naturally through ``jax.grad`` without any straight-through
hackery. The bias of the relaxation (different physics in detail than
hard CP) is the price; it's documented and bounded.

**Reference implementations:**
- *Phase 2 REINFORCE*: kept as the always-correct fallback. Exact CPM
  dynamics, score-function gradient. Slow but bias-free.
- *Phase 3b Gumbel-softmax (int state)*: stochastic exploration kernel for
  annealing schedules — useful but not a gradient path. A soft-state
  Gumbel variant (Phase 3.5, Task #33) would combine 3b's stochasticity
  with 3c's differentiability.

### Phase 3.5 — soft-state Gumbel *(landed)*

``update.soft_gumbel_step`` — same (H, W, K) logits state as soft-lattice,
but each step samples Gumbel(0, 1) noise, computes the energy on the
Gumbel-softmax-perturbed soft state, and gradient-descents on that
*perturbed* energy. Combines 3c's differentiability (gradients flow
naturally through ``jax.grad``) with 3b's stochastic exploration (different
keys produce different updates, helping escape local minima).

8 tests green. Registered as ``kernel="soft_gumbel"`` in ``simulate.run``.

## Phase 4 — MorpheusML import/export *(both landed)*

`io_morpheus.read(path) → MorpheusModel` parses the CP-proper subset of
MorpheusML (Space/Lattice, CellTypes/VolumeConstraint, CPM/Interaction/Contact,
MonteCarloSampler/MetropolisKinetics). `MorpheusModel.to_jax() → (Lattice,
Hamiltonian)`.

`io_morpheus.write(model, path)` serialises a `MorpheusModel` back to
MorpheusML XML. Emits a minimal-but-runnable Morpheus 2.x file:
the CP-proper subset the reader covers, plus default
`Time`/`CellPopulations`/`Analysis` blocks chosen so the output is a
valid standalone model. The headline use case is *fitted-parameter
export* — parse a source spec, fit J/lam_A/target_A/temperature in
cpjax, `dataclasses.replace(...)`, write back to share with the
COMBINE / Morpheus community.

41 tests green (14 reader + 27 writer). Round-trip
``read(write(read(spec))) == read(spec)`` confirmed on all 5 authored
benchmark specs: J matrix, target_A, lam_A, temperature, celltype names,
lattice shape, neighbour order, and boundary all preserved to float
tolerance.

Scope clearly bounded: out-of-scope for v0 are `<Global>` PDE systems,
`<Chemotaxis>` (Phase 5 with bioelectric coupling), `<CellPopulations>`
initialisation (user supplies state separately), `<Analysis>` plugins.

**External-spec import demonstrators** *(both landed)*. Two vendored
real-world MorpheusML models live under
``src/cpjax/benchmarks/_specs/_external/`` with paired demonstrators:

| model | source | demonstrator | regression tests |
|---|---|---|---|
| ``MODEL2009210001_TwoLayerCircuit`` (Mulberry & Edelstein-Keshet 2020) | BioModels (CC0) | ``examples/08_biomodels_import.py`` | 5 in ``tests/test_phase4_biomodels_import.py`` |
| ``VascularPatterning`` (Köhn-Luque et al. 2013) | Morpheus repo Examples/Multiscale (BSD-3) | ``examples/09_morpheus_repo_vasculogenesis.py`` | 5 in ``tests/test_phase4_morpheus_repo_import.py`` |

Each demonstrator parses the spec verbatim through
``io_morpheus.read``, runs ``to_jax`` to get a ``(Lattice, Hamiltonian)``
pair, executes a 5-step soft-lattice smoke rollout, and emits a
diagnostic that buckets every MorpheusML element into Phase-4 capture
(consumed end-to-end into ``(Lattice, Hamiltonian)``), Phase-6 capture
(parsed verbatim into the :class:`MorpheusModel` parse layer; not yet
evaluated), or still-lost (no consumer yet anywhere in cpjax). For any
future MorpheusML model, users can see at a glance which features
cpjax already handles and which would need a Phase-6.1+ extension.

## Phase 5 — BETSE-JAX coupling *(spine + 2 demonstrators landed)*

| component | tests | status |
|---|---|---|
| ``hamiltonian.effective_J(params, vmem)`` — apply user-supplied V_mem coupling | 3 | green |
| ``hamiltonian.energy_with_vmem(state, params, vmem)`` — V_mem-modulated adhesion | 3 | green |
| ``inverse.shape_inverse(...)`` — gradient descent on V_mem with soft-lattice rollout (SGD + Adam paths) | 4 | green |
| ``examples/06_bioelectric_coupling.py`` — synthetic-target demonstrator | 3 | green |
| ``examples/07_cell_sorting_inverse.py`` — real-benchmark demonstrator (cell_sorting DAH restoration) | 3 | green |

The architectural payoff: target morphology → ``jax.grad`` → soft-lattice
dynamics → ``effective_J(vmem)`` coupling → back to V_mem pattern.
End-to-end differentiable shape control from bioelectric prepattern.

**Phase-5 demonstrator landed** *(99.9% shape distance reduction; 89% V_mem
error reduction)*. ``examples/06_bioelectric_coupling.py`` picks a known
true V_mem, rolls the soft lattice forward to produce a target shape, then
recovers V_mem from a wrong initial guess via ``shape_inverse(optimizer="adam")``.

Why Adam, not the original SGD path: the bilinear V_mem coupling
amplifies the loss gradient sharply (gradients of order O(150) on a
loss of order O(1000)). The SGD path with the per-step ±0.5 clip
saturates and **oscillates** between two clipped states — the same
failure mode the Phase-2 acceptance work hit. ``shape_inverse`` now
exposes ``optimizer="adam"``, which adaptively rescales by the running
RMS of past gradients and converges cleanly.

**Coupling near-degeneracy**: the bilinear form
``J_eff = J_base * (1 + vmem[i] + vmem[j])`` is invariant under adding a
constant to every V_mem component (modulo the area term), so V_mem
recovery saturates faster than shape recovery. That's a property of the
coupling parameterization, not the optimizer — documented in the
demonstrator's output. Regression-pinned in
``tests/test_phase5_demonstrator.py``.

**Real-benchmark demonstrator** *(notebook 07, cell_sorting)*. The
headline application: take the cell_sorting ground-truth J, knock
``J[1, 2]`` below the DAH envelopment threshold (11 → 6), and ask the
optimizer to find the V_mem pattern that restores sorting. With an
off-diagonal multiplicative coupling (which is locally more expressive
than the bilinear form for this problem), the optimizer:

* recovers effective ``J[1, 2]`` to **10.98** (target 11.0, 0.2% error)
* restores the DAH envelopment condition (21.95 > 18.0)
* closes 100% of the shape-distance gap

The found V_mem has medium ≈ −0.42 and both biological types ≈ +0.41 —
exactly the gauge in which medium-cell adhesion stays invariant while
heterotypic adhesion is boosted. This is the "bioelectric prepattern
instructs morphology" story end-to-end on a canonical CP benchmark.
Regression-pinned in ``tests/test_phase5_cell_sorting_inverse.py``.

## Phase 6 — MorpheusML PDE/Chemotaxis parse layer *(landed)*

Phase 6.0 closes the gap between "the reader parses CP-proper" and
"the reader parses every MorpheusML element the two vendored external
specs use." The reader now lands the following onto the
:class:`MorpheusModel` as structured data, verbatim from XML:

* ``Global/Field`` — lattice-PDE state variables with declared
  ``Diffusion`` rate. Captured as :class:`Field`.
* ``Global/Variable`` — global scalar state. Captured as
  :class:`Variable` (the BioModels MODEL2009210001 style).
* ``Global/Constant`` — global symbolic constants.
* ``Global/System`` — ODE/algebraic system block with its ``solver``,
  ``time-step``, ``Constant``s, and ``DiffEqn``/``Rule`` entries.
  Each :class:`Equation` preserves its raw ``Expression`` string.
* Per-``CellType`` ``Property``, ``Chemotaxis``, ``System``, and
  ``NeighborhoodReporter``.

| component | tests | status |
|---|---|---|
| New dataclasses (Field, Variable, Constant, Equation, System, Property, Chemotaxis, NeighborhoodReporter) on ``MorpheusModel`` | covered by 13 | green |
| ``_parse_global`` helper — Field, Variable, global Constants, Systems | 4 | green |
| ``_ingest_celltype_extras`` helper — Property, Chemotaxis, NeighborhoodReporter, per-cell System | 5 | green |
| Authored vasculogenesis + chemotaxis specs now expose their Field + Chemotaxis layer through the same parse surface | 2 | green |
| Three CP-pure authored specs (cell_sorting / gastrulation_toy / tumour_stroma) remain empty under the Phase-6 attributes — backward compatibility | 1 | green |

**Explicit non-goals for Phase 6.0:**

1. **Expression evaluation**: ``Expression`` strings are kept verbatim
   as Python ``str``. A Phase-6.1 MuParser-style evaluator (or a
   ``sympy``-backed adapter) is the natural next layer.
2. **PDE integration**: ``Field`` diffusion + reaction terms are not
   yet stepped. A Phase-6.1 JAX finite-difference Laplacian + Heun
   step is the natural next layer.
3. **Chemotaxis coupling**: ``Chemotaxis`` is captured but not yet
   added into the Hamiltonian's ``delta_h``. Phase-5's bioelectric
   coupling spine is the natural place to plug a static-gradient or
   field-coupled chemotaxis adapter in.
4. **Writer round-trip of the Phase-6 surface**: the writer still
   emits only the CP-proper subset. The Phase-4 round-trip
   ``read(write(read(spec))) == read(spec)`` continues to hold on
   that subset; round-trip of the Phase-6 surface is deferred.

Demonstrators 08 and 09 now report a three-bucket diagnostic
(Phase-4 capture / Phase-6 capture / still-lost) rather than a
two-bucket "supported / lost" diagnostic. Both demonstrators continue
to pass their regression tests after the bucket refactor.

## Status

**Phases 0–6.0 landed.** 261 tests passing, 1 skipped (real-oracle integration,
gated on local CC3D/Morpheus binaries). Soft-lattice elected the production
kernel after the Phase-3 bake-off. Phase-2 acceptance criterion hit on all
five benchmarks; MorpheusML round-trip parity confirmed on all five authored
specs plus two vendored external specs (BioModels MODEL2009210001 and the
Morpheus-repo VascularPatterning example); BETSE-JAX V_mem→shape coupling
spine in place. The reader now captures every MorpheusML element the two
vendored external specs use; evaluating the parsed Expressions and stepping
the parsed PDE Fields is the Phase-6.1 follow-up.

Other follow-ups: ongoing wet-lab data integration as
[`anatomical-compiler/docs/wetlab-program.md`](https://github.com/biopunk-lab/anatomical-compiler/blob/master/docs/wetlab-program.md)
cycles produce segmented shape data.
