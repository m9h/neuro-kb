# cpjax examples

End-to-end demonstrators for each phase. The underlying capability
(oracle parity, forward parity, inverse recovery, bioelectric coupling)
is implemented and tested in ``src/cpjax/`` and ``tests/``; the entries
below are reader-friendly executables that exercise the public API.

| # | demonstrator | underlying capability | form |
|---|---|---|---|
| 1 | `01_cell_sorting`      | Graner & Glazier 1992 — forward parity + Phase-2 J recovery | not yet bundled — see `tests/test_cell_sorting.py`, `tests/integration/test_cell_sorting_oracle_parity.py`, `scripts/phase2_acceptance_benchmark_v5.py` |
| 2 | `02_vasculogenesis`    | Merks & Glazier 2005 — forward + J[1,1] recovery | not yet bundled — see `tests/test_vasculogenesis.py`, `scripts/phase2_vasculogenesis_benchmark_v2.py` |
| 3 | `03_chemotaxis`        | Savill & Hogeweg 1997 — single + collective | not yet bundled — see `tests/test_chemotaxis.py` |
| 4 | `04_gastrulation_toy`  | Drasdo & Forgacs 2000 — staged J recovery | not yet bundled — see `tests/test_gastrulation_toy.py`, `scripts/phase2_gastrulation_toy_benchmark.py` |
| 5 | `05_tumour_stroma`     | Szabó & Merks 2013 — joint J Frobenius | not yet bundled — see `tests/test_tumour_stroma.py`, `scripts/phase2_tumour_stroma_benchmark.py` |
| 6 | [`06_bioelectric_coupling.py`](06_bioelectric_coupling.py) | Phase 5 spine: synthetic V_mem recovery | **landed** (script + tests/test_phase5_demonstrator.py) |
| 7 | [`07_cell_sorting_inverse.py`](07_cell_sorting_inverse.py) | Phase 5 on real benchmark: V_mem restores cell_sorting DAH | **landed** (script + tests/test_phase5_cell_sorting_inverse.py) |

## Notebook 6 — Phase 5 demonstrator (landed)

End-to-end bioelectric → shape control. Given a *known true* V_mem, the
script rolls the soft lattice forward to produce a target shape, then asks
``inverse.shape_inverse`` (Adam, with a wrong initial guess) to recover a
V_mem whose rollout matches the target.

```bash
python examples/06_bioelectric_coupling.py                                    # numbers
python examples/06_bioelectric_coupling.py --save examples/06_bioelectric_coupling.png   # + figure
```

Convergence on the default config (80 outer steps, 20 inner soft-lattice
steps, Adam lr=0.05):

* Shape distance to target: **1216 → 1.0** (99.9% reduction)
* V_mem error vs ground truth: **0.36 → 0.04** (89% reduction)

The bilinear coupling ``J_eff = J_base · (1 + vmem[i] + vmem[j])`` has a
near-degeneracy under constant offsets in V_mem, so V_mem recovery saturates
faster than shape recovery — that's a property of the coupling, not the
optimizer. Regression-pinned in ``tests/test_phase5_demonstrator.py``.

## Notebook 7 — Phase 5 on a real benchmark (landed)

The headline Phase-5 result: a bioelectric prepattern restores
DAH-driven sorting on the canonical cell_sorting benchmark after the
heterotypic adhesion energy has been *broken* below the envelopment
threshold.

```bash
python examples/07_cell_sorting_inverse.py
python examples/07_cell_sorting_inverse.py --save examples/07_cell_sorting_inverse.png
```

Setup: take the ground-truth J from ``cpjax.benchmarks.cell_sorting``,
knock ``J[1, 2]`` from 11 down to 6 (breaks ``2·J[1,2] > J[1,1] + J[2,2]``),
and use an off-diagonal multiplicative coupling
``J_eff[i, j] = J_base[i, j] · (1 + v[i] + v[j])`` for ``i ≠ j``.
``inverse.shape_inverse`` (Adam) is asked to find the V_mem pattern
that, applied to the broken J, drives soft-lattice dynamics back to the
ground-truth equilibrium.

Result:

* Shape distance to target: **1211 → 0.18** (100% reduction)
* Effective ``J[1, 2]`` after optimization: **10.98** (ground truth 11.0)
* DAH envelopment condition: **restored** (21.95 > 18.0)
* V_mem: ``medium = −0.42, Condensing = +0.41, NonCondensing = +0.42``
  — the optimizer found the gauge in which medium-cell adhesion stays
  invariant while heterotypic adhesion is boosted.

Regression-pinned in ``tests/test_phase5_cell_sorting_inverse.py``.

## Demonstrators 1–5 — not yet bundled

The full Phase-0 oracle parity, Phase-1 forward parity, and Phase-2
inverse-recovery results exist as tests and scripts (see the table above
for paths). A future pass will package them as runnable demonstrators
similar to ``06_bioelectric_coupling.py``. Each will:

1. Load the oracle trajectory (CC3D or Morpheus output, cached `.npy`).
2. Run the same model in `cpjax` (Phase 1 forward kernel).
3. Compute the forward-parity metric (cluster count, mean cluster size,
   anisotropy, perimeter/area).
4. Recover the known Hamiltonian parameters from a fragment of the oracle
   trajectory; compare to ground truth (Phase 2 acceptance recipe).

See [`anatomical-compiler/docs/differentiable-cp.md`](https://github.com/biopunk-lab/anatomical-compiler/blob/master/docs/differentiable-cp.md)
for the full plan and §6 for the BioModels benchmark provenance.
