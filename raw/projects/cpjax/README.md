# 🧬 cpjax — Differentiable Cellular Potts in JAX

[![Status: phases 0–5 landed](https://img.shields.io/badge/status-phases%200--5%20landed-brightgreen)](docs/plan.md)
[![Tests: 265 passing](https://img.shields.io/badge/tests-265%20passing%20%2F%201%20skipped-brightgreen)](tests/)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue)](pyproject.toml)
[![JAX 0.10+](https://img.shields.io/badge/jax-0.10%2B-orange)](pyproject.toml)
[![License: Apache 2.0](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![CURE-compliant](https://img.shields.io/badge/CURE-compliant-purple)](src/cpjax/benchmarks/_provenance.py)
[![MorpheusML import](https://img.shields.io/badge/MorpheusML-import-ff69b4)](src/cpjax/io_morpheus.py)

> **A JAX-native Cellular Potts simulator with end-to-end gradients through the Hamiltonian dynamics — the differentiable shape-layer for inverse morphology design.**

`cpjax` is the differentiable sister to [BETSE-JAX](https://github.com/biopunk-lab/betse-unified) (bioelectric) and a substrate component of the [anatomical-compiler](https://github.com/biopunk-lab/anatomical-compiler) regulome project. Where BETSE-JAX makes Vₘₑₘ inverse-designable, **`cpjax` makes cell shape, division, adhesion, and movement inverse-designable** — closing the regulome → form gap.

---

## ✨ What's inside

| layer | what you get |
|---|---|
| 🟢 **Forward simulator** | `jax.jit` / `lax.scan` CPM on 2D lattices, periodic BCs, Moore + von Neumann neighbourhoods |
| 🟢 **Three gradient kernels** | exact REINFORCE • soft-lattice relaxation (40× faster, winner) • soft-state Gumbel-softmax |
| 🟢 **Five canonical benchmarks** | cell sorting, vasculogenesis, chemotaxis, gastrulation toy, tumour–stroma — all with oracle parity |
| 🟢 **Oracle harness** | run real CC3D and Morpheus, parse VTK frames, cache trajectories, SHA-256 manifests |
| 🟢 **MorpheusML import + export** | `io_morpheus.read` parses BioModels-CP / Morpheus specs to JAX; `io_morpheus.write` exports fitted params back to MorpheusML XML for community sharing |
| 🟢 **MorpheusML PDE/Chemotaxis parse layer** | reader captures `Global/Field`+`Diffusion`+`System`, `Variable`, per-CellType `Property`/`Chemotaxis`/`NeighborhoodReporter` as structured data on `MorpheusModel` (parsed verbatim; Phase 6.1 evaluator pending) |
| 🟢 **Inverse design** | `recover_adhesion()` from observed trajectories • `shape_inverse()` from V_mem prepattern |
| 🟢 **BETSE-JAX coupling** | `effective_J(params, vmem)` — V_mem-modulated adhesion, end-to-end differentiable through shape physics |

---

## 🚀 Quick start

```bash
pip install -e .
pytest -q                    # 261 passed, 1 skipped (real-oracle integration)
```

### Forward simulation

```python
import jax
from cpjax.lattice import Lattice
from cpjax.hamiltonian import Hamiltonian
from cpjax.simulate import run

# Build a 32×32 lattice initialised as Voronoi-filled cells around the centre.
lattice0 = Lattice.init(shape=(32, 32), n_cells=8, kind="unique_voronoi", seed=0)

params = Hamiltonian(
    J=jax.numpy.array([[0., 16., 16.], [16., 2., 11.], [16., 11., 16.]]),  # cell_sorting truth
    lam_A=2.0, target_A=25.0, temperature=10.0,
)

trajectory = run(lattice0, params, n_mcs=100, kernel="metropolis", seed=0)
# trajectory: (n_frames, H, W) — same shape Morpheus/CC3D would produce.
```

### Inverse: recover adhesion energies from observed trajectories

```python
from cpjax.inverse import recover_adhesion
from cpjax.benchmarks import cell_sorting

oracle_traj = ...  # cached from CC3D or Morpheus via cpjax.oracles
theta_init = Hamiltonian(J=initial_guess_J, lam_A=2.0, target_A=25.0)

theta_hat = recover_adhesion(
    oracle_traj, theta_init, n_steps=40,
    kernel="reinforce", n_seeds=48, lr=0.1,
)
# theta_hat.J should converge to cell_sorting.ground_truth()["J"] within ~10% rel_err.
```

### Inverse: design V_mem prepattern for a target shape

```python
from cpjax.inverse import shape_inverse

vmem_hat = shape_inverse(
    target_state=target_lattice,
    init_state=init_lattice,
    theta=hamiltonian_with_vmem_coupling,
    vmem_init=initial_vmem_pattern,
    n_steps=200, lr=0.05,
)
# Gradient descent flows through soft-lattice rollout → effective_J(vmem) → vmem.
```

---

## 🧪 The five benchmarks

Every benchmark exposes the same four-function contract — `oracle_config()`, `init_lattice()`, `ground_truth()`, `acceptance_metric()` — and ships with a `provenance.toml` (paper citation, original params, CURE compliance) plus a Morpheus model spec for oracle parity tests.

| benchmark | source | biology | acceptance metric |
|---|---|---|---|
| 🔵 **cell_sorting** | Graner & Glazier 1992 | DAH-driven envelopment, 2 cell types | cluster count ↓ |
| 🩸 **vasculogenesis** | Merks & Glazier 2005 | VEGF chemotaxis → endothelial network | endo–medium perimeter |
| 🧭 **chemotaxis** | Savill & Hogeweg 1997 | single + collective migration up gradient | centroid drift |
| 🥚 **gastrulation_toy** | Drasdo & Forgacs 2000 | two-layer DAH folding | interface curvature |
| 🧫 **tumour_stroma** | Szabó & Merks 2013 | heterotypic adhesion, non-invasive | compactness vs invasion |

All five benchmarks have **dual-oracle parity** — authored CC3D + Morpheus specs and 4 parity tests each (round-trip + behaviour, per oracle). `cell_sorting` additionally ships a vendored CC3D `cellsort_2D` demo (MIT-licensed) as the canonical Phase-0 reference.

---

## 🧠 Gradient strategy — three kernels, one winner

The design doc reserved space for three update kernels. All three are implemented and benchmarked:

| kernel | gradient path | wall-clock / MCS (24×24) | bias | use it for |
|---|---|---:|---|---|
| Phase 1 `metropolis` | none (forward only) | 120 ms (baseline) | exact CPM | sanity, oracle parity |
| Phase 2 `reinforce` | score-function | 120 ms (1.0×) | **none** | correctness reference, exact CPM gradients |
| Phase 3b `gumbel_softmax` | none (int state severs grad) | 137 ms (1.1×) | low | stochastic annealing |
| 🏆 Phase 3c `soft_lattice` | direct `jax.grad` | **2.9 ms (40× faster)** | bounded relaxation | **production** |
| Phase 3.5 `soft_gumbel` | direct `jax.grad` | comparable to 3c | bounded + stochastic | escaping local minima |

**Verdict (design-doc §4 bake-off): soft-lattice wins.** Vectorised gradient descent on the whole `(H, W, K)` softmax field beats `lax.scan` over `H·W` sequential single-site attempts by two orders of magnitude per step, and the gradient flows through `jax.grad` without straight-through hackery. REINFORCE is kept as the always-correct fallback.

---

## 📈 Phase 2 inverse-recovery results

End-to-end gradient descent on REINFORCE recovers known adhesion energies from oracle trajectories. Target: **≤10% relative error**. Achieved on all five benchmarks.

| benchmark | what's recovered | metric | rel_err |
|---|---|---|---:|
| cell_sorting | full J matrix (warm-start + per-entry continuation) | cluster count | **9.4%** ✓ |
| vasculogenesis | J[1,1] (endo–endo) | endo–medium perimeter | **9.5%** ✓ |
| chemotaxis | J[1,1] | centroid drift | hit ✓ |
| gastrulation_toy | full J via staged per-entry | interface curvature | **6.7%** ✓ |
| tumour_stroma | full J (joint Frobenius) | compactness | **6.7%** ✓ |

The recipe — **Adam per-entry + focused continuation on the load-bearing J entry + no bias correction on warm starts** — is captured in [`scripts/phase2_acceptance_benchmark_v5.py`](scripts/phase2_acceptance_benchmark_v5.py) and replicated across all four other benchmarks.

> **Subtlety we surfaced during tuning:** below the Steinberg DAH phase boundary (`J_C_NC < 8` in cell_sorting), the loss landscape is *genuinely flat in J* — cells never sort, so the acceptance metric is constant. REINFORCE has no signal there. Not a bug in the gradient; a property of the physics.

---

## 🏗️ Architecture

```
                                                                
        ┌─────────────────────────────────────────────────┐    
        │  cpjax.io_morpheus.read(...)  ──► (L, H)        │    
        │   ✅ any BioModels CP entry → differentiable     │    
        └────────────────────────┬────────────────────────┘    
                                 │                              
   ┌───────────────────┐         ▼         ┌────────────────┐  
   │  cpjax.oracles    │    ┌─────────┐    │ benchmarks/    │  
   │  (CC3D, Morpheus) │───▶│ Lattice │◀───│ 5 cases +      │  
   │  trajectories →   │    └────┬────┘    │ provenance     │  
   │  .npy + manifest  │         │         └────────────────┘  
   └───────────────────┘         ▼                              
                          ┌────────────┐                        
                          │Hamiltonian │ ← V_mem coupling      
                          │  (J, λ_A,  │   from BETSE-JAX      
                          │  λ_P, …)   │                        
                          └─────┬──────┘                        
                                │                               
                ┌───────────────┼───────────────┐               
                ▼               ▼               ▼               
        metropolis       reinforce       soft_lattice (🏆)     
         (forward)      (score-fn ∇)      (jax.grad ∇)         
                │               │               │               
                └───────┬───────┴───────────────┘               
                        ▼                                       
              cpjax.simulate.run → trajectory                   
                        │                                       
                        ▼                                       
                cpjax.inverse                                   
                ├─ recover_adhesion (params from trajectory)    
                └─ shape_inverse    (V_mem from target shape)   
```

---

## 📂 Layout

```
src/cpjax/
├── lattice.py              # Lattice PyTree, neighbours, area, perimeter
├── hamiltonian.py          # adhesion/area/perimeter energies, effective_J(vmem)
├── update.py               # 4 kernels: metropolis · reinforce · soft_lattice · soft_gumbel
├── simulate.py             # run / run_batched — lax.scan over MCS
├── inverse.py              # reinforce_grad · recover_adhesion · shape_inverse
├── io_morpheus.py          # MorpheusML → (Lattice, Hamiltonian)
├── oracles/
│   ├── cc3d.py             # run CC3D, parse VTK, cache trajectory
│   ├── morpheus.py         # run Morpheus, same interface
│   ├── _manifest.py        # SHA-256 trajectory manifests
│   └── _vtk_parse.py       # ASCII STRUCTURED_POINTS parser
└── benchmarks/
    ├── cell_sorting.py     # + vendored CC3D demo (MIT)
    ├── vasculogenesis.py
    ├── chemotaxis.py
    ├── gastrulation_toy.py
    ├── tumour_stroma.py
    └── _specs/             # MorpheusML XML specs (one per benchmark)
        └── _external/      # vendored MorpheusML — BioModels + Morpheus repo

scripts/                    # phase-2 acceptance benchmarks (v1 → v5)
tests/                      # 261 unit + integration tests
docs/plan.md                # phase-by-phase status
```

---

## 🤝 Compatibility & community standards

- **[CURE](https://arxiv.org/abs/2502.15597) compliance:** every benchmark ships `provenance.toml` with paper citation, original parameter values, and implementation status. Validated by 15 tests in `tests/test_provenance.py`.
- **MorpheusML round-trip:** parsing every authored benchmark spec yields a `Hamiltonian` whose `J` matches `ground_truth()` to float tolerance (14 tests, `tests/test_morpheus_import.py`).
- **External-spec import:** the reader has been validated against a CC0 [BioModels](https://www.ebi.ac.uk/biomodels/MODEL2009210001) MorpheusML model (Mulberry & Edelstein-Keshet 2020) and against the canonical Morpheus-distribution `VascularPatterning` example (Köhn-Luque et al. 2013). Both ship as vendored XMLs under `src/cpjax/benchmarks/_specs/_external/` with demonstrators (`examples/08_biomodels_import.py`, `examples/09_morpheus_repo_vasculogenesis.py`) that surface exactly which MorpheusML features each model uses and which the v0 reader drops (5+5 regression tests).
- **CC3D oracle parity:** real `cellsort_2D` demo run through `cpjax.oracles.cc3d.run` produces ≥30% cluster-count reduction (Phase-0 acceptance milestone).
- **Morpheus oracle parity:** same for the authored Morpheus specs across all 5 benchmarks.

---

## 🛣️ Phase status

| phase | scope | status |
|---|---|---|
| 0 — Oracle harness | wrap CC3D + Morpheus, 5 benchmarks, cached trajectories | 🟢 done; dual-oracle parity on all 5 benchmarks |
| 1 — Forward-only JAX CP | jittable `lax.scan` Metropolis CPM | 🟢 done |
| 1.5 — Cell-id vs cell-type | per-cell identity tracking | 🟢 landed |
| 2 — REINFORCE gradient | exact score-function ∇; parameter recovery | 🟢 acceptance hit (5/5 benchmarks) |
| 3 — Relaxed gradients | Gumbel + soft-lattice + bake-off | 🟢 soft-lattice winner (40×) |
| 3.5 — Soft-state Gumbel | stochastic + differentiable hybrid | 🟢 landed |
| 4 — MorpheusML import + export | BioModels-CP ↔ JAX | 🟢 reader + writer landed; round-trip on 5 benchmark specs + 2 vendored external specs (BioModels + Morpheus repo) |
| 6.0 — PDE/Chemotaxis parse layer | Field/System/Chemotaxis → structured `MorpheusModel` data | 🟢 reader now captures every MorpheusML element the two external specs use; Expression evaluation + PDE integration deferred to Phase 6.1 |
| 5 — BETSE-JAX coupling | V_mem → effective J → shape | 🟢 spine + 2 demonstrators landed (synthetic 99.9% + cell_sorting DAH restored, J[1,2] 0.2% err) |

Full breakdown — including the v1→v5 acceptance-tuning story — is in [`docs/plan.md`](docs/plan.md).

---

## 📚 Related work

- **U-Net CPM surrogates** ([2505.00316](https://arxiv.org/abs/2505.00316)) — forward-acceleration surrogates; complementary to `cpjax`'s differentiable-dynamics goal.
- **DDPM CPM surrogates** ([2505.09630](https://arxiv.org/abs/2505.09630), Glazier co-author) — score-function-gradient possibility from CC3D community.
- **CURE guidelines** ([2502.15597](https://arxiv.org/abs/2502.15597), Sauro & Glazier) — community standards; `cpjax` already compliant.
- **Forward-physics oracles:** [CompuCell3D](https://compucell3d.org), [Morpheus](https://morpheus.gitlab.io), [BioModels](https://www.ebi.ac.uk/biomodels/) (Cellular Potts entries).
- **Architectural precedent:** [jax-md](https://github.com/jax-md/jax-md) (Schoenholz & Cubuk 2020) — differentiable lattice/MC physics in JAX.

---

## 🔗 See also

- 📐 Design doc: [`anatomical-compiler/docs/differentiable-cp.md`](https://github.com/biopunk-lab/anatomical-compiler/blob/master/docs/differentiable-cp.md)
- ⚡ Bioelectric sister: [`betse-unified`](https://github.com/biopunk-lab/betse-unified) — differentiable V_mem layer
- 🧬 Compositional parent: [`anatomical-compiler`](https://github.com/biopunk-lab/anatomical-compiler) — regulome / control / educational track

---

## 📄 License

Apache-2.0. Vendored CC3D `cellsort_2D` demo retains its original MIT license; see `src/cpjax/benchmarks/_specs/cell_sorting/`.
