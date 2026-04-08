---
title: "Quantum Compute for Computational Cognitive Modeling — bridging quantum computing and computational cognitive neuroscience"
author:
  - Morgan Hough
date: "2026-03-20"
---


# Introduction


## QCCCM: Quantum Compute for Computational Cognitive Modeling

## QCCCM — Quantum Compute for Computational Cognitive Modeling

> *The same Hamiltonian that describes a disordered magnet describes a society of agents with heterogeneous relationships. The same tools that find ground states of spin glasses find Nash equilibria of social systems. This library makes that isomorphism computational.*

[[Tests](https://github.com/m9h/quantum-cognition/actions/workflows/ci.yml/badge.svg)](https://github.com/m9h/quantum-cognition/actions)
[Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue)
[JAX](https://img.shields.io/badge/JAX-0.4%2B-green)
[PennyLane](https://img.shields.io/badge/PennyLane-0.35%2B-purple)
[Tests](https://img.shields.io/badge/tests-146%20passing-brightgreen)

---

### Vision

There is currently **no open-source quantum cognition Python library**. The field has rich theory (Busemeyer, Pothos, Khrennikov, Fuss & Navarro) but no reusable software. Meanwhile, quantum computing frameworks are mature, hardware access is increasingly free, and the mathematical isomorphism between **disordered magnets** and **multi-agent social systems** is exact.

QCCCM fills this gap — a JAX-native library that lets researchers move between:

```
┌─────────────────────┐     ┌──────────────────────┐     ┌─────────────────────┐
│   MATERIALS SCIENCE  │     │  QUANTUM COMPUTING   │     │   SOCIAL SCIENCE    │
│                      │     │                      │     │                     │
│  Spin glass          │◄───►│  VQE / QAOA / D-Wave │◄───►│  Multi-agent game   │
│  Disordered magnet   │     │  PennyLane circuits  │     │  Opinion dynamics   │
│  Phase transitions   │     │  Quantum walks       │     │  Consensus/conflict │
│                      │     │  Error mitigation    │     │                     │
│  H = -Σ Jᵢⱼ sᵢsⱼ   │     │                      │     │  H = -Σ Jᵢⱼ sᵢsⱼ  │
└─────────────────────┘     └──────────────────────┘     └─────────────────────┘
         ▲                                                         ▲
         │              SAME HAMILTONIAN                           │
         └─────────────────────────────────────────────────────────┘
```

#### The Materials ↔ Social Correspondence

| Materials Science | Multi-Agent Social Systems |
|---|---|
| Spin s_i ∈ {↑,↓} | Agent opinion s_i ∈ {A,B} |
| Coupling J_ij > 0 (ferromagnetic) | Agreement incentive (conformity) |
| Coupling J_ij < 0 (antiferromagnetic) | Disagreement incentive (competition) |
| Disorder in J_ij | Heterogeneous social relationships |
| Temperature T | Bounded rationality / noise |
| Transverse field Γ | Quantum cognitive flexibility |
| Ground state | Nash equilibrium / social optimum |
| Frustrated triangle | Social dilemma (enemy of my enemy) |
| Spin glass phase (m≈0, q_EA>0) | Polarization lock-in without consensus |
| Phase transition at T_c | Tipping point in public opinion |

---

### Architecture

```
src/qcccm/
├── core/                     # Quantum primitives (JAX, jit-compatible)
│   ├── states.py             #   |0⟩, |+⟩, Bell, GHZ state preparation
│   ├── density_matrix.py     #   von Neumann entropy, partial trace, QMI, purity, fidelity
│   └── quantum_walk.py       #   Hadamard/biased coins, QW evolution (lax.scan), FPT density
│
├── models/                   # Classical ↔ quantum bridges
│   ├── bridge.py             #   Szegedy stochastic→unitary, beliefs↔ρ, quantum EFE
│   └── alf_bridge.py         #   ALF active inference integration, QuantumEFEAgent
│
├── spin_glass/               # Disordered magnet / social simulation
│   ├── hamiltonians.py       #   SK, EA models, PennyLane Hamiltonian construction
│   ├── order_params.py       #   q_EA, P(q) overlap distribution, Binder cumulant, χ_SG
│   └── solvers.py            #   Metropolis, PIMC (Trotter), VQE, QAOA
│
├── games/                    # Multi-agent game theory
│   ├── minority.py           #   Minority game with quantum agents, phase transition at α_c
│   └── agreement.py          #   Ising agreement model, frustration, Schelling segregation
│
├── circuits/                 # PennyLane quantum circuits
│   ├── templates.py          #   Variational ansatze, amplitude encoding
│   ├── interference.py       #   Quantum interference for decision models
│   ├── belief_circuits.py    #   Bayesian belief update on circuit
│   └── export.py             #   PennyLane → Qiskit conversion
│
├── networks/                 # Multi-agent quantum cognitive networks
│   ├── topology.py           #   Complete, ring, star, random graph construction
│   ├── multi_agent.py        #   Density matrix evolution with coupling + decoherence
│   └── observables.py        #   Entropy, polarization, fidelity, coherence
│
├── fitting/                  # Parameter estimation
│   ├── likelihoods.py        #   Choice, QW RT, interference log-likelihoods (JAX)
│   ├── mle.py                #   MLE with JAX autodiff, AIC/BIC model comparison
│   └── data.py               #   ChoiceData, FitResult containers
│
├── annealing/                # Quantum annealing
│   ├── qubo.py               #   EFE → QUBO mapping, policy assignment
│   └── solve.py              #   Brute force, simulated, D-Wave solvers
│
├── mitigation/               # Quantum error mitigation
│   └── zne.py                #   Zero-noise extrapolation (Richardson)
│
├── benchmarks/               # Performance profiling
│   ├── bench_jax.py          #   JIT compilation, walk scaling, density matrix ops
│   └── bench_networks.py     #   Network evolution, observables scaling
│
└── viz/                      # Visualization
    ├── bloch.py              #   Bloch sphere for single-qubit states
    └── walks.py              #   QW probability evolution, spreading, FPT plots

autoresearch/                 # Autonomous research loop (Karpathy-style)
├── program.md                #   Agent instructions — the "skill file"
├── prepare.py                #   Infrastructure (read-only): solvers, metrics, logging
└── experiment.py             #   THE FILE THE AI AGENT MODIFIES

notebooks/
├── 01_bits_to_qubits.ipynb           # Superposition, density matrices, von Neumann entropy
├── 02_quantum_walks_decision.ipynb   # QW vs classical RW, FPT, evidence accumulation
├── 03_quantum_vs_classical_efe.ipynb # Quantum EFE, active inference, policy selection
├── 04_quantum_minority_game.ipynb    # Phase transition at α_c, quantum agent coordination
└── 05_spins_phases_collective.ipynb  # Ising model, agreement dynamics, Nayebi scaling
```

---

### Quick Start

```bash
## Install
uv sync

## Run tests (146 passing)
uv run pytest -v

## Run a spin glass experiment
uv run python autoresearch/experiment.py
```

```python
import jax.numpy as jnp
from qcccm.core import (
    plus_state, pure_state_density_matrix,
    von_neumann_entropy, purity,
    QuantumWalkParams, quantum_walk_evolution,
)
from qcccm.spin_glass import sk_couplings, metropolis_spin_glass, edwards_anderson_q
from qcccm.spin_glass.hamiltonians import SocialSpinGlassParams
import numpy as np

## --- Quantum cognition: density matrices as generalized beliefs ---
psi = plus_state()
rho = pure_state_density_matrix(psi)
print(f"Purity: {purity(rho):.2f}, S(ρ): {von_neumann_entropy(rho):.4f}")

## --- Quantum walk: ballistic spreading ---
params = QuantumWalkParams(n_sites=101, n_steps=50, start_pos=50)
probs = quantum_walk_evolution(params)  # (51, 101) — O(t²) spreading

## --- Spin glass: social equilibrium search ---
N = 12
adj, J = sk_couplings(N, seed=42)
sg_params = SocialSpinGlassParams(N, adj, J, np.zeros(N), temperature=0.3)
result = metropolis_spin_glass(sg_params, n_steps=5000)
print(f"Ground state energy: {result.energy:.4f}")
print(f"Edwards-Anderson q_EA: {edwards_anderson_q(result.trajectory):.4f}")
```

---

### Solvers

Four methods for finding social equilibria / spin glass ground states:

| Solver | Method | Best For | N Range |
|--------|--------|----------|---------|
| **Metropolis MC** | Classical single-spin-flip | Baseline, any system | Any |
| **PIMC** | Path-integral Monte Carlo (Trotter slices) | Quantum tunneling through barriers | Any |
| **VQE** | Variational Quantum Eigensolver (PennyLane) | Small systems, exact ground state | ≤ 16 |
| **QAOA** | Quantum Approximate Optimization (PennyLane) | Combinatorial structure | ≤ 16 |

```python
from qcccm.spin_glass.solvers import (
    metropolis_spin_glass,    # Classical baseline
    transverse_field_mc,      # Quantum-inspired PIMC
    vqe_ground_state,         # PennyLane VQE
    qaoa_ground_state,        # PennyLane QAOA
)
```

---

### Autoresearch

Inspired by [Karpathy's autoresearch](https://github.com/karpathy/autoresearch) (45K stars). An autonomous AI research loop that iterates on sociophysics quantum architecture:

```
LOOP FOREVER:
  1. Read results.tsv — what worked? what's unexplored?
  2. Modify experiment.py — new model, parameters, solver
  3. Run experiment (10-min timeout)
  4. Compute quantum_advantage = (E_classical - E_quantum) / |E_exact|
  5. If advantage > 0.01 → KEEP (advance branch)
  6. If not → DISCARD (git reset)
  7. Log to results.tsv
  8. Generate next hypothesis
```

The agent explores the space of sociophysics models (SK, EA, Voter, Majority Rule, Sznajd, Schelling, Minority Game) × topologies (complete, lattice, random, scale-free) × solvers (Metropolis, PIMC, VQE, QAOA) to find regimes where quantum methods outperform classical.

```bash
## Kick off with any LLM coding agent:
## "Read autoresearch/program.md and start the research loop"
uv run python autoresearch/experiment.py
```

---

### Notebooks

| # | Notebook | Topics | Prerequisites |
|---|----------|--------|---------------|
| 01 | From Bits to Qubits | Superposition, density matrices, von Neumann entropy, entanglement | Linear algebra |
| 02 | Quantum Walks & Decision | QW vs classical RW, FPT, interference fringes, evidence accumulation | NB 01, DDM concepts |
| 03 | Quantum EFE | Quantum active inference, policy selection, quantumness parameter | NB 01, AIF concepts |
| 04 | Quantum Minority Game | Phase transition at α_c, quantum agent coordination, volatility | NB 03 |
| 05 | Spins & Collective Behavior | Ising model, agreement dynamics, frustration, Nayebi scaling | NB 04 |

No quantum mechanics prerequisites. Dirac notation introduced gradually (NB 01-02 use matrix notation only).

---

### Optional Dependencies

```bash
uv pip install "qcccm[ibm]"        # Qiskit + Aer (IBM hardware access)
uv pip install "qcccm[annealing]"   # D-Wave Ocean SDK (quantum annealing)
uv pip install "qcccm[mitiq]"       # Mitiq error mitigation
```

---

### Key References

**Sociophysics:**
- Mullick & Sen (2025). "Sociophysics models inspired by the Ising model." [arXiv:2506.23837](https://arxiv.org/abs/2506.23837) — comprehensive review, 118 references
- Brock & Durlauf (2001). "Discrete Choice with Social Interactions." *Rev. Econ. Studies* 68:235 — mean-field Ising = logistic choice
- Challet, Marsili, Zecchina (2000). "Statistical Mechanics of Minority Games." *PRL* 84:1824 — replica solution

**Quantum Cognition:**
- Busemeyer & Bruza (2012). *Quantum Models of Cognition and Decision* — foundational textbook
- Pothos & Busemeyer (2013). "Can quantum probability provide a new direction?" *Psych. Review*
- Khrennikov (2010). *Ubiquitous Quantum Structure* — quantum-like models outside physics

**Quantum Computing for Social/Materials Science:**
- Farhi et al. (2022). "QAOA and the SK Model at Infinite Size." *Quantum* 6:759 — QAOA surpasses SDP at depth p=11
- Abbas et al. (2021). "Power of quantum neural networks." *Nature Comp. Sci.* — QNN effective dimension
- Andreev, Cattan et al. (2023). "pyRiemann-qiskit." *RIO Journal* — Riemannian + quantum classifiers

**Autoresearch:**
- Karpathy (2026). [autoresearch](https://github.com/karpathy/autoresearch) — autonomous AI-driven ML research

---

### License

MIT


# Methodology


## Autonomous Sociophysics Quantum Research Loop

## Sociophysics Quantum Compute Autoresearch

### Goal

Find sociophysics models and parameter regimes where **quantum methods outperform classical methods** for finding ground states (social equilibria) of disordered multi-agent systems.

### The Loop

You are an autonomous research agent. Run this loop indefinitely:

1. **Read** `results.tsv` to understand what has been tried and what worked
2. **Modify** `experiment.py` with a new experimental idea
3. **Commit** with a short description: `git commit -am "experiment: <description>"`
4. **Run**: `uv run python autoresearch/experiment.py > autoresearch/run.log 2>&1`
5. **Extract results**: `grep "^RESULT|" autoresearch/run.log`
6. If empty → crash → `tail -n 50 autoresearch/run.log` → attempt fix
7. **Log** to `autoresearch/results.tsv` (append, untracked)
8. If `quantum_advantage > 0.01` → **KEEP** (advance branch)
9. If `quantum_advantage <= 0.01` → **DISCARD** (`git checkout -- autoresearch/experiment.py`)
10. **NEVER STOP.** Do not ask the human. Run indefinitely.

### Metric

The primary metric is `quantum_advantage`:

```
quantum_advantage = (E_classical - E_quantum) / |E_exact|
```

where:
- `E_classical` = best energy found by Metropolis MC (5000 sweeps)
- `E_quantum` = best energy found by quantum method (PIMC, VQE, or QAOA)
- `E_exact` = exact ground state energy (brute force for N ≤ 20, or best known)

Secondary metrics (always report):
- `q_EA_classical` = Edwards-Anderson order parameter from Metropolis trajectory
- `q_EA_quantum` = Edwards-Anderson order parameter from quantum trajectory (if available)
- `frustration_index` = fraction of frustrated triangles
- `wall_time_classical` = seconds for classical solver
- `wall_time_quantum` = seconds for quantum solver
- `n_agents` = system size

### What You Can Change

Everything in `experiment.py` is fair game:

#### Model Parameters
- **Topology**: complete, square lattice, chain, ring, star, random (Erdos-Renyi), scale-free
- **Disorder**: SK (Gaussian J_ij), EA bimodal (±J), EA Gaussian, uniform, custom
- **System size**: N = 4 to 20 (must be brute-force tractable for exact comparison)
- **Temperature**: T = 0.01 to 10.0
- **External field**: h_i = 0 (no field), uniform, random, site-dependent
- **Frustration level**: 0.0 to 1.0 (fraction of negative bonds for bimodal)

#### Solver Parameters
- **Classical**: n_sweeps, n_equilibrate, initial configuration
- **PIMC**: n_trotter (2-32), transverse_field strength (Gamma)
- **VQE**: n_layers (1-6), learning_rate, max_steps, ansatz type
- **QAOA**: depth p (1-10), learning_rate, max_steps

#### Dynamics (from Mullick & Sen 2025)
- Glauber dynamics (heat bath)
- Metropolis dynamics
- Kawasaki dynamics (conserved order parameter — segregation)
- Voter model dynamics
- Majority rule dynamics
- Sznajd dynamics
- Generalized (z, y) model: p_i(σ) = (1/2)[1 - σ_i F_i(σ)]

#### Observables
- Energy E and energy per spin E/N
- Magnetization |m|
- Edwards-Anderson q_EA
- Overlap distribution P(q) from multiple replicas
- Binder cumulant U_L
- Glass susceptibility χ_SG
- Consensus time τ (for dynamics experiments)
- Exit probability E(x)
- Persistence P(t)

### What You Cannot Change

- `prepare.py` — the infrastructure (qcccm library imports, solver wrappers)
- The metric definition
- The 10-minute wall-clock timeout per experiment
- The requirement to report all secondary metrics

### Strategy Guidance

1. **Start simple**: N=6-8, SK model, compare Metropolis vs PIMC. Find the regime where PIMC wins.
2. **Increase frustration**: Frustrated systems are where quantum tunneling should help most.
3. **Sweep transverse field**: Find optimal Gamma for PIMC at each (N, T, frustration).
4. **Try VQE/QAOA**: For small N (4-8), compare against exact ground state.
5. **Vary topology**: Does quantum advantage depend on graph structure?
6. **Try dynamics**: Consensus time τ — does quantum tunneling speed up consensus in frustrated networks?
7. **Combine insights**: If you find quantum advantage in a specific regime, characterize it systematically.

### Results Format

Each experiment appends one line to `results.tsv`:

```
commit  model  topology  disorder  N  T  Gamma  frustration  method  E_best  E_exact  quantum_advantage  q_EA  wall_time  status  description
```

### If You Get Stuck

- Re-read results.tsv for patterns. What worked? What's unexplored?
- Try combining two previous near-misses
- Try a radically different topology or dynamics
- Increase system size (more room for frustration)
- Try disorder in the external field (random field Ising model)
- Read the correspondence table in program.md and pick a new social scenario

### Reference: Materials ↔ Social Correspondence

| Materials | Social | Hamiltonian |
|---|---|---|
| Ferromagnet | Consensus game | H = -J Σ s_i s_j, J > 0 |
| Antiferromagnet | Competition game | H = -J Σ s_i s_j, J < 0 |
| Spin glass (SK) | Heterogeneous social network | J_ij ~ N(0, 1/√N) |
| Spin glass (EA) | Local trust/distrust network | J_ij = ±1 on lattice |
| Random field Ising | Diverse private information | h_i ~ N(0, σ_h) |
| Transverse field Ising | Quantum cognitive flexibility | H_x = -Γ Σ X_i |
| Frustrated triangle | Social dilemma | Mixed-sign J on triangle |
| Schelling segregation | Kawasaki Ising below T_c | Conserved magnetization |
| Minority game | p-spin glass | Replica RSB at α_c |
