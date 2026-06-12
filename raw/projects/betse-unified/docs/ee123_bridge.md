# EE 123 → BETSE bridge

A map from the Tufts **EE 123 / BME 123 Computational Bioelectricity** course
(Joel Grodstein, Michael Levin — <https://www.ece.tufts.edu/ee/123/>) to the
corresponding modules and demo configs in this repo.

EE 123 teaches the fundamentals using its own pedagogical framework
("Bitsey": `sim.py`, `main_sim_SS.py`, `main_sim_QSS.py`,
`main_sim_neuron.py`). BETSE is the research-grade follow-on: same physics,
plus a Voronoi tissue mesh, gap-junction-coupled networks, gene-regulatory
coupling, and (with `BETSE_JAX=1`) gradient-based inverse design. This
document is meant for a student finishing EE 123 who wants to scale a final
project up to tissue-level work.

## Course unit → BETSE module map

### Unit 0 — Course intro / circuit fundamentals
Start with the repo `README.rst` and `sample_sim.yaml`. The simplest entry
point is a single-tissue patch with one ion channel — read the comment
sections of `sample_sim.yaml` top to bottom.

### Unit 1a — Steady-state cellular voltage generation
**Worked example:** `examples/ee123_lab2_steady_state.py` — a direct
reproduction of the Lab 2 lesson (current-clamp at different Jn levels →
exponential settle to steady-state Vmem, with a linear I-V relationship at
equilibrium). Run with `BETSE_JAX=1 .venv/bin/python
examples/ee123_lab2_steady_state.py`; output lands in
`RESULTS/ee123_lab2_steady_state.png`.

The "hardware" layer in BETSE. Each module is a HH-style channel kinetics
class implementing `ChannelsABC`:

| EE 123 concept | BETSE module |
|----------------|--------------|
| Voltage-gated Na⁺ | `src/betse/science/channels/vg_na.py` |
| Voltage-gated K⁺  | `src/betse/science/channels/vg_k.py` |
| Voltage-gated Ca²⁺ | `src/betse/science/channels/vg_ca.py` |
| Voltage-gated Cl⁻ | `src/betse/science/channels/vg_cl.py` |
| Non-selective cation | `src/betse/science/channels/cation.py` |
| Abstract channel base | `src/betse/science/channels/channelsabc.py` |

Bitsey analogue: `main_sim_SS.py`. BETSE analogue: any config with a single
tissue patch and one or two channels enabled — fork `sample_sim.yaml`.

### Unit 1b — Quasi-steady-state neuronal bioelectricity
Same channel set plus:

- `src/betse/science/channels/vg_funny.py` — HCN / "funny" current.
- `src/betse/science/channels/vg_morrislecar.py` — Morris–Lecar two-variable
  spiking model on a whole-cell patch.

Bitsey analogue: `main_sim_QSS.py`.

### Unit 2a — Neurons, surface-EMG, prosthetics
BETSE simulates tissue rather than single-axon cable equations, but the
membrane physics is the same family used by `main_sim_neuron.py`. Use
`vg_morrislecar.py` on a single-cell patch for spiking-style demos. For
gap-junction-coupled multi-cell behaviour see
`src/betse/science/channels/gap_junction.py`.

### Unit 2b — Nervous system, electroceuticals, intervention
The "electroceutical" framing in this unit maps onto BETSE's inverse-design
pipeline: search for an external current schedule that drives the system
toward a target bioelectric state. See **Going beyond EE 123** below.

### Unit 3 — Cardiac bioelectricity
- `demo_cardiac_atrial.yaml` and `demo_lee2019_fresh_cardiac.yaml` — atrial
  tissue configs.
- `src/betse/science/jax/cardiac/courtemanche.py` — Courtemanche atrial cell
  model, JIT-able under `BETSE_JAX=1`.
- `src/betse/science/jax/cardiac/integrator.py` — explicit-Euler driver for
  the cardiac model.

This is the easiest unit to take past EE 123: the Courtemanche model is
already wired into the JAX path, and the demo YAMLs supply atrial geometry.

### Unit 4a/4b — Biology backgrounder, morphogenesis & developmental biology
This is where BETSE significantly exceeds Bitsey. The Voronoi tissue mesh
plus gap junctions implements the multi-cellular pattern formation the
unit covers.

- `examples/pietak_2018_reproduction.py` and
  `notebooks/pietak_2018_reproduction.ipynb` — reproduction of categorical
  Vmem patterns on a planarian silhouette from Pietak & Levin 2018 PBMB.
- `demo_drosophila.yaml` — Drosophila-shaped tissue config.

### Unit 4c — Worm / cancer topics
- `demo_planaria.yaml` — planarian-shaped tissue, a natural starting point
  for regeneration / head-vs-tail reprogramming experiments.

## Going beyond EE 123

### Inverse design — the "anatomical compiler" objective
The JAX overlay (`BETSE_JAX=1`) adds gradient-based search for parameters
that drive Vmem toward a target morphology. This is the computational
realisation of the anatomical-compiler framing used in the Levin lab.

**Worked example:** `examples/ee123_inverse_design.py` — same 10-cell
strip as the Lab 2 example, but solves the *inverse* problem: pick a
left-to-right Vmem gradient as the target, let `optimize_pattern` with
`n_steps=200` discover the Jn that produces it. Loss drops ~10,000× over
300 epochs; the converged Jn matches the analytic answer. Output:
`RESULTS/ee123_inverse_design.png`.

- `src/betse/science/jax/inverse.py` — `optimize_pattern` (line 86) and a
  multi-step loss (line 28) that integrates the full physics through
  `lax.scan` before scoring against the target.
- `src/betse/science/jax/bridge.py` — `make_target_pattern` (line 181)
  resolves named morphologies (`ring`, `gradient-x`, `gradient-y`,
  `flat-depolarized`, `flat-hyperpolarized`) into per-membrane Vmem arrays;
  `optimize_from_parameters` (line 237) runs the whole pipeline from a
  parsed YAML `inverse:` block.
- `src/betse/science/config/model/confinverse.py` — `SimConfInverse`, the
  YAML schema (`enabled`, `epochs`, `learning rate`, `loss function`,
  `target pattern`).

Canonical reference: `docs/JAX_REFAC.md`.

### Gene regulatory network coupling
EE 123 keeps genes and bioelectricity separate. BETSE couples them: the same
simulation can advance a per-cell GRN ODE while bioelectric state biases
transcription.

- `extra_configs/grn_basic.yaml` — minimal GRN config reproducing Karlebach
  & Shamir 2008 Fig. 2.
- `src/betse/science/jax/regulome.py` — Vm → transcriptional-bias coupling
  on the JAX path.

### Spatial transcriptomics → bioelectric mesh
- `demo_visium_brain.yaml` — Visium spatial-transcriptomics geometry.
- `src/betse/science/jax/regulome.py` — AnnData/H5AD projection onto a mesh.

## Suggested EE 123 final-project paths

The course leaves final projects open. Three that BETSE makes tractable:

1. **Categorical morph reproduction.** Reproduce one of the five Pietak &
   Levin 2018 Vmem morphs on the planarian silhouette, then use the
   `inverse:` YAML block to search for channel-conductance settings that
   flip it to a different morph. Starting point:
   `examples/pietak_2018_reproduction.py`.
2. **Cardiac tissue scaling.** Take a Unit 3 single-cell Courtemanche model
   up to atrial tissue and characterise propagation. Starting point:
   `demo_cardiac_atrial.yaml` + `src/betse/science/jax/cardiac/`.
3. **GRN ↔ bioelectric coupling.** Wire a toy two-gene GRN to a
   voltage-sensitive channel and look for bistability in the joint state.
   Starting point: `extra_configs/grn_basic.yaml`.

## Running

The legacy pipeline runs under stock NumPy:

```bash
.venv/bin/python -m betse seed sample_sim.yaml
.venv/bin/python -m betse init sample_sim.yaml
.venv/bin/python -m betse sim  sample_sim.yaml
```

The JAX overlay (inverse design, fused `step_pure`, Courtemanche, regulome
projection) requires `BETSE_JAX=1`:

```bash
BETSE_JAX=1 .venv/bin/python -m pytest tests/betse_test/a00_unit/science/
```
