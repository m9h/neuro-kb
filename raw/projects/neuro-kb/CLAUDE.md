# neuro-kb — Shared Knowledge Base for Neuroimaging Projects

This is an LLM-maintained knowledge base (following the Karpathy "LLM Wiki" pattern) that serves as shared infrastructure across the neuroimaging project family in `~/dev/`.

## Purpose

These projects study the same underlying problem — imaging and simulating the brain — through different physics and sensor modalities. This wiki compiles the cross-cutting domain knowledge so that each project's agent doesn't rediscover it from scratch.

## Connected projects

| Project | Domain | Key shared concepts |
|---|---|---|
| `hgx` | Hypergraph neural networks (JAX/Equinox) | Topology, higher-order structure, connectomics |
| `vbjax` | Whole-brain simulation (JAX) | Neural mass models, connectomes, forward models |
| `setae` | Bio-inspired surface mechanics (JAX) | Contact mechanics, tissue properties, FEM |
| `qcccm` | Quantum cognition (JAX/PennyLane) | Spin glass ↔ neural dynamics isomorphisms |
| `jaxctrl` | Differentiable control theory (JAX) | Controllability, Lyapunov, hypergraph dynamics |
| `alf` | Active inference (JAX) | Generative models, Bayesian inference, agents |
| `RatInABox` | Spatial navigation / hippocampus | Place cells, grid cells, synthetic neural data |
| `organoid-hgx-benchmark` | Organoid gene regulatory networks | scRNA-seq, ATAC-seq, GRN inference |
| `organoid_regulomes` | Organoid regulatory genomics | Trajectory analysis, perturbation, CROP-seq |
| `neuro-nav` | RL for neuroscience | Successor representations, spatial cognition |
| `PGMax` | Probabilistic graphical models (JAX) | Belief propagation, factor graphs |
| `agentsciml` | Multi-agent scientific ML | Automated experiment design, swarm optimization |
| `libspm` | Statistical parametric mapping (C) | Registration, segmentation, tissue classification |
| `LAYNII` | Layer-fMRI analysis (C++) | Cortical layers, laminar fMRI, flattening |
| `neurotech-primer-book` | Neurotechnology education (MyST) | All modalities, sensor physics, safety |

## Directory structure

```
raw/            ← Immutable source documents. The LLM reads these but NEVER modifies them.
  papers/       ← PDFs and references
  head-models/  ← Documentation for MIDA, SCI, and other head models
  datasheets/   ← Sensor specs, tissue property tables, standards docs
wiki/           ← LLM-generated and LLM-maintained interlinked markdown
  index.md      ← Master catalog of all wiki pages
  log.md        ← Chronological audit trail of changes
```

## Entity types and schema

Wiki pages use YAML frontmatter. Every page MUST have `type`, `title`, and `related` fields.

### Modality
Imaging or stimulation modality (EEG, MEG, fNIRS, fUS, TMS, tDCS, DBS, etc.)
```yaml
type: modality
title: <name>
physics: <electromagnetic | hemodynamic | acoustic | mechanical | optical>
measurement: <what is measured>
spatial_resolution: <typical range>
temporal_resolution: <typical range>
related: [<other page filenames>]
```

### Physics
Physical principles underlying a modality or simulation.
```yaml
type: physics
title: <name>
governing_equations: <brief>
related: [<other page filenames>]
```

### Tissue
Biological tissue type with properties relevant across modalities.
```yaml
type: tissue
title: <name>
properties:
  conductivity_S_m: <value or range>
  relative_permittivity: <value or range>
  acoustic_impedance_MRayl: <value or range>
  optical_absorption_1_cm: <value or range>
  optical_scattering_1_cm: <value or range>
  density_kg_m3: <value or range>
sources: [<citation keys>]
related: [<other page filenames>]
```

### Head model
Anatomical head model with metadata.
```yaml
type: head-model
title: <name>
source: <provider/URL>
tissues: <number of tissue classes>
resolution: <voxel size or mesh detail>
formats: [<available file formats>]
local_path: <path under /data/datasets/>
related: [<other page filenames>]
```

### Method
Computational method shared across projects.
```yaml
type: method
title: <name>
category: <FEM | BEM | spectral | differentiable | optimization | inference>
implementations: [<project:module pairs>]
related: [<other page filenames>]
```

### Coordinate system
Spatial reference frame.
```yaml
type: coordinate-system
title: <name>
definition: <brief>
used_by: [<projects or tools>]
related: [<other page filenames>]
```

### Concept
General domain concept that doesn't fit the above.
```yaml
type: concept
title: <name>
related: [<other page filenames>]
```

## Agent conventions

1. **Cross-reference aggressively.** When updating a page, check `index.md` for related pages and add backlinks. The value of this wiki is in the connections.
2. **Cite sources.** When a fact comes from `raw/`, link to the source file. When from a project, use `project:file:line` format.
3. **Prefer specificity over generality.** "CSF conductivity is 1.79 S/m at 10 Hz (Gabriel 1996)" beats "CSF is conductive."
4. **Update `index.md`** after creating or renaming any page.
5. **Append to `log.md`** after every ingest or significant edit session, with date and summary.
6. **One concept per page.** Split if a page grows beyond ~300 lines.
7. **Never modify `raw/`.** Corrections go in the wiki page with a note about the source error.
8. **Frontmatter is queryable.** Keep it accurate — downstream agents filter on `type`, `physics`, `related`.
