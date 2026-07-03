---
type: modality
title: Transcranial Electric Stimulation E-Field Modeling
description: "FEM computation of the electric field delivered to the brain by scalp electrodes in tDCS/tACS/tES, on individualized head models."
timestamp: 2026-07-03T00:00:00-07:00
physics: electromagnetic
measurement: induced electric field (V/m) in brain tissue
spatial_resolution: mm (mesh-dependent)
temporal_resolution: quasi-static (DC / low-frequency)
implementations: ["simnibs:simnibs/simulation", "simnibs:simnibs/segmentation", "simnibs:simnibs/mesh_tools"]
related: [tms.md, method-fem.md, tissue-electrical-properties.md, structural-mri.md, head-model-mida.md, eeg.md]
---

# Transcranial Electric Stimulation E-Field Modeling

Transcranial electric stimulation (tES: tDCS, tACS, tRNS) drives weak currents through scalp electrodes to modulate cortical excitability. Because the skull spreads and attenuates current, the *delivered* electric field in the brain is non-obvious — so dosing relies on **subject-specific E-field simulation** rather than electrode montage alone. This is the stimulation counterpart to the EEG forward problem (same volume-conduction physics, sources and sinks swapped).

## Pipeline

1. **Segmentation** — MRI → tissue labels (SimNIBS **CHARM**), assigning conductivities (see [tissue-electrical-properties.md](tissue-electrical-properties.md)).
2. **Meshing** — tetrahedral head mesh across scalp/skull/CSF/GM/WM (`simnibs:mesh_tools`).
3. **FEM solve** — quasi-static current-conduction problem `∇·(σ∇φ) = 0` with electrode boundary conditions (`simnibs:simulation`), yielding the E-field `E = −∇φ`.

The governing equation and FEM machinery are shared with [method-fem.md](method-fem.md) and the [electromagnetic forward problem](physics-electromagnetic.md); tES differs mainly in boundary conditions (injected current at electrode pads) and the quasi-static assumption.

## Uses

- Montage optimization for target focality/intensity
- Dose reproducibility across subjects and sessions
- Explaining inter-subject variability in tDCS response

## Implementations

- **simnibs** — individualized head models (CHARM segmentation), tetrahedral meshing, FEM E-field solver for tES and TMS.

## Citations

[1] Saturnino et al. (2019). SimNIBS 2.1: TMS and tES electric field modelling. In: Brain and Human Body Modeling.
[2] Huang et al. (2019). Realistic volumetric-approach to simulate transcranial electric stimulation — ROAST.

## See Also

- [tms.md](tms.md) - magnetic-stimulation sibling
- [method-fem.md](method-fem.md) - numerical solver
- [physics-electromagnetic.md](physics-electromagnetic.md) - shared forward physics
- [head-model-mida.md](head-model-mida.md) - anatomical head models
