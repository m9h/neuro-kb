---
type: modality
title: In-Vitro MEA Cortical Cultures
description: "Dissociated or organotypic neuronal cultures grown on multi-electrode arrays, and their in-silico simulation, used as a reduced experimental model of learning networks."
timestamp: 2026-07-03T00:00:00-07:00
physics: electromagnetic
measurement: extracellular potentials / spikes from a planar electrode grid
spatial_resolution: electrode pitch (10-200 µm)
temporal_resolution: <1 ms
implementations: ["bl1:src", "neurosim:adapter.py", "neurosim:autoresearch"]
related: [eeg.md, method-active-inference.md, neural-mass-models.md, foundation-models.md]
---

# In-Vitro MEA Cortical Cultures

Dissociated cortical neurons cultured on a **multi-electrode array (MEA)** self-organize into spontaneously active networks whose spiking can be recorded and stimulated in closed loop. They are a tractable, fully observable testbed for plasticity, criticality, and — following the DishBrain "Pong" result — for whether cultured networks can learn under structured feedback. This corpus works with the **in-silico** counterpart: GPU simulators of culture-on-MEA dynamics.

## Modelled physics

- **Neuron models** — Izhikevich / adaptive-exponential (AdEx) spiking units with conductance-based synapses (`bl1:src`).
- **Plasticity** — multi-timescale rules (fast STDP through slow homeostatic/structural scales).
- **Closed loop** — sensory encoding → MEA stimulation, motor readout → environment, with free-energy / prediction-error feedback (links to [method-active-inference.md](method-active-inference.md)).

## Why it matters here

MEA cultures give a bottom-up, biologically instantiated learning system to compare against top-down whole-brain models ([neural-mass-models.md](neural-mass-models.md)) and against foundation-model representations of neural data ([foundation-models.md](foundation-models.md)). The simulators also serve as synthetic-data generators and as environments for embodied/agentic experiments.

## Implementations

- **bl1** — JAX in-silico cortical-culture-on-MEA simulator: spiking neurons, conductance synapses, 4-timescale plasticity, DishBrain-style Pong.
- **neurosim** — same core plus adapters and an autoresearch harness.

## Citations

[1] Kagan et al. (2022). In vitro neurons learn and exhibit sentience when embodied in a simulated game-world (DishBrain). Neuron.
[2] Wagenaar, Pine & Potter (2006). An extremely rich repertoire of bursting patterns during the development of cortical cultures. BMC Neurosci.

## See Also

- [method-active-inference.md](method-active-inference.md) - closed-loop free-energy feedback
- [neural-mass-models.md](neural-mass-models.md) - macroscale counterpart
- [eeg.md](eeg.md) - extracellular-potential measurement physics
