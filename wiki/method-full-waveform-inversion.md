---
type: method
title: Full-Waveform Inversion
description: "PDE-constrained inversion that recovers a spatial physical-property map (e.g. speed-of-sound) by minimizing the misfit between simulated and measured wavefields via adjoint gradients."
timestamp: 2026-07-03T00:00:00-07:00
category: optimization
implementations: ["brain-fwi:src/brain_fwi/simulation/forward.py", "brain-fwi:src/brain_fwi/simulation/checkpointed_scan.py", "stride:stride/optimisation/optimisation_loop.py", "jwave:jwave/acoustics/time_varying.py"]
related: [physics-acoustic.md, tus.md, method-fem.md, method-pinn.md, tissue-acoustic-properties.md, diffuse-optical-tomography.md]
---

# Full-Waveform Inversion

Full-waveform inversion (FWI) is a PDE-constrained optimization that recovers a spatially varying medium property — for transcranial ultrasound, the **speed-of-sound / density map of the head** — by iteratively minimizing the L2 misfit between simulated and measured wavefields. Unlike ray-based or travel-time tomography, FWI uses the *full* recorded waveform (amplitude + phase), giving sub-wavelength resolution at the cost of a nonconvex objective prone to cycle-skipping.

## Method

Each iteration solves a forward wave simulation, computes the data residual, back-propagates it as an adjoint field, and forms the gradient by zero-lag correlation of forward and adjoint wavefields:

```
m_{k+1} = m_k - α ∇_m J,   J = ½ Σ ‖ u_sim(m) - u_obs ‖²
```

Memory is the bottleneck: the adjoint needs the forward wavefield at every timestep, so implementations use **checkpointed time-stepping** (`brain-fwi:simulation/checkpointed_scan`) to trade recompute for storage. Differentiable forward solvers (JAX/Devito) obtain `∇_m J` by automatic differentiation rather than a hand-derived adjoint.

## Cycle-skipping mitigation

- Multi-scale (low → high frequency) continuation
- Good starting model (e.g. **pseudo-CT skull** from `mr-to-pct`, atlas priors)
- Neural-operator surrogates (FNO/UNO, `brain-fwi:surrogate/fno3d`) as fast approximate solvers or initializers

## Implementations

- **brain-fwi** — 3D transcranial speed-of-sound FWI in JAX; pseudospectral forward, checkpointed adjoint, FNO3D surrogate, BrainWeb phantoms.
- **stride** — FDTD ultrasound modelling + medical-tomography optimisation loop (Devito, CPU/GPU).
- **jwave** — differentiable k-Wave port supplying the transient/Helmholtz forward operator.

## Citations

[1] Tarantola (1984). Inversion of seismic reflection data in the acoustic approximation.
[2] Guasch et al. (2020). Full-waveform inversion imaging of the human brain. npj Digital Medicine.

## See Also

- [physics-acoustic.md](physics-acoustic.md) - governing wave equation
- [tus.md](tus.md) - transcranial ultrasound modality
- [method-fem.md](method-fem.md) - alternative forward discretization
- [method-pinn.md](method-pinn.md) - PINN inversion alternative
- [diffuse-optical-tomography.md](diffuse-optical-tomography.md) - analogous PDE-constrained optical inversion
