---
type: modality
title: Diffuse Optical Tomography
description: "3D reconstruction of tissue optical properties from near-infrared light measured on the scalp, modelled by the photon diffusion equation."
timestamp: 2026-07-03T00:00:00-07:00
physics: optical
measurement: near-infrared light attenuation / photon fluence at the scalp
spatial_resolution: 1-3 cm
temporal_resolution: 10-100 ms
implementations: ["dot-jax:src/dot_jax/forward.py", "dot-jax:src/dot_jax/recon.py", "redbirdpy:redbirdpy/forward.py", "redbirdpy:redbirdpy/recon.py"]
related: [fnirs.md, physics-diffusion-equation.md, method-fem.md, method-monte-carlo.md, tissue-optical-properties.md]
---

# Diffuse Optical Tomography

Diffuse optical tomography (DOT) reconstructs a **3D image of tissue optical properties** (absorption μ_a, scattering μ_s′, and thence oxy-/deoxy-hemoglobin) from near-infrared light injected and detected by source–detector pairs on the scalp. It is the tomographic, image-forming generalization of [fNIRS](fnirs.md): where fNIRS reports channel-wise concentration changes, DOT solves an inverse problem over a volumetric mesh.

## Forward model

In highly scattering tissue the radiative transfer equation reduces to the **photon diffusion equation** (see [physics-diffusion-equation.md](physics-diffusion-equation.md)):

```
-∇ · (D(r) ∇Φ(r)) + μ_a(r) Φ(r) = q(r),   D = 1 / (3(μ_a + μ_s′))
```

solved by FEM on a head mesh (`redbirdpy:forward`, `dot-jax:mesh`) or by mesh-based **Monte Carlo** photon transport (the MMC/Redbird lineage; see [method-monte-carlo.md](method-monte-carlo.md)).

## Inverse problem

Reconstruction minimizes the misfit between measured and predicted boundary fluence, regularized (Tikhonov / spatial priors). Differentiable forward solvers (JAX/Equinox in `dot-jax`) give the Jacobian by autodiff; classic solvers (`redbirdpy`) form it explicitly and iterate a nonlinear Gauss–Newton update. Spectral (multi-wavelength) constraints tie μ_a to chromophore concentrations (`dot-jax:spectral`).

## Implementations

- **dot-jax** — differentiable JAX/Equinox DOT + fNIRS: FEM forward, autodiff reconstruction, chromophore spectroscopy.
- **redbirdpy** — Python port of Redbird: FEM diffusion forward + nonlinear DOT reconstruction (Q. Fang).

## Citations

[1] Boas et al. (2001). Imaging the body with diffuse optical tomography. IEEE Signal Process Mag.
[2] Eggebrecht et al. (2014). Mapping distributed brain function and networks with diffuse optical tomography. Nat Photonics.

## See Also

- [fnirs.md](fnirs.md) - channel-space optical modality
- [physics-diffusion-equation.md](physics-diffusion-equation.md) - photon diffusion forward physics
- [method-fem.md](method-fem.md) / [method-monte-carlo.md](method-monte-carlo.md) - forward solvers
- [tissue-optical-properties.md](tissue-optical-properties.md) - μ_a / μ_s′ values
