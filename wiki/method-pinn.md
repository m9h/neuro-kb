---
type: method
title: Physics-Informed Neural Networks
description: "Neural networks trained with the governing PDE/ODE residual as a loss term, used here to invert biophysical models without meshing the forward problem."
timestamp: 2026-07-03T00:00:00-07:00
category: differentiable
implementations: ["dmijl:src/pinn", "dmipy:dmipy_jax/pinns", "neurojax:geometry/fem_forward.py", "dot-jax:src/dot_jax/forward.py"]
related: [method-fem.md, physics-diffusion-equation.md, diffusion-mri.md, method-full-waveform-inversion.md, method-variational-inference.md, method-sbi.md]
---

# Physics-Informed Neural Networks

Physics-informed neural networks (PINNs) represent a field (or a parameter map) as a neural network and add the **residual of the governing equation** to the training loss, so the network is regularized to satisfy the physics between data points. In neuroimaging they appear as a mesh-free alternative to FEM/FD forward solvers and as a route to invert biophysical models directly.

## Formulation

For a PDE `𝒩[u](x) = 0` with boundary/data constraints, the network `u_θ(x)` minimizes:

```
L(θ) = λ_data · ‖u_θ − u_obs‖²  +  λ_phys · ‖𝒩[u_θ]‖²  (+ boundary terms)
```

Derivatives in `𝒩` are obtained by automatic differentiation of the network. For **inversion**, unknown physical parameters are trained jointly with (or instead of) the field.

## Uses in this corpus

- **Diffusion MRI microstructure** — restricted-diffusion / Bloch–Torrey residuals to recover axon radius and compartment fractions (`dmijl:src/pinn` AxCaliber; `dmipy:dmipy_jax/pinns`), an alternative to [method-sbi.md](method-sbi.md).
- **Differentiable head modelling** — gradients flow through the forward operator to fit conductivity/qMRI parameters (`neurojax`).
- **Optical / acoustic forward operators** — differentiable PDE solves that double as PINN-style constraints (`dot-jax`), complementary to [method-full-waveform-inversion.md](method-full-waveform-inversion.md).

Trade-offs: PINNs avoid meshing and give smooth, differentiable inverses, but training can be stiff (loss-term balancing, spectral bias toward low frequencies) relative to a well-conditioned FEM solve.

## Citations

[1] Raissi, Perdikaris & Karniadakis (2019). Physics-informed neural networks. J Comput Phys.
[2] Karniadakis et al. (2021). Physics-informed machine learning. Nat Rev Phys.

## See Also

- [method-fem.md](method-fem.md) - meshed forward-problem alternative
- [diffusion-mri.md](diffusion-mri.md) - microstructure inversion target
- [method-sbi.md](method-sbi.md) - simulation-based inference alternative
- [method-full-waveform-inversion.md](method-full-waveform-inversion.md) - adjoint-based PDE inversion
