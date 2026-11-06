---
type: modality
title: Transcranial Ultrasound
physics: acoustic
measurement: Acoustic pressure transmitted through the skull, including focal pressure for neuromodulation or ablation, transmission waveforms for USCT/FWI, and secondary radiation-force tissue displacement (MR-ARFI)
spatial_resolution: ~1.5–8 mm wavelength in brain (1 MHz–180 kHz); 1–2 mm grids for FWI sound-speed maps
temporal_resolution: sub-µs (acoustic waveform); ~100 ms per MR-ARFI image; minutes–hours per FWI reconstruction
related: [physics-acoustic.md, tissue-acoustic-properties.md, tissue-skull.md, fus.md, head-model-mida.md, sci-head-model.md, method-fem.md, method-sbi.md, jax-ecosystem.md, physics-bloch.md, acoustoelectric-imaging.md, structural-mri.md, tissue-csf.md, tissue-gray-matter.md, tissue-white-matter.md, tissue-scalp.md, compute-infrastructure.md, temporal-interference.md, tms.md]
description: "Transcranial ultrasound (TUS) sends acoustic energy through the intact skull into the brain."
timestamp: 2026-11-05T15:59:53-08:00
tags: [modality, acoustic, ultrasound, stimulation]
---

# Transcranial Ultrasound

Transcranial ultrasound (TUS) sends acoustic energy through the intact skull into the brain. The term covers three uses that share the same forward physics:

- **Therapy and neuromodulation (TFUS).** Focused beams from 1 to 256+ elements are steered by per-element delays.
- **Imaging.** Transcranial ultrasound computed tomography (USCT) uses full waveform inversion (FWI) to recover a 3D speed-of-sound map.
- **MRI-coupled monitoring.** MR-ARFI images tissue displacement and PRF thermometry tracks heating [@kaye2013mrarfi; @rieke2008mr].

The skull is the dominant obstacle. It attenuates, aberrates and reflects the beam, so accurate subject-specific skull modelling is the central computational problem. Functional ultrasound (power Doppler, ULM) is a related but distinct hemodynamic imaging modality, covered in [fus.md](fus.md).

## Physics

TUS simulation in this project family uses the linear acoustic equations with heterogeneous sound speed $c(\mathbf{x})$, density $\rho_0(\mathbf{x})$ and power-law absorption $\alpha_0$. See [physics-acoustic.md](physics-acoustic.md) for the full treatment.

**Time domain (j-Wave/k-Wave first-order system).** The equations follow eq. 2.1 of the k-Wave manual and are integrated with a first-order symplectic scheme. Source: `jwave:docs/api_time_varying.md`.

$$
\partial_t \mathbf{u} = -\tfrac{1}{\rho_0}\nabla p,\qquad
\partial_t \rho_\varepsilon = -\rho_0\,\partial_\varepsilon u_\varepsilon + S_M,\qquad
p = c_0^2 \textstyle\sum_\varepsilon \rho_\varepsilon
$$

**Frequency domain (Helmholtz with Stokes absorption).** This form is solved with GMRES and is used for single-frequency steady-state problems. Absorption $\alpha_0$ is specified in dB/(MHz^y cm). Source: `jwave:docs/api_time_harmonic.md`.

$$
\nabla^2 P - \tfrac{1}{\rho_0}\nabla\rho_0\cdot\nabla P + \tfrac{2i\omega^3\alpha_0}{c_0}P + \tfrac{\omega^2}{c_0^2}P = i\omega S_M
$$

**Boundaries.** Open boundaries use a PML that modifies each partial derivative, $\partial^2_\varepsilon \to \gamma_\varepsilon^{-1}\partial_\varepsilon\gamma_\varepsilon^{-1}\partial_\varepsilon$ (Bermúdez et al. 2007).

**Radiation force and displacement.** The secondary effects are modelled in two steps:

1. Absorbed intensity produces a body force $F = 2\alpha I/c$.
2. Tissue displacement $u$ follows elastodynamics, $\mu\nabla^2 u + F = \rho\ddot{u}$.

Typical TFUS displacements are 1–10 µm, which is within MR-ARFI's ~0.1 µm sensitivity [@kaye2011mrarfi; @kaye2013mrarfi]. Source: `sbi4dwi:docs/openlifu/README.md`.

### Wavelength vs. frequency

Wavelength is computed as λ = c/f.

| Frequency | λ in brain/water (c≈1500 m/s) | λ in cortical bone (c=2800 m/s) |
|---|---|---|
| 180 kHz | 8.3 mm | 15.6 mm |
| 400 kHz | 3.75 mm | 7.0 mm |
| 500 kHz | 3.0 mm | 5.6 mm |
| 1 MHz | 1.5 mm | 2.8 mm |

Focal width scales with λ times the array f-number. Lower frequency therefore trades focal precision for skull penetration. The skull (~5–10 mm thick) is on the order of one wavelength, which is why reverberation and standing waves inside the bone matter.

## Skull and Tissue Acoustic Properties

The ITRUSST benchmark sound speeds [@aubry2022itrusst] are used in `brain-fwi:brain_fwi/phantoms/` and `sbi4dwi:dmipy_jax/biophysics/acoustic.py`. The acoustic module also provides continuous property estimation from CT Hounsfield units.

| Tissue | Sound speed (m/s) | Density (kg/m³) | Notes |
|---|---|---|---|
| Cortical bone | 2800 | 1850 | ITRUSST [@aubry2022itrusst] |
| Trabecular (diploë) bone | 2300 | 1700 | ITRUSST [@aubry2022itrusst] |
| Brain | 1560 | 1040 | Soft-tissue reference |
| CSF / water | 1500 | 1000 | Coupling medium and baseline |

Notes on these values:

- The densities and soft-tissue speeds are carried over from the prior revision of this page and should be verified against [tissue-acoustic-properties.md](tissue-acoustic-properties.md).
- Some brain-fwi synthetic phantoms use a different "true" skull speed, 4080 m/s, in the multi-frequency run below. Compare speeds only within a single phantom definition.

Per-tissue detail is on [tissue-skull.md](tissue-skull.md), [tissue-scalp.md](tissue-scalp.md), [tissue-csf.md](tissue-csf.md), [tissue-gray-matter.md](tissue-gray-matter.md) and [tissue-white-matter.md](tissue-white-matter.md).

**Mechanical properties for ARFI.** Region-specific brain shear moduli come from Constitutive Artificial Neural Networks [@linka2023cann]. Under the same radiation force, corpus callosum displaces about 3.4× more than cortex. Bayesian CANNs add uncertainty distributions over these moduli [@linka2025bayesian].

## Simulation Results (sbi4dwi + j-Wave)

### 2D simulations on the SCI head model

The SCI head model has 8 tissues on a 208×256×256 grid at 1 mm [@warner2019highresolution]; see [sci-head-model.md](sci-head-model.md). All 9/9 experiments ran on a Modal A100. Source: `sbi4dwi:docs/openlifu/README.md`.

| Experiment | Result |
|---|---|
| Skull attenuation, 400 kHz | **93.0%** (p_target 2.13e-4 Pa in water vs 1.49e-5 Pa with skull) |
| Multi-target attenuation (cortex / mid-brain / thalamus) | 55.4% / 55.3% / 53.9% |
| Frequency sweep, 180 kHz / 400 kHz / 1 MHz | 32.0% / 55.1% / 34.3% attenuation |
| Sensitivity, mean \|dp/dc\| | Skull 1.41e-7 (**232×** water); brain 9.27e-7; water 6.09e-10 |
| Grid convergence, 0.4 / 0.2 / 0.1 mm | p_max 0.502 / 0.311 / 0.188, still converging |
| Helmholtz (GMRES) vs PSTD on water | p_max 0.287 vs 0.311, correlation **r = 0.82** |

**Interpretation and caveats:**

- 180 kHz gives the best penetration, consistent with the frequency range chosen for OpenLIFU.
- The 1 MHz result is non-monotonic: 34.3% attenuation is lower than at 400 kHz. This is likely a slab-geometry resonance or standing-wave artefact and should not be read as better penetration at high frequency.
- The 232× skull sensitivity confirms that uncertainty in skull properties dominates focal accuracy.

### Delay optimization

Per-element delays are optimized by gradient descent through the differentiable forward model (`sbi4dwi:dmipy_jax/biophysics/tus_optimizer.py`), with 20 iterations in each case.

| Array | Loss (start → end) | Focal gain | Time/iter |
|---|---|---|---|
| 4 elements (2D) | −0.0904 → −0.0907 | 1.0× | 2.1 s |
| 16 elements (2D) | −0.1216 → −0.1545 | 1.3× | 1.8 s |
| 32 elements (2D) | −0.2991 → −0.4138 | 1.4× | 1.8 s |
| 16 elements (3D, 4 mm grid) | −1.66e-5 → −7.79e-5 | **4.7×** | 2.6 s |

The optimized 3D delay pattern alternates between about 0 ns and about 1200 ns.

### 3D volumetric simulation

At a 4 mm grid (52×64×64, 213K voxels), the 3D run gives:

- 94.3% attenuation
- 0.0 mm focal shift
- 13.0 s for the water and skull runs combined

The 2 mm run (1.7M voxels) reported −7.6% attenuation. This is flagged in the source as a target/source positioning artefact that is still under investigation.

## Full Waveform Inversion (brain-fwi)

FWI recovers $c(\mathbf{x})$ by minimizing waveform misfit, with gradients taken by JAX autodiff through j-Wave. This replaces the hand-coded adjoint-state gradients of Stride/Devito used by Guasch et al. [@guasch2020fullwaveform; @cueto2022stride]. Source: `brain-fwi:README.md`.

Design choices:

- Multi-frequency banding: 50–100 → 100–200 → 200–300 kHz.
- Envelope loss, which is robust to cycle-skipping through thick skull.
- Sigmoid-bounded velocity parameterisation.
- Two-level nested-scan gradient checkpointing, reducing memory from O(N) to O(√N).

| Run | Setup | Result |
|---|---|---|
| Single-band 3D | 96³ at 2 mm, 256-element helmet, 500 kHz, 5 iterations | 50.3 s total (10 s/iter); c range 1411–2166 m/s |
| 3-band 3D | 50→200→500 kHz, 8 sources × 32 sensors | Skull recovered to **4312 m/s** (true 4080); brain MSE 2776; 222 s on A100 |
| USCT reconstruction | 96³ at 2 mm, 128 elements, DGX Spark | Brain RMSE **52.7 m/s**; skull RMSE 1255.6 m/s; 13.3–44.5 min |
| 2D FWI on Ernie/MNI152 | 256 elements, 555 kHz | 2.6 s/iter; forward pass 7.5 s |

**Cost at publication scale** (192³ grid at 1 mm, 256 elements, 3 bands × 20 iterations):

- A single forward shot takes about 35 s.
- One FWI iteration (8 shots plus gradients) takes about 50 min.
- The full inversion takes about **23 h**.
- Naive autodiff needs 218 GB for carries; checkpointing reduces this to about 13 GB, which fits on a 128 GB GB10.

See [compute-infrastructure.md](compute-infrastructure.md).

## Device Range

| Tier | Elements | Examples | Status in sbi4dwi |
|---|---|---|---|
| Entry | 1–4 | NeuroFUS CTX-500, Sonic Concepts | Supported |
| Mid-range | 16–32 | OpenLIFU (Openwater Health) | Primary target, full optimization pipeline |
| High-end | 128–256 | Oxford-UCL 256-element helmet (semi-ellipsoidal Fibonacci spiral, MRI-compatible) [@martin2025ultrasound; @martin2025tfus] | Scalable architecture |
| Research / custom | Arbitrary | BabelBrain-supported transducers [@pichardo2023babelbrain], Insightec ExAblate | Any geometry via array API |

The prior revision listed ExAblate as mid-range. It has been moved to the custom tier to match `sbi4dwi:docs/openlifu/README.md`.

## Solver Landscape

| Solver | Discretization | Gradients | Notes |
|---|---|---|---|
| j-Wave [@stanziola2022jwave; @stanziola2023jwave] | Fourier pseudospectral / FD (JaxDF) | JAX autodiff | Time-domain and Helmholtz; GPU/TPU; k-Wave port |
| k-Wave | k-space pseudospectral | — | MATLAB/C++ reference |
| Stride / Devito [@cueto2022stride] | Finite difference, C codegen | Adjoint-state | Imperial brain-FWI group |
| BabelBrain [@pichardo2023babelbrain] | Multi-solver planning | — | Clinical planning, MRI/CT input |
| PRESTUS | k-Wave + SimNIBS | — | Subject-specific pipelines |

The FEM-based coupled acoustic–electric solvers are covered in [acoustoelectric-imaging.md](acoustoelectric-imaging.md) and [method-fem.md](method-fem.md).

## Multimodal MRI–TFUS–ARFI Chain

```
j-Wave pressure → F = 2αI/c → µ∇²u + F = ρü → Bloch sim with motion (MR-ARFI phase)
```

**MR-ARFI acquisition.** MR-ARFI uses GRE with bipolar motion-sensitizing gradients. The encoding principle is the same as diffusion encoding, but applied to deterministic displacement. It achieves about 0.1 µm sensitivity at about 100 ms per image [@kaye2011mrarfi; @kaye2013mrarfi]. PRF thermometry provides temperature monitoring [@rieke2008mr].

**Simulator choice.** KomaMRI is the preferred simulator because it is Pulseq-compatible and supports time-dependent displacement through Motion objects. POSSUM lacks MSG support, and JEMRIS is slow. See [physics-bloch.md](physics-bloch.md).

**Gap.** The remaining missing link is coupling the displacement field to simulated spin positions (`sbi4dwi:dmipy_jax/.../mr_arfi`, `multimodal_tus`).

## Relevant Projects

- **sbi4dwi**
  - `biophysics/acoustic.py`: label-to-property and HU mapping
  - `mesh_rasterizer.py`: SCI tetrahedral mesh to grid
  - `jwave_adapter.py`, `tus_optimizer.py`, `tus_solution_export.py`: export to the OpenLIFU Solution format
  - `radiation_force`, `fwi_us`, `mr_arfi`, `multimodal_tus`
- **jwave** (upstream ucl-bug): differentiable pseudospectral time-domain and Helmholtz acoustic solver
- **brain-fwi**
  - 3D FWI with helmet arrays and checkpointed scans
  - BrainWeb, MIDA and synthetic phantoms with ITRUSST properties
- **openlifu-python**
  - Clinical bridge for transducer arrays, skull segmentation and phase correction
  - `HeterogeneousSkullSegmentation` branch
- **open-fus-plan**: open functional-ultrasound bench (power Doppler, ULM, clutter filtering). It is a companion imaging modality, not transcranial therapy; see [fus.md](fus.md).
- **ae-sim**: acoustoelectric imaging, which reuses the acoustic forward model.

## Key Challenges

1. **Skull heterogeneity.** Focal pressure is 232× more sensitive to skull speed than to water. CT/HU or FWI estimates are needed.
2. **Standing waves and reverberation.** These are visible as the non-monotonic attenuation-vs-frequency result above.
3. **Cycle-skipping in FWI.** It requires low-frequency starting bands and envelope losses.
4. **Memory and compute.** 3D autodiff through more than 1000 time steps requires gradient checkpointing.
5. **Grid convergence.** Pressure has not yet converged at 0.1 mm in 2D tests.
6. **Safety.** Thermal dose and mechanical index constraints apply, monitored with PRF thermometry [@rieke2008mr].

## See Also

- [physics-acoustic.md](physics-acoustic.md): wave equation, absorption and PML
- [tissue-acoustic-properties.md](tissue-acoustic-properties.md): cross-tissue acoustic property tables
- [tissue-skull.md](tissue-skull.md): cortical and trabecular bone properties
- [fus.md](fus.md): functional ultrasound imaging (hemodynamic)
- [sci-head-model.md](sci-head-model.md) and [head-model-mida.md](head-model-mida.md): head models used in simulations
- [acoustoelectric-imaging.md](acoustoelectric-imaging.md): coupled acoustic–electric imaging
- [physics-bloch.md](physics-bloch.md): MR-ARFI signal simulation
- [method-sbi.md](method-sbi.md): posterior inference over skull properties
- [jax-ecosystem.md](jax-ecosystem.md): JAX tooling used by j-Wave and brain-fwi
- [structural-mri.md](structural-mri.md): anatomical input for skull segmentation
- [temporal-interference.md](temporal-interference.md) and [tms.md](tms.md): alternative non-invasive stimulation modalities
