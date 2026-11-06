---
type: modality
title: Acoustoelectric Neuroimaging
physics: electromagnetic   # current-density readout; spatial selectivity comes from acoustic coupling (see physics-acoustic.md)
measurement: Low-frequency current density (neural or injected), tagged by a focused ultrasound pulse through pressure-induced conductivity modulation and read as a voltage on electrode pairs
spatial_resolution: ~1–8 mm; set by the ultrasound focus (4.72 × 7.85 × 4.46 mm through human skull, Allard 2025; < 2 mm in rat, Tianjin line)
temporal_resolution: Sub-ms to ms; the current waveform is demodulated from a 0.5–1 MHz carrier (e.g. 4 kHz pulse repetition in Allard 2025), but low SNR means averaging over repeats
related: [physics-acoustic.md, physics-electromagnetic.md, method-fem.md, method-bem.md, tus.md, fus.md, eeg.md, temporal-interference.md, method-source-imaging.md, neuronal-current-mri.md, tissue-electrical-properties.md, tissue-acoustic-properties.md, tissue-skull.md, tissue-csf.md, jax-ecosystem.md, compute-infrastructure.md, data-formats.md]
description: "Acoustoelectric (AE) imaging uses a focused ultrasound beam to tag the electrical current flowing at the focus."
timestamp: 2026-11-05T15:59:53-08:00
tags: [modality, acoustic, electromagnetic, ultrasound]
---

# Acoustoelectric Neuroimaging

Acoustoelectric (AE) imaging uses a focused ultrasound beam to **tag** the electrical current flowing at the focus. Pressure modulates tissue conductivity by a small fraction. Electrodes, which can be on the scalp or in depth, then pick up a voltage at the ultrasound frequency, and that voltage is proportional to the local current density. Scanning the focus produces a current-density image at roughly ultrasound resolution. EEG, by contrast, is limited by the smoothing effect of the volume conductor. The same physics run in reverse is **acoustoelectric heterodyning**. There, an applied field plus an ultrasound beam *produces* a localised field at the sum and difference frequencies. This links AE imaging to [temporal-interference.md](temporal-interference.md)-style focal stimulation (source: `ae-sim:README.md`).

## Physical principle

**Conductivity modulation.** The coupling is defined by

  Δσ / σ₀ = −K ΔP

Here K is the acoustoelectric coupling constant, a fractional conductivity change per pascal. The value of K is unsettled; see the table below.

**Imaging mode (reciprocity).** A recording pair with lead field **J_L** (current density per ampere injected through the pair) sees the following voltage:

  V_AE(t) = −K ∫ ΔP(**r**, t) ρ₀ **J_L**(**r**) · **J_I**(**r**) dV

**J_I** is the current being imaged. The signal is therefore the pressure-weighted overlap of the lead field and the source current. This is the same reciprocity structure used in [method-source-imaging.md](method-source-imaging.md) and the EEG forward problem ([eeg.md](eeg.md)) (source: `ae-sim:README.md`).

**Heterodyning / forward mode.** The AE potential φ₁ satisfies a Poisson problem with a pressure-weighted source:

  −∇·(σ∇φ₁) = −∇·(σ K P **E₀**)

For uniform σ this reduces to ∇²φ₁ = K **E₀**·∇P. In the plane-wave limit it gives |E₁| = K P E₀ exactly, which serves as a unit test in `ae_sim.spectral`. Inside a focal spot, the volume conductor cancels a fraction N of the local modulation, so that E₁ = −N K P E₀. N is the depolarisation factor of the spot shape. Measured values are:

- sphere: 0.334
- prolate spheroid of aspect 8 oriented along E₀: 0.030
- the same spheroid oriented across E₀: 0.487

As a result, the aim of the beam relative to the stimulation field changes the AE current by about 2× (source: `ae-sim:docs/research/05-prototype-findings.md`). The governing acoustics are described in [physics-acoustic.md](physics-acoustic.md) and the quasi-static electrodynamics in [physics-electromagnetic.md](physics-electromagnetic.md).

## The coupling constant K

Every AE number scales linearly with K. Published values span two orders of magnitude.

| Source | K (1/Pa) | K (% / MPa) | Medium | Note |
|---|---|---|---|---|
| ae-sim default (`song`) | 1e-9 | 0.100 | — | "from Xizi Song's paper", per the Rintoul solver comments |
| Preston 2018 (Witte group) | 3.4e-10 | 0.034 | 0.9 % NaCl | Value used by the Arizona line |
| Wang 2026, IEEE TBME | 2.4e-10 | 0.024 | saline | Direct measurement |
| Wang 2026, IEEE TBME | 2.2e-10 | 0.022 | porcine brain | Direct measurement |
| Rintoul 2023 solver comment | 4.6e-10 | 0.046 | water | "Physical" adiabatic value |
| Rintoul 2023 solver (hard-coded) | 2e-8 | 2 | saline | Chosen to fit the tank data |

Source: `ae-sim:docs/research/04-coupling-constant.md`.

The measured values are 0.022–0.034 %/MPa. Under those values, real-head estimates fall ~3–4× and the required averages rise ~10–16×. In ae-sim, K is a runtime parameter (`AE_SIM_K=witte | wang-brain | 0.034%/MPa | <value>`). Each result records the K it used, and injected-current results are reported in µV/(mA·MPa).

## Experimental benchmarks and replications

There are two experimental lines:

- **Witte / O'Donnell line (Arizona).** Uses 0.6–1 MHz, saline, gel and human-skull phantoms, and reports *absolute* sensitivities.
- **Tianjin line.** Rat in vivo work with phased arrays and skull aberration correction, mostly paywalled.

The **Rintoul / Grossman** papers (Imperial) are the most completely specified open setups (source: `ae-sim:docs/research/01-replication-targets.md`, `02-rintoul-heterodyning-and-in-vivo.md`).

| Experiment | Setup | Measured | ae-sim simulation |
|---|---|---|---|
| Preston 2018, UMB | Medtronic 3389 DBS lead (1.27 mm × 1.5 mm contacts), 0.9 % NaCl, 1 MHz focused at 63 mm, 200 Hz current | **0.52 µV/(mA·MPa)**; 0.56 through a 6 mm skullcap; threshold 1.75 mA at 1 MPa | 0.47 with a Gaussian beam; 0.83 with the BabelViscoFDTD A392S beam, both at K = 0.1 %/MPa. Peak 0.43 / 0.10 mm from the contact (paper: < 0.41 mm) |
| Kang 2023, TMI; Cheong 2026 | Dipole in an 8 cm saline cube (1.2 S/m), 18 × 7 array at 0.6 MHz, focus 35 mm, 1 MPa, 3-axis leads | COMSOL + Field II / k-Wave maps | Planned (~8 M FDTD cells at 0.6 MHz) |
| Allard 2025, TBME; Barragan 2020, JNE | Human skull + agarose, 12-contact depth electrode, 126-element 0.6 MHz array, 0.7 MPa through skull (60–70 % attenuation) | **1.5 µV/mA at 0.7 MPa**; resolution 4.72 × 7.85 × 4.46 mm; NEC 360 µA (2020); 28 µA with a Wiener filter (Abu Farha 2026) | Planned (SimNIBS `ernie` with gel σ) |
| Rintoul 2023, Comms Phys (heterodyning) | 50 × 20 × 20 cm tank split by a 12 µm PVDF film; 500 kHz bowl, ROC 63 mm, 1 MPa; Pt-Ir pins 3 mm apart, 250 V/m at 8 kHz | ~1 V/m at sum/difference frequencies; parallel/perpendicular ratio 4.12 | 0.23 V/m at K = 1e-9 with real pins; ratio 3.54 |
| Rintoul 2026, Nat Commun (neuromodulation) | Mouse, CW 500 kHz, 0.5–2.5 MPa, mouth + head loop electrodes | ~9 mV onset potential, ~6 V/m estimated | Digimouse: |E₁| 0.12 V/m at 20 V, 1 MPa |
| Tianjin rat DBS (Zhou 2024/2026) | Rat DBS contact, transcranial focus | 167.74 µV V⁻¹ MPa⁻¹, < 2 mm resolution | SIGMA rat head, generic parameters |

**The Rintoul "K gap".** The 2023 simulation at K = 1e-9 fell 20–40× short of the tank measurement. Two ae-sim findings bear on this:

- Modelling the real pins alone reduces the gap to ×4.3, so K ≈ 4.3e-9 would fit.
- A fresh FFT solve of the stated equation on the authors' own deposited pressure grid gives fields **35× larger** than their deposited `ae_data_kspace_*` outputs, in both orientations, with matching shape.

The "K ≈ 2e-8 needed" conclusion therefore appears to rest on a numerical factor rather than on the physics. This interpretation is unconfirmed and remains an open question for the authors. The PVDF film and the oil side change the result by < 0.1 % (source: `ae-sim:docs/research/02-rintoul-heterodyning-and-in-vivo.md`).

## Key quantitative findings (ae-sim prototype, K = 1e-9 /Pa)

| Quantity | Value | Note |
|---|---|---|
| Real head (SimNIBS `ernie`): 20 nA·m dipole 15 mm under C3, 0.3 MPa, C3–C4 pair | 32 pV at 1 MHz; 268 pV at 0.25 MHz | Lower frequency is the strongest lever |
| Finite insulated bath vs infinite medium | ~5× gain (14.0 vs 2.69 nV) | Boundaries matter |
| FDTD piston near field vs ideal beam | ~4× | — |
| Mesh-granularity uncertainty (real head) | 1.5–3× | Worst at high frequency; trilinear resampling drops 31.5 → 11.0 pV |
| Digimouse head + gel, 20 V drive | 42 mA, 386 V/m median in brain, φ₁ peak 510 µV, motor−visual +54 µV | E₁ = 0.22 K P E₀ |
| Somatic polarisation from E₁ = 0.12 V/m | ~23 µV (0.2 mV per V/m) | 1/50 of the 6 V/m estimated in vivo |
| Cone artefact at 500 kHz | 22 mV/V common mode; 0.77 mV/V on the motor−visual pair | Capacitive film path |

Source: `ae-sim:docs/research/05-prototype-findings.md`, `ae-sim:README.md`.

**Physics lessons:**

- **A radiated pulse has no DC.** The AE trace is the coupling band-pass filtered by the pulse spectrum. Smooth currents therefore appear only through their axial gradients, and CW or long bursts image gradients only. Pulses of about one cycle preserve the low-frequency content.
- **Electrode orientation.** Crossed electrode pairs null the centre line.
- **Circuit state at each frequency.** Electrode circuits matter as much as fields. In the Preston replication, the stimulating contact acts as a sink at 1 MHz (50 Ω generator), while idle Pt-Ir contacts (~1 µF double layer) are open at 200 Hz. Getting these wrong moves the peak to the wrong contact, or cancels it entirely.
- **Skull and the lead field.** A nearer scalp pair does not help, because the skull sets the cortical lead field ([tissue-skull.md](tissue-skull.md)).
- **Neuromodulation mechanism.** The heterodyned field is at best a modulator. The 9 mV onset response needs another mechanism; the 500 kHz field's activating function is 4 orders of magnitude larger.

## Simulation pipeline

ae-sim builds both modes from seven blocks: head model, acoustics, lead fields, sources, mixing, acquisition noise, and validation. A 39-stage BATS pipeline asserts the numbers against CPU baselines.

| Block | Tool | Notes |
|---|---|---|
| Head model | SimNIBS 4.6 `charm` / `create_mesh` | Meshes any labelled volume; rodent element sizes 0.1–0.5 mm |
| Acoustics | BabelViscoFDTD 1.2.7 (CPU / CUDA / OpenCL) | λ/6 grid. The Preston transducer needs 12 M cells and runs in 326 s on 4 threads. A mouse skull (~0.3 mm) needs a ~0.1 mm grid |
| Lead fields / forward | scikit-fem P1; `AEForwardSolver` with pyamg smoothed aggregation + CG | 442 k nodes: 16 s setup, 6 s per pressure snapshot; discrete reciprocity exact to 1e-8 |
| Complex quasi-static | `ComplexForwardSolver`, σ + iωε at 500 kHz | Cone artefact, 159 k unknowns |
| Spectral cross-check | `ae_sim.spectral` FFT Poisson solve | Plane-wave limit as a unit test |

FEM details are in [method-fem.md](method-fem.md); conductivities are in [tissue-electrical-properties.md](tissue-electrical-properties.md) and [tissue-csf.md](tissue-csf.md); acoustic parameters are in [tissue-acoustic-properties.md](tissue-acoustic-properties.md).

**Complex Helmholtz solves (fem-trials).** These are needed for CW acoustics with air as a pressure-release boundary. Benchmarks used dolfinx 0.11 complex with PML. With Fedora's SuperLU_DIST, only MMD ordering works, because the METIS ordering aborts. Under MMD, 132 k DOFs exceed 10 GB. python-mumps with METIS ordering scales much better:

| DOFs | Factorisation (x86 laptop) | Factorisation (DGX Spark, 4 threads) | Memory |
|---|---|---|---|
| 68,921 | 3.5 s | 10.3 s | 1.19 GB |
| 274,625 / 357,911 | 32.5 s (275 k) | 210 s (358 k) | 7.17 / 10.3 GB |
| 1,030,301 | — | 1481 s; 23 s per re-solve | 42.0 GB |

Accuracy of P1 elements at 10 points per wavelength is about 14.5 % median error against the free-space Green's function. Percent-level accuracy needs P2 elements or ≥ 20 ppw. A rodent head at 1 MHz (~28 mm) is therefore in the 1e6-DOF regime. On Grace cores, `OMP_NUM_THREADS=4` is recommended (source: `fem-trials:README.md`; hosts in [compute-infrastructure.md](compute-infrastructure.md)).

**Differentiable acoustics.** j-Wave [@stanziola2022jwave] provides JAX pseudospectral time-domain and Helmholtz (GMRES, PML) solvers, partly ported from k-Wave. ae-sim does not currently use it, but it is the natural differentiable replacement for the FDTD block ([jax-ecosystem.md](jax-ecosystem.md)). brain-fwi already differentiates through j-Wave to recover skull and brain speed-of-sound maps by full-waveform inversion [@guasch2020fullwaveform]. Those maps are exactly the skull-acoustics input that AE head models still lack. Transcranial simulation accuracy is benchmarked across solvers in [@Aubry2022benchmark].

## Open head models and data

| Resource | Content | Licence | Size |
|---|---|---|---|
| SimNIBS `ernie` | Low-res human head | CC BY-NC 4.0 | 413 MB |
| Digimouse | 0.1 mm mouse labels; 58 k-node / 307 k-tet mesh | CC BY-SA 4.0 | 0.9 + 4.2 MB |
| SIGMA v2 | Wistar rat whole head with skull maps; classified at 0.18 mm | CC BY 4.0 | 1.5 GB |
| Rintoul 2023 tank | k-space pressure phasor (101 × 101 × 1401), AE outputs, streams | CC BY 4.0 (code CC BY-NC-SA 4.0, reference only) | 229–653 MB |

Large files are fetched by checksummed `scripts/fetch-*.sh` and never packaged in the COPR (source: `ae-sim:docs/research/03-rodent-head-models.md`; formats in [data-formats.md](data-formats.md)).

## Relevant Projects

- **ae-sim** — end-to-end AE imaging and heterodyning simulator. Includes the coupling-constant registry, FEM forward path, Preston and Rintoul replications, and Digimouse / SIGMA rodent heads.
- **fem-trials** — dated solver experiments for ae-sim and ti-sim: dolfinx complex path, SuperLU_DIST vs MUMPS memory scaling, DGX Spark timings.
- **jwave** — differentiable JAX acoustic solvers (time-domain and Helmholtz with PML), a candidate for differentiable AE acoustics.
- **brain-fwi** — transcranial FWI on j-Wave; supplies skull speed-of-sound maps for transcranial focusing.
- **ti-sim** — shares the FEM field-solve stack and the focal-stimulation framing (see [temporal-interference.md](temporal-interference.md)).
- **neuroelectric-mri** — alternative route to imaging neuronal current density ([neuronal-current-mri.md](neuronal-current-mri.md)).
- **neurofedora** — packages SimNIBS, BabelViscoFDTD, dolfinx and MUMPS used by the pipeline.

## See Also

- [physics-acoustic.md](physics-acoustic.md) — wave propagation, focusing, attenuation
- [physics-electromagnetic.md](physics-electromagnetic.md) — quasi-static volume conduction
- [tus.md](tus.md), [fus.md](fus.md) — transcranial ultrasound stimulation and functional ultrasound
- [eeg.md](eeg.md), [method-source-imaging.md](method-source-imaging.md) — lead fields and reciprocity
- [temporal-interference.md](temporal-interference.md) — other frequency-mixing focal stimulation
- [neuronal-current-mri.md](neuronal-current-mri.md) — competing current-density imaging modality
- [method-fem.md](method-fem.md), [method-bem.md](method-bem.md) — forward solvers
- [tissue-skull.md](tissue-skull.md), [tissue-acoustic-properties.md](tissue-acoustic-properties.md), [tissue-electrical-properties.md](tissue-electrical-properties.md), [tissue-csf.md](tissue-csf.md)
- [jax-ecosystem.md](jax-ecosystem.md), [compute-infrastructure.md](compute-infrastructure.md), [data-formats.md](data-formats.md)
