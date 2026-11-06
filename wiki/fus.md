---
type: modality
title: Functional Ultrasound Imaging
physics: acoustic
measurement: >-
  Power Doppler intensity of red-blood-cell backscatter after spatiotemporal (SVD)
  clutter filtering. It is approximately proportional to cerebral blood volume (CBV)
  in small vessels, so it serves as a haemodynamic proxy for neural activity through
  neurovascular coupling. The ULM variant localises and tracks individual contrast
  microbubbles.
spatial_resolution: >-
  "Sub-100 µm" (open-fus-plan paper abstract). Domain rule of thumb, not from the
  source files: ~100 µm in-plane at ~15 MHz in rodents, a few hundred µm in elevation,
  and roughly mm-scale at transcranial frequencies. ULM resolves vessels of a few µm.
temporal_resolution: >-
  "Sub-second" Doppler maps (paper abstract). Raw compounded frames run at hundreds
  of Hz to kHz, for example 500 Hz in Open-H-RF twente-vortexflow. The open-bench
  target is live power Doppler at ≥ 10 fps.
related: [tus.md, physics-acoustic.md, tissue-skull.md, tissue-acoustic-properties.md, physics-hemodynamic.md, physics-cerebrovascular.md, fmri.md, fnirs.md, acoustoelectric-imaging.md, head-model-mida.md, jax-ecosystem.md, data-formats.md, compute-infrastructure.md, method-sbi.md, coordinate-systems.md, neural-mass-models.md, method-full-waveform-inversion.md]
description: "Functional ultrasound imaging maps changes in cerebral blood volume (CBV) using ultrafast plane-wave Doppler."
timestamp: 2026-11-05T15:59:53-08:00
tags: [modality, acoustic, hemodynamic, ultrasound]
---

# Functional Ultrasound Imaging (fUS / fUSI)

Functional ultrasound imaging maps changes in **cerebral blood volume (CBV)** using ultrafast plane-wave Doppler. It produces fMRI-like functional maps at sub-second, sub-100 µm resolution (`open-fus-plan:paper/main.tex`, abstract).

The signal is haemodynamic, like [fMRI](fmri.md) BOLD and [fNIRS](fnirs.md). It is coupled to neural activity through the mechanisms described in [physics-hemodynamic.md](physics-hemodynamic.md) and [physics-cerebrovascular.md](physics-cerebrovascular.md). The physics, however, is acoustic ([physics-acoustic.md](physics-acoustic.md)): the scanner measures Doppler-shifted backscatter from moving red blood cells.

Two features separate fUS from conventional colour Doppler:

- **Ultrafast plane-wave acquisition.** Each transmit insonifies the whole field of view, so the frame rate is limited only by the round-trip time of flight.
- **Spatiotemporal (SVD) clutter filtering.** This separates the slow-flow blood signal from tissue motion that is orders of magnitude stronger.

## History

All facts in this section come from `open-fus-plan:paper/main.tex` §2. That paper marks the Caltech primate citations as unverified.

- **Origin (2011, ESPCI Paris).** fUS began in Mickael Tanter's group at ESPCI Paris, now Physics for Medicine Paris. The key insight was that imaging at thousands of frames per second raises Doppler sensitivity enough to see CBV changes in small vessels. The group, with Lenkei and Osmanski, then showed resting-state connectivity in rats, followed by awake and freely moving animals.
- **Commercialisation.** Iconeus was founded in 2016, and Inserm, CNRS and ESPCI licensed the technology in February 2018. The Iconeus One is a portable plane-wave station with a motorised probe platform for head-fixed animals. The same hardware performs ULM at a resolution of a few µm. Its price is on request and estimated at six figures (`open-fus-plan:docs/BOM.md`).
- **Human BCI branch.** At Caltech, Norman with the Andersen and Shapiro labs decoded movement intentions from fUS in non-human primates. This line of work led to Forest Neurotech, later renamed Arbor Neuroscience. In March 2025 Forest began a 3.5-year NHS exploratory study of Forest 1, a wearable fUS imager placed over **craniectomy windows** and funded with £6.5M from ARIA. Merge Labs was spun out of Forest; the spin-out was reported in December 2025, and Merge emerged in January 2026 with a $252M seed round.
- **Transcranial ULM (June 2026).** Aleph Neuro reported 3D ULM of the human cerebral vasculature through the intact skull, using Butterfly Ultrasound-on-Chip hardware. Acquisition took about 4 min under continuous microbubble infusion. The result is company-reported, not peer reviewed, and depends on contrast agent. Aleph released the pipeline (`alephneuro/microbubbles`) and a 96 GB dataset under the MIT licence.

## Signal chain

The canonical pipeline is implemented in `open-fus-plan` → `nvus-fus` (an SVD/power-Doppler reference) and in Aleph's `microbubbles` (GPU MACH beamforming, temporal SVD, tracking). It has five stages:

1. **Transmit.** Fire N tilted plane waves. The bench plan uses 32 angles (`open-fus-plan:docs/PLAN.md`, Phase 1).
2. **Beamform.** Apply delay-and-sum (DAS) to raw RF or demodulated IQ, giving one complex image per angle. Forest's `mach` CUDA DAS kernel is reported at ~10× the prior state of the art (`open-fus-plan:docs/SOFTWARE.md`).
3. **Coherent compounding.** Sum the angle images into one compounded frame. N_t consecutive frames form an *ensemble*.
4. **Clutter filter.** Reshape the ensemble into a Casorati matrix **S** ∈ ℂ^(N_pix × N_t). Factor it as **S** = **U Σ V**ᴴ and remove the first k singular components:
   **S**_f = Σ_{i>k} σ_i **u**_i **v**_iᴴ.
   The high-energy, temporally coherent components are tissue; the remaining subspace is blood.
5. **Power Doppler.** Compute PD(x) = (1/N_t) Σ_t |S_f(x,t)|². This is proportional to the number of moving scatterers in each voxel, i.e. relative CBV. Functional maps come from regressing PD(x, t) against a stimulus or task regressor, analogous to a GLM on BOLD.

The **ULM** branch diverges after step 4. Instead of integrating power, it localises isolated microbubbles at sub-voxel precision and links them into tracks; Aleph uses Kalman–Hungarian linking. The accumulated track density and velocity give vessel maps.

`mach` and Aleph's pipeline both consume **demodulated IQ**. Beamformed IQ, or compounded frames, is therefore the natural published data boundary between device and reconstruction. This mirrors the moment-level boundary in the Kernel Flow fNIRS Holoscan app (`open-fus-plan:docs/HOLOSCAN-PATTERNS.md`).

The bench stores frames in HDF5 using the zea format, which keeps them compatible with Open-H-RF (see [data-formats.md](data-formats.md)). Skull-corrected delay tables, element phases and SVD bases are treated as precomputed operators. They are versioned with the head model and the device serial.

Dedicated wiki pages on beamforming, SVD clutter filtering and ULM do not exist yet. This section is the current reference for them.

## Parameters

### Open bench (open-fus-plan)

All values come from `open-fus-plan:docs/BOM.md`, `docs/PLAN.md` and `paper/main.tex`.

| Parameter | Value | Notes |
|---|---|---|
| Front-end channels | 128 TX / 64 RX | TI AFE5932EVM (2× AFE5932 RX, 2× TX73L64 TX) |
| Data path | 100 GbE RoCE straight into GPU memory | Terasic Holoscan Sensor Bridge (Stratix 10 TX) → DGX Spark ConnectX-7 |
| Shot-level control | ~1,500 transmit updates/s, set by the GPU | NVIDIA Holoscan Ultrasound Imaging Reference Design, Apache-2.0 |
| Probe | Sonosray L12-5, 5–12 MHz linear | Rodent or through-window imaging; **not** transcranial human |
| Compounding | 32 plane-wave angles | Phase 1 target |
| Live output target | Power Doppler of a flow phantom at ≥ 10 fps | Phase 1 deliverable |
| Imaging hardware cost | $10–18k est | Compare "six figures est" for the Iconeus One |

### Benchmark data and timings

| Item | Value | Source |
|---|---|---|
| Aleph transcranial ULM dataset | 96 GB; 216 acquisitions, each 240 frames × 4 TX × 8 elevation rows × 120 elements IQ | `open-fus-plan:docs/PLAN.md` |
| `mach` DAS kernel | 29 ms per 30-frame chunk on GB10 and A10G; 13 ms on H100 (~3 s per acquisition) | `nvus-fus/beam/mach_kernel_bench.py` |
| Default grid, one acquisition | 154 × 25 × 275 voxels. On the Spark: 9.3 s after GPU compounding (was 188 s), RSS 6.5 GB. With spatial TGC: 56 s (was 10 min 50 s) | `gpu-compound` branch of `m9h/microbubbles` |
| Full Aleph sample on Spark | ~35 min without TGC; ~3.4 h with TGC | `open-fus-plan:docs/PLAN.md` |
| Open-H-RF `twente-vortexflow` | Flow phantom with optical ground truth; curved array, 3.4 MHz; 750 frames at 500 Hz; 6 × 1.5 GB | First replay target, run on CPU with zea + `nvus-fus` |
| Open-H-RF `ulmshare` | 99 transcranial mouse-brain ULM acquisitions; Verasonics 256; ~180 GB each | Stream; do not download |

The timings carry a practical lesson: once the beamforming kernel is fast, **host-side array handling dominates**. On the GB10, NumPy's default `NUMPY_MADVISE_HUGEPAGE=1` made a 2 GB copy take 3.8 s instead of 0.4 s. Compounding on the GPU and copying once fixed the problem, and the output was bit-identical. Peak host memory with TGC fell from 87 GB to 21 GB of the shared 120 GB pool. See [compute-infrastructure.md](compute-infrastructure.md).

## The skull problem

fUS works best through thin bone, a craniotomy or craniectomy window, or the infant fontanelle. Skull bone differs sharply from soft tissue in its acoustic properties (see [tissue-skull.md](tissue-skull.md) and [tissue-acoustic-properties.md](tissue-acoustic-properties.md)):

| Medium | Sound speed (m/s) | Source |
|---|---|---|
| Water | 1500 | `brain-fwi:docs/datasets/icl-dual-probe-2023.md` |
| Water + brain-phantom path (path-weighted) | 1527 (fit); brain phantom ~1540 | First-arrival regression on ICL dual-probe synthetic traces [@robins2023dualprobe] |
| Trabecular (diploë) skull | 2300 | ITRUSST benchmark [@Aubry2022benchmark] |
| Cortical skull | 2800 | ITRUSST benchmark [@Aubry2022benchmark] |

This contrast in sound speed and impedance causes three problems:

- aberration of the transmit and receive wavefronts;
- mode conversion;
- attenuation that rises steeply with frequency.

The result is a direct trade-off. Transcranial penetration needs hundreds of kHz up to ~1 MHz, while fUS resolution needs multi-MHz frequencies. There are four current routes around it:

1. **Acoustic windows**, as used by Forest 1.
2. **Contrast agents with ULM**, as used by Aleph.
3. **Skull-corrected beamforming delay tables.** These can be precomputed per subject from CT/MR using pseudospectral or FDTD solvers: k-Wave, j-Wave [@stanziola2022jwave] or BabelViscoFDTD.
4. **Quantitative sound-speed maps of skull and brain** recovered by full-waveform inversion (FWI) [@guasch2020fullwaveform]. Experimental support includes a 2-D low-frequency acquisition device [@CudeiroBlanco2022device], transducer calibration [@Cueto2021sri_brain; @Cueto2022sri_robust], the Stride platform [@Cueto2022stride], and a PVA-skull phantom study [@robins2023dualprobe].

FWI images *acoustic properties*, not blood volume. It complements fUS, because an FWI sound-speed map is exactly the medium needed to build aberration-corrected delays for transcranial Doppler.

### brain-fwi parameters

These values come from the `brain-fwi:README.md` 192³ configuration.

| Parameter | Value |
|---|---|
| Grid | 192³ at 1 mm; >10 points per wavelength |
| Array | 256-element helmet (3D) or ring (2D) |
| Frequency bands | 50–100 → 100–200 → 200–300 kHz, 20 iterations each |
| Single forward shot | ~35 s on GPU |
| Full inversion | ~23 h on GB10 (128 GB unified memory) |
| Peak memory | 218 GB naive autodiff → ~13 GB with two-level O(√N) checkpointing [@griewank2000algorithm] |
| Optimiser / loss | Adam [@kingma2014adam]; envelope loss against cycle-skipping; sigmoid-bounded velocity |
| Head models | BrainWeb (1 mm, 12 classes); [MIDA](head-model-mida.md) (500 µm, 153 structures, 3 skull layers) |

The ICL dual-probe dataset adds real PVA-skull transmission data: 2× P4-1 probes, 384 sources, 442,368 traces of 3,328 samples, with dt inferred as 44 ns (22.727 MHz). In the 200 kHz–1 MHz band, synthetic and experimental traces have a median correlation of 0.871 (`brain-fwi:docs/datasets/icl-dual-probe-2023.md`).

## Comparison with other haemodynamic modalities

Resolution and depth values in this table are general domain figures, not taken from the source files.

| | fUS | [fMRI](fmri.md) (BOLD) | [fNIRS](fnirs.md) |
|---|---|---|---|
| Contrast | CBV (Doppler power) | Deoxy-Hb susceptibility | HbO/HbR absorption |
| Spatial resolution | ~100 µm (window or rodent) | ~1–3 mm (sub-mm laminar at 7 T) | ~1–3 cm |
| Depth | Whole rodent brain; human limited by skull | Whole brain | ~1.5–2 cm of cortex |
| Portability | Wearable possible (Forest 1) | No | Wearable, e.g. Kernel Flow [@Ban2022kernelflow] |
| Skull | Major obstacle | Transparent | Scattering, but passable |

## Safety

fUS runs at diagnostic exposure levels, but long ultrafast ensembles still accumulate heat. The bench's safety supervisor therefore applies one set of limits to both imaging and [TUS](tus.md) stimulation. It tracks four quantities against ITRUSST 2025 levels, keeping FDA intensities for reference:

- MI_tc;
- temperature rise;
- TI exposure-time limits;
- CEM43.

The supervisor is `nvus-fus:nvus_fus/safety.py`: version v0, with 21 tests and advisor review pending (`open-fus-plan:docs/PLAN.md`). In Phase 2 it gates the HV enable, and every sonication is logged.

## Relevant Projects

- **open-fus-plan** — Plan for an open fUS + LIFU phantom bench. It covers:
  - the Holoscan acquisition chain;
  - the `nvus-fus` SVD/power-Doppler reference, zea replayer and safety supervisor (33 tests);
  - the Aleph ULM replay with the `mach` GPU-compounding fix.
  
  Key files are `docs/PLAN.md`, `docs/SOFTWARE.md`, `docs/HOLOSCAN-PATTERNS.md` and `docs/BOM.md`. Phase 4 targets one shared frame for probe, array, phantom and plan (see [coordinate-systems.md](coordinate-systems.md)).
- **brain-fwi** — Recovers transcranial sound speed by FWI, using JAX autodiff through j-Wave. Its outputs are medium models for skull-corrected beamforming. It includes an ICL dual-probe loader and a Phase-0 data factory of 10⁴–10⁵ (θ, d) pairs for amortised inference ([method-sbi.md](method-sbi.md)).
- **jwave** — Differentiable pseudospectral acoustic solver [@stanziola2022jwave], licensed LGPL-3.0. It is the forward model for delay tables, FWI and transmit planning, and runs on the DGX Spark GPU ([jax-ecosystem.md](jax-ecosystem.md)).
- **ae-sim** — Acoustoelectric forward model built on the same acoustic core (BabelViscoFDTD) and the same arrays ([acoustoelectric-imaging.md](acoustoelectric-imaging.md)).
- **vbjax** (indirect) — Neural-mass simulations ([neural-mass-models.md](neural-mass-models.md)). Coupled to a haemodynamic forward model, these could predict CBV-like fUS signals.

## See Also

- [Full-Waveform Inversion](method-full-waveform-inversion.md)
- [tus.md](tus.md) — transcranial ultrasound *stimulation*, using the same arrays and solvers
- [physics-acoustic.md](physics-acoustic.md) — wave equation, attenuation, pseudospectral and FDTD solvers
- [tissue-skull.md](tissue-skull.md), [tissue-acoustic-properties.md](tissue-acoustic-properties.md) — ITRUSST acoustic properties
- [physics-hemodynamic.md](physics-hemodynamic.md), [physics-cerebrovascular.md](physics-cerebrovascular.md) — neurovascular coupling and CBV
- [acoustoelectric-imaging.md](acoustoelectric-imaging.md) — electrical readout path for the same arrays
- [fmri.md](fmri.md), [fnirs.md](fnirs.md) — other haemodynamic modalities
- [head-model-mida.md](head-model-mida.md) — head model used for acoustic simulation
- [data-formats.md](data-formats.md) — HDF5, zea, Open-H-RF
- [compute-infrastructure.md](compute-infrastructure.md) — DGX Spark memory caps and timings
- [method-sbi.md](method-sbi.md) — amortised posteriors over FWI sound-speed maps
