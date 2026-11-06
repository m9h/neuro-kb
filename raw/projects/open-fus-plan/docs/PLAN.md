# Plan

Five subsystems, four phases, phantom-only. Weeks are indicative for a student team working
part time; phases overlap.

## Subsystems

1. Imaging front end — Holoscan Ultrasound Imaging Reference Design on the DGX Spark.
2. Stimulation front end — Open-LIFU 64-element array, transmit module and console (or a
   single-element transducer + RF amplifier to start).
3. Planning — SimNIBS head model → BabelBrain / k-Wave / j-Wave → element phases.
4. Navigation — SlicerOpenLIFU and the Open-LIFU 3D-scanner workflow; optical tracking later.
5. Readout and closed loop — GPU fUS pipeline now; acoustoelectric decoding (`ae-sim`) later.

## Phase 0 — software only (weeks 1–6)

- Install NDK, j-Wave, BabelBrain, openlifu-python on the Spark (containers, aarch64).
  *openlifu-python 0.18 and j-Wave already present in `~/dev`; NDK imports on aarch64 as of
  2026-10-26 (devito 4.7.1 needs `codepy<2025`; set `DEVITO_ARCH=gcc`); BabelBrain pending.*
- Reproduce a transcranial focus through a CT-derived skull in each tool; compare.
- Run `ae-sim` end to end; document inputs and outputs.
- Replay public raw-channel data (Open-H-RF) through a beamform → SVD → power-Doppler chain.
  *Done 2026-10-25 on CPU: `twente-vortexflow` flow phantom via zea + `nvus-fus`; GPU run
  pending JAX CUDA 13 on aarch64. Holoscan reference design's `./nvus run --dry-run` and
  `tests/installation.py` are the no-hardware checks for its container; not yet run.*
- Build Forest's `mach` beamformer for sm_121 and run Aleph's ULM pipeline on its 96 GB
  sample (download in progress on the T9). *Done 2026-10-25: `mach` builds and imports on the
  GB10 (arch 121); Aleph sample (96 GB, 216 acquisitions of 240 × 4 tx × 8 elevation rows × 120
  elements IQ) on the T9; one acquisition beamformed at coarse grid (25 × 38 × 34 voxels) and
  tracked (21 microbubble tracks). Spark timings for one acquisition (240 frames, 4 transmits,
  `acq0_only.h5`): coarse grid 38 × 25 × 34 in 7 s (RSS 4 GB); default grid 154 × 25 × 275 in
  188 s without spatial TGC (RSS 27 GB, 2.06 GB output) and 10 min 50 s with `--spatial-tgc`
  (the global-TGC SVD is CPU-bound; host peak 87 GB of the shared 120 GB). Cloud comparison
  (`nvus-fus/beam/modal_aleph_bench.py`, 2026-10-26, x86_64 PyPI `mach-beamform` wheel, 8 chunks,
  8 vCPU; the Beam variant is blocked on workspace credits): Modal A10G, coarse 16 s, default
  20 s without TGC and 123 s with TGC; Modal H100, coarse 12 s, default 37 s without TGC and
  117 s with TGC (RSS 28–31 GB in all cases). The H100 is no faster than the A10G, so on x86 the
  default grid is bound by host work, not the kernel. The Spark was initially ~10× slower on the
  same run (188–218 s, independent of T9 vs NVMe). Profiling (nsys + cProfile + a kernel-only
  microbenchmark, `nvus-fus/beam/mach_kernel_bench.py`) showed the mach kernel takes the same
  29 ms per 30-frame chunk on the GB10 as on the A10G (13 ms on H100), ~3 s per acquisition; the
  other ~110 s were NumPy `stack`/`astype`/`sum` over 4 × 2 GB on the host, which on the Spark
  run 10–15× slower than expected because NumPy's default hugepage `madvise` (`NUMPY_MADVISE_HUGEPAGE=1`)
  makes multi-GB copies crawl on this kernel (2 GB copy: 3.8 s vs 0.4 s with it off). Fix on the
  `gpu-compound` branch of the `m9h/microbubbles` fork (PR to alephneuro pending): compound over
  angles on the GPU and copy once. Spark default grid now **9.3 s** (RSS 6.5 GB, output bit-identical), i.e. 2× faster
  than the Modal A10G run of upstream code; with `--spatial-tgc` and `NUMPY_MADVISE_HUGEPAGE=0`
  **56 s** (was 10 min 50 s; host peak 21 GB instead of 87 GB), so the full 216-acquisition
  sample is ~35 min without TGC or ~3.4 h with it on the Spark. The ~12–16 s coarse floor on Modal is container start
  plus the 0.45 GB read.*
- Write the safety supervisor (MI_tc, temperature, TI exposure-time table, CEM43 against
  ITRUSST 2025 levels; FDA intensities for reference) as pure software with tests; advisor review.
  *v0 written with 21 tests (`nvus_fus/safety.py`); advisor review pending.*

Deliverable: reproduced-simulation report; safety supervisor v0.

## Phase 1 — imaging bench (weeks 6–16)

- Buy BOM part A. Bring up HSB 100G + AFE5932EVM + probe on the Spark.
- B-mode → 32-angle plane-wave compounding → Doppler on a flow phantom.
- Implement SVD clutter filter and power Doppler as Holoscan operators (`nvus-fus`).

Deliverable: live power Doppler of a flow phantom at ≥ 10 fps, with a validation note.

## Phase 2 — stimulation bench (weeks 12–24)

- Buy BOM part C (hydrophone, tank, positioner, degasser) before part B.
- Fabricate or buy the Open-LIFU array and console; hydrophone-map it; fit calibration.
- Planning chain to a target in a skull phantom; verify focus position and pressure.
- Safety supervisor gates HV enable; every sonication logged.

Deliverable: calibrated, navigated, phantom-verified LIFU focus with logged exposure.

## Phase 3 — acoustoelectric (weeks 16–30)

- Saline tank with electrodes and an injected current; heterodyne receiver at the ultrasound
  frequency; compare measured modulation with `ae-sim` predictions.
- Move to a conductive head phantom; electrode-layout design from `ae-sim`.

Deliverable: reproduction of focused acoustoelectric decoding (Rintoul 2023) on open hardware.

## Phase 4 — integration (weeks 24–36)

- Interleaved stimulate-then-image on a phantom with a measurable response.
- One coordinate frame for probe, array, phantom and plan.
- Closed-loop demo: a stimulation parameter updated from the imaging readout.

Deliverable: the bench; a methods paper; reproducible containers and calibration files.

## Governance

- Phantom and bench only. Any human or animal step needs institutional review, ITRUSST
  limits as a floor, and the advisor's sign-off on the safety supervisor.
- Hardware derived from Open-LIFU stays CERN-OHL-S; our software Apache-2.0.
- Altera FPGA IP stays in NVIDIA's repository (its licence is Altera-products-only).
- No MimickNet weights (CC BY-NC) in anything shared; IDC data filtered to CC BY.
- Memory caps on every CUDA process on the shared DGX Spark.
