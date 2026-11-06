# Software to build on

Checked 2026-10-25 via the GitHub API. Stars are indicative. Forks are under github.com/m9h.

| Project | Origin | Licence | Last push | Role | Fork |
|---|---|---|---|---|---|
| holoscan-ultrasound-imaging | NVIDIA | Apache-2.0; FPGA IP dir under Altera licence | 2026-10-22 | Raw-channel imaging, GPU beamforming, shot control; fUS engine (v0.1) | m9h/holoscan-ultrasound-imaging |
| openlifu-python | Openwater | Apache-2.0 | 2026-10-19 | Planning, simulation, transducer control | m9h/openlifu-python |
| openlifu-sdk | Openwater | Apache-2.0 | 2026-10-23 | Standalone hardware I/O (TX module, HV controller); the integration point for Holoscan shot control | m9h/openlifu-sdk |
| openlifu-desktop-application | Openwater | Apache-2.0 | 2026-10-12 | Planning/control GUI | m9h/openlifu-desktop-application |
| SlicerOpenLIFU | Openwater | Apache-2.0 (GitHub licence API; draft 0.1 wrongly said AGPL) | 2026-10-09 | 3D Slicer neuronavigation extension (we ran it on aarch64) | m9h/SlicerOpenLIFU |
| openlifu-transmitter-fw, openlifu-console-fw, *-bl | Openwater | Apache-2.0 | 2026-10-23 | STM32L443 TX firmware, console power firmware, secure-boot loaders | m9h/openlifu-transmitter-fw, m9h/openlifu-console-fw |
| openlifu-verification-tank | Openwater | Apache-2.0 | 2026-10-19 | Tank control software for hydrophone characterisation | m9h/openlifu-verification-tank |
| OpenLIFU-3DScanner / OpenLIFU-3DRegistration | Openwater | Apache-2.0 / AGPL-3.0 | 2026-10-09 | Mesh capture and registration for transducer localisation | m9h/OpenLIFU-3DScanner |
| openlifu-closed-loop-eeg | Openwater | Apache-2.0 | 2026-10-09 | EEG-triggered sonication example | m9h/openlifu-closed-loop-eeg |
| openlifu-sample-database | Openwater | CC-BY-4.0 | 2026-10-22 | Transducer configs, protocols, sample data | — |
| openlifu-electrical | Openwater | CERN-OHL-S-2.0 | 2026-10-09 | 64-element array flex, HV transmit module Rev 0.4.1, console PCB; Gerbers + BOMs | m9h/openlifu-electrical |
| openlifu-mechanical | Openwater | CERN-OHL-S-2.0 | 2026-10-09 | Transducer housings, console, verification-tank fixtures (STEP) | m9h/openlifu-mechanical |
| openwater-neuromod-hardware / -software | Openwater | AGPL-3.0 | 2026-10-09 | Previous-gen wearable steerable LIFU headset + acoustic validation (Bawiec 2025, JUM) | m9h/openwater-neuromod-hardware, m9h/openwater-neuromod-software |
| opw_ustx, st-ustx* | Openwater | AGPL-3.0 | 2026-10-09 | Ultrasound transmit module lineage | m9h/opw_ustx |
| openmotion-sdk, openmotion-bloodflow-app, openmotion-sensor-fw, openmotion-console-fw | Openwater | Apache-2.0 | 2026-10-23 | Near-infrared optical blood-flow / micro-motion readout | m9h/openmotion-sdk, m9h/openmotion-bloodflow-app |
| openmotion-*-fpga (safety, seed, TA, camera) | Openwater | AGPL-3.0 | 2026-10-09 | Console and camera FPGA projects | — |
| openmotion-electrical / -mechanical | Openwater | CERN-OHL-S-2.0 | 2026-10-09 | Open-Motion PCBs and enclosures | m9h/openmotion-electrical |
| openwater-phantoms | Openwater | AGPL-3.0 | 2026-10-09 | `ultrasound_and_AO/`: gelatin/agarose phantom instructions, Gen2 phantom construction notes (67 MB PDF), rat-body mould + vein/artery target moulds (STL/STEP/3dm), Souza 2016 reference; `optical/`: Gen2 optical phantom recipe, headset/flow phantom CAD, static-phantom tester. Instructions on the Openwater wiki | m9h/openwater-phantoms |
| openwater-reference-meshes | Openwater | CC-BY-4.0 (meshes), Apache-2.0 (code) | 2026-10-09 | **Mostly scaffolding as of 2026-10-25**: README/changelog promise skull, head, brain STL from the April 2026 Neuro Hackathon with provenance YAMLs, but the repo holds one 25 MB `meshes/head/texturedMesh.obj`, empty `skull/` and `brain/` folders, a metadata template and two loader scripts. Not a usable skull source yet | m9h/openwater-reference-meshes |
| openwater-regulatory, openwater-patents, openwater-commons, openwater-docs | Openwater | AGPL-3.0 / CC-BY-4.0 | 2026-10 | FDA correspondence; 68 filings + Patent Pledge; governance; docs | m9h/openwater-patents |
| openwater-oncolysis-data/-hardware/-software, openwater-laser | Openwater | AGPL-3.0 | 2026-10-09 | Preclinical therapeutic-ultrasound tissue-effect data; not for this bench | — |
| alephneuro/microbubbles (Ultratrace ULM) | Aleph Neuro | MIT | 2026-07-31 | GPU MACH beamforming (CUDA 12/13) from demodulated IQ, temporal SVD clutter filter, 3D bubble localisation, Kalman–Hungarian tracking, viewers; 96 GB public transcranial dataset (contrast) | m9h/microbubbles |
| neurotechdevkit (NDK) | AE Studio (agencyenterprise) | Apache-2.0 | 2026-04-30 | TUS simulation + visualisation for teaching | m9h/neurotechdevkit |
| BabelBrain | Pichardo lab (ProteusMRIgHIFU) | BSD-3-Clause | 2026-10-24 | CT/MR → transcranial plan, GPU FDTD | m9h/BabelBrain |
| BabelViscoFDTD | Pichardo lab | NOASSERTION on GitHub — check | 2026-10-11 | Viscoelastic FDTD (CUDA/Metal/OpenCL); used by `ae-sim`; we package it for Fedora | m9h/BabelViscoFDTD |
| k-Wave | UCL (Treeby, Cox) | LGPL-3.0 | 2026-05-05 | Reference MATLAB acoustic solver | m9h/k-wave |
| j-Wave | UCL (Stanziola) | LGPL-3.0 | 2026-03-22 | Differentiable JAX acoustics; runs on the Spark GPU | m9h/jwave |
| Stride | Imperial | AGPL-3.0 | 2026-10-09 | Full-waveform inversion / modelling | — |
| PRESTUS | Donders | GPL-3.0 | 2026-06-18 | End-to-end TUS protocol pipeline | m9h/PRESTUS |
| SimNIBS | DTU / DRCMR | GPL-3.0 | 2026-03-03 | Head models (charm), lead fields; our aarch64 packages | (packaged in neurofedora/neurodebian) |
| mSOUND | Jing lab | GPL-3.0 | 2025-06-05 | Frequency-domain focus estimates | — |
| ae-sim | ours (private, in progress) | TBD | — | Acoustoelectric forward model: BabelViscoFDTD × SimNIBS lead fields, scikit-fem, cupy | — |
| Acoustoelectric (GitHub org) | likely Rintoul — check | check | — | k-space Poisson, FEM, neural-recording analysis | — |
| mach (`mach-beamform`) | Forest Neurotech (Guan, Rockhill, Sode, Pinton; J Med Imaging 13(6):062203, 2026) | BSD-3-Clause-Clear | 2026-10-23 | CUDA delay-and-sum, ~10x prior SOTA, built for fUSI; what Aleph's pipeline calls. x86_64 wheels only: on the Spark build from source with arch 121 (`~/dev/mach`) | — (clone) |
| zea | TU/e (tue-bmd) | Apache-2.0 | 2026-10 | Keras-3 ultrasound pipeline (demodulate, beamform, …) and the Open-H-RF file format/loader; CPU and GPU. `init_device()` fails on GB10 (nvidia-smi memory `[N/A]`) — skip it | — |
| Open-H-RF v1.0 (`nvidia/Open-H-RF` on Hugging Face) | Stanford / TU/e / NVIDIA + 38 institutions (arXiv:2610.13616, 2026-10-09) | CC BY 4.0 | 2026-10-04 | 39 TB, 19,471 zea HDF5 files, 33 subsets. fUS-relevant: `ulmshare` (99 transcranial mouse-brain ULM acquisitions, Verasonics 256, ~180 GB each — stream, do not download), `twente-vortexflow` (flow phantom with optical ground truth, 6 × 1.5 GB, curved array 3.4 MHz, 750 frames at 500 Hz; our first replay target), `resolvestroke` (transcranial flow), `stanford-murine`, `waterloo-carotid`/`-femoralvein` (vector Doppler), `twente-microbubblesim`. Per-subset `reconstruct.py` in github.com/open-h/Open-H-RF | — (clone) |
| nvus-fus | ours (`~/dev/nvus-fus`, local git) | Apache-2.0 | 2026-10-25 | Safety supervisor v0 (ITRUSST), SVD/power-Doppler reference, zea replayer, vortexflow script; 33 tests | — |
| 3D Slicer, idc-index | Kitware / NCI | BSD / data CC BY | — | Visualisation, registration, CC-BY head CT/MR for skull models | — |

## Gaps (nothing open found)

- GPU fUS pipeline (compounding, SVD clutter filter, power Doppler) as streaming Holoscan operators.
  Aleph's pipeline has GPU beamforming (via Forest's `mach`) and SVD offline from IQ files; zea has
  GPU beamforming for Open-H-RF files; `nvus-fus` has the SVD/PD reference. The Holoscan wrapper is
  still missing.
- Driver for Open-LIFU under Holoscan shot-level control (start from openlifu-sdk).

## Closed (for reference, not usable)

- Butterfly Ultrasound-on-Chip / Poseidon: Embedded licence only; Merge Labs has BCI R&D exclusivity.
- Nudge Zero, Iconeus One, Insightec Exablate, Attune ATTN201, BrainSonix Pulsar: closed.
- Safety supervisor computing derated ISPPA/ISPTA/MI/TI from live drive settings and gating HV.
- Driver putting Open-LIFU's transmit module under Holoscan shot-level control.
- Acoustoelectric decoding beyond research scripts.
