# open-fus-plan

A plan for a student-built, open focused- and functional-ultrasound bench for brain
research, of the kind Arbor Neuroscience (formerly Forest Neurotech) is building at
scale: ultrafast imaging plus steered low-intensity stimulation, simulation-based
planning, neuronavigation, calibrated safety limits, and an acoustoelectric readout path.

Prepared for neurotech@berkeley, with Jean Rintoul as advisor. Phantom and bench work
only; no human or animal use is proposed.

- `paper/main.tex` — the article (history, architecture, software, BOM, plan). Build with
  `tectonic main.tex` or `latexmk -pdf main.tex`. A built PDF is committed for convenience.
- `docs/PLAN.md` — phases and deliverables.
- `docs/SOFTWARE.md` — the open projects to build on, with licences and our forks.
- `docs/BOM.md` — the purchase list with estimate flags.
- `docs/IP.md` — the patent landscape (Forest/Merge, Stanford/Williams, Attune, Openwater pledge, Butterfly, Insightec) and the rules we follow.
- `docs/QUESTIONS-OPENWATER.md` — what to ask Openwater before replicating their meshes, phantoms and tank.

## Building blocks (all forked under github.com/m9h on 2026-10-25)

| Upstream | Role |
|---|---|
| nvidia-holoscan/holoscan-ultrasound-imaging | Raw-channel imaging engine (128 TX / 64 RX, RoCE into GPU, per-shot control); the fUS acquisition chain |
| OpenwaterHealth/openlifu-python, openlifu-sdk, openlifu-desktop-application, SlicerOpenLIFU, openlifu-transmitter-fw, openlifu-console-fw, openlifu-verification-tank, OpenLIFU-3DScanner, openlifu-closed-loop-eeg, openlifu-electrical, openlifu-mechanical | Open-LIFU stimulation stack: 64-element array, HV transmit module, console, firmware, planner, neuronavigation, tank |
| OpenwaterHealth/openwater-neuromod-hardware, -software, opw_ustx | Previous-generation wearable steerable LIFU headset with validation data |
| OpenwaterHealth/openmotion-sdk, openmotion-bloodflow-app, openmotion-electrical | Open-Motion near-infrared blood-flow readout |
| OpenwaterHealth/openwater-phantoms, openwater-reference-meshes, openwater-patents | Gelatin/agarose and acousto-optic phantom recipes with rat/vessel moulds; a reference-mesh repo that so far holds one textured head scan (skull/brain folders empty); 68-filing patent list + Patent Pledge |
| alephneuro/microbubbles | Aleph Neuro's MIT-licensed 3D transcranial ULM pipeline (GPU beamforming, SVD clutter filter, tracking) + 96 GB dataset |
| agencyenterprise/neurotechdevkit | AE Studio's Neurotech Development Kit: TUS simulation for teaching and prototyping |
| ProteusMRIgHIFU/BabelBrain, BabelViscoFDTD | CT/MR-based TUS planning, GPU FDTD; also the acoustics core of our `ae-sim` |
| ucl-bug/k-wave, jwave | Reference solvers from the UCL group behind the 2025 human helmet array |
| Donders-Institute/PRESTUS | End-to-end TUS protocol pipeline used in human studies |

Plus our own work: `ae-sim` (acoustoelectric forward model on BabelViscoFDTD + SimNIBS lead
fields), and the neurofedora / neurodebian aarch64 packaging of SimNIBS, FreeSurfer, 3D
Slicer and the Open-LIFU Slicer extension.

## Status

Draft 0.2 (adds Openwater's full portfolio, Butterfly Embedded partners Aleph Neuro / Merge Labs /
Bridge Neurotech / Midjourney Medical, Gestala, and an IP section). Facts were checked against the sources in the bibliography on 2026-10-25;
items marked `[check]` in the paper and `est` in the BOM are unverified or estimated.
Nothing has been purchased or built.
