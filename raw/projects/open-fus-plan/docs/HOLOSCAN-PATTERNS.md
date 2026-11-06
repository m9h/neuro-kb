# Lessons from the Kernel Flow fNIRS app on Holoscan

Source: HoloHub `applications/bci_visualization` ("Kernel Flow BCI Real-Time
Visualization", NVIDIA + Kernel, Apache-2.0, merged 2026-01-12, Kernel SDK live path
2026-01-16, now under `examples/legacy` on the `archive-4x` branch), the NeurIPS 2025
demo abstract, and the Hugging Face dataset `KernelCo/holohub_bci_visualization`
(74 GB). Read 2026-10-25.

## What they built

- Python Holoscan app (SDK >= 3.7, tested 3.7/3.9), x86_64 and aarch64, developed on a
  Jetson Thor dev kit with a Kernel Flow 2 headset.
- Six reconstruction operators (stream, build RHS, normalise, regularised solver, convert
  to voxels, voxel to volume) feeding a ClaraViz volume renderer and Holoviz on a second
  thread under the event-based scheduler.
- Physics frozen offline. A per-device "mega Jacobian" (channels x 3 moments x 2
  wavelengths x voxels x {mua, musp}) is precomputed on an average head and shipped as
  data: 13.3 GB for the Flow 2.3 headset, 1.4 GB for the dev kit. The runtime never runs a
  photon simulation.
- Inverse is small and cached: dual Tikhonov system (J J^T + lambda sqrt(||J J^T||) I),
  lambda = 0.1, formed once per wavelength; each frame is one `cupy.linalg.solve` plus a
  J^T back-substitution at ~9,000 unknowns. Input is frame-to-frame difference of log
  moment 0 plus the raw first and second moments; normalisers are hard-coded.
- Thin hardware contract. The headset (120 sources, 240 detectors, 3,000+ channels,
  100 ps pulses at 20 MHz, ~1 GB/min compressed) computes TOF-histogram moments on-device;
  the host receives only moments at 4.75 Hz over the Kernel SDK socket or from a SNIRF file.
- Replay first: SNIRF replayer and a 472 MB recording are the default run mode; hardware
  is a flag.

## What we take for the fUS bench

1. **Split device from reconstruction at a published data boundary.** Ours is beamformed
   IQ or compounded frames out of the acquisition chain. Aleph's pipeline and Forest's
   `mach` already take demodulated IQ, so the layers line up.
2. **Precompute the expensive operator and ship it as data.** Skull-corrected delay tables,
   element phases, SVD bases and `ae-sim` lead fields are our Jacobians. Version them with
   the head model and device serial; keep the runtime to matrix products. Budget storage:
   their per-device assets are 1-13 GB.
3. **Cache the system matrix, solve the small dual.** Fits acoustoelectric decoding: few
   electrodes, many voxels, so solve in channel space and back-project.
4. **Recorded-data replayer before hardware.** Open container (HDF5; zea format for
   OpenH-RF compatibility) for our own frames; develop every operator against replay.
5. **Separate threads for reconstruction and rendering, drop-latest queue policy,
   `ConditionType.NONE` on the passthrough output** so the GUI never stalls acquisition.
6. **Host expectations are modest once the device compresses.** Thor or any Holoscan
   platform, Python + cupy, no NIC/FPGA. Raw-channel ultrasound is the opposite case: the
   acquisition node needs the Spark's ConnectX-7 and the sensor bridge, but reconstruction
   and decoding could run on any aarch64 GPU box.
7. **Open what they did not.** Headset, SDK, Jacobian generator and head model are closed;
   the Jacobian is Apache-2.0 data but not reproducible. Our forward-model generation
   (SimNIBS, BabelViscoFDTD, k-Wave) must stay in the repo and rerunnable.
8. **Rough edges to avoid.** Hard-coded normalisers, affine from four random voxels, a
   global Hessian cache keyed only by wavelength, no latency numbers, no exposure logging.
   Our safety supervisor has no analogue in their stack.
