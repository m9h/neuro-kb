# Running ae-sim (legion, DGX Spark container, pipeline)

Prototype for Morgan Hough's acoustoelectric (AE) imaging feasibility study. A focused
ultrasound pulse modulates tissue conductivity (d(rho)/rho0 = -K dP, K ~ 1e-9/Pa in saline);
a recording electrode pair sees, by reciprocity,

    V_AE(t) = -K * integral( dP(r, t) * rho0 * J_L(r) . J_I(r) dV )

where J_L is the lead field of the pair (current density per ampere injected through it) and
J_I the neural or injected current being imaged. The seven building blocks and where they stand:

| block | what | implementation |
|---|---|---|
| 1 head model | SimNIBS charm `m2m_<subj>`; Digimouse whole-body mouse mesh + 0.1 mm labels | `scripts/fetch-ernie.sh` (example head); `scripts/fetch-digimouse.sh` + `ae_sim.digimouse` (per-organ σ, electrode patches, loads straight into `ae_sim.forward`) |
| 2 acoustics | time-resolved pressure field | `ae_sim.acoustics` wraps BabelViscoFDTD (CPU / CUDA / OpenCL); skull not yet in the path (BabelBrain next) |
| 3 lead fields | J_L of the recording pair | `ae_sim.leadfield` (scikit-fem box phantom), `ae_sim.head` (SimNIBS on a real head) |
| 4 sources | the imaged current | `ae_sim.fields` (point pairs), `ae_sim.sources` (FEM dipoles) |
| 5 mixing | the operator above | `ae_sim.mixing` |
| 6 acquisition | Johnson + amplifier noise, averaging | `ae_sim.noise` |
| 7 validation | tests + machine-checked examples | `tests/`, `tests/pipeline.bats` |

## Running

The project is a `uv` project whose venv is created with `--system-site-packages` so it sees
the system solvers (python3-babelviscofdtd, python3-simnibs from the mhough/neurofedora COPR).
Everything else (numpy < 2.5 for the system numba, scipy, scikit-fem, matplotlib) comes from uv.

```sh
uv venv --system-site-packages --python /usr/bin/python3 && uv sync      # once
uv run pytest -q                                                          # ~70 s, head test skipped
AE_SIM_M2M=/path/to/m2m_ernie uv run pytest -q                           # + SimNIBS head test
scripts/run-pipeline.sh                                                   # the whole pipeline (BATS)
```

`scripts/run-pipeline.sh` runs `tests/pipeline.bats` in a clean environment (no conda, no stray
`PYTHON*`): toolchain, `uv sync`, solver imports, backend check, unit tests, FDTD on the selected
backend, example head and Digimouse (fetch with `AE_SIM_FETCH=1`), SimNIBS lead field, the
Digimouse tests, the examples with their numbers asserted against the recorded CPU baselines,
GPU-vs-CPU at production size, and a
`results/summary.json`. Example figures and JSON land in `results/`.

Cache keys include the head path and injected current for SimNIBS lead fields, and the
effective element sizes and facet distances for head meshes. Older cache names are left
in place but are not reused. The corrected gmsh scalp size grid and zero-mean Hann pulse
also use new cache versions. Their first run rebuilds the affected mesh or pulse record;
the historical one-cycle pulse numbers need a new FDTD run before comparison. The gmsh
baselines also need revalidation after rebuilding with the corrected scalp refinement.

Environment variables (all optional): `AE_SIM_K` (coupling constant: `song` (default, 1e-9 /Pa),
`witte`, `wang-saline`, `wang-brain`, `water`, `rintoul-fit`, a value in 1/Pa, or `0.034%/MPa`;
every example records the K it used and, where a current is injected, the sensitivity in
µV/(mA·MPa)), `AE_SIM_BACKEND` (cpu | cuda | opencl), `AE_SIM_GPU`
(device-name substring), `AE_SIM_M2M`, `AE_SIM_DATA`, `AE_SIM_FETCH`, `AE_SIM_WORK` (SimNIBS cache),
`AE_SIM_RESULTS`, `AE_SIM_OUT` (JSON path for a single example), `AE_SIM_F0` (ultrasound frequency
for `examples/head_feasibility.py`), `OMP_NUM_THREADS`. `examples/dbs_lead_saline.py` (Preston
2018, stage 36) has its own knobs: `AE_SIM_BEAM_FWHM`, `AE_SIM_SIGMA_T`, `AE_SIM_H_FINE`,
`AE_SIM_STIM`/`AE_SIM_REC`, `AE_SIM_FLOATING`, `AE_SIM_STIM_SINK`, `AE_SIM_GROUND`, `AE_SIM_TIP_X`. `AE_SIM_BEAM=fdtd` swaps the Gaussian beam for a BabelViscoFDTD run of the paper's
transducer (stage 38, opt-in with `AE_SIM_HEAVY=1` or a GPU backend; `AE_SIM_CPW` cells per
wavelength, default 6; the record is cached under `AE_SIM_WORK`). `examples/ae_field_map.py` (FEM forward path, stage 37): `AE_SIM_E0` (V/m),
`AE_SIM_P0` (Pa), `AE_SIM_H_FINE`. `examples/digimouse_head.py` (tutorial 1, stages 39 and 41):
`AE_SIM_DIGIMOUSE` (default `data/digimouse`, from `scripts/fetch-digimouse.sh`), `AE_SIM_MODEL`
(`head`: the painted label volume meshed by SimNIBS, default when simnibs imports, ~4 min once
and cached under `AE_SIM_WORK`; `body`: the whole-body mesh with skin-patch electrodes),
`AE_SIM_SIGMA_GEL` (S/m, default 1.0), `AE_SIM_V0` (volts between the mouth and head loops,
default 20), `AE_SIM_P0`. `examples/digimouse_skull.py` (tutorial 1b, stage 42): `AE_SIM_H`
(acoustic grid, default 0.2e-3 m; 0.1e-3 for the atlas resolution, ~10× the cells),
`AE_SIM_PERIODS` (CW cycles, default 30), `AE_SIM_BACKEND`. `examples/cone_artefact.py`
(tutorial 1c, stage 43, needs the SimNIBS cone mesh, ~5 min once): `AE_SIM_SIGMA_WATER` (cone
water, default 5e-3 S/m), `AE_SIM_FILM_UM` (default 127). Stage 46 reruns `digimouse_skull.py`
with `AE_SIM_SKULL_MM=0.3` (the atlas bone thinned from the outside to 0.3 mm,
`ae_sim.mousehead.thin_skull`), `AE_SIM_RUNS=water,skull` (skip the no-skull run) and
`AE_SIM_TAG=_thin` (figure suffix); `AE_SIM_AIR=1` puts air around the head (currently unstable
in Babel, see 05). `examples/sigma_rat_dbs.py` (tutorial 2, stage 45, needs the SimNIBS rat mesh,
~3 min once): `AE_SIM_SIGMA` (default `data/sigma-v2`), `AE_SIM_TARGET` (atlas structure, default
"Subthalamic nucleus"), `AE_SIM_SIDE`, `AE_SIM_I0` (A, default 1e-3), `AE_SIM_BEAM_FWHM` (m, default
1.5e-3), `AE_SIM_BEAM_SIGMA_AX` (m, default 4e-3), `AE_SIM_SCAN_MM` (default 3), `AE_SIM_SCAN_STEP_MM`
(default 0.5; the FWHM is quantised by it), `AE_SIM_P0`, `AE_SIM_K`, `AE_SIM_TAG` (figure suffix), and the
lead: `AE_SIM_LEAD=voxel` (default: the painted 0.72 mm column meshed by SimNIBS) or `gmsh` (the CAD lead
of `ae_sim.leadmesh`, needs `python3-gmsh`; the mesher runs in a child interpreter and is cached in
`AE_SIM_WORK` as `leadmesh_<hash>.npz`) with `AE_SIM_LEAD_MM` (diameter, default 0.25), `AE_SIM_CONTACT_MM`
(tip contact length, default 0.5), `AE_SIM_LEAD_SECTION` (round | square), `AE_SIM_LEAD_BIPOLAR=1` with
`AE_SIM_CONTACT_GAP_MM` (default 0.5: a second ring of the same length that gap above the contact is the return
instead of the neck; stage 51), `AE_SIM_LEAD_PAINTED=1` (the
painted column's own geometry, for the mesher comparison of stage 49) and the mesh sizes in mm
`AE_SIM_H_CONTACT` / `_LEAD` / `_SKULL` / `_FAR` / `_SCALP` (0.06 / 0.15 / 0.25 / 0.5 / 0.3). Stages 49, 50 and 51
run it with the gmsh lead (painted geometry, the round 0.25 mm lead, the bipolar lead). `examples/sigma_rat_skull.py` (tutorial 2b, stage 47; 3 CW runs of 2–4 min, phasors cached in
`AE_SIM_WORK` by their parameters): `AE_SIM_H` (default 0.15e-3), `AE_SIM_PERIODS`, `AE_SIM_RUNS`
(default water,skull; the stage adds no_skull), `AE_SIM_F0` (1e6), `AE_SIM_BOWL_RADIUS` (10e-3),
`AE_SIM_ROC` (25e-3). `examples/sigma_rat_pulse.py` (tutorial 2c, stage 52; water + skull pulse runs of ~2 min
each, traces cached in `AE_SIM_WORK` as `ratpulse_<hash>.npz`): the same knobs plus `AE_SIM_CYCLES` (Hann burst
length in periods, default 2), `AE_SIM_SENSOR_HALF_MM` (recorded box around the contact, default 4),
`AE_SIM_TIME_STRIDE` (4), `AE_SIM_LEAD` (voxel | gmsh), `AE_SIM_AMP_NV_RTHZ` (amplifier noise density for the
one-pulse SNR, default 5), `AE_SIM_PRF` (1000), `AE_SIM_TAG`; stage 53 (opt-in, `AE_SIM_HEAVY=1` or a GPU)
reruns it at `AE_SIM_F0=0.5e6` with `AE_SIM_BEAM_FWHM=3e-3`. `examples/onset_response.py` (tutorial 1d, stage 48, seconds): `AE_SIM_E1`
(V/m at the focus, default 0.38), `AE_SIM_V0`, `AE_SIM_P0`, `AE_SIM_LAMBDA_P` (0.2e-3 m), `AE_SIM_TAU_M`
(20e-3 s), `AE_SIM_DF` (modulation frequencies, default 10,100,1000). `examples/rintoul_tank.py` (stage 26): `AE_SIM_RINTOUL` (default
`data/rintoul-2023`, from `scripts/fetch-rintoul-2023.sh`), `AE_SIM_E0` (V/m between the pins,
default 250), `AE_SIM_H_FINE`. Data fetchers: `scripts/fetch-ernie.sh`, `fetch-digimouse.sh`,
`fetch-rintoul-2023.sh [DATA] [sets]`, `fetch-sigma.sh`, all checksum-pinned and idempotent.

## GPU (legion)

BabelViscoFDTD's CUDA backend needs `cupy`, which compiles the FDTD kernel with NVRTC at run time.
The `gpu` extra installs `cupy-cuda13x` plus the NVRTC/cudart wheels (legion: RTX 2080, driver
615.x, CUDA 13.4, Fedora 44, Python 3.14; a CUDA 12 driver would need the `-cu12` names instead).

```sh
uv sync --extra gpu
AE_SIM_BACKEND=cuda scripts/run-pipeline.sh          # add AE_SIM_GPU=RTX to pin the device
AE_SIM_BACKEND=cuda uv run pytest -q tests/test_acoustics.py   # just the wrapper + GPU-vs-CPU
```

On the AMD NUC the GPU exposes no OpenCL device, so the CPU backend (OpenMP) is the baseline there.
Legion runs one heavy job at a time: keep `OMP_NUM_THREADS` small (the CPU half of stage 40 and
SimNIBS use it) and coordinate before starting.

### Moving the project to legion

```sh
# on the NUC: scripts/bundle-for-legion.sh  -> ae-sim-<sha>.bundle (git history, no data)
# on legion:
git clone ae-sim-<sha>.bundle ae-sim && cd ae-sim
sudo dnf install python3-babelviscofdtd python3-simnibs bats     # COPR mhough/neurofedora
uv venv --system-site-packages --python /usr/bin/python3 && uv sync --extra gpu
AE_SIM_FETCH=1 AE_SIM_BACKEND=cuda OMP_NUM_THREADS=2 scripts/run-pipeline.sh
```

The example head (413 MB) is downloaded once into `data/` and checksum-verified. The SimNIBS
runs are cached under `data/ae-sim-work/` by montage.

## Container on the DGX Spark (alternative to legion)

The whole pipeline, SimNIBS included, runs in one container. The Spark (gx10-dgx-spark) is
aarch64 (GB10, sm_121), Ubuntu 24.04, driver 580.x / CUDA 13.0, Docker 29 with the NVIDIA toolkit
(`--gpus all` proven). The image is `fedora:44` (already pulled there) plus the mhough/neurofedora
COPR, which has fedora-44-aarch64 builds of python3-simnibs 4.6.0-21 and python3-babelviscofdtd
1.2.7-5: the NUC's stack exactly, no source builds. A dry-run resolution of those two packages
against Fedora 44 aarch64 from the NUC succeeds (557 packages, no conflicts). cupy and the
NVRTC/cudart wheels come from the `gpu` extra; libcuda is injected by the container runtime.

The box is shared and its 120 GB is one CPU+GPU pool, so cupy's pool is capped at 16 GiB by
default (`CUPY_GPU_MEMORY_LIMIT`) and the thread count stays small; the container runs as the
invoking user with `--rm`. Keep inputs, outputs and images on /mnt/t9, not on `/` (22 GB free).
Tell the Spark's operator (the neurospark session) before a large run.

```sh
mkdir -p ~/dev && cd ~/dev && git clone /mnt/t9/ae-sim/inputs/ae-sim-<sha>.bundle ae-sim && cd ae-sim
export AE_SIM_HOST_DIR=/mnt/t9/ae-sim      # inputs/ -> /data, outputs/ -> /results
scripts/container-run.sh build             # fedora:44 + COPR rpms + uv venv with the gpu extra
scripts/container-run.sh smoke             # nvidia-smi, cupy device count, simnibs import, FDTD tests on CUDA
scripts/container-run.sh run               # the whole pipeline; the ernie head is fetched into inputs/ once
scripts/container-run.sh head              # just examples/head_feasibility.py (AE_SIM_F0, AE_SIM_CPW, AE_SIM_LATERAL)
```

Validated on the NUC (x86_64, rootless podman, `AE_SIM_NO_GPU=1`, image `ae-sim:cpu`): the same
Containerfile builds (fedora:44 + COPR rpms, ~10 GB, ~10 min) and the whole pipeline passes inside
the container as a non-root user with the mounted inputs/outputs layout, SimNIBS runs and all,
GPU stage skipped. Remaining unknowns for the Spark: the aarch64 rpms installing (resolved from
the NUC, not yet executed), cupy-cuda13x on sm_121 (Babel's kernel is compiled by NVRTC at run
time, so Blackwell support rests on the NVRTC wheel), and numba/SimNIBS JIT on aarch64 (the COPR
changelog says SimNIBS was measured on aarch64 by neurospark). `smoke` answers these in minutes.
Rootless podman notes: the runner uses `--userns=keep-id` and `:Z` mounts, and treats the
podman-docker shim as podman.

**Portable lead fields** (optional). `examples/head_feasibility.py` can also export its two
SimNIBS lead fields as regular grids (`AE_SIM_EXPORT_LEADFIELDS=<dir>`, 60 mm cube around the
target, `AE_SIM_EXPORT_H` spacing, ~1 MB each at 1 mm, `ae_sim.gridfield`) and load them with
`AE_SIM_LEADFIELDS=<dir>` on a host without SimNIBS. Caveat: the mesh lookup is element-wise
constant and the export is trilinear, and a short pulse is sensitive to that sub-mm difference
(pipeline stage 35 records the ratio; see results below).

## Results so far (CPU baselines the pipeline asserts against)

- Saline bath, 1 MPa, 1 MHz, 10 mA injected: 0.18 nV at the centre for a 1-cycle pulse; the
  image shows electrode lobes up to 12 nV. A radiated pulse has no DC, so smooth current fields
  show only through their axial gradients; wavelength-scale structure shows directly.
- FEM + FDTD phantom: 14.0 nV in the finite insulated bath vs 2.69 nV with infinite-medium fields
  (the bath walls raise the signal ~5x); the ideal non-diffracting beam gives 0.70 nV (~4x low).
- Bath dipole 20 nA m at 0.73 MPa: ~1 pV total (band-pass cancellation over the 3 mm blob);
  ~1e13 averages against 2.4 uV noise over 2 MHz.
- Real head (ernie), 20 nA m dipole 15 mm under C3, 0.3 MPa at the target, pair C3-C4, 3.5 uV
  noise: 1 MHz 32 pV (1e11 averages), 0.5 MHz 146 pV, 0.25 MHz 268 pV (1.6e9 averages). Lower
  ultrasound frequency (pulse length ~ source size) is the strongest lever; a nearer scalp pair
  does not help because the skull sets the cortical lead field. Point/DC upper bound 1.8 nV.

- Lead-field representation (stage 35): with the SimNIBS lead field resampled trilinearly on a
  1 mm grid instead of looked up element-wise on the mesh, the far-pair peak drops from 31.5 to
  11.0 pV at 1 MHz (23.4 pV with a 0.5 mm grid), 146 to 63 pV at 0.5 MHz and 268 to 168 pV at
  0.25 MHz. A short pulse band-passes the sub-mm jumps of piecewise-constant element data, so the
  real-head numbers carry a 1.5-3x uncertainty from mesh granularity, worst at high frequency; a
  finer head mesh or a smoothness study is needed before quoting them to better than a factor 2.

Not yet modelled: skull in the acoustic path (BabelBrain headless), the dipole's return current on
the head mesh (~15% in the bath), matched-filter receive bandwidth, larger synchronous sources.
