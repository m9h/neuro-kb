---
type: concept
title: "Compute: NSG, NeuroFedora, NeuroDebian and Containers"
related: [jax-ecosystem.md, method-eeg-foundation-models.md, eeg.md, meg.md, data-formats.md, structural-mri.md, diffusion-mri.md, coordinate-systems.md, method-sbi.md]
description: "This page covers the shared compute and packaging layer under the neuroimaging project family."
timestamp: 2026-11-05T15:59:53-08:00
tags: [concept, compute, hpc, packaging]
---

# Compute: NSG, NeuroFedora, NeuroDebian and Containers

This page covers the shared compute and packaging layer under the neuroimaging project family. It answers three questions: where GPU jobs run for free, how the native toolchain (ITK, ANTs, LSL) is packaged, and which containers and conventions keep environments reproducible. It has four parts:

1. **NSG**, the Neuroscience Gateway on SDSC Expanse, for free V100 batch compute.
2. **NeuroDebian / NeuroFedora**, OS-level packages for tools missing from the distributions.
3. **Neurocontainers / Neurodesk**, containerized neuroimaging applications.
4. **research-infra**, cross-repo conventions and scheduling for agents.

There are no bibliography keys for this material. Every fact is cited to its project file (`project:file`). Facts are tagged by how they were established:

- **[measured]**: observed on real jobs.
- **[doc]**: from official NSG/SDSC docs.

---

## 1. Neuroscience Gateway (NSG → SDSC Expanse)

NSG is a science gateway that submits jobs to NSF HPC, currently **SDSC Expanse**. It is free to academic users because NSG holds a pooled ACCESS allocation. You need no allocation of your own and no SSH. NSG is built on the CIPRES Workbench, which is why the REST path is `cipresrest` and the auth header is `cipres-appkey` (nsg-agent-kit:docs/FINDINGS.md §1).

### Job model

- **Input.** Upload **one `.zip` whose top level is exactly one directory**.
- **Tool.** Choose a *tool*, which is a fixed Singularity environment.
- **Parameters.** Set `filename` (entry script, default `input.py`), `subdirname`, `runtime`, `number_gpus`, and the other tool parameters.
- **Execution.** NSG runs the job under Slurm and returns the **entire working directory** as `output.tar.gz`.
- **Front doors.**
  - Web portal: `https://nsgprod.sdsc.edu:8443/portal2/`
  - NSG-R REST: `https://nsgr.sdsc.edu:8443/cipresrest/v1`
  - Wrapper: nsg-agent-kit ships `nsgr/nsgr.sh` with `submit`, `list` and `fetch` commands.
- **Data policy.** Human-subject data must be de-identified.

### Platform parameters

| Parameter | Value | Status / source |
|---|---|---|
| GPU | `Tesla V100-SXM2-32GB`, compute capability 7.0 (sm_70, Volta) | [measured] FINDINGS §3b, §7 |
| GPUs per node (`number_gpus`) | default 1, max 4 | [doc] REST param spec |
| Nodes (`number_nodes`) | max 72 | [doc] |
| Memory per node (`number_gbmemorypernode`) | max 243 GB | [doc] |
| Wallclock (`runtime`) | default 0.5 h, max 48 h | [doc]/[measured] |
| CPU cores | docs: 64-core AMD EPYC; **GPU nodes measured at 40 cores** (2026-09-22) | discrepancy, see FINDINGS header |
| Bundled Python | 3.11.4 | [measured] |
| Image torch (`PYTORCH_PY_EXPANSE`) | 2.0.1+cu117 (unchanged through 2026-10) | [measured] FINDINGS §6a |
| NVIDIA driver | 580.82.07 (CUDA-12 capable) | [measured] |
| Network egress from compute node | **true** (pypi.org:443, 8.8.8.8:53 reachable) | [measured] probe on `exp-2-58` |
| NEMAR datasets on disk | 552 OpenNeuro EEG/MEG/iEEG `ds######` (was 547 in 2026-07) | [measured] |
| `$NEMARPATH` | `/expanse/projects/nemar/openneuro/` (read-only, synced daily via DataLad) | [measured]/[doc] |
| Queue latency | 7 min → 1 h 52 min observed; README estimate ~8–10 min | [measured] FINDINGS §8 |
| Per-user concurrency | ~3 jobs passed to Slurm at once; others held in `SUBMITTING` for 2–20 h | [measured] FINDINGS §9 |
| Quota/usage API | none; `/usage`, `/quota`, `/allocation` return 404; only signal is `canSubmit` on `/user/<you>` | [measured] LIMITATIONS §1 |

### Tool catalog (selected, 2026-07-21)

| Tool id / portal name | Version | Kind | Relevance |
|---|---|---|---|
| `PYTORCH_PY_EXPANSE` | torch 2.0.1+cu117 | GPU | EEG foundation-model sweeps ([method-eeg-foundation-models.md](method-eeg-foundation-models.md)) |
| `PY_EXPANSE` (Python on Expanse) | 3.11.4 | CPU | BMTK, Brian2, HNN-Core, NetPyNE 1.0.4.2, PyNN, NEST ([neural-mass-models.md](neural-mass-models.md)) |
| Python on Expanse GPU | 3.8.10 | GPU | TensorFlow-oriented |
| EEGLAB on Expanse | latest | — | Default NEMAR path ([eeg.md](eeg.md)) |
| FREESURFER on Expanse | 7.1.1 | — | Surface reconstruction ([structural-mri.md](structural-mri.md)) |
| MRTrix (NeuroDesk) | 3.0.3 | — | Tractography ([diffusion-mri.md](diffusion-mri.md)); shows NSG runs Neurodesk containers |
| SpikeInterface | 0.101.0 | — | Spike sorting |
| AMICA | 17 | — | ICA for EEG |

The GPU tool lacks `nrnivmodl`, so NEURON mechanisms must compile on the CPU tool (FINDINGS header).

### Dependency rules (measured failure modes)

1. **The venv is read-only.** Installing to the default location fails with `OSError [Errno 30] Read-only file system`.
   - Use `pip install --target "$TMPDIR/libs" --ignore-installed <pkgs>` and add the directory to `PYTHONPATH`.
   - `--user` is not an option because the environment is a venv.
2. **Install to `$TMPDIR`, not `./`.** NSG returns the whole working directory, so a `jax[cuda12]` install in `./` produces a 3.3 GB download.
3. **Shadowing torch kills the V100.**
   - `peft` hard-requires torch. Unpinned, pip pulls the latest **torch 2.13.0+cu130** into `--target`.
   - That wheel dropped sm_70, so the first forward pass fails with `no kernel image is available for execution on the device`.
   - **Fix:** pin `torch==2.4.1` (PyPI default build is cu121) *first* in the same install call. This build:
     - still ships Volta kernels;
     - provides `torch.nn.attention`, which REVE needs (torch ≥2.1);
     - matches the 580.x driver.
   - Do not use cu130 or nightly torch on the V100 (FINDINGS §6a, §7).
4. **Other required pins and shims:**
   - `numpy<2`.
   - `pandas` listed alongside numpy in the same call, to avoid a C-extension ABI mismatch.
   - `transformers==4.46.3` (avoid 5.x).
   - `peft==0.21.2`. The earlier pin `peft==0.14.0` failed 33/59 adapter cells because it lacks `target_parameters`, `all-linear` support for IA3, and the newer OFT semantics.
   - The image's Python lacks `_sqlite3`; shim it with `pysqlite3-binary` aliased as `sqlite3` (FINDINGS §8).
5. **Apptainer is the escape hatch.** It is only needed when a dependency needs a newer CUDA than the driver supports, or a torch build without Volta kernels.
6. **Hugging Face downloads.**
   - Anonymous HF downloads get rate-limited on the node's shared IP (198.202.103.196).
   - Upstream re-runs `snapshot_download()` per cell, so even a token can hit HTTP 429.
   - **Fix:**
     - Ship a token with the job.
     - Call `prefetch()` once per shard.
     - Set `HF_HUB_OFFLINE=1` for the cells.
   - Measured: arithmetic prefetch takes 61 s for 510 files, then the offline load takes 4 s (FINDINGS §9a).
7. **The NSG workspace disk fills up periodically** (`No space left on device`). Affected jobs report `COMPLETED` but return no tarball; this has recurred, so plan resubmission (`submit --only SHARD --resubmit`).

### Measured throughput (OpenEEGBench on NSG)

| Run | GPU work | Queue | Outcome |
|---|---|---|---|
| Smoke shard (BIOT frozen, arithmetic) | pip 149 s + shard 53 s | 7 min | bal. acc **0.576389**, bit-identical to a local RTX 2080 |
| bench1 (7 backbones × 9 strategies, arithmetic) | 59 cells in 1,544 s | 1 h 52 min | 24/59 completed with peft 0.14; per-cell cost 1–245 s |
| bench1c (EEGPT, batch 16) | 2 cells in 1,692 s | 55 min | Peak 8.8–10.1 GiB vs >30 GiB at batch 64 (protocol deviation) |
| public1 (26 shards, seed 0) | 20 shards fetched over a 120 h deadline loop | — | 427 rows, 359 completed |

Practical consequences:

- **Queue time, not compute, sets wall-clock** at small-dataset scale.
- **The 48 h limit bites only for large sets** such as ISRUC (13 GB, about 10× the windows).
- **EEGPT gradient strategies at protocol batch size exceed 32 GB.** They are marked `KNOWN_V100_OOM_CELLS`.
- **Two failure classes are upstream bugs, not our pins.** LaBraM × LoRA-family and CBraMod/REVE × OFT are recorded as `KNOWN_UPSTREAM_BROKEN` (FINDINGS §9b).

Model-level results are covered in [method-eeg-foundation-models.md](method-eeg-foundation-models.md).

---

## 2. OS packaging: NeuroDebian and NeuroFedora

**neurodebian** holds Debian source packaging (`packages/<name>/debian/`) for tools missing from NeuroDebian, Debian or Ubuntu. They are ported from the Fedora COPR roster `mhough/neurofedora` (neurodebian:README.md).

- **Targets:** Ubuntu 24.04 (noble), amd64 on the laptop and arm64 on the DGX Spark.
- **Architecture:** the `debian/` tree is shared and architecture-neutral.

| Package | Version | Role |
|---|---|---|
| `liblsl` | 1.17.7-1 | Lab Streaming Layer core (liblsl2 + -dev), for real-time EEG streaming ([eeg.md](eeg.md)) |
| `pylsl` | 1.18.2-1 | Python LSL binding against system liblsl2 |
| `pyxdf` | 1.17.5-1 | XDF reader ([data-formats.md](data-formats.md)) |
| `insighttoolkit5` | 5.4.6-1~noble2 | ITK 5.4.6 with 8 extra remote modules (see below) |
| `ants` | 2.6.5-1~noble1 | ANTs built against the system ITK above |

ITK details:

- **"Batteries-included" build.** It adds 8 remote modules: GenericLabelInterpolator, AdaptiveDenoising, MorphologicalContourInterpolation, GrowCut, SimpleITKFilters, LabelErodeDilate, MGHIO and IOScanco.
- **Foundational package.** ANTs, ITK-SNAP, SimpleITK, MITK and 3D Slicer all depend on those modules.
- **ANTs downstream.** ANTs registration underpins template normalization ([coordinate-systems.md](coordinate-systems.md), [structural-mri.md](structural-mri.md)).

Build notes:

- **Build command:** `dpkg-buildpackage -us -uc -b` against the upstream `orig` tarball.
- **Memory:** ITK template translation units peak at about **3 GB each**, so bound parallelism with `DEB_BUILD_OPTIONS="parallel=N"`.
- **Version tracking:** `versions.tsv` tracks upstream, COPR, Debian and local versions; regenerate it with `refresh_versions.py --net`.
- **Roadmap:** `neurodebian_validated_roadmap.md` records the EXISTS/MISSING/OUTDATED status of the COPR roster.

**neurofedora** is the upstream Fedora COPR roster (`gitlab.com/morgan.hough/neurofedora`). Its README is currently an unedited GitLab template, so roster contents are known only indirectly through neurodebian's tracker. Treat neurodebian's `versions.tsv` as the authoritative cross-channel view.

---

## 3. Containers: Neurocontainers / Neurodesk

**neurocontainers** is the Neurodesk repository for automatically building and testing the application containers that ship in Neurodesktop (neurocontainers:README.md). Adding a tool follows the Neurodesk "new tools" tutorial at neurodesk.org.

Connection to NSG:

- **NSG already runs Neurodesk containers.** The catalog's `MRTrix (NeuroDesk) on Expanse` is the evidence, and all NSG tools execute in Singularity.
- **Apptainer is first-class** for escaping the stock environment.
- **Still unverified:** whether NSG accepts an *arbitrary user* Apptainer image. Open question for nsghelp@sdsc.edu (FINDINGS §5).

Recommended layering:

| Need | Layer | Example |
|---|---|---|
| Native C/C++ libraries on workstations or DGX | `.deb` / COPR | ITK 5.4.6, ANTs 2.6.5, liblsl |
| Pinned Python ML stack on a free GPU | NSG + `pip --target $TMPDIR` | torch 2.4.1+cu121, peft 0.21.2 |
| Full GUI or CLI neuroimaging applications, reproducible | Neurodesk container | MRtrix 3.0.3, FreeSurfer |
| Dependencies beyond driver/Volta limits on NSG | Custom Apptainer image | CUDA-13-only wheels |

---

## 4. research-infra: conventions and scheduling

research-infra (`rinf` CLI) is the cross-repo backbone for agentic workflows (research-infra:README.md).

- **Shared Python stack:**
  - `uv` for packaging;
  - `hatchling` builds;
  - `ruff` for linting;
  - `pytest` + `beartype` for testing;
  - the Kidger stack: Equinox, Diffrax, Optax, Lineax ([jax-ecosystem.md](jax-ecosystem.md)).
- **No-mock policy.** `rinf validate --check no-mock` and a pre-commit hook reject `unittest.mock`, `MagicMock`, `@patch` and `mocker.`. This matches nsg-agent-kit's live-probe testing style: contract tests are RED until real job results arrive, then GREEN.
- **Autoresearch scheduler.**
  - `rinf autoresearch next` round-robins across projects that contain `autoresearch/program.md`.
  - It scans `~/dev/` and `~/Workspace/` for those projects.
  - Built-in templates exist for alf, hgx, jaxctrl, neurojax, setae, qcccm and organoid-hgx-benchmark, among others.
- **Manuscript and slides.** `rinf build manuscript` (pandoc + XeLaTeX) and `rinf build slides` (Beamer) assemble output from `.md` frontmatter. The frontmatter fields are `category`, `section`, `weight` (0–999), `status` and `slide_summary`.

---

## Relevant Projects

- **nsg-agent-kit:** NSG-R wrapper (`nsgr/nsgr.sh`), job templates, probes with TDD contracts, the OpenEEGBench sweep driver (`sweep/oeb_plan.py`, `sweep/oeb_shard.py`), and agent skills (`nemar-search`, `nemar-inspect`, `nemar-job`).
- **neurodebian:** `.deb` packaging for ITK, ANTs, LSL and XDF on noble (amd64 and arm64).
- **neurofedora:** Fedora COPR roster (`mhough/neurofedora`), the source of the Debian ports.
- **neurocontainers:** Neurodesk container build and test pipeline.
- **research-infra:** conventions, validation, manuscript builds and autoresearch scheduling.
- **Consumers:**
  - `emeg-fm`, `nanoeeg` and `zuna-eeg-pipeline` run foundation-model sweeps on NSG.
  - `hnn-jax` uses HNN-Core on the CPU tool.
  - `libspm` and `LAYNII` use native registration and segmentation toolchains.
  - `neurojax` and other JAX projects use the JAX CUDA-12 wheels on the V100 via `--target`.

## See Also

- [jax-ecosystem.md](jax-ecosystem.md): JAX/Equinox stack shared across projects.
- [method-eeg-foundation-models.md](method-eeg-foundation-models.md): REVE, BIOT, CBraMod and other models benchmarked on NSG.
- [eeg.md](eeg.md) and [meg.md](meg.md): modalities behind the NEMAR/OpenNeuro datasets.
- [data-formats.md](data-formats.md): BIDS, XDF and the file formats moved through these pipelines.
- [structural-mri.md](structural-mri.md) and [diffusion-mri.md](diffusion-mri.md): FreeSurfer, ANTs and MRtrix workflows.
- [coordinate-systems.md](coordinate-systems.md): template spaces reached via ANTs registration.
- [method-sbi.md](method-sbi.md): simulation-heavy inference workloads that suit batch GPU compute.
