# Getting the papers' files, and turning them into packaged tutorial material

*Survey date: 2026-10-23. Question from Morgan: how do we get files from these papers so we
can add more of them to the packages as tutorial material?*

Short answer: of the nine papers in [references.md](references.md), only the three Rintoul
papers deposited anything beyond the PDF, and their data are CC BY 4.0 on Figshare. The Witte /
O'Donnell papers (Preston, Barragan, Allard, Kang, Cheong) have no data deposits at all, only
supplementary movies under the journal's copyright; what we can package from them is our own
replication of their published numbers, which is what `examples/` already is. The two rodent
atlases are open and small enough to package outright. Everything below was checked against
the live repositories (Figshare API, Zenodo API, the USC download pages, the local
`fedora-license-data`).

## What exists, paper by paper

| source | what is deposited | licence | size | packageable in the free COPR? |
|---|---|---|---|---|
| Rintoul 2023, Commun Phys ([c.6365098](https://doi.org/10.6084/m9.figshare.c.6365098)) | k-space simulation outputs `ae_data_kspace_a0degrees_f0.5MHz.npz`, `..._a90degrees_...npz`, `pressure0.5MHz.npz`; tank streams `df10hz_stream.npy`, `df492khz_stream.npy`, `12.0_demod_8_stream.npy`; four field GIFs | **CC BY 4.0** on every Figshare item | 653 + 653 + 229 MB; 480 + 480 + 32 MB; 11–19 MB each | yes (allowed-content) |
| Rintoul 2023 code (`Acoustoelectric-1.0.zip`, Figshare 21804039) | `kspace_solver.py`, `focused.m`, `pressure_3Diser.py`, demodulation and plotting scripts | Figshare says CC BY 4.0, but `LICENSE.txt` and the README inside say **CC BY-NC-SA 4.0** | 7.4 MB | **no**: the inner licence governs; NC is not allowed in Fedora. Re-derive, never port |
| Rintoul 2023 Fig 4 2-D maps (Figshare 21802183) | a Dropbox link | — | 0 MB placeholder, dead | ask Jean |
| Rintoul 2026, Nat Commun ([c.7909283](https://doi.org/10.6084/m9.figshare.c.7909283)) | figure data zips F2–F8 and Supp 9; analysis code | **CC BY 4.0** data; code **Apache-2.0** | 49 MB to 1.05 GB per figure, ~4.2 GB total; code 0.7 MB | yes, but only selected figures (size) |
| Rintoul 2026, Commun Eng ([c.7754537](https://doi.org/10.6084/m9.figshare.c.7754537)) | `ae.npy`, `ac_iso.npy` single-shot recording example | **CC BY 4.0** | 480 + 480 MB | yes |
| Preston 2018 UMB | one supplementary movie | journal (Elsevier) | 1 file | no; parameters only |
| Barragan 2020 JNE | none found | — | — | no; parameters only |
| Allard 2025 TBME (PMC12153511) | three MP4 movies, no data or code statement | IEEE / author manuscript | 1–2 MB each | no; parameters only |
| Kang 2023 TMI | none found (Perkins 2022 MSc thesis at Arizona is the nearest open text) | — | — | no |
| Cheong 2026 Ultrasonics | unknown: ScienceDirect refuses automated fetches; check the Data availability section in a browser | — | — | tbd |
| Wang 2026 TBME (K in saline and brain) | none found | — | — | no; the K values are in `ae_sim.coupling` |
| Digimouse (USC) | `atlas.zip` (0.1 mm label volume, 380 × 992 × 208, ANALYZE, 22 labels incl. brain sub-regions and skeleton), `tesselated_atlas.zip` (**whole-body tetrahedral mesh**: 58,244 nodes, 306,773 tets, 17 organ labels, MATLAB .mat), `ct_data.zip` | **CC BY-SA 4.0** (`LICENSE.txt` in the archive) | 0.9 MB, 4.2 MB, 20 MB compressed | yes (allowed-content) |
| SIGMA v2 rat ([Zenodo 10635831](https://zenodo.org/records/10635831)) | templates and atlases, one zip | **CC BY 4.0** | 1.5 GB | yes, as a fetch or a trimmed subpackage |
| SimNIBS `ernie` | m2m folder | CC BY-NC 4.0 | 413 MB | no: `simnibs-example-data.spec` already targets the nonfree project |

Fedora's licence data (`fedora-license-data` 1.76): `CC-BY-4.0` and `CC-BY-SA-4.0` are
*allowed-content* (not for code); `CC-BY-NC-4.0` and `CC-BY-NC-SA-4.0` are *not-allowed*.
`ci/copr-submit.sh` in the neurofedora repo already refuses NC licences for the free project
(the COPR API returned no `mhough/neurofedora-nonfree` project today, so that side still has to
be created before `simnibs-example-data` can land anywhere).

Two things to ask the 2023 authors for directly: the Fig 4 2-D maps whose Dropbox link is
dead, and whether the code can be re-deposited under the CC BY 4.0 the Figshare record already
claims. A one-line relicensing note would let the k-space solver ship as a tutorial; without it
we keep re-deriving.

## Mechanics: how a file becomes a package

The pattern already in `neurofedora/`: `simnibs-example-data.spec`, `mitk-data.spec`,
`fsl-data-*.spec`. Noarch, `Source0:` a stable URL, `%prep` unpacks and `%install` copies
into `/usr/share/<name>/`, a `LICENSE` file as `%license`, `Requires:` the tool that reads it.
Points specific to these sources:

- **Figshare URLs.** `https://ndownloader.figshare.com/files/<id>` 302-redirects to a signed
  S3 URL with a 10 s lifetime and a `Content-Disposition` filename. `spectool -g` and COPR's
  SCM fetch follow redirects, but the spec must name the file:
  `Source0: https://ndownloader.figshare.com/files/38686022#/pressure0.5MHz.npz`. Checksums
  belong in the spec as comments (`sha256sum` of each download) because the files are not
  versioned; the Figshare *article* version is the version string (`Version: 1`, `%global
  figshare_article 21802165`).
- **USC Digimouse.** Plain HTTPS, no registration, stable paths under
  `https://neuroimage.usc.edu/mouse_atlas/`. Checksums measured today:
  `atlas.zip` 6f7cbd8a…fe11761, `tesselated_atlas.zip` 78c09b56…6d6415d.
- **Zenodo.** Versioned DOIs and a JSON API; 1.5 GB is large for an RPM. Either a
  `sigma-v2-data` package that ships only the template, the brain mask and the atlas labels
  at the resolution we use, or a `fetch-sigma.sh` next to `scripts/fetch-ernie.sh` with the
  checksum pinned.
- **Size policy.** Keep RPMs under ~100 MB installed where the tutorial allows it: Digimouse
  (whole), the Rintoul demodulation stream (32 MB), one figure zip, the GIFs. Put the 0.5–1 GB
  streams and simulation outputs behind a checksummed fetch script instead; a package can
  still carry the script, the checksums and the citation, which is the part people lose.
- **Attribution.** CC BY and CC BY-SA require the creator, the licence, a link and a note of
  modifications. Each data package carries the upstream `LICENSE` verbatim, a `CITATION.cff`
  with the paper DOI, and a `README` saying what was changed (format conversion, cropping).
  CC BY-SA on Digimouse means any *derived* mesh we ship (a brain-only tet mesh, a NIfTI
  conversion) is CC BY-SA too; that is fine for data, and the code that made it stays Apache-2.0.

## What becomes a tutorial

Each tutorial is one example script in `ae-sim`, one data package (or fetch script) it reads,
and a pipeline stage that asserts the numbers, which is the form `examples/dbs_lead_saline.py`
and stages 36–38 already have. Proposed set, in the order the data allows:

1. **Digimouse head in the forward solver** (data: `digimouse-data`, 5 MB). Load
   `Tesselated_Atlas.mat` as `MeshTet(r.T * 1e-3, (T - 1).T)` with per-element σ from the 17
   organ labels (brain = 2, skeleton = 13, muscle = 1/14/15), drop in the Rintoul 2026 mouth
   and head loop electrodes, and compute the AE field for a 500 kHz focus at motor cortex.
   This is the only tutorial that needs nothing from anyone: the mesh is 306 k tets, which
   `AEForwardSolver` handles in seconds. The 0.1 mm label volume feeds `create_mesh` for a
   finer head-only mesh later.
   **Done 2026-10-23**: `scripts/fetch-digimouse.sh`, `ae_sim.digimouse` (loader for both
   files, per-organ σ table, electrode patches), `tests/test_digimouse.py`,
   `examples/digimouse_head.py`, pipeline stages 22, 23 and 39. Two things the data taught
   us: the 2006 mesh has ~11 k sliver tets with micrometre edges (the loader collapses
   edges under 0.05 mm, half a voxel, keeping every organ volume; a few dozen inverted
   slivers remain and the AMG and LU solves still agree to 1e-9), and the mesh and the 2024
   label volume share a frame with brain centroids 2.5 mm apart (older segmentation), so
   atlas outlines can be drawn over mesh fields but the finer head mesh should come from
   the volume.
2. **Rintoul 2023 tank cross-check** (data: `pressure0.5MHz.npz` 229 MB + one
   `ae_data_kspace_*.npz` 653 MB, fetch script). Load their pressure field, run our forward
   solve with K = 2e-8 to match their solver, then at K = 1e-9, and show the parallel /
   perpendicular ratio against the measured 4.12. The 2-D field GIFs (11–19 MB, CC BY 4.0) can
   ship in the package as the reference figures.
3. **Rintoul demodulation** (data: `12.0_demod_8_stream.npy`, 32 MB). Their IQ demodulation of
   an 8 kHz field heterodyned onto 500 kHz, re-implemented from the paper's description (their
   script is NC-SA), run on their recorded stream. Small enough to ship whole, and it is the
   acquisition-side tutorial we do not have.
4. **Rintoul 2026 in vivo onset potential** (data: one figure zip, e.g. Fig 5 at 49 MB). The
   ~9 mV onset waveform next to the Digimouse prediction from tutorial 1.
5. **Preston, Barragan, Allard, Kang** stay as they are: parameter tables in
   [01](01-replication-targets.md) and our replications; the tutorial text cites the paper
   and shows our numbers beside theirs. Nothing of theirs is redistributed.
6. **SIGMA v2 rat**: fetched, loaded (`ae_sim.sigma_rat`) and used by tutorial 2
   (`examples/sigma_rat_dbs.py`, stage 45); the Tianjin methods, once in hand, set the
   electrode and beam parameters that are generic placeholders today.

## Package layout proposal

```
ae-sim                 the library, examples and docs (Apache-2.0)          ~1 MB
ae-sim-tutorials       the tutorial notebooks/scripts + pipeline stages     ~1 MB, Requires: ae-sim
digimouse-data         Digimouse label volume + tet mesh + LICENSE (CC BY-SA 4.0)   ~90 MB installed
ae-rintoul-2023-data   demod stream, GIFs, README, CITATION, fetch script for the 2.5 GB rest (CC BY 4.0)   ~90 MB
sigma-v2-data          trimmed rat template + atlas (CC BY 4.0), or fetch script only
```

`ae-sim` itself is not packaged yet (roadmap, Platform). It should be, before the data
packages: a data package without the tool that reads it is what the SimNIBS spec calls
"useless without the tools".

## Decision (Morgan, 2026-10-23)

Large data stays out of the COPR: only fetch scripts (checksummed, one per dataset, in
`scripts/`) are packaged, with the licence and citation beside them. The README now carries the
resource table. Small redistributable items (Digimouse, the GIFs, the 32 MB stream) may still
ship in packages.

Status 2026-10-24 (evening): tutorial 1 has three parts now: 1 (`digimouse_head.py`, stages
39/41, with the neural readout), 1b (`digimouse_skull.py`, stage 42, the bowl through the skull
by Rayleigh + CW FDTD) and 1c (`cone_artefact.py`, stage 43, the 500 kHz pick-up through the
cone). Tutorial 2 also reads the `kspace` set when present and reports the factor 35 between
the deposited field and the stated equation. SIGMA v2 (2026-10-25): `ae_sim.sigma_rat` loads the
ex vivo head (template, tissue maps, CT, atlas; the atlas header is 1 mm off the template in y and
the loader verifies the voxel grids coincide, 96 % of labelled voxels inside the brain mask),
classifies and downsamples it to 0.18 mm, paints a 0.72 mm DBS lead and meshes it (115 k nodes,
664 k tets, 160 s), or cuts a CAD lead out of the box with gmsh (`ae_sim.leadmesh`, 0.8 M tets in
10–160 s, stages 49–50); `examples/sigma_rat_dbs.py` is the rat DBS mapping case (stage 45) and
`examples/sigma_rat_skull.py` the CT skull in its acoustic path at 1 MHz (stage 47), `examples/sigma_rat_pulse.py`
the same with a short burst and the AE trace in time (stage 52). Tutorial 1d
(`examples/onset_response.py`, stage 48) is the HH point neuron under the mouse field.
Earlier the same day: all three scripts exist. `fetch-digimouse.sh` (sha256), `fetch-rintoul-2023.sh`
(md5 from the Figshare API, sets `pressure demod gifs kspace streams all`, default the first
three, 300 MB; the NC-SA code zip is deliberately not a set) and `fetch-sigma.sh` (md5 from the
Zenodo API, 1.5 GB). Tutorial 2 (the tank cross-check) runs on the fetched pressure field:
`examples/rintoul_tank.py`, stage 26. Tutorial 1 now has the gel, loop and mouth fixtures
painted into the label volume (`ae_sim.mousehead`, stages 24 and 41).

## Open questions for Morgan

- Whether to ask the 2023 authors about relicensing the code (cost: one email; gain: the
  k-space solver as a shipped tutorial instead of a re-derivation).
- Cheong 2026's data statement needs a browser.
