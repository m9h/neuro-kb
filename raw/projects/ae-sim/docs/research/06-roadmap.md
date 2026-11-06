# Roadmap

Ordered by what unblocks what. Each item names the research note that motivates it.

## Code changes the survey asks for

- [x] **K as a parameter**: `ae_sim.coupling` (table of literature values, `AE_SIM_K`), every
      example records its K and the injected-current examples report µV / (mA·MPa).
      ([04](04-coupling-constant.md))
- [x] **Cylindrical contact source** in the scikit-fem lead field, for the Medtronic 3389
      geometry: `ae_sim.dbs`, `cylinder_density`, graded tensor meshes, variable conductivity,
      interior Dirichlet sinks, `TensorPoissonField`. ([01](01-replication-targets.md) target 1)
- [ ] **Focused rectangular array source** in `ae_sim.acoustics` (Babel accepts arbitrary source
      maps): 18 × 7 at 0.6 MHz, and later the 126-element array. ([01](01-replication-targets.md)
      targets 2 and 3)
- [ ] **Bowl transducer source**: 60 mm, ROC 63 mm, 500 kHz, CW and pulsed.
      ([02](02-rintoul-heterodyning-and-in-vivo.md))
- [x] **Poisson solve with a volume source term** −∇·(σ K P E₀) for the heterodyning
      (neuromodulation-mode) problem, alongside the reciprocity-based imaging-mode operator:
      `ae_sim.forward` (mesh-agnostic, operator set up once, pyamg, `P1Locator`,
      `grid_pressure`), `examples/ae_field_map.py`, stage 37. ([05](05-prototype-findings.md))
      FDTD snapshots feed it through `acoustics.snapshot_grid` + `grid_pressure` (φ₁ tracks
      the pulse, tested). Digimouse loads as `MeshTet` with per-organ σ (`ae_sim.digimouse`,
      2026-10-23; `solve_dirichlet` drives electrode pairs). Follow-ups: SimNIBS meshes, tensor
      conductivity, a GPU backend behind `_solve`.

## Replications

1. [x] Preston 2018: 0.52 µV / (mA·MPa) in saline → **0.47** with the idealised beam
       (2026-10-23, stage 36) and **0.83** with the FDTD run of the A392S transducer
       (`AE_SIM_BEAM=fdtd`, stage 38, ring-delay focusing in `ae_sim.acoustics`).
       Follow-ups: the three dipole configurations (3-2, 3-1, 3-0), the skullcap pressure
       drop, λ/8 on a GPU host to check grid dispersion.
2. [x] Rintoul 2023 tank: cross-check against the published `.npz` (2026-10-24,
       `examples/rintoul_tank.py`, stage 26, `scripts/fetch-rintoul-2023.sh`). Their
       `pressure0.5MHz.npz` phasor (1 MPa at z = 60 mm, lateral FWHM 4.2 mm) through our FEM
       path with real pins (0.25 mm, 1.4 mm tips, 3 mm apart) at 250 V/m: |E₁| at the
       midpoint **0.23 V/m** parallel (paper ~1 V/m) and 0.065 V/m perpendicular, ratio
       **3.54** (paper 4.12), at K = 1e-9. The pins alone close the gap to ×4.3 (K ≈ 4.3e-9
       would match), not the ×20 their solver needed; the recording electrode is a symmetry
       zero exactly between pins across the beam, so the scanned maximum is what to compare.
       Done 2026-10-24/25: their `ae_data_kspace_*` outputs sit a constant factor 35 below the
       equation they state (see 02); the PVDF film / oil side (saline cut at an insulating
       plane 5, 10 or 20 mm before the focus) changes |E₁| by < 0.1 % and the recording by
       1–2 %, so neither closes the gap.
3. [ ] Kang 2023 cube with the 3-axis leads; acoustics cross-checked against Cheong 2026.
4. [ ] Allard 2025 / Barragan 2020 skull + gel: 1.5 µV / mA at 0.7 MPa.
5. [ ] Rintoul 2026 in vivo mouse on Digimouse: standing waves and the 9 mV onset potential.
       Started: `examples/digimouse_head.py` (stage 39, 2026-10-23) has the mouth and head
       loops at 20 V and a 1 MPa Gaussian focus over motor cortex on the whole-body mesh:
       16 mA (1.2 kΩ), 313 V/m in brain, φ₁ peak 299 µV, motor − visual electrodes +98 µV.
       Gel and loop geometry done (2026-10-24, `ae_sim.mousehead`, stages 24 and 41): the
       head cut from the 0.1 mm volume with a gel blanket, the 2 cm wire loop in it and an
       oral cavity with the mouth wire, meshed by SimNIBS (130 k nodes, 732 k tets, 4 min,
       cached): 42 mA (478 Ω), 386 V/m median in brain (p90 1.6 kV/m next to the mouth wire
       under the olfactory bulbs), φ₁ peak 510 µV, motor − visual +54 µV, E₁ = 0.22 K P E₀.
       Done since (2026-10-24): the neural readout (`ae_sim.activation`: |E₁| 0.12 V/m →
       23 µV polarisation, 1/50 of the paper's 6 V/m), the cone artefact (stage 43: 0.77 mV
       per transducer volt on the pair through the 127 µm film) and the skull in the acoustic
       path (stage 42, `examples/digimouse_skull.py`: transmission 0.78, standing waves
       1.7, and the real beam column gives |E₁| 0.38 V/m, 3.4× the Gaussian spot, so the
       field is 1/16 of the paper's 6 V/m, not 1/50). 2026-10-25: stage 46 thins the atlas
       bone to 0.3 mm (`thin_skull`); air around the head is blocked by FDTD stability (see 05).
       The onset response: stage 48 (`ae_sim.neuron`, an HH point neuron: a spike needs
       9.6 V/m, 25× the field here).
6. [x] Tianjin rat DBS mapping on SIGMA v2 (2026-10-25, `examples/sigma_rat_dbs.py`, stage 45;
       order of magnitude, the methods being paywalled): a 0.72 mm lead with a 1 mm contact in
       the right STN against the neck, scalp patches 3 mm lateral / 8 mm posterior, 1 MHz
       Gaussian spot (FWHM 1.5 mm) scanned by the reciprocal operator
       (`AEForwardSolver.lead_field` + `reciprocal_weights`: 959 foci in 48 s, equal to the
       direct solve to 1e-4): 1.08 kΩ, peak **31 µV / (mA·MPa) = 29 µV / (V·MPa)** against
       the abstracts' 167.74 µV V⁻¹ MPa⁻¹, image FWHM 1.5 × 1.5 × 9.3 mm (the beam), peak on
       the contact. The CT skull in the acoustic path at 1 MHz is stage 47
       (`examples/sigma_rat_skull.py`: transmission 0.61, CW standing waves 3.5, coherent
       readout ×0.09 of the spot by travelling-wave cancellation, envelope ×0.65, so 20 µV /
       (mA·MPa) short-pulse). Later the same day: the lead as CAD (`ae_sim.leadmesh`, gmsh,
       stages 49–50): same electrode under two meshers gives the same per-volt sensitivity
       (28.7 vs 28.8), and a 0.25–1 mm lead sweep stays within 19–30 µV / (V·MPa), so the
       electrode geometry is excluded as the factor ~6. Open: the real Tianjin transducer /
       pulse-length / recording-montage parameters. Pulsed FDTD done (stage 52, `examples/sigma_rat_pulse.py`): a 2-cycle burst
       gives 3.8 µV through the skull and a 1-cycle burst 4.1, 0.19 of the CW-envelope proxy:
       a zero-mean packet reads the kernel band-passed at the wavelength; the A-line localises
       the contact to 0.8 mm. At 0.5 MHz (stage 53) the burst reads 0.40 of the spot and the
       skull transmits 0.77: 4.8× the signal for twice the beam width.
       Bipolar lead done (stage 51): the return through a second ring makes the
       scalp pair 25× less sensitive (0.6 µV / (V·MPa)), so the return path, not the
       contact, is the first-order electrode parameter.

7. [x] Independent FEM cross-check and a direct-solver backend (2026-10-25, with the
       neurofedora-t2 session): SimNIBS's TMSFEM reproduces the AE forward solve to 1.5e-3 on
       ernie (`tests/test_simnibs_crosscheck.py`); `AEForwardSolver(method="mumps")`
       factorises once (12.6 s at 140 k nodes) and solves in 0.10 s. Open: DOLFINx for a
       Helmholtz-with-PML phasor solve with air (framework's `python3-dolfinx-openmpi`
       0.11.0-5 is in the COPR; Fedora's PETSc is real-only) and for EMI cell models.
       neurofedora-t2's trial (2026-10-25, fedora:44 container): the complex assembly is
       right (film layer vs the series circuit to 1.8e-13, ω → 0 vs the real solve 1.3e-10,
       p = 0 plane exact, 0.6 mm bone slab vs the 1-D transfer matrix ×1.10 at 10 points per
       wavelength), but Fedora's superlu_dist lacks the METIS ordering (`get_perm_c.c:545`
       abort; MMD/COLAMD only), so 133 k complex unknowns blow 10 GB, while python-mumps with
       METIS does 275 k in 32 s / 7 GB and 1e6 would be ~40 GB (the Spark). Plan: assemble in
       dolfinx (`la.MatrixCSR.to_scipy()`), solve with python-mumps (zmumps), P2 or 20+ points
       per wavelength for percent accuracy. The cone artefact already runs on
       `ComplexForwardSolver(method="mumps")`.

## Model building

- [x] SIGMA v2 downloaded (2026-10-24, `data/sigma-v2`, 1.5 GB: in vivo and ex vivo
      templates, anatomical and functional atlases, brain meshes, CT and FDG templates); the
      rat case written 2026-10-25 (`ae_sim.sigma_rat`: ex vivo space, CT bone > 8 or skull map
      > 0.5, GM/WM/CSF by largest probability, air by template threshold + fill; block-majority
      to 0.18 mm; lead painted; `mesh_rat_volume` via the mouse cache; `tests/test_sigma_rat.py`,
      stage 44). The CT skull over the cortex is 0.72–0.90 mm 2–3 mm off the midline. Digimouse done
      (`scripts/fetch-digimouse.sh`; labels verified in `tests/test_digimouse.py`: 17 organ
      labels on the mesh, 22 on the volume, brain centroids within 2.5 mm).
- [x] `create_mesh` on a rodent label volume: `ae_sim.mousehead.mesh_head_volume` (SimNIBS,
      0.2–0.5 mm in skull and wires, 0.4–1.2 mm elsewhere, cached), with the fixtures
      painted into the volume first (`build_head_volume`: gel, loop, oral cavity, wire).
      Rat case on SIGMA done 2026-10-25 (`ae_sim.sigma_rat.mesh_rat_volume`; a 2 × 2 voxel
      0.36 mm lead did not survive create_mesh's spike removal, 4 × 4 = 0.72 mm does; the
      1 mm contact label is trimmed too, so its nodes are taken geometrically). Superseded
      for the electrode by `ae_sim.leadmesh` (2026-10-25): the lead is CAD cut out of the box
      by gmsh, the tissue tagged from the voxels, sizes graded from the contact (stages 49–50).
- [x] Skull in the acoustic path (2026-10-24, `examples/digimouse_skull.py`, stage 42): the
      bowl propagated analytically (Rayleigh, `acoustics.bowl_plane_phasor`, O'Neil-exact at
      the focus) to a plane at the top of the gel, a CW BabelViscoFDTD run from that plane on
      the head box (`simulate_cw_plane_source`, quantised phasor source, demodulated to the
      complex phasor), water / no-skull / skull runs, then `grid_pressure` into the forward
      solver. Open: mouse-specific skull values (Estrada 2016) instead of generic bone and a
      skull thinned to the measured 0.2–0.5 mm (the atlas gives 1.6 mm along the oblique beam
      axis at 0.2 mm, so the skull effect is currently an upper bound), shear
      in the skull, air instead of water around the head, the 0.1 mm run on the Spark
      (`AE_SIM_H=0.1e-3`), and the human case via BabelBrain headless.
- [x] Activating-function readout (2026-10-24, `ae_sim.activation`: ∂²φ₁/∂s² along the
      cortical normal from the brain mask, somatic polarisation, thresholds from Reato 2010 /
      Radman 2009 / the paper's 6 V/m, pressure and voltage needed to reach each). Done
      2026-10-25: `ae_sim.neuron` + `examples/onset_response.py` (stage 48), an HH point
      neuron under the field: no spike at 0.38 V/m (76 µV), a spike needs 9.6 V/m = 25× the
      field (508 V on the loops or 25 MPa), which is where the paper's 6 V/m sits. Next: a
      cortical cell model (LFPykit / NEURON) instead of the squid set.
- [x] Cone artefact (2026-10-24, `examples/cone_artefact.py`, stage 43): σ + iωε at 500 kHz
      (`ComplexForwardSolver`), film / no film / thick film. Open: the actual transducer
      voltage and grounding scheme (ask the authors). The oil/PVDF side of the 2023 tank is
      done (stage 26, `AE_SIM_FILM_MM`): no effect beyond 2 %.
- [ ] Dipole return current on the head mesh (~15 % effect in the bath).
- [ ] Matched-filter receive-bandwidth noise model.
- [ ] Larger synchronous sources via LFPykit.
- [ ] Fetch scripts, not data packages (decided 2026-10-23, [07](07-data-and-tutorials.md)):
      `scripts/fetch-digimouse.sh`, `fetch-rintoul-2023.sh`, `fetch-sigma.sh` (all three
      done 2026-10-24), each checksummed with licence and citation beside it; only Digimouse (5 MB) and the small
      Rintoul files are candidates for an actual package; `ae-sim` itself packaged first.

## Platform

- [ ] First GPU run of the pipeline on the DGX Spark (`scripts/container-run.sh smoke` answers
      the three unknowns in minutes).
