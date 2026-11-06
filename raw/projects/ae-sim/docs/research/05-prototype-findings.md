# What the prototype has found so far

Numbers the pipeline (`tests/pipeline.bats`) asserts against, and the physics lessons behind
them. All at K = 1e-9 /Pa (see [04](04-coupling-constant.md) before quoting them).

## Physics lessons

1. **A radiated pulse has no DC component.** Even a stress source with net DC input leaves only
   a static pressure offset in the FDTD grid. The AE trace is therefore the coupling along the
   beam axis *band-pass filtered by the pulse spectrum*: smooth current fields show only
   through their axial gradients, and wavelength-scale (mm) structure shows directly. Long
   tone bursts or CW (the BabelBrain default) are narrow-band and image gradients only;
   ~1-cycle pulses keep the low frequencies. An earlier "A-line of the coupling" claim was
   wrong; it relied on a non-zero-mean pulse.
2. **Crossed electrode pairs null the centre line** (perpendicular fields). Use parallel pairs.
3. **Boundaries matter.** A finite insulated bath whose wall at z = 0 is the transducer face
   raises the signal ~5× over infinite-medium analytic fields (14.0 vs 2.69 nV). The idealised
   non-diffracting beam underestimates the FDTD piston near field ~4× (0.70 nV).
4. **Lead-field representation.** Resampling the SimNIBS lead field trilinearly on a regular
   grid instead of the element-wise mesh lookup drops the 1 MHz far-pair peak from 31.5 to
   11.0 pV (1 mm grid) or 23.4 pV (0.5 mm grid). A short pulse band-passes the sub-mm jumps of
   piecewise-constant element data, so real-head numbers carry a 1.5–3× uncertainty from mesh
   granularity, worst at high frequency. Pipeline stage 35 records the ratio.
5. **Lower ultrasound frequency is the strongest lever** on the real head (pulse length ~
   source size). A nearer scalp pair (C3-C1) does not help because the skull sets the cortical
   lead field.

## First replication: Preston 2018, DBS lead in saline (2026-10-23)

`examples/dbs_lead_saline.py`, scikit-fem on a graded tensor mesh (255 k nodes, two solves in
~40 s), Medtronic 3389 geometry, monopole configuration, idealised 1 MHz Gaussian beam
(FWHM 2.5 mm, 4.40 MPa peak-to-peak), 0.4 µs pulse, 0.33 mm scan along the lead.

| quantity | paper | Gaussian beam (K = 0.1 %/MPa) | FDTD beam (K = 0.1 %/MPa) |
|---|---|---|---|
| sensitivity, µV / (mA · MPa pk-pk) | 0.52 | **0.47** | **0.83** |
| peak position vs stimulating contact | within 0.41 mm | 0.43 mm | 0.10 mm |
| monopole extent, lateral × axial FWHM | 3.67 × 3.11 mm | 2.64 × 3.09 mm | 2.97 × 2.20 mm |

The FDTD column (`AE_SIM_BEAM=fdtd`, stage 38, 2026-10-23) is a BabelViscoFDTD run of the
paper's transducer (Olympus A392S, 38 mm aperture focused at 63 mm, ring time delays, 12 M
cells at λ/6, 326 s on 4 CPU threads, cached) scaled to 4.40 MPa peak-to-peak on the axis at
the lead depth. Its beam at the lead is 3.25 mm wide (FWHM) rather than the 2.5 mm the
Gaussian assumed, because the on-axis pressure maximum of the pulsed, F# 1.65 source sits at
56 mm, before the geometric focus, so the contacts at 63 mm see a beam past its waist. The
wider beam integrates more of the current around the contact: the sensitivity rises 1.8× and
the lateral extent moves towards the paper's. The axial extent shortens because the radiated
pulse is shorter than the source burst (no DC, see lesson 1).

Two lessons, both about the circuit rather than the physics:

- **The two solves see two circuits.** With the stimulating contact modelled as a floating
  conductor in the lead-field solve, the lead field enters it on one face and leaves on the
  other, the coupling changes sign across the contact and the AE signal at the stimulating
  contact cancels to ~0.3 µV per mA while a 17 µV per mA artefact appears two contacts away.
  In the experiment that contact hangs on the generator's 50 Ω output, so at 1 MHz it is a
  *sink*. With that (interior Dirichlet nodes, `dirichlet_nodes`) the peak lands on the
  stimulating contact and the amplitude is right.
- **Idle contacts are open at 200 Hz.** Their Pt-Ir double layer (~1 µF, hundreds of ohms at
  200 Hz, comparable to the access resistance) blocks through-flow; modelling them as metal in
  the stimulation solve moves the peak to the wrong contact and doubles the lateral width
  (4.95 mm). At 1 MHz the same double layer is a short, so in the lead-field solve they are
  metal. The example's defaults encode this; `AE_SIM_FLOATING` and `AE_SIM_STIM_SINK` bracket it.

Caveats: the result scales with K: the Gaussian-beam 0.47 becomes 0.16 at the 0.034 %/MPa
Preston's group quotes (three times low, the same direction as the Rintoul gap,
[02](02-rintoul-heterodyning-and-in-vivo.md)), and the FDTD-beam 0.83 becomes 0.28; the two
beams bracket the measurement from either side at K = 0.1 %/MPa and both undershoot at
0.034 %/MPa, so K ≈ 0.06 %/MPa would reconcile the FDTD model with the paper; the paper does not say
whether 0.52 is referred to the electrode or to the output of the 40 dB preamplifier; the
generator impedance splits the lead-field return between the stimulating contact and the far
ground, modelled here as a perfect sink; the lateral width depends on the assumed transducer
FWHM (a 3.3 mm beam would reproduce 3.67 mm).

## The FEM forward path (2026-10-23)

`ae_sim.forward.AEForwardSolver` solves −∇·(σ∇φ₁) = −∇·(σ K P E₀) on any tetrahedral mesh
(scalar, per-element or callable σ; insulated walls, ground walls, pinned contacts), with the
operator assembled and set up once (pyamg smoothed aggregation + CG; 442 k nodes: 16 s setup,
6 s per pressure snapshot) and every beam position or time step a new right-hand side. The
weak form takes the source as ∫ j_src·∇v with j_src = σ K P E₀, so P needs no derivative and
can be an FDTD snapshot (`grid_pressure`). `P1Locator` is a vertex-star element finder for
graded and general meshes (20 k points on a 50 k-node graded mesh in 0.7 s; scikit-fem's
finder asked for 41 GiB on the same job).

What it adds to the reciprocity operator: φ₁ and E₁ = −∇φ₁ everywhere from one solve, for
any number of recording sites, and the quantity the heterodyning / neuromodulation question
needs. Checks:

- **Discrete reciprocity is exact.** V_cd = φ_L · b_src on a shared mesh, to 1e-8 with AMG,
  on a tensor box and on a rotated two-compartment mesh with per-element σ; the grid-sampled
  `mixing.ae_voltage_direct` agrees to 5 %.
- **Depolarisation factors.** A uniform spot of pressure P in a uniform E₀ is a conductivity
  inclusion δ = K P, so inside it E₁ = −N K P E₀ with N the depolarisation factor of the shape.
  Measured N: sphere 0.334 (1/3), prolate spheroid of aspect 8 along E₀ 0.030 (0.0285),
  across E₀ 0.487 (0.486).
- **`examples/ae_field_map.py` (stage 37):** 500 kHz Gaussian spot (σ 1.3 × 8 mm, 1 MPa) in
  E₀ = 10 V/m between plates: N = 0.41 with the beam across E₀, 0.04 along it, so the AE
  current actually flowing in the spot is 0.59 or 0.96 × σ K P E₀ depending on how the
  transducer is aimed relative to the stimulation field. φ₁ peaks at 5.5 µV (across) and
  2.2 µV (along) at K = 0.1 %/MPa; linear in K, P and E₀.
- **Digimouse head (`examples/digimouse_head.py`, stage 39, 2026-10-23):** the Rintoul 2026
  layout on the whole-body mouse mesh (CC BY-SA 4.0, `scripts/fetch-digimouse.sh`): mouth
  loop and a dorsal head patch (the gel the 2 cm loop sits in) at 20 V, one low-frequency σ
  per organ, a 1 MPa Gaussian focus (σ 1.3 × 8 mm) 1 mm under the cortex over motor cortex.
  The loops drive 16 mA (1.2 kΩ), |E₀| in brain is 313 V/m (median), φ₁ peaks at 299 µV and
  the motor − visual electrode pair sees +98 µV at K = 0.1 %/MPa; |E₁| at the focus is
  0.37 × K P E₀, the "across" geometry of the field map (the beam from above cuts the
  nose-to-tail field). The 2006 mesh needed its ~11 k sliver tets collapsed (edges under
  0.05 mm; organ volumes preserved; AMG and LU agree to 1e-9). Not yet in the model: gel and
  cone as conductors, the skull in the acoustic path, the neural onset response.
- **Digimouse head model with gel and loops (`AE_SIM_MODEL=head`, stages 24 and 41,
  2026-10-24):** the head cut from the 0.1 mm label volume with the fixtures painted in
  (`ae_sim.mousehead`: a 3 mm gel blanket under a 22 mm cone footprint, the 2 cm loop as a
  0.3 mm wire torus in the gel, a 9 mm oral cavity along the snout midline with the mouth
  wire in it) and meshed by SimNIBS `create_mesh` (130 k nodes, 732 k tets, 4 min, cached).
  Same drive and focus: 42 mA (478 Ω, the gel halves the resistance), 386 V/m median in
  brain, φ₁ peak 510 µV, motor − visual +54 µV, E₁ = 0.22 K P E₀. Against the body model
  the electrode-pair number drops by half and the current entry moves off the scalp into
  the gel; the mouth wire ends under the olfactory bulbs, where the field reaches 1.6 kV/m.
- **Rintoul 2023 tank cross-check (`examples/rintoul_tank.py`, stage 26, 2026-10-24):**
  their k-space pressure phasor (CC BY 4.0) through our forward path as a complex source
  (`ae_potential` solves real and imaginary parts; `grid_pressure` and `P1Locator` keep
  phasors). Real pins at 250 V/m, 1 MPa, K = 1e-9: |E₁| at the midpoint 0.23 V/m with the
  pins along the beam (paper ~1 V/m) and 0.065 V/m across it, ratio 3.54 (paper 4.12).
  The pins alone close their 20–40× gap to ×4.3; K ≈ 4.3e-9 would match outright. The
  point exactly between pins placed across the beam is a symmetry zero of φ₁, so the
  recording number to compare is the maximum over their 0.5 mm scan (137 vs 75 µV).
- **Their k-space outputs vs the stated equation (stage 26 with the `kspace` set, 2026-10-24):**
  a fresh FFT solve of ∇²φ₁ = K E₀·∇P (`ae_sim.spectral`, plane-wave limit |E₁| = K P E₀ as a
  unit test) on their own grid with their K = 2e-8 and E₀ = 250 V/m gives 4.38 V/m
  (0.88 K P E₀) along the beam; their deposited E is 0.125 V/m. The deposited field is a
  constant factor 35 below the equation in both orientations while the shape agrees
  (parallel / perpendicular 5.3 vs 5.4; our FEM pins agree with the spectral solve to 5 %).
  The 2023 "K ≈ 2e-8 needed" rests on that factor; with the equation as stated the gap at
  K = 1e-9 is ×4.5 ([02](02-rintoul-heterodyning-and-in-vivo.md)).
- **Neural readout (`ae_sim.activation`, tutorial 1, 2026-10-24):** |E₁| at the focus is
  0.12 V/m at 20 V and 1 MPa in both Digimouse models, i.e. ~23 µV of somatic polarisation
  (0.2 mV per V/m, Radman 2009), 0.6 × the 0.2 V/m spike-phase entrainment field (Reato
  2010) and 1/50 of the ~6 V/m the 2026 paper estimates in vivo; reaching 6 V/m at 20 V
  would take ~50 MPa, or 300 V at 1 MPa. The heterodyned field is therefore a modulator at
  best; the 9 mV onset response needs another mechanism (the applied 500 kHz field itself
  has an activating function of 1e5–1e6 V/m² at the focus, 4 orders above E₁'s).
- **Cone artefact (`examples/cone_artefact.py`, stage 43, 2026-10-24):** the water-filled
  cone and its 127 µm film painted onto the head (`build_head_volume(cone=True)`, 159 k
  nodes / 885 k tets), the operator σ + iωε at 500 kHz (`ComplexForwardSolver`, pyamg on
  the complex matrix with BiCGStab), 1 V on the transducer face against the body, the
  unmeshed cone as a 4.5 kΩ series impedance. The brain sits at 22 mV per volt common mode
  and the motor − visual pair sees 0.77 mV per volt; the film is only a 5 kΩ capacitor at
  500 kHz, so removing it changes the artefact by ×1.3 and a ten-times-thicker film by
  ×3.5. A tens-of-volts drive puts tens of millivolts of 500 kHz pick-up on the differential
  recording, far above any imaging-mode AE signal, which is what the F21 attenuation
  experiment is about.
- **SIGMA rat DBS mapping (`examples/sigma_rat_dbs.py`, tutorial 2, stage 45, 2026-10-25):** the
  SIGMA v2 ex vivo head (`ae_sim.sigma_rat`: CT bone, GM / WM / CSF, soft tissue at 0.18 mm;
  the CT calvarium is 0.72–0.90 mm thick 2–3 mm off the midline) with a 0.72 mm insulated
  lead and a 1 mm contact in the right subthalamic nucleus driven against the cut neck, scalp
  patches 3 mm lateral and 8 mm posterior of the lead, a 1 MHz Gaussian spot (FWHM 1.5 mm,
  σ_ax 4 mm), 1 mA, 1 MPa, K = 1e-9. The image comes from the reciprocal operator on the
  pinned mesh (`AEForwardSolver.lead_field` + `reciprocal_weights`, one quadrature sum per
  focus; 959 foci in 48 s; equal to a direct `ae_potential` solve to 1e-4): contact 1.08 kΩ,
  |E₀| 650 V/m at 0.75 mm, peak **31 µV / (mA·MPa) = 29 µV / (V·MPa)** on the contact, image
  FWHM 1.5 × 1.5 mm laterally and 9.3 mm axially (the beam itself: a point-like source
  convolved with the focal spot), against the Tianjin abstracts' 167.74 µV V⁻¹ MPa⁻¹ and
  "< 2 mm". The factor ~6 is within what the unknown electrode and transducer parameters can
  move; the per-ampere value is 60× Preston's saline lead because the contact is 30× smaller
  in area and the recording electrode sits 3 mm away on the head rather than on the lead.
- **A CAD lead in the rat (`ae_sim.leadmesh`, gmsh, stages 49–50, 2026-10-25):** the painted
  4 × 4 voxel column was the CGAL mesher's compromise (a 0.36 mm column did not survive its
  spike removal, the 1 mm contact label was trimmed to 0.2 mm³, ~25 contact nodes). `leadmesh`
  cuts a round or square shaft with ring contacts out of the volume's box with gmsh's
  OpenCASCADE kernel (`python3-gmsh` 4.15 from Fedora), grades the mesh from 0.06 mm at the
  contact through 0.15 mm along the shaft to 0.5 mm in tissue, with bone, CSF and the scalp
  held at 0.25 / 0.3 mm by a Structured background field written from the voxels, tags the
  tissue from the voxel under each element's centroid and drops the air; HXT meshes the rat
  box (0.8 M tets) in 10–160 s in a child interpreter (HXT aborted inside the pytest process
  twice and never standalone; a shaft thinner than the elements gets the sizes capped at the
  diameter and a Delaunay retry). Same electrode, two meshers (stage 49): 913 Ω against CGAL's
  1079 (the exact 3.6 mm² contact surface conducts better than the trimmed voxel contact),
  26.2 against 31.0 µV / (mA·MPa), and the **same 28.7 against 28.8 µV / (V·MPa)**. The lead
  sweep (tip contact = 2 × diameter, 0.25 mm scan step):

  | lead | contact | area | Z | µV / (mA·MPa) | µV / (V·MPa) | image FWHM |
  |---|---|---|---|---|---|---|
  | round 0.125 mm | 0.25 mm | 0.11 mm² | 5.54 kΩ | 63.4 | 11.4 | 1.5 × 1.5 mm |
  | round 0.25 mm | 0.5 mm | 0.44 mm² | 2.70 kΩ | 52.3 | 19.3 | 1.5 × 1.5 mm |
  | round 0.5 mm | 1.0 mm | 1.77 mm² | 1.31 kΩ | 38.8 | 29.6 | 1.5 × 1.5 mm |
  | square 0.72 mm (painted) | 1.08 mm | 3.63 mm² | 0.91 kΩ | 26.2 | 28.7 | 2.0 × 1.75 mm |
  | round 1.0 mm | 1.5 mm | 5.50 mm² | 0.70 kΩ | 19.4 | 27.9 | 2.0 × 1.75 mm |

  Per ampere a small contact gains (the AE source is K P σ E₀ and E₀ concentrates at the
  contact), per volt the gain is more than paid back by the impedance: from 1 mm down to
  0.25 mm the per-volt sensitivity stays within 19–30 µV / (V·MPa) and a 0.125 mm wire falls
  to 11, so **the electrode geometry cannot carry the factor ~6 to the Tianjin 167.74
  µV V⁻¹ MPa⁻¹** (a smaller electrode moves the figure the wrong way); what remains is their
  beam, pulse length (stage 47: coherent ×0.09, envelope ×0.65), coupling constant, pressure
  convention and recording montage. The image FWHM is the 1.5 mm beam convolved with the
  contact (2.0 mm for the 1 mm lead). Resolution: h_contact 0.06 → 0.03 mm moves Z by +3 %
  and the peak by −0.5 %.
- **Bipolar lead (stage 51, 2026-10-25):** the same 0.25 mm lead with a second 0.5 mm ring as
  the return (`rat_dbs_setup(return_contact=...)`, `AE_SIM_LEAD_BIPOLAR=1`), gap 0.5 mm: 3.6 kΩ,
  peak **2.0 µV / (mA·MPa) = 0.6 µV / (V·MPa)**, 25× below the monopolar lead against the neck
  (52 / 19); gap 1.5 mm: 3.9 kΩ, 3.6 µV / (mA·MPa) = 0.9 µV / (V·MPa). The reason is in the
  uniform-pressure identity: V = K P × (mutual impedance between the stimulating pair and the
  recording pair) × I, and a local dipole on the lead has almost no mutual impedance with a
  scalp pair 3 and 8 mm away, whereas the monopolar current passes the scalp electrodes on its
  way to the neck. So the AE visibility of DBS on the scalp depends first on the return path:
  monopolar (case return) configurations image 25× better than bipolar ones, and a 167.74
  µV V⁻¹ MPa⁻¹ figure cannot be a bipolar lead seen from the scalp. The image width is the
  beam either way.
- **The pulse itself through the rat skull (`examples/sigma_rat_pulse.py`, tutorial 2c, stage
  52, 2026-10-25):** the bowl pattern on the plane plays a 2-cycle Hann burst
  (`acoustics.simulate_pulse_plane_source`), the pressure time series is recorded on the box
  around the contact (151 snapshots at 0.15 mm), the recording pair's reciprocal weights are
  folded onto that grid once (`forward.grid_weights`, exact against `reciprocal_readout`) and
  the AE voltage trace is one dot product per snapshot. Pulse transmission through the 0.6 mm
  of bone on the axis 0.52 (the CW 0.61 includes the standing-wave build-up). The trace's
  envelope peaks at **3.8 µV through the skull** (water 6.4, bone as soft tissue 6.5), i.e.
  0.12 of the Gaussian spot's 31 µV and 0.19 of the CW-envelope proxy of 2b (20 µV); a
  4-cycle burst gave 3.0 (0.10), tending to the CW phasor's 2.7 (0.09). The historical
  1-cycle result, 4.1 µV (0.13), used a Hann source with a nonzero time integral and is
  withdrawn as evidence for the pulse-length comparison. The source now removes its
  window-weighted mean for every phase; the 1-cycle case needs an FDTD rerun. A
  radiated pulse has no DC, so at every instant the packet changes sign across the ~1 mm
  coupling region around the contact and the trace reads the sensitivity kernel band-passed
  at the 1.5 mm wavelength, about 0.12 of its DC content; the envelope |P| (all positive) is
  the DC limit and overstates a pulsed readout by ~5× at 1 MHz for this contact (the
  band-pass finding of the first saline tests, now on the real head). A longer wavelength
  would read more of the kernel, at the cost of resolution; so would a narrower kernel, and it
  does: the 0.25 mm gmsh contact (`AE_SIM_LEAD=gmsh`) reads 0.28 of its own spot at 1 MHz,
  14.3 µV through the skull (water 24.3) against the 0.72 mm voxel contact's 3.8, 3.8× per
  milliampere and 1.5× per volt (5.3 against 3.5 µV / (V·MPa) through 2.7 and 1.1 kΩ), with an
  A-line error of 0.10 mm. For pulsed imaging the small contact wins per volt too, the
  opposite of the CW-spot ranking of stage 50. The
  A-line, the envelope against time of flight, peaks 0.8 mm shallow of the contact (9.7 mm
  below the plane): one pulse localises the contact along the beam. Sensitivity 3.8 µV /
  (mA·MPa) per free-field MPa, 7.2 per in-situ MPa. Against the noise
  (`noise.matched_filter_snr`: the pair's 2.97 kΩ from its own lead field, 7.1 nV/√Hz Johnson
  at 310 K plus a 5 nV/√Hz amplifier, matched filter on the computed trace): one pulse at 1 mA
  and 1 MPa gives SNR 0.44 through the skull (0.74 in water), so SNR 10 needs 510 pulses, 0.5 s
  at 1 kHz; at 0.5 MHz one pulse gives SNR 3.0 and 11 pulses suffice. Transcranial AE mapping
  of a DBS contact in the rat is seconds of averaging per focus, not hours.
- **The rat case at 0.5 MHz (stage 53, opt-in, 2026-10-25):** the same bowl, head and lead at
  half the frequency (`AE_SIM_F0=0.5e6`, spot FWHM scaled to 3 mm, giving a 45 µV spot
  readout). Skull: pulse transmission 0.77 (1 MHz: 0.52), CW 0.93 with a standing-wave
  ratio of 2.1 (1 MHz: 0.61 and 3.5). Readouts through the skull: 2-cycle burst **18.3 µV**
  (water 21.0), 0.40 of the spot (1 MHz: 3.8 µV, 0.12); the historical 1-cycle 14.5 (0.32)
  is likewise withdrawn pending a rerun with the zero-mean source; CW phasor
  20.2 (0.44; 1 MHz: 2.7, 0.09); CW envelope 31.7 (0.70). A-line error 0.14 mm. So halving
  the frequency multiplies the pulsed scalp signal by 4.8 and the CW signal by 7.4, from two
  factors: the 3 mm wavelength reads 3–4× more of the ~1 mm sensitivity kernel than 1.5 mm
  does, and the 0.6 mm calvarium transmits 1.5× more. The price is the beam: 3 mm instead of
  1.5 mm, which is the Tianjin "< 2 mm" resolution claim's frequency. For a scalp recording,
  sensitivity favours 0.5 MHz; resolution favours 1 MHz; the sensitivity kernel's own width
  (the contact's near field) sets the floor either way.
- **Onset response of a point neuron (`ae_sim.neuron`, `examples/onset_response.py`, tutorial
  1d, stage 48, 2026-10-25):** a Hodgkin–Huxley point neuron (1952 squid set) driven by the AE
  field through the Radman polarisation (0.2 mV per V/m, applied through the resting
  small-signal conductance so the steady polarisation is exactly λ_p E), with the field
  switched on as a step and modulated at 10 / 100 / 1000 Hz. At the 0.38 V/m of the real beam
  column (20 V, 1 MPa): 76 µV steady, 151 µV at the onset peak (the sodium transient doubles
  the step), no spike; the passive membrane passes 62 % of a 10 Hz modulation and 0.8 % of
  1 kHz. A spike from rest needs **9.6 V/m** (1.9 mV of polarisation: the HH current
  threshold of ~2.2 µA/cm² through 1.2 mS/cm²), 25× the field here, i.e. ~500 V on the
  loops at 1 MPa or 25 MPa at 20 V. That the model's spike threshold lands where the paper
  puts its 6 V/m estimate (with λ_p = 0.3 mV per V/m it would be 6.4 V/m) is the one place
  where our numbers and theirs agree without adjustment; what differs is the field the
  loops produce at the focus, 1/16 to 1/50 of it.
- **Rat CT skull in the acoustic path at 1 MHz (`examples/sigma_rat_skull.py`, tutorial 2b,
  stage 47, 2026-10-25):** a 1 MHz bowl (20 mm / ROC 25 mm; the Tianjin transducer is
  unknown) aimed at the STN contact, Rayleigh to the cut scalp 9.7 mm above it, CW
  BabelViscoFDTD at 0.15 mm on a 16 × 16 × 16 mm box (1.2 M cells, 2–4 min per run, phasors
  cached): the head transmits 0.61 of the free-field focal pressure (skull alone 0.64)
  through 0.6 mm of bone on the axis, and the CW standing-wave ratio within 3 mm of the
  contact is 3.5 (λ = 1.5 mm in a 10 mm brain cavity between calvarium and skull base).
  Through the reciprocal operator the field gives two readouts: the coherent phasor, 2.7 µV
  against 31 µV for the in-phase spot (×0.09: the travelling wave's phase turns through 2π
  every 1.5 mm across the ~1 mm coupling region of the contact and the sum cancels, the same
  effect that makes tone bursts image axial gradients only), and the amplitude envelope,
  the short-pulse limit the spot stands for: 20 µV (×0.65), or 26 µV (×0.83) for the
  free-field envelope scaled by the transmission (no CW reverberation). So the rat
  short-pulse sensitivity through the skull is **20 µV / (mA·MPa) per free-field MPa (33 per
  in-situ MPa)**, and the CW / long-burst value an order of magnitude lower; which one the
  Tianjin 167.74 µV V⁻¹ MPa⁻¹ corresponds to depends on their pulse length.
- **Skull in the acoustic path (`examples/digimouse_skull.py`, stage 42, 2026-10-24):** the
  60 mm / ROC 63 mm bowl propagated by the Rayleigh integral to the top of the gel 5.8 mm
  above the focus, then three CW BabelViscoFDTD runs on a 20 × 28 × 20 mm box at 0.2 mm
  (1.4 M cells, 30 periods, 3.5 min each on 4 CPU threads): water, head with the skull as
  soft tissue, head with the skull (fluid bone, 1900 kg/m³, 2900 m/s, 40 Np/m; air treated
  as water). The skull transmits 0.78 of the focal pressure and raises the on-axis
  standing-wave ratio in the brain from 1.08 to 1.70; the beam remains a ~1 MPa column
  through the whole head (the free-field plateau is 8 mm long), not an 8 mm spot. Fed to
  the forward solver that matters more than the skull: |E₁| at the focus is 0.38 V/m with
  the skull (0.45 without) against 0.11 V/m for the Gaussian spot, and the motor − visual
  pair 344 µV against 54 µV, ×3.4 and ×6.3. With the real beam the in vivo field is 1/16 of
  the paper's 6 V/m rather than 1/50; the polarisation proxy becomes ~75 µV. Two caveats
  of the setup: the imposed-stress source plane reads about twice the Rayleigh focus in
  water (calibration factor 0.47, absorbed by the water run), and the beam crosses the
  0.2 mm atlas skull obliquely, 1.6 mm of bone along the axis where a mouse parietal bone
  is 0.2–0.5 mm thick, so the transmission 0.78 and the standing-wave ratio are upper
  bounds on the skull's effect until the atlas skull is thinned to measured values.
- **Thinned calvarium and air (stage 46, 2026-10-25):** `ae_sim.mousehead.thin_skull` trims the
  atlas bone from the outside to 0.3 mm (keeping the outer surface; the deeper bone becomes
  what lies under it). With it the head transmits 0.91 of the free-field focal pressure
  instead of 0.80, while the on-axis standing-wave ratio stays 1.60 (1.70 before): the
  standing waves belong to the head as a whole (skull base, soft tissue boundaries), not to
  the calvarium, and |E₁| at the focus becomes 3.7× the Gaussian spot (3.4× before). Air
  around the head is not yet in the FDTD: BabelViscoFDTD's staggered grid goes NaN for a
  material lighter than ~300 kg/m³ next to water and for any attenuation on a slow
  material; a 300 kg/m³ / 150 m/s surrogate (reflection −0.94) is stable in every slab probe
  (against water, soft tissue and bone, at the source plane, 30 periods) yet the voxelised
  head diverges with it. `AE_SIM_AIR=1` keeps the knob, the surrogate and the boundary
  clean-up (enclosed pockets filled, one-cell spikes shaved) for the solver that takes it;
  air around the head would mostly change the lateral walls, which the bowl's beam does not
  reach at the focus.

## Baselines

| case | configuration | result |
|---|---|---|
| Saline bath (`examples/saline_scan.py`) | 10 mA, 1 MPa, 1 MHz, 1-cycle pulse | 0.18 nV at centre; electrode lobes up to 12 nV |
| FEM + FDTD phantom (`examples/babel_fem_phantom.py`) | finite insulated bath / infinite medium / ideal beam | 14.0 / 2.69 / 0.70 nV |
| Bath dipole (`examples/dipole_feasibility.py`) | 20 nA·m, 0.73 MPa at source, pair 3 cm apart 1.5 cm away, 2 MHz band | ~1 pV vs 2.4 µV noise, ~5e13 averages |
| Real head, `ernie` (`examples/head_feasibility.py`) | 20 nA·m dipole 15 mm under C3, 0.3 MPa at target, pair C3-C4, 3.5 µV noise | 1 MHz 31.5 pV (1e11 averages); 0.5 MHz 146 pV (5e9); 0.25 MHz 268 pV (1.6e9, ~400 h at 1 kHz PRF); point/DC upper bound 1.8 nV |
| Grid vs mesh lead field (stage 35) | 1 mm / 0.5 mm grid at 1 MHz; 0.5 MHz; 0.25 MHz | 31.5→11.0 / 23.4 pV; 146→63 pV; 268→168 pV |
| Preston 2018 DBS lead (`examples/dbs_lead_saline.py`, stage 36) | monopole, 4.40 MPa pk-pk, 1 MHz | 0.47 µV / (mA·MPa) vs 0.52 measured; peak 0.43 mm from the contact; 2.64 × 3.09 mm |
| Preston 2018 with the FDTD beam (`AE_SIM_BEAM=fdtd`, stage 38) | A392S 38 mm / 63 mm, λ/6, 4.40 MPa pk-pk | 0.83 µV / (mA·MPa); peak 0.10 mm from the contact; 2.97 × 2.20 mm; beam FWHM 3.25 mm at the lead |
| AE field map (`examples/ae_field_map.py`, stage 37) | 10 V/m plates, 500 kHz spot σ 1.3 × 8 mm, 1 MPa | depolarisation factor 0.41 across E₀ / 0.04 along; φ₁ 5.5 / 2.2 µV |
| Digimouse head, body model (`examples/digimouse_head.py`, stage 39) | mouth + head loops 20 V, 1 MPa Gaussian focus over motor cortex, per-organ σ | 16.2 mA (1.23 kΩ); 313 V/m in brain; φ₁ peak 299 µV; motor − visual +98 µV; E₁ = 0.37 K P E₀ |
| Digimouse head, painted head model (`AE_SIM_MODEL=head`, stage 41) | gel blanket 1 S/m, wire loop in the gel, oral cavity + mouth wire, SimNIBS mesh | 41.9 mA (478 Ω); 386 V/m in brain; φ₁ peak 510 µV; motor − visual +54 µV; E₁ = 0.22 K P E₀ |
| Rintoul 2023 tank (`examples/rintoul_tank.py`, stage 26) | their 500 kHz phasor, pins 3 mm apart at 250 V/m, 1 MPa, K = 1e-9 | |E₁| 0.23 V/m parallel / 0.065 perpendicular (paper ~1 V/m); ratio 3.54 (paper 4.12); scan max 137 / 75 µV; deposited k-space field / stated equation 0.0285 (both orientations) |
| Rintoul 2023 tank with the PVDF film (stage 26, `AE_SIM_FILM_MM=5,10,20`) | saline cut at an insulating plane 5 / 10.6 / 21 mm before the focus (oil side carries no current) | |E₁| at the midpoint ×1.000 ± 0.001 of the open box, recording ×0.99–1.02; ratio 3.54–3.55 unchanged |
| Neural readout at the focus (tutorial 1, both models) | 20 V, 1 MPa, K = 1e-9 | |E₁| 0.12 V/m → 23 µV polarisation, 0.58 × entrainment (0.2 V/m), 6 V/m needs 51–53 MPa at 20 V |
| Skull in the acoustic path (`examples/digimouse_skull.py`, stage 42) | 0.2 mm, 30 periods, generic bone, air as water | transmission 0.78 (head/water 0.80); SWR 1.70 vs 1.08; |E₁| 0.38 V/m (no skull 0.45, spot 0.11); motor − visual 344 µV (spot 54) |
| SIGMA rat DBS mapping (`examples/sigma_rat_dbs.py`, stage 45) | 0.72 mm lead, 1 mm contact in the right STN, scalp patches 3 / 8 mm, 1 MHz spot 1.5 mm, 1 mA, 1 MPa | 1.08 kΩ; peak 31 µV / (mA·MPa) = 29 µV / (V·MPa) (Tianjin 167.74); FWHM 1.5 × 1.5 × 9.3 mm; reciprocal = direct to 1e-4 |
| SIGMA rat, painted lead re-meshed by gmsh (`ae_sim.leadmesh`, stage 49) | the same 0.72 mm square column and 1.08 mm contact as CAD, 0.06 / 0.15 / 0.25 / 0.5 / 0.3 mm grading, 0.25 mm scan step | 913 Ω (CGAL 1079); 26.2 µV / (mA·MPa) (CGAL 31.0); 28.7 µV / (V·MPa) (CGAL 28.8); FWHM 2.0 × 1.75 × 9.25 mm; reciprocal = direct to 1e-4 |
| SIGMA rat, round 0.25 mm lead (stage 50) | 0.5 mm tip contact (0.44 mm²), otherwise as stage 45 | 2.71 kΩ; |E₀| 970 V/m at 0.75 mm; 52 µV / (mA·MPa) = 19 µV / (V·MPa); FWHM 1.5 × 1.5 × 9 mm; sweep 0.25–1 mm: 19–30 µV / (V·MPa) |
| SIGMA rat, bipolar 0.25 mm lead (stage 51) | return ring 0.5 mm long, 0.5 mm above the 0.5 mm contact (1.5 mm: second row) | 3.6 kΩ; 2.0 µV / (mA·MPa) = 0.6 µV / (V·MPa), ×0.04 of monopolar; gap 1.5 mm: 3.9 kΩ, 3.6 = 0.9 |
| Rat skull, pulsed (`examples/sigma_rat_pulse.py`, stage 52) | 2-cycle 1 MHz Hann burst, 0.15 mm, water / skull / no-skull, grid weights on the recorded box | pulse transmission 0.52; AE trace peak 3.8 µV (water 6.4, no skull 6.5) = 0.12 × spot, 0.19 × CW envelope; 4-cycle burst 3.0 µV (0.10, CW phasor 0.09); 1-cycle baseline withdrawn pending zero-mean-source rerun; A-line peak 0.8 mm shallow of the contact; 7.2 µV / (mA·MPa) in situ; one-pulse matched-filter SNR 0.44 (2.97 kΩ pair + 5 nV/√Hz), SNR 10 in 510 pulses; 0.25 mm gmsh contact: 14.3 µV = 0.28 × its spot, A-line error 0.10 mm |
| Rat skull at 0.5 MHz (stage 53, `AE_SIM_F0=0.5e6`, spot 3 mm) | pulsed (1 / 2 cycles) and CW, water / skull / no-skull | pulse transmission 0.77, CW 0.93, SWR 2.1; trace peak 18.3 µV (2 cycles; 1 MHz 3.8) = 0.40 × spot; CW phasor 20.2 (0.44), envelope 31.7 (0.70); A-line error 0.14 mm; one-pulse SNR 3.0, SNR 10 in 11 pulses |
| Thinned calvarium (`AE_SIM_SKULL_MM=0.3`, stage 46) | 0.2 mm, 30 periods, bone 0.3 mm from the outside, water + skull runs | head transmission 0.91 (0.80 at 1.6 mm); SWR 1.60; |E₁| 0.42 V/m = 3.7× the spot; motor − visual 322 µV = 5.9× |
| Onset response (`examples/onset_response.py`, stage 48) | HH 1952 point neuron, λ_p 0.2 mV per V/m, τ_m 20 ms, E₁ 0.38 V/m | 76 µV steady / 151 µV onset peak, no spike; spike needs 9.6 V/m = 25× (508 V on the loops, or 25 MPa at 20 V); 10 Hz gain 0.62 |
| Rat CT skull at 1 MHz (`examples/sigma_rat_skull.py`, stage 47) | 20 mm / ROC 25 mm bowl, 0.15 mm, 30 periods, water / skull / no-skull | head transmission 0.61 (skull 0.64), 0.6 mm bone on axis, SWR 3.5; readout vs spot: phasor ×0.09, envelope ×0.65, free-field envelope ×0.83; 20 µV / (mA·MPa) short-pulse |
| Cone artefact (`examples/cone_artefact.py`, stage 43) | 1 V on the transducer face, 127 µm film, gel 1 S/m, cone water 5e-3 S/m | motor − visual 0.77 mV/V (film), 0.97 (no film), 0.22 (10× film); brain common mode 22 mV/V; column |Z| 4.5 kΩ |

## Verification record

- scikit-fem lead field (Dirichlet-matched box) reproduces the analytic point-pair fields on
  the FDTD sensor grid to 0.4 %.
- An independent FEM reproduces the AE forward solve (2026-10-25, suggested and first run by
  the neurofedora-t2 session): SimNIBS's TMSFEM with the AE source as its dA/dt term
  (E = −∇v − ∂A/∂t, so ∂A/∂t := −K P E₀ gives J = σ(K P E₀ − ∇φ₁)) matches
  `AEForwardSolver` to 7.3e-3 in relative L2 on its coarse 4.6 k node sphere (P per element
  against per quadrature point) and to 1.5e-3 on the ernie low-res head (223 k nodes, 1.3 M
  tets). `tests/test_simnibs_crosscheck.py` keeps the sphere case (2 s). The same exercise
  gave the solver a MUMPS backend (`method="mumps"`, python-mumps, what SimNIBS uses on
  Fedora; Fedora's serial PETSc has no MUMPS): on the 140 k node rat lead mesh 12.6 s to
  factorise then 0.10 s per right-hand side, against pyamg's 2.3 s + 1.4 s per solve and
  SuperLU's 232 s; MUMPS and SuperLU agree to 5e-13, AMG to 2e-10. The complex solver has the
  same backend (zmumps; equal to complex SuperLU to 1e-9 in the test).
- The gmsh lead mesh (stages 49–50): a 0.25 mm contact in a uniform grounded box gives the
  point-source potential drop between 0.5 and 1.5 mm within 5 % and |E| at 1 mm within 10 %
  (sphere-averaged; a point-sampled P1 gradient is 7–9 % off at 1–1.5 mm, which is the
  mesh-granularity effect of the real heads in miniature).
- BabelViscoFDTD CPU backend (OpenMP) runs 77 × 77 × 186 at λ/8 in ~19 s on the AMD NUC.
- Ring-delay focusing in the Babel wrapper (`focus=`): the pressure maximum of an 8 mm / 20 mm
  source lands at 14.8 mm (focal shift of a low-gain source) with a 1.75 mm FWHM against the
  1.91 mm of 1.02 λ F#; the 38 mm / 63 mm Preston transducer takes 326 s at λ/6 on 4 threads.
- `TensorPoissonField` (nodal-grid J, Jacobi-PCG) solves 255 k nodes in ~20 s; the P1 projection
  path took 230 s for 80 k nodes and scikit-fem's element finder is brute force on graded meshes.
- `AEForwardSolver` (pyamg + CG) sets up 442 k nodes in 16 s and solves each snapshot in 6 s;
  the discrete reciprocity identity holds to 1e-8 and the spheroid depolarisation factors to 0.5 %.
- CPU pipeline on the NUC: 16/16 stages, ~9 min; inside the `fedora:44` + COPR container as a
  non-root user, SimNIBS included: 16/16.
- CPU baseline timings on the NUC: pytest 64 s, saline 71 s, phantom 67 s (6 GB RSS), dipole
  227 s (9.8 GB RSS), head 20 s cached.

## Hosts

| host | role | notes |
|---|---|---|
| AMD NUC (Fedora 44) | development, CPU baseline, container validation (rootless podman) | the GPU exposes no OpenCL device |
| legion | RTX 2080 8 GB, driver 615 / CUDA 13.4, Fedora 44 | one heavy job at a time |
| DGX Spark | aarch64 GB10 (sm_121), Ubuntu 24.04, driver 580 / CUDA 13.0, Docker 29, 120 GB unified memory | chosen production host; everything runs in the container; cupy capped at 16 GiB; data on /mnt/t9 |

Open unknowns on the Spark: the aarch64 COPR rpms installing (resolved from the NUC, not yet
executed), cupy-cuda13x on sm_121 (Babel's kernel is NVRTC-compiled at run time), and
numba / SimNIBS JIT on aarch64.
