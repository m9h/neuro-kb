# The Rintoul papers: heterodyning tank, in vivo neuromodulation, neural recording

Jean Rintoul's three papers (Imperial College, Grossman lab) are the most completely specified
acoustoelectric experiments in the open literature, and the 2023 tank data and solver are
public. They are the first replication we intend to run end to end.

## Rintoul et al. 2023, Communications Physics: acoustoelectric heterodyning

DOI [10.1038/s42005-023-01198-w](https://doi.org/10.1038/s42005-023-01198-w)

### Setup

| item | value |
|---|---|
| tank | 50 × 20 × 20 cm, split by a 12 µm PVDF film: transducer side in mineral oil, electrode side in 0.9 % saline (1.6 S/m) |
| transducer | 500 kHz curved PZT, 60 mm diameter, radius of curvature 63 mm, 1 MPa at focus |
| hydrophone | 0.2 mm, 52 mV/kPa |
| applied field | Pt-Ir pins, 0.25 mm wire, 1.4 mm exposed tips, 3 mm spacing; 8 kHz (also 499.99 kHz); ±20 V transformer source; ~250 V/m at the measurement point |
| recording | electrode between the pins vs a remote reference; SR560 preamp, gain 500, 100 kHz high-pass |
| scan | 0.5 mm steps |

### Results

- 250 V/m applied gives ~1 V/m at the sum / difference frequencies at 1 MPa.
- Parallel / perpendicular ratio of the AE field: 4.12.
- Their simulation with K = 1e-9 gives a coupling of ~55 pPa⁻¹ against ~2000 pPa⁻¹ measured.
  The authors say K ≈ 2e-8 would be needed, and the published solver hard-codes
  `adiabatic_compressibility = 2e-8` (code comments: 4.6e-10 is the physical value; 1e-9
  "from Xizi Song's paper").

That 20–40× gap is the interesting number. Our own prototype found a ×5 boundary gain from a
finite insulated bath and a ×4 near-field correction relative to an ideal beam, which between
them span the gap without touching K. Testing whether real pins, the PVDF film and the
reference electrode close it is the point of the replication.

**Cross-check done (2026-10-24, `examples/rintoul_tank.py`, stage 26).** Their deposited
`pressure0.5MHz.npz` phasor (101 × 101 × 1401 at 0.15 / 0.1 mm, normalised to 1 MPa at
z = 60 mm, lateral FWHM 4.2 mm) drives our FEM forward path with the pins modelled as pinned
wires (0.25 mm, 1.4 mm tips, 3 mm apart) at 250 V/m. At K = 1e-9: |E₁| at the midpoint
0.23 V/m with the pins along the beam and 0.065 V/m across it, ratio 3.54 against the measured
4.12. So real pins already bring the discrepancy from ×20–40 to ×4.3, and K ≈ 4.3e-9 would close
it; the 2e-8 their solver hard-codes is not needed. Two cautions: the point exactly between
pins placed across the beam is a symmetry zero of φ₁ (the recording electrode's reading there
depends on sub-millimetre placement, so the scanned maximum, 137 vs 75 µV, is the comparable
number). **The PVDF film and the oil side are in the box since 2026-10-25** (`AE_SIM_FILM_MM`,
default 5, 10, 20 mm before the focus: the saline ends at an insulating plane there, the oil
carries the beam but no current): |E₁| at the midpoint changes by less than 0.1 % and the
recording electrode by 1–2 % for any of the three distances. The AE source that reaches the
electrode is the few millimetres of beam around the pins; the near-field ripple on the
transducer side contributes nothing measurable. The film cannot be part of the discrepancy.

**Their k-space outputs, checked (2026-10-24, fetch set `kspace`).** `ae_data_kspace_a0degrees`
and `a90degrees_f0.5MHz.npz` hold φ and E on the 101 × 101 × 800 grid (z = 20–100 mm). At the
focus their E_AE is 0.125 V/m along the beam (φ 63.5 µV) and 0.009 V/m on the axis across it
(0.024 V/m within 3 mm of the axis). We solved the equation their paper states,
∇²φ₁ = K E₀·∇P with uniform E₀ = 250 V/m and their K = 2e-8, on their own pressure grid
(`ae_sim.spectral`, a fresh FFT implementation; the plane-wave limit |E₁| = K P E₀ is exact
and is a unit test): 4.38 V/m = 0.88 K P E₀ along the beam, 0.81 V/m near the axis across it.
The deposited field is a constant factor **35** below that in *both* orientations
(0.0285 and 0.0289), while the shape agrees: parallel / perpendicular 5.3 (theirs) vs 5.4
(spectral) vs 3.5 (our FEM with real pins) vs 4.12 measured. Our FEM pins and the spectral
uniform-E₀ solve agree to 5 % at the same K. So the 2023 "K ≈ 2e-8 is needed" conclusion rests
on a factor of 35 in the simulation, not in the physics: with the equation as stated, K = 1e-9
gives 0.22 V/m against the measured ~1 V/m (×4.5, K ≈ 4.5e-9 would fit), and the physical
water value 4.6e-10 is ×10 short. We cannot see which step produced the factor (the code is
NC-SA and was not run; a units choice in the source term or the k-vector, or a smaller E₀ in
the run, would do it); it is a question for the authors.

### Code and data

GitHub organisation [Acoustoelectric](https://github.com/Acoustoelectric):

| repo | content | licence |
|---|---|---|
| `Acoustoelectric` | k-space Poisson solver ∇·E_AE = E₀ β·∇P (`kspace_solver.py`); `focused.m` Bessel-beam bowl, a = 30 mm, ROC = 63 mm, c₀ = 1488 m/s, dx = 0.1 mm | CC BY-NC-SA 4.0 |
| `Acoustoelectric_Neural_Recording` | IQ demodulation example | — |
| `Acoustoelectric_neuromodulation_and_its_contribution_to_US_stimulation` | figure data only | Apache-2.0 |

Figshare collection [c.6365098](https://doi.org/10.6084/m9.figshare.c.6365098) (CC BY 4.0):

- 21802165: simulation outputs `ae_data_kspace_a0degrees_f0.5MHz.npz` and
  `..._a90degrees_...npz` (653 MB each) and `pressure0.5MHz.npz` (229 MB)
- 21802171: `df10hz_stream.npy`, `df492khz_stream.npy` (480 MB each)
- 21802183: Fig 4 2-D maps, a **dead 0 MB Dropbox placeholder**

Other collections: c.7754537 (recording), c.7909283 (neuromodulation figure data). Raw in
vivo data "on reasonable request". Imperial has a patent application on the technology.

## Rintoul et al. 2026, Nature Communications: acoustoelectric neuromodulation

DOI [10.1038/s41467-026-73826-2](https://doi.org/10.1038/s41467-026-73826-2) (PMC13276402)

| item | value |
|---|---|
| ultrasound | same transducer, CW 500 kHz, 1 MPa (0.5–2.5 MPa in vivo) |
| animals | mice; Pt-Ir 0.5 mm recording wires |
| recording site | motor cortex, AP 0 / ML −2 mm; reference V1, AP −3.5 / ML +2.25 mm |
| stimulating electrodes | loop in the mouth (1 cm) and loop above the head (2 cm) |
| AE potential | ~9 mV at ultrasound onset, ~1.5 mm between extrema, estimated ~6 V/m |
| acoustics model | Sim4Life mouse head, standing waves reported in the Supplement |
| phantom | 4 cm petri dish |

This is the target for the mouse head model in
[03-rodent-head-models.md](03-rodent-head-models.md): the geometry is fully specified, the
mouth-to-head applied field is far from uniform, and the 3 mm wavelength is a large fraction
of the head, so a quasi-static Poisson solve on a Digimouse mesh plus our FDTD is a real test.

## Rintoul et al. 2026, Communications Engineering: acoustoelectric neural recording

DOI [10.1038/s44172-026-00598-4](https://doi.org/10.1038/s44172-026-00598-4) (PMC12920806)

500 kHz CW at 1 MPa; SR560 gain 2000 / 5000, 0.1–1 MHz band-pass; 2 MHz 14-bit ADC; phantom
with 1 Vpp at 8 kHz across pins 7 mm apart.

## Replication plan

1. **Solver cross-check.** scikit-fem Poisson solve with source term −∇(K P)·E₀ driven by a
   BabelViscoFDTD bowl-transducer field, compared against the published `.npz` outputs
   (same K = 2e-8 as the solver, to isolate the numerics).
2. **Geometry test.** Same solve with real pins, the PVDF film / insulating walls and the
   reference electrode at K = 1e-9, to see how much of the 20–40× gap is geometry.
3. **In vivo prediction.** Digimouse head, her electrode positions, FDTD standing-wave
   field, predict the 9 mV / 6 V/m.

## Questions for the author

- Fig 4 map data in volts, the gain chain, and exact electrode positions (the Figshare item
  is empty).
- The hydrophone scan of the transducer.
- Her view on the K discrepancy, and specifically the factor of 35 between the deposited
  `ae_data_kspace_*` fields and the stated equation solved on the same grid (above).
- In vivo applied voltage and electrode positions; the Sim4Life mouse model if shareable.
- Licence intent for the 2023 solver (CC BY-NC-SA; we use it for reference only).
- Whether a joint imaging-mode µV / (mA·MPa) measurement on her rig is possible.
