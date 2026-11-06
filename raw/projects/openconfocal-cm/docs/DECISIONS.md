# Architectural decisions, and corrections to widely-repeated bad advice

Every claim here is either sourced with a URL or explicitly flagged as inferred.
Used-equipment prices are **asking** prices from listings, not transacted.

---

## D1. ~~Stage-scanning, not galvo-scanning~~ **SUPERSEDED by D11**

**Decision.** Park the excitation spot on axis; scan the sample.

**Why.** In a stage-scanning confocal every pixel is imaged at the *same on-axis
field point*. Field curvature, lateral colour, coma and astigmatism all scale
with field height and are identically zero on axis, by symmetry. They are never
sampled. Therefore:

- No scan lens. No tube-lens/scan-lens telecentric pair. No descan path.
- An engineered tube lens (Thorlabs TTL200-A, ~$500–650) buys a flat, colour-
  corrected *field* that this instrument never uses. A plain achromat
  (**AC254-200-A, ~$117**) is essentially penalty-free. What still matters on
  axis — longitudinal chromatic and on-axis spherical — is exactly what an
  achromat corrects, and the objective dominates the spherical budget anyway.

**Cost.** Speed. Galvos scan at kHz line rates; a stage does not. For a static
sample that is the right trade.

**Sources.** Thorlabs tube-lens tutorial; MicroscopyU infinity optical systems;
ASI *Infinity Microscope Basics* (http://asiimaging.com/docs/infinity_microscope_basics).

---

## D2. ~~Camera at the image plane, not a pinhole and a point detector~~ **SUPERSEDED by D11**

**Decision.** Read a small camera ROI at each scan position. The raw datum is a
cube `(D, Ny, Nx)`, not an image.

**Why.** Three things fall out, and the third is the reason.

1. **The pinhole becomes software.** Summing detector elements inside a disk is
   *exactly* convolving the detection PSF with that pinhole — see
   `test_ism_stack_summed_over_a_disk_equals_the_confocal_psf`, which pins the
   two code paths together to 1e-9. So you choose the aperture after the
   experiment, sweep it, or select it by information content (D6). Nothing to
   buy, mount, or align — and **pinhole alignment is the hardest step in a
   conventional confocal**.
2. **Sheppard reassignment is free.** `sqrt(2)` resolution over widefield with no
   photon loss (~2x after deconvolution), because you kept the detector
   coordinate instead of integrating it away.
3. **Self-metrology.** Each scan position yields an *image*, so consecutive
   sub-images register against each other and recover the trajectory the stage
   actually followed. A point detector returns one scalar per position and can
   never do this. Given D3, this is decisive.

**Cost.** Camera read noise, versus a photon-counting SPAD's ~25–500 cps dark
rate. Only binds on genuinely dim fluorescence — see the Phase 3 detector plan.

**Sampling check.** An IMX477 (1.55 µm pixels) behind a 40x/0.65 at 520 nm puts
**~25 pixels across the Airy diameter**. Generous; binnable 2x2.

---

## D3. Resolution is not accuracy *(the arithmetic stands; the conclusion narrowed — see D11)*

**Decision.** Never trust a commanded stage position. Record where the sample
actually was, and reconstruct on those coordinates (`regrid.py`).

**Why.** Every stage in this price range quotes a *resolution* and hides its
*accuracy*.

| Stage | Headline (resolution) | Reality | Source |
|---|---|---|---|
| Open-loop stepper, 256x, 0.5 mm pitch | 9.77 nm/microstep | ±125 nm step accuracy (±5% of a full step, non-cumulative); ~300–600 nm smallest reliable move | ATO NEMA-17 datasheet; Lin Engineering; Faulhaber; ADI *Analog Dialogue* |
| Zaber X-LSM050B | 0.19 µm — Zaber's own column header reads *"Microstep Size (Resolution)"* | **<16 µm backlash, 25 µm accuracy**, no encoder | zaber.com/products/linear-stages/X-LSM/specs |
| Prior OptiScan ES111 | "100 nm step" | **<3.4 µm bidirectional repeatability, <14.4 µm accuracy** | prior.com/product/optiscan-es111 |
| OpenFlexure microscope stage | 67 ± 6 nm/microstep | **12.1 ± 1.9 µm backlash (XY)**; <1 µm small-move repeatability, ~15 µm at 1 mm; exponential viscoelastic creep | Sharkey et al., *RSI* 87:025104 (2016); Knapper et al., arXiv:2109.06842 |
| Renishaw TONiC + RGSZ20 | "1 nm resolution" | ±30 nm sub-divisional error; ±5 µm/m grating | Renishaw TONiC datasheet |
| PI P-733.2CD (closed-loop capacitive) | <0.3 nm resolution | **<2 nm repeatability, 0.03% linearity** | PI P-733 datasheet |

Only the last two rows position better than a micrometre. **A $5,500 Zaber is
less accurate (25 µm) than the $200 OpenFlexure's small-move repeatability
(<1 µm).** The commonly-cited $200 → $1,000 → $5,500 ladder is not a precision
ladder; it is a travel-range-and-convenience ladder.

**The stepper mechanism.** Torque from advancing one microstep is
`holding_torque · sin(90°/microsteps)`. At 256x that is **0.61% of holding
torque**, while *detent* torque alone is 5–20%. Faulhaber, verbatim: "taking a
microstep does not mean the motor will actually move… successive microsteps will
have to be realized until the accumulated torque exceeds the load torque." Then
it jumps. Microstepping buys smoothness and resolution. It does not buy accuracy,
and the vendors say so.

**Beware the OpenFlexure number conflation.** The widely-quoted "12 nm step,
1.6 µm/16 h drift" figures belong to the **Block Stage**, a 2×2×2 mm fibre-
alignment device (Meng et al., *Opt. Express* 28:4763, 2020) — *not* the
microscope stage. They are constantly cited as if they were the same instrument.

**Consequence.** Positioning is a metrology problem, not a mechanics problem.
Position can come from a closed-loop piezo's capacitive sensor monitor (digitize
it per pixel, ~$50 of ADC) or, per D2, from the sub-images themselves ($0).

---

## D4. Thermal drift is the binding constraint, and it is why you buy speed

Aluminium is **23 ppm/K**: a 100 mm structural loop moves **~2.3 µm per kelvin**.
A 147² Nyquist raster at 40x/0.65 runs 0.7 min with a piezo and **6.0 hours**
with a stepper's settling. Drift error is proportional to acquisition duration.

So the piezo scanner is not bought for its "step size." It is bought because it
settles in milliseconds instead of seconds, which shortens the scan by ~500x,
which is the only defence against drift that money can buy. Everything else is
free: enclosure, warm-up, fiducial correction (SMLM reaches <2 nm *residual*).

PLA is ~70 ppm/K, three times worse than aluminium. **The OpenFlexure stage stays
in the design for coarse navigation and for the DPC illumination geometry. It
must not be in the precision structural loop.** Mount the objective and the
sample on the same metal breadboard.

Open-loop piezo is not a shortcut either: PI's own actuator tutorial gives
**10–15% large-signal hysteresis** (of commanded stroke, not of range) and
**1–2% creep per time decade**, logarithmic. Disqualifying for quantitative Z
sectioning unless closed-loop — or unless measured and modelled.

**Vibration.** A light board on sorbothane or tennis balls resonates at 10–30 Hz,
*inside* the 5–30 Hz building-disturbance band, so it **amplifies**. Mass buys the
low corner frequency: an inner tube under a granite slab lands at ~1–3 Hz. The
NTNU HardwareX chamber (springs + steel-spheres-in-honey dashpots, $400) measured
**1.2 Hz** (Vestad & Steinert, 2022, PMC9058575).

---

## D5. Objective: Nikon or Olympus. Never Zeiss or Leica on a bare achromat.

**Correction to widely-repeated advice, including my own first draft.** The
correction placement is the opposite of what is usually asserted:

- **Zeiss and Leica put chromatic correction in the TUBE LENS.** The matched
  proprietary tube lens is required; Zeiss ICS deliberately leaves ~1.5% lateral
  colour in the objective for the tube lens to finish.
- **Nikon and Olympus correct within the OBJECTIVE.** Mix-and-match tolerant.

Since D1 puts a $117 achromat in the tube-lens position, **buy Nikon or Olympus**.
A Zeiss 63x/1.4 on a bare achromat shows visible residual colour.

Tube-lens focal lengths: Nikon 200 mm, Leica 200 mm, Olympus/Evident 180 mm,
Zeiss 165 mm. Engraved magnification is exact only with the native tube lens — an
Olympus 60x on a 200 mm achromat runs at ~54x. Source: ASI; MicroscopyU.

---

## D6. Choose the pinhole by information, not by folklore

"Run at 1 Airy unit" is a heuristic. Because D2 makes the aperture a software
choice, you can instead *measure* which aperture carries the most information
about the sample, using IDEAL's decoder-free estimator:

`I(Y;X) = H(Y) − H(Y|X)`, with `H(Y|X)` from Poisson shot-noise physics and
`H(Y)` from the measurements themselves. No reconstruction algorithm has to be
trained to rank designs. Pinkard, Kabuli, … Waller, NeurIPS 2025;
[arXiv:2405.20559](https://arxiv.org/abs/2405.20559);
[Waller-Lab/EncodingInformation](https://github.com/Waller-Lab/EncodingInformation).

`design.py` implements the Gaussian entropy **upper** bound. Two honest caveats,
both in the docstrings:

- It is an upper bound on `H(Y)` and hence on the MI. Per the paper's guidance,
  when comparing estimators prefer the **lowest** estimate. Swap in
  EncodingInformation's PixelCNN for publication-grade numbers. The Gaussian
  bound is adequate for *ranking apertures of one instrument*, which is all this
  module claims.
- A 2D in-focus simulation gives an aperture no out-of-focus background to
  reject, so it **understates** the value of closing down. Run the sweep on a 3D
  sample or on real data. On real data, that is one scan.

---

## D7. Software: pycro-manager, not ImSwitch. A licence decision.

- **ImSwitch (GPL-3.0)** is the most confocal-ready open engine — a real
  `PointScanTTLCycleDesigner`, per-pixel dwell, APD line readout. Linking it
  would relicense this repository, which is BSD-3 like `bsccm-jax`.
- **pycro-manager (BSD-3) on Micro-Manager (LGPL-2.1)** is licence-compatible and
  gives scripted, hardware-triggered stage sequences. Since we write the
  reconstruction anyway, this is the right trade.
- **OpenScan (BSD-2)** is a real laser-scanning stack, but it is a *galvo* engine
  with no stage-scan mode, and it is Windows + NI-DAQ only.
- **Cockpit / python-microscope (GPL-3)** has no demonstrated point-scanning.
- **ScanImage** is a galvo/resonant engine, needs MATLAB, and the free tier is
  gone (last free release 5.7R1).

---

## D8. What does not exist, and why this repo does

No fully-open, buildable, low-cost, *genuinely confocal* design exists except the
UW/Vaughan spinning disk (Biomed. Opt. Express 13:1102, 2022; disk $400–1,000,
module ~$7k atop an existing laser epi stand). openUC2 has no pinhole modality.
Two prior OpenFlexure-confocal adaptations exist on GitHub
(`pawel1010/Openflexure-confocal-microscope`,
`SebasM2000/Confocal_microscope_Openflexure_adaptation`); neither is a
characterized, published instrument.

Gaps the OpenFlexure papers leave open, which this instrument is positioned to
fill: a single hysteresis figure, a thermal coefficient (nm/K), a creep time
constant, and any confocal-specific characterization at all.

---

## D9. ~~Scientific purpose: paired label-free + fluorescence~~ **RETRACTED — see D11**

`bsccm-jax` does quantitative phase from coded illumination and targets virtual
staining (Cytoland/VisCy) against BSCCM's six fluorescence bands. The scope in
that repo produces label-free contrast and **no fluorescence ground truth**.

This instrument supplies it: paired label-free (DPC/FPM, transmission, LED head
on the condenser dovetail) and sectioned fluorescence (confocal/ISM, epi) through
the same objective, on the same sample, in the same coordinate frame, sharing a
camera. That is what makes the multi-line laser (matching BSCCM's bands) and the
piezo Z (stacks, not planes) worth their cost — and what makes TCSPC/FLIM, the
most tempting way to spend surplus budget, the wrong purchase.

---

## Known uncertainties

- Piezo and PMT/SPAD "new" pricing is **quote-only** across every vendor. The
  $8–18k figures circulating are inference, not sourced.
- Used prices are **asking** prices from listings. The only hard used data point
  found was a PI P-721K186 PIFOC refurb at $1,199.99.
- The scalar model in `confocal.py` **understates axial extent above NA ~0.7**.
  Do not quote an absolute sectioning number at NA 1.4 without the Chromatix
  oracle.
- `optics.optical_section_thickness` uses the Zeiss quadrature expression. The
  older `4nλ/NA²` depth-of-field formula gives ~2x larger numbers (33 µm vs
  14.4 µm at NA 0.25). Different quantities; do not mix them.
- Whether a 60x/1.4 oil objective (WD ~130 µm) physically clears a piezo scanner
  stacked on the OpenFlexure coarse stage is **unverified** and may force focus
  entirely onto the PIFOC.

---

## D10. Drift is not identifiable from a single raster (a negative result)

**Found by a test that failed for the right reason.** The first spec for
`regrid.reconstruct_joint` asserted that one raster could recover its own drift.
It cannot.

In a single pass, a given object row is visited within a narrow window of scan
time. A slow drift along the slow scan axis is therefore **confounded with a
stretch of the object**, and no optimizer separates them. Worse, it fails
*silently*: seeding the object by scattering the data at the commanded positions
bakes the drift into the object and creates a local minimum at zero drift whose
loss (0.353) is *lower* than the loss at the true drift (1.336). The solve
converges, reports success, and has found nothing.

Measured, at 1.8 px drift amplitude (`tests/test_joint.py`):

| Passes over the field | Recovered trajectory error | Object correlation |
|---|---|---|
| 1 | 1.805 px (i.e. the full drift amplitude) | 0.754 |
| 2 | 1.325 px | 0.899 |
| **3** | **0.010 px** | **0.988** |

**Consequence for acquisition.** If you intend to recover drift from the data
alone, **scan the field at least three times** rather than once at 3x the dwell.
Same photons, same duration, and each object point is then sampled under several
different drift states. This is free, and it is the difference between a
recovered trajectory and a confidently wrong one.

The alternatives, any one of which also suffices: a fiducial bead (absolute, no
accumulation), a capacitive sensor readback (~$50 of ADC), or ISM sub-image
registration (D2, $0 — and it is a *local* increment, so it does not suffer this
confound; it suffers a random walk instead, which is why you anchor it).

`test_single_pass_drift_is_not_identifiable` pins the negative result so that
nobody later "fixes" the solver to chase it.

---

## D11. What this instrument became, and why it changed three times

This design was rebuilt from scratch three times. Each reversal followed a fact,
not a preference, and the facts arrived in a particular order. The log is here so
nobody re-derives the dead ends.

### v1 — Stage-scanning confocal with camera-based ISM

**Premise:** "speed isn't a priority for static imaging."
**Argument:** every pixel is imaged on-axis, so field aberrations vanish, a $117
achromat is a penalty-free tube lens, no scan lens is needed, the pinhole becomes
software, and the scan trajectory can be recovered from the sub-images.
**Killed by:** the word *workhorse*. Run `scripts/design_scope.py`: one 147² field
at 40×/0.65 takes **0.7 min with a $5,000 piezo and 6 hours without one** — for one
plane, one channel. A lab instrument needs z-stacks, channels, and several users.
Stage-scanning cannot do that at any budget.

**What died with it:** `regrid.py` (reconstruct on measured coordinates),
`drift.py` (recover the trajectory), `scan.py` (unidirectional rasters). All were
tested and worked; they solved a problem this instrument no longer has. Recoverable
from git history if a stage ever re-enters the imaging path.

### v2 — Spinning disk (DIY, then a used Yokogawa CSU)

**Premise:** fixed samples make the microlens-free disk's ~95% light loss tolerable.
**Killed by:** the samples growing thicker. Shimozawa *et al.*, *PNAS* 110:3399
(2013): on a standard CSU-X1 (5× pitch/diameter), contrast is **completely lost at
~30 µm** and background exceeds 60% of signal by ~80 µm.

And a second error inside the first: I argued that *cleared* tissue would be fine
because clearing removes scattering. **It does not matter.** Disk crosstalk is
driven by out-of-focus *fluorescence* leaking through neighbouring pinholes, and
clearing does not reduce out-of-focus fluorescence — a densely labelled cleared
organoid produces more of it, not less. Buying a CSU-10 would have put $7,500 plus
a $3–8k relay into an instrument that cannot see into the samples.

### v3 — Galvo point-scanning confocal (current)

Fixed + cleared + thick is exactly the regime where a single pinhole wins: no
crosstalk term at all, and *fixed* means slow scanning costs nothing. Every
objection to a point scanner — months to build, owner-operator, slow frames —
either evaporates (slow doesn't matter) or was already accepted.

### The advice that inverted along the way

- **"Never buy a Zaber or Prior stage"** was conditioned on stage-scanning, where
  the stage *is* the pixel clock. For navigation and tiling, micron repeatability
  is correct and those stages are the right purchase. `stages.py` keeps the
  arithmetic and narrows the conclusion.
- **"A $117 achromat is penalty-free"** depended on every pixel being on-axis. A
  scanned field needs a real scan lens (CLS-SL, $3,262.86). Do not cut this corner.
- **"The $2,225 machine-vision camera is fine"** was justified on bright fixed
  samples. It is 2.4× oversampled at 60×, uncooled, ~3 e⁻ read noise. Buy the
  photon budget.

### D9 retracted

The paired label-free + sectioned-fluorescence rationale (DPC through the same
objective, for virtual-staining ground truth) was inferred from `bsccm-jax`, not
stated. It is not this instrument's purpose. The OpenFlexure-derived hardware that
served it has been removed, and with it the CERN-OHL-S obligation: this repository
is now BSD-3 throughout.

---

## D12. The clearing protocol is an optics decision

Solvent protocols (iDISCO, uDISCO, BABB, ECi — all n ≈ 1.56) are, per the *Nature
Protocols* tutorial, "capable of dissolving glues and plastics, **including those
used in the assembly of most objective lenses**." They force quartz chambers, FEP
or PTFE wetted parts, UV-cure or cyanoacrylate adhesives, a solvent-resistant
dipping cap, and a $28k objective.

Aqueous protocols (CUBIC ~1.48, CLARITY/RIMS ~1.46) preserve fluorescent proteins,
tolerate cheap optics, and don't destroy the chamber.

**Choose the protocol before the lens.** The reverse ordering is expensive and
irreversible. See `CLEARING.md`.

## D13. The objective, not the microscope, is the budget

`aberration.py` computes it. Index-mismatch spherical aberration grows **linearly
with depth** and scales as **NA⁴** once refocusing absorbs the piston and defocus
terms. At 520 nm, Strehl 0.8:

| | depth |
|---|---|
| 60×/1.4 oil into iDISCO (n=1.56) | **7.2 µm** |
| 60×/1.2 oil into iDISCO | 28.6 µm |
| 20×/0.75 air into iDISCO | 9.3 µm |
| 10×/0.45 air into iDISCO | 113.1 µm |
| 25×/1.1 index-matched | unlimited (WD-limited) |

Corroboration: axial resolution degrades ~380% by 90 µm for an oil objective into
an aqueous sample; immersion objectives "lose their excellent imaging properties at
depths exceeding ~10 µm."

Two independent walls: aberration, and working distance. A 60×/1.4 oil has ~130 µm
of WD — even index-matched it could never reach the middle of a 300 µm organoid.
This is why every cleared-tissue objective is low-NA and long-WD. That is not a
compromise; it is the optimum.
