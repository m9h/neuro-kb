# Which recent acoustoelectric neuroimaging papers can we replicate now?

*Survey date: 2026-10-22. Question from Morgan: "Of the recent acoustoelectric neuroimaging
papers what would be something we can now replicate?"*

The field splits into two lines. The **Witte / O'Donnell line** (Arizona, 0.6 MHz arrays,
saline and gel phantoms, then human skull) reports *absolute* sensitivities in µV per mA per
MPa, which is exactly what a simulator can be checked against. The **Tianjin line** (rat
in vivo, phased arrays, skull aberration correction) publishes behind IEEE/IOP paywalls and
needs an animal model, so it was out of reach at survey time (see
[03-rodent-head-models.md](03-rodent-head-models.md) for how that changed).

## Recommended targets, in order

### 1. Preston et al. 2018, Ultrasound Med Biol: DBS lead contacts in saline

DOI [10.1016/j.ultrasmedbio.2018.06.021](https://doi.org/10.1016/j.ultrasmedbio.2018.06.021)

| parameter | value |
|---|---|
| electrode | Medtronic 3389 DBS lead, contacts 1.27 mm diameter, 1.5 mm long |
| medium | 0.9 % NaCl |
| ultrasound | 1 MHz, focused at 63 mm |
| current | 200 Hz sine |
| sensitivity | **0.52 µV / (mA·MPa)**; 0.56 through a 6 mm skullcap |
| skullcap effect | ~80 % pressure attenuation, 13.8 dB SNR loss |
| detection threshold | 1.75 mA at 1 MPa |
| K quoted | 0.034 % / MPa |

**Done 2026-10-23** (`examples/dbs_lead_saline.py`, pipeline stage 36): 0.47 µV / (mA·MPa) at
K = 0.1 %/MPa against the measured 0.52, peak 0.43 mm from the stimulating contact, extent
2.64 × 3.09 mm against 3.67 × 3.11. Getting there required modelling the two circuits
correctly (stimulating contact a sink at 1 MHz, idle contacts open at 200 Hz); see
[05](05-prototype-findings.md). With the BabelViscoFDTD run of the A392S transducer instead
of the Gaussian beam (`AE_SIM_BEAM=fdtd`, stage 38): 0.83 µV / (mA·MPa), peak 0.10 mm from
the contact, 2.97 × 2.20 mm. The two beams bracket the measurement; K is the open question.

### 2. Kang et al. 2023, IEEE Trans Med Imaging: dipole in a saline cube, 3-axis leads

DOI [10.1109/TMI.2022.3215748](https://doi.org/10.1109/TMI.2022.3215748)

| parameter | value |
|---|---|
| medium | 8 cm saline cube, 1.2 S/m |
| source | dipole at the centre |
| recording | 3 orthogonal gold-cup lead pairs, 10 mm leads |
| array | 18 × 7 elements, 0.6 MHz, 48.6 × 29.3 mm aperture, pitch 2.7 / 4.18 mm, geometric focus 35 mm, mounted in the floor |
| pressure | 1 MPa |
| pulses | Gaussian, 45 / 60 / 90 % fractional bandwidth |
| their model | COMSOL + Field II |

Cheong et al. 2026, Ultrasonics, DOI
[10.1016/j.ultras.2026.108229](https://doi.org/10.1016/j.ultras.2026.108229), publishes
k-Wave fields for the same array, which gives an independent acoustic check. For us:
scikit-fem lead fields in the cube plus BabelViscoFDTD with a rectangular focused array as
the source map (~8 M cells at 0.6 MHz).

### 3. Allard et al. 2025, IEEE TBME, and Barragan et al. 2020, J Neural Eng: human skull + gel

- Allard 2025, DOI [10.1109/TBME.2024.3514553](https://doi.org/10.1109/TBME.2024.3514553)
  (open access, PMC12153511)
- Barragan 2020, DOI [10.1088/1741-2552/abb63a](https://doi.org/10.1088/1741-2552/abb63a)

| parameter | value |
|---|---|
| phantom | human skull filled with homogeneous agarose gel |
| electrode | 12-contact depth electrode, 2.41 mm contacts, 5 mm pitch, 30 mm deep (51 mm from the array); gold cup on the skull |
| array | 0.6 MHz, 126 elements |
| pressure | 0.7 MPa peak negative, measured through the skull (skull attenuation 60–70 %) |
| current | 3-cycle 200 Hz bursts, PRR 4 kHz, receive band 0.3–0.9 MHz |
| sensitivity | **1.5 µV / mA at 0.7 MPa**; threshold < 500 µA with 10 averages |
| resolution | 4.72 × 7.85 × 4.46 mm |
| noise-equivalent current | 360 µA (2020) |

Abu Farha et al. 2026, J Neural Eng, DOI
[10.1088/1741-2552/ae7764](https://doi.org/10.1088/1741-2552/ae7764) (paywalled) pushes the
limit to 28 µA for an ictal waveform with a Wiener filter.

For us: SimNIBS `ernie` with the brain compartments set to the gel conductivity, or a
scikit-fem skull shell. The acoustic skull step is *not* needed because they report the
pressure measured through the skull.

## Out of reach at survey time

The Tianjin in vivo rat work: Chen 2023 (SSVEP), Zhou 2024 and 2026 (DBS lead mapping in
rat), Zhou 2023 Front Physiol, Zhang 2023 / 2026 (skull-aberration-corrected phased array),
Guo 2026 Sensors (simulation platform; MDPI fetch returned 403). Abstract-level parameters
only: < 2 mm resolution, SNR 20 dB, 167.74 µV V⁻¹ MPa⁻¹, 3 × 3 mm scan areas. These need a
rat head volume conductor and the paywalled methods.

## What the survey changes in the code

1. **K must become a parameter.** We hard-code K = 1e-9 /Pa (0.1 % / MPa). Witte's group uses
   0.034 % / MPa; Wang et al. 2026, IEEE TBME, DOI
   [10.1109/TBME.2026.3682610](https://doi.org/10.1109/TBME.2026.3682610), measured
   0.024 % / MPa in saline and 0.022 % / MPa in porcine brain. Our head pV numbers scale down
   ~4× under those values. See [04-coupling-constant.md](04-coupling-constant.md).
2. Cylindrical contact source in the FEM (target 1).
3. Focused rectangular array source in `acoustics.py` (Babel takes arbitrary source maps).
4. Report sensitivity in µV / (mA·MPa) so results compare directly with the papers.
