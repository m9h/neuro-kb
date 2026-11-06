# Bill of materials (draft, unquoted)

All prices are order-of-magnitude estimates (`est`), not quotes. The Holoscan repository
is said to include a component shopping list; use it as the authoritative source for part A.

## A. Imaging bench — Holoscan Ultrasound Imaging Reference Design

| Item | Est. | Note |
|---|---|---|
| DGX Spark (GB10) | have | ConnectX-7 ports for RoCE. `docs/setup.md` in the reference design has a DGX Spark host tab and shows the HSB on Spark QSFP port 0, so Spark is a documented host; the README still says "the supplied image targets PGX. Qualify other hosts separately." Morgan is asking NVIDIA about Spark qualification |
| Terasic Holoscan Sensor Bridge 100G (Stratix 10 TX) | $5–8k est | 2× 100 GbE, one active |
| TI AFE5932EVM (2× AFE5932 RX, 2× TX73L64 TX; 128 TX / 64 RX) | $2–4k est | plus any TI controller/power boards the EVM needs — check |
| Sonosray L12-5 D linear probe | $1–3k est | 5–12 MHz; rodent / through-window imaging, not transcranial human |
| QSFP28 DAC cable, PSU, cabling, enclosure | $0.5k est | |
| Flow phantom + pump (CIRS or agar/graphite) | $0.3–3k est | for Doppler validation |
| **Subtotal** | **$10–18k est** | consistent with the ~$10k in the announcement |

## B. Stimulation bench — Open-LIFU

| Item | Est. | Note |
|---|---|---|
| Open-LIFU console + 64-element transducer (Openwater) | on request | OEM configurations exist |
| or fabricate from openlifu-electrical / -mechanical | $3–8k est | PCB fab + assembly; piezo elements and lens are the hard part |
| Alternative start: single-element focused transducer 250–500 kHz (e.g. Sonic Concepts H-series) | $8–15k est | |
| RF amplifier (e.g. E&I 1040L class) | $8–12k est | |
| Arbitrary waveform generator | $1–2k est | |

## C. Characterisation and safety — buy before B

| Item | Est. | Note |
|---|---|---|
| Needle / fibre-optic hydrophone + preamp (Onda, Precision Acoustics) | $5–15k est | |
| Water tank, 3-axis positioner, degasser | $2–6k est | Open-LIFU verification tank design or 3D-printer gantry |
| Radiation force balance | $3–6k est | or home-built |
| Skull phantoms (3D-printed from CT) | $0.5–2k est | ex-vivo only under institutional rules |
| Oscilloscope ≥ 200 MHz, current probe | $2–4k est | |

## D. Navigation and acoustoelectric

| Item | Est. | Note |
|---|---|---|
| Optical tracking (NDI Polaris Vega) | $15–25k est | later |
| or webcams + ArUco with Open-LIFU 3D-scanner workflow | $0.2k est | start here |
| High-rate multichannel ADC for acoustoelectric heterodyne (Red Pitaya / Digilent / Intan + ADC, or imaging AFE in RX-only) | $0.5–3k est | bandwidth must exceed the ultrasound frequency |
| Saline tank, electrodes, current source | $0.3k est | |
| Open-Motion optical blood-flow sensor + console (fabricate from openmotion-electrical/-mechanical) | $1–3k est | optional second haemodynamic readout during sonication |

Not purchasable: Butterfly Ultrasound-on-Chip modules (Embedded licence only); Nudge Zero; Iconeus One is a commercial animal system (price on request, six figures est).

## E. Compute

DGX Spark covers acquisition, simulation (BabelViscoFDTD/cupy, j-Wave/JAX) and planning.
Shared 120 GB CPU+GPU pool: cap every CUDA process.
