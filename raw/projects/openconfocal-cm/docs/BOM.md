# Bill of materials

`[S]` price read off a live vendor page. `[U]` used-market **asking** price from a
listing, not transacted. `[Q]` quote-only, no public price exists. `[I]` inferred.

Piezo, PMT, SPAD and cleared-tissue-objective "new" pricing is quote-only across
*every* vendor. Treat `[I]` as an order of magnitude.

## Phase 1 — substrate (~$9–20k)

| Item | Price | |
|---|---|---|
| Used Olympus IX81, bare working (needs IX2-UCB controller) | $2,500–5,000 | `[S]` |
| — or refurb, loaded | $11,995–13,995 | `[S]` |
| Nikon Ti-E refurb, loaded | $15,000–25,000 | `[I]` |
| Used cooled sCMOS: Andor Zyla 4.2 | $3,000 | `[U]` |
| — or Hamamatsu Flash4.0 refurb | $5,795 | `[S]` |
| LED illuminator (CoolLED / Lumencor, used) | $3,000–8,000 | `[I]` |
| Micro-Manager + pycro-manager | $0 | |

**Camera note.** 6.5 µm sCMOS pixels are exactly Nyquist at 60× (108 nm/px, 2.3
samples per resolution element). The cheap machine-vision route (Sony IMX540, FLIR
Blackfly BFS-U3-244S8M-C, **$2,225** `[S]`) is **2.4× oversampled at 60×**,
uncooled, ~3 e⁻ read noise. Defensible for bright fixed samples with binning;
wrong for anything dim. Confocal is photon-starved.

## Phase 2 — scan head (~$11–15k)

| Item | Price | |
|---|---|---|
| Galvo pair + mount | ~$2,000 | `[I]` |
| Thorlabs **CLS-SL** scan lens, EFL 70 mm | $3,262.86 | `[S]` |
| Thorlabs **TTL200** tube lens | $500 | `[S]` |
| Used Hamamatsu **H7422P-40** cooled GaAsP PMT (~100 cps dark, ~40% peak QE) | ~$1,200 | `[U]` |
| — + M13414 PSU $757 `[S]`, A7423 heatsink $268 `[S]` | ~$1,025 | `[S]` |
| Used **NI USB-6363** DAQ (4 AO 16-bit 2.86 MS/s, 4× 32-bit counters) | $500–1,200 | `[U]` |
| Used **Cobolt Skyra** 3-line combiner, 50 mW/line at fibre | $6,750 | `[S]` |
| — or used Coherent OBIS 488 nm 50 mW | $1,500 | `[U]` |
| Dichroics, emission filters, pinhole, relay | ~$2,000 | `[I]` |
| Breadboard, cage, mounts | ~$2,000 | `[I]` |
| Micro-Manager + **OpenScan** (BSD-2) + pycro-manager | $0 | |

**Thorlabs LSKGG4** packaged galvo head: price not retrievable (JS shell);
**~$6,000–7,000 `[I]`** — get a quote before believing it.

**Do not DIY the laser launch.** NicoLase measured single-mode coupling of 68% at
405 nm, 58% at 488, 55% at 561, **31% at 640**. No cheap 561 nm diode exists —
DPSS, $4–10k for that line alone. Sub-micron alignment stability; a 1 °C swing
detunes it.

## Phase 3 — the objective

| Objective | Mag/NA | WD | RI | Price |
|---|---|---|---|---|
| Low-NA air lens to start (**NA ≤ 0.3**, see CLEARING.md) | 10×/0.30 | ~4 mm | air | $500–1,500 `[I]` |
| Olympus XLPLN25XSVMP | 25×/1.00 | 4 mm | multi | **$28,090** `[S]` |
| Nikon CFI75 Apo 25XC W 1300 | 25×/1.10 | 2 mm | **water only** | **$31,500** `[S]` |
| Zeiss Clr Plan-Neofluar 20×/1.0 | 20×/1.00 | 6.4 mm | nd 1.53±0.05 | ~$25–40k `[I]` |
| Special Optics / ASI 54-12-8 | ~15×/0.70 | 10 mm | **1.33–1.56** | $30k+ `[I]` |

**Trap:** the Nikon 25XC W 1300 is a water-dipping two-photon lens. Its collar
corrects coverslip thickness, **not** high-RI clearing media.

## Deliberately not bought

| Item | Avoided | Why |
|---|---|---|
| Yokogawa CSU spinning-disk head | $7,500–37,500 `[S]` | Pinhole crosstalk kills sectioning past ~30–40 µm. Clearing does not help. |
| DIY multi-line laser launch | $3–6k of parts | 31% coupling at 640 nm; no cheap 561 nm. A used Skyra undercuts the parts. |
| Zaber X-LSM / Prior ES111 as a *scanning* stage | $3,000–5,500 | 25 µm and 14.4 µm accuracy behind 100 nm headlines. (As a *navigation* stage they are fine.) |
| Swabian Time Tagger / TCSPC | $7,000–9,000 `[I]` | FLIM is orthogonal to this instrument. |
| Mirrorcle MEMS scanner | $6,950–8,490 `[S]` | Sub-3 mm aperture cannot fill a Ø8–14 mm back pupil. |
| Engineered tube lens over the scan lens | — | You need a *real* scan lens; that is not the corner to cut. |

## Environment — best return per dollar in the whole BOM

| Item | Effect | Price |
|---|---|---|
| Granite slab on inner tubes | ~1–3 Hz corner frequency | ~$300 `[I]` |
| Draught-proof enclosure + 2 h warm-up | biggest cheap win | ~$200 `[I]` |
| PI-controlled slight over-temperature | ~5 nm/hour (Adler & Pham 2003) | ~$300 `[I]` |

**Not** tennis balls or bare sorbothane under a light board: those resonate at
10–30 Hz, inside the 5–30 Hz building band, and **amplify**. Mass buys the low
corner. The NTNU HardwareX chamber (springs + steel-spheres-in-honey dashpots,
$400) measured 1.2 Hz.
