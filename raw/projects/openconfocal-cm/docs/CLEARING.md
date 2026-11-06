# Clearing protocols, refractive index, and the objective you must then buy

**The clearing protocol and the objective are one decision, not two.** Pick the
protocol first only if you understand what index it leaves you with.

## Refractive index by protocol

| Protocol | Chemistry | Final RI | Size change | Fluorescent protein | Immuno |
|---|---|---|---|---|---|
| ScaleA2 | urea/glycerol/Triton | **~1.38** | expands ~1.25× | good | limited |
| ClearT2 | formamide + PEG | ~1.44 | mild | moderate | limited |
| CLARITY / PACT | hydrogel + SDS | **~1.45–1.46** | mild swell | good | good |
| RIMS / EasyIndex | Histodenz mount | 1.46–1.48 | n/a | good | n/a |
| ScaleS | sorbitol/urea | ~1.47 | slight expand | good, lipid-preserving | ok |
| **CUBIC** | aminoalcohol hyperhydration | **~1.48** | slight expansion | **good** | good |
| FRUIT | fructose + urea | ~1.48 | minimal | good | moderate |
| SeeDB | fructose | ~1.49 | minimal (~1%) | good | limited (poor Ab penetration) |
| Ce3D | iohexol + NMA | ~1.49 | minimal | ok | good |
| TDE | thiodiethanol | tunable to ~1.52 | minimal | good | good |
| SeeDB2G | iohexol | 1.52 | minimal | good | moderate |
| **iDISCO / iDISCO+** | MeOH / DBE | **1.56** | **shrinks 10–30%** | **poor (quenches GFP)** | **excellent** |
| 3DISCO | THF / DBE | 1.56 | shrinks | **very poor** (FP half-life 1–2 d) | limited |
| uDISCO | tBuOH + BABB-D | 1.56 | **shrinks 30–65%** | good (FP stable months) | moderate |
| FDISCO / sDISCO | pH/antioxidant-tuned | 1.56 | shrinks | good | moderate |
| BABB | benzyl alcohol/benzoate | ~1.55–1.56 | shrinks | poor | limited |
| ECi | ethyl cinnamate | **1.56** | low shrink | moderate–good | moderate |

Three regimes: **aqueous/hydrogel 1.38–1.48**, **high-aqueous 1.49–1.52**, and
**solvent 1.55–1.56**.

## Why solvent clearing is expensive beyond the reagents

Organic solvents (DBE, BABB, ECi) are, per the *Nature Protocols* tutorial
([PMC10542857](https://pmc.ncbi.nlm.nih.gov/articles/PMC10542857/)), "capable of
dissolving glues and plastics, **including those used in the assembly of most
objective lenses**." Consequences you must budget for:

- Chambers in **glass or quartz** only. Polystyrene, PMMA, PDMS and most 3D-print
  resins craze or dissolve. FEP and PTFE are inert and safe.
- **Agarose does not survive** dehydration or solvent. Use FEP tubes/foil (n≈1.34)
  or glue the sample to a glass rod with UV-cure or cyanoacrylate — epoxies and
  hot-melt fail.
- The objective front element needs a solvent-resistant dipping cap.
- DBE forms peroxides on storage, which quench fluorescence. That is exactly why
  FDISCO and sDISCO add antioxidants.

**Failure modes seen in practice:** chamber crazing dropping the sample mid-scan;
RI drift as the medium absorbs water or forms peroxides; bubbles and menisci;
agarose creep during a long stack; a correction collar left at the wrong index,
silently reintroducing the aberration you paid to remove.

## Objectives

Prices are list. The used market for these is thin — do not plan on it.

| Objective | Mag / NA | WD | RI range | Price |
|---|---|---|---|---|
| Nikon CFI75 Apo 25XC W 1300 | 25×/1.10 | 2.0 mm | **water-dipping ~1.33** | **$31,500** `[S]` |
| Nikon CFI90 20XC Glyc | 20×/1.00 | 8.2 mm | 1.44–1.50 (collar) | ~$25–35k `[I]` |
| Olympus XLPLN10XSVMP | 10×/0.60 | **8 mm** | ~1.33–1.52 (collar) | ~$15–20k `[I]` |
| Olympus XLPLN25XSVMP | 25×/1.00 | 4 mm | ScaleView / silicone / multi | **$28,090** `[S]` |
| Zeiss Clr Plan-Neofluar 20×/1.0 | 20×/1.00 | 6.4 mm | **nd = 1.53 ± 0.05** (variant-specific) | ~$25–40k `[I]` |
| Special Optics / ASI 54-12-8 | ~15×/0.70 | 10 mm | **1.33–1.56**, no collar | $30k+ `[I]` |
| Leica HC PL FLUOTAR 5×/0.15 | 5×/0.15 | 13.7 mm (dry) | air | ~$1–2k |

**Trap:** the Nikon 25XC W 1300 is a *water-dipping two-photon* lens. Its collar
corrects coverslip thickness, **not** high-RI clearing media. Do not buy it
expecting DBE compatibility.

## What an ordinary 60×/1.4 oil actually does

It fails within the first ~10 µm. Run `openconfocal_cm.aberration`:

```
Depth at which Strehl falls to 0.8 (Marechal, diffraction-limited), 520 nm:
  60x/1.4 oil  -> iDISCO n=1.56        7.2 um
  60x/1.2 oil  -> iDISCO n=1.56       28.6 um
  20x/0.75 air -> iDISCO n=1.56        9.3 um
  20x/0.75 air -> CUBIC  n=1.48        9.8 um
  10x/0.45 air -> iDISCO n=1.56      113.1 um
  25x/1.1 multi-immersion, matched    unlimited (working-distance limited)
```

Aberration scales as **NA⁴** once piston and defocus are removed — refocusing
absorbs those two terms, and what is left is pure spherical. Doubling NA costs a
factor of ~16 in usable depth. That is why every cleared-tissue objective you can
name is low-NA and long-working-distance, and it is not a compromise: it is the
optimum.

Independent corroboration: axial resolution degrades **~380% by 90 µm** for an oil
objective into an aqueous sample, and immersion objectives "lose their excellent
imaging properties at depths exceeding ~10 µm" through a coverslip into aqueous
solution.

**Two independent walls.** Aberration, and working distance. A 60×/1.4 oil has
~130 µm of WD: even perfectly index-matched it could never reach the middle of a
300 µm organoid. `aberration.usable_depth_um` returns the lesser of the two.

## Recommendation for a lab starting out

Choose an **aqueous protocol** (CUBIC ~1.48, or CLARITY/PACT with RIMS ~1.46).
You keep fluorescent proteins, you keep your plastics, and you keep your objective.

Then be honest about the air lens. Depth depends on **NA and the two indices only** —
magnification is not in the physics:

| Objective | into CUBIC (1.48) | into iDISCO (1.56) |
|---|---|---|
| 60×/1.4 oil | 6.6 µm | 7.2 µm |
| 40×/1.3 oil | 15.5 µm | 15.3 µm |
| **20×/0.75 air** | **9.8 µm** | **9.3 µm** |
| 10× or 20× /0.45 air | 119.8 µm | 113.1 µm |
| 10×/0.30 air | 681.5 µm | 641.3 µm |
| 4×/0.16 air | 8.98 mm | 8.42 mm |
| 25×/0.95 water dipping | 21.7 µm | 16.1 µm |
| 20×/1.0 glycerol (n=1.47) | **319.0 µm** | 41.1 µm |
| 25×/1.0 multi, collar at 1.48 | **unlimited** | 47.2 µm |
| 20×/1.0 multi, collar at 1.53 | 71.6 µm | 138.6 µm |

Reading it: **a high-NA air lens is worse than oil** — air against 1.56 is a bigger
mismatch than oil is. To use an air lens in cleared tissue you must drop to
NA ≤ 0.3, and then you have thrown away the resolution you came for.

Reaching the centre of a 300 µm organoid needs 150 µm of depth. Only the low-NA air
lenses, the glycerol lens in a matched medium, and the properly-collared
multi-immersion lenses get there.

**The correction collar is not a detail.** A 25×/1.0 multi-immersion set to 1.48 is
unlimited in CUBIC and dies at 47 µm in iDISCO. You paid $28,000 to remove that
aberration; leaving the collar on the wrong index puts it straight back.

Choose a solvent protocol only when you need whole-mount immunolabelling
(iDISCO's real strength) or months-stable FP (uDISCO/FDISCO), and budget for
quartz, FEP, a dipping cap, and a $28k objective.
