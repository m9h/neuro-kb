# openconfocal-cm

Design tools for a **galvo point-scanning confocal microscope**, built for fixed,
cleared tissue: sections, organoids, thick slices.

Differentiable forward model, closed-form design formulas, and a costed
construction plan — plus honest documentation of the several instruments this is
*not*, and when you should build one of those instead.

## Why a point scanner

- **One pinhole, so no crosstalk.** A spinning disk's pinholes sit close enough
  that out-of-focus light leaks between them: contrast is *completely lost at
  ~30 µm* and background exceeds 60% of signal by ~80 µm (Shimozawa, *PNAS*
  110:3399, 2013). **Clearing does not fix this** — the crosstalk is driven by
  out-of-focus fluorescence, not scattering.
- **Fixed samples make slow scanning free.** No phototoxicity, no dynamics.
  Integrate as long as you like.
- **Coverslip mounting.** For individual small cleared organoids this is the
  field's default, and a light-sheet's dipping chamber is not worth it below ~1 mm.

## The number that decides your budget

Not the microscope. **The objective.**

```
$ uv run python scripts/design_scope.py --na 0.65 --magnification 40

Depth at which Strehl falls to 0.8 (diffraction-limited), 520 nm:
  60x/1.4 oil  -> iDISCO n=1.56        7.2 um
  60x/1.2 oil  -> iDISCO n=1.56       28.6 um
  20x/0.75 air -> iDISCO n=1.56        9.3 um
  10x/0.45 air -> iDISCO n=1.56      113.1 um
  25x/1.1 multi-immersion, matched    unlimited (working-distance limited)
```

Index-mismatch spherical aberration grows **linearly with depth** and scales as
**NA⁴** once refocusing absorbs the piston and defocus terms. A 60×/1.4 oil lens
is diffraction-limited for ~7 µm into cleared tissue — and its 130 µm working
distance means it could never have reached the middle of a 300 µm organoid anyway.
Two independent walls.

That is why every cleared-tissue objective is low-NA and long-WD, and why they
cost **$28,090–$31,500** with no used market. **The clearing protocol is an optics
decision** — see [`docs/CLEARING.md`](docs/CLEARING.md).

## Modules

| Module | What |
|---|---|
| `optics.py` | Airy units, sectioning, pinhole and relay sizing. Pure Python. |
| `confocal.py` | Scalar wave-optics forward model: `h_ill · (h_det ⊛ P)`. |
| `aberration.py` | Depth-dependent spherical aberration. **Decides which objective you buy.** |
| `scanhead.py` | Galvo field of view, pupil fill, and the parts that don't fit. |
| `design.py` | Choose the pinhole diameter by information content (IDEAL), not folklore. |
| `ism.py` | Image scanning microscopy — same physics, plausible upgrade path. |
| `stages.py` | Resolution-vs-accuracy arithmetic. The stage only navigates now. |

`confocal.py` is a hand-rolled scalar model (the `dpc.py` / `chromatix_oracle.py`
split from [`bsccm-jax`](https://github.com/m9h/bsccm-jax)). **It understates axial
extent above NA ~0.7** — validate against Chromatix before quoting a number at
high NA.

## Two traps `scanhead.py` exists to catch

- `f_obj = f_tube_design / M_engraved`. An Olympus 20× on a Thorlabs TTL200 runs
  at **22.2×**.
- A **low**-magnification objective needs a **bigger** galvo. The 60×/1.4 needs a
  2.94 mm beam; the 20×/0.75 needs **4.72 mm** and clips a 4 mm aperture. Backwards
  from intuition, and where scan-head builds get bought wrong.

## Quickstart

```bash
uv sync --extra dev
uv run pytest -q                                      # 46 tests, no GPU
uv run python scripts/design_scope.py --na 0.65 --magnification 40
```

## Documentation

- [`designs/`](designs/) — **start here.** A ladder of instruments by sample
  thickness, each costed. This confocal is one rung.
- [`docs/CLEARING.md`](docs/CLEARING.md) — refractive index by protocol, the
  objectives that follow, and why solvents dissolve your lens.
- [`docs/BUILD.md`](docs/BUILD.md) — phased construction plan.
- [`docs/BOM.md`](docs/BOM.md) — sourced prices, with every inference flagged.
- [`docs/DECISIONS.md`](docs/DECISIONS.md) — architecture, corrections to
  widely-repeated bad advice, and a log of the times this design was wrong.
- [`docs/REFERENCES.md`](docs/REFERENCES.md) — open-source confocal, light-sheet,
  light-field and platform projects, with licences and honest maturity notes.
- [`docs/LEARNING.md`](docs/LEARNING.md) — how to actually learn this.
- [`CONTRIBUTING.md`](CONTRIBUTING.md) — red-green TDD and the evidence standard.

## Status

Phase 0 (simulate). **Nothing has been built.** Every number in `docs/` is sourced
with a URL or explicitly flagged as inferred; used-equipment prices are *asking*
prices from listings, not transacted.

## Licence

**BSD-3-Clause** throughout. This repository contains no OpenFlexure-derived
geometry and carries no CERN-OHL obligation.
