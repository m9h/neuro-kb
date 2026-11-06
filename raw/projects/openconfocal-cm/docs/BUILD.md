# Construction plan

Read [`../designs/README.md`](../designs/README.md) first — it tells you whether
this is even the right instrument for your samples. Then
[`DECISIONS.md`](DECISIONS.md) for the reasoning and the corrections.

---

## Phase 0 — Simulate before you buy ($0)

You have Chromatix as a differentiable wave-optics oracle. Almost nobody building
a microscope can simulate it before committing to parts.

1. Decide your **clearing protocol**, because it fixes your refractive index and
   therefore your objective ([`CLEARING.md`](CLEARING.md)). This ordering is easy
   to get backwards and expensive to undo.
2. Run the numbers:
   ```bash
   uv run python scripts/design_scope.py --na 0.75 --magnification 20 --n-immersion 1.0
   ```
3. Extend `chromatix_oracle.py` with the confocal PSF `h_ill · (h_det ⊛ P)` and the
   depth-dependent pupil phase from `aberration.mismatch_opd`. The scalar model in
   `confocal.py` understates axial extent above NA ~0.7 — do not trust an absolute
   sectioning number at high NA without the oracle.
4. Use `design.information_vs_pinhole` to choose the pinhole diameter on a
   simulated 3D sample. Do not accept 1.0 AU on faith. (A 2D in-focus simulation
   gives the aperture no out-of-focus background to reject, so it *understates* the
   value of closing down. Use a 3D phantom, or real data.)
5. Use `scanhead.check_relay` before ordering galvos.

## Phase 1 — The substrate (~$9–20k)

Common to every design in [`../designs/`](../designs/), so nothing here is wasted
if you change your mind:

- Used motorised inverted stand with a **free infinity side port**: Olympus IX81
  or Nikon Ti-E. **Not Zeiss** (ZEN dongle, ApoTome lock-in), **not Leica** (least
  hackable). The side port is where the scan head goes; it is the whole reason to
  care which stand you buy.
- Used cooled sCMOS (Zyla 4.2 / Flash4.0) for navigation and for
  [design 02](../designs/02-widefield-deconvolution.md).
- Micro-Manager + pycro-manager.

**This alone is a working widefield + deconvolution instrument**, and per
*Communications Biology* (2025) depth-variant deconvolution reaches subnuclear
axial resolution to ~500 µm. Build it first. It is the fallback if the scan head
takes longer than you hoped, and the reconstruction is your discipline.

## Phase 2 — The scan head (~$11–15k more)

Galvos, CLS-SL scan lens, TTL200 tube lens, pinhole, cooled GaAsP PMT, NI DAQ,
OpenScan. See [`BOM.md`](BOM.md).

**Do not DIY the laser launch.** Buy a used combiner. One fibre coupler cannot be
optimal across 405–640 nm — the NicoLase build measured 68% coupling at 405 nm and
**31% at 640 nm** — and there is no cheap 561 nm diode at any effort level.

**Alignment.** Front-surface mirror at the sample plane, scan Z, record the axial
reflectance PSF: its FWHM is your optical section thickness, and you need no
fluorophore. Then 100 nm TetraSpeck beads (Thermo **T7279**) for the lateral PSF.
Then measure yourself against a QUAREP-LiMi protocol.

## Phase 3 — The objective ($1k, or $28k)

Start with a **20×/0.75 air** (~$1–2k used) and an **aqueous** clearing protocol.
Upgrade to a multi-immersion dipping lens ($28,090–$31,500 list, no used market)
only when the science demands the depth. See [`CLEARING.md`](CLEARING.md).

---

## What would change the plan

- **Samples above ~1 mm** → build a light-sheet instead
  ([design 03](../designs/03-lightsheet-thick-cleared.md)). Above 1 mm a confocal
  is 10× slower, bleaches the whole excitation cone every plane, and stops
  penetrating around 3 mm.
- **Live thick scattering tissue** → no single-photon confocal reaches it. That is
  two-photon, and the femtosecond source is most of the cost.
- **Thin samples only** → stop at Phase 1 and put the money into deconvolution.
