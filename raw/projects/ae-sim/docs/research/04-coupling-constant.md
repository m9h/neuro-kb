# The acoustoelectric coupling constant K

Every number in this project scales linearly with K, the fractional conductivity change per
unit pressure: Δσ / σ₀ = −K ΔP. The prototype hard-codes **K = 1e-9 /Pa (0.1 % / MPa)**. The
literature does not agree with that value, and the disagreement runs in both directions.

| source | K | medium | note |
|---|---|---|---|
| ae-sim (current) | 1e-9 /Pa = 0.100 % / MPa | — | hard-coded; "from Xizi Song's paper" per the Rintoul solver comments |
| Preston 2018 (Witte group) | 0.034 % / MPa | 0.9 % NaCl | the value the Arizona line uses |
| Wang et al. 2026, IEEE TBME, DOI [10.1109/TBME.2026.3682610](https://doi.org/10.1109/TBME.2026.3682610) | 0.024 % / MPa | saline | direct measurement |
| Wang et al. 2026 | 0.022 % / MPa | porcine brain | direct measurement |
| Rintoul 2023 solver comment | 4.6e-10 /Pa = 0.046 % / MPa | water | "real" adiabatic value |
| Rintoul 2023, value needed to match the tank data | 2e-8 /Pa = 2 % / MPa | saline | hard-coded in `kspace_solver.py` |

## Consequences

- **Imaging-mode numbers go down.** Under the measured 0.022–0.034 % / MPa, our real-head
  picovolt estimates scale down by ~3–4× and the required averages go up by ~10–16×.
- **The Rintoul gap is a geometry question, not a K question, until proven otherwise.** The
  20–40× shortfall between her simulation and her tank measurement coincides with our two
  prototype findings: a finite insulated bath raises the AE signal ~5× over infinite-medium
  fields, and the FDTD piston near field is ~4× above the ideal non-diffracting beam. The
  replication in [02](02-rintoul-heterodyning-and-in-vivo.md) tests that.
- **K is a parameter.** `ae_sim.coupling` holds the table above as `K_VALUES`; every example
  reads `AE_SIM_K` (a name, a value in 1/Pa, or `0.034%/MPa`), records the K it used in its
  JSON, and the injected-current examples report sensitivity in µV / (mA·MPa) so they compare
  directly with the experimental papers. The pipeline baselines are at the default `song`.
