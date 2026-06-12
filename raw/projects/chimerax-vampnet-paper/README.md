# `chimerax-vampnet` research plan (paper draft)

A LaTeX manuscript that doubles as the research plan + pre-registration
for the `chimerax-vampnet` ChimeraX bundle. Builds with
[tectonic](https://tectonic-typesetting.github.io/), no system TeX
install needed.

## Build

```bash
make            # → paper.pdf
```

First build downloads ~150 MB of LaTeX packages from the tectonic
package cache. Subsequent builds are fast.

## Live preview during editing

```bash
make watch      # auto-rebuild on save (needs `entr`)
```

## Files

| File | Purpose |
|---|---|
| `paper.tex` | the manuscript itself |
| `references.bib` | BibTeX bibliography (curated to match the citations in `paper.tex`) |
| `Makefile` | tectonic build invocation |
| `figures/` | placeholder for figures (filled as results land) |
| `sections/` | placeholder for per-section split files if `paper.tex` gets too long |

## What this document is and isn't

- **Is**: a research plan / pre-registration for the bundle and the
  Notch1 NRR demonstration. Methods and pre-registered hypotheses are
  final at the time of building.
- **Isn't**: a final paper. Results sections contain placeholders; the
  Conclusion is a stub. Updated in-place as the project executes.

## Companion documents

- `../htgaa-protein-design-supplement.md` — the broader stack
  (ChimeraX, LIVIA, Stack A / Stack B) the bundle slots into.
- `../chimerax-vampnet-plan.md` — the implementation plan with
  day-by-day breakdown, compute budget, risk register, locked
  decisions.
