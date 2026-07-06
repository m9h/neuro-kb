# neuro-kb

An LLM-maintained **shared knowledge base** for a family of neuroimaging projects — the cross-cutting domain knowledge (modalities, physics, tissue properties, methods, head models) that every project's agent would otherwise rediscover from scratch. It follows the Karpathy "LLM wiki" pattern: a directory of interlinked markdown files with YAML frontmatter, generated and maintained by agents.

The corpus lives in [`wiki/`](wiki/) (start at [`wiki/index.md`](wiki/index.md)); immutable sources live in [`raw/`](raw/); agent conventions and the entity schema are in [`CLAUDE.md`](CLAUDE.md).

## Open Knowledge Format (OKF) conformance

This bundle is **conformant with [Google's Open Knowledge Format (OKF) v0.1](https://github.com/GoogleCloudPlatform/knowledge-catalog/tree/main/okf)** — the vendor-neutral spec that formalizes exactly this markdown-wiki pattern. neuro-kb predates OKF and converged on the same shape independently; the [2026-07-03 conformance pass](wiki/log.md) closed the remaining deltas.

Conformance criteria (OKF §9) — all met:

- ✅ Every non-reserved `.md` file has parseable YAML frontmatter with a non-empty `type`
- ✅ Reserved files follow their required structure: `index.md` carries only `okf_version: "0.1"` at the bundle root; `log.md` carries no frontmatter
- ✅ Concept identity = file path; the concept graph is expressed as body markdown links (`## See Also`)

### Gap analysis — status

| # | OKF expectation | Status | Notes |
|---|---|---|---|
| 1 | Only `type` strictly required | ✅ done | present on all 46 concept pages since inception |
| 2 | `index.md`/`log.md` are reserved (no frontmatter; root index may hold only `okf_version`) | ✅ resolved | frontmatter stripped from `log.md`; `index.md` reduced to `okf_version` |
| 3 | Declare `okf_version` | ✅ resolved | `okf_version: "0.1"` in root `index.md` |
| 4 | Concept graph via body markdown links | ✅ already present | `## See Also` sections; the `related:` frontmatter array is retained as a (tolerated) extension key |
| 5 | `description` (RECOMMENDED) | ✅ resolved | backfilled on all 46 pages |
| 6 | `timestamp` (RECOMMENDED, ISO 8601) | ✅ resolved | backfilled from git last-commit date |
| 7 | `## Citations` conventional heading | ✅ resolved | 43 reference headings normalized |
| 8 | `resource` / `tags` (RECOMMENDED) | ✅ resolved | `tags` backfilled on all 55 concept pages; `head-model` pages also carry `source`/`local_path` (→ `resource`) |
| 9 | Index entries carry descriptions | ✅ resolved | every `index.md` entry has an inline description |

Legend: ✅ resolved.

All RECOMMENDED fields are now populated; the bundle is fully conformant with OKF v0.1 with no outstanding gaps.

### Frontmatter example (OKF-conformant concept page)

```yaml
---
type: method
title: Full-Waveform Inversion
description: "PDE-constrained inversion that recovers a speed-of-sound map via adjoint gradients."
timestamp: 2026-07-03T00:00:00-07:00
category: optimization
implementations: ["brain-fwi:src/brain_fwi/simulation/forward.py", "stride:stride/optimisation/optimisation_loop.py"]
related: [physics-acoustic.md, tus.md, method-fem.md]
---
```

`type`, `title`, `description`, `timestamp` are OKF reserved fields; `category`, `implementations`, `related` are neuro-kb extension keys (OKF requires consumers to tolerate unknown keys).

## Repo coverage

The wiki maps cross-cutting concepts to their implementations across the `~/dev/` neuroimaging project family via the `implementations:` field (`repo:module` form). The most recent [repo sweep](wiki/log.md) (2026-07-03, ~80 repos) added 11 cross-cutting pages spanning full-waveform inversion, connectome harmonics, QSM, diffuse optical tomography, PINNs, masked autoencoding, tES E-field modeling, in-vitro MEA cultures, FM diagnostics, brain-age benchmarking, and spiking neural decoding.

## Development

The wiki renders to a self-contained interactive graph via
[`scripts/build_viz.py`](scripts/build_viz.py) → `okf-visualizer.html` (gitignored).

A tracked git hook rebuilds it automatically after any commit that touches
`wiki/`. Enable it once per clone:

```bash
scripts/setup-hooks.sh   # sets core.hooksPath=scripts/hooks
```
