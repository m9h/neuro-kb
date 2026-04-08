---
category: infrastructure
section: appendix
weight: 50
title: "Setae Architecture and Developer Guide"
status: draft
slide_summary: "Five-layer architecture from contact primitives to optimization, with SI units, NamedTuple pytrees, float64 precision, and experimental validation against published data."
tags: [jax, architecture, developer-guide, testing, setae]
---

# setae -- Bio-Inspired Surface Mechanics in JAX

## Quick start

```bash
cd /home/mhough/dev/setae
pytest setae/tests/ -v
```

## Architecture

| Layer | Modules | Purpose |
|-------|---------|---------|
| 0 | `_contact`, `_surface_energy`, `_friction` | Contact and surface primitives |
| 1 | `_materials`, `_beam`, `_shell`, `_hierarchical` | Structural mechanics |
| 2 | `_capillary`, `_wetting`, `_drag` | Fluid-surface interaction |
| 3 | `_gecko`, `_tree_frog`, `_octopus`, `_shark`, ... | Bio-system models |
| 4 | `_optimize`, `_evolutionary`, `_parameterize` | Optimization and design |

## Key conventions

- All units SI (meters, newtons, pascals, joules)
- Pure functions on NamedTuple state pytrees
- float64 in tests (contact mechanics needs precision)
- `_` prefix for private modules; public API via `__init__.py`
- Each bio model validates against published experimental data

## Testing

```bash
pytest setae/tests/ -v                 # all tests
pytest setae/tests/test_contact.py     # just contact mechanics
```
