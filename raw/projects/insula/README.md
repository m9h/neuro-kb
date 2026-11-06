# Insula

Insula is a modernised fork of the core of the
[BrainVISA](https://brainvisa.info/) neuroimaging suite: the AIMS image, mesh,
texture and graph libraries and tools, the soma-io file-format layer, the
soma-base Python core, and the Anatomist viewer, with their Python bindings.

It exists to make that code buildable and maintainable on current Linux
distributions: Qt 6 only, Qt's own `AUTOMOC`/`AUTOUIC`/`AUTORCC`, and stock
CMake (exports, `GNUInstallDirs`, presets, `ctest`) in place of the custom
`brainvisa-cmake` build layer and `bv_maker`. See [docs/PLAN.md](docs/PLAN.md).

Repository: <https://github.com/m9h/insula>.

**Status: stage 1.** This repository currently holds the upstream code and its
history, unchanged, plus the Fedora fix commits that the Fedora packages carry
today. No build logic has been converted yet.

## Credit

All of this code was written by the BrainVISA developers at CEA / NeuroSpin
(formerly the Institut Fédératif de Recherche 49) and their collaborators, and
by the Populse project for soma-base. Their history, with authorship, is kept
in full in this repository: each directory is an upstream repository imported
with its commits. Insula is not affiliated with or endorsed by the CEA, and
"BrainVISA" and "Anatomist" remain the names of their projects. Python import
names (`soma`, `aims`, `anatomist`) are kept so existing scripts keep working.

## Layout

| Directory | Upstream | Licence |
|---|---|---|
| `soma-base/` | populse/soma-base | CeCILL-B |
| `soma-io/` | brainvisa/soma-io | CeCILL-B |
| `aims-free/` | brainvisa/aims-free | CeCILL-B |
| `aims-gpl/` | brainvisa/aims-gpl | CeCILL-2.1 |
| `anatomist-free/` | brainvisa/anatomist-free | CeCILL-B |
| `anatomist-gpl/` | brainvisa/anatomist-gpl | CeCILL-2.0 |
| `brainvisa-cmake/` | brainvisa/brainvisa-cmake | CeCILL-B |
| `brainvisa-share/` | brainvisa/brainvisa-share | CeCILL-B |

Some directories contain third-party code under other licences; [NOTICE](NOTICE)
lists them all and the texts are in [LICENSES/](LICENSES/).

## What is not here

The University of Louvain ECAT library, which `aims-free` embeds, is licensed
for non-commercial use only and may not be redistributed in modified form. It
has been removed from the tree and from the history of this repository. Morphologist
is not included: its core is proprietary.
