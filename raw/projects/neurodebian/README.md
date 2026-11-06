# neurodebian — Debian/NeuroDebian packaging for neuroimaging tools

Debian source packaging (`debian/` trees) for neuroimaging and electrophysiology
tools that are **missing from NeuroDebian / Debian / Ubuntu**, ported from the
Fedora COPR roster [`mhough/neurofedora`](https://gitlab.com/morgan.hough/neurofedora).

Built and maintained on Ubuntu 24.04 (noble): amd64 on the laptop, arm64 on the
DGX Spark. Packaging is architecture-neutral (a shared `debian/` tree).

## Packages

| package | version | what |
|---|---|---|
| `liblsl` | 1.17.7-1 | Lab Streaming Layer core library (liblsl2 + -dev) |
| `pylsl` | 1.18.2-1 | Python LSL binding (uses system liblsl2) |
| `pyxdf` | 1.17.5-1 | Python XDF file reader |
| `insighttoolkit5` | 5.4.6-1~noble2 | ITK 5.4.6 **batteries-included** — adds the 8 remote modules (GenericLabelInterpolator, AdaptiveDenoising, MorphologicalContourInterpolation, GrowCut, SimpleITKFilters, LabelErodeDilate, MGHIO, IOScanco) that the downstream neuro stack needs |
| `ants` | 2.6.5-1~noble1 | Advanced Normalization Tools, built against the system ITK above |

The batteries-included ITK is the foundational package: ANTs, ITK-SNAP,
SimpleITK, MITK and 3D Slicer all depend on those remote modules.

## Layout

```
packages/<name>/debian/   # the Debian packaging for each tool
versions.tsv              # cross-channel version tracker (upstream/copr/debian/ours)
refresh_versions.py       # regenerate the tracker (--net pulls upstream via gh)
neurodebian_validated_roadmap.md   # validated EXISTS/MISSING/OUTDATED status of the COPR roster
```

## Building a package

```sh
cd packages/<name>
# place the upstream orig tarball as ../<name>_<ver>.orig.tar.{gz,xz}, extract,
# copy in debian/, then:
dpkg-buildpackage -us -uc -b
```

ITK/ANTs are large C++ builds; use `DEB_BUILD_OPTIONS="parallel=N"` to bound
memory (ITK template TUs peak ~3 GB each).
