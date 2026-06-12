# charm-gems-cli

FSL-style command-line binaries for [`charm-gems`](https://github.com/simnibs/charm-gems) — the C++ segmentation kernels (`KvlImage`, `KvlMesh`, `KvlGMM`, `KvlEMSegmenter`) underneath SimNIBS CHARM.

`charm-gems` itself ships only as a Python library. This package exposes its functionality as standalone CLI tools that follow [FSL](https://fsl.fmrib.ox.ac.uk/) conventions: one binary per task, 12-character left-aligned key/value output, single-letter flags.

## Status

| Binary | Wraps | Status |
|---|---|---|
| `cgemsinfo` | `KvlImage` | shipped (v0.1) |
| `cgemsmesh` | `KvlMesh`, `KvlMeshCollection` | reserved |
| `cgemsreg`  | `KvlAffineRegistration`, `KvlRigidRegistration` | reserved |

**Note on missing classes.** `KvlGMM` and `KvlEMSegmenter` (the GMM-EM tissue clustering primitives) are *not* exposed in `charm-gems` v1.3.3 — only `KvlCostAndGradientCalculator`. Full atlas-prior segmentation also requires the CHARM atlas, which isn't shipped with `charm-gems` (it lives in SimNIBS' `simnibs/resources/templates/`).

## Install

```sh
pip install charm-gems-cli
```

Requires **Python 3.10 or newer**. Upstream `charm-gems` only publishes wheels for cp38–cp311 on PyPI (v1.3.3), so newer Python versions must build it from source.

On macOS, you should install via Homebrew (`brew install m9h/neuro/charm-gems`) which handles the source build for modern Pythons (including 3.13), and then install this package.

## cgemsinfo

```sh
$ cgemsinfo HeadSegmentation.nrrd
filename     /data/sci/HeadSegmentation.nrrd
data_type    UINT8
dim0         3
dim1         256
dim2         256
dim3         180
pixdim1      1.000000
pixdim2      1.000000
pixdim3      1.000000
voxels       11796480
min          0
max          8
xorient      Right-to-Left
yorient      Posterior-to-Anterior
zorient      Inferior-to-Superior

$ cgemsinfo -L HeadSegmentation.nrrd | tail -10
label              voxels   fraction
0             10245112     0.8685
1                12348     0.0010
2               523099     0.0443
3               428103     0.0363
...

$ cgemsinfo --json HeadSegmentation.nrrd | jq .dim
[256, 256, 180]
```

### Flags
- `-L`, `--labels` — append per-label voxel counts (integer images only)
- `-j`, `--json` — emit machine-readable JSON instead of FSL-style text
- `-h`, `--help` — usage

## License

GPL-3.0-or-later (matches the upstream `charm-gems` license, since this package is a derivative work that links it at runtime).
