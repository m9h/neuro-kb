# libspm

Standalone C library packaging the numerical core of [SPM](https://www.fil.ion.ucl.ac.uk/spm/) (Statistical Parametric Mapping).

SPM's ~10% C code — the numerical solvers, image resampling, and histogram routines in `src/` — is separable from the MATLAB codebase. This project compiles those GPL-2.0 routines as a native shared library (`libspm.so` / `libspm.dylib` / `spm.dll`) for use from any language.

## Modules

| Module | Header | Description |
|--------|--------|-------------|
| **diffeo** | `spm/diffeo.h` | Diffeomorphic registration — composition, Jacobians, push/pull warping, inverse deformation |
| **field** | `spm/field.h` | Full Multigrid (FMG) and Conjugate Gradient PDE solvers (3-component and N-component) |
| **regularisers** | `spm/regularisers.h` | Membrane, bending, and linear-elasticity regularisation |
| **bsplines** | `spm/bsplines.h` | B-spline interpolation, degrees 0–7, mirror/wrap boundaries (Thevenaz & Unser) |
| **histogram** | `spm/histogram.h` | 2D joint histogram with affine mapping for mutual-information registration |
| **gmm** | `spm/gmm.h` | Gaussian Mixture Models for tissue segmentation with missing-data handling |
| **expm** | `spm/expm.h` | Matrix exponential (2×2, 3×3) and Lie-group exponential |
| **boundary** | `spm/boundary.h` | Boundary conditions: circulant, Neumann, Dirichlet, sliding |
| **openmp** | `spm/openmp.h` | Thread count management (respects `SPM_NUM_THREADS`) |

## Building

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
```

### Options

| Option | Default | Description |
|--------|---------|-------------|
| `SPM_USE_OPENMP` | `ON` | Enable OpenMP parallelisation |
| `SPM_BUILD_SHARED` | `ON` | Build shared library |
| `SPM_BUILD_STATIC` | `ON` | Build static library |
| `SPM_BUILD_EXAMPLES` | `ON` | Build example programs |
| `SPM_BUILD_TESTS` | `ON` | Build tests |

### Install

```bash
cmake --install build --prefix /usr/local
```

This installs headers to `include/spm/`, the library to `lib/`, and a `libspm.pc` for pkg-config.

## Usage

```c
#include <spm/libspm.h>

int main(void) {
    /* Set Neumann boundary conditions */
    spm_set_boundary(SPM_BOUND_NEUMANN);

    /* 3x3 matrix exponential */
    float A[9] = {0, -0.1f, 0, 0.1f, 0, 0, 0, 0, 0};
    float L[9];
    spm_expm33(A, L);

    /* Control OpenMP threads */
    spm_set_num_threads(4);

    return 0;
}
```

Link with:
```bash
gcc -o myapp myapp.c $(pkg-config --cflags --libs libspm)
# or directly:
gcc -o myapp myapp.c -lspm -lm
```

## Dependencies

- **C99 compiler** (GCC, Clang, MSVC)
- **CMake** >= 3.16
- **libm** (standard math library)
- **OpenMP** (optional, for parallelisation)

No external numerical libraries (BLAS, LAPACK, FFTW) are required. All linear algebra, interpolation, and PDE solvers are self-contained.

## Origin

All C source code is extracted from the [SPM](https://github.com/spm/spm) software developed by the [Wellcome Centre for Human Neuroimaging](https://www.fil.ion.ucl.ac.uk/spm/) at UCL. Principal authors of the C code: John Ashburner, Yael Balbastre, Mikael Brudfors, Jesper Andersson, Guillaume Flandin.

The upstream SPM `Makefile` already contains a `libSPM.so` target for the shoot/diffeo/histogram/GMM core. This project extends that to a full standalone library with a CMake build system and clean public headers.

## License

GPL-2.0-only — same as SPM. See [LICENSE](LICENSE).
